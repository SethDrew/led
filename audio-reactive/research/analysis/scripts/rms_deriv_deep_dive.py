#!/usr/bin/env python3
"""
Deep dive: what does RMS derivative look like AT each user tap?

Instead of peak detection → F1 score, let's look at the raw signal:
  1. What's the RMS derivative value at each tap? (distribution)
  2. Is the signal reliably positive (energy arriving) at taps?
  3. Does HPSS percussive + RMS derivative do better?
  4. Can combining bass flux + RMS deriv help? (both positive = beat)
  5. What's the signal-to-noise ratio at taps vs random times?
"""

import sys
import numpy as np
import librosa
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
HARMONIX = SEGMENTS / 'harmonix'

TRACKS = [
    (SEGMENTS / 'opiate_intro.wav',
     SEGMENTS / 'opiate_intro.annotations.yaml',
     'beat', 'Opiate Intro'),
    (HARMONIX / 'aroundtheworld.wav',
     HARMONIX / 'aroundtheworld.annotations.yaml',
     'sethbeat', 'Around The World'),
    (HARMONIX / 'blackened.wav',
     HARMONIX / 'blackened.annotations.yaml',
     'sethbeat', 'Blackened'),
    (HARMONIX / 'constantmotion.wav',
     HARMONIX / 'constantmotion.annotations.yaml',
     'sethbeat', 'Constant Motion'),
    (HARMONIX / 'floods.wav',
     HARMONIX / 'floods.annotations.yaml',
     'sethbeat', 'Floods'),
    (HARMONIX / 'limelight.wav',
     HARMONIX / 'limelight.annotations.yaml',
     'sethbeat', 'Limelight'),
    (HARMONIX / 'cameraeye.wav',
     HARMONIX / 'cameraeye.annotations.yaml',
     'sethbeat', 'Camera Eye'),
]

HOP = 512
SR = 44100


def compute_signals(audio, sr):
    """Compute all candidate beat detection signals."""
    # 1. RMS and its derivative
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=HOP)[0]
    dt = HOP / sr
    rms_deriv = np.diff(rms, prepend=rms[0]) / dt
    rms_deriv_pos = np.maximum(rms_deriv, 0)

    # 2. Bass spectral flux
    spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_mask = (freqs >= 20) & (freqs <= 250)
    bass_spec = spec[bass_mask, :]
    bass_flux = np.sum(np.maximum(np.diff(bass_spec, axis=1), 0), axis=0)
    bass_flux = np.insert(bass_flux, 0, 0)

    # 3. HPSS percussive signal → RMS derivative
    harm, perc = librosa.decompose.hpss(spec, margin=3.0)
    perc_audio = librosa.istft(perc, hop_length=HOP, length=len(audio))
    perc_rms = librosa.feature.rms(y=perc_audio, frame_length=2048, hop_length=HOP)[0]
    perc_rms_deriv = np.diff(perc_rms, prepend=perc_rms[0]) / dt
    perc_rms_deriv_pos = np.maximum(perc_rms_deriv, 0)

    # 4. Broadband spectral flux (full spectrum)
    broad_flux = np.sum(np.maximum(np.diff(spec, axis=1), 0), axis=0)
    broad_flux = np.insert(broad_flux, 0, 0)

    # 5. Combined: bass flux * rms_deriv_pos (both must be high)
    # Normalize each to [0,1] range first
    bf_norm = bass_flux / (np.max(bass_flux) + 1e-10)
    rd_norm = rms_deriv_pos / (np.max(rms_deriv_pos) + 1e-10)
    combined = bf_norm * rd_norm  # geometric mean-ish

    # 6. Percussive bass flux: bass flux on percussive component only
    perc_bass = perc[bass_mask, :]
    perc_bass_flux = np.sum(np.maximum(np.diff(perc_bass, axis=1), 0), axis=0)
    perc_bass_flux = np.insert(perc_bass_flux, 0, 0)

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)

    return {
        'times': times,
        'rms_deriv+': rms_deriv_pos,
        'bass_flux': bass_flux,
        'perc_rms_deriv+': perc_rms_deriv_pos,
        'broad_flux': broad_flux,
        'combined': combined,
        'perc_bass_flux': perc_bass_flux,
    }


def sample_at_beats(signal, times, beats, window_sec=0.05):
    """Get signal values at each beat time (max within ±window)."""
    values = []
    for b in beats:
        mask = (times >= b - window_sec) & (times <= b + window_sec)
        if np.any(mask):
            values.append(np.max(signal[mask]))
        else:
            values.append(0)
    return np.array(values)


def sample_random(signal, times, n, min_gap_sec=0.1):
    """Get signal values at n random times."""
    valid = np.arange(len(signal))
    chosen = np.random.choice(valid, size=min(n * 3, len(valid)), replace=False)
    return signal[chosen[:n]]


def score_detections(detected, ground_truth, tolerance_sec=0.1):
    """Compute precision, recall, F1."""
    if len(detected) == 0 or len(ground_truth) == 0:
        return 0, 0, 0
    matched_gt = set()
    tp = 0
    for d in detected:
        diffs = np.abs(ground_truth - d)
        nearest_idx = np.argmin(diffs)
        if diffs[nearest_idx] <= tolerance_sec and nearest_idx not in matched_gt:
            tp += 1
            matched_gt.add(nearest_idx)
    p = tp / len(detected) if len(detected) > 0 else 0
    r = tp / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def detect_peaks(signal, times, threshold_frac, cooldown_sec):
    """Simple peak detection at fraction of max."""
    sig_max = np.max(signal)
    if sig_max == 0:
        return np.array([])
    thresh = sig_max * threshold_frac
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    cooldown_frames = max(1, int(cooldown_sec / dt))
    peaks = []
    last = -cooldown_frames
    for i in range(len(signal)):
        if signal[i] > thresh and (i - last) >= cooldown_frames:
            peaks.append(times[i])
            last = i
    return np.array(peaks)


def best_f1(signal, times, beats, cooldowns=[0.1, 0.15, 0.2],
            fracs=[0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]):
    """Grid search for best F1."""
    best = (0, 0, 0, 0, 0, 0)  # f1, p, r, frac, cool, n_det
    for frac in fracs:
        for cool in cooldowns:
            det = detect_peaks(signal, times, frac, cool)
            p, r, f1 = score_detections(det, beats)
            if f1 > best[0]:
                best = (f1, p, r, frac, cool, len(det))
    return best


def analyze_track(wav_path, ann_path, layer, label):
    """Analyze one track."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    with open(ann_path) as f:
        beats = np.array(yaml.safe_load(f)[layer])
    print(f"  {len(beats)} taps, {len(audio)/sr:.1f}s")

    signals = compute_signals(audio, sr)
    times = signals['times']

    # Signal-to-noise analysis: values at beats vs random
    print(f"\n  Signal values at taps vs random times (higher ratio = better discriminant):")
    print(f"  {'Signal':<20s} {'@Taps':>8s} {'@Random':>8s} {'Ratio':>8s} {'Best F1':>8s}")
    print(f"  {'-'*55}")

    np.random.seed(42)
    results = {}
    for name, sig in signals.items():
        if name == 'times':
            continue
        at_beats = sample_at_beats(sig, times, beats)
        at_random = sample_random(sig, times, len(beats) * 5)
        mean_beat = np.mean(at_beats) if len(at_beats) > 0 else 0
        mean_random = np.mean(at_random) if len(at_random) > 0 else 0
        ratio = mean_beat / mean_random if mean_random > 0 else float('inf')

        f1, p, r, frac, cool, n_det = best_f1(sig, times, beats)
        print(f"  {name:<20s} {mean_beat:>8.2f} {mean_random:>8.2f} {ratio:>8.2f} {f1:>8.3f}")
        results[name] = {
            'f1': f1, 'precision': p, 'recall': r,
            'snr': ratio, 'best_frac': frac, 'best_cool': cool
        }

    return label, results, signals, beats


def main():
    print("\n  RMS Derivative Deep Dive — Signal Quality at User Taps")
    print("  " + "=" * 55)

    all_results = []
    all_signals = []
    all_beats = []
    labels = []

    for wav_path, ann_path, layer, label in TRACKS:
        if not wav_path.exists():
            print(f"  SKIP: {label}")
            continue
        l, r, s, b = analyze_track(wav_path, ann_path, layer, label)
        all_results.append(r)
        all_signals.append(s)
        all_beats.append(b)
        labels.append(l)

    # Cross-track summary
    signal_names = [k for k in all_results[0].keys()]
    print(f"\n\n{'='*65}")
    print(f"  CROSS-TRACK AVERAGE F1 and SNR")
    print(f"{'='*65}")
    print(f"  {'Signal':<20s} {'Avg F1':>8s} {'Avg SNR':>8s}")
    print(f"  {'-'*38}")

    summary = []
    for name in signal_names:
        avg_f1 = np.mean([r[name]['f1'] for r in all_results])
        avg_snr = np.mean([r[name]['snr'] for r in all_results])
        summary.append((name, avg_f1, avg_snr))
        print(f"  {name:<20s} {avg_f1:>8.3f} {avg_snr:>8.2f}")

    summary.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Ranked by F1:")
    for i, (name, f1, snr) in enumerate(summary):
        marker = " ◀ BEST" if i == 0 else ""
        print(f"  {i+1}. {name:<20s} F1={f1:.3f}  SNR={snr:.2f}{marker}")

    # Detailed visualization for the first two tracks
    fig, axes = plt.subplots(len(labels), 1, figsize=(18, 3.5 * len(labels)),
                             constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]

    sig_colors = {
        'rms_deriv+': '#e74c3c',
        'bass_flux': '#f39c12',
        'perc_rms_deriv+': '#9b59b6',
        'broad_flux': '#3498db',
        'combined': '#2ecc71',
        'perc_bass_flux': '#e91e63',
    }

    for ax, label, signals, beats in zip(axes, labels, all_signals, all_beats):
        times = signals['times']
        # Normalize all signals to [0,1] for comparison
        for name, sig in signals.items():
            if name == 'times':
                continue
            sig_norm = sig / (np.max(sig) + 1e-10)
            color = sig_colors.get(name, '#888')
            ax.plot(times, sig_norm, color=color, alpha=0.5, linewidth=0.6, label=name)

        # Beat annotations as vertical lines
        for b in beats:
            ax.axvline(b, color='white', alpha=0.4, linewidth=0.5)

        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('All Signals (normalized) vs User Taps', fontsize=13, color='#ddd')
    fig.patch.set_facecolor('#0e0e1a')

    plot_path = Path(__file__).resolve().parent / 'rms_deriv_deep_dive.png'
    fig.savefig(str(plot_path), dpi=150, facecolor=fig.get_facecolor())
    print(f"\n  Saved: {plot_path}")


if __name__ == '__main__':
    main()
