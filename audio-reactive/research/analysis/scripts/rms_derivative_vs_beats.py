#!/usr/bin/env python3
"""
RMS Derivative vs User Taps — analysis across all annotated tracks.

Question: Can RMS derivative (rate of energy change) be used as a beat
detection signal? The hypothesis is that beats show as sharp positive
spikes in RMS derivative (energy arriving fast), and the signal is more
robust than bass-band spectral flux because it captures ALL energy
arrivals, not just bass.

Analyzes:
  - RMS derivative at different window sizes
  - Rolling averages at 1s, 5s, 10s for adaptive thresholding
  - Peak detection with various thresholds relative to rolling averages
  - F1 score / precision / recall against user tap annotations
"""

import sys
import os
import yaml
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
HARMONIX = SEGMENTS / 'harmonix'

# All tracks with user tap annotations
TRACKS = [
    # (wav_path, annotation_path, beat_layer_name, label)
    (SEGMENTS / 'opiate_intro.wav',
     SEGMENTS / 'opiate_intro.annotations.yaml',
     'beat', 'Opiate Intro (Tool)'),

    (HARMONIX / 'aroundtheworld.wav',
     HARMONIX / 'aroundtheworld.annotations.yaml',
     'sethbeat', 'Around The World (Daft Punk)'),

    (HARMONIX / 'blackened.wav',
     HARMONIX / 'blackened.annotations.yaml',
     'sethbeat', 'Blackened (Metallica)'),

    (HARMONIX / 'cameraeye.wav',
     HARMONIX / 'cameraeye.annotations.yaml',
     'sethbeat', 'The Camera Eye (Rush)'),

    (HARMONIX / 'constantmotion.wav',
     HARMONIX / 'constantmotion.annotations.yaml',
     'sethbeat', 'Constant Motion (Dream Theater)'),

    (HARMONIX / 'floods.wav',
     HARMONIX / 'floods.annotations.yaml',
     'sethbeat', 'Floods (Pantera)'),

    (HARMONIX / 'limelight.wav',
     HARMONIX / 'limelight.annotations.yaml',
     'sethbeat', 'Limelight (Rush)'),
]


def load_annotations(ann_path, layer):
    """Load beat timestamps from YAML annotation file."""
    with open(ann_path) as f:
        data = yaml.safe_load(f)
    return np.array(data[layer], dtype=np.float64)


def compute_rms_derivative(audio, sr, hop_length=512):
    """Compute RMS energy and its derivative."""
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    dt = hop_length / sr
    # Derivative: rate of change of RMS
    rms_deriv = np.diff(rms, prepend=rms[0]) / dt
    return times, rms, rms_deriv


def compute_rolling_stats(signal, times, window_secs):
    """Compute rolling mean and std of a signal over a time window."""
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    window_frames = max(1, int(window_secs / dt))
    # Cumulative sum trick for rolling mean
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    rolling_mean = np.zeros_like(signal)
    rolling_std = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window_frames)
        n = i - start
        if n > 0:
            rolling_mean[i] = (cumsum[i] - cumsum[start]) / n
        else:
            rolling_mean[i] = signal[i]
    # Rolling std via second pass
    for i in range(len(signal)):
        start = max(0, i - window_frames)
        n = i - start
        if n > 1:
            window_slice = signal[start:i]
            rolling_std[i] = np.std(window_slice)
        else:
            rolling_std[i] = 0.0
    return rolling_mean, rolling_std


def detect_peaks_adaptive(signal, times, rolling_mean, rolling_std,
                          threshold_multiplier=2.0, cooldown_sec=0.1):
    """Detect peaks where signal exceeds rolling_mean + threshold_multiplier * rolling_std."""
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    cooldown_frames = max(1, int(cooldown_sec / dt))

    peaks = []
    last_peak = -cooldown_frames
    for i in range(len(signal)):
        thresh = rolling_mean[i] + threshold_multiplier * rolling_std[i]
        if signal[i] > thresh and (i - last_peak) >= cooldown_frames:
            peaks.append(times[i])
            last_peak = i
    return np.array(peaks)


def detect_peaks_fixed(signal, times, threshold, cooldown_sec=0.1):
    """Detect peaks above a fixed threshold on the positive derivative."""
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    cooldown_frames = max(1, int(cooldown_sec / dt))

    peaks = []
    last_peak = -cooldown_frames
    for i in range(len(signal)):
        if signal[i] > threshold and (i - last_peak) >= cooldown_frames:
            peaks.append(times[i])
            last_peak = i
    return np.array(peaks)


def score_detections(detected, ground_truth, tolerance_sec=0.1):
    """Compute precision, recall, F1 between detected beats and ground truth."""
    if len(detected) == 0 or len(ground_truth) == 0:
        return 0, 0, 0

    # Match each detection to nearest ground truth
    matched_gt = set()
    true_positives = 0
    for d in detected:
        diffs = np.abs(ground_truth - d)
        nearest_idx = np.argmin(diffs)
        if diffs[nearest_idx] <= tolerance_sec and nearest_idx not in matched_gt:
            true_positives += 1
            matched_gt.add(nearest_idx)

    precision = true_positives / len(detected) if len(detected) > 0 else 0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def analyze_track(wav_path, ann_path, beat_layer, label, hop_length=512):
    """Full analysis of one track."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Load
    audio, sr = librosa.load(wav_path, sr=44100, mono=True)
    beats = load_annotations(ann_path, beat_layer)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s, {len(beats)} taps")
    if len(beats) > 1:
        ibis = np.diff(beats)
        median_bpm = 60.0 / np.median(ibis)
        print(f"  Median tap tempo: {median_bpm:.0f} BPM")

    # Compute RMS derivative
    times, rms, rms_deriv = compute_rms_derivative(audio, sr, hop_length)

    # Half-wave rectify: only positive changes (energy arriving)
    rms_deriv_pos = np.maximum(rms_deriv, 0)

    # Rolling averages at different timescales
    windows = [0.5, 1.0, 2.0, 5.0]
    print(f"\n  --- Adaptive threshold (RMS deriv+ vs rolling mean+std) ---")
    print(f"  {'Window':>6s} {'Mult':>5s} {'Cool':>5s} | {'Det':>4s} {'Prec':>5s} {'Rec':>5s} {'F1':>5s}")
    print(f"  {'-'*50}")

    best_f1 = 0
    best_config = None

    for window_sec in windows:
        r_mean, r_std = compute_rolling_stats(rms_deriv_pos, times, window_sec)
        for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for cooldown in [0.1, 0.15, 0.2]:
                detected = detect_peaks_adaptive(
                    rms_deriv_pos, times, r_mean, r_std, mult, cooldown)
                p, r, f1 = score_detections(detected, beats, tolerance_sec=0.1)
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = (window_sec, mult, cooldown, len(detected), p, r, f1)

    w, m, c, n, p, r, f1 = best_config
    print(f"  Best: win={w:.1f}s mult={m:.1f} cool={c:.2f}s | "
          f"det={n:3d} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # Also test fixed threshold on raw positive derivative
    print(f"\n  --- Fixed threshold on RMS deriv+ ---")
    rms_deriv_max = np.max(rms_deriv_pos) if np.max(rms_deriv_pos) > 0 else 1
    best_fixed_f1 = 0
    best_fixed = None
    for frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        threshold = rms_deriv_max * frac
        for cooldown in [0.1, 0.15, 0.2]:
            detected = detect_peaks_fixed(rms_deriv_pos, times, threshold, cooldown)
            p, r, f1 = score_detections(detected, beats, tolerance_sec=0.1)
            if f1 > best_fixed_f1:
                best_fixed_f1 = f1
                best_fixed = (frac, cooldown, len(detected), p, r, f1)

    frac, c, n, p, r, f1 = best_fixed
    print(f"  Best: thresh={frac:.0%} of max, cool={c:.2f}s | "
          f"det={n:3d} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # Compare with bass-band spectral flux (our current approach)
    print(f"\n  --- Bass spectral flux (current bass_pulse approach) ---")
    spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_mask = (freqs >= 20) & (freqs <= 250)
    bass_spec = spec[bass_mask, :]
    # Half-wave rectified spectral flux
    bass_flux = np.sum(np.maximum(np.diff(bass_spec, axis=1), 0), axis=0)
    bass_flux = np.insert(bass_flux, 0, 0)  # prepend 0 for alignment
    flux_times = librosa.frames_to_time(np.arange(len(bass_flux)), sr=sr, hop_length=hop_length)

    best_bass_f1 = 0
    best_bass = None
    flux_max = np.max(bass_flux) if np.max(bass_flux) > 0 else 1
    for frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        threshold = flux_max * frac
        for cooldown in [0.1, 0.15, 0.2]:
            detected = detect_peaks_fixed(bass_flux, flux_times, threshold, cooldown)
            p, r, f1 = score_detections(detected, beats, tolerance_sec=0.1)
            if f1 > best_bass_f1:
                best_bass_f1 = f1
                best_bass = (frac, cooldown, len(detected), p, r, f1)

    frac, c, n, p, r, f1 = best_bass
    print(f"  Best: thresh={frac:.0%} of max, cool={c:.2f}s | "
          f"det={n:3d} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    return {
        'label': label,
        'duration': duration,
        'n_taps': len(beats),
        'adaptive_f1': best_config[6],
        'adaptive_config': best_config[:3],
        'fixed_f1': best_fixed[5],
        'bass_flux_f1': best_bass[5] if best_bass else 0,
        'times': times,
        'rms': rms,
        'rms_deriv': rms_deriv,
        'rms_deriv_pos': rms_deriv_pos,
        'beats': beats,
    }


def plot_comparison(results, save_path=None):
    """Plot RMS derivative vs beats for all tracks."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        times = res['times']
        rms_deriv_pos = res['rms_deriv_pos']
        beats = res['beats']

        # Plot positive RMS derivative
        ax.plot(times, rms_deriv_pos, color='#e74c3c', alpha=0.7, linewidth=0.5, label='RMS deriv+')

        # Rolling averages
        for win, color, alpha in [(1.0, '#3498db', 0.6), (5.0, '#2ecc71', 0.6)]:
            r_mean, r_std = compute_rolling_stats(rms_deriv_pos, times, win)
            ax.plot(times, r_mean, color=color, alpha=alpha, linewidth=1.0,
                    label=f'{win:.0f}s mean')
            ax.plot(times, r_mean + 2.0 * r_std, color=color, alpha=alpha * 0.5,
                    linewidth=0.8, linestyle='--', label=f'{win:.0f}s +2σ')

        # Beat annotations
        for b in beats:
            ax.axvline(b, color='white', alpha=0.5, linewidth=0.5)

        ax.set_ylabel(res['label'].split('(')[0].strip(), fontsize=9)
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#888')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # F1 scores in corner
        ax.text(0.98, 0.95,
                f"Adaptive F1={res['adaptive_f1']:.3f}  |  Bass Flux F1={res['bass_flux_f1']:.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color='#ccc', fontfamily='monospace')

    axes[0].legend(loc='upper left', fontsize=7, ncol=5, framealpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('RMS Derivative (positive) vs User Taps — All Tracks', fontsize=13, color='#ddd')
    fig.patch.set_facecolor('#0e0e1a')

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"\n  Saved plot: {save_path}")
    else:
        plt.show()


def main():
    print("\n  RMS Derivative vs User Taps — Cross-Track Analysis")
    print("  " + "=" * 55)

    results = []
    for wav_path, ann_path, layer, label in TRACKS:
        if not wav_path.exists():
            print(f"\n  SKIP: {label} — file not found: {wav_path}")
            continue
        res = analyze_track(wav_path, ann_path, layer, label)
        results.append(res)

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — Best F1 scores per approach")
    print(f"{'='*70}")
    print(f"  {'Track':<30s} {'Adaptive':>8s} {'Fixed':>8s} {'BassFlux':>8s} {'Winner':>10s}")
    print(f"  {'-'*66}")

    adaptive_wins = 0
    bass_wins = 0
    for res in results:
        a, f, b = res['adaptive_f1'], res['fixed_f1'], res['bass_flux_f1']
        best = max(a, f, b)
        if best == a:
            winner = "Adaptive"
            adaptive_wins += 1
        elif best == f:
            winner = "Fixed"
        else:
            winner = "BassFlux"
            bass_wins += 1
        print(f"  {res['label']:<30s} {a:>8.3f} {f:>8.3f} {b:>8.3f} {winner:>10s}")

    # Averages
    avg_a = np.mean([r['adaptive_f1'] for r in results])
    avg_f = np.mean([r['fixed_f1'] for r in results])
    avg_b = np.mean([r['bass_flux_f1'] for r in results])
    print(f"  {'-'*66}")
    print(f"  {'AVERAGE':<30s} {avg_a:>8.3f} {avg_f:>8.3f} {avg_b:>8.3f}")

    # Plot
    plot_path = Path(__file__).resolve().parent / 'rms_derivative_vs_beats.png'
    plot_comparison(results, save_path=str(plot_path))


if __name__ == '__main__':
    main()
