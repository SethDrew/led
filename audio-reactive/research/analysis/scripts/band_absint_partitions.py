#!/usr/bin/env python3
"""
Band-specific abs-integrals with magnitude partitioning.

Hypothesis: computing abs-integral on separate frequency bands (bass/mid/high)
gives cleaner instrument separation than partitioning broadband by magnitude.

Also tests: do magnitude partitions within each band correspond to different
instruments/events? (e.g., bass top 70% = kick drums, bass 30-70% = bass notes)

And: what does the second derivative look like? Is it useful for timing?
"""

import numpy as np
import librosa
import yaml
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from pathlib import Path

SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
HARMONIX = SEGMENTS / 'harmonix'
SR = 44100
HOP = 512
FRAME_LEN = 2048


def bandpass_rms(audio, sr, low, high, frame_length, hop_length):
    """Compute RMS energy of a bandpass-filtered signal."""
    # Butterworth bandpass filter
    nyq = sr / 2
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    sos = butter(4, [low_n, high_n], btype='band', output='sos')
    filtered = sosfilt(sos, audio)
    rms = librosa.feature.rms(y=filtered, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def compute_band_absint(audio, sr, bands, window_sec=0.15):
    """Compute abs-integral of RMS derivative for each frequency band."""
    results = {}
    dt = FRAME_LEN / sr  # time per RMS frame

    for name, (low, high) in bands.items():
        rms = bandpass_rms(audio, sr, low, high, FRAME_LEN, HOP)
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)

        # Derivative
        rms_deriv = np.diff(rms, prepend=rms[0]) / dt

        # Abs-integral over window
        abs_d = np.abs(rms_deriv)
        window_frames = max(1, int(window_sec / dt))
        kernel = np.ones(window_frames) * dt
        abs_int = np.convolve(abs_d, kernel, mode='same')

        # Also compute second derivative for comparison
        rms_deriv2 = np.diff(rms_deriv, prepend=rms_deriv[0]) / dt

        results[name] = {
            'rms': rms,
            'deriv': rms_deriv,
            'deriv2': rms_deriv2,
            'abs_int': abs_int,
            'times': times,
        }

    # Also broadband for comparison
    rms_full = librosa.feature.rms(y=audio, frame_length=FRAME_LEN, hop_length=HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms_full)), sr=sr, hop_length=HOP)
    rms_deriv_full = np.diff(rms_full, prepend=rms_full[0]) / dt
    abs_d_full = np.abs(rms_deriv_full)
    window_frames = max(1, int(window_sec / dt))
    kernel = np.ones(window_frames) * dt
    abs_int_full = np.convolve(abs_d_full, kernel, mode='same')
    results['broadband'] = {
        'rms': rms_full,
        'deriv': rms_deriv_full,
        'deriv2': np.diff(rms_deriv_full, prepend=rms_deriv_full[0]) / dt,
        'abs_int': abs_int_full,
        'times': times,
    }

    return results


def analyze_partitions(abs_int, times, beats, band_name):
    """Check if magnitude partitions separate different beat types."""
    # Compute percentiles of the abs_int signal
    p30 = np.percentile(abs_int, 70)  # top 30%
    p50 = np.percentile(abs_int, 50)  # top 50%
    p70 = np.percentile(abs_int, 30)  # top 70%
    p90 = np.percentile(abs_int, 10)  # top 90%

    # Sample abs_int at beat times (±50ms window)
    beat_values = []
    for b in beats:
        mask = (times >= b - 0.05) & (times <= b + 0.05)
        if np.any(mask):
            beat_values.append(np.max(abs_int[mask]))

    beat_values = np.array(beat_values)

    if len(beat_values) == 0:
        return {}

    # What percentile do beats fall in?
    strong = np.mean(beat_values > p30)   # beats in top 30%
    medium = np.mean((beat_values > p50) & (beat_values <= p30))
    weak = np.mean((beat_values > p70) & (beat_values <= p50))
    miss = np.mean(beat_values <= p70)

    return {
        'strong_pct': strong,
        'medium_pct': medium,
        'weak_pct': weak,
        'miss_pct': miss,
        'beat_mean': np.mean(beat_values),
        'beat_median': np.median(beat_values),
        'signal_median': np.median(abs_int),
        'p30': p30,
        'p50': p50,
    }


def plot_track(title, results, beats, save_path):
    """Visualize band-specific abs-integrals with partitions."""
    bands = [k for k in results.keys() if k != 'broadband']
    n_panels = len(bands) + 2  # bands + broadband + partition timeline

    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 2.8 * n_panels),
                             constrained_layout=True)

    band_colors = {
        'bass': '#e74c3c',
        'mid': '#f39c12',
        'high': '#3498db',
        'broadband': '#2ecc71',
    }

    # Plot each band's abs-integral
    for ax, name in zip(axes[:-1], list(bands) + ['broadband']):
        data = results[name]
        times = data['times']
        sig = data['abs_int']
        color = band_colors.get(name, '#888')

        # Normalize for display
        sig_max = np.max(sig) if np.max(sig) > 0 else 1
        sig_norm = sig / sig_max

        ax.fill_between(times, sig_norm, color=color, alpha=0.4)
        ax.plot(times, sig_norm, color=color, alpha=0.8, linewidth=0.6)

        # Percentile lines
        p70 = np.percentile(sig_norm, 70)
        p90 = np.percentile(sig_norm, 90)
        ax.axhline(p70, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
        ax.axhline(p90, color='white', alpha=0.5, linewidth=0.5, linestyle='--')
        ax.text(times[-1] * 0.99, p90 + 0.02, 'top 10%', fontsize=7,
                color='white', alpha=0.5, ha='right')
        ax.text(times[-1] * 0.99, p70 + 0.02, 'top 30%', fontsize=7,
                color='white', alpha=0.3, ha='right')

        for b in beats:
            ax.axvline(b, color='white', alpha=0.4, linewidth=0.5)

        ax.set_ylabel(name, fontsize=10, color=color)
        ax.set_ylim(-0.02, 1.1)

    # Bottom panel: partition assignments per beat
    ax = axes[-1]
    for i, name in enumerate(bands):
        data = results[name]
        times = data['times']
        sig = data['abs_int']
        p50 = np.percentile(sig, 50)
        p70 = np.percentile(sig, 70)
        p90 = np.percentile(sig, 90)
        color = band_colors.get(name, '#888')

        for b in beats:
            mask = (times >= b - 0.05) & (times <= b + 0.05)
            if np.any(mask):
                val = np.max(sig[mask])
                if val > p90:
                    size = 80
                    alpha = 1.0
                elif val > p70:
                    size = 40
                    alpha = 0.7
                elif val > p50:
                    size = 15
                    alpha = 0.4
                else:
                    size = 5
                    alpha = 0.2
                ax.scatter(b, i, s=size, color=color, alpha=alpha, zorder=3)

    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels(bands, fontsize=9)
    ax.set_ylabel('Band', fontsize=10)
    ax.set_xlabel('Time (s)')
    ax.set_ylim(-0.5, len(bands) - 0.5)

    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(results['broadband']['times'][0],
                    results['broadband']['times'][-1])

    fig.suptitle(f'{title} — Band Abs-Integrals + Magnitude Partitions',
                 fontsize=13, color='#ddd')
    fig.patch.set_facecolor('#0e0e1a')
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    print("\n  Band-Specific Abs-Integrals — Partition Analysis")
    print("  " + "=" * 52)

    BANDS = {
        'bass': (20, 250),
        'mid': (250, 2000),
        'high': (2000, 10000),
    }

    tracks = [
        (SEGMENTS / 'opiate_intro.wav',
         SEGMENTS / 'opiate_intro.annotations.yaml',
         'beat', 'Opiate Intro'),
        (HARMONIX / 'aroundtheworld.wav',
         HARMONIX / 'aroundtheworld.annotations.yaml',
         'sethbeat', 'Around The World'),
        (HARMONIX / 'constantmotion.wav',
         HARMONIX / 'constantmotion.annotations.yaml',
         'sethbeat', 'Constant Motion'),
        (HARMONIX / 'floods.wav',
         HARMONIX / 'floods.annotations.yaml',
         'sethbeat', 'Floods'),
        (HARMONIX / 'limelight.wav',
         HARMONIX / 'limelight.annotations.yaml',
         'sethbeat', 'Limelight'),
    ]

    out_dir = Path(__file__).resolve().parent

    for wav_path, ann_path, layer, title in tracks:
        if not wav_path.exists():
            continue

        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")

        audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        with open(ann_path) as f:
            beats = np.array(yaml.safe_load(f)[layer])

        results = compute_band_absint(audio, sr, BANDS)

        # Partition analysis per band
        print(f"\n  Band partition analysis — where do your beats land?")
        print(f"  {'Band':<12s} {'Top10%':>7s} {'10-30%':>7s} {'30-50%':>7s} "
              f"{'Below50%':>8s} {'BeatMed':>8s} {'SigMed':>8s} {'Ratio':>6s}")
        print(f"  {'-'*68}")

        for name in list(BANDS.keys()) + ['broadband']:
            data = results[name]
            p = analyze_partitions(data['abs_int'], data['times'], beats, name)
            if not p:
                continue
            beat_ratio = p['beat_median'] / p['signal_median'] if p['signal_median'] > 0 else 0
            print(f"  {name:<12s} {p['strong_pct']:>7.0%} {p['medium_pct']:>7.0%} "
                  f"{p['weak_pct']:>7.0%} {p['miss_pct']:>8.0%} "
                  f"{p['beat_median']:>8.4f} {p['signal_median']:>8.4f} {beat_ratio:>6.1f}x")

        # Second derivative analysis
        print(f"\n  Second derivative at beats (timing potential):")
        for name in list(BANDS.keys()) + ['broadband']:
            data = results[name]
            d2 = data['deriv2']
            times = data['times']

            # At beats: is second derivative crossing zero? (inflection = peak arrival)
            d2_at_beats = []
            for b in beats:
                mask = (times >= b - 0.03) & (times <= b + 0.03)
                if np.any(mask):
                    # Look for zero crossing
                    local_d2 = d2[mask]
                    has_crossing = np.any(np.diff(np.sign(local_d2)) != 0)
                    d2_at_beats.append(has_crossing)

            crossing_rate = np.mean(d2_at_beats) if d2_at_beats else 0
            print(f"    {name:<12s} — d²R/dt² zero-crossing at {crossing_rate:.0%} of beats (±30ms)")

        # Save visualization
        plot_path = out_dir / f'band_absint_{title.lower().replace(" ", "_")}.png'
        plot_track(title, results, beats, plot_path)
        print(f"\n  Saved: {plot_path.name}")


if __name__ == '__main__':
    main()
