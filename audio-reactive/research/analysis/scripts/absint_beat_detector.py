#!/usr/bin/env python3
"""
Absolute-value integral beat detector — exploring a new approach.

Observation: RMS derivative pulses positive then negative on each beat.
The absolute value integral over a short window captures the "perturbation"
from a beat — high when a beat just happened, low during steady-state.

Step 1: Detect beats "late" (after they happen) via the abs-integral signal.
Step 2: Predict next beat from recent intervals and fire "on time".

This script visualizes the signal and tests both steps.
"""

import numpy as np
import librosa
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
HARMONIX = SEGMENTS / 'harmonix'

SR = 44100
HOP = 512


def compute_rms_derivative(audio, sr, hop):
    """RMS energy and its time derivative."""
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    dt = hop / sr
    rms_deriv = np.diff(rms, prepend=rms[0]) / dt
    return times, rms, rms_deriv


def compute_abs_integral(rms_deriv, times, window_sec=0.25):
    """
    Absolute-value integral of RMS derivative over a trailing window.

    High value = big energy changes happened recently (beat just occurred).
    Low value = steady state (between beats).

    This is essentially: sum(|d(RMS)/dt|) over the last `window_sec` seconds.
    """
    dt = times[1] - times[0] if len(times) > 1 else HOP / SR
    window_frames = max(1, int(window_sec / dt))

    abs_deriv = np.abs(rms_deriv)

    # Rolling sum via cumulative sum (much faster than loop)
    cumsum = np.cumsum(np.insert(abs_deriv * dt, 0, 0))  # multiply by dt for true integral
    abs_integral = np.zeros(len(abs_deriv))
    for i in range(len(abs_deriv)):
        start = max(0, i - window_frames)
        abs_integral[i] = cumsum[i + 1] - cumsum[start]

    return abs_integral


def detect_beats_late(abs_integral, times, threshold_frac=0.3, cooldown_sec=0.2):
    """
    Step 1: "Late" beat detection.

    Detect when abs_integral crosses above threshold (= beat just happened).
    The detection is "late" because the integral needs time to accumulate.

    Returns detected beat times (these are LATE — after the actual beat).
    """
    peak = np.max(abs_integral)
    threshold = peak * threshold_frac
    dt = times[1] - times[0]
    cooldown_frames = max(1, int(cooldown_sec / dt))

    detections = []
    last_det = -cooldown_frames
    for i in range(len(abs_integral)):
        if abs_integral[i] > threshold and (i - last_det) >= cooldown_frames:
            detections.append(times[i])
            last_det = i

    return np.array(detections)


def predict_beats(late_detections, times, abs_integral, max_history=8):
    """
    Step 2: Beat prediction.

    From late detections, compute inter-beat interval.
    Predict when the NEXT beat should happen and register on-time.

    Algorithm:
      - After 2+ detections, compute median interval from recent history
      - Predicted beat time = last detection + interval (adjusted for detection lag)
      - Keep predicting until evidence changes (new detection updates the interval)

    Returns (predicted_times, intervals_over_time).
    """
    if len(late_detections) < 2:
        return np.array([]), np.array([])

    # Compute intervals between consecutive late detections
    intervals = np.diff(late_detections)

    # For each detection after the first two, predict the next beat
    predicted = []
    interval_history = []

    for i in range(1, len(late_detections)):
        # Use median of recent intervals for robustness
        recent = intervals[max(0, i-1-max_history):i]
        if len(recent) > 0:
            est_interval = np.median(recent)
            interval_history.append(est_interval)

            # The late detection happened AFTER the beat.
            # Estimate the lag: how far into the window is the peak?
            # For now, assume the beat was ~half a window before the detection.
            # We'll measure this more precisely below.
            predicted_next = late_detections[i] + est_interval
            predicted.append(predicted_next)

    return np.array(predicted), np.array(interval_history)


def estimate_detection_lag(late_detections, ground_truth, tolerance=0.3):
    """Measure systematic lag between late detections and nearest ground truth beats."""
    lags = []
    for d in late_detections:
        diffs = ground_truth - d  # negative = detection is late
        nearest_idx = np.argmin(np.abs(diffs))
        if np.abs(diffs[nearest_idx]) < tolerance:
            lags.append(diffs[nearest_idx])
    return np.array(lags)


def score(detected, ground_truth, tolerance=0.1):
    """Precision, recall, F1."""
    if len(detected) == 0 or len(ground_truth) == 0:
        return 0, 0, 0
    matched = set()
    tp = 0
    for d in detected:
        diffs = np.abs(ground_truth - d)
        idx = np.argmin(diffs)
        if diffs[idx] <= tolerance and idx not in matched:
            tp += 1
            matched.add(idx)
    p = tp / len(detected) if detected.size else 0
    r = tp / len(ground_truth) if ground_truth.size else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    return p, r, f1


def analyze_and_plot(wav_path, ann_path, beat_layer, title):
    """Full analysis + visualization for one track."""
    audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    with open(ann_path) as f:
        data = yaml.safe_load(f)
    beats = np.array(data[beat_layer])

    times, rms, rms_deriv = compute_rms_derivative(audio, sr, HOP)

    # Test multiple window sizes
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"  {len(beats)} beats in {len(audio)/sr:.1f}s")
    if len(beats) > 1:
        print(f"  Beat interval: {np.median(np.diff(beats))*1000:.0f}ms "
              f"({60/np.median(np.diff(beats)):.0f} BPM)")
    print(f"{'='*65}")

    # Try different window sizes for the abs integral
    windows = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    print(f"\n  --- Abs-integral window sweep (late detection) ---")
    print(f"  {'Win':>5s} {'ThFrac':>6s} {'Cool':>5s} | {'Det':>4s} {'Prec':>5s} {'Rec':>5s} "
          f"{'F1':>5s} {'Lag':>7s}")
    print(f"  {'-'*52}")

    best_f1 = 0
    best_config = None
    best_signal = None

    for win in windows:
        abs_int = compute_abs_integral(rms_deriv, times, win)
        for th_frac in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            for cool in [0.15, 0.2, 0.25, 0.3]:
                dets = detect_beats_late(abs_int, times, th_frac, cool)
                p, r, f1 = score(dets, beats, tolerance=0.15)  # 150ms tolerance for late detection
                if f1 > best_f1:
                    lags = estimate_detection_lag(dets, beats)
                    median_lag = np.median(lags) * 1000 if len(lags) > 0 else 0
                    best_f1 = f1
                    best_config = (win, th_frac, cool, len(dets), p, r, f1, median_lag)
                    best_signal = abs_int

    w, th, c, n, p, r, f1, lag = best_config
    print(f"  Best: win={w:.2f}s th={th:.2f} cool={c:.2f}s | "
          f"det={n:3d} P={p:.3f} R={r:.3f} F1={f1:.3f} lag={lag:+.0f}ms")

    # Re-detect with best config for detailed analysis
    abs_int_best = compute_abs_integral(rms_deriv, times, w)
    dets = detect_beats_late(abs_int_best, times, th, c)
    lags = estimate_detection_lag(dets, beats)

    if len(lags) > 0:
        print(f"\n  Detection lag statistics:")
        print(f"    Median: {np.median(lags)*1000:+.0f}ms")
        print(f"    Mean:   {np.mean(lags)*1000:+.0f}ms")
        print(f"    Std:    {np.std(lags)*1000:.0f}ms")
        print(f"    Range:  {np.min(lags)*1000:+.0f} to {np.max(lags)*1000:+.0f}ms")

    # Step 2: prediction
    if len(dets) >= 3:
        predicted, intervals = predict_beats(dets, times, abs_int_best)
        if len(predicted) > 0:
            # Shift predictions back by the median lag to compensate
            median_lag_sec = np.median(lags) if len(lags) > 0 else 0
            predicted_corrected = predicted + median_lag_sec  # lag is negative, so this subtracts

            p2, r2, f12 = score(predicted_corrected, beats, tolerance=0.1)
            print(f"\n  Step 2 — Predicted beats (lag-corrected):")
            print(f"    {len(predicted_corrected)} predictions, P={p2:.3f} R={r2:.3f} F1={f12:.3f}")
            print(f"    (at 100ms tolerance, stricter than late detection)")

            # Also check with 50ms tolerance
            p3, r3, f13 = score(predicted_corrected, beats, tolerance=0.05)
            print(f"    At 50ms tolerance: P={p3:.3f} R={r3:.3f} F1={f13:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1]})

    # Panel 1: Waveform
    ax = axes[0]
    wave_times = np.linspace(0, len(audio)/sr, len(audio))
    ax.plot(wave_times, audio, color='#4a9eff', alpha=0.5, linewidth=0.3)
    for b in beats:
        ax.axvline(b, color='#ff6b6b', alpha=0.7, linewidth=1)
    ax.set_ylabel('Waveform')
    ax.set_title(f'{title} — Abs-Integral Beat Detector', fontsize=12, color='#ddd')

    # Panel 2: RMS derivative (raw)
    ax = axes[1]
    ax.plot(times, rms_deriv, color='#e74c3c', alpha=0.7, linewidth=0.6, label='RMS derivative')
    ax.axhline(0, color='#555', linewidth=0.5)
    ax.fill_between(times, rms_deriv, 0, where=rms_deriv > 0,
                     color='#e74c3c', alpha=0.3, label='positive (energy arriving)')
    ax.fill_between(times, rms_deriv, 0, where=rms_deriv < 0,
                     color='#3498db', alpha=0.3, label='negative (energy leaving)')
    for b in beats:
        ax.axvline(b, color='white', alpha=0.5, linewidth=0.8)
    ax.set_ylabel('d(RMS)/dt')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.3)

    # Panel 3: Absolute integral
    ax = axes[2]
    ax.plot(times, abs_int_best, color='#2ecc71', alpha=0.8, linewidth=1.0,
            label=f'|RMS\'| integral ({w:.2f}s window)')
    peak = np.max(abs_int_best)
    ax.axhline(peak * th, color='#f39c12', linewidth=1, linestyle='--',
               alpha=0.7, label=f'threshold ({th:.0%} of peak)')
    for b in beats:
        ax.axvline(b, color='white', alpha=0.5, linewidth=0.8)
    for d in dets:
        ax.axvline(d, color='#e91e63', alpha=0.8, linewidth=1.5, linestyle=':')
    ax.set_ylabel('Abs integral')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.3)

    # Panel 4: Detection timeline
    ax = axes[3]
    # Ground truth beats
    ax.scatter(beats, np.ones(len(beats)) * 0.8, color='white', s=50, marker='|',
               linewidths=2, label='Your taps', zorder=3)
    # Late detections
    ax.scatter(dets, np.ones(len(dets)) * 0.5, color='#e91e63', s=50, marker='|',
               linewidths=2, label='Late detections', zorder=3)
    # Predictions (if available)
    if len(dets) >= 3:
        predicted, intervals = predict_beats(dets, times, abs_int_best)
        if len(predicted) > 0:
            median_lag_sec = np.median(lags) if len(lags) > 0 else 0
            predicted_corrected = predicted + median_lag_sec
            ax.scatter(predicted_corrected, np.ones(len(predicted_corrected)) * 0.2,
                       color='#2ecc71', s=50, marker='|', linewidths=2,
                       label='Predicted (lag-corrected)', zorder=3)

    ax.set_ylim(0, 1)
    ax.set_ylabel('Timeline')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.3, ncol=3)
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_yticklabels(['Predicted', 'Late det.', 'Your taps'], fontsize=8)

    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(times[0], times[-1])

    fig.patch.set_facecolor('#0e0e1a')
    return fig


def main():
    print("\n  Absolute-Value Integral Beat Detector")
    print("  " + "=" * 45)

    # Primary test: aroundtheworld_5s
    tracks = [
        (HARMONIX / 'aroundtheworld_5s.wav',
         HARMONIX / 'aroundtheworld_5s.annotations.yaml',
         'beats', 'Around The World (5s)'),

        (HARMONIX / 'aroundtheworld.wav',
         HARMONIX / 'aroundtheworld.annotations.yaml',
         'sethbeat', 'Around The World (60s, sethbeat)'),

        (SEGMENTS / 'opiate_intro.wav',
         SEGMENTS / 'opiate_intro.annotations.yaml',
         'beat', 'Opiate Intro (sethbeat)'),

        (HARMONIX / 'constantmotion.wav',
         HARMONIX / 'constantmotion.annotations.yaml',
         'sethbeat', 'Constant Motion (sethbeat)'),

        (HARMONIX / 'floods.wav',
         HARMONIX / 'floods.annotations.yaml',
         'sethbeat', 'Floods (sethbeat)'),

        (HARMONIX / 'limelight.wav',
         HARMONIX / 'limelight.annotations.yaml',
         'sethbeat', 'Limelight (sethbeat)'),

        (HARMONIX / 'blackened.wav',
         HARMONIX / 'blackened.annotations.yaml',
         'sethbeat', 'Blackened (sethbeat)'),

        (HARMONIX / 'cameraeye.wav',
         HARMONIX / 'cameraeye.annotations.yaml',
         'sethbeat', 'Camera Eye (sethbeat)'),
    ]

    figs = []
    for wav_path, ann_path, layer, title in tracks:
        if not wav_path.exists():
            print(f"  SKIP: {title}")
            continue
        fig = analyze_and_plot(wav_path, ann_path, layer, title)
        figs.append((fig, title))

    # Save all figures
    out_dir = Path(__file__).resolve().parent
    for fig, title in figs:
        safe_name = title.split('(')[0].strip().lower().replace(' ', '_')
        path = out_dir / f'absint_{safe_name}.png'
        fig.savefig(str(path), dpi=150, facecolor=fig.get_facecolor())
        print(f"  Saved: {path.name}")

    plt.close('all')


if __name__ == '__main__':
    main()
