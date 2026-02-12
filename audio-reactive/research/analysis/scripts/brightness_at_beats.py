#!/usr/bin/env python3
"""
Compare brightness at sethbeat times: absint_prop vs absint_pred.

Simulates both effects on each track, sampling the rendered brightness
at each user tap time. Higher brightness at beats = better effect.

Also measures:
  - Brightness at beats vs between beats (discrimination)
  - Timing accuracy: is brightness peaking AT the beat or slightly after?
"""

import sys
import os
import numpy as np
import yaml
import librosa
from pathlib import Path

# Add effects to path
EFFECTS_DIR = Path(__file__).resolve().parents[3] / 'effects'
sys.path.insert(0, str(EFFECTS_DIR))

from absint_proportional import AbsIntProportionalEffect
from absint_predictive import AbsIntPredictiveEffect

SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
HARMONIX = SEGMENTS / 'harmonix'

SR = 44100
CHUNK_SIZE = 1024
LED_FPS = 30
FRAME_DT = 1.0 / LED_FPS

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


def simulate_effect(effect, audio, sr, beats):
    """
    Run effect on audio, sample brightness at each LED frame.
    Returns (frame_times, brightness_values) arrays.
    """
    chunk_size = CHUNK_SIZE
    total_samples = len(audio)

    # We'll collect brightness at every LED frame
    frame_times = []
    brightness_values = []

    audio_pos = 0
    next_frame_time = 0.0
    wall_time = 0.0

    while audio_pos < total_samples:
        # Feed one audio chunk
        chunk_end = min(audio_pos + chunk_size, total_samples)
        chunk = audio[audio_pos:chunk_end].astype(np.float32)
        effect.process_audio(chunk)
        audio_pos = chunk_end

        wall_time = audio_pos / sr

        # Render frames as needed
        while next_frame_time <= wall_time:
            frame = effect.render(FRAME_DT)
            # Brightness = max channel of first pixel (they're all the same)
            b = frame[0].max() / 255.0
            frame_times.append(next_frame_time)
            brightness_values.append(b)
            next_frame_time += FRAME_DT

    return np.array(frame_times), np.array(brightness_values)


def sample_brightness_at_beats(frame_times, brightness, beats, window_ms=50):
    """
    Get the peak brightness within ±window_ms of each beat.
    Also get brightness at "anti-beats" (midpoints between consecutive beats).
    """
    window_sec = window_ms / 1000.0
    beat_brightness = []
    for b in beats:
        mask = (frame_times >= b - window_sec) & (frame_times <= b + window_sec)
        if np.any(mask):
            beat_brightness.append(np.max(brightness[mask]))
        else:
            beat_brightness.append(0.0)

    # Anti-beats: midpoints between consecutive beats
    anti_brightness = []
    for i in range(len(beats) - 1):
        mid = (beats[i] + beats[i+1]) / 2.0
        mask = (frame_times >= mid - window_sec) & (frame_times <= mid + window_sec)
        if np.any(mask):
            anti_brightness.append(np.max(brightness[mask]))
        else:
            anti_brightness.append(0.0)

    return np.array(beat_brightness), np.array(anti_brightness)


def find_peak_offset(frame_times, brightness, beats, search_ms=100):
    """
    For each beat, find the time offset of the brightness peak relative to beat time.
    Negative = peak before beat (early), Positive = peak after beat (late).
    """
    search_sec = search_ms / 1000.0
    offsets = []
    for b in beats:
        mask = (frame_times >= b - search_sec) & (frame_times <= b + search_sec)
        if np.any(mask):
            local_times = frame_times[mask]
            local_bright = brightness[mask]
            peak_idx = np.argmax(local_bright)
            offset = local_times[peak_idx] - b
            offsets.append(offset)
    return np.array(offsets)


def analyze_track(wav_path, ann_path, layer, label):
    """Compare prop vs pred on one track."""
    audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    with open(ann_path) as f:
        beats = np.array(yaml.safe_load(f)[layer])

    duration = len(audio) / sr
    print(f"\n{'='*70}")
    print(f"  {label} — {len(beats)} taps, {duration:.1f}s")
    if len(beats) > 1:
        median_ibi = np.median(np.diff(beats))
        print(f"  Median tap tempo: {60/median_ibi:.0f} BPM ({median_ibi*1000:.0f}ms)")
    print(f"{'='*70}")

    results = {}

    for name, EffectClass in [('Proportional', AbsIntProportionalEffect),
                               ('Predictive', AbsIntPredictiveEffect)]:
        effect = EffectClass(num_leds=197, sample_rate=SR)
        ftimes, bright = simulate_effect(effect, audio, sr, beats)

        at_beats, at_anti = sample_brightness_at_beats(ftimes, bright, beats)
        offsets = find_peak_offset(ftimes, bright, beats)

        mean_at_beat = np.mean(at_beats)
        mean_at_anti = np.mean(at_anti) if len(at_anti) > 0 else 0
        ratio = mean_at_beat / mean_at_anti if mean_at_anti > 0 else float('inf')

        # What fraction of beats have brightness > 0.3 (visible)?
        visible = np.mean(at_beats > 0.3)
        # What fraction of beats have brightness > 0.5 (strong flash)?
        strong = np.mean(at_beats > 0.5)

        median_offset_ms = np.median(offsets) * 1000 if len(offsets) > 0 else 0
        std_offset_ms = np.std(offsets) * 1000 if len(offsets) > 0 else 0

        diag = effect.get_diagnostics()

        print(f"\n  [{name}]")
        print(f"    Brightness at beats:   mean={mean_at_beat:.3f}  "
              f"(>{0.3}: {visible:.0%}  >{0.5}: {strong:.0%})")
        print(f"    Brightness at anti:    mean={mean_at_anti:.3f}")
        print(f"    Beat/Anti ratio:       {ratio:.2f}x")
        print(f"    Peak offset from beat: {median_offset_ms:+.0f}ms median  "
              f"(±{std_offset_ms:.0f}ms)")
        if 'bpm' in diag:
            print(f"    Estimated BPM: {diag['bpm']}  "
                  f"(confidence: {diag['ac_conf']})")
            print(f"    Confirmed: {diag['confirmed']}  "
                  f"Predicted: {diag['predicted']}")

        results[name] = {
            'mean_at_beat': mean_at_beat,
            'mean_at_anti': mean_at_anti,
            'ratio': ratio,
            'visible': visible,
            'strong': strong,
            'offset_ms': median_offset_ms,
            'offset_std_ms': std_offset_ms,
            'diag': diag,
        }

    return label, results


def main():
    print("\n  Brightness at Beats — Proportional vs Predictive")
    print("  " + "=" * 50)

    all_results = []
    for wav_path, ann_path, layer, label in TRACKS:
        if not wav_path.exists():
            print(f"  SKIP: {label}")
            continue
        label, results = analyze_track(wav_path, ann_path, layer, label)
        all_results.append((label, results))

    # Summary table
    print(f"\n\n{'='*75}")
    print(f"  SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Track':<22s} | {'Proportional':^25s} | {'Predictive':^25s}")
    print(f"  {'':22s} | {'@Beat':>6s} {'Ratio':>6s} {'Vis%':>5s} {'Off':>5s}"
          f" | {'@Beat':>6s} {'Ratio':>6s} {'Vis%':>5s} {'Off':>5s}")
    print(f"  {'-'*75}")

    prop_beats = []
    pred_beats = []
    prop_ratios = []
    pred_ratios = []

    for label, results in all_results:
        p = results['Proportional']
        d = results['Predictive']
        short = label[:22]
        print(f"  {short:<22s} | {p['mean_at_beat']:>6.3f} {p['ratio']:>6.2f} "
              f"{p['visible']:>4.0%} {p['offset_ms']:>+4.0f}ms"
              f" | {d['mean_at_beat']:>6.3f} {d['ratio']:>6.2f} "
              f"{d['visible']:>4.0%} {d['offset_ms']:>+4.0f}ms")

        prop_beats.append(p['mean_at_beat'])
        pred_beats.append(d['mean_at_beat'])
        prop_ratios.append(p['ratio'])
        pred_ratios.append(d['ratio'])

    print(f"  {'-'*75}")
    print(f"  {'AVERAGE':<22s} | {np.mean(prop_beats):>6.3f} {np.mean(prop_ratios):>6.2f} "
          f"{'':>5s} {'':>5s}"
          f" | {np.mean(pred_beats):>6.3f} {np.mean(pred_ratios):>6.2f}")

    # Winner per metric
    print(f"\n  Winner by brightness at beats: ", end='')
    if np.mean(pred_beats) > np.mean(prop_beats):
        pct = (np.mean(pred_beats) / np.mean(prop_beats) - 1) * 100
        print(f"Predictive (+{pct:.0f}%)")
    else:
        pct = (np.mean(prop_beats) / np.mean(pred_beats) - 1) * 100
        print(f"Proportional (+{pct:.0f}%)")

    print(f"  Winner by beat/anti ratio:     ", end='')
    if np.mean(pred_ratios) > np.mean(prop_ratios):
        print(f"Predictive ({np.mean(pred_ratios):.2f}x vs {np.mean(prop_ratios):.2f}x)")
    else:
        print(f"Proportional ({np.mean(prop_ratios):.2f}x vs {np.mean(pred_ratios):.2f}x)")


if __name__ == '__main__':
    main()
