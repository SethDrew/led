#!/usr/bin/env python3
"""
Evaluate librosa onset strength against Harmonix beat annotations.

For each track:
1. Compute onset strength envelope + detect onset peaks
2. Compare onset peaks to Harmonix beat times
3. Report precision, recall, F1 at various tolerance windows

Also checks: does onset strength *value* at beat times differ from non-beat times?
(i.e., even if peak detection fails, is the raw signal informative?)
"""

import os
import sys
import numpy as np
import yaml
import librosa

HARMONIX_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'audio-segments', 'harmonix')

# Tolerance windows to test (seconds)
TOLERANCES = [0.025, 0.050, 0.070, 0.100]


def load_track(name):
    """Load audio and beat annotations for a harmonix track."""
    wav_path = os.path.join(HARMONIX_DIR, f'{name}.wav')
    ann_path = os.path.join(HARMONIX_DIR, f'{name}.annotations.yaml')

    y, sr = librosa.load(wav_path, sr=None, mono=True)

    with open(ann_path) as f:
        ann = yaml.safe_load(f)

    beats = np.array(ann.get('beats', []))
    return y, sr, beats


def evaluate_onsets_vs_beats(onset_times, beat_times, tolerance):
    """Precision/recall/F1 of onset peaks against beat times."""
    if len(onset_times) == 0 or len(beat_times) == 0:
        return 0, 0, 0

    # For each onset, find closest beat
    tp_onset = 0
    for ot in onset_times:
        dists = np.abs(beat_times - ot)
        if np.min(dists) <= tolerance:
            tp_onset += 1

    # For each beat, find closest onset
    tp_beat = 0
    for bt in beat_times:
        dists = np.abs(onset_times - bt)
        if np.min(dists) <= tolerance:
            tp_beat += 1

    precision = tp_onset / len(onset_times) if len(onset_times) > 0 else 0
    recall = tp_beat / len(beat_times) if len(beat_times) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def onset_strength_at_beats(onset_env, times, beat_times, sr, hop_length):
    """Compare onset strength values at beat vs non-beat frames."""
    # Get onset strength at each beat time
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    beat_frames = beat_frames[beat_frames < len(onset_env)]

    beat_strengths = onset_env[beat_frames]

    # Non-beat frames: all frames NOT within 1 frame of a beat
    all_frames = np.arange(len(onset_env))
    beat_set = set()
    for bf in beat_frames:
        for offset in range(-1, 2):
            if 0 <= bf + offset < len(onset_env):
                beat_set.add(bf + offset)
    non_beat_mask = ~np.isin(all_frames, list(beat_set))
    non_beat_strengths = onset_env[non_beat_mask]

    return {
        'beat_mean': float(np.mean(beat_strengths)),
        'beat_median': float(np.median(beat_strengths)),
        'nonbeat_mean': float(np.mean(non_beat_strengths)),
        'nonbeat_median': float(np.median(non_beat_strengths)),
        'ratio_mean': float(np.mean(beat_strengths) / np.mean(non_beat_strengths)) if np.mean(non_beat_strengths) > 0 else 0,
        'ratio_median': float(np.median(beat_strengths) / np.median(non_beat_strengths)) if np.median(non_beat_strengths) > 0 else 0,
    }


def main():
    # Find all tracks with both wav and annotations
    tracks = []
    for f in sorted(os.listdir(HARMONIX_DIR)):
        if f.endswith('.wav') and not f.endswith('_5s.wav') and not f.endswith('_10s.wav'):
            name = f.replace('.wav', '')
            ann_path = os.path.join(HARMONIX_DIR, f'{name}.annotations.yaml')
            if os.path.exists(ann_path):
                tracks.append(name)

    print(f"Found {len(tracks)} tracks: {', '.join(tracks)}\n")

    hop_length = 512
    all_results = []

    for name in tracks:
        print(f"{'=' * 60}")
        print(f"Track: {name}")
        print(f"{'=' * 60}")

        y, sr, beats = load_track(name)
        duration = len(y) / sr

        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

        # Detect onset peaks
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        print(f"  Duration: {duration:.1f}s, Beats: {len(beats)}, Onsets detected: {len(onset_times)}")
        print(f"  Beat density: {len(beats)/duration:.1f}/s, Onset density: {len(onset_times)/duration:.1f}/s")
        print()

        # Peak detection evaluation
        print("  Peak detection vs beats:")
        track_f1s = {}
        for tol in TOLERANCES:
            p, r, f1 = evaluate_onsets_vs_beats(onset_times, beats, tol)
            print(f"    tol={tol*1000:.0f}ms: P={p:.3f} R={r:.3f} F1={f1:.3f}")
            track_f1s[tol] = f1
        print()

        # Strength value analysis
        strength_stats = onset_strength_at_beats(onset_env, times, beats, sr, hop_length)
        print(f"  Onset strength at beats vs non-beats:")
        print(f"    Beat mean:    {strength_stats['beat_mean']:.3f}")
        print(f"    NonBeat mean: {strength_stats['nonbeat_mean']:.3f}")
        print(f"    Ratio (mean): {strength_stats['ratio_mean']:.2f}x")
        print(f"    Beat median:    {strength_stats['beat_median']:.3f}")
        print(f"    NonBeat median: {strength_stats['nonbeat_median']:.3f}")
        print(f"    Ratio (median): {strength_stats['ratio_median']:.2f}x")
        print()

        all_results.append({
            'name': name,
            'n_beats': len(beats),
            'n_onsets': len(onset_times),
            'duration': duration,
            'f1_50ms': track_f1s[0.050],
            'f1_70ms': track_f1s[0.070],
            'strength_ratio': strength_stats['ratio_mean'],
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Track':<20} {'Beats':>5} {'Onsets':>6} {'F1@50ms':>8} {'F1@70ms':>8} {'Str.Ratio':>10}")
    print("-" * 60)

    f1_50_vals = []
    f1_70_vals = []
    ratio_vals = []

    for r in all_results:
        print(f"{r['name']:<20} {r['n_beats']:>5} {r['n_onsets']:>6} {r['f1_50ms']:>8.3f} {r['f1_70ms']:>8.3f} {r['strength_ratio']:>10.2f}x")
        f1_50_vals.append(r['f1_50ms'])
        f1_70_vals.append(r['f1_70ms'])
        ratio_vals.append(r['strength_ratio'])

    print("-" * 60)
    print(f"{'MEAN':<20} {'':>5} {'':>6} {np.mean(f1_50_vals):>8.3f} {np.mean(f1_70_vals):>8.3f} {np.mean(ratio_vals):>10.2f}x")
    print(f"{'STD':<20} {'':>5} {'':>6} {np.std(f1_50_vals):>8.3f} {np.std(f1_70_vals):>8.3f} {np.std(ratio_vals):>10.2f}x")

    print(f"\nInterpretation:")
    mean_f1 = np.mean(f1_70_vals)
    mean_ratio = np.mean(ratio_vals)
    if mean_f1 > 0.6:
        print(f"  Peak detection: GOOD (mean F1@70ms = {mean_f1:.3f})")
    elif mean_f1 > 0.4:
        print(f"  Peak detection: MODERATE (mean F1@70ms = {mean_f1:.3f})")
    else:
        print(f"  Peak detection: POOR (mean F1@70ms = {mean_f1:.3f})")

    if mean_ratio > 2.0:
        print(f"  Strength signal: STRONG discriminator (beats are {mean_ratio:.1f}x stronger)")
    elif mean_ratio > 1.5:
        print(f"  Strength signal: MODERATE discriminator (beats are {mean_ratio:.1f}x stronger)")
    else:
        print(f"  Strength signal: WEAK discriminator (beats are only {mean_ratio:.1f}x stronger)")


if __name__ == '__main__':
    main()
