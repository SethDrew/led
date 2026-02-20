#!/usr/bin/env python3
"""
Check complex_beat_glue annotations for human error,
then compare BeatPredictor (absint) vs OnsetTempoTracker.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'effects'))

import numpy as np
import yaml
import librosa
from signals import (OverlapFrameAccumulator, AbsIntegral, BeatPredictor,
                     OnsetTempoTracker)

SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
SR = 44100


def check_annotations():
    """Check beat annotations for human error / outliers."""
    ann = SEGMENTS / 'complex_beat_glue.annotations.yaml'
    with open(ann) as f:
        data = yaml.safe_load(f)
    beats = np.array(data['beat'], dtype=float)

    print(f"\n  Annotation check: {len(beats)} beats")
    print(f"  Range: {beats[0]:.3f}s to {beats[-1]:.3f}s")

    ibi = np.diff(beats)
    median_ibi = np.median(ibi)
    mean_ibi = np.mean(ibi)
    std_ibi = np.std(ibi)
    bpm = 60.0 / median_ibi

    print(f"  Median IBI: {median_ibi*1000:.0f}ms ({bpm:.1f} BPM)")
    print(f"  Mean IBI:   {mean_ibi*1000:.0f}ms")
    print(f"  Std IBI:    {std_ibi*1000:.1f}ms")
    print(f"  Range:      [{np.min(ibi)*1000:.0f}, {np.max(ibi)*1000:.0f}]ms")

    # Flag outliers (> 2 std from median, or > 30% deviation)
    print(f"\n  --- Potential outliers (>25% deviation from median) ---")
    flagged = []
    for i, interval in enumerate(ibi):
        dev = abs(interval - median_ibi) / median_ibi
        if dev > 0.25:
            flagged.append(i)
            before = beats[i]
            after = beats[i + 1]
            print(f"  Beat {i+1:3d}→{i+2:3d}: {before:.3f}→{after:.3f}  "
                  f"IBI={interval*1000:.0f}ms  "
                  f"({dev*100:+.0f}% from {median_ibi*1000:.0f}ms median)")

    if not flagged:
        print("  None found!")

    # Show IBI histogram
    print(f"\n  --- IBI distribution ---")
    bins = [300, 350, 400, 420, 440, 460, 480, 500, 520, 540, 560, 600, 700]
    for j in range(len(bins) - 1):
        count = np.sum((ibi * 1000 >= bins[j]) & (ibi * 1000 < bins[j+1]))
        if count > 0:
            bar = '#' * count
            print(f"  {bins[j]:4d}-{bins[j+1]:4d}ms: {count:3d} {bar}")

    # Check for monotonicity
    non_mono = np.where(np.diff(beats) <= 0)[0]
    if len(non_mono) > 0:
        print(f"\n  WARNING: Non-monotonic beats at indices: {non_mono}")

    return beats


def compare_detectors(beats):
    """Run both detectors on the track, compare to ground truth."""
    wav = SEGMENTS / 'complex_beat_glue.wav'
    if not wav.exists():
        print(f"  SKIP: no wav file")
        return

    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    true_bpm = 60.0 / np.median(np.diff(beats))

    print(f"\n  {'='*65}")
    print(f"  Detector comparison — complex_beat_glue")
    print(f"  True BPM: {true_bpm:.1f}  ({len(beats)} beats in {audio.shape[0]/SR:.1f}s)")
    print(f"  {'='*65}")

    # --- BeatPredictor (absint) ---
    accum1 = OverlapFrameAccumulator()
    absint = AbsIntegral(sample_rate=SR)
    predictor = BeatPredictor(rms_fps=absint.rms_fps)

    bp_checkpoints = {}
    bp_checkpoint_times = {5, 10, 15, 30, 45}
    bp_confirmed = []
    bp_predicted = []

    for frame in accum1.feed(audio):
        normalized = absint.update(frame)
        events = predictor.feed(absint.raw, normalized, absint.time_acc)
        t = absint.time_acc

        for cp in list(bp_checkpoint_times):
            if t >= cp:
                bp_checkpoints[cp] = predictor.bpm
                bp_checkpoint_times.discard(cp)

        for e in events:
            if e['type'] == 'confirmed':
                bp_confirmed.append(t)
            elif e['type'] == 'predicted':
                bp_predicted.append(t)

    bp_bpm = predictor.bpm
    bp_err = abs(bp_bpm - true_bpm) / true_bpm * 100 if true_bpm else 100
    cp_str = ', '.join(f"{t}s:{bp_checkpoints.get(t, 0):.0f}" for t in [5, 10, 15, 30, 45])

    print(f"\n  BeatPredictor (absint):")
    print(f"    Final BPM: {bp_bpm:.1f}  (err: {bp_err:.1f}%)")
    print(f"    BPM over time: {cp_str}")
    print(f"    Confirmed beats: {len(bp_confirmed)}")
    print(f"    Predicted beats: {len(bp_predicted)}")
    print(f"    Confidence: {predictor.confidence:.3f}")

    # Score confirmed+predicted vs ground truth
    all_bp_beats = sorted(bp_confirmed + bp_predicted)
    if all_bp_beats:
        p, r, f1 = score(np.array(all_bp_beats), beats, tolerance=0.1)
        print(f"    Beat F1 (100ms): P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # --- OnsetTempoTracker (no prior) ---
    accum2 = OverlapFrameAccumulator()
    tracker = OnsetTempoTracker(sample_rate=SR, prior_sigma=99999)

    ot_checkpoints = {}
    ot_checkpoint_times = {5, 10, 15, 30, 45}

    for frame in accum2.feed(audio):
        tracker.feed_frame(frame)
        t = tracker.time_acc

        for cp in list(ot_checkpoint_times):
            if t >= cp:
                ot_checkpoints[cp] = tracker.bpm
                ot_checkpoint_times.discard(cp)

    ot_bpm = tracker.bpm
    ot_err = abs(ot_bpm - true_bpm) / true_bpm * 100 if true_bpm else 100

    # Check octave
    octave = ""
    if ot_err > 20:
        for mult, label in [(2, "2x"), (0.5, "0.5x")]:
            if abs(ot_bpm - true_bpm * mult) / (true_bpm * mult) < 0.05:
                octave = f" ({label})"

    cp_str = ', '.join(f"{t}s:{ot_checkpoints.get(t, 0):.0f}" for t in [5, 10, 15, 30, 45])

    print(f"\n  OnsetTempoTracker (no prior):")
    print(f"    Final BPM: {ot_bpm:.1f}  (err: {ot_err:.1f}%{octave})")
    print(f"    BPM over time: {cp_str}")
    print(f"    Confidence: {tracker.confidence:.3f}")

    # --- OnsetTempoTracker (rap prior for reference) ---
    accum3 = OverlapFrameAccumulator()
    tracker_prior = OnsetTempoTracker(sample_rate=SR,
                                       prior_center=100, prior_sigma=40)

    for frame in accum3.feed(audio):
        tracker_prior.feed_frame(frame)

    op_bpm = tracker_prior.bpm
    op_err = abs(op_bpm - true_bpm) / true_bpm * 100 if true_bpm else 100
    print(f"\n  OnsetTempoTracker (prior=100±40, for reference):")
    print(f"    Final BPM: {op_bpm:.1f}  (err: {op_err:.1f}%)")


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
    p = tp / len(detected) if len(detected) else 0
    r = tp / len(ground_truth) if len(ground_truth) else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def main():
    print("\n  complex_beat_glue — Annotation Check & Detector Comparison")
    print("  " + "=" * 55)

    beats = check_annotations()
    compare_detectors(beats)


if __name__ == '__main__':
    main()
