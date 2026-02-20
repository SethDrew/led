#!/usr/bin/env python3
"""
Test OnsetTempoTracker on the rap tracks to verify the implementation
matches the offline analysis results.
"""

import sys
from pathlib import Path

# Add effects dir so we can import signals
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'effects'))

import numpy as np
import librosa
import yaml
from signals import OverlapFrameAccumulator, OnsetTempoTracker

SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
HARMONIX = SEGMENTS / 'harmonix'
SR = 44100


def load_track(name):
    wav = HARMONIX / f'{name}.wav'
    ann = HARMONIX / f'{name}.annotations.yaml'
    if not wav.exists():
        return None, None, None
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    with open(ann) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            f.seek(0)
            data = yaml.unsafe_load(f)
    beats = np.array(data['harmonix_beats'], dtype=float)
    true_bpm = 60.0 / np.median(np.diff(beats)) if len(beats) > 1 else 0
    return audio, beats, true_bpm


def test_track(name, title):
    audio, beats, true_bpm = load_track(name)
    if audio is None:
        print(f"  SKIP: {title}")
        return

    accum = OverlapFrameAccumulator()
    tracker = OnsetTempoTracker(sample_rate=SR)

    # Track BPM over time
    checkpoints = {}
    checkpoint_times = {5, 10, 15, 30, 45}

    for frame in accum.feed(audio):
        tracker.feed_frame(frame)
        t = tracker.time_acc
        for cp in list(checkpoint_times):
            if t >= cp:
                checkpoints[cp] = tracker.bpm
                checkpoint_times.discard(cp)

    final_bpm = tracker.bpm
    err = abs(final_bpm - true_bpm) / true_bpm * 100 if true_bpm else 100
    status = "OK" if err < 5 else "CLOSE" if err < 15 else "FAIL"

    cp_str = ', '.join(f"{t}s:{checkpoints.get(t, 0):.0f}" for t in [5, 10, 15, 30, 45])
    print(f"  [{status:5s}] {title:35s} true={true_bpm:.0f}  final={final_bpm:.0f}  "
          f"err={err:.1f}%  conf={tracker.confidence:.3f}")
    print(f"           BPM over time: {cp_str}")


def test_all(prior_center, prior_sigma, label):
    tracks = [
        ('0026_blackandyellow_60s', 'Black and Yellow', 'harmonix_beats'),
        ('0141_indaclub_60s', 'In Da Club', 'harmonix_beats'),
        ('0439_lovethewayyoulie_60s', 'Love The Way You Lie', 'harmonix_beats'),
        ('0607_cantholdus_60s', "Can't Hold Us", 'harmonix_beats'),
        ('aroundtheworld', 'Around The World', 'sethbeat'),
        ('constantmotion', 'Constant Motion', 'sethbeat'),
        ('floods', 'Floods', 'sethbeat'),
        ('limelight', 'Limelight', 'sethbeat'),
        ('blackened', 'Blackened', 'sethbeat'),
    ]

    print(f"\n  {label}")
    print(f"  prior_center={prior_center}, prior_sigma={prior_sigma}")
    print("  " + "=" * 70)

    for name, title, layer in tracks:
        wav = HARMONIX / f'{name}.wav'
        ann = HARMONIX / f'{name}.annotations.yaml'
        if not wav.exists():
            print(f"  SKIP: {title}")
            continue
        audio, _ = librosa.load(str(wav), sr=SR, mono=True)
        with open(ann) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                f.seek(0)
                data = yaml.unsafe_load(f)
        beats = np.array(data[layer], dtype=float)
        true_bpm = 60.0 / np.median(np.diff(beats)) if len(beats) > 1 else 0

        accum = OverlapFrameAccumulator()
        tracker = OnsetTempoTracker(sample_rate=SR,
                                     prior_center=prior_center,
                                     prior_sigma=prior_sigma)
        checkpoints = {}
        checkpoint_times = {5, 10, 15, 30, 45}
        for frame in accum.feed(audio):
            tracker.feed_frame(frame)
            t = tracker.time_acc
            for cp in list(checkpoint_times):
                if t >= cp:
                    checkpoints[cp] = tracker.bpm
                    checkpoint_times.discard(cp)

        final_bpm = tracker.bpm
        err = abs(final_bpm - true_bpm) / true_bpm * 100 if true_bpm else 100

        # Check if it's a clean octave error
        octave_note = ""
        if err > 20:
            for mult, label_o in [(2, "2x"), (0.5, "0.5x")]:
                if abs(final_bpm - true_bpm * mult) / (true_bpm * mult) < 0.05:
                    octave_note = f" ({label_o})"
                    break

        status = "OK" if err < 5 else "CLOSE" if err < 15 else "FAIL"
        cp_str = ', '.join(f"{t}s:{checkpoints.get(t, 0):.0f}"
                          for t in [5, 10, 15, 30])
        print(f"  [{status:5s}] {title:25s} true={true_bpm:5.0f}  "
              f"est={final_bpm:5.0f}  err={err:5.1f}%{octave_note:6s}  "
              f"[{cp_str}]")


def main():
    test_all(100, 40, "WITH RAP PRIOR")
    test_all(100, 99999, "NO PRIOR (flat)")
    test_all(120, 99999, "NO PRIOR (flat, center irrelevant)")


if __name__ == '__main__':
    main()
