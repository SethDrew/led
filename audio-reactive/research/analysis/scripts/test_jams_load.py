#!/usr/bin/env python3
"""
Test script to demonstrate loading Harmonix Set JAMS annotations.

This shows how to extract beat times, downbeats, tempo, and segments
from the JAMS format for validation purposes.
"""

import jams
from pathlib import Path

# Paths
DATASET_DIR = Path("/Users/KO16K39/Documents/led/audio-reactive/research/datasets/harmonix/dataset")
JAMS_DIR = DATASET_DIR / "jams"

# Test with a key track: Metallica - ...and Justice for All
test_file = "0010_andjusticeforall.jams"
test_path = JAMS_DIR / test_file


def explore_jams_file(jams_path):
    """Load and explore a JAMS annotation file."""
    print(f"Loading: {jams_path.name}")
    print("=" * 80)
    print()

    # Load JAMS file
    jam = jams.load(str(jams_path))

    # File metadata
    print("METADATA:")
    print(f"  Duration: {jam.file_metadata.duration:.2f} seconds")
    if jam.file_metadata.title:
        print(f"  Title: {jam.file_metadata.title}")
    if jam.file_metadata.artist:
        print(f"  Artist: {jam.file_metadata.artist}")
    print()

    # List all annotation types
    print("ANNOTATION TYPES AVAILABLE:")
    for ann in jam.annotations:
        print(f"  - {ann.namespace} ({len(ann.data)} observations)")
    print()

    # Extract beats
    print("BEAT ANNOTATIONS:")
    beat_ann = jam.search(namespace='beat')[0]
    beat_times = [obs.time for obs in beat_ann.data]
    print(f"  Total beats: {len(beat_times)}")
    print(f"  First 10 beat times: {beat_times[:10]}")
    print()

    # Compute tempo from beats
    import numpy as np
    if len(beat_times) > 1:
        ibi = np.diff(beat_times)
        avg_ibi = np.mean(ibi)
        avg_bpm = 60.0 / avg_ibi
        print(f"  Average BPM from beats: {avg_bpm:.1f}")
        print(f"  IBI std dev: {np.std(ibi):.4f} (lower = more stable)")
    print()

    # Extract segments
    print("SEGMENT ANNOTATIONS:")
    segment_ann = jam.search(namespace='segment_open')[0]
    for i, obs in enumerate(segment_ann.data[:10]):
        print(f"  {obs.time:8.3f}s - {obs.time + obs.duration:8.3f}s: {obs.value}")
    if len(segment_ann.data) > 10:
        print(f"  ... ({len(segment_ann.data) - 10} more segments)")
    print()

    # Check for tempo annotation
    print("TEMPO ANNOTATION:")
    tempo_ann = jam.search(namespace='tempo')
    if tempo_ann:
        for obs in tempo_ann[0].data:
            print(f"  {obs.time:.3f}s: {obs.value:.1f} BPM (confidence: {obs.confidence:.2f})")
    else:
        print("  No tempo annotation found")
    print()

    return jam


def compare_multiple_tracks():
    """Compare annotations across different tracks."""
    print("\n" + "=" * 80)
    print("COMPARING MULTIPLE TRACKS")
    print("=" * 80)
    print()

    tracks = [
        ("0010_andjusticeforall.jams", "Metallica - Justice (6/4, prog)"),
        ("0161_limelight.jams", "Rush - Limelight (mixed meter)"),
        ("0012_aroundtheworld.jams", "Daft Punk - Around The World (electronic)"),
    ]

    for jams_file, description in tracks:
        jams_path = JAMS_DIR / jams_file
        if not jams_path.exists():
            print(f"⚠ File not found: {jams_file}")
            continue

        jam = jams.load(str(jams_path))
        beat_ann = jam.search(namespace='beat')[0]
        beat_times = [obs.time for obs in beat_ann.data]

        # Compute tempo
        import numpy as np
        ibi = np.diff(beat_times)
        avg_bpm = 60.0 / np.mean(ibi)
        tempo_stability = np.std(ibi) / np.mean(ibi)

        print(f"{description:50s}")
        print(f"  Beats: {len(beat_times):4d}  Avg BPM: {avg_bpm:6.1f}  Stability: {tempo_stability:.4f}")
        print()


if __name__ == "__main__":
    # Test loading a single file
    if test_path.exists():
        jam = explore_jams_file(test_path)
        print("\n✓ Successfully loaded JAMS file!")
        print()
        print("To use in validation:")
        print("  1. Load JAMS file with jams.load()")
        print("  2. Extract beat times: jam.search(namespace='beat')[0].data")
        print("  3. Run beat tracking on audio")
        print("  4. Compare detected beats to ground truth using mir_eval.beat.f_measure()")
        print()
    else:
        print(f"⚠ Test file not found: {test_path}")
        print("Make sure Harmonix Set is cloned to the expected location.")
        print()

    # Compare multiple tracks
    compare_multiple_tracks()
