#!/usr/bin/env python3
"""
Validate Onset Detector Against Test Audio Files

Compares BassFluxDetector vs OnsetDetector offline on our test audio segments:
- Opiate Intro.wav (rock with kick drums)
- electronic_beat.wav (electronic with continuous bass)
- ambient.wav (ambient electronic)

For Opiate, we also score against consistent-beat ground truth annotations.

Usage:
    python validate_onset_detector.py
"""

import sys
import os
import numpy as np
import yaml
import soundfile as sf

# Add tools directory to path for detector imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tools'))

from realtime_onset_led import (
    BassFluxDetector,
    OnsetDetector,
    SAMPLE_RATE,
    CHUNK_SIZE,
    N_FFT
)

# Audio segments directory
SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), '../../audio-segments')

# Test files
TEST_FILES = [
    'opiate_intro.wav',
    'electronic_beat.wav',
    'ambient.wav'
]

# Annotation file for Opiate
OPIATE_ANNOTATIONS = 'opiate_intro.annotations.yaml'


def load_annotations(audio_file):
    """Load ground truth annotations if available."""
    base_name = os.path.splitext(audio_file)[0]
    annotations_path = os.path.join(SEGMENTS_DIR, f"{base_name}.annotations.yaml")

    if not os.path.exists(annotations_path):
        return None

    with open(annotations_path, 'r') as f:
        return yaml.safe_load(f)


def compute_f1_score(detected_times, ground_truth_times, tolerance=0.07):
    """
    Compute precision, recall, F1 for detected beats vs ground truth.

    Args:
        detected_times: List of detected beat times (seconds)
        ground_truth_times: List of ground truth beat times (seconds)
        tolerance: Time window for matching (seconds)
    """
    if len(detected_times) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(ground_truth_times)

    if len(ground_truth_times) == 0:
        return 0.0, 0.0, 0.0, 0, len(detected_times), 0

    # Convert to numpy arrays
    detected = np.array(sorted(detected_times))
    ground_truth = np.array(sorted(ground_truth_times))

    # For each ground truth beat, check if there's a detection within tolerance
    true_positives = 0
    matched_detections = set()

    for gt_time in ground_truth:
        # Find closest detection
        diffs = np.abs(detected - gt_time)
        closest_idx = np.argmin(diffs)
        closest_diff = diffs[closest_idx]

        if closest_diff <= tolerance and closest_idx not in matched_detections:
            true_positives += 1
            matched_detections.add(closest_idx)

    false_positives = len(detected) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / len(detected) if len(detected) > 0 else 0.0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, true_positives, false_positives, false_negatives


def process_audio_file(audio_path, bass_detector, onset_detector):
    """Process audio file with both detectors. Returns (bass_beats, onset_beats)."""
    # Load audio
    audio, sr = sf.read(audio_path)

    # Mix to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed (simple decimation/interpolation)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr))

    # Reset detector state
    bass_detector.prev_spectrum = None
    bass_detector.flux_history = []
    bass_detector.last_beat_time = -999.0  # Allow first beat immediately
    bass_detector.beat_count = 0
    bass_detector.audio_buffer = np.zeros(bass_detector.n_fft)

    onset_detector.prev_mel_spectrum = None
    onset_detector.onset_history = []
    onset_detector.last_beat_time = -999.0  # Allow first beat immediately
    onset_detector.beat_count = 0
    onset_detector.audio_buffer = np.zeros(onset_detector.n_fft)

    # Process in chunks
    bass_beats = []
    onset_beats = []
    bass_values = []
    onset_values = []

    num_chunks = len(audio) // CHUNK_SIZE
    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = audio[start:end]

        # Current time (center of chunk for better accuracy)
        chunk_time = (i + 0.5) * CHUNK_SIZE / SAMPLE_RATE

        # Process with both detectors (use_realtime=False for offline processing)
        bass_beat, bass_val, bass_thresh = bass_detector.process_chunk(chunk, use_realtime=False)
        onset_beat, onset_val, onset_thresh = onset_detector.process_chunk(chunk, use_realtime=False)

        # Record beat times
        if bass_beat:
            bass_beats.append(chunk_time)

        if onset_beat:
            onset_beats.append(chunk_time)

        bass_values.append((chunk_time, bass_val, bass_thresh))
        onset_values.append((chunk_time, onset_val, onset_thresh))

    return bass_beats, onset_beats


def analyze_overlap(bass_beats, onset_beats, tolerance=0.07):
    """Compute overlap statistics between two beat lists."""
    if len(bass_beats) == 0 or len(onset_beats) == 0:
        return 0, 0, 0

    bass_only = 0
    onset_only = 0
    both = 0

    matched_onset = set()

    for bass_time in bass_beats:
        # Check if there's a matching onset beat
        diffs = np.abs(np.array(onset_beats) - bass_time)
        if len(diffs) > 0:
            closest_idx = np.argmin(diffs)
            if diffs[closest_idx] <= tolerance:
                both += 1
                matched_onset.add(closest_idx)
            else:
                bass_only += 1
        else:
            bass_only += 1

    onset_only = len(onset_beats) - len(matched_onset)

    return bass_only, onset_only, both


def main():
    print("\n" + "="*70)
    print("  Onset Detector Validation")
    print("="*70)

    # Initialize detectors
    bass_detector = BassFluxDetector()
    onset_detector = OnsetDetector()

    results = []

    for audio_file in TEST_FILES:
        audio_path = os.path.join(SEGMENTS_DIR, audio_file)

        if not os.path.exists(audio_path):
            print(f"\n  Warning: {audio_file} not found, skipping...")
            continue

        print(f"\n  Processing: {audio_file}")
        print("  " + "-"*60)

        # Process audio
        bass_beats, onset_beats = process_audio_file(audio_path, bass_detector, onset_detector)

        print(f"    Bass flux detections:  {len(bass_beats)}")
        print(f"    Onset detections:      {len(onset_beats)}")

        # Analyze overlap
        bass_only, onset_only, both = analyze_overlap(bass_beats, onset_beats)
        print(f"    Overlap:               Bass-only={bass_only}, Onset-only={onset_only}, Both={both}")

        # Load ground truth if available
        annotations = load_annotations(audio_file)

        result = {
            'file': audio_file,
            'bass_count': len(bass_beats),
            'onset_count': len(onset_beats),
            'bass_only': bass_only,
            'onset_only': onset_only,
            'both': both
        }

        if annotations and 'consistent-beat' in annotations:
            ground_truth = annotations['consistent-beat']
            print(f"    Ground truth beats:    {len(ground_truth)}")

            # Score bass detector
            bass_prec, bass_rec, bass_f1, bass_tp, bass_fp, bass_fn = compute_f1_score(
                bass_beats, ground_truth
            )

            # Score onset detector
            onset_prec, onset_rec, onset_f1, onset_tp, onset_fp, onset_fn = compute_f1_score(
                onset_beats, ground_truth
            )

            print(f"\n    BASS FLUX vs Ground Truth:")
            print(f"      Precision: {bass_prec:.3f}  Recall: {bass_rec:.3f}  F1: {bass_f1:.3f}")
            print(f"      TP={bass_tp}, FP={bass_fp}, FN={bass_fn}")

            print(f"\n    ONSET vs Ground Truth:")
            print(f"      Precision: {onset_prec:.3f}  Recall: {onset_rec:.3f}  F1: {onset_f1:.3f}")
            print(f"      TP={onset_tp}, FP={onset_fp}, FN={onset_fn}")

            result['ground_truth_count'] = len(ground_truth)
            result['bass_f1'] = bass_f1
            result['bass_precision'] = bass_prec
            result['bass_recall'] = bass_rec
            result['onset_f1'] = onset_f1
            result['onset_precision'] = onset_prec
            result['onset_recall'] = onset_rec

        results.append(result)

    # Summary table
    print("\n\n" + "="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    print(f"  {'File':<20s} {'Bass':<8s} {'Onset':<8s} {'Overlap':<15s} {'F1 Score':<20s}")
    print("  " + "-"*68)

    for r in results:
        overlap_str = f"{r['both']}/{r['bass_count']}"

        if 'bass_f1' in r:
            f1_str = f"B:{r['bass_f1']:.3f} O:{r['onset_f1']:.3f}"
        else:
            f1_str = "-"

        print(f"  {r['file']:<20s} {r['bass_count']:<8d} {r['onset_count']:<8d} "
              f"{overlap_str:<15s} {f1_str:<20s}")

    # Key insights
    print("\n" + "="*70)
    print("  KEY INSIGHTS")
    print("="*70)

    for r in results:
        print(f"\n  {r['file']}:")

        if r['bass_count'] == 0:
            print("    - Bass detector found NO beats (fails on continuous sub-bass)")
        elif r['onset_count'] > r['bass_count'] * 2:
            print(f"    - Onset detector found {r['onset_count']/r['bass_count']:.1f}x more beats (captures hi-hats/snares)")

        if 'onset_f1' in r and 'bass_f1' in r:
            if r['onset_f1'] > r['bass_f1'] * 1.5:
                print(f"    - Onset detector significantly better (F1: {r['onset_f1']:.3f} vs {r['bass_f1']:.3f})")
            elif r['bass_f1'] > r['onset_f1'] * 1.5:
                print(f"    - Bass flux significantly better (F1: {r['bass_f1']:.3f} vs {r['onset_f1']:.3f})")
            else:
                print(f"    - Similar performance (F1: Bass={r['bass_f1']:.3f}, Onset={r['onset_f1']:.3f})")

        if r['both'] > 0 and r['bass_count'] > 0:
            overlap_pct = r['both'] / r['bass_count'] * 100
            print(f"    - {overlap_pct:.0f}% of bass beats also detected by onset")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
