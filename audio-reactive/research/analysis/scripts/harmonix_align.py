#!/usr/bin/env python3
"""
Find the time offset between our YouTube-downloaded audio and
Harmonix beat annotations using cross-correlation.

Method:
  1. Compute onset strength envelope from our audio
  2. Build a beat pulse train from the annotation timestamps
  3. Cross-correlate them
  4. The lag at peak correlation = the offset to apply to annotations

Usage:
    python harmonix_align.py                    # all 4 rap tracks
    python harmonix_align.py 0141_indaclub      # single track
"""

import sys
import os
import numpy as np
import librosa

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SEGMENTS_DIR = os.path.join(_REPO, 'research', 'audio-segments')
BEATS_DIR = os.path.join(_REPO, 'research', 'datasets',
                         'harmonix', 'dataset', 'beats_and_downbeats')

TRACKS = {
    '0141_indaclub': 'harmonix/0141_indaclub.wav',
    '0026_blackandyellow': 'harmonix/0026_blackandyellow.wav',
    '0439_lovethewayyoulie': 'harmonix/0439_lovethewayyoulie.wav',
    '0607_cantholdus': 'harmonix/0607_cantholdus.wav',
}

SR = 22050
HOP = 512


def load_beat_times(harmonix_id):
    """Load beat times from Harmonix annotation file."""
    path = os.path.join(BEATS_DIR, f'{harmonix_id}.txt')
    times = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                times.append(float(parts[0]))
    return np.array(times)


def find_offset(harmonix_id, wav_filename, search_range_sec=5.0):
    """Find the time offset between audio and annotations.

    Returns offset in seconds: annotation_time + offset = audio_time
    (positive offset means annotations are early / audio has extra leading silence)
    """
    # Load audio
    wav_path = os.path.join(SEGMENTS_DIR, wav_filename)
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    duration = len(y) / sr

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=HOP)

    # Load beat annotations
    beat_times = load_beat_times(harmonix_id)
    beat_times = beat_times[beat_times < duration]

    # Build beat pulse train at same resolution as onset envelope
    beat_pulse = np.zeros_like(onset_env)
    for bt in beat_times:
        idx = int(round(bt * sr / HOP))
        if 0 <= idx < len(beat_pulse):
            beat_pulse[idx] = 1.0

    # Cross-correlate within search range
    search_frames = int(search_range_sec * sr / HOP)
    # Normalize both signals
    onset_norm = onset_env - np.mean(onset_env)
    beat_norm = beat_pulse - np.mean(beat_pulse)

    # Full cross-correlation
    corr = np.correlate(onset_norm, beat_norm, mode='full')
    # The zero-lag is at index len(beat_norm) - 1
    zero_lag = len(beat_norm) - 1

    # Search within ±search_range
    lo = max(0, zero_lag - search_frames)
    hi = min(len(corr), zero_lag + search_frames + 1)
    search_corr = corr[lo:hi]
    best_idx = np.argmax(search_corr) + lo

    offset_frames = best_idx - zero_lag
    offset_sec = offset_frames * HOP / sr

    # Confidence: ratio of peak to mean correlation in search window
    peak_val = corr[best_idx]
    mean_val = np.mean(np.abs(search_corr))
    confidence = peak_val / mean_val if mean_val > 0 else 0

    # Validate: compute beat alignment score at this offset
    shifted_beats = beat_times + offset_sec
    # For each shifted beat, find nearest onset peak
    onset_peaks = onset_times[librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=5
    )]

    if len(onset_peaks) > 0 and len(shifted_beats) > 0:
        dists = []
        for bt in shifted_beats:
            nearest = onset_peaks[np.argmin(np.abs(onset_peaks - bt))]
            dists.append(abs(nearest - bt))
        median_dist = np.median(dists)
        pct_within_50ms = np.mean(np.array(dists) < 0.050) * 100
    else:
        median_dist = float('inf')
        pct_within_50ms = 0

    return {
        'offset_sec': offset_sec,
        'confidence': confidence,
        'median_beat_error_ms': median_dist * 1000,
        'pct_within_50ms': pct_within_50ms,
        'n_beats': len(beat_times),
    }


def main():
    if len(sys.argv) > 1:
        track_id = sys.argv[1]
        if track_id not in TRACKS:
            print(f"Unknown track: {track_id}")
            print(f"Available: {', '.join(TRACKS.keys())}")
            sys.exit(1)
        tracks = {track_id: TRACKS[track_id]}
    else:
        tracks = TRACKS

    print(f"{'Track':<30} {'Offset':>8} {'Conf':>6} {'Med err':>8} {'<50ms':>6} {'Beats':>6}")
    print('-' * 70)

    for harmonix_id, wav_file in tracks.items():
        result = find_offset(harmonix_id, wav_file)
        sign = '+' if result['offset_sec'] >= 0 else ''
        print(f"{harmonix_id:<30} {sign}{result['offset_sec']:>7.3f}s "
              f"{result['confidence']:>5.1f}x "
              f"{result['median_beat_error_ms']:>6.1f}ms "
              f"{result['pct_within_50ms']:>5.1f}% "
              f"{result['n_beats']:>5d}")


if __name__ == '__main__':
    main()
