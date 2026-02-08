#!/usr/bin/env python3
"""
Harmonix Set Dataset Explorer

Analyzes the Harmonix dataset structure, computes statistics,
and identifies tracks relevant for beat tracking validation
(especially rock and electronic genres).
"""

import csv
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np


# Dataset paths
DATASET_DIR = Path("/Users/KO16K39/Documents/led/interactivity-research/datasets/harmonix/dataset")
METADATA_FILE = DATASET_DIR / "metadata.csv"
BEATS_DIR = DATASET_DIR / "beats_and_downbeats"
SEGMENTS_DIR = DATASET_DIR / "segments"
JAMS_DIR = DATASET_DIR / "jams"
OUTPUT_DIR = Path("/Users/KO16K39/Documents/led/interactivity-research/analysis")


def load_metadata():
    """Load and parse the metadata CSV file."""
    tracks = []
    with open(METADATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracks.append(row)
    return tracks


def analyze_beat_file(filepath):
    """Analyze a single beat annotation file."""
    beats = []
    downbeats = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                timestamp = float(parts[0])
                position = int(parts[1])
                bar_num = int(parts[2])

                beats.append(timestamp)
                if position == 1:  # Downbeat
                    downbeats.append(timestamp)

    # Compute inter-beat intervals for tempo consistency check
    if len(beats) > 1:
        ibi = np.diff(beats)
        avg_ibi = np.mean(ibi)
        std_ibi = np.std(ibi)
        avg_bpm = 60.0 / avg_ibi if avg_ibi > 0 else 0
        tempo_stability = std_ibi / avg_ibi if avg_ibi > 0 else 0  # Lower is more stable
    else:
        avg_bpm = 0
        tempo_stability = 0

    return {
        'num_beats': len(beats),
        'num_downbeats': len(downbeats),
        'num_bars': downbeats[-1] if downbeats else 0,
        'avg_bpm': avg_bpm,
        'tempo_stability': tempo_stability,
        'duration': beats[-1] if beats else 0
    }


def analyze_segment_file(filepath):
    """Analyze a single segment annotation file."""
    segments = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                timestamp = float(parts[0])
                label = parts[1].strip()
                segments.append((timestamp, label))

    # Get segment labels (excluding 'end')
    labels = [label for _, label in segments if label != 'end']

    return {
        'num_segments': len(segments) - 1,  # Exclude 'end' marker
        'segment_labels': labels
    }


def compute_statistics(tracks):
    """Compute comprehensive dataset statistics."""
    stats = {
        'total_tracks': len(tracks),
        'genres': Counter(),
        'time_signatures': Counter(),
        'tempo_distribution': [],
        'duration_distribution': [],
        'genre_tempo_map': defaultdict(list),
        'tracks_by_genre': defaultdict(list)
    }

    for track in tracks:
        genre = track['Genre']
        bpm = float(track['BPM'])
        duration = float(track['Duration'])
        time_sig = track['Time Signature']

        stats['genres'][genre] += 1
        stats['time_signatures'][time_sig] += 1
        stats['tempo_distribution'].append(bpm)
        stats['duration_distribution'].append(duration)
        stats['genre_tempo_map'][genre].append(bpm)
        stats['tracks_by_genre'][genre].append(track)

    # Compute tempo statistics
    tempos = np.array(stats['tempo_distribution'])
    stats['tempo_stats'] = {
        'min': float(np.min(tempos)),
        'max': float(np.max(tempos)),
        'mean': float(np.mean(tempos)),
        'median': float(np.median(tempos)),
        'std': float(np.std(tempos))
    }

    # Compute duration statistics
    durations = np.array(stats['duration_distribution'])
    stats['duration_stats'] = {
        'min': float(np.min(durations)),
        'max': float(np.max(durations)),
        'mean': float(np.mean(durations)),
        'median': float(np.median(durations)),
        'std': float(np.std(durations)),
        'total_hours': float(np.sum(durations) / 3600)
    }

    return stats


def identify_rock_electronic_tracks(tracks):
    """
    Identify tracks that are rock or electronic for beat tracking validation.
    Categories based on the genre field.
    """
    rock_keywords = ['Rock', 'Metal', 'Alternative', 'Prog']
    electronic_keywords = ['Electronic', 'Dance']

    rock_tracks = []
    electronic_tracks = []

    for track in tracks:
        genre = track['Genre']

        # Check if any rock keyword is in the genre
        if any(keyword in genre for keyword in rock_keywords):
            rock_tracks.append(track)

        # Check if any electronic keyword is in the genre
        if any(keyword in genre for keyword in electronic_keywords):
            electronic_tracks.append(track)

    return rock_tracks, electronic_tracks


def analyze_tempo_range_by_genre(stats):
    """Analyze tempo ranges for each genre."""
    genre_tempo_analysis = {}

    for genre, tempos in stats['genre_tempo_map'].items():
        if tempos:
            tempo_array = np.array(tempos)
            genre_tempo_analysis[genre] = {
                'count': len(tempos),
                'min_bpm': float(np.min(tempo_array)),
                'max_bpm': float(np.max(tempo_array)),
                'mean_bpm': float(np.mean(tempo_array)),
                'median_bpm': float(np.median(tempo_array)),
                'std_bpm': float(np.std(tempo_array))
            }

    return genre_tempo_analysis


def save_summary(stats, rock_tracks, electronic_tracks, genre_tempo_analysis):
    """Save analysis summary to JSON file."""
    summary = {
        'dataset_overview': {
            'total_tracks': stats['total_tracks'],
            'total_duration_hours': stats['duration_stats']['total_hours'],
            'genre_distribution': dict(stats['genres']),
            'time_signature_distribution': dict(stats['time_signatures'])
        },
        'tempo_statistics': stats['tempo_stats'],
        'duration_statistics': stats['duration_stats'],
        'genre_tempo_analysis': genre_tempo_analysis,
        'rock_tracks': {
            'count': len(rock_tracks),
            'examples': [
                {
                    'file': t['File'],
                    'title': t['Title'],
                    'artist': t['Artist'],
                    'bpm': float(t['BPM']),
                    'genre': t['Genre'],
                    'duration': float(t['Duration'])
                }
                for t in sorted(rock_tracks, key=lambda x: x['Title'])[:20]
            ]
        },
        'electronic_tracks': {
            'count': len(electronic_tracks),
            'examples': [
                {
                    'file': t['File'],
                    'title': t['Title'],
                    'artist': t['Artist'],
                    'bpm': float(t['BPM']),
                    'genre': t['Genre'],
                    'duration': float(t['Duration'])
                }
                for t in sorted(electronic_tracks, key=lambda x: x['Title'])[:20]
            ]
        }
    }

    output_file = OUTPUT_DIR / "harmonix_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {output_file}")
    return summary


def print_summary(stats, rock_tracks, electronic_tracks):
    """Print human-readable summary to console."""
    print("=" * 80)
    print("HARMONIX SET DATASET EXPLORATION")
    print("=" * 80)
    print()

    print(f"Total Tracks: {stats['total_tracks']}")
    print(f"Total Duration: {stats['duration_stats']['total_hours']:.2f} hours")
    print()

    print("GENRE DISTRIBUTION:")
    for genre, count in sorted(stats['genres'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_tracks']) * 100
        print(f"  {genre:20s}: {count:3d} tracks ({percentage:5.2f}%)")
    print()

    print("TEMPO STATISTICS:")
    print(f"  Range: {stats['tempo_stats']['min']:.1f} - {stats['tempo_stats']['max']:.1f} BPM")
    print(f"  Mean: {stats['tempo_stats']['mean']:.1f} BPM")
    print(f"  Median: {stats['tempo_stats']['median']:.1f} BPM")
    print(f"  Std Dev: {stats['tempo_stats']['std']:.1f} BPM")
    print()

    print("TIME SIGNATURE DISTRIBUTION:")
    for sig, count in sorted(stats['time_signatures'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_tracks']) * 100
        print(f"  {sig:10s}: {count:3d} tracks ({percentage:5.2f}%)")
    print()

    print(f"ROCK TRACKS: {len(rock_tracks)} total")
    print("  Examples:")
    for track in sorted(rock_tracks, key=lambda x: x['Title'])[:10]:
        bpm = float(track['BPM'])
        print(f"    - {track['Title']:40s} by {track['Artist']:30s} ({bpm:6.1f} BPM, {track['Genre']})")
    print()

    print(f"ELECTRONIC TRACKS: {len(electronic_tracks)} total")
    print("  Examples:")
    for track in sorted(electronic_tracks, key=lambda x: x['Title'])[:10]:
        bpm = float(track['BPM'])
        print(f"    - {track['Title']:40s} by {track['Artist']:30s} ({bpm:6.1f} BPM, {track['Genre']})")
    print()

    print("=" * 80)
    print("RELEVANCE FOR BEAT TRACKING VALIDATION")
    print("=" * 80)
    print()
    print("Rock tracks are particularly useful for testing beat tracking algorithms")
    print("because they often have:")
    print("  - Complex drum patterns (fills, syncopation)")
    print("  - Hi-hat subdivision ambiguity (8th vs quarter notes)")
    print("  - Tempo variations and ritardandos")
    print("  - Distorted guitars that can confuse onset detection")
    print()
    print("Electronic tracks are useful for testing because they have:")
    print("  - Precise, metronomic beats (good baseline)")
    print("  - Spectral complexity in bass range")
    print("  - Synthesized sounds with different attack profiles")
    print()
    print("NOTE: This dataset contains ANNOTATIONS ONLY (no audio files).")
    print("To use for validation, you would need to:")
    print("  1. Match tracks to your own music collection")
    print("  2. Use YouTube URLs provided in the dataset")
    print("  3. Use MusicBrainz/AcoustID identifiers to find audio")
    print("  4. Download mel-spectrograms from Dropbox (~1.2GB)")
    print()


def main():
    """Main execution function."""
    print("Loading metadata...")
    tracks = load_metadata()

    print("Computing statistics...")
    stats = compute_statistics(tracks)

    print("Identifying rock and electronic tracks...")
    rock_tracks, electronic_tracks = identify_rock_electronic_tracks(tracks)

    print("Analyzing tempo ranges by genre...")
    genre_tempo_analysis = analyze_tempo_range_by_genre(stats)

    print("Saving summary...")
    save_summary(stats, rock_tracks, electronic_tracks, genre_tempo_analysis)

    print_summary(stats, rock_tracks, electronic_tracks)

    # Additional analysis: Check if we have beat files for all tracks
    print("Verifying dataset completeness...")
    beats_files = set(f.stem for f in BEATS_DIR.glob("*.txt"))
    segments_files = set(f.stem for f in SEGMENTS_DIR.glob("*.txt"))
    jams_files = set(f.stem for f in JAMS_DIR.glob("*.jams"))
    metadata_files = set(t['File'] for t in tracks)

    print(f"  Beat annotation files: {len(beats_files)}")
    print(f"  Segment annotation files: {len(segments_files)}")
    print(f"  JAMS files: {len(jams_files)}")
    print(f"  Metadata entries: {len(metadata_files)}")

    if len(beats_files) == len(metadata_files):
        print("  ✓ All tracks have beat annotations")
    else:
        missing = metadata_files - beats_files
        if missing:
            print(f"  ⚠ Missing beat annotations for {len(missing)} tracks")

    print()
    print("Exploration complete!")


if __name__ == "__main__":
    main()
