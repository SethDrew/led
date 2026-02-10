#!/usr/bin/env python3
"""
Harmonix Beat Detector Hyperparameter Optimization

Runs our real-time beat detector (BeatDetector from realtime_beat_led.py) on
Harmonix Set tracks with ground truth annotations, and sweeps hyperparameters
to find optimal settings for different genres.

Usage:
    python harmonix_optimize.py
    python harmonix_optimize.py --quick  # Coarse grid for testing

Outputs:
    - harmonix_optimization.md (results report)
    - harmonix_optimal_params.yaml (best params per genre)
    - harmonix_optimization.png (visualization)
"""

import sys
import os
import time
import argparse
import yaml
import json
from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np
import librosa
import jams
import mir_eval
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

# ── Constants ───────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent.parent / 'datasets' / 'harmonix'
AUDIO_DIR = DATASET_DIR / 'audio'
JAMS_DIR = DATASET_DIR / 'dataset' / 'jams'
TRACK_MAPPING = AUDIO_DIR / 'track_mapping.yaml'
OUTPUT_DIR = Path(__file__).parent.parent

# Hyperparameter search space
PARAM_GRID = {
    'THRESHOLD_MULTIPLIER': [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    'BASS_HIGH_HZ': [150, 200, 250, 300, 400],
    'MIN_BEAT_INTERVAL_SEC': [0.2, 0.3, 0.4, 0.5],
    'FLUX_HISTORY_SEC': [2.0, 3.0, 5.0]
}

# Quick grid for testing
QUICK_PARAM_GRID = {
    'THRESHOLD_MULTIPLIER': [1.5, 2.0, 2.5],
    'BASS_HIGH_HZ': [200, 250, 300],
    'MIN_BEAT_INTERVAL_SEC': [0.3, 0.4],
    'FLUX_HISTORY_SEC': [3.0]
}

# ── BeatDetector (copied from realtime_beat_led.py) ────────────────
class BeatDetector:
    """Real-time bass-band spectral flux beat detection."""

    def __init__(self, sample_rate=44100, n_fft=2048,
                 bass_low_hz=20, bass_high_hz=250,
                 threshold_multiplier=1.5,
                 min_beat_interval_sec=0.3,
                 flux_history_sec=3.0,
                 chunk_size=1024):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.chunk_size = chunk_size

        # Parameters
        self.bass_low_hz = bass_low_hz
        self.bass_high_hz = bass_high_hz
        self.threshold_multiplier = threshold_multiplier
        self.min_beat_interval_sec = min_beat_interval_sec

        # Frequency bin indices for bass range
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self.bass_bins = np.where((freqs >= bass_low_hz) & (freqs <= bass_high_hz))[0]

        # State
        self.prev_spectrum = None
        self.flux_history = []
        self.max_history = int(flux_history_sec * sample_rate / chunk_size)
        self.last_beat_time = 0
        self.beat_times = []  # Track all beat times

        # Windowing
        self.window = np.hanning(n_fft)

        # Audio buffer for overlapping FFT
        self.audio_buffer = np.zeros(n_fft)

    def process_chunk(self, audio_chunk, chunk_time):
        """Process an audio chunk. Returns (is_beat, flux_value, threshold)."""
        # Shift buffer and add new audio
        chunk_len = len(audio_chunk)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = audio_chunk

        # Windowed FFT
        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Extract bass band
        bass_spectrum = spectrum[self.bass_bins]

        if self.prev_spectrum is None:
            self.prev_spectrum = bass_spectrum
            return False, 0.0, 0.0

        # Spectral flux: sum of positive differences (half-wave rectified)
        diff = bass_spectrum - self.prev_spectrum
        flux = np.sum(np.maximum(diff, 0))
        self.prev_spectrum = bass_spectrum

        # Adaptive threshold
        self.flux_history.append(flux)
        if len(self.flux_history) > self.max_history:
            self.flux_history.pop(0)

        if len(self.flux_history) < 10:
            return False, flux, 0.0

        mean_flux = np.mean(self.flux_history)
        std_flux = np.std(self.flux_history)
        threshold = mean_flux + self.threshold_multiplier * std_flux

        # Beat detection with minimum interval
        is_beat = (flux > threshold and
                   (chunk_time - self.last_beat_time) > self.min_beat_interval_sec)

        if is_beat:
            self.last_beat_time = chunk_time
            self.beat_times.append(chunk_time)

        return is_beat, flux, threshold


def load_ground_truth_beats(jams_path):
    """Load beat times from Harmonix JAMS file."""
    jam = jams.load(str(jams_path))
    beat_ann = jam.search(namespace='beat')[0]
    beat_times = np.array([obs.time for obs in beat_ann.data])
    return beat_times


def compute_reference_tempo(beat_times):
    """Compute tempo from ground truth beat times."""
    if len(beat_times) < 2:
        return None
    ibi = np.median(np.diff(beat_times))
    tempo = 60.0 / ibi
    return tempo


def run_beat_detection(audio_path, params):
    """
    Run beat detection on audio file with given parameters.

    Returns detected beat times array.
    """
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=44100, mono=True)

    # Create detector
    detector = BeatDetector(
        sample_rate=sr,
        n_fft=2048,
        chunk_size=1024,
        bass_low_hz=20,
        bass_high_hz=params['BASS_HIGH_HZ'],
        threshold_multiplier=params['THRESHOLD_MULTIPLIER'],
        min_beat_interval_sec=params['MIN_BEAT_INTERVAL_SEC'],
        flux_history_sec=params['FLUX_HISTORY_SEC']
    )

    # Process in chunks (simulate real-time)
    chunk_size = 1024
    for i in range(0, len(y) - chunk_size, chunk_size):
        chunk = y[i:i+chunk_size]
        chunk_time = i / sr
        detector.process_chunk(chunk, chunk_time)

    return np.array(detector.beat_times)


def evaluate_beat_detection(detected_beats, reference_beats):
    """
    Evaluate beat detection using mir_eval.

    Returns dict with F-measure, precision, recall, tempo accuracy.
    """
    if len(detected_beats) < 2 or len(reference_beats) < 2:
        return {
            'f_measure': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'detected_tempo': 0.0,
            'reference_tempo': 0.0,
            'tempo_error': 0.0,
            'tempo_ratio': 0.0,
            'tempo_issue': 'NO_BEATS'
        }

    # Compute F-measure with 70ms tolerance
    f_measure = mir_eval.beat.f_measure(
        reference_beats,
        detected_beats,
        f_measure_threshold=0.07
    )

    # Compute precision and recall
    # Use mir_eval's internal matching logic
    matching = mir_eval.util.match_events(
        reference_beats,
        detected_beats,
        0.07  # 70ms window
    )
    precision = len(matching) / len(detected_beats) if len(detected_beats) > 0 else 0.0
    recall = len(matching) / len(reference_beats) if len(reference_beats) > 0 else 0.0

    # Compute tempo
    detected_tempo = compute_reference_tempo(detected_beats)
    reference_tempo = compute_reference_tempo(reference_beats)

    if detected_tempo is None or reference_tempo is None:
        tempo_error = 0.0
        tempo_ratio = 0.0
        tempo_issue = 'NO_TEMPO'
    else:
        tempo_error = detected_tempo - reference_tempo
        tempo_ratio = detected_tempo / reference_tempo

        # Detect tempo doubling/halving
        if 1.9 <= tempo_ratio <= 2.1:
            tempo_issue = "DOUBLED"
        elif 0.45 <= tempo_ratio <= 0.55:
            tempo_issue = "HALVED"
        elif 0.9 <= tempo_ratio <= 1.1:
            tempo_issue = "OK"
        else:
            tempo_issue = "OFF"

    return {
        'f_measure': f_measure,
        'precision': precision,
        'recall': recall,
        'detected_tempo': detected_tempo or 0.0,
        'reference_tempo': reference_tempo or 0.0,
        'tempo_error': tempo_error,
        'tempo_ratio': tempo_ratio,
        'tempo_issue': tempo_issue
    }


def optimize_track(audio_path, jams_path, param_grid):
    """
    Optimize beat detector parameters for a single track.

    Returns list of dicts with params and scores.
    """
    # Load ground truth
    reference_beats = load_ground_truth_beats(jams_path)

    # Generate all parameter combinations
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combinations = list(product(*param_values))

    results = []

    for combo in tqdm(combinations, desc=f"  Testing {audio_path.stem}", leave=False):
        params = dict(zip(param_names, combo))

        # Run detection
        detected_beats = run_beat_detection(audio_path, params)

        # Evaluate
        scores = evaluate_beat_detection(detected_beats, reference_beats)

        # Store result
        result = {**params, **scores}
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Optimize beat detector hyperparameters')
    parser.add_argument('--quick', action='store_true', help='Use coarse grid for testing')
    parser.add_argument('--fallback', action='store_true', help='Use existing test audio if Harmonix unavailable')
    args = parser.parse_args()

    param_grid = QUICK_PARAM_GRID if args.quick else PARAM_GRID

    print("\n" + "="*70)
    print("  Harmonix Beat Detector Hyperparameter Optimization")
    print("="*70)
    print(f"\n  Parameter grid:")
    for k, v in param_grid.items():
        print(f"    {k}: {v}")
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n  Total combinations per track: {total_combinations}")

    # Load track mapping
    if not TRACK_MAPPING.exists() or args.fallback:
        print("\n  ⚠️  Using fallback: existing test audio")
        tracks = [
            {
                'filename': 'Opiate Intro.wav',
                'genre': 'rock',
                'harmonix_id': None,
                'artist': 'Tool',
                'title': 'Opiate (Intro)',
                'bpm': 83
            }
        ]
        audio_dir = Path(__file__).parent.parent.parent / 'audio-segments'
        jams_dir = None
    else:
        with open(TRACK_MAPPING) as f:
            mapping = yaml.safe_load(f)
        tracks = mapping['tracks']
        audio_dir = AUDIO_DIR
        jams_dir = JAMS_DIR

    print(f"\n  Found {len(tracks)} tracks\n")

    # Run optimization on each track
    all_results = {}

    for track in tracks:
        print(f"\n{'─'*70}")
        print(f"  {track['artist']} - {track['title']}")
        print(f"  Genre: {track['genre']} | BPM: {track.get('bpm', 'unknown')}")
        print(f"{'─'*70}")

        audio_path = audio_dir / track['filename']

        if not audio_path.exists():
            print(f"  ⚠️  Audio file not found: {audio_path}")
            continue

        # Handle JAMS path
        if jams_dir and track.get('harmonix_id'):
            jams_path = jams_dir / f"{track['harmonix_id']}.jams"
        elif track['harmonix_id'] is None:
            # Use user tap annotations for Opiate
            jams_path = audio_dir.parent / 'opiate_intro.annotations.yaml'
        else:
            print(f"  ⚠️  JAMS file path unknown")
            continue

        if not jams_path.exists():
            print(f"  ⚠️  Annotations not found: {jams_path}")
            continue

        # Handle different annotation formats
        if jams_path.suffix == '.yaml':
            # User tap annotations
            with open(jams_path) as f:
                annotations = yaml.safe_load(f)
            # Use consistent-beat layer
            reference_beats = np.array(annotations['consistent-beat'])
        else:
            # Harmonix JAMS
            reference_beats = load_ground_truth_beats(jams_path)

        print(f"  Reference beats: {len(reference_beats)}")
        ref_tempo = compute_reference_tempo(reference_beats)
        print(f"  Reference tempo: {ref_tempo:.1f} BPM\n")

        # Optimize
        track_results = optimize_track(audio_path, jams_path, param_grid)

        # Find best result
        best = max(track_results, key=lambda r: r['f_measure'])
        print(f"\n  Best F1: {best['f_measure']:.3f}")
        print(f"    Threshold: {best['THRESHOLD_MULTIPLIER']}")
        print(f"    Bass high: {best['BASS_HIGH_HZ']} Hz")
        print(f"    Min interval: {best['MIN_BEAT_INTERVAL_SEC']}s")
        print(f"    Flux history: {best['FLUX_HISTORY_SEC']}s")
        print(f"    Detected tempo: {best['detected_tempo']:.1f} BPM ({best['tempo_issue']})")

        all_results[track['title']] = {
            'track': track,
            'reference_beats': reference_beats,
            'results': track_results,
            'best': best
        }

    # Analyze results by genre
    print(f"\n\n{'='*70}")
    print("  GENRE ANALYSIS")
    print(f"{'='*70}\n")

    genre_results = defaultdict(list)
    for title, data in all_results.items():
        genre = data['track']['genre']
        genre_results[genre].extend(data['results'])

    optimal_params = {}

    for genre in ['electronic', 'rock', 'metal']:
        if genre not in genre_results:
            continue

        results = genre_results[genre]

        # Find best params for this genre (average F1)
        param_scores = defaultdict(list)
        for r in results:
            key = (r['THRESHOLD_MULTIPLIER'], r['BASS_HIGH_HZ'],
                   r['MIN_BEAT_INTERVAL_SEC'], r['FLUX_HISTORY_SEC'])
            param_scores[key].append(r['f_measure'])

        best_key = max(param_scores.keys(), key=lambda k: np.mean(param_scores[k]))
        best_f1 = np.mean(param_scores[best_key])

        print(f"  {genre.upper()}")
        print(f"  {'─'*70}")
        print(f"    Best average F1: {best_f1:.3f}")
        print(f"    THRESHOLD_MULTIPLIER: {best_key[0]}")
        print(f"    BASS_HIGH_HZ: {best_key[1]}")
        print(f"    MIN_BEAT_INTERVAL_SEC: {best_key[2]}")
        print(f"    FLUX_HISTORY_SEC: {best_key[3]}")
        print()

        optimal_params[genre] = {
            'threshold_multiplier': float(best_key[0]),
            'bass_high_hz': int(best_key[1]),
            'min_beat_interval_sec': float(best_key[2]),
            'flux_history_sec': float(best_key[3]),
            'f1_score': float(best_f1)
        }

    # Universal best params
    all_genre_results = []
    for results in genre_results.values():
        all_genre_results.extend(results)

    param_scores = defaultdict(list)
    for r in all_genre_results:
        key = (r['THRESHOLD_MULTIPLIER'], r['BASS_HIGH_HZ'],
               r['MIN_BEAT_INTERVAL_SEC'], r['FLUX_HISTORY_SEC'])
        param_scores[key].append(r['f_measure'])

    universal_best_key = max(param_scores.keys(), key=lambda k: np.mean(param_scores[k]))
    universal_best_f1 = np.mean(param_scores[universal_best_key])

    print(f"  UNIVERSAL (all genres)")
    print(f"  {'─'*70}")
    print(f"    Best average F1: {universal_best_f1:.3f}")
    print(f"    THRESHOLD_MULTIPLIER: {universal_best_key[0]}")
    print(f"    BASS_HIGH_HZ: {universal_best_key[1]}")
    print(f"    MIN_BEAT_INTERVAL_SEC: {universal_best_key[2]}")
    print(f"    FLUX_HISTORY_SEC: {universal_best_key[3]}")
    print()

    optimal_params['universal'] = {
        'threshold_multiplier': float(universal_best_key[0]),
        'bass_high_hz': int(universal_best_key[1]),
        'min_beat_interval_sec': float(universal_best_key[2]),
        'flux_history_sec': float(universal_best_key[3]),
        'f1_score': float(universal_best_f1)
    }

    # Save optimal params
    params_path = OUTPUT_DIR / 'harmonix_optimal_params.yaml'
    with open(params_path, 'w') as f:
        yaml.dump(optimal_params, f, default_flow_style=False, sort_keys=False)
    print(f"  ✓ Saved optimal params to {params_path}")

    # Save detailed results
    results_path = OUTPUT_DIR / 'harmonix_optimization_results.json'
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for title, data in all_results.items():
        serializable_results[title] = {
            'track': data['track'],
            'reference_beats': data['reference_beats'].tolist(),
            'best': data['best']
        }
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"  ✓ Saved detailed results to {results_path}")

    # Generate visualization
    print(f"\n  Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beat Detector Hyperparameter Optimization', fontsize=16, fontweight='bold')

    # 1. F1 scores by track
    ax = axes[0, 0]
    track_names = []
    track_f1s = []
    track_genres = []
    for title, data in all_results.items():
        track_names.append(f"{data['track']['artist']}\n{title}")
        track_f1s.append(data['best']['f_measure'])
        track_genres.append(data['track']['genre'])

    colors = {'electronic': '#3498db', 'rock': '#e74c3c', 'metal': '#9b59b6'}
    bar_colors = [colors.get(g, '#95a5a6') for g in track_genres]

    bars = ax.barh(range(len(track_names)), track_f1s, color=bar_colors, alpha=0.7)
    ax.set_yticks(range(len(track_names)))
    ax.set_yticklabels(track_names, fontsize=8)
    ax.set_xlabel('F1 Score', fontweight='bold')
    ax.set_title('Best F1 Score by Track', fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0.95, color='green', linestyle='--', alpha=0.5, label='Excellent (0.95)')
    ax.axvline(0.85, color='orange', linestyle='--', alpha=0.5, label='Good (0.85)')
    ax.legend(fontsize=8)

    # 2. Tempo accuracy
    ax = axes[0, 1]
    tempo_errors = []
    tempo_labels = []
    for title, data in all_results.items():
        tempo_errors.append(data['best']['tempo_ratio'])
        tempo_labels.append(f"{data['track']['artist']}\n{title}")

    bars = ax.barh(range(len(tempo_labels)), tempo_errors, color=bar_colors, alpha=0.7)
    ax.set_yticks(range(len(tempo_labels)))
    ax.set_yticklabels(tempo_labels, fontsize=8)
    ax.set_xlabel('Tempo Ratio (detected / reference)', fontweight='bold')
    ax.set_title('Tempo Accuracy', fontweight='bold')
    ax.axvline(1.0, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Perfect (1.0x)')
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.5, label='Doubled (2.0x)')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Halved (0.5x)')
    ax.set_xlim(0, 2.5)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Parameter sensitivity (F1 vs threshold multiplier)
    ax = axes[1, 0]
    for genre, results in genre_results.items():
        threshold_vals = sorted(set(r['THRESHOLD_MULTIPLIER'] for r in results))
        f1_by_threshold = defaultdict(list)
        for r in results:
            f1_by_threshold[r['THRESHOLD_MULTIPLIER']].append(r['f_measure'])

        mean_f1s = [np.mean(f1_by_threshold[t]) for t in threshold_vals]
        ax.plot(threshold_vals, mean_f1s, 'o-', label=genre, linewidth=2, markersize=8)

    ax.set_xlabel('Threshold Multiplier', fontweight='bold')
    ax.set_ylabel('Average F1 Score', fontweight='bold')
    ax.set_title('Parameter Sensitivity: Threshold Multiplier', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Parameter sensitivity (F1 vs bass high Hz)
    ax = axes[1, 1]
    for genre, results in genre_results.items():
        bass_vals = sorted(set(r['BASS_HIGH_HZ'] for r in results))
        f1_by_bass = defaultdict(list)
        for r in results:
            f1_by_bass[r['BASS_HIGH_HZ']].append(r['f_measure'])

        mean_f1s = [np.mean(f1_by_bass[b]) for b in bass_vals]
        ax.plot(bass_vals, mean_f1s, 's-', label=genre, linewidth=2, markersize=8)

    ax.set_xlabel('Bass High Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Average F1 Score', fontweight='bold')
    ax.set_title('Parameter Sensitivity: Bass Range', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    viz_path = OUTPUT_DIR / 'harmonix_optimization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {viz_path}")

    # Generate markdown report
    report_path = OUTPUT_DIR / 'harmonix_optimization.md'
    with open(report_path, 'w') as f:
        f.write("# Beat Detector Hyperparameter Optimization Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n\n")
        f.write(f"**Total combinations tested:** {total_combinations} per track\n\n")

        f.write("## Optimal Parameters by Genre\n\n")
        for genre, params in optimal_params.items():
            f.write(f"### {genre.upper()}\n\n")
            f.write(f"- **F1 Score:** {params['f1_score']:.3f}\n")
            f.write(f"- **Threshold Multiplier:** {params['threshold_multiplier']}\n")
            f.write(f"- **Bass High Hz:** {params['bass_high_hz']}\n")
            f.write(f"- **Min Beat Interval:** {params['min_beat_interval_sec']}s\n")
            f.write(f"- **Flux History:** {params['flux_history_sec']}s\n\n")

        f.write("## Per-Track Results\n\n")
        for title, data in all_results.items():
            track = data['track']
            best = data['best']
            f.write(f"### {track['artist']} - {title}\n\n")
            f.write(f"- **Genre:** {track['genre']}\n")
            f.write(f"- **Reference BPM:** {track.get('bpm', 'unknown')}\n")
            f.write(f"- **Best F1:** {best['f_measure']:.3f}\n")
            f.write(f"- **Detected BPM:** {best['detected_tempo']:.1f}\n")
            f.write(f"- **Tempo Issue:** {best['tempo_issue']}\n")
            f.write(f"- **Best Params:**\n")
            f.write(f"  - Threshold: {best['THRESHOLD_MULTIPLIER']}\n")
            f.write(f"  - Bass High: {best['BASS_HIGH_HZ']} Hz\n")
            f.write(f"  - Min Interval: {best['MIN_BEAT_INTERVAL_SEC']}s\n")
            f.write(f"  - Flux History: {best['FLUX_HISTORY_SEC']}s\n\n")

        f.write("## Visualization\n\n")
        f.write(f"![Optimization Results](harmonix_optimization.png)\n\n")

        f.write("## Key Findings\n\n")
        f.write("TBD - Add interpretation of results here.\n")

    print(f"  ✓ Saved report to {report_path}")

    print(f"\n{'='*70}")
    print("  ✓ OPTIMIZATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
