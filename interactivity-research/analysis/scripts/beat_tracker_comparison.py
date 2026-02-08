#!/usr/bin/env python3
"""
Beat Tracker Comparison Script

Compares multiple beat tracking approaches against user tap annotations
for Tool's "Opiate Intro" - a rock song where librosa's default beat_track
incorrectly detected 161.5 BPM (actual tempo is ~101-109 BPM based on user taps).

Approaches tested:
1. librosa default (baseline - known bad)
2. librosa with constrained tempo range
3. librosa with bass-only onset detection
4. madmom RNN beat tracker (if available)
5. Spectral flux beat tracking (custom)

Evaluation metric: F-measure against user taps on steady groove sections.
"""

import numpy as np
import librosa
import soundfile as sf
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Paths
AUDIO_FILE = "/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.wav"
ANNOTATIONS_FILE = "/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.annotations.yaml"
OUTPUT_DIR = Path("/Users/KO16K39/Documents/led/interactivity-research/analysis")
OUTPUT_REPORT = OUTPUT_DIR / "beat_tracker_comparison.md"
OUTPUT_DATA = OUTPUT_DIR / "beat_tracker_comparison.yaml"

# Analysis parameters
MATCH_TOLERANCE = 0.100  # ±100ms window for matching beats to taps
STEADY_GROOVE_START = 8.0  # Start of section B (after first change marker)
STEADY_GROOVE_END = 24.0   # End of section C (before rapid burst section)


def load_annotations() -> Dict[str, List[float]]:
    """Load user tap annotations from YAML."""
    with open(ANNOTATIONS_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data


def load_audio() -> Tuple[np.ndarray, int]:
    """Load audio file."""
    y, sr = librosa.load(AUDIO_FILE, sr=None)
    return y, sr


def filter_steady_groove_taps(taps: List[float], changes: List[float]) -> List[float]:
    """
    Filter user taps to only the steady groove sections (B and C).
    Based on change markers, sections are:
    - A: 0-8s (intro, variable tempo)
    - B: 8-17s (steady groove)
    - C: 17-24s (steady groove continues)
    - D: 24-29s (fills/rapid bursts)
    - E: 29-34s (more variation)
    - F: 34-38s (more variation)
    """
    steady_taps = [t for t in taps if STEADY_GROOVE_START <= t <= STEADY_GROOVE_END]
    return steady_taps


def estimate_tempo_from_taps(taps: List[float]) -> float:
    """Estimate tempo from tap intervals."""
    if len(taps) < 2:
        return 0.0
    intervals = np.diff(taps)
    median_interval = np.median(intervals)
    bpm = 60.0 / median_interval if median_interval > 0 else 0.0
    return bpm


def compute_f_measure(detected_beats: np.ndarray, reference_taps: List[float],
                      tolerance: float = MATCH_TOLERANCE) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.

    A detected beat is "correct" if it's within ±tolerance of a reference tap.
    """
    if len(detected_beats) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'num_detected': 0, 'num_reference': len(reference_taps)}

    if len(reference_taps) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'num_detected': len(detected_beats), 'num_reference': 0}

    reference_taps = np.array(reference_taps)

    # Count true positives from detected beats perspective (precision)
    true_positives_detected = 0
    for beat in detected_beats:
        if np.any(np.abs(reference_taps - beat) <= tolerance):
            true_positives_detected += 1

    # Count true positives from reference taps perspective (recall)
    true_positives_reference = 0
    for tap in reference_taps:
        if np.any(np.abs(detected_beats - tap) <= tolerance):
            true_positives_reference += 1

    precision = true_positives_detected / len(detected_beats) if len(detected_beats) > 0 else 0.0
    recall = true_positives_reference / len(reference_taps) if len(reference_taps) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_detected': len(detected_beats),
        'num_reference': len(reference_taps)
    }


def approach_1_librosa_default(y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    """Approach 1: librosa default beat_track (baseline - known bad)."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times


def approach_2_librosa_constrained(y: np.ndarray, sr: int, start_bpm: int) -> Tuple[float, np.ndarray]:
    """Approach 2: librosa with constrained tempo range."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=start_bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times


def approach_3_librosa_bass_onset(y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    """Approach 3: librosa with bass-only onset detection."""
    # Compute mel spectrogram focused on bass frequencies (20-250Hz)
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmin=20, fmax=250, n_mels=8)

    # Compute onset strength from bass-only spectrogram
    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max), sr=sr)

    # Run beat tracking on bass onset envelope
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times


def approach_4_madmom_rnn(audio_file: str) -> Optional[Tuple[float, np.ndarray]]:
    """Approach 4: madmom RNN beat tracker (if available)."""
    try:
        import madmom

        # Process with RNN beat processor
        proc = madmom.features.beats.RNNBeatProcessor()
        act = proc(audio_file)

        # Track beats with DBN
        beat_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        beat_times = beat_proc(act)

        # Estimate tempo from beat intervals
        if len(beat_times) > 1:
            intervals = np.diff(beat_times)
            median_interval = np.median(intervals)
            tempo = 60.0 / median_interval if median_interval > 0 else 0.0
        else:
            tempo = 0.0

        return tempo, np.array(beat_times)

    except ImportError:
        return None
    except Exception as e:
        print(f"madmom error: {e}")
        return None


def approach_5_spectral_flux(y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    """Approach 5: Spectral flux beat tracking (custom)."""
    # Compute STFT
    D = np.abs(librosa.stft(y))

    # Compute spectral flux (frame-to-frame difference)
    flux = np.zeros(D.shape[1])
    for i in range(1, D.shape[1]):
        # Positive differences only (half-wave rectification)
        diff = D[:, i] - D[:, i-1]
        flux[i] = np.sum(np.maximum(diff, 0))

    # Normalize
    flux = flux / np.max(flux) if np.max(flux) > 0 else flux

    # Find peaks in spectral flux
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(flux, height=0.1, distance=int(sr/512 * 0.2))  # Min 200ms apart
    peak_times = librosa.frames_to_time(peaks, sr=sr)

    # Estimate tempo using autocorrelation of peak signal
    # Create binary peak signal
    peak_signal = np.zeros_like(flux)
    peak_signal[peaks] = 1.0

    # Autocorrelation
    autocorr = np.correlate(peak_signal, peak_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags

    # Look for peaks in autocorrelation corresponding to 80-130 BPM
    min_lag = int((60.0 / 130.0) * sr / 512)  # 130 BPM in frames
    max_lag = int((60.0 / 80.0) * sr / 512)   # 80 BPM in frames

    if max_lag < len(autocorr):
        tempo_region = autocorr[min_lag:max_lag]
        if len(tempo_region) > 0:
            best_lag = np.argmax(tempo_region) + min_lag
            tempo = 60.0 / (best_lag * 512 / sr)
        else:
            tempo = 0.0
    else:
        tempo = 0.0

    return tempo, peak_times


def main():
    print("=" * 80)
    print("BEAT TRACKER COMPARISON")
    print("=" * 80)
    print()

    # Load data
    print("Loading audio and annotations...")
    y, sr = load_audio()
    annotations = load_annotations()

    all_taps = annotations.get('beat', [])
    changes = annotations.get('changes', [])

    print(f"Audio: {len(y)/sr:.1f}s at {sr}Hz")
    print(f"Total user taps: {len(all_taps)}")
    print(f"Change markers: {len(changes)}")
    print()

    # Filter to steady groove sections
    steady_taps = filter_steady_groove_taps(all_taps, changes)
    print(f"Steady groove taps ({STEADY_GROOVE_START:.1f}s - {STEADY_GROOVE_END:.1f}s): {len(steady_taps)}")

    # Estimate reference tempo from user taps
    tap_tempo = estimate_tempo_from_taps(steady_taps)
    print(f"Estimated tempo from user taps: {tap_tempo:.1f} BPM")
    print()

    # Storage for results
    results = {}
    all_beats = {}

    # Approach 1: librosa default
    print("-" * 80)
    print("APPROACH 1: librosa default (baseline)")
    print("-" * 80)
    tempo, beats = approach_1_librosa_default(y, sr)

    # Filter beats to steady groove section for evaluation
    steady_beats = beats[(beats >= STEADY_GROOVE_START) & (beats <= STEADY_GROOVE_END)]
    metrics = compute_f_measure(steady_beats, steady_taps)

    results['librosa_default'] = {
        'tempo': float(tempo),
        'metrics': metrics,
        'notes': 'Known to double tempo on rock music (locks onto hi-hat instead of kick)'
    }
    all_beats['librosa_default'] = beats.tolist()

    print(f"Detected tempo: {tempo:.1f} BPM")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Detected beats in groove: {metrics['num_detected']}")
    print()

    # Approach 2: librosa with constrained tempo (try multiple start BPMs)
    for start_bpm in [100, 105, 110]:
        print("-" * 80)
        print(f"APPROACH 2: librosa constrained (start_bpm={start_bpm})")
        print("-" * 80)
        tempo, beats = approach_2_librosa_constrained(y, sr, start_bpm)

        steady_beats = beats[(beats >= STEADY_GROOVE_START) & (beats <= STEADY_GROOVE_END)]
        metrics = compute_f_measure(steady_beats, steady_taps)

        key = f'librosa_constrained_{start_bpm}'
        results[key] = {
            'tempo': float(tempo),
            'metrics': metrics,
            'notes': f'Constrained to start at {start_bpm} BPM'
        }
        all_beats[key] = beats.tolist()

        print(f"Detected tempo: {tempo:.1f} BPM")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")
        print(f"Detected beats in groove: {metrics['num_detected']}")
        print()

    # Approach 3: librosa with bass-only onset detection
    print("-" * 80)
    print("APPROACH 3: librosa with bass-only onset (20-250Hz)")
    print("-" * 80)
    tempo, beats = approach_3_librosa_bass_onset(y, sr)

    steady_beats = beats[(beats >= STEADY_GROOVE_START) & (beats <= STEADY_GROOVE_END)]
    metrics = compute_f_measure(steady_beats, steady_taps)

    results['librosa_bass_onset'] = {
        'tempo': float(tempo),
        'metrics': metrics,
        'notes': 'Uses only bass frequencies (20-250Hz) for onset detection'
    }
    all_beats['librosa_bass_onset'] = beats.tolist()

    print(f"Detected tempo: {tempo:.1f} BPM")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Detected beats in groove: {metrics['num_detected']}")
    print()

    # Approach 4: madmom RNN (if available)
    print("-" * 80)
    print("APPROACH 4: madmom RNN beat tracker")
    print("-" * 80)
    result = approach_4_madmom_rnn(AUDIO_FILE)
    if result is not None:
        tempo, beats = result

        steady_beats = beats[(beats >= STEADY_GROOVE_START) & (beats <= STEADY_GROOVE_END)]
        metrics = compute_f_measure(steady_beats, steady_taps)

        results['madmom_rnn'] = {
            'tempo': float(tempo),
            'metrics': metrics,
            'notes': 'Neural network trained on annotated data including rock'
        }
        all_beats['madmom_rnn'] = beats.tolist()

        print(f"Detected tempo: {tempo:.1f} BPM")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")
        print(f"Detected beats in groove: {metrics['num_detected']}")
    else:
        print("madmom not available (installation failed due to complex dependencies)")
        results['madmom_rnn'] = {
            'tempo': None,
            'metrics': None,
            'notes': 'Could not be installed (requires Cython build environment)'
        }
    print()

    # Approach 5: Spectral flux
    print("-" * 80)
    print("APPROACH 5: Spectral flux beat tracking (custom)")
    print("-" * 80)
    tempo, beats = approach_5_spectral_flux(y, sr)

    steady_beats = beats[(beats >= STEADY_GROOVE_START) & (beats <= STEADY_GROOVE_END)]
    metrics = compute_f_measure(steady_beats, steady_taps)

    results['spectral_flux'] = {
        'tempo': float(tempo),
        'metrics': metrics,
        'notes': 'Custom: spectral flux peaks + autocorrelation, constrained to 80-130 BPM'
    }
    all_beats['spectral_flux'] = beats.tolist()

    print(f"Detected tempo: {tempo:.1f} BPM")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Detected beats in groove: {metrics['num_detected']}")
    print()

    # Rank by F1 score
    print("=" * 80)
    print("RANKING BY F1 SCORE (on steady groove sections)")
    print("=" * 80)
    print()

    ranked = [(name, data) for name, data in results.items() if data['metrics'] is not None]
    ranked.sort(key=lambda x: x[1]['metrics']['f1'], reverse=True)

    for rank, (name, data) in enumerate(ranked, 1):
        m = data['metrics']
        print(f"{rank}. {name:30s} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  Tempo={data['tempo']:.1f} BPM")
    print()

    # Write markdown report
    print("Writing markdown report...")
    write_markdown_report(results, ranked, tap_tempo, steady_taps, all_taps)

    # Write YAML data
    print("Writing YAML data...")
    write_yaml_data(results, all_beats, tap_tempo, steady_taps, all_taps)

    print()
    print("=" * 80)
    print("DONE")
    print(f"Report: {OUTPUT_REPORT}")
    print(f"Data: {OUTPUT_DATA}")
    print("=" * 80)


def write_markdown_report(results: Dict, ranked: List, tap_tempo: float,
                         steady_taps: List[float], all_taps: List[float]):
    """Write detailed markdown report."""

    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# Beat Tracker Comparison Report\n\n")
        f.write("**Song**: Tool - Opiate (Intro)\n\n")
        f.write("**Audio File**: `Opiate Intro.wav` (~40s)\n\n")
        f.write("**Problem**: librosa's default `beat_track()` incorrectly detected 161.5 BPM when the actual tempo is ~101-109 BPM (based on user tap annotations).\n\n")

        f.write("## Reference Data\n\n")
        f.write(f"- **Total user taps**: {len(all_taps)}\n")
        f.write(f"- **Steady groove taps** ({STEADY_GROOVE_START:.1f}s - {STEADY_GROOVE_END:.1f}s): {len(steady_taps)}\n")
        f.write(f"- **Estimated tempo from taps**: {tap_tempo:.1f} BPM\n")
        f.write(f"- **Match tolerance**: ±{MATCH_TOLERANCE*1000:.0f}ms\n\n")

        f.write("### Song Structure (from change markers)\n\n")
        f.write("Based on user annotations, the song has distinct sections:\n\n")
        f.write("- **Section A (0-8s)**: Intro, variable tempo, rapid hits at ~160 BPM\n")
        f.write("- **Section B (8-17s)**: Steady groove at ~105 BPM ← **PRIMARY EVALUATION SECTION**\n")
        f.write("- **Section C (17-24s)**: Groove continues ← **PRIMARY EVALUATION SECTION**\n")
        f.write("- **Section D (24-29s)**: Fills with rapid bursts\n")
        f.write("- **Section E+ (29-40s)**: More variation and fills\n\n")

        f.write("**Evaluation focuses on sections B and C** (8-24s) where the user's taps are most metronomic.\n\n")

        f.write("## Approaches Tested\n\n")
        for i, (approach, desc) in enumerate([
            ("librosa default", "Baseline: `librosa.beat.beat_track()` with default parameters"),
            ("librosa constrained", "Constrained tempo: `start_bpm` parameter set to 100, 105, or 110"),
            ("librosa bass-only onset", "Bass-focused: onset detection using only 20-250Hz frequency range"),
            ("madmom RNN", "Neural network: RNN beat processor trained on annotated data (if available)"),
            ("spectral flux", "Custom: spectral flux peaks + autocorrelation, constrained to 80-130 BPM")
        ], 1):
            f.write(f"{i}. **{approach}**: {desc}\n")
        f.write("\n")

        f.write("## Results\n\n")
        f.write("### Comparison Table\n\n")
        f.write("| Rank | Approach | Tempo (BPM) | Precision | Recall | F1 | Beats Detected | Notes |\n")
        f.write("|------|----------|-------------|-----------|--------|----|-----------------|-------|\n")

        for rank, (name, data) in enumerate(ranked, 1):
            m = data['metrics']
            tempo_str = f"{data['tempo']:.1f}" if data['tempo'] is not None else "N/A"
            note = data['notes'].replace('|', '\\|')  # Escape pipes in notes
            f.write(f"| {rank} | {name} | {tempo_str} | {m['precision']:.3f} | {m['recall']:.3f} | "
                   f"{m['f1']:.3f} | {m['num_detected']} | {note} |\n")

        # Add madmom if it wasn't available
        if 'madmom_rnn' in results and results['madmom_rnn']['metrics'] is None:
            note = results['madmom_rnn']['notes'].replace('|', '\\|')
            f.write(f"| - | madmom_rnn | N/A | N/A | N/A | N/A | N/A | {note} |\n")

        f.write("\n")

        f.write("### Metrics Explained\n\n")
        f.write(f"- **Precision**: % of detected beats that match a user tap (within ±{MATCH_TOLERANCE*1000:.0f}ms)\n")
        f.write(f"- **Recall**: % of user taps that have a matching detected beat (within ±{MATCH_TOLERANCE*1000:.0f}ms)\n")
        f.write("- **F1**: Harmonic mean of precision and recall (balanced accuracy)\n\n")

        f.write("## Analysis\n\n")

        if len(ranked) > 0:
            best_name, best_data = ranked[0]
            f.write(f"### Best Performing Approach: **{best_name}**\n\n")
            f.write(f"- **Detected tempo**: {best_data['tempo']:.1f} BPM (reference: {tap_tempo:.1f} BPM)\n")
            f.write(f"- **F1 score**: {best_data['metrics']['f1']:.3f}\n")
            f.write(f"- **Precision**: {best_data['metrics']['precision']:.3f}\n")
            f.write(f"- **Recall**: {best_data['metrics']['recall']:.3f}\n")
            f.write(f"- **Notes**: {best_data['notes']}\n\n")

        f.write("### Why librosa Default Failed\n\n")
        if 'librosa_default' in results:
            default = results['librosa_default']
            f.write(f"The default approach detected **{default['tempo']:.1f} BPM** — approximately double the true tempo. ")
            f.write("This is a common failure mode for rock music where:\n\n")
            f.write("- The algorithm locks onto hi-hat or eighth-note subdivisions instead of the kick drum\n")
            f.write("- Without tempo constraints, it finds local maxima at the wrong periodicity\n")
            f.write("- Rock's dense, distorted instrumentation creates many onset candidates\n\n")

        f.write("## Recommendations\n\n")
        f.write("Based on this comparison:\n\n")

        if len(ranked) > 0 and ranked[0][1]['metrics']['f1'] > 0.7:
            f.write(f"1. **Use {ranked[0][0]}** for rock beat tracking on this type of material\n")
            if len(ranked) > 1:
                f.write(f"2. **{ranked[1][0]}** is a good fallback (F1={ranked[1][1]['metrics']['f1']:.3f})\n")
        else:
            f.write("1. **None of the tested approaches achieved strong performance** (best F1 < 0.7)\n")
            f.write("2. Consider user tap data as the ground truth for LED programming\n")

        f.write("3. **User tap annotations are richer than beat tracking**: They capture subdivision changes, fills, and structural transitions\n")
        f.write("4. **For LED effects**, the imperfections in user taps ARE the data — they reflect musical intensity and variation\n\n")

        f.write("## Data Output\n\n")
        f.write(f"All detected beat times are saved to `{OUTPUT_DATA.name}` for further analysis.\n\n")

        f.write("---\n\n")
        f.write("*Generated by `beat_tracker_comparison.py`*\n")


def write_yaml_data(results: Dict, all_beats: Dict, tap_tempo: float,
                   steady_taps: List[float], all_taps: List[float]):
    """Write beat times and results to YAML."""

    data = {
        'reference': {
            'tempo_from_taps_bpm': float(tap_tempo),
            'all_user_taps': all_taps,
            'steady_groove_taps': steady_taps,
            'steady_groove_range': [STEADY_GROOVE_START, STEADY_GROOVE_END],
            'num_total_taps': len(all_taps),
            'num_steady_taps': len(steady_taps)
        },
        'approaches': results,
        'beat_times': all_beats,
        'evaluation': {
            'match_tolerance_seconds': MATCH_TOLERANCE,
            'match_tolerance_ms': MATCH_TOLERANCE * 1000
        }
    }

    with open(OUTPUT_DATA, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    main()
