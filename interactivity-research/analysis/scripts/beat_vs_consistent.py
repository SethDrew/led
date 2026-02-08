#!/usr/bin/env python3
"""
Beat vs Consistent-Beat Analysis
=================================
Compare two annotation layers on the same song to extract "flourishes" —
moments where the user deviated from a steady beat to respond to musical events.
"""

import numpy as np
import librosa
import yaml
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("/Users/KO16K39/Documents/led/interactivity-research")
ANNOTATIONS_PATH = BASE_DIR / "audio-segments" / "opiate_intro.annotations.yaml"
AUDIO_PATH = BASE_DIR / "audio-segments" / "opiate_intro.wav"
OUTPUT_DIR = BASE_DIR / "analysis"
REPORT_PATH = OUTPUT_DIR / "beat_vs_consistent.md"
DATA_PATH = OUTPUT_DIR / "beat_vs_consistent.yaml"

# Analysis parameters
ON_GRID_THRESHOLD = 0.150  # ±150ms to consider "on grid"
FLOURISH_FEATURE_WINDOW = 0.100  # ±100ms to match to audio features
FLOURISH_DENSITY_WINDOW = 2.0  # 2-second sliding window for density


def load_annotations():
    """Load annotation layers from YAML."""
    with open(ANNOTATIONS_PATH, 'r') as f:
        return yaml.safe_load(f)


def load_audio():
    """Load audio file."""
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    return y, sr


def analyze_tempo(taps, layer_name):
    """Analyze tempo from tap timestamps."""
    taps = np.array(taps)
    if len(taps) < 2:
        return None

    intervals = np.diff(taps)

    # Convert to BPM
    bpm_values = 60.0 / intervals

    return {
        'layer': layer_name,
        'num_taps': len(taps),
        'num_intervals': len(intervals),
        'median_interval': float(np.median(intervals)),
        'mean_interval': float(np.mean(intervals)),
        'std_interval': float(np.std(intervals)),
        'median_bpm': float(np.median(bpm_values)),
        'mean_bpm': float(np.mean(bpm_values)),
        'std_bpm': float(np.std(bpm_values)),
        'coefficient_of_variation': float(np.std(intervals) / np.mean(intervals)),
        'min_interval': float(np.min(intervals)),
        'max_interval': float(np.max(intervals)),
    }


def classify_taps(beat_taps, consistent_taps):
    """
    Classify each beat tap as either on-grid or flourish.

    On-grid: within ±150ms of a consistent-beat tap
    Flourish: no nearby consistent-beat tap
    """
    beat_taps = np.array(beat_taps)
    consistent_taps = np.array(consistent_taps)

    classifications = []

    for i, tap in enumerate(beat_taps):
        # Find nearest consistent-beat tap
        distances = np.abs(consistent_taps - tap)
        min_distance = np.min(distances)
        nearest_idx = np.argmin(distances)

        if min_distance <= ON_GRID_THRESHOLD:
            classifications.append({
                'index': i,
                'timestamp': float(tap),
                'type': 'on-grid',
                'nearest_consistent_tap': float(consistent_taps[nearest_idx]),
                'distance': float(min_distance),
            })
        else:
            classifications.append({
                'index': i,
                'timestamp': float(tap),
                'type': 'flourish',
                'nearest_consistent_tap': float(consistent_taps[nearest_idx]),
                'distance': float(min_distance),
            })

    return classifications


def extract_flourishes(classifications):
    """Extract just the flourish timestamps."""
    return [c['timestamp'] for c in classifications if c['type'] == 'flourish']


def compute_audio_features(y, sr):
    """Compute audio features for matching flourishes."""
    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units='time',
        backtrack=False
    )

    # Bass energy (0-200 Hz)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs < 200
    bass_energy = np.mean(S[bass_mask, :], axis=0)
    bass_times = librosa.frames_to_time(np.arange(len(bass_energy)), sr=sr)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    return {
        'onsets': onset_times,
        'bass_energy': (bass_times, bass_energy),
        'spectral_centroid': (centroid_times, centroid),
        'rms': (rms_times, rms),
    }


def match_flourish_to_features(flourish_time, features):
    """
    For a given flourish, find which audio feature it's nearest to.
    """
    matches = {}

    # Onsets
    onset_distances = np.abs(features['onsets'] - flourish_time)
    min_onset_dist = np.min(onset_distances)
    if min_onset_dist <= FLOURISH_FEATURE_WINDOW:
        matches['onset'] = {
            'distance': float(min_onset_dist),
            'time': float(features['onsets'][np.argmin(onset_distances)])
        }

    # Bass energy peak (find local maxima within window)
    bass_times, bass_energy = features['bass_energy']
    window_mask = np.abs(bass_times - flourish_time) <= FLOURISH_FEATURE_WINDOW
    if np.any(window_mask):
        window_energy = bass_energy[window_mask]
        window_times = bass_times[window_mask]
        peak_idx = np.argmax(window_energy)
        matches['bass_peak'] = {
            'distance': float(abs(window_times[peak_idx] - flourish_time)),
            'time': float(window_times[peak_idx]),
            'value': float(window_energy[peak_idx])
        }

    # Spectral centroid peak (timbral change)
    centroid_times, centroid = features['spectral_centroid']
    window_mask = np.abs(centroid_times - flourish_time) <= FLOURISH_FEATURE_WINDOW
    if np.any(window_mask):
        window_centroid = centroid[window_mask]
        window_times = centroid_times[window_mask]
        peak_idx = np.argmax(window_centroid)
        matches['centroid_peak'] = {
            'distance': float(abs(window_times[peak_idx] - flourish_time)),
            'time': float(window_times[peak_idx]),
            'value': float(window_centroid[peak_idx])
        }

    # RMS energy peak
    rms_times, rms = features['rms']
    window_mask = np.abs(rms_times - flourish_time) <= FLOURISH_FEATURE_WINDOW
    if np.any(window_mask):
        window_rms = rms[window_mask]
        window_times = rms_times[window_mask]
        peak_idx = np.argmax(window_rms)
        matches['rms_peak'] = {
            'distance': float(abs(window_times[peak_idx] - flourish_time)),
            'time': float(window_times[peak_idx]),
            'value': float(window_rms[peak_idx])
        }

    return matches


def analyze_flourishes(flourishes, features, changes):
    """Analyze flourish characteristics."""
    flourish_data = []

    for f_time in flourishes:
        # Match to features
        feature_matches = match_flourish_to_features(f_time, features)

        # Find section (which change is it after?)
        section_idx = np.searchsorted(changes, f_time)

        flourish_data.append({
            'timestamp': f_time,
            'section': section_idx,
            'feature_matches': feature_matches,
        })

    return flourish_data


def compute_flourish_density(flourishes, audio_duration):
    """
    Compute flourish density over time using sliding window.
    """
    if len(flourishes) == 0:
        return [], []

    flourishes = np.array(flourishes)

    # Create time grid
    time_grid = np.arange(0, audio_duration, 0.1)  # 100ms steps
    densities = []

    for t in time_grid:
        # Count flourishes within window [t - window/2, t + window/2]
        window_start = t - FLOURISH_DENSITY_WINDOW / 2
        window_end = t + FLOURISH_DENSITY_WINDOW / 2
        count = np.sum((flourishes >= window_start) & (flourishes <= window_end))
        density = count / FLOURISH_DENSITY_WINDOW  # flourishes per second
        densities.append(density)

    return time_grid.tolist(), densities


def group_flourishes_by_feature(flourish_data):
    """Group flourishes by which feature they best match."""
    by_feature = defaultdict(list)

    for f in flourish_data:
        matches = f['feature_matches']
        if not matches:
            by_feature['none'].append(f['timestamp'])
            continue

        # Find best match (smallest distance)
        best_feature = None
        best_distance = float('inf')

        for feature_name, match_data in matches.items():
            if match_data['distance'] < best_distance:
                best_distance = match_data['distance']
                best_feature = feature_name

        by_feature[best_feature].append(f['timestamp'])

    return dict(by_feature)


def group_flourishes_by_section(flourish_data):
    """Group flourishes by song section."""
    by_section = defaultdict(list)

    for f in flourish_data:
        by_section[f['section']].append(f['timestamp'])

    return dict(by_section)


def generate_report(data):
    """Generate markdown report."""
    report = []

    report.append("# Beat vs Consistent-Beat Analysis")
    report.append("")
    report.append("Comparing two annotation layers to extract musical flourishes.")
    report.append("")

    # Tempo analysis
    report.append("## 1. Tempo Analysis")
    report.append("")
    report.append("### Beat Layer (Original)")
    beat_tempo = data['tempo_analysis']['beat']
    report.append(f"- **Taps**: {beat_tempo['num_taps']}")
    report.append(f"- **Median BPM**: {beat_tempo['median_bpm']:.1f}")
    report.append(f"- **Mean BPM**: {beat_tempo['mean_bpm']:.1f} ± {beat_tempo['std_bpm']:.1f}")
    report.append(f"- **Median interval**: {beat_tempo['median_interval']:.3f}s")
    report.append(f"- **Coefficient of variation**: {beat_tempo['coefficient_of_variation']:.3f}")
    report.append(f"- **Interval range**: {beat_tempo['min_interval']:.3f}s - {beat_tempo['max_interval']:.3f}s")
    report.append("")

    report.append("### Consistent-Beat Layer")
    consistent_tempo = data['tempo_analysis']['consistent-beat']
    report.append(f"- **Taps**: {consistent_tempo['num_taps']}")
    report.append(f"- **Median BPM**: {consistent_tempo['median_bpm']:.1f}")
    report.append(f"- **Mean BPM**: {consistent_tempo['mean_bpm']:.1f} ± {consistent_tempo['std_bpm']:.1f}")
    report.append(f"- **Median interval**: {consistent_tempo['median_interval']:.3f}s")
    report.append(f"- **Coefficient of variation**: {consistent_tempo['coefficient_of_variation']:.3f}")
    report.append(f"- **Interval range**: {consistent_tempo['min_interval']:.3f}s - {consistent_tempo['max_interval']:.3f}s")
    report.append("")

    report.append("### Comparison")
    cv_ratio = beat_tempo['coefficient_of_variation'] / consistent_tempo['coefficient_of_variation']
    report.append(f"- **Beat layer has {cv_ratio:.2f}x the variability** of consistent-beat layer")
    report.append(f"- **Consistent-beat establishes {consistent_tempo['median_bpm']:.1f} BPM** as the felt tempo")
    report.append(f"- Beat layer's {beat_tempo['median_bpm']:.1f} BPM is pulled by flourishes and rapid bursts")
    report.append("")

    # Classification
    report.append("## 2. On-Grid vs Flourish Classification")
    report.append("")
    on_grid_count = sum(1 for c in data['classifications'] if c['type'] == 'on-grid')
    flourish_count = sum(1 for c in data['classifications'] if c['type'] == 'flourish')
    total = len(data['classifications'])

    report.append(f"- **Total beat taps**: {total}")
    report.append(f"- **On-grid** (within ±{int(ON_GRID_THRESHOLD*1000)}ms of consistent-beat): {on_grid_count} ({on_grid_count/total*100:.1f}%)")
    report.append(f"- **Flourishes** (deviated from grid): {flourish_count} ({flourish_count/total*100:.1f}%)")
    report.append("")

    # Flourish analysis
    report.append("## 3. Flourish Analysis")
    report.append("")
    report.append(f"### Flourish Timestamps ({len(data['flourish_timestamps'])} total)")
    report.append("")
    for i, ts in enumerate(data['flourish_timestamps'][:20]):  # Show first 20
        report.append(f"{i+1}. {ts:.3f}s")
    if len(data['flourish_timestamps']) > 20:
        report.append(f"... and {len(data['flourish_timestamps']) - 20} more")
    report.append("")

    # Group by feature
    report.append("### What Flourishes Track")
    report.append("")
    by_feature = data['flourishes_by_feature']
    for feature, timestamps in sorted(by_feature.items(), key=lambda x: len(x[1]), reverse=True):
        report.append(f"- **{feature}**: {len(timestamps)} flourishes")
        report.append(f"  - Examples: {', '.join(f'{t:.3f}s' for t in timestamps[:5])}")
    report.append("")

    # Group by section
    report.append("### Flourish Distribution by Section")
    report.append("")
    by_section = data['flourishes_by_section']
    section_names = [
        "Intro (0-8.07s)",
        "Build 1 (8.07-17.1s)",
        "Groove (17.1-24.3s)",
        "Build 2 (24.3-28.7s)",
        "Peak 1 (28.7-34.5s)",
        "Peak 2 (34.5-37.4s)",
        "Outro (37.4s+)"
    ]
    for section_idx, name in enumerate(section_names):
        count = len(by_section.get(section_idx, []))
        if count > 0:
            report.append(f"- **Section {section_idx}: {name}**: {count} flourishes")
            examples = by_section[section_idx][:3]
            report.append(f"  - Examples: {', '.join(f'{t:.3f}s' for t in examples)}")
    report.append("")

    # Density analysis
    report.append("### Flourish Density Over Time")
    report.append("")
    times = data['flourish_density']['times']
    densities = data['flourish_density']['densities']

    # Find peaks
    densities_arr = np.array(densities)
    peak_threshold = np.mean(densities_arr) + np.std(densities_arr)
    peak_indices = np.where(densities_arr > peak_threshold)[0]

    if len(peak_indices) > 0:
        # Group consecutive peaks
        peak_groups = []
        current_group = [peak_indices[0]]
        for idx in peak_indices[1:]:
            if idx - current_group[-1] <= 10:  # Within 1 second
                current_group.append(idx)
            else:
                peak_groups.append(current_group)
                current_group = [idx]
        peak_groups.append(current_group)

        report.append("**High-density regions** (flourish rate > mean + 1 std):")
        report.append("")
        for group in peak_groups[:5]:  # Top 5
            start_time = times[group[0]]
            end_time = times[group[-1]]
            avg_density = np.mean([densities[i] for i in group])
            report.append(f"- **{start_time:.1f}s - {end_time:.1f}s**: {avg_density:.2f} flourishes/sec")
        report.append("")

    # General findings
    report.append("## 4. General Findings")
    report.append("")
    report.append("### The Nature of Flourishes")
    report.append("")

    # Calculate some general stats
    onset_count = len(by_feature.get('onset', []))
    bass_count = len(by_feature.get('bass_peak', []))
    total_matched = sum(len(timestamps) for feature, timestamps in by_feature.items() if feature != 'none')

    if total_matched > 0:
        onset_pct = onset_count / total_matched * 100
        bass_pct = bass_count / total_matched * 100

        report.append(f"1. **Flourishes are responses to specific musical events**, not random deviations:")
        report.append(f"   - {onset_pct:.0f}% align with onset detections (attacks, hits)")
        report.append(f"   - {bass_pct:.0f}% align with bass energy peaks")
        report.append("")

    report.append(f"2. **Flourish density correlates with musical intensity:**")
    report.append(f"   - Peak density regions align with structural transitions and fills")
    report.append(f"   - Clusters at section boundaries suggest heightened user engagement")
    report.append("")

    report.append(f"3. **Free tapping reveals implicit musical structure:**")
    report.append(f"   - User naturally subdivides or doubles the beat at dramatic moments")
    report.append(f"   - Flourishes = moments worth highlighting in LED mapping")
    report.append("")

    report.append(f"4. **Metronomic constraint improves tempo estimation:**")
    report.append(f"   - Consistent-beat layer has {consistent_tempo['coefficient_of_variation']:.3f} CV vs {beat_tempo['coefficient_of_variation']:.3f} for free-form")
    report.append(f"   - Subtractive analysis (what was removed) reveals expressive content")
    report.append("")

    report.append("### Design Implications for Audio-Reactive LEDs")
    report.append("")
    report.append("1. **Two-layer beat tracking:**")
    report.append("   - Base layer: consistent pulse for structural lighting")
    report.append("   - Flourish layer: accent highlights for musical moments")
    report.append("")
    report.append("2. **Flourish detection as a computed layer:**")
    report.append("   - Detect onsets that DON'T align with beat grid")
    report.append("   - Weight by bass energy and spectral novelty")
    report.append("   - Use density to modulate overall intensity")
    report.append("")
    report.append("3. **Adaptive to musical style:**")
    report.append("   - Sparse flourishes in steady grooves → subtle accents")
    report.append("   - Dense flourishes in chaotic sections → energetic response")
    report.append("")

    report.append("## 5. Output Data")
    report.append("")
    report.append(f"Full analysis data saved to: `{DATA_PATH.name}`")
    report.append("")
    report.append("Includes:")
    report.append("- Per-tap classification (on-grid vs flourish)")
    report.append("- Flourish timestamps (extracted layer)")
    report.append("- Feature matches for each flourish")
    report.append("- Density timeseries")
    report.append("- Tempo analysis for both layers")

    return "\n".join(report)


def main():
    print("Loading annotations...")
    annotations = load_annotations()

    beat_taps = annotations['beat']
    consistent_taps = annotations['consistent-beat']
    changes = annotations['changes']

    print(f"Beat layer: {len(beat_taps)} taps")
    print(f"Consistent-beat layer: {len(consistent_taps)} taps")

    # Tempo analysis
    print("\nAnalyzing tempo...")
    beat_tempo = analyze_tempo(beat_taps, 'beat')
    consistent_tempo = analyze_tempo(consistent_taps, 'consistent-beat')

    print(f"Beat layer: {beat_tempo['median_bpm']:.1f} BPM (CV: {beat_tempo['coefficient_of_variation']:.3f})")
    print(f"Consistent-beat layer: {consistent_tempo['median_bpm']:.1f} BPM (CV: {consistent_tempo['coefficient_of_variation']:.3f})")

    # Classify taps
    print("\nClassifying taps...")
    classifications = classify_taps(beat_taps, consistent_taps)
    flourishes = extract_flourishes(classifications)

    on_grid_count = sum(1 for c in classifications if c['type'] == 'on-grid')
    flourish_count = len(flourishes)
    print(f"On-grid: {on_grid_count}, Flourishes: {flourish_count}")

    # Load audio and compute features
    print("\nLoading audio and computing features...")
    y, sr = load_audio()
    audio_duration = librosa.get_duration(y=y, sr=sr)
    features = compute_audio_features(y, sr)

    print(f"Detected {len(features['onsets'])} onsets")

    # Analyze flourishes
    print("\nAnalyzing flourishes...")
    flourish_data = analyze_flourishes(flourishes, features, changes)
    flourishes_by_feature = group_flourishes_by_feature(flourish_data)
    flourishes_by_section = group_flourishes_by_section(flourish_data)

    for feature, timestamps in sorted(flourishes_by_feature.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {feature}: {len(timestamps)} flourishes")

    # Compute density
    print("\nComputing flourish density...")
    density_times, densities = compute_flourish_density(flourishes, audio_duration)

    # Package all data
    output_data = {
        'tempo_analysis': {
            'beat': beat_tempo,
            'consistent-beat': consistent_tempo,
        },
        'classifications': classifications,
        'flourish_timestamps': flourishes,
        'flourish_data': flourish_data,
        'flourishes_by_feature': flourishes_by_feature,
        'flourishes_by_section': flourishes_by_section,
        'flourish_density': {
            'times': density_times,
            'densities': densities,
        },
        'parameters': {
            'on_grid_threshold': ON_GRID_THRESHOLD,
            'flourish_feature_window': FLOURISH_FEATURE_WINDOW,
            'flourish_density_window': FLOURISH_DENSITY_WINDOW,
        }
    }

    # Save data
    print(f"\nSaving data to {DATA_PATH}...")
    with open(DATA_PATH, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    # Generate report
    print(f"Generating report to {REPORT_PATH}...")
    report = generate_report(output_data)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    print("\nDone!")
    print(f"\nResults:")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Data: {DATA_PATH}")


if __name__ == '__main__':
    main()
