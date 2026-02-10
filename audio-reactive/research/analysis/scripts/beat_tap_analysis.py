#!/usr/bin/env python3
"""
Analyze user tap annotations to understand what audio features they're tracking.
Does NOT snap taps to onsets - the goal is to UNDERSTAND the relationship.
"""

import librosa
import numpy as np
import yaml
from pathlib import Path
from scipy.signal import find_peaks

# File paths
AUDIO_PATH = "/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.wav"
ANNOTATIONS_PATH = "/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.annotations.yaml"
OUTPUT_REPORT = "/Users/KO16K39/Documents/led/audio-reactive/research/analysis/beat_tap_analysis.md"
OUTPUT_DATA = "/Users/KO16K39/Documents/led/audio-reactive/research/analysis/beat_tap_analysis.yaml"

def load_annotations(path):
    """Load tap annotations from YAML."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def find_nearest_peak(tap_time, peak_times, window_ms=100):
    """Find the nearest peak to a tap time within a window.

    Returns:
        distance_ms: signed distance to nearest peak (negative if peak is before tap)
        peak_time: time of nearest peak (or None if no peak in window)
    """
    if len(peak_times) == 0:
        return None, None

    window_s = window_ms / 1000.0
    distances = peak_times - tap_time

    # Filter to window
    in_window = np.abs(distances) <= window_s
    if not np.any(in_window):
        return None, None

    # Find closest
    distances_in_window = distances[in_window]
    closest_idx = np.argmin(np.abs(distances_in_window))
    distance_s = distances_in_window[closest_idx]
    peak_time = tap_time + distance_s

    return distance_s * 1000, peak_time  # Convert to ms

def detect_local_peaks(envelope, times, height_percentile=30):
    """Detect local peaks in an envelope.

    Args:
        envelope: signal values
        times: corresponding time points
        height_percentile: minimum height as percentile of envelope

    Returns:
        peak_times: array of peak times
    """
    # Require peaks to be above a threshold
    threshold = np.percentile(envelope, height_percentile)

    # Find peaks with minimum distance of ~50ms worth of frames
    min_distance = int(0.05 / (times[1] - times[0]))

    peaks, properties = find_peaks(envelope, height=threshold, distance=min_distance)

    return times[peaks]

def compute_spectral_flux(S, times):
    """Compute spectral flux (frame-to-frame spectral difference)."""
    flux = np.zeros(S.shape[1])
    flux[1:] = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    return flux, times

def segment_taps_by_changes(beat_times, change_times):
    """Segment beat taps into sections based on 'changes' layer timestamps.

    Returns:
        list of dicts with 'start', 'end', 'taps' (tap times in section)
    """
    if len(change_times) == 0:
        return [{'start': 0, 'end': beat_times[-1] if len(beat_times) > 0 else 0,
                 'taps': beat_times, 'section_id': 0}]

    # Add implicit boundaries at start and end
    boundaries = [0] + sorted(change_times) + [beat_times[-1] + 1 if len(beat_times) > 0 else 100]

    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Find taps in this section
        in_section = (beat_times >= start) & (beat_times < end)
        section_taps = beat_times[in_section]

        if len(section_taps) > 0:
            sections.append({
                'start': start,
                'end': end,
                'taps': section_taps,
                'section_id': i
            })

    return sections

def analyze_section(section_taps):
    """Analyze rhythmic properties of a section."""
    if len(section_taps) < 2:
        return None

    intervals = np.diff(section_taps)
    median_interval = np.median(intervals)
    std_interval = np.std(intervals)
    bpm = 60.0 / median_interval if median_interval > 0 else 0

    return {
        'bpm': bpm,
        'median_interval_ms': median_interval * 1000,
        'std_interval_ms': std_interval * 1000,
        'consistency': std_interval / median_interval if median_interval > 0 else 0,  # coefficient of variation
        'num_taps': len(section_taps)
    }

def main():
    print("Loading audio and annotations...")

    # Load audio
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    duration = len(y) / sr
    print(f"Audio: {duration:.2f}s at {sr} Hz")

    # Load annotations
    annotations = load_annotations(ANNOTATIONS_PATH)
    beat_times = np.array(annotations.get('beat', []))
    change_times = annotations.get('changes', [])

    print(f"Found {len(beat_times)} beat taps, {len(change_times)} change markers")

    # Compute audio features at high resolution
    hop_length = 512

    print("\nComputing audio features...")

    # 1. Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

    # 2. Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                               backtrack=False, units='frames')
    onset_detect_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # 3. RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_peaks = detect_local_peaks(rms, rms_times, height_percentile=40)

    # 4. Spectral flux
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    flux, flux_times = compute_spectral_flux(S, rms_times)
    flux_peaks = detect_local_peaks(flux, flux_times, height_percentile=40)

    # 5. Bass energy (20-250 Hz in mel bands)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length,
                                         n_mels=128, fmin=20, fmax=8000)
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)
    bass_bins = mel_freqs <= 250
    bass_energy = np.sum(mel[bass_bins, :], axis=0)
    bass_times = librosa.frames_to_time(np.arange(len(bass_energy)), sr=sr, hop_length=hop_length)
    bass_peaks = detect_local_peaks(bass_energy, bass_times, height_percentile=40)

    # 6. Spectral centroid rate-of-change
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_deriv = np.zeros_like(centroid)
    centroid_deriv[1:] = np.abs(np.diff(centroid))
    centroid_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length)
    centroid_peaks = detect_local_peaks(centroid_deriv, centroid_times, height_percentile=40)

    print(f"Detected {len(onset_detect_times)} onsets")
    print(f"Detected {len(rms_peaks)} RMS peaks")
    print(f"Detected {len(bass_peaks)} bass peaks")
    print(f"Detected {len(flux_peaks)} spectral flux peaks")
    print(f"Detected {len(centroid_peaks)} centroid change peaks")

    # Analyze each tap
    print("\nAnalyzing taps...")
    tap_analysis = []
    key_repeat_candidates = []

    for i, tap_time in enumerate(beat_times):
        # Find nearest features
        onset_dist, onset_peak = find_nearest_peak(tap_time, onset_detect_times, window_ms=100)
        rms_dist, rms_peak = find_nearest_peak(tap_time, rms_peaks, window_ms=100)
        bass_dist, bass_peak = find_nearest_peak(tap_time, bass_peaks, window_ms=100)
        flux_dist, flux_peak = find_nearest_peak(tap_time, flux_peaks, window_ms=100)
        centroid_dist, centroid_peak = find_nearest_peak(tap_time, centroid_peaks, window_ms=100)

        # Check for key repeat
        is_key_repeat = False
        if i > 0:
            prev_gap_ms = (tap_time - beat_times[i-1]) * 1000
            if prev_gap_ms < 100:
                # Could be key repeat - check if both are near distinct onsets
                prev_onset_dist, _ = find_nearest_peak(beat_times[i-1], onset_detect_times, window_ms=100)
                if onset_dist is None or prev_onset_dist is None or abs(onset_dist) > 50:
                    is_key_repeat = True
                    key_repeat_candidates.append(i)

        tap_analysis.append({
            'tap_id': i,
            'tap_time': float(tap_time),
            'onset_dist_ms': float(onset_dist) if onset_dist is not None else None,
            'rms_peak_dist_ms': float(rms_dist) if rms_dist is not None else None,
            'bass_peak_dist_ms': float(bass_dist) if bass_dist is not None else None,
            'flux_peak_dist_ms': float(flux_dist) if flux_dist is not None else None,
            'centroid_peak_dist_ms': float(centroid_dist) if centroid_dist is not None else None,
            'is_key_repeat_candidate': is_key_repeat
        })

    # Segment taps by changes
    print("\nSegmenting by change markers...")
    sections = segment_taps_by_changes(beat_times, change_times)

    # Analyze each section
    section_analysis = []
    for section in sections:
        rhythmic = analyze_section(section['taps'])
        if rhythmic is None:
            continue

        # Find which feature aligns best in this section
        section_tap_indices = [i for i, t in enumerate(beat_times) if section['start'] <= t < section['end']]

        feature_alignments = {
            'onset': [],
            'rms_peak': [],
            'bass_peak': [],
            'flux_peak': [],
            'centroid_peak': []
        }

        for idx in section_tap_indices:
            tap = tap_analysis[idx]
            if tap['onset_dist_ms'] is not None:
                feature_alignments['onset'].append(abs(tap['onset_dist_ms']))
            if tap['rms_peak_dist_ms'] is not None:
                feature_alignments['rms_peak'].append(abs(tap['rms_peak_dist_ms']))
            if tap['bass_peak_dist_ms'] is not None:
                feature_alignments['bass_peak'].append(abs(tap['bass_peak_dist_ms']))
            if tap['flux_peak_dist_ms'] is not None:
                feature_alignments['flux_peak'].append(abs(tap['flux_peak_dist_ms']))
            if tap['centroid_peak_dist_ms'] is not None:
                feature_alignments['centroid_peak'].append(abs(tap['centroid_peak_dist_ms']))

        # Find best alignment
        best_feature = None
        best_median_dist = float('inf')
        for feature, dists in feature_alignments.items():
            if len(dists) > 0:
                median_dist = np.median(dists)
                if median_dist < best_median_dist:
                    best_median_dist = median_dist
                    best_feature = feature

        section_analysis.append({
            'section_id': section['section_id'],
            'time_range': f"{section['start']:.2f}s - {section['end']:.2f}s",
            'start': float(section['start']),
            'end': float(section['end']),
            'bpm': rhythmic['bpm'],
            'median_interval_ms': rhythmic['median_interval_ms'],
            'std_interval_ms': rhythmic['std_interval_ms'],
            'consistency': rhythmic['consistency'],
            'num_taps': rhythmic['num_taps'],
            'best_aligned_feature': best_feature,
            'best_alignment_median_dist_ms': float(best_median_dist),
            'feature_alignments': {k: {'median_ms': float(np.median(v)) if len(v) > 0 else None,
                                       'count': len(v)}
                                  for k, v in feature_alignments.items()}
        })

    # Overall statistics
    print("\nComputing overall statistics...")
    all_onset_dists = [abs(t['onset_dist_ms']) for t in tap_analysis if t['onset_dist_ms'] is not None]
    all_rms_dists = [abs(t['rms_peak_dist_ms']) for t in tap_analysis if t['rms_peak_dist_ms'] is not None]
    all_bass_dists = [abs(t['bass_peak_dist_ms']) for t in tap_analysis if t['bass_peak_dist_ms'] is not None]
    all_flux_dists = [abs(t['flux_peak_dist_ms']) for t in tap_analysis if t['flux_peak_dist_ms'] is not None]

    overall_stats = {
        'total_taps': len(beat_times),
        'key_repeat_candidates': len(key_repeat_candidates),
        'onset_alignment': {
            'within_50ms': sum(1 for d in all_onset_dists if d <= 50),
            'within_30ms': sum(1 for d in all_onset_dists if d <= 30),
            'median_dist_ms': float(np.median(all_onset_dists)) if len(all_onset_dists) > 0 else None,
            'coverage': len(all_onset_dists)
        },
        'rms_peak_alignment': {
            'within_50ms': sum(1 for d in all_rms_dists if d <= 50),
            'within_30ms': sum(1 for d in all_rms_dists if d <= 30),
            'median_dist_ms': float(np.median(all_rms_dists)) if len(all_rms_dists) > 0 else None,
            'coverage': len(all_rms_dists)
        },
        'bass_peak_alignment': {
            'within_50ms': sum(1 for d in all_bass_dists if d <= 50),
            'within_30ms': sum(1 for d in all_bass_dists if d <= 30),
            'median_dist_ms': float(np.median(all_bass_dists)) if len(all_bass_dists) > 0 else None,
            'coverage': len(all_bass_dists)
        },
        'flux_peak_alignment': {
            'within_50ms': sum(1 for d in all_flux_dists if d <= 50),
            'within_30ms': sum(1 for d in all_flux_dists if d <= 30),
            'median_dist_ms': float(np.median(all_flux_dists)) if len(all_flux_dists) > 0 else None,
            'coverage': len(all_flux_dists)
        }
    }

    # Write YAML output
    print(f"\nWriting structured data to {OUTPUT_DATA}...")
    output_data = {
        'metadata': {
            'audio_file': AUDIO_PATH,
            'duration_s': float(duration),
            'sample_rate': int(sr),
            'hop_length': hop_length
        },
        'overall_statistics': overall_stats,
        'sections': section_analysis,
        'taps': tap_analysis,
        'key_repeat_candidates': key_repeat_candidates
    }

    Path(OUTPUT_DATA).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATA, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    # Write markdown report
    print(f"Writing report to {OUTPUT_REPORT}...")

    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# Beat Tap Analysis: Tool - Opiate\n\n")
        f.write("Analysis of user tap annotations to understand what audio features they track.\n\n")

        f.write("## Overall Statistics\n\n")
        f.write(f"- Total taps: {overall_stats['total_taps']}\n")
        f.write(f"- Key repeat candidates: {overall_stats['key_repeat_candidates']}\n\n")

        f.write("### Feature Alignment Summary\n\n")
        f.write("| Feature | Within 30ms | Within 50ms | Median Dist (ms) | Coverage |\n")
        f.write("|---------|-------------|-------------|------------------|----------|\n")

        for feature_name, key in [('Onsets', 'onset_alignment'),
                                   ('RMS Peaks', 'rms_peak_alignment'),
                                   ('Bass Peaks', 'bass_peak_alignment'),
                                   ('Spectral Flux', 'flux_peak_alignment')]:
            stats = overall_stats[key]
            pct_30 = f"{stats['within_30ms']}/{stats['coverage']}" if stats['coverage'] > 0 else "N/A"
            pct_50 = f"{stats['within_50ms']}/{stats['coverage']}" if stats['coverage'] > 0 else "N/A"
            median = f"{stats['median_dist_ms']:.1f}" if stats['median_dist_ms'] is not None else "N/A"
            coverage = f"{stats['coverage']}/{overall_stats['total_taps']}"
            f.write(f"| {feature_name} | {pct_30} | {pct_50} | {median} | {coverage} |\n")

        f.write("\n## Section Analysis\n\n")
        f.write("Sections defined by 'changes' layer timestamps:\n\n")

        f.write("| Section | Time Range | Taps | BPM | Consistency | Best Feature | Median Dist (ms) |\n")
        f.write("|---------|------------|------|-----|-------------|--------------|------------------|\n")

        for sec in section_analysis:
            consistency_desc = "tight" if sec['consistency'] < 0.1 else "moderate" if sec['consistency'] < 0.2 else "loose"
            f.write(f"| {sec['section_id']} | {sec['time_range']} | {sec['num_taps']} | "
                   f"{sec['bpm']:.1f} | {consistency_desc} ({sec['consistency']:.2f}) | "
                   f"{sec['best_aligned_feature']} | {sec['best_alignment_median_dist_ms']:.1f} |\n")

        f.write("\n### Section Details\n\n")
        for sec in section_analysis:
            f.write(f"#### Section {sec['section_id']}: {sec['time_range']}\n\n")
            f.write(f"- **BPM**: {sec['bpm']:.1f} (median interval: {sec['median_interval_ms']:.1f}ms, "
                   f"std: {sec['std_interval_ms']:.1f}ms)\n")
            f.write(f"- **Consistency**: {sec['consistency']:.3f} (coefficient of variation)\n")
            f.write(f"- **Best aligned feature**: {sec['best_aligned_feature']} "
                   f"(median {sec['best_alignment_median_dist_ms']:.1f}ms)\n\n")

            f.write("Feature alignments in this section:\n\n")
            for feat, data in sec['feature_alignments'].items():
                if data['median_ms'] is not None:
                    f.write(f"- {feat}: {data['median_ms']:.1f}ms median ({data['count']} taps)\n")
                else:
                    f.write(f"- {feat}: no matches\n")
            f.write("\n")

        f.write("## Key Repeat Candidates\n\n")
        if len(key_repeat_candidates) > 0:
            f.write(f"Found {len(key_repeat_candidates)} potential key-repeat artifacts "
                   f"(taps <100ms apart, not aligned with distinct onsets):\n\n")
            for idx in key_repeat_candidates:
                tap = tap_analysis[idx]
                prev_tap = tap_analysis[idx-1]
                gap_ms = (tap['tap_time'] - prev_tap['tap_time']) * 1000
                f.write(f"- Tap {idx} at {tap['tap_time']:.3f}s "
                       f"({gap_ms:.1f}ms after previous tap)\n")
        else:
            f.write("No key repeat artifacts detected.\n")

        f.write("\n## Recommendations\n\n")

        # Determine if onset snapping makes sense
        onset_pct_50 = (overall_stats['onset_alignment']['within_50ms'] /
                       max(overall_stats['onset_alignment']['coverage'], 1) * 100)
        bass_pct_50 = (overall_stats['bass_peak_alignment']['within_50ms'] /
                      max(overall_stats['bass_peak_alignment']['coverage'], 1) * 100)

        f.write(f"- **Onset alignment**: {onset_pct_50:.1f}% of taps within 50ms of an onset\n")
        f.write(f"- **Bass peak alignment**: {bass_pct_50:.1f}% of taps within 50ms of a bass peak\n\n")

        if onset_pct_50 > 70:
            f.write("**Recommendation**: Onset-snapping makes sense. Most taps track onsets well.\n")
        elif bass_pct_50 > onset_pct_50:
            f.write("**Recommendation**: Taps track bass peaks more than general onsets. "
                   "Consider bass-specific beat tracking.\n")
        else:
            f.write("**Recommendation**: Taps don't consistently track a single feature. "
                   "User may be tracking multiple aspects (kick, hi-hat, fills). "
                   "Consider context-aware snapping or preserving user timing.\n")

        f.write("\n### Section-specific insights\n\n")
        for sec in section_analysis:
            f.write(f"- **Section {sec['section_id']}** ({sec['time_range']}): "
                   f"Tracks {sec['best_aligned_feature']}, {sec['bpm']:.1f} BPM, "
                   f"{sec['num_taps']} taps\n")

        if len(set(s['best_aligned_feature'] for s in section_analysis)) > 1:
            f.write("\nUser appears to track **different features in different sections**. "
                   "This richness may be valuable for LED scene transitions.\n")

    print("\nDone!")
    print(f"Report: {OUTPUT_REPORT}")
    print(f"Data: {OUTPUT_DATA}")

if __name__ == '__main__':
    main()
