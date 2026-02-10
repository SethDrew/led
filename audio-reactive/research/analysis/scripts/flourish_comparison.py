#!/usr/bin/env python3
"""
Flourish Comparison Analysis

Compares the user's beat-flourish annotation layer against computed flourishes
(beat taps that don't align with consistent-beat) to determine if the beat-flourish
layer adds genuinely new information.
"""

import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

# Constants
MATCH_THRESHOLD = 0.15  # 150ms window for "overlap"
WINDOW_SIZE = 2.0  # seconds for density analysis

def load_annotations(path):
    """Load annotation layers from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_beat_vs_consistent(path):
    """
    Load previous beat vs consistent analysis.
    Parse manually to avoid numpy object deserialization issues.
    """
    flourishes = []
    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse line by line looking for flourish timestamps
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('timestamp:'):
            timestamp = float(line.split(':')[1].strip())
            # Check next line for type
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line == 'type: flourish':
                    flourishes.append(timestamp)
        i += 1

    return sorted(flourishes)

def compute_flourishes_from_layers(beat_taps, consistent_taps, threshold=0.15):
    """
    Recompute flourishes: beat taps that are >threshold from any consistent tap.
    """
    flourishes = []
    for beat_tap in beat_taps:
        distances = [abs(beat_tap - cons_tap) for cons_tap in consistent_taps]
        if min(distances) > threshold:
            flourishes.append(beat_tap)
    return sorted(flourishes)

def find_nearest(timestamp, tap_list):
    """Find nearest tap in a list and return (tap, distance)."""
    if not tap_list:
        return None, float('inf')
    distances = [abs(timestamp - tap) for tap in tap_list]
    min_idx = np.argmin(distances)
    return tap_list[min_idx], distances[min_idx]

def analyze_overlap(beat_flourish_taps, computed_flourishes, threshold=0.15):
    """
    For each beat-flourish tap, check if it overlaps with computed flourishes.
    Returns: overlaps, new_taps, missed_flourishes
    """
    overlaps = []
    new_taps = []

    for bf_tap in beat_flourish_taps:
        nearest_cf, distance = find_nearest(bf_tap, computed_flourishes)
        if distance <= threshold:
            overlaps.append({
                'beat_flourish': bf_tap,
                'computed_flourish': nearest_cf,
                'distance': distance
            })
        else:
            new_taps.append({
                'beat_flourish': bf_tap,
                'nearest_computed': nearest_cf,
                'distance': distance
            })

    # Check which computed flourishes are missed by beat-flourish
    missed = []
    for cf_tap in computed_flourishes:
        nearest_bf, distance = find_nearest(cf_tap, beat_flourish_taps)
        if distance > threshold:
            missed.append({
                'computed_flourish': cf_tap,
                'nearest_beat_flourish': nearest_bf,
                'distance': distance
            })

    return overlaps, new_taps, missed

def analyze_new_taps_context(new_taps, consistent_taps, changes_taps, air_taps,
                             beat_taps, threshold=0.15):
    """
    Analyze what each new tap is near.
    """
    analysis = []

    for item in new_taps:
        bf_tap = item['beat_flourish']

        # Check proximity to different layers
        nearest_cons, dist_cons = find_nearest(bf_tap, consistent_taps)
        nearest_change, dist_change = find_nearest(bf_tap, changes_taps)
        nearest_air, dist_air = find_nearest(bf_tap, air_taps)
        nearest_beat, dist_beat = find_nearest(bf_tap, beat_taps)

        context = {
            'beat_flourish': bf_tap,
            'near_consistent': dist_cons <= threshold,
            'dist_consistent': dist_cons,
            'near_change': dist_change <= threshold * 2,  # Wider window for changes
            'dist_change': dist_change,
            'near_air': dist_air <= threshold,
            'dist_air': dist_air,
            'in_original_beat': dist_beat <= threshold,
            'dist_beat': dist_beat
        }
        analysis.append(context)

    return analysis

def compute_density(taps, duration, window_size=2.0, step=0.1):
    """
    Compute tap density over time using sliding window.
    Returns: time_points, density_values
    """
    time_points = np.arange(0, duration, step)
    densities = []

    for t in time_points:
        window_start = max(0, t - window_size / 2)
        window_end = t + window_size / 2
        count = sum(1 for tap in taps if window_start <= tap <= window_end)
        densities.append(count / window_size)  # taps per second

    return time_points, np.array(densities)

def analyze_intervals(taps):
    """Analyze inter-tap intervals."""
    if len(taps) < 2:
        return None

    intervals = np.diff(sorted(taps))

    return {
        'num_intervals': len(intervals),
        'median': float(np.median(intervals)),
        'mean': float(np.mean(intervals)),
        'std': float(np.std(intervals)),
        'min': float(np.min(intervals)),
        'max': float(np.max(intervals)),
        'cv': float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0,
        'q25': float(np.percentile(intervals, 25)),
        'q75': float(np.percentile(intervals, 75))
    }

def correlate_layers(taps_a, taps_b, threshold=0.15):
    """
    Compute correlation between two tap layers.
    Returns fraction of taps_a that have a nearby tap in taps_b.
    """
    if not taps_a:
        return 0.0

    matches = 0
    for tap_a in taps_a:
        _, distance = find_nearest(tap_a, taps_b)
        if distance <= threshold:
            matches += 1

    return matches / len(taps_a)

def analyze_density_patterns(beat_flourish_taps, computed_flourishes, beat_taps,
                             duration, window_size=2.0):
    """
    Compare density patterns across layers.
    """
    time_points, bf_density = compute_density(beat_flourish_taps, duration, window_size)
    _, cf_density = compute_density(computed_flourishes, duration, window_size)
    _, beat_density = compute_density(beat_taps, duration, window_size)

    # Find moments where beat-flourish is denser or sparser than computed
    bf_denser = np.sum(bf_density > cf_density)
    bf_sparser = np.sum(bf_density < cf_density)
    bf_similar = np.sum(np.abs(bf_density - cf_density) < 0.5)  # Within 0.5 taps/sec

    # Correlation between densities
    correlation = np.corrcoef(bf_density, cf_density)[0, 1] if len(bf_density) > 1 else 0

    return {
        'num_time_points': len(time_points),
        'bf_denser_count': int(bf_denser),
        'bf_sparser_count': int(bf_sparser),
        'similar_count': int(bf_similar),
        'density_correlation': float(correlation),
        'bf_mean_density': float(np.mean(bf_density)),
        'cf_mean_density': float(np.mean(cf_density)),
        'beat_mean_density': float(np.mean(beat_density))
    }

def main():
    # Paths
    base_path = Path("/Users/KO16K39/Documents/led/audio-reactive/research")
    annotations_path = base_path / "audio-segments" / "opiate_intro.annotations.yaml"
    beat_vs_consistent_path = base_path / "analysis" / "beat_vs_consistent.yaml"
    audio_path = base_path / "audio-segments" / "opiate_intro.wav"
    output_path = base_path / "analysis" / "flourish_comparison.md"

    # Load data
    print("Loading annotations...")
    annotations = load_annotations(annotations_path)

    beat_taps = sorted(annotations['beat'])
    consistent_taps = sorted(annotations['consistent-beat'])
    beat_flourish_taps = sorted(annotations['beat-flourish'])
    changes_taps = sorted(annotations['changes'])
    air_taps = sorted(annotations['air'])

    print(f"Loaded {len(beat_taps)} beat taps")
    print(f"Loaded {len(consistent_taps)} consistent-beat taps")
    print(f"Loaded {len(beat_flourish_taps)} beat-flourish taps")
    print(f"Loaded {len(changes_taps)} change markers")
    print(f"Loaded {len(air_taps)} air taps")

    # Load computed flourishes
    print("\nLoading computed flourishes from beat_vs_consistent.yaml...")
    computed_flourishes = load_beat_vs_consistent(beat_vs_consistent_path)
    print(f"Loaded {len(computed_flourishes)} computed flourishes")

    # Verify by recomputing
    print("\nVerifying by recomputing flourishes...")
    recomputed = compute_flourishes_from_layers(beat_taps, consistent_taps, MATCH_THRESHOLD)
    print(f"Recomputed {len(recomputed)} flourishes")

    if len(recomputed) != len(computed_flourishes):
        print(f"WARNING: Counts differ! Using recomputed version.")
        computed_flourishes = recomputed

    # Get audio duration
    import soundfile as sf
    audio_info = sf.info(audio_path)
    duration = audio_info.duration
    print(f"\nAudio duration: {duration:.2f} seconds")

    # 1. OVERLAP ANALYSIS
    print("\n" + "="*80)
    print("1. OVERLAP ANALYSIS")
    print("="*80)

    overlaps, new_taps, missed = analyze_overlap(
        beat_flourish_taps, computed_flourishes, MATCH_THRESHOLD
    )

    print(f"\nTotal beat-flourish taps: {len(beat_flourish_taps)}")
    print(f"Overlapping with computed flourishes: {len(overlaps)} ({len(overlaps)/len(beat_flourish_taps)*100:.1f}%)")
    print(f"New (not in computed): {len(new_taps)} ({len(new_taps)/len(beat_flourish_taps)*100:.1f}%)")
    print(f"\nComputed flourishes missed by beat-flourish: {len(missed)} ({len(missed)/len(computed_flourishes)*100:.1f}%)")

    # 2. ANALYZE NEW TAPS
    print("\n" + "="*80)
    print("2. WHAT ARE THE NEW TAPS?")
    print("="*80)

    new_taps_analysis = analyze_new_taps_context(
        new_taps, consistent_taps, changes_taps, air_taps, beat_taps, MATCH_THRESHOLD
    )

    # Summarize
    near_consistent = sum(1 for t in new_taps_analysis if t['near_consistent'])
    near_change = sum(1 for t in new_taps_analysis if t['near_change'])
    near_air = sum(1 for t in new_taps_analysis if t['near_air'])
    in_beat = sum(1 for t in new_taps_analysis if t['in_original_beat'])

    print(f"\nOf {len(new_taps)} new taps:")
    print(f"  Near consistent-beat tap: {near_consistent} ({near_consistent/len(new_taps)*100:.1f}%)")
    print(f"  Near a change marker: {near_change} ({near_change/len(new_taps)*100:.1f}%)")
    print(f"  Near an air tap: {near_air} ({near_air/len(new_taps)*100:.1f}%)")
    print(f"  In original beat layer: {in_beat} ({in_beat/len(new_taps)*100:.1f}%)")

    # 3. DENSITY COMPARISON
    print("\n" + "="*80)
    print("3. DENSITY COMPARISON")
    print("="*80)

    density_stats = analyze_density_patterns(
        beat_flourish_taps, computed_flourishes, beat_taps, duration, WINDOW_SIZE
    )

    print(f"\nMean tap density (taps/second):")
    print(f"  Beat-flourish: {density_stats['bf_mean_density']:.2f}")
    print(f"  Computed flourishes: {density_stats['cf_mean_density']:.2f}")
    print(f"  Original beat: {density_stats['beat_mean_density']:.2f}")
    print(f"\nDensity correlation (beat-flourish vs computed): {density_stats['density_correlation']:.3f}")
    print(f"Time points where beat-flourish is:")
    print(f"  Denser: {density_stats['bf_denser_count']} ({density_stats['bf_denser_count']/density_stats['num_time_points']*100:.1f}%)")
    print(f"  Sparser: {density_stats['bf_sparser_count']} ({density_stats['bf_sparser_count']/density_stats['num_time_points']*100:.1f}%)")
    print(f"  Similar: {density_stats['similar_count']} ({density_stats['similar_count']/density_stats['num_time_points']*100:.1f}%)")

    # 4. TEMPORAL PATTERN ANALYSIS
    print("\n" + "="*80)
    print("4. TEMPORAL PATTERN ANALYSIS")
    print("="*80)

    bf_intervals = analyze_intervals(beat_flourish_taps)
    cf_intervals = analyze_intervals(computed_flourishes)
    beat_intervals = analyze_intervals(beat_taps)
    cons_intervals = analyze_intervals(consistent_taps)

    print(f"\nInter-tap intervals (seconds):")
    print(f"\nBeat-flourish:")
    print(f"  Median: {bf_intervals['median']:.3f}s ({60/bf_intervals['median']:.1f} BPM)")
    print(f"  Mean: {bf_intervals['mean']:.3f}s")
    print(f"  Std: {bf_intervals['std']:.3f}s")
    print(f"  CV: {bf_intervals['cv']:.3f}")
    print(f"  Range: {bf_intervals['min']:.3f}s - {bf_intervals['max']:.3f}s")

    print(f"\nComputed flourishes:")
    print(f"  Median: {cf_intervals['median']:.3f}s ({60/cf_intervals['median']:.1f} BPM)")
    print(f"  Mean: {cf_intervals['mean']:.3f}s")
    print(f"  Std: {cf_intervals['std']:.3f}s")
    print(f"  CV: {cf_intervals['cv']:.3f}")
    print(f"  Range: {cf_intervals['min']:.3f}s - {cf_intervals['max']:.3f}s")

    print(f"\nConsistent-beat (for reference):")
    print(f"  Median: {cons_intervals['median']:.3f}s ({60/cons_intervals['median']:.1f} BPM)")
    print(f"  CV: {cons_intervals['cv']:.3f}")

    # 5. CORRELATION WITH EXISTING LAYERS
    print("\n" + "="*80)
    print("5. CORRELATION WITH EXISTING LAYERS")
    print("="*80)

    bf_to_air = correlate_layers(beat_flourish_taps, air_taps, MATCH_THRESHOLD)
    bf_to_consistent = correlate_layers(beat_flourish_taps, consistent_taps, MATCH_THRESHOLD)
    bf_to_changes = correlate_layers(beat_flourish_taps, changes_taps, MATCH_THRESHOLD * 2)
    cf_to_air = correlate_layers(computed_flourishes, air_taps, MATCH_THRESHOLD)

    print(f"\nFraction of beat-flourish taps that align with:")
    print(f"  Air taps: {bf_to_air:.1%}")
    print(f"  Consistent-beat taps: {bf_to_consistent:.1%}")
    print(f"  Change markers: {bf_to_changes:.1%}")

    print(f"\nFor comparison, computed flourishes aligned with air: {cf_to_air:.1%}")

    # 6. VERDICT
    print("\n" + "="*80)
    print("6. VERDICT")
    print("="*80)

    # Determine verdict based on evidence
    overlap_pct = len(overlaps) / len(beat_flourish_taps) * 100
    new_pct = len(new_taps) / len(beat_flourish_taps) * 100
    missed_pct = len(missed) / len(computed_flourishes) * 100

    print(f"\nKey findings:")
    print(f"  - {overlap_pct:.1f}% of beat-flourish taps overlap with computed flourishes")
    print(f"  - {new_pct:.1f}% are genuinely new")
    print(f"  - {missed_pct:.1f}% of computed flourishes are missed by beat-flourish")
    print(f"  - Density correlation: {density_stats['density_correlation']:.3f}")
    print(f"  - Beat-flourish has {bf_intervals['cv']:.2f} CV vs computed {cf_intervals['cv']:.2f} CV")

    if new_pct > 30 or missed_pct > 30:
        verdict = "ADDS SIGNIFICANT NEW INFORMATION"
        explanation = f"Beat-flourish captures {'substantially different' if new_pct > 30 else 'a different perspective on'} flourish moments compared to computed flourishes."
    elif overlap_pct > 70 and abs(density_stats['density_correlation']) > 0.7:
        verdict = "MOSTLY REDUNDANT"
        explanation = "Beat-flourish largely captures the same flourish moments as the computed method."
    else:
        verdict = "PARTIALLY REDUNDANT, SOME NEW INFO"
        explanation = "Beat-flourish overlaps significantly with computed flourishes but captures some unique moments."

    print(f"\n>>> {verdict}")
    print(f"\n{explanation}")

    # Generate detailed report
    print("\n" + "="*80)
    print("Generating detailed report...")
    print("="*80)

    with open(output_path, 'w') as f:
        f.write("# Flourish Comparison Analysis\n\n")
        f.write(f"**Analysis Date:** {Path(__file__).stat().st_mtime}\n\n")
        f.write("## Question\n\n")
        f.write("Does the user's `beat-flourish` annotation layer add genuinely new information ")
        f.write("beyond what we already captured in computed flourishes (beat taps that don't align ")
        f.write("with consistent-beat)?\n\n")

        f.write("## Summary\n\n")
        f.write(f"**Verdict:** {verdict}\n\n")
        f.write(f"{explanation}\n\n")

        f.write("## Datasets\n\n")
        f.write(f"- **Beat-flourish layer:** {len(beat_flourish_taps)} taps (user tapped trying to hit only flourishes)\n")
        f.write(f"- **Computed flourishes:** {len(computed_flourishes)} taps (beat taps >150ms from any consistent-beat tap)\n")
        f.write(f"- **Audio duration:** {duration:.2f} seconds\n\n")

        f.write("## 1. Overlap Analysis\n\n")
        f.write(f"Using ±{MATCH_THRESHOLD*1000:.0f}ms matching window:\n\n")
        f.write(f"- **Overlaps:** {len(overlaps)} taps ({overlap_pct:.1f}%)\n")
        f.write(f"  - These beat-flourish taps matched computed flourishes — we already had this info\n")
        f.write(f"- **New:** {len(new_taps)} taps ({new_pct:.1f}%)\n")
        f.write(f"  - These beat-flourish taps did NOT match computed flourishes — genuinely new\n")
        f.write(f"- **Missed:** {len(missed)} taps ({missed_pct:.1f}%)\n")
        f.write(f"  - Computed flourishes that beat-flourish layer missed\n\n")

        if overlaps:
            f.write("### Sample Overlaps\n\n")
            f.write("| Beat-Flourish | Computed Flourish | Distance |\n")
            f.write("|---------------|-------------------|----------|\n")
            for item in overlaps[:10]:
                f.write(f"| {item['beat_flourish']:.3f}s | {item['computed_flourish']:.3f}s | {item['distance']*1000:.0f}ms |\n")
            f.write("\n")

        f.write("## 2. What Are the New Taps?\n\n")
        f.write(f"Analyzing {len(new_taps)} new beat-flourish taps that don't match computed flourishes:\n\n")
        f.write(f"- **Near consistent-beat:** {near_consistent} ({near_consistent/len(new_taps)*100:.1f}%)\n")
        f.write(f"  - User tapped moments they previously called \"consistent\" as flourishes\n")
        f.write(f"  - Suggests ambiguity or context-dependent perception\n")
        f.write(f"- **Near change markers:** {near_change} ({near_change/len(new_taps)*100:.1f}%)\n")
        f.write(f"  - Flourishes at structural boundaries\n")
        f.write(f"- **Near air taps:** {near_air} ({near_air/len(new_taps)*100:.1f}%)\n")
        f.write(f"  - Airy moments perceived as flourish-worthy\n")
        f.write(f"- **In original beat layer:** {in_beat} ({in_beat/len(new_taps)*100:.1f}%)\n")
        f.write(f"  - Present in original beat but aligned with consistent-beat (so not in computed flourishes)\n\n")

        if new_taps_analysis:
            f.write("### Sample New Taps with Context\n\n")
            f.write("| Beat-Flourish | Dist to Consistent | Dist to Change | Dist to Air | In Beat? |\n")
            f.write("|---------------|-------------------|----------------|-------------|----------|\n")
            for item in new_taps_analysis[:15]:
                f.write(f"| {item['beat_flourish']:.3f}s | ")
                f.write(f"{item['dist_consistent']*1000:.0f}ms | ")
                f.write(f"{item['dist_change']*1000:.0f}ms | ")
                f.write(f"{item['dist_air']*1000:.0f}ms | ")
                f.write(f"{'Yes' if item['in_original_beat'] else 'No'} |\n")
            f.write("\n")

        f.write("## 3. Density Comparison\n\n")
        f.write(f"Using {WINDOW_SIZE}s sliding window:\n\n")
        f.write(f"| Layer | Mean Density (taps/sec) |\n")
        f.write(f"|-------|------------------------|\n")
        f.write(f"| Beat-flourish | {density_stats['bf_mean_density']:.2f} |\n")
        f.write(f"| Computed flourishes | {density_stats['cf_mean_density']:.2f} |\n")
        f.write(f"| Original beat | {density_stats['beat_mean_density']:.2f} |\n\n")

        f.write(f"**Density correlation:** {density_stats['density_correlation']:.3f}\n\n")
        f.write(f"Of {density_stats['num_time_points']} time points:\n")
        f.write(f"- Beat-flourish **denser:** {density_stats['bf_denser_count']} ({density_stats['bf_denser_count']/density_stats['num_time_points']*100:.1f}%)\n")
        f.write(f"- Beat-flourish **sparser:** {density_stats['bf_sparser_count']} ({density_stats['bf_sparser_count']/density_stats['num_time_points']*100:.1f}%)\n")
        f.write(f"- **Similar:** {density_stats['similar_count']} ({density_stats['similar_count']/density_stats['num_time_points']*100:.1f}%)\n\n")

        f.write("## 4. Temporal Pattern Analysis\n\n")
        f.write("### Beat-Flourish Intervals\n\n")
        f.write(f"- **Median:** {bf_intervals['median']:.3f}s ({60/bf_intervals['median']:.1f} BPM)\n")
        f.write(f"- **Mean:** {bf_intervals['mean']:.3f}s ± {bf_intervals['std']:.3f}s\n")
        f.write(f"- **CV:** {bf_intervals['cv']:.3f}\n")
        f.write(f"- **Range:** {bf_intervals['min']:.3f}s to {bf_intervals['max']:.3f}s\n")
        f.write(f"- **Quartiles:** Q25={bf_intervals['q25']:.3f}s, Q75={bf_intervals['q75']:.3f}s\n\n")

        f.write("### Computed Flourishes Intervals\n\n")
        f.write(f"- **Median:** {cf_intervals['median']:.3f}s ({60/cf_intervals['median']:.1f} BPM)\n")
        f.write(f"- **Mean:** {cf_intervals['mean']:.3f}s ± {cf_intervals['std']:.3f}s\n")
        f.write(f"- **CV:** {cf_intervals['cv']:.3f}\n")
        f.write(f"- **Range:** {cf_intervals['min']:.3f}s to {cf_intervals['max']:.3f}s\n\n")

        f.write("### Interpretation\n\n")
        if bf_intervals['cv'] > cf_intervals['cv']:
            f.write("Beat-flourish has **higher variability** (higher CV), suggesting more irregular tapping.\n")
        else:
            f.write("Beat-flourish has **lower variability** (lower CV), suggesting more regular tapping.\n")

        if abs(bf_intervals['median'] - cons_intervals['median']) < 0.1:
            f.write("Beat-flourish median interval is **similar to consistent-beat**, suggesting user fell into a groove.\n")
        else:
            f.write("Beat-flourish median interval **differs from consistent-beat**, suggesting truly off-grid tapping.\n")
        f.write("\n")

        f.write("## 5. Correlation with Existing Layers\n\n")
        f.write(f"Fraction of beat-flourish taps within ±{MATCH_THRESHOLD*1000:.0f}ms of:\n\n")
        f.write(f"- **Air taps:** {bf_to_air:.1%}\n")
        f.write(f"- **Consistent-beat taps:** {bf_to_consistent:.1%}\n")
        f.write(f"- **Change markers:** {bf_to_changes:.1%} (using ±{MATCH_THRESHOLD*2*1000:.0f}ms window)\n\n")
        f.write(f"For comparison:\n")
        f.write(f"- **Computed flourishes aligned with air:** {cf_to_air:.1%}\n\n")

        if bf_to_air > 0.3:
            f.write("Beat-flourish shows **strong correlation with air layer** — airy moments perceived as flourish-heavy.\n\n")

        if bf_to_consistent > 0.2:
            f.write("Beat-flourish shows **unexpected alignment with consistent-beat** — suggests ambiguity in what counts as a flourish.\n\n")

        f.write("## 6. Conclusion\n\n")
        f.write(f"### {verdict}\n\n")
        f.write(f"{explanation}\n\n")

        if new_pct > 30:
            f.write("### Specific New Insights\n\n")
            f.write("The beat-flourish layer captures:\n\n")
            if near_consistent > len(new_taps) * 0.3:
                f.write(f"1. **Ambiguous moments** — {near_consistent} taps near consistent-beat that user now perceives as flourishes\n")
            if near_air > len(new_taps) * 0.3:
                f.write(f"2. **Airy flourishes** — {near_air} taps correlating with the air layer\n")
            if near_change > len(new_taps) * 0.3:
                f.write(f"3. **Transitional flourishes** — {near_change} taps near structural changes\n")
            f.write("\n")

        if missed_pct > 30:
            f.write("### Missed Flourishes\n\n")
            f.write(f"The beat-flourish layer **missed {len(missed)} computed flourishes** ({missed_pct:.1f}%).\n")
            f.write("This suggests:\n")
            f.write("- User's focused flourish-tapping missed some off-grid moments from original beat layer\n")
            f.write("- Or user's perception of \"flourish\" evolved between tapping sessions\n")
            f.write("- Or some computed flourishes weren't actually flourish-worthy to the user\n\n")

            if missed:
                f.write("#### Sample Missed Flourishes\n\n")
                f.write("| Computed Flourish | Nearest Beat-Flourish | Distance |\n")
                f.write("|-------------------|----------------------|----------|\n")
                for item in missed[:10]:
                    f.write(f"| {item['computed_flourish']:.3f}s | ")
                    if item['nearest_beat_flourish'] is not None:
                        f.write(f"{item['nearest_beat_flourish']:.3f}s | {item['distance']*1000:.0f}ms |\n")
                    else:
                        f.write(f"None | N/A |\n")
                f.write("\n")

        if overlap_pct > 70:
            f.write("### Redundancy Assessment\n\n")
            f.write(f"With {overlap_pct:.1f}% overlap, the beat-flourish layer is **largely redundant** ")
            f.write("with computed flourishes. For practical LED applications, the computed method ")
            f.write("(beat taps that don't align with consistent-beat) is likely sufficient.\n\n")

        f.write("## Recommendations\n\n")
        if new_pct > 30 or missed_pct > 30:
            f.write("- **Use beat-flourish as ground truth** — it captures user's current perception of flourishes\n")
            f.write("- **Investigate new taps** — understand what audio features characterize these moments\n")
            f.write("- **Investigate missed flourishes** — understand why user didn't tap them\n")
        else:
            f.write("- **Computed flourishes are sufficient** — beat-flourish doesn't add substantial new information\n")
            f.write("- **Original beat + consistent-beat decomposition works well** for this track\n")

        f.write("\n---\n\n")
        f.write(f"*Analysis threshold: ±{MATCH_THRESHOLD*1000:.0f}ms*\n")
        f.write(f"*Density window: {WINDOW_SIZE}s*\n")

    print(f"\nReport written to: {output_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
