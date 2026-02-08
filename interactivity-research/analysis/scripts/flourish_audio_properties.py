#!/usr/bin/env python3
"""
Analyze audio properties that characterize flourish moments.

Compares flourish taps vs on-beat taps vs rest of track to find
detectable audio signatures for automated flourish detection.
"""

import numpy as np
import librosa
import yaml
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Paths
AUDIO_PATH = Path("/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.wav")
ANNOTATIONS_PATH = Path("/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.annotations.yaml")
OUTPUT_PATH = Path("/Users/KO16K39/Documents/led/interactivity-research/analysis/flourish_audio_properties.md")

# Analysis parameters
HOP_LENGTH = 512
FLOURISH_WINDOW = 0.25  # ±250ms window around flourish taps
BEAT_TOLERANCE = 0.150  # 150ms tolerance for "on-beat" matching

def load_annotations():
    """Load tap annotations from YAML."""
    with open(ANNOTATIONS_PATH) as f:
        data = yaml.safe_load(f)
    return data

def build_flourish_ground_truth(annotations):
    """
    Build confidence-weighted flourish dataset.

    Returns:
        flourishes: dict of {timestamp: confidence_level}
        on_beat: list of consistent-beat timestamps
    """
    # Get consistent beat taps (ground truth for "not flourish")
    consistent_beat = np.array(annotations['consistent-beat'])

    # Get all potential flourish sources
    beat_taps = np.array(annotations['beat'])
    flourish1_taps = np.array(annotations['beat-flourish'])
    flourish2_taps = np.array(annotations['beat-flourish2'])

    flourishes = {}  # {timestamp: confidence}

    # 1. Computed flourishes: beat taps that are >150ms from any consistent-beat tap
    for tap in beat_taps:
        if np.min(np.abs(tap - consistent_beat)) > BEAT_TOLERANCE:
            flourishes[tap] = flourishes.get(tap, 0) + 1

    # 2. beat-flourish taps that are >150ms from consistent-beat
    for tap in flourish1_taps:
        if np.min(np.abs(tap - consistent_beat)) > BEAT_TOLERANCE:
            flourishes[tap] = flourishes.get(tap, 0) + 1

    # 3. beat-flourish2 taps that are >150ms from consistent-beat
    for tap in flourish2_taps:
        if np.min(np.abs(tap - consistent_beat)) > BEAT_TOLERANCE:
            flourishes[tap] = flourishes.get(tap, 0) + 1

    # Merge nearby flourishes (within 50ms)
    merged_flourishes = {}
    sorted_times = sorted(flourishes.keys())
    skip_idx = set()

    for i, t in enumerate(sorted_times):
        if i in skip_idx:
            continue

        # Find all taps within 50ms
        cluster = [t]
        cluster_conf = [flourishes[t]]

        for j in range(i+1, len(sorted_times)):
            if sorted_times[j] - t < 0.050:
                cluster.append(sorted_times[j])
                cluster_conf.append(flourishes[sorted_times[j]])
                skip_idx.add(j)
            else:
                break

        # Use mean time, sum confidence
        merged_flourishes[np.mean(cluster)] = sum(cluster_conf)

    print(f"Found {len(merged_flourishes)} unique flourish moments")
    print(f"  Confidence 1: {sum(1 for c in merged_flourishes.values() if c == 1)}")
    print(f"  Confidence 2: {sum(1 for c in merged_flourishes.values() if c == 2)}")
    print(f"  Confidence 3: {sum(1 for c in merged_flourishes.values() if c == 3)}")

    return merged_flourishes, consistent_beat

def extract_features(y, sr):
    """
    Extract comprehensive audio features at high resolution.

    Returns dict of features, each aligned to same time grid.
    """
    features = {}

    # Frame-level spectral features
    spec = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )[0]
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )[0]
    features['spectral_flatness'] = librosa.feature.spectral_flatness(
        y=y, hop_length=HOP_LENGTH
    )[0]
    features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH
    ), axis=0)
    features['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    # HPSS
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    h_rms = librosa.feature.rms(y=y_harmonic, hop_length=HOP_LENGTH)[0]
    p_rms = librosa.feature.rms(y=y_percussive, hop_length=HOP_LENGTH)[0]
    features['harmonic_ratio'] = h_rms / (h_rms + p_rms + 1e-8)
    features['percussive_energy'] = p_rms

    # Spectral flux (frame-to-frame change)
    spectral_flux = np.zeros(spec.shape[1])
    spectral_flux[1:] = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
    features['spectral_flux'] = spectral_flux

    # Onset strength
    features['onset_strength'] = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )

    # Onset density (onsets per second in ±1s window)
    onset_env = features['onset_strength']
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

    onset_density = np.zeros(len(onset_env))
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP_LENGTH)
    for i, t in enumerate(times):
        nearby_onsets = np.sum((onset_times >= t - 1.0) & (onset_times <= t + 1.0))
        onset_density[i] = nearby_onsets / 2.0  # per second
    features['onset_density'] = onset_density

    # Spectral novelty (difference from recent average)
    lookback_frames = int(2.0 * sr / HOP_LENGTH)  # 2 seconds
    spectral_novelty = np.zeros(spec.shape[1])
    for i in range(spec.shape[1]):
        start = max(0, i - lookback_frames)
        if start < i:
            recent_mean = np.mean(spec[:, start:i], axis=1, keepdims=True)
            spectral_novelty[i] = np.sqrt(np.sum((spec[:, i:i+1] - recent_mean)**2))
    features['spectral_novelty'] = spectral_novelty

    # Energy derivative
    energy_deriv = np.zeros(len(features['rms']))
    energy_deriv[1:] = np.diff(features['rms'])
    features['energy_derivative'] = energy_deriv

    # Centroid derivative
    centroid_deriv = np.zeros(len(features['spectral_centroid']))
    centroid_deriv[1:] = np.diff(features['spectral_centroid'])
    features['centroid_derivative'] = centroid_deriv

    # Active frequency bands (sparseness)
    # Count bins with energy > 5% of max per frame
    active_bands = np.sum(spec > 0.05 * np.max(spec, axis=0, keepdims=True), axis=0)
    features['active_bands'] = active_bands

    # Bass-to-mids ratio
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_mels=64, fmin=20, fmax=8000
    )
    bass_energy = np.sum(mel_spec[0:8, :], axis=0)  # ~20-200 Hz
    mids_energy = np.sum(mel_spec[8:32, :], axis=0)  # ~200-2000 Hz
    features['bass_to_mids'] = bass_energy / (mids_energy + 1e-8)

    # High frequency energy ratio
    highs_energy = np.sum(mel_spec[32:, :], axis=0)  # >2000 Hz
    features['high_freq_ratio'] = highs_energy / (np.sum(mel_spec, axis=0) + 1e-8)

    # Zero crossing rate
    features['zcr'] = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]

    return features, times

def get_feature_windows(features, times, timestamps, window_size=FLOURISH_WINDOW):
    """
    Extract feature values in windows around timestamps.

    Returns list of feature vectors, one per timestamp.
    """
    feature_vectors = defaultdict(list)

    for ts in timestamps:
        # Find frames within window
        mask = (times >= ts - window_size) & (times <= ts + window_size)

        if np.sum(mask) > 0:
            for name, values in features.items():
                # Use max for peak-sensitive features, mean for others
                if name in ['spectral_flux', 'onset_strength', 'spectral_novelty',
                           'energy_derivative', 'onset_density']:
                    feature_vectors[name].append(np.max(values[mask]))
                else:
                    feature_vectors[name].append(np.mean(values[mask]))

    return feature_vectors

def cohen_d(x, y):
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def compare_distributions(flourish_vals, comparison_vals, feature_name):
    """
    Compare distributions and return statistics.
    """
    # Effect size
    effect = cohen_d(flourish_vals, comparison_vals)

    # Statistical test
    if len(flourish_vals) > 0 and len(comparison_vals) > 0:
        stat, pval = stats.mannwhitneyu(flourish_vals, comparison_vals, alternative='two-sided')
    else:
        pval = 1.0

    return {
        'feature': feature_name,
        'flourish_mean': np.mean(flourish_vals),
        'flourish_std': np.std(flourish_vals),
        'comparison_mean': np.mean(comparison_vals),
        'comparison_std': np.std(comparison_vals),
        'effect_size': effect,
        'p_value': pval,
        'significant': pval < 0.01
    }

def main():
    print("Loading audio and annotations...")
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    annotations = load_annotations()

    print("\nBuilding flourish ground truth...")
    flourishes, on_beat = build_flourish_ground_truth(annotations)

    print(f"\nFound {len(on_beat)} on-beat taps")
    print(f"Found {len(flourishes)} flourish moments")

    print("\nExtracting audio features...")
    features, times = extract_features(y, sr)
    print(f"Computed {len(features)} features over {len(times)} frames")
    print(f"Feature names: {list(features.keys())}")

    # Get feature windows for different sets
    print("\nExtracting feature windows...")
    flourish_times = list(flourishes.keys())
    high_conf_flourishes = [t for t, c in flourishes.items() if c >= 2]

    flourish_features = get_feature_windows(features, times, flourish_times)
    high_conf_features = get_feature_windows(features, times, high_conf_flourishes)
    on_beat_features = get_feature_windows(features, times, on_beat)

    # Get random non-flourish, non-beat samples for baseline
    all_event_times = set(flourish_times) | set(on_beat)
    non_event_times = []
    for t in times[::10]:  # Sample every 10th frame
        if not any(abs(t - et) < FLOURISH_WINDOW for et in all_event_times):
            non_event_times.append(t)
    non_event_features = get_feature_windows(features, times, non_event_times[:100])

    print(f"  Flourish windows: {len(flourish_times)}")
    print(f"  High-confidence flourish windows: {len(high_conf_flourishes)}")
    print(f"  On-beat windows: {len(on_beat)}")
    print(f"  Non-event baseline windows: {len(non_event_times[:100])}")

    # Compare flourishes vs on-beat (key comparison!)
    print("\n=== Comparing Flourishes vs On-Beat ===")
    flourish_vs_beat = []
    for name in features.keys():
        result = compare_distributions(
            flourish_features[name],
            on_beat_features[name],
            name
        )
        flourish_vs_beat.append(result)

    flourish_vs_beat.sort(key=lambda x: abs(x['effect_size']), reverse=True)

    # Compare flourishes vs baseline
    print("\n=== Comparing Flourishes vs Non-Events ===")
    flourish_vs_baseline = []
    for name in features.keys():
        result = compare_distributions(
            flourish_features[name],
            non_event_features[name],
            name
        )
        flourish_vs_baseline.append(result)

    flourish_vs_baseline.sort(key=lambda x: abs(x['effect_size']), reverse=True)

    # Compare high-confidence flourishes vs on-beat
    print("\n=== High-Confidence Flourishes vs On-Beat ===")
    high_conf_vs_beat = []
    for name in features.keys():
        result = compare_distributions(
            high_conf_features[name],
            on_beat_features[name],
            name
        )
        high_conf_vs_beat.append(result)

    high_conf_vs_beat.sort(key=lambda x: abs(x['effect_size']), reverse=True)

    # Load air analysis if it exists for comparison
    air_features = None
    air_report_path = Path("/Users/KO16K39/Documents/led/interactivity-research/analysis/air_audio_properties.md")
    if air_report_path.exists():
        print("\nLoading air analysis for comparison...")
        # We'll note this in the report

    # Generate report
    print("\nGenerating report...")
    with open(OUTPUT_PATH, 'w') as f:
        f.write("# Flourish Audio Properties Analysis\n\n")
        f.write("Analysis of audio features that characterize musical flourishes — ")
        f.write("off-beat moments that humans naturally emphasize.\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Audio file**: {AUDIO_PATH.name}\n")
        f.write(f"- **Duration**: {len(y)/sr:.1f} seconds\n")
        f.write(f"- **Flourish moments analyzed**: {len(flourishes)} total\n")
        f.write(f"  - High confidence (2+ sources): {len(high_conf_flourishes)}\n")
        f.write(f"  - Confidence distribution: ")
        for c in [1, 2, 3]:
            count = sum(1 for conf in flourishes.values() if conf == c)
            f.write(f"{c}={count} ")
        f.write("\n")
        f.write(f"- **On-beat moments**: {len(on_beat)}\n")
        f.write(f"- **Feature window**: ±{FLOURISH_WINDOW*1000:.0f}ms\n")
        f.write(f"- **Features computed**: {len(features)}\n\n")

        # Key findings
        f.write("## Key Findings\n\n")

        # Top discriminating features
        top_features = flourish_vs_beat[:5]
        f.write("### Top Features Distinguishing Flourishes from On-Beat\n\n")
        for i, result in enumerate(top_features, 1):
            direction = "higher" if result['flourish_mean'] > result['comparison_mean'] else "lower"
            f.write(f"{i}. **{result['feature']}** (effect size: {result['effect_size']:.3f})\n")
            f.write(f"   - Flourish: {result['flourish_mean']:.4f} ± {result['flourish_std']:.4f}\n")
            f.write(f"   - On-beat: {result['comparison_mean']:.4f} ± {result['comparison_std']:.4f}\n")
            f.write(f"   - Flourishes are **{direction}** (p={result['p_value']:.4f})\n\n")

        # Characterization
        f.write("### Flourish Characterization\n\n")
        f.write("Based on the top features, flourishes are characterized by:\n\n")

        # Analyze top features
        high_in_flourish = [r for r in top_features if r['flourish_mean'] > r['comparison_mean']]
        low_in_flourish = [r for r in top_features if r['flourish_mean'] < r['comparison_mean']]

        if high_in_flourish:
            f.write("**Higher than on-beat:**\n")
            for r in high_in_flourish:
                pct_increase = ((r['flourish_mean'] - r['comparison_mean']) / r['comparison_mean']) * 100
                f.write(f"- {r['feature']}: {pct_increase:+.1f}% increase\n")
            f.write("\n")

        if low_in_flourish:
            f.write("**Lower than on-beat:**\n")
            for r in low_in_flourish:
                pct_decrease = ((r['flourish_mean'] - r['comparison_mean']) / r['comparison_mean']) * 100
                f.write(f"- {r['feature']}: {pct_decrease:+.1f}% change\n")
            f.write("\n")

        # Detection strategy
        f.write("### Proposed Detection Strategy\n\n")

        # Find features with large effect and significance
        strong_predictors = [r for r in flourish_vs_beat if abs(r['effect_size']) > 0.5 and r['significant']]

        if strong_predictors:
            f.write(f"**Strong predictors** (|effect| > 0.5, p < 0.01): {len(strong_predictors)} features\n\n")
            f.write("Recommended multi-feature detection rule:\n\n")
            f.write("```python\n")
            f.write("def detect_flourish(features):\n")
            f.write("    score = 0\n")
            for r in strong_predictors[:3]:
                if r['flourish_mean'] > r['comparison_mean']:
                    threshold = r['comparison_mean'] + 0.5 * r['comparison_std']
                    f.write(f"    if features['{r['feature']}'] > {threshold:.4f}:\n")
                    f.write(f"        score += {abs(r['effect_size']):.2f}  # effect size as weight\n")
                else:
                    threshold = r['comparison_mean'] - 0.5 * r['comparison_std']
                    f.write(f"    if features['{r['feature']}'] < {threshold:.4f}:\n")
                    f.write(f"        score += {abs(r['effect_size']):.2f}\n")
            f.write("    return score > 1.0  # threshold to tune\n")
            f.write("```\n\n")
        else:
            f.write("No single feature shows very strong discrimination. ")
            f.write("Flourishes may require multi-dimensional feature space or ")
            f.write("temporal context (preceding/following frames).\n\n")

        # Full comparison tables
        f.write("## Detailed Results\n\n")

        f.write("### Flourishes vs On-Beat\n\n")
        f.write("Features ranked by effect size (Cohen's d).\n\n")
        f.write("| Rank | Feature | Effect Size | Flourish Mean | On-Beat Mean | p-value | Sig |\n")
        f.write("|------|---------|-------------|---------------|--------------|---------|-----|\n")
        for i, r in enumerate(flourish_vs_beat, 1):
            sig = "✓" if r['significant'] else ""
            f.write(f"| {i} | {r['feature']} | {r['effect_size']:+.3f} | ")
            f.write(f"{r['flourish_mean']:.4f} | {r['comparison_mean']:.4f} | ")
            f.write(f"{r['p_value']:.4f} | {sig} |\n")
        f.write("\n")

        f.write("### High-Confidence Flourishes vs On-Beat\n\n")
        f.write("Same analysis using only flourishes with 2+ source agreement.\n\n")
        f.write("| Rank | Feature | Effect Size | High-Conf Mean | On-Beat Mean | p-value | Sig |\n")
        f.write("|------|---------|-------------|----------------|--------------|---------|-----|\n")
        for i, r in enumerate(high_conf_vs_beat, 1):
            sig = "✓" if r['significant'] else ""
            f.write(f"| {i} | {r['feature']} | {r['effect_size']:+.3f} | ")
            f.write(f"{r['flourish_mean']:.4f} | {r['comparison_mean']:.4f} | ")
            f.write(f"{r['p_value']:.4f} | {sig} |\n")
        f.write("\n")

        f.write("### Flourishes vs Non-Events (Baseline)\n\n")
        f.write("Comparing flourishes to random non-event moments.\n\n")
        f.write("| Rank | Feature | Effect Size | Flourish Mean | Baseline Mean | p-value | Sig |\n")
        f.write("|------|---------|-------------|---------------|---------------|---------|-----|\n")
        for i, r in enumerate(flourish_vs_baseline, 1):
            sig = "✓" if r['significant'] else ""
            f.write(f"| {i} | {r['feature']} | {r['effect_size']:+.3f} | ")
            f.write(f"{r['flourish_mean']:.4f} | {r['comparison_mean']:.4f} | ")
            f.write(f"{r['p_value']:.4f} | {sig} |\n")
        f.write("\n")

        # Comparison to air taps
        f.write("## Comparison to Air Taps\n\n")
        if air_report_path.exists():
            f.write("See `air_audio_properties.md` for detailed air tap analysis.\n\n")
            f.write("**Key question**: Do flourishes and air moments share acoustic properties?\n\n")
            f.write("Both represent 'moments worth highlighting' but may differ:\n")
            f.write("- Air: sustained quality, texture, atmosphere\n")
            f.write("- Flourish: transient events, rhythmic surprises, fills\n\n")
            f.write("Compare the top features from both analyses to see overlap.\n\n")
        else:
            f.write("Air tap analysis not yet available. Run `air_audio_properties.py` for comparison.\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        f.write("### What Makes a Flourish?\n\n")

        # Categorize top features
        spectral_features = [r for r in top_features if 'spectral' in r['feature'] or 'centroid' in r['feature']]
        temporal_features = [r for r in top_features if 'onset' in r['feature'] or 'flux' in r['feature'] or 'derivative' in r['feature']]
        energy_features = [r for r in top_features if 'rms' in r['feature'] or 'energy' in r['feature'] or 'percussive' in r['feature']]

        if spectral_features:
            f.write("**Spectral characteristics**:\n")
            for r in spectral_features:
                direction = "brighter/wider" if r['flourish_mean'] > r['comparison_mean'] else "darker/narrower"
                f.write(f"- {r['feature']}: {direction} spectrum during flourishes\n")
            f.write("\n")

        if temporal_features:
            f.write("**Temporal characteristics**:\n")
            for r in temporal_features:
                direction = "more" if r['flourish_mean'] > r['comparison_mean'] else "less"
                f.write(f"- {r['feature']}: {direction} activity/change during flourishes\n")
            f.write("\n")

        if energy_features:
            f.write("**Energy characteristics**:\n")
            for r in energy_features:
                direction = "louder" if r['flourish_mean'] > r['comparison_mean'] else "quieter"
                f.write(f"- {r['feature']}: {direction} during flourishes\n")
            f.write("\n")

        f.write("### Flourish vs Beat: The Core Distinction\n\n")
        f.write("On-beat moments are **predictable, periodic, metronomic**.\n")
        f.write("Flourish moments are **surprising, non-periodic, noteworthy**.\n\n")
        f.write("The acoustic difference between them tells us what makes a moment 'flourish-worthy':\n\n")

        # Find the most discriminating features
        best_discriminator = flourish_vs_beat[0]
        f.write(f"**Primary acoustic signature**: {best_discriminator['feature']}\n")
        f.write(f"- Effect size: {best_discriminator['effect_size']:.3f}\n")
        f.write(f"- This is a {'LARGE' if abs(best_discriminator['effect_size']) > 0.8 else 'MEDIUM' if abs(best_discriminator['effect_size']) > 0.5 else 'SMALL'} effect\n\n")

        # Generalizability
        f.write("### Generalizability to Other Songs\n\n")
        f.write("These findings are based on Tool's 'Opiate Intro' (psych rock, heavy, complex rhythms).\n\n")
        f.write("**Likely to generalize**:\n")
        f.write("- High onset strength/spectral flux at flourish moments (transient events)\n")
        f.write("- Spectral novelty (unexpected timbral changes)\n")
        f.write("- Percussive energy spikes (cymbal crashes, fills)\n\n")
        f.write("**May be genre-specific**:\n")
        f.write("- Exact threshold values (will vary by production style)\n")
        f.write("- Bass-to-mids ratio (depends on mix balance)\n")
        f.write("- Harmonic ratio (varies by instrumentation)\n\n")
        f.write("**Recommendation**: Test on multiple genres (electronic, ambient, jazz) to find universal features.\n\n")

        # Next steps
        f.write("## Next Steps\n\n")
        f.write("1. **Validate detection rule** on held-out segments of this track\n")
        f.write("2. **Test on other songs** in the audio-segments catalog\n")
        f.write("3. **Compare to air tap features** to see if they're orthogonal or overlapping\n")
        f.write("4. **Temporal context**: Do flourishes need preceding quiet/steady state?\n")
        f.write("5. **Multi-scale analysis**: Are there micro-flourishes vs macro-flourishes?\n")
        f.write("6. **LED mapping**: How should flourish intensity map to LED effects?\n\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write(f"- **Ground truth**: {len(flourishes)} flourish moments from 3 sources\n")
        f.write(f"  - Computed from free-form beat taps (off-beat taps)\n")
        f.write(f"  - beat-flourish intentional annotation layer\n")
        f.write(f"  - beat-flourish2 second annotation layer\n")
        f.write(f"  - Confidence = number of agreeing sources (1-3)\n")
        f.write(f"- **Comparison sets**:\n")
        f.write(f"  - {len(on_beat)} on-beat moments (consistent-beat taps)\n")
        f.write(f"  - {len(non_event_times[:100])} random non-event moments (baseline)\n")
        f.write(f"- **Feature extraction**: {len(features)} features at {sr} Hz, hop={HOP_LENGTH}\n")
        f.write(f"- **Feature window**: ±{FLOURISH_WINDOW*1000:.0f}ms around each tap\n")
        f.write(f"  - Peak value for onset/flux/novelty features\n")
        f.write(f"  - Mean value for spectral/energy features\n")
        f.write(f"- **Statistics**: Cohen's d effect size, Mann-Whitney U test\n")
        f.write(f"- **Significance threshold**: p < 0.01\n\n")

    print(f"\n✓ Report saved to {OUTPUT_PATH}")
    print("\nTop 5 features discriminating flourishes from on-beat:")
    for i, r in enumerate(flourish_vs_beat[:5], 1):
        print(f"  {i}. {r['feature']}: effect={r['effect_size']:+.3f}, p={r['p_value']:.4f}")

if __name__ == "__main__":
    main()
