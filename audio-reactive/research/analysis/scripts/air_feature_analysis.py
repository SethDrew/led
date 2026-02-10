#!/usr/bin/env python3
"""
Analyze audio features that correlate with "airiness" annotations.

Compares frame-level and contextual features between air-tapped moments
and the rest of the track to identify which features predict the functional
sense of "airiness as preparation/transition."
"""

import numpy as np
import librosa
import yaml
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
AUDIO_PATH = Path("/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.wav")
ANNOTATIONS_PATH = Path("/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.annotations.yaml")
OUTPUT_MD = Path("/Users/KO16K39/Documents/led/audio-reactive/research/analysis/air_feature_analysis.md")
OUTPUT_YAML = Path("/Users/KO16K39/Documents/led/audio-reactive/research/analysis/air_feature_analysis.yaml")

# Ensure output directory exists
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

# Parameters
HOP_LENGTH = 512  # ~11.6ms at 44.1kHz
WINDOW_MS = 250  # Window around each tap to consider "airy"
LONG_CONTEXT_SEC = 2.0  # For novelty detection
ANTICIPATION_SEC = 2.5  # Look-ahead for energy trajectory

def load_annotations():
    """Load air tap annotations from YAML."""
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = yaml.safe_load(f)

    # Extract air layer taps - flat structure with 'air' key at root
    air_taps = data.get('air', [])

    return air_taps

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def compute_spectral_flatness(y, sr, hop_length):
    """Compute spectral flatness (noise-like vs tonal)."""
    return librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

def compute_spectral_centroid(y, sr, hop_length):
    """Compute spectral centroid (brightness)."""
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

def compute_spectral_bandwidth(y, sr, hop_length):
    """Compute spectral bandwidth (spread of frequency content)."""
    return librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]

def compute_rms_energy(y, hop_length):
    """Compute RMS energy (overall loudness)."""
    return librosa.feature.rms(y=y, hop_length=hop_length)[0]

def compute_harmonic_ratio(y, sr, hop_length):
    """Compute ratio of harmonic to percussive energy."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Compute RMS for each
    rms_harmonic = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    rms_percussive = librosa.feature.rms(y=y_percussive, hop_length=hop_length)[0]

    # Ratio (avoid division by zero)
    total_energy = rms_harmonic + rms_percussive + 1e-10
    harmonic_ratio = rms_harmonic / total_energy

    return harmonic_ratio

def compute_spectral_contrast(y, sr, hop_length):
    """Compute mean spectral contrast across bands."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    return np.mean(contrast, axis=0)  # Average across bands

def compute_sparseness(y, sr, hop_length):
    """
    Count how many mel frequency bands have significant energy.
    Fewer active bands = more isolated/sparse sound.
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # For each frame, count bands above threshold (within 10dB of frame max)
    sparseness = []
    for frame in mel_spec_db.T:
        frame_max = np.max(frame)
        significant_bands = np.sum(frame > (frame_max - 10))
        sparseness.append(significant_bands)

    return np.array(sparseness)

def compute_novelty(y, sr, hop_length):
    """
    Compute spectral novelty using self-similarity.
    High novelty = this spectral pattern hasn't appeared recently.
    """
    # Use chromagram for timbral novelty
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Compute self-similarity matrix
    rec_matrix = librosa.segment.recurrence_matrix(
        chroma,
        mode='affinity',
        metric='cosine',
        width=int(LONG_CONTEXT_SEC * sr / hop_length)  # 2-second context
    )

    # Novelty is inverse of self-similarity
    # For each frame, average similarity to recent frames
    novelty = 1 - np.mean(rec_matrix, axis=0)

    return novelty

def compute_onset_density(y, sr, hop_length):
    """
    Count onsets in a ±1 second window around each frame.
    Low density = more space/openness.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units='frames')

    n_frames = len(onset_env)
    window_frames = int(1.0 * sr / hop_length)  # 1 second window

    onset_density = np.zeros(n_frames)
    for i in range(n_frames):
        start = max(0, i - window_frames)
        end = min(n_frames, i + window_frames)
        onset_density[i] = np.sum((onsets >= start) & (onsets < end))

    return onset_density

def compute_energy_trajectory(y, sr, hop_length):
    """
    Compute future energy increase.
    Is energy INCREASING in the next 2-3 seconds?
    Captures "preparing for what comes next."
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    lookahead_frames = int(ANTICIPATION_SEC * sr / hop_length)
    n_frames = len(rms)

    energy_trajectory = np.zeros(n_frames)
    for i in range(n_frames):
        current_energy = rms[i]
        # Look ahead and compute max energy increase
        future_end = min(n_frames, i + lookahead_frames)
        if future_end > i:
            future_max = np.max(rms[i:future_end])
            energy_trajectory[i] = future_max - current_energy
        else:
            energy_trajectory[i] = 0

    return energy_trajectory

def compute_active_sources(y, sr, hop_length):
    """
    Estimate number of active sound sources using spectral peaks.
    More peaks = more instruments/layers.
    """
    spec = np.abs(librosa.stft(y, hop_length=hop_length))

    active_sources = []
    for frame in spec.T:
        # Find peaks in spectrum
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(frame, height=np.max(frame) * 0.1)
        active_sources.append(len(peaks))

    return np.array(active_sources)

def frames_to_time(frames, sr, hop_length):
    """Convert frame indices to time in seconds."""
    return frames * hop_length / sr

def time_to_frames(times, sr, hop_length):
    """Convert time in seconds to frame indices."""
    return np.round(np.array(times) * sr / hop_length).astype(int)

def get_tap_windows(air_taps, sr, hop_length, n_frames):
    """
    Get frame indices for windows around each tap.
    Returns (cluster1_frames, cluster2_frames, all_air_frames)
    """
    window_frames = int((WINDOW_MS / 1000.0) * sr / hop_length)

    # Separate into clusters based on time
    cluster1_taps = [t for t in air_taps if t < 10]  # First ~10 seconds
    cluster2_taps = [t for t in air_taps if t >= 10]

    def get_window_frames(taps):
        frames = set()
        for tap in taps:
            center = int(tap * sr / hop_length)
            for f in range(max(0, center - window_frames), min(n_frames, center + window_frames)):
                frames.add(f)
        return frames

    cluster1_frames = get_window_frames(cluster1_taps)
    cluster2_frames = get_window_frames(cluster2_taps)
    all_air_frames = cluster1_frames | cluster2_frames

    return cluster1_frames, cluster2_frames, all_air_frames

def analyze_feature(feature_values, air_frames, non_air_frames, feature_name):
    """Analyze a single feature comparing air vs non-air regions."""
    air_vals = feature_values[list(air_frames)]
    non_air_vals = feature_values[list(non_air_frames)]

    # Statistics
    air_mean = np.mean(air_vals)
    air_std = np.std(air_vals)
    non_air_mean = np.mean(non_air_vals)
    non_air_std = np.std(non_air_vals)

    # Effect size
    effect_size = cohens_d(air_vals, non_air_vals)

    # T-test
    t_stat, p_value = stats.ttest_ind(air_vals, non_air_vals)

    return {
        'feature': feature_name,
        'air_mean': float(air_mean),
        'air_std': float(air_std),
        'non_air_mean': float(non_air_mean),
        'non_air_std': float(non_air_std),
        'effect_size': float(effect_size),
        'p_value': float(p_value)
    }

def main():
    print("Loading audio and annotations...")
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    air_taps = load_annotations()

    print(f"Found {len(air_taps)} air taps")
    print(f"Audio: {len(y)/sr:.2f} seconds at {sr} Hz")

    # Compute all features
    print("\nComputing features...")

    print("  1. Spectral flatness...")
    flatness = compute_spectral_flatness(y, sr, HOP_LENGTH)

    print("  2. Spectral centroid...")
    centroid = compute_spectral_centroid(y, sr, HOP_LENGTH)

    print("  3. Spectral bandwidth...")
    bandwidth = compute_spectral_bandwidth(y, sr, HOP_LENGTH)

    print("  4. RMS energy...")
    rms = compute_rms_energy(y, HOP_LENGTH)

    print("  5. Harmonic ratio...")
    harmonic_ratio = compute_harmonic_ratio(y, sr, HOP_LENGTH)

    print("  6. Spectral contrast...")
    contrast = compute_spectral_contrast(y, sr, HOP_LENGTH)

    print("  7. Sparseness...")
    sparseness = compute_sparseness(y, sr, HOP_LENGTH)

    print("  8. Novelty...")
    novelty = compute_novelty(y, sr, HOP_LENGTH)

    print("  9. Onset density...")
    onset_density = compute_onset_density(y, sr, HOP_LENGTH)

    print("  10. Energy trajectory...")
    energy_trajectory = compute_energy_trajectory(y, sr, HOP_LENGTH)

    print("  11. Active sources...")
    active_sources = compute_active_sources(y, sr, HOP_LENGTH)

    # Ensure all features have same length (take minimum)
    n_frames = min(len(flatness), len(centroid), len(bandwidth), len(rms),
                   len(harmonic_ratio), len(contrast), len(sparseness),
                   len(novelty), len(onset_density), len(energy_trajectory),
                   len(active_sources))

    # Truncate all to same length
    features = {
        'spectral_flatness': flatness[:n_frames],
        'spectral_centroid': centroid[:n_frames],
        'spectral_bandwidth': bandwidth[:n_frames],
        'rms_energy': rms[:n_frames],
        'harmonic_ratio': harmonic_ratio[:n_frames],
        'spectral_contrast': contrast[:n_frames],
        'sparseness': sparseness[:n_frames],
        'novelty': novelty[:n_frames],
        'onset_density': onset_density[:n_frames],
        'energy_trajectory': energy_trajectory[:n_frames],
        'active_sources': active_sources[:n_frames]
    }

    # Get frame windows
    print("\nIdentifying air-tapped and non-air regions...")
    cluster1_frames, cluster2_frames, all_air_frames = get_tap_windows(air_taps, sr, HOP_LENGTH, n_frames)
    non_air_frames = set(range(n_frames)) - all_air_frames

    print(f"  Cluster 1 (early): {len(cluster1_frames)} frames")
    print(f"  Cluster 2 (late): {len(cluster2_frames)} frames")
    print(f"  All air: {len(all_air_frames)} frames ({100*len(all_air_frames)/n_frames:.1f}%)")
    print(f"  Non-air: {len(non_air_frames)} frames")

    # Analyze each feature
    print("\nAnalyzing features...")

    results = {
        'all_air': [],
        'cluster1': [],
        'cluster2': []
    }

    for feat_name, feat_values in features.items():
        # All air vs non-air
        results['all_air'].append(
            analyze_feature(feat_values, all_air_frames, non_air_frames, feat_name)
        )

        # Cluster 1 vs non-air
        results['cluster1'].append(
            analyze_feature(feat_values, cluster1_frames, non_air_frames, feat_name)
        )

        # Cluster 2 vs non-air
        results['cluster2'].append(
            analyze_feature(feat_values, cluster2_frames, non_air_frames, feat_name)
        )

    # Sort by absolute effect size
    for key in results:
        results[key].sort(key=lambda x: abs(x['effect_size']), reverse=True)

    # Generate markdown report
    print("\nGenerating report...")

    report = []
    report.append("# Airiness Feature Analysis\n")
    report.append("## Overview\n")
    report.append(f"- **Audio file**: {AUDIO_PATH.name}\n")
    report.append(f"- **Duration**: {len(y)/sr:.2f} seconds\n")
    report.append(f"- **Air taps**: {len(air_taps)} annotations\n")
    report.append(f"- **Analysis window**: ±{WINDOW_MS}ms around each tap\n")

    cluster1_taps = [t for t in air_taps if t < 10]
    cluster2_taps = [t for t in air_taps if t >= 10]

    if cluster1_taps:
        report.append(f"- **Cluster 1** (early guitar trail-off): {len(cluster1_taps)} taps at {min(cluster1_taps):.2f}-{max(cluster1_taps):.2f}s\n")
    else:
        report.append(f"- **Cluster 1** (early guitar trail-off): 0 taps\n")

    if cluster2_taps:
        report.append(f"- **Cluster 2** (vocal build): {len(cluster2_taps)} taps at {min(cluster2_taps):.2f}-{max(cluster2_taps):.2f}s\n")
    else:
        report.append(f"- **Cluster 2** (vocal build): 0 taps\n")

    report.append("\n## Feature Rankings (by Effect Size)\n")
    report.append("\n### All Air Regions vs Non-Air\n")
    report.append("Features ranked by how well they discriminate air-tapped moments from the rest of the track.\n\n")
    report.append("| Rank | Feature | Air Mean | Air Std | Non-Air Mean | Non-Air Std | Effect Size | p-value |\n")
    report.append("|------|---------|----------|---------|--------------|-------------|-------------|----------|\n")

    for i, r in enumerate(results['all_air'], 1):
        report.append(f"| {i} | **{r['feature']}** | {r['air_mean']:.4f} | {r['air_std']:.4f} | "
                     f"{r['non_air_mean']:.4f} | {r['non_air_std']:.4f} | **{r['effect_size']:.3f}** | {r['p_value']:.4f} |\n")

    report.append("\n### Cluster 1 (Early Guitar) vs Non-Air\n")
    report.append("| Rank | Feature | Air Mean | Non-Air Mean | Effect Size | p-value |\n")
    report.append("|------|---------|----------|--------------|-------------|----------|\n")

    for i, r in enumerate(results['cluster1'], 1):
        report.append(f"| {i} | **{r['feature']}** | {r['air_mean']:.4f} | "
                     f"{r['non_air_mean']:.4f} | **{r['effect_size']:.3f}** | {r['p_value']:.4f} |\n")

    report.append("\n### Cluster 2 (Vocal Build) vs Non-Air\n")
    report.append("| Rank | Feature | Air Mean | Non-Air Mean | Effect Size | p-value |\n")
    report.append("|------|---------|----------|--------------|-------------|----------|\n")

    for i, r in enumerate(results['cluster2'], 1):
        report.append(f"| {i} | **{r['feature']}** | {r['air_mean']:.4f} | "
                     f"{r['non_air_mean']:.4f} | **{r['effect_size']:.3f}** | {r['p_value']:.4f} |\n")

    # Interpretation
    report.append("\n## Interpretation\n")

    report.append("\n### Top Predictive Features (Shared Function)\n")
    report.append("Features that appear in top 5 for BOTH clusters suggest they capture the functional essence of 'airiness':\n\n")

    top_all = set([r['feature'] for r in results['all_air'][:5]])
    top_c1 = set([r['feature'] for r in results['cluster1'][:5]])
    top_c2 = set([r['feature'] for r in results['cluster2'][:5]])
    shared = top_c1 & top_c2

    if shared:
        for feat in shared:
            # Get effect sizes from both clusters
            c1_effect = next(r['effect_size'] for r in results['cluster1'] if r['feature'] == feat)
            c2_effect = next(r['effect_size'] for r in results['cluster2'] if r['feature'] == feat)
            all_effect = next(r['effect_size'] for r in results['all_air'] if r['feature'] == feat)

            c1_mean = next(r['air_mean'] for r in results['cluster1'] if r['feature'] == feat)
            c2_mean = next(r['air_mean'] for r in results['cluster2'] if r['feature'] == feat)
            non_air = next(r['non_air_mean'] for r in results['all_air'] if r['feature'] == feat)

            direction = "HIGHER" if all_effect > 0 else "LOWER"

            report.append(f"- **{feat}**: Effect size = {all_effect:.3f} (C1: {c1_effect:.3f}, C2: {c2_effect:.3f})\n")
            report.append(f"  - Air regions: {c1_mean:.4f} (C1), {c2_mean:.4f} (C2)\n")
            report.append(f"  - Non-air: {non_air:.4f}\n")
            report.append(f"  - **{direction} during airy moments**\n\n")
    else:
        report.append("*No features appear in top 5 for both clusters — suggests different acoustic mechanisms.*\n\n")

    report.append("\n### Cluster-Specific Features (Form-Dependent)\n")
    report.append("Features that work well for only one cluster:\n\n")

    c1_specific = top_c1 - top_c2
    c2_specific = top_c2 - top_c1

    if c1_specific:
        report.append("**Cluster 1 only** (guitar trail-off):\n")
        for feat in c1_specific:
            c1_effect = next(r['effect_size'] for r in results['cluster1'] if r['feature'] == feat)
            report.append(f"- {feat} (effect: {c1_effect:.3f})\n")
        report.append("\n")

    if c2_specific:
        report.append("**Cluster 2 only** (vocal build):\n")
        for feat in c2_specific:
            c2_effect = next(r['effect_size'] for r in results['cluster2'] if r['feature'] == feat)
            report.append(f"- {feat} (effect: {c2_effect:.3f})\n")
        report.append("\n")

    report.append("\n### Effect Size Interpretation\n")
    report.append("- **|d| < 0.2**: Negligible difference\n")
    report.append("- **|d| = 0.2-0.5**: Small effect\n")
    report.append("- **|d| = 0.5-0.8**: Medium effect\n")
    report.append("- **|d| > 0.8**: Large effect\n")
    report.append("\n**Positive effect size** means the feature is HIGHER during airy moments.\n")
    report.append("**Negative effect size** means the feature is LOWER during airy moments.\n")

    # Write report
    with open(OUTPUT_MD, 'w') as f:
        f.writelines(report)

    print(f"Report written to: {OUTPUT_MD}")

    # Generate YAML output with per-tap feature values
    print("\nGenerating YAML data...")

    yaml_data = {
        'metadata': {
            'audio_file': str(AUDIO_PATH),
            'duration_sec': float(len(y) / sr),
            'sample_rate': int(sr),
            'hop_length': HOP_LENGTH,
            'n_frames': n_frames,
            'window_ms': WINDOW_MS,
            'n_air_taps': len(air_taps)
        },
        'feature_rankings': {
            'all_air': results['all_air'],
            'cluster1': results['cluster1'],
            'cluster2': results['cluster2']
        },
        'per_tap_features': []
    }

    # Extract feature values at each tap
    for tap_time in air_taps:
        tap_frame = int(tap_time * sr / HOP_LENGTH)
        if tap_frame >= n_frames:
            continue

        tap_data = {
            'time_sec': float(tap_time),
            'frame': int(tap_frame),
            'cluster': 1 if tap_time < 10 else 2,
            'features': {}
        }

        for feat_name, feat_values in features.items():
            tap_data['features'][feat_name] = float(feat_values[tap_frame])

        yaml_data['per_tap_features'].append(tap_data)

    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"YAML data written to: {OUTPUT_YAML}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nTop 5 features overall (by effect size):")
    for i, r in enumerate(results['all_air'][:5], 1):
        direction = "↑" if r['effect_size'] > 0 else "↓"
        print(f"  {i}. {r['feature']:20s} {direction} d={r['effect_size']:+.3f}  p={r['p_value']:.4f}")

    if shared:
        print(f"\nShared function features (both clusters): {', '.join(shared)}")
    else:
        print(f"\nNo strongly shared features — different acoustic mechanisms")

if __name__ == '__main__':
    main()
