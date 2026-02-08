#!/usr/bin/env python3
"""
Compare BUILD vs CLIMAX regions in Opiate Intro
Analyzes vocal phrases "like meeeeee" that feel different contextually
"""

import numpy as np
import librosa
import librosa.display
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
AUDIO_PATH = "/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.wav"
OUTPUT_PATH = "/Users/KO16K39/Documents/led/interactivity-research/analysis/build_vs_climax.md"

# Region definitions (in seconds)
BUILD_REGION = (28.9, 33.5)  # "airy" - first two "like meeeeee"
CLIMAX_REGION = (34.0, 40.0)  # culmination - last two "like meeeeee"

def extract_region(y, sr, start, end):
    """Extract audio samples for a time region"""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return y[start_sample:end_sample]

def compute_spectral_features(y, sr):
    """Compute spectral features for audio segment"""
    # Short-time Fourier transform
    S = np.abs(librosa.stft(y))

    # Basic spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    return {
        'centroid': centroid,
        'bandwidth': bandwidth,
        'flatness': flatness,
        'contrast': contrast  # 7 bands
    }

def compute_energy_features(y, sr):
    """Compute energy-based features"""
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # Mel spectrogram for frequency band energy
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=20, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Frequency band definitions (in mel bin indices, approximate)
    # For 128 mel bins from 20-8000 Hz:
    sub_bass_bins = (0, 8)      # 20-80 Hz
    bass_bins = (8, 20)          # 80-250 Hz
    mids_bins = (20, 70)         # 250-2000 Hz
    high_mids_bins = (70, 110)   # 2000-6000 Hz
    treble_bins = (110, 128)     # 6000-8000 Hz

    # Compute energy ratios
    total_energy = np.sum(mel, axis=0)
    sub_bass_energy = np.sum(mel[sub_bass_bins[0]:sub_bass_bins[1], :], axis=0) / (total_energy + 1e-10)
    bass_energy = np.sum(mel[bass_bins[0]:bass_bins[1], :], axis=0) / (total_energy + 1e-10)
    mids_energy = np.sum(mel[mids_bins[0]:mids_bins[1], :], axis=0) / (total_energy + 1e-10)
    high_mids_energy = np.sum(mel[high_mids_bins[0]:high_mids_bins[1], :], axis=0) / (total_energy + 1e-10)
    treble_energy = np.sum(mel[treble_bins[0]:treble_bins[1], :], axis=0) / (total_energy + 1e-10)

    return {
        'rms': rms,
        'sub_bass_ratio': sub_bass_energy,
        'bass_ratio': bass_energy,
        'mids_ratio': mids_energy,
        'high_mids_ratio': high_mids_energy,
        'treble_ratio': treble_energy
    }

def compute_hpss_features(y, sr):
    """Compute harmonic-percussive separation features"""
    # HPSS
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # RMS of each component
    rms_harmonic = librosa.feature.rms(y=y_harmonic)[0]
    rms_percussive = librosa.feature.rms(y=y_percussive)[0]
    rms_total = librosa.feature.rms(y=y)[0]

    # Harmonic ratio
    harmonic_ratio = rms_harmonic / (rms_total + 1e-10)

    return {
        'harmonic_ratio': harmonic_ratio,
        'percussive_ratio': rms_percussive / (rms_total + 1e-10)
    }

def compute_onset_features(y, sr):
    """Compute onset detection features"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    duration = len(y) / sr
    onset_density = len(onsets) / duration

    return {
        'onset_strength': onset_env,
        'onsets': onsets,
        'onset_density': onset_density
    }

def compute_trajectory(values):
    """Compute linear trend (slope) of a time series"""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    return slope

def analyze_region(y, sr, region_name):
    """Comprehensive analysis of an audio region"""
    print(f"Analyzing {region_name}...")

    # Compute all features
    spectral = compute_spectral_features(y, sr)
    energy = compute_energy_features(y, sr)
    hpss = compute_hpss_features(y, sr)
    onsets = compute_onset_features(y, sr)

    # Aggregate statistics
    results = {
        'name': region_name,
        'duration': len(y) / sr,

        # RMS energy
        'rms_mean': np.mean(energy['rms']),
        'rms_std': np.std(energy['rms']),
        'rms_max': np.max(energy['rms']),
        'rms_min': np.min(energy['rms']),
        'rms_range': np.max(energy['rms']) - np.min(energy['rms']),
        'rms_trajectory': compute_trajectory(energy['rms']),

        # Spectral features
        'centroid_mean': np.mean(spectral['centroid']),
        'centroid_std': np.std(spectral['centroid']),
        'centroid_trajectory': compute_trajectory(spectral['centroid']),

        'bandwidth_mean': np.mean(spectral['bandwidth']),
        'bandwidth_std': np.std(spectral['bandwidth']),

        'flatness_mean': np.mean(spectral['flatness']),
        'flatness_std': np.std(spectral['flatness']),

        'contrast_mean': np.mean(spectral['contrast']),
        'contrast_std': np.std(spectral['contrast']),

        # Frequency band energies
        'sub_bass_mean': np.mean(energy['sub_bass_ratio']),
        'bass_mean': np.mean(energy['bass_ratio']),
        'mids_mean': np.mean(energy['mids_ratio']),
        'high_mids_mean': np.mean(energy['high_mids_ratio']),
        'treble_mean': np.mean(energy['treble_ratio']),

        # HPSS
        'harmonic_ratio_mean': np.mean(hpss['harmonic_ratio']),
        'harmonic_ratio_std': np.std(hpss['harmonic_ratio']),
        'harmonic_ratio_trajectory': compute_trajectory(hpss['harmonic_ratio']),

        'percussive_ratio_mean': np.mean(hpss['percussive_ratio']),

        # Onsets
        'onset_density': onsets['onset_density'],
        'onset_strength_mean': np.mean(onsets['onset_strength']),
        'onset_strength_max': np.max(onsets['onset_strength']),
        'onset_strength_std': np.std(onsets['onset_strength']),
        'num_onsets': len(onsets['onsets']),

        # Raw data for phrase analysis
        '_raw': {
            'rms': energy['rms'],
            'centroid': spectral['centroid'],
            'onsets': onsets['onsets'],
            'onset_strength': onsets['onset_strength']
        }
    }

    return results

def identify_vocal_phrases(rms, onset_strength, sr, hop_length=512):
    """Identify individual vocal phrases based on energy peaks"""
    # Use onset strength to find phrase boundaries
    # Look for local maxima with minimum spacing
    from scipy.signal import find_peaks

    # Find peaks in onset strength
    peaks, properties = find_peaks(
        onset_strength,
        distance=int(0.5 * sr / hop_length),  # Min 0.5s between phrases
        prominence=np.std(onset_strength) * 0.5
    )

    # Convert frame indices to time
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # Define phrase windows (±0.75s around each peak)
    phrase_windows = []
    for t in times:
        start = max(0, t - 0.75)
        end = min(len(rms) * hop_length / sr, t + 0.75)
        phrase_windows.append((start, end))

    return phrase_windows, times

def analyze_phrases(y, sr, region_results):
    """Analyze individual vocal phrases within a region"""
    rms = region_results['_raw']['rms']
    onset_strength = region_results['_raw']['onset_strength']

    phrase_windows, peak_times = identify_vocal_phrases(rms, onset_strength, sr)

    phrases = []
    for i, (start, end) in enumerate(phrase_windows):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        phrase_y = y[start_sample:end_sample]

        if len(phrase_y) < sr * 0.1:  # Skip too-short segments
            continue

        # Quick features for this phrase
        phrase_rms = librosa.feature.rms(y=phrase_y)[0]
        phrase_centroid = librosa.feature.spectral_centroid(y=phrase_y, sr=sr)[0]
        phrase_flatness = librosa.feature.spectral_flatness(y=phrase_y)[0]

        phrases.append({
            'index': i,
            'start': start,
            'end': end,
            'peak_time': peak_times[i],
            'rms_mean': np.mean(phrase_rms),
            'rms_max': np.max(phrase_rms),
            'centroid_mean': np.mean(phrase_centroid),
            'flatness_mean': np.mean(phrase_flatness),
            'duration': end - start
        })

    return phrases

def compute_differences(build, climax):
    """Compute relative differences between regions"""
    differences = {}

    # Skip non-numeric fields
    skip_keys = ['name', 'duration', '_raw']

    for key in build.keys():
        if key in skip_keys:
            continue

        if isinstance(build[key], (int, float, np.number)):
            build_val = build[key]
            climax_val = climax[key]

            # Absolute difference
            abs_diff = climax_val - build_val

            # Relative difference (percentage)
            if abs(build_val) > 1e-10:
                rel_diff = (abs_diff / abs(build_val)) * 100
            else:
                rel_diff = 0.0

            differences[key] = {
                'build': build_val,
                'climax': climax_val,
                'abs_diff': abs_diff,
                'rel_diff': rel_diff,
                'abs_rel_diff': abs(rel_diff)
            }

    return differences

def format_report(build, climax, differences, build_phrases, climax_phrases):
    """Generate markdown report"""

    # Sort differences by absolute relative difference
    sorted_diffs = sorted(
        differences.items(),
        key=lambda x: x[1]['abs_rel_diff'],
        reverse=True
    )

    report = []
    report.append("# Build vs Climax Analysis: Opiate Intro")
    report.append("")
    report.append("Analysis of two regions containing similar vocal phrases ('like meeeeee') that feel different.")
    report.append("")
    report.append("## Regions")
    report.append("")
    report.append(f"- **BUILD**: {BUILD_REGION[0]}s - {BUILD_REGION[1]}s ({build['duration']:.2f}s)")
    report.append("  - User annotation: 'airy', anticipation")
    report.append("  - First two 'like meeeeee' vocals")
    report.append("")
    report.append(f"- **CLIMAX**: {CLIMAX_REGION[0]}s - {CLIMAX_REGION[1]}s ({climax['duration']:.2f}s)")
    report.append("  - User annotation: payoff, arrival, culmination")
    report.append("  - Last two 'like meeeeee' vocals")
    report.append("")

    # Top distinguishing features
    report.append("## Top Distinguishing Features")
    report.append("")
    report.append("Features with largest relative difference between regions:")
    report.append("")
    report.append("| Feature | Build | Climax | Change | % Change |")
    report.append("|---------|-------|--------|--------|----------|")

    for key, diff in sorted_diffs[:15]:
        feature_name = key.replace('_', ' ').title()
        build_val = diff['build']
        climax_val = diff['climax']
        abs_diff = diff['abs_diff']
        rel_diff = diff['rel_diff']

        # Format based on magnitude
        if abs(build_val) < 0.01:
            build_str = f"{build_val:.6f}"
            climax_str = f"{climax_val:.6f}"
            diff_str = f"{abs_diff:+.6f}"
        elif abs(build_val) < 1:
            build_str = f"{build_val:.4f}"
            climax_str = f"{climax_val:.4f}"
            diff_str = f"{abs_diff:+.4f}"
        else:
            build_str = f"{build_val:.2f}"
            climax_str = f"{climax_val:.2f}"
            diff_str = f"{abs_diff:+.2f}"

        report.append(f"| {feature_name} | {build_str} | {climax_str} | {diff_str} | {rel_diff:+.1f}% |")

    report.append("")

    # Full feature comparison table
    report.append("## Complete Feature Comparison")
    report.append("")

    # Energy features
    report.append("### Energy Features")
    report.append("")
    report.append("| Feature | Build | Climax | Difference |")
    report.append("|---------|-------|--------|------------|")
    energy_features = [
        'rms_mean', 'rms_std', 'rms_max', 'rms_min', 'rms_range', 'rms_trajectory'
    ]
    for feat in energy_features:
        if feat in differences:
            d = differences[feat]
            report.append(f"| {feat.replace('_', ' ').title()} | {d['build']:.4f} | {d['climax']:.4f} | {d['abs_diff']:+.4f} ({d['rel_diff']:+.1f}%) |")
    report.append("")

    # Spectral features
    report.append("### Spectral Features")
    report.append("")
    report.append("| Feature | Build | Climax | Difference |")
    report.append("|---------|-------|--------|------------|")
    spectral_features = [
        'centroid_mean', 'centroid_std', 'centroid_trajectory',
        'bandwidth_mean', 'bandwidth_std',
        'flatness_mean', 'flatness_std',
        'contrast_mean', 'contrast_std'
    ]
    for feat in spectral_features:
        if feat in differences:
            d = differences[feat]
            report.append(f"| {feat.replace('_', ' ').title()} | {d['build']:.2f} | {d['climax']:.2f} | {d['abs_diff']:+.2f} ({d['rel_diff']:+.1f}%) |")
    report.append("")

    # Frequency bands
    report.append("### Frequency Band Energy Ratios")
    report.append("")
    report.append("| Band | Build | Climax | Difference |")
    report.append("|------|-------|--------|------------|")
    band_features = [
        'sub_bass_mean', 'bass_mean', 'mids_mean', 'high_mids_mean', 'treble_mean'
    ]
    for feat in band_features:
        if feat in differences:
            d = differences[feat]
            report.append(f"| {feat.replace('_mean', '').replace('_', ' ').title()} | {d['build']:.4f} | {d['climax']:.4f} | {d['abs_diff']:+.4f} ({d['rel_diff']:+.1f}%) |")
    report.append("")

    # Harmonic/Percussive
    report.append("### Harmonic-Percussive Separation")
    report.append("")
    report.append("| Feature | Build | Climax | Difference |")
    report.append("|---------|-------|--------|------------|")
    hpss_features = [
        'harmonic_ratio_mean', 'harmonic_ratio_std', 'harmonic_ratio_trajectory',
        'percussive_ratio_mean'
    ]
    for feat in hpss_features:
        if feat in differences:
            d = differences[feat]
            report.append(f"| {feat.replace('_', ' ').title()} | {d['build']:.4f} | {d['climax']:.4f} | {d['abs_diff']:+.4f} ({d['rel_diff']:+.1f}%) |")
    report.append("")

    # Onset features
    report.append("### Onset Features")
    report.append("")
    report.append("| Feature | Build | Climax | Difference |")
    report.append("|---------|-------|--------|------------|")
    onset_features = [
        'onset_density', 'onset_strength_mean', 'onset_strength_max',
        'onset_strength_std', 'num_onsets'
    ]
    for feat in onset_features:
        if feat in differences:
            d = differences[feat]
            report.append(f"| {feat.replace('_', ' ').title()} | {d['build']:.4f} | {d['climax']:.4f} | {d['abs_diff']:+.4f} ({d['rel_diff']:+.1f}%) |")
    report.append("")

    # Trajectory interpretation
    report.append("## Trajectory Analysis")
    report.append("")
    report.append("Changes over time within each region:")
    report.append("")

    # RMS trajectory
    rms_traj_build = build['rms_trajectory']
    rms_traj_climax = climax['rms_trajectory']
    report.append(f"**Energy trajectory (RMS slope)**:")
    report.append(f"- Build: {rms_traj_build:+.6f} ({'increasing' if rms_traj_build > 0 else 'decreasing'})")
    report.append(f"- Climax: {rms_traj_climax:+.6f} ({'increasing' if rms_traj_climax > 0 else 'decreasing'})")
    report.append("")

    # Centroid trajectory
    cent_traj_build = build['centroid_trajectory']
    cent_traj_climax = climax['centroid_trajectory']
    report.append(f"**Brightness trajectory (Centroid slope)**:")
    report.append(f"- Build: {cent_traj_build:+.2f} Hz/frame ({'brightening' if cent_traj_build > 0 else 'darkening'})")
    report.append(f"- Climax: {cent_traj_climax:+.2f} Hz/frame ({'brightening' if cent_traj_climax > 0 else 'darkening'})")
    report.append("")

    # Harmonic ratio trajectory
    harm_traj_build = build['harmonic_ratio_trajectory']
    harm_traj_climax = climax['harmonic_ratio_trajectory']
    report.append(f"**Harmonic ratio trajectory**:")
    report.append(f"- Build: {harm_traj_build:+.6f} ({'more harmonic' if harm_traj_build > 0 else 'more percussive'})")
    report.append(f"- Climax: {harm_traj_climax:+.6f} ({'more harmonic' if harm_traj_climax > 0 else 'more percussive'})")
    report.append("")

    # Individual phrase analysis
    report.append("## Individual Vocal Phrase Analysis")
    report.append("")
    report.append(f"Detected {len(build_phrases)} phrases in BUILD region and {len(climax_phrases)} phrases in CLIMAX region.")
    report.append("")

    if build_phrases:
        report.append("### BUILD Region Phrases")
        report.append("")
        report.append("| # | Time (s) | Duration | RMS Mean | RMS Max | Centroid | Flatness |")
        report.append("|---|----------|----------|----------|---------|----------|----------|")
        for p in build_phrases:
            report.append(f"| {p['index']+1} | {p['peak_time']:.2f} | {p['duration']:.2f}s | {p['rms_mean']:.4f} | {p['rms_max']:.4f} | {p['centroid_mean']:.0f} Hz | {p['flatness_mean']:.4f} |")
        report.append("")

    if climax_phrases:
        report.append("### CLIMAX Region Phrases")
        report.append("")
        report.append("| # | Time (s) | Duration | RMS Mean | RMS Max | Centroid | Flatness |")
        report.append("|---|----------|----------|----------|---------|----------|----------|")
        for p in climax_phrases:
            report.append(f"| {p['index']+1} | {p['peak_time']:.2f} | {p['duration']:.2f}s | {p['rms_mean']:.4f} | {p['rms_max']:.4f} | {p['centroid_mean']:.0f} Hz | {p['flatness_mean']:.4f} |")
        report.append("")

    # Compare phrases if we have matching counts
    if len(build_phrases) > 0 and len(climax_phrases) > 0:
        report.append("### Phrase-Level Comparison")
        report.append("")

        # Average across phrases
        build_avg_rms = np.mean([p['rms_mean'] for p in build_phrases])
        build_avg_centroid = np.mean([p['centroid_mean'] for p in build_phrases])
        build_avg_flatness = np.mean([p['flatness_mean'] for p in build_phrases])

        climax_avg_rms = np.mean([p['rms_mean'] for p in climax_phrases])
        climax_avg_centroid = np.mean([p['centroid_mean'] for p in climax_phrases])
        climax_avg_flatness = np.mean([p['flatness_mean'] for p in climax_phrases])

        report.append("Average features across individual phrases:")
        report.append("")
        report.append("| Feature | Build Phrases | Climax Phrases | Difference |")
        report.append("|---------|---------------|----------------|------------|")
        report.append(f"| RMS Mean | {build_avg_rms:.4f} | {climax_avg_rms:.4f} | {climax_avg_rms - build_avg_rms:+.4f} ({((climax_avg_rms - build_avg_rms) / build_avg_rms * 100):+.1f}%) |")
        report.append(f"| Centroid | {build_avg_centroid:.0f} Hz | {climax_avg_centroid:.0f} Hz | {climax_avg_centroid - build_avg_centroid:+.0f} Hz ({((climax_avg_centroid - build_avg_centroid) / build_avg_centroid * 100):+.1f}%) |")
        report.append(f"| Flatness | {build_avg_flatness:.4f} | {climax_avg_flatness:.4f} | {climax_avg_flatness - build_avg_flatness:+.4f} ({((climax_avg_flatness - build_avg_flatness) / build_avg_flatness * 100):+.1f}%) |")
        report.append("")

    # Interpretation
    report.append("## Interpretation")
    report.append("")
    report.append("### What Makes the CLIMAX Feel Like Climax?")
    report.append("")

    # Energy interpretation
    rms_diff = differences['rms_mean']
    if rms_diff['rel_diff'] > 10:
        report.append(f"**1. Higher Energy**: Climax is {rms_diff['rel_diff']:.1f}% louder (RMS: {rms_diff['climax']:.4f} vs {rms_diff['build']:.4f}). Pure amplitude increase contributes to the sense of arrival.")
    elif rms_diff['rel_diff'] < -10:
        report.append(f"**1. Lower Energy**: Surprisingly, climax is {abs(rms_diff['rel_diff']):.1f}% quieter — the 'arrival' feeling may be more about spectral/contextual changes than raw volume.")
    else:
        report.append(f"**1. Similar Energy**: RMS only differs by {rms_diff['rel_diff']:.1f}% — the 'arrival' feeling is NOT primarily about loudness.")
    report.append("")

    # Spectral brightness
    cent_diff = differences['centroid_mean']
    if cent_diff['rel_diff'] > 5:
        report.append(f"**2. Brighter Spectrum**: Climax is {cent_diff['rel_diff']:.1f}% brighter (centroid: {cent_diff['climax']:.0f} Hz vs {cent_diff['build']:.0f} Hz). More high-frequency content = more 'present' and 'forward'.")
    elif cent_diff['rel_diff'] < -5:
        report.append(f"**2. Darker Spectrum**: Climax is {abs(cent_diff['rel_diff']):.1f}% darker — the 'weight' of lower frequencies may add to the sense of resolution.")
    else:
        report.append(f"**2. Similar Brightness**: Spectral centroid only differs by {cent_diff['rel_diff']:.1f}% — not a primary distinguisher.")
    report.append("")

    # Harmonic content
    harm_diff = differences['harmonic_ratio_mean']
    if harm_diff['rel_diff'] > 5:
        report.append(f"**3. More Harmonic**: Climax is {harm_diff['rel_diff']:.1f}% more harmonic (ratio: {harm_diff['climax']:.3f} vs {harm_diff['build']:.3f}). Stronger pitched content = more melodic 'fullness'.")
    elif harm_diff['rel_diff'] < -5:
        report.append(f"**3. More Percussive**: Climax is {abs(harm_diff['rel_diff']):.1f}% less harmonic — more rhythmic attack, less tonal sustain.")
    else:
        report.append(f"**3. Similar Harmonic Content**: Harmonic ratio only differs by {harm_diff['rel_diff']:.1f}%.")
    report.append("")

    # Frequency band balance
    bass_diff = differences['bass_mean']
    mids_diff = differences['mids_mean']
    treble_diff = differences['treble_mean']

    report.append("**4. Frequency Balance Shifts**:")
    if abs(bass_diff['rel_diff']) > 10 or abs(mids_diff['rel_diff']) > 10 or abs(treble_diff['rel_diff']) > 10:
        report.append(f"- Bass: {bass_diff['rel_diff']:+.1f}% (climax: {bass_diff['climax']:.3f}, build: {bass_diff['build']:.3f})")
        report.append(f"- Mids: {mids_diff['rel_diff']:+.1f}% (climax: {mids_diff['climax']:.3f}, build: {mids_diff['build']:.3f})")
        report.append(f"- Treble: {treble_diff['rel_diff']:+.1f}% (climax: {treble_diff['climax']:.3f}, build: {treble_diff['build']:.3f})")
    else:
        report.append("- Frequency band balance is very similar between regions")
    report.append("")

    # Trajectory interpretation
    report.append("**5. Temporal Context (Trajectory)**:")
    if abs(rms_traj_build - rms_traj_climax) > 0.001:
        report.append(f"- BUILD energy trajectory: {rms_traj_build:+.6f} ({'rising' if rms_traj_build > 0 else 'falling'})")
        report.append(f"- CLIMAX energy trajectory: {rms_traj_climax:+.6f} ({'rising' if rms_traj_climax > 0 else 'falling'})")
        if rms_traj_build > 0.002 and rms_traj_climax < 0.002:
            report.append("- BUILD is **rising** while CLIMAX has **plateaued** — the change in trajectory itself signals arrival")
    report.append("")

    # Dynamic range
    range_diff = differences['rms_range']
    report.append(f"**6. Dynamic Range**: Climax has {range_diff['rel_diff']:+.1f}% more dynamic variation (range: {range_diff['climax']:.4f} vs {range_diff['build']:.4f}).")
    report.append("")

    # Onset density
    onset_diff = differences['onset_density']
    report.append(f"**7. Rhythmic Density**: {onset_diff['rel_diff']:+.1f}% change in onset density ({onset_diff['climax']:.2f} vs {onset_diff['build']:.2f} onsets/sec).")
    report.append("")

    # Summary
    report.append("### Summary")
    report.append("")
    report.append("The distinction between BUILD and CLIMAX appears to be driven by:")
    report.append("")

    # Find top 3 distinguishers
    top_3 = sorted_diffs[:3]
    for i, (key, diff) in enumerate(top_3, 1):
        direction = "higher" if diff['abs_diff'] > 0 else "lower"
        report.append(f"{i}. **{key.replace('_', ' ').title()}**: Climax is {abs(diff['rel_diff']):.1f}% {direction}")

    report.append("")

    # Spectral vs contextual
    if len(build_phrases) > 0 and len(climax_phrases) > 0:
        phrase_rms_diff = abs((climax_avg_rms - build_avg_rms) / build_avg_rms * 100)
        region_rms_diff = abs(rms_diff['rel_diff'])

        if phrase_rms_diff > region_rms_diff * 0.8:
            report.append("**Individual vocal phrases ARE spectrally different** between regions — the 'like meeeeee' vocals themselves have changed.")
        else:
            report.append("**Individual vocal phrases are similar** — the difference is more about **what's around them** (context, layering, trajectory).")

    report.append("")
    report.append("---")
    report.append(f"*Analysis completed: {len(build_phrases)} build phrases, {len(climax_phrases)} climax phrases detected*")

    return '\n'.join(report)

def main():
    print("Loading audio...")
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    print(f"Loaded: {len(y)/sr:.2f}s at {sr} Hz")

    # Extract regions
    build_y = extract_region(y, sr, BUILD_REGION[0], BUILD_REGION[1])
    climax_y = extract_region(y, sr, CLIMAX_REGION[0], CLIMAX_REGION[1])

    # Analyze regions
    build_results = analyze_region(build_y, sr, "BUILD")
    climax_results = analyze_region(climax_y, sr, "CLIMAX")

    # Analyze phrases
    print("Identifying vocal phrases...")
    build_phrases = analyze_phrases(build_y, sr, build_results)
    climax_phrases = analyze_phrases(climax_y, sr, climax_results)

    print(f"Found {len(build_phrases)} phrases in BUILD, {len(climax_phrases)} in CLIMAX")

    # Compute differences
    differences = compute_differences(build_results, climax_results)

    # Generate report
    print("Generating report...")
    report = format_report(build_results, climax_results, differences, build_phrases, climax_phrases)

    # Save report
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    print(f"\nReport saved to: {OUTPUT_PATH}")
    print(f"Report length: {len(report)} characters, {len(report.splitlines())} lines")

if __name__ == "__main__":
    main()
