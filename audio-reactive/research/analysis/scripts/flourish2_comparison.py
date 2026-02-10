#!/usr/bin/env python3
"""
Comprehensive analysis of beat-flourish2 annotation layer.
Compares against: consistent-beat, beat-flourish, and computed flourishes.
"""

import yaml
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from collections import defaultdict

# Paths
AUDIO_DIR = Path("/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments")
OUTPUT_DIR = Path("/Users/KO16K39/Documents/led/audio-reactive/research/analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

ANNOTATIONS_FILE = AUDIO_DIR / "opiate_intro.annotations.yaml"
AUDIO_FILE = AUDIO_DIR / "opiate_intro.wav"

# Load annotations
with open(ANNOTATIONS_FILE, 'r') as f:
    annotations = yaml.safe_load(f)

# Extract layers
consistent_beat = np.array(annotations['consistent-beat'])
beat_flourish = np.array(annotations['beat-flourish'])
beat_flourish2 = np.array(annotations['beat-flourish2'])
changes = np.array(annotations['changes'])

# Compute flourishes (beat minus consistent-beat)
beat_layer = np.array(annotations['beat'])
computed_flourishes = []
for bt in beat_layer:
    if not np.any(np.abs(consistent_beat - bt) < 0.15):
        computed_flourishes.append(bt)
computed_flourishes = np.array(computed_flourishes)

# Load audio
y, sr = sf.read(AUDIO_FILE)
if y.ndim > 1:
    y = y.mean(axis=1)

# Extract audio features
hop_length = 512
n_fft = 2048

# Onsets
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

# Spectral features
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# Bass (20-250 Hz)
bass_mask = (freqs >= 20) & (freqs <= 250)
bass_energy = S[bass_mask].sum(axis=0)
times = librosa.frames_to_time(np.arange(len(bass_energy)), sr=sr, hop_length=hop_length)

# Spectral centroid
centroid = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=hop_length)[0]

# RMS
rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

# Spectral flux
flux = np.diff(S, axis=1)
flux = np.sqrt((flux**2).sum(axis=0))
flux = np.concatenate([[0], flux])  # Pad to match length

# Helper functions
def interval_stats(timestamps):
    """Compute inter-tap interval statistics."""
    if len(timestamps) < 2:
        return {}
    intervals = np.diff(timestamps)
    return {
        'median': np.median(intervals),
        'mean': np.mean(intervals),
        'std': np.std(intervals),
        'cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0,
        'min': np.min(intervals),
        'max': np.max(intervals),
        'bpm_from_median': 60.0 / np.median(intervals) if np.median(intervals) > 0 else 0
    }

def grid_alignment(test_taps, reference_taps, threshold=0.15):
    """Count what % of test_taps align with reference_taps within threshold."""
    aligned = 0
    for tap in test_taps:
        if np.any(np.abs(reference_taps - tap) < threshold):
            aligned += 1
    return aligned / len(test_taps) if len(test_taps) > 0 else 0

def find_nearest_peak(tap_time, feature_values, feature_times, window=0.3):
    """Find nearest peak in feature within window of tap_time."""
    mask = np.abs(feature_times - tap_time) < window
    if not np.any(mask):
        return None, float('inf')

    local_vals = feature_values[mask]
    local_times = feature_times[mask]

    # Find local maximum
    if len(local_vals) == 0:
        return None, float('inf')

    peak_idx = np.argmax(local_vals)
    peak_time = local_times[peak_idx]
    distance = abs(peak_time - tap_time)

    return peak_time, distance

def feature_alignment_analysis(taps):
    """For each tap, find which feature it best aligns with."""
    alignments = {
        'onset': [],
        'bass': [],
        'centroid': [],
        'rms': [],
        'flux': []
    }

    for tap in taps:
        distances = {}

        # Onset
        onset_dists = np.abs(onset_times - tap)
        distances['onset'] = np.min(onset_dists) if len(onset_dists) > 0 else float('inf')

        # Bass peak
        _, bass_dist = find_nearest_peak(tap, bass_energy, times)
        distances['bass'] = bass_dist

        # Centroid peak
        _, centroid_dist = find_nearest_peak(tap, centroid, times)
        distances['centroid'] = centroid_dist

        # RMS peak
        _, rms_dist = find_nearest_peak(tap, rms, times)
        distances['rms'] = rms_dist

        # Flux peak
        _, flux_dist = find_nearest_peak(tap, flux, times)
        distances['flux'] = flux_dist

        # Best match
        best_feature = min(distances.keys(), key=lambda k: distances[k])
        alignments[best_feature].append(tap)

    return alignments

def section_density(taps, section_bounds):
    """Compute tap density per section."""
    densities = []
    sections = []

    for i in range(len(section_bounds) - 1):
        start = section_bounds[i]
        end = section_bounds[i + 1]
        duration = end - start

        count = np.sum((taps >= start) & (taps < end))
        density = count / duration

        densities.append(density)
        sections.append(f"Section {i+1} ({start:.1f}s - {end:.1f}s)")

    return sections, densities

def find_new_taps(test_taps, *reference_layers, threshold=0.15):
    """Find taps in test_taps that don't align with any reference layer."""
    new_taps = []
    for tap in test_taps:
        is_new = True
        for ref_layer in reference_layers:
            if np.any(np.abs(ref_layer - tap) < threshold):
                is_new = False
                break
        if is_new:
            new_taps.append(tap)
    return np.array(new_taps)

# ============================================================================
# ANALYSIS
# ============================================================================

report_lines = []

def add_line(text=""):
    report_lines.append(text)

add_line("# Beat-Flourish2 Analysis")
add_line()
add_line("Comprehensive comparison of the beat-flourish2 layer against existing annotations and computed flourishes.")
add_line()

# ============================================================================
# 1. BASIC STATS
# ============================================================================

add_line("## 1. Basic Statistics")
add_line()

bf2_stats = interval_stats(beat_flourish2)
bf_stats = interval_stats(beat_flourish)
cb_stats = interval_stats(consistent_beat)
cf_stats = interval_stats(computed_flourishes)

add_line("### Inter-tap Intervals")
add_line()
add_line("| Layer | Count | Median (s) | Mean (s) | Std | CV | BPM (median) |")
add_line("|-------|-------|------------|----------|-----|-----|--------------|")
add_line(f"| **beat-flourish2** | {len(beat_flourish2)} | {bf2_stats['median']:.3f} | {bf2_stats['mean']:.3f} | {bf2_stats['std']:.3f} | {bf2_stats['cv']:.3f} | {bf2_stats['bpm_from_median']:.1f} |")
add_line(f"| beat-flourish | {len(beat_flourish)} | {bf_stats['median']:.3f} | {bf_stats['mean']:.3f} | {bf_stats['std']:.3f} | {bf_stats['cv']:.3f} | {bf_stats['bpm_from_median']:.1f} |")
add_line(f"| consistent-beat | {len(consistent_beat)} | {cb_stats['median']:.3f} | {cb_stats['mean']:.3f} | {cb_stats['std']:.3f} | {cb_stats['cv']:.3f} | {cb_stats['bpm_from_median']:.1f} |")
add_line(f"| computed flourishes | {len(computed_flourishes)} | {cf_stats['median']:.3f} | {cf_stats['mean']:.3f} | {cf_stats['std']:.3f} | {cf_stats['cv']:.3f} | {cf_stats['bpm_from_median']:.1f} |")
add_line()

add_line("**Interpretation:**")
if bf2_stats['cv'] < bf_stats['cv']:
    add_line(f"- beat-flourish2 is MORE regular than beat-flourish (CV {bf2_stats['cv']:.3f} vs {bf_stats['cv']:.3f})")
else:
    add_line(f"- beat-flourish2 is LESS regular than beat-flourish (CV {bf2_stats['cv']:.3f} vs {bf_stats['cv']:.3f})")

if bf2_stats['cv'] > 0.5:
    add_line(f"- High CV ({bf2_stats['cv']:.3f}) suggests irregular tapping (good for flourishes)")
else:
    add_line(f"- Low CV ({bf2_stats['cv']:.3f}) suggests user fell into a groove (not ideal for flourishes)")

add_line()

# ============================================================================
# 2. GRID ALIGNMENT
# ============================================================================

add_line("## 2. Grid Alignment")
add_line()

cb_alignment = grid_alignment(beat_flourish2, consistent_beat, threshold=0.15)
cf_alignment = grid_alignment(beat_flourish2, computed_flourishes, threshold=0.15)
bf_cb_alignment = grid_alignment(beat_flourish, consistent_beat, threshold=0.15)

add_line("### Overlap with Existing Layers (±150ms threshold)")
add_line()
add_line("| Test Layer | vs. Layer | Overlap % |")
add_line("|------------|-----------|-----------|")
add_line(f"| **beat-flourish2** | consistent-beat | **{cb_alignment*100:.1f}%** |")
add_line(f"| beat-flourish | consistent-beat | {bf_cb_alignment*100:.1f}% |")
add_line(f"| **beat-flourish2** | computed flourishes | **{cf_alignment*100:.1f}%** |")
add_line()

add_line("**Interpretation:**")
if cb_alignment < bf_cb_alignment:
    add_line(f"- ✅ beat-flourish2 is CLEANER — only {cb_alignment*100:.1f}% overlap with consistent-beat vs. {bf_cb_alignment*100:.1f}% for beat-flourish")
else:
    add_line(f"- ❌ beat-flourish2 is NOT cleaner — {cb_alignment*100:.1f}% overlap with consistent-beat vs. {bf_cb_alignment*100:.1f}% for beat-flourish")

if cf_alignment > 0.6:
    add_line(f"- High overlap with computed flourishes ({cf_alignment*100:.1f}%) — mostly redundant")
elif cf_alignment > 0.3:
    add_line(f"- Moderate overlap with computed flourishes ({cf_alignment*100:.1f}%) — some agreement, some new info")
else:
    add_line(f"- Low overlap with computed flourishes ({cf_alignment*100:.1f}%) — captures different events")

add_line()

# Find genuinely new taps
new_taps = find_new_taps(beat_flourish2, consistent_beat, computed_flourishes, threshold=0.15)
new_pct = len(new_taps) / len(beat_flourish2) * 100 if len(beat_flourish2) > 0 else 0

add_line(f"### New Information")
add_line()
add_line(f"- **{len(new_taps)} taps ({new_pct:.1f}%)** are genuinely NEW (not in consistent-beat or computed flourishes)")
add_line()

if len(new_taps) > 0:
    add_line(f"New tap timestamps: {', '.join([f'{t:.2f}s' for t in new_taps[:10]])}" + ("..." if len(new_taps) > 10 else ""))
    add_line()

# ============================================================================
# 3. FEATURE ALIGNMENT
# ============================================================================

add_line("## 3. Feature Alignment")
add_line()

bf2_alignments = feature_alignment_analysis(beat_flourish2)
cf_alignments = feature_alignment_analysis(computed_flourishes)

add_line("### What do the taps track?")
add_line()
add_line("| Feature | beat-flourish2 | computed flourishes |")
add_line("|---------|----------------|---------------------|")
for feature in ['onset', 'bass', 'centroid', 'rms', 'flux']:
    bf2_count = len(bf2_alignments[feature])
    bf2_pct = bf2_count / len(beat_flourish2) * 100 if len(beat_flourish2) > 0 else 0
    cf_count = len(cf_alignments[feature])
    cf_pct = cf_count / len(computed_flourishes) * 100 if len(computed_flourishes) > 0 else 0
    add_line(f"| {feature.capitalize()} | {bf2_count} ({bf2_pct:.0f}%) | {cf_count} ({cf_pct:.0f}%) |")
add_line()

# Determine dominant feature
bf2_dominant = max(bf2_alignments.keys(), key=lambda k: len(bf2_alignments[k]))
cf_dominant = max(cf_alignments.keys(), key=lambda k: len(cf_alignments[k]))

add_line("**Interpretation:**")
add_line(f"- beat-flourish2 primarily tracks **{bf2_dominant}** events")
add_line(f"- computed flourishes primarily track **{cf_dominant}** events")

if bf2_dominant != cf_dominant:
    add_line(f"- Different dominant features suggest they capture different musical aspects")
else:
    add_line(f"- Same dominant feature suggests similar musical focus")

add_line()

# ============================================================================
# 4. REGULARITY CHECK
# ============================================================================

add_line("## 4. Regularity Check")
add_line()

# Analyze interval distribution
intervals_bf2 = np.diff(beat_flourish2)
interval_bins = np.histogram(intervals_bf2, bins=20)

# Check for clustering around a tempo
# If most intervals are within ±20% of median, it's regular
median_interval = np.median(intervals_bf2)
tolerance = 0.2 * median_interval
regular_count = np.sum(np.abs(intervals_bf2 - median_interval) < tolerance)
regular_pct = regular_count / len(intervals_bf2) * 100 if len(intervals_bf2) > 0 else 0

add_line(f"- Median interval: {median_interval:.3f}s ({60/median_interval:.1f} BPM)")
add_line(f"- {regular_count}/{len(intervals_bf2)} intervals ({regular_pct:.1f}%) within ±20% of median")
add_line()

if regular_pct > 60:
    add_line(f"**Verdict: REGULAR tapping** — user fell into a {60/median_interval:.1f} BPM groove")
elif regular_pct > 40:
    add_line(f"**Verdict: SEMI-REGULAR** — some groove, but with variation")
else:
    add_line(f"**Verdict: IRREGULAR** — truly varied timing (good for flourishes)")

add_line()

# Interval distribution stats
add_line(f"Interval range: {bf2_stats['min']:.3f}s - {bf2_stats['max']:.3f}s (ratio: {bf2_stats['max']/bf2_stats['min']:.1f}x)")
add_line()

# ============================================================================
# 5. SECTION-BY-SECTION DENSITY
# ============================================================================

add_line("## 5. Section Density")
add_line()

# Use changes layer as section boundaries, plus start and end
section_bounds = [beat_flourish2[0], *changes, beat_flourish2[-1]]
section_bounds = sorted(section_bounds)

sections_bf2, densities_bf2 = section_density(beat_flourish2, section_bounds)
sections_cf, densities_cf = section_density(computed_flourishes, section_bounds)
sections_bf, densities_bf = section_density(beat_flourish, section_bounds)

add_line("### Tap Density by Section (taps per second)")
add_line()
add_line("| Section | beat-flourish2 | computed flourishes | beat-flourish |")
add_line("|---------|----------------|---------------------|---------------|")
for i, section in enumerate(sections_bf2):
    add_line(f"| {section} | {densities_bf2[i]:.2f} | {densities_cf[i]:.2f} | {densities_bf[i]:.2f} |")
add_line()

# Compare density patterns (correlation)
if len(densities_bf2) > 1 and len(densities_cf) > 1:
    corr_cf = np.corrcoef(densities_bf2, densities_cf)[0, 1]
    corr_bf = np.corrcoef(densities_bf2, densities_bf)[0, 1]

    add_line("**Density Pattern Correlation:**")
    add_line(f"- beat-flourish2 vs. computed flourishes: r = {corr_cf:.3f}")
    add_line(f"- beat-flourish2 vs. beat-flourish: r = {corr_bf:.3f}")
    add_line()

    if corr_cf > 0.7:
        add_line("- High correlation with computed flourishes → similar density pattern")
    elif corr_cf < -0.3:
        add_line("- Negative correlation with computed flourishes → opposite density pattern")
    else:
        add_line("- Low correlation with computed flourishes → different density pattern")

add_line()

# ============================================================================
# 6. VERDICT
# ============================================================================

add_line("## 6. Verdict")
add_line()

# Criteria for "better":
# 1. Lower consistent-beat overlap than beat-flourish
# 2. CV > 0.5 (irregular)
# 3. Adds new info (>20% new taps)
# 4. Not redundant with computed flourishes (overlap < 70%)

cleaner = cb_alignment < bf_cb_alignment
irregular = bf2_stats['cv'] > 0.5
adds_new = new_pct > 20
not_redundant = cf_alignment < 0.7

add_line("### Checklist")
add_line()
add_line(f"- [ ] Cleaner than beat-flourish? {'✅ YES' if cleaner else '❌ NO'} ({cb_alignment*100:.1f}% vs {bf_cb_alignment*100:.1f}% consistent-beat overlap)")
add_line(f"- [ ] Irregular timing? {'✅ YES' if irregular else '❌ NO'} (CV = {bf2_stats['cv']:.3f})")
add_line(f"- [ ] Adds new information? {'✅ YES' if adds_new else '❌ NO'} ({new_pct:.1f}% new taps)")
add_line(f"- [ ] Not redundant with computed flourishes? {'✅ YES' if not_redundant else '❌ NO'} ({cf_alignment*100:.1f}% overlap)")
add_line()

score = sum([cleaner, irregular, adds_new, not_redundant])

add_line("### Recommendation")
add_line()

if score >= 3:
    add_line(f"**✅ USE beat-flourish2** ({score}/4 criteria met)")
    add_line()
    add_line("This layer is cleaner and adds value beyond computed flourishes.")
    if new_pct > 50:
        add_line(f"The {new_pct:.0f}% new taps capture unique flourish moments.")
    add_line()
    add_line("**Suggested use:** Primary flourish layer for LED effects.")

elif score == 2:
    add_line(f"**⚠️ MIXED** ({score}/4 criteria met)")
    add_line()
    add_line("beat-flourish2 has some merit but also limitations.")
    add_line()
    if not_redundant and adds_new:
        add_line("**Suggested use:** Combine with computed flourishes — use beat-flourish2 for sections where user tapped, computed for full coverage.")
    else:
        add_line("**Suggested use:** Stick with computed flourishes unless specific sections need the user's interpretation.")

else:
    add_line(f"**❌ DON'T USE beat-flourish2** ({score}/4 criteria met)")
    add_line()
    add_line("This layer does not add sufficient value over existing data.")
    add_line()
    if not irregular:
        add_line(f"Primary issue: User fell into a regular {bf2_stats['bpm_from_median']:.0f} BPM groove instead of tapping flourishes.")
    if not adds_new:
        add_line(f"Primary issue: Only {new_pct:.0f}% new taps — mostly duplicates existing layers.")
    if not not_redundant:
        add_line(f"Primary issue: {cf_alignment*100:.0f}% overlap with computed flourishes — redundant.")
    add_line()
    add_line("**Suggested use:** Use computed flourishes instead.")

add_line()

# ============================================================================
# SAVE REPORT
# ============================================================================

output_file = OUTPUT_DIR / "flourish2_comparison.md"
with open(output_file, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"Analysis complete. Report saved to: {output_file}")
print()
print("=" * 70)
print("QUICK SUMMARY")
print("=" * 70)
print(f"beat-flourish2: {len(beat_flourish2)} taps, CV={bf2_stats['cv']:.3f}, {bf2_stats['bpm_from_median']:.1f} BPM")
print(f"Consistent-beat overlap: {cb_alignment*100:.1f}% (beat-flourish was {bf_cb_alignment*100:.1f}%)")
print(f"Computed flourish overlap: {cf_alignment*100:.1f}%")
print(f"New taps: {new_pct:.1f}%")
print(f"Score: {score}/4 criteria met")
print("=" * 70)
