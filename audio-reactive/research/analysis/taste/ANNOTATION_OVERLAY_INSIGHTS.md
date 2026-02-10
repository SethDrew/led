# Opiate Intro — Annotation Overlay Analysis

## Overview
This document summarizes insights from overlaying user tap annotations on audio analysis features for Tool's "Opiate Intro" (40.57 seconds).

## Visualization File
`/Users/KO16K39/Documents/led/audio-reactive/research/analysis/opiate_annotation_overlay.png`

## User Annotation Summary

### Beat Taps (n=72)
- **Mean interval**: 0.563s
- **Implied tempo**: 106.5 BPM
- **Standard deviation**: 0.409s
- The high standard deviation indicates tempo/subdivision changes throughout the song

### Change Markers (n=6)
Divides the song into 7 distinct sections:
- **Section A** (0-8.07s): 8.07s — Intro drone/buildup
- **Section B** (8.07-17.10s): 9.03s — Main groove enters
- **Section C** (17.10-24.27s): 7.17s — Groove continues
- **Section D** (24.27-28.70s): 4.43s — Intensity change/fill section
- **Section E** (28.70-34.50s): 5.80s — Rhythmic variation
- **Section F** (34.50-37.37s): 2.88s — Final fill/transition
- **Section G** (37.37-40.57s): 3.20s — Ending

### Air Taps (n=24)
Concentrated in **2 distinct regions**:
1. **Region 1** (1.61-3.90s): 2.28s duration — Early intro ambience
2. **Region 2** (28.92-33.55s): 4.64s duration — Mid-song atmospheric section

## Key Findings

### 1. Beat Detection Failure Confirmed
- **User tempo**: 106.5 BPM (ground truth from taps)
- **Algorithm detected**: 161.5 BPM
- **Ratio**: 1.52x user tempo (tempo doubling)
- **Panel 4 shows**: Red user taps align with strong onset peaks; yellow algorithmic beats often fall between them or on subdivisions

**Why it matters**: librosa's beat tracker locked onto hi-hat/subdivision patterns instead of the primary kick/snare pulse. This confirms the need for bass-weighted onset detection or tempo constraints for rock music.

### 2. Air Taps Correlate with Harmonic Dominance
**Panel 5 (HPSS) reveals**:
- Air Region 1 (1.61-3.90s): Purple harmonic line dominates over orange percussive line
- Air Region 2 (28.92-33.55s): Strong harmonic presence, percussive drops relative to other sections
- **Hypothesis validated**: User's "air" feeling maps to moments where harmonic content dominates over percussive elements

**Actionable**: For LED mapping, "airy" sections could:
- Increase brightness/saturation
- Use slower, flowing animations
- Emphasize mid-high frequency LED zones
- Reduce transient/strobe effects

### 3. Section Changes Align with Energy Shifts
**Panel 3 (Band Energies) shows**:
- Section A: Low energy across all bands (intro buildup)
- Section B: Sharp increase in bass/mid energy (main riff enters at ~8s)
- Section D: Peak energy burst (fill section at ~24-29s)
- Section E-G: Declining energy toward end

**Change markers** (white dashed lines) occur at visible transitions in:
- Waveform amplitude (Panel 1)
- Spectral content (Panel 2)
- Band energy distribution (Panel 3)

This validates that user's structural perception aligns with measurable audio features.

### 4. Tempo Instability in User Taps
**Beat interval std dev = 0.409s** is high relative to mean interval (0.563s).

Examining the tap timeline:
- Early sections: More regular tapping (quarter notes)
- Section D (28-29s): Rapid burst of taps (fills/subdivisions)
- The "imperfections" capture **musical intent** (tempo changes, fills, emphasis shifts)

**Key insight**: Don't try to quantize or "fix" user taps. The variability IS the data — it represents where the user felt compelled to tap, which correlates with musical emphasis.

### 5. Mel Spectrogram Shows Tonal Shifts
**Panel 2** with air tap overlays:
- Air Region 1 (cyan highlights): Lower frequency content, fewer harmonics visible
- Air Region 2 (cyan highlights): More complex harmonic structure in mid-range (500-2000 Hz)
- Both regions show **less percussive transients** (vertical stripes) compared to surrounding sections

**Pattern**: "Air" doesn't just mean harmonic dominance — it also correlates with **sustained tones** vs. sharp transients.

## Implications for LED Mapping

### For "Beat" Layer
- Don't rely on algorithmic beat detection for rock/complex music
- Use user taps to create a "feel template" that can adapt to new audio
- Consider training a model: onset strength + bass energy → beat probability

### For "Air" Layer
- **Primary feature**: Harmonic-to-percussive ratio (HPSS)
- **Secondary feature**: Spectral flatness (low flatness = tonal/harmonic)
- **Tertiary feature**: Sustained energy vs. transient energy
- Threshold-based detection likely insufficient — needs multi-feature scoring

### For "Changes" Layer
- Spectral flux across multiple bands (detect simultaneous changes)
- RMS energy derivatives (sudden increases/decreases)
- Timbral novelty (madmom or librosa chromagram comparison)
- User's 6 change markers are **sparse** — algorithm should be conservative

## Next Steps

1. **Correlate air taps with features**:
   - Compute harmonic/percussive ratio at each air tap timestamp
   - Compute spectral centroid, flatness, rolloff at each tap
   - Build feature vector: `[H/P ratio, flatness, centroid, ...]`
   - Find thresholds or train classifier

2. **Test on other segments**:
   - Do air taps in "ambient.wav" show same HPSS correlation?
   - Do beat taps in "electronic_beat.wav" match algorithm better (simpler rhythm)?

3. **Build real-time feature extractor**:
   - Streaming HPSS (need buffering strategy)
   - Rolling harmonic/percussive RMS (attack/decay envelope)
   - Onset strength with bass-weighting

4. **Create "feeling template" format**:
   ```yaml
   air:
     features:
       - harmonic_percussive_ratio: {min: 1.2, weight: 0.6}
       - spectral_flatness: {max: 0.3, weight: 0.3}
       - transient_density: {max: 0.5, weight: 0.1}
     led_response:
       brightness: increase
       saturation: increase
       animation_speed: decrease
       zones: [mid, high]
   ```

## Files Generated
- **Script**: `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/scripts/annotation_overlay.py`
- **Visualization**: `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/opiate_annotation_overlay.png`
- **This document**: `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/ANNOTATION_OVERLAY_INSIGHTS.md`
