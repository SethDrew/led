# Airiness Feature Analysis - Key Findings

## Executive Summary

Analyzed 24 "air" annotations in Tool's "Opiate" intro to identify which audio features correlate with the user's sense of "airiness as functional preparation/transition." Found that **airiness is NOT a single acoustic signature** but rather a context-dependent phenomenon with different acoustic mechanisms serving the same functional purpose.

## The Two Clusters

### Cluster 1: Guitar Trail-Off (1.6-3.9s, 9 taps)
**Acoustic profile**: Single guitar pick trailing off alone in the mix
- **Very low** spectral centroid (575 Hz vs 3339 Hz average)
- **Very low** spectral bandwidth (2305 Hz vs 3903 Hz average)
- **Very low** onset density (1.1 vs 5.0 average)
- **Very low** energy trajectory (-1.3 effect size)
- **High** harmonic ratio (0.89 vs 0.78 average)

**Interpretation**: This is "airy" because it's **sparse, isolated, and static**. A single tonal element decaying with no competing sounds and no future energy increase. The *novelty* of this sound (first time hearing it) contributes to the feeling.

### Cluster 2: Vocal Build (28.9-33.5s, 15 taps)
**Acoustic profile**: Drawn-out vocals building tension before crescendo
- **High** spectral centroid (4018 Hz vs 3339 Hz average)
- **High** spectral bandwidth (4353 Hz vs 3903 Hz average)
- **High** onset density (7.1 vs 5.0 average)
- **High** RMS energy (0.19 vs 0.13 average)
- **Positive** energy trajectory (0.28 effect size)

**Interpretation**: This is "airy" because it's **sustained, building, and anticipatory**. A vocal passage with rising energy that *prepares you* for the explosive section that follows. The higher frequency content (vocal sibilance/breath) and continuous onset activity create the tension.

## Shared Functional Features

Despite wildly different acoustic profiles, three features appear in the top 5 for BOTH clusters:

### 1. Spectral Bandwidth
- **Overall effect**: -0.23 (small)
- **Cluster 1**: -1.95 (huge) - much narrower bandwidth
- **Cluster 2**: +0.57 (medium) - wider bandwidth
- **Paradox**: Goes opposite directions but still discriminates in both cases
- **Interpretation**: "Airiness" correlates with bandwidth that *differs from the norm*. Cluster 1 is narrow (isolated), Cluster 2 is wide (complex/breathy).

### 2. Spectral Centroid (Brightness)
- **Overall effect**: -0.25 (small)
- **Cluster 1**: -1.74 (huge) - very dark
- **Cluster 2**: +0.44 (medium) - brighter
- **Paradox**: Same as bandwidth
- **Interpretation**: "Airiness" happens at frequency extremes — either very low (guitar) or upper-mid/high (vocals).

### 3. Onset Density
- **Overall effect**: +0.09 (negligible)
- **Cluster 1**: -1.89 (huge) - very sparse
- **Cluster 2**: +1.05 (large) - dense
- **Paradox**: Completely opposite directions
- **Interpretation**: "Airiness" correlates with onset density that creates *contrast with the groove*. Cluster 1 is sparse (no groove yet), Cluster 2 is dense (building toward groove explosion).

## Cluster-Specific Features

### Cluster 1 Only
- **Energy trajectory** (-1.32 effect): No anticipation, just decay
- **Spectral contrast** (-1.58 effect): Low contrast (simple tone)

### Cluster 2 Only
- **RMS energy** (+0.80 effect): Louder sustained sound
- **Novelty** (-0.35 effect): Less novel (familiar vocal texture)

## Top Overall Predictor: Novelty (-0.60 effect)

**Novelty** (how different the current spectral pattern is from recent context) is the single best predictor across all air moments, with LOWER novelty during airy sections.

**Counterintuitive finding**: You'd expect "airy" moments to be novel, but they're actually *familiar patterns used in a preparatory context*. The guitar trail-off and vocal build both use recognizable textures, just deployed functionally as transitions.

## What This Means for LED Mapping

### Don't Look for "The Airy Sound"
There is no single acoustic signature. Cluster 1 and Cluster 2 sound completely different but feel the same functionally.

### Look for Contextual Markers Instead
The shared features (bandwidth, centroid, onset density) work because they detect **deviation from the track's norm**:
- Cluster 1: Deviates by being sparse, dark, and low-energy
- Cluster 2: Deviates by being dense, bright, and building

### Use Multi-Dimensional Detection
A good "airiness detector" might be:
```
airiness_score = (
    abs(spectral_bandwidth - track_mean_bandwidth) * w1 +
    abs(spectral_centroid - track_mean_centroid) * w2 +
    abs(onset_density - track_mean_density) * w3 +
    low_novelty_bonus * w4 +
    energy_trajectory_factor * w5
)
```

Where the detector rewards **difference from norm** rather than specific values.

### LED Mapping Strategies

**Option A: Single "Airiness" Visual**
- Treat both types as "preparation/transition" moments
- Use consistent visual (e.g., slow pulse, color fade, reduced saturation)
- Ignores the acoustic differences, focuses on function

**Option B: Two Types of Airiness**
- Cluster 1 → "Sparse Air" (fade out, isolate single element, dim)
- Cluster 2 → "Building Air" (pulse acceleration, brightness ramp, tension colors)
- Respects the acoustic differences and maps them differently

**Option C: Context-Aware Airiness**
- Use novelty + energy trajectory to distinguish:
  - Low energy trajectory → fading/decaying visual
  - Positive energy trajectory → building/anticipating visual
- Feature values determine visual parameters automatically

## Recommended Next Steps

1. **Test on more songs**: Does this pattern hold for other genres/artists?
2. **Annotate more layers**: Analyze "heavy," "tense," etc. to build feeling vocabulary
3. **Build real-time detector**: Implement multi-feature detection with sliding window
4. **User feedback loop**: Show LED responses to detected airiness, refine weights
5. **Temporal context**: Current analysis uses ±250ms window; try longer contexts (1-5 seconds)

## Technical Notes

- **Analysis window**: ±250ms around each tap (capture local context)
- **Novelty window**: 2-second lookback (detect spectral similarity to recent past)
- **Energy trajectory**: 2.5-second lookahead (detect future energy increase)
- **Effect sizes**: Cohen's d comparing air vs non-air frames
- **Statistical significance**: All top features p < 0.001

## Files Generated

- `/Users/KO16K39/Documents/led/interactivity-research/analysis/air_feature_analysis.md` - Full statistical report
- `/Users/KO16K39/Documents/led/interactivity-research/analysis/air_feature_analysis.yaml` - Per-tap feature data
- `/Users/KO16K39/Documents/led/interactivity-research/analysis/scripts/air_feature_analysis.py` - Reusable analysis script
