# Flourish Audio Properties Analysis

Analysis of audio features that characterize musical flourishes — off-beat moments that humans naturally emphasize.

## Summary

- **Audio file**: opiate_intro.wav
- **Duration**: 40.6 seconds
- **Flourish moments analyzed**: 91 total
  - High confidence (2+ sources): 19
  - Confidence distribution: 1=72 2=17 3=2 
- **On-beat moments**: 41
- **Feature window**: ±250ms
- **Features computed**: 17

## Key Findings

### Top Features Distinguishing Flourishes from On-Beat

1. **percussive_energy** (effect size: -0.533)
   - Flourish: 0.0340 ± 0.0223
   - On-beat: 0.0451 ± 0.0168
   - Flourishes are **lower** (p=0.0048)

2. **rms** (effect size: -0.475)
   - Flourish: 0.1315 ± 0.0766
   - On-beat: 0.1648 ± 0.0509
   - Flourishes are **lower** (p=0.0817)

3. **onset_density** (effect size: -0.469)
   - Flourish: 2.9670 ± 1.0608
   - On-beat: 3.4390 ± 0.8424
   - Flourishes are **lower** (p=0.0200)

4. **spectral_flux** (effect size: -0.459)
   - Flourish: 102.1610 ± 67.8090
   - On-beat: 131.4644 ± 51.9067
   - Flourishes are **lower** (p=0.1010)

5. **spectral_novelty** (effect size: -0.419)
   - Flourish: 141.1264 ± 94.6722
   - On-beat: 178.3989 ± 72.2143
   - Flourishes are **lower** (p=0.1127)

### Flourish Characterization

Based on the top features, flourishes are characterized by:

**Lower than on-beat:**
- percussive_energy: -24.7% change
- rms: -20.2% change
- onset_density: -13.7% change
- spectral_flux: -22.3% change
- spectral_novelty: -20.9% change

### Proposed Detection Strategy

**Strong predictors** (|effect| > 0.5, p < 0.01): 1 features

Recommended multi-feature detection rule:

```python
def detect_flourish(features):
    score = 0
    if features['percussive_energy'] < 0.0367:
        score += 0.53
    return score > 1.0  # threshold to tune
```

## Detailed Results

### Flourishes vs On-Beat

Features ranked by effect size (Cohen's d).

| Rank | Feature | Effect Size | Flourish Mean | On-Beat Mean | p-value | Sig |
|------|---------|-------------|---------------|--------------|---------|-----|
| 1 | percussive_energy | -0.533 | 0.0340 | 0.0451 | 0.0048 | ✓ |
| 2 | rms | -0.475 | 0.1315 | 0.1648 | 0.0817 |  |
| 3 | onset_density | -0.469 | 2.9670 | 3.4390 | 0.0200 |  |
| 4 | spectral_flux | -0.459 | 102.1610 | 131.4644 | 0.1010 |  |
| 5 | spectral_novelty | -0.419 | 141.1264 | 178.3989 | 0.1127 |  |
| 6 | energy_derivative | -0.371 | 0.0440 | 0.0560 | 0.0692 |  |
| 7 | zcr | -0.362 | 0.0645 | 0.0766 | 0.1005 |  |
| 8 | spectral_centroid | -0.349 | 3407.7172 | 3820.3039 | 0.0907 |  |
| 9 | onset_strength | -0.340 | 4.3575 | 5.1707 | 0.1368 |  |
| 10 | active_bands | -0.319 | 71.8836 | 88.6095 | 0.1336 |  |
| 11 | spectral_flatness | -0.286 | 0.0021 | 0.0026 | 0.1538 |  |
| 12 | harmonic_ratio | +0.258 | 0.7769 | 0.7614 | 0.1892 |  |
| 13 | high_freq_ratio | -0.255 | 0.0484 | 0.0584 | 0.2116 |  |
| 14 | spectral_bandwidth | -0.168 | 3997.3998 | 4107.5590 | 0.4087 |  |
| 15 | bass_to_mids | +0.135 | 251.9153 | 98.6553 | 0.0178 |  |
| 16 | centroid_derivative | -0.064 | -5.5658 | -3.2043 | 0.3041 |  |
| 17 | spectral_contrast | +0.012 | 19.4307 | 19.4164 | 0.7680 |  |

### High-Confidence Flourishes vs On-Beat

Same analysis using only flourishes with 2+ source agreement.

| Rank | Feature | Effect Size | High-Conf Mean | On-Beat Mean | p-value | Sig |
|------|---------|-------------|----------------|--------------|---------|-----|
| 1 | spectral_flatness | -0.265 | 0.0021 | 0.0026 | 0.4944 |  |
| 2 | zcr | -0.202 | 0.0705 | 0.0766 | 0.7029 |  |
| 3 | energy_derivative | +0.184 | 0.0614 | 0.0560 | 0.3608 |  |
| 4 | onset_density | +0.168 | 3.5789 | 3.4390 | 0.6203 |  |
| 5 | onset_strength | -0.158 | 4.7514 | 5.1707 | 0.9051 |  |
| 6 | rms | +0.153 | 0.1726 | 0.1648 | 0.2526 |  |
| 7 | active_bands | -0.142 | 80.4321 | 88.6095 | 0.7870 |  |
| 8 | bass_to_mids | -0.116 | 64.9324 | 98.6553 | 0.0921 |  |
| 9 | spectral_centroid | -0.110 | 3711.4346 | 3820.3039 | 0.3821 |  |
| 10 | spectral_flux | +0.079 | 135.5343 | 131.4644 | 0.4696 |  |
| 11 | spectral_contrast | +0.050 | 19.4752 | 19.4164 | 0.9114 |  |
| 12 | spectral_novelty | +0.039 | 181.2178 | 178.3989 | 0.7088 |  |
| 13 | spectral_bandwidth | +0.023 | 4121.4688 | 4107.5590 | 0.9746 |  |
| 14 | harmonic_ratio | -0.019 | 0.7602 | 0.7614 | 0.8612 |  |
| 15 | centroid_derivative | -0.019 | -3.8201 | -3.2043 | 0.7386 |  |
| 16 | high_freq_ratio | -0.014 | 0.0579 | 0.0584 | 0.6795 |  |
| 17 | percussive_energy | -0.007 | 0.0450 | 0.0451 | 0.7029 |  |

### Flourishes vs Non-Events (Baseline)

Comparing flourishes to random non-event moments.

| Rank | Feature | Effect Size | Flourish Mean | Baseline Mean | p-value | Sig |
|------|---------|-------------|---------------|---------------|---------|-----|
| 1 | spectral_bandwidth | +1.319 | 3997.3998 | 3054.1747 | 0.0000 | ✓ |
| 2 | bass_to_mids | -1.243 | 251.9153 | 5065.7148 | 0.0000 | ✓ |
| 3 | spectral_contrast | +1.240 | 19.4307 | 17.5027 | 0.0000 | ✓ |
| 4 | spectral_centroid | +1.162 | 3407.7172 | 1850.9750 | 0.0000 | ✓ |
| 5 | harmonic_ratio | -1.105 | 0.7769 | 0.8492 | 0.0000 | ✓ |
| 6 | onset_density | +1.054 | 2.9670 | 1.6842 | 0.0000 | ✓ |
| 7 | zcr | +0.910 | 0.0645 | 0.0333 | 0.0000 | ✓ |
| 8 | rms | +0.848 | 0.1315 | 0.0683 | 0.0002 | ✓ |
| 9 | percussive_energy | +0.831 | 0.0340 | 0.0158 | 0.0000 | ✓ |
| 10 | active_bands | +0.798 | 71.8836 | 36.8213 | 0.0000 | ✓ |
| 11 | spectral_flux | +0.691 | 102.1610 | 54.9726 | 0.0001 | ✓ |
| 12 | spectral_novelty | +0.679 | 141.1264 | 75.7319 | 0.0000 | ✓ |
| 13 | high_freq_ratio | +0.542 | 0.0484 | 0.0269 | 0.0001 | ✓ |
| 14 | energy_derivative | +0.515 | 0.0440 | 0.0263 | 0.0423 |  |
| 15 | onset_strength | +0.321 | 4.3575 | 3.4086 | 0.0001 | ✓ |
| 16 | spectral_flatness | +0.172 | 0.0021 | 0.0015 | 0.0000 | ✓ |
| 17 | centroid_derivative | +0.095 | -5.5658 | -11.4293 | 0.1561 |  |

## Comparison to Air Taps

Air tap analysis not yet available. Run `air_audio_properties.py` for comparison.

## Interpretation

### What Makes a Flourish?

**Spectral characteristics**:
- spectral_flux: darker/narrower spectrum during flourishes
- spectral_novelty: darker/narrower spectrum during flourishes

**Temporal characteristics**:
- onset_density: less activity/change during flourishes
- spectral_flux: less activity/change during flourishes

**Energy characteristics**:
- percussive_energy: quieter during flourishes
- rms: quieter during flourishes

### Flourish vs Beat: The Core Distinction

On-beat moments are **predictable, periodic, metronomic**.
Flourish moments are **surprising, non-periodic, noteworthy**.

The acoustic difference between them tells us what makes a moment 'flourish-worthy':

**Primary acoustic signature**: percussive_energy
- Effect size: -0.533
- This is a MEDIUM effect

### Generalizability to Other Songs

These findings are based on Tool's 'Opiate Intro' (psych rock, heavy, complex rhythms).

**Likely to generalize**:
- High onset strength/spectral flux at flourish moments (transient events)
- Spectral novelty (unexpected timbral changes)
- Percussive energy spikes (cymbal crashes, fills)

**May be genre-specific**:
- Exact threshold values (will vary by production style)
- Bass-to-mids ratio (depends on mix balance)
- Harmonic ratio (varies by instrumentation)

**Recommendation**: Test on multiple genres (electronic, ambient, jazz) to find universal features.

## Next Steps

1. **Validate detection rule** on held-out segments of this track
2. **Test on other songs** in the audio-segments catalog
3. **Compare to air tap features** to see if they're orthogonal or overlapping
4. **Temporal context**: Do flourishes need preceding quiet/steady state?
5. **Multi-scale analysis**: Are there micro-flourishes vs macro-flourishes?
6. **LED mapping**: How should flourish intensity map to LED effects?

## Methodology

- **Ground truth**: 91 flourish moments from 3 sources
  - Computed from free-form beat taps (off-beat taps)
  - beat-flourish intentional annotation layer
  - beat-flourish2 second annotation layer
  - Confidence = number of agreeing sources (1-3)
- **Comparison sets**:
  - 41 on-beat moments (consistent-beat taps)
  - 38 random non-event moments (baseline)
- **Feature extraction**: 17 features at 44100 Hz, hop=512
- **Feature window**: ±250ms around each tap
  - Peak value for onset/flux/novelty features
  - Mean value for spectral/energy features
- **Statistics**: Cohen's d effect size, Mann-Whitney U test
- **Significance threshold**: p < 0.01

