# Airiness Feature Analysis
## Overview
- **Audio file**: opiate_intro.wav
- **Duration**: 40.57 seconds
- **Air taps**: 24 annotations
- **Analysis window**: ±250ms around each tap
- **Cluster 1** (early guitar trail-off): 9 taps at 1.61-3.90s
- **Cluster 2** (vocal build): 15 taps at 28.92-33.55s

## Feature Rankings (by Effect Size)

### All Air Regions vs Non-Air
Features ranked by how well they discriminate air-tapped moments from the rest of the track.

| Rank | Feature | Air Mean | Air Std | Non-Air Mean | Non-Air Std | Effect Size | p-value |
|------|---------|----------|---------|--------------|-------------|-------------|----------|
| 1 | **novelty** | 0.9819 | 0.0048 | 0.9855 | 0.0062 | **-0.604** | 0.0000 |
| 2 | **spectral_contrast** | 18.3783 | 2.3184 | 19.3362 | 2.2755 | **-0.420** | 0.0000 |
| 3 | **spectral_centroid** | 2932.4983 | 1691.8843 | 3338.9107 | 1634.1271 | **-0.247** | 0.0000 |
| 4 | **spectral_bandwidth** | 3707.6090 | 1005.2383 | 3902.9553 | 839.8290 | **-0.225** | 0.0000 |
| 5 | **energy_trajectory** | 0.1001 | 0.0778 | 0.1197 | 0.0898 | **-0.222** | 0.0000 |
| 6 | **active_sources** | 9.2469 | 9.5051 | 12.4032 | 15.2359 | **-0.218** | 0.0000 |
| 7 | **rms_energy** | 0.1405 | 0.0853 | 0.1266 | 0.0849 | **0.164** | 0.0004 |
| 8 | **harmonic_ratio** | 0.7940 | 0.0965 | 0.7806 | 0.1145 | **0.119** | 0.0091 |
| 9 | **sparseness** | 2.5674 | 1.1793 | 2.8044 | 2.3113 | **-0.109** | 0.0169 |
| 10 | **onset_density** | 5.2259 | 2.9072 | 5.0120 | 2.1094 | **0.095** | 0.0386 |
| 11 | **spectral_flatness** | 0.0016 | 0.0037 | 0.0022 | 0.0074 | **-0.095** | 0.0388 |

### Cluster 1 (Early Guitar) vs Non-Air
| Rank | Feature | Air Mean | Non-Air Mean | Effect Size | p-value |
|------|---------|----------|--------------|-------------|----------|
| 1 | **spectral_bandwidth** | 2305.1644 | 3902.9553 | **-1.951** | 0.0000 |
| 2 | **onset_density** | 1.1222 | 5.0120 | **-1.889** | 0.0000 |
| 3 | **spectral_centroid** | 575.2443 | 3338.9107 | **-1.741** | 0.0000 |
| 4 | **spectral_contrast** | 15.8127 | 19.3362 | **-1.579** | 0.0000 |
| 5 | **energy_trajectory** | 0.0049 | 0.1197 | **-1.316** | 0.0000 |
| 6 | **rms_energy** | 0.0294 | 0.1266 | **-1.179** | 0.0000 |
| 7 | **novelty** | 0.9785 | 0.9855 | **-1.128** | 0.0000 |
| 8 | **harmonic_ratio** | 0.8875 | 0.7806 | **0.957** | 0.0000 |
| 9 | **active_sources** | 3.9944 | 12.4032 | **-0.568** | 0.0000 |
| 10 | **sparseness** | 1.9444 | 2.8044 | **-0.382** | 0.0000 |
| 11 | **spectral_flatness** | 0.0000 | 0.0022 | **-0.311** | 0.0001 |

### Cluster 2 (Vocal Build) vs Non-Air
| Rank | Feature | Air Mean | Non-Air Mean | Effect Size | p-value |
|------|---------|----------|--------------|-------------|----------|
| 1 | **onset_density** | 7.1151 | 5.0120 | **1.051** | 0.0000 |
| 2 | **rms_energy** | 0.1917 | 0.1266 | **0.799** | 0.0000 |
| 3 | **spectral_bandwidth** | 4353.2357 | 3902.9553 | **0.565** | 0.0000 |
| 4 | **spectral_centroid** | 4017.6791 | 3338.9107 | **0.437** | 0.0000 |
| 5 | **novelty** | 0.9834 | 0.9855 | **-0.350** | 0.0000 |
| 6 | **energy_trajectory** | 0.1440 | 0.1197 | **0.282** | 0.0000 |
| 7 | **harmonic_ratio** | 0.7509 | 0.7806 | **-0.267** | 0.0000 |
| 8 | **spectral_contrast** | 19.5594 | 19.3362 | **0.101** | 0.0606 |
| 9 | **active_sources** | 11.6650 | 12.4032 | **-0.050** | 0.3534 |
| 10 | **sparseness** | 2.8542 | 2.8044 | **0.023** | 0.6756 |
| 11 | **spectral_flatness** | 0.0023 | 0.0022 | **0.010** | 0.8506 |

## Interpretation

### Top Predictive Features (Shared Function)
Features that appear in top 5 for BOTH clusters suggest they capture the functional essence of 'airiness':

- **spectral_bandwidth**: Effect size = -0.225 (C1: -1.951, C2: 0.565)
  - Air regions: 2305.1644 (C1), 4353.2357 (C2)
  - Non-air: 3902.9553
  - **LOWER during airy moments**

- **spectral_centroid**: Effect size = -0.247 (C1: -1.741, C2: 0.437)
  - Air regions: 575.2443 (C1), 4017.6791 (C2)
  - Non-air: 3338.9107
  - **LOWER during airy moments**

- **onset_density**: Effect size = 0.095 (C1: -1.889, C2: 1.051)
  - Air regions: 1.1222 (C1), 7.1151 (C2)
  - Non-air: 5.0120
  - **HIGHER during airy moments**


### Cluster-Specific Features (Form-Dependent)
Features that work well for only one cluster:

**Cluster 1 only** (guitar trail-off):
- energy_trajectory (effect: -1.316)
- spectral_contrast (effect: -1.579)

**Cluster 2 only** (vocal build):
- rms_energy (effect: 0.799)
- novelty (effect: -0.350)


### Effect Size Interpretation
- **|d| < 0.2**: Negligible difference
- **|d| = 0.2-0.5**: Small effect
- **|d| = 0.5-0.8**: Medium effect
- **|d| > 0.8**: Large effect

**Positive effect size** means the feature is HIGHER during airy moments.
**Negative effect size** means the feature is LOWER during airy moments.
