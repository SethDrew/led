# Build vs Climax Analysis: Opiate Intro

Analysis of two regions containing similar vocal phrases ('like meeeeee') that feel different.

## Regions

- **BUILD**: 28.9s - 33.5s (4.60s)
  - User annotation: 'airy', anticipation
  - First two 'like meeeeee' vocals

- **CLIMAX**: 34.0s - 40.0s (6.00s)
  - User annotation: payoff, arrival, culmination
  - Last two 'like meeeeee' vocals

## Top Distinguishing Features

Features with largest relative difference between regions:

| Feature | Build | Climax | Change | % Change |
|---------|-------|--------|--------|----------|
| Centroid Trajectory | 0.0360 | 2.1139 | +2.0779 | +5778.9% |
| Harmonic Ratio Trajectory | 0.000055 | -0.000046 | -0.000101 | -183.7% |
| Rms Trajectory | -0.000015 | -0.000034 | -0.000019 | -128.0% |
| Bass Mean | 0.1893 | 0.0588 | -0.1305 | -69.0% |
| Mids Mean | 0.0525 | 0.0861 | +0.0336 | +63.9% |
| Flatness Std | 0.003896 | 0.005266 | +0.001370 | +35.2% |
| Num Onsets | 21.00 | 28.00 | +7.00 | +33.3% |
| Flatness Mean | 0.002282 | 0.003037 | +0.000755 | +33.1% |
| High Mids Mean | 0.0342 | 0.0433 | +0.0092 | +26.9% |
| Harmonic Ratio Std | 0.0765 | 0.0954 | +0.0188 | +24.6% |
| Sub Bass Mean | 0.7206 | 0.8084 | +0.0878 | +12.2% |
| Rms Range | 0.2571 | 0.2319 | -0.0252 | -9.8% |
| Bandwidth Std | 313.98 | 341.81 | +27.83 | +8.9% |
| Rms Std | 0.0457 | 0.0423 | -0.0034 | -7.4% |
| Onset Strength Max | 7.26 | 7.76 | +0.50 | +6.9% |

## Complete Feature Comparison

### Energy Features

| Feature | Build | Climax | Difference |
|---------|-------|--------|------------|
| Rms Mean | 0.1890 | 0.1899 | +0.0009 (+0.5%) |
| Rms Std | 0.0457 | 0.0423 | -0.0034 (-7.4%) |
| Rms Max | 0.3704 | 0.3468 | -0.0236 (-6.4%) |
| Rms Min | 0.1134 | 0.1150 | +0.0016 (+1.4%) |
| Rms Range | 0.2571 | 0.2319 | -0.0252 (-9.8%) |
| Rms Trajectory | -0.0000 | -0.0000 | -0.0000 (-128.0%) |

### Spectral Features

| Feature | Build | Climax | Difference |
|---------|-------|--------|------------|
| Centroid Mean | 4059.08 | 4253.48 | +194.40 (+4.8%) |
| Centroid Std | 684.62 | 683.95 | -0.67 (-0.1%) |
| Centroid Trajectory | 0.04 | 2.11 | +2.08 (+5778.9%) |
| Bandwidth Mean | 4394.96 | 4445.20 | +50.24 (+1.1%) |
| Bandwidth Std | 313.98 | 341.81 | +27.83 (+8.9%) |
| Flatness Mean | 0.00 | 0.00 | +0.00 (+33.1%) |
| Flatness Std | 0.00 | 0.01 | +0.00 (+35.2%) |
| Contrast Mean | 19.47 | 19.11 | -0.36 (-1.8%) |
| Contrast Std | 12.38 | 12.55 | +0.17 (+1.4%) |

### Frequency Band Energy Ratios

| Band | Build | Climax | Difference |
|------|-------|--------|------------|
| Sub Bass | 0.7206 | 0.8084 | +0.0878 (+12.2%) |
| Bass | 0.1893 | 0.0588 | -0.1305 (-69.0%) |
| Mids | 0.0525 | 0.0861 | +0.0336 (+63.9%) |
| High Mids | 0.0342 | 0.0433 | +0.0092 (+26.9%) |
| Treble | 0.0034 | 0.0034 | -0.0000 (-0.1%) |

### Harmonic-Percussive Separation

| Feature | Build | Climax | Difference |
|---------|-------|--------|------------|
| Harmonic Ratio Mean | 0.7947 | 0.8191 | +0.0244 (+3.1%) |
| Harmonic Ratio Std | 0.0765 | 0.0954 | +0.0188 (+24.6%) |
| Harmonic Ratio Trajectory | 0.0001 | -0.0000 | -0.0001 (-183.7%) |
| Percussive Ratio Mean | 0.2684 | 0.2542 | -0.0142 (-5.3%) |

### Onset Features

| Feature | Build | Climax | Difference |
|---------|-------|--------|------------|
| Onset Density | 4.5652 | 4.6667 | +0.1014 (+2.2%) |
| Onset Strength Mean | 0.9771 | 1.0161 | +0.0391 (+4.0%) |
| Onset Strength Max | 7.2564 | 7.7602 | +0.5038 (+6.9%) |
| Onset Strength Std | 0.8108 | 0.8456 | +0.0348 (+4.3%) |
| Num Onsets | 21.0000 | 28.0000 | +7.0000 (+33.3%) |

## Trajectory Analysis

Changes over time within each region:

**Energy trajectory (RMS slope)**:
- Build: -0.000015 (decreasing)
- Climax: -0.000034 (decreasing)

**Brightness trajectory (Centroid slope)**:
- Build: +0.04 Hz/frame (brightening)
- Climax: +2.11 Hz/frame (brightening)

**Harmonic ratio trajectory**:
- Build: +0.000055 (more harmonic)
- Climax: -0.000046 (more percussive)

## Individual Vocal Phrase Analysis

Detected 8 phrases in BUILD region and 9 phrases in CLIMAX region.

### BUILD Region Phrases

| # | Time (s) | Duration | RMS Mean | RMS Max | Centroid | Flatness |
|---|----------|----------|----------|---------|----------|----------|
| 1 | 0.03 | 0.78s | 0.1789 | 0.2774 | 4377 Hz | 0.0043 |
| 2 | 0.55 | 1.30s | 0.1886 | 0.2818 | 4121 Hz | 0.0029 |
| 3 | 1.07 | 1.50s | 0.1943 | 0.3299 | 4023 Hz | 0.0029 |
| 4 | 1.99 | 1.50s | 0.1917 | 0.3300 | 4051 Hz | 0.0026 |
| 5 | 2.52 | 1.50s | 0.1895 | 0.2839 | 3953 Hz | 0.0023 |
| 6 | 3.44 | 1.50s | 0.1832 | 0.2856 | 4021 Hz | 0.0020 |
| 7 | 3.96 | 1.40s | 0.1826 | 0.3619 | 4195 Hz | 0.0024 |
| 8 | 4.49 | 0.87s | 0.1886 | 0.3619 | 4057 Hz | 0.0019 |

### CLIMAX Region Phrases

| # | Time (s) | Duration | RMS Mean | RMS Max | Centroid | Flatness |
|---|----------|----------|----------|---------|----------|----------|
| 1 | 0.12 | 0.87s | 0.2045 | 0.3299 | 3709 Hz | 0.0018 |
| 2 | 0.66 | 1.41s | 0.1956 | 0.3299 | 3941 Hz | 0.0027 |
| 3 | 1.22 | 1.50s | 0.1928 | 0.2817 | 3921 Hz | 0.0024 |
| 4 | 1.74 | 1.50s | 0.1891 | 0.2967 | 3937 Hz | 0.0024 |
| 5 | 2.64 | 1.50s | 0.1909 | 0.2968 | 4055 Hz | 0.0026 |
| 6 | 3.18 | 1.50s | 0.1923 | 0.2943 | 4271 Hz | 0.0027 |
| 7 | 4.10 | 1.50s | 0.1805 | 0.2943 | 4745 Hz | 0.0055 |
| 8 | 4.98 | 1.50s | 0.1854 | 0.3462 | 4681 Hz | 0.0051 |
| 9 | 5.53 | 1.23s | 0.1883 | 0.3462 | 4525 Hz | 0.0033 |

### Phrase-Level Comparison

Average features across individual phrases:

| Feature | Build Phrases | Climax Phrases | Difference |
|---------|---------------|----------------|------------|
| RMS Mean | 0.1872 | 0.1911 | +0.0039 (+2.1%) |
| Centroid | 4100 Hz | 4198 Hz | +98 Hz (+2.4%) |
| Flatness | 0.0027 | 0.0032 | +0.0005 (+19.5%) |

## Interpretation

### What Makes the CLIMAX Feel Like Climax?

**1. Similar Energy**: RMS only differs by 0.5% — the 'arrival' feeling is NOT primarily about loudness.

**2. Similar Brightness**: Spectral centroid only differs by 4.8% — not a primary distinguisher.

**3. Similar Harmonic Content**: Harmonic ratio only differs by 3.1%.

**4. Frequency Balance Shifts**:
- Bass: -69.0% (climax: 0.059, build: 0.189)
- Mids: +63.9% (climax: 0.086, build: 0.053)
- Treble: -0.1% (climax: 0.003, build: 0.003)

**5. Temporal Context (Trajectory)**:

**6. Dynamic Range**: Climax has -9.8% more dynamic variation (range: 0.2319 vs 0.2571).

**7. Rhythmic Density**: +2.2% change in onset density (4.67 vs 4.57 onsets/sec).

### Summary

The distinction between BUILD and CLIMAX appears to be driven by:

1. **Centroid Trajectory**: Climax is 5778.9% higher
2. **Harmonic Ratio Trajectory**: Climax is 183.7% lower
3. **Rms Trajectory**: Climax is 128.0% lower

**Individual vocal phrases ARE spectrally different** between regions — the 'like meeeeee' vocals themselves have changed.

---
*Analysis completed: 8 build phrases, 9 climax phrases detected*