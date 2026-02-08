# Flourish Comparison Analysis

**Analysis Date:** 1770350352.0650084

## Question

Does the user's `beat-flourish` annotation layer add genuinely new information beyond what we already captured in computed flourishes (beat taps that don't align with consistent-beat)?

## Summary

**Verdict:** ADDS SIGNIFICANT NEW INFORMATION

Beat-flourish captures substantially different flourish moments compared to computed flourishes.

## Datasets

- **Beat-flourish layer:** 76 taps (user tapped trying to hit only flourishes)
- **Computed flourishes:** 39 taps (beat taps >150ms from any consistent-beat tap)
- **Audio duration:** 40.57 seconds

## 1. Overlap Analysis

Using ±150ms matching window:

- **Overlaps:** 25 taps (32.9%)
  - These beat-flourish taps matched computed flourishes — we already had this info
- **New:** 51 taps (67.1%)
  - These beat-flourish taps did NOT match computed flourishes — genuinely new
- **Missed:** 12 taps (30.8%)
  - Computed flourishes that beat-flourish layer missed

### Sample Overlaps

| Beat-Flourish | Computed Flourish | Distance |
|---------------|-------------------|----------|
| 5.232s | 5.346s | 114ms |
| 7.216s | 7.276s | 60ms |
| 8.028s | 8.002s | 26ms |
| 8.571s | 8.481s | 90ms |
| 9.102s | 9.048s | 54ms |
| 11.735s | 11.690s | 45ms |
| 14.691s | 14.746s | 55ms |
| 15.198s | 15.312s | 114ms |
| 15.874s | 15.901s | 27ms |
| 17.735s | 17.704s | 31ms |

## 2. What Are the New Taps?

Analyzing 51 new beat-flourish taps that don't match computed flourishes:

- **Near consistent-beat:** 27 (52.9%)
  - User tapped moments they previously called "consistent" as flourishes
  - Suggests ambiguity or context-dependent perception
- **Near change markers:** 6 (11.8%)
  - Flourishes at structural boundaries
- **Near air taps:** 9 (17.6%)
  - Airy moments perceived as flourish-worthy
- **In original beat layer:** 22 (43.1%)
  - Present in original beat but aligned with consistent-beat (so not in computed flourishes)

### Sample New Taps with Context

| Beat-Flourish | Dist to Consistent | Dist to Change | Dist to Air | In Beat? |
|---------------|-------------------|----------------|-------------|----------|
| 1.526s | 24ms | 6547ms | 86ms | Yes |
| 4.402s | 2900ms | 3671ms | 507ms | No |
| 6.258s | 4756ms | 1815ms | 2363ms | No |
| 10.147s | 946ms | 2074ms | 6252ms | No |
| 11.086s | 7ms | 3013ms | 7191ms | Yes |
| 12.457s | 193ms | 4384ms | 8562ms | No |
| 13.442s | 41ms | 3663ms | 9547ms | Yes |
| 14.227s | 90ms | 2878ms | 10332ms | Yes |
| 16.354s | 105ms | 751ms | 12459ms | No |
| 17.174s | 28ms | 69ms | 11745ms | No |
| 20.132s | 30ms | 3027ms | 8787ms | Yes |
| 20.331s | 169ms | 3226ms | 8588ms | No |
| 21.142s | 270ms | 3129ms | 7777ms | No |
| 21.612s | 40ms | 2659ms | 7307ms | No |
| 22.138s | 103ms | 2133ms | 6781ms | Yes |

## 3. Density Comparison

Using 2.0s sliding window:

| Layer | Mean Density (taps/sec) |
|-------|------------------------|
| Beat-flourish | 1.86 |
| Computed flourishes | 0.94 |
| Original beat | 1.75 |

**Density correlation:** -0.095

Of 406 time points:
- Beat-flourish **denser:** 282 (69.5%)
- Beat-flourish **sparser:** 66 (16.3%)
- **Similar:** 58 (14.3%)

## 4. Temporal Pattern Analysis

### Beat-Flourish Intervals

- **Median:** 0.452s (132.7 BPM)
- **Mean:** 0.518s ± 0.345s
- **CV:** 0.666
- **Range:** 0.149s to 2.876s
- **Quartiles:** Q25=0.359s, Q75=0.585s

### Computed Flourishes Intervals

- **Median:** 0.606s (98.9 BPM)
- **Mean:** 1.044s ± 1.183s
- **CV:** 1.133
- **Range:** 0.082s to 6.505s

### Interpretation

Beat-flourish has **lower variability** (lower CV), suggesting more regular tapping.
Beat-flourish median interval **differs from consistent-beat**, suggesting truly off-grid tapping.

## 5. Correlation with Existing Layers

Fraction of beat-flourish taps within ±150ms of:

- **Air taps:** 14.5%
- **Consistent-beat taps:** 38.2%
- **Change markers:** 10.5% (using ±300ms window)

For comparison:
- **Computed flourishes aligned with air:** 12.8%

Beat-flourish shows **unexpected alignment with consistent-beat** — suggests ambiguity in what counts as a flourish.

## 6. Conclusion

### ADDS SIGNIFICANT NEW INFORMATION

Beat-flourish captures substantially different flourish moments compared to computed flourishes.

### Specific New Insights

The beat-flourish layer captures:

1. **Ambiguous moments** — 27 taps near consistent-beat that user now perceives as flourishes

### Missed Flourishes

The beat-flourish layer **missed 12 computed flourishes** (30.8%).
This suggests:
- User's focused flourish-tapping missed some off-grid moments from original beat layer
- Or user's perception of "flourish" evolved between tapping sessions
- Or some computed flourishes weren't actually flourish-worthy to the user

#### Sample Missed Flourishes

| Computed Flourish | Nearest Beat-Flourish | Distance |
|-------------------|----------------------|----------|
| 0.329s | 1.526s | 1197ms |
| 0.675s | 1.526s | 851ms |
| 1.050s | 1.526s | 476ms |
| 4.841s | 5.232s | 391ms |
| 6.009s | 6.258s | 249ms |
| 6.610s | 6.258s | 352ms |
| 7.682s | 8.028s | 346ms |
| 9.651s | 10.147s | 496ms |
| 12.146s | 12.457s | 311ms |
| 12.817s | 12.457s | 360ms |

## Recommendations

- **Use beat-flourish as ground truth** — it captures user's current perception of flourishes
- **Investigate new taps** — understand what audio features characterize these moments
- **Investigate missed flourishes** — understand why user didn't tap them

---

*Analysis threshold: ±150ms*
*Density window: 2.0s*
