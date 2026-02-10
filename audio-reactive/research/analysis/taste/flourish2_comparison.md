# Beat-Flourish2 Analysis

Comprehensive comparison of the beat-flourish2 layer against existing annotations and computed flourishes.

## 1. Basic Statistics

### Inter-tap Intervals

| Layer | Count | Median (s) | Mean (s) | Std | CV | BPM (median) |
|-------|-------|------------|----------|-----|-----|--------------|
| **beat-flourish2** | 65 | 0.514 | 0.504 | 0.219 | 0.434 | 116.7 |
| beat-flourish | 76 | 0.452 | 0.518 | 0.345 | 0.666 | 132.7 |
| consistent-beat | 41 | 0.732 | 0.970 | 1.387 | 1.431 | 81.9 |
| computed flourishes | 39 | 0.606 | 1.044 | 1.183 | 1.133 | 98.9 |

**Interpretation:**
- beat-flourish2 is MORE regular than beat-flourish (CV 0.434 vs 0.666)
- Low CV (0.434) suggests user fell into a groove (not ideal for flourishes)

## 2. Grid Alignment

### Overlap with Existing Layers (±150ms threshold)

| Test Layer | vs. Layer | Overlap % |
|------------|-----------|-----------|
| **beat-flourish2** | consistent-beat | **60.0%** |
| beat-flourish | consistent-beat | 38.2% |
| **beat-flourish2** | computed flourishes | **16.9%** |

**Interpretation:**
- ❌ beat-flourish2 is NOT cleaner — 60.0% overlap with consistent-beat vs. 38.2% for beat-flourish
- Low overlap with computed flourishes (16.9%) — captures different events

### New Information

- **16 taps (24.6%)** are genuinely NEW (not in consistent-beat or computed flourishes)

New tap timestamps: 8.73s, 9.49s, 10.36s, 11.85s, 21.25s, 21.98s, 22.67s, 23.52s, 24.02s, 25.06s...

## 3. Feature Alignment

### What do the taps track?

| Feature | beat-flourish2 | computed flourishes |
|---------|----------------|---------------------|
| Onset | 30 (46%) | 10 (26%) |
| Bass | 2 (3%) | 8 (21%) |
| Centroid | 18 (28%) | 5 (13%) |
| Rms | 9 (14%) | 10 (26%) |
| Flux | 6 (9%) | 6 (15%) |

**Interpretation:**
- beat-flourish2 primarily tracks **onset** events
- computed flourishes primarily track **onset** events
- Same dominant feature suggests similar musical focus

## 4. Regularity Check

- Median interval: 0.514s (116.7 BPM)
- 18/64 intervals (28.1%) within ±20% of median

**Verdict: IRREGULAR** — truly varied timing (good for flourishes)

Interval range: 0.144s - 1.063s (ratio: 7.4x)

## 5. Section Density

### Tap Density by Section (taps per second)

| Section | beat-flourish2 | computed flourishes | beat-flourish |
|---------|----------------|---------------------|---------------|
| Section 1 (8.1s - 8.1s) | 125.00 | 0.00 | 0.00 |
| Section 2 (8.1s - 17.1s) | 1.22 | 1.11 | 1.33 |
| Section 3 (17.1s - 24.3s) | 2.51 | 0.84 | 2.37 |
| Section 4 (24.3s - 28.7s) | 2.49 | 0.23 | 2.49 |
| Section 5 (28.7s - 34.5s) | 2.07 | 1.03 | 2.41 |
| Section 6 (34.5s - 37.4s) | 2.09 | 1.04 | 2.43 |
| Section 7 (37.4s - 40.3s) | 1.68 | 1.01 | 2.69 |

**Density Pattern Correlation:**
- beat-flourish2 vs. computed flourishes: r = -0.743
- beat-flourish2 vs. beat-flourish: r = -0.888

- Negative correlation with computed flourishes → opposite density pattern

## 6. Verdict

### Checklist

- [ ] Cleaner than beat-flourish? ❌ NO (60.0% vs 38.2% consistent-beat overlap)
- [ ] Irregular timing? ❌ NO (CV = 0.434)
- [ ] Adds new information? ✅ YES (24.6% new taps)
- [ ] Not redundant with computed flourishes? ✅ YES (16.9% overlap)

### Recommendation

**⚠️ MIXED** (2/4 criteria met)

beat-flourish2 has some merit but also limitations.

**Suggested use:** Combine with computed flourishes — use beat-flourish2 for sections where user tapped, computed for full coverage.
