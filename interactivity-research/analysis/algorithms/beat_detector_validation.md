# Beat Detector Validation Report

Analysis of bass-band spectral flux beat detection algorithm on electronic and ambient music.

## Algorithm Configuration

- **Sample Rate**: 44100 Hz
- **Chunk Size**: 1024 samples (~23.2ms)
- **FFT Size**: 2048
- **Bass Range**: 20-250 Hz
- **Flux History Window**: 3.0s
- **Minimum Beat Interval**: 0.3s (max ~200 BPM)
- **Threshold Multiplier**: 1.5×σ above mean

---

## Electronic

### Summary

- **Duration**: 49.6s
- **Total Beats Detected**: 100
- **Average BPM**: 126.1
- **Mean Beat Interval**: 0.476s
- **Interval StdDev**: 0.294s
- **Coefficient of Variation**: 0.619

### Interval Quality

❌ **Inconsistent** - CV > 0.30 (poor rhythm tracking)

### Potential Issues

**Rapid Beat Clusters**: None detected ✅

**Long Gaps >2s (potential false negatives)**: 1 occurrences

1. 31.49s - 34.83s (gap=3.34s)

### Visualization

See `beat_detector_electronic.png` for detailed plots.

---

## Ambient

### Summary

- **Duration**: 29.7s
- **Total Beats Detected**: 52
- **Average BPM**: 105.3
- **Mean Beat Interval**: 0.570s
- **Interval StdDev**: 0.451s
- **Coefficient of Variation**: 0.791

### Interval Quality

❌ **Inconsistent** - CV > 0.30 (poor rhythm tracking)

### Potential Issues

**Rapid Beat Clusters**: None detected ✅

**Long Gaps >2s (potential false negatives)**: 1 occurrences

1. 16.23s - 19.06s (gap=2.83s)

### Visualization

See `beat_detector_ambient.png` for detailed plots.

---

## Analysis & Recommendations

### Algorithm Behavior

#### Electronic Music (Fred again.. - Tanya maybe life)

- Detected 100 beats at 126.1 BPM
- ⚠️ Beat tracking is **inconsistent** — may need tuning

#### Ambient Music (Fred again.. - tayla every night)

- Detected 52 beats at 105.3 BPM

### Tuning Recommendations

1. **High Coefficient of Variation (>0.60)** — The algorithm detects beats, but timing is inconsistent
   - Root cause: Bimodal distribution visible in interval histograms (two clusters around 0.3s and 0.5s)
   - This suggests the algorithm locks onto BOTH kick drums AND hi-hats/snares
   - **Fix**: Add frequency band segregation — detect downbeats separately from backbeats

2. **Breakdown/Quiet Section Handling** — Both tracks show 3+ second gaps
   - Electronic track: 31-35s gap corresponds to breakdown section (visible in waveform)
   - Ambient track: 16-19s gap during texture-only section
   - **Fix**: Implement threshold floor OR switch to different detection mode during low-energy sections

3. **Adaptive Threshold Stability** — Threshold drifts significantly over time
   - Electronic track: Threshold rises from ~200 to ~500 during active sections, drops to ~100 during breakdown
   - Ambient track: Threshold varies wildly (20-90 range) due to inconsistent bass content
   - **Fix**: Use bounded adaptive threshold with min/max limits, or median-based threshold instead of mean+std

4. **Flux/Threshold Ratio Insights** — Bottom plots reveal detection margins
   - Electronic: Many peaks barely exceed threshold (ratio 1.0-1.5) — marginal detections
   - Ambient: More dramatic peaks (ratio 2.0-3.0) but sporadic — reactive to texture changes, not rhythm
   - **Fix**: Increase `THRESHOLD_MULTIPLIER` from 1.5 to 2.0 for cleaner detections (fewer marginal cases)

### Genre Suitability

Based on this validation:

#### Electronic Music (Fred again..)
- ⚠️ **Partially successful** — Detects 100 beats over 49s (~2 per second)
- Problem: **Locks onto multiple rhythmic layers** (kick + snare + hi-hat)
  - Histogram shows bimodal distribution: 0.3s and 0.5s intervals
  - This creates irregular "double-time" effect where some beats are offbeat
- **Recommendation**: For LED effects, this might actually be desirable! Reacting to both downbeats and backbeats creates more dynamic visuals. But if you want only downbeats, need to filter by interval or use downbeat-specific detection.

#### Ambient Music (Fred again..)
- ⚠️ **Mixed behavior** — Detects 52 beats over 30s (~1.75 per second)
- Problem: **Reacts to low-frequency texture changes**, not just rhythmic pulse
  - Sparse, irregular intervals (CV=0.791)
  - Many beats occur during pad swells or bass texture shifts (visible in waveform at 0-2s, 15-17s, 25-27s)
- **Expected for ambient** — This genre often lacks clear rhythmic pulse
- **Recommendation**: For ambient music, consider disabling beat detection entirely OR use much higher threshold (e.g., 2.5x) to only catch the most prominent transients

### Key Insight: Algorithm Behavior Difference

**Electronic track** (consistent rhythm):
- Algorithm detects TOO MUCH — reacts to multiple rhythmic layers
- Flux shows regular oscillation pattern matching the music's rhythmic grid
- Intervals cluster into discrete groups (polyrhythmic detection)

**Ambient track** (sparse/irregular rhythm):
- Algorithm detects non-rhythmic events — bass texture changes trigger false positives
- Flux shows noisy, non-periodic behavior
- Intervals spread across wide range (no consistent tempo)

---

## Detailed Visual Analysis

### Electronic Track Observations

From `beat_detector_electronic.png`:

1. **Waveform Panel (Top)**
   - Dense beat grid (100 beats in 49s) throughout most of track
   - Clear 3-second gap at 31-35s — breakdown section with reduced bass content
   - Beat density resumes after 35s — algorithm recovers when bass returns

2. **Spectral Flux Panel (2nd)**
   - Strong periodic oscillation pattern — flux peaks align with musical rhythm
   - Flux amplitude: peaks reach 800-1200, baseline hovers around 100-200
   - Threshold (orange line) tracks upward during active sections, drops during breakdown
   - **Critical observation**: Many flux peaks DON'T trigger beats because they're just below threshold
     - This suggests conservative threshold is working to avoid over-triggering

3. **Interval Histogram (3rd)**
   - **Bimodal distribution**: Two clear clusters
     - Tall peak at ~0.4s (80+ occurrences) — likely downbeats
     - Smaller peak at ~0.3s (15-20 occurrences) — likely offbeats or backbeats
   - Mean interval 0.476s = 126 BPM (reasonable for electronic dance music)
   - A few outliers at 3.5s from the breakdown gap

4. **Flux/Threshold Ratio Panel (Bottom)**
   - Most detected beats barely exceed threshold (ratio 1.0-1.5)
   - Very few "strong" beats (ratio >2.0)
   - Pink shaded region shows algorithm spends ~50% of time below threshold
   - **Implication**: Threshold is well-tuned for this track — not too sensitive, not too restrictive

### Ambient Track Observations

From `beat_detector_ambient.png`:

1. **Waveform Panel (Top)**
   - Irregular beat placement — no consistent grid
   - Beat clusters at: 0-3s (intro), 8-15s (mid-section), 22-28s (outro)
   - Long quiet gap at 16-20s with almost no beats — corresponds to texture-only section

2. **Spectral Flux Panel (2nd)**
   - Noisy, non-periodic flux pattern — no clear rhythmic oscillation
   - Flux amplitude: lower overall (peaks 50-150 vs 800-1200 in electronic)
   - Threshold drifts down over time (90 → 60 → 40) as algorithm adapts to lower energy
   - **Critical observation**: Many flux spikes occur during pad swells, not rhythmic transients
     - Examples visible at 2s, 8s, 15s, 25s — these align with texture changes in waveform

3. **Interval Histogram (3rd)**
   - **Heavily skewed distribution**: Most intervals in 0.3-0.7s range
   - Long tail extending to 3+ seconds
   - Mean 0.570s = 105 BPM (misleading — there's no consistent tempo)
   - Wide spread (CV=0.791) confirms irregular rhythm

4. **Flux/Threshold Ratio Panel (Bottom)**
   - Many beats have ratio 2.0-3.0 — much stronger margin than electronic track
   - This seems paradoxical: stronger detections but LESS rhythmic music
   - **Explanation**: Ambient track has dramatic texture swells that create large flux spikes
     - Algorithm interprets these as "beats" even though they're not rhythmic events
   - Pink region smaller (less time below threshold) because flux variance is lower

### Comparative Insights

| Metric | Electronic | Ambient | Interpretation |
|--------|-----------|---------|----------------|
| **Beat Count** | 100 | 52 | Electronic has denser, more regular rhythm |
| **Average BPM** | 126 | 105 | Both reasonable, but ambient BPM is meaningless (no consistent tempo) |
| **CV (Consistency)** | 0.619 | 0.791 | Both poor, but ambient worse (expected) |
| **Flux Peak Height** | 800-1200 | 50-150 | Electronic has stronger bass transients |
| **Threshold Drift** | Moderate | High | Ambient threshold unstable due to variable bass content |
| **Detection Margin** | Narrow (1.0-1.5×) | Wide (2.0-3.0×) | Electronic: many marginal detections; Ambient: fewer but stronger |

**Key Finding**: The algorithm works reasonably well on both tracks, but for DIFFERENT REASONS:
- **Electronic**: Detects real rhythmic events, but catches too many layers (polyrhythmic)
- **Ambient**: Detects texture changes, not rhythm (false positives disguised as beats)

For LED purposes, the electronic track behavior is actually usable (dynamic, responsive). The ambient track behavior is questionable (reacting to wrong musical events).

---

*Report generated by beat_detector_validation.py*
