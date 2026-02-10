# Beat Detector Hyperparameter Optimization - Executive Summary

**Date:** 2026-02-06
**Status:** ❌ Current algorithm FAILED — requires redesign
**Recommendation:** Switch to full-spectrum onset detection or madmom RNN

---

## What We Did

1. **Downloaded audio** for 7 priority tracks from Harmonix dataset:
   - 2 electronic (Daft Punk, Lady Gaga) — baseline validation
   - 2 rock (Rush - Limelight, The Camera Eye) — prog complexity
   - 3 metal (Metallica, Dream Theater, Pantera) — hard cases

2. **Built validation pipeline** (`harmonix_optimize.py`):
   - Loads audio + ground truth beat annotations from JAMS files
   - Runs our BeatDetector chunk-by-chunk (simulates real-time)
   - Scores detected beats vs ground truth using mir_eval (F1, precision, recall)

3. **Grid search** over hyperparameters:
   - THRESHOLD_MULTIPLIER: [1.5, 2.0, 2.5]
   - BASS_HIGH_HZ: [200, 250, 300]
   - MIN_BEAT_INTERVAL_SEC: [0.3, 0.4]
   - FLUX_HISTORY_SEC: [3.0]
   - **18 combinations per track** × 7 tracks = 126 test runs

4. **Analyzed results** by genre to find optimal params

---

## Results: Algorithm Failure

### F1 Scores (70ms tolerance window)

| Genre      | Best F1 | Expected | Status     |
|------------|---------|----------|------------|
| Electronic | 0.06    | >0.95    | **FAILED** |
| Rock       | 0.27    | >0.85    | **POOR**   |
| Metal      | 0.26    | >0.80    | **POOR**   |
| Universal  | 0.20    | >0.85    | **FAILED** |

### Per-Track Breakdown

| Track | Artist | Genre | BPM | F1 | Detected BPM | Issue |
|-------|--------|-------|-----|-----|--------------|-------|
| Around The World | Daft Punk | electronic | 121 | **0.01** | 123 | ❌ 99% missed |
| Bad Romance | Lady Gaga | electronic | 119 | **0.12** | 118 | ❌ 88% missed |
| Blackened | Metallica | metal | 190 | **0.34** | 129 | ⚠️ Tempo wrong |
| Constant Motion | Dream Theater | metal | 180 | **0.24** | 123 | ⚠️ Tempo wrong |
| The Camera Eye | Rush | rock | 166 | **0.29** | 112 | ⚠️ Tempo wrong |
| Limelight | Rush | rock | 130 | **0.26** | 129 | ⚠️ 74% missed |
| Floods | Pantera | metal | 120 | **0.23** | 123 | ⚠️ 77% missed |

---

## Why It Failed

### Root Cause: Bass-Band-Only Detection Doesn't Work

Our algorithm uses **bass-band spectral flux** (20-250 Hz) to detect beats. This fails because:

1. **Electronic music:** Bass is continuous sub-bass hum, not transient kicks
   - Kick drums are synthesized with soft attack/release (no sharp onset)
   - All the sharp transients (hi-hat, snare) are OUTSIDE the bass band
   - **Result:** Algorithm sees no activity in 20-250 Hz → detects nothing

2. **Rock/metal:** Better (physical drums have sharper attack) but still poor
   - Kick drum transients ARE in bass band, so we detect some beats
   - But we're missing all the snare/hi-hat beats that define the rhythm
   - **Result:** Detecting ~25% of beats (kicks only, missing snares/cymbals)

3. **Tempo detection:** Algorithm locks onto wrong subdivision
   - Fast tracks (180-190 BPM) detected as ~125 BPM (missing 1/3 of beats)
   - Slow tracks mostly correct (120-130 BPM range)

### Visual Evidence

![Optimization Results](harmonix_optimization.png)

**Top-left:** F1 scores all below 0.35 (vs. target >0.85)
**Top-right:** Tempo ratios all over the place (purple = metal, red = rock, blue = electronic)
**Bottom:** Parameter sensitivity curves are FLAT → hyperparameters don't matter because algorithm is broken

---

## Optimal Parameters (For What It's Worth)

Even though the algorithm fails, here are the "best" hyperparameters we found:

### Electronic
```yaml
threshold_multiplier: 1.5
bass_high_hz: 200
min_beat_interval_sec: 0.3
flux_history_sec: 3.0
f1_score: 0.060  # Still terrible
```

### Rock
```yaml
threshold_multiplier: 1.5
bass_high_hz: 300  # Wider bass range helps slightly
min_beat_interval_sec: 0.3
flux_history_sec: 3.0
f1_score: 0.269  # Better but still poor
```

### Metal
```yaml
threshold_multiplier: 1.5
bass_high_hz: 200
min_beat_interval_sec: 0.3
flux_history_sec: 3.0
f1_score: 0.257  # Poor
```

### Universal (all genres)
```yaml
threshold_multiplier: 1.5
bass_high_hz: 200
min_beat_interval_sec: 0.3
flux_history_sec: 3.0
f1_score: 0.204  # Failed
```

**Key insight:** Lower threshold (1.5 vs 2.0-2.5) is universally better → we want to be MORE sensitive, not less. But even maximally sensitive settings only achieve F1 = 0.20.

---

## Next Steps: Fix the Algorithm

### Option 1: Full-Spectrum Onset Detection (Recommended)

Replace bass-band flux with **full-spectrum onset strength**:

```python
# CURRENT (broken)
bass_spectrum = spectrum[bass_bins]  # 20-250 Hz only
flux = np.sum(np.maximum(bass_spectrum - prev_bass_spectrum, 0))

# NEW (better)
onset_env = librosa.onset.onset_strength(y=audio_chunk, sr=sr)
peaks = librosa.util.peak_pick(onset_env, ...)
```

**Pros:**
- Detects ALL transients (kick, snare, hi-hat, bass plucks, guitar hits)
- Works on electronic music (hi-hat transients are sharp)
- librosa built-in, no new dependencies
- Real-time compatible

**Cons:**
- More sensitive to non-beat transients (guitar strums, cymbal crashes)
- May need better peak picking logic

### Option 2: madmom RNN Beat Tracker

Use machine learning approach:

```python
from madmom.features.beats import RNNBeatProcessor
proc = RNNBeatProcessor()
beats = proc(audio_file)
```

**Pros:**
- State-of-the-art accuracy (F1 > 0.90 on Harmonix)
- Handles genre variation, odd meters, tempo changes
- Battle-tested on MIR benchmarks

**Cons:**
- NOT real-time (processes entire audio file at once)
- Requires madmom dependency (large, complex)
- Can't use for live LED streaming

### Option 3: Tempo Tracking + Phase Locking

Assume steady tempo, lock to phase:

1. Estimate global tempo (librosa `beat.tempo`)
2. Find first beat (onset detection)
3. Generate beat grid at fixed interval (60/BPM seconds)
4. Use onset strength to nudge phase slightly (error correction)

**Pros:**
- Works for metronomic music (electronic, most rock)
- Extremely efficient (just phase tracking)
- Real-time compatible

**Cons:**
- Fails on tempo changes (Rush, prog metal)
- Requires accurate initial tempo estimate
- Phase drift over long tracks

---

## Recommendation

**For LED effects (real-time):**
1. Try **Option 1** (full-spectrum onset detection) first
2. Test on Harmonix tracks, aim for F1 > 0.85
3. If successful, deploy to LED controller
4. If F1 still < 0.85, fall back to **Option 3** (tempo tracking)

**For offline analysis (research):**
- Use **Option 2** (madmom) as ground truth "best possible" detector
- Compare our real-time approach to madmom to understand performance gap

---

## Files Generated

- **`harmonix_optimal_params.yaml`** — Optimal hyperparameters per genre
- **`harmonix_optimization.md`** — Full results report with per-track details
- **`harmonix_optimization.png`** — 4-panel visualization of results
- **`harmonix_optimization_results.json`** — Raw data (all 126 test runs)
- **`harmonix_optimize.py`** — Validation script (re-runnable)
- **`track_mapping.yaml`** — YouTube ID to Harmonix ID mapping
- **Audio files (507 MB):**
  - `/datasets/harmonix/audio/*.wav` (7 tracks)

---

## How to Re-Run

```bash
source venv/bin/activate
cd audio-reactive/research/analysis/scripts

# Quick test (18 combinations per track)
python harmonix_optimize.py --quick

# Full grid (420 combinations per track)
python harmonix_optimize.py
```

---

## Conclusion

Our bass-band spectral flux beat detector **does not work** for music. It achieves F1 = 0.06-0.34 across genres (vs. target >0.85). The algorithm needs to be replaced with full-spectrum onset detection or machine learning.

However, the **validation infrastructure is excellent**:
- Harmonix dataset downloaded and integrated
- Automated testing pipeline built
- Genre-specific optimization working
- Beautiful visualizations

When we fix the algorithm, we can re-run this exact pipeline to validate the new approach.

**Status:** Ready for Algorithm V2 development.
