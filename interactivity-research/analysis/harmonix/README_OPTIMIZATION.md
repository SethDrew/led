# Beat Detector Hyperparameter Optimization — Quick Reference

**Created:** 2026-02-06
**Scope:** Harmonix dataset validation of bass-band spectral flux beat detector
**Result:** Algorithm failed (F1 = 0.06-0.34), needs replacement with full-spectrum onset detection

---

## Quick Start

### View Results

1. **Executive summary:** `OPTIMIZATION_SUMMARY.md`
2. **Optimal params:** `harmonix_optimal_params.yaml`
3. **Full report:** `harmonix_optimization.md`
4. **Visualization:** `harmonix_optimization.png`

### Re-Run Optimization

```bash
source venv/bin/activate
cd analysis/scripts
python harmonix_optimize.py --quick  # Fast test (18 combos/track)
python harmonix_optimize.py          # Full grid (420 combos/track)
```

---

## Files Created

### Analysis Scripts
- `analysis/scripts/harmonix_optimize.py` — Validation pipeline (grid search + evaluation)
  - Loads audio + JAMS annotations
  - Runs BeatDetector chunk-by-chunk
  - Evaluates with mir_eval (F1, precision, recall, tempo)
  - Generates reports and visualizations

### Data
- `datasets/harmonix/audio/` — 7 downloaded tracks (517 MB)
  - `0012_aroundtheworld.wav` — Daft Punk (electronic, 121 BPM)
  - `qrO4YZeyl0I.wav` — Lady Gaga (electronic, 119 BPM)
  - `DU_ggFovJNo.wav` — Metallica (metal, 190 BPM)
  - `jbwBsAGCMLE.wav` — Dream Theater (metal, 180 BPM)
  - `td-v6vG2Xhs.wav` — Pantera (metal, 120 BPM)
  - `fjrHJhMHyIM.wav` — Rush "Camera Eye" (rock, 166 BPM)
  - `ZiRuj2_czzw.wav` — Rush "Limelight" (rock, 130 BPM)

- `datasets/harmonix/audio/track_mapping.yaml` — YouTube ID → Harmonix ID mapping

### Results
- `analysis/harmonix_optimal_params.yaml` — Best hyperparameters per genre
- `analysis/harmonix_optimization.md` — Full report (Markdown)
- `analysis/harmonix_optimization.png` — 4-panel visualization
- `analysis/harmonix_optimization_results.json` — Raw data (126 test runs)
- `analysis/OPTIMIZATION_SUMMARY.md` — Executive summary
- `analysis/README_OPTIMIZATION.md` — This file

---

## Key Findings

### ❌ Algorithm Failed

**Bass-band spectral flux does not work for beat detection.**

| Genre      | F1 Score | Status     | Why?                                |
|------------|----------|------------|-------------------------------------|
| Electronic | 0.06     | **FAILED** | No bass transients (continuous hum) |
| Rock       | 0.27     | **POOR**   | Missing snare/hi-hat beats          |
| Metal      | 0.26     | **POOR**   | Tempo detection wrong               |
| Universal  | 0.20     | **FAILED** | Fundamentally broken                |

**Expected:** F1 > 0.85 (mir_eval standard for "good" beat tracking)
**Achieved:** F1 = 0.06-0.34 (catastrophic failure)

### Why Electronic Music Failed Hardest

Electronic music has **no bass transients**:
- Kick drums are synthesized with slow attack/release
- Bass is continuous sub-bass hum (80-120 Hz)
- All sharp transients (hi-hat, snare) are OUTSIDE bass band (>250 Hz)
- **Result:** Algorithm sees nothing in 20-250 Hz → F1 = 0.01-0.12

Rock/metal have physical drums with sharper attack → slightly better (F1 = 0.23-0.34) but still unusable.

### Optimal Hyperparameters (For Reference)

Even though the algorithm failed, optimal params are:

```yaml
threshold_multiplier: 1.5  # Lower = more sensitive (1.5 is best)
bass_high_hz: 200-300      # Rock prefers 300 Hz, others 200 Hz
min_beat_interval_sec: 0.3 # Allows up to 200 BPM
flux_history_sec: 3.0      # Sufficient for adaptive threshold
```

**Key insight:** Parameters don't matter much because the algorithm is fundamentally broken. Even optimal settings → F1 = 0.20.

---

## Next Steps

### Recommended: Full-Spectrum Onset Detection

Replace bass-band flux with full-spectrum onset strength:

```python
# CURRENT (broken)
bass_spectrum = spectrum[bass_bins]  # 20-250 Hz only
flux = np.sum(np.maximum(bass_spectrum - prev_bass_spectrum, 0))

# NEW (better)
onset_env = librosa.onset.onset_strength(y=audio_chunk, sr=sr)
peaks = librosa.util.peak_pick(
    onset_env,
    pre_max=3, post_max=3,
    pre_avg=3, post_avg=5,
    delta=0.5, wait=10
)
```

**Why this should work:**
- Detects ALL transients (kick, snare, hi-hat, guitar hits)
- Works on electronic music (hi-hat transients are sharp)
- Real-time compatible (chunk-by-chunk processing)
- librosa built-in (no new dependencies)

**Expected F1:** 0.70-0.85 (based on librosa benchmarks)

### Alternative: madmom RNN

For offline validation (not real-time):

```python
from madmom.features.beats import RNNBeatProcessor
proc = RNNBeatProcessor()
beats = proc(audio_file)
```

**Expected F1:** 0.90-0.95 (state-of-the-art)

Use madmom as "ground truth" to compare our real-time approach.

---

## Validation Pipeline Architecture

The `harmonix_optimize.py` script is **reusable** for testing any beat detector:

1. **Input:** Audio file + JAMS annotation
2. **Process:** Run beat detector chunk-by-chunk (simulate real-time)
3. **Evaluate:** Compare detected beats to ground truth (mir_eval)
4. **Output:** F1, precision, recall, tempo accuracy

**To test a new algorithm:**
1. Replace `BeatDetector` class in `harmonix_optimize.py`
2. Re-run: `python harmonix_optimize.py --quick`
3. Compare F1 scores to current baseline (0.20)
4. If F1 > 0.85 → deploy to LED controller

---

## Dataset Details

### Harmonix Set
- **Location:** `/datasets/harmonix/`
- **Size:** 912 tracks, 56+ hours annotated
- **Annotations:** Beat times, downbeats, segments (intro/verse/chorus)
- **Format:** JAMS (JSON for music annotations)

### Priority Tracks (7 downloaded)

**Electronic (baseline — should be easiest):**
- Daft Punk "Around The World" (121 BPM, 4/4) — F1 = **0.01** ❌
- Lady Gaga "Bad Romance" (119 BPM, 4/4) — F1 = **0.12** ❌

**Rock (prog complexity):**
- Rush "Limelight" (130 BPM, 7/4 + 4/4) — F1 = **0.26** ⚠️
- Rush "The Camera Eye" (166 BPM, 4/4) — F1 = **0.29** ⚠️

**Metal (hardest):**
- Metallica "Blackened" (190 BPM, 4/4) — F1 = **0.34** ⚠️
- Dream Theater "Constant Motion" (180 BPM, 4/4) — F1 = **0.24** ⚠️
- Pantera "Floods" (120 BPM, 4/4) — F1 = **0.23** ⚠️

**Genre performance:** Metal > Rock > Electronic (counterintuitive!)

---

## Dependencies Added

```txt
tqdm>=4.67.0      # Progress bars (for optimization script)
mir_eval>=0.7     # Beat tracking evaluation metrics (already installed)
jams>=0.3.4       # JAMS annotation format (already installed)
yt-dlp>=2024.0.0  # YouTube audio download (already installed)
```

---

## Reproduction Instructions

### Download More Tracks

```bash
cd datasets/harmonix/audio
source ../../../venv/bin/activate

# Download from YouTube using JAMS URLs
yt-dlp -x --audio-format wav --audio-quality 0 \
  -o "%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Run Full Optimization (420 combos/track)

```bash
cd analysis/scripts
python harmonix_optimize.py  # ~30 min for 7 tracks
```

Parameter grid:
- THRESHOLD_MULTIPLIER: [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] (7 values)
- BASS_HIGH_HZ: [150, 200, 250, 300, 400] (5 values)
- MIN_BEAT_INTERVAL_SEC: [0.2, 0.3, 0.4, 0.5] (4 values)
- FLUX_HISTORY_SEC: [2.0, 3.0, 5.0] (3 values)

Total: 7 × 5 × 4 × 3 = **420 combinations per track**

---

## Comparison to Our Previous Test

### Tool "Opiate Intro" (user taps)
- **Algorithm:** Bass-band spectral flux
- **Ground truth:** User tap annotations (`consistent-beat` layer)
- **Reference BPM:** 83 BPM (from user taps)
- **Result:** Not formally evaluated, but subjectively "looked okay"

### Harmonix Validation (professional annotations)
- **Same algorithm** on professional annotations
- **Result:** F1 = 0.06-0.34 (catastrophic failure)

**Lesson:** User taps are forgiving; professional validation is brutal. Our algorithm doesn't work.

---

## Success Metrics for Next Algorithm

To deploy to LED controller, we need:

- ✅ **F1 > 0.85** on electronic music (metronomic baseline)
- ✅ **F1 > 0.80** on rock music (prog complexity)
- ✅ **F1 > 0.75** on metal music (fast/heavy)
- ✅ **Tempo accuracy:** Detected BPM within ±5 of reference
- ✅ **Real-time compatible:** <50ms latency per chunk

Current algorithm: **0/5** criteria met

---

## Acknowledgments

- **Harmonix Set:** Nieto et al., ISMIR 2019 (912 professionally annotated tracks)
- **mir_eval:** Raffel et al. (standard MIR evaluation library)
- **librosa:** McFee et al. (audio analysis library)
- **yt-dlp:** YouTube audio extraction (research/development only)

---

## Status

**Date:** 2026-02-06
**Current algorithm:** ❌ Failed validation (F1 = 0.20)
**Next task:** Implement full-spectrum onset detection
**Validation pipeline:** ✅ Ready to test new algorithms

When new algorithm is ready:
1. Drop it into `harmonix_optimize.py` (replace `BeatDetector` class)
2. Run `python harmonix_optimize.py --quick`
3. Check if F1 > 0.85
4. If yes → deploy to LED controller
5. If no → iterate on algorithm

**Infrastructure is ready. Algorithm needs work.**
