# Madmom Beat Tracking Results

## Installation Status

✅ **SUCCESS** - madmom 0.17.dev0 installed successfully

Installation method: `pip install git+https://github.com/CPJKU/madmom.git`

The initial `pip install madmom` (v0.16.1 from PyPI) failed on Python 3.10 with `ImportError: cannot import name 'MutableSequence' from 'collections'`. This is a known compatibility issue with Python 3.10+ where `collections.MutableSequence` was moved to `collections.abc.MutableSequence`. The development version from GitHub has this fix applied.

## Audio File

**Tool - Opiate (Intro)**, ~40 seconds, progressive rock

## User Tap Annotations

| Layer | Taps | Estimated Tempo | Description |
|-------|------|-----------------|-------------|
| beat | 72 | 111.3 BPM | All beat taps (mixed tempos) |
| consistent-beat | 41 | 81.9 BPM | Steady groove pulse (mostly 11-40s) |

## Madmom Results

### 1. RNNBeatProcessor + DBNBeatTrackingProcessor

- **Detected beats**: 106
- **Estimated tempo**: 162.2 BPM
- First 10 beats: [0.07, 0.52, 0.97, 1.4, 1.83, 2.27, 2.68, 3.11, 3.56, 3.97]

### 2. TempoEstimationProcessor

- Primary tempo: 83.3 BPM (strength: 0.224)
- Secondary tempo: 166.7 BPM (strength: 0.222)

Top 5 tempo candidates:

| Rank | Tempo (BPM) | Strength |
|------|-------------|----------|
| 1 | 83.3 | 0.224 |
| 2 | 166.7 | 0.222 |
| 3 | 41.4 | 0.108 |
| 4 | 55.0 | 0.093 |
| 5 | 65.9 | 0.089 |

### 3. Downbeat Detection

✅ Successfully detected 109 beat positions

- **Downbeats (position 1)**: 27
- **Other beats**: 82
- **Assumed meter**: 4/4 time signature

This indicates madmom can track not just beats, but which beat in the bar (1, 2, 3, 4), useful for identifying downbeats and musical structure.

## Comparison with User Annotations

F1 score calculated with ±100ms tolerance window.

| Comparison | Precision | Recall | F1 | TP | FP | FN | Notes |
|------------|-----------|--------|----|----|----|----|-------|
| vs. beat layer (all taps) | 0.406 | 0.597 | 0.483 | 43 | 63 | 29 | Mixed tempos |
| vs. consistent-beat | 0.245 | 0.634 | 0.354 | 26 | 80 | 15 | 82 BPM pulse |
| vs. steady groove (11-40s) | 0.312 | 0.641 | 0.420 | 25 | 55 | 14 | Best section |

## Tempo Analysis

| Method | Tempo (BPM) | Notes |
|--------|-------------|-------|
| User taps (beat layer) | 111.3 | Mixed, includes tempo changes |
| User taps (consistent-beat) | 81.9 | Steady groove pulse |
| User taps (groove 11-40s) | 81.9 | Most stable section |
| Madmom (overall) | 162.2 | Full track |
| Madmom (groove 11-40s) | 166.7 | Groove section |

## Interpretation

✅ **TempoEstimationProcessor is accurate!** The tempo estimation correctly identified 83.3 BPM as the primary tempo, matching the user's 81.9 BPM taps. However, the DBNBeatTrackingProcessor locked onto the doubled tempo (subdivisions) instead.

⚠️ **Tempo doubled** - madmom detected ~162 BPM, which is 2x the user's ~82 BPM pulse. Locked onto subdivisions (hi-hats) instead of main pulse (kick/snare).

⚠️ **Moderate F1 score** - madmom catches some beats but misses others or detects extra beats.

## Key Findings

### Tempo Estimation vs. Beat Tracking Discrepancy

There's an important split in madmom's results:

1. **TempoEstimationProcessor** (histogram-based) correctly identified 83.3 BPM
2. **DBNBeatTrackingProcessor** (tracking-based) locked onto 166.7 BPM (doubled)

This suggests:
- The tempo **estimation** algorithm is working correctly
- The beat **tracking** algorithm prefers the subdivision level
- DBN parameters may need tuning (tempo constraints, transition matrix)
- Could initialize DBN with the TempoEstimationProcessor result to bias toward correct tempo

## Recommendation

✅ **RECOMMEND FOR TEMPO ESTIMATION ONLY**

While beat tracking locked onto subdivisions, the TempoEstimationProcessor accurately identified the correct tempo. Consider:

**Option 1: Use madmom for tempo, custom for beats**
- Use `TempoEstimationProcessor` to get accurate BPM
- Use librosa or custom onset detection for beat times
- Constrains the search space effectively

**Option 2: Tune DBN parameters**
```python
# Initialize DBN with tempo estimate
dbn = DBNBeatTrackingProcessor(
    min_bpm=60, max_bpm=120,  # Constrain to half-time range
    transition_lambda=100,    # Reduce tempo fluctuation
    fps=100
)
```

**Option 3: Post-process to select correct metric level**
- Detect beats at multiple metric levels
- Choose level closest to TempoEstimationProcessor result

## Context: Previous librosa Results

For comparison, librosa's `beat_track()` on this same audio:
- Detected tempo: 161.5 BPM
- Actual tempo: ~82-109 BPM (depending on which pulse you track)
- Problem: Tempo doubling, locked onto hi-hat subdivisions

The user's tap annotations provide ground truth showing the perceptual pulse around 82 BPM (consistent-beat layer) with variations during different sections (beat layer shows tempo changes and flourishes).
