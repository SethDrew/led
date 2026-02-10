# Madmom Installation & Test Summary

## Quick Summary

✅ **madmom successfully installed and tested**

- **Installation**: Requires development version from GitHub (v0.17.dev0) for Python 3.10 compatibility
- **Tempo Estimation**: ✅ Accurate (83.3 BPM detected vs. 81.9 BPM user taps)
- **Beat Tracking**: ⚠️ Tempo doubling issue (locked onto 166.7 BPM subdivisions)
- **Recommendation**: Use for tempo estimation, not beat tracking (without parameter tuning)

---

## Installation Instructions

### For Python 3.10+

The PyPI release (v0.16.1) has a Python 3.10 compatibility issue. Install from GitHub instead:

```bash
# Activate your venv first
source venv/bin/activate  # or: venv/bin/activate.fish, etc.

# Install development version
pip install git+https://github.com/CPJKU/madmom.git
```

### Error You'll See with PyPI Version

```
ImportError: cannot import name 'MutableSequence' from 'collections'
```

This is because Python 3.10+ moved `MutableSequence` to `collections.abc`. The GitHub version fixes this.

---

## Test Results on Tool - Opiate

### What madmom detected:

| Component | Result | Notes |
|-----------|--------|-------|
| **TempoEstimationProcessor** | 83.3 BPM (primary) | ✅ Correct! Matches user's 81.9 BPM taps |
| | 166.7 BPM (secondary) | Subdivision level |
| **DBNBeatTrackingProcessor** | 162.2 BPM | ❌ Tempo doubled |
| | 106 beats detected | Too many (tracking subdivisions) |
| **DownBeatProcessor** | 27 downbeats, 82 other beats | Works, but at wrong metric level |

### Comparison with User Taps

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision | 0.312 | Low - many false positives (extra beats) |
| Recall | 0.641 | Moderate - catches most user taps (doubled) |
| F1 Score | 0.420 | Moderate - not great for direct use |

**The issue**: Beat tracker locked onto hi-hat/subdivision level instead of kick/snare level.

**The win**: Tempo estimator correctly identified the right BPM!

---

## Comparison vs. librosa

Both made the **same mistake** on this rock track:

| Library | Method | Detected Tempo | Correct Tempo | Issue |
|---------|--------|----------------|---------------|-------|
| librosa | `beat_track()` | 161.5 BPM | ~82 BPM | Tempo doubled |
| madmom | `DBNBeatTrackingProcessor` | 162.2 BPM | ~82 BPM | Tempo doubled |
| madmom | `TempoEstimationProcessor` | **83.3 BPM** | ~82 BPM | ✅ Correct |

---

## Recommended Usage

### Option 1: Tempo Estimation Only (Recommended)

Use madmom to get accurate BPM, then use other methods for beat detection:

```python
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor

# Get RNN activations
rnn = RNNBeatProcessor()
activations = rnn('audio.wav')

# Get accurate tempo estimate
tempo_proc = TempoEstimationProcessor(fps=100)
tempos = tempo_proc(activations)
bpm = tempos[0][0]  # Primary tempo

print(f"Detected tempo: {bpm:.1f} BPM")

# Then use librosa or custom onset detection with this tempo constraint
```

### Option 2: Tune DBN Parameters

Constrain the beat tracker to avoid tempo doubling:

```python
from madmom.features.beats import DBNBeatTrackingProcessor

dbn = DBNBeatTrackingProcessor(
    min_bpm=60,           # Lower bound
    max_bpm=120,          # Upper bound (prevents doubling to 160+)
    transition_lambda=100,  # Reduce tempo fluctuation
    fps=100
)
beats = dbn(activations)
```

### Option 3: Multi-Level Detection + Selection

```python
# Detect at multiple tempo ranges
dbn_half = DBNBeatTrackingProcessor(min_bpm=40, max_bpm=100, fps=100)
dbn_full = DBNBeatTrackingProcessor(min_bpm=80, max_bpm=200, fps=100)

beats_half = dbn_half(activations)
beats_full = dbn_full(activations)

# Choose level closest to TempoEstimationProcessor result
tempo_estimate = tempos[0][0]
tempo_half = estimate_tempo(beats_half)
tempo_full = estimate_tempo(beats_full)

if abs(tempo_half - tempo_estimate) < abs(tempo_full - tempo_estimate):
    beats = beats_half
else:
    beats = beats_full
```

---

## Adding to Requirements

Since the GitHub version must be used, you can't add it to `requirements.txt` in the standard way.

### Option A: requirements.txt with git URL

```txt
# requirements.txt
sounddevice>=0.4.6
soundfile>=0.12.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
pyyaml>=6.0
matplotlib>=3.7.0
git+https://github.com/CPJKU/madmom.git
```

### Option B: Separate install step

Keep requirements.txt as-is, and document separate install:

```bash
pip install -r requirements.txt
pip install git+https://github.com/CPJKU/madmom.git
```

### Option C: Wait for PyPI release

Monitor https://github.com/CPJKU/madmom for the next PyPI release (likely v0.17.0 or v0.18.0) that includes Python 3.10+ fixes.

---

## When to Use madmom

### ✅ Good for:
- **Tempo estimation** on rock/metal/complex rhythms
- Genres where librosa struggles with tempo doubling/halving
- Getting multiple tempo candidates with confidence scores
- Downbeat detection (which beat is "1" in the bar)

### ⚠️ Use with caution:
- Direct beat tracking without parameter tuning
- Real-time applications (RNN processing is slower than librosa)
- Environments where you need PyPI-only dependencies

### ❌ Not ideal for:
- Simple, clean electronic music (librosa is faster and sufficient)
- Extremely low-latency requirements
- Systems without Python 3.10+ or where you can't install from git

---

## Further Testing Recommended

Before fully adopting madmom, test on:

1. More rock/metal tracks with complex rhythms
2. Electronic music with clear beats
3. Tracks with tempo changes
4. Different time signatures (3/4, 5/4, 7/8)

The Tool - Opiate test shows promise for **tempo estimation** but reveals **beat tracking** needs tuning for this genre.

---

## Full Results

See `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/madmom_results.md` for detailed test results including:
- F1 scores vs. user tap annotations
- Beat-by-beat comparison
- Tempo analysis across different sections
- Downbeat detection results
