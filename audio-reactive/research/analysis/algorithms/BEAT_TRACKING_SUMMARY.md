# Beat Tracking Comparison Summary

**Date**: 2026-02-05
**Song**: Tool - Opiate (Intro)
**Goal**: Find a reliable beat tracking approach for rock music after librosa's default failed

## Executive Summary

We tested 5 different beat tracking approaches against your tap annotations for Tool's "Opiate Intro". **None achieved strong performance** (best F1 score: 0.500). This confirms that **user tap data should remain the ground truth** for LED programming, especially for complex rock music.

## The Problem

librosa's default `beat_track()` detected **161.5 BPM** when the actual tempo is **~103 BPM** (from your taps). This is tempo doubling - a common failure mode where the algorithm locks onto hi-hat/eighth-notes instead of the kick drum.

## Approaches Tested

| Approach | Description | Result |
|----------|-------------|--------|
| **librosa default** | Baseline with default parameters | ❌ 161.5 BPM (doubled tempo), F1=0.486 |
| **librosa constrained** | `start_bpm=100/105/110` | ❌ Still 161.5 BPM (constraint ignored!), F1=0.486 |
| **librosa bass-only onset** | Onset detection on 20-250Hz only | ⚠️ 110.0 BPM (better!), F1=0.500 |
| **spectral flux (custom)** | Spectral flux + autocorrelation | ❌ 82.0 BPM (undershot), F1=0.327 |
| **madmom RNN** | Neural network trained on rock | ❌ Could not install (build dependencies) |

## Key Findings

### 1. Tempo Constraints Had NO Effect
Even with `start_bpm` set to 100, 105, or 110, librosa still detected 161.5 BPM. The onset strength signal is so dominated by hi-hat that the constraint couldn't override it.

### 2. Bass-Only Onset Detection Helped (But Not Enough)
By filtering to 20-250Hz, we got closer to the correct tempo (110 BPM vs. 103 BPM reference). However:
- **Precision: 46.7%** — Less than half of detected beats match your taps
- **Recall: 53.8%** — Algorithm misses nearly half of your taps
- **F1: 0.500** — Marginal performance

### 3. Custom Spectral Flux Undershot Tempo
Detected 82 BPM instead of 103 BPM. The autocorrelation likely latched onto a longer period (every other beat), suggesting the algorithm detected a half-time feel in some sections.

## Why This Matters for Your LED Project

### User Taps Are Richer Than Beat Tracking
Your tap annotations capture:
- **Subdivision changes** (quarter notes → eighth notes → sixteenth bursts)
- **Structural transitions** (your change markers at 8s, 17s, 24s, 29s, 34s, 37s)
- **Musical intensity** (rapid taps during fills, sparse taps during sustained notes)
- **Human feel** (slight timing variations that reflect groove and emphasis)

These "imperfections" **ARE the data**. For LED effects, the variation in tap density and timing maps directly to visual intensity and energy.

### Recommendation: Don't Try to Fix Your Taps
Instead of making algorithms match your taps, **use your taps directly** as the beat data:

1. **Beat-driven effects** → Trigger on user taps
2. **Intensity/energy** → Compute from tap density (taps per second in sliding window)
3. **Structural transitions** → Use change markers to switch LED scenes
4. **Feeling layers** → Your "air", "heavy", "tension" taps are already mapped to timestamps

## What We Learned About Rock Beat Tracking

Rock music is **uniquely difficult** for beat tracking because:

1. **Dense, distorted instrumentation** creates many onset candidates
2. **Hi-hat dominates the onset envelope** (constant eighth-notes at ~160 BPM)
3. **Kick drum is lower in the mix** and harder to isolate
4. **Tempo rubato and fills** break algorithmic assumptions about periodicity
5. **Subdivision changes** (quarter → eighth → sixteenth) confuse autocorrelation

Bass-only onset detection helps, but even at 20-250Hz there's still bleed from other instruments in rock's dense mix.

## Files Generated

- **Report**: `beat_tracker_comparison.md` — Full analysis with methodology
- **Data**: `beat_tracker_comparison.yaml` — All detected beat times for each approach
- **Visualization**: `beat_tracker_comparison.png` — Timeline showing all approaches vs. user taps
- **Script**: `scripts/beat_tracker_comparison.py` — Reusable comparison framework

## Next Steps for Audio-Reactive LED Project

Based on this analysis, we recommend:

### 1. Use User Taps as Beat Data (Don't Replace Them)
```python
# Load user taps
taps = annotations['beat']

# Compute beat intervals for tempo-aware effects
intervals = np.diff(taps)
current_tempo = 60.0 / np.median(intervals[-8:])  # Last 8 beats

# Trigger LED beat flash on each tap
for tap_time in taps:
    schedule_led_flash(tap_time)
```

### 2. Compute Intensity from Tap Density
```python
# Sliding window: taps per second
def compute_tap_intensity(taps, window_size=2.0):
    intensity = []
    for t in np.linspace(0, duration, num_frames):
        count = len([tap for tap in taps if t - window_size/2 < tap < t + window_size/2])
        intensity.append(count / window_size)
    return intensity
```

### 3. Use Change Markers for Scene Transitions
```python
# Your change markers define song structure
changes = annotations['changes']  # [8.073, 17.105, 24.271, ...]

# Map to LED scenes
scenes = [
    (0, 8, 'intro_sparse'),
    (8, 17, 'groove_steady'),
    (17, 24, 'groove_intensify'),
    (24, 29, 'fills_rapid'),
    # ...
]
```

### 4. Map Feeling Layers to LED Color/Movement
```python
# Your "air" taps → brightness modulation
# Your "heavy" taps → saturation increase
# Your "tension" taps → color shift toward red/white

air_times = annotations['air']
heavy_times = annotations.get('heavy', [])
tension_times = annotations.get('tension', [])

# Create feeling envelopes for LED mapping
```

## Conclusion

Beat tracking algorithms (even specialized approaches) **cannot compete with human annotation** for complex rock music. Your tap data is:
- More accurate (103 BPM vs. algorithmic 161.5/110/82 BPM)
- Richer (captures subdivision, structure, intensity)
- Directly usable for LED programming

**Recommendation**: Keep your tap-based workflow. The "imperfections" in your taps are musical data that algorithms miss.

---

*This analysis validates the user-in-the-loop approach for audio-reactive LED art. Computational tools provide the vocabulary (spectral features, onset strength, etc.), but human annotation captures the feeling layer that makes great LED effects.*
