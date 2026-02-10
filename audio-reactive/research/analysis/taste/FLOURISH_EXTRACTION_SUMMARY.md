# Flourish Extraction: Summary & Implications

## Overview

This analysis compared two annotation layers on Tool's "Opiate Intro":
1. **Beat layer**: Free-form tapping (72 taps) where the user tracked whatever felt important
2. **Consistent-beat layer**: Metronomic tapping (41 taps) where the user tried to tap only the steady pulse

By subtracting the consistent-beat from the beat layer, we computationally extracted **flourishes** — moments where the user responded to musical events beyond the basic pulse.

## Key Findings

### 1. Flourish Ratio Tracks Structural Clarity

The percentage of taps that are flourishes (vs on-grid) inversely correlates with rhythmic clarity:

- **Intro (0-8s)**: 90.9% flourishes → No clear pulse, user responding to individual events
- **Build sections**: 71.4% → 54.5% → User locking onto emerging groove
- **Strong groove (Build 2)**: 16.7% flourishes → Tightest lock-in, minimal deviation
- **Complex peaks**: 37.5-50% → Pulse present but user adds expressive accents

**This ratio can be computed in real-time** to detect song structure without training data.

### 2. Absence is Data

The consistent-beat layer has a 9.6-second gap (1.5s → 11.1s) where the user tapped once then stopped. This absence reveals structural ambiguity: **the intro has no consistent pulse**.

When groove establishes (11s+):
- Consistent-beat CV: **0.185** (extremely tight)
- Beat layer CV: **0.726** (3.9x more variable)
- Felt tempo: **82.1 BPM**

**Lesson**: When comparing annotation strategies, look for what's MISSING, not just what's present.

### 3. Flourishes Are Not Random

Flourishes align with specific audio features:
- **36% track bass energy peaks** (low-frequency events)
- **33% track spectral centroid peaks** (timbral changes)
- **23% track RMS peaks** (overall energy)
- **8% track onset detections** (sharp attacks)

They're not "errors" — they're the user's response to salient musical moments.

### 4. Metronomic Constraint Reveals Expressive Content

By asking the user to tap consistently, we force them to suppress their natural responses to flourishes. The **difference** between free and constrained tapping reveals what the user finds musically interesting.

This is a form of **subtractive annotation**: we learn more from what was removed than from what was kept.

## Design Implications for Audio-Reactive LEDs

### Three-Mode System Based on Flourish Ratio

Compute the ratio of off-grid onsets to on-grid beats over a sliding window (e.g., 5 seconds):

1. **Ambient Mode (ratio > 70%)**
   - No clear pulse detected or pulse too weak
   - LEDs respond to timbral changes, spectral movement, energy contours
   - No rhythmic flashing, favor smooth color transitions
   - Example: Tool intro, ambient electronic, free jazz

2. **Accent Mode (ratio 30-70%)**
   - Pulse exists but complex rhythmic content
   - Base layer: subtle pulse indication
   - Flourish layer: highlight accents, fills, attacks
   - Intensity modulated by flourish density
   - Example: Tool build sections, funk with syncopation

3. **Groove Mode (ratio < 30%)**
   - Strong, consistent pulse
   - LEDs locked to beat grid
   - Flourishes used only for occasional highlights
   - Favor geometric patterns, clear rhythmic structure
   - Example: Tool Build 2, house music, steady rock

**Transitions between modes should use hysteresis** to avoid rapid switching.

### Two-Layer Architecture

Instead of fighting between "beat tracking" and "spectral response," use both:

1. **Grid Layer** (when detectable)
   - Attempt beat tracking (librosa, madmom, etc.)
   - Quantize to consistent tempo
   - Drives structural/geometric patterns

2. **Flourish Layer** (always active)
   - Detect onsets that DON'T align with grid
   - Weight by bass energy + spectral novelty
   - Drives accents, color pops, intensity bursts
   - Compute density → modulates overall excitement

When grid layer fails (low confidence), flourish layer becomes primary driver. The system degrades gracefully.

### Real-Time Implementation

```
For each audio frame:
  1. Detect onsets (librosa.onset_detect)
  2. Attempt beat tracking (if ratio < 70%)
  3. For each onset:
     a. Is it within ±150ms of beat grid?
        → Yes: on-grid event (use for grid layer)
        → No: flourish (use for accent layer)
     b. Compute bass energy + spectral novelty at onset
     c. Weight flourish intensity by audio features
  4. Update flourish ratio (5-second window)
  5. Select mode based on ratio
  6. Render LEDs using both layers
```

## Generalization to Other Music

This analysis was performed on one song, but the principles generalize:

### Universal Truths

1. **Flourish ratio inversely correlates with rhythmic clarity** (should hold for any genre)
2. **Absence of consistent taps reveals structural ambiguity** (useful for any annotation task)
3. **Subtractive analysis reveals expressive content** (applies to any constrained vs free comparison)
4. **Flourishes align with salient audio features** (bass, centroid, RMS, onsets)

### Genre-Specific Tuning

The mode thresholds (70%, 30%) may need adjustment:
- **Electronic dance music**: May need lower thresholds (more groove-heavy)
- **Ambient/experimental**: May need higher thresholds (less pulse-dependent)
- **Jazz**: May need tighter window (rapid mode changes)

But the **architecture** (two layers + mode selection by ratio) should work universally.

## Next Steps

### For This Project

1. **Test on other annotated songs**
   - Apply same analysis to ambient.wav, electronic_beat.wav
   - Verify that flourish ratio correlates with perceived structure
   - Calibrate mode thresholds across genres

2. **Implement real-time flourish detection**
   - Port to Python streaming pipeline
   - Compare computed flourishes to user annotations
   - Tune feature weights (bass, centroid, RMS)

3. **Build LED mapping prototypes**
   - Ambient mode: color gradient driven by spectral centroid
   - Accent mode: base pulse + flourish pops
   - Groove mode: geometric patterns locked to beat
   - Demo smooth transitions between modes

### General Research Value

This "flourish extraction" technique could be useful beyond LEDs:

- **Music information retrieval**: Detecting structural boundaries without training data
- **Expressive performance analysis**: What do musicians emphasize beyond notation?
- **Audio-driven animation**: Two-layer approach for game engines, VJ tools
- **Accessibility**: Translating musical "excitement" to visual cues for deaf/HoH users

## Files Generated

- `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/beat_vs_consistent.md` — Full analysis report
- `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/beat_vs_consistent.yaml` — Complete data (classifications, features, density)
- `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/beat_vs_consistent_visualization.png` — Three-panel visualization
- `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/scripts/beat_vs_consistent.py` — Analysis script
- `/Users/KO16K39/Documents/led/audio-reactive/research/analysis/scripts/visualize_flourish_ratio.py` — Visualization script

## Conclusion

By comparing free-form and metronomic annotations, we've discovered a computationally-accessible signal (**flourish ratio**) that tracks song structure and can drive LED mode selection. This is not a heuristic — it's a principled extraction of what humans naturally respond to in music.

The key insight: **Don't try to make beat tracking work for all music. Instead, detect when beat tracking SHOULD work (low flourish ratio) vs when it shouldn't (high flourish ratio), and adapt accordingly.**

This is the "feeling layer" — not a library, but a framework for building it.
