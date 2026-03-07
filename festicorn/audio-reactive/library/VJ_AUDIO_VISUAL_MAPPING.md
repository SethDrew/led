# VJ Audio-Visual Mapping Conventions

How professional VJ systems map audio features to visual parameters. Based on research across Resolume, TouchDesigner, VDMX, MadMapper, LedFx, WLED-SR, and academic literature on music-movement perception.

## Quick Reference

### The Visual Parameter Space

Six dimensions VJ software exposes for audio-reactive control:

| Visual Parameter | Primary Audio Feature | Secondary | Direction | ESP32 Feasible |
|---|---|---|---|---|
| **Brightness** | RMS energy | Onset strength | Louder → brighter | Yes |
| **Color / hue** | Spectral centroid | Frequency bands | Higher centroid → cooler (blue) | Yes |
| **Speed** | Tempo | Spectral flux | Faster tempo → faster animation | Moderate |
| **Complexity** | Spectral density | Polyphony estimate | More harmonics → more sparkles | Moderate |
| **Spatial position** | Frequency band | — | Bass at base, treble at tips | Yes |
| **Rhythm sync** | Beat / onset detection | — | Triggers, strobes, transitions | Yes |

### What VJ Software Actually Uses

No major VJ platform (Resolume, TouchDesigner, VDMX, MadMapper, Magic Music Visuals) exposes ZCR, spectral rolloff, or MFCCs. Their audio analysis pipelines are:

1. **FFT → 3 bands** (low/mid/high) — the universal foundation
2. **RMS / amplitude** — overall energy envelope
3. **Beat / onset detection** — rhythmic triggers
4. **BPM sync** (often via Ableton Link or MIDI clock)

Advanced tools (TouchDesigner, Resolume Wire) add spectral centroid and spectral flux but these are niche.

---

## Audio → Visual Mappings in Detail

### Brightness (most intuitive mapping)

**Audio**: RMS energy, onset strength
**Visual**: LED brightness, opacity, flash intensity

Constants from successful projects:
- Gamma correction: 2.2 for perceptual linearity on LEDs
- Power law: x^0.9 (scottlawsonbc) to x^2.2 for dynamic range compression
- Asymmetric smoothing: rise α=0.7-0.99 (instant flash), decay α=0.1-0.3 (smooth fade)

**Anti-pattern**: Per-frame min-max normalization destroys dynamics — quiet and loud sections look identical.

### Color Temperature

**Audio**: Spectral centroid (brightness of the sound)
**Visual**: Warm (red/orange) ↔ cool (blue/cyan)

```
centroid = sum(frequencies * magnitudes) / sum(magnitudes)
hue = map(centroid, min_freq, max_freq, 0, 360)
```

The HSV model decouples color from brightness — spectral centroid drives hue while RMS drives value. Spectral flatness can drive saturation (tonal = saturated, noisy = desaturated).

ZCR correlates with spectral centroid at r=0.85 (measured across 8000 FMA Small tracks). Spectral centroid is strictly better — ZCR is redundant if FFT is available.

### Speed / Motion Rate

**Audio**: Tempo (BPM), spectral flux
**Visual**: Animation speed, crawl rate, pulse frequency

From motion research (Frontiers in Psychology, 2013):
- High-frequency spectral flux (6400-12800 Hz) → head and hand movement speed
- Percussive sounds → sharp, staccato visual movements
- Sustained sounds → smooth, flowing visual movements

Clean octave multiples of detected tempo are acceptable for LEDs (from our `no-prior-for-leds` finding).

### Complexity / Density

**Audio**: Spectral density, harmonic richness, polyphony
**Visual**: Number of active elements, sparkle density, pattern width

Spectral flatness as proxy:
- Flat spectrum (noise-like) → high complexity, many visual elements
- Peaked spectrum (tonal) → low complexity, few focused elements

### Spatial Position

**Audio**: Frequency band
**Visual**: Physical position on LED strip/tree

Universal VJ convention: bass at base, treble at tips. Maps naturally to tree topology (trunk = bass, branches = treble).

### Rhythm Sync

**Audio**: Beat tracking, onset detection
**Visual**: Pattern changes, strobes, color shifts on beat

Two approaches in practice:
- **Event-triggered**: Onset/beat fires a one-shot animation (flash, chase, color change)
- **Phase-locked**: Animation runs continuously, phase-locked to detected tempo

---

## Perceptual Science

### Circumplex Model (Valence-Arousal)

The dominant framework in music emotion research. Two axes:

**Arousal** (calm ↔ activated) — maps cleanly to visuals:
- High arousal (fast, loud, bright timbre) → brighter, faster, more complex
- Low arousal (slow, quiet, soft timbre) → dimmer, slower, simpler
- Computable from: tempo, RMS, spectral centroid

**Valence** (negative ↔ positive) — hard to compute:
- Positive (major mode, consonance) → warmer colors?
- Negative (minor mode, dissonance) → cooler colors?
- Requires harmonic analysis — not feasible on ESP32 in real-time

### Music Emotion Recognition (MER) — 8 Dimensions

| Dimension | Computable in Real-Time? | Notes |
|---|---|---|
| **Dynamics** (loudness, attack/decay) | Yes | RMS, onset strength |
| **Rhythm** (tempo, meter, syncopation) | Moderate | Needs 2-4s buffer for tempo |
| **Tone color** (timbre, spectral shape) | Yes | Spectral centroid, MFCCs |
| **Melody** (pitch contour, intervals) | No | Requires pitch tracking + context |
| **Harmony** (chords, consonance) | No | Requires harmonic analysis |
| **Expressivity** (micro-timing, vibrato) | No | Sub-frame precision needed |
| **Texture** (mono/polyphonic) | Moderate | Spectral flatness as proxy |
| **Form** (structure, repetition) | No | Requires long-term memory |

Only dynamics, rhythm, and tone color are tractable for real-time embedded.

---

## Proven Constants

From `RESEARCH_AUDIO_VISUAL_MAPPING.md` and WLED-SR analysis:

### Smoothing (Asymmetric Attack/Decay)

| Context | Rise α | Decay α | Time Constant (60fps) | Behavior |
|---------|--------|---------|----------------------|----------|
| Bass flash (red) | 0.99 | 0.2 | ~5 frames (83ms) | Sticky peaks |
| Transient (green) | 0.3 | 0.05 | ~20 frames (333ms) | Smooth |
| Balanced (blue) | 0.5 | 0.1 | ~10 frames (167ms) | Mid-ground |
| WLED-SR AGC attack | — | — | 80ms | Fast response |
| WLED-SR AGC decay | — | — | 1400ms | Slow release |

### Frequency Splits

| Band | Range | Maps To |
|------|-------|---------|
| Bass | 200-800 Hz | Brightness, spatial base, pulse |
| Mids | 800-3000 Hz | Color, spatial middle |
| Highs | 3000-12000 Hz | Sparkle, spatial tips, complexity |

### Normalization

Rolling window peak normalization (track recent max, divide by it):
- Peak decay α: 0.1-0.2 (remembers ~2-5 seconds)
- Bounds adaptation to prevent pumping
- Never use per-frame min-max (destroys dynamics)

---

## Sources

### Code Analysis
- scottlawsonbc/audio-reactive-led-strip — visualization.py, config.py, dsp.py
- LedFx/LedFx — effects/spectrum.py, effects/energy.py, effects/bands.py
- WLED AudioReactive — audio_reactive.cpp, FX.cpp (30 audio-reactive effects)

### VJ Software Documentation
- Resolume Wire FFT, TouchDesigner Audio Spectrum CHOP, VDMX Audio Analysis, MadMapper Audio Analyser

### Academic
- Burger et al. (2013) "Influences of Rhythm- and Timbre-Related Musical Features on Characteristics of Music-Induced Movement" — Frontiers in Psychology
- Yang & Chen — "Machine Recognition of Music Emotion: A Review"
- Hevner's Adjective Circle — mood descriptor taxonomy on valence-arousal plane

### Dataset Validation
- FMA Small (8000 tracks, 8 genres): ZCR-centroid correlation r=0.85, ZCR-RMS correlation weak — confirms ZCR is redundant with FFT-based features
