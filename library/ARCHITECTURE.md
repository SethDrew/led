# Architecture Document -- LED Effect Design Space

Companion to the research ledger. The ledger records findings; this document organizes them into a framework for creative decision-making.

A shared mental model, not a spec. The goal is to compress context so that any session can start from a common understanding of the design space. Context management is the expensive resource, not code -- investment goes into shared mental models, not code abstractions. (Ledger: `context-expensive-code-cheap`)

**Everything below is illustrative, not exhaustive.** The seven axes are the framework; specific items are examples from current research.

---

## 1. The Design Space (Seven Axes)

Any effect is a point (or region) in this space -- a combination of choices along each axis. The axes are largely independent. Additional axes may emerge; these seven capture the dimensions we've found useful.

### Axis 1: Topologies

Sculpture shapes and their spatial properties. Defined in `hardware/sculptures.json`, mapped in `runner.py:apply_topology()`. Current sculptures: `cob_flat` (80 LEDs), `cob_diamond` (73, 3 branches, height-mapped), `strip_150` (150), `tree_flat` (197, ESP32, 3 pins).

Key spatial concepts: zones (frequency-mapped regions), height gradients (base-to-tip), branch independence, gamma curves for geometry compensation. Two mapping modes: `branch` (each branch gets own slice) and `height` (single height axis, all branches sample from it).

### Axis 2: Musical Events

What happens in music that LEDs should respond to.

**Build phases** (from `build-taxonomy`): Primer (cyclic ramps), Sustained (upward trajectory), Bridge (chaos, highest RMS), Drop (sustained plateau, NOT peak intensity). Independent, not sequential.

**Other structural events**: Dropout/reintroduction, crescendo/climax (genre-dependent signatures), riser, transition, harmonic section.

**Behavioral modes** (from `accent-vs-groove-effects`): Accent (individual hits, sparse percussion, low flourish ratio <30%) vs Groove (continuous rhythmic feel, dense percussion, high flourish ratio >70%).

**Genre is excluded as an axis** -- it's derivative of observable feature properties. See ledger: `genre-awareness-is-derivative`.

### Axis 3: Audio Features

What we can compute. Two key principles; full catalog in `AUDIO_FEATURES.md` (library).

**Derivatives over absolutes:** Build vs climax has identical static features (RMS +/-0.5%) but climax brightens 58x faster. The derivative is the signal, not position. (Ledger: `derivatives-over-absolutes`)

**Context deviation:** All features are relative to a running baseline. The adaptation time constant IS the temporal scope. A subjective quality is not a fixed spectral shape -- it's deviation from the surrounding context's norm. (Ledger: `feelings-are-contextual`)

### Axis 4: LED Behaviors

The visual vocabulary, independent of what triggers them. Three categories: foreground (transient, additive -- sparkle, pulse, flash, traveling pulse, collision), background (persistent, base layer -- breathe, zone coloring, flow, gradient shift), and standalone (self-contained composition -- sidechain pump, band proportional, growth).

Foreground vs background is a composition decision, not a behavior property. The C++ `Effect.h` formalizes this: `BackgroundEffect` uses REPLACE, `ForegroundEffect` uses ADD.

Full visual pattern catalog in `MATHY_EFFECTS_CATALOG.md` (library).

### Axis 5: Temporal Scope

The time horizon of both music and effects. To detect a phrase-level musical event, you need phrase-level memory in the effect.

| Scope | Duration | Musical events (examples) | Features needed | Effect memory | Example effects |
|---|---|---|---|---|---|
| **Frame** | ~11ms (2048 samples) | Individual transients, onset timing | Onset strength, spectral flux, ZCR | None (stateless) | Sparkle trigger, percussive flash |
| **Beat** | 300-600ms | Rhythmic pulse, individual hits | Absint (150ms window), beat predictor | ~1s (cooldown, last beat time) | `impulse`, `bass_pulse`, `band_zone_pulse` |
| **Phrase** | 4-30s | Builds, sections, crescendos, drops | Rolling integral (5s), autocorrelation (5s) | 5-30s (ring buffers, running stats) | `impulse_sections`, `longint_sections` |
| **Song** | Minutes | Overall energy arc, narrative shape | Running baseline, adaptation constants | Minutes (slow-decay peaks) | Adaptation baselines, per-band normalization |

**The adaptation time constant IS the temporal scope.** When you compute a feature relative to a 5-second running average, you're operating at phrase scope. When you use a 150ms window, you're at beat scope.

**Current gap: no streaming event detection.** All event detection code lives in `viewer.py` as batch-only visualization. Foote's checkerboard novelty is O(n^2) -- fundamentally non-streaming. A streaming event detector needs rolling integral slope or per-band dropout detection. (Ledger: `event-detection-viewer-only`)

### Axis 6: Composition

How effects combine.

**Blend modes**: REPLACE (background overwrites), ADD (foreground overlay), MULTIPLY (masking, unused), ALPHA (crossfades, unused). Critical lesson: pulses must REPLACE background at their pixels, not blend additively -- additive shifts pulse hue. (Ledger: `pulse-compositing-replace-not-blend`)

**Spatial composition**: Different effects on different topology zones. Bass at base, treble at tips.

**Temporal composition**: Different effects in different sections. Build -> anticipation, drop -> full-brightness, breakdown -> ambient.

**DAW concepts worth stealing** (from `daw-concepts-for-leds`): Sidechain (one feature suppresses another visual), sends, ADSR envelopes, wet/dry mix. The meta-insight: best DAW engineers create relationships between elements, not independent mappings.

### Axis 7: Perceptual Mapping

The bridge between audio features and visual parameters.

| Audio | Visual | Direction |
|---|---|---|
| Loudness (RMS) | Brightness | Louder -> brighter (gamma 2.2) |
| Spectral centroid | Color temperature | Higher -> cooler (blue) |
| Tempo | Motion speed | Faster tempo -> faster animation |
| Spectral richness | Visual density | More harmonics -> more sparkles |
| Frequency band | Spatial position | Bass at base, treble at tips |
| Onset strength | Flash intensity | Harder hit -> brighter flash (x^0.9) |

VJ prior art and proven constants in `VJ_AUDIO_VISUAL_MAPPING.md` (library). Perceptual color on hardware in `COLOR_ENGINEERING.md` (library).

---

## 2. Shared Vocabulary

Precise definitions for terms used throughout the project. Prevents re-deriving definitions each session.

- **Build phases:** Primer, sustained, bridge, drop -- four independent phases. Bridge has HIGHER RMS than drop. Drop is sustained intensity, not peak. (`build-taxonomy`)
- **Accent effect:** Responds to individual hits. Sparse percussion. `band_zone_pulse`. (`accent-vs-groove-effects`)
- **Groove effect:** Responds to continuous rhythmic feel. Dense percussion. `tempo_pulse`, `rap_pulse`. (`accent-vs-groove-effects`)
- **Foreground / Background / Standalone:** Foreground = transient, ADD blend. Background = persistent, REPLACE blend. Standalone = self-contained composition.
- **Derivatives vs absolutes:** Rate-of-change is the signal. Build vs climax have identical RMS but climax brightens 58x faster. (`derivatives-over-absolutes`)
- **Context deviation:** Feelings are relative to surrounding context, not absolute values. Use deviation-from-running-average. (`feelings-are-contextual`)
- **The two quality axes:** Audio decomposition and LED mapping quality are independent. WLED proves crude audio + great visuals = great product. (`two-quality-axes`)
- **Flourish ratio:** Off-grid vs on-grid tap ratio. Ambient (>70%), accent (30-70%), groove (<30%). (`flourish-ratio`)
- **Absint:** |d(RMS)/dt| integrated over 150ms. Measures how much loudness is *changing*. Primary beat detection for non-dense music.
- **Onset envelope:** FFT -> mel bands -> log energy -> diff -> half-wave rectify -> mean. Right signal for tempo autocorrelation on percussive music. (`onset-envelope-for-tempo`)

---

## 3. The Two Quality Axes

The project's guiding principle.

```
              LED Mapping Quality (visual design)
                         ^
              Great      |      Great analysis,
              analysis,  |      great LEDs
              bad LEDs   |      (THE GOAL)
           -------------+--------------> Audio Decomposition
              Crude      |      Crude analysis,
              analysis,  |      great LEDs
              bad LEDs   |      (WLED, Fluora)
```

WLED proves the bottom-right works: crude audio + great visuals = great product. The project's thesis is that the top-right is strictly better -- section-aware effects and build anticipation require phrase-level understanding that creates something qualitatively different. Don't over-invest in decomposition at the expense of visual design -- improve both in parallel.

---

## 4. Effect Design Patterns

Recurring patterns observed across our effects. Not implementations -- structural archetypes.

| Pattern | Description | Audio | Topology | Temporal Scope | Examples |
|---|---|---|---|---|---|
| **Accent** | Responds to individual hits with per-event visual reactions | Per-band peak detection, HPSS, spectral flux | Frequency zones mapped to positions | Beat | `band_zone_pulse`, `band_sparkle_flux` |
| **Groove** | Responds to continuous rhythmic feel with tempo-locked oscillation | Onset-envelope autocorrelation, phase-lock | Whole-strip or zoned, uniform | Phrase | `tempo_pulse`, `rap_pulse` |
| **Section** | Responds to structural changes over 4-30s windows | Rolling integral, long-horizon RMS | Fibonacci sections, height-mapped | Phrase to song | `impulse_sections`, `longint_sections` |
| **Ambient** | Responds to slow spectral change with smooth evolution | Centroid (smoothed), dominant band, RMS baseline | Full-strip gradients, zone coloring | Phrase to song | `band_sparkles` (background) |
| **Proportional** | Maps feature value directly to visual parameter, no thresholds | Any continuous feature | Any | Beat | `impulse_glow`, `rms_meter` |
| **Composite** | Layers multiple behaviors using compositor (BG replace + FG add) | Multiple | Multiple | Multiple | `band_zone_pulse`, `three_voices` |

### Cross-reference: Patterns x Topologies

| Pattern | Linear strip | Diamond (height) | Tree (flat) |
|---|---|---|---|
| Accent | Zone sparkles | Height-mapped sparkles | Full-strip sparkles |
| Groove | Whole-strip pulse | All branches pulse together | Whole-tree pulse |
| Section | Fibonacci sections | Base-to-tip growth | Fibonacci sections |
| Ambient | Color gradient | Height gradient | Color gradient |
| Proportional | Meter (base up) | Height fill | Meter |

---

## 5. Input Roles

Effects declare abstract input roles, not specific signals. The same role can be filled by different sources (audio feature, user knob, MIDI clock, ML model). This separation enables swapping signal sources without changing effect code. (Ledger: `input-role-taxonomy`)

| Role | What It Controls | Timescale | Example Sources |
|---|---|---|---|
| **EVENT** | Triggers discrete action (spawn, flash) | Instantaneous | Onset detection, beat prediction, button press |
| **RATE** | Speed or frequency of continuous behavior | Frame-level | Tempo, onset density, MIDI clock |
| **INTENSITY** | Magnitude (brightness, force, size) | Frame-level | RMS, absint, per-band energy, user fader |
| **MOOD** | Qualitative character (warm/cool, tense/relaxed) | Slow-moving | Spectral centroid, dominant band, user palette |
| **TRAJECTORY** | Directional change over time (rising/falling) | Phrase-level | Rolling integral slope, section detector state |
| **TEXTURE** | Roughness, density, granularity | Frame-level | Spectral flatness, onset density, ZCR |
| **NOVELTY** | Deviation from recent context | Phrase-level | MFCC distance, spectral flux variance |
| **DENSITY** | How many entities/events coexist | Beat-level | Per-band onset count, spectral complexity |
| **SPACE** | Spatial position or spread | Frame-level | Per-band energy, stereo field |
| **POSITION** | Specific point or region on sculpture | Frame-level | Frequency-to-height mapping, random walk |
| **FAMILIARITY** | How well-known the current material is | Song-level | Repetition detector, chorus vs verse confidence |
| **SURPRISE** | Unexpected events (dropout, sudden change) | Instantaneous | Per-band dropout, RMS step function |
| **PHASE** | Cyclic position within repeating pattern (0-1) | Beat-level | Beat phase, bar phase, LFO |

### Role Composition Patterns

- **A: Event + Intensity** (trigger-and-scale) -- EVENT triggers discrete action, INTENSITY scales how dramatic it is.
- **B: Rate + Mood** (continuous character) -- RATE controls speed, MOOD shifts character.
- **C: Trajectory + Density** (arc-and-population) -- TRAJECTORY drives long-term direction, DENSITY determines how much is happening.
- **D: Event + Surprise + Novelty** (structural response) -- responds to musical structure at multiple levels.
- **E: Space + Position + Phase** (spatial animation) -- spatially-distributed animation with cyclic motion.
- **F: Intensity + Texture + Mood** (ambient character) -- continuous ambient texture with no discrete events.

Full P/S mapping table across all effects in `INPUT_ROLE_MATRIX.md` (library). Per-band peak-decay normalization — the shared preprocessing that puts all INTENSITY sources on a 0-1 scale — in `PER_BAND_NORMALIZATION.md` (library).

---

## 6. Spatial Architecture

Sculptures have three distinct coordinate systems. Confusing them leads to wrong design decisions.

| System | Dimension | What It Means | When to Use |
|---|---|---|---|
| **Index space** | 1D | `led[i]` -- array offset in data buffer | Serial output, memory layout, simple effects |
| **Topology space** | 1D per branch + height | Position along branch (0=base, 1=tip), branch identity, shared height axis. Provided by `apply_topology()` via `sculptures.json` | Branch-aware effects, height-mapped effects. **Already implemented.** |
| **World space** | 3D | (x,y,z) physical position in centimeters | Noise fields, wave propagation, ripples. **Not yet implemented.** |

**What 3D coordinates unlock:** World-space rendering correct from every viewing angle. Perlin noise, plasma, voronoi, metaballs, ripple, reaction-diffusion all become significantly better. **Practical path:** photograph sculpture with ruler, annotate key LEDs, interpolate, store as `"positions": [[x,y,z], ...]` in `sculptures.json`. Even approximate coordinates (within 1-2cm) suffice.

Topology-aware idioms and 3D-dependent effects in `NATURE_TOPOLOGY_EFFECTS_CATALOG.md` (library).

---

## 7. Show Architecture

What it takes to automate a multi-hour show without a human VJ.

**What VJs actually do:** (1) effect selection based on musical section, (2) parameter modulation -- tweaking intensity/speed/color continuously, (3) transitions -- crossfading at musical boundaries. VJs **prepare** structure and **improvise** details.

### Detection Stack

| Layer | Scope | Status | Key Capabilities |
|---|---|---|---|
| **1: Frame-level** | ~11ms | Implemented | RMS, onset/transient, spectral centroid, per-band energy, absint |
| **2: Beat-level** | 300-600ms | Partial | Tempo, beat prediction, onset envelope. Gap: true downbeat detection |
| **3: Phrase-level** | 4-30s | **The critical gap** | Build/drop/breakdown detection, section boundaries. Needs rolling integral slope, per-band dropout |
| **4: Song-level** | Minutes | Future | Song transitions, genre/mood shift, energy arc, key detection |

Real-time phrase/section detection is **not solved**. Beat/downbeat tracking is solved (BeatNet, BEAST >80% accuracy). But semantic section detection relies on self-similarity matrices requiring the full song -- fundamentally non-streaming. Simple heuristics (rolling integral slope, onset density gradients) are crude but causal and fast.

### Show Controller

```
Audio Features --> [ Section Detector: Build/Drop/Break state, boundaries ]
                   [ Role Binder: audio/knob/ML -> input roles             ]
                   [ Effect Selector: section->effect map, transitions     ]
                   [ Parameter Modulator: role->visual, energy curve       ]
                   [ Palette Manager: mood->palette, rotation timer        ]
```

### Section -> Effect Mapping (Starting Point)

| Detected Section | Effect Family | Intensity | Palette | Speed |
|---|---|---|---|---|
| **Intro / Low Energy** | Ambient (noise field, bioluminescence, breathing) | Low | Neutral | Slow |
| **Verse / Groove** | Groove (tempo-locked pulse, flowing noise) | Medium | Warm | Locked to tempo |
| **Build** | Convergence + rising complexity | Rising | Shifts warmer | Accelerating |
| **Drop** | Full-intensity (root pulse, fire, storm) | High, sustained | Hot (reds, whites) | Fast |
| **Breakdown** | Ambient + sparkle (embers, fireflies) | Low-medium | Cool (blues, purples) | Slow |
| **Climax / Peak** | Max complexity (composite: BG + sparkle + pulse) | Maximum | White-hot | Maximum |
| **Outro / Fade** | Retreat (reverse growth, cooling embers) | Decreasing | Cooling | Decelerating |

### Preventing Visual Fatigue

- **Effect rotation** -- don't use the same effect for more than N consecutive sections of the same type.
- **Palette cycling** -- rotate through palette families over 10-20 minute periods.
- **Complexity curve** -- track long-horizon visual complexity; force simpler sections if too high for too long.
- **Spatial rotation** -- alternate between topology modes (full-strip, zoned, height-mapped, branch-independent).
- **Novelty injection** -- periodically introduce a rare effect; surprise prevents habituation.

Multi-entity interaction types in `ENTITY_INTERACTIONS.md` (library).
