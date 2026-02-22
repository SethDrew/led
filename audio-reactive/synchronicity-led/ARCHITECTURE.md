# Architecture Document — LED Effect Design Space

Companion to the research ledger. The ledger records findings; this document organizes them into a framework for creative decision-making.

This is a shared mental model, not a spec. No function signatures, no refactoring plan. The goal is to compress context so that any session can start from a common understanding of the design space.

**Why this document exists:** Context management is the expensive resource, not code. AI can write one-off effects cheaply. Effects can all be one-offs and likely will need to be for ESP32 performance tuning. No unified transformation framework needed. Investment goes into shared mental models (this document, the ledger) not code abstractions. (Ledger: `context-expensive-code-cheap`)

**Everything below is illustrative, not exhaustive.** The seven axes are the framework; the specific items listed under each are examples drawn from current research. All lists can be extended, revised, or reorganized as understanding deepens. When this document says "behaviors include sparkle, pulse, flash" it means "here are some behaviors we've identified" — not "these are the only behaviors."

---

## 1. The Design Space

Seven axes we've identified so far for thinking about the design space. Any effect is a point (or region) in this space — a combination of choices along each axis. The axes are largely independent: you can change the topology without changing the temporal scope, or change the audio features without changing the LED behavior. Additional axes may emerge; these seven capture the dimensions we've found useful.

### Axis 1: Topologies

Sculpture shapes and their spatial properties. Defined in `hardware/sculptures.json`, mapped in `runner.py:apply_topology()`. These are just represenative examples, not an exhaustive list.

| Topology | Physical LEDs | Spatial Properties | Modes |
|---|---|---|---|
| `cob_flat` | 80 (WS2811 12V) | Linear strip, single branch | Linear zones, gradients |
| `cob_diamond` | 73 (3 branches) | Height-mapped: up1(42) + down(20) + up2(11) | Height axis shared across branches |
| `strip_150` | 150 | Linear strip, single branch | Linear zones, gradients |
| `tree_flat` | 197 (ESP32, 3 pins) | Flat: all 197 treated as single strip | Linear zones, full-strip effects |

**Spatial concepts each topology supports:**

- **Zones** — divide the strip into frequency-mapped regions (all topologies)
- **Height gradients** — base-to-tip mapping, meaningful on diamond where branches share a height axis
- **Branch independence** — different effects on different branches (diamond only, currently unused)
- **Gamma curves** — non-linear height mapping per branch (`gamma` parameter in sculptures.json). Diamond's `up2` uses gamma=0.55 to compensate for physical geometry where the branch rises steeply then levels off

**Two mapping modes (so far!!):**
- `branch` (default): each branch gets its own slice of the logical LED array
- `height`: logical array is a single height axis (0=base, 1=peak); all branches sample from it at their own resolution

### Axis 2: Musical Events

What happens in music that LEDs should respond to. Examples of vocabulary drawn from validated ledger entries so far — this list grows as we encounter new musical structures.

**Build phases** (from `build-taxonomy`) — one example of structural decomposition:
- **Primer** — cyclic ramps, anticipation without commitment
- **Sustained** — upward trajectory, building energy
- **Bridge** — chaos, highest RMS (higher than drop), transition turbulence
- **Drop** — sustained plateau, NOT peak intensity — sustained intensity

These phases are independent, not sequential. A song can skip phases or reorder them. Other songs may exhibit structural patterns that don't fit this taxonomy at all.

**Some other structural events we've identified:**
- **Dropout** — band goes silent; reference freezes so reintroduction hits hard (from `per-band-normalization-with-dropout-handling`)
- **Crescendo/Climax** — genre-dependent: rock crescendo = gradual swell; EDM drop = sudden sustained intensity. Same word, different audio signatures
- **Riser** — sweeping frequency buildup, often before drops
- **Transition** — section boundary, change in texture/rhythm
- **Harmonic section** — sustained chords/pads, low onset density

**Behavioral modes identified so far** (from `accent-vs-groove-effects`):
- **Accent** — music characterized by individual hits; sparse percussion, prominent transients. Band zone pulse works here. Low flourish ratio (<30% off-grid taps)
- **Groove** — music characterized by continuous rhythmic feel; dense percussion, locked beat. Needs tempo-locked pulsing, not per-hit reactions. High flourish ratio (>70% off-grid taps) for ambient sections

**Why genre is excluded as an axis:** Genre awareness (electronic, rock, hip-hop) is derivative of observable feature properties. Whether music is "accent" or "groove" comes from onset density, spectral continuity, and flourish ratio — not from a genre label. Genre is an emergent label on top of these properties, not an independent input. See ledger entry `genre-awareness-is-derivative`.

### Axis 3: Audio Features

What we can compute. Organized into three categories so far, plus per-band expansion. New features and categories will emerge as the project evolves.

#### Standard features (librosa / numpy)

| Feature | What it measures | Real-time? | Temporal scope |
|---|---|---|---|
| RMS | Loudness (energy) | Yes | Frame (46ms) |
| Onset strength | Instantaneous spectral flux | Yes | Frame |
| Spectral centroid | Brightness / timbral center | Yes | Frame |
| Spectral flatness | Noise-like vs tonal | Yes | Frame |
| ZCR (zero-crossing rate) | Noisiness | Yes | Frame |
| MFCCs | Timbral fingerprint (13 coefficients) | Yes | Frame |
| Chroma | Pitch class distribution (12 bins) | Yes | Frame |
| HPSS | Harmonic/percussive separation | Yes (streaming temporal-median) | Frame |
| Mel spectrogram | Perceptually-weighted frequency energy | Yes | Frame |

#### Calculus features (custom, first-class)

Our custom signal processing primitives in `effects/signals.py`. These are not standard library features — they emerged from project-specific research and are treated as first-class alongside librosa features.

| Feature | What it measures | Real-time? | Temporal scope | Source |
|---|---|---|---|---|
| AbsIntegral | \|d(RMS)/dt\| summed over 150ms — rate of loudness change | Yes | Beat (~150ms window) | `signals.py:AbsIntegral` |
| Multi-band onset envelope | FFT → 6 mel bands → log energy → diff → half-wave rectify → mean | Yes | Frame | `signals.py:OnsetTempoTracker` |
| Autocorrelation tempo | Periodicity detection from 5s buffer of absint or onset signal | Yes | Phrase (5s) | `signals.py:BeatPredictor`, `OnsetTempoTracker` |
| Rolling integral | Sum of energy over 5s window per band — sustained energy detector | Yes | Phrase (5s) | `viewer.py` (visualization only) |
| Rolling integral slope | d/dt of rolling integral — rising (build) vs falling (breakdown) | Untested | Phrase | Concept only |

#### Per-band expansion

Any scalar feature can be computed independently per frequency band. With 5 bands, this multiplies the feature space by 5x. Not all features benefit equally:

- **Benefits strongly**: RMS, onset strength, spectral flux, absint (each band has independent energy dynamics)
- **Benefits moderately**: spectral centroid (sub-band centroid gives finer frequency detail)
- **Doesn't benefit**: overall spectral flatness, chroma (already frequency-decomposed)

Standard band definitions (from `band_zone_pulse.py`): sub-bass (20-80Hz), bass (80-250Hz), low-mid (250-1kHz), high-mid (1-4kHz), treble (4-16kHz).

#### Key principle: derivatives over absolutes

Build vs climax has identical static features (RMS ±0.5%) but climax brightens 58x faster. The derivative is the signal, not position. (Ledger: `derivatives-over-absolutes`)

#### Key principle: context deviation

All features are relative to a running baseline. The adaptation time constant IS the temporal scope. "Airy" is not a fixed spectral shape — it's deviation from the surrounding context's norm. (Ledger: `airiness-context-deviation`)

### Axis 4: LED Behaviors

Examples of the visual vocabulary, independent of what triggers them. These are behaviors we've built or identified — not a closed set.

#### Foreground behaviors (transient, additive) — e.g.

| Behavior | Description | Natural drivers | Topology fit |
|---|---|---|---|
| Sparkle | Single-pixel flash, fast decay | Percussive onset, band peak | All |
| Pulse | Whole-strip or zone flash, exponential decay | Beat detection, absint threshold | All |
| Traveling pulse | Gaussian blob that moves along strip | Beat + velocity from strength | Linear, tree |
| Flash | Instant full-brightness, snap off | Strong onset, downbeat | All |
| Collision | Two pulses meet and explode | Beat pairs, stereo onsets | Linear |

#### Background behaviors (persistent, base layer) — e.g.

| Behavior | Description | Natural drivers | Topology fit |
|---|---|---|---|
| Breathe | Slow sine wave brightness oscillation | Tempo (1/2 or 1/4 rate), RMS | All |
| Zone coloring | Different colors for different strip regions | Dominant band (5s vote) | Zoned topologies |
| Flow / crawl | Continuous movement along strip | Tempo, spectral centroid | Linear, tree |
| Volume meter | Lit LED count proportional to energy | RMS, absint | Linear |
| Gradient shift | Color temperature drifts with spectral content | Centroid (warm→cool) | All |

#### Standalone behaviors — e.g.

| Behavior | Description | Natural drivers | Topology fit |
|---|---|---|---|
| Sidechain pump | Brightness dips on hits, recovers between | Bass energy, kick detection | All |
| Band proportional | RGB channels map to bass/mid/high energy | Per-band absint | All |
| Growth | LEDs light progressively from base | Rolling integral, build detection | Height-mapped |

#### Foreground vs background is a composition decision, not a behavior property

The C++ static-animations system (`Effect.h`) formalizes this: `BackgroundEffect` uses REPLACE blend mode, `ForegroundEffect` uses ADD. The same visual behavior (e.g., a pulse) could be background (full-strip replace) or foreground (additive sparkle overlay).

### Axis 5: Temporal Scope

The time horizon of both music and effects. Collapsed from "time scales" and "memory" because to detect a phrase-level musical event, you need phrase-level memory in the effect. The scopes below are rough bins, not hard boundaries — real music bleeds across them.

| Scope | Duration | Musical events (examples) | Features needed | Effect memory | Example effects |
|---|---|---|---|---|---|
| **Frame** | ~11ms (2048 samples @ 44.1kHz) | Individual transients, onset timing | Onset strength, spectral flux, ZCR | None (stateless) | Sparkle trigger, percussive flash |
| **Beat** | 300-600ms | Rhythmic pulse, individual hits | Absint (150ms window), beat predictor | ~1s (cooldown, last beat time) | `impulse`, `bass_pulse`, `band_zone_pulse` |
| **Phrase** | 4-30s | Builds, sections, crescendos, drops | Rolling integral (5s), autocorrelation (5s), onset tempo | 5-30s (ring buffers, running stats) | `impulse_sections`, `longint_sections`, `band_sparkles` |
| **Song** | Minutes | Overall energy arc, narrative shape | Running baseline, adaptation constants | Minutes (slow-decay peaks, reference levels) | Adaptation baselines, per-band normalization |

**The adaptation time constant IS the temporal scope.** When you compute a feature relative to a 5-second running average, you're operating at phrase scope. When you use a 150ms window, you're at beat scope. The musical event you can detect is bounded by the memory you maintain.

**Per-band normalization with dropout handling** (from `per-band-normalization-with-dropout-handling`) bridges song and phrase scope: asymmetric EMA with instant attack / constant decay, absolute dropout threshold, and reference freeze during dropout — so that EDM drops naturally produce massive normalized values when energy returns after 15-30s of silence.

**Current gap: no streaming event detection.** All event detection code (dropout, re-entry, crescendo, climax) lives in `viewer.py` as batch-only visualization. No effect can consume it at runtime. Foote's checkerboard novelty requires an O(n^2) similarity matrix — fundamentally non-streaming. A streaming event detector needs a different approach, likely rolling integral slope (rising = build, falling = breakdown) or per-band dropout detection. This is the prerequisite for section-aware effects. (Ledger: `event-detection-viewer-only`)

### Axis 6: Composition

How effects combine. This is where the system becomes more than the sum of its parts. The modes and patterns below are what we've implemented or borrowed from DAW concepts — other composition strategies exist.

#### Blend modes (implemented so far)

From `Effect.h` (C++) and `band_zone_pulse.py` (Python):

| Mode | Behavior | Use case |
|---|---|---|
| **REPLACE** | Background overwrites buffer | Base layer: nebula, solid color, zone coloring |
| **ADD** | Sum with cap at 255 | Foreground overlay: sparkles, traveling pulses |
| **MULTIPLY** | Per-channel multiply (defined but unused) | Theoretical: masking, shadows |
| **ALPHA** | Opacity blending (defined but unused) | Theoretical: crossfades |

**Critical compositing lesson** (from `pulse-compositing-replace-not-blend`): Pulses must REPLACE the background at their pixels, not blend additively. Additive shifts pulse hue. Per-channel max lets wrong channels win. Correct approach: pulse replaces background, fades to black, then background fades back via per-pixel opacity.

#### Spatial composition

Different effects on different topology zones:
- Bass energy drives the base of the strip; treble drives the tips (frequency-to-position mapping)
- `band_zone_pulse` assigns 5 frequency zones to strip positions
- `three_voices` maps bass foundation (bottom), harmonic body (full tree), percussive flash (peaks)

#### Temporal composition

Different effects in different sections:
- Build sections → anticipation effects (growth, rising intensity)
- Drop sections → full-brightness sustained effects
- Breakdown → ambient/breathe effects
- Currently manual (effect selection at runtime); automatic section detection is a future direction

#### DAW concepts worth stealing

From `daw-concepts-for-leds`:

| DAW concept | LED application | Status |
|---|---|---|
| **Sidechain** | One audio feature suppresses another visual element (bass ducks treble sparkles) | Spark — highest priority to prototype |
| **Sends** | One signal drives multiple destinations at different amounts | Concept only |
| **ADSR envelopes** | Tunable attack/decay/sustain/release per effect parameter | Partially implemented (attack/decay in smoothing) |
| **Wet/dry mix** | Blend audio-reactive with ambient baseline; prevents dead LEDs in silence | Concept only |

**The meta-insight:** Best DAW engineers create relationships between elements, not independent mappings. Audio features should modulate each other's influence on visuals.

#### Current compositor pattern

From `effects.ino`:
```
clearBuffer()
background.render(buffer, REPLACE)
foreground.render(buffer, ADD)
applyBrightness(buffer, GLOBAL_BRIGHTNESS)
applyMinThreshold(buffer, MIN_LED_VALUE)
pushToStrip(buffer)
```

Python runner uses a simpler model: single effect renders full frame, topology mapping applied after.

### Axis 7: Perceptual Mapping

The bridge between audio features and visual parameters. How human perception creates cross-modal correspondences. These mappings are starting points from VJ prior art and our own experiments — novel correspondences are likely waiting to be discovered.

#### Some established correspondences

| Audio | Visual | Direction | Notes |
|---|---|---|---|
| Loudness (RMS) | Brightness | Louder → brighter | Most intuitive mapping; gamma correction (2.2) for perceptual linearity |
| Spectral centroid | Color temperature | Higher centroid → cooler (blue); lower → warmer (red/orange) | VJ prior art confirmed across scottlawsonbc, LedFx, WLED |
| Tempo | Motion speed | Faster tempo → faster crawl/pulse rate | Clean octave multiples acceptable for LEDs (from `no-prior-for-leds`) |
| Spectral richness | Visual density/complexity | More harmonics → more sparkles, wider patterns | Spectral flatness as proxy |
| Frequency band | Spatial position | Bass at base, treble at tips | Standard in VJ systems |
| Onset strength | Flash intensity | Harder hit → brighter flash | Power law scaling (x^0.9 from scottlawsonbc) |

#### Proven constants from VJ prior art

From `research/landscape/RESEARCH_AUDIO_VISUAL_MAPPING.md`:

**Smoothing (asymmetric attack/decay):**
- ExpFilter with separate rise/fall alphas is the core primitive across all successful projects
- scottlawsonbc RGB channel-specific: Red (rise 0.99, decay 0.1), Green (rise 0.3, decay 0.1), Blue (rise 0.1, decay 0.05) — red responds fastest, blue is smoothest
- LedFx: multiplier-based energy scaling with configurable sensitivity

**Normalization:**
- Rolling window peak normalization (preferred) — track recent maximum, scale to it
- Per-frame min-max (anti-pattern — causes flicker)
- Our approach: peak-decay normalization (decay constant 0.998 per frame)

**Spatial smoothing:**
- Gaussian blur along strip (σ = 0.2–4.0) prevents harsh pixel boundaries
- Higher σ for background, lower for sparkles

**Frequency analysis:**
- 24 mel bins (standard), 200-12000 Hz range
- FFT size 512-2048 (we use 2048)
- Power law scaling (x^0.9) for energy-to-brightness
- Gamma correction 2.2 for perceived brightness linearity

**What works (proven across multiple projects):**
- Asymmetric filtering (fast attack, slow decay)
- Mel-scale frequency spacing
- Rolling peak normalization
- Power law brightness scaling
- Spatial smoothing

**What doesn't work:**
- Per-frame normalization (flicker)
- Symmetric smoothing (mushy transients)
- Linear frequency spacing (bass swamps everything)
- Too much smoothing (kills responsiveness)

---

## 2. Shared Vocabulary

Precise definitions for terms used throughout the project so far. Prevents re-deriving definitions each session. New terms get added as the vocabulary evolves.

**Build phases:** Primer, sustained, bridge, drop — four independent phases of energy evolution in music. Bridge has HIGHER RMS than drop. Drop is sustained intensity, not peak intensity. (from `build-taxonomy`)

**Accent effect:** Responds to individual hits. Works for sparse percussion (reggae, electronic). `band_zone_pulse` is the canonical example. (from `accent-vs-groove-effects`)

**Groove effect:** Responds to continuous rhythmic feel. Works for dense percussion (rap, funk). `tempo_pulse` and `rap_pulse` are examples. Needs tempo-locked pulsing, not per-hit reactions. (from `accent-vs-groove-effects`)

**Foreground effect:** Transient, additive. Overlays on top of whatever is already there. Sparkles, flashes, traveling pulses. Uses ADD blend mode.

**Background effect:** Persistent, base layer. Defines the canvas. Nebula, zone coloring, solid color. Uses REPLACE blend mode.

**Standalone effect:** Self-contained, handles its own composition. Band zone pulse manages its own background + foreground internally.

**Derivatives vs absolutes:** Rate-of-change is the signal, not position. Build vs climax have identical RMS but climax brightens 58x faster. Always prefer `d(feature)/dt` over `feature`. (from `derivatives-over-absolutes`)

**Context deviation:** Feelings are relative to surrounding context, not absolute acoustic properties. "Airy" is whatever deviates from the local norm. Use deviation-from-running-average, not fixed thresholds. (from `airiness-context-deviation`)

**The two quality axes:** Audio decomposition quality (can we extract features?) and LED mapping quality (do the LEDs look/feel right?) are independent. WLED proves crude audio + great visuals = great product. (from `two-quality-axes`)

**Flourish ratio:** Off-grid vs on-grid tap ratio. Ambient (>70%), accent (30-70%), groove (<30%). Currently computed from user taps only, not from audio alone. (from `flourish-ratio`)

**Absint (abs-integral):** |d(RMS)/dt| integrated over 150ms window. Measures how much loudness is *changing*, not how loud it is. Normalized via slow-decay peak. The primary beat detection signal for non-dense music.

**Onset envelope:** FFT → mel bands → log energy → diff → half-wave rectify → mean across bands. The right signal for tempo autocorrelation, especially on percussive music where absint fails. (from `onset-envelope-for-tempo`)

---

## 3. The Two Quality Axes

This is the project's guiding principle, formalized as a compass for investment decisions.

```
                    LED Mapping Quality
                    (visual design)
                         ▲
                         │
              Great      │      Great
              analysis,  │      analysis,
              bad LEDs   │      great LEDs
              ─────────  │  ─────────────
              (academic  │  (the goal)
               paper)    │
           ─────────────┼──────────────► Audio Decomposition
                         │                Quality (feature extraction)
              Crude      │      Crude
              analysis,  │      analysis,
              bad LEDs   │      great LEDs
              ─────────  │  ─────────────
              (nothing)  │  (WLED, Fluora)
                         │
```

**WLED Sound Reactive** proves the bottom-right quadrant works: their beat detection is a simple bin threshold — no spectral flux, no tempo tracking, no feeling layer. Their magic is visual effects, not audio analysis. (from `wled-sr-reimplemented`)

**Fluora/PixelAir** confirms the same: polished consumer LED product ($250-500, $221K Indiegogo), "music sync" is basic amplitude/rhythm via phone mic. No spectral decomposition at all.

**Implication:** Don't over-invest in decomposition at the expense of visual design. The cheapest path to better results is often better LED mapping (smoother fades, better color choices, more interesting spatial patterns), not more sophisticated audio analysis.

**But also:** The project's thesis is that the top-right quadrant is strictly better — that understanding music more deeply enables visual responses that crude analysis cannot achieve. Section-aware effects, build anticipation, dropout detection — these require phrase-level understanding. The bet is that this understanding, combined with good visual design, creates something qualitatively different from WLED.

---

## 4. Feature Catalog

Features currently implemented or identified. This table is a snapshot — new features will be added as research progresses.

| Feature | Category | Measures | Real-time? | Temporal Scope | Events it detects | Behaviors it drives |
|---|---|---|---|---|---|---|
| RMS | Standard | Loudness | Yes | Frame | Silence, loud passages | Volume meter, brightness |
| Onset strength | Standard | Spectral flux (one-sided) | Yes | Frame | Transients, percussion | Sparkle triggers, flash |
| Spectral centroid | Standard | Timbral brightness | Yes | Frame | Instrument changes, filter sweeps | Color temperature shift |
| Spectral flatness | Standard | Noise vs tone | Yes | Frame | Noise bursts, tonal sections | Sparkle density |
| ZCR | Standard | High-frequency content | Yes | Frame | Percussion, sibilance | Treble sparkle rate |
| MFCCs (13) | Standard | Timbral fingerprint | Yes | Frame | Instrument/section changes | Similarity detection |
| Chroma (12) | Standard | Pitch class energy | Yes | Frame | Key changes, chord progressions | Color palette selection |
| HPSS harmonic | Standard | Sustained tonal content | Yes | Frame | Chords, pads, vocals | Background color, glow |
| HPSS percussive | Standard | Transient content | Yes | Frame | Drums, clicks, consonants | Sparkle triggers |
| AbsIntegral | Calculus | \|d(RMS)/dt\| over 150ms | Yes | Beat | Beats, energy changes | Pulse, breathe, snake, meter |
| Onset envelope | Calculus | Multi-band onset flux | Yes | Beat | Onsets (better than absint on dense material) | Tempo estimation input |
| Beat prediction | Calculus | Autocorrelation + threshold | Yes | Phrase | Beat timing | Pulse, predicted beats |
| Tempo estimate | Calculus | Onset autocorrelation period | Yes | Phrase | Tempo (octave-ambiguous) | Pulse rate, crawl speed |
| Rolling integral (5s) | Calculus | Sustained energy per band | Untested | Phrase | Drops, builds, sections | Background color, growth |
| Per-band RMS (×5) | Expansion | Energy per frequency zone | Yes | Frame | Band-specific events | Zone coloring, band prop |
| Per-band absint (×5) | Expansion | Energy change per zone | Yes | Beat | Band-specific beats | Zone pulse, band sparkles |
| Per-band onset (×5) | Expansion | Onset per frequency zone | Yes | Frame | Band-specific transients | Zone sparkle, band flux |
| Dominant band (5s vote) | Derived | Which band has most energy | Yes | Phrase | Instrument dominance shifts | Background color selection |
| Autocorrelation confidence | Derived | Periodicity strength | Yes | Phrase | Rhythmic vs arrhythmic sections | Beat prediction enable/disable |

**Dead ends** (features that don't work for our use case):
- Equal-loudness weighting on mastered music — double-corrects because mix engineers already compensate (from `equal-loudness-weighting-dead-end`)
- Absint autocorrelation for tempo on percussive music — symmetric bump-dip destroys timing (from `absint-fails-percussive`)
- Spectral centroid for sparkle position — clusters mid-zone, tree feels underutilized (from `centroid-position-clustering`)
- NMF for band-level transient detection — overkill, less stable in streaming (from `nmf-not-suited-for-sparkles`)

---

## 5. Effect Design Patterns

Not implementations, but recurring patterns we've observed across our effects. Other patterns will emerge — these are the ones identified so far.

### Pattern: Accent Effect

Responds to individual hits with per-event visual reactions.

- **Audio**: Per-band peak detection with cooldowns, HPSS or spectral flux for transient isolation
- **Visual**: Sparkle, flash, or traveling pulse per detected event
- **Topology**: Frequency zones mapped to strip positions
- **Temporal scope**: Beat (individual event detection)
- **Works well for**: Sparse percussion, reggae, electronic, rock
- **Fails on**: Dense percussive material (rap vocals, constant hi-hats) — too many triggers
- **Examples**: `band_zone_pulse`, `band_sparkle_flux`, `impulse_bands`

### Pattern: Groove Effect

Responds to continuous rhythmic feel with tempo-locked oscillation.

- **Audio**: Onset-envelope autocorrelation for tempo; phase-lock to confirmed beats
- **Visual**: Pulsing, breathing, or crawling at estimated tempo
- **Topology**: Whole-strip or zoned, uniform across zones
- **Temporal scope**: Phrase (tempo estimation requires 5s buffer)
- **Works well for**: Dense rhythmic material, rap, funk, house
- **Fails on**: Arrhythmic or rubato sections (autocorrelation confidence drops)
- **Examples**: `tempo_pulse`, `rap_pulse`, `impulse_breathe`

### Pattern: Section Effect

Responds to structural changes over 4-30 second windows.

- **Audio**: Rolling integral, long-horizon RMS, section boundary detection
- **Visual**: Gradual intensity changes, color palette shifts, growth/shrink
- **Topology**: Fibonacci-sized sections, height-mapped zones
- **Temporal scope**: Phrase to song
- **Works well for**: Music with clear sections (verse/chorus, build/drop)
- **Currently limited by**: No automatic section detection yet (rolling integral untested)
- **Examples**: `impulse_sections`, `longint_sections`

### Pattern: Ambient Effect

Responds to slow spectral change with smooth visual evolution.

- **Audio**: Spectral centroid (smoothed), dominant band (5s vote), RMS baseline
- **Visual**: Color temperature drift, zone coloring, gentle brightness modulation
- **Topology**: Full-strip gradients, zone coloring
- **Temporal scope**: Phrase to song
- **Examples**: `band_sparkles` (background component), `basic_sparkles` (non-reactive)

### Pattern: Proportional Mapping

Maps feature value directly to visual parameter without event detection.

- **Audio**: Any continuous feature (absint, RMS, per-band energy)
- **Visual**: Brightness proportional to feature value; no threshold, no binary on/off
- **Key advantage**: False positives stay proportionally dim instead of triggering full flash
- **Examples**: `impulse_glow`, `rms_meter`, `band_prop`

### Pattern: Composite Effect (Background + Foreground)

Layers multiple visual behaviors using the compositor.

- **Background**: Replace blend — zone coloring, nebula, solid color
- **Foreground**: Add blend — sparkles, flashes, traveling pulses
- **Composition**: Background renders first (REPLACE), foreground overlays (ADD)
- **Key lesson**: Foreground pulses must REPLACE background at their pixels, not add, to prevent hue shift
- **Examples**: All C++ static-animations use this pattern; `band_zone_pulse` manages internal layers; `three_voices` has three depth-mapped layers

### Cross-reference: Patterns × Topologies

| Pattern | Linear strip | Diamond (height) | Tree (flat) |
|---|---|---|---|
| Accent | Zone sparkles | Height-mapped sparkles | Full-strip sparkles |
| Groove | Whole-strip pulse | All branches pulse together | Whole-tree pulse |
| Section | Fibonacci sections | Base-to-tip growth | Fibonacci sections |
| Ambient | Color gradient | Height gradient | Color gradient |
| Proportional | Meter (base up) | Height fill | Meter |

---

## 6. Current Assets Inventory

Quick-reference of what exists and where.

### Effects (21 custom + 4 WLED)

**Signal effects** (ScalarSignalEffect — output 0-1 intensity, composable with palette):

| Registry name | File | Description | Default palette |
|---|---|---|---|
| `impulse` | absint_pulse.py | Beat detection via absint, whole-tree pulse with exponential decay | amber |
| `impulse_glow` | absint_proportional.py | Absint mapped directly to brightness, no threshold | amber |
| `impulse_predict` | absint_predictive.py | Absint + autocorrelation for predicted beats | reds |
| `impulse_breathe` | absint_breathe.py | Symmetric fade-on/off from absint | reds |
| `impulse_downbeat` | absint_downbeat.py | Pulse only every 4th detected beat | night_dim |
| `impulse_sections` | absint_sections.py | Fibonacci-sized sections with proportional brightness | fib_orange_purple |
| `longint_sections` | longint_sections.py | 80% long-horizon RMS + 20% bass absint | fib_orange_purple |
| `tempo_pulse` | tempo_pulse.py | Free-running pulse at autocorrelation-estimated tempo | reds |
| `bass_pulse` | bass_pulse.py | Half-wave-rectified spectral flux in bass band | amber |

**Full effects** (AudioReactiveEffect — own RGB rendering):

| Registry name | File | Description |
|---|---|---|
| `band_zone_pulse` | band_zone_pulse.py | Frequency-zoned percussive pulses via streaming HPSS |
| `band_sparkle_flux` | band_sparkle_flux.py | Spectral flux variant for sparkle triggers |
| `band_sparkles` | band_sparkles.py | Twinkles over dominant-band-shifted base color |
| `band_tempo_sparkles` | band_tempo_sparkles.py | Wide sparkles triggered by predicted beats |
| `basic_sparkles` | basic_sparkles.py | Non-reactive dim red-magenta twinkles |
| `rms_meter` | rms_meter.py | Simple volume meter |
| `impulse_meter` | absint_meter.py | Absint-driven volume meter |
| `impulse_bands` | absint_band_pulse.py | Multi-band beat detection → traveling Gaussian pulses |
| `impulse_snake` | absint_snake.py | Beats spawn traveling pulses scaled by strength |
| `band_prop` | band_prop.py | RGB channels = bass/mid/high absint |
| `hpss_voices` | three_voices.py | Three depth-mapped layers via streaming HPSS |
| `rap_pulse` | rap_pulse.py | Slow background pulse at 1/6th tempo |

**WLED Sound Reactive reimplementations** (in `effects/wled_sr/`, no registry_name):

| Class | File | Notes |
|---|---|---|
| WLEDVolumeReactive | volume_reactive.py | Simple bin threshold |
| WLEDFrequencyReactive | frequency_reactive.py | Frequency-to-color mapping |
| WLEDBeatReactive | beat_reactive.py | Beat detection via amplitude |
| WLEDAllEffects | all_effects.py | Combined WLED effects |

### Topologies (4 sculptures)

Defined in `hardware/sculptures.json`. See Axis 1 above.

### Datasets

| Dataset | Size | What it provides | Access | Ledger audit |
|---|---|---|---|---|
| **Harmonix Set** | 912 tracks | Beats, downbeats, segments (hip-hop, dance, rock, metal) | YouTube URLs + annotations (fragile) | `harmonix-set-audit` |
| **Yadati EDM** | 402 CC tracks | Drop/build/break annotations | OSF bundle (download and go) | `yadati-edm-drops-audit` |
| **SALAMI** | 1,300+ tracks | Hierarchical structure (coarse/fine/functional), dual annotator | Annotations only, multiple audio sources (hard) | `salami-dataset-audit` |
| **GiantSteps** | 664 tempo / 604 key | EDM tempo and key annotations | Beatport previews via scripts | `giantsteps-tempo-key-audit` |
| **FMA** | 8,000+ tracks | Genre labels, audio features | Direct download | Landscape doc only |
| **User annotations** | ~10 segments | Semantic annotations (builds, drops, airiness, taps) | Local (`research/audio-segments/`) | Various |

**Cross-cutting finding** (from `mir-dataset-landscape-gap`): No MIR dataset has both redistributable audio AND rigorous multi-annotator methodology. Every dataset requires understanding its specific failure modes before treating it as ground truth.

### Signal Processing Primitives

In `effects/signals.py`:
- `OverlapFrameAccumulator` — feed audio chunks, yield overlapped frames
- `AbsIntegral` — normalized abs-integral of RMS derivative
- `BeatPredictor` — autocorrelation tempo + predicted/confirmed beats
- `OnsetTempoTracker` — onset-envelope autocorrelation for tempo estimation

In `effects/feature_computer.py`:
- `FeatureComputer` — thread-safe computation of abs_integral, RMS, spectral centroid, autocorrelation confidence

### Research Documents

| Document | Location | Contents |
|---|---|---|
| Research ledger | `research/ledger.yaml` | All findings, 100+ entries |
| Ledger guide | `research/LEDGER_GUIDE.md` | Format and conventions |
| This document | `synchronicity-led/ARCHITECTURE.md` | Design space framework |
| Algorithm specs | `synchronicity-led/algorithms/` | Formal algorithm specifications |
| Reference library | `synchronicity-led/library/` | Research summaries, guides (see INDEX.md) |
| Audio-visual mapping | `research/landscape/RESEARCH_AUDIO_VISUAL_MAPPING.md` | VJ prior art, constants, code snippets |
| Audio analysis landscape | `research/landscape/RESEARCH_AUDIO_ANALYSIS.md` | MIR techniques survey |
| ESP32 audio DSP | `research/landscape/ESP32_AUDIO_DSP.md` | Hardware constraints |
| Existing datasets | `research/landscape/EXISTING_DATASETS.md` | Dataset survey |
| Research findings | `research/landscape/RESEARCH_FINDINGS.md` | Early findings summary |
