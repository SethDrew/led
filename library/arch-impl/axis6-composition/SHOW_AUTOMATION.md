# Automated Show Architecture

How to run a multi-hour LED show without a human VJ. Extracted from the original AUTOMATED-LIGHTSHOW.md design document; the visual idiom and input role sections now live in their respective library docs.

---

## What VJs Actually Do

VJs make three categories of real-time decisions:

1. **Effect Selection** — which visual runs right now (based on musical section)
2. **Parameter Modulation** — tweaking intensity, speed, color, complexity (continuous)
3. **Transitions** — crossfading between states at musical boundaries (section changes)

The key insight: VJs **prepare** the structure (asset library, color palettes, energy arc) and **improvise** the details (which exact effect, which parameter tweaks, crowd response).

For automated LED sculpture shows, the analogy is:
- **Prepare**: effect library, palette sets, section->effect mappings, energy curves
- **Automate**: real-time section detection, parameter modulation from audio, transition logic
- **Manual override**: human can intervene via controller for taste decisions

## Detection Stack

What needs to be detected, in order of feasibility and impact:

### Layer 1: Frame-Level (Already Have)

| Detection | Status | Source |
|---|---|---|
| RMS / loudness | Implemented | `signals.py`, `feature_computer.py` |
| Onset / transient | Implemented | Per-band spectral flux |
| Spectral centroid | Implemented | `feature_computer.py` |
| Per-band energy | Implemented | 5-band mel decomposition |
| Absint (rate of change) | Implemented | `signals.py:AbsIntegral` |

### Layer 2: Beat-Level (Partially Have)

| Detection | Status | Gap |
|---|---|---|
| Tempo (BPM) | Implemented | Octave-ambiguous |
| Beat prediction | Implemented | `signals.py:BeatPredictor` |
| Onset envelope | Implemented | `signals.py:OnsetTempoTracker` |
| Downbeat detection | Partial | Every-4th heuristic, not true downbeat |
| Vocal presence | Not implemented | Could use spectral shape heuristics or ML |

### Layer 3: Phrase-Level (The Critical Gap)

This is where automated show control lives — and it's largely unimplemented.

| Detection | Description | Approach | Difficulty |
|---|---|---|---|
| **Build detection** | Rising energy over 4-16 bars | Rolling integral slope > threshold for N seconds | Medium — rolling integral exists but untested |
| **Drop detection** | Sudden sustained intensity after build | RMS jump after build phase, bass energy spike | Medium — combine build detection + RMS derivative |
| **Breakdown detection** | Energy withdrawal, sparser texture | Rolling integral slope < 0, spectral flatness drops | Medium |
| **Section boundary** | Any structural change | Self-similarity novelty, spectral flux over 2-4s window | Hard — Foote's method is O(n^2), needs streaming alternative |
| **Riser detection** | Sweeping frequency buildup | Rising spectral centroid + increasing RMS over 2-8s | Medium |
| **Dropout detection** | Band goes silent then returns | Per-band energy drops to near-zero, reintroduction | Easy — per-band normalization already handles this |

**Priority**: Build detection -> drop detection -> breakdown -> section boundary. These four cover 80% of automated show control decisions.

**Streaming Section Detection Alternatives** (Foote's is non-streaming):
- Rolling integral slope: positive = build, negative = breakdown, near-zero = sustained
- Per-band onset density: sparse onsets = breakdown, dense = groove
- Spectral flux variance: high variance = transition, low = stable section
- **Timbral moment-picking (one validated approach):** Frame-vs-EMA distance on raw MFCCs 0-12 with cubic eagerness curve (N decays 3σ→1σ over 120s). Picks artistically good audio moments to switch visuals. The eagerness models human visual fatigue — after ~2 minutes of the same look, the detector gets more willing to switch. Cheap and streaming. There are many ways to approach this problem — this is one that works well for our use case. (Ledger: `timbral-shift-detection-eagerness`)
- **Timbral section drift (one exploring approach):** Dual-EMA (5s fast + 45s slow) on L2-normalized MFCCs detects gradual timbral evolution (ambient, dark electronic) that frame-level misses. Better for identifying THAT the section changed. Also just one of many possible approaches. (Ledger: `dual-ema-section-level-detection`)

### Layer 4: Song-Level (Future / Optional)

| Detection | Description | Use Case | Approach |
|---|---|---|---|
| **Song transition** | One track fading into another (DJ context) | Reset adaptation, change palette | Detect simultaneous energy in different keys, or BPM instability |
| **Genre/mood shift** | Overall energy character changes | Select effect family | Sliding window over spectral shape, onset density, tempo stability |
| **Energy arc** | Set-wide intensity trajectory | Prevent visual fatigue, manage intensity over hours | Very long rolling average (minutes), trend detection |
| **Key detection** | Musical key of current section | Color palette selection (major=warm, minor=cool) | Chroma analysis — feasible but low priority |

## Show Controller Architecture

```
                     +-----------------------------+
                     |      Show Controller         |
                     |                               |
  Audio Features --> |  Section Detector             |
  (frame-level)      |    +-- Build/Drop/Break state  |
                     |    +-- Section boundaries       |
                     |                               |
                     |  Role Binder                   | <-- binds input roles to sources
                     |    +-- Audio feature -> role    |
                     |    +-- User knob -> role         |
                     |    +-- ML model -> role           |
                     |                               |
                     |  Effect Selector               | --> Active Effect
                     |    +-- Section -> effect map     |
                     |    +-- Transition logic          |
                     |    +-- Fatigue prevention        |
                     |                               |
                     |  Parameter Modulator           | --> Effect params (via roles)
                     |    +-- Role -> visual mapping   |
                     |    +-- Energy curve tracking     |
                     |                               |
                     |  Palette Manager               | --> Color palette
                     |    +-- Mood -> palette map        |
                     |    +-- Rotation timer             |
                     |                               |
                     +-----------------------------+
```

## Section -> Effect Mapping (Starting Point)

Not prescriptive — this is a default mapping that a human would override for taste. The idea is that it works *reasonably* without intervention.

| Detected Section | Effect Family | Intensity | Palette Temperature | Speed |
|---|---|---|---|---|
| **Intro / Low Energy** | Ambient (noise field, bioluminescence, breathing) | Low | Warm/cool neutral | Slow |
| **Verse / Groove** | Groove (tempo-locked pulse, flowing noise) | Medium | Warm | Medium, locked to tempo |
| **Build** | Convergence + rising complexity | Rising | Shifts warmer | Accelerating |
| **Drop** | Full-intensity (root pulse, fire, electrical storm) | High, sustained | Hot (reds, whites) | Fast |
| **Breakdown** | Ambient + sparkle (embers, fireflies, bioluminescence) | Low-medium, falling | Cool (blues, purples) | Slow, decoupled from tempo |
| **Climax / Peak** | Maximum complexity (composite: background + sparkle + pulse) | Maximum | White-hot | Maximum |
| **Outro / Fade** | Retreat (reverse growth, cooling embers, fading noise) | Decreasing to zero | Cooling | Decelerating |

## Preventing Visual Fatigue

For multi-hour automated shows, the system needs strategies to stay interesting:

1. **Effect Rotation** — don't use the same effect for more than N consecutive sections of the same type. After 3 drops with Fire, switch to Root Pulse or Electrical Storm.
2. **Palette Cycling** — can be timer-based (rotate through palette families every 10-20 minutes) or audio-driven. One approach that works: the timbral moment-picking detector's eagerness curve naturally increases willingness to switch as time passes, picking musically appropriate moments. (Ledger: `timbral-shift-detection-eagerness`)
3. **Complexity Curve** — track a long-horizon "visual complexity" metric. If it's been high for too long, force a simpler section. Mirror the concept of dynamic range in mastering.
4. **Spatial Rotation** — alternate between topology modes (full-strip, zoned, height-mapped, branch-independent) to keep the spatial experience fresh.
5. **Novelty Injection** — periodically introduce a rare effect (frost crystallization, reaction-diffusion) that hasn't been used recently. The surprise factor prevents habituation.

## What SoundSwitch / MaestroDMX Claim (and What's Actually True)

Commercial automated lighting systems market phrase detection:

- **SoundSwitch**: Phrase detection (intro/verse/chorus/bridge/drop/outro). Ships with 32 pre-built "Autoloops." But this is **pre-analysis** — tracks must be analyzed before playback. Not real-time.
- **MaestroDMX**: Claims real-time "autonomous lighting designer." The only commercial tool claiming streaming section detection, but algorithm is proprietary and unverifiable. User reviews describe it as "not much more sophisticated than enhanced sound-to-light."
- **Engine Lighting**: "Industry-leading automated phrase detection" — also pre-analysis, not streaming.

**The honest assessment**: Real-time phrase/section detection is **not solved**. Beat/downbeat tracking is solved (BeatNet, BEAST achieve >80% accuracy with <50ms latency). But semantic section detection (verse/chorus/build/drop) relies on self-similarity matrices that require the **full song** — fundamentally non-streaming. No published causal algorithm matches offline methods.

**What exists for real-time**: Simple heuristics — rolling integral slope, kick dropout detection, onset density gradients, spectral centroid derivative. These are crude but causal and fast (<1ms). They won't generalize across genres without tuning, and even for our use case they will likely need significant R&D to feel right. This is an active research area for this project, not a solved problem we can import.

**Song-level energy arc** (10+ minute trajectory) is actually easier — long timescales are forgiving, and multi-scale rolling averages work well. This may be more important than phrase-level detection for our use case.

The "two quality axes" framework still applies: detection quality matters, but mapping quality is what people see. We should improve both in parallel.

## Implementation Phases

**Phase 1: Streaming Section Detector**
- Implement rolling integral slope as build/drop/breakdown detector
- Add per-band dropout detection (already have normalization)
- Expose detected section as state in runner.py
- Test against annotated audio segments

**Phase 2: Effect Library Expansion**
- Implement 4-6 new effects from the Tier 1/2 priority list
- Each effect declares its input roles (not specific audio features)
- Build as composable layers (background + foreground pattern)

**Phase 3: Show Controller with Role Binding**
- Section -> effect mapping with configurable defaults
- Role binder: maps audio features, user knobs, and other sources to effect input roles
- Transition logic (crossfade, cut, morph between effects)
- Palette manager with rotation
- Fatigue prevention heuristics

**Phase 4: Spatial Rendering**
- Add 3D coordinates to `sculptures.json` for tree and diamond
- Implement world-space rendering path in effect base class
- Upgrade noise field, ripple, and metaball effects to use 3D coordinates
- Effects fall back to topology space when 3D coordinates are unavailable

**Phase 5: Multi-Hour Autonomy**
- Energy arc tracking over minutes
- Effect rotation memory (don't repeat too soon)
- Long-horizon adaptation (overall set intensity curve)
- Manual override via controller input
