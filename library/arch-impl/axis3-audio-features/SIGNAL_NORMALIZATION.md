# Signal Normalization Primitive

How raw audio features become usable effect inputs. A single normalization primitive exposes multiple outputs tuned to different consumer needs.

**Relationship to PER_BAND_NORMALIZATION.md (library, Axis 6):** That document covers per-band peak-decay as a composition concern -- putting all frequency bands on a common 0-1 scale. This document covers the normalization primitive itself: why EMA (mean-tracking) supersedes peak-decay (max-tracking) for section-aware work, and how one primitive serves multiple consumers through different output taps.

---

## 1. The Problem with Peak-Decay

Peak-decay normalization (`peak = max(energy, peak * decay)`) is industry standard and works well for beat-level visualization. But it has a structural limitation: it tracks the *maximum*, so within any stable section the output converges toward binary 0/100% as the peak locks to the ceiling.

Concrete evidence: 10-minute Chase & Status DnB set with peak-decay at 0.9995 (~46s half-life) -- ceiling and floor converge within a minute, erasing the multi-minute energy arc entirely. The listener hears builds, plateaus, and drops spanning minutes that the 46s window adapts away. (Ledger: `perception-is-layered-by-timescale`)

Max-tracking erases section dynamics because the reference ratchets to the loudest moment and only slowly decays. Mean-tracking (EMA) preserves section dynamics because the reference sits at the *center* of the signal, not the ceiling.

---

## 2. The Multi-Output Primitive

One EMA tracker, three output taps. Each tap serves a different consumer profile.

```
raw energy ──┬── .peak_normalized   (0-1, legacy peak-decay)
             │
             ├── .ema_ratio         (unbounded, centered ~1.0)
             │
             └── .sigmoid           (0-1, centered ~0.5)
                    ▲
                    │
               EMA tracker
          (time constant in seconds)
```

### `.peak_normalized` -- Legacy Peak-Decay

Standard peak-decay: `peak = max(energy, peak * decay); output = energy / peak`.

- **Range:** 0 to 1.0
- **Character:** Binary at steady state, responsive to new peaks
- **Use:** Beat-level triggers, backward compatibility with existing effects

### `.ema_ratio` -- Raw EMA Ratio

`output = energy / ema` where EMA tracks the running mean.

- **Range:** Unbounded, centered around 1.0
- **Character:** >1 means "above recent average," <1 means "below." Quiet-to-loud transitions produce large spikes (3x-5x). Preserves section-level contrast that peak-decay erases.
- **Use:** Section boundary detection (threshold on ratio magnitude), routing decisions, trajectory/novelty signals. This is the signal that tells you "something changed" at the section level.

### `.sigmoid` -- Sigmoid-Compressed EMA Ratio

`output = ratio / (ratio + 1.0)` — the simplest sigmoid that maps 0→0, 1→0.5, ∞→1.

This is equivalent to the logistic `1 / (1 + exp(-k * (ratio - 1)))` with k=1, but cheaper (no exp). Use this form unless a specific reason requires tunable steepness.

- **Range:** 0 to 1.0, centered at 0.5 when ratio = 1.0
- **Character:** Proportional response with soft limits. A ratio of 1.0 (at average) maps to 0.5. Above-average maps toward 1.0, below-average toward 0.0, with smooth saturation at extremes. A strong beat at 2x the section mean maps to 0.67; at 3x, 0.75.
- **Use:** Proportional effects (brightness, speed, density) that need bounded output but must preserve section dynamics. Replaces peak-decay for any continuous mapping where 0-1 range matters.

---

## 3. Why EMA Preserves Section Dynamics

The core insight: mean-tracking and max-tracking have fundamentally different steady-state behavior.

| Property | Peak-Decay (max-tracking) | EMA (mean-tracking) |
|---|---|---|
| Reference tracks | Recent maximum | Running mean |
| Steady-state output | Converges to 0 or 1 (binary) | Fluctuates around 1.0 (preserves variance) |
| Section transition | Slow recovery (decay from old peak) | Immediate ratio shift (new mean vs old) |
| Information preserved | "How close to recent peak" | "How different from recent average" |

This aligns with the context-deviation principle: what matters is deviation from context, not absolute position or distance from a maximum. (Ledger: `feelings-are-contextual`)

---

## 4. Time Constant Design

### Wall-Clock Seconds, Not Per-Frame Decay

Time constants are specified in wall-clock seconds. The per-frame alpha is derived:

```
alpha = 2.0 / (time_constant_seconds * fps + 1)
```

This ensures frame-rate independence -- the same perceptual behavior at 30fps, 44fps, or any other rate. Per-frame decay factors (0.9995, 0.998, etc.) silently change behavior when frame rate changes.

### Choosing the Time Constant

The adaptation time constant IS the temporal scope (see Axis 5 in ARCHITECTURE.md). Rule of thumb: set the time constant to ~0.5-0.7x the section length you want to preserve.

| Time constant | Preserves dynamics at | Adapts away |
|---|---|---|
| 2-5s | Phrase level | Beat-to-beat variation |
| 10-20s | Section level | Phrase-level variation |
| 30-60s | Song level | Section-level variation |

For section-aware normalization, 10-20s is the primary operating range. Shorter constants for beat-responsive effects; longer constants risk the same convergence problem as peak-decay.

---

## 5. Feature-Specific Normalization

Different audio features have different statistical distributions and need different normalization strategies. One size does not fit all.

| Feature type | Examples | Recommended normalization | Rationale |
|---|---|---|---|
| **Energy / amplitude** | RMS, band energy, absint | EMA ratio + sigmoid | Unbounded, heavy-tailed; EMA centers the distribution |
| **Derivative / rate-of-change** | d(RMS)/dt, spectral flux | EMA ratio + sigmoid | Same reasoning as energy; derivatives amplify variance |
| **Pre-integrated** | Rolling integral, sustained energy | Self-relative (value / own recent range) | Already accumulated; further averaging over-smooths |
| **Position / frequency** | Spectral centroid, dominant band | Adaptive range (local min-max) | Bounded by nature; the interesting signal is *where* within the range |
| **Binary / event** | Onset detection, beat trigger | No normalization (threshold only) | Already discrete; normalization destroys the binary character |

This is the practical consequence of "perception is layered by timescale" -- each feature type carries information at a different scale, and normalization must respect that. (Ledger: `perception-is-layered-by-timescale`)

---

## 6. EMA Ratio as Section Boundary Detector

The `.ema_ratio` output doubles as a section boundary signal. When the music transitions from quiet to loud (or vice versa), the ratio produces a transient spike or dip before the EMA catches up. The magnitude of the spike is proportional to the contrast between sections.

This means section detection does not require a separate algorithm -- it falls out of the normalization primitive. A simple threshold on `abs(ema_ratio - 1.0)` flags boundaries. The same signal drives both continuous normalization and discrete event detection.

This connects normalization to the events layer without conflating them -- the normalization primitive produces a clean signal, and downstream consumers decide what constitutes a "boundary." (Ledger: `normalization-events-separation`)

---

## 7. ESP32 Feasibility

Per feature, per frame:
- **EMA update:** 1 multiply, 1 multiply, 1 add (3 FLOPs)
- **Ratio:** 1 divide
- **Sigmoid:** 1 subtract, 1 multiply, 1 exp, 1 add, 1 divide

Total: ~8 FLOPs per feature per frame. For 5 bands at 30fps, that is ~1200 FLOPs/second -- negligible on ESP32. Memory: 2 floats per feature (EMA state + peak state for legacy output).

---

## 8. Threshold Effects: Split Detection from Brightness

The codebase has five effects that use thresholds: `absint_pulse`, `absint_downbeat`, `absint_band_pulse`, `bass_pulse`, and `absint_predictive` (BeatPredictor). An audit of all five found that none are pure binary detectors. Each uses the threshold for onset *detection* but also uses the normalized signal value for *brightness* or *strength* after the trigger fires. They are hybrid: discrete trigger, continuous response.

This means all five should migrate to EMA normalization alongside the proportional effects. The threshold behavior and the brightness behavior simply consume different output taps from the same primitive.

### 8.1 The Split Signal Path

After migration, each threshold effect reads two outputs from the normalization primitive:

- **Detection path:** Use raw `.ema_ratio` with the threshold reframed as a multiplier of the local mean. A threshold of 1.4 means "fire when instantaneous energy is 40% above the section mean." This replaces the old peak-decay threshold (e.g., 0.30 on a 0-1 scale) with a semantically equivalent condition on the EMA ratio.
- **Brightness path:** Use `.sigmoid` for the brightness/strength value after detection. This gives bounded 0-1 output with mid-range centering, so the visual intensity tracks how *much* above the threshold the signal is -- not just that it crossed.

### 8.2 Threshold Recalibration

Old peak-decay thresholds do not transfer directly to EMA ratios. The table below provides starting points for recalibration, derived from the statistical relationship between the two normalizations.

| Effect | Current threshold (peak-decay) | EMA ratio starting point | Meaning |
|---|---|---|---|
| `absint_pulse` | 0.30 | 1.4 | 40% above section mean |
| `absint_downbeat` | 0.30 | 1.4 | same |
| `absint_band_pulse` | 0.30 per band | 1.4 | same |
| `bass_pulse` | 0.55 | 1.6 | higher bar (bass was already stricter) |
| `absint_predictive` (BeatPredictor) | 0.30 | 1.4 | same |

These are *starting points*. Perceptual validation (eyes on LEDs) is required before committing final threshold values. The old thresholds emerged from tuning against peak-decay output; EMA ratios have different distributions, and the subjective feel may shift even when the statistical mapping is correct.

### 8.3 Pattern in Pseudo-Code

```python
raw_ratio = self.absint.update_ema(frame)  # value / ema, centered ~1.0
if raw_ratio > self.threshold and time_since_beat > self.cooldown:  # threshold ~1.4
    self.brightness = raw_ratio / (raw_ratio + 1.0)  # sigmoid for brightness
```

The detection branch reads `.ema_ratio` directly. The brightness branch applies the simplified sigmoid `ratio / (ratio + 1)` to compress the unbounded ratio into a 0-1 range centered at 0.5. This ensures that a beat detected at 1.4x the mean produces noticeably less brightness than a beat detected at 2.0x -- preserving dynamic contrast within the triggered response.

### 8.4 Refactoring Note

`absint_band_pulse` currently performs inline normalization rather than using the shared AbsIntegral signal processor. It should be refactored to use the shared class so that all threshold effects consume the same multi-output primitive and benefit from consistent time-constant management.

---

## 9. Related Ledger Entries

| Entry | Relevance |
|---|---|
| `perception-is-layered-by-timescale` | Foundation: each timescale needs its own normalization |
| `feelings-are-contextual` | Foundation: deviation from context is the signal |
| `normalization-events-separation` | Normalization and event detection are separate layers |
| `peak-decay-demand-charge-analogy` | Why peak-decay converges (demand charge ratchet) |
| `normalization-prior-art-audit` | Peak-decay matches industry standard (which this extends) |
| `per-band-normalization-with-dropout-handling` | Historical: original combined design, now split |
| `proportional-mapping-needs-midrange-centering` | Why brightness path needs mid-range centering (equal headroom above and below) |
