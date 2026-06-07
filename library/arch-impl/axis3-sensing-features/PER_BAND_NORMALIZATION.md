# Per-Band EMA Normalization

Normalize each frequency band's energy against its own running mean. The validated direction for this project's "capture musical feeling across full songs and DJ sets" use case — preserves both within-section dynamics and inter-section contrast where peak-decay erases them.

**Status:** Implemented as `PerBandEMANormalize` in `audio-reactive/effects/signals.py`. Used by `EnergyWaterfall3BandEffect` and other band-decomposition RMS amplitude effects.

---

## 1. Problem

### Why per-band normalization?

Raw bass energy is ~100x treble energy. Without normalization, bass always dominates. Normalizing each band against its own recent history puts all bands on a 0-1 scale where 1.0 = "loud for this band right now" — what matters is deviation from the band's own recent context, not absolute energy levels (ledger: `feelings-are-contextual`).

### Why not perceptual weighting?

Equal-loudness curves (A/C-weighting, ISO 226) applied to mastered music **double-correct** — mix engineers already compensate for human hearing sensitivity (ledger: `equal-loudness-weighting-dead-end`).

### Why EMA over peak-decay?

Peak-decay normalization tracks the loudest moment in recent memory and divides by it. Loud sections set a high reference; subsequent quiet sections get crushed toward zero (or take ~1 minute to recover). Section-to-section dynamics are erased.

EMA-ratio normalization divides by a *running mean* rather than a peak. The mean tracks the current section's energy context (ledger: `ema-normalization-replaces-peak-decay`):

- **Within a section**: beat hits register as deviations above the local mean → bright pulses on a dim background.
- **Drop**: energy spikes well above the still-low EMA → output spikes above 1.0 momentarily before EMA catches up.
- **Breakdown**: energy falls below the still-high EMA → output dips near zero before EMA catches up.

These transient mismatches mirror human auditory adaptation (ledger: `ema-section-boundary-contrast`).

---

## 2. Algorithm

```python
# Per band, per frame

def update(rms):
    ema += alpha * (rms - ema)              # exponential moving average
    if rms < noise_floor_rms:
        return 0.0                          # noise gate (see §4)
    if ema < EPS:
        return 0.0                          # no signal yet
    ratio = rms / ema
    return clip(ratio / max_ratio, 0.0, 1.0)
```

Three knobs:

- `ema_tc` — time constant in seconds (see §3)
- `max_ratio` — ratio that maps to 1.0 (default 2.0 → "twice the running mean = full")
- `noise_floor_rms` — absolute RMS below which output is gated to 0 (see §4)

The EMA always adapts to raw RMS (including during noise-gated periods) so the running mean tracks the actual ambient floor. Only the *output* is gated.

---

## 3. Time Constant Selection

The EMA time constant should be approximately 0.5-0.7× the typical section length of the music (ledger: `ema-time-constant-is-section-relative`):

| Music | Sections | Recommended TC |
|---|---|---|
| EDM | ~30s | **20s** |
| Rock / pop | ~60s | 30-40s |
| DJ sets | longer | 60-90s |
| Classical | very long | 120s+ |

Default: **30s** (EDM-leaning, works for most modern music). Same TC across all bands — bass and treble share the section structure of the music.

---

## 4. Noise Floor Gate

Per-band normalization on a noise-dominated band amplifies hiss to full scale. This is real for INMP441 deployments:

- Bass band at 3 ft from speaker: -30 dB vs system audio (ledger: `inmp441-frequency-response`)
- Treble band always: -36 to -43 dB (`inmp441-validation-2026-04-26`)

Per `inmp441-normalization-limits`: *normalization cannot fix the frequency imbalance — bass after per-band normalization is noise, not music*. The honest framing — gating doesn't recover signal that's actually in noise; it just makes the silence honest instead of hallucinating beats from hiss.

### Mechanism

Hard absolute-RMS gate: `if raw_rms < noise_floor_rms: output = 0`. Cheap, deterministic, per-band configurable.

### Calibration

- **System audio** (BlackHole loopback): noise near zero → set `noise_floor_rms` ~ 1e-5 (effectively off).
- **INMP441**: per-band thresholds depend on room. Measure during silence, set 3-6 dB above the silent-RMS reading. Test vectors at `library/test-vectors/inmp441-validation/` provide reference recordings.

Adaptive SNR estimation (short-vs-long energy ratio per band, continuous gate) is a future refinement.

---

## 5. Expected Behavior

### Steady beat (techno kick)

EMA tracks beat-averaged energy. Each kick spikes above EMA → output 0.7-1.0. Between kicks, output falls as EMA catches up partially. Pattern is dynamic, not flat. Peak-decay would give near-1.0 every beat (no breathing); EMA gives a kick-and-recover envelope.

### Loud section → quiet section

EMA still high from loud section. Quiet section's RMS / high EMA → output near zero. Over ~30s the EMA decays to match, output gradually rises. *Perceptually correct*: the quiet section *should* feel dim relative to what came before.

### Quiet section → loud section (drop)

EMA still low. Loud section's RMS / low EMA → output spikes above 1.0 (clipped at `max_ratio`). The drop announces itself before settling.

### Band dropout

Bass disappears: EMA decays slowly. Bass output near zero. Treble unaffected (independent normalization). Bass returns: if EMA still partially adapted, moderate output; if fully decayed, output spikes briefly.

---

## 6. ESP32 Feasibility

- Memory: 1 float per band for EMA + 1 for floor. ~24 bytes for 3 bands.
- Compute: 1 multiply, 1 add, 1 compare, 1 divide per band per frame.
- Trivially within budget.

---

## 7. What This Primitive Does NOT Do

Per `normalization-events-separation`, the following belong downstream, not in normalization:

- **Dropout reintroduction emphasis** — events layer
- **Build / drop / section detection** — temporal layer
- **Discrete event emission** — downstream consumers (e.g., `PulseDriver`)

The normalization layer's job is to produce clean 0-1 values per band. What the system does with dramatic transitions in those values is a separate design decision.

---

## 8. Composability

Per-band EMA normalization is the foundation. The standard pipeline for "band decomposition RMS amplitude effects":

```
FFT → per-band RMS → PerBandEMANormalize → continuous brightness
                              ↓
                     PerBandAbsIntegral → PulseDriver → discrete pulses
```

Effects can tap any layer:

- **Continuous proportional** (waterfall, sparkles): use the EMA-normalized output directly.
- **Pulse-driven** (worley, accent flashes): use `PulseDriver` events.
- **Hybrid** (waterfall + transient overlay): both — continuous brightness with pulse-emission overlay in the dominant band's color.

---

## 9. Prior Art

| System | Approach |
|--------|----------|
| **scottlawsonbc** | Per-filterbank peak-decay (α=0.001) |
| **band_sparkles.py** (legacy) | Per-band peak-decay (0.9995) |
| **WLED SR** | Global AGC (not per-band) |
| **PCEN** | Per-channel adaptive normalization (ML domain) |

Per-band peak-decay is the LED-VJ industry standard. We diverged to EMA-ratio because the loudest-moment-as-reference behavior of peak-decay erases the section-to-section dynamics that the project cares about (ledger: `normalization-prior-art-audit`, `ema-normalization-replaces-peak-decay`).

---

## 10. Related Ledger Entries

| Entry | Relevance |
|-------|-----------|
| `ema-normalization-replaces-peak-decay` | Core motivation for moving from peak-decay to EMA |
| `ema-time-constant-is-section-relative` | TC tuning per genre/section length |
| `ema-section-boundary-contrast` | Why drops/breakdowns feel right with EMA |
| `inmp441-normalization-limits` | Noise floor amplification problem |
| `inmp441-signal-impact` | Per-band decomposition limitation on INMP441 |
| `inmp441-frequency-response` | Distance-dependent bass loss; treble in noise |
| `feelings-are-contextual` | Philosophical foundation: feelings = context deviation |
| `normalization-prior-art-audit` | Industry standard is peak-decay (we diverge) |
| `normalization-events-separation` | What belongs downstream, not in normalization |
| `equal-loudness-weighting-dead-end` | Why we rejected perceptual weighting curves |
