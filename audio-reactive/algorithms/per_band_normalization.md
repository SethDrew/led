# Per-Band Peak-Decay Normalization

Normalize each frequency band's energy against its own recent peak. Industry-standard approach used across VJ and LED visualization systems.

**Status:** Implemented in existing effects (e.g., `band_sparkles.py`), not yet unified as shared primitive.

---

## 1. Problem

### Why not perceptual weighting?

Equal-loudness curves (A/C-weighting, ISO 226) applied to mastered music **double-correct** — mix engineers already compensate for human hearing sensitivity. No phon level produces balanced band energy from professionally mixed music (ledger: `equal-loudness-weighting-dead-end`).

- A-weighting (40 phon): eliminates sub-bass entirely
- C-weighting: nearly flat (useless for band discrimination)
- ISO 226 at concert levels (100-115 phon): over-attenuates bass

### Why per-band normalization?

Raw bass energy is ~100x treble energy. Without normalization, bass always dominates. Normalizing each band against its own recent history puts all bands on a 0-1 scale where 1.0 = "loud for this band right now."

This matches the "context deviation" principle — what matters is deviation from a band's own recent context, not absolute energy levels (ledger: `context-deviation`, originally `airiness-context-deviation`).

---

## 2. Algorithm

Peak-decay normalization per band. Instant attack, slow exponential decay.

```python
# Per band, per frame (~30 fps)

def normalize_band(band, energy):
    # Instant attack: new peaks immediately adopted
    # Slow decay: reference drifts down over ~1 minute
    peak[band] = max(energy, peak[band] * DECAY)

    # Normalize: 0 = silence, 1.0 = recent peak
    if peak[band] > FLOOR:
        return energy / peak[band]
    else:
        return 0.0
```

That's the entire algorithm. No special cases, no freeze, no asymmetric EMA. This matches scottlawsonbc and `band_sparkles.py`.

---

## 3. Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Decay factor | 0.9995 per frame | ~46s half-life → ~1 minute effective memory |
| Floor | 1e-10 | Prevents division by zero; not a silence detector |
| Bands | 5 (sub-bass, bass, low-mid, high-mid, treble) | Standard band definitions from `band_zone_pulse.py` |

### Decay factor = reference window

The decay factor determines how far back the reference "remembers." No separate ring buffer needed — the exponential decay IS the window.

| Decay/frame (30fps) | Half-life | Effective memory |
|---|---|---|
| 0.998 | ~12s | Short, responsive |
| 0.999 | ~23s | Half minute |
| **0.9995** | **~46s** | **~1 minute** |
| 0.9997 | ~77s | ~1.3 minutes |

At 0.9995, a peak from 1 minute ago has decayed to ~25% of its original value. A peak from 2 minutes ago is at ~6%. This gives roughly a minute of meaningful history while still adapting across songs.

---

## 4. Expected Behavior

### Steady beat (techno kick)

- Peak tracks kick amplitude, decays slightly between kicks
- Every hit: output ≈ 0.9-1.0 (consistent brightness)
- Between hits: output ≈ 0 (clean gaps)
- Pattern is stable indefinitely

### Loud section → quiet section

- During loud section: peak adapted to loud level
- Quiet section: energy / high peak = low normalized values (dim)
- Peak slowly decays over ~1 minute, normalized values gradually rise
- This is correct: the quiet section *should* look dim relative to what came before

### Quiet section → loud section

- During quiet section: peak has decayed
- Loud section arrives: energy > peak, peak instantly jumps up
- First loud frame: output can briefly exceed 1.0 (peak hasn't fully caught up in a single frame — but with instant `max()` it catches up in one frame)
- Actually: `peak = max(energy, peak * decay)` means peak ≥ energy always, so output ≤ 1.0
- The transition is clean: output jumps from low (dim) to high (bright) in one frame

### Band dropout (bass disappears, treble continues)

- Bass peak slowly decays over ~1 minute
- Bass output = 0 (no energy / decaying peak ≈ 0)
- Treble unaffected (independent normalization)
- When bass returns: if within ~1 minute, peak is still partially high → moderate output. If after >1 minute, peak has decayed far → output jumps to near 1.0 immediately

**Note:** Dropout emphasis (making reintroductions visually dramatic) is a separate concern for the events/temporal layer, not the normalization layer. This algorithm just provides clean 0-1 normalized values per band.

---

## 5. ESP32 Feasibility

- Memory: 1 float per band = 20 bytes for 5 bands
- Compute: 1 max, 1 multiply, 1 divide per band per frame
- Trivial cost within the 18-58ms latency budget

---

## 6. Prior Art (ledger: `normalization-prior-art-audit`)

This is the standard approach. Matches:

| System | Implementation |
|--------|---------------|
| **scottlawsonbc** | Per-filterbank peak with slow decay (α=0.001) |
| **band_sparkles.py** | Per-band peak with decay 0.9995 |
| **WLED SR** | Global AGC (not per-band, but same peak-tracking concept) |
| **PCEN** | Per-channel adaptive normalization (ML domain, same principle) |

### What this algorithm does NOT do

- **Dropout detection** — no threshold, no freeze, no special silence handling
- **Reintroduction emphasis** — no boost mechanism for returning bands
- **Event detection** — no builds, drops, or section boundaries

These belong in downstream consumers (events layer, temporal scope), not in normalization. The normalization layer's job is to produce clean 0-1 values per band. What the system does with dramatic transitions in those values is a separate design decision.

---

## 7. Related Ledger Entries

| Entry | Relevance |
|-------|-----------|
| `per-band-normalization-with-dropout-handling` | Original design (dropout handling now deferred to events layer) |
| `equal-loudness-weighting-dead-end` | Why we rejected perceptual weighting curves |
| `airiness-context-deviation` | Philosophical foundation: feelings = context deviation |
| `adaptation-time-constant-is-temporal-scope` | Decay factor = temporal scope of normalization |
| `normalization-prior-art-audit` | Confirmed this matches industry standard |
