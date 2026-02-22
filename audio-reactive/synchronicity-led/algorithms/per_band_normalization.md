# Per-Band Normalization with Dropout Handling

Algorithm specification for normalizing frequency band energy against its own recent context, with asymmetric adaptation and dropout detection.

**Status:** Untested design (ledger: `per-band-normalization-with-dropout-handling`)

---

## 1. Problem

### Why not perceptual weighting?

Equal-loudness curves (A/C-weighting, ISO 226) applied to mastered music **double-correct** — mix engineers already compensate for human hearing sensitivity. No phon level produces balanced band energy from professionally mixed music (ledger: `equal-loudness-weighting-dead-end`).

- A-weighting (40 phon): eliminates sub-bass entirely
- C-weighting: nearly flat (useless for band discrimination)
- ISO 226 at concert levels (100-115 phon): over-attenuates bass

### Why per-band normalization?

Normalize each band against its own recent history instead. This:

- Matches the "context deviation" principle — feelings are deviations from context, not absolute levels (ledger: `airiness-context-deviation`)
- Works in real-time on ESP32 — only needs what it's seen so far, not the full track
- Captures musical intention of dropouts/reintroductions without explicit structure detection
- Avoids the fundamental problem of static weighting on dynamic music

---

## 2. Components

### 2.1 Asymmetric EMA Reference

Track a running reference level per band using an exponential moving average with different attack and release rates.

```
if energy > reference:
    reference += (energy - reference) * attack    # fast: adopt new peaks
else:
    reference += (energy - reference) * release   # slow: drift down gradually
reference *= decay                                # constant decay per frame
```

**Why asymmetric:** Fast attack means new peaks immediately register at full brightness. Slow release means the reference drifts down after peaks, so subsequent hits of the same magnitude still appear bright.

**Why constant decay:** Without it, a perfectly steady signal would normalize to exactly 1.0 and appear flat. The decay factor ensures the reference is always slightly below actual energy, so steady patterns (e.g., techno kick) always show >1.0 on each hit.

### 2.2 Absolute Dropout Threshold

A fixed energy floor per band, below which the band is considered silent.

```
if energy < absolute_floor:
    output = 0           # hard zero — clean darkness
    freeze reference     # stop updating
```

**Why absolute (not relative to reference):** Prevents false dropout detection in genres with sustained low-energy bands. A techno kick pattern that's monotonic but not silent would falsely trigger dropout if the threshold were relative to the current reference.

### 2.3 Reference Freeze During Dropout

When a band drops below the absolute floor:

1. Output hard zero (clean visual silence)
2. Stop updating the reference — freeze it at the pre-dropout value
3. When energy returns above the floor, unfreeze and resume normal EMA updates

**Effect:** After a 15-30 second dropout, the frozen reference is much lower than the returning energy. The normalized output spikes to 1.5x-4.0x for the entire reintroduction section, naturally capturing the "crash" feel of EDM drops without any explicit structure detection.

---

## 3. Pseudocode

```python
# Per band, per frame (30 fps)

def normalize_band(band, energy):
    # Dropout detection
    if energy < ABSOLUTE_FLOOR[band]:
        frozen[band] = True
        return 0.0

    # Unfreeze if coming out of dropout
    if frozen[band]:
        frozen[band] = False

    # Asymmetric EMA
    if energy > reference[band]:
        reference[band] += (energy - reference[band]) * ATTACK
    else:
        reference[band] += (energy - reference[band]) * RELEASE

    # Constant decay (prevents normalization settling)
    reference[band] *= DECAY

    # Normalize
    return energy / max(reference[band], ABSOLUTE_FLOOR[band])
```

---

## 4. Parameters

### Known

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Attack coefficient | 0.3 | ~3 frames to adopt new peak |
| Release coefficient | 0.002 | ~500 frames / ~17s to decay |
| Decay factor | 0.9999 per frame | ~0.003 dB/frame drift, prevents settling |
| Dropout output | 0 (hard zero) | Clean visual silence |

### Unresolved

| Parameter | Range to explore | Impact |
|-----------|-----------------|--------|
| Absolute floor per band | ~0.01 × calibration max | Dropout detection sensitivity |
| Band definitions | Standard 5-band (sub/bass/low-mid/high-mid/treble) | Which frequencies to track independently |
| Whether decay needs per-band tuning | 0.999 - 0.99999 | Faster decay = more headroom but less stable |

---

## 5. Expected Behavior

### Steady beat (techno kick, constant energy)

- Reference tracks slightly below kick peaks due to constant decay
- Every hit: output ~1.0-1.2x (always visible)
- Between hits: output near 0 (clean gaps)
- Pattern sustains indefinitely without going flat

### EDM drop/reintroduction (15-30s dropout)

- **Pre-dropout:** Reference at steady state, output ~1.0x
- **During dropout:** Output = 0 (hard zero), reference frozen
- **Reintroduction:** Frozen reference << returning energy → output spikes to 1.5-4.0x
- **Recovery:** Fast attack gradually brings reference up, output decays to ~1.0x over ~10-30s

Dropout duration vs boost magnitude (approximate):
- 15s dropout → ~1.3x boost
- 30s dropout → ~2.0x boost
- 45s dropout → ~4.0x boost

### Sustained low-energy band (ambient sub-bass)

- Energy above absolute floor → no false dropout
- Reference tracks the low energy level normally
- Output stays ~1.0-1.2x (visible, not exaggerated)
- Constant decay prevents reference from settling exactly at energy level

### Gradual build (crescendo over 30s)

- Fast attack keeps reference close behind rising energy
- Output ~1.05-1.15x throughout — gradual brightening, not sudden spikes
- This is correct: the build *feeling* should be gradual, not binary

---

## 6. ESP32 Feasibility

- Memory: ~12 bytes per band (reference float + frozen bool + floor float) = ~60 bytes for 5 bands
- Compute: 3 comparisons, 2 multiplies, 1 division per band per frame
- Easily fits within the 18-58ms latency budget

---

## 7. Related Ledger Entries

| Entry | Relevance |
|-------|-----------|
| `per-band-normalization-with-dropout-handling` | Primary entry for this algorithm |
| `equal-loudness-weighting-dead-end` | Why we rejected perceptual weighting curves |
| `airiness-context-deviation` | Philosophical foundation: feelings = context deviation |
| `adaptation-time-constant-is-temporal-scope` | Window size determines what events register |
| `rolling-integral-sustained-energy` | Related concept: sliding-window energy tracking |
| `accent-vs-groove-effects` | Accent effects use per-band peak detection this normalizes |
