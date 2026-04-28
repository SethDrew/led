# Band Recombination Strategies

How to collapse 3 per-band normalized signals (`bass`, `mids`, `treble` — each already in [0,1] via `PerBandEMANormalize`) into a single drive signal for LED effects, **without bass dominating**, and ideally as a reusable primitive across many future effects.

---

## Recommendation

**Strategy 5 (per-band EMA → softmax-weighted sum) as the headline single-channel primitive, paired with Strategy 7 (activity count as an explicit second channel).** The softmax-weighted sum keeps the user's "per-band EMA → sum" intuition (each band normalized then combined, so chronic bass loses its edge) but adds a contrast term: when one band is *much* louder than the others, that band drives the output; when all three are roughly equal, it averages. This avoids the two pathologies — the always-bright `max` (Strategy 1) and the muddy `mean` (Strategy 3) — while staying fully linear and ESP32-cheap. The bigger structural win is recognizing that **collapsing to one signal is a lossy choice**: keeping `(intensity, density)` as two channels gives downstream effects far more to work with than any single-channel scheme. Strategy 7 should ride alongside whatever single-channel choice wins.

The absint-deemphasizes-bass hypothesis is **partially true but does not solve the problem on its own** — see §"Absint hypothesis" below.

---

## Notation

Let `b`, `m`, `t` be the per-band EMA-normalized values (each clipped to `[0,1]`, where 1.0 = `max_ratio` × the band's running mean — see `PerBandEMANormalize` in `signals.py`). All formulas operate per-frame.

For "3-bands-firing" cases I assume `b=0.9, m=0.7, t=0.6` (drop, all hot).
For "bass-only-loud" I assume `b=0.9, m=0.05, t=0.05` (sub bassline over silence).
For "mids+treble, no bass" I assume `b=0.05, m=0.7, t=0.6` (vocal + cymbals over a breakdown).

---

## Strategy 1 — `max(b, m, t)` (current 3-band waterfall)

```
y = max(b, m, t)
```

- **Preserves**: whichever band is loudest *relative to its own history*. Pulses survive.
- **Destroys**: contrast between "one band hot" vs "all bands hot" — both read 0.9. Loses the *texture* of a multi-band hit.
- **3-bands-firing**: `y = 0.9`. Same as bass-only.
- **Bass-only-loud**: `y = 0.9`. Same as 3-bands.
- **Reusability**: high (1 line, monotone, bounded). But the always-something-is-firing problem the user already noted: per-band EMA guarantees *some* band sits near 1.0 most of the time, so max-of-bands rarely dips. Output is uniformly bright with little dynamic shape.

---

## Strategy 2 — `mean(b, m, t)`

```
y = (b + m + t) / 3
```

- **Preserves**: gross overall energy. Still contextual (per-band EMA already removed absolute bass dominance).
- **Destroys**: single-band transients. A snare-only hit at `m=0.9` reads 0.3 — perceptually loud, visually dim.
- **3-bands-firing**: `y = 0.73`.
- **Bass-only-loud**: `y = 0.33`.
- **Reusability**: moderate. Smooth, well-behaved, bounded. But the "transient gets muddied" problem hits any percussive effect.

---

## Strategy 3 — `sum(b, m, t)` clipped to [0,1]

```
y = clip(b + m + t, 0, 1)
```

This is the user's stated "per-band EMA → sum" hypothesis in its simplest form.

- **Preserves**: dimensional density. More active bands → brighter, monotonically. Each band's *relative* loudness already deemphasized by EMA.
- **Destroys**: above the clip ceiling, the texture of how-much-louder collapses. With 3 active bands you spend most of the time at the cap.
- **3-bands-firing**: `b+m+t = 2.2`, clipped to 1.0.
- **Bass-only-loud**: `y = 0.9`. Bass alone can saturate — the per-band EMA does the heavy lifting for *chronic* bass dominance, but a momentary bass spike still hits 1.0.
- **Reusability**: cheap, but the clip cliff makes it a poor primitive for proportional effects (lots of saturation = lost contrast).

**Verdict on the user's hypothesis specifically**: per-band EMA handles bass dominance only on the *time-averaged* axis. Within any single frame a bass kick can still spike to 1.0 because EMA tracks the mean, not the peak — so kicks (which are above-mean by definition) will saturate the sum. The hypothesis is correct that bass *over a section* loses its edge; it's wrong that *per-frame* bass spikes lose theirs. To attenuate per-frame, you need either a softer combiner (Strategies 4-5) or active competition (Strategy 6).

---

## Strategy 4 — softmax / weighted-by-self mean

```
w_i  = b_i^p                    # p ≥ 1, sharpening exponent
y    = Σ w_i · b_i / Σ w_i
```

(Self-weighted: louder bands count more, but *all* bands contribute. With `p=1` this is RMS-like; with `p→∞` it converges to `max`.)

- **Preserves**: emphasizes whoever is loudest while still acknowledging the others.
- **Destroys**: less than max; more than mean. Tunable.
- **3-bands-firing** (`p=2`): `(0.81·0.9 + 0.49·0.7 + 0.36·0.6) / (0.81+0.49+0.36) = 1.295/1.66 = 0.78`.
- **Bass-only-loud** (`p=2`): `(0.81·0.9 + 0.0025·0.05 + 0.0025·0.05) / (0.815) ≈ 0.895`.
- **Reusability**: good, one extra knob (`p`). For the recombination problem specifically though, this still gives bass-only and 3-bands roughly the same headline number when bass is loudest, so it doesn't really change the contrast story.

---

## Strategy 5 — softmax-weighted sum with floor (recommended)

A small twist on Strategy 4: combine softmax weighting (favors the loudest) with a sum (rewards multi-band activity). One reasonable form:

```
w_i  = b_i^p                    # p≈2
y    = clip( Σ w_i · b_i / max(w_i)  +  α · Σ b_i / 3 ,  0, 1 )
```

The first term ≈ "loudest band, sharpened" → close to `max` for one-hot cases. The second term, weighted by α (≈0.3), adds a "density bonus" for multiple active bands. Total stays bounded.

- **Preserves**: a single hot band drives output (good for solo transients). Multiple hot bands push it higher than any single hot band would (good for density).
- **Destroys**: nothing dramatic; soft saturation near 1.
- **3-bands-firing** (`p=2, α=0.3`): first term ≈ 0.78 (Strategy 4 value), density bonus ≈ 0.22 → `y ≈ 1.0` (saturates).
- **Bass-only-loud**: first term ≈ 0.895, density bonus ≈ 0.10 → `y ≈ 0.99`.
- **Mids+treble, no bass**: first term ≈ 0.66, density bonus ≈ 0.45/3·0.3·… ≈ 0.13 → `y ≈ 0.79`. *This* is where it differs from `max`: a hot-bass-only frame and a hot-mids+treble frame both get bright, and so does the all-three-firing frame, but bass-only no longer wins automatically.
- **Reusability**: good. Two knobs (`p`, `α`). Bounded, smooth, monotone in each band.

**Why this over Strategy 3 (raw sum)**: the softmax term means a single slightly-hot band doesn't compete with the loudest band for headline brightness, so output more closely tracks "what is the dominant audio event right now" rather than "how much energy is sitting around in any band". Combined with the density term, you still distinguish "all three firing" from "one firing".

---

## Strategy 6 — competitive normalization (lateral inhibition)

```
total = b + m + t + ε
shares: B = b/total, M = m/total, T = t/total          # always sum to 1
y     = max(B, M, T)                                    # or any scalar of shares
```

A band only contributes if it's *winning*. If bass is consistently 90% of the energy, `B = 0.9, M ≈ 0.05, T ≈ 0.05` → `y = 0.9`. But the difference: `y` is now a *share*, not a level. To get a brightness, multiply by overall level:

```
level   = (b + m + t) / 3          # or sum, or max
shares  = (b, m, t) / total
y       = level · max(shares)
```

- **Preserves**: bands that win their slice. Single-band hits look distinct from broadband loud passages (latter has lower max-share).
- **Destroys**: a balanced loud passage reads dim (max-share ≈ 1/3, level ≈ 0.7 → y ≈ 0.23). Possibly desirable for a "punch" detector, undesirable for general brightness.
- **3-bands-firing**: shares = `(0.41, 0.32, 0.27)`, level = 0.73 → `y = 0.41 · 0.73 = 0.30`.
- **Bass-only-loud**: shares = `(0.9, 0.05, 0.05)`, level = 0.33 → `y = 0.9 · 0.33 = 0.30`.
- **Reusability**: niche. The shares vector is genuinely useful (it's the "where is the energy concentrated" signal — see Strategy 7 for the related dimension), but as a single drive signal it's biased toward percussive single-band hits.

---

## Strategy 7 — encode density as a separate dimension (the bigger win)

Don't collapse to one channel. Output two:

```
intensity  = one of Strategies 1-5      # "how loud right now"
density    = #{ i : b_i > θ } / 3       # 0, 1/3, 2/3, or 1 (with θ ≈ 0.3)
   # or smoothly: density = (b + m + t) / (max(b,m,t) + ε)   # 1.0 if one band, ≈3 if balanced
   # normalized to [0,1] by dividing by max possible (3) and subtracting 1/3 floor
```

- **Preserves**: both axes — *how strong* the moment is and *how broad-band* it is. A solo bass kick reads `(intensity=0.9, density=0.33)`. A mid-drop with everything firing reads `(0.95, 1.0)`. A breakdown with sustained mid alone reads `(0.4, 0.33)`.
- **Reusability**: very high. Effects map them differently:
  - Brightness ← intensity, palette/hue ← density.
  - Pulse rate ← intensity, pulse width ← density.
  - Particle speed ← intensity, particle count ← density.
  - Fog density ← 1 − density (sparse moments → atmospheric).
- **Why this matters**: the user's framing in the prompt is "we have to choose how to recombine 3 → 1". That premise is the source of the trouble. Two channels removes the false choice between "favor transients" and "favor density" — you express both.

A clean continuous form for density:
```
density_raw = (b + m + t) / (max(b,m,t) + ε)     # range: [1, 3]
density     = (density_raw - 1) / 2              # range: [0, 1]
```
1.0 = perfectly balanced across bands; 0.0 = single-band only.

---

## Absint hypothesis

The user proposed: maybe `absint` of broadband already deemphasizes bass (because bass sustains, lower per-frame |dRMS/dt|), avoiding the need for a per-band split.

Reading `AbsIntegral` in `audio-reactive/effects/signals.py:158-219`:
- It computes `|d(RMS)/dt|` and integrates over a 150ms ring buffer.
- Then peak-decay normalizes the result to 0-1.

**What's true about the hypothesis:**
- Sustained tones (a held bass note, droning pad) produce near-zero per-frame derivative, so they get suppressed. ✓
- A snare hit and a kick hit both produce sharp dRMS/dt spikes during their attack envelope. The snare's spike isn't necessarily bigger.

**What's false about the hypothesis:**
- A kick attack is *fast* (5-20ms attack envelope) and big (raw RMS jumps from quiet to loud). The derivative magnitude scales with the *amplitude swing*, not the spectral content. A kick still produces a large |dRMS/dt| spike — often larger than a snare's because the kick's RMS swing is bigger.
- After peak-decay normalization, the kick will hit 1.0 just like the snare. Bass *transient* dominance is preserved.
- The one place it helps: a sustained sub bassline (no attack envelope) *is* deemphasized by absint relative to broadband RMS. So it does help if "bass" means "drone bass". It doesn't help if "bass" means "kick".

**Conclusion**: absint deemphasizes *steady* loud signals (any band — sustained vocals are also damped) but does nothing about transient bass. For an LED system that wants to keep kick-driven feel without bass crowding everything else, absint is necessary but not sufficient. You still want per-band normalization upstream so bass kicks can be put on the same scale as snare hits.

A useful composite primitive that combines both ideas:
```
per-band RMS → PerBandEMANormalize → PerBandAbsIntegral → recombine
```
This is essentially what `EnergyWaterfall3BandEffect` already does (via the `PulseDriver` max). The recombination strategy is the missing knob.

---

## Summary table

| # | Strategy | Bass-only | 3-bands hot | Mids+tre hot | Bounded? | Knobs | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `max` | 0.90 | 0.90 | 0.70 | yes | none | always-bright; current pain |
| 2 | `mean` | 0.33 | 0.73 | 0.45 | yes | none | safe but transients muddied |
| 3 | `sum` clipped | 0.90 | 1.00 | 1.00 | yes | none | clip cliff; bass still spikes |
| 4 | softmax (`p=2`) | 0.90 | 0.78 | 0.66 | yes | `p` | very close to `max` for the cases that matter |
| 5 | **softmax + density floor** | 0.99 | 1.00 | 0.79 | yes | `p, α` | distinguishes density without losing transients |
| 6 | competitive (max-share × level) | 0.30 | 0.30 | 0.30 | yes | none | diagnostic only; bad as drive signal |
| 7 | **2-channel (intensity, density)** | (0.9, 0.33) | (1.0, 1.0) | (0.7, 0.93) | yes | varies | the real win — don't collapse |

---

## Test plan recommendation

Worth building a quick A/B harness that runs the same audio (one EDM track, one rap track, one ambient) through all five single-channel strategies plus the 2-channel scheme, dumping each as a brightness trace and as a rendered LED-o-gram (`tools/preview_effect.py`). Hard to pick a winner from formulas alone — taste decision, requires eyes on the actual signal envelopes.

The cheapest first test: take `EnergyWaterfall3BandEffect` and swap the `band_brightness.max(axis=0)` collapse on line 232 for each strategy in turn. Same color, same scroll, only the recombination changes.
