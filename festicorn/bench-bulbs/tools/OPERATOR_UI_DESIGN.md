# bench-bulbs Operator UI — Design

## Goal
Replace single-char serial (`b`/`g`/`s`/`t`/`f`/`m`/`i`/`w`/`x`/`c`) + reflash-to-tune loop with a web UI that (1) picks an effect and (2) exposes the per-effect knobs Seth actually iterates on. Same shape as `color_probe.py`: local Python HTTP server, browser UI, writes serial lines to ESP32.

## Effects + knobs

Source: `src/bench.cpp`. Knobs listed are the ones that have artistic meaning (skip noise gates, internal smoothing alphas, etc.). Type/range columns are proposals.

### `b` — quiet_bloom  (ALG_QUIET_BLOOM)
Motion-triggered flash + slow breathing palette sweep. **Most-tuned effect** (Seth iterated BLOOM_FLASH_BUMP 20% → 3% → 6% and BLOOM_BRIGHTNESS_CAP 25% → 50% live).

| Knob | Default | Type | Range | Notes |
|---|---|---|---|---|
| BLOOM_FLASH_BUMP | 0.06 | float | 0.00–0.30 | per-hit brightness add (3%–20% tested range) |
| BLOOM_ACCEL_THRESH | 1.5 g | float | 1.05–3.0 | tap threshold above 1g rest |
| BLOOM_BRIGHTNESS_CAP | 0.50 | float | 0.05–1.00 | output cap after gamma |
| flash decay rate | 1.5 /s | float | 0.5–5.0 | `expf(-rate*dt)` on bloomFlash |
| BLOOM_BREATH_MIN_PERIOD | 3.0 s | float | 1–10 | breath cycle min |
| BLOOM_BREATH_MAX_PERIOD | 8.0 s | float | 2–20 | breath cycle max |
| BLOOM_BREATH_FLOOR | 0.15 | float | 0–0.5 | dark-side floor |
| hue drift speed | 1/15..1/45 | float | 1/120–1/5 | palette wander rate |

Palette colors (BLOOM_HUE_A/B/FLASH RGB) — leave as reflash for now. Color tuning is a different mode.

### `g` — gravity_particle  (ALG_GRAVITY_PARTICLE)
N particles fall along strip, accel = tilt.

| Knob | Default | Type | Range |
|---|---|---|---|
| GS_PARTICLE_COUNT | 7 | int | 1–20 (requires `resetGravitySparkle()` on change) |
| GS_GRAVITY_SCALE | 40.0 | float | 5–200 |
| GS_VELOCITY_DAMP | 0.92 | float | 0.5–0.999 |
| GS_BOUNCE_REBOUND | 0.5 | float | 0–1 |
| GS_SPLAT_RADIUS | 2.5 | float | 0.5–8 |
| GS_BRIGHTNESS_CAP | 0.45 | float | 0.05–1.0 |

### `t` — sparkle_twinkle  (ALG_SPARKLE_TWINKLE)
Free-running ambient white sparkle.

| Knob | Default | Type | Range |
|---|---|---|---|
| TWINKLE_SPAWN_RATE | 60 /s | float | 1–300 |
| TWINKLE_ATTACK_S | 0.067 | float | 0.005–0.5 |
| TWINKLE_TAU_S | 0.10 | float | 0.02–2.0 |
| TWINKLE_PEAK_MIN | 0.6 | float | 0–1 |
| TWINKLE_PEAK_MAX | 1.0 | float | 0–1 |
| SPARKLE_BRIGHTNESS_CAP | 0.05 | float | 0.01–0.5 | shared w/ syllable |

### `y` / `s` — sparkle_syllable  (ALG_SPARKLE_SYLLABLE)
Audio-onset triggered sparkle bursts. Knobs mostly internal alphas + cooldown; few are artistic.

| Knob | Default | Type | Range |
|---|---|---|---|
| ignite cooldown | 0.060 s | float | 0.01–0.5 |
| ignite fraction base | 0.25 | float | 0.05–0.8 |
| ignite fraction scale | 0.25 | float | 0–0.75 |
| decay base | 0.92 | float | 0.7–0.99 |
| SPARKLE_DEADBAND | 0.08 | float | 0–0.3 |

### `f` / `m` — fire_flicker / fire_meld  (ALG_FIRE_*)
Two variants share `renderFire()`; `withDropout` flag differs.

| Knob | Default | Type | Range |
|---|---|---|---|
| FIRE_FLICKER_SCALE | 3.0 | float | 0.5–10 |
| FIRE_DEADBAND | 0.08 | float | 0–0.3 |
| FIRE_DROPOUT_DEPTH | 0.85 | float | 0–1 |
| BRIGHTNESS_CAP (shared) | 0.25 | float | 0.05–1 |

### `i` — idle  (rainbow wash)
| Knob | Default | Type | Range |
|---|---|---|---|
| IDLE_BRIGHTNESS | 0.30 | float | 0–1 |
| IDLE_SPEED | 0.03 | float | 0–0.5 |

### Global (already wired via `!B`/`!S` UDP slider — but **not serial**)
| Knob | Default | Range |
|---|---|---|
| globalBrightness | 1.0 | 0–2 (val/128) |
| globalSensitivity | 1.0 | 0.05–2 (val/128) |

Currently `handleSliderCommand()` is UDP-only. Should also accept serial.

## Runtime-tunable vs reflash

**Runtime (cheap, just variables):** all numeric knobs above. Convert each `#define` to a `static float` (or `static int` for counts) with the define as the initializer. Touching a count knob (`GS_PARTICLE_COUNT`) also calls the relevant `reset…()`.

**Reflash:** palette RGB triples (BLOOM_HUE_A/B/FLASH, FIRE amber/red/white), pin assignments, LED_COUNT, baud, SSID.

## Serial protocol additions

Existing: single chars (`b g s t y f m i w x c ?`), `#RRGGBBWW`, `!B<n>` `!D<0|1>` `!F<n>` (raw-color helpers).

Add a generic parameter set command. Keep it line-oriented so it composes with the existing `processSerialLine()`:

```
!P <KEY>=<value>\n
```

Examples:
```
!P BLOOM_FLASH_BUMP=0.06
!P GS_GRAVITY_SCALE=40
!P GS_PARTICLE_COUNT=12     ← triggers reset
!P GLOBAL_BRIGHTNESS=1.0
```

Firmware keeps a small table: `{name, type, float* / int*, optional reset fn}`. Unknown keys logged + ignored. On set: parse, clamp to range, assign, optionally call reset, echo `[PARAM] <KEY>=<value>`.

Also add `!P?\n` → dump all current values (one per line, `[PARAM] KEY=VALUE`). UI uses this on connect to populate sliders.

## UI layout (text sketch)

```
┌─ bench-bulbs operator ─────────────────────────┐
│ [● connected /dev/cu.usbserial-0001]           │
│                                                │
│ Effect:  ( ) off  (●) bloom  ( ) gravity       │
│          ( ) twinkle  ( ) syllable             │
│          ( ) fire_flicker  ( ) fire_meld       │
│          ( ) idle  ( ) raw_color               │
│                                                │
│ ─ Global ──────────────────────────────────    │
│ Brightness  [────●──────] 1.00                 │
│ Sensitivity [──●────────] 0.80                 │
│ [Calibrate]                                    │
│                                                │
│ ─ bloom knobs ─────────────────────────────    │
│ flash bump        [──●──────] 0.06             │
│ accel threshold   [───●─────] 1.5 g            │
│ brightness cap    [──────●──] 0.50             │
│ flash decay       [───●─────] 1.5 /s           │
│ breath min period [──●──────] 3.0 s            │
│ breath max period [────●────] 8.0 s            │
│ breath floor      [──●──────] 0.15             │
│ hue drift max     [───●─────] 1/15             │
│                                                │
│ [Reset to defaults]                            │
│                                                │
│ status: !P BLOOM_FLASH_BUMP=0.06 → ok          │
└────────────────────────────────────────────────┘
```

One panel of knobs visible at a time, switched by the effect radio. Lazy approach: render every effect's knobs as a `<details>` block under the radio; expand the active one.

Style: lift from `color_probe.py` (dark bg, Menlo, range+number pairs, fetch on input).

## Implementation plan

1. **Firmware param table** — `src/bench.cpp`:
   - Convert listed `#define`s to `static float`/`static int` runtime vars with same names lowercased (`bloomFlashBump`, etc.). Keep `#define` defaults as initializers (`= 0.06f`).
   - Update existing uses (e.g. `renderQuietBloom` references `bloomFlash * expf(-1.5f * dt)` — make the `1.5f` into `bloomFlashDecayRate`).
   - Add `struct Param { const char* name; enum {F32,I32} t; void* ptr; float lo, hi; void (*onChange)(); }` and a `static const Param PARAMS[] = {…}`.
   - Extend `processSerialLine()`: handle `!P` lines — parse `KEY=val`, lookup, clamp, assign, log. Handle `!P?` — iterate table, print each.
   - Verify build: `cd festicorn/bench-bulbs && ../../.venv/bin/pio run`.

2. **Python operator tool** — `tools/operator.py` (new, sibling to `color_probe.py`):
   - HTTP server on a fresh port (8322).
   - On launch: open serial, send `!P?\n`, parse `[PARAM] …` responses into a dict, serve as JSON to the UI.
   - Routes: `GET /` HTML; `GET /params` JSON of current values; `POST /set?key=…&v=…` writes `!P KEY=v\n`; `POST /alg?c=b` writes single char.
   - One HTML file embedded (color_probe-style), one fetch per slider change. Debounce 50 ms.

3. **Sanity test** — set every effect, twiddle every knob, confirm `[PARAM]` echo + visual change. Test `GS_PARTICLE_COUNT` reset path.

## Out of scope (for v1)
- Saving/loading presets (just write defaults in firmware).
- Palette editor for bloom/fire colors.
- Multi-board control (single serial port only).
- Persisting tuned values across reboot (no NVS write yet — Seth promotes good values into source by editing the `#define`).
