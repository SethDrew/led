# SK6812 RGBW Output Pipeline

How we get from float brightness to flicker-free output on SK6812 RGBW strips. Solves the 8-bit quantization problem that makes low-brightness fades look stepped.

For color generation (hue, chroma, rainbow LUT), see COLOR_ENGINEERING.md. This document covers what happens *after* you have a color — the output stage.

## The hardware problem

SK6812 RGBW uses strictly linear 8-bit PWM on all four channels — 256 levels, uniformly spaced. Unlike WS2812B, which has an undocumented internal nonlinear mapping giving ~2048 effective levels (per cpldcpu measurements), SK6812 has no such compensation.

At the low end (values 0–10), every step is perceptually huge. Human JND is ~1% at these levels, and the jump from 0→1 is 0.4% of full output. Without compensation, smooth fades look like staircases.

The W channel is worst because warm white phosphor has higher luminous efficacy than individual RGB dies — the same PWM step produces a more visible brightness change.

## The pipeline

Seven stages, from animation math to wire. Each stage addresses a specific problem.

```
float brightness  (animation)
    │
    ├─ 1. gamma 2.4         — perceptual-to-linear
    ├─ 2. RGBW routing      — pure W below threshold, blend in RGB above
    ├─ 3. 8.8 fixed point   — preserve sub-LSB precision
    ├─ 4. noise gate        — per-channel snap-to-zero
    ├─ 5. delta-sigma       — temporal dithering (aperiodic)
    │
    └─► uint8 R, G, B, W    (to strip)
```

### 1. Gamma correction (perceptual → linear)

```cpp
float linBright = powf(bright, GAMMA) * BRIGHTNESS_CAP;  // GAMMA = 2.4
```

Apply gamma to the **brightness multiplier**, not per-channel. Per-channel gamma crushes small values to zero and destroys color balance (see COLOR_ENGINEERING.md). The result scales the color channels:

```cpp
float oR = colR * linBright;
float oG = colG * linBright;
float oB = colB * linBright;
```

### 2. RGBW channel routing

SK6812 RGBW has four separate dies. At low brightness, unequal R/G/B values cause the weaker channels to toggle between 0 and 1 — visible as strobe on individual dies. The fix: below a brightness threshold, route all light to the W channel only.

```cpp
#define PURE_W_CEIL   0.10f   // below this: pure W, no RGB
#define PURE_W_BLEND  0.15f   // blend range above CEIL to full RGB

float maxCh_f = fmaxf(oR, fmaxf(oG, oB));
float bFrac = maxCh_f / 255.0f;
float rgbBlend = clampf((bFrac - PURE_W_CEIL) / PURE_W_BLEND, 0.0f, 1.0f);
float avgRGB = (oR + oG + oB) / 3.0f;

float fR = oR * rgbBlend;
float fG = oG * rgbBlend;
float fB = oB * rgbBlend;
float fW = avgRGB * (1.0f - rgbBlend);
```

Below ~10% brightness: RGB channels are zero, all light comes from the W die. Between 10–25%: smooth crossfade. Above 25%: full RGB color. The W channel's single warm white phosphor handles low-brightness smoothly — no die strobe.

### 3. 8.8 fixed-point conversion

Convert float channel values to 16-bit (8.8 format) to preserve sub-LSB fractional information for the ditherer:

```cpp
uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);
```

The high byte is the integer brightness (what the LED gets this frame). The low byte is the fractional residual (what delta-sigma accumulates across frames).

### 4. Sub-LSB noise gate

Snap **each channel independently** to zero when its target16 < 256 (i.e., the 8-bit integer part is 0). This prevents the visible 0↔1 die strobe that delta-sigma produces at sub-LSB targets.

```cpp
if (tR16 < 256) tR16 = 0;
if (tG16 < 256) tG16 = 0;
if (tB16 < 256) tB16 = 0;
if (tW16 < 256) tW16 = 0;
```

**Why per-channel, not coordinated:** An earlier design snapped all channels together when `max(channels) < threshold`. This fails for colored effects with unequal channel ratios — e.g., a teal glow with G=50, B=400: the dominant channel keeps the max above threshold, so the weak channel's target16 passes through and delta-sigma dithers it between 0 and 1. On SK6812, value=1 is a 0.4% duty-cycle pulse (1.2kHz internal PWM), so the 0↔1 toggle is visible as intermittent die flashing. Per-channel gating eliminates this. The visual trade-off is slightly purer color at very low brightness (the weak channel drops out before the dominant one), which is acceptable for most effects. See ledger entry `per-channel-noise-gate-kills-die-strobe` and telemetry data in `audio-reactive/research/datasets/sk6812-pipeline-telemetry/`.

### 5. Delta-sigma modulation

First-order delta-sigma ditherer, applied independently to all four RGBW channels per pixel:

```cpp
// delta_sigma.h
static inline uint8_t deltaSigma(uint16_t &accum, uint16_t target16) {
    accum += target16;
    uint8_t out = accum >> 8;
    accum &= 0xFF;
    return out;
}
```

Each pixel has a persistent accumulator per channel (4 × uint16_t = 8 bytes per pixel). The accumulator integrates the fractional residual across frames. When it overflows the low byte, the output ticks up by 1 for that frame — spreading quantization error in time.

```cpp
uint8_t r = deltaSigma(dsR[i], tR16);
uint8_t g = deltaSigma(dsG[i], tG16);
uint8_t b = deltaSigma(dsB[i], tB16);
uint8_t w = deltaSigma(dsW[i], tW16);
strip.setPixelColor(i, r, g, b, w);
```

**Why this works:** Delta-sigma produces aperiodic switching patterns. The quantization noise is noise-shaped — pushed to higher temporal frequencies where flicker fusion makes it invisible. At 300+ fps on ESP32-C3, the eye integrates across enough frames that sub-LSB fractional brightness is perceived smoothly.

**Why ordered dithering failed:** 16-phase ordered dithering (`frameCount & 15`) at 300 fps produces 18.75 Hz periodic ripple — within visible flicker range for peripheral vision. The periodicity was the problem, not the concept of temporal dithering.

## Accumulator seeding

Accumulators are seeded at boot with spread values to decorrelate per-pixel dither patterns. Without seeding, adjacent pixels at the same target would toggle in lockstep — visible as a coordinated flash.

```cpp
for (uint16_t i = 0; i < LED_COUNT; i++) {
    uint16_t seed = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
    dsR[i] = seed;
    dsG[i] = (seed + 64) & 0xFF;
    dsB[i] = (seed + 128) & 0xFF;
    dsW[i] = (seed + 192) & 0xFF;
}
```

The 64-offset between channels ensures R, G, B, W don't dither in phase on the same pixel either.

## Frame rate

The pipeline requires **uncapped frame rate** — no `delay()` in the render loop. ESP32-C3 with 50 SK6812 LEDs achieves 300+ fps. More frames = more temporal samples for the eye to integrate = smoother perceived brightness.

Animation timing is decoupled from frame rate using `millis()`-based dt, so visual speed is independent of render rate. A 1ms yield (`delay(1)`) keeps WiFi alive without meaningfully reducing throughput.

## All channels, not just W

The original A/B test showed W-alone is sufficient for warm white fades — multi-channel RGB spreading adds no visible benefit *when delta-sigma is already applied*. But the production implementation applies delta-sigma to all four channels anyway. This is the right choice:

- Colored effects (fire amber, red ember) need smooth dithering on RGB channels too
- The cost is trivial (8 bytes per pixel total, 3 ops per channel)
- It means every effect gets correct dithering without thinking about it — the output stage is effect-agnostic

## Implementation cost

| Resource | Per pixel | 50 LEDs |
|----------|-----------|---------|
| RAM (accumulators) | 8 bytes (4 channels × 2 bytes) | 400 bytes |
| CPU (deltaSigma) | 3 ops per channel (add, shift, mask) | negligible |
| CPU (gamma) | 1 powf() per pixel | ~50μs total |

## Source files

| File | Role |
|------|------|
| `festicorn/lib/delta_sigma/delta_sigma.h` | Shared delta-sigma function |
| `festicorn/original-duck/src/gyro_mic_fade.cpp` | Standalone gyro+mic with full pipeline |
| `festicorn/budget-skylight/src/bloom_standalone.cpp` | Bloom/fire/leaf-wind/crawler effects with full pipeline |

## Ledger references

- `delta-sigma-eliminates-low-brightness-flicker` (research) — the validation experiment, why ordered dithering failed
- `temporal-dithering-smooths-low-brightness` (research, superseded) — the 16-phase predecessor
- `sk6812-w-channel-worse-low-brightness` (engineering, resolved) — the hardware constraint that motivated this work
- `rgbw-pure-w-at-low-brightness` (engineering, resolved) — the RGB die strobe problem and pure-W crossover fix
- `rgb-color-gamut-limitations` (research) — broader context: gamma, dithering, and diffusion as complementary solutions

## External sources

- cpldcpu (Tim), "Does the WS2812 have integrated gamma correction?" (2022) — https://cpldcpu.com/2022/08/15/does-the-ws2812-have-integrated-gamma-correction/ — SK6812 linear PWM measurement, WS2812 nonlinear comparison
- Micah Elizabeth Scott, Fadecandy firmware — first-order delta-sigma temporal dithering for LED strips, the direct inspiration for this pipeline
