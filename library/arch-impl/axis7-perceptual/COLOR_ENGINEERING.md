# Color Engineering

Quick reference for how we handle color on LED strips. Three tools: **hue picking** (rainbow LUT), **brightness curves** (hybrid gamma), and **chroma** (saturation control). All validated on hardware.

For deep details, see the research ledger entries at the bottom.

## How it works

We generate correct RGB from scratch using OKLCH — a color space where equal steps look like equal steps to your eye. This avoids the usual approach of starting with wrong RGB and trying to fix it afterward.

Three independent controls:

1. **Hue** — index into the rainbow LUT (`oklchVarL[0–255]`). Each entry is a ready-to-send RGB triplet at max saturation with per-hue tuned brightness. No runtime math.

2. **Brightness** — scale all three channels by one gamma-corrected number. Never apply gamma per-channel (it kills subtle color components like the warmth in orange).

3. **Chroma** — blend any color toward its perceptual gray. 255 = full color, 0 = neutral gray. Works out of the box — did not need the per-hue tuning that hue and brightness required.

## WS2812B board specifics

What we learned from testing on actual strips:

**The green problem.** WS2812B's green LED puts out roughly 2x the light of red and 4x blue. Our eyes are also most sensitive to green. Result: naive RGB looks green-dominated. The OKLCH rainbow compensates by lowering lightness for greens and raising it for reds/purples.

**Brightness needs gamma.** The strips use linear PWM but our eyes hear brightness logarithmically. We use a gamma=2.8 curve with a linear ramp at the top so max brightness is reachable. Critical rule: apply gamma to the *brightness multiplier*, not to individual R/G/B channels — per-channel gamma crushes small values to zero and destroys color balance.

**Hue needs lightness shaping.** A flat-lightness rainbow (even in OKLCH) washes out reds and purples. Our LUT dips lightness via cosine curves: reds to L≈0.52, purples to L≈0.38, greens/cyans stay at L=0.75. Tuned by A/B testing on hardware.

**Chroma just worked.** Desaturation via BT.601 luminance blending (blend toward perceptual gray) gives satisfying results with no extra tuning. Available as both a runtime slider and precomputed sweep palettes (blue, red, green, purple, gold wash).

**Low-brightness hue shift.** Colors with unequal RGB ratios shift hue as they fade — smaller channels round to 0 first in 8-bit math (orange goes red, then black). Clamp non-zero channels to min 1 to reduce this.

## Source files

| File | Purpose |
|------|---------|
| `audio-reactive/effects/color/oklch.py` | OKLCH math + `RAINBOW_LUT` (single source of truth) |
| `audio-reactive/effects/color/gen_firmware_lut.py` | Emits `festicorn/lib/oklch_lut/oklch_lut.cpp` from `oklch.py` |
| `festicorn/gen_palettes.py` | Chroma sweep palette generator |
| `festicorn/src/effects.cpp` | LUT arrays, rendering, palette data |
| `firmware/tree/src/tree_rainbow_test.cpp` | Runtime chroma + gamma, web UI slider |

## Ledger references

Grep by entry ID in the research ledger for full details:

- `oklch-perceptual-rainbow` — all 6 hue experiments, why variable-L won
- `hybrid-gamma-brightness` — gamma crossover design, hardware test
- `ws2812b-channel-brightness` — LED die specs, photopic response
- `ws2812b-perceptual-research-gap` — community survey (FastLED, Adafruit, WLED, Fadecandy)
- `oklch-color-solid-coverage` — 3-axis coverage roadmap
- `rgb-color-gamut-limitations` — full problem stack, future solution layers (RGBW, dithering, diffusion)
- `hue-shift-low-brightness` — low-brightness failure mode

## External sources

- Björn Ottosson, "A perceptual color space for image processing" (2020) — https://bottosson.github.io/posts/oklab/ — our color space
- Worldsemi, "WS2812B-V5" datasheet — channel brightness and die material specs
- ITU-R BT.601-7 (2011) — luminance coefficients for chroma blending (Y = 0.299R + 0.587G + 0.114B)
- Adafruit, "LED Tricks: Gamma Correction" (2014) — https://learn.adafruit.com/led-tricks-gamma-correction — source of the gamma=2.8 table we use
