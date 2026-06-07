# Subharmonic Flicker Analysis

## The Problem

When continuous RGB values are quantized to 8-bit output via temporal dithering,
channels with target values between 0 and ~3/255 produce pulse patterns with
periods long enough for the human eye to resolve as individual flashes.

At 60fps, a target of 0.4/255 means ~1 pulse per 2.5 frames — a 24Hz pattern
sitting in peak flicker sensitivity (8–25Hz worst, detectable up to ~60Hz).

## Three Sources in Any LED Pipeline

1. **Quantization floor** — gamma expansion maps the bottom ~15% of linear input
   into the 0–3 range, widening the danger zone
2. **Scaling compression** — global brightness scalar pushes all channels toward
   zero proportionally
3. **Color imbalance** — warm whites / pastels have weak secondary channels that
   land in the danger zone even when the pixel looks bright overall

## Simulation Results (tree-of-record bloom, first-order delta-sigma)

Danger zone occupancy (channels with 0 < target16 < 256):
- brightness 1.0:  minimal
- brightness 0.15: 163/300 pixel-channels per frame (biolum default)
- brightness 0.05: most channels
- brightness 0.01: nearly all

Lone flashes (single 0→1→0 isolated by >200ms both sides):
- 0.15: 41 flashes across 35/300 channels
- 0.05: 250 flashes across 131/300
- 0.01: 1128 flashes across 295/300

Max inter-pulse gap: 2.7s at 0.15, 6s at 0.01.

The offending channel is always the WEAK one on each pixel (R for teal bloom).

## The Fundamental Tradeoff

You cannot have all three simultaneously at sub-LSB levels:
- Flicker-free
- Correct brightness
- Correct color

Pick two. Each policy makes a different sacrifice:

| Policy | Flicker | Brightness | Color | Best for |
|--------|---------|-----------|-------|----------|
| snap (zero below threshold) | eliminated | -28% at 0.15, -100% at 0.01 | desaturates toward dominant | fade-to-black aesthetics |
| floor (round up to 1/255) | eliminated | +51% at 0.15, +1604% at 0.01 | correct but no fade | always-on installations |
| redistribute (snap + boost survivors) | eliminated | preserved | desaturates | general-purpose default |
| none (raw dither) | present | correct | correct | sparkle/texture effects |

## Pipeline Stage Design

The fix is not in the dither algorithm (first-order delta-sigma is correct).
The fix is a pluggable stage between continuous RGB and the quantizer:

    float RGB → [effect pipeline] → [flicker policy] → [quantizer/dither] → 8-bit out

The policy stage knows frame rate and threshold. Installation designer picks
which tradeoff they want per-effect or per-installation.

## Hardware Test Plan

Full-strip test at near-zero brightness with each policy in isolation.
The entire strip should be running the same marginal signal so the effect
(or absence) is visible everywhere simultaneously — no hunting for which
pixel is flickering. Compare:
1. Raw dither (baseline — see the flicker)
2. Snap to zero
3. Floor to 1
4. Redistribute

## References

- Simulation: tree-of-record/tools/sim/
- Delta-sigma impl: festicorn/lib/delta_sigma/delta_sigma.h
- Bloom source: festicorn/biolum/src/bloom_b.cpp
