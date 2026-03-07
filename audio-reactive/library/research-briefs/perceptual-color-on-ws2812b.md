# Perceptual Color Control for WS2812B LED Strips

## Problem

WS2812B strips produce wrong-looking color when driven with naive RGB values. Three causes compound: (1) the green LED die outputs ~2x the luminous intensity of red and ~4x blue [1], (2) human photopic sensitivity peaks at green (555nm), amplifying the imbalance, and (3) the driver IC applies linear PWM with no gamma correction, so brightness perception is nonlinear. The result: greens dominate, reds and purples wash out, and dimming curves feel abrupt.

## Prior work

No published perceptual color appearance research exists for WS2812B. FastLED applies a fixed RGB scaling (G to 70%, B to 94%). Adafruit provides a gamma=2.8 lookup table [2]. Fadecandy used gamma=2.5 with temporal dithering. WLED inherits FastLED-derived corrections. All are approximated without controlled perceptual testing. cpldcpu confirmed datasheet brightness values with a light sensor but did not address color appearance.

## Our approach

Instead of correcting bad RGB after the fact, we generate perceptually correct RGB from scratch using OKLCH [3], a perceptually uniform color space. Three independent controls:

**Hue.** A 256-entry LUT maps hue angle to RGB at maximum chroma per hue (binary search at 98% sRGB gamut boundary). Lightness varies by hue via cosine curves — reds dip to L=0.52, purples to L=0.38, greens stay at L=0.75 — tuned by A/B comparison on hardware. Zero runtime cost.

**Brightness.** Gamma correction (exponent 2.8 [2]) applied to a scalar brightness multiplier, not per-channel. Per-channel gamma destroys color balance by crushing small channel values to zero. A linear crossover ramp above 60% input ensures full brightness is reachable.

**Chroma.** Desaturation by blending toward BT.601 luminance-weighted gray [4]. Provides a smooth 0-255 saturation slider. Unlike hue and brightness, this required no per-hue tuning — works out of the box.

## Validation

Each axis was validated by side-by-side comparison on WS2812B strips (150 LEDs, folded for direct A/B viewing). Six hue approaches were tested; OKLCH variable-lightness won decisively. Hybrid gamma was compared against linear dimming. Chroma blending was confirmed across five hue families (blue, red, green, purple, gold).

## References

[1] Worldsemi, "WS2812B-V5 Intelligent control LED integrated light source" datasheet. Luminous intensity: G 1100-1400 mcd, R 550-700 mcd, B 200-400 mcd.

[2] P. Burgess, "LED Tricks: Gamma Correction," Adafruit, 2014. https://learn.adafruit.com/led-tricks-gamma-correction

[3] B. Ottosson, "A perceptual color space for image processing," 2020. https://bottosson.github.io/posts/oklab/

[4] ITU-R BT.601-7, "Studio encoding parameters of digital television," 2011. Luminance coefficients: Y = 0.299R + 0.587G + 0.114B.
