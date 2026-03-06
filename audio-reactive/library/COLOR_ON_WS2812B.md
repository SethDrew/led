# Color on WS2812B

How we produce perceptually correct color on WS2812B LED strips. Three independent axes — hue, brightness, and chroma — each with its own solution.

## Background: why WS2812B color is hard

WS2812B contains three separate LED dies (AlGaInP red, InGaN green, InGaN blue). They are not equal:

| Channel | Die material | Datasheet (mcd) | Relative brightness |
|---------|-------------|-----------------|-------------------|
| Green | InGaN | 1100–1400 | 1× (reference) |
| Red | AlGaInP | 550–700 | ~0.5× |
| Blue | InGaN | 200–400 | ~0.25× |

Green appears brightest due to both higher luminous intensity and peak human photopic sensitivity at 555nm. This means naively sending equal RGB values produces visually green-dominated light.

The LED community has no rigorous perceptual color research for these strips. FastLED applies an eyeballed color correction (scale G to 70%, B to 94%). Adafruit provides a gamma=2.8 LUT that is "not super-scientific." WLED inherits FastLED's values. Nobody has published peer-reviewed work.

Our approach: generate perceptually correct RGB from scratch using a perceptually uniform color space, rather than trying to correct bad RGB after the fact.

## Axis 1: Hue — OKLCH variable-L rainbow

**Status: validated on hardware.**

We use Björn Ottosson's OKLCH color space (the cylindrical form of OKLab) to generate a 256-entry hue LUT. OKLCH has three coordinates:

- **L** (lightness): 0 = black, 1 = white
- **C** (chroma): 0 = gray, higher = more saturated
- **H** (hue): 0–360° angle

### Per-hue max chroma

For each of 256 hue steps, we binary-search for the maximum chroma that stays within 98% of the sRGB gamut boundary at the chosen lightness. This means every hue is as vivid as the strip can physically render.

### Variable lightness

A constant L across all hues makes reds and purples look washed out (L=0.75 is too bright for those hues). Our lightness profile uses cosine dips:

```
L_base = 0.75

Red dip:    center=30°,  depth=0.23, half-width=55°  → L≈0.52
Purple dip: center=300°, depth=0.37, half-width=50°  → L≈0.38
```

Green, cyan, and yellow stay at L=0.75. The cosine taper ensures smooth transitions — no abrupt lightness changes between hue regions. These values were tuned by A/B testing on actual WS2812B strips.

### Generator and output

`festicorn/gen_oklch_varL_lut.py` generates the LUT with no external dependencies (manual OKLab implementation). Output is a C array:

```cpp
static const uint8_t oklchVarL[256][3] = {
    {255, 74,  0}, {255, 71,  0}, ...  // red region (low L, high R)
    ...
    { 0, 255, 14}, { 0, 255, 20}, ...  // green region (high L, max G)
    ...
    { 99,  0,255}, { 85,  0,241}, ...  // purple region (lowest L)
};
```

Usage: index `i` = hue step (0–255 maps to 0°–360°). Each entry is a ready-to-send RGB triplet. No runtime math.

### What the rainbow covers and misses

The rainbow is a 1D slice through the 3D color solid — all hues at max saturation with tuned lightness. It cannot represent:
- Desaturated colors (pastels, earth tones, muted hues)
- Near-white or near-black
- Any color at less than full chroma for its hue

These require the chroma axis (see below).

## Axis 2: Brightness — hybrid gamma

**Status: validated on hardware.**

LED strips use linear PWM, but human brightness perception is nonlinear. Halving the PWM duty cycle does not look half as bright — it looks nearly the same, then suddenly drops. Gamma correction maps perceptual brightness to linear PWM.

### Gamma table

We use Adafruit's `gamma8` LUT (gamma exponent 2.8), a 256-entry uint8_t table. This was confirmed on hardware to produce visibly smoother dimming than linear scaling.

### How to apply: brightness multiplier, not per-channel

The critical insight: apply gamma to the **scalar brightness multiplier**, not to individual R, G, B channel values.

**Wrong** (per-channel):
```
R_out = gamma8[R_in]    // A pixel (200, 3, 0) becomes (137, 0, 0)
G_out = gamma8[G_in]    // The trace of green warmth is destroyed
B_out = gamma8[B_in]
```

**Right** (brightness multiplier):
```
brightness_out = gamma8[brightness_in]
scale = brightness_out / 255.0
R_out = R_in * scale
G_out = G_in * scale
B_out = B_in * scale
```

Per-channel gamma crushes small channel values to zero because gamma8 maps small inputs aggressively (e.g., gamma8[3] = 0). This destroys color balance. Applying gamma to a single brightness scalar preserves the hue and saturation of every pixel.

### Hybrid crossover for full brightness

Pure gamma=2.8 never quite reaches maximum output at the top of the input range. The top 20% of input covers 46% of the output range, so a brightness slider spends too long in dim territory and can't hit full brightness.

Solution: follow the gamma curve up to 80% input, then linear ramp to 255:

```cpp
const uint8_t CROSSOVER = 204;  // 80% of 255

uint8_t gammaHybrid(uint8_t v) {
    if (v <= CROSSOVER) return gamma8[v];
    uint8_t base = gamma8[CROSSOVER];  // = 137
    return base + (uint16_t)(v - CROSSOVER) * (255 - base) / (255 - CROSSOVER);
}
```

Below 80%: perceptually uniform dimming via gamma curve.
Above 80%: linear ramp that reaches full brightness at input=255.

Verified on hardware: two OKLCH rainbows side by side cycling through brightness, one with hybrid gamma and one linear. No visible discontinuity at the crossover point. At maximum brightness both produce identical RGB values.

## Axis 3: Chroma (saturation) — not yet implemented

**Status: unimplemented. Next step for palette design.**

The rainbow always uses maximum chroma for each hue. To access the full renderable color solid, we need a chroma control that can reduce saturation from max toward zero.

### What this would unlock

- **Pastels**: high L, low C (e.g., pastel blue, soft pink)
- **Earth tones**: mid L, low C (e.g., terracotta, olive, sand)
- **Near-whites**: very high L, near-zero C
- **"Muted" or "dusty" variants**: any hue at reduced vividness
- **Saturation gradients**: e.g., deep vivid blue → pastel blue → white

### Design direction

At C=0, all hues collapse to the same neutral gray at whatever lightness is set. So chroma reduction with the OKLCH rainbow would smoothly desaturate toward a gray ramp. The generator script already has the `max_chroma_for(L, h_deg)` function — scaling its output by a 0.0–1.0 saturation factor is straightforward.

This would give effects a third parameter axis: in addition to "which hue" and "how bright," they could control "how vivid." Palette gradients that sweep through chroma space (deep color → pastel → white) would become possible without manually picking RGB stops.

### Implementation sketch

Extend the LUT generator to produce a 2D table indexed by (hue, chroma_level), or compute chroma reduction at runtime:

```python
C = max_chroma_for(L, h_deg) * saturation   # saturation: 0.0 to 1.0
r, g, b = oklch_to_srgb8(L, C, h_deg)
```

On ESP32, a small runtime OKLCH-to-RGB conversion would work for static palettes. For real-time effects, precomputing a few chroma levels (100%, 50%, 25%, 0%) as separate LUTs may be more practical.

## Combining the axes

The three axes are orthogonal. To render any color:

1. **Pick a hue** → index into `oklchVarL[hue_index]` to get full-brightness, full-saturation RGB
2. **Apply brightness** → `scale = gammaHybrid(brightness) / 255.0`, multiply all channels
3. **Apply chroma** (future) → reduce saturation in OKLCH space before step 1, or interpolate toward gray after

For current effects, steps 1 and 2 cover the full (H, L) plane at maximum chroma — all vivid colors at any brightness. Step 3 would complete coverage of the full renderable color solid.

## Source files

| File | Purpose |
|------|---------|
| `festicorn/gen_oklch_varL_lut.py` | OKLCH LUT generator (Python, no deps) |
| `festicorn/src/effects.cpp` | Contains `oklchVarL[256][3]` and `oklchConstL[256][3]` |
| `firmware/tree/src/gamma_test.cpp` | Hardware validation firmware (hybrid gamma + rainbow) |

## Ledger references

- `oklch-perceptual-rainbow` — full experiment history, winner rationale
- `hybrid-gamma-brightness` — brightness validation, crossover design
- `ws2812b-channel-brightness` — datasheet die specs, photopic response
- `ws2812b-perceptual-research-gap` — community survey, comparison table
- `oklch-color-solid-coverage` — 3-axis roadmap
- `rgb-color-gamut-limitations` — original problem framing
