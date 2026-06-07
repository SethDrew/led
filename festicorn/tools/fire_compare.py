#!/usr/bin/env python3
"""Trace ACTUAL output RGB for fire effects in bulb-fleet vs tree-of-record.

Reimplements the fire noise math (identical in both) and BOTH output
pipelines, then compares the final 8-bit values that hit the LED.
"""
import math

# ── shared LUTs (copied from festicorn/lib/fast_math/fast_math.h) ──
sinLUT = [
    int(round(math.sin(i / 256.0 * 2 * math.pi) * 32767)) for i in range(256)
]
gammaLUT = [
    int(round((i / 255.0) ** 2.4 * 65535)) for i in range(256)
]

def fastSin(radians):
    idx = int(radians * 40.7436654)
    return sinLUT[idx & 0xFF] * (1.0 / 32767.0)

def fastGamma24(x):
    if x <= 0.0:
        return 0.0
    idx = int(x * 255.0)
    if idx > 255:
        idx = 255
    return gammaLUT[idx] * (1.0 / 65535.0)

def clampf(v, lo, hi):
    return max(lo, min(hi, v))

def deltaSigma(accum, target16):
    # accum is 16-bit; returns (out8, new_accum)
    accum = (accum + target16) & 0xFFFF
    out = accum >> 8
    accum = accum & 0xFF
    return out, accum

# ── fire constants (identical in both files) ──
FIRE_FLICKER_SCALE = 3.0
FIRE_DEADBAND = 0.08
FIRE_DROPOUT_DEPTH = 0.85
BRIGHTNESS_CAP = 0.60          # bulb-fleet only
BULB_GLOBAL_BRIGHTNESS = 0.50  # bulb-fleet default slider

WHITE_BLEND_THRESHOLD = 0.15
RED_FULL = 0.5
amberR, amberG, amberB = 255.0, 140.0, 30.0
redR, redG, redB = 200.0, 20.0, 0.0
whiteR, whiteG, whiteB = 180.0, 170.0, 160.0


def fire_color(ce):
    """Base flame color from fireColorEnergy. Same in both."""
    if ce < WHITE_BLEND_THRESHOLD:
        tw = 1.0 - (ce / WHITE_BLEND_THRESHOLD)
        return (amberR * (1 - tw) + whiteR * tw,
                amberG * (1 - tw) + whiteG * tw,
                amberB * (1 - tw) + whiteB * tw)
    tr = (ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD)
    tr = min(tr, 1.0)
    return (amberR * (1 - tr) + redR * tr,
            amberG * (1 - tr) + redG * tr,
            amberB * (1 - tr) + redB * tr)


def fire_pixel_bright_and_color(i, base, flickerInt, ce, t, sOff, withDropout, dropoutAmount):
    """The per-pixel fire math — IDENTICAL in both firmwares.

    Returns (bright[0..1], colR, colG, colB) in 0..255 color range, pre-gamma."""
    baseColR, baseColG, baseColB = fire_color(ce)

    fi = float(i) + sOff
    noise = (fastSin(fi * 7.3 + t * 2.5) *
             fastSin(fi * 3.7 + t * 1.4) * 0.5 + 0.5)
    noiseAmp = max(0.15 * FIRE_FLICKER_SCALE,
                   0.10 * FIRE_FLICKER_SCALE / max(base, 0.1))
    bright = (base * (1.0 + noiseAmp * (noise - 0.5))
              + flickerInt * (noise - 0.5) * 0.25 * FIRE_FLICKER_SCALE)

    perLedDim = 0.0
    colorRedShift = 0.0
    if withDropout and dropoutAmount > 0.0:
        resilience = (fastSin(fi * 13.7 + t * 0.3) *
                      fastSin(fi * 9.1 + t * 0.2) * 0.5 + 0.5)
        perLedDim = clampf((dropoutAmount - resilience * 0.7) / 0.3,
                           0.0, 1.0) * FIRE_DROPOUT_DEPTH
        colorRedShift = clampf(perLedDim / 0.3, 0.0, 1.0)

    bright *= (1.0 - perLedDim)
    bright = clampf(bright, 0.0, 1.0)

    colR = baseColR * (1 - colorRedShift) + redR * colorRedShift
    colG = baseColG * (1 - colorRedShift) + redG * colorRedShift
    colB = baseColB * (1 - colorRedShift) + redB * colorRedShift
    return bright, colR, colG, colB


# ── bulb-fleet output pipeline ──
def bulbfleet_output(bright, colR, colG, colB, globalBrightness=BULB_GLOBAL_BRIGHTNESS):
    linBright = fastGamma24(bright) * BRIGHTNESS_CAP * globalBrightness
    oR = colR * linBright
    oG = colG * linBright
    oB = colB * linBright
    return (int(clampf(oR, 0, 255)),
            int(clampf(oG, 0, 255)),
            int(clampf(oB, 0, 255)))


# ── tree-of-record output pipeline (no palette rotation: hue=0,sat=0 → no-op) ──
def treeofrecord_output(bright, colR, colG, colB, renderBrightness, accums):
    # effect: linBright = fastGamma24(bright); writes colR*linBright (0..255) to setPixelDither
    linBright = fastGamma24(bright)
    fr = clampf(colR * linBright, 0.0, 255.0)
    fg = clampf(colG * linBright, 0.0, 255.0)
    fb = clampf(colB * linBright, 0.0, 255.0)
    # applyGlobalPalette no-op (hue=0,sat=0)
    fr *= renderBrightness
    fg *= renderBrightness
    fb *= renderBrightness
    t16R = int(min(fr * 256.0, 65535.0))
    t16G = int(min(fg * 256.0, 65535.0))
    t16B = int(min(fb * 256.0, 65535.0))
    aR, aG, aB = accums
    if (t16R | t16G | t16B) == 0:
        aR = aG = aB = 0
    r8, aR = deltaSigma(aR, t16R)
    g8, aG = deltaSigma(aG, t16G)
    b8, aB = deltaSigma(aB, t16B)
    return (r8, g8, b8), (aR, aG, aB)


def treeofrecord_target_float(bright, colR, colG, colB, renderBrightness):
    """The ideal (pre-dither) value tree-of-record is aiming at, in float 8-bit."""
    linBright = fastGamma24(bright)
    fr = clampf(colR * linBright, 0.0, 255.0) * renderBrightness
    fg = clampf(colG * linBright, 0.0, 255.0) * renderBrightness
    fb = clampf(colB * linBright, 0.0, 255.0) * renderBrightness
    return fr, fg, fb


def settle_dither(bright, colR, colG, colB, renderBrightness, frames=64):
    """Average tree-of-record dithered output over N frames → effective brightness."""
    accums = (0, 0, 0)
    sums = [0, 0, 0]
    for _ in range(frames):
        (r8, g8, b8), accums = treeofrecord_output(bright, colR, colG, colB, renderBrightness, accums)
        sums[0] += r8; sums[1] += g8; sums[2] += b8
    return (sums[0] / frames, sums[1] / frames, sums[2] / frames)


# ── Drive the effect state to a steady value for a given energy ──
# We bypass the temporal EMAs and just set the steady-state values the EMAs
# converge to, so we compare the rendering pipelines fairly at matched state.
def steady_state(energy, onset=0.0):
    isSilent = energy < 0.001
    isPercussiveOnly = (not isSilent) and energy < 0.15 and onset > 0.5
    # fireBaseBrightness converges to targetBrightness
    targetBrightness = 0.25 if isSilent else max(0.25, energy)
    base = targetBrightness
    if base < FIRE_DEADBAND:
        base = 0.0
    # fireFlickerIntensity converges to (isSilent?0:onset)
    flickerInt = 0.0 if isSilent else onset
    # fireColorEnergy converges to colorTarget
    if isPercussiveOnly:
        colorTarget = 0.0
    elif isSilent:
        colorTarget = 0.3
    else:
        colorTarget = max(0.3, energy)
    return base, flickerInt, colorTarget


def run():
    scenarios = [
        ("silent (energy=0)", 0.0, 0.0),
        ("mid    (energy=0.5)", 0.5, 0.0),
        ("loud   (energy=1.0)", 1.0, 0.0),
    ]
    # Use a fixed time / strip offset so the noise term is deterministic.
    t = 1.234
    sOff = 0.0
    withDropout = False
    dropoutAmount = 0.0

    sample_pixels = [0, 1, 10, 25, 50, 75, 99]

    print("=" * 100)
    print("FIRE OUTPUT COMPARISON — fire math is BYTE-IDENTICAL; only the output pipeline differs")
    print("=" * 100)
    print()
    print("bulb-fleet : out = colorRGB * fastGamma24(bright) * 0.60(CAP) * 0.50(globalBright)  -> direct 8-bit")
    print("tree-rec   : out = deltaSigma( colorRGB * fastGamma24(bright) * renderBrightness )  -> dithered 8-bit")
    print()

    for label, energy, onset in scenarios:
        base, flickerInt, ce = steady_state(energy, onset)
        print(f"\n#### {label}   [steady-state: base={base:.3f} flicker={flickerInt:.3f} colorEnergy={ce:.3f}]")
        print(f"{'pixel':>5} | {'fire bright':>11} {'colorRGB(pre-gamma)':>22} | "
              f"{'bulb-fleet 8bit':>18} | {'tree@RB=0.30 (dith avg)':>26} | {'tree@RB matching':>16}")
        print("-" * 110)
        for i in sample_pixels:
            bright, colR, colG, colB = fire_pixel_bright_and_color(
                i, base, flickerInt, ce, t, sOff, withDropout, dropoutAmount)
            bf = bulbfleet_output(bright, colR, colG, colB)
            # tree-of-record at its likely default. Find what renderBrightness reproduces bulb-fleet.
            # bulb-fleet effective scale = CAP*globalBright = 0.60*0.50 = 0.30
            tor_target = treeofrecord_target_float(bright, colR, colG, colB, 0.30)
            tor_dith = settle_dither(bright, colR, colG, colB, 0.30)
            print(f"{i:>5} | {bright:>11.4f} ({colR:5.1f},{colG:5.1f},{colB:5.1f}) | "
                  f"({bf[0]:3d},{bf[1]:3d},{bf[2]:3d})       | "
                  f"({tor_dith[0]:5.2f},{tor_dith[1]:5.2f},{tor_dith[2]:5.2f})     | "
                  f"target=({tor_target[0]:5.2f},{tor_target[1]:5.2f},{tor_target[2]:5.2f})")

    # ── Find the renderBrightness that makes tree-of-record == bulb-fleet ──
    print("\n" + "=" * 100)
    print("MATCHING renderBrightness: at what tens-knob does tree-of-record reproduce bulb-fleet default?")
    print("=" * 100)
    # bulb-fleet effective = fastGamma24(bright)*colR*0.30
    # tree target          = fastGamma24(bright)*colR*renderBrightness
    # These are equal iff renderBrightness == 0.30 (= BRIGHTNESS_CAP * globalBrightness).
    print("\nbulb-fleet effective output scale = BRIGHTNESS_CAP * globalBrightness = 0.60 * 0.50 = 0.30")
    print("tree-of-record output scale       = renderBrightness = lerp(0.01, 1.0, tens/9)")
    print("=> EXACT match when renderBrightness = 0.30")
    # solve lerp(0.01,1.0,tens/9)=0.30 -> tens = 9*(0.30-0.01)/(1.0-0.01)
    tens = 9 * (0.30 - 0.01) / (1.0 - 0.01)
    print(f"   lerp(0.01,1.0,tens/9)=0.30  =>  tens = {tens:.2f}  (knob position ~2.6 of 0..9)")
    print(f"   tens=2 -> renderBrightness={0.01+(1.0-0.01)*2/9:.3f};  tens=3 -> {0.01+(1.0-0.01)*3/9:.3f}")

    # numeric proof at energy=1.0, pixel 50
    base, flickerInt, ce = steady_state(1.0, 0.0)
    bright, colR, colG, colB = fire_pixel_bright_and_color(50, base, flickerInt, ce, t, sOff, False, 0.0)
    bf = bulbfleet_output(bright, colR, colG, colB)
    print(f"\nProof @ energy=1.0 pixel=50: bulb-fleet = {bf}")
    for rb in (0.20, 0.30, 0.40, 0.50, 1.00):
        avg = settle_dither(bright, colR, colG, colB, rb)
        tgt = treeofrecord_target_float(bright, colR, colG, colB, rb)
        print(f"   tree renderBrightness={rb:.2f}: dither-avg=({avg[0]:6.2f},{avg[1]:6.2f},{avg[2]:6.2f})  "
              f"float-target=({tgt[0]:6.2f},{tgt[1]:6.2f},{tgt[2]:6.2f})")


if __name__ == "__main__":
    run()
