"""
Quiet bloom bioluminescence algorithm simulation.

Traces the exact math pipeline from receiver_3bulbs.cpp for a single LED
through its breathing cycle and flash events. Identifies value bottlenecks.
"""
import math

# -- Constants (from receiver_3bulbs.cpp) --
GAMMA          = 2.4
BRIGHTNESS_CAP = 0.30
NOISE_GATE     = 24

BREATH_MIN_PEAK = 0.50
BREATH_MAX_PEAK = 0.70
BREATH_FLOOR    = 0.03

HUE_A_G = 20.0
HUE_A_B = 100.0
HUE_B_G = 70.0
HUE_B_B = 110.0

FLASH_G = 150.0
FLASH_B = 170.0
FLASH_PEAK = 0.85

W_ONSET = 0.5


def clampf(v, lo, hi):
    return max(lo, min(hi, v))


def lerpf(a, b, t):
    return a + (b - a) * t


def trace_pipeline(glow, hueT, is_flash=False, label=""):
    """Trace the exact math pipeline for a given glow level."""
    # Color selection
    if is_flash:
        colG = FLASH_G
        colB = FLASH_B
    else:
        colG = lerpf(HUE_A_G, HUE_B_G, hueT)
        colB = lerpf(HUE_A_B, HUE_B_B, hueT)

    # Brightness: gamma + cap
    linBright = (glow ** GAMMA) * BRIGHTNESS_CAP

    oG = colG * linBright
    oB = colB * linBright

    # W channel
    wFrac = clampf((glow - W_ONSET) / (1.0 - W_ONSET), 0.0, 1.0)
    oW = wFrac * linBright * 200.0

    # Scale to 16-bit
    tG16 = int(clampf(oG * 256.0, 0, 65535))
    tB16 = int(clampf(oB * 256.0, 0, 65535))
    tW16 = int(clampf(oW * 256.0, 0, 65535))

    # Noise gate
    maxCh = max(tG16, tB16, tW16)
    passes_gate = maxCh >= NOISE_GATE

    if not passes_gate:
        tG16_out = 0
        tB16_out = 0
        tW16_out = 0
    else:
        tG16_out = tG16
        tB16_out = tB16
        tW16_out = tW16

    # Approximate 8-bit output (without delta-sigma dither, just >> 8)
    approx_G8 = tG16_out >> 8
    approx_B8 = tB16_out >> 8
    approx_W8 = tW16_out >> 8

    return {
        "label": label,
        "glow": glow,
        "linBright": linBright,
        "colG": colG, "colB": colB,
        "oG": oG, "oB": oB, "oW": oW,
        "tG16": tG16, "tB16": tB16, "tW16": tW16,
        "maxCh": maxCh,
        "passes_gate": passes_gate,
        "approx_G8": approx_G8, "approx_B8": approx_B8, "approx_W8": approx_W8,
        "is_flash": is_flash,
        "wFrac": wFrac,
    }


def print_header():
    print(f"{'Label':<22} {'glow':>6} {'linBrt':>8} {'colG':>5} {'colB':>5}"
          f" {'oG':>7} {'oB':>7} {'oW':>7}"
          f" {'tG16':>6} {'tB16':>6} {'tW16':>6}"
          f" {'maxCh':>6} {'gate?':>6}"
          f" {'~G8':>4} {'~B8':>4} {'~W8':>4}")
    print("-" * 130)


def print_row(r):
    gate_str = "PASS" if r["passes_gate"] else "KILL"
    print(f"{r['label']:<22} {r['glow']:>6.4f} {r['linBright']:>8.5f}"
          f" {r['colG']:>5.0f} {r['colB']:>5.0f}"
          f" {r['oG']:>7.3f} {r['oB']:>7.3f} {r['oW']:>7.3f}"
          f" {r['tG16']:>6} {r['tB16']:>6} {r['tW16']:>6}"
          f" {r['maxCh']:>6} {gate_str:>6}"
          f" {r['approx_G8']:>4} {r['approx_B8']:>4} {r['approx_W8']:>4}")


# =============================================================================
# PART 1: Full breathing cycle at stillness=1.0
# =============================================================================
print("=" * 130)
print("PART 1: BREATHING CYCLE TRACE (stillness=1.0)")
print("=" * 130)

# Representative LED parameters
for hue_label, hueT_val in [("hueT=0.0 (deep blue)", 0.0),
                              ("hueT=0.5 (mid)", 0.5),
                              ("hueT=1.0 (teal)", 1.0)]:
    for peak_label, breathPeak in [("minPeak=0.50", BREATH_MIN_PEAK),
                                    ("maxPeak=0.70", BREATH_MAX_PEAK)]:
        print(f"\n--- {hue_label}, {peak_label} ---")
        print_header()

        # Breathing cycle key points
        # breath = sin(phase * 2pi) * 0.5 + 0.5
        # phase=0.00: breath=0.5 (mid-rise)
        # phase=0.25: breath=1.0 (peak)
        # phase=0.50: breath=0.5 (mid-fall)
        # phase=0.75: breath=0.0 (trough)
        key_points = [
            (0.75, "trough (breath=0)"),
            (0.875, "quarter-rise"),
            (0.00, "mid-rise (breath=.5)"),
            (0.125, "3/4 rise"),
            (0.25, "PEAK (breath=1)"),
            (0.375, "3/4 fall"),
            (0.50, "mid-fall"),
            (0.625, "quarter-fall"),
        ]

        for phase, phase_label in key_points:
            breath = math.sin(phase * 2.0 * math.pi) * 0.5 + 0.5
            # breathGlow = BREATH_FLOOR + breath * (breathPeak - BREATH_FLOOR)
            breathGlow = BREATH_FLOOR + breath * (breathPeak - BREATH_FLOOR)
            # at stillness=1.0, breathGlow *= 1.0 (no change)
            glow = breathGlow  # no flash

            r = trace_pipeline(glow, hueT_val, is_flash=False,
                               label=f"ph={phase:.3f} {phase_label}")
            print_row(r)


# =============================================================================
# PART 2: FLASH EVENT TRACE
# =============================================================================
print("\n\n" + "=" * 130)
print("PART 2: FLASH EVENT TRACE")
print("=" * 130)
print_header()

# Flash glow values from initial hit to decay
# FLASH_PEAK = 0.85 initially, then decays by flashDecay^(dt*30)
# Typical flashDecay ~0.935 (midpoint), dt=0.04s (25Hz)
flash_decay = 0.935
dt = 0.04
flash_g = FLASH_PEAK

for step in range(40):
    t = step * dt
    if step == 0:
        label = f"t={t:.2f}s FLASH HIT"
    else:
        label = f"t={t:.2f}s"

    r = trace_pipeline(flash_g, 0.5, is_flash=(flash_g > 0.03),
                       label=label)
    print_row(r)

    # Decay
    flash_g *= flash_decay ** (dt * 30.0)
    if flash_g < 0.005:
        flash_g = 0.0
        # One more row to show the zero
        r = trace_pipeline(flash_g, 0.5, is_flash=False,
                           label=f"t={(step+1)*dt:.2f}s (zeroed)")
        print_row(r)
        break


# =============================================================================
# PART 3: CRITICAL THRESHOLD ANALYSIS
# =============================================================================
print("\n\n" + "=" * 130)
print("PART 3: CRITICAL THRESHOLD ANALYSIS")
print("=" * 130)
print("\nFinding the minimum glow that passes the noise gate (for breathing, hueT=0.5)...")

hueT_val = 0.5
# Binary search for noise gate threshold
lo, hi = 0.0, 1.0
for _ in range(50):
    mid = (lo + hi) / 2
    r = trace_pipeline(mid, hueT_val, is_flash=False)
    if r["passes_gate"]:
        hi = mid
    else:
        lo = mid

gate_threshold_glow = hi
print(f"\n  Minimum glow to pass noise gate: {gate_threshold_glow:.6f}")

# What breath value produces this glow?
# breathGlow = BREATH_FLOOR + breath * (breathPeak - BREATH_FLOOR)
# Solving: breath = (glow - BREATH_FLOOR) / (breathPeak - BREATH_FLOOR)
for bp_label, bp in [("minPeak=0.50", 0.50), ("maxPeak=0.70", 0.70)]:
    breath_needed = (gate_threshold_glow - BREATH_FLOOR) / (bp - BREATH_FLOOR)
    if breath_needed > 1.0:
        print(f"  With {bp_label}: NEVER passes gate (needs breath={breath_needed:.3f} > 1.0)")
    elif breath_needed < 0.0:
        print(f"  With {bp_label}: ALWAYS passes gate (floor alone is sufficient)")
    else:
        # breath = sin(phase*2pi)*0.5+0.5, so sin = (breath-0.5)/0.5 = 2*breath - 1
        sin_val = 2 * breath_needed - 1
        if abs(sin_val) <= 1.0:
            phase_rad = math.asin(sin_val)
            phase_frac = phase_rad / (2 * math.pi)
            # fraction of cycle above threshold = time between the two crossing points
            # The sin is above sin_val for phase in [asin(sin_val), pi - asin(sin_val)]
            # In our phase space that means...
            upper_phase = math.pi - phase_rad
            duty = (upper_phase - phase_rad) / (2 * math.pi)
            print(f"  With {bp_label}: passes gate when breath >= {breath_needed:.4f}"
                  f" ({duty*100:.1f}% of cycle visible)")
        else:
            print(f"  With {bp_label}: breath_needed={breath_needed:.4f}")


# =============================================================================
# PART 4: GLOW SWEEP — fine-grained view of the gamma crush zone
# =============================================================================
print("\n\n" + "=" * 130)
print("PART 4: GLOW SWEEP (hueT=0.5, breathing color)")
print("=" * 130)
print("\nFine-grained glow sweep showing where values live:")
print_header()

glow_values = [
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 1.00,
]

for g in glow_values:
    r = trace_pipeline(g, 0.5, is_flash=False, label=f"glow={g:.2f}")
    print_row(r)


# =============================================================================
# PART 5: DELTA-SIGMA DITHERING SIMULATION
# =============================================================================
print("\n\n" + "=" * 130)
print("PART 5: DELTA-SIGMA DITHERING OVER MULTIPLE FRAMES")
print("=" * 130)
print("\nThe delta-sigma ditherer spreads sub-LSB values over time.")
print("Simulating 256 frames at steady glow levels to show effective brightness:\n")


def simulate_delta_sigma(target16, n_frames=256):
    """Simulate delta-sigma dithering, return average 8-bit output."""
    accum = 128  # typical initial accumulator
    total = 0
    outputs = []
    for _ in range(n_frames):
        accum = (accum + target16) & 0xFFFF
        out = accum >> 8
        accum &= 0xFF
        total += out
        outputs.append(out)
    return total / n_frames, outputs


print(f"{'glow':>6} {'tB16':>6} {'avg B8 (256fr)':>15} {'output pattern (first 16 frames)':>40}")
print("-" * 75)

test_glows = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
for g in test_glows:
    r = trace_pipeline(g, 0.5, is_flash=False)
    if r["passes_gate"]:
        avg, outs = simulate_delta_sigma(r["tB16"], 256)
        pattern = " ".join(str(o) for o in outs[:16])
        print(f"{g:>6.2f} {r['tB16']:>6} {avg:>15.2f}   {pattern}")
    else:
        print(f"{g:>6.2f} {r['tB16']:>6}  (gated to 0)      ---")


# =============================================================================
# PART 6: SUMMARY — WHERE VALUES GET CRUSHED
# =============================================================================
print("\n\n" + "=" * 130)
print("PART 6: BOTTLENECK ANALYSIS SUMMARY")
print("=" * 130)

# Compute some key numbers
peak_glow_min = BREATH_FLOOR + 1.0 * (BREATH_MIN_PEAK - BREATH_FLOOR)  # = BREATH_MIN_PEAK
peak_glow_max = BREATH_FLOOR + 1.0 * (BREATH_MAX_PEAK - BREATH_FLOOR)  # = BREATH_MAX_PEAK
trough_glow = BREATH_FLOOR

peak_linBright_min = peak_glow_min ** GAMMA * BRIGHTNESS_CAP
peak_linBright_max = peak_glow_max ** GAMMA * BRIGHTNESS_CAP
trough_linBright = trough_glow ** GAMMA * BRIGHTNESS_CAP

# Max possible color channel values at breathing peak
colB_mid = lerpf(HUE_A_B, HUE_B_B, 0.5)  # 105
peak_oB_min = colB_mid * peak_linBright_min
peak_oB_max = colB_mid * peak_linBright_max
trough_oB = colB_mid * trough_linBright

peak_tB16_min = int(peak_oB_min * 256)
peak_tB16_max = int(peak_oB_max * 256)
trough_tB16 = int(trough_oB * 256)

flash_linBright = FLASH_PEAK ** GAMMA * BRIGHTNESS_CAP
flash_oB = FLASH_B * flash_linBright
flash_tB16 = int(flash_oB * 256)

print(f"""
PIPELINE STAGE ANALYSIS:

1. GAMMA COMPRESSION (gamma={GAMMA}):
   - glow=0.03 (trough) -> {0.03**GAMMA:.6f}    (crushed to ~0.04% of original)
   - glow=0.50 (minPeak) -> {0.50**GAMMA:.6f}    (crushed to ~19% of original)
   - glow=0.70 (maxPeak) -> {0.70**GAMMA:.6f}    (crushed to ~42% of original)
   - glow=0.85 (flash)   -> {0.85**GAMMA:.6f}    (crushed to ~72% of original)
   Gamma 2.4 is standard sRGB — this compensates for LED nonlinearity.

2. BRIGHTNESS CAP ({BRIGHTNESS_CAP}):
   - Multiplied after gamma, so peak linBright = glow^gamma * {BRIGHTNESS_CAP}
   - At breathPeak=0.50: linBright = {peak_linBright_min:.6f}
   - At breathPeak=0.70: linBright = {peak_linBright_max:.6f}
   - At flash peak=0.85: linBright = {flash_linBright:.6f}
   - At trough (0.03):   linBright = {trough_linBright:.8f}

3. COLOR MULTIPLY (colB_mid={colB_mid:.0f}):
   - Peak oB (minPeak): {peak_oB_min:.4f}
   - Peak oB (maxPeak): {peak_oB_max:.4f}
   - Flash peak oB:     {flash_oB:.4f}
   - Trough oB:         {trough_oB:.6f}

4. 16-BIT SCALE (* 256):
   - Peak tB16 (minPeak): {peak_tB16_min}   (8-bit: {peak_tB16_min >> 8})
   - Peak tB16 (maxPeak): {peak_tB16_max}   (8-bit: {peak_tB16_max >> 8})
   - Flash peak tB16:     {flash_tB16}  (8-bit: {flash_tB16 >> 8})
   - Trough tB16:         {trough_tB16}     (8-bit: {trough_tB16 >> 8})

5. NOISE GATE (threshold={NOISE_GATE}):
   - Trough tB16={trough_tB16}: {"KILLED" if trough_tB16 < NOISE_GATE else "passes"}
   - Gate threshold glow: ~{gate_threshold_glow:.4f}

6. DYNAMIC RANGE SUMMARY:
   - Breathing range: glow {BREATH_FLOOR:.2f} to {BREATH_MAX_PEAK:.2f}
   - After gamma+cap+color+scale: {trough_tB16} to {peak_tB16_max} (16-bit B channel)
   - After noise gate: 0 to {peak_tB16_max}
   - Effective 8-bit range: 0 to {peak_tB16_max >> 8}
   - Flash 8-bit peak: {flash_tB16 >> 8}

   The 16-bit delta-sigma dithering rescues sub-8-bit values:
   - A tB16 of {peak_tB16_min} averages to ~{peak_tB16_min/256:.1f} effective 8-bit brightness
   - This is WHERE the smoothness lives — the ditherer is doing real work
""")

# Final: what fraction of the breathing cycle is visible (passes noise gate)?
print("VISIBILITY DUTY CYCLE (fraction of breath cycle that passes noise gate):")
for bp_label, bp in [("breathPeak=0.50 (dimmest)", 0.50),
                       ("breathPeak=0.60 (mid)", 0.60),
                       ("breathPeak=0.70 (brightest)", 0.70)]:
    n_visible = 0
    n_total = 1000
    for i in range(n_total):
        phase = i / n_total
        breath = math.sin(phase * 2.0 * math.pi) * 0.5 + 0.5
        breathGlow = BREATH_FLOOR + breath * (bp - BREATH_FLOOR)
        r = trace_pipeline(breathGlow, 0.5, is_flash=False)
        if r["passes_gate"]:
            n_visible += 1
    pct = n_visible / n_total * 100
    print(f"  {bp_label}: {pct:.1f}% of cycle visible")

print()
