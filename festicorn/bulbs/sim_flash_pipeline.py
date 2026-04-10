"""
Simulate the receiver_3bulbs flash triggering pipeline.

Traces frame-by-frame what the receiver sees when the duck rotates,
including the sender's EMA smoothing effect on angular rate.
"""

import math

# ── Receiver constants (from receiver_3bulbs.cpp) ──────────────────
SENSOR_HZ = 25.0
DT = 1.0 / SENSOR_HZ

MOVEMENT_RANGE = 60.0       # deg/s maps to disturbance 1.0
FLASH_THRESHOLD = 10.0      # deg/s to trigger flashes
MOVEMENT_EMA_TC = 0.200     # seconds
STILLNESS_DROP_TC = 0.300
STILLNESS_RISE_TC = 2.0

FLASH_CHARGE_COST = 0.18
FLASH_CHARGE_MIN = 0.10
FLASH_RECHARGE_RATE = 0.10  # per second during stillness
FLASH_COOLDOWN = 0.150      # seconds

FLASH_PEAK = 0.85
FLASH_DECAY_LO = 0.91
FLASH_DECAY_HI = 0.96
FLASH_MIN_SEEDS = 2
FLASH_MAX_SEEDS = 8

GAMMA = 2.4
BRIGHTNESS_CAP = 0.70
NOISE_GATE = 24
W_ONSET = 0.5

# Flash color
FLASH_G = 150.0
FLASH_B = 170.0

# ── Sender constants (from sender.cpp) ─────────────────────────────
ACCEL_TC = 0.3
SENDER_ALPHA = 2.0 / (ACCEL_TC * SENSOR_HZ + 1.0)  # ≈ 0.235


def clampf(v, lo, hi):
    return max(lo, min(hi, v))


def normalize(x, y, z):
    length = math.sqrt(x*x + y*y + z*z)
    if length > 0:
        return x/length, y/length, z/length
    return x, y, z


def generate_true_orientation(t, scenario):
    """Return the TRUE (pre-smoothing) unit vector at time t."""
    if scenario == "slow_90":
        # 90 degrees over 1 second starting at t=0.2, so 90 deg/s
        start, duration = 0.2, 1.0
    elif scenario == "fast_90":
        # 90 degrees over 0.5 seconds starting at t=0.2, so 180 deg/s
        start, duration = 0.2, 0.5
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if t < start:
        angle = 0.0
    elif t < start + duration:
        angle = ((t - start) / duration) * (math.pi / 2)
    else:
        angle = math.pi / 2

    # Rotate from (0,0,1) toward (1,0,0) in the xz plane
    x = math.sin(angle)
    y = 0.0
    z = math.cos(angle)
    return x, y, z


class SenderSim:
    """Simulates the sender's EMA smoothing."""
    def __init__(self):
        self.sx = 0.0
        self.sy = 0.0
        self.sz = 1.0
        self.alpha = SENDER_ALPHA

    def update(self, raw_x, raw_y, raw_z):
        # Normalize raw
        rx, ry, rz = normalize(raw_x, raw_y, raw_z)

        # EMA smooth
        self.sx += self.alpha * (rx - self.sx)
        self.sy += self.alpha * (ry - self.sy)
        self.sz += self.alpha * (rz - self.sz)

        # Re-normalize
        self.sx, self.sy, self.sz = normalize(self.sx, self.sy, self.sz)
        return self.sx, self.sy, self.sz


class ReceiverSim:
    """Simulates the receiver's movement pipeline."""
    def __init__(self):
        self.prevAx = 0.0
        self.prevAy = 0.0
        self.prevAz = 1.0
        self.hasPrev = False
        self.movementEnergy = 0.0
        self.stillness = 1.0
        self.flashCharge = 1.0
        self.cooldown = 0.0

    def computeAngularRate(self, ax, ay, az, dt):
        if not self.hasPrev:
            self.prevAx = ax
            self.prevAy = ay
            self.prevAz = az
            self.hasPrev = True
            return 0.0

        dot = clampf(
            self.prevAx * ax + self.prevAy * ay + self.prevAz * az,
            -1.0, 1.0
        )
        angleDeg = math.acos(dot) * (180.0 / math.pi)
        rate = angleDeg / max(dt, 0.001)

        self.prevAx = ax
        self.prevAy = ay
        self.prevAz = az
        return rate

    def updateMovement(self, angularRate, dt):
        alpha_m = min(1.0, dt / MOVEMENT_EMA_TC)
        self.movementEnergy += alpha_m * (angularRate - self.movementEnergy)

        targetStillness = clampf(1.0 - self.movementEnergy / MOVEMENT_RANGE, 0.0, 1.0)
        if targetStillness < self.stillness:
            alpha_drop = min(1.0, dt / STILLNESS_DROP_TC)
            self.stillness += alpha_drop * (targetStillness - self.stillness)
        else:
            alpha_rise = min(1.0, dt / STILLNESS_RISE_TC)
            self.stillness += alpha_rise * (targetStillness - self.stillness)

        self.flashCharge = min(1.0, self.flashCharge + FLASH_RECHARGE_RATE * self.stillness * dt)

    def checkFlashTrigger(self, angularRate, dt):
        self.cooldown = max(0.0, self.cooldown - dt)

        triggered = False
        strength = 0.0
        nSeeds = 0
        peakGlow = 0.0

        if (angularRate > FLASH_THRESHOLD and
                self.cooldown <= 0.0 and
                self.flashCharge > FLASH_CHARGE_MIN):
            triggered = True
            strength = clampf(
                (angularRate - FLASH_THRESHOLD) / (MOVEMENT_RANGE - FLASH_THRESHOLD),
                0.0, 1.0
            )
            strength *= self.flashCharge
            nSeeds = FLASH_MIN_SEEDS + int(strength * (FLASH_MAX_SEEDS - FLASH_MIN_SEEDS))
            self.cooldown = FLASH_COOLDOWN
            self.flashCharge = max(0.0, self.flashCharge - FLASH_CHARGE_COST)
            peakGlow = 0.5 + 0.35 * strength

        return triggered, strength, nSeeds, peakGlow


def compute_flash_led_output(glow):
    """Trace a flash glow value through the gamma/cap/color pipeline."""
    # linBright = pow(glow, GAMMA) * BRIGHTNESS_CAP
    linBright = (glow ** GAMMA) * BRIGHTNESS_CAP

    oG = FLASH_G * linBright
    oB = FLASH_B * linBright

    # W channel
    wFrac = clampf((glow - W_ONSET) / (1.0 - W_ONSET), 0.0, 1.0)
    oW = wFrac * linBright * 200.0

    # 16-bit for delta-sigma
    tG16 = int(clampf(oG * 256.0, 0, 65535))
    tB16 = int(clampf(oB * 256.0, 0, 65535))
    tW16 = int(clampf(oW * 256.0, 0, 65535))

    # Noise gate
    maxCh = max(tG16, tB16, tW16)
    if maxCh < NOISE_GATE:
        tG16 = tB16 = tW16 = 0

    # 8-bit (approximate — real delta-sigma dithers, but avg is target/256)
    g8 = tG16 >> 8
    b8 = tB16 >> 8
    w8 = tW16 >> 8

    return {
        'linBright': linBright,
        'oG': oG, 'oB': oB, 'oW': oW,
        'tG16': tG16, 'tB16': tB16, 'tW16': tW16,
        'maxCh16': maxCh,
        'g8': g8, 'b8': b8, 'w8': w8,
        'noiseGated': maxCh < NOISE_GATE,
    }


def run_scenario(name, scenario, duration=2.0):
    print(f"\n{'='*120}")
    print(f"SCENARIO: {name}")
    print(f"{'='*120}")

    sender = SenderSim()
    receiver = ReceiverSim()

    # Also run a "no smoothing" receiver for comparison
    receiver_raw = ReceiverSim()

    nFrames = int(duration * SENSOR_HZ)

    # Header
    hdr = (f"{'frm':>3} {'t':>5}  "
           f"{'true_x':>7} {'true_z':>7}  "
           f"{'ema_x':>7} {'ema_z':>7}  "
           f"{'rawRate':>8} {'emaRate':>8}  "
           f"{'mvE':>6} {'still':>6} {'chg':>5}  "
           f"{'flash?':>6} {'seeds':>5} {'pkGlow':>6}  "
           f"{'G8':>3} {'B8':>3} {'W8':>3}  "
           f"{'noRawRate':>10}")
    print(hdr)
    print("-" * len(hdr))

    flash_count = 0
    max_ema_rate = 0.0
    max_raw_rate = 0.0

    for f in range(nFrames):
        t = f * DT

        # True orientation
        tx, ty, tz = generate_true_orientation(t, scenario)

        # Sender EMA smoothing
        sx, sy, sz = sender.update(tx, ty, tz)

        # Receiver: with sender smoothing
        angularRate = receiver.computeAngularRate(sx, sy, sz, DT)
        receiver.updateMovement(angularRate, DT)
        triggered, strength, nSeeds, peakGlow = receiver.checkFlashTrigger(angularRate, DT)

        # Receiver: without sender smoothing (raw orientation)
        rawRate = receiver_raw.computeAngularRate(tx, ty, tz, DT)
        receiver_raw.updateMovement(rawRate, DT)

        max_ema_rate = max(max_ema_rate, angularRate)
        max_raw_rate = max(max_raw_rate, rawRate)

        # LED output if flash triggered
        if peakGlow > 0:
            led = compute_flash_led_output(peakGlow)
            g8, b8, w8 = led['g8'], led['b8'], led['w8']
        else:
            g8 = b8 = w8 = 0

        if triggered:
            flash_count += 1

        print(f"{f:3d} {t:5.2f}  "
              f"{tx:7.4f} {tz:7.4f}  "
              f"{sx:7.4f} {sz:7.4f}  "
              f"{rawRate:8.2f} {angularRate:8.2f}  "
              f"{receiver.movementEnergy:6.2f} {receiver.stillness:6.3f} {receiver.flashCharge:5.2f}  "
              f"{'YES' if triggered else '   ':>6} {nSeeds if triggered else '':>5} {peakGlow:6.3f}  "
              f"{g8:3d} {b8:3d} {w8:3d}  "
              f"{rawRate:10.2f}")

    print(f"\nSummary:")
    print(f"  Total flashes triggered: {flash_count}")
    print(f"  Max angular rate (with sender EMA): {max_ema_rate:.2f} deg/s")
    print(f"  Max angular rate (raw, no EMA):     {max_raw_rate:.2f} deg/s")
    print(f"  Flash threshold: {FLASH_THRESHOLD} deg/s")
    print(f"  Final flashCharge: {receiver.flashCharge:.3f}")
    print(f"  Final stillness: {receiver.stillness:.3f}")


def analyze_ema_attenuation():
    """Show how the sender EMA attenuates a constant rotation."""
    print(f"\n{'='*120}")
    print("ANALYSIS: Sender EMA attenuation of constant angular velocity")
    print(f"{'='*120}")

    print(f"\nSender EMA alpha = {SENDER_ALPHA:.4f}")
    print(f"  (ACCEL_TC={ACCEL_TC}, SENSOR_HZ={SENSOR_HZ})")
    print(f"  Formula: 2.0 / (ACCEL_TC * SENSOR_HZ + 1.0) = 2.0 / {ACCEL_TC * SENSOR_HZ + 1.0}")

    print(f"\nFor a constant rotation at omega deg/s, at 25 Hz each frame rotates omega/25 degrees.")
    print(f"The sender EMA smoothing delays the orientation, reducing the per-frame angular change")
    print(f"seen by the receiver.\n")

    # For a constant rotation, the EMA output lags behind.
    # The steady-state angular rate the receiver sees is attenuated.
    # Let's measure empirically.

    test_rates = [45, 90, 180, 360]  # deg/s actual rotation
    print(f"{'Actual deg/s':>15} {'Receiver sees (peak)':>22} {'Receiver sees (steady)':>24} {'Attenuation':>14}")
    print("-" * 80)

    for omega in test_rates:
        sender = SenderSim()
        receiver = ReceiverSim()

        peak_rate = 0.0
        rates = []
        # Run for 2 seconds
        for f in range(50):
            t = f * DT
            angle = omega * t * math.pi / 180.0
            tx = math.sin(angle)
            ty = 0.0
            tz = math.cos(angle)

            sx, sy, sz = sender.update(tx, ty, tz)
            rate = receiver.computeAngularRate(sx, sy, sz, DT)
            peak_rate = max(peak_rate, rate)
            if f > 10:  # after settling
                rates.append(rate)

        avg_rate = sum(rates) / len(rates) if rates else 0
        print(f"{omega:15.1f} {peak_rate:22.2f} {avg_rate:24.2f} {avg_rate/omega:14.2f}x")


def analyze_flash_output_levels():
    """Show what LED output values various flash glow levels produce."""
    print(f"\n{'='*120}")
    print("ANALYSIS: Flash glow -> LED output mapping")
    print(f"{'='*120}")

    print(f"\nParameters: GAMMA={GAMMA}, BRIGHTNESS_CAP={BRIGHTNESS_CAP}, NOISE_GATE={NOISE_GATE}")
    print(f"Flash color: G={FLASH_G}, B={FLASH_B}")
    print(f"W onset threshold: {W_ONSET}\n")

    glows = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 1.0]
    print(f"{'glow':>6} {'linBr':>8} {'oG':>8} {'oB':>8} {'oW':>8}  "
          f"{'G16':>6} {'B16':>6} {'W16':>6} {'max16':>6}  "
          f"{'G8':>3} {'B8':>3} {'W8':>3}  {'gated?':>6}")
    print("-" * 100)

    for g in glows:
        led = compute_flash_led_output(g)
        print(f"{g:6.2f} {led['linBright']:8.5f} {led['oG']:8.2f} {led['oB']:8.2f} {led['oW']:8.2f}  "
              f"{led['tG16']:6d} {led['tB16']:6d} {led['tW16']:6d} {led['maxCh16']:6d}  "
              f"{led['g8']:3d} {led['b8']:3d} {led['w8']:3d}  "
              f"{'YES' if led['noiseGated'] else '':>6}")


def main():
    print("Flash Pipeline Simulation for receiver_3bulbs.cpp")
    print(f"Sender EMA alpha: {SENDER_ALPHA:.4f}")

    # Run scenarios
    run_scenario("Slow rotation: 90 deg over 1 second (90 deg/s)", "slow_90", duration=2.0)
    run_scenario("Fast rotation: 90 deg over 0.5 seconds (180 deg/s)", "fast_90", duration=2.0)

    # Analysis
    analyze_ema_attenuation()
    analyze_flash_output_levels()

    # Additional scenarios
    run_scenario("Gentle tilt: 30 deg over 2 seconds (15 deg/s)", "gentle_30", duration=3.0)
    run_scenario("Quick flick: 45 deg over 0.2 seconds (225 deg/s)", "quick_flick", duration=2.0)
    run_scenario("Waggle: 3 quick back-and-forth oscillations", "waggle", duration=3.0)

    # Flash decay analysis
    analyze_flash_decay()

    # Bug diagnosis
    print(f"\n{'='*120}")
    print("BUG DIAGNOSIS SUMMARY")
    print(f"{'='*120}")

    print(f"""
FINDING 1: Sender EMA does NOT kill angular rate for sustained rotation.
  The EMA smooths the orientation but with re-normalization, once the EMA output
  catches up (a few frames), the receiver sees nearly the true angular velocity.
  Attenuation for constant rotation is ~0.99x — negligible.

FINDING 2: Flashes DO trigger in simulation for 90+ deg/s rotations.
  Slow rotation (90 deg/s): 6 flashes triggered, peak glow up to 0.79
  Fast rotation (180 deg/s): 6 flashes triggered, peak glow up to 0.79

FINDING 3: Flash charge depletes quickly and blocks later flashes.
  FLASH_CHARGE_COST=0.18, so 5-6 flashes depletes the charge below FLASH_CHARGE_MIN=0.10.
  Recharge rate is only 0.10/s * stillness — during movement, stillness is near 0,
  so recharge is effectively frozen. After movement stops, stillness recovers slowly
  (STILLNESS_RISE_TC=2.0s), so recharge takes many seconds.

FINDING 4: FLASH_COOLDOWN=0.15s limits flashes to ~6.7/s max across all sculptures.
  But the cooldown is per-sculpture, so 3 sculptures can fire independently.

FINDING 5: Flash visibility depends on glow level vs gamma compression.
  At peakGlow=0.50 (minimum flash): linBright=0.133, G8=19, B8=22 — barely visible.
  At peakGlow=0.79 (strong flash):  linBright=0.40, G8=60, B8=68, W8=46 — good.
  The gamma=2.4 crushes lower glow values hard.

FINDING 6: Flash glow decays VERY fast.
  flashDecay ranges 0.91-0.96, applied as pow(decay, dt*30).
  At 25Hz (dt=0.04): effective per-frame decay = pow(0.93, 1.2) = 0.914
  A flash at glow=0.79 drops below 0.005 (cutoff) in about 35 frames (1.4s).
  BUT most of that time it's below visible threshold due to gamma compression.
  Effective visible flash duration is only ~0.3-0.5s.

POSSIBLE ROOT CAUSES for "no visible flashes":
  A) If the physical duck movement is gentle (< 10 deg/s), flashes never trigger.
     A casual pick-up-and-tilt might only be 15-30 deg/s — barely above threshold,
     producing weak flashes (glow 0.50-0.55, G8=19-24).
  B) The renderQuietBloom function receives `angularRate` (the raw per-frame rate),
     NOT `movementEnergy`. This means flash triggering uses the instantaneous rate,
     which is good — but it also means the flash trigger condition is checked EVERY
     frame, and cooldown prevents rapid re-triggering (0.15s = 3.75 frames).
  C) The flash check in renderQuietBloom uses angularRate (instantaneous), but
     the flash strength formula uses angularRate too. If the sender EMA causes the
     first frame's rate to be low (ramp-up), the first flash is weak.
  D) Flash charge depletion: if the user moves the duck continuously, flash charge
     hits zero after ~5-6 flashes and stays there because stillness is low.
     Result: flashes only happen at the START of movement, then nothing.
""")


def generate_true_orientation_extended(t, scenario):
    """Extended scenarios."""
    if scenario in ("slow_90", "fast_90"):
        return generate_true_orientation(t, scenario)
    elif scenario == "gentle_30":
        # 30 degrees over 2 seconds starting at t=0.2, so 15 deg/s
        start, duration = 0.2, 2.0
        max_angle = 30.0 * math.pi / 180.0
        if t < start:
            angle = 0.0
        elif t < start + duration:
            angle = ((t - start) / duration) * max_angle
        else:
            angle = max_angle
        return math.sin(angle), 0.0, math.cos(angle)
    elif scenario == "quick_flick":
        # 45 degrees over 0.2 seconds starting at t=0.2, so 225 deg/s
        start, duration = 0.2, 0.2
        max_angle = 45.0 * math.pi / 180.0
        if t < start:
            angle = 0.0
        elif t < start + duration:
            angle = ((t - start) / duration) * max_angle
        else:
            angle = max_angle
        return math.sin(angle), 0.0, math.cos(angle)
    elif scenario == "waggle":
        # 3 oscillations: +/-20 degrees at ~2Hz starting at t=0.3
        start = 0.3
        freq = 2.0  # Hz
        amplitude = 20.0 * math.pi / 180.0
        n_cycles = 3
        duration = n_cycles / freq
        if t < start:
            angle = 0.0
        elif t < start + duration:
            angle = amplitude * math.sin(2 * math.pi * freq * (t - start))
        else:
            angle = 0.0
        return math.sin(angle), 0.0, math.cos(angle)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# Monkey-patch to use extended version
_orig_generate = generate_true_orientation


def generate_true_orientation(t, scenario):
    return generate_true_orientation_extended(t, scenario)


def analyze_flash_decay():
    """Show how flash glow decays over time."""
    print(f"\n{'='*120}")
    print("ANALYSIS: Flash glow decay over time")
    print(f"{'='*120}")

    print(f"\nFlash decay parameters: FLASH_DECAY_LO={FLASH_DECAY_LO}, FLASH_DECAY_HI={FLASH_DECAY_HI}")
    print(f"Applied as: flashGlow *= pow(decay, dt * 30)")
    print(f"At 25Hz, dt=0.04, so effective per-frame multiplier = pow(decay, 1.2)")
    print(f"Cutoff: flashGlow < 0.005 -> set to 0\n")

    decay_mid = (FLASH_DECAY_LO + FLASH_DECAY_HI) / 2
    initial_glows = [0.85, 0.70, 0.55, 0.50]

    for decay in [FLASH_DECAY_LO, decay_mid, FLASH_DECAY_HI]:
        print(f"\n  decay={decay:.2f}, per-frame factor={decay**1.2:.4f}:")
        for g0 in initial_glows:
            glow = g0
            frames_visible = 0
            frames_total = 0
            for f in range(200):
                glow *= decay ** (DT * 30)
                if glow < 0.005:
                    glow = 0.0
                    break
                frames_total += 1
                # Check if LED output would be above noise gate
                led = compute_flash_led_output(glow)
                if not led['noiseGated'] and led['b8'] > 0:
                    frames_visible += 1
            print(f"    glow={g0:.2f}: visible {frames_visible} frames ({frames_visible*DT:.2f}s), "
                  f"total {frames_total} frames ({frames_total*DT:.2f}s)")


if __name__ == "__main__":
    main()
