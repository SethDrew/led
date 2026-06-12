#!/usr/bin/env python3
"""Replay a rec_N.audio clip through the firmware's audio feature pipeline.

Mirrors playbackAudioTick + audioDeriveFeaturesRms in main.cpp exactly
(8 kHz unsigned bytes, RMS in the 24-bit domain via <<WF_QUANT_SHIFT).
Prints a timeline of RMS / floor / energy and every onset the detector
fires, under selectable gate variants, so thresholds can be tuned against
real field recordings instead of guesses.

Usage: sim_onset.py rec_0.audio [--gate none|presence] [--risefrac 0.5]
"""

import argparse
import math
import sys

RATE = 8000
QUANT_SHIFT = 11
DT = 1.0 / 60.0                 # firmware render tick

SNS_RMS_FLOOR_MIN = 20000.0
SNS_RMS_CEIL_MIN = 130000.0
SNS_ONSET_FLOOR = 6000.0
SNS_FLOOR_HEADROOM = 1.4
SNS_FLOOR_LEAK = 0.005
SNS_FLOOR_SNAP_EPS = 0.05
SNS_FLOOR_SOFT_SIG = 0.6


def simulate(samples, gate, risefrac, refrac):
    floor = 0.0
    ceil = SNS_RMS_CEIL_MIN
    below_cnt = 0
    fast = slow = lock = 0.0
    t = 0.0
    idx = 0
    n_per_tick = int(RATE * DT + 0.5)

    ticks = []      # (t, rms, floor_eff, energy, fast)
    onsets = []     # (t, strength, rms, fast, slow)

    while idx < len(samples):
        chunk = samples[idx:idx + n_per_tick]
        idx += n_per_tick
        if not chunk:
            break
        acc = [(b - 128) << QUANT_SHIFT for b in chunk]
        rms = math.sqrt(sum(a * a for a in acc) / len(acc))

        # ── ceiling ──
        ceil = max(SNS_RMS_CEIL_MIN, ceil * math.exp(-0.0025 * DT))
        if rms > ceil:
            ceil = rms

        # ── adaptive floor (snsUpdateFloor) ──
        if floor < 1.0:
            floor = max(rms, SNS_RMS_FLOOR_MIN)
        elif rms < floor * (1.0 + SNS_FLOOR_SNAP_EPS):
            below_cnt += 1
            if below_cnt >= 3:
                target = max(rms, SNS_RMS_FLOOR_MIN)
                floor += min(1.0, DT / 0.11) * (target - floor)
        else:
            below_cnt = 0
            ratio = rms / max(floor, 1.0)
            d = (ratio - 1.0) / SNS_FLOOR_SOFT_SIG
            floor *= 1.0 + SNS_FLOOR_LEAK * DT * math.exp(-(d * d))
        floor = max(floor, SNS_RMS_FLOOR_MIN)
        eff_floor = floor * SNS_FLOOR_HEADROOM

        # ── energy ──
        if rms < eff_floor:
            energy = 0.0
        else:
            db = 20.0 * math.log10(rms / eff_floor)
            db_range = max(1.0, 20.0 * math.log10(ceil / eff_floor))
            energy = min(1.0, max(0.0, db / db_range))

        # ── onset ──
        fast += min(1.0, DT / 0.030) * (rms - fast)
        slow += min(1.0, DT / 0.300) * (rms - slow)
        rise = fast - slow
        thresh = max(SNS_ONSET_FLOOR, risefrac * slow)
        lock = max(0.0, lock - DT)
        present = True
        if gate == "presence":
            present = fast > eff_floor
        if present and rise > thresh and lock <= 0.0:
            lock = refrac
            strength = min(1.0, max(0.0, rise / max(slow, 1.0)))
            onsets.append((t, strength, rms, fast, slow))

        ticks.append((t, rms, eff_floor, energy, fast))
        t += DT

    return ticks, onsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--gate", default="presence", choices=["none", "presence"])
    ap.add_argument("--risefrac", type=float, default=0.5)
    ap.add_argument("--refrac", type=float, default=0.120)
    ap.add_argument("--bucket", type=float, default=0.25)
    args = ap.parse_args()

    samples = open(args.clip, "rb").read()
    print(f"{args.clip}: {len(samples)} samples = {len(samples)/RATE:.2f}s "
          f"| gate={args.gate} risefrac={args.risefrac}")

    ticks, onsets = simulate(samples, args.gate, args.risefrac, args.refrac)

    print(f"\n{'t':>5}  {'rms_avg':>8} {'rms_max':>8} {'effFloor':>8} "
          f"{'energy':>6}  onsets")
    nb = int(args.bucket / DT)
    for b in range(0, len(ticks), nb):
        chunk = ticks[b:b + nb]
        t0 = chunk[0][0]
        rms_avg = sum(c[1] for c in chunk) / len(chunk)
        rms_max = max(c[1] for c in chunk)
        floor_avg = sum(c[2] for c in chunk) / len(chunk)
        e_max = max(c[3] for c in chunk)
        fired = [o for o in onsets if t0 <= o[0] < t0 + args.bucket]
        marks = " ".join(f"*{o[1]:.2f}" for o in fired)
        print(f"{t0:5.2f}  {rms_avg:8.0f} {rms_max:8.0f} {floor_avg:8.0f} "
              f"{e_max:6.2f}  {marks}")

    print(f"\n{len(onsets)} onsets fired:")
    for t, s, rms, fast, slow in onsets:
        print(f"  t={t:5.2f}s  strength={s:.2f}  rms={rms:7.0f}  "
              f"fast={fast:7.0f}  slow={slow:7.0f}")


if __name__ == "__main__":
    main()
