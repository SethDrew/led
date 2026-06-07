#!/usr/bin/env python3
"""Drill into the lone-flash population under the CURRENT policy, and quantify
brightness loss from snap vs floor. Run at one brightness."""
import subprocess, sys, csv, io
from collections import defaultdict

SIM = "./bloom_dither_sim"
CHANNELS = ("R", "G", "B")


def run(b, policy, frames):
    out = subprocess.run([SIM, "--brightness", str(b), "--policy", policy,
                          "--frames", str(frames)],
                         capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(out)))


def main():
    b = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    # CURRENT: classify each pulse by the gap that PRECEDES it. A "lone flash"
    # is an isolated out==1 with a long dark gap on both sides.
    rows = run(b, "current", frames)
    pulses = defaultdict(list)
    for r in rows:
        f = int(r["frame"]); p = int(r["pixel"])
        for ch in CHANNELS:
            if int(r[f"{ch.lower()}8"]) >= 1:
                pulses[(p, ch)].append(f)

    lone = 0          # pulses isolated by >40-frame (200ms) gaps on both sides
    pixchan_with_lone = set()
    for key, fs in pulses.items():
        for i, f in enumerate(fs):
            prev_gap = f - fs[i-1] if i > 0 else 10**9
            next_gap = fs[i+1] - f if i+1 < len(fs) else 10**9
            if prev_gap > 40 and next_gap > 40:
                lone += 1
                pixchan_with_lone.add(key)
    print(f"brightness={b}")
    print(f"  CURRENT lone flashes (isolated >200ms both sides): {lone}")
    print(f"  pixel-channels exhibiting >=1 lone flash: {len(pixchan_with_lone)} / 300")

    # Brightness loss: mean per-pixel total 8-bit energy, current vs snap vs floor.
    def total_energy(policy):
        rs = run(b, policy, frames)
        tot = 0
        for r in rs:
            tot += int(r["r8"]) + int(r["g8"]) + int(r["b8"])
        return tot / frames  # per-frame total across all 100 px

    e_cur = total_energy("current")
    e_snap = total_energy("snap")
    e_floor = total_energy("floor")
    print(f"  per-frame total 8-bit energy (100px x 3ch):")
    print(f"    current={e_cur:.1f}  snap={e_snap:.1f} ({100*(e_snap-e_cur)/e_cur:+.1f}%)"
          f"  floor={e_floor:.1f} ({100*(e_floor-e_cur)/e_cur:+.1f}%)")


if __name__ == "__main__":
    main()
