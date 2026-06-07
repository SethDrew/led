#!/usr/bin/env python3
"""Analyze bloom dither sim CSV for low-brightness flicker.

Runs the sim for a given brightness across all three gate policies and reports:
  - danger-zone occupancy (0 < t16 < 256) per frame
  - inter-pulse interval distribution for danger-zone channels
  - mean brightness per channel (to check the fix doesn't darken)
"""
import subprocess, sys, csv, io
from collections import defaultdict

SIM = "./bloom_dither_sim"
LEDS = 100
CHANNELS = ("R", "G", "B")


def run(brightness, policy, frames):
    out = subprocess.run(
        [SIM, "--brightness", str(brightness), "--policy", policy, "--frames", str(frames)],
        capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(out)))


def analyze(rows, frames):
    n = len(rows)
    danger_cells = 0                       # rows*channels with 0<t16<256
    pulse_frames = defaultdict(list)       # (pixel,ch) -> [frame indices where out==1]
    out_sum = defaultdict(int)             # (pixel,ch) -> sum of 8-bit output
    danger_pixchan = set()                 # (pixel,ch) that ever entered danger zone

    for r in rows:
        f = int(r["frame"]); p = int(r["pixel"])
        for ch in CHANNELS:
            t16 = int(r[f"t16{ch}"])
            o8 = int(r[f"{ch.lower()}8"])
            out_sum[(p, ch)] += o8
            if 0 < t16 < 256:
                danger_cells += 1
                danger_pixchan.add((p, ch))
            if o8 >= 1:
                pulse_frames[(p, ch)].append(f)

    # inter-pulse intervals, restricted to channels that spent time in danger zone
    intervals = []
    for key in danger_pixchan:
        fs = pulse_frames.get(key, [])
        for a, b in zip(fs, fs[1:]):
            intervals.append(b - a)

    mean_bright = {ch: sum(out_sum[(p, ch)] for p in range(LEDS)) / (LEDS * frames)
                   for ch in CHANNELS}

    return {
        "danger_per_frame": danger_cells / frames,
        "danger_pixchan": len(danger_pixchan),
        "intervals": intervals,
        "mean_bright": mean_bright,
    }


def pct(vals, q):
    if not vals: return float("nan")
    s = sorted(vals); k = int(q * (len(s) - 1)); return s[k]


def main():
    brightness = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    print(f"=== brightness={brightness}  frames={frames}  (200 fps) ===\n")
    for policy in ("current", "snap", "floor"):
        rows = run(brightness, policy, frames)
        a = analyze(rows, frames)
        iv = a["intervals"]
        # flicker = lone pulses spaced far apart. >13 frames @200fps => <15 Hz pulse rate.
        slow = sum(1 for x in iv if x > 13)
        print(f"--- policy: {policy} ---")
        print(f"  danger-zone (0<t16<256) cells/frame : {a['danger_per_frame']:.1f}"
              f"   (of {LEDS*3} pix-channels)")
        print(f"  distinct pix-channels ever in danger: {a['danger_pixchan']}")
        if iv:
            print(f"  inter-pulse interval (frames): "
                  f"med={pct(iv,0.5)} p90={pct(iv,0.9)} p99={pct(iv,0.99)} max={max(iv)}")
            print(f"  pulses spaced >13 frames (visible-flicker risk): "
                  f"{slow}/{len(iv)} ({100*slow/len(iv):.1f}%)")
        else:
            print("  no pulses in danger-zone channels")
        mb = a["mean_bright"]
        print(f"  mean 8-bit out  R={mb['R']:.4f} G={mb['G']:.4f} B={mb['B']:.4f}\n")


if __name__ == "__main__":
    main()
