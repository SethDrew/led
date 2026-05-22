#!/usr/bin/env python3
"""QA analysis of sim output.

Reads RGBW frame data (200 bytes/frame) and produces:
1. Summary statistics (JSON)
2. Text-based frame timeline (for LLM consumption)
3. Optional: LED-o-gram PNG

Usage:
    ./sim --effect sparkle --json < recording.bin > frames.bin 2> diag.jsonl
    python qa_analyze.py frames.bin --diag diag.jsonl --effect sparkle
"""

import argparse
import json
import struct
import sys
import numpy as np

LED_COUNT = 50
FRAME_SIZE = LED_COUNT * 4  # RGBW

def load_frames(path):
    with open(path, "rb") as f:
        data = f.read()
    n_frames = len(data) // FRAME_SIZE
    frames = np.frombuffer(data[:n_frames * FRAME_SIZE], dtype=np.uint8).reshape(n_frames, LED_COUNT, 4)
    return frames

def load_diag(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines

def compute_stats(frames, effect_name):
    n = len(frames)
    if n == 0:
        return {"error": "no frames"}

    # Per-frame brightness (sum of all channels)
    brightness = frames.astype(np.float32).sum(axis=(1, 2)) / (LED_COUNT * 4 * 255)

    # Lit pixels per frame (any channel > 0)
    lit = (frames.max(axis=2) > 0).sum(axis=1)

    # Frame-to-frame change (motion)
    if n > 1:
        diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
        motion = diffs.sum(axis=(1, 2)) / (LED_COUNT * 4 * 255)
    else:
        motion = np.array([0.0])

    # Spatial variance per frame (are pixels different from each other?)
    spatial_var = frames.astype(np.float32).std(axis=1).mean(axis=1)

    # Color analysis: dominant channel
    r_total = frames[:, :, 0].sum()
    g_total = frames[:, :, 1].sum()
    b_total = frames[:, :, 2].sum()
    w_total = frames[:, :, 3].sum()
    total = r_total + g_total + b_total + w_total
    if total > 0:
        color_mix = {"r": float(r_total/total), "g": float(g_total/total),
                     "b": float(b_total/total), "w": float(w_total/total)}
    else:
        color_mix = {"r": 0, "g": 0, "b": 0, "w": 0}

    # Dead pixels (never lit across all frames)
    ever_lit = (frames.max(axis=2) > 0).any(axis=0)
    dead_pixels = int((~ever_lit).sum())

    stats = {
        "effect": effect_name,
        "frames": n,
        "duration_s": n / 25.0,
        "brightness": {
            "mean": float(brightness.mean()),
            "max": float(brightness.max()),
            "min": float(brightness.min()),
            "std": float(brightness.std()),
        },
        "lit_pixels": {
            "mean": float(lit.mean()),
            "min": int(lit.min()),
            "max": int(lit.max()),
        },
        "motion": {
            "mean": float(motion.mean()),
            "max": float(motion.max()),
            "std": float(motion.std()),
        },
        "spatial_variance_mean": float(spatial_var.mean()),
        "color_mix": color_mix,
        "dead_pixels": dead_pixels,
    }

    # Effect-specific checks
    verdicts = []
    if effect_name in ("sparkle", "simple"):
        # Should have bursts (high motion peaks) interspersed with calm
        if motion.max() < 0.01:
            verdicts.append("FAIL: no sparkle bursts detected (motion never spikes)")
        elif motion.std() < 0.002:
            verdicts.append("WARN: very uniform motion — sparkles may not be triggering distinctly")
        else:
            verdicts.append("OK: sparkle bursts detected")
    elif effect_name == "gravity":
        # Should have spatial structure (not uniform) and respond to tilt
        if spatial_var.mean() < 1.0:
            verdicts.append("FAIL: no spatial variance — particles may be stuck")
        else:
            verdicts.append("OK: spatial variance present (particles spread)")
        if motion.mean() < 0.001:
            verdicts.append("WARN: very low motion — particles may not be moving")
    elif effect_name in ("fire", "flicker"):
        if brightness.mean() < 0.001:
            verdicts.append("FAIL: fire is dark")
        elif motion.mean() < 0.001:
            verdicts.append("FAIL: fire has no flicker")
        else:
            verdicts.append("OK: fire flickering with non-zero brightness")
    elif effect_name == "bloom":
        if brightness.mean() < 0.001:
            verdicts.append("FAIL: bloom is dark")
        else:
            verdicts.append("OK: bloom has brightness")
        if motion.std() < 0.0001:
            verdicts.append("WARN: bloom very static — breathing may not be visible")

    stats["verdicts"] = verdicts
    return stats

def text_timeline(frames, diag_lines, max_lines=50):
    """Generate a compact text timeline for LLM consumption."""
    n = len(frames)
    step = max(1, n // max_lines)
    lines = []
    lines.append(f"{'frame':>5} {'t':>6} {'lit':>3} {'avgBr':>6} {'maxCh':>5} {'rms':>5} {'bar'}")
    lines.append("-" * 60)

    for i in range(0, n, step):
        f = frames[i]
        lit = int((f.max(axis=1) > 0).sum())
        avg_br = f.mean() / 255.0
        max_ch = int(f.max())

        # Get RMS from diag if available
        rms = 0
        if i < len(diag_lines) and "rms" in diag_lines[i]:
            rms = diag_lines[i]["rms"]

        # ASCII bar showing brightness distribution across strip
        bar_width = 25
        bar = ""
        chunk = LED_COUNT // bar_width
        for c in range(bar_width):
            chunk_max = f[c*chunk:(c+1)*chunk].max()
            if chunk_max == 0:
                bar += " "
            elif chunk_max < 10:
                bar += "."
            elif chunk_max < 50:
                bar += "o"
            elif chunk_max < 150:
                bar += "O"
            else:
                bar += "#"

        t = i / 25.0
        lines.append(f"{i:5d} {t:6.2f} {lit:3d} {avg_br:6.4f} {max_ch:5d} {rms:5d} |{bar}|")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="QA analysis of sim output")
    parser.add_argument("frames", help="Path to raw RGBW frames file")
    parser.add_argument("--diag", help="Path to JSONL diagnostics from sim")
    parser.add_argument("--effect", default="unknown", help="Effect name for verdict logic")
    parser.add_argument("--timeline", action="store_true", help="Print text timeline")
    parser.add_argument("--png", help="Output LED-o-gram PNG path")
    args = parser.parse_args()

    frames = load_frames(args.frames)
    diag_lines = load_diag(args.diag) if args.diag else []

    stats = compute_stats(frames, args.effect)
    print(json.dumps(stats, indent=2))

    if args.timeline:
        print("\n--- TIMELINE ---")
        print(text_timeline(frames, diag_lines))

    if args.png:
        try:
            from PIL import Image
            # LED-o-gram: time (y) × LED position (x), RGB only
            rgb = frames[:, :, :3]
            img = Image.fromarray(rgb, 'RGB')
            # Scale up for visibility
            w, h = img.size
            scale = max(1, 800 // w)
            img = img.resize((w * scale, min(h, 2000)), Image.NEAREST)
            img.save(args.png)
            print(f"\nLED-o-gram saved to {args.png}")
        except ImportError:
            print("\nInstall Pillow for PNG output: pip install Pillow", file=sys.stderr)

if __name__ == "__main__":
    main()
