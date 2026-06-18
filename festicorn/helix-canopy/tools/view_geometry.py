#!/usr/bin/env python3
"""
view_geometry.py — 3D "LED-o-gram" for the helix-canopy sculpture.

Renders the LED layout (and, later, effect frames) to a multi-angle composite
PNG using matplotlib's offscreen Agg backend. No GUI, no hardware — built to be
driven inside an AI loop: generate -> render -> an agent (or you) LOOKS at the
PNG and reasons about it.

Two modes:
  1. Geometry preview (default): color each LED by strip, or by height.
     python3 view_geometry.py
  2. Frame preview: overlay actual effect colors from a sim CSV with columns
     frame,pixel,x,y,z,r,g,b  (r,g,b in 0..255). Renders one chosen frame.
     python3 view_geometry.py --frames sim.csv --frame 120

Output: geometry/preview.png  (four camera angles in a 2x2 grid)
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")  # offscreen — required for headless / AI-loop use
import matplotlib.pyplot as plt
import numpy as np


def load_layout(geom_dir):
    with open(os.path.join(geom_dir, "layout.json")) as f:
        layout = json.load(f)
    pos = np.array([p["point"] for p in layout], dtype=float)  # (N,3)
    with open(os.path.join(geom_dir, "meta.json")) as f:
        meta = json.load(f)
    return pos, meta


def strip_id_per_led(meta, n):
    sid = np.zeros(n, dtype=int)
    for i, s in enumerate(meta["strips"]):
        sid[s["start"]:s["start"] + s["count"]] = i
    return sid


def colors_by_strip(meta, n):
    sid = strip_id_per_led(meta, n)
    palette = plt.get_cmap("tab10")
    return palette(sid % 10)[:, :3]


def colors_by_height(pos):
    z = pos[:, 2]
    zn = (z - z.min()) / max(1e-6, (z.max() - z.min()))
    return plt.get_cmap("viridis")(zn)[:, :3]


def load_frame_colors(csv_path, frame, n):
    import csv
    rgb = np.full((n, 3), 0.02)  # near-black default (unlit pixels visible faintly)
    want = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    frames = sorted({int(r["frame"]) for r in rows})
    if not frames:
        return rgb
    want = frame if frame is not None else frames[len(frames) // 2]
    # snap to nearest available frame
    want = min(frames, key=lambda fr: abs(fr - want))
    for r in rows:
        if int(r["frame"]) != want:
            continue
        i = int(r["pixel"])
        if 0 <= i < n:
            rgb[i] = [float(r["r"]) / 255.0, float(r["g"]) / 255.0, float(r["b"]) / 255.0]
    return rgb, want


def render(pos, rgb, meta, out_path, title, point_size, frame_label=None):
    # Equal aspect via shared bounds; emphasize the tall sculpture.
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    ctr = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0 * 1.05

    views = [("front",  20, -90),
             ("3/4",    22, -55),
             ("side",   12,   0),
             ("top",    88, -90)]

    fig = plt.figure(figsize=(12, 12), facecolor="#0b0b0f")
    for k, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.set_facecolor("#0b0b0f")
        # faint dark base point so unlit LEDs still show position
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=rgb, s=point_size, depthshade=False,
                   edgecolors="none")
        ax.view_init(elev=elev, azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0, 0, 0, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0.06))
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(min(0, ctr[2] - span), ctr[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(colors="#666", labelsize=6)
        ax.set_title(name, color="#aaa", fontsize=10)
        ax.grid(False)

    sub = f"{title}   |   {meta['n_total']} LEDs"
    if frame_label is not None:
        sub += f"   |   frame {frame_label}"
    fig.suptitle(sub, color="#ddd", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    geom_dir = os.path.normpath(os.path.join(here, "..", "geometry"))
    ap.add_argument("--geom", default=geom_dir)
    ap.add_argument("--color", choices=["strip", "height"], default="strip")
    ap.add_argument("--frames", default=None, help="sim CSV: frame,pixel,x,y,z,r,g,b")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--point-size", type=float, default=8.0)
    args = ap.parse_args()

    pos, meta = load_layout(args.geom)
    n = len(pos)
    out = args.out or os.path.join(args.geom, "preview.png")

    frame_label = None
    if args.frames:
        rgb, frame_label = load_frame_colors(args.frames, args.frame, n)
        title = "effect frame"
    elif args.color == "height":
        rgb = colors_by_height(pos)
        title = "geometry (height)"
    else:
        rgb = colors_by_strip(meta, n)
        title = "geometry (by strip)"

    render(pos, rgb, meta, out, title, args.point_size, frame_label)


if __name__ == "__main__":
    main()
