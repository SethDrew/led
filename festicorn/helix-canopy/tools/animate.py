#!/usr/bin/env python3
"""
animate.py — watch a helix-canopy effect play over time.

Reads a sim CSV (frame,t_s,pixel,x,y,z,r,g,b) and renders the whole sequence to
a video (MP4 via ffmpeg, GIF fallback). One 3/4 camera by default; optionally
rotate the camera slowly so you can read the 3D structure.

  python3 animate.py --frames sim/pour.csv
  python3 animate.py --frames sim/pour.csv --rotate --out pour.mp4
"""

import argparse
import csv
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def load_csv(path):
    frames = {}
    pos = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            fr = int(row["frame"])
            i = int(row["pixel"])
            pos[i] = (float(row["x"]), float(row["y"]), float(row["z"]))
            frames.setdefault(fr, {})[i] = (
                int(row["r"]) / 255.0, int(row["g"]) / 255.0, int(row["b"]) / 255.0)
    n = max(pos) + 1
    P = np.array([pos[i] for i in range(n)])
    order = sorted(frames)
    C = np.zeros((len(order), n, 3))
    T = []
    for k, fr in enumerate(order):
        for i, rgb in frames[fr].items():
            C[k, i] = rgb
    # recover t_s for fps
    with open(path) as f:
        ts = {}
        for row in csv.DictReader(f):
            ts[int(row["frame"])] = float(row["t_s"])
    T = [ts[fr] for fr in order]
    return P, C, np.array(T)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rotate", action="store_true",
                    help="slowly orbit the camera over the clip")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--elev-start", type=float, default=-20.0,
                    help="elevation at clip start when --rotate (below = negative)")
    ap.add_argument("--elev-end", type=float, default=40.0,
                    help="elevation at clip end when --rotate (above = positive)")
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--point-size", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--glow", action="store_true",
                    help="scale point size with brightness (fake bloom)")
    args = ap.parse_args()

    P, C, T = load_csv(args.frames)
    n = P.shape[0]
    n_frames = C.shape[0]
    fps = 30.0
    if len(T) > 1:
        dt = np.median(np.diff(T))
        if dt > 0:
            fps = 1.0 / dt

    mins, maxs = P.min(0), P.max(0)
    ctr = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.05

    fig = plt.figure(figsize=(6, 7), facecolor="#08080c")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#08080c")
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(min(0, ctr[2] - span), ctr[2] + span)
    ax.set_box_aspect((1, 1, 1.1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0, 0, 0, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0.05))
    ax.tick_params(colors="#555", labelsize=6)
    ax.grid(False)

    base = np.full(n, args.point_size)
    scat = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C[0], s=base,
                      depthshade=False, edgecolors="none")
    title = ax.set_title("", color="#ccc", fontsize=10)

    def update(k):
        scat.set_color(np.clip(C[k], 0, 1))
        if args.glow:
            lum = C[k].max(axis=1)
            scat.set_sizes(base * (0.35 + 1.4 * lum))
        if args.rotate:
            frac = k / max(1, n_frames - 1)
            elev = args.elev_start + (args.elev_end - args.elev_start) * frac
            ax.view_init(elev=elev, azim=args.azim + 360.0 * frac)
        else:
            ax.view_init(elev=args.elev, azim=args.azim)
        title.set_text(f"t = {T[k]:5.2f}s   frame {k}/{n_frames}")
        return scat, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)

    out = args.out or os.path.splitext(args.frames)[0] + ".mp4"
    if shutil.which("ffmpeg") and out.endswith(".mp4"):
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=args.dpi)
    else:
        out = os.path.splitext(out)[0] + ".gif"
        anim.save(out, writer=PillowWriter(fps=min(fps, 25)), dpi=max(70, args.dpi - 30))
    plt.close(fig)
    print(f"wrote {out}  ({n_frames} frames @ {fps:.0f}fps)")


if __name__ == "__main__":
    main()
