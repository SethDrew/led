#!/usr/bin/env python3
"""
build_frames.py — pack a sim CSV + geometry into a compact browser payload.

Phase 0 of the LED IDE: we don't run effects.h live yet. Instead we take the
frames the existing host sim already produces and turn them into something the
Three.js viewer can fetch and play back interactively (orbit / scrub / bloom).

Reads:
  ../geometry/layout.json   (OPC: flat array of {"point":[x,y,z]})
  ../geometry/meta.json     (n_total, strips/kinds, bounds)
  <csv>                     (sim dump: frame,t_s,pixel,x,y,z,r,g,b)

Writes (into viewer/data/, self-contained so the viewer can be served alone):
  data/scene.json   {n_leds, n_frames, fps, bounds, positions, kinds}
  data/frames.bin   Uint8 RGB, frame-major: [frame][pixel][r,g,b]

Stdlib only (no numpy) so it runs against system python with no venv.
"""

import argparse
import array
import csv
import json
import os
import sys

KIND_ID = {"helix": 0, "canopy": 1, "root": 2}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    geom_default = os.path.normpath(os.path.join(here, "..", "geometry"))
    csv_default = os.path.normpath(
        os.path.join(here, "..", "tools", "sim", "nebula_frames.csv"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=csv_default)
    ap.add_argument("--geom", default=geom_default)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.join(here, "data"))
    args = ap.parse_args()

    with open(os.path.join(args.geom, "layout.json")) as f:
        layout = json.load(f)
    with open(os.path.join(args.geom, "meta.json")) as f:
        meta = json.load(f)

    n_leds = meta["n_total"]
    positions = [p["point"] for p in layout]
    assert len(positions) == n_leds, f"{len(positions)} pts vs n_total {n_leds}"

    # per-pixel kind id (for optional coloring / filtering in the viewer)
    kinds = [0] * n_leds
    for s in meta["strips"]:
        k = KIND_ID.get(s["kind"], 0)
        for i in range(s["start"], s["start"] + s["count"]):
            kinds[i] = k

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    bounds = {"min": [min(xs), min(ys), min(zs)],
              "max": [max(xs), max(ys), max(zs)]}

    # ---- frames: read CSV, bucket by frame, fill a dense Uint8 buffer --------
    frame_ids = set()
    rows = []
    with open(args.csv) as f:
        for r in csv.reader(f):
            if r[0] == "frame":          # header
                continue
            fr = int(r[0]); px = int(r[2])
            rr = int(r[6]); gg = int(r[7]); bb = int(r[8])
            rows.append((fr, px, rr, gg, bb))
            frame_ids.add(fr)

    if not frame_ids:
        sys.exit("no frames parsed from CSV")
    n_frames = max(frame_ids) + 1
    buf = array.array("B", bytes(n_frames * n_leds * 3))
    for fr, px, rr, gg, bb in rows:
        o = (fr * n_leds + px) * 3
        buf[o] = rr; buf[o + 1] = gg; buf[o + 2] = bb

    os.makedirs(args.out, exist_ok=True)
    scene = {
        "n_leds": n_leds,
        "n_frames": n_frames,
        "fps": args.fps,
        "bounds": bounds,
        "positions": positions,
        "kinds": kinds,
    }
    with open(os.path.join(args.out, "scene.json"), "w") as f:
        json.dump(scene, f)
    with open(os.path.join(args.out, "frames.bin"), "wb") as f:
        buf.tofile(f)

    mb = len(buf) / 1e6
    print(f"wrote {args.out}/scene.json  ({n_leds} LEDs)")
    print(f"wrote {args.out}/frames.bin  ({n_frames} frames, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
