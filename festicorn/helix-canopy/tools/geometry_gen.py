#!/usr/bin/env python3
"""
geometry_gen.py — parametric geometry for the helix-canopy sculpture.

The sculpture is a "T": a vertical DOUBLE HELIX post rising from the floor to a
FILLED HEPTAGON ceiling canopy ~10 ft up. This script is the SINGLE SOURCE OF
TRUTH for where every LED lives in 3D space.

Anti-drift principle (the ledsim doctrine, applied to geometry):
  The position math lives in exactly ONE place — here. We compute every LED's
  (x, y, z) once and emit it as literal data to two consumers:
    - geometry/layout.json  — OPC-style flat array, for the 3D viewer / tooling
    - geometry/topology.h   — Vec3 LED_POS[] literal table, for the firmware
  The device does NOT recompute geometry (no duplicated trig to drift). It reads
  the table. If you change the shape, re-run this script; both outputs regenerate
  together and stay in lockstep.

Units: meters. Z is up. The T-axis is the z axis through (x=0, y=0).

Wire/strip model (matches festicorn's multi-strip setPixel(s, i) convention):
  strip 0 — helix strand A   (floor -> ceiling)
  strip 1 — helix strand B   (floor -> ceiling, 180 deg phase)
  strip 2 — heptagon canopy  (serpentine fill, one continuous run)
Each strip is a contiguous range in the flat LED_POS[] array.
"""

import json
import math
import os

# ─────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS — everything you'd want to change lives here.
# These are FIRST-DRAFT guesses. Adjust to the real build, re-run, re-view.
# ─────────────────────────────────────────────────────────────────────────
P = {
    # ── Double helix (the post of the T) ─────────────────────────────────
    "helix_strands":      2,        # double helix
    "helix_radius_m":     0.30,     # radius of the spiral about the central axis
    "helix_height_m":     3.05,     # 10 ft, floor to canopy underside
    "helix_turns":        4.0,      # full revolutions floor -> ceiling
    "helix_led_pitch_m":  0.0333,   # spacing between LEDs along the strand (~30/m)
    "helix_z_start_m":    0.10,     # lift the bottom LED slightly off the floor

    # ── Heptagon canopy (the top of the T), a FILLED grid ────────────────
    "hept_sides":         7,
    "hept_circumradius_m": 1.50,    # center -> vertex
    "canopy_row_pitch_m": 0.15,     # spacing BETWEEN serpentine rows
    "canopy_led_pitch_m": 0.05,     # spacing between LEDs ALONG a row (~20/m)
    "canopy_rotation_deg": 0.0,     # rotate the heptagon about z (0 = vertex toward +y... see note)
    # canopy sits at the top of the helix:
    #   canopy_z = helix_z_start + helix_height   (computed below)
}


# ─────────────────────────────────────────────────────────────────────────
# Geometry builders
# ─────────────────────────────────────────────────────────────────────────
def build_helix(p):
    """Return list of strands; each strand is a list of (x, y, z)."""
    R = p["helix_radius_m"]
    H = p["helix_height_m"]
    turns = p["helix_turns"]
    z0 = p["helix_z_start_m"]
    n_strands = p["helix_strands"]

    # Arc length of one strand: a helix is a straight line on an unrolled cylinder.
    horiz = 2.0 * math.pi * R * turns
    arc_len = math.hypot(horiz, H)
    n_leds = max(2, int(round(arc_len / p["helix_led_pitch_m"])))

    strands = []
    for j in range(n_strands):
        phase = 2.0 * math.pi * j / n_strands
        pts = []
        for i in range(n_leds):
            t = i / (n_leds - 1)
            theta = 2.0 * math.pi * turns * t + phase
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            z = z0 + H * t
            pts.append((x, y, z))
        strands.append(pts)
    return strands, arc_len, n_leds


def _heptagon_vertices(p):
    n = p["hept_sides"]
    R = p["hept_circumradius_m"]
    rot = math.radians(p["canopy_rotation_deg"]) + math.pi / 2.0  # first vertex toward +y
    return [
        (R * math.cos(rot + 2 * math.pi * k / n),
         R * math.sin(rot + 2 * math.pi * k / n))
        for k in range(n)
    ]


def _row_xspan(verts, y):
    """For convex polygon `verts`, return (xmin, xmax) where horizontal line
    y=const crosses the boundary, or None if it doesn't intersect."""
    xs = []
    n = len(verts)
    for k in range(n):
        x0, y0 = verts[k]
        x1, y1 = verts[(k + 1) % n]
        if (y0 <= y < y1) or (y1 <= y < y0):  # edge straddles y
            frac = (y - y0) / (y1 - y0)
            xs.append(x0 + frac * (x1 - x0))
    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def build_canopy(p):
    """Serpentine (boustrophedon) fill of the heptagon at the ceiling plane.
    One continuous run: even rows left->right, odd rows right->left."""
    verts = _heptagon_vertices(p)
    canopy_z = p["helix_z_start_m"] + p["helix_height_m"]
    ys = [v[1] for v in verts]
    y_lo, y_hi = min(ys), max(ys)

    row_pitch = p["canopy_row_pitch_m"]
    led_pitch = p["canopy_led_pitch_m"]
    # center the rows in the polygon's vertical extent
    span = y_hi - y_lo
    n_rows = max(1, int(math.floor(span / row_pitch)))
    y_start = y_lo + (span - (n_rows - 1) * row_pitch) / 2.0

    pts = []
    for r in range(n_rows):
        y = y_start + r * row_pitch
        sp = _row_xspan(verts, y)
        if sp is None:
            continue
        xmin, xmax = sp
        width = xmax - xmin
        n_in_row = max(1, int(round(width / led_pitch)))
        # place LEDs centered in the span
        if n_in_row == 1:
            row_pts = [((xmin + xmax) / 2.0, y, canopy_z)]
        else:
            margin = (width - (n_in_row - 1) * led_pitch) / 2.0
            row_pts = [(xmin + margin + i * led_pitch, y, canopy_z)
                       for i in range(n_in_row)]
        if r % 2 == 1:
            row_pts.reverse()  # serpentine
        pts.extend(row_pts)
    return pts, verts, canopy_z


# ─────────────────────────────────────────────────────────────────────────
# Emit
# ─────────────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "geometry"))
    os.makedirs(out_dir, exist_ok=True)

    canopy_z = P["helix_z_start_m"] + P["helix_height_m"]

    strands, helix_arc, helix_n = build_helix(P)
    canopy_pts, hept_verts, _ = build_canopy(P)

    # Flatten in wire order, recording strip ranges.
    flat = []          # list of (x, y, z)
    strips = []        # list of dicts {name, kind, start, count}

    def add_strip(name, kind, pts):
        start = len(flat)
        flat.extend(pts)
        strips.append({"name": name, "kind": kind, "start": start, "count": len(pts)})

    for j, strand in enumerate(strands):
        add_strip(f"helix_{chr(ord('A') + j)}", "helix", strand)
    add_strip("canopy", "canopy", canopy_pts)

    n_total = len(flat)

    # ── layout.json (OPC-style flat array) ──────────────────────────────
    layout = [{"point": [round(x, 5), round(y, 5), round(z, 5)]} for (x, y, z) in flat]
    with open(os.path.join(out_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=0)

    # ── meta.json (params + strip table; for viewer & humans) ───────────
    meta = {
        "params": P,
        "n_total": n_total,
        "canopy_z_m": round(canopy_z, 4),
        "helix_arc_len_m": round(helix_arc, 4),
        "helix_leds_per_strand": helix_n,
        "heptagon_vertices_xy": [[round(x, 5), round(y, 5)] for (x, y) in hept_verts],
        "strips": strips,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── topology.h (Vec3 literal table for firmware) ────────────────────
    write_topology_h(os.path.join(out_dir, "topology.h"), flat, strips, P, canopy_z)

    # ── console summary (react to the counts!) ──────────────────────────
    print(f"helix-canopy geometry generated -> {out_dir}")
    print(f"  total LEDs: {n_total}")
    for s in strips:
        print(f"    strip {s['start']:>5}..{s['start']+s['count']-1:<5} "
              f"[{s['kind']:>6}] {s['name']:<10} {s['count']:>5} LEDs")
    print(f"  helix arc length/strand: {helix_arc:.2f} m  ({helix_n} LEDs/strand)")
    print(f"  canopy plane z: {canopy_z:.2f} m")
    est_w = n_total * 0.06  # ~60mW/LED rough at moderate white
    print(f"  rough power @ ~60mW/LED avg: ~{est_w:.0f} W  "
          f"(peak full-white ~{n_total*0.3:.0f} W) — sanity-check controllers/PSU")


def write_topology_h(path, flat, strips, p, canopy_z):
    n = len(flat)
    lines = []
    lines.append("// topology.h — helix-canopy 3D pixel map.")
    lines.append("// GENERATED by tools/geometry_gen.py — DO NOT EDIT BY HAND.")
    lines.append("// Single source of truth for geometry; re-run the generator to change shape.")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("struct Vec3 { float x, y, z; };")
    lines.append("")
    lines.append(f"#define HC_NUM_LEDS {n}")
    lines.append(f"#define HC_NUM_STRIPS {len(strips)}")
    lines.append("")
    lines.append("// Per-LED world position (meters). Index = wire order.")
    lines.append("static const Vec3 LED_POS[HC_NUM_LEDS] = {")
    for (x, y, z) in flat:
        lines.append(f"  {{ {x:.5f}f, {y:.5f}f, {z:.5f}f }},")
    lines.append("};")
    lines.append("")
    lines.append("// Strip ranges into LED_POS[]. kind: 0=helix, 1=canopy.")
    lines.append("struct HcStrip { uint16_t start, count; uint8_t kind; };")
    kind_map = {"helix": 0, "canopy": 1}
    lines.append(f"static const HcStrip HC_STRIPS[HC_NUM_STRIPS] = {{")
    for s in strips:
        lines.append(f"  {{ {s['start']}, {s['count']}, {kind_map[s['kind']]} }}, "
                     f"// {s['name']}")
    lines.append("};")
    lines.append("")
    # handy bounds for normalized effects
    xs = [v[0] for v in flat]; ys = [v[1] for v in flat]; zs = [v[2] for v in flat]
    lines.append("// Axis-aligned bounds of the whole sculpture (meters).")
    lines.append(f"static const Vec3 HC_BOUND_MIN = {{ {min(xs):.5f}f, {min(ys):.5f}f, {min(zs):.5f}f }};")
    lines.append(f"static const Vec3 HC_BOUND_MAX = {{ {max(xs):.5f}f, {max(ys):.5f}f, {max(zs):.5f}f }};")
    lines.append(f"static const float HC_CANOPY_Z = {canopy_z:.5f}f;")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
