#!/usr/bin/env python3
"""
geometry_gen.py — parametric geometry for the helix-canopy sculpture.

A "T"/spire form:
  - a vertical DOUBLE HELIX trunk that TAPERS from a wide base to a narrow top,
  - a FILLED HEPTAGON canopy partway up, which the helix POKES THROUGH and
    continues above,
  - a ROOT SYSTEM of strips splaying randomly from the trunk base along the floor.

Anti-drift principle (the ledsim doctrine, applied to geometry):
  The position math lives in exactly ONE place — here. We compute every LED's
  (x, y, z) once and emit it as literal data to two consumers:
    - geometry/layout.json  — OPC-style flat array, for the 3D viewer / tooling
    - geometry/topology.h   — Vec3 LED_POS[] literal table, for the firmware
  The device reads the table; it never recomputes geometry, so nothing drifts.

Units: meters internally. Z is up. The trunk axis is the z axis through (0, 0).

Wire/strip model (festicorn multi-strip setPixel(s, i) convention):
  strip 0..1  — helix strand A / B (floor -> top, tapering)
  strip 2     — heptagon canopy (serpentine fill, with a hole for the trunk)
  strip 3..9  — roots (one strip each)
"""

import json
import math
import os
import random

FT = 0.3048   # meters per foot
IN = 0.0254   # meters per inch

# ─────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS — physical dimensions in ft/in for legibility.
# ─────────────────────────────────────────────────────────────────────────
P = {
    # ── Tapered double helix (the trunk) ─────────────────────────────────
    "helix_strands":      2,
    "helix_base_radius_m": 1.25 * FT,   # ~2.5 ft wide at the base
    "helix_top_radius_m":  3.0 * IN,    # ~6 in wide at the top
    "helix_height_m":     15.0 * FT,    # rises to 15 ft (pokes 5 ft above canopy)
    "helix_turns":        6.0,          # full revolutions base -> top
    "helix_led_pitch_m":  0.0333,       # ~30 LEDs/m along the strand
    "helix_z_start_m":    0.0,          # base sits at the floor (meets the roots)

    # ── Heptagon canopy (filled, the helix passes through it) ────────────
    "hept_sides":         7,
    "hept_side_len_m":    10.0 * FT,    # each edge 10 ft -> circumradius ~11.5 ft
    "canopy_height_m":    10.0 * FT,    # ceiling plane at 10 ft
    "canopy_row_pitch_m": 0.33,         # bulb-string row spacing
    "canopy_led_pitch_m": 0.33,         # bulb spacing along a string (~300 total)
    "canopy_rotation_deg": 0.0,
    "canopy_center_hole_m": 0.35,       # radius of the hole the trunk passes through

    # ── Root system ──────────────────────────────────────────────────────
    "roots_count":        7,
    "roots_len_min_m":    3.0 * FT,
    "roots_len_max_m":    8.0 * FT,
    "roots_led_pitch_m":  0.05,
    "roots_wander_deg":   22.0,         # max heading change per LED step (random walk)
    "roots_seed":         7,            # deterministic layout (re-run = same roots)
}


# ── small vector helpers ─────────────────────────────────────────────────
def _d3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _lerp(a, b, f):
    return (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2]))


def resample_polyline(pts, pitch):
    """Return points equidistant (~pitch apart) along the polyline `pts`."""
    if len(pts) < 2:
        return list(pts)
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + _d3(pts[i], pts[i-1]))
    total = cum[-1]
    n = max(2, int(round(total / pitch)))
    out, j = [], 0
    for i in range(n):
        d = total * i / (n - 1)
        while j < len(cum) - 2 and cum[j+1] < d:
            j += 1
        seg = cum[j+1] - cum[j]
        f = 0.0 if seg < 1e-9 else (d - cum[j]) / seg
        out.append(_lerp(pts[j], pts[j+1], f))
    return out, total


# ─────────────────────────────────────────────────────────────────────────
# Geometry builders
# ─────────────────────────────────────────────────────────────────────────
def build_helix(p):
    """Tapered conical double helix. Returns (strands, arc_len, leds_per_strand)."""
    R0 = p["helix_base_radius_m"]
    R1 = p["helix_top_radius_m"]
    H = p["helix_height_m"]
    turns = p["helix_turns"]
    z0 = p["helix_z_start_m"]
    n_strands = p["helix_strands"]
    SAMPLES = 8000  # dense base curve, then resample to equidistant LEDs

    strands, arc_len, n_leds = [], 0.0, 0
    for j in range(n_strands):
        phase = 2.0 * math.pi * j / n_strands
        raw = []
        for s in range(SAMPLES + 1):
            t = s / SAMPLES
            r = R0 + t * (R1 - R0)                 # linear taper
            th = 2.0 * math.pi * turns * t + phase
            raw.append((r * math.cos(th), r * math.sin(th), z0 + H * t))
        pts, total = resample_polyline(raw, p["helix_led_pitch_m"])
        strands.append(pts)
        arc_len = total
        n_leds = len(pts)
    return strands, arc_len, n_leds


def _heptagon_vertices(p):
    n = p["hept_sides"]
    R = p["hept_side_len_m"] / (2.0 * math.sin(math.pi / n))   # circumradius from side
    rot = math.radians(p["canopy_rotation_deg"]) + math.pi / 2.0
    verts = [(R * math.cos(rot + 2*math.pi*k/n), R * math.sin(rot + 2*math.pi*k/n))
             for k in range(n)]
    return verts, R


def _row_xspan(verts, y):
    xs = []
    n = len(verts)
    for k in range(n):
        x0, y0 = verts[k]
        x1, y1 = verts[(k+1) % n]
        if (y0 <= y < y1) or (y1 <= y < y0):
            frac = (y - y0) / (y1 - y0)
            xs.append(x0 + frac * (x1 - x0))
    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def build_canopy(p):
    """Serpentine fill of the heptagon, with a center hole for the trunk."""
    verts, R = _heptagon_vertices(p)
    canopy_z = p["canopy_height_m"]
    hole = p["canopy_center_hole_m"]
    ys = [v[1] for v in verts]
    y_lo, y_hi = min(ys), max(ys)
    row_pitch, led_pitch = p["canopy_row_pitch_m"], p["canopy_led_pitch_m"]

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
        n_in = max(1, int(round(width / led_pitch)))
        if n_in == 1:
            row = [((xmin + xmax) / 2.0, y, canopy_z)]
        else:
            margin = (width - (n_in - 1) * led_pitch) / 2.0
            row = [(xmin + margin + i * led_pitch, y, canopy_z) for i in range(n_in)]
        if r % 2 == 1:
            row.reverse()
        # punch the trunk hole
        row = [pt for pt in row if math.hypot(pt[0], pt[1]) > hole]
        pts.extend(row)
    return pts, verts, canopy_z


def build_roots(p):
    """N roots splaying randomly from the trunk base, wandering along the floor."""
    rng = random.Random(p["roots_seed"])
    pitch = p["roots_led_pitch_m"]
    wander = math.radians(p["roots_wander_deg"])
    roots = []
    for k in range(p["roots_count"]):
        heading = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(p["roots_len_min_m"], p["roots_len_max_m"])
        n = max(2, int(round(length / pitch)))
        x, y, z = 0.0, 0.0, 0.01
        pts = []
        for i in range(n):
            pts.append((x, y, max(0.0, z)))
            heading += rng.uniform(-wander, wander)        # random walk
            x += pitch * math.cos(heading)
            y += pitch * math.sin(heading)
            z = 0.01 + 0.015 * math.sin(i * 0.4)           # gentle ground undulation
        roots.append(pts)
    return roots


# ─────────────────────────────────────────────────────────────────────────
# Emit
# ─────────────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "geometry"))
    os.makedirs(out_dir, exist_ok=True)

    strands, helix_arc, helix_n = build_helix(P)
    canopy_pts, hept_verts, canopy_z = build_canopy(P)
    roots = build_roots(P)

    flat, strips = [], []

    def add_strip(name, kind, pts):
        start = len(flat)
        flat.extend(pts)
        strips.append({"name": name, "kind": kind, "start": start, "count": len(pts)})

    for j, strand in enumerate(strands):
        add_strip(f"helix_{chr(ord('A') + j)}", "helix", strand)
    add_strip("canopy", "canopy", canopy_pts)
    for k, root in enumerate(roots):
        add_strip(f"root_{k}", "root", root)

    n_total = len(flat)

    layout = [{"point": [round(x, 5), round(y, 5), round(z, 5)]} for (x, y, z) in flat]
    with open(os.path.join(out_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=0)

    meta = {
        "params": P,
        "n_total": n_total,
        "canopy_z_m": round(canopy_z, 4),
        "helix_top_z_m": round(P["helix_z_start_m"] + P["helix_height_m"], 4),
        "helix_arc_len_m": round(helix_arc, 4),
        "helix_leds_per_strand": helix_n,
        "heptagon_vertices_xy": [[round(x, 5), round(y, 5)] for (x, y) in hept_verts],
        "strips": strips,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    write_topology_h(os.path.join(out_dir, "topology.h"), flat, strips, P, canopy_z)

    # ── console summary ─────────────────────────────────────────────────
    print(f"helix-canopy geometry generated -> {out_dir}")
    print(f"  total LEDs: {n_total}")
    by_kind = {}
    for s in strips:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + s["count"]
    for kind, c in by_kind.items():
        print(f"    {kind:>7}: {c:>5} LEDs")
    print(f"  helix: {helix_n} LEDs/strand, arc {helix_arc:.2f} m, "
          f"top z {P['helix_z_start_m']+P['helix_height_m']:.2f} m "
          f"(canopy at {canopy_z:.2f} m -> pokes "
          f"{(P['helix_z_start_m']+P['helix_height_m']-canopy_z)/FT:.1f} ft above)")
    print(f"  roots: {P['roots_count']}")
    est_w = n_total * 0.06
    print(f"  rough power @ ~60mW/LED avg: ~{est_w:.0f} W  "
          f"(peak full-white ~{n_total*0.3:.0f} W) — sanity-check controllers/PSU")


def write_topology_h(path, flat, strips, p, canopy_z):
    n = len(flat)
    L = []
    L.append("// topology.h — helix-canopy 3D pixel map.")
    L.append("// GENERATED by tools/geometry_gen.py — DO NOT EDIT BY HAND.")
    L.append("// Single source of truth for geometry; re-run the generator to change shape.")
    L.append("#pragma once")
    L.append("#include <stdint.h>")
    L.append("")
    L.append("struct Vec3 { float x, y, z; };")
    L.append("")
    L.append(f"#define HC_NUM_LEDS {n}")
    L.append(f"#define HC_NUM_STRIPS {len(strips)}")
    L.append("")
    L.append("// Per-LED world position (meters). Index = wire order.")
    L.append("static const Vec3 LED_POS[HC_NUM_LEDS] = {")
    for (x, y, z) in flat:
        L.append(f"  {{ {x:.5f}f, {y:.5f}f, {z:.5f}f }},")
    L.append("};")
    L.append("")
    L.append("// Strip ranges into LED_POS[]. kind: 0=helix, 1=canopy, 2=root.")
    L.append("struct HcStrip { uint16_t start, count; uint8_t kind; };")
    kind_map = {"helix": 0, "canopy": 1, "root": 2}
    L.append("static const HcStrip HC_STRIPS[HC_NUM_STRIPS] = {")
    for s in strips:
        L.append(f"  {{ {s['start']}, {s['count']}, {kind_map[s['kind']]} }}, // {s['name']}")
    L.append("};")
    L.append("")
    xs = [v[0] for v in flat]; ys = [v[1] for v in flat]; zs = [v[2] for v in flat]
    L.append("// Axis-aligned bounds of the whole sculpture (meters).")
    L.append(f"static const Vec3 HC_BOUND_MIN = {{ {min(xs):.5f}f, {min(ys):.5f}f, {min(zs):.5f}f }};")
    L.append(f"static const Vec3 HC_BOUND_MAX = {{ {max(xs):.5f}f, {max(ys):.5f}f, {max(zs):.5f}f }};")
    L.append(f"static const float HC_CANOPY_Z = {canopy_z:.5f}f;")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
