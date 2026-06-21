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
    # Real hardware: 12mm WS2811 bullet strands, 50 nodes @ 100mm pitch (10/m, 5m).
    # The canopy is radial spokes — one strand per spoke, center hole -> rim.
    "canopy_spoke_count":  7,           # spokes (default = one per heptagon vertex)
    "canopy_pixel_pitch_m": 0.10,       # 100mm node pitch (the real default)
    "canopy_strand_nodes": 50,          # nodes per real strand (caps a spoke's length)
    "canopy_rotation_deg": 0.0,
    "canopy_center_hole_m": 0.35,       # radius of the hole the trunk passes through

    # ── Root system ──────────────────────────────────────────────────────
    "roots_count":        7,
    "roots_len_min_m":    9.0 * FT,     # ~= straight-line reach (endpoint anchored on spoke)
    "roots_len_max_m":    10.0 * FT,    # tight variance so roots are comparable
    "roots_led_pitch_m":  0.05,
    "roots_wave_amp_m":   0.22,         # lateral waviness amplitude (0 at base + tip)
    "roots_wave_freq":    2.0,          # ~wave cycles along a root
    "roots_jitter_deg":   4.0,          # +/- jitter on the even radial spoke (symmetry)
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


def _point_in_poly(x, y, verts):
    """Ray-cast point-in-polygon test (verts: list of (x,y))."""
    inside = False
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]; xj, yj = verts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def build_canopy(p):
    """Radial spokes from the trunk to the rim — one real bullet strand per spoke.

    Each spoke marches outward from the center-hole edge at the real 100mm node
    pitch until it leaves the heptagon (or hits the strand's node cap), so spoke
    length is clipped to the actual canopy rim. Spokes are evenly spaced and, when
    spoke_count == hept_sides, point straight at the vertices. Returns the strand
    occupancy so we can report nodes-used vs the 50-node strand length.
    """
    verts, R = _heptagon_vertices(p)
    canopy_z = p["canopy_height_m"]
    hole = p["canopy_center_hole_m"]
    pitch = p["canopy_pixel_pitch_m"]
    count = p["canopy_spoke_count"]
    cap = p["canopy_strand_nodes"]

    # align spoke 0 with vertex 0 (same rotation as the heptagon)
    base = math.radians(p["canopy_rotation_deg"]) + math.pi / 2.0

    spokes = []
    for s in range(count):
        theta = base + 2.0 * math.pi * s / count
        ct, st = math.cos(theta), math.sin(theta)
        spoke = []
        for i in range(cap):
            r = hole + i * pitch
            x, y = r * ct, r * st
            if not _point_in_poly(x, y, verts):
                break
            spoke.append((x, y, canopy_z))
        spokes.append(spoke)
    return spokes, verts, canopy_z


def build_roots(p):
    """N roots splaying evenly from the trunk base along the floor.

    Symmetry and waviness are decoupled. Each root's ENDPOINT is anchored on an
    evenly-spaced radial spoke (2*pi*k/count + small jitter) at the chosen reach,
    so the tips fan out evenly. Organic waviness is a LATERAL offset perpendicular
    to the spoke, modulated by a sin(pi*s) envelope that is zero at both the base
    and the tip — so the root meanders in the middle but always starts at the
    trunk and lands exactly on its spoke. Reach ~= length by construction.
    """
    rng = random.Random(p["roots_seed"])
    pitch = p["roots_led_pitch_m"]
    amp = p["roots_wave_amp_m"]
    basefreq = p["roots_wave_freq"]
    jitter = math.radians(p["roots_jitter_deg"])
    count = p["roots_count"]
    roots = []
    for k in range(count):
        base = 2.0 * math.pi * k / count + rng.uniform(-jitter, jitter)
        reach = rng.uniform(p["roots_len_min_m"], p["roots_len_max_m"])
        n = max(2, int(round(reach / pitch)))
        cb, sb = math.cos(base), math.sin(base)
        # two random sinusoids per root for non-repeating organic wave
        f1 = basefreq * rng.uniform(0.8, 1.2); ph1 = rng.uniform(0, 2 * math.pi)
        f2 = basefreq * rng.uniform(1.8, 2.4); ph2 = rng.uniform(0, 2 * math.pi)
        a2 = rng.uniform(0.3, 0.6)
        pts = []
        for i in range(n):
            s = i / (n - 1)                                  # 0..1 along the root
            rr = s * reach                                   # radial distance out the spoke
            env = math.sin(math.pi * s)                      # 0 at base + tip, 1 mid
            lateral = env * amp * (math.sin(2 * math.pi * f1 * s + ph1)
                                   + a2 * math.sin(2 * math.pi * f2 * s + ph2))
            x = rr * cb - lateral * sb                       # spoke + perpendicular wiggle
            y = rr * sb + lateral * cb
            z = 0.01 + 0.015 * math.sin(i * 0.4)             # gentle ground undulation
            pts.append((x, y, max(0.0, z)))
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
    canopy_spokes, hept_verts, canopy_z = build_canopy(P)
    roots = build_roots(P)

    flat, strips = [], []

    def add_strip(name, kind, pts):
        start = len(flat)
        flat.extend(pts)
        strips.append({"name": name, "kind": kind, "start": start, "count": len(pts)})

    for j, strand in enumerate(strands):
        add_strip(f"helix_{chr(ord('A') + j)}", "helix", strand)
    for k, spoke in enumerate(canopy_spokes):       # one real strand per spoke
        add_strip(f"canopy_{k}", "canopy", spoke)
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
