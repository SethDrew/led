#!/usr/bin/env python3
"""
geometry_gen.py — parametric geometry for the strap-canopy sculpture.

Final playa form (2026-08-20, canopy-plan-nopull-butterfly + wiring session;
physical narrative in ../../WIRING.md):
  - REGULAR HEPTAGON canopy, side fixed 10 ft, NO edge pull. Alpine-butterfly
    knots push each spoke-attach point 5.5 in outboard of the corner.
  - CANOPY CHAINS of WS2811 bulbs (8 runs of 2 x 50-node strands, draped):
    from ONE double-run short spoke, two chains of four runs wrap the rim in
    opposite directions; the two final runs are cut ~25 bulbs short so both
    chains terminate on the vertex diametrically opposite the start. Y-split
    pins pair mirror-twin runs: the whole canopy reflects about the
    start-spoke/terminal-vertex axis.
  - 7 LONG-SPOKE RUNS: one 50-node bulb strand each, coupling edge -> attach
    point (taut), then the GUY continues from the attach point down-and-out
    to a ground anchor, dressed with one full 5 m / 150-LED (30/m) WS2812B
    strip — the anchor sits where the strip runs out.
  - CENTER HELIX: WS2812B strips (same product as guys) wound around a
    2 in-diameter, 10 ft pole under the canopy center.
  - ROOT SYSTEM of paper-mache poles on the floor, one 150-LED strip each,
    laid as a double meander.

Anti-drift principle (the ledsim doctrine, applied to geometry):
  The position math lives in exactly ONE place — here. We compute every LED's
  (x, y, z) once and emit it as literal data to two consumers:
    - geometry/layout.json  — OPC-style flat array, for the 3D viewer / tooling
    - geometry/topology.h   — Vec3 LED_POS[] literal table, for the firmware
  The device reads the table; it never recomputes geometry, so nothing drifts.

Units: meters internally. Z is up. The center axis is the z axis through (0, 0).

Wire/strip model (festicorn multi-strip setPixel(s, i) convention):
  strip 0..6    — sector runs (short spoke -> rim wrap)
  strip 7..20   — long-spoke / guy pairs, interleaved in chain order
  strip 21..22  — helix strips on the center pole
  strip 23..26  — roots (one strip each)
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
    # ── Heptagon canopy (nopull-butterfly plan) ──────────────────────────
    "hept_sides":          7,
    "hept_side_len_m":     10.0 * FT,   # FIXED side -> circumradius 138.3 in
    "canopy_height_m":     10.0 * FT,   # the canopy is ONE flat plane
    "coupling_radius_m":   3.0 * IN,    # hub coupling plate, diameter 6 in
    "butterfly_offset_m":  5.5 * IN,    # attach point outboard of each corner
    "canopy_rotation_deg": 0.0,

    # ── WS2811 bulb strands (12mm bullets, 50 nodes, 140 in lit) ─────────
    "bulb_strand_nodes":   50,
    # Canopy chains: from ONE double-run short spoke, two chains of four
    # 100-bulb runs wrap the rim in opposite directions (each run: out its own
    # short spoke, then one side-length of rim to the next short spoke). The
    # two final runs are CUT ~25 bulbs short so both terminate on the vertex
    # diametrically opposite the start spoke. Y-splitters pair each run with
    # its mirror twin on one pin: 4 pins x 2 strips, whole-canopy reflection
    # about the start-spoke/terminal-vertex axis.
    "canopy_start_spoke":  0,
    "pair_run_nodes":      100,
    "pair_spoke_sep_m":    0.05,        # the double-run strands hang ~5cm apart

    # ── Guys: one full WS2812B 5 m / 150-LED (30/m) strip per long spoke ─
    "guy_strip_leds":      150,
    "guy_strip_pitch_m":   1.0 / 30.0,

    # ── Center helix: same strip product, wound on the center pole ───────
    "helix_strips":        2,
    "helix_strip_leds":    150,
    "helix_strip_pitch_m": 1.0 / 30.0,
    "helix_pole_radius_m": 1.0 * IN,    # 2 in-diameter pole
    "helix_height_m":      10.0 * FT,

    # ── Root system: one standard IP67 strip per paper-mache pole ────────
    "roots_count":        4,
    "root_len_m":         6.0 * FT,     # straight pole length
    "root_radius_m":      3.5 * IN,     # ~7 in diameter paper-mache root
    "root_inner_m":       1.0 * FT,     # pole's inner end, out from center
    "root_strip_leds":    150,          # the real strip: 150 LEDs @ 30/m = 5 m
    "root_strip_pitch_m": 1.0 / 30.0,
    "root_meander_cycles": 3.0,         # sway cycles per pass
    "root_meander_sep_rad": 1.1,        # azimuth gap between out and back snakes
    "roots_jitter_deg":   4.0,          # +/- jitter on the even radial spoke (symmetry)
    "roots_seed":         7,            # deterministic layout (re-run = same roots)
}


# ── small vector helpers ─────────────────────────────────────────────────
def _d3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _lerp(a, b, f):
    return (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2]))


def _place_n(waypoints, n):
    """Place exactly n nodes evenly (by arc length) along a 3D polyline.

    Physical strands are longer than the structural path and get DRAPED to
    fit, so even spacing along the path is the honest pixel model.
    """
    cum = [0.0]
    for a, b in zip(waypoints, waypoints[1:]):
        cum.append(cum[-1] + _d3(a, b))
    total = cum[-1]
    pts, j = [], 0
    for i in range(n):
        d = total * i / (n - 1)
        while j < len(cum) - 2 and cum[j + 1] < d:
            j += 1
        seg = cum[j + 1] - cum[j]
        f = 0.0 if seg < 1e-9 else (d - cum[j]) / seg
        pts.append(_lerp(waypoints[j], waypoints[j + 1], f))
    return pts, total


# ─────────────────────────────────────────────────────────────────────────
# Geometry builders
# ─────────────────────────────────────────────────────────────────────────
def _frame(p):
    n = p["hept_sides"]
    R = p["hept_side_len_m"] / (2.0 * math.sin(math.pi / n))
    base = math.radians(p["canopy_rotation_deg"]) + math.pi / 2.0
    verts = [(R * math.cos(base + 2*math.pi*k/n), R * math.sin(base + 2*math.pi*k/n))
             for k in range(n)]
    mid_az = [base + 2.0 * math.pi * (k + 0.5) / n for k in range(n)]
    return verts, mid_az, R, base


def build_pairs(p, verts, mid_az):
    """Canopy chains. From the start short spoke, two chains of four 100-bulb
    runs wrap the rim in opposite directions; each run goes out its own short
    spoke (coupling edge -> mid-edge) then one side-length of rim to the next
    short-spoke junction. Every short spoke carries one run; the start spoke
    carries two (~5cm apart). The two final runs are CUT where they reach the
    corner diametrically opposite the start spoke (~node 74 of 100), so both
    chains terminate on that vertex. The cut bulbs don't exist; each pin
    still clocks a full field.

    Y-splitters pair run j of one chain with run j of the other on one pin —
    identical data at mirror wire positions = whole-canopy reflection about
    the start-spoke/terminal-vertex axis.

    Returns a list of (name, points, split), chains interleaved in pin order
    ccw0, cw0, ccw1, cw1, ... — split = bulb count of the spoke segment
    (bulbs [0,split) hub->mid-edge, the rest is rim).
    """
    n = len(verts)
    z = p["canopy_height_m"]
    cr = p["coupling_radius_m"]
    nodes = p["pair_run_nodes"]
    sep = p["pair_spoke_sep_m"] / 2.0
    f = p["canopy_start_spoke"]
    depth = (n + 1) // 2                     # 4 runs per chain on a heptagon

    def mid(k):
        ax_, ay_ = verts[k % n]
        bx_, by_ = verts[(k + 1) % n]
        return ((ax_ + bx_) / 2.0, (ay_ + by_) / 2.0)

    runs = []
    for j in range(depth):
        for dirn in (+1, -1):
            k = (f + j * dirn) % n           # this run's feed spoke
            th = mid_az[k]
            ox = oy = 0.0
            if j == 0:                       # only the start spoke is doubled
                ox, oy = -math.sin(th) * sep * dirn, math.cos(th) * sep * dirn
            start = (cr * math.cos(th) + ox, cr * math.sin(th) + oy, z)
            m0x, m0y = mid(k)
            m0 = (m0x + ox, m0y + oy, z)
            corner = verts[(k + 1) % n] if dirn > 0 else verts[k]
            far = mid(k + 1) if dirn > 0 else mid(k - 1)
            pts, total = _place_n([start, m0, corner + (z,), far + (z,)], nodes)
            pitch_eff = total / (nodes - 1)
            split = int(_d3(start, m0) / pitch_eff) + 1
            if j == depth - 1:               # final runs terminate on the far vertex
                s_corner = _d3(start, m0) + _d3(m0, corner + (z,))
                pts = pts[:int(s_corner / pitch_eff) + 1]
            runs.append((f"{'ccw' if dirn > 0 else 'cw'}{j}", pts, split))
    return runs


def build_long_spokes(p, base):
    """Long spoke k: one taut 50-node bulb strand, coupling edge -> attach
    point (corner azimuth, butterfly_offset outboard of the corner radius).
    """
    n = p["hept_sides"]
    z = p["canopy_height_m"]
    cr = p["coupling_radius_m"]
    R = p["hept_side_len_m"] / (2.0 * math.sin(math.pi / n))
    ar = R + p["butterfly_offset_m"]
    spokes, attach = [], []
    for k in range(n):
        th = base + 2.0 * math.pi * k / n
        a = (cr * math.cos(th), cr * math.sin(th), z)
        b = (ar * math.cos(th), ar * math.sin(th), z)
        pts, _ = _place_n([a, b], p["bulb_strand_nodes"])
        spokes.append(pts)
        attach.append(b)
    return spokes, attach


def build_guys(p, attach):
    """Guy k continues the long-spoke chain: from the attach point (top)
    straight down-and-out on the spoke's azimuth to a ground anchor placed
    where the full 5 m strip runs out. LED 0 at the TOP, so normalized
    position 0 reads as "inner/high" like every other kind.
    """
    lit = (p["guy_strip_leds"] - 1) * p["guy_strip_pitch_m"]
    h = p["canopy_height_m"]
    horiz = math.sqrt(max(0.0, lit * lit - h * h))
    guys = []
    for (tx, ty, tz) in attach:
        r = math.hypot(tx, ty)
        ux, uy = tx / r, ty / r
        anchor = (tx + ux * horiz, ty + uy * horiz, 0.0)
        pts, _ = _place_n([(tx, ty, tz), anchor], p["guy_strip_leds"])
        guys.append(pts)
    return guys, horiz


def build_helix(p):
    """Center pole helix: helix_strips strips (same product as the guys)
    wound around the pole, phase-staggered, chained serpentine — strip 0
    descends from the hub, strip 1 climbs back, and so on. The winding
    pitch falls out of the strip length vs pole height.
    """
    ns = p["helix_strips"]
    nled = p["helix_strip_leds"]
    r = p["helix_pole_radius_m"]
    h = p["helix_height_m"]
    lit = (nled - 1) * p["helix_strip_pitch_m"]
    wrap = math.sqrt(max(0.0, lit * lit - h * h)) / r   # total wind angle (rad)
    strips = []
    for j in range(ns):
        phase = 2.0 * math.pi * j / ns
        down = (j % 2 == 0)
        pts = []
        for i in range(nled):
            f = i / (nled - 1)
            z = h * (1.0 - f) if down else h * f
            th = phase + wrap * f
            pts.append((r * math.cos(th), r * math.sin(th), z))
        strips.append(pts)
    return strips, wrap


def build_roots(p):
    """Root poles on the floor, each dressed with ONE standard strip laid as
    a DOUBLE MEANDER: out along the pole with the azimuth swaying
    sinusoidally, around the far end cap, back as a parallel snake at a
    constant azimuth offset — the two runs never cross. The sway amplitude
    is FIT numerically so the path consumes exactly the strip's length
    (root_strip_leds x root_strip_pitch), no cut and no leftover.
    LED 0 is at the inner (center) end.
    (Earlier shape candidates — spiral wrap, serpentine cage, plain
    out-and-back — live at git 821e223 if we want them back.)
    """
    rng = random.Random(p["roots_seed"])
    R = p["root_radius_m"]
    L = p["root_len_m"]
    pitch = p["root_strip_pitch_m"]
    nled = p["root_strip_leds"]
    ncyc = p["root_meander_cycles"]
    dlt = p["root_meander_sep_rad"] / 2.0
    jitter = math.radians(p["roots_jitter_deg"])
    count = p["roots_count"]
    top = math.pi / 2.0
    target = (nled - 1) * pitch

    def plen(raw):
        return sum(_d3(raw[i], raw[i - 1]) for i in range(1, len(raw)))

    roots = []
    fit_A = 0.0
    for k in range(count):
        th = 2.0 * math.pi * k / count + rng.uniform(-jitter, jitter)
        ux, uy = math.cos(th), math.sin(th)
        vx, vy = -uy, ux
        x0, y0 = p["root_inner_m"] * ux, p["root_inner_m"] * uy

        def pt(s, phi):
            c = R * math.cos(phi)
            return (x0 + ux * s + c * vx, y0 + uy * s + c * vy,
                    R + R * math.sin(phi))

        def path(A):
            raw = []
            for i in range(513):
                s = L * i / 512
                raw.append(pt(s, top + dlt
                              + A * math.sin(2.0 * math.pi * ncyc * s / L)))
            for i in range(1, 24):
                raw.append(pt(L, top + dlt - 2.0 * dlt * i / 24))
            for i in range(513):
                s = L * (1.0 - i / 512)
                raw.append(pt(s, top - dlt
                              + A * math.sin(2.0 * math.pi * ncyc * s / L)))
            return raw

        lo, hi = 0.0, 2.4
        for _ in range(48):
            A = 0.5 * (lo + hi)
            if plen(path(A)) < target + 0.5 * pitch:
                lo = A
            else:
                hi = A
        fit_A = 0.5 * (lo + hi)
        raw = path(fit_A)
        cum = [0.0]
        for i in range(1, len(raw)):
            cum.append(cum[-1] + _d3(raw[i], raw[i - 1]))
        pts, j = [], 0
        for i in range(nled):
            d = min(i * pitch, cum[-1])
            while j < len(cum) - 2 and cum[j + 1] < d:
                j += 1
            seg = cum[j + 1] - cum[j]
            f = 0.0 if seg < 1e-9 else (d - cum[j]) / seg
            pts.append(_lerp(raw[j], raw[j + 1], f))
        roots.append(pts)
    return roots, fit_A


# ─────────────────────────────────────────────────────────────────────────
# Emit
# ─────────────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "geometry"))
    os.makedirs(out_dir, exist_ok=True)

    verts, mid_az, R, base = _frame(P)
    canopy_z = P["canopy_height_m"]
    pairs = build_pairs(P, verts, mid_az)
    spokes, attach = build_long_spokes(P, base)
    guys, guy_horiz = build_guys(P, attach)
    helices, helix_wrap = build_helix(P)
    roots, meander_amp = build_roots(P)

    flat, strips = [], []

    def add_strip(name, kind, pts, axn=0, axsplit=0, axoff=0):
        start = len(flat)
        flat.extend(pts)
        strips.append({"name": name, "kind": kind, "start": start,
                       "count": len(pts), "axn": axn,
                       "axsplit": axsplit, "axoff": axoff})

    # each chain's rim segments concatenate into ONE perimeter particle
    # array (mid-S0 -> terminal vertex); the other chain is its mirror
    rim_off, off = {}, {"ccw": 0, "cw": 0}
    for name, run, split in pairs:
        rim_off[name] = off[name[:-1]]
        off[name[:-1]] += len(run) - split
    assert off["ccw"] == off["cw"], "chain rim arrays must mirror"
    rim_total = off["ccw"]

    for name, run, split in pairs:
        add_strip(name, "canopy", run, rim_total,
                  axsplit=split, axoff=rim_off[name])
    for k in range(len(spokes)):
        # spoke_k and guy_k are ONE physical chain (bulbs, then strip)
        add_strip(f"spoke_{k}", "canopy", spokes[k], len(spokes[k]))
        add_strip(f"guy_{k}", "strap", guys[k])
    # helix axn = height raster resolution, matched to the spokes' node pitch
    helix_axn = P["bulb_strand_nodes"]
    for j, hx in enumerate(helices):
        add_strip(f"helix_{j}", "helix", hx, helix_axn)
    root_axn = int(round(P["root_len_m"] / P["root_strip_pitch_m"])) + 1
    for k, root in enumerate(roots):
        add_strip(f"root_{k}", "root", root, root_axn)

    n_total = len(flat)

    layout = [{"point": [round(x, 5), round(y, 5), round(z, 5)]} for (x, y, z) in flat]
    with open(os.path.join(out_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=0)

    meta = {
        "params": P,
        "n_total": n_total,
        "canopy_z_m": round(canopy_z, 4),
        "attach_radius_m": round(R + P["butterfly_offset_m"], 5),
        "guy_anchor_past_attach_m": round(guy_horiz, 4),
        "helix_wrap_turns": round(helix_wrap / (2.0 * math.pi), 2),
        "heptagon_vertices_xy": [[round(x, 5), round(y, 5)] for (x, y) in verts],
        "strips": strips,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    write_topology_h(os.path.join(out_dir, "topology.h"), flat, strips, P, canopy_z)

    # ── console summary ─────────────────────────────────────────────────
    print(f"strap-canopy geometry generated -> {out_dir}")
    print(f"  total LEDs: {n_total}")
    by_kind = {}
    for s in strips:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + s["count"]
    for kind, c in by_kind.items():
        print(f"    {kind:>7}: {c:>5} LEDs")
    cuts = [(nm, len(r)) for nm, r, _ in pairs if len(r) < P["pair_run_nodes"]]
    print(f"  canopy chains: {len(pairs)} runs, double-run spoke "
          f"S{P['canopy_start_spoke']}; cut runs: {cuts}")
    print(f"  canopy field: spoke split at bulb {pairs[0][2]}, "
          f"rim array {rim_total} bulbs per chain")
    print(f"  long spokes: {len(spokes)} x {len(spokes[0])} bulbs, "
          f"guys {len(guys)} x {P['guy_strip_leds']} LEDs "
          f"(anchor {guy_horiz/FT:.1f} ft past attach)")
    print(f"  helix: {len(helices)} x {P['helix_strip_leds']} LEDs, "
          f"{helix_wrap/(2*math.pi):.1f} turns each")
    print(f"  roots: {len(roots)} x double-meander, {P['root_strip_leds']} LEDs "
          f"each; fitted sway amplitude {math.degrees(meander_amp):.0f} deg")
    est_w = n_total * 0.06
    print(f"  rough power @ ~60mW/LED avg: ~{est_w:.0f} W  "
          f"(peak full-white ~{n_total*0.3:.0f} W) — sanity-check controllers/PSU")


def write_topology_h(path, flat, strips, p, canopy_z):
    n = len(flat)
    L = []
    L.append("// topology.h — strap-canopy 3D pixel map.")
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
    L.append("// Strip ranges into LED_POS[]. kind: 0=helix, 1=canopy, 2=root, 3=strap/guy.")
    L.append("// axn: axial raster resolution for wrapped/folded strips (0 = index space).")
    L.append("// Canopy chain runs split at the mid-edge junction: bulbs [0,axsplit)")
    L.append("// run the spoke, bulbs [axsplit,count) are rim — axoff is their offset")
    L.append("// into the chain's rim particle array and axn its total bulb count.")
    L.append("struct HcStrip { uint16_t start, count; uint8_t kind; uint16_t axn, axsplit, axoff; };")
    kind_map = {"helix": 0, "canopy": 1, "root": 2, "strap": 3}
    L.append("static const HcStrip HC_STRIPS[HC_NUM_STRIPS] = {")
    for s in strips:
        L.append(f"  {{ {s['start']}, {s['count']}, {kind_map[s['kind']]}, "
                 f"{s['axn']}, {s['axsplit']}, {s['axoff']} }}, // {s['name']}")
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
