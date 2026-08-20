#!/usr/bin/env python3
"""
geometry_gen.py — parametric geometry for the strap-canopy sculpture.

Canopy-on-straps form (2026-08-17 reshape — the middle helix is gone):
  - a FILLED HEPTAGON canopy held up at each tip,
  - one RATCHET STRAP per tip (decorated), running from the tip radially
    down-and-out to a ground anchor — the bm26 n7-radial single-guy layout,
  - a ROOT SYSTEM of paper-mache poles radiating on the floor, each decorated
    with an IP67 strip in a different shape (spiral / serpentine cage /
    out-and-back / meander).

Anti-drift principle (the ledsim doctrine, applied to geometry):
  The position math lives in exactly ONE place — here. We compute every LED's
  (x, y, z) once and emit it as literal data to two consumers:
    - geometry/layout.json  — OPC-style flat array, for the 3D viewer / tooling
    - geometry/topology.h   — Vec3 LED_POS[] literal table, for the firmware
  The device reads the table; it never recomputes geometry, so nothing drifts.

Units: meters internally. Z is up. The center axis is the z axis through (0, 0).

Wire/strip model (festicorn multi-strip setPixel(s, i) convention):
  strip 0..13  — canopy spokes (center hole -> rim, then along the rim)
  strip 14..20 — ratchet straps (canopy tip -> ground anchor)
  strip 21..27 — roots (one strip each)
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
    # ── Ratchet straps (canopy support, decorated) ───────────────────────
    # One strap per canopy tip, on the tip's radial azimuth (bm26 n7-radial,
    # single guy per pole): tip at canopy height -> ground anchor strap_run
    # horizontally past the tip's floor projection.
    "strap_run_m":        96.0 * IN,    # bm26 GUY_RUN default (8 ft past the tip)
    "strap_led_pitch_m":  0.05,         # decorated with 20/m strip along the strap

    # ── Heptagon canopy ──────────────────────────────────────────────────
    "hept_sides":         7,
    "hept_side_len_m":    10.0 * FT,    # each edge 10 ft -> circumradius ~11.5 ft
    "canopy_height_m":    10.0 * FT,    # the canopy is ONE flat plane
    # The net pulls each side inward: the rim bows toward the center between
    # tips, deepest at each side's midpoint (concave scalloped outline).
    "canopy_edge_pull_m": 1.5 * FT,
    # Real hardware: 12mm WS2811 bullet strands, 50 nodes @ 100mm pitch (10/m, 5m).
    # The canopy is radial spokes — one strand per spoke, center hole -> rim.
    "canopy_spoke_count":  14,          # spokes (14 = vertices + edge midpoints; rim arcs meet)
    "canopy_pixel_pitch_m": 0.10,       # 100mm node pitch (the real default)
    "canopy_strand_nodes": 50,          # nodes per real strand (caps a spoke's length)
    "canopy_rotation_deg": 0.0,
    "canopy_center_hole_m": 0.35,       # center hole where the spokes start

    # ── Root system: one standard IP67 strip per paper-mache pole ──────────
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
def build_straps(p, verts, canopy_z):
    """One TAUT ratchet strap per canopy tip: straight line from the tip
    radially down-and-out to a ground anchor. LED index 0 is at the TIP
    (top), matching the canopy spokes' center-out convention: normalized
    position 0 reads as "inner/high" on both kinds.
    """
    run = p["strap_run_m"]
    pitch = p["strap_led_pitch_m"]
    straps = []
    for (vx, vy), z in ((v, canopy_z) for v in verts):
        r = math.hypot(vx, vy)
        ux, uy = vx / r, vy / r
        tip = (vx, vy, z)
        anchor = (vx + ux * run, vy + uy * run, 0.0)
        pts, _ = resample_polyline([tip, anchor], pitch)
        straps.append(pts)
    return straps


def _heptagon_vertices(p):
    n = p["hept_sides"]
    R = p["hept_side_len_m"] / (2.0 * math.sin(math.pi / n))   # circumradius from side
    rot = math.radians(p["canopy_rotation_deg"]) + math.pi / 2.0
    verts = [(R * math.cos(rot + 2*math.pi*k/n), R * math.sin(rot + 2*math.pi*k/n))
             for k in range(n)]
    return verts, R


def _ray_poly_hit(theta, verts):
    """Where a ray from the origin at angle theta exits the (convex) polygon.

    Returns (t, hitpoint, edge_index, u) for the nearest forward boundary
    crossing; edge_index is the edge verts[k]->verts[k+1] that was hit.
    """
    dx, dy = math.cos(theta), math.sin(theta)
    n = len(verts)
    best = None
    for k in range(n):
        ax, ay = verts[k]; bx, by = verts[(k + 1) % n]
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        t = (ax * ey - ay * ex) / denom          # origin at (0,0)
        u = (ax * dy - ay * dx) / denom
        if t > 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            if best is None or t < best[0]:
                best = (t, (dx * t, dy * t), k, u)
    return best


def _march_polyline(waypoints, pitch, cap):
    """Place up to `cap` nodes at `pitch` arc-length spacing along a 3D polyline."""
    segs = []
    for a, b in zip(waypoints, waypoints[1:]):
        L = _d3(a, b)
        if L < 1e-9:
            continue
        segs.append((a, ((b[0]-a[0])/L, (b[1]-a[1])/L, (b[2]-a[2])/L), L))
    pts = []
    for k in range(cap):
        s = k * pitch
        acc = 0.0
        for (a, u, L) in segs:
            if s <= acc + L:
                t = s - acc
                pts.append((a[0] + u[0]*t, a[1] + u[1]*t, a[2] + u[2]*t))
                break
            acc += L
        else:
            break                                 # ran off the end of the polyline
    return pts


def build_canopy(p):
    """Radial spokes that bend along the rim — one real bullet strand per spoke.

    Each strand is a single continuous run at the real 100mm node pitch: it goes
    radially from the center-hole edge out to the heptagon rim, then BENDS and
    continues horizontally along the perimeter (CCW) until the strand's node cap
    is used up. So a 50-node strand spends ~32 nodes on the spoke and its ~18
    spare nodes wrapping around the edge, instead of draping down. Spokes are
    evenly spaced (spoke_count == hept_sides -> one per vertex).
    """
    verts, R = _heptagon_vertices(p)
    canopy_z = p["canopy_height_m"]
    hole = p["canopy_center_hole_m"]
    pitch = p["canopy_pixel_pitch_m"]
    count = p["canopy_spoke_count"]
    cap = p["canopy_strand_nodes"]
    base = math.radians(p["canopy_rotation_deg"]) + math.pi / 2.0

    # Concave rim: each heptagon side sampled densely and pulled toward the
    # center with a sin(pi t) profile — 0 at the tips, canopy_edge_pull deep
    # at the side midpoints. Star-shaped wrt center, so a spoke ray still
    # crosses it exactly once and _ray_poly_hit works unchanged.
    pull = p["canopy_edge_pull_m"]
    n = len(verts)
    SAMP = 48
    rim = []
    for k in range(n):
        ax_, ay_ = verts[k]
        bx_, by_ = verts[(k + 1) % n]
        for i in range(SAMP):
            t = i / SAMP
            x = ax_ + (bx_ - ax_) * t
            y = ay_ + (by_ - ay_) * t
            d = math.hypot(x, y)
            f = pull * math.sin(math.pi * t) / d
            rim.append((x * (1.0 - f), y * (1.0 - f)))

    spokes, radial_counts = [], []
    for s in range(count):
        theta = base + 2.0 * math.pi * s / count
        start = (hole * math.cos(theta), hole * math.sin(theta), canopy_z)
        hit = _ray_poly_hit(theta, rim)
        if hit is None:                            # degenerate; fall back to straight spoke
            pts = _march_polyline(
                [start, (R * math.cos(theta), R * math.sin(theta), canopy_z)],
                pitch, cap)
            spokes.append(pts)
            radial_counts.append(len(pts))
            continue
        _, hitpt, edge_idx, u = hit
        # waypoints: hole -> rim hit -> CCW along the scalloped rim (enough
        # points to cover the strand's full length)
        hit3 = (hitpt[0], hitpt[1], canopy_z)
        waypoints = [start, hit3]
        waypoints += [(rim[(edge_idx + 1 + j) % len(rim)][0],
                       rim[(edge_idx + 1 + j) % len(rim)][1],
                       canopy_z) for j in range(len(rim) // 2)]
        spokes.append(_march_polyline(waypoints, pitch, cap))
        # LEDs on the RADIAL section (before the rim bend) — the renderer's
        # spoke/rim boundary
        radial_counts.append(min(cap, int(_d3(start, hit3) / pitch) + 1))
    return spokes, radial_counts, verts, canopy_z


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

    canopy_spokes, canopy_radial, hept_verts, canopy_z = build_canopy(P)
    straps = build_straps(P, hept_verts, canopy_z)
    roots, meander_amp = build_roots(P)

    flat, strips = [], []

    def add_strip(name, kind, pts, axn=0):
        start = len(flat)
        flat.extend(pts)
        strips.append({"name": name, "kind": kind, "start": start,
                       "count": len(pts), "axn": axn})

    for k, spoke in enumerate(canopy_spokes):       # one real strand per spoke
        # canopy axn = radial-section LED count: the spoke/rim boundary
        add_strip(f"canopy_{k}", "canopy", spoke, canopy_radial[k])
    for k, strap in enumerate(straps):
        add_strip(f"strap_{k}", "strap", strap)
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
        "strap_leds_each": len(straps[0]) if straps else 0,
        "heptagon_vertices_xy": [[round(x, 5), round(y, 5)] for (x, y) in hept_verts],
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
    print(f"  straps: {len(straps)} x {len(straps[0])} LEDs "
          f"(tip z {canopy_z/FT:.0f} ft -> anchor {P['strap_run_m']/FT:.1f} ft past tip)")
    wire = (P["root_strip_leds"] - 1) * P["root_strip_pitch_m"]
    print(f"  roots: {len(roots)} x double-meander, {P['root_strip_leds']} LEDs "
          f"({wire:.2f} m strip) each; fitted sway amplitude "
          f"{math.degrees(meander_amp):.0f} deg")
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
    L.append("// Strip ranges into LED_POS[]. kind: 0=helix(retired), 1=canopy, 2=root, 3=strap.")
    L.append("// axn: axial raster resolution for wrapped/folded strips (0 = index space).")
    L.append("// A renderer rasterizing a root at axn pixels and sampling by each LED's")
    L.append("// axial position makes effects travel DOWN the pole, whatever the wire path.")
    L.append("struct HcStrip { uint16_t start, count; uint8_t kind; uint16_t axn; };")
    kind_map = {"helix": 0, "canopy": 1, "root": 2, "strap": 3}
    L.append("static const HcStrip HC_STRIPS[HC_NUM_STRIPS] = {")
    for s in strips:
        L.append(f"  {{ {s['start']}, {s['count']}, {kind_map[s['kind']]}, {s['axn']} }}, // {s['name']}")
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
