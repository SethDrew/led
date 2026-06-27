#!/usr/bin/env python3
"""
analyze.py — spatial data analysis for helix-canopy effects.

This is the PRIMARY interface for the AI iteration loop: it answers questions
about an effect from its pixel data, not from a picture. Reads a sim CSV
(frame,t_s,pixel,x,y,z,r,g,b) + geometry/meta.json (for strip/kind labels) and
reports topology-aware metrics.

Per the ledsim doctrine, the *analysis* layer is the committable part (it
operates on data and doesn't drift). Add functions here as real questions arise.

  python3 analyze.py --frames sim/pour.csv
  python3 analyze.py --frames sim/pour.csv --json     # machine-readable

Functions are importable for ad-hoc analysis:
  from analyze import load, wavefront_z, cross_form_timing, coverage
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# Perceptual luminance weights (Rec.709), inputs 0..255 -> 0..1
_LW = np.array([0.2126, 0.7152, 0.0722]) / 255.0


def load(csv_path, meta_path=None):
    df = pd.read_csv(csv_path)
    df["lum"] = df[["r", "g", "b"]].to_numpy() @ _LW
    if meta_path is None:
        here = os.path.dirname(os.path.abspath(csv_path))
        guess = os.path.normpath(os.path.join(here, "..", "..", "geometry", "meta.json"))
        meta_path = guess if os.path.exists(guess) else None
    strip_name = {}
    strip_kind = {}
    if meta_path and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        for s in meta["strips"]:
            for i in range(s["start"], s["start"] + s["count"]):
                strip_name[i] = s["name"]
                strip_kind[i] = s["kind"]
        df["strip"] = df["pixel"].map(strip_name)
        df["kind"] = df["pixel"].map(strip_kind)
    else:
        df["strip"] = "all"
        df["kind"] = "all"
    return df


def basics(df):
    fr = sorted(df["frame"].unique())
    dur = df["t_s"].max()
    fps = (len(fr) - 1) / dur if dur > 0 else float("nan")
    return {"n_frames": len(fr), "n_leds": int(df["pixel"].nunique()),
            "duration_s": round(float(dur), 3), "fps": round(float(fps), 2)}


def coverage(df, thresh=0.05):
    """Fraction of LEDs that ever exceed `thresh` luminance, overall + per kind."""
    peak = df.groupby("pixel")["lum"].max()
    kind = df.drop_duplicates("pixel").set_index("pixel")["kind"]
    out = {"overall": round(float((peak > thresh).mean()), 3)}
    for k in sorted(kind.unique()):
        px = kind[kind == k].index
        out[k] = round(float((peak.loc[px] > thresh).mean()), 3)
    return out


def energy_envelope(df):
    """Total luminance per frame, and per kind. Returns a DataFrame indexed by t_s."""
    g = df.groupby(["t_s", "kind"])["lum"].sum().unstack(fill_value=0.0)
    g["total"] = g.sum(axis=1)
    return g


def wavefront_z(df, lum_floor=0.02, kind=None):
    """Brightness-weighted mean z per frame (the 'height of the light').
    Pass kind='helix' to track the descending front on the post only — the
    canopy's 800+ co-planar LEDs otherwise dominate the weighted mean and pin
    the global centroid near the ceiling (a real metric trap).
    Returns DataFrame: t_s, z_centroid, lum_sum."""
    d = df[df["lum"] > lum_floor]
    if kind is not None:
        d = d[d["kind"] == kind]
    grp = d.groupby("t_s")
    z = grp.apply(lambda x: np.average(x["z"], weights=x["lum"]), include_groups=False)
    lsum = grp["lum"].sum()
    return pd.DataFrame({"z_centroid": z, "lum_sum": lsum}).reset_index()


def descent_velocity(wf, z_top, z_bot):
    """Fit descent speed on the longest monotonically-falling run of z_centroid.
    Returns (v_mps, r2, frac_of_height_covered)."""
    z = wf["z_centroid"].to_numpy()
    t = wf["t_s"].to_numpy()
    # find longest strictly-decreasing run
    best = (0, 0)
    i = 0
    while i < len(z) - 1:
        j = i
        while j < len(z) - 1 and z[j + 1] < z[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = max(j, i + 1)
    a, b = best
    if b - a < 2:
        return None
    tt, zz = t[a:b + 1], z[a:b + 1]
    A = np.vstack([tt, np.ones_like(tt)]).T
    (slope, _), res, *_ = np.linalg.lstsq(A, zz, rcond=None)
    pred = A @ np.linalg.lstsq(A, zz, rcond=None)[0]
    ss_res = float(((zz - pred) ** 2).sum())
    ss_tot = float(((zz - zz.mean()) ** 2).sum()) or 1e-9
    r2 = 1 - ss_res / ss_tot
    frac = abs(zz[-1] - zz[0]) / max(1e-9, (z_top - z_bot))
    return {"v_mps": round(float(-slope), 4), "r2": round(r2, 4),
            "height_frac": round(float(frac), 3),
            "t_window": [round(float(tt[0]), 3), round(float(tt[-1]), 3)]}


def spatial_coherence(df, k=6):
    """How much does a LED's brightness resemble its NEAREST-IN-3D-SPACE
    neighbors? This is the litmus test for 'topology-native': a coordinate-driven
    effect varies smoothly through space (high coherence), while an index-driven
    effect scatters spatially-adjacent LEDs across the wave (low coherence).

    Returns Pearson r between each LED's luminance and its spatial-neighbor mean
    (over all frames), plus a scale-free roughness. r->1 coherent, r->0 not."""
    from scipy.spatial import cKDTree
    pos = df.drop_duplicates("pixel").sort_values("pixel")[["x", "y", "z"]].to_numpy()
    n = pos.shape[0]
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)        # includes self at col 0
    nbr = idx[:, 1:]                          # (n, k)

    lum_mat = df.pivot_table(index="frame", columns="pixel", values="lum",
                             fill_value=0.0).to_numpy()  # (F, n)
    nbr_mean = lum_mat[:, nbr].mean(axis=2)              # (F, n)
    a = lum_mat.ravel()
    b = nbr_mean.ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    mean_lum = max(1e-9, float(a.mean()))
    roughness = float(np.abs(a - b).mean() / mean_lum)
    return {"neighbor_r": round(r, 4), "roughness": round(roughness, 4), "k": k}


def _sorted_lum_matrix(df):
    """(frames x pixel) luminance matrix, columns sorted by wire index."""
    mat = df.pivot_table(index="frame", columns="pixel", values="lum", fill_value=0.0)
    return mat.reindex(sorted(mat.columns), axis=1).to_numpy()


def instant_section_energy(df):
    """Per-frame fraction of total luminance in each section, averaged over frames.
    Unlike coverage() ('ever lit'), this is INSTANTANEOUS — where the light *is* at
    any moment. An along-the-wire effect pours energy into the trunk (35% of the
    wire is helix); an across-the-bearing effect can't light the sparse helix and
    piles into the canopy. This separates effects that coverage() calls identical."""
    tot = df.groupby("frame")["lum"].sum()
    perk = df.groupby(["frame", "kind"])["lum"].sum().unstack(fill_value=0.0)
    frac = perk.div(tot.replace(0, np.nan), axis=0).mean()
    return {k: round(float(frac[k]), 3) for k in frac.sort_values(ascending=False).index}


def index_coherence(df, k=6):
    """Like spatial_coherence, but neighbors are taken in WIRE-INDEX space (i±k)
    instead of 3D space. Comparing the two tells you which coordinate the light is
    organized in: an along-strand (index-native) effect is coherent in BOTH; an
    across-strand effect is coherent in 3D but NOT in index (it lights one bearing
    across many strands, scattered through the wire). r->1 contiguous along wire."""
    mat = _sorted_lum_matrix(df)
    acc = np.zeros_like(mat)
    for s in range(1, k + 1):
        acc += np.roll(mat, s, axis=1) + np.roll(mat, -s, axis=1)
    nbr = acc / (2 * k)
    a, b = mat.ravel(), nbr.ravel()
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
    return {"index_neighbor_r": round(r, 4), "k": k}


def index_runs(df, thresh=0.05):
    """Per frame, group lit pixels into maximal consecutive-wire-index runs (the
    'segments' the eye reads as one creature). Along-strand light = a few LONG runs;
    across-strand light = many SHORT runs scattered through the index. Means over
    frames."""
    lit = _sorted_lum_matrix(df) > thresh
    lens, counts = [], []
    for row in lit:
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        runs = edges[1::2] - edges[0::2]
        counts.append(len(runs))
        if len(runs):
            lens.append(runs.mean())
    return {"mean_run_len": round(float(np.mean(lens)), 2) if lens else 0.0,
            "mean_run_count": round(float(np.mean(counts)), 2),
            "lit_per_frame": round(float(lit.sum(1).mean()), 1)}


def compare_structure(df_a, df_b, label_a="A", label_b="B"):
    """Side-by-side of the structural metrics that distinguish along-strand from
    across-strand light — the ones coverage()/spatial_coherence() are blind to."""
    a = {"section": instant_section_energy(df_a), "c3d": spatial_coherence(df_a)["neighbor_r"],
         "cix": index_coherence(df_a)["index_neighbor_r"], "runs": index_runs(df_a)}
    b = {"section": instant_section_energy(df_b), "c3d": spatial_coherence(df_b)["neighbor_r"],
         "cix": index_coherence(df_b)["index_neighbor_r"], "runs": index_runs(df_b)}
    print("\nWHAT THE DATA SHOWS — structural comparison (what coverage/3D-coherence miss)")
    print(f"  {'metric':<28}{label_a:>14}{label_b:>14}")
    for sec in ["helix", "canopy", "root"]:
        print(f"  instant energy · {sec:<11}{a['section'].get(sec,0):>13.0%}{b['section'].get(sec,0):>14.0%}")
    print(f"  {'coherence · 3D neighbors':<28}{a['c3d']:>14}{b['c3d']:>14}")
    print(f"  {'coherence · wire-index nbrs':<28}{a['cix']:>14}{b['cix']:>14}")
    print(f"  {'mean run length (LEDs)':<28}{a['runs']['mean_run_len']:>14}{b['runs']['mean_run_len']:>14}")
    print(f"  {'mean run count':<28}{a['runs']['mean_run_count']:>14}{b['runs']['mean_run_count']:>14}")
    return a, b


def cross_form_timing(df):
    """When does the canopy peak vs the helix, in each cycle? Reports the lag
    between canopy energy peak and helix energy peak (first cycle)."""
    env = energy_envelope(df)
    if "canopy" not in env or "helix" not in env:
        return None
    t = env.index.to_numpy()
    cz = env["canopy"].to_numpy()
    hz = env["helix"].to_numpy()
    tc = float(t[int(np.argmax(cz))])
    th = float(t[int(np.argmax(hz))])
    return {"canopy_peak_t": round(tc, 3), "helix_peak_t": round(th, 3),
            "lag_s": round(th - tc, 3),
            "canopy_peak_lum": round(float(cz.max()), 2),
            "helix_peak_lum": round(float(hz.max()), 2)}


def report(csv_path, meta_path=None, as_json=False):
    df = load(csv_path, meta_path)
    z_top = float(df["z"].max())
    z_bot = float(df["z"].min())
    # Track the descending front on the helix only (canopy mass biases a global
    # centroid toward the ceiling). Fall back to all pixels if no helix kind.
    has_helix = (df["kind"] == "helix").any()
    wf = wavefront_z(df, kind="helix" if has_helix else None)
    out = {
        "file": os.path.basename(csv_path),
        "basics": basics(df),
        "coverage": coverage(df),
        "wavefront": {
            "tracked_on": "helix" if has_helix else "all",
            "z_top": round(z_top, 3), "z_bot": round(z_bot, 3),
            "z_centroid_range": [round(float(wf["z_centroid"].min()), 3),
                                 round(float(wf["z_centroid"].max()), 3)],
            "descent": descent_velocity(wf, z_top, z_bot),
        },
        "cross_form_timing": cross_form_timing(df),
        "spatial_coherence": spatial_coherence(df),
    }
    if as_json:
        print(json.dumps(out, indent=2))
        return out

    b = out["basics"]
    print(f"\n=== {out['file']} ===")
    print(f"WHAT THE DATA SHOWS")
    print(f"  {b['n_frames']} frames, {b['n_leds']} LEDs, "
          f"{b['duration_s']}s @ {b['fps']}fps")
    cov = out["coverage"]
    print(f"  coverage (ever lit): overall {cov['overall']:.0%}  " +
          "  ".join(f"{k} {v:.0%}" for k, v in cov.items() if k != "overall"))
    w = out["wavefront"]
    print(f"  light-height (z centroid, on {w['tracked_on']}) ranged "
          f"{w['z_centroid_range'][0]} -> {w['z_centroid_range'][1]} m  "
          f"(sculpture {w['z_bot']}..{w['z_top']} m)")
    d = w["descent"]
    if d:
        print(f"  descent: {d['v_mps']} m/s over {d['t_window']}s, "
              f"covered {d['height_frac']:.0%} of height, linearity R^2={d['r2']}")
    ct = out["cross_form_timing"]
    if ct:
        print(f"  cross-form: canopy peaks @ {ct['canopy_peak_t']}s, "
              f"helix @ {ct['helix_peak_t']}s, lag {ct['lag_s']}s")
    sc = out["spatial_coherence"]
    print(f"  spatial coherence (neighbor r): {sc['neighbor_r']}  "
          f"roughness {sc['roughness']}  (r->1 topology-native, r->0 index-native)")
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report(args.frames, args.meta, args.json)


if __name__ == "__main__":
    main()
