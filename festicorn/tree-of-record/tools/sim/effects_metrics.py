"""Reusable brightness + motion metrics for tree-of-record effect CSVs.

Operates on the tidy long-form CSV emitted by effects_sim.cpp:
    effect,frame,t_s,strip,pixel,r,g,b   (8-bit output pixels)

Unlike the sim (which mirrors drifting firmware) these functions operate on
data and don't drift, so they're safe to commit and reuse.

Luminance is perceptual Rec.709: 0.2126R + 0.7152G + 0.0722B on the 8-bit
output (0..255). Metrics:

  per_pixel_brightness(df) -> dict
    mean_lum  : mean per-pixel luminance over all frames+pixels (0..255)
    peak_lum  : max per-pixel luminance seen in the run
    frac_lit  : fraction of (frame,pixel) samples with luminance > 0

  motion_score(df) -> dict
    motion_lum_per_s : mean over pixels of mean |Δluminance| between
                       consecutive frames, divided by dt -> lum/s. Primary
                       "speed" metric: how fast the picture changes.
    drift_lum_per_s  : coarse spatial drift proxy = mean over frames of the
                       mean |luminance[i+1]-luminance[i]| spatial gradient,
                       cheap structure indicator (not a true optical-flow).
"""
import numpy as np
import pandas as pd

_LUM = np.array([0.2126, 0.7152, 0.0722])


def _luminance(df):
    return df["r"].to_numpy() * _LUM[0] + \
           df["g"].to_numpy() * _LUM[1] + \
           df["b"].to_numpy() * _LUM[2]


def _frame_pixel_lum(df):
    """Return (n_frames, n_pixels) luminance array sorted by frame then (strip,pixel)."""
    df = df.sort_values(["frame", "strip", "pixel"])
    lum = _luminance(df)
    n_frames = df["frame"].nunique()
    n_pix = len(df) // n_frames
    return lum.reshape(n_frames, n_pix)


def per_pixel_brightness(df):
    lum = _luminance(df)
    lit = lum[lum > 0]
    return {
        "mean_lum": float(lum.mean()),          # field average (all pixels)
        "lit_mean_lum": float(lit.mean()) if lit.size else 0.0,  # mean over lit pixels only
        "peak_lum": float(lum.max()),
        "frac_lit": float((lum > 0).mean()),
    }


def motion_score(df, dt):
    arr = _frame_pixel_lum(df)              # (frames, pixels)
    # temporal: mean |Δ| between consecutive frames, per pixel, /dt
    dframe = np.abs(np.diff(arr, axis=0))   # (frames-1, pixels)
    motion_lum_per_s = float(dframe.mean() / dt)
    # spatial drift proxy: mean abs gradient along the pixel axis, /dt-equiv
    dspace = np.abs(np.diff(arr, axis=1))   # (frames, pixels-1)
    drift = float(dspace.mean())
    return {"motion_lum_per_s": motion_lum_per_s, "drift_spatial": drift}


def analyze_file(path, dt):
    df = pd.read_csv(path)
    b = per_pixel_brightness(df)
    m = motion_score(df, dt)
    return {"effect": df["effect"].iloc[0], **b, **m}


if __name__ == "__main__":
    import sys, glob
    DT = 1.0 / 200.0
    files = sys.argv[1:] or sorted(glob.glob("/tmp/effects_csv/fx_*.csv"))
    rows = [analyze_file(f, DT) for f in files]
    out = pd.DataFrame(rows)
    # firmware enum order
    order = ["AMBIENT_BLOOM", "FIRE", "HEAT_DIFFUSION", "WORLEY",
             "GRAVITY_PARTICLE", "SPARKLE_SYLLABLE", "CREATURES",
             "POLYCULE", "LEAF_WIND", "LIGHT_THROUGH", "NEBULA", "RAINBOW"]
    out["__o"] = out["effect"].map({n: i for i, n in enumerate(order)})
    out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    pd.set_option("display.width", 140)
    print(out.to_string(index=False))
