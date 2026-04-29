#!/usr/bin/env python3
"""
Extract per-episode gesture features from MPU-6050 IMU CSVs.

Inputs: a CSV (t,ax,ay,az,gx,gy,gz,rms) at 200 Hz and an episodes JSON
(list of {start_s, end_s, ...} or {name, start_s, end_s}).

Output: a unified per-episode feature dict, written to a JSON and printed
as a markdown table.

The features are designed to support a (kind, severity) classifier:

KIND-discriminating features
  crest_factor      peak|a| / rms|a|   (>5 impulsive, ~3 smooth/sinusoid)
  ac_strength       autocorr peak height of |a|, lag 0.05–0.5 s
  ac_freq_hz        frequency of that autocorr peak
  n_peaks           sharp peaks in |a| (height>3*rms, dist>30 ms)
  stroke_rate_hz    n_peaks / duration
  grav_angle_deg    angle between gravity at episode start and end
  duration_s
  dom_accel_axis    X / Y / Z

SEVERITY-discriminating features
  peak_amag         clip-prone above ~58k counts
  rms_amag          best severity proxy when peak saturates
  integrated_amag   sum |a|*dt (impulse-like total, not normalized by duration)
  peak_gmag, rms_gmag
  gyro_sat_frac     fraction of samples where any gyro axis hit ±32767
  accel_clip_frac   fraction of samples where any accel axis hit ±32767

Gravity baseline: We use a 1.0 s rolling mean per accel axis to track gravity
as it moves during the recording (matches duck_gestures method, supersedes
hanging's static-baseline method so both captures are comparable).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FS = 200.0  # Hz
DT = 1.0 / FS
ACCEL_CLIP = 32767
GYRO_CLIP = 32767  # at ±250 °/s setting


def rolling_mean(x: np.ndarray, win_samples: int) -> np.ndarray:
    """Centered rolling mean using cumulative sum."""
    if win_samples < 2:
        return x.copy()
    pad = win_samples // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    csum = np.cumsum(xp, dtype=np.float64)
    csum = np.concatenate([[0.0], csum])
    out = (csum[win_samples:] - csum[:-win_samples]) / win_samples
    return out[: len(x)]


def remove_gravity(a_xyz: np.ndarray, win_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (ac_coupled_accel, gravity_baseline). Both shape (N,3)."""
    win = max(2, int(round(win_s * FS)))
    grav = np.stack([rolling_mean(a_xyz[:, i], win) for i in range(3)], axis=1)
    return a_xyz - grav, grav


def autocorr_peak(sig: np.ndarray, min_lag: int, max_lag: int) -> tuple[float, float]:
    """Find the highest autocorrelation peak in [min_lag, max_lag] (samples).

    Returns (strength, freq_hz). Strength is normalized so 1.0 means perfect
    self-similarity at that lag.
    """
    s = sig - sig.mean()
    denom = float(np.dot(s, s))
    if denom <= 0 or len(s) <= max_lag + 2:
        return 0.0, float("nan")
    best_strength = 0.0
    best_lag = 0
    for lag in range(min_lag, max_lag + 1):
        num = float(np.dot(s[:-lag], s[lag:]))
        strength = num / denom
        if strength > best_strength:
            best_strength = strength
            best_lag = lag
    if best_lag == 0:
        return 0.0, float("nan")
    return best_strength, FS / best_lag


def find_peaks_simple(sig: np.ndarray, height: float, min_dist: int) -> list[int]:
    """Local maxima above height, separated by at least min_dist samples."""
    out: list[int] = []
    for i in range(1, len(sig) - 1):
        if sig[i] > height and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
            if not out or i - out[-1] >= min_dist:
                out.append(i)
            elif sig[i] > sig[out[-1]]:
                out[-1] = i
    return out


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def episode_features(
    accel_raw: np.ndarray,
    gyro_raw: np.ndarray,
    grav: np.ndarray,
    accel_ac: np.ndarray,
    i0: int,
    i1: int,
) -> dict:
    """Compute features over [i0, i1)."""
    a = accel_ac[i0:i1]               # gravity-removed accel, (N,3)
    g_raw = gyro_raw[i0:i1]           # gyro raw counts (gyro bias subtracted later)
    grav_seg = grav[i0:i1]            # gravity baseline trace
    a_full_raw = accel_raw[i0:i1]

    # Gyro bias: estimate as mean over the whole record (caller passes the
    # bias-corrected gyro). Here we just use it directly.
    g = g_raw

    n = max(1, len(a))
    duration_s = n * DT

    amag = np.linalg.norm(a, axis=1)
    gmag = np.linalg.norm(g, axis=1)

    peak_amag = float(amag.max()) if n else 0.0
    rms_amag = float(math.sqrt(float(np.mean(amag * amag)))) if n else 0.0
    integrated_amag = float(amag.sum() * DT)
    peak_gmag = float(gmag.max()) if n else 0.0
    rms_gmag = float(math.sqrt(float(np.mean(gmag * gmag)))) if n else 0.0

    crest = peak_amag / rms_amag if rms_amag > 0 else 0.0

    # Clipping fractions
    accel_clip = (np.abs(a_full_raw) >= ACCEL_CLIP - 1).any(axis=1).mean()
    gyro_sat = (np.abs(g) >= GYRO_CLIP - 1).any(axis=1).mean()

    # Peak detection on |a|: peaks > 2.5 * rms, min distance 25 ms
    peak_idx = find_peaks_simple(amag, height=2.5 * rms_amag, min_dist=int(0.025 * FS))
    n_peaks = len(peak_idx)
    stroke_rate_hz = n_peaks / duration_s if duration_s > 0 else 0.0

    # Autocorrelation: lag range ~50 ms (20 Hz) to ~min(0.5 s, half-window)
    min_lag = max(2, int(0.05 * FS))
    max_lag = min(int(0.5 * FS), max(min_lag + 1, n // 3))
    if max_lag > min_lag:
        ac_str, ac_freq = autocorr_peak(amag, min_lag, max_lag)
    else:
        ac_str, ac_freq = 0.0, float("nan")

    # Axis dominance from AC-coupled accel
    energy_xyz = np.sum(a * a, axis=0)
    total = float(energy_xyz.sum())
    if total > 0:
        axis_share = (energy_xyz / total).tolist()
    else:
        axis_share = [0.0, 0.0, 0.0]
    dom_accel_axis = ["X", "Y", "Z"][int(np.argmax(energy_xyz))]

    # Gravity rotation: angle between mean gravity in first 100 ms and last 100 ms
    edge = max(1, int(0.1 * FS))
    g_pre = grav_seg[:edge].mean(axis=0)
    g_post = grav_seg[-edge:].mean(axis=0)
    grav_angle = angle_between(g_pre, g_post)

    return {
        "duration_s": round(duration_s, 3),
        "peak_amag": round(peak_amag, 1),
        "rms_amag": round(rms_amag, 1),
        "integrated_amag": round(integrated_amag, 1),
        "crest": round(crest, 2),
        "peak_gmag": round(peak_gmag, 1),
        "rms_gmag": round(rms_gmag, 1),
        "n_peaks": n_peaks,
        "stroke_rate_hz": round(stroke_rate_hz, 2),
        "ac_strength": round(ac_str, 3),
        "ac_freq_hz": round(ac_freq, 2) if not math.isnan(ac_freq) else None,
        "dom_accel_axis": dom_accel_axis,
        "axis_share_xyz": [round(v, 3) for v in axis_share],
        "accel_clip_frac": round(float(accel_clip), 3),
        "gyro_sat_frac": round(float(gyro_sat), 3),
        "grav_angle_deg": round(grav_angle, 1),
    }


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (accel_int16, gyro_int16) shape (N,3) each."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, comments="#")
    accel = arr[:, 1:4].astype(np.int32)
    gyro = arr[:, 4:7].astype(np.int32)
    return accel, gyro


def estimate_gyro_bias(gyro: np.ndarray, calm_seconds: float = 5.0) -> np.ndarray:
    n = min(len(gyro), int(calm_seconds * FS))
    return gyro[:n].mean(axis=0)


def process_capture(csv_path: Path, episodes_path: Path) -> list[dict]:
    accel, gyro = load_csv(csv_path)
    gyro_bias = estimate_gyro_bias(gyro, calm_seconds=5.0)
    gyro_corr = (gyro - gyro_bias).astype(np.float64)
    accel_f = accel.astype(np.float64)

    accel_ac, grav = remove_gravity(accel_f, win_s=1.0)

    eps_raw = json.loads(episodes_path.read_text())
    out: list[dict] = []
    for ep in eps_raw:
        i0 = int(round(ep["start_s"] * FS))
        i1 = int(round(ep["end_s"] * FS))
        i0 = max(0, min(i0, len(accel_f) - 1))
        i1 = max(i0 + 1, min(i1, len(accel_f)))
        feat = episode_features(accel, gyro_corr, grav, accel_ac, i0, i1)
        feat["start_s"] = ep["start_s"]
        feat["end_s"] = ep["end_s"]
        feat["name"] = str(ep.get("name", ep.get("ep", "?")))
        feat["capture"] = csv_path.stem
        out.append(feat)
    return out


def md_table(rows: list[dict]) -> str:
    cols = [
        "capture", "name", "duration_s",
        "peak_amag", "rms_amag", "integrated_amag", "crest",
        "peak_gmag", "rms_gmag", "gyro_sat_frac", "accel_clip_frac",
        "n_peaks", "stroke_rate_hz", "ac_strength", "ac_freq_hz",
        "dom_accel_axis", "grav_angle_deg",
    ]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path,
                    default=Path("/Users/sethdrew/Documents/projects/led/festicorn/gyro-sense/data/recordings/gesture_features.json"))
    args = ap.parse_args()

    captures = [
        (
            Path("/Users/sethdrew/Documents/projects/led/festicorn/gyro-sense/data/recordings/hanging/20260426T195344.csv"),
            Path("/Users/sethdrew/Documents/projects/led/festicorn/gyro-sense/data/recordings/hanging/20260426T195344_episodes.json"),
        ),
        (
            Path("/Users/sethdrew/Documents/projects/led/festicorn/gyro-sense/data/recordings/duck_gestures/20260426T210735.csv"),
            Path("/Users/sethdrew/Documents/projects/led/festicorn/gyro-sense/data/recordings/duck_gestures/20260426T210735_episodes.json"),
        ),
    ]

    all_rows: list[dict] = []
    for csv_path, eps_path in captures:
        all_rows.extend(process_capture(csv_path, eps_path))

    args.out_json.write_text(json.dumps(all_rows, indent=2))
    print(md_table(all_rows))
    print(f"\n[{len(all_rows)} episodes -> {args.out_json}]")


if __name__ == "__main__":
    main()
