#!/usr/bin/env python3
"""
Downsampler simulator + receiver-side feature recovery for the bulb-telemetry
wire format.

Pipeline:
  CSV (200 Hz raw int16 IMU)
    -> on-device gravity tracker (1 s rolling mean)
    -> 40 ms windows (8 samples), one packet per window
    -> packet bytes (16 B base, 17 B with mic)
    -> wire-decoded view
    -> receiver feature recovery (per-episode)

Run:
  /Users/sethdrew/Documents/projects/led/audio-reactive/venv/bin/python \
    tools/simulate_wire.py [--mic]

Output:
  data/recordings/wire_packets/<capture>.bin   (raw packet bytes, replay)
  data/recordings/wire_packets/<capture>.json  (decoded packets, debug)
  data/recordings/gesture_features_recovered.json  (receiver-side features)
  printed fidelity-loss table comparing raw vs recovered features.

Wire format (16 B base) — Option A semantics (matches production sender.cpp):
  off  type   field           encoding
  0-1  uint16 seq             packet counter, little-endian
  2    int8   ax_max          ac-coupled per-axis max  ((rawMax - rawMean) >> 8)
  3    int8   ay_max
  4    int8   az_max
  5    int8   ax_min          ac-coupled per-axis min  ((rawMin - rawMean) >> 8)
  6    int8   ay_min
  7    int8   az_min
  8    int8   ax_mean         per-axis raw mean (rawMean >> 8) — carries the gravity vector
  9    int8   ay_mean
  10   int8   az_mean
  11   uint8  amag_max        sqrt-companded |a_raw| max  (gravity INCLUDED, ~1g at rest)
  12   uint8  amag_mean       sqrt-companded mean(|a_raw|)
  13   uint8  gmag_max        sqrt-companded |g| max
  14   uint8  gmag_mean       sqrt-companded |g| mean
  15   uint8  flags           bit0:accel_clip bit1:gyro_sat bit2:bgMotion bit3..7:rsv
  (16  uint8  rawRms          OPTIONAL log-companded mic RMS — 0 when audio off)

OPTION-A SEMANTICS — IMPORTANT for downstream consumers:

  - Per-axis bytes (ax/ay/az_max, _min) ARE AC-coupled at the sender:
    sender computes `(rawMax - rawMean) >> 8` so the byte is signed deviation
    from the window's gravity baseline. Receiver consumes these directly as
    "AC accel" without further gravity subtraction.
  - Per-axis MEAN bytes (ax/ay/az_mean) are the RAW per-axis window mean
    (`rawMean >> 8`) — they carry the gravity vector trajectory. These are
    NOT gravity-removed.
  - Magnitude bytes (amag_*, gmag_*) are computed from RAW accel/gyro per
    sample, then maxed/meaned over the 8-sample window, then sqrt-companded.
    `amag_*` therefore INCLUDES gravity: at rest, amag_max ≈ amag_mean ≈ 1g.
    Effects that want "AC magnitude" can reconstruct from per-axis bytes as
    sqrt(ax_max² + ay_max² + az_max²); they cannot get it from amag_max.
  - The crest-factor proxy `(amag_max_byte / amag_mean_byte)²` is therefore
    always close to 1 (rest=1; a 2g lateral tap gives 4). The historical
    "≥ 4.5 = impulsive" gate assumed gravity-removed amag and is structurally
    unreachable under Option A. Don't use it as a kind discriminator.

Why Option A: lets the sender ship raw |a| (one sqrt per sample, no extra
gravity-tracker work for the magnitude path) while still giving the receiver
a per-axis AC view via the int8 axis bytes. Receivers needing a clean gravity
vector during dynamic motion can EMA the per-axis _mean bytes (~1–2 s tau)
on the receiver side — no extra wire bytes required.

Companding (sqrt) — production-matched constants (sender.cpp:90):
  uint8 = round( clip( sqrt(value / RANGE) * 255, 0, 255 ) )
  decode: value = (byte/255)^2 * RANGE
  RANGE_AMAG = 57000     # AMAG_FS — sender.cpp #define
  RANGE_GMAG = 57000     # GMAG_FS — sender.cpp #define
At ±4g (8192 counts/g), 57000 corresponds to ~6.96 g — leaves headroom past
the rail without wasting bits.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

# Reuse the raw-feature extractor for ground truth comparison
import sys
sys.path.insert(0, str(Path(__file__).parent))
import extract_gesture_features as efx

FS = 200.0
WINDOW_SAMPLES = 8           # 40 ms windows -> 25 Hz packet rate
PACKET_HZ = FS / WINDOW_SAMPLES
# Production-matched companding full-scales (sender.cpp:90 — AMAG_FS / GMAG_FS).
# Single source of truth — if the sender constants change, update here too.
RANGE_AMAG = 57000.0
RANGE_GMAG = 57000.0
ACCEL_CLIP = 32767
GYRO_CLIP = 32767


def sqrt_compand(values: np.ndarray, full_scale: float) -> np.ndarray:
    """Encode float -> uint8 with sqrt companding."""
    norm = np.clip(values / full_scale, 0.0, 1.0)
    return np.round(np.sqrt(norm) * 255.0).astype(np.uint8)


def sqrt_decompand(bytes_in: np.ndarray, full_scale: float) -> np.ndarray:
    n = bytes_in.astype(np.float64) / 255.0
    return n * n * full_scale


def axis_int8_pack(values: np.ndarray) -> np.ndarray:
    """Pack int16 values into int8 by /256 (high byte)."""
    return np.clip(np.round(values / 256.0), -128, 127).astype(np.int8)


def axis_int8_unpack(bytes_in: np.ndarray) -> np.ndarray:
    return bytes_in.astype(np.int32) * 256


def encode_packets(
    accel_raw: np.ndarray,
    gyro_corr: np.ndarray,
    accel_ac: np.ndarray,
    grav: np.ndarray,
    with_mic: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Turn 200 Hz traces into a sequence of 16(/17) B packets.

    Returns (raw_bytes_array, decoded_list).
    """
    n_full = (len(accel_raw) // WINDOW_SAMPLES) * WINDOW_SAMPLES
    accel_raw = accel_raw[:n_full]
    gyro_corr = gyro_corr[:n_full]
    accel_ac = accel_ac[:n_full]
    grav = grav[:n_full]

    n_pkt = n_full // WINDOW_SAMPLES
    pkt_size = 17 if with_mic else 16
    raw = np.zeros((n_pkt, pkt_size), dtype=np.uint8)
    decoded: list[dict] = []

    # Per-window aggregations, vectorized.
    # NOTE: production sender uses the per-window mean (rawSum/8) as gravity
    # baseline for the AC max/min — NOT the 1s rolling mean. Mirror that here.
    a_ac_w   = accel_ac.reshape(n_pkt, WINDOW_SAMPLES, 3)        # 1s-rolling-mean AC (debug only)
    g_w      = gyro_corr.reshape(n_pkt, WINDOW_SAMPLES, 3)
    a_full_w = accel_raw.reshape(n_pkt, WINDOW_SAMPLES, 3)
    grav_w   = grav.reshape(n_pkt, WINDOW_SAMPLES, 3)

    # Production: ax_max/min/mean computed from RAW per-window samples.
    # ax_max/min are signed AC deviations (rawMax - rawMean); ax_mean is rawMean.
    raw_window_mean = a_full_w.mean(axis=1)                      # (P,3) = win.axSum/8 etc.
    raw_window_max  = a_full_w.max(axis=1)
    raw_window_min  = a_full_w.min(axis=1)
    a_max  = raw_window_max - raw_window_mean                    # AC max
    a_min  = raw_window_min - raw_window_mean                    # AC min
    a_mean = raw_window_mean                                      # RAW mean (gravity)

    # amag computed from RAW per-sample |a| (gravity included), maxed/meaned over the window.
    amag = np.linalg.norm(a_full_w, axis=2)                      # (P,8) raw |a|, includes gravity
    gmag = np.linalg.norm(g_w, axis=2)
    amag_max  = amag.max(axis=1)
    amag_mean = amag.mean(axis=1)
    gmag_max  = gmag.max(axis=1)
    gmag_mean = gmag.mean(axis=1)

    accel_clip_w = (np.abs(a_full_w) >= ACCEL_CLIP - 1).any(axis=(1, 2))
    gyro_sat_w = (np.abs(g_w) >= GYRO_CLIP - 1).any(axis=(1, 2))

    grav_mean_w = grav_w.mean(axis=1)  # not transmitted in base packet, kept for debug

    # Pack
    a_max_p = axis_int8_pack(a_max)
    a_min_p = axis_int8_pack(a_min)
    a_mean_p = axis_int8_pack(a_mean)
    amag_max_p = sqrt_compand(amag_max, RANGE_AMAG)
    amag_mean_p = sqrt_compand(amag_mean, RANGE_AMAG)
    gmag_max_p = sqrt_compand(gmag_max, RANGE_GMAG)
    gmag_mean_p = sqrt_compand(gmag_mean, RANGE_GMAG)

    for i in range(n_pkt):
        seq = i & 0xFFFF
        raw[i, 0] = seq & 0xFF
        raw[i, 1] = (seq >> 8) & 0xFF
        raw[i, 2:5] = a_max_p[i].view(np.uint8)
        raw[i, 5:8] = a_min_p[i].view(np.uint8)
        raw[i, 8:11] = a_mean_p[i].view(np.uint8)
        raw[i, 11] = amag_max_p[i]
        raw[i, 12] = amag_mean_p[i]
        raw[i, 13] = gmag_max_p[i]
        raw[i, 14] = gmag_mean_p[i]
        flags = 0
        if accel_clip_w[i]:
            flags |= 0x01
        if gyro_sat_w[i]:
            flags |= 0x02
        raw[i, 15] = flags
        if with_mic:
            raw[i, 16] = 0  # audio disabled on this board

        decoded.append({
            "seq": seq,
            "t_s": i / PACKET_HZ,
            "a_max": a_max_p[i].tolist(),
            "a_min": a_min_p[i].tolist(),
            "a_mean": a_mean_p[i].tolist(),
            "amag_max_byte": int(amag_max_p[i]),
            "amag_mean_byte": int(amag_mean_p[i]),
            "gmag_max_byte": int(gmag_max_p[i]),
            "gmag_mean_byte": int(gmag_mean_p[i]),
            "accel_clip": bool(accel_clip_w[i]),
            "gyro_sat": bool(gyro_sat_w[i]),
            # bonus fields (NOT on the wire) for fidelity comparison
            "_grav_mean": grav_mean_w[i].tolist(),
        })

    return raw, decoded


def decode_stream(decoded: list[dict]) -> dict:
    """Receiver-side: reconstruct float-valued time-series from decoded packets."""
    n = len(decoded)
    a_max = np.array([axis_int8_unpack(np.array(p["a_max"], dtype=np.int8)) for p in decoded], dtype=np.float64)
    a_min = np.array([axis_int8_unpack(np.array(p["a_min"], dtype=np.int8)) for p in decoded], dtype=np.float64)
    a_mean = np.array([axis_int8_unpack(np.array(p["a_mean"], dtype=np.int8)) for p in decoded], dtype=np.float64)
    amag_max = sqrt_decompand(np.array([p["amag_max_byte"] for p in decoded], dtype=np.uint8), RANGE_AMAG)
    amag_mean = sqrt_decompand(np.array([p["amag_mean_byte"] for p in decoded], dtype=np.uint8), RANGE_AMAG)
    gmag_max = sqrt_decompand(np.array([p["gmag_max_byte"] for p in decoded], dtype=np.uint8), RANGE_GMAG)
    gmag_mean = sqrt_decompand(np.array([p["gmag_mean_byte"] for p in decoded], dtype=np.uint8), RANGE_GMAG)
    accel_clip = np.array([p["accel_clip"] for p in decoded], dtype=bool)
    gyro_sat = np.array([p["gyro_sat"] for p in decoded], dtype=bool)
    # NOT on the wire — used for grav_angle fidelity sanity check only:
    grav_mean = np.array([p["_grav_mean"] for p in decoded], dtype=np.float64)

    return {
        "a_max": a_max,
        "a_min": a_min,
        "a_mean": a_mean,
        "amag_max": amag_max,
        "amag_mean": amag_mean,
        "gmag_max": gmag_max,
        "gmag_mean": gmag_mean,
        "accel_clip": accel_clip,
        "gyro_sat": gyro_sat,
        "_grav_mean": grav_mean,
    }


def autocorr_peak(sig: np.ndarray, min_lag: int, max_lag: int) -> tuple[float, float]:
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
    return best_strength, PACKET_HZ / best_lag


def find_peaks_simple(sig: np.ndarray, height: float, min_dist: int) -> list[int]:
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


def recover_episode_features(stream: dict, ep_start_s: float, ep_end_s: float) -> dict:
    """Compute the same feature vector as extract_gesture_features.episode_features,
    but using only the wire stream (25 Hz packets)."""
    i0 = int(round(ep_start_s * PACKET_HZ))
    i1 = int(round(ep_end_s * PACKET_HZ))
    n_total = len(stream["amag_max"])
    i0 = max(0, min(i0, n_total - 1))
    i1 = max(i0 + 1, min(i1, n_total))

    amag_max = stream["amag_max"][i0:i1]
    amag_mean = stream["amag_mean"][i0:i1]
    gmag_max = stream["gmag_max"][i0:i1]
    gmag_mean = stream["gmag_mean"][i0:i1]
    a_max = stream["a_max"][i0:i1]
    a_min = stream["a_min"][i0:i1]
    a_mean = stream["a_mean"][i0:i1]
    accel_clip = stream["accel_clip"][i0:i1]
    gyro_sat = stream["gyro_sat"][i0:i1]

    n = max(1, i1 - i0)
    duration_s = n / PACKET_HZ

    # peak |a| = max over all windows of amag_max
    peak_amag = float(amag_max.max()) if n else 0.0
    # rms |a| approximated by RMS of per-window amag_mean (loses sub-window variance)
    rms_amag = float(math.sqrt(float(np.mean(amag_mean * amag_mean)))) if n else 0.0
    # integrated |a| = sum amag_mean / PACKET_HZ
    integrated_amag = float(amag_mean.sum() / PACKET_HZ)

    peak_gmag = float(gmag_max.max()) if n else 0.0
    rms_gmag = float(math.sqrt(float(np.mean(gmag_mean * gmag_mean)))) if n else 0.0

    crest = peak_amag / rms_amag if rms_amag > 0 else 0.0

    # Clipping fractions: fraction of windows with the flag set
    accel_clip_frac = float(accel_clip.mean())
    gyro_sat_frac = float(gyro_sat.mean())

    # Peak detection on amag_max (per-packet samples)
    rms_for_peaks = max(rms_amag, 1.0)
    peak_idx = find_peaks_simple(amag_max, height=2.5 * rms_for_peaks, min_dist=1)
    n_peaks = len(peak_idx)
    stroke_rate_hz = n_peaks / duration_s if duration_s > 0 else 0.0

    # Autocorrelation on amag_max (25 Hz signal)
    min_lag = max(1, int(0.05 * PACKET_HZ))    # 50 ms ~ 1 sample at 25 Hz; bump to >=2
    min_lag = max(min_lag, 2)
    max_lag = min(int(0.5 * PACKET_HZ), max(min_lag + 1, n // 3))
    if max_lag > min_lag:
        ac_str, ac_freq = autocorr_peak(amag_max, min_lag, max_lag)
    else:
        ac_str, ac_freq = 0.0, float("nan")

    # Axis dominance from per-axis mean (energy proxy)
    a_centered = a_mean - a_mean.mean(axis=0, keepdims=True)
    energy_xyz = np.sum(a_centered * a_centered, axis=0)
    total = float(energy_xyz.sum())
    axis_share = (energy_xyz / total).tolist() if total > 0 else [0.0, 0.0, 0.0]
    dom_accel_axis = ["X", "Y", "Z"][int(np.argmax(energy_xyz))]

    # Gravity rotation: best we can do without the dedicated grav bytes is to
    # take the per-axis MEAN of `a_mean` early vs late. Since a_mean is the
    # ac-coupled accel mean (gravity already stripped on the C3), this only
    # captures gravity drift indirectly through whatever the C3's rolling-mean
    # baseline missed. NOT a faithful recovery — use the bonus _grav_mean field
    # (not on the wire) for the fidelity comparison.
    edge = max(1, int(0.1 * PACKET_HZ))
    grav_mean = stream["_grav_mean"][i0:i1]
    g_pre = grav_mean[:edge].mean(axis=0)
    g_post = grav_mean[-edge:].mean(axis=0)
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
        "accel_clip_frac": round(accel_clip_frac, 3),
        "gyro_sat_frac": round(gyro_sat_frac, 3),
        "grav_angle_deg": round(grav_angle, 1),
    }


def fidelity_table(raw_rows: list[dict], rec_rows: list[dict]) -> str:
    cols_pct = ["peak_amag", "rms_amag", "integrated_amag", "peak_gmag",
                "rms_gmag", "crest", "stroke_rate_hz", "ac_strength"]
    cols_abs = ["grav_angle_deg", "n_peaks", "accel_clip_frac", "gyro_sat_frac"]
    lines = ["| capture | ep | feature | raw | recovered | err |", "|---|---|---|---|---|---|"]
    for r, rc in zip(raw_rows, rec_rows):
        for c in cols_pct:
            rv = r.get(c) or 0.0
            xv = rc.get(c) or 0.0
            err = abs(xv - rv) / abs(rv) * 100.0 if rv else (0.0 if xv == 0 else float("inf"))
            lines.append(f"| {r['capture']} | {r['name']} | {c} | {rv:g} | {xv:g} | {err:.1f}% |")
        for c in cols_abs:
            rv = r.get(c) or 0.0
            xv = rc.get(c) or 0.0
            lines.append(f"| {r['capture']} | {r['name']} | {c} | {rv:g} | {xv:g} | Δ={xv - rv:g} |")
    return "\n".join(lines)


def per_feature_summary(raw_rows: list[dict], rec_rows: list[dict]) -> str:
    """Median-abs-percent error per feature across all 13 episodes."""
    cols_pct = ["peak_amag", "rms_amag", "integrated_amag", "peak_gmag",
                "rms_gmag", "crest", "stroke_rate_hz", "ac_strength"]
    cols_abs = ["grav_angle_deg", "n_peaks", "accel_clip_frac", "gyro_sat_frac"]
    lines = ["| feature | median abs err | max abs err | unit |", "|---|---|---|---|"]
    for c in cols_pct:
        errs = []
        for r, rc in zip(raw_rows, rec_rows):
            rv = r.get(c) or 0.0
            xv = rc.get(c) or 0.0
            if rv != 0:
                errs.append(abs(xv - rv) / abs(rv) * 100.0)
        if errs:
            lines.append(f"| {c} | {np.median(errs):.1f}% | {max(errs):.1f}% | percent |")
    for c in cols_abs:
        diffs = []
        for r, rc in zip(raw_rows, rec_rows):
            rv = r.get(c) or 0.0
            xv = rc.get(c) or 0.0
            diffs.append(abs(xv - rv))
        if diffs:
            unit = "deg" if "angle" in c else ("count" if c == "n_peaks" else "fraction")
            lines.append(f"| {c} | {np.median(diffs):.2f} | {max(diffs):.2f} | {unit} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="Include 17th rawRms byte")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Users/sethdrew/Documents/projects/led/festicorn/gyro-sense/data/recordings/wire_packets"))
    ap.add_argument("--out-features", type=Path,
                    default=Path("/Users/sethdrew/Documents/projects/led/festicorn/gyro-sense/data/recordings/gesture_features_recovered.json"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    all_raw: list[dict] = []
    all_rec: list[dict] = []

    for csv_path, eps_path in captures:
        accel, gyro = efx.load_csv(csv_path)
        gyro_bias = efx.estimate_gyro_bias(gyro, calm_seconds=5.0)
        gyro_corr = (gyro - gyro_bias).astype(np.float64)
        accel_f = accel.astype(np.float64)
        accel_ac, grav = efx.remove_gravity(accel_f, win_s=1.0)

        raw_bytes, decoded = encode_packets(
            accel.astype(np.int32), gyro_corr, accel_ac, grav, with_mic=args.mic,
        )
        bin_path = args.out_dir / f"{csv_path.stem}.bin"
        json_path = args.out_dir / f"{csv_path.stem}.json"
        bin_path.write_bytes(raw_bytes.tobytes())
        json_path.write_text(json.dumps(decoded, indent=2))

        # Decode (a no-op since we already have decoded structure, but matches what
        # a real receiver would do)
        stream = decode_stream(decoded)

        # Episodes
        eps = json.loads(eps_path.read_text())
        for ep in eps:
            i0 = int(round(ep["start_s"] * FS))
            i1 = int(round(ep["end_s"] * FS))
            i0 = max(0, min(i0, len(accel_f) - 1))
            i1 = max(i0 + 1, min(i1, len(accel_f)))
            raw_feat = efx.episode_features(accel, gyro_corr, grav, accel_ac, i0, i1)
            raw_feat["start_s"] = ep["start_s"]
            raw_feat["end_s"] = ep["end_s"]
            raw_feat["name"] = str(ep.get("name", ep.get("ep", "?")))
            raw_feat["capture"] = csv_path.stem
            all_raw.append(raw_feat)

            rec_feat = recover_episode_features(stream, ep["start_s"], ep["end_s"])
            rec_feat["start_s"] = ep["start_s"]
            rec_feat["end_s"] = ep["end_s"]
            rec_feat["name"] = raw_feat["name"]
            rec_feat["capture"] = raw_feat["capture"]
            all_rec.append(rec_feat)

        print(f"[{csv_path.name}] {len(decoded)} packets -> {bin_path.name} ({raw_bytes.nbytes} B)")

    args.out_features.write_text(json.dumps(all_rec, indent=2))

    print("\n## Fidelity loss (median abs err across 13 episodes)")
    print(per_feature_summary(all_raw, all_rec))
    print("\n## Per-episode comparison")
    print(fidelity_table(all_raw, all_rec))
    print(f"\n[Recovered features -> {args.out_features}]")


if __name__ == "__main__":
    main()
