#!/usr/bin/env python3
"""Reciprocity + linearity analysis of an rf-bench matrix run.

RSSI_ij = T_i - PL_ij + R_j, and path loss is reciprocal (PL_ij == PL_ji), so
    RSSI_ij - RSSI_ji = (T_i - R_i) - (T_j - R_j) = D_i - D_j
which isolates each board's transmit-vs-receive balance with no knowledge of
geometry. Antenna gain is reciprocal too, so it cancels out of D entirely: D is
a property of the silicon's PA-vs-LNA balance, not of the antenna.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    d = json.loads(Path(path).read_text())
    labels = [k for k in d["boards"]]
    chips = {k: v["chip"] for k, v in d["boards"].items()}
    # per (level, tx, rx) collect RSSI samples across the repeated bursts
    rssi = defaultdict(list)
    ack = defaultdict(list)
    lat = defaultdict(list)
    readback = {}
    for r in d["results"]:
        q = r["txq_req"]
        readback[(q, r["tx"])] = r["txq_readback"]
        b = r["burst"]
        if b["n"]:
            ack[(q, r["tx"], r["dst"])].append(100.0 * b["ack"] / b["n"])
            lat[(q, r["tx"], r["dst"])].append((b["ack_lat_mean_us"], b["ack_lat_sd_us"]))
        for rx, st in r["rx"].items():
            if st.get("n"):
                rssi[(q, r["tx"], rx)].append(st["rssi_mean"])
    levels = sorted({k[0] for k in rssi})
    return d, labels, chips, levels, rssi, ack, lat, readback


def med(xs):
    return float(np.median(xs)) if xs else None


def main():
    path = sys.argv[1]
    d, labels, chips, levels, rssi, ack, lat, readback = load(path)
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}

    print(f"boards: " + ", ".join(f"{l}({chips[l]})" for l in labels))
    print(f"levels: {levels}\n")

    print("=" * 78)
    print("1. TX POWER READBACK (quarter-dBm actually accepted by the driver)")
    print("=" * 78)
    print(f"{'req':>5s}" + "".join(f"{l:>9s}" for l in labels))
    for q in levels:
        print(f"{q:5d}" + "".join(f"{readback.get((q, l), float('nan')):9.0f}" for l in labels))

    print()
    print("=" * 78)
    print("2. RECIPROCITY -> per-board TX-minus-RX balance D (dB, gauge: mean=0)")
    print("   antenna gain cancels; this is PA-vs-LNA silicon balance")
    print("=" * 78)
    print(f"{'req':>5s}" + "".join(f"{l:>9s}" for l in labels) + f"{'resid_rms':>11s}")
    Dtab = {}
    for q in levels:
        rows, rhs = [], []
        for i in labels:
            for j in labels:
                if idx[i] >= idx[j]:
                    continue
                a, b = med(rssi.get((q, i, j), [])), med(rssi.get((q, j, i), []))
                if a is None or b is None:
                    continue
                r = np.zeros(n)
                r[idx[i]], r[idx[j]] = 1.0, -1.0
                rows.append(r)
                rhs.append(a - b)
        if not rows:
            continue
        rows.append(np.ones(n))  # gauge fix: mean(D) = 0
        rhs.append(0.0)
        A, y = np.array(rows), np.array(rhs)
        D, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = A[:-1] @ D - y[:-1]
        rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 0.0
        Dtab[q] = D
        print(f"{q:5d}" + "".join(f"{D[idx[l]]:9.2f}" for l in labels) + f"{rms:11.2f}")

    print()
    print("=" * 78)
    print("3. LINEARITY: slope of RSSI vs commanded dBm, per link")
    print("   slope 1.0 = radiated power tracks the command exactly")
    print("=" * 78)
    print(f"{'link':>14s} {'slope':>7s} {'r2':>6s} {'span_cmd':>9s} {'span_rssi':>10s}")
    for tx in labels:
        for rx in labels:
            if tx == rx:
                continue
            xs, ys = [], []
            for q in levels:
                v = med(rssi.get((q, tx, rx), []))
                rb = readback.get((q, tx))
                if v is None or rb is None:
                    continue
                xs.append(rb / 4.0)
                ys.append(v)
            if len(xs) < 3:
                continue
            xs, ys = np.array(xs), np.array(ys)
            if xs.max() - xs.min() < 1:
                continue
            sl, ic = np.polyfit(xs, ys, 1)
            pred = sl * xs + ic
            ss = 1 - np.sum((ys - pred) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-9)
            print(f"{tx + '->' + rx:>14s} {sl:7.2f} {ss:6.2f} "
                  f"{xs.max() - xs.min():9.2f} {ys.max() - ys.min():10.2f}")

    print()
    print("=" * 78)
    print("4. RSSI vs COMMANDED POWER (median dBm per link) - cliff detection")
    print("=" * 78)
    links = [(tx, rx) for tx in labels for rx in labels if tx != rx]
    print(f"{'req':>5s}" + "".join(f"{t + '>' + r:>12s}" for t, r in links))
    for q in levels:
        row = f"{q:5d}"
        for tx, rx in links:
            v = med(rssi.get((q, tx, rx), []))
            row += f"{v:12.1f}" if v is not None else f"{'-':>12s}"
        print(row)

    print()
    print("=" * 78)
    print("5. UNICAST ACK DELIVERY % and MAC ACK LATENCY (us)")
    print("=" * 78)
    print(f"{'req':>5s}" + "".join(f"{t + '>' + r:>12s}" for t, r in links))
    for q in levels:
        row = f"{q:5d}"
        for tx, rx in links:
            v = med(ack.get((q, tx, rx), []))
            row += f"{v:12.1f}" if v is not None else f"{'-':>12s}"
        print(row)

    print("\nMAC ACK latency, median across all levels (us mean / us sd):")
    for tx in labels:
        parts = []
        for rx in labels:
            if tx == rx:
                continue
            allv = [x for q in levels for x in lat.get((q, tx, rx), [])]
            if allv:
                parts.append(f"{rx}={med([a for a, _ in allv]):.0f}/{med([s for _, s in allv]):.0f}")
        print(f"  {tx:6s} " + "  ".join(parts))

    print()
    print("=" * 78)
    print("6. CHIP-FAMILY SUMMARY of D (TX-minus-RX balance)")
    print("=" * 78)
    for q in levels:
        if q not in Dtab:
            continue
        c3 = [Dtab[q][idx[l]] for l in labels if chips[l] == "esp32c3"]
        e32 = [Dtab[q][idx[l]] for l in labels if chips[l] != "esp32c3"]
        if c3 and e32:
            print(f"  req={q:3d}  C3 mean={np.mean(c3):6.2f} (spread {np.ptp(c3):4.2f})   "
                  f"ESP32 mean={np.mean(e32):6.2f} (spread {np.ptp(e32):4.2f})   "
                  f"C3-ESP32 = {np.mean(c3) - np.mean(e32):+6.2f} dB")


if __name__ == "__main__":
    main()
