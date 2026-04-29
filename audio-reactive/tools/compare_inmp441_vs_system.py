#!/usr/bin/env python3
"""Compare INMP441 acoustic capture vs system loopback (BlackHole).

Loads two WAVs, aligns via envelope cross-correlation, then computes per-band
PSD ratios (system - inmp441 in dB) plus broadband stats. Saves spectrograms
to the same directory.

Usage:
    python compare_inmp441_vs_system.py inmp441_<stamp>.wav system_<stamp>.wav
    python compare_inmp441_vs_system.py             # auto-pick newest pair
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import correlate, spectrogram, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "library", "test-vectors", "inmp441-validation",
)

BANDS = [
    ("sub-bass",  20,    80),
    ("bass",      80,    250),
    ("mids",      250,   2000),
    ("high-mid",  2000,  6000),
    ("treble",    6000,  8000),
]

LEDGER_TARGETS_DB = {
    "sub-bass": -15.0,
    "bass":     -28.0,
    "mids":     -21.0,
    "treble":   0.0,        # "comparable" → ~0 dB
}


@dataclass
class Wav:
    samples: np.ndarray   # float32 in [-1, 1]
    rate: int
    path: str


def load_wav(path: str) -> Wav:
    data, rate = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return Wav(samples=data.astype(np.float32), rate=rate, path=path)


def envelope(x: np.ndarray, win: int) -> np.ndarray:
    """Box-RMS envelope (downsamples by `win`)."""
    n = (x.size // win) * win
    blocks = x[:n].reshape(-1, win)
    return np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)


def find_lag(env_ref: np.ndarray, env_test: np.ndarray, max_lag: int) -> int:
    """Return integer lag (in env samples) of test relative to ref."""
    ref = (env_ref - env_ref.mean()) / (env_ref.std() + 1e-12)
    tst = (env_test - env_test.mean()) / (env_test.std() + 1e-12)
    n = min(ref.size, tst.size)
    ref, tst = ref[:n], tst[:n]
    xc = correlate(tst, ref, mode="full")
    center = n - 1
    lo = max(0, center - max_lag)
    hi = min(xc.size, center + max_lag + 1)
    window = xc[lo:hi]
    peak = np.argmax(window) + lo
    return peak - center


def band_power_db(x: np.ndarray, rate: int, lo: float, hi: float,
                  freqs: np.ndarray, psd: np.ndarray) -> float:
    """Integrate PSD over [lo, hi] and return 10*log10."""
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return float("nan")
    power = np.trapezoid(psd[mask], freqs[mask])
    return 10.0 * np.log10(power + 1e-20)


def plot_spectrogram(wav: Wav, out_path: str, title: str):
    f, t, S = spectrogram(wav.samples, fs=wav.rate, nperseg=2048,
                          noverlap=1024, scaling="density")
    S_db = 10 * np.log10(S + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(t, f, S_db, shading="auto",
                       vmin=S_db.max() - 80, vmax=S_db.max())
    ax.set_yscale("log")
    ax.set_ylim(20, 8000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inmp441", nargs="?", default=None)
    ap.add_argument("system", nargs="?", default=None)
    args = ap.parse_args()

    if not args.inmp441 or not args.system:
        inmps = sorted(glob.glob(os.path.join(VAL_DIR, "**", "inmp441_*.wav"),
                                 recursive=True))
        syss  = sorted(glob.glob(os.path.join(VAL_DIR, "**", "system_*.wav"),
                                 recursive=True))
        if not inmps or not syss:
            print("No capture pair found under", VAL_DIR, file=sys.stderr)
            sys.exit(1)
        args.inmp441 = inmps[-1]
        args.system  = syss[-1]
        print(f"auto-pick: {os.path.relpath(args.inmp441, VAL_DIR)} + "
              f"{os.path.relpath(args.system, VAL_DIR)}")

    inmp = load_wav(args.inmp441)
    sysw = load_wav(args.system)
    print(f"INMP441: {inmp.samples.size} samples @ {inmp.rate} Hz "
          f"({inmp.samples.size / inmp.rate:.2f}s)")
    print(f"System:  {sysw.samples.size} samples @ {sysw.rate} Hz "
          f"({sysw.samples.size / sysw.rate:.2f}s)")

    if inmp.rate != sysw.rate:
        print(f"sample-rate mismatch: {inmp.rate} vs {sysw.rate}", file=sys.stderr)
        sys.exit(1)
    rate = inmp.rate

    # ── Align via envelope cross-correlation ───────────────────────
    env_win = rate // 100   # 10 ms envelope
    e_inmp = envelope(inmp.samples, env_win)
    e_sys  = envelope(sysw.samples, env_win)
    n_env = min(e_inmp.size, e_sys.size)
    e_inmp, e_sys = e_inmp[:n_env], e_sys[:n_env]
    max_lag_env = int(0.5 * rate / env_win)   # ±0.5s tolerance
    lag = find_lag(e_sys, e_inmp, max_lag_env)
    lag_samples = lag * env_win
    lag_ms = 1000.0 * lag_samples / rate
    print(f"alignment lag (inmp441 vs system): {lag} env-blocks "
          f"= {lag_samples} samples = {lag_ms:.1f} ms")

    # Apply lag to INMP441 (positive lag → INMP441 ahead → trim its head)
    if lag > 0:
        inmp_aligned = inmp.samples[lag_samples:]
        sys_aligned  = sysw.samples[:inmp_aligned.size]
    else:
        sys_aligned  = sysw.samples[-lag_samples:]
        inmp_aligned = inmp.samples[:sys_aligned.size]
    n = min(inmp_aligned.size, sys_aligned.size)
    inmp_aligned = inmp_aligned[:n]
    sys_aligned  = sys_aligned[:n]
    print(f"aligned length: {n} samples ({n / rate:.2f}s)")

    # Re-do envelopes on aligned signals for correlation
    e_inmp2 = envelope(inmp_aligned, env_win)
    e_sys2  = envelope(sys_aligned, env_win)
    m = min(e_inmp2.size, e_sys2.size)
    pearson = float(np.corrcoef(e_inmp2[:m], e_sys2[:m])[0, 1])
    print(f"envelope Pearson correlation: {pearson:+.3f}")

    # ── Per-band PSDs (Welch) ──────────────────────────────────────
    f_i, p_i = welch(inmp_aligned, fs=rate, nperseg=4096, noverlap=2048)
    f_s, p_s = welch(sys_aligned,  fs=rate, nperseg=4096, noverlap=2048)

    print("\nper-band power (dB) and delta (system - inmp441):")
    print(f"  {'band':10}  {'lo-hi (Hz)':14}  {'sys dB':>9}  "
          f"{'inmp dB':>9}  {'Δ dB':>8}  {'ledger':>8}  {'match':>8}")
    rows = []
    for name, lo, hi in BANDS:
        sys_db  = band_power_db(sys_aligned,  rate, lo, hi, f_s, p_s)
        inmp_db = band_power_db(inmp_aligned, rate, lo, hi, f_i, p_i)
        delta = sys_db - inmp_db   # positive = system louder than inmp
        # Ledger states inmp loss as negative (inmp - system). Convert: -delta.
        inmp_minus_sys = -delta
        target = LEDGER_TARGETS_DB.get(name)
        if target is None:
            tgt_str, match_str = "—", "—"
        else:
            diff = inmp_minus_sys - target
            tgt_str = f"{target:+.1f}"
            match_str = "ok" if abs(diff) <= 5.0 else f"{diff:+.1f}"
        print(f"  {name:10}  {lo:5.0f}-{hi:5.0f}    "
              f"{sys_db:+8.2f}  {inmp_db:+8.2f}  {inmp_minus_sys:+7.2f}  "
              f"{tgt_str:>8}  {match_str:>8}")
        rows.append((name, lo, hi, sys_db, inmp_db, inmp_minus_sys, target))

    # Broadband loudness comparison
    sys_rms = float(np.sqrt(np.mean(sys_aligned ** 2)))
    inmp_rms = float(np.sqrt(np.mean(inmp_aligned ** 2)))
    sys_rms_db  = 20 * np.log10(sys_rms + 1e-12)
    inmp_rms_db = 20 * np.log10(inmp_rms + 1e-12)
    print(f"\nbroadband RMS:  system={sys_rms_db:+.2f} dBFS  "
          f"inmp441={inmp_rms_db:+.2f} dBFS  "
          f"delta(inmp - sys)={inmp_rms_db - sys_rms_db:+.2f} dB")

    # ── Spectrograms ───────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.inmp441))[0].split("_", 1)[1]
    out_dir = os.path.dirname(os.path.abspath(args.inmp441))
    inmp_png = os.path.join(out_dir, f"spec_inmp441_{base}.png")
    sys_png  = os.path.join(out_dir, f"spec_system_{base}.png")
    plot_spectrogram(Wav(inmp_aligned, rate, args.inmp441), inmp_png,
                     "INMP441 (acoustic, aligned)")
    plot_spectrogram(Wav(sys_aligned, rate, args.system),  sys_png,
                     "System loopback (BlackHole)")
    print(f"\nspectrograms:\n  {inmp_png}\n  {sys_png}")

    # ── Combined PSD plot ──────────────────────────────────────────
    psd_png = os.path.join(out_dir, f"psd_compare_{base}.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(f_s, 10 * np.log10(p_s + 1e-20), label="system (BlackHole)",
                color="tab:blue")
    ax.semilogx(f_i, 10 * np.log10(p_i + 1e-20), label="INMP441 (acoustic)",
                color="tab:orange")
    ax.set_xlim(20, 8000)
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Power spectral density: INMP441 vs system loopback")
    ax.grid(True, which="both", alpha=0.3)
    for name, lo, hi in BANDS:
        ax.axvspan(lo, hi, alpha=0.05, color="black")
    ax.legend()
    fig.tight_layout()
    fig.savefig(psd_png, dpi=110)
    plt.close(fig)
    print(f"  {psd_png}")


if __name__ == "__main__":
    main()
