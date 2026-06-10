#!/usr/bin/env python3
"""replay.py — load a recorded RMS-envelope fixture and replay it frame-by-frame.

Lets you build/tune an audio-reactive effect offline against real telemetry,
no mic or hardware needed. Two fixtures ship alongside this file:

  audio_envelope_profile.csv        coarse (~1 Hz), rich: music + speech, 2k..140k
  audio_envelope_dense_sentence.csv dense (~25 Hz), thin: one sentence + silence

Both carry RMS in the 24-bit (rawI2S>>8) domain, full-scale RMS_FS = 200000.

Usage:
  python replay.py audio_envelope_dense_sentence.csv
  # or import and feed your own detector:
  for t, mean, mx in load("audio_envelope_profile.csv"): ...
"""
import csv, sys

RMS_FS = 200000.0


def load(path):
    """Yield (t_seconds, rms_mean, rms_max). Skips '#' comment lines."""
    with open(path) as f:
        rows = [r for r in f if not r.lstrip().startswith("#")]
    rdr = csv.reader(rows)
    header = next(rdr)
    tcol = 0  # first column is time (t_s or t_ms)
    t_is_ms = header[0].endswith("ms")
    for row in rdr:
        if not row:
            continue
        t = float(row[0]) / (1000.0 if t_is_ms else 1.0)
        yield t, float(row[1]), float(row[2])


def encode_byte(rms, fs=RMS_FS):
    """Re-companding a decoded RMS back to its 0..255 transport byte."""
    r = max(0.0, min(1.0, rms / fs))
    return round((r ** 0.5) * 255.0)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "audio_envelope_dense_sentence.csv"
    samples = list(load(path))
    means = [m for _, m, _ in samples]
    maxs = [x for _, _, x in samples]
    dur = samples[-1][0] - samples[0][0] if samples else 0.0
    print(f"{path}: {len(samples)} samples, {dur:.1f}s span")
    print(f"  rms_mean: min={min(means):.0f} max={max(means):.0f}")
    print(f"  rms_max : min={min(maxs):.0f} max={max(maxs):.0f}")
