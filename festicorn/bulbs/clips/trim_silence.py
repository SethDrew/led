#!/usr/bin/env python3
"""Trim silence/noise from WAV files based on RMS threshold."""

import wave
import numpy as np
import struct
import os

CLIPS_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = ["hmUmmmmmm.wav", "humm.wav", "singing.wav", "speaking.wav"]
WINDOW_SIZE = 1024
THRESHOLD = 0.003
PADDING_SEC = 0.2


def read_wav(path):
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        max_val = 2147483648.0
    elif sampwidth == 1:
        dtype = np.uint8
        max_val = 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if sampwidth == 1:
        samples = samples - 128.0
    samples /= max_val

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, framerate, n_channels, sampwidth


def write_wav(path, samples, framerate, n_channels, sampwidth):
    if sampwidth == 2:
        max_val = 32767
        dtype = np.int16
    elif sampwidth == 4:
        max_val = 2147483647
        dtype = np.int32
    elif sampwidth == 1:
        max_val = 127
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    int_samples = np.clip(samples * (max_val + 1), -max_val - 1, max_val).astype(dtype)
    if sampwidth == 1:
        int_samples = (int_samples + 128).astype(np.uint8)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)  # we mono-mixed on read
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(int_samples.tobytes())


def trim_clip(filepath):
    samples, framerate, n_channels, sampwidth = read_wav(filepath)
    original_duration = len(samples) / framerate

    # Compute RMS per window
    n_windows = len(samples) // WINDOW_SIZE
    rms_values = np.zeros(n_windows)
    for i in range(n_windows):
        window = samples[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        rms_values[i] = np.sqrt(np.mean(window ** 2))

    # Find first and last windows above threshold
    above = np.where(rms_values > THRESHOLD)[0]
    if len(above) == 0:
        print(f"  WARNING: No audio above threshold, skipping")
        return

    first_window = above[0]
    last_window = above[-1]

    # Convert to sample indices with padding
    padding_samples = int(PADDING_SEC * framerate)
    start_sample = max(0, first_window * WINDOW_SIZE - padding_samples)
    end_sample = min(len(samples), (last_window + 1) * WINDOW_SIZE + padding_samples)

    trimmed = samples[start_sample:end_sample]
    trimmed_duration = len(trimmed) / framerate

    write_wav(filepath, trimmed, framerate, 1, sampwidth)

    print(f"  Before: {original_duration:.2f}s  After: {trimmed_duration:.2f}s  "
          f"(removed {original_duration - trimmed_duration:.2f}s)")


if __name__ == "__main__":
    for fname in FILES:
        path = os.path.join(CLIPS_DIR, fname)
        print(f"Processing {fname}...")
        trim_clip(path)
    print("\nDone!")
