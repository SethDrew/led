#!/usr/bin/env python3
"""Dual capture: INMP441 (over USB serial) + system audio (BlackHole loopback).

Threads share a started/stop Event so both streams begin within ~ms of each
other. The serial thread reads raw int16 LE @ 16 kHz from /dev/cu.usbmodem*
(produced by the raw_audio_sender firmware). The BlackHole thread captures
2ch float32 from a sounddevice input and downmixes to mono.

Output: two WAVs in audio-reactive/research/inmp441-validation/, both at 16 kHz
mono — comparable in length so per-band FFTs line up.

Usage:
    python dual_capture.py [duration_seconds] [--port /dev/cu.usbmodem114301]
                           [--blackhole-index 3] [--baud 460800]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import serial

SAMPLE_RATE = 16000
SERIAL_PORT_DEFAULT = "/dev/cu.usbmodem114301"
SERIAL_BAUD_DEFAULT = 460800
BLACKHOLE_INDEX_DEFAULT = 3   # `BlackHole 2ch` from sd.query_devices()
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "library", "test-vectors", "inmp441-validation",
)


def serial_capture(port: str, baud: int, start_evt: threading.Event,
                   stop_evt: threading.Event, out_path: str,
                   stats: dict):
    """Read raw int16 LE from serial, write mono WAV at SAMPLE_RATE."""
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except (serial.SerialException, OSError) as e:
        stats["error"] = f"serial open failed: {e}"
        start_evt.set()
        return

    # Drain any old buffer junk before the synchronized start.
    ser.reset_input_buffer()
    start_evt.set()
    start_evt.wait()  # everyone stand together

    chunks: list[bytes] = []
    bytes_total = 0
    t_start = time.monotonic()
    try:
        while not stop_evt.is_set():
            data = ser.read(4096)
            if data:
                chunks.append(data)
                bytes_total += len(data)
    finally:
        try:
            ser.close()
        except Exception:
            pass

    elapsed = time.monotonic() - t_start
    raw = b"".join(chunks)
    # Drop a trailing odd byte if present.
    if len(raw) % 2 == 1:
        raw = raw[:-1]
    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size > 0:
        sf.write(out_path, samples, SAMPLE_RATE, subtype="PCM_16")
    stats["bytes"] = bytes_total
    stats["samples"] = int(samples.size)
    stats["elapsed"] = elapsed
    stats["peak"] = int(np.max(np.abs(samples))) if samples.size else 0


def blackhole_capture(device_index: int, start_evt: threading.Event,
                      stop_evt: threading.Event, out_path: str,
                      stats: dict):
    """Capture system audio via BlackHole, downmix to mono, write 16 kHz WAV."""
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    info = sd.query_devices(device_index)
    in_channels = min(2, info["max_input_channels"])
    in_rate = int(info["default_samplerate"])

    try:
        stream = sd.InputStream(
            device=device_index,
            channels=in_channels,
            samplerate=in_rate,
            blocksize=1024,
            dtype="float32",
            callback=callback,
        )
    except Exception as e:
        stats["error"] = f"blackhole open failed: {e}"
        start_evt.set()
        return

    stream.start()
    start_evt.set()
    start_evt.wait()

    t_start = time.monotonic()
    try:
        while not stop_evt.is_set():
            time.sleep(0.05)
    finally:
        stream.stop()
        stream.close()

    elapsed = time.monotonic() - t_start
    if not chunks:
        stats["error"] = "no audio captured from BlackHole"
        return

    audio = np.concatenate(chunks, axis=0)
    if audio.ndim > 1:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    if in_rate != SAMPLE_RATE:
        # Linear interpolation resample — fine for spectral comparison.
        n_in = mono.shape[0]
        n_out = int(round(n_in * SAMPLE_RATE / in_rate))
        x_in = np.linspace(0.0, 1.0, n_in, endpoint=False)
        x_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
        mono = np.interp(x_out, x_in, mono).astype(np.float32)

    sf.write(out_path, mono, SAMPLE_RATE, subtype="PCM_16")
    stats["samples"] = int(mono.size)
    stats["elapsed"] = elapsed
    stats["peak"] = float(np.max(np.abs(mono))) if mono.size else 0.0
    stats["in_rate"] = in_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("duration", nargs="?", type=float, default=60.0,
                    help="capture duration in seconds (default 60)")
    ap.add_argument("--port", default=SERIAL_PORT_DEFAULT)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD_DEFAULT)
    ap.add_argument("--blackhole-index", type=int,
                    default=BLACKHOLE_INDEX_DEFAULT)
    ap.add_argument("--prefix", default=None,
                    help="filename prefix (default: timestamp)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = args.prefix or datetime.now().strftime("%Y%m%d_%H%M%S")
    inmp_path = os.path.abspath(os.path.join(OUT_DIR, f"inmp441_{stamp}.wav"))
    sys_path  = os.path.abspath(os.path.join(OUT_DIR, f"system_{stamp}.wav"))

    print(f"out dir: {os.path.abspath(OUT_DIR)}")
    print(f"  inmp441 -> {os.path.basename(inmp_path)}")
    print(f"  system  -> {os.path.basename(sys_path)}")

    start_evt = threading.Event()
    serial_ready = threading.Event()
    blackhole_ready = threading.Event()
    stop_evt = threading.Event()
    serial_stats: dict = {}
    blackhole_stats: dict = {}

    # Each thread sets its own ready event, then waits on the shared start.
    def serial_runner():
        ser_local_start = threading.Event()
        # Re-use serial_capture with serial_ready as the local "started" gate.
        # We chain by replacing start_evt below.
        # Implemented inline so we can share start_evt cleanly:
        try:
            ser = serial.Serial(args.port, args.baud, timeout=0.1)
        except Exception as e:
            serial_stats["error"] = f"serial open failed: {e}"
            serial_ready.set()
            return
        ser.reset_input_buffer()
        serial_ready.set()
        start_evt.wait()
        chunks = []
        bytes_total = 0
        t0 = time.monotonic()
        try:
            while not stop_evt.is_set():
                data = ser.read(4096)
                if data:
                    chunks.append(data)
                    bytes_total += len(data)
        finally:
            try:
                ser.close()
            except Exception:
                pass
        raw = b"".join(chunks)
        if len(raw) % 2 == 1:
            raw = raw[:-1]
        samples = np.frombuffer(raw, dtype="<i2")
        if samples.size > 0:
            sf.write(inmp_path, samples, SAMPLE_RATE, subtype="PCM_16")
        serial_stats["bytes"] = bytes_total
        serial_stats["samples"] = int(samples.size)
        serial_stats["elapsed"] = time.monotonic() - t0
        serial_stats["peak"] = int(np.max(np.abs(samples))) if samples.size else 0

    def blackhole_runner():
        chunks = []

        def cb(indata, frames, time_info, status):
            chunks.append(indata.copy())

        try:
            info = sd.query_devices(args.blackhole_index)
            in_channels = min(2, info["max_input_channels"])
            in_rate = int(info["default_samplerate"])
            stream = sd.InputStream(
                device=args.blackhole_index,
                channels=in_channels,
                samplerate=in_rate,
                blocksize=1024,
                dtype="float32",
                callback=cb,
            )
            stream.start()
        except Exception as e:
            blackhole_stats["error"] = f"blackhole open failed: {e}"
            blackhole_ready.set()
            return

        blackhole_ready.set()
        start_evt.wait()
        t0 = time.monotonic()
        try:
            while not stop_evt.is_set():
                time.sleep(0.05)
        finally:
            stream.stop()
            stream.close()

        if not chunks:
            blackhole_stats["error"] = "no audio captured"
            return
        audio = np.concatenate(chunks, axis=0)
        mono = audio.mean(axis=1) if audio.ndim > 1 else audio
        if in_rate != SAMPLE_RATE:
            n_in = mono.shape[0]
            n_out = int(round(n_in * SAMPLE_RATE / in_rate))
            x_in = np.linspace(0.0, 1.0, n_in, endpoint=False)
            x_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
            mono = np.interp(x_out, x_in, mono).astype(np.float32)
        sf.write(sys_path, mono, SAMPLE_RATE, subtype="PCM_16")
        blackhole_stats["samples"] = int(mono.size)
        blackhole_stats["elapsed"] = time.monotonic() - t0
        blackhole_stats["peak"] = float(np.max(np.abs(mono)))
        blackhole_stats["in_rate"] = in_rate

    t_ser = threading.Thread(target=serial_runner, daemon=True)
    t_bh = threading.Thread(target=blackhole_runner, daemon=True)
    t_ser.start()
    t_bh.start()

    print("waiting for both streams to open...")
    serial_ready.wait(timeout=5.0)
    blackhole_ready.wait(timeout=5.0)
    if "error" in serial_stats:
        print(f"  serial: {serial_stats['error']}", file=sys.stderr)
    if "error" in blackhole_stats:
        print(f"  blackhole: {blackhole_stats['error']}", file=sys.stderr)

    print(f"capturing {args.duration:.1f}s...")
    start_evt.set()
    time.sleep(args.duration)
    stop_evt.set()

    t_ser.join(timeout=3.0)
    t_bh.join(timeout=3.0)

    print("\nresults:")
    print(f"  inmp441: {serial_stats}")
    print(f"  system:  {blackhole_stats}")
    if os.path.exists(inmp_path):
        print(f"  -> {inmp_path}  ({os.path.getsize(inmp_path)} B)")
    if os.path.exists(sys_path):
        print(f"  -> {sys_path}  ({os.path.getsize(sys_path)} B)")


if __name__ == "__main__":
    main()
