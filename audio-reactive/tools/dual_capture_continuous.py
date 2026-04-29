#!/usr/bin/env python3
"""Continuous dual capture for narrated sessions.

Streams INMP441 (USB serial) + BlackHole loopback to disk incrementally so a
crash doesn't lose data. Writes session.yaml with the wall-clock start time;
narrated segments are aligned offline by computing offsets from start_iso.

Run forever until SIGINT/SIGTERM:
    dual_capture_continuous.py [--port /dev/cu.usbmodem114301]
                               [--blackhole-index N] [--out-dir DIR]

Writes:
    {out_dir}/inmp441.wav   — 16 kHz mono PCM_16
    {out_dir}/system.wav    — 16 kHz mono PCM_16 (resampled from BlackHole rate)
    {out_dir}/session.yaml  — start_iso, sample rate, paths
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd
import soundfile as sf
import serial
import yaml

SAMPLE_RATE = 16000
SERIAL_PORT_DEFAULT = "/dev/cu.usbmodem114301"
SERIAL_BAUD_DEFAULT = 460800

stop_evt = threading.Event()


def find_blackhole_index() -> int:
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if "BlackHole" in d["name"] and d["max_input_channels"] > 0:
            return i
    return 3  # fall back to dual_capture.py default


def serial_loop(port, baud, start_evt, writer, stats):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except Exception as e:
        stats["error"] = f"serial open failed: {e}"
        start_evt.set()
        return
    ser.reset_input_buffer()
    start_evt.set()
    start_evt.wait()  # synchronized go
    try:
        while not stop_evt.is_set():
            data = ser.read(4096)
            if not data:
                continue
            if len(data) % 2 == 1:
                data = data[:-1]
            samples = np.frombuffer(data, dtype="<i2")
            writer.write(samples)
            stats["samples"] += int(samples.size)
    finally:
        try:
            ser.close()
        except Exception:
            pass


def blackhole_loop(device_index, start_evt, writer, stats):
    info = sd.query_devices(device_index)
    in_channels = min(2, info["max_input_channels"])
    in_rate = int(info["default_samplerate"])
    stats["in_rate"] = in_rate

    def cb(indata, frames, time_info, status):
        if stop_evt.is_set() or not start_evt.is_set():
            return
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
        if in_rate != SAMPLE_RATE:
            n_in = mono.shape[0]
            n_out = int(round(n_in * SAMPLE_RATE / in_rate))
            x_in = np.linspace(0.0, 1.0, n_in, endpoint=False)
            x_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
            mono = np.interp(x_out, x_in, mono).astype(np.float32)
        writer.write(mono)
        stats["samples"] += int(mono.size)

    stream = sd.InputStream(
        device=device_index,
        channels=in_channels,
        samplerate=in_rate,
        blocksize=1024,
        dtype="float32",
        callback=cb,
    )
    stream.start()
    start_evt.set()
    start_evt.wait()
    try:
        while not stop_evt.is_set():
            time.sleep(0.1)
    finally:
        stream.stop()
        stream.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=SERIAL_PORT_DEFAULT)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD_DEFAULT)
    ap.add_argument("--blackhole-index", type=int, default=None,
                    help="BlackHole device index (auto-detected if omitted)")
    ap.add_argument("--out-dir", default=None,
                    help="output session directory (auto-named if omitted)")
    args = ap.parse_args()

    bh_index = args.blackhole_index if args.blackhole_index is not None \
               else find_blackhole_index()

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..",
            "library", "test-vectors", "inmp441-validation",
            f"session_{stamp}",
        ))
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    inmp_path = os.path.join(out_dir, "inmp441.wav")
    sys_path = os.path.join(out_dir, "system.wav")
    session_path = os.path.join(out_dir, "session.yaml")

    def handler(signum, frame):
        stop_evt.set()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f"out dir: {out_dir}", flush=True)
    print(f"blackhole device index: {bh_index}", flush=True)

    inmp_writer = sf.SoundFile(inmp_path, "w", SAMPLE_RATE, 1, subtype="PCM_16")
    sys_writer = sf.SoundFile(sys_path, "w", SAMPLE_RATE, 1, subtype="PCM_16")

    serial_ready = threading.Event()
    blackhole_ready = threading.Event()
    serial_stats = {"samples": 0}
    blackhole_stats = {"samples": 0}

    t_ser = threading.Thread(
        target=serial_loop,
        args=(args.port, args.baud, serial_ready, inmp_writer, serial_stats),
        daemon=True,
    )
    t_bh = threading.Thread(
        target=blackhole_loop,
        args=(bh_index, blackhole_ready, sys_writer, blackhole_stats),
        daemon=True,
    )
    t_ser.start()
    t_bh.start()

    serial_ready.wait(timeout=5.0)
    blackhole_ready.wait(timeout=5.0)
    if "error" in serial_stats:
        print(f"serial: {serial_stats['error']}", file=sys.stderr, flush=True)
        return
    if "error" in blackhole_stats:
        print(f"blackhole: {blackhole_stats['error']}", file=sys.stderr, flush=True)
        return

    start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    with open(session_path, "w") as f:
        yaml.safe_dump({
            "start_iso": start_iso,
            "sample_rate_hz": SAMPLE_RATE,
            "wav_files": {"inmp441": "inmp441.wav", "system": "system.wav"},
            "port": args.port,
            "blackhole_index": bh_index,
            "blackhole_in_rate_hz": blackhole_stats.get("in_rate"),
        }, f)

    print(f"start: {start_iso}", flush=True)
    print("recording ... send SIGINT/SIGTERM to stop", flush=True)

    try:
        while not stop_evt.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_evt.set()

    t_ser.join(timeout=2.0)
    t_bh.join(timeout=2.0)
    inmp_writer.close()
    sys_writer.close()

    elapsed = time.monotonic() - t0
    print(f"\nstopped after {elapsed:.1f}s", flush=True)
    print(f"  inmp441 samples: {serial_stats['samples']} "
          f"({serial_stats['samples']/SAMPLE_RATE:.1f}s)", flush=True)
    print(f"  system  samples: {blackhole_stats['samples']} "
          f"({blackhole_stats['samples']/SAMPLE_RATE:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
