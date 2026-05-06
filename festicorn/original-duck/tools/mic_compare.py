#!/usr/bin/env python3
"""
mic_compare.py — simultaneous multi-mic recording for cross-device comparison.

Records two streams in parallel:
  1. ESP32-C3 INMP441 over serial (mic_profile firmware)
  2. macOS default input (or selected device) via sounddevice

On start: prints "CLAP NOW for sync" and counts down 5 s, then records for
DURATION seconds. Outputs two timestamped WAVs in --out-dir.

Run inside a venv:
  python -m venv .venv
  .venv/bin/pip install pyserial sounddevice numpy
  .venv/bin/python tools/mic_compare.py --duration 30
"""
import argparse
import math
import os
import struct
import sys
import threading
import time
import wave

import serial

try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    sys.stderr.write("ERROR: install sounddevice + numpy in a venv:\n")
    sys.stderr.write("  python -m venv .venv && .venv/bin/pip install sounddevice numpy pyserial\n")
    sys.exit(1)

SYNC = b"\xAA\x55\xA5\x5A"
ESP32_RATE = 16000


def list_devices():
    print(sd.query_devices())


def esp32_recorder(port, baud, out_path, stop_evt, started_evt):
    """Capture ESP32 serial stream into a WAV. Sets started_evt once aligned."""
    ser = serial.Serial(port, baud, timeout=0.1)
    wav = wave.open(out_path, "wb")
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(ESP32_RATE)

    leftover = b""
    aligned = False

    def read_exact(n):
        nonlocal leftover
        while len(leftover) < n:
            if stop_evt.is_set():
                return None
            chunk = ser.read(n - len(leftover))
            if chunk:
                leftover += chunk
        out = leftover[:n]
        leftover = leftover[n:]
        return out

    try:
        while not stop_evt.is_set():
            # find sync
            while not stop_evt.is_set():
                idx = leftover.find(SYNC)
                if idx >= 0:
                    leftover = leftover[idx + 4:]
                    break
                # keep tail in case sync straddles
                if len(leftover) > 4:
                    leftover = leftover[-4:]
                chunk = ser.read(256)
                if chunk:
                    leftover += chunk
            if stop_evt.is_set():
                break
            hdr = read_exact(4)
            if hdr is None:
                break
            _bn, n_samples = struct.unpack("<HH", hdr)
            if n_samples == 0 or n_samples > 4096:
                continue
            payload = read_exact(n_samples * 2)
            if payload is None:
                break
            wav.writeframes(payload)
            if not aligned:
                aligned = True
                started_evt.set()
    finally:
        wav.close()
        ser.close()


def mac_recorder(device, samplerate, channels, out_path, duration_s, started_evt, stop_evt):
    """Record default-input audio to WAV. Waits for started_evt before opening WAV."""
    started_evt.wait(timeout=10.0)
    frames = []

    def cb(indata, frame_count, time_info, status):
        if status:
            sys.stderr.write(f"# [mac status] {status}\n")
        frames.append(indata.copy())

    with sd.InputStream(device=device, channels=channels, samplerate=samplerate,
                        dtype="int16", callback=cb):
        t_end = time.time() + duration_s
        while time.time() < t_end and not stop_evt.is_set():
            time.sleep(0.05)

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, channels), dtype=np.int16)
    with wave.open(out_path, "wb") as wav:
        wav.setnchannels(channels); wav.setsampwidth(2); wav.setframerate(samplerate)
        wav.writeframes(audio.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem1114201",
                    help="ESP32 serial port")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--mac-device", default=None,
                    help="sounddevice input device (name or index). Default = system default.")
    ap.add_argument("--mac-rate", type=int, default=48000)
    ap.add_argument("--mac-channels", type=int, default=1)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    esp_path = os.path.join(args.out_dir, f"esp32_{ts}.wav")
    mac_path = os.path.join(args.out_dir, f"mac_{ts}.wav")

    stop_evt = threading.Event()
    started_evt = threading.Event()

    print(f"ESP32 → {esp_path}")
    print(f"Mac   → {mac_path}")
    print()
    print("CLAP NOW for sync — recording starts in:")
    for n in range(5, 0, -1):
        print(f"  {n}...")
        time.sleep(1.0)
    print(f"RECORDING for {args.duration:.1f} s")

    esp_thread = threading.Thread(
        target=esp32_recorder,
        args=(args.port, args.baud, esp_path, stop_evt, started_evt),
        daemon=True,
    )
    esp_thread.start()

    try:
        mac_recorder(args.mac_device, args.mac_rate, args.mac_channels,
                     mac_path, args.duration, started_evt, stop_evt)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        esp_thread.join(timeout=3.0)

    print("Done.")
    print(f"  {esp_path}")
    print(f"  {mac_path}")


if __name__ == "__main__":
    main()
