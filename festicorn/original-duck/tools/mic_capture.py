#!/usr/bin/env python3
"""
mic_capture.py — capture INMP441 stream from ESP32-C3 (mic_profile fw).

Reads framed binary blocks (AA 55 A5 5A | u16 block# | u16 n | n*int16le)
from the serial port, writes a 16 kHz / 16-bit / mono WAV. Live RMS meter
on stderr. Status lines from firmware ('#'-prefixed) are echoed to stderr.

Run inside a venv:  python -m venv .venv && .venv/bin/pip install pyserial
                    .venv/bin/python tools/mic_capture.py
"""
import argparse
import math
import signal
import struct
import sys
import time
import wave

import serial

SYNC = b"\xAA\x55\xA5\x5A"
SAMPLE_RATE = 16000


def find_sync(ser, leftover=b""):
    """Read until SYNC magic is found. Returns leftover bytes after magic."""
    buf = leftover
    while True:
        idx = buf.find(SYNC)
        if idx >= 0:
            # Anything before the magic that starts with '#' = status line.
            preamble = buf[:idx]
            if preamble:
                for line in preamble.split(b"\n"):
                    s = line.strip()
                    if s.startswith(b"#"):
                        sys.stderr.write(s.decode("utf-8", "replace") + "\n")
                        sys.stderr.flush()
            return buf[idx + 4:]
        # Keep tail in case magic straddles reads.
        if len(buf) > 4:
            # Echo any complete '#' lines in buf before discarding.
            *lines, tail = buf.split(b"\n")
            for line in lines:
                s = line.strip()
                if s.startswith(b"#"):
                    sys.stderr.write(s.decode("utf-8", "replace") + "\n")
            buf = tail
        chunk = ser.read(256)
        if not chunk:
            continue
        buf += chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem1114201")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--out",  default=None,
                    help="Output WAV path (default: mic_<timestamp>.wav)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="Stop after N seconds (0 = until Ctrl-C)")
    args = ap.parse_args()

    out_path = args.out or time.strftime("mic_%Y%m%d_%H%M%S.wav")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    sys.stderr.write(f"# Opened {args.port} @ {args.baud}, writing {out_path}\n")

    wav = wave.open(out_path, "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)

    stopping = {"flag": False}

    def stop(*_):
        stopping["flag"] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    leftover = b""
    last_block = None
    drops = 0
    total_samples = 0
    last_meter_t = time.time()
    sumsq_window = 0.0
    n_window = 0
    t_start = time.time()

    try:
        while not stopping["flag"]:
            if args.seconds > 0 and (time.time() - t_start) >= args.seconds:
                break

            leftover = find_sync(ser, leftover)
            # need 4 bytes header (block#, n)
            while len(leftover) < 4:
                chunk = ser.read(4 - len(leftover))
                if not chunk:
                    if stopping["flag"]:
                        break
                    continue
                leftover += chunk
            if len(leftover) < 4:
                break
            block_no, n_samples = struct.unpack("<HH", leftover[:4])
            leftover = leftover[4:]

            if n_samples == 0 or n_samples > 4096:
                # bogus — resync
                drops += 1
                continue

            need = n_samples * 2
            while len(leftover) < need:
                chunk = ser.read(need - len(leftover))
                if not chunk:
                    if stopping["flag"]:
                        break
                    continue
                leftover += chunk
            if len(leftover) < need:
                break
            payload = leftover[:need]
            leftover = leftover[need:]

            if last_block is not None:
                expected = (last_block + 1) & 0xFFFF
                if block_no != expected:
                    gap = (block_no - expected) & 0xFFFF
                    drops += gap
                    sys.stderr.write(f"# [GAP] {gap} blocks (got {block_no}, expected {expected})\n")
            last_block = block_no

            wav.writeframes(payload)
            total_samples += n_samples

            # RMS for meter
            samples = struct.unpack(f"<{n_samples}h", payload)
            for s in samples:
                sumsq_window += s * s
            n_window += n_samples

            now = time.time()
            if now - last_meter_t >= 0.25:
                rms = math.sqrt(sumsq_window / n_window) if n_window else 0.0
                # 0..32768 → ~0..90 dBFS. Show both.
                dbfs = 20 * math.log10(rms / 32768.0) if rms > 0 else -120.0
                bars = int(min(40, max(0, (dbfs + 60) * 40 / 60)))
                meter = "#" * bars + "-" * (40 - bars)
                sys.stderr.write(f"\rrms={rms:7.0f}  {dbfs:6.1f} dBFS  [{meter}]  drops={drops}")
                sys.stderr.flush()
                sumsq_window = 0.0
                n_window = 0
                last_meter_t = now
    finally:
        wav.close()
        ser.close()
        dur = total_samples / SAMPLE_RATE
        sys.stderr.write(f"\n# Wrote {out_path}: {total_samples} samples ({dur:.2f}s), drops={drops}\n")


if __name__ == "__main__":
    main()
