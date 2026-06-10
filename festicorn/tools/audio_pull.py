#!/usr/bin/env python3
"""
audio_pull.py — drive the bench audio_recorder and pull a playable .wav.

Companion to:
  - festicorn/original-duck/src/sender_audio.cpp  (duck streams the waveform)
  - festicorn/bench-bulbs/src/audio_recorder.cpp  (bench records + dumps)

Flow: open the recorder's serial port, tell it to record for N seconds while
you talk to the duck, stop, then pull the LittleFS file as framed hex and wrap
it as 8 kHz / 8-bit unsigned mono WAV.

Run with the project venv python (has pyserial):
  .venv/bin/python festicorn/tools/audio_pull.py --port /dev/cu.usbserial-XXXX

The recorder runs at 460800 baud. Channel is fixed to 1 on both ends.
"""
from __future__ import annotations
import argparse, glob, sys, time, wave
import serial

BAUD = 460800


def find_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    cands = sorted(glob.glob("/dev/cu.usbserial-*"))
    if len(cands) == 1:
        print(f"[port] auto-detected {cands[0]}")
        return cands[0]
    if not cands:
        sys.exit("No /dev/cu.usbserial-* found. Pass --port explicitly.")
    sys.exit("Multiple serial ports found; pass --port:\n  " + "\n  ".join(cands))


def read_until(ser: serial.Serial, marker: str, timeout: float) -> list[str]:
    """Read lines until one contains `marker`. Returns all lines (incl. marker)."""
    deadline = time.time() + timeout
    lines: list[str] = []
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("ascii", "replace").strip()
        if line:
            lines.append(line)
            if marker in line:
                return lines
    raise TimeoutError(f"timed out waiting for '{marker}'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull a .wav from the bench audio recorder.")
    ap.add_argument("--port", help="recorder serial port (default: auto-detect)")
    ap.add_argument("--seconds", type=float, default=30.0, help="record duration (default 30)")
    ap.add_argument("--out", default=None, help="output .wav path (default audio_capture.wav)")
    args = ap.parse_args()

    port = find_port(args.port)
    out = args.out or "audio_capture.wav"

    with serial.Serial(port, BAUD, timeout=1.0) as ser:
        time.sleep(0.3)
        ser.reset_input_buffer()

        # ── Record ────────────────────────────────────────────────
        print(f"[rec] starting {args.seconds:.0f}s capture — TALK TO THE DUCK NOW")
        ser.write(b"R")
        for line in read_until(ser, "[REC] start", timeout=3.0):
            print("   ", line)

        # Countdown while the board records.
        end = time.time() + args.seconds
        while time.time() < end:
            remaining = end - time.time()
            print(f"\r   recording… {remaining:4.1f}s left ", end="", flush=True)
            time.sleep(0.2)
        print()

        ser.write(b"S")
        for line in read_until(ser, "[REC] stop", timeout=5.0):
            print("   ", line)

        # ── Dump ──────────────────────────────────────────────────
        print("[dump] pulling file…")
        ser.reset_input_buffer()
        ser.write(b"D")

        # Header: "[DUMP] start size=N rate=R bits=8 enc=u8"
        size = None
        rate = 8000
        for line in read_until(ser, "[DUMP] start", timeout=5.0):
            if "[DUMP] start" in line:
                for tok in line.split():
                    if tok.startswith("size="):
                        size = int(tok.split("=", 1)[1])
                    elif tok.startswith("rate="):
                        rate = int(tok.split("=", 1)[1])
        if size is None:
            sys.exit("dump header missing size=")
        print(f"   size={size} bytes  rate={rate} Hz")

        data = bytearray()
        deadline = time.time() + 60.0
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", "replace").strip()
            if not line:
                continue
            if "[DUMP] end" in line:
                break
            try:
                data.extend(bytes.fromhex(line))
            except ValueError:
                print(f"   [skip non-hex] {line}")
        else:
            print("   WARNING: hit dump timeout before [DUMP] end")

        print(f"   received {len(data)} bytes")
        if size is not None and len(data) != size:
            print(f"   WARNING: byte count {len(data)} != reported size {size}")

    if not data:
        sys.exit("No audio data received — is the duck flashed with sender_audio and streaming?")

    # ── Wrap WAV (8-bit unsigned PCM, mono) ───────────────────────
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)            # 8-bit; wave treats this as unsigned, mid=128
        w.setframerate(rate)
        w.writeframes(bytes(data))

    dur = len(data) / rate
    print(f"[ok] wrote {out}  ({len(data)} samples, {dur:.1f}s @ {rate} Hz)")
    print("     play it, then nudge QUANT_SHIFT in sender_audio.cpp if too quiet/clipped.")


if __name__ == "__main__":
    main()
