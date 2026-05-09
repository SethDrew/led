#!/usr/bin/env python3
"""Provision a Nano streaming-receiver with an EEPROM device_id.

Sends [0xFD, <id>] at 1 Mbps, expects echo [0xFD, <id>] back.
Run once per Nano:
    python provision_nano.py --port /dev/cu.usbserial-1110 --id 1
"""
import argparse
import sys
import time

import serial


CMD_SET_ID = 0xFD


def provision(port: str, device_id: int, baud: int = 1_000_000, settle: float = 2.0) -> int:
    if not (0 <= device_id <= 255):
        print(f"id must be 0..255, got {device_id}", file=sys.stderr)
        return 2

    with serial.Serial(port, baud, timeout=1.0) as ser:
        time.sleep(settle)  # wait for bootloader
        ser.reset_input_buffer()
        ser.write(bytes([CMD_SET_ID, device_id]))
        ser.flush()
        resp = ser.read(2)

    if len(resp) != 2:
        print(f"no/short response: {resp!r}", file=sys.stderr)
        return 1
    if resp[0] != CMD_SET_ID or resp[1] != device_id:
        print(f"bad echo: {resp!r} (expected {bytes([CMD_SET_ID, device_id])!r})", file=sys.stderr)
        return 1

    print(f"OK: {port} now device_id={device_id}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--id", type=int, required=True, dest="device_id")
    p.add_argument("--baud", type=int, default=1_000_000)
    args = p.parse_args()
    return provision(args.port, args.device_id, args.baud)


if __name__ == "__main__":
    sys.exit(main())
