#!/usr/bin/env python3
"""Record sensor packets from UDP port 4210 to a binary file.

Usage:
    python record_sensor.py [--port 4210] [--duration 30] [--out recording.bin]

Each packet is 15 bytes (SensorPacket struct). Timestamps stored in sidecar .times file.
"""

import argparse
import socket
import struct
import time
import sys

PACKET_SIZE = 15

def main():
    parser = argparse.ArgumentParser(description="Record sensor UDP packets")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds to record")
    parser.add_argument("--out", default="recording.bin")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print(f"Listening on UDP port {args.port} for {args.duration}s → {args.out}")

    start = time.time()
    count = 0
    with open(args.out, "wb") as f, open(args.out + ".times", "w") as tf:
        while time.time() - start < args.duration:
            try:
                data, addr = sock.recvfrom(64)
            except socket.timeout:
                continue
            if len(data) != PACKET_SIZE:
                continue
            f.write(data)
            tf.write(f"{time.time() - start:.4f}\n")
            count += 1
            if count % 25 == 0:
                elapsed = time.time() - start
                print(f"\r  {count} packets ({count/elapsed:.1f}/s)", end="", flush=True)

    elapsed = time.time() - start
    print(f"\nDone: {count} packets in {elapsed:.1f}s ({count/elapsed:.1f}/s)")

if __name__ == "__main__":
    main()
