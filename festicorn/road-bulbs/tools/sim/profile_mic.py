#!/usr/bin/env python3
"""Profile phone mic RMS from sensor packets.

Listens on UDP 4210, collects rawRms values, prints statistics.

Usage:
    python profile_mic.py [--port 4210] [--duration 30]

Output: min, max, mean, median, p25, p75, p90, p95, histogram.
Use this to set RMS_FLOOR and RMS_CEILING in receiver.cpp.
"""

import argparse
import socket
import struct
import time
import sys

PACKET_SIZE = 15

def main():
    parser = argparse.ArgumentParser(description="Profile phone mic RMS")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print(f"Listening on UDP {args.port} for {args.duration}s...")
    print("Make some noise, be quiet, vary the volume.\n")

    start = time.time()
    rms_values = []
    mic_on_count = 0

    while time.time() - start < args.duration:
        try:
            data, addr = sock.recvfrom(64)
        except socket.timeout:
            continue
        if len(data) != PACKET_SIZE:
            continue
        ax, ay, az, gx, gy, gz, rms, mic = struct.unpack('<hhhhhh HB', data)
        rms_values.append(rms)
        if mic:
            mic_on_count += 1
        if len(rms_values) % 25 == 0:
            print(f"\r  {len(rms_values)} samples, latest rms={rms}, mic={'ON' if mic else 'OFF'}", end="", flush=True)

    sock.close()
    print(f"\n\nCollected {len(rms_values)} samples in {time.time()-start:.1f}s")
    print(f"Mic enabled in {mic_on_count}/{len(rms_values)} packets")

    if not rms_values:
        print("No data!")
        return

    rms_values.sort()
    n = len(rms_values)

    def percentile(p):
        idx = int(n * p / 100)
        return rms_values[min(idx, n-1)]

    print(f"\n{'='*50}")
    print(f"  min:    {rms_values[0]}")
    print(f"  p5:     {percentile(5)}")
    print(f"  p25:    {percentile(25)}")
    print(f"  median: {percentile(50)}")
    print(f"  mean:   {sum(rms_values)/n:.0f}")
    print(f"  p75:    {percentile(75)}")
    print(f"  p90:    {percentile(90)}")
    print(f"  p95:    {percentile(95)}")
    print(f"  max:    {rms_values[-1]}")
    print(f"{'='*50}")

    # Histogram
    print("\nHistogram:")
    max_rms = max(rms_values)
    n_bins = 20
    bin_width = max(1, (max_rms + 1) // n_bins)
    bins = [0] * n_bins
    for v in rms_values:
        b = min(v // bin_width, n_bins - 1)
        bins[b] += 1
    max_count = max(bins)
    for i, count in enumerate(bins):
        lo = i * bin_width
        hi = lo + bin_width - 1
        bar = '#' * int(40 * count / max_count) if max_count > 0 else ''
        print(f"  {lo:5d}-{hi:5d} [{count:4d}] {bar}")

    # Suggestions
    print(f"\nSuggested thresholds:")
    print(f"  RMS_FLOOR   = {percentile(75):.0f}  (p75 — noise floor for this environment)")
    print(f"  RMS_CEILING = {percentile(95):.0f}  (p95 — loud events)")
    print(f"  SIMPLE_SPARKLE_FLOOR = {percentile(75):.0f}")

if __name__ == "__main__":
    main()
