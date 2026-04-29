#!/usr/bin/env python3
"""Send 'r' to start a 5 s recording in-place, then dump a few samples per
second from a fresh capture so we can confirm whether the IMU is reading live
motion. Run:

    python imu_live.py
    # then physically tilt the device 90° while it records

Reads back 5 s of data and prints a/g for ~10 evenly-spaced samples.
"""
import argparse, base64, struct, time, sys
import serial

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem114301")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    s = serial.Serial(args.port, 460800, timeout=2.0)
    time.sleep(0.5)
    s.reset_input_buffer()

    # erase any existing file (defensive)
    s.write(b"e\n")
    time.sleep(0.3)
    while s.in_waiting: s.read(s.in_waiting)

    print(f"Recording for {args.seconds:.1f} s — TILT THE DEVICE NOW")
    s.write(b"r\n")
    time.sleep(args.seconds)
    s.write(b"s\n")
    time.sleep(0.5)
    while s.in_waiting:
        line = s.readline().decode("utf-8", errors="replace").rstrip()
        if line: print(f"  [device] {line}")

    s.write(b"d\n")
    while True:
        line = s.readline().decode("utf-8", errors="replace").strip()
        if line.startswith("DUMPBEGIN"):
            n = int(line.split("bytes=", 1)[1])
            break

    encoded = []
    while True:
        line = s.readline().decode("utf-8", errors="replace").strip()
        if line == "DUMPEND": break
        if line: encoded.append(line)
    raw = base64.b64decode("".join(encoded))

    s.write(b"e\n")
    time.sleep(0.3)
    s.close()

    magic, ver, hz, count, _ = struct.unpack("<IHHII", raw[:16])
    body = raw[16:]
    print(f"\nGot {count} samples @ {hz} Hz ({count/hz:.2f} s)")
    step = max(1, count // 12)
    print(f"\n{'t(s)':>6}  {'ax':>6} {'ay':>6} {'az':>6}    {'gx':>6} {'gy':>6} {'gz':>6}    rms")
    for i in range(0, count, step):
        ax, ay, az, gx, gy, gz, rms = struct.unpack("<hhhhhhH", body[i*14:(i+1)*14])
        print(f"{i/hz:6.2f}  {ax:6} {ay:6} {az:6}    {gx:6} {gy:6} {gz:6}    {rms}")

if __name__ == "__main__":
    main()
