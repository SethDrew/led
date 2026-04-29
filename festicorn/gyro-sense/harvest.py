#!/usr/bin/env python3
"""Harvest a sender_recorder capture from the duck-recorder firmware.

Usage:
    python harvest.py <label>                       # default: dump + erase
    python harvest.py <label> --port /dev/cu.x      # explicit port
    python harvest.py <label> --keep                # keep the device file

Saves raw binary to data/recordings/<label>/<timestamp>.bin plus a parsed CSV.
The device file is erased after a successful dump unless --keep is passed,
so the next power-cycle on a battery will start a fresh capture.
"""
import argparse, base64, datetime, json, os, struct, sys, time

import serial

DEFAULT_PORT = "/dev/cu.usbmodem114201"
BAUD         = 460800
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "recordings")
HEADER_FMT   = "<IHHII"           # magic, ver, hz, count, cfg
SAMPLE_FMT   = "<hhhhhhH"         # ax,ay,az, gx,gy,gz, rms (14 B)
MAGIC        = 0x4B435544         # 'DUCK'

ACCEL_RANGE_G   = {0: 2,   1: 4,   2: 8,    3: 16}
GYRO_RANGE_DPS  = {0: 250, 1: 500, 2: 1000, 3: 2000}


def decode_cfg(ver: int, cfg: int) -> tuple[int, int]:
    """Return (accel_g, gyro_dps) for the capture's IMU configuration.

    ver=1 firmware predates the cfg field; assume the MPU-6050 power-on
    defaults (±2g, ±250°/s). ver>=2 packs AFS_SEL in byte 0 and FS_SEL
    in byte 1 of the 4-byte cfg word.
    """
    if ver < 2:
        return 2, 250
    afs = cfg & 0x03
    fs  = (cfg >> 8) & 0x03
    return ACCEL_RANGE_G.get(afs, 2), GYRO_RANGE_DPS.get(fs, 250)


def open_port(port: str) -> serial.Serial:
    s = serial.Serial(port, BAUD, timeout=2)
    time.sleep(0.5)
    s.reset_input_buffer()
    return s


def wait_for_line(s: serial.Serial, prefix: str, timeout_sec: float = 5.0) -> str:
    end = time.time() + timeout_sec
    while time.time() < end:
        line = s.readline().decode("utf-8", errors="replace").strip()
        if line.startswith(prefix):
            return line
        if line:
            print(f"  [device] {line}")
    raise TimeoutError(f"never got {prefix!r}")


def parse_and_save(raw: bytes, bin_path: str, csv_path: str) -> None:
    with open(bin_path, "wb") as f:
        f.write(raw)
    print(f"  Saved binary: {bin_path}")
    if len(raw) < 16:
        print("  ERR: file too short to parse")
        return
    magic, ver, hz, count, cfg = struct.unpack(HEADER_FMT, raw[:16])
    if magic != MAGIC:
        print(f"  ERR: bad magic 0x{magic:08x}")
        return
    accel_g, gyro_dps = decode_cfg(ver, cfg)
    body = raw[16:]
    n = len(body) // 14
    if n != count:
        print(f"  NOTE: header count={count}, body has {n} samples (using body)")
    with open(csv_path, "w") as out:
        out.write(f"# accel_g={accel_g} gyro_dps={gyro_dps} ver={ver}\n")
        out.write("t,ax,ay,az,gx,gy,gz,rms\n")
        for i in range(n):
            ax, ay, az, gx, gy, gz, rms = struct.unpack(SAMPLE_FMT, body[i*14:(i+1)*14])
            t = i / hz
            out.write(f"{t:.4f},{ax},{ay},{az},{gx},{gy},{gz},{rms}\n")
    meta_path = os.path.splitext(csv_path)[0] + ".meta.json"
    with open(meta_path, "w") as mf:
        json.dump({
            "header_version": ver,
            "sample_hz": hz,
            "sample_count": n,
            "accel_g": accel_g,
            "gyro_dps": gyro_dps,
        }, mf, indent=2)
    print(f"  Saved CSV:    {csv_path}")
    print(f"  Saved meta:   {meta_path}")
    print(f"  Summary:      {n} samples @ {hz} Hz = {n/hz:.2f} s "
          f"(±{accel_g}g / ±{gyro_dps}°/s)")


def harvest(port: str, label: str, erase: bool = True) -> None:
    out_dir = os.path.join(DATA_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    stamp   = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    s = open_port(port)
    s.write(b"s\n")        # stop any active recording so the file is finalized
    time.sleep(0.3)
    while s.in_waiting:
        print(f"  [device] {s.readline().decode('utf-8', errors='replace').rstrip()}")

    print("Sending dump request...")
    s.write(b"d\n")
    begin = wait_for_line(s, "DUMPBEGIN", timeout_sec=10.0)
    # DUMPBEGIN files=N
    n_files = int(begin.split("files=", 1)[1])
    print(f"  expected {n_files} files")

    cur_idx = None
    cur_bytes = 0
    cur_encoded: list[str] = []
    saved = 0

    def flush():
        nonlocal saved, cur_idx, cur_bytes, cur_encoded
        if cur_idx is None:
            return
        raw = base64.b64decode("".join(cur_encoded))
        if len(raw) != cur_bytes:
            print(f"  WARN: rec{cur_idx}: got {len(raw)} bytes, expected {cur_bytes}")
        bin_path = os.path.join(out_dir, f"{stamp}_rec{cur_idx}.bin")
        csv_path = os.path.join(out_dir, f"{stamp}_rec{cur_idx}.csv")
        print(f"rec{cur_idx} ({cur_bytes} bytes):")
        parse_and_save(raw, bin_path, csv_path)
        saved += 1
        cur_idx = None
        cur_bytes = 0
        cur_encoded = []

    while True:
        line = s.readline().decode("utf-8", errors="replace").strip()
        if line == "DUMPEND":
            flush()
            break
        if not line:
            continue
        if line.startswith("FILE idx="):
            flush()
            # FILE idx=K bytes=M
            try:
                idx_part, bytes_part = line.split(" bytes=", 1)
                cur_idx = int(idx_part.split("=", 1)[1])
                cur_bytes = int(bytes_part)
                cur_encoded = []
            except Exception as e:
                print(f"  WARN: bad header {line!r}: {e}")
            continue
        cur_encoded.append(line)

    print(f"\nSaved {saved}/{n_files} files to {out_dir}/")

    if erase:
        s.write(b"e\n")
        time.sleep(0.5)
        while s.in_waiting:
            print(f"  [device] {s.readline().decode('utf-8', errors='replace').rstrip()}")

    s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="capture label (e.g. duck, hard, hanging)")
    ap.add_argument("--port",  default=DEFAULT_PORT)
    ap.add_argument("--keep",  action="store_true",
                    help="keep the device file after dump (default: erase)")
    args = ap.parse_args()

    try:
        harvest(args.port, args.label, erase=not args.keep)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
