#!/usr/bin/env python3
"""Capture an ESP-NOW stream from the stream_bridge ESP32 over USB serial.

Companion to stream_sender.cpp + stream_bridge.cpp. The bridge forwards each
ESP-NOW packet over USB serial as:
    0xAA 0x55  <len:u8>  <payload[len]>  <crc8>

Each ESP-NOW payload contains:
    StreamHeader (8 B): magic 'STR1' (0x31525453), uint16 seq,
                        uint8 sample_count, uint8 cfg
    cfg low nibble  = AFS_SEL (0=±2g, 1=±4g, 2=±8g, 3=±16g)
    cfg high nibble = FS_SEL  (0=±250, 1=±500, 2=±1000, 3=±2000 °/s)
    cfg=0 from a legacy sender → assume ±2g/±250°/s.
    Sample[N]   (14 B each): int16 ax,ay,az,gx,gy,gz; uint16 rms

Saves output in the same on-disk format as harvest.py:
    16 B header: 'DUCK', uint16 ver, uint16 sample_hz=200,
                 uint32 sample_count, uint32 cfg
    body: 14 B/sample
plus a parsed CSV (with `# accel_g=N gyro_dps=M ver=V` comment line) and
a `<basename>.meta.json` sidecar so the analysis pipeline can scale counts
to physical units.

Usage:
    python stream_capture.py --label duck_stream --seconds 20
"""
import argparse, datetime, json, os, struct, sys, time

import serial

DEFAULT_PORT  = "/dev/cu.usbserial-0001"
BAUD          = 1_000_000
DATA_DIR      = os.path.join(os.path.dirname(__file__), "data", "recordings")
SAMPLE_HZ     = 200

# On-disk format (matches harvest.py)
DISK_HEADER_FMT = "<IHHII"           # magic, ver, hz, count, cfg
DISK_MAGIC      = 0x4B435544         # 'DUCK'
DISK_VER        = 2                  # ver 2 carries cfg byte
SAMPLE_FMT      = "<hhhhhhH"         # ax,ay,az, gx,gy,gz, rms (14 B)
SAMPLE_BYTES    = 14

# Wire framing
SYNC0           = 0xAA
SYNC1           = 0x55
STREAM_MAGIC    = 0x31525453         # 'STR1' little-endian
STREAM_HDR_FMT  = "<IHBB"            # magic, seq, sample_count, cfg
STREAM_HDR_BYTES = 8

ACCEL_RANGE_G  = {0: 2,   1: 4,   2: 8,    3: 16}
GYRO_RANGE_DPS = {0: 250, 1: 500, 2: 1000, 3: 2000}


def decode_cfg_byte(cfg: int) -> tuple[int, int]:
    """cfg=0 → legacy ±2g/±250°/s. Otherwise low nibble = AFS_SEL,
    high nibble = FS_SEL."""
    afs = cfg & 0x0F
    fs  = (cfg >> 4) & 0x0F
    return ACCEL_RANGE_G.get(afs, 2), GYRO_RANGE_DPS.get(fs, 250)


def crc8(data: bytes) -> int:
    """CRC-8/CCITT, poly 0x07, init 0x00 — matches stream_bridge.cpp."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def read_frames(s: serial.Serial, deadline: float):
    """Generator that yields validated ESP-NOW payloads until `deadline`."""
    state = 0
    plen = 0
    payload = bytearray()
    while time.time() < deadline:
        # Block briefly so we don't burn CPU; timeout is set on the port.
        chunk = s.read(1)
        if not chunk:
            continue
        b = chunk[0]
        if state == 0:
            if b == SYNC0:
                state = 1
        elif state == 1:
            state = 2 if b == SYNC1 else (1 if b == SYNC0 else 0)
        elif state == 2:
            plen = b
            if plen == 0 or plen > 250:
                state = 0
                continue
            payload = bytearray()
            state = 3
        elif state == 3:
            payload.append(b)
            if len(payload) == plen:
                state = 4
        elif state == 4:
            crc = b
            if crc8(bytes(payload)) == crc:
                yield bytes(payload)
            # else: silent drop, host stats track only seq gaps
            state = 0


def parse_payload(payload: bytes):
    """Returns (seq, cfg, samples_bytes) or None when payload is malformed."""
    if len(payload) < STREAM_HDR_BYTES:
        return None
    magic, seq, count, cfg = struct.unpack(STREAM_HDR_FMT, payload[:STREAM_HDR_BYTES])
    if magic != STREAM_MAGIC:
        return None
    body = payload[STREAM_HDR_BYTES:]
    expected = count * SAMPLE_BYTES
    if len(body) < expected:
        return None
    return seq, cfg, body[:expected]


def capture(port: str, label: str, seconds: float) -> str:
    out_dir  = os.path.join(DATA_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    stamp    = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    bin_path = os.path.join(out_dir, f"{stamp}.bin")
    csv_path = os.path.join(out_dir, f"{stamp}.csv")

    s = serial.Serial(port, BAUD, timeout=0.05)
    time.sleep(0.3)
    s.reset_input_buffer()

    print(f"Capturing {seconds:.1f}s from {port} → {bin_path}")
    deadline = time.time() + seconds

    body = bytearray()
    pkt_count   = 0
    sample_count = 0
    drops       = 0
    last_seq    = None
    first_seq   = None
    seen_cfg    = None

    for payload in read_frames(s, deadline):
        parsed = parse_payload(payload)
        if parsed is None:
            continue
        seq, cfg, samples = parsed
        if seen_cfg is None:
            seen_cfg = cfg
        elif cfg != seen_cfg:
            # Sender shouldn't change cfg mid-capture; warn so we notice.
            print(f"WARN: cfg byte changed mid-capture: 0x{seen_cfg:02x} → 0x{cfg:02x}")
            seen_cfg = cfg
        if last_seq is not None:
            gap = (seq - last_seq) & 0xFFFF
            if gap == 0:
                # duplicate seq — count as drop heuristically
                pass
            elif gap > 1:
                drops += gap - 1
        else:
            first_seq = seq
        last_seq = seq
        body.extend(samples)
        sample_count += len(samples) // SAMPLE_BYTES
        pkt_count += 1

    s.close()

    cfg_byte = seen_cfg if seen_cfg is not None else 0
    accel_g, gyro_dps = decode_cfg_byte(cfg_byte)
    # Re-pack into the disk header's 32-bit cfg word (byte 0 AFS, byte 1 FS)
    afs = cfg_byte & 0x0F
    fs  = (cfg_byte >> 4) & 0x0F
    disk_cfg = afs | (fs << 8)

    # ── Write .bin (DUCK header + body) ───────────────────────────
    hdr = struct.pack(DISK_HEADER_FMT, DISK_MAGIC, DISK_VER, SAMPLE_HZ, sample_count, disk_cfg)
    with open(bin_path, "wb") as f:
        f.write(hdr)
        f.write(body)
    print(f"Saved binary: {bin_path}")

    # ── Write .csv ────────────────────────────────────────────────
    with open(csv_path, "w") as out:
        out.write(f"# accel_g={accel_g} gyro_dps={gyro_dps} ver={DISK_VER}\n")
        out.write("t,ax,ay,az,gx,gy,gz,rms\n")
        for i in range(sample_count):
            ax, ay, az, gx, gy, gz, rms = struct.unpack(
                SAMPLE_FMT, body[i*SAMPLE_BYTES:(i+1)*SAMPLE_BYTES])
            t = i / SAMPLE_HZ
            out.write(f"{t:.4f},{ax},{ay},{az},{gx},{gy},{gz},{rms}\n")
    print(f"Saved CSV:    {csv_path}")

    # ── Write sidecar metadata ────────────────────────────────────
    meta_path = os.path.splitext(csv_path)[0] + ".meta.json"
    with open(meta_path, "w") as mf:
        json.dump({
            "header_version": DISK_VER,
            "sample_hz": SAMPLE_HZ,
            "sample_count": sample_count,
            "accel_g": accel_g,
            "gyro_dps": gyro_dps,
            "stream_cfg_byte": cfg_byte,
        }, mf, indent=2)
    print(f"Saved meta:   {meta_path}")
    print(f"Range:        ±{accel_g}g / ±{gyro_dps}°/s (cfg=0x{cfg_byte:02x})")

    # ── Stats ─────────────────────────────────────────────────────
    expected = int(seconds * SAMPLE_HZ)
    rate = (sample_count / expected * 100.0) if expected else 0.0
    expected_pkts = pkt_count + drops
    drop_rate = (drops / expected_pkts * 100.0) if expected_pkts else 0.0
    print()
    print(f"Packets received:   {pkt_count}")
    print(f"Packet drops:       {drops}  ({drop_rate:.2f}%)")
    print(f"Samples received:   {sample_count}")
    print(f"Samples expected:   {expected}  ({rate:.1f}% of expected)")
    if first_seq is not None and last_seq is not None:
        span = (last_seq - first_seq) & 0xFFFF
        print(f"Seq span:           {first_seq} → {last_seq}  ({span+1} expected)")
    return bin_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",   required=True,
                    help="capture label (e.g. duck_stream, hanging_stream)")
    ap.add_argument("--port",    default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    try:
        capture(args.port, args.label, args.seconds)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
