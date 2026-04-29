#!/usr/bin/env python3
"""
register.py — flash sender firmware to one or more ESP32-C3 bulb-sender
boards, capture validation telemetry, and upsert each board into the
fleet catalog at festicorn/catalog/boards.yaml (matched by MAC address).

Usage:
    register.py PORT [PORT ...]
    register.py --no-flash PORT [PORT ...]      # skip flashing, just validate
    register.py --capture-secs 10 PORT          # longer capture window

Existing catalog entries with the same MAC are updated in place.
New MACs are appended. The script never deletes — pruning is manual.
"""
import argparse
import datetime
import re
import subprocess
import sys
import time
from pathlib import Path

import serial
import yaml

REPO_ROOT       = Path(__file__).resolve().parents[3]
GYRO_SENSE_DIR  = REPO_ROOT / "festicorn" / "gyro-sense"
SENDER_SRC      = GYRO_SENSE_DIR / "src" / "sender.cpp"
CATALOGUE_PATH  = REPO_ROOT / "festicorn" / "catalog" / "boards.yaml"
PIO             = REPO_ROOT / ".venv" / "bin" / "pio"


def git_sha_short() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def sender_dirty() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain", str(SENDER_SRC)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


def flash(port: str) -> bool:
    print(f"  [flash] {port}")
    r = subprocess.run(
        [str(PIO), "run", "-e", "sender", "-t", "upload", "--upload-port", port],
        cwd=GYRO_SENSE_DIR, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("  [flash] FAILED")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        return False
    return True


def capture(port: str, secs: float) -> list[str]:
    s = serial.Serial(port, 460800, timeout=1)
    s.setDTR(False); s.setRTS(True); time.sleep(0.15); s.setRTS(False)
    end = time.time() + secs
    lines: list[str] = []
    while time.time() < end:
        line = s.readline()
        if line:
            try:
                t = line.decode("utf-8", errors="replace").rstrip()
                if t:
                    lines.append(t)
            except Exception:
                pass
    s.close()
    return lines


# ── Parsers ──────────────────────────────────────────────────────────
RE_MAC      = re.compile(r"MAC=([0-9A-Fa-f:]{17})")
RE_CH_BANNER = re.compile(r"ch=(\d+)\s+MAC=")
RE_FOUND_CH = re.compile(r"Found '[^']+' on ch=(\d+)")
RE_IMU_OK   = re.compile(r"^IMU:\s*ok")
RE_IMU_NF   = re.compile(r"^IMU:\s*NOT\s*FOUND")
RE_SEQ      = re.compile(r"^seq=\s*(\d+)")


def parse(lines: list[str]) -> dict:
    rec: dict = {}
    seqs: list[int] = []
    for ln in lines:
        if (m := RE_MAC.search(ln)):
            rec["mac"] = m.group(1).upper()
        if (m := RE_CH_BANNER.search(ln)):
            rec["channel"] = int(m.group(1))
        elif (m := RE_FOUND_CH.search(ln)) and "channel" not in rec:
            rec["channel"] = int(m.group(1))
        if RE_IMU_OK.match(ln):
            rec["imu_status"] = "ok"
        elif RE_IMU_NF.match(ln):
            rec["imu_status"] = "not_found"
        if (m := RE_SEQ.match(ln)):
            seqs.append(int(m.group(1)))
    if seqs:
        rec["first_seq"] = seqs[0]
        rec["last_seq"]  = seqs[-1]
    return rec


# ── Catalogue I/O ────────────────────────────────────────────────────
def load_cat() -> dict:
    if not CATALOGUE_PATH.exists():
        return {"schema_version": 2, "boards": []}
    with CATALOGUE_PATH.open() as f:
        return yaml.safe_load(f) or {"schema_version": 2, "boards": []}


def save_cat(cat: dict) -> None:
    with CATALOGUE_PATH.open("w") as f:
        yaml.safe_dump(cat, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def upsert(cat: dict, mac: str, port: str, parsed: dict, sha: str, dirty: bool) -> tuple[str, dict]:
    """Insert or update entry. Returns (action, entry)."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    validation = {
        "utc": now,
        "port": port,
        "channel": parsed.get("channel"),
        "imu_status": parsed.get("imu_status", "unknown"),
        "first_seq": parsed.get("first_seq"),
        "last_seq": parsed.get("last_seq"),
        "git_sha": sha,
        "dirty": dirty,
    }
    boards = cat.setdefault("boards", [])
    for entry in boards:
        if entry.get("mac", "").upper() == mac:
            entry["last_validated"] = validation
            return "updated", entry
    entry = {
        "mac": mac,
        "id": "",
        "chip": "esp32-c3",
        "role": "bulb-sender",
        "i2c_addr": 0x68,
        "first_seen": now,
        "last_validated": validation,
        "notes": "",
    }
    boards.append(entry)
    return "added", entry


# ── Main ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ports", nargs="+", help="serial ports to register")
    ap.add_argument("--no-flash", action="store_true",
                    help="skip flashing; reset+capture only")
    ap.add_argument("--capture-secs", type=float, default=6.0)
    args = ap.parse_args()

    sha   = git_sha_short()
    dirty = sender_dirty()
    cat   = load_cat()
    print(f"git: {sha}{' (sender.cpp dirty)' if dirty else ''}")
    print(f"catalog: {CATALOGUE_PATH} ({len(cat.get('boards', []))} entries)\n")

    summary = []
    for port in args.ports:
        print(f"--- {port} ---")
        if not args.no_flash:
            if not flash(port):
                summary.append((port, "FAIL-flash", None))
                continue

        try:
            lines = capture(port, args.capture_secs)
        except Exception as e:
            print(f"  [capture] FAILED: {e}")
            summary.append((port, "FAIL-capture", None))
            continue

        parsed = parse(lines)
        mac = parsed.get("mac")
        if not mac:
            print("  [parse] no MAC in boot log — capturing unsuccessful")
            summary.append((port, "FAIL-no-mac", None))
            continue

        action, _ = upsert(cat, mac, port, parsed, sha, dirty)
        save_cat(cat)
        status = "PASS" if parsed.get("imu_status") == "ok" else "WARN"
        print(f"  [{action}] mac={mac}  ch={parsed.get('channel')}  "
              f"imu={parsed.get('imu_status')}  "
              f"seq={parsed.get('first_seq')}->{parsed.get('last_seq')}")
        summary.append((port, status, mac))

    print("\n=== summary ===")
    for port, status, mac in summary:
        print(f"  {status:5}  {port}  {mac or ''}")
    print(f"\ncatalog now has {len(cat.get('boards', []))} entries.")
    return 0 if all(s == "PASS" for _, s, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
