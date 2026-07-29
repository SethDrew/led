#!/usr/bin/env python3
"""Substitution test: measure a remote autonomous node against fixed bench references.

The remote node has no serial link. It broadcasts beacons at 10 Hz carrying, in the
payload, the RSSI stats it measured for each bench board. So one bench-side capture
recovers BOTH directions:
    node -> bench   from the bench's own promiscuous RSSI of the beacons
    bench -> node   from the report embedded in those beacons

Run this once with the C3 at the remote position and once with the classic ESP32 at
the IDENTICAL position. Geometry is then held constant and the difference between
runs is the chip (silicon + antenna together).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import serial


class Board:
    def __init__(self, port):
        self.port = port
        self.ser = serial.Serial(port, 115200, timeout=0.3)
        time.sleep(1.2)
        self.mac = None
        self.chip = None
        self.label = port

    def drain(self, settle=0.15):
        deadline = time.time() + 1.5
        while time.time() < deadline:
            if not self.ser.read(4096):
                time.sleep(settle)
                if not self.ser.in_waiting:
                    return

    def send(self, line):
        self.ser.write((line + "\n").encode())
        self.ser.flush()

    def read_until(self, pred, timeout=15.0):
        out, deadline = [], time.time() + timeout
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            s = raw.decode(errors="replace").strip()
            if not s:
                continue
            out.append(s)
            if pred(s):
                return out
        raise TimeoutError(f"{self.label}: timeout; got {out[-6:]}")

    def cmd(self, line, until, timeout=15.0, retries=2):
        last = None
        for attempt in range(retries + 1):
            self.drain()
            self.send(line)
            try:
                return self.read_until(lambda s: s.startswith(until), timeout)
            except TimeoutError as e:
                last = e
                print(f"  [resync] {self.label}: retry {attempt+1} of '{line}'", file=sys.stderr)
        raise last

    def identify(self):
        for ln in self.cmd("ID", "ID"):
            if ln.startswith("ID "):
                kv = dict(p.split("=", 1) for p in ln.split()[1:] if "=" in p)
                self.mac = kv["mac"].upper()
                self.chip = kv["chip"]
        return self.mac

    def set_txq(self, q, tries=15):
        rb = None
        for _ in range(tries):
            for ln in self.cmd(f"TXP {q}", "TXP"):
                if ln.startswith("TXP "):
                    kv = dict(p.split("=", 1) for p in ln.split()[1:] if "=" in p)
                    rb = int(kv["readback"])
            if rb == q:
                return rb
        return rb


def parse_kv(line, prefix):
    return dict(p.split("=", 1) for p in line[len(prefix):].split() if "=" in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True, help="comma-separated bench board ports")
    ap.add_argument("--node-mac", required=True, help="MAC of the remote node under test")
    ap.add_argument("--tag", required=True, help="phase label, e.g. 'c3-remote' or 'esp32-remote'")
    ap.add_argument("--txq", type=int, default=27)
    ap.add_argument("--listen", type=float, default=20.0, help="seconds of passive beacon capture")
    ap.add_argument("--count", type=int, default=200, help="packets per bench->node burst")
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()

    node_mac = args.node_mac.upper()
    boards = [Board(p) for p in args.ports.split(",")]
    for b in boards:
        b.identify()
        b.label = f"{b.chip}:{b.mac[-5:]}"
        print(f"bench {b.label}  port={b.port}")
        if b.mac == node_mac:
            sys.exit(f"ERROR: {b.port} IS the node under test; it must not be a bench reference")

    for b in boards:
        rb = b.set_txq(args.txq)
        print(f"  {b.label}: txq req={args.txq} readback={rb} ({rb/4:.2f} dBm)"
              + ("" if rb == args.txq else "   <-- MISMATCH, comparison is compromised"))
        for other in boards:
            if other is not b:
                b.cmd(f"PEER {other.mac}", "PEER")
        b.cmd(f"PEER {node_mac}", "PEER")

    print(f"\nlistening {args.listen:.0f}s for node beacons...")
    for b in boards:
        b.cmd("RXCLEAR", "RXCLEAR")
    time.sleep(args.listen)

    rec = {
        "tag": args.tag,
        "utc": datetime.now(timezone.utc).isoformat(),
        "node_mac": node_mac,
        "txq_req": args.txq,
        "listen_s": args.listen,
        "bench": {b.label: {"mac": b.mac, "chip": b.chip} for b in boards},
        "node_to_bench": {},
        "node_heard": {},
        "bench_to_node_burst": {},
    }

    for b in boards:
        for ln in b.cmd("RXSTAT", "RXSTAT end", timeout=20):
            if ln.startswith("RX src="):
                kv = parse_kv(ln, "RX ")
                if kv["src"].upper() == node_mac:
                    rec["node_to_bench"][b.label] = {
                        k: (float(v) if "." in v or "rssi" in k or "loss" in k else
                            (int(v) if v.lstrip("-").isdigit() else v))
                        for k, v in kv.items() if k != "src"
                    }
        for ln in b.cmd("NODESTAT", "NODESTAT end", timeout=20):
            if ln.startswith("NODE src="):
                kv = parse_kv(ln, "NODE ")
                if kv["src"].upper() == node_mac:
                    rec["node_meta"] = kv
            elif ln.startswith("NODEHEARD src="):
                kv = parse_kv(ln, "NODEHEARD ")
                if kv["src"].upper() == node_mac:
                    rec["node_heard"][kv["heard"].upper()] = kv

    before = dict(rec["node_heard"])

    print("\nbench -> node bursts...")
    for b in boards:
        for ln in b.cmd(f"BURST {node_mac} {args.count} {args.interval} 32 0",
                        "BURST", timeout=60):
            if ln.startswith("BURST "):
                rec["bench_to_node_burst"][b.label] = parse_kv(ln, "BURST ")

    time.sleep(4.0)
    for b in boards:
        for ln in b.cmd("NODESTAT", "NODESTAT end", timeout=20):
            if ln.startswith("NODEHEARD src="):
                kv = parse_kv(ln, "NODEHEARD ")
                if kv["src"].upper() == node_mac:
                    rec["node_heard"][kv["heard"].upper()] = kv

    rec["node_heard_before"] = before
    rec["burst_delta"] = {}
    for mac, aft in rec["node_heard"].items():
        pre = before.get(mac)
        dn = int(aft["n"]) - (int(pre["n"]) if pre else 0)
        ds = int(aft["rssi_sum"]) - (int(pre["rssi_sum"]) if pre else 0)
        rec["burst_delta"][mac] = {
            "n": dn,
            "rssi_mean": round(ds / dn, 2) if dn else None,
            "rssi_min": int(aft["rssi_min"]),
            "rssi_max": int(aft["rssi_max"]),
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = Path(__file__).resolve().parent.parent / "data" / f"node_{args.tag}_{ts}.json"
    out.write_text(json.dumps(rec, indent=2))

    print(f"\n{'='*66}\nNODE -> BENCH  (beacon RSSI measured at each bench board)")
    for lab, st in rec["node_to_bench"].items():
        print(f"  {lab:16s} n={st.get('n'):>5}  rssi={st.get('rssi_mean'):>7}  "
              f"min={st.get('rssi_min')} max={st.get('rssi_max')}  loss={st.get('loss_pct')}%")
    if not rec["node_to_bench"]:
        print("  (nothing heard from the node)")

    print(f"\nBENCH -> NODE  (RSSI the node measured during the burst)")
    macs = {b.mac: b.label for b in boards}
    for mac, kv in rec["burst_delta"].items():
        print(f"  {macs.get(mac, mac):16s} n={kv['n']:>5}  rssi={kv['rssi_mean']}  "
              f"min={kv['rssi_min']} max={kv['rssi_max']}")
    if not any(v["n"] for v in rec["burst_delta"].values()):
        print("  (node reported no receptions during the burst)")

    print(f"\nBENCH -> NODE unicast ACK")
    for lab, kv in rec["bench_to_node_burst"].items():
        n, a = int(kv.get("n", 0)), int(kv.get("ack", 0))
        print(f"  {lab:16s} {a}/{n} = {100*a/max(n,1):.1f}%  "
              f"lat_min={kv.get('ack_lat_min_us')}us")

    if "node_meta" in rec:
        m = rec["node_meta"]
        print(f"\nnode: chip={m.get('chip')} txq={m.get('txq')} ({m.get('dbm')} dBm) "
              f"uptime={m.get('uptime_s')}s window={m.get('window')}")
        if int(m.get("txq", -1)) != args.txq:
            print(f"  *** NODE TX POWER IS {m.get('txq')}, NOT {args.txq} — "
                  f"this run is NOT comparable to a run at {args.txq}")
    else:
        print("\n*** no node metadata received — node not heard at all")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
