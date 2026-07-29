#!/usr/bin/env python3
"""Drive the rf-bench fleet over serial and record ESP-NOW radio measurements."""

import argparse
import json
import re
import statistics
from collections import defaultdict
import sys
import time
from pathlib import Path

import serial

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# esp_wifi_set_max_tx_power quarter-dBm request values spanning the documented
# mapping table; readback reveals each chip's actual quantization.
TXQ_LEVELS = [8, 20, 28, 34, 44, 52, 56, 60, 66, 72, 78, 80, 84]


class Board:
    def __init__(self, port, baud=115200, timeout=0.2):
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.mac = None
        self.chip = None
        self.label = port
        info = self.identify()
        self.mac = info["mac"]
        self.chip = info["chip"]
        self.idf = info.get("idf", "?")

    def send(self, line):
        self.ser.write((line + "\n").encode())
        self.ser.flush()

    def read_until(self, pred, timeout=15.0):
        lines = []
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            chunk = self.ser.read(4096)
            if chunk:
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    s = raw.decode(errors="replace").strip()
                    if not s:
                        continue
                    lines.append(s)
                    if pred(s):
                        return lines
        raise TimeoutError(f"{self.label}: timeout waiting; got {lines[-5:]}")

    def drain(self, settle=0.15):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if not self.ser.read(4096):
                time.sleep(settle)
                if not self.ser.in_waiting:
                    return

    def cmd(self, line, until, timeout=15.0, retries=2):
        last = None
        for attempt in range(retries + 1):
            self.drain()
            self.send(line)
            try:
                return self.read_until(lambda s: s.startswith(until), timeout)
            except TimeoutError as e:
                last = e
                print(f"  [resync] {self.label}: retry {attempt + 1} of '{line}'", file=sys.stderr)
        raise last

    def identify(self):
        for _ in range(3):
            try:
                lines = self.cmd("ID", "ID ", timeout=5.0)
                return kv(next(l for l in lines if l.startswith("ID ")))
            except (TimeoutError, StopIteration):
                continue
        raise RuntimeError(f"{self.port}: no ID response")

    def set_txq(self, q):
        line = self.cmd(f"TXP {q}", "TXP ")[-1]
        return kv(line)

    def add_peer(self, mac):
        self.cmd(f"PEER {mac}", "PEER ")

    def rx_clear(self):
        self.cmd("RXCLEAR", "RXCLEAR ok", timeout=5.0)

    def rx_stat(self):
        lines = self.cmd("RXSTAT", "RXSTAT end", timeout=10.0)
        out = {}
        for l in lines:
            if l.startswith("RX "):
                d = kv(l)
                out[d["src"]] = d
        return out

    def burst(self, dst, count=50, interval=5, length=32, echo=0):
        timeout = 10.0 + count * (interval + 110) / 1000.0
        lines = self.cmd(f"BURST {dst} {count} {interval} {length} {echo}", "BURST ", timeout)
        return kv(lines[-1])

    def scan(self):
        lines = self.cmd("SCAN", "SCAN end", timeout=40.0)
        return [kv(l) for l in lines if l.startswith("AP ")]

    def noise(self, ms=3000):
        lines = self.cmd(f"NOISE {ms}", "NOISE ", timeout=ms / 1000.0 + 15.0)
        return kv(lines[-1])

    def close(self):
        self.ser.close()


def kv(line):
    out = {}
    for m in re.finditer(r"(\w+)=(\S+)", line):
        k, v = m.group(1), m.group(2)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def connect(ports):
    boards = []
    for p in ports:
        b = Board(p)
        boards.append(b)
    c3n = e32n = 0
    for b in boards:
        if b.chip == "esp32c3":
            c3n += 1
            b.label = f"C3-{chr(ord('A') + c3n - 1)}"
        else:
            e32n += 1
            b.label = f"E32-{chr(ord('A') + e32n - 1)}"
    for b in boards:
        for o in boards:
            if o is not b:
                b.add_peer(o.mac)
    return boards


def save(name, payload):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = DATA_DIR / f"{name}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {path}")
    return path


def cmd_ids(boards, args):
    for b in boards:
        print(f"{b.label:6s} {b.mac}  {b.chip:8s} port={b.port} idf={b.idf}")


def cmd_txp(boards, args):
    rows = {}
    print(f"{'req':>5s} " + " ".join(f"{b.label:>8s}" for b in boards))
    for q in TXQ_LEVELS:
        line = f"{q:5d} "
        for b in boards:
            r = b.set_txq(q)
            rows.setdefault(b.label, {})[q] = r
            line += f" {r['readback']:3d}/{r['dbm']:5.2f}"
        print(line)
    print("\n(values are readback_quarter_dBm/dBm)")
    save("txp_quantization", {b.label: {"mac": b.mac, "chip": b.chip, "levels": rows[b.label]} for b in boards})


def cmd_scan(boards, args):
    results = {}
    samples = {b.label: defaultdict(list) for b in boards}
    for rep in range(args.repeat):
        for b in boards:
            for a in b.scan():
                samples[b.label][a["bssid"]].append(a["rssi"])
        print(f"  scan pass {rep + 1}/{args.repeat} done")
    for b in boards:
        aps = [{"bssid": k, "rssi": statistics.median(v), "n": len(v)}
               for k, v in samples[b.label].items()]
        results[b.label] = {"mac": b.mac, "chip": b.chip, "aps": aps}
        print(f"{b.label}: {len(aps)} distinct BSSIDs")

    # only BSSIDs seen in most passes by every board are trustworthy references
    common = None
    for b in boards:
        s = {a["bssid"] for a in results[b.label]["aps"]
             if a["n"] >= max(2, args.repeat // 2)}
        common = s if common is None else (common & s)

    print(f"\nBSSIDs heard by all {len(boards)} boards: {len(common)}")
    if common:
        hdr = f"{'bssid':<18s} " + " ".join(f"{b.label:>7s}" for b in boards)
        print(hdr)
        print("-" * len(hdr))
        deltas = {b.label: [] for b in boards}
        ref = boards[0].label
        for bssid in sorted(common):
            vals = {}
            for b in boards:
                a = next(x for x in results[b.label]["aps"] if x["bssid"] == bssid)
                vals[b.label] = a["rssi"]
            print(f"{bssid:<18s} " + " ".join(f"{vals[b.label]:7.1f}" for b in boards))
            for b in boards:
                deltas[b.label].append(vals[b.label] - vals[ref])
        print("-" * len(hdr))
        print(f"{'mean vs ' + ref:<18s} " + " ".join(
            f"{statistics.mean(deltas[b.label]):7.1f}" for b in boards))
        print(f"{'median vs ' + ref:<18s} " + " ".join(
            f"{statistics.median(deltas[b.label]):7.1f}" for b in boards))
        results["_common_bssids"] = sorted(common)
    save("ap_scan", results)


def cmd_noise(boards, args):
    out = {}
    for b in boards:
        n = b.noise(args.ms)
        out[b.label] = {"mac": b.mac, "chip": b.chip, **n}
        print(f"{b.label:6s} frames={n['frames']:6d} ({n['frames_per_s']:6.1f}/s) "
              f"noise_floor mean={n['nf_mean']:7.2f} min={n['nf_min']:4d} max={n['nf_max']:4d} "
              f"ambient_rssi_mean={n['rssi_mean']:7.2f}")
    save("noise_floor", out)


def cmd_matrix(boards, args):
    levels = [args.txq] if args.txq else TXQ_LEVELS
    run = {"boards": {b.label: {"mac": b.mac, "chip": b.chip} for b in boards},
           "count": args.count, "length": args.length, "results": []}

    for q in levels:
        readback = {}
        for b in boards:
            readback[b.label] = b.set_txq(q)["readback"]
        print(f"\n=== TXQ request {q} (readback {readback}) ===")
        hdr = f"{'tx':>6s} {'dst':>6s} {'ack%':>6s} {'lat_us':>8s} {'sd':>7s} " + \
              " ".join(f"{'rx@' + b.label:>10s}" for b in boards)
        print(hdr)
        for tx in boards:
            for dst in boards:
                if dst is tx:
                    continue
                for b in boards:
                    b.rx_clear()
                r = tx.burst(dst.mac, args.count, args.interval, args.length)
                rssi = {}
                for b in boards:
                    if b is tx:
                        continue
                    st = b.rx_stat().get(tx.mac, {})
                    rssi[b.label] = st
                ackpct = 100.0 * r["ack"] / r["n"] if r["n"] else 0
                cells = []
                for b in boards:
                    if b is tx:
                        cells.append(f"{'-':>10s}")
                    else:
                        st = rssi.get(b.label, {})
                        cells.append(f"{st.get('rssi_mean', float('nan')):6.1f}/{st.get('n', 0):<3d}"
                                     if st.get("n") else f"{'none':>10s}")
                print(f"{tx.label:>6s} {dst.label:>6s} {ackpct:6.1f} "
                      f"{r['ack_lat_mean_us']:8.1f} {r['ack_lat_sd_us']:7.1f} " + " ".join(cells))
                run["results"].append({
                    "txq_req": q, "txq_readback": readback[tx.label],
                    "tx": tx.label, "dst": dst.label, "burst": r,
                    "rx": {k: v for k, v in rssi.items()},
                })
    save("matrix" + (f"_txq{args.txq}" if args.txq else ""), run)


def cmd_rate(boards, args):
    out = []
    for b in boards:
        b.set_txq(args.txq)
    print(f"{'tx':>6s} {'dst':>6s} {'len':>4s} {'n':>5s} {'ack%':>6s} "
          f"{'pkt/s':>8s} {'kB/s':>7s} {'lat_us':>8s} {'sd':>7s} {'min':>7s} {'max':>8s}")
    for tx in boards:
        for dst in boards:
            if dst is tx:
                continue
            for length in args.lengths:
                r = tx.burst(dst.mac, args.count, 0, length)
                rate = r["n"] * 1e6 / r["elapsed_us"] if r["elapsed_us"] else 0
                print(f"{tx.label:>6s} {dst.label:>6s} {length:4d} {r['n']:5d} "
                      f"{100.0 * r['ack'] / r['n']:6.1f} {rate:8.1f} {rate * length / 1000:7.2f} "
                      f"{r['ack_lat_mean_us']:8.1f} {r['ack_lat_sd_us']:7.1f} "
                      f"{r['ack_lat_min_us']:7d} {r['ack_lat_max_us']:8d}")
                out.append({"tx": tx.label, "tx_chip": tx.chip, "dst": dst.label,
                            "dst_chip": dst.chip, "length": length, "pkt_per_s": rate, **r})
    save("throughput", {"boards": {b.label: {"mac": b.mac, "chip": b.chip} for b in boards},
                        "results": out})


def cmd_cliff(boards, args):
    """Fine TX-power sweep: for each board as transmitter, does its signal survive?"""
    run = {"boards": {b.label: {"mac": b.mac, "chip": b.chip} for b in boards},
           "safe_txq": args.safe, "results": []}
    others = {b.label: [o for o in boards if o is not b] for b in boards}
    for b in boards:
        b.set_txq(args.safe)
    print(f"all boards held at txq={args.safe} ({args.safe / 4.0:.2f} dBm); "
          f"only the transmitter under test is varied\n")
    print(f"{'txq':>4s} {'dBm':>6s}  " +
          "  ".join(f"{b.label + ' ack%/rssi':>18s}" for b in boards))
    for q in range(args.lo, args.hi + 1):
        row = f"{q:4d} {q / 4.0:6.2f}  "
        for tx in boards:
            rb = tx.set_txq(q)["readback"]
            dst = others[tx.label][0]
            for b in boards:
                b.rx_clear()
            r = tx.burst(dst.mac, args.count, 3, 32)
            heard = [b.rx_stat().get(tx.mac, {}) for b in others[tx.label]]
            heard = [h for h in heard if h.get("n")]
            best = max((h["rssi_mean"] for h in heard), default=float("nan"))
            ackp = 100.0 * r["ack"] / r["n"] if r["n"] else 0
            row += f"{ackp:7.1f}/{best:7.1f}(rb{rb:2d})  "
            run["results"].append({"txq": q, "tx": tx.label, "readback": rb,
                                   "ack_pct": ackp, "best_rssi": best,
                                   "n_heard_by": len(heard), "burst": r})
            tx.set_txq(args.safe)
        print(row)
    save("cliff", run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True,
                    help="comma-separated serial ports")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ids").set_defaults(fn=cmd_ids)
    sub.add_parser("txp").set_defaults(fn=cmd_txp)
    p = sub.add_parser("scan")
    p.add_argument("--repeat", type=int, default=1)
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("noise")
    p.add_argument("--ms", type=int, default=5000)
    p.set_defaults(fn=cmd_noise)

    p = sub.add_parser("matrix")
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--interval", type=int, default=4)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--txq", type=int, default=None)
    p.set_defaults(fn=cmd_matrix)

    p = sub.add_parser("rate")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 250])
    p.add_argument("--txq", type=int, default=52)
    p.set_defaults(fn=cmd_rate)

    p = sub.add_parser("cliff")
    p.add_argument("--lo", type=int, default=52)
    p.add_argument("--hi", type=int, default=72)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--safe", type=int, default=52)
    p.set_defaults(fn=cmd_cliff)

    args = ap.parse_args()
    boards = connect([p for p in args.ports.split(",") if p])
    try:
        args.fn(boards, args)
    finally:
        for b in boards:
            b.close()


if __name__ == "__main__":
    main()
