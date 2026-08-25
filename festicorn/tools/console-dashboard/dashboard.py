#!/usr/bin/env python3
"""
Axis Mundi console dashboard — one server, every console.

Supersedes clicky-pots/dashboard.py and kettle-knob/dashboard.py.

Run:   .venv/bin/python festicorn/tools/console-dashboard/dashboard.py
Open:  http://localhost:8080/

Consoles on usbmodem* ports are auto-discovered by probing for a known
signature. usbserial* ports are NOT swept: opening one resets the attached
ESP32 even with DTR/RTS deasserted, and the sculpture receiver lives on one.
Reach a console there with --port clicky=/dev/cu.usbserial-2120 (or --include
to let it be probed) — both reset that board once, deliberately.

The ESP-NOW bridge decoder is wired but inert until the bridge board is
reflashed to channel 1.

HTTP API (stable — other local consumers may poll it):
  /api/config              roster, labels, regions
  /api/state               full live state, all roles
  /api/events?since=<id>   append-only event log, monotonic cursor
  /api/duck/stream?since=  every duck rms sample since <i>, for the duck scope
                           (the /api/state poll would alias the 25 Hz stream)
  /api/console/<role>      one console, flat — same shape the retired
                           per-console dashboards served at /api/state
"""

import argparse
import glob
import json
import os
import socket
import struct
import threading
import time
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "ui.html")

OFFLINE_AFTER = 3.0
RATE_WINDOW = 2.0
SPIN_WINDOW = 0.35
TURN_COALESCE = 0.4

CLICKY_MAGIC = 0xC11C
CLICKY_VER = 1
KETTLE_MAGIC = 0xEC11
KETTLE_VER = 1
DUCK_MAGIC = 0xD0CC
DUCK_VER = 1
DUCK_RMS_FS = 200000.0
DUCK_STREAM_KEEP = 6000

KETTLE_COUNTS_PER_DETENT = 2
KETTLE_DETENTS_PER_REV = 30

KETTLE_BTNS = [
    {"label": "knob", "sub": "knob push · GPIO 21 · sandbox: ease all spin "
                            "integrals home", "region": None},
    {"label": "flywheel", "sub": "top · GPIO 9 · toggle · spin keeps coasting",
     "region": "top"},
    {"label": "twist", "sub": "left · GPIO 7 · hold+spin · fans rings out of phase",
     "region": "left"},
    {"label": "counter", "sub": "right · GPIO 6 · toggle · alternate rings reverse",
     "region": "right"},
    {"label": "crackle", "sub": "bottom · GPIO 5 · hold · sparks pinned in place",
     "region": "bottom"},
    {"label": "aux-mom", "sub": "new · GPIO 3 · momentary · unassigned", "region": None},
    {"label": "keep-temp", "sub": "new · GPIO 4 · toggle (latched) · unassigned",
     "region": None},
]
REGIONS = ["top", "left", "right", "bottom"]

CLICKY_POTS = ["P0", "P1", "P2", "P3 ⇄", "P4", "P5"]
CLICKY_BTNS = ["25", "26", "14", "27", "16", "13", "17 tgl"]
CLICKY_BRIGHTNESS_POT = 5

ROSTER = [
    {"role": "clicky", "title": "CLICKY-POTS", "baud": 230400,
     "detail": "6 push-pull pots + 6 buttons"},
    {"role": "kettle", "title": "KETTLE-KNOB", "baud": 115200,
     "detail": "30-detent encoder + 7 buttons"},
    {"role": "duck", "title": "DUCK-SENDER", "baud": 115200,
     "detail": "mic energy waterfall · tilt recolors the water"},
]
ROLES = [r["role"] for r in ROSTER]
BAUD_BY_ROLE = {r["role"]: r["baud"] for r in ROSTER}
BRIDGE_BAUD = 460800
PROBE_BAUDS = [230400, 115200, 460800]


def open_serial(port, baud, timeout):
    # Open with the control lines left alone. Forcing DTR/RTS low resets both
    # board classes here: the C3's USB-serial-JTAG implements the reset sequence
    # in hardware, and a CH340 resets on open regardless. Non-console boards are
    # kept safe by not opening their port at all — see discover().
    return serial.Serial(port, baud, timeout=timeout)


def blank(role):
    return {
        "role": role,
        "online": False,
        "source": None,
        "ts": 0.0,
        "packets": 0,
        "rate": 0.0,
        "reboots": 0,
        "seq_gaps": 0,
        "up_ms": None,
        "state": {},
        "derived": {},
    }


class Registry:
    def __init__(self):
        self.lock = threading.Lock()
        self.roles = {r: blank(r) for r in ROLES}
        self.events = deque(maxlen=500)
        self.next_id = 1
        self._stamps = {r: deque() for r in ROLES}
        self._prev = {r: None for r in ROLES}
        self._prev_seq = {r: None for r in ROLES}
        self._enc_hist = deque()
        self._waves = {r: {"count": 0, "ts": 0.0} for r in REGIONS}
        self._turn_from = None
        self._turn_last = 0.0
        self._duck_stream = deque(maxlen=DUCK_STREAM_KEEP)
        self._duck_n = 0

    def log(self, role, text, kind="info"):
        self.events.append({
            "id": self.next_id,
            "t": time.time(),
            "role": role,
            "kind": kind,
            "text": text,
        })
        self.next_id += 1

    def attach(self, role, source):
        with self.lock:
            rec = self.roles[role]
            if rec["source"] != source:
                rec["source"] = source
                self.log(role, f"source attached — {source}", "source")

    def detach(self, role, why):
        with self.lock:
            rec = self.roles[role]
            rec["online"] = False
            rec["source"] = None
            self.log(role, f"source lost — {why}", "source")

    def ingest(self, role, fields, source, seq=None, up_ms=None):
        now = time.time()
        with self.lock:
            rec = self.roles[role]
            rec["source"] = source
            prev = self._prev[role]

            if up_ms is not None:
                if rec["up_ms"] is not None and up_ms < rec["up_ms"]:
                    rec["reboots"] += 1
                    self._prev[role] = None
                    self._prev_seq[role] = None
                    self._enc_hist.clear()
                    self._turn_from = None
                    prev = None
                    self.log(role, "*** BOARD REBOOTED ***", "reboot")
                rec["up_ms"] = up_ms

            if seq is not None:
                psq = self._prev_seq[role]
                if psq is not None:
                    gap = (seq - psq) & 0xFF
                    if gap > 1:
                        rec["seq_gaps"] += gap - 1
                self._prev_seq[role] = seq

            stamps = self._stamps[role]
            stamps.append(now)
            while stamps and now - stamps[0] > RATE_WINDOW:
                stamps.popleft()
            rec["rate"] = len(stamps) / RATE_WINDOW

            rec["packets"] += 1
            rec["ts"] = now
            rec["online"] = True
            rec["state"] = fields

            if role == "kettle":
                rec["derived"] = self._kettle_derived(fields, prev, now)
            elif role == "clicky":
                rec["derived"] = self._clicky_derived(fields, prev, now)
            elif role == "duck":
                rec["derived"] = self._duck_derived(fields)
                if "rms_mean" in fields:
                    self._duck_n += 1
                    self._duck_stream.append({
                        "i": self._duck_n,
                        "t": now,
                        "m": fields["rms_mean"],
                        "x": fields["rms_max"],
                        "s": seq,
                    })

            self._prev[role] = fields

    def _duck_derived(self, s):
        if "rms_mean" not in s:
            return {}
        dec = lambda b: (b / 255.0) ** 2 * DUCK_RMS_FS
        return {"rms_mean": dec(s["rms_mean"]), "rms_max": dec(s["rms_max"]),
                "imu": not (s.get("flags", 0) & 0x01)}

    def _flush_turn(self, det, rev, now):
        delta = det - self._turn_from
        if abs(delta) >= 0.5:
            d = "CW" if delta > 0 else "CCW"
            self.log("kettle",
                     f"turn {d} {abs(delta):.1f} det → {det:+.1f} ({rev:+.2f} rev)", "turn")
        self._turn_from = None
        self._turn_last = now

    def _kettle_derived(self, s, prev, now):
        c = s.get("c", 0)
        det = c / KETTLE_COUNTS_PER_DETENT
        rev = det / KETTLE_DETENTS_PER_REV

        self._enc_hist.append((now, det))
        while self._enc_hist and now - self._enc_hist[0][0] > SPIN_WINDOW:
            self._enc_hist.popleft()
        spin = 0.0
        if len(self._enc_hist) >= 2:
            t0, d0 = self._enc_hist[0]
            dt = now - t0
            if dt > 1e-3:
                spin = (det - d0) / dt

        btn = s.get("btn", [0] * 5)
        counts = s.get("n", [0] * 5)
        for i, meta in enumerate(KETTLE_BTNS):
            if meta["region"] and i < len(counts):
                self._waves[meta["region"]]["count"] = counts[i]

        if prev:
            pb = prev.get("btn", [0] * 5)
            n = counts
            if s.get("c") != prev.get("c"):
                if self._turn_from is None:
                    self._turn_from = prev.get("c", c) / KETTLE_COUNTS_PER_DETENT
                if now - self._turn_last >= TURN_COALESCE:
                    self._flush_turn(det, rev, now)
            elif self._turn_from is not None and now - self._turn_last >= TURN_COALESCE:
                self._flush_turn(det, rev, now)
            for i, meta in enumerate(KETTLE_BTNS):
                if btn[i] and not pb[i]:
                    self.log("kettle", f"{meta['label']} press #{n[i]}", "press")
                    if meta["region"]:
                        self._waves[meta["region"]]["ts"] = now
                elif not btn[i] and pb[i]:
                    self.log("kettle", f"{meta['label']} release", "release")

        return {
            "detents": det,
            "revs": rev,
            "spin": spin,
            "hold": bool(btn[0]),
            "dir": "CW" if spin > 0.05 else ("CCW" if spin < -0.05 else "—"),
        }

    def _clicky_derived(self, s, prev, now):
        p = s.get("p", [0.0] * 6)
        pull = s.get("pull", [0] * 6)
        btn = s.get("btn", [0] * 7)
        if prev:
            for i in range(min(len(btn), len(prev.get("btn", [])))):
                if btn[i] and not prev["btn"][i]:
                    self.log("clicky", f"btn {CLICKY_BTNS[i]} press", "press")
                elif not btn[i] and prev["btn"][i]:
                    self.log("clicky", f"btn {CLICKY_BTNS[i]} release", "release")
            for i in range(min(len(pull), len(prev.get("pull", [])))):
                if pull[i] != prev["pull"][i]:
                    verb = "PULLED" if pull[i] else "pushed"
                    self.log("clicky", f"pot {i} {verb}", "pull")
        return {
            "brightness": p[CLICKY_BRIGHTNESS_POT] if len(p) > CLICKY_BRIGHTNESS_POT else 0.0,
            "pulled": [i for i, v in enumerate(pull) if v],
        }

    def snapshot(self):
        now = time.time()
        with self.lock:
            for rec in self.roles.values():
                if rec["online"] and now - rec["ts"] > OFFLINE_AFTER:
                    rec["online"] = False
                    rec["rate"] = 0.0
            waves = {
                r: {"count": w["count"], "age": (now - w["ts"]) if w["ts"] else None}
                for r, w in self._waves.items()
            }
            return {
                "now": now,
                "roles": {k: dict(v) for k, v in self.roles.items()},
                "sculpture": {
                    "waves": waves,
                    "spin": self.roles["kettle"]["derived"].get("spin", 0.0),
                    "hold": self.roles["kettle"]["derived"].get("hold", False),
                    "brightness": self.roles["clicky"]["derived"].get("brightness"),
                    "telemetry": None,
                },
            }

    def flat(self, role):
        now = time.time()
        with self.lock:
            rec = self.roles[role]
            online = rec["online"] and now - rec["ts"] <= OFFLINE_AFTER
            out = dict(rec["state"])
            out.update({
                "role": role,
                "connected": online,
                "rate": rec["rate"],
                "reboots": rec["reboots"],
                "ts": rec["ts"],
                "derived": dict(rec["derived"]),
            })
            return out

    def duck_stream(self, cursor):
        with self.lock:
            samples = [s for s in self._duck_stream if s["i"] > cursor]
            # A client that has fallen off the back of the ring must be told, or
            # it would feed the WASM probe a gap as if it were continuous time.
            oldest = self._duck_stream[0]["i"] if self._duck_stream else self._duck_n + 1
            return {"cursor": self._duck_n, "gap": cursor > 0 and oldest > cursor + 1,
                    "samples": samples}

    def since(self, cursor):
        with self.lock:
            evs = [e for e in self.events if e["id"] > cursor]
            return {"cursor": self.next_id - 1, "events": evs}


REG = Registry()


def parse_kettle_json(d):
    if "c" not in d or "up" not in d:
        return None
    return d, d.get("seq"), d.get("up")


def parse_clicky_json(d):
    if "p" not in d or "wraw" not in d:
        return None
    return d, None, d.get("ms")


JSON_PARSERS = {"kettle": parse_kettle_json, "clicky": parse_clicky_json}


def classify_json(d):
    for role, fn in JSON_PARSERS.items():
        if fn(d):
            return role
    return None


def decode_espnow(payload):
    n = len(payload)
    if n == 13:
        magic, ver, seq, btnbits, enc, upms = struct.unpack("<HBBBiI", payload)
        if magic != KETTLE_MAGIC or ver != KETTLE_VER:
            return None
        btn = [(btnbits >> i) & 1 for i in range(5)]
        fields = {"c": enc, "a": 1, "b": 1, "btn": btn,
                  "n": [0] * 5, "up": upms, "seq": seq}
        return "kettle", fields, seq, upms
    if n == 18:
        vals = struct.unpack("<HBBBB6H", payload)
        magic, ver, seq, pullbits, btnbits = vals[:5]
        if magic != CLICKY_MAGIC or ver != CLICKY_VER:
            return None
        pos = vals[5:]
        fields = {
            "ms": 0,
            "p": [v / 10000.0 for v in pos],
            "pull": [(pullbits >> i) & 1 for i in range(6)],
            "btn": [(btnbits >> i) & 1 for i in range(7)],
            "raw": [0] * 6,
            "wraw": [0] * 6,
        }
        return "clicky", fields, seq, None
    if n == 14:
        magic, ver, seq, rms_mean, rms_max, ax, ay, az, flags, upms = \
            struct.unpack("<HBBBB3bBI", payload)
        if magic != DUCK_MAGIC or ver != DUCK_VER:
            return None
        fields = {"seq": seq, "rms_mean": rms_mean, "rms_max": rms_max,
                  "ax": ax, "ay": ay, "az": az, "flags": flags, "up": upms}
        return "duck", fields, seq, upms
    if n == 16:
        seq = struct.unpack("<H", payload[:2])[0]
        ax, ay, az = struct.unpack("<3b", payload[2:5])
        amag_max, amag_mean, gmag_max, gmag_mean = struct.unpack("<4B", payload[11:15])
        fields = {"seq": seq, "ax": ax, "ay": ay, "az": az,
                  "amag_max": amag_max, "amag_mean": amag_mean,
                  "gmag_max": gmag_max, "gmag_mean": gmag_mean}
        return "duck", fields, seq & 0xFF, None
    return None


class SerialJsonSource(threading.Thread):
    daemon = True

    def __init__(self, port, role, baud):
        super().__init__(name=f"serial-{role}")
        self.port = port
        self.role = role
        self.baud = baud

    def run(self):
        tag = f"serial {os.path.basename(self.port)}"
        while True:
            try:
                ser = open_serial(self.port, self.baud, 1)
                REG.attach(self.role, tag)
                print(f"[{self.role}] {self.port} @ {self.baud}")
                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parser = JSON_PARSERS.get(self.role)
                    if parser is None:
                        continue
                    parsed = parser(d)
                    if not parsed:
                        continue
                    fields, seq, up = parsed
                    REG.ingest(self.role, fields, tag, seq=seq, up_ms=up)
            except (serial.SerialException, OSError) as e:
                REG.detach(self.role, str(e))
                print(f"[{self.role}] lost: {e} — retry in 2s")
                time.sleep(2)


class BridgeSource(threading.Thread):
    daemon = True

    def __init__(self, port, baud=BRIDGE_BAUD):
        super().__init__(name="bridge")
        self.port = port
        self.baud = baud

    def run(self):
        tag = f"bridge {os.path.basename(self.port)}"
        while True:
            try:
                ser = open_serial(self.port, self.baud, 1)
                print(f"[bridge] {self.port} @ {self.baud}")
                for payload in self._frames(ser):
                    got = decode_espnow(payload)
                    if not got:
                        continue
                    role, fields, seq, up = got
                    REG.ingest(role, fields, tag, seq=seq, up_ms=up)
            except (serial.SerialException, OSError) as e:
                print(f"[bridge] lost: {e} — retry in 2s")
                time.sleep(2)

    def _frames(self, ser):
        while True:
            b = ser.read(1)
            if not b or b[0] != 0xA5:
                continue
            b = ser.read(1)
            if not b or b[0] != 0x5A:
                continue
            b = ser.read(1)
            if not b:
                continue
            n = b[0]
            if n == 0 or n > 250:
                continue
            payload = ser.read(n)
            if len(payload) != n:
                continue
            chk = ser.read(1)
            if not chk:
                continue
            x = n
            for v in payload:
                x ^= v
            if x == chk[0]:
                yield payload


class ReplayDuckSource(threading.Thread):
    """Replay a duck-scope export (or any {"samples":[{t,m,x}]} file) as if the
    bridge were delivering it — same 14-byte encode/decode path as the wire, so
    the panel can be tuned and demoed with no hardware attached."""

    daemon = True

    def __init__(self, path, loop=True):
        super().__init__(name="replay-duck")
        self.path = path
        self.loop = loop

    def run(self):
        with open(self.path) as f:
            samples = json.load(f)["samples"]
        if not samples:
            print(f"[replay] {self.path} — no samples")
            return
        tag = f"replay {os.path.basename(self.path)}"
        print(f"[replay] {self.path} — {len(samples)} duck samples")
        seq = 0
        up = 0
        upBase = 0
        while True:
            t0 = samples[0]["t"]
            start = time.time()
            for s in samples:
                wait = (s["t"] - t0) - (time.time() - start)
                if wait > 0:
                    time.sleep(wait)
                up = int((s["t"] - t0) * 1000) + upBase
                payload = struct.pack("<HBBBB3bBI", DUCK_MAGIC, DUCK_VER, seq & 0xFF,
                                      int(s["m"]) & 0xFF, int(s.get("x", s["m"])) & 0xFF,
                                      0, 0, 0, 0x01, up)
                seq += 1
                got = decode_espnow(payload)
                if got:
                    role, fields, sq, upms = got
                    REG.ingest(role, fields, tag, seq=sq, up_ms=upms)
            if not self.loop:
                return
            upBase = up + 40


def probe(port, window=1.5):
    for baud in PROBE_BAUDS:
        try:
            ser = open_serial(port, baud, 0.3)
        except (serial.SerialException, OSError):
            return None
        deadline = time.time() + window
        try:
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                if raw.startswith(b"\xa5\x5a"):
                    return "bridge", baud
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = classify_json(d)
                if role:
                    return role, baud
        finally:
            ser.close()
    return None


def discover(exclude, include):
    found = {}
    for port in sorted(glob.glob("/dev/cu.usb*")):
        if port in exclude:
            print(f"[probe] {port} — excluded")
            continue
        # Opening a usbserial bridge resets the attached ESP32 even with DTR/RTS
        # deasserted (verified on CH340 + macOS), so never sweep one blind — the
        # sculpture receiver lives on one and a probe would reboot the show.
        if "usbserial" in port and port not in include:
            print(f"[probe] {port} — skipped (usbserial resets on open; "
                  f"pass --include {port} to probe it)")
            continue
        print(f"[probe] {port} …")
        got = probe(port)
        if not got:
            print(f"[probe] {port} — no known signature")
            continue
        role, baud = got
        print(f"[probe] {port} → {role} @ {baud}")
        found[role] = (port, baud)
    return found


def json_response(handler, obj):
    body = json.dumps(obj).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


VIZ_DIR = os.path.join(HERE, "viz")
VIZ_MIME = {
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".wasm": "application/wasm",
    ".json": "application/json",
}

CONFIG = {
    "roster": ROSTER,
    "kettle_btns": KETTLE_BTNS,
    "regions": REGIONS,
    "clicky_pots": CLICKY_POTS,
    "clicky_btns": CLICKY_BTNS,
    "brightness_pot": CLICKY_BRIGHTNESS_POT,
    "offline_after": OFFLINE_AFTER,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/state":
            json_response(self, REG.snapshot())
        elif u.path == "/api/events":
            q = parse_qs(u.query)
            try:
                cursor = int(q.get("since", ["0"])[0])
            except ValueError:
                cursor = 0
            json_response(self, REG.since(cursor))
        elif u.path == "/api/duck/stream":
            q = parse_qs(u.query)
            try:
                cursor = int(q.get("since", ["0"])[0])
            except ValueError:
                cursor = 0
            json_response(self, REG.duck_stream(cursor))
        elif u.path.startswith("/api/console/"):
            role = u.path.rsplit("/", 1)[-1]
            if role not in ROLES:
                self.send_error(404)
                return
            json_response(self, REG.flat(role))
        elif u.path == "/api/config":
            json_response(self, CONFIG)
        elif u.path in ("/", "/index.html"):
            with open(UI_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path.startswith("/viz/"):
            path = os.path.realpath(os.path.join(VIZ_DIR, u.path[len("/viz/"):]))
            if not path.startswith(VIZ_DIR + os.sep) or not os.path.isfile(path):
                self.send_error(404)
                return
            with open(path, "rb") as f:
                body = f.read()
            ext = os.path.splitext(path)[1]
            self.send_response(200)
            self.send_header("Content-Type", VIZ_MIME.get(ext, "application/octet-stream"))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Axis Mundi console dashboard")
    ap.add_argument("--port", action="append", default=[], metavar="ROLE=PATH",
                    help="pin a source, e.g. kettle=/dev/cu.usbmodem101 or bridge=/dev/...")
    ap.add_argument("--http", type=int, default=8080)
    ap.add_argument("--exclude", action="append", default=[], metavar="PATH",
                    help="never probe this port (repeatable); non-console boards "
                         "such as the sculpture receiver")
    ap.add_argument("--include", action="append", default=[], metavar="PATH",
                    help="probe this usbserial port anyway (repeatable) — it will "
                         "reset the attached board once, so name it deliberately")
    ap.add_argument("--no-discover", action="store_true",
                    help="use only --port pins")
    ap.add_argument("--replay-duck", metavar="PATH",
                    help="replay a duck-scope export as the duck source (no hardware)")
    ap.add_argument("--replay-once", action="store_true",
                    help="play --replay-duck through once instead of looping")
    args = ap.parse_args()

    pinned = {}
    for spec in args.port:
        if "=" not in spec:
            ap.error(f"--port needs ROLE=PATH, got {spec!r}")
        role, path = spec.split("=", 1)
        if role not in ROLES and role != "bridge":
            ap.error(f"unknown role {role!r}")
        pinned[role] = (path, BRIDGE_BAUD if role == "bridge" else BAUD_BY_ROLE[role])

    sources = dict(pinned)
    if not args.no_discover:
        skip = {p for p, _ in pinned.values()} | set(args.exclude)
        for role, val in discover(skip, set(args.include)).items():
            sources.setdefault(role, val)

    for role, (path, baud) in sources.items():
        if role == "bridge":
            BridgeSource(path, baud).start()
        else:
            SerialJsonSource(path, role, baud).start()

    if args.replay_duck:
        ReplayDuckSource(args.replay_duck, loop=not args.replay_once).start()

    if not sources and not args.replay_duck:
        print("[warn] no sources — dashboard will show every console offline")

    # IPv4-only on purpose: the macOS application firewall silently drops IPv6
    # TCP to python, so a dual-stack bind makes Chrome hang on localhost (::1
    # never gets a RST to trigger IPv4 fallback).
    class Server(ThreadingHTTPServer):
        daemon_threads = True

    httpd = Server(("0.0.0.0", args.http), Handler)
    print(f"Dashboard: http://localhost:{args.http}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
