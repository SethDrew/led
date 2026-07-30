#!/usr/bin/env python3
"""
Board bench — click-around USB debug UI for the festicorn board fleet.

Run:   .venv/bin/python festicorn/tools/board-bench/server.py
Open:  http://localhost:8090/

Lists USB serial ports, identifies boards by MAC against catalog/boards.yaml,
shows wiring/hazard notes, and offers a per-port serial monitor with send,
plus reset. Every serial action is an explicit click: opening a port resets
the attached ESP32 (CH340 resets on open; C3 USB-JTAG likewise), so nothing
here touches a port on its own.
"""

import argparse
import glob
import json
import os
import plistlib
import re
import subprocess
import threading
import time
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import serial
from serial.tools import list_ports
import yaml

MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")

HERE = os.path.dirname(os.path.abspath(__file__))
LED_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CATALOG_PATH = os.path.join(LED_ROOT, "festicorn", "catalog", "boards.yaml")
ESPTOOL = os.path.join(LED_ROOT, ".venv", "bin", "esptool.py")
UI_PATH = os.path.join(HERE, "ui.html")

BAUDS = [115200, 230400, 460800, 921600]
MONITOR_BUFFER = 4000


def load_catalog():
    with open(CATALOG_PATH) as f:
        doc = yaml.safe_load(f)
    by_mac = {}
    for b in doc.get("boards", []):
        if b.get("mac"):
            by_mac[b["mac"].upper()] = b
    return {
        "by_mac": by_mac,
        "boards": doc.get("boards", []),
        "roles": doc.get("firmware_by_role", {}),
        "mtime": os.path.getmtime(CATALOG_PATH),
    }


def loc_key(loc):
    bus = loc >> 24
    chain = []
    for shift in range(20, -1, -4):
        d = (loc >> shift) & 0xF
        if d == 0:
            break
        chain.append(str(d))
    return f"{bus}-{'.'.join(chain)}" if chain else str(bus)


def serial_ports_by_loc():
    out = {}
    for p in list_ports.comports():
        if "usb" in p.device.lower() and p.location:
            out[p.location.split(":")[0]] = p
    return out


def usb_tree():
    r = subprocess.run(["ioreg", "-a", "-p", "IOUSB", "-l", "-w0"],
                       capture_output=True, timeout=15)
    root = plistlib.loads(r.stdout)
    ttys = serial_ports_by_loc()

    def node(entry):
        loc = entry.get("locationID")
        n = {
            "name": entry.get("USB Product Name") or entry.get("IORegistryEntryName", "?"),
            "vendor": entry.get("USB Vendor Name"),
            "sn": entry.get("USB Serial Number"),
            "loc": loc_key(loc) if isinstance(loc, int) else None,
            "tty": None,
            "children": [node(c) for c in entry.get("IORegistryEntryChildren", [])],
        }
        p = ttys.get(n["loc"])
        if p:
            n["tty"] = p.device
        return n

    buses = []
    for bus in root.get("IORegistryEntryChildren", []):
        kids = bus.get("IORegistryEntryChildren", [])
        if kids:
            buses.append({"name": bus.get("IORegistryEntryName", "bus"),
                          "vendor": None, "sn": None, "loc": None, "tty": None,
                          "children": [node(c) for c in kids]})
    return buses


class Bench:
    def __init__(self):
        self.lock = threading.Lock()
        self.catalog = load_catalog()
        self.identified = {}
        self.monitors = {}
        self.log = deque(maxlen=400)
        self.log_id = 1

    def event(self, port, text, kind="info"):
        with self.lock:
            self.log.append({
                "id": self.log_id, "t": time.time(),
                "port": os.path.basename(port) if port else "",
                "kind": kind, "text": text,
            })
            self.log_id += 1

    def events_since(self, cursor):
        with self.lock:
            return {"cursor": self.log_id - 1,
                    "events": [e for e in self.log if e["id"] > cursor]}

    def refresh_catalog(self):
        if os.path.getmtime(CATALOG_PATH) != self.catalog["mtime"]:
            self.catalog = load_catalog()
            self.event(None, "boards.yaml changed — catalog reloaded", "catalog")

    def ports(self):
        self.refresh_catalog()
        # ESP32-C3/S3 native-USB ports report the chip MAC as the USB serial
        # number, so those boards are identified with no port open, no reset.
        usb_sn = {p.device: p.serial_number for p in list_ports.comports()
                  if p.serial_number and MAC_RE.match(p.serial_number)}
        out = []
        for path in sorted(glob.glob("/dev/cu.usb*")):
            mon = self.monitors.get(path)
            ident = self.identified.get(path)
            if not ident and path in usb_sn:
                ident = {"mac": usb_sn[path].upper(), "chip": None,
                         "method": "usb-sn", "ts": None}
            rec = {
                "port": path,
                "monitoring": mon.baud if mon and mon.alive() else None,
                "identified": ident,
                "board": None,
            }
            if ident and ident.get("mac"):
                rec["board"] = self.board_for_mac(ident["mac"])
            out.append(rec)
        return out

    def topology(self):
        try:
            tree = usb_tree()
        except (subprocess.SubprocessError, plistlib.InvalidFileException, OSError) as e:
            return {"error": str(e), "tree": []}
        boards = {p["port"]: p for p in self.ports()}

        def annotate(n):
            if n["tty"] and n["tty"] in boards:
                b = boards[n["tty"]]
                if b["identified"]:
                    n["mac"] = b["identified"]["mac"]
                if b["board"] and b["board"].get("known"):
                    n["board_id"] = b["board"]["entry"].get("id") or b["board"]["entry"].get("role")
                elif b["board"]:
                    n["board_id"] = "not in catalog"
                n["monitoring"] = b["monitoring"]
            for c in n["children"]:
                annotate(c)

        for bus in tree:
            annotate(bus)
        return {"tree": tree}

    def board_for_mac(self, mac):
        b = self.catalog["by_mac"].get(mac.upper())
        if not b:
            return {"known": False, "mac": mac}
        role = b.get("role")
        return {"known": True, "mac": mac, "entry": b,
                "role_info": self.catalog["roles"].get(role)}

    def identify(self, port):
        self.stop_monitor(port, quiet=True)
        self.event(port, "identify — running esptool read_mac (resets board)")
        try:
            r = subprocess.run(
                [ESPTOOL, "--port", port, "read_mac"],
                capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            self.event(port, "identify timed out", "error")
            return {"error": "esptool timed out (board busy or not in sync?)"}
        text = r.stdout + r.stderr
        mac = None
        m = re.search(r"MAC:\s+([0-9a-fA-F:]{17})", text)
        if m:
            mac = m.group(1).upper()
        chip = None
        m = re.search(r"(?:Chip is|Detecting chip type\.+\s*)(.+)", text)
        if m:
            chip = m.group(1).strip()
        if not mac:
            tail = "\n".join(text.strip().splitlines()[-4:])
            self.event(port, f"identify failed: {tail}", "error")
            return {"error": f"no MAC in esptool output:\n{tail}"}
        ident = {"mac": mac, "chip": chip, "method": "esptool", "ts": time.time()}
        self.identified[port] = ident
        board = self.board_for_mac(mac)
        name = board["entry"].get("id") or board["entry"].get("role") if board["known"] else "NOT IN CATALOG"
        self.event(port, f"identified {mac} → {name}", "identify")
        return {"identified": ident, "board": board}

    def start_monitor(self, port, baud):
        self.stop_monitor(port, quiet=True)
        mon = Monitor(port, baud, self)
        try:
            mon.open()
        except (serial.SerialException, OSError) as e:
            self.event(port, f"monitor open failed: {e}", "error")
            return {"error": str(e)}
        self.monitors[port] = mon
        mon.start()
        self.event(port, f"monitor started @ {baud} (open resets board)", "monitor")
        return {"ok": True}

    def stop_monitor(self, port, quiet=False):
        mon = self.monitors.pop(port, None)
        if mon:
            mon.stop()
            if not quiet:
                self.event(port, "monitor stopped", "monitor")
        return {"ok": True}

    def read_monitor(self, port, since):
        mon = self.monitors.get(port)
        if not mon:
            return {"cursor": since, "lines": [], "alive": False}
        return mon.since(since)

    def send(self, port, text):
        mon = self.monitors.get(port)
        if not mon or not mon.alive():
            return {"error": "no monitor open on this port"}
        mon.send(text)
        self.event(port, f"sent: {text!r}", "send")
        return {"ok": True}

    def reset(self, port):
        mon = self.monitors.get(port)
        try:
            if mon and mon.alive():
                mon.pulse_reset()
            else:
                ser = open_port(port, 115200, 0.2)
                pulse_reset(ser)
                ser.close()
        except (serial.SerialException, OSError) as e:
            self.event(port, f"reset failed: {e}", "error")
            return {"error": str(e)}
        self.event(port, "reset pulsed (EN low via RTS)", "reset")
        return {"ok": True}


def open_port(port, baud, timeout):
    # Default open asserts DTR+RTS: a C3 USB-Serial/JTAG stays silent, and
    # line transitions can drop it into ROM download mode. Recipe verified on
    # kettle-knob: pre-lower both, open, then pulse an explicit run-mode
    # reset (RTS=EN low with DTR low) so the board boots the app regardless
    # of what state the open left it in.
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = timeout
    ser.dtr = False
    ser.rts = False
    ser.open()
    pulse_reset(ser)
    return ser


def pulse_reset(ser):
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False


class Monitor(threading.Thread):
    daemon = True

    def __init__(self, port, baud, bench):
        super().__init__(name=f"mon-{os.path.basename(port)}")
        self.port = port
        self.baud = baud
        self.bench = bench
        self.ser = None
        self.running = True
        self.lock = threading.Lock()
        self.lines = deque(maxlen=MONITOR_BUFFER)
        self.next_id = 1

    def open(self):
        self.ser = open_port(self.port, self.baud, 0.5)

    def alive(self):
        return self.running and self.ser is not None

    def _push(self, text, kind="rx"):
        with self.lock:
            self.lines.append({"id": self.next_id, "t": time.time(),
                               "kind": kind, "text": text})
            self.next_id += 1

    def run(self):
        buf = b""
        while self.running:
            try:
                chunk = self.ser.read(4096)
            except (serial.SerialException, OSError, TypeError) as e:
                if self.running:
                    self._push(f"[monitor lost: {e}]", "err")
                    self.bench.event(self.port, f"monitor lost: {e}", "error")
                self.running = False
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._push(line.decode("utf-8", errors="replace").rstrip("\r"))
            if len(buf) > 4096:
                self._push(buf.decode("utf-8", errors="replace"))
                buf = b""

    def since(self, cursor):
        with self.lock:
            return {"cursor": self.next_id - 1, "alive": self.alive(),
                    "lines": [l for l in self.lines if l["id"] > cursor]}

    def send(self, text):
        self.ser.write(text.encode() + b"\n")
        self._push(text, "tx")

    def pulse_reset(self):
        pulse_reset(self.ser)
        self._push("[reset pulsed]", "err")

    def stop(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except (serial.SerialException, OSError):
                pass
            self.ser = None


BENCH = Bench()


def json_response(handler, obj, code=200):
    body = json.dumps(obj, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def q(self):
        return parse_qs(urlparse(self.path).query)

    def qp(self, name, default=None):
        return self.q().get(name, [default])[0]

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/ports":
            json_response(self, {"ports": BENCH.ports(), "bauds": BAUDS})
        elif path == "/api/topology":
            json_response(self, BENCH.topology())
        elif path == "/api/catalog":
            BENCH.refresh_catalog()
            json_response(self, {"boards": BENCH.catalog["boards"],
                                 "roles": BENCH.catalog["roles"]})
        elif path == "/api/monitor/read":
            port = self.qp("port")
            since = int(self.qp("since", "0"))
            json_response(self, BENCH.read_monitor(port, since))
        elif path == "/api/log":
            json_response(self, BENCH.events_since(int(self.qp("since", "0"))))
        elif path in ("/", "/index.html"):
            with open(UI_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        port = self.qp("port")
        if not port or not port.startswith("/dev/cu."):
            json_response(self, {"error": "bad port"}, 400)
            return
        if path == "/api/identify":
            json_response(self, BENCH.identify(port))
        elif path == "/api/monitor/start":
            baud = int(self.qp("baud", "115200"))
            json_response(self, BENCH.start_monitor(port, baud))
        elif path == "/api/monitor/stop":
            json_response(self, BENCH.stop_monitor(port))
        elif path == "/api/reset":
            json_response(self, BENCH.reset(port))
        elif path == "/api/send":
            n = int(self.headers.get("Content-Length", 0))
            text = self.rfile.read(n).decode("utf-8", errors="replace")
            json_response(self, BENCH.send(port, text))
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Festicorn board bench")
    ap.add_argument("--http", type=int, default=8090)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.http), Handler)
    server.daemon_threads = True
    print(f"Board bench: http://localhost:{args.http}/")
    print(f"Catalog: {CATALOG_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
