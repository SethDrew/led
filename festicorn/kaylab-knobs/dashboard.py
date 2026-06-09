#!/usr/bin/env python3
"""
Kay Lab knob dashboard — live web UI reading ESP-NOW sniffer serial.

Run:   ../../.venv/bin/python dashboard.py [--port /dev/cu.usbmodem114301] [--http 8080]
Open:  http://localhost:8080/
"""

import argparse
import json
import re
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import serial

# Sniffer output format (from sniffer.cpp):
# seq     ones tens hund dout dmid din  | up(s)  boot | rawH  rawT  rawO  adsT  adsO  aRef | changes
# %-7d %4d %4d %4d %4d %4d %4d | %5lu %4d | %5d %5d %5d %5d %5d %4d | %s
LINE_RE = re.compile(
    r"^\s*(\d+)\s+"           # seq
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+"  # ones tens hund
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\|\s*"  # dout dmid din
    r"(\d+)\s+(-?\d+)\s*\|\s*"  # up boot
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+"  # rawH rawT rawO
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\|\s*"  # adsT adsO aRef
    r"(.*)"                   # changes
)

state = {
    "seq": 0, "ones": -1, "tens": -1, "hundreds": -1,
    "decOuter": -1, "decMid": -1, "decInner": -1,
    "uptime": 0, "boot": -1,
    "rawH": 0, "rawT": 0, "rawO": 0,
    "adsT": 0, "adsO": 0, "adsRef": 0,
    "changes": "", "ts": 0, "connected": False,
}
state_lock = threading.Lock()


def serial_reader(port, baud=115200):
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            with state_lock:
                state["connected"] = True
            print(f"[serial] connected to {port}")
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                m = LINE_RE.match(line)
                if not m:
                    continue
                g = m.groups()
                with state_lock:
                    state["seq"] = int(g[0])
                    state["ones"] = int(g[1])
                    state["tens"] = int(g[2])
                    state["hundreds"] = int(g[3])
                    state["decOuter"] = int(g[4])
                    state["decMid"] = int(g[5])
                    state["decInner"] = int(g[6])
                    state["uptime"] = int(g[7])
                    state["boot"] = int(g[8])
                    state["rawH"] = int(g[9])
                    state["rawT"] = int(g[10])
                    state["rawO"] = int(g[11])
                    state["adsT"] = int(g[12])
                    state["adsO"] = int(g[13])
                    state["adsRef"] = int(g[14])
                    state["changes"] = g[15].strip()
                    state["ts"] = time.time()
        except (serial.SerialException, OSError) as e:
            with state_lock:
                state["connected"] = False
            print(f"[serial] lost: {e} — retrying in 2s")
            time.sleep(2)


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Kay Lab Knobs</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#eee; font-family:'SF Mono',Menlo,monospace; padding:24px; }
  h1 { font-size:18px; color:#888; margin-bottom:16px; }
  .status { font-size:12px; color:#666; margin-bottom:20px; }
  .status.live { color:#4a4; }
  .status.dead { color:#a44; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:24px; }
  .card { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:16px; text-align:center; }
  .card .label { font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1px; }
  .card .value { font-size:36px; font-weight:bold; margin:8px 0; color:#fff; }
  .card .value.changed { color:#f92; }
  .card .sub { font-size:11px; color:#555; }
  .section { font-size:13px; color:#666; margin:16px 0 8px; text-transform:uppercase; letter-spacing:1px; }
  .raw-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px; }
  .raw { background:#1a1a1a; border:1px solid #262626; border-radius:6px; padding:10px; text-align:center; }
  .raw .label { font-size:10px; color:#666; }
  .raw .value { font-size:18px; color:#aaa; margin-top:4px; }
  .changes { margin-top:16px; font-size:13px; color:#f92; min-height:20px; }
  .combo { font-size:14px; color:#888; margin-bottom:20px; }
  .combo span { color:#fff; font-size:20px; font-weight:bold; }
</style>
</head><body>
<h1>Kay Lab 508-25 — Knob Dashboard</h1>
<div class="status dead" id="status">disconnected</div>
<div class="combo" id="combo"></div>
<div class="grid" id="knobs"></div>
<div class="section">Decade Knob</div>
<div class="grid" id="decade"></div>
<div class="section">Raw ADC</div>
<div class="raw-grid" id="raw"></div>
<div class="section">Diagnostics</div>
<div class="raw-grid" id="diag"></div>
<div class="changes" id="changes"></div>
<script>
let prev = {};
function render(s) {
  // Status
  const el = document.getElementById('status');
  const age = Date.now()/1000 - s.ts;
  if (!s.connected || age > 5) {
    el.className = 'status dead';
    el.textContent = s.connected ? 'stale (' + age.toFixed(0) + 's)' : 'disconnected';
  } else {
    el.className = 'status live';
    el.textContent = 'live — seq ' + s.seq;
  }

  // Combo display
  const hStr = s.hundreds >= 0 ? s.hundreds : '?';
  const tStr = s.tens >= 0 ? s.tens : '?';
  const oStr = s.ones >= 0 ? (s.ones === 0 ? '1.02' : (s.ones+1)) : '?';
  const dStr = (s.decOuter >= 0 ? s.decOuter : '?') + '' +
               (s.decMid >= 0 ? s.decMid : '?') + '' +
               (s.decInner >= 0 ? s.decInner : '?');
  document.getElementById('combo').innerHTML =
    'Effect: <span>' + hStr + tStr + oStr + '</span> &nbsp; Decade: <span>' + dStr + '</span>';

  // Knob cards
  const knobs = [
    {key:'hundreds', label:'Hundreds', sub:'GPIO 35'},
    {key:'tens', label:'Tens', sub:'GPIO 32'},
    {key:'ones', label:'Ones', sub:'GPIO 34'},
  ];
  let html = '';
  for (const k of knobs) {
    const changed = prev[k.key] !== undefined && prev[k.key] !== s[k.key];
    html += '<div class="card"><div class="label">' + k.label + '</div>' +
      '<div class="value' + (changed?' changed':'') + '">' + s[k.key] + '</div>' +
      '<div class="sub">' + k.sub + '</div></div>';
  }
  document.getElementById('knobs').innerHTML = html;

  // Decade cards
  const dec = [
    {key:'decOuter', label:'Outer (100s)', sub:'ADS Ch0'},
    {key:'decMid', label:'Mid (10s)', sub:'interpolated'},
    {key:'decInner', label:'Inner (1s)', sub:'ADS Ch1'},
  ];
  html = '';
  for (const k of dec) {
    const changed = prev[k.key] !== undefined && prev[k.key] !== s[k.key];
    html += '<div class="card"><div class="label">' + k.label + '</div>' +
      '<div class="value' + (changed?' changed':'') + '">' + s[k.key] + '</div>' +
      '<div class="sub">' + k.sub + '</div></div>';
  }
  document.getElementById('decade').innerHTML = html;

  // Raw ADC
  const raws = [
    {key:'rawH', label:'Hundreds raw'},
    {key:'rawT', label:'Tens raw'},
    {key:'rawO', label:'Ones raw'},
    {key:'adsT', label:'ADS Tens'},
    {key:'adsO', label:'ADS Ones'},
    {key:'adsRef', label:'ADS Ref'},
  ];
  html = '';
  for (const r of raws) {
    html += '<div class="raw"><div class="label">' + r.label + '</div>' +
      '<div class="value">' + s[r.key] + '</div></div>';
  }
  document.getElementById('raw').innerHTML = html;

  // Diagnostics
  const mins = Math.floor(s.uptime / 60);
  const secs = s.uptime % 60;
  html = '<div class="raw"><div class="label">Uptime</div><div class="value">' +
    mins + 'm ' + secs + 's</div></div>' +
    '<div class="raw"><div class="label">Boot #</div><div class="value">' +
    s.boot + '</div></div>' +
    '<div class="raw"><div class="label">Seq</div><div class="value">' +
    s.seq + '</div></div>';
  document.getElementById('diag').innerHTML = html;

  // Changes
  document.getElementById('changes').textContent = s.changes || '';

  prev = Object.assign({}, s);
}

function poll() {
  fetch('/api/state').then(r => r.json()).then(render).catch(() => {});
  setTimeout(poll, 250);
}
poll();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            with state_lock:
                body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Kay Lab knob dashboard")
    ap.add_argument("--port", default="/dev/cu.usbmodem114301",
                    help="sniffer serial port")
    ap.add_argument("--http", type=int, default=8080, help="HTTP port")
    args = ap.parse_args()

    t = threading.Thread(target=serial_reader, args=(args.port,), daemon=True)
    t.start()

    httpd = HTTPServer(("0.0.0.0", args.http), Handler)
    print(f"Dashboard: http://localhost:{args.http}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
