#!/usr/bin/env python3
"""
Faroudja panel dashboard — live web UI reading the board's serial JSON debug.

Run:   ../../.venv/bin/python dashboard.py [--port /dev/cu.usbserial-2120] [--http 8081]
Open:  http://localhost:8081/
"""

import argparse
import json
import socket
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import serial

state = {
    "p": [0.0] * 6,
    "pull": [0] * 6,
    "btn": [0] * 7,
    "raw": [0] * 6,
    "wraw": [0] * 6,
    "ts": 0,
    "connected": False,
    "lines": 0,
}
state_lock = threading.Lock()


def serial_reader(port, baud=230400):
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
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "p" not in d or "wraw" not in d:
                    continue
                with state_lock:
                    for k in ("p", "pull", "btn", "raw", "wraw"):
                        state[k] = d[k]
                    state["ts"] = time.time()
                    state["lines"] += 1
        except (serial.SerialException, OSError) as e:
            with state_lock:
                state["connected"] = False
            print(f"[serial] lost: {e} — retrying in 2s")
            time.sleep(2)


RREF = 330.0
VCC = 3.3
WHITE_PINS = ["VP/36", "VN/39", "34", "35", "32", "33"]
BTN_PINS = ["25", "26", "14", "27", "16", "13", "17 (toggle)"]

HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Faroudja Panel</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#eee; font-family:'SF Mono',Menlo,monospace; padding:24px; }
  h1 { font-size:18px; color:#888; margin-bottom:12px; }
  .status { font-size:12px; color:#666; margin-bottom:20px; }
  .status.live { color:#4a4; }
  .status.dead { color:#a44; }
  .pots { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:24px; }
  .pot { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:14px; }
  .pot.pulled { border-color:#c83; }
  .pot .label { font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1px; display:flex; justify-content:space-between; }
  .pot .mode { color:#555; }
  .pot.pulled .mode { color:#f92; }
  .pot .value { font-size:32px; font-weight:bold; margin:6px 0; color:#fff; }
  .pot .value.changed { color:#f92; }
  .bar { height:8px; background:#262626; border-radius:4px; overflow:hidden; margin:6px 0; }
  .bar .fill { height:100%; background:#4a8; transition:width 0.15s; }
  .pot.pulled .bar .fill { background:#c83; }
  .pot .sub { font-size:10px; color:#555; line-height:1.5; }
  .btns { display:grid; grid-template-columns:repeat(auto-fit,minmax(90px,1fr)); gap:8px; margin-bottom:24px; }
  .btn { background:#1a1a1a; border:1px solid #262626; border-radius:6px; padding:12px; text-align:center; }
  .btn.on { background:#2a3; border-color:#4c5; color:#fff; }
  .btn .label { font-size:10px; color:#666; }
  .btn.on .label { color:#cfc; }
  .btn .state { font-size:16px; font-weight:bold; margin-top:4px; }
  .section { font-size:13px; color:#666; margin:16px 0 8px; text-transform:uppercase; letter-spacing:1px; }
</style>
</head><body>
<h1>Faroudja LD100 Panel — Dev Dashboard</h1>
<div class="status dead" id="status">disconnected</div>
<div class="section">Pots</div>
<div class="pots" id="pots"></div>
<div class="section">Buttons</div>
<div class="btns" id="btns"></div>
<script>
const WHITE_PINS = __WHITE_PINS__;
const BTN_PINS = __BTN_PINS__;
const RREF = __RREF__, VCC = __VCC__;
let prev = {p:[]};

function adsToOhms(raw) {
  const v = raw * 0.000125;
  if (v <= 0.001) return 0;
  if (v >= VCC - 0.05) return Infinity;
  return v * RREF / (VCC - v);
}

function render(s) {
  const el = document.getElementById('status');
  const age = Date.now()/1000 - s.ts;
  if (!s.connected || age > 5) {
    el.className = 'status dead';
    el.textContent = s.connected ? 'stale (' + age.toFixed(0) + 's)' : 'disconnected';
  } else {
    el.className = 'status live';
    el.textContent = 'live — ' + s.lines + ' frames';
  }

  let html = '';
  for (let i = 0; i < 6; i++) {
    const pulled = s.pull[i] ? ' pulled' : '';
    const changed = prev.p[i] !== undefined && Math.abs(prev.p[i] - s.p[i]) > 0.005;
    const ohms = adsToOhms(s.raw[i]);
    const ohmsStr = isFinite(ohms) ? ohms.toFixed(0) + 'Ω' : 'open';
    const wv = (s.wraw[i] * VCC / 4095).toFixed(2);
    html += '<div class="pot' + pulled + '">' +
      '<div class="label"><span>Pot ' + i + (i===3?' ⇄':'') + '</span>' +
      '<span class="mode">' + (s.pull[i] ? 'PULLED' : 'pressed') + '</span></div>' +
      '<div class="value' + (changed?' changed':'') + '">' + s.p[i].toFixed(3) + '</div>' +
      '<div class="bar"><div class="fill" style="width:' + (s.p[i]*100) + '%"></div></div>' +
      '<div class="sub">ads ' + s.raw[i] + ' → ' + ohmsStr +
      '<br>white ' + s.wraw[i] + ' → ' + wv + 'V (' + WHITE_PINS[i] + ')</div></div>';
  }
  document.getElementById('pots').innerHTML = html;

  html = '';
  for (let i = 0; i < BTN_PINS.length; i++) {
    html += '<div class="btn' + (s.btn[i] ? ' on' : '') + '">' +
      '<div class="label">GPIO ' + BTN_PINS[i] + '</div>' +
      '<div class="state">' + (s.btn[i] ? 'ON' : '—') + '</div></div>';
  }
  document.getElementById('btns').innerHTML = html;

  prev = {p: s.p.slice()};
}

function poll() {
  fetch('/api/state', {signal: AbortSignal.timeout(1500)})
    .then(r => r.json()).then(render).catch(() => {})
    .finally(() => setTimeout(poll, 33));
}
poll();
</script>
</body></html>"""

HTML = (HTML
        .replace("__WHITE_PINS__", json.dumps(WHITE_PINS))
        .replace("__BTN_PINS__", json.dumps(BTN_PINS))
        .replace("__RREF__", str(RREF))
        .replace("__VCC__", str(VCC)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
    ap = argparse.ArgumentParser(description="Faroudja panel dashboard")
    ap.add_argument("--port", default="/dev/cu.usbserial-2120",
                    help="board serial port")
    ap.add_argument("--http", type=int, default=8081, help="HTTP port")
    args = ap.parse_args()

    t = threading.Thread(target=serial_reader, args=(args.port,), daemon=True)
    t.start()

    class DualStackServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6

        def server_bind(self):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()

    httpd = DualStackServer(("::", args.http), Handler)
    print(f"Dashboard: http://localhost:{args.http}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
