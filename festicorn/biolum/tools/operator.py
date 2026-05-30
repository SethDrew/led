#!/usr/bin/env python3
"""
biolum operator — runtime knob UI over ESP-NOW bridge.

Sends parameter commands through the espnow-bridge board, which
broadcasts them on ch6. The biolum receiver picks them up as
CMD_PKT_MAGIC (0xC0) prefixed packets.

Usage:
  python operator.py              # auto-detect bridge port
  python operator.py /dev/cu...   # explicit port
"""

import sys
import glob
import time
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import serial
except ImportError:
    print("pip install pyserial")
    sys.exit(1)

BAUD = 460800
HTTP_PORT = 8323
CMD_MAGIC = 0xC0

ser = None
last_status = "ready"


def probe_port(port):
    try:
        s = serial.Serial(port, BAUD, timeout=0.8)
        s.reset_input_buffer()
        s.write(b'\n?\n')
        time.sleep(0.5)
        data = s.read(2048).decode('utf-8', 'replace')
        s.close()
        for line in data.split('\n'):
            if '[BOOT]' in line and 'role=bridge' in line:
                return True
        return False
    except Exception:
        return False


def find_bridge():
    candidates = sorted(glob.glob('/dev/cu.usb*'))
    for port in candidates:
        if probe_port(port):
            return port
    if candidates:
        print(f"Warning: no bridge detected, using {candidates[0]}")
        return candidates[0]
    return None


def send_cmd(cmd_str):
    payload = bytes([CMD_MAGIC]) + cmd_str.encode('utf-8')
    length = len(payload)
    if length > 250:
        return
    xor = length
    for b in payload:
        xor ^= b
    frame = bytes([0xA5, 0x5A, length]) + payload + bytes([xor & 0xFF])
    ser.write(frame)


KNOBS = [
    ('BRIGHTNESS_CAP_A', 'brightness cap (0–64)',  0.05, 1.0,  0.01),
    ('BRIGHTNESS_CAP_B', 'brightness cap (65–99)', 0.05, 1.0,  0.01),
    ('REST_G',           'rest threshold (g)',      0.9,  2.0,  0.01),
    ('BUFFER_DRAIN',     'buffer drain rate',       0.5,  20.0, 0.5),
    ('FLASH_DECAY_RATE', 'flash decay rate',        0.2,  10.0, 0.1),
    ('BREATH_FLOOR',     'breath floor',            0.0,  0.5,  0.01),
]

params = {}
defaults = {
    'BRIGHTNESS_CAP_A': 0.10,
    'BRIGHTNESS_CAP_B': 0.50,
    'REST_G': 1.07,
    'BUFFER_DRAIN': 4.0,
    'FLASH_DECAY_RATE': 1.5,
    'BREATH_FLOOR': 0.15,
}
for k in KNOBS:
    key = k[0]
    params[key] = defaults.get(key, (k[2] + k[3]) / 2)


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>biolum operator</title>
<style>
  body { background: #1a1a1a; color: #ccc; font-family: 'Menlo', monospace;
         max-width: 640px; margin: 24px auto; padding: 0 16px; }
  h2 { margin-bottom: 4px; }
  h3 { color: #aaa; font-size: 13px; margin: 16px 0 8px; text-transform: uppercase;
       letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 4px; }
  .knob { display: flex; align-items: center; margin: 6px 0; gap: 10px; }
  .knob label { width: 180px; font-size: 12px; color: #999; }
  .knob input[type=range] { flex: 1; accent-color: #2a6; }
  .knob input[type=number] { width: 78px; background: #2a2a2a; color: white;
    border: 1px solid #444; font-family: inherit; font-size: 12px;
    text-align: right; padding: 4px 6px; }
  #status { color: #666; font-size: 11px; margin-top: 16px;
            border-top: 1px solid #333; padding-top: 8px;
            white-space: pre-wrap; max-height: 80px; overflow-y: auto; }
  .conn { font-size: 11px; color: #888; }
  .conn .mode { color: #2a6; font-weight: bold; }
</style></head><body>
<h2>biolum operator</h2>
<div class="conn"><span class="mode">ESP-NOW bridge</span> via __PORT__</div>

<h3>bloom knobs</h3>
<div id="knobs"></div>

<div id="status">--</div>

<script>
let KNOBS = __KNOBS_JSON__;
let params = __PARAMS_JSON__;

function setStatus(s) {
  let el = document.getElementById('status');
  let t = new Date().toLocaleTimeString();
  el.textContent = '[' + t + '] ' + s + '\\n' + el.textContent;
  if (el.textContent.length > 1500) el.textContent = el.textContent.slice(0, 1500);
}

async function setParam(key, val) {
  await fetch('/set?k=' + encodeURIComponent(key) + '&v=' + encodeURIComponent(val));
  params[key] = parseFloat(val);
  setStatus('!P ' + key + '=' + val);
}

function knobRow(k) {
  let [key, label, lo, hi, step] = k;
  let cur = params[key];
  if (cur === undefined) cur = lo;
  let row = document.createElement('div');
  row.className = 'knob';
  row.innerHTML =
    '<label>' + label + '</label>' +
    '<input type="range" min="' + lo + '" max="' + hi + '" step="' + step + '" value="' + cur + '" id="s_' + key + '">' +
    '<input type="number" min="' + lo + '" max="' + hi + '" step="' + step + '" value="' + cur + '" id="n_' + key + '">';
  return row;
}

function wireKnob(k) {
  let [key] = k;
  let s = document.getElementById('s_' + key);
  let n = document.getElementById('n_' + key);
  if (!s) return;
  let timer = null;
  function fire(v) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => setParam(key, parseFloat(v)), 40);
  }
  s.oninput = () => { n.value = s.value; fire(s.value); };
  n.onchange = () => { s.value = n.value; fire(n.value); };
}

let box = document.getElementById('knobs');
for (let k of KNOBS) { box.appendChild(knobRow(k)); wireKnob(k); }
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global last_status
        if self.path == '/' or self.path == '/index.html':
            html = (HTML
                    .replace('__KNOBS_JSON__', json.dumps(KNOBS))
                    .replace('__PARAMS_JSON__', json.dumps(params))
                    .replace('__PORT__', ser.port if ser else '?'))
            self._respond(200, 'text/html', html)
        elif self.path.startswith('/set?'):
            p = self._params()
            k = p.get('k', '')
            v = p.get('v', '')
            if k and v:
                send_cmd(f'!P {k}={v}')
                last_status = f'!P {k}={v}'
                try:
                    params[k] = float(v)
                except ValueError:
                    pass
            self._respond(200, 'text/plain', last_status)
        else:
            self._respond(404, 'text/plain', 'not found')

    def _params(self):
        from urllib.parse import unquote
        return {p.split('=')[0]: unquote(p.split('=', 1)[1])
                for p in self.path.split('?')[1].split('&') if '=' in p}

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def log_message(self, format, *args):
        pass


def main():
    global ser
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = find_bridge()
        if not port:
            print("No serial port found")
            sys.exit(1)

    ser = serial.Serial(port, BAUD, timeout=0.1)
    print(f"Bridge: {port}")
    print(f"UI:     http://localhost:{HTTP_PORT}")

    server = HTTPServer(('127.0.0.1', HTTP_PORT), Handler)
    webbrowser.open(f'http://localhost:{HTTP_PORT}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        ser.close()
        server.server_close()


if __name__ == '__main__':
    main()
