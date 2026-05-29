#!/usr/bin/env python3
"""
bench-bulbs operator UI — effect selector + per-effect runtime knobs.

Opens a browser UI. Talks to ESP32 over serial:
  - single-char effect commands (b/g/t/y/f/m/i/w/x/c)
  - `!P KEY=value` to set runtime params
  - `!P?` on startup to read current values

Usage:
  python operator.py [serial_port]
"""

import sys
import time
import threading
import serial
import webbrowser
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = '/dev/cu.usbserial-0001'
BAUD = 460800
HTTP_PORT = 8322

ser = None
params = {}        # name -> current float/int value
params_lock = threading.Lock()
last_status = "ready"


# ── per-effect knob definitions (must mirror firmware PARAMS table) ──
# Each knob: (key, label, lo, hi, step, is_int)
EFFECTS = {
    'bloom': {
        'cmd': 'b',
        'label': 'bloom',
        'knobs': [
            ('BLOOM_FLASH_BUMP',       'flash bump',        0.0,  0.30, 0.005, False),
            ('BLOOM_ACCEL_THRESH',     'accel threshold',   1.05, 3.0,  0.05,  False),
            ('BLOOM_BRIGHTNESS_CAP',   'brightness cap',    0.05, 1.0,  0.01,  False),
            ('BLOOM_FLASH_DECAY_RATE', 'flash decay rate',  0.5,  5.0,  0.1,   False),
            ('BLOOM_BREATH_FLOOR',     'breath floor',      0.0,  0.5,  0.01,  False),
        ],
    },
    'gravity': {
        'cmd': 'g',
        'label': 'gravity',
        'knobs': [
            ('GS_PARTICLE_COUNT',  'particle count',   1,    20,   1,     True),
            ('GS_GRAVITY_SCALE',   'gravity scale',    5.0,  200., 1.0,   False),
            ('GS_VELOCITY_DAMP',   'velocity damp',    0.5,  0.999,0.001, False),
            ('GS_BOUNCE_REBOUND',  'bounce rebound',   0.0,  1.0,  0.01,  False),
            ('GS_SPLAT_RADIUS',    'splat radius',     0.5,  8.0,  0.1,   False),
            ('GS_BRIGHTNESS_CAP',  'brightness cap',   0.05, 1.0,  0.01,  False),
        ],
    },
    'twinkle': {
        'cmd': 't',
        'label': 'twinkle',
        'knobs': [
            ('TWINKLE_SPAWN_RATE',     'spawn rate /s',    1.0,   300., 1.0,   False),
            ('TWINKLE_ATTACK_S',       'attack (s)',       0.005, 0.5,  0.005, False),
            ('TWINKLE_TAU_S',          'decay tau (s)',    0.02,  2.0,  0.01,  False),
            ('TWINKLE_PEAK_MIN',       'peak min',         0.0,   1.0,  0.01,  False),
            ('TWINKLE_PEAK_MAX',       'peak max',         0.0,   1.0,  0.01,  False),
            ('SPARKLE_BRIGHTNESS_CAP', 'brightness cap',   0.01,  0.5,  0.005, False),
        ],
    },
    'syllable': {
        'cmd': 'y',
        'label': 'syllable',
        'knobs': [
            ('SPARKLE_BRIGHTNESS_CAP', 'brightness cap',   0.01, 0.5, 0.005, False),
        ],
    },
    'fire_flicker': {
        'cmd': 'f',
        'label': 'fire flicker',
        'knobs': [],
    },
    'fire_meld': {
        'cmd': 'm',
        'label': 'fire meld',
        'knobs': [],
    },
    'idle': {
        'cmd': 'i',
        'label': 'idle',
        'knobs': [
            ('IDLE_BRIGHTNESS', 'brightness', 0.0, 1.0, 0.01, False),
            ('IDLE_SPEED',      'speed',      0.0, 0.5, 0.005, False),
        ],
    },
    'white_stress': {'cmd': 'w', 'label': 'white stress', 'knobs': []},
    'off':          {'cmd': 'x', 'label': 'off',          'knobs': []},
}

GLOBAL_KNOBS = [
    ('GLOBAL_BRIGHTNESS',  'brightness',  0.0,  2.0, 0.01, False),
    ('GLOBAL_SENSITIVITY', 'sensitivity', 0.05, 2.0, 0.01, False),
]


def serial_reader():
    """Drain serial; capture [PARAM] lines into the params dict."""
    buf = b''
    while True:
        try:
            data = ser.read(256)
            if not data:
                continue
            buf += data
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                s = line.decode('utf-8', 'replace').strip()
                if s.startswith('[PARAM]'):
                    rest = s[len('[PARAM]'):].strip()
                    if '=' in rest:
                        k, v = rest.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        try:
                            fv = float(v)
                            with params_lock:
                                params[k] = fv
                        except ValueError:
                            pass
        except Exception:
            time.sleep(0.05)


def request_param_dump():
    ser.write(b'!P?\n')


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>bench-bulbs operator</title>
<style>
  body { background: #1a1a1a; color: #ccc; font-family: 'Menlo', monospace;
         max-width: 640px; margin: 24px auto; padding: 0 16px; }
  h2 { margin-bottom: 4px; }
  h3 { color: #aaa; font-size: 13px; margin: 16px 0 8px; text-transform: uppercase;
       letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 4px; }
  .effects { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
  .effect-btn { background: #2a2a2a; color: #ccc; border: 1px solid #444;
                padding: 8px 14px; font-family: inherit; font-size: 13px;
                cursor: pointer; border-radius: 3px; }
  .effect-btn:hover { background: #333; }
  .effect-btn.on { background: #2a6; color: white; border-color: #2a6; }
  .knob { display: flex; align-items: center; margin: 6px 0; gap: 10px; }
  .knob label { width: 160px; font-size: 12px; color: #999; }
  .knob input[type=range] { flex: 1; accent-color: #2a6; }
  .knob input[type=number] { width: 78px; background: #2a2a2a; color: white;
    border: 1px solid #444; font-family: inherit; font-size: 12px;
    text-align: right; padding: 4px 6px; }
  .actions { display: flex; gap: 8px; margin: 12px 0; }
  .actions button { background: #2a2a2a; color: #ccc; border: 1px solid #444;
    padding: 6px 14px; font-family: inherit; font-size: 12px; cursor: pointer; }
  .actions button:hover { background: #333; }
  #status { color: #666; font-size: 11px; margin-top: 16px;
            border-top: 1px solid #333; padding-top: 8px;
            white-space: pre-wrap; max-height: 80px; overflow-y: auto; }
</style></head><body>
<h2>bench-bulbs operator</h2>
<div id="conn" style="font-size:11px;color:#888;">--</div>

<h3>effect</h3>
<div class="effects" id="effects"></div>

<h3>global</h3>
<div id="global"></div>

<h3 id="knobsHeader">knobs</h3>
<div id="knobs"></div>

<div class="actions">
  <button onclick="cmd('c')">calibrate</button>
  <button onclick="refresh()">refresh</button>
</div>

<div id="status">--</div>

<script>
let EFFECTS = __EFFECTS_JSON__;
let GLOBALS = __GLOBALS_JSON__;
let currentEffect = 'bloom';
let params = {};

function setStatus(s) {
  let el = document.getElementById('status');
  let t = new Date().toLocaleTimeString();
  el.textContent = '[' + t + '] ' + s + '\\n' + el.textContent;
  if (el.textContent.length > 1500) el.textContent = el.textContent.slice(0, 1500);
}

async function refresh() {
  let r = await fetch('/params');
  params = await r.json();
  renderGlobal();
  renderKnobs();
  setStatus('refreshed (' + Object.keys(params).length + ' params)');
}

async function cmd(c) {
  await fetch('/cmd?c=' + encodeURIComponent(c));
  setStatus('cmd: ' + c);
}

async function setParam(key, val) {
  await fetch('/set?k=' + encodeURIComponent(key) + '&v=' + encodeURIComponent(val));
  params[key] = parseFloat(val);
  setStatus('!P ' + key + '=' + val);
}

function knobRow(k) {
  let [key, label, lo, hi, step, isInt] = k;
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
  let [key, label, lo, hi, step, isInt] = k;
  let s = document.getElementById('s_' + key);
  let n = document.getElementById('n_' + key);
  if (!s) return;
  let timer = null;
  function fire(v) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      let vv = isInt ? Math.round(parseFloat(v)) : parseFloat(v);
      setParam(key, vv);
    }, 40);
  }
  s.oninput = () => { n.value = s.value; fire(s.value); };
  n.onchange = () => { s.value = n.value; fire(n.value); };
}

function renderGlobal() {
  let g = document.getElementById('global');
  g.innerHTML = '';
  for (let k of GLOBALS) g.appendChild(knobRow(k));
  for (let k of GLOBALS) wireKnob(k);
}

function renderKnobs() {
  let h = document.getElementById('knobsHeader');
  let box = document.getElementById('knobs');
  box.innerHTML = '';
  let eff = EFFECTS[currentEffect];
  h.textContent = eff.label + ' knobs';
  if (!eff.knobs.length) {
    box.innerHTML = '<div style="color:#666;font-size:12px;">(no runtime knobs)</div>';
    return;
  }
  for (let k of eff.knobs) box.appendChild(knobRow(k));
  for (let k of eff.knobs) wireKnob(k);
}

function renderEffects() {
  let bar = document.getElementById('effects');
  bar.innerHTML = '';
  for (let name in EFFECTS) {
    let eff = EFFECTS[name];
    let b = document.createElement('button');
    b.className = 'effect-btn' + (name === currentEffect ? ' on' : '');
    b.textContent = eff.label;
    b.onclick = async () => {
      currentEffect = name;
      await cmd(eff.cmd);
      renderEffects();
      renderKnobs();
    };
    bar.appendChild(b);
  }
}

document.getElementById('conn').textContent = 'serial: __PORT__';
renderEffects();
refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global last_status
        if self.path == '/' or self.path == '/index.html':
            html = (HTML
                    .replace('__EFFECTS_JSON__', json.dumps(EFFECTS))
                    .replace('__GLOBALS_JSON__', json.dumps(GLOBAL_KNOBS))
                    .replace('__PORT__', PORT))
            self._respond(200, 'text/html', html)
        elif self.path == '/params':
            request_param_dump()
            time.sleep(0.25)
            with params_lock:
                snap = dict(params)
            self._respond(200, 'application/json', json.dumps(snap))
        elif self.path.startswith('/cmd?'):
            p = self._params()
            c = p.get('c', '')
            if c:
                ser.write(c.encode() + b'\n')
                last_status = f"cmd {c}"
            self._respond(200, 'text/plain', last_status)
        elif self.path.startswith('/set?'):
            p = self._params()
            k = p.get('k', '')
            v = p.get('v', '')
            if k and v:
                line = f'!P {k}={v}\n'
                ser.write(line.encode())
                last_status = line.strip()
                try:
                    with params_lock:
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
    global ser, PORT
    PORT = sys.argv[1] if len(sys.argv) > 1 else PORT
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    print(f"Serial: {PORT}")
    print(f"UI: http://localhost:{HTTP_PORT}")

    threading.Thread(target=serial_reader, daemon=True).start()
    time.sleep(0.3)
    request_param_dump()
    time.sleep(0.5)

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
