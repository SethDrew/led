#!/usr/bin/env python3
"""
bench-bulbs operator UI — effect selector + per-effect runtime knobs.

Auto-detects ESP-NOW bridge or direct bench-bulbs serial port.
Opens a browser UI on localhost:8322.

Usage:
  python operator.py              # auto-detect
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
HTTP_PORT = 8322
CMD_MAGIC = 0xC0

ser = None
use_bridge = False
last_status = "ready"


def probe_port(port):
    """Open port briefly, send newline to trigger boot echo, read role."""
    try:
        s = serial.Serial(port, BAUD, timeout=0.8)
        s.reset_input_buffer()
        s.write(b'\n?\n')
        time.sleep(0.5)
        data = s.read(2048).decode('utf-8', 'replace')
        s.close()
        for line in data.split('\n'):
            if '[BOOT]' in line and 'role=' in line:
                role = line.split('role=')[1].split()[0].strip()
                return role
        return None
    except Exception:
        return None


def find_serial_port():
    """Auto-detect bridge or bench-bulbs by role in boot line."""
    candidates = sorted(glob.glob('/dev/cu.usb*'))
    bridge_port = None
    direct_port = None
    for port in candidates:
        role = probe_port(port)
        if role == 'bridge':
            bridge_port = port
        elif role and 'bench' in role:
            direct_port = port
    if bridge_port:
        return bridge_port, True
    if direct_port:
        return direct_port, False
    if candidates:
        print(f"Warning: no role detected, using {candidates[0]} as direct serial")
        return candidates[0], False
    return None, False


def send_cmd(cmd_str):
    """Send command — framed for bridge, raw for direct serial."""
    if use_bridge:
        payload = bytes([CMD_MAGIC]) + cmd_str.encode('utf-8')
        length = len(payload)
        if length > 250:
            return
        xor = length
        for b in payload:
            xor ^= b
        frame = bytes([0xA5, 0x5A, length]) + payload + bytes([xor & 0xFF])
        ser.write(frame)
    else:
        ser.write((cmd_str + '\n').encode())


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
    'fire_flicker': {'cmd': 'f', 'label': 'fire flicker', 'knobs': []},
    'fire_meld':    {'cmd': 'm', 'label': 'fire meld',    'knobs': []},
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
    ('GLOBAL_BRIGHTNESS',  'brightness',  0.0,  1.0, 0.01, False),
]

params = {}
for eff in EFFECTS.values():
    for k in eff['knobs']:
        key, _, lo, hi, _, _ = k
        params[key] = (lo + hi) / 2
for k in GLOBAL_KNOBS:
    key, _, lo, hi, _, _ = k
    params[key] = 1.0 if 'BRIGHTNESS' in key or 'SENSITIVITY' in key else (lo + hi) / 2


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
  .conn { font-size: 11px; color: #888; }
  .conn .mode { color: #2a6; font-weight: bold; }
</style></head><body>
<h2>bench-bulbs operator</h2>
<div class="conn">__MODE__ via __PORT__</div>

<h3>effect</h3>
<div class="effects" id="effects"></div>

<h3>global</h3>
<div id="global"></div>

<h3 id="knobsHeader">knobs</h3>
<div id="knobs"></div>

<div class="actions">
  <button onclick="cmd('c')">calibrate</button>
</div>

<div id="status">--</div>

<script>
let EFFECTS = __EFFECTS_JSON__;
let GLOBALS = __GLOBALS_JSON__;
let currentEffect = 'bloom';
let params = __PARAMS_JSON__;

function setStatus(s) {
  let el = document.getElementById('status');
  let t = new Date().toLocaleTimeString();
  el.textContent = '[' + t + '] ' + s + '\\n' + el.textContent;
  if (el.textContent.length > 1500) el.textContent = el.textContent.slice(0, 1500);
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

renderEffects();
renderGlobal();
renderKnobs();
cmd(EFFECTS[currentEffect].cmd);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global last_status
        if self.path == '/' or self.path == '/index.html':
            mode = 'ESP-NOW bridge' if use_bridge else 'serial'
            html = (HTML
                    .replace('__EFFECTS_JSON__', json.dumps(EFFECTS))
                    .replace('__GLOBALS_JSON__', json.dumps(GLOBAL_KNOBS))
                    .replace('__PARAMS_JSON__', json.dumps(params))
                    .replace('__MODE__', mode)
                    .replace('__PORT__', ser.port if ser else '?'))
            self._respond(200, 'text/html', html)
        elif self.path.startswith('/cmd?'):
            p = self._params()
            c = p.get('c', '')
            if c:
                send_cmd(c)
                last_status = f"cmd {c}"
            self._respond(200, 'text/plain', last_status)
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
    global ser, use_bridge
    if len(sys.argv) > 1:
        port = sys.argv[1]
        use_bridge = '--bridge' in sys.argv
        ser = serial.Serial(port, BAUD, timeout=0.1)
    else:
        port, use_bridge = find_serial_port()
        if not port:
            print("No serial port found")
            sys.exit(1)
        ser = serial.Serial(port, BAUD, timeout=0.1)

    mode = 'ESP-NOW bridge' if use_bridge else 'serial (direct)'
    print(f"Port: {port} ({mode})")
    print(f"UI:   http://localhost:{HTTP_PORT}")

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
