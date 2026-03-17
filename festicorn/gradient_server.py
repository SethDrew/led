#!/usr/bin/env python3
"""Gradient picker web UI that streams to Arduino Nano over serial.

Mirrors the Butterfly (festicorn) gradient picker but renders server-side
and sends frames to a Nano via the standard serial protocol:
  [0xFF][0xAA][RGB × N][XOR checksum]  at 1 Mbps

Usage:
    python festicorn/gradient_server.py [--port /dev/cu.usbserial-1240] [--leds 300]
"""

import argparse
import json
import numpy as np
import serial
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# ── Adafruit gamma8 table (gamma=2.8) ────────────────────────────────
GAMMA8 = [
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,
    5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,
   10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
   17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
   25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,
   37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,
   51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,
   69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,
   90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,
  115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,
  144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,
  177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,
  215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255,
]
CROSSOVER = 153


def gamma_hybrid(v):
    if v == 0:
        return 0
    if v <= CROSSOVER:
        g = GAMMA8[v]
    else:
        base = GAMMA8[CROSSOVER]  # 137
        g = int(base + (v - CROSSOVER) * (255 - base) / (255 - CROSSOVER))
    return max(g, 2)


# ── Palette definitions (from festicorn/src/effects.cpp) ─────────────

PALETTES = {
    'sap_flow': [
        [0,40,0], [0,70,0], [10,110,10], [34,180,34], [80,255,80],
    ],
    'oklch_rainbow': [
        [251,3,64], [188,82,1], [2,155,72], [3,129,252], [184,1,169],
    ],
    'red_blue': [
        [251,3,64], [251,15,90], [252,26,117], [252,30,140],
        [252,20,162], [239,1,181], [179,1,167], [126,1,145],
        [86,1,121], [57,0,100], [40,0,86], [29,0,79],
        [24,0,82], [23,0,130], [25,6,248], [26,48,250],
    ],
    'cyan_gold': [
        [2,144,145], [2,139,184], [2,131,236], [29,118,252],
        [34,80,251], [25,6,248], [23,0,90], [53,0,97],
        [191,1,171], [252,30,127], [224,1,43], [150,1,7],
        [125,12,0], [151,37,1], [187,76,1], [166,100,1],
    ],
    'green_purple': [
        [2,163,12], [2,157,51], [2,153,82], [2,149,108],
        [2,146,132], [2,143,156], [2,139,182], [2,135,212],
        [2,129,252], [25,120,252], [40,108,252], [30,67,251],
        [24,17,249], [23,0,149], [23,0,84], [40,0,86],
    ],
    'orange_teal': [
        [137,27,0], [158,43,1], [179,63,1], [188,82,1],
        [179,95,1], [158,103,1], [138,111,1], [116,119,1],
        [92,129,1], [65,139,1], [31,153,1], [2,163,14],
        [2,158,45], [2,155,71], [2,152,92], [2,149,112],
    ],
    'magenta_cyan': [
        [252,20,162], [251,17,93], [202,1,31], [148,1,6],
        [126,10,0], [140,29,0], [176,59,1], [186,90,1],
        [155,104,1], [120,118,1], [80,133,1], [26,155,1],
        [2,159,39], [2,153,80], [2,149,114], [2,144,145],
    ],
    'sunset_sky': [
        [50,40,130], [70,40,125], [95,45,120], [125,50,115],
        [150,55,105], [175,50,85], [195,50,70], [210,60,55],
        [220,90,60], [215,115,55], [200,120,40], [185,95,20],
        [175,60,5], [170,40,2], [185,70,1], [210,110,15],
    ],
    'blue_wash': [
        [32,72,251], [37,74,220], [43,76,191], [49,77,165],
        [56,78,140], [63,78,118], [70,78,97], [78,78,78],
    ],
    'red_wash': [
        [145,1,4], [128,7,7], [111,12,11], [95,18,16],
        [79,23,20], [65,28,25], [50,32,31], [37,37,37],
    ],
    'green_wash': [
        [2,163,12], [16,156,23], [30,148,35], [45,141,48],
        [60,133,62], [76,125,76], [92,116,91], [108,108,108],
    ],
    'purple_wash': [
        [29,0,79], [27,3,67], [24,6,55], [22,8,45],
        [20,10,36], [18,12,28], [17,14,21], [15,15,15],
    ],
    'gold_wash': [
        [166,100,1], [157,102,13], [148,104,25], [139,105,39],
        [131,106,54], [123,107,70], [115,107,88], [108,108,108],
    ],
}


# ── Rendering ────────────────────────────────────────────────────────

def _lerp_palette_wrap(colors, t):
    """Interpolate palette with wrap-around (last stop blends back to first)."""
    n = len(colors)
    idx = t * n
    lo = int(idx) % n
    hi = (lo + 1) % n
    frac = idx - int(idx)
    return colors[lo] * (1 - frac) + colors[hi] * frac


def render_gradient(palette_name, num_leds, phase=0.0):
    colors = np.array(PALETTES.get(palette_name, PALETTES['oklch_rainbow']),
                      dtype=np.float32)
    frame = np.zeros((num_leds, 3), dtype=np.float32)
    for i in range(num_leds):
        t = i / max(num_leds, 1) + phase
        t -= int(t)
        frame[i] = _lerp_palette_wrap(colors, t)
    return np.clip(frame, 0, 255).astype(np.uint8)


def render_solid(color, num_leds):
    return np.tile(np.array(color, dtype=np.uint8), (num_leds, 1))


def apply_chroma(frame, chroma):
    """Desaturate toward per-pixel BT.601 luminance grey.
    Source: ITU-R BT.601-7 (2011), Y = 0.299R + 0.587G + 0.114B.
    Cited in COLOR_ENGINEERING.md — not yet validated on hardware (ledger
    entry oklch-color-solid-coverage, status: spark, confidence: high)."""
    if chroma >= 255:
        return frame
    f = frame.astype(np.float32)
    grey = (0.299 * f[:, 0] + 0.587 * f[:, 1] + 0.114 * f[:, 2])
    t = chroma / 255.0
    for c in range(3):
        f[:, c] = grey * (1 - t) + f[:, c] * t
    return np.clip(f, 0, 255).astype(np.uint8)


def apply_brightness(frame, brightness):
    br = gamma_hybrid(brightness)
    return (frame.astype(np.uint16) * br // 255).astype(np.uint8)


# ── Serial protocol ─────────────────────────────────────────────────

def send_frame(ser, frame):
    rgb = frame.flatten()
    checksum = int(np.bitwise_xor.reduce(rgb))
    packet = bytearray([0xFF, 0xAA])
    packet.extend(rgb.tobytes())
    packet.append(checksum)
    try:
        ser.write(packet)
        ser.flush()
    except serial.SerialException:
        pass


# ── State ────────────────────────────────────────────────────────────

state = {
    'mode': 'gradient',       # gradient | solid
    'palette': 'oklch_rainbow',
    'brightness': 128,
    'chroma': 255,
    'cycle_ms': 8000,
    'solid_color': [255, 140, 0],
    'phase': 0.0,
}
state_lock = threading.Lock()


def stream_loop(ser, num_leds):
    frame_interval = 1.0 / 30
    next_time = time.time()
    while True:
        with state_lock:
            mode = state['mode']
            palette = state['palette']
            brightness = state['brightness']
            chroma = state['chroma']
            cycle_ms = state['cycle_ms']
            solid = list(state['solid_color'])
            phase = state['phase']

        if mode == 'gradient':
            frame = render_gradient(palette, num_leds, phase)
        else:
            frame = render_solid(solid, num_leds)

        frame = apply_chroma(frame, chroma)
        frame = apply_brightness(frame, brightness)
        send_frame(ser, frame)

        # Advance phase
        if mode == 'gradient' and cycle_ms > 0:
            with state_lock:
                state['phase'] += frame_interval / (cycle_ms / 1000.0)
                if state['phase'] >= 1.0:
                    state['phase'] -= 1.0

        next_time += frame_interval
        sleep = next_time - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_time = time.time()


# ── HTML UI ──────────────────────────────────────────────────────────

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nano Strip</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    min-height: 100vh; padding: 20px;
    display: flex; flex-direction: column; align-items: center;
  }
  h1 { font-size: 1.8em; margin-bottom: 4px; color: #fff; }
  .section { width: 100%; max-width: 400px; margin-bottom: 20px; }
  .section-label { font-size: 0.85em; color: #888; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px; }
  .btn-group { display: flex; gap: 8px; }
  .btn {
    flex: 1; padding: 12px; border: 2px solid #333; border-radius: 8px;
    background: #16213e; color: #e0e0e0; font-size: 1em; cursor: pointer;
    transition: all 0.2s; text-align: center;
  }
  .btn:hover { border-color: #555; }
  .btn.active { border-color: #e94560; background: #e94560; color: #fff; }
  .slider-container { display: flex; align-items: center; gap: 12px; }
  .slider-container input[type=range] { flex: 1; accent-color: #e94560; }
  .slider-val { min-width: 40px; text-align: right; font-variant-numeric: tabular-nums; }
  .palette-group-label { font-size: 0.75em; color: #666; text-transform: uppercase;
    letter-spacing: 1px; margin: 10px 0 4px 0; }
  .palette-group-label:first-child { margin-top: 0; }
  .palette-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .palette-btn {
    height: 48px; border-radius: 8px; border: 2px solid #333;
    cursor: pointer; transition: all 0.2s;
  }
  .palette-btn.active { border-color: #e94560; box-shadow: 0 0 8px rgba(233,69,96,0.5); }
  .palette-btn:hover { border-color: #555; }
  .color-row { display: flex; align-items: center; gap: 12px; }
  .color-row input[type=color] {
    width: 60px; height: 40px; border: 2px solid #333; border-radius: 8px;
    background: #16213e; cursor: pointer; padding: 2px;
  }
  .color-hex { font-family: monospace; font-size: 1.1em; color: #ccc; }
  .status {
    margin-top: 20px; padding: 12px; background: #0f3460; border-radius: 8px;
    font-size: 0.85em; text-align: center; width: 100%; max-width: 400px;
  }
  .hidden { display: none; }
</style>
</head>
<body>
<h1>Nano Strip</h1>

<div class="section">
  <div class="section-label">Mode</div>
  <div class="btn-group">
    <button class="btn" id="btn-gradient" onclick="setMode('gradient')">Gradient</button>
    <button class="btn" id="btn-solid" onclick="setMode('solid')">Solid</button>
  </div>
</div>

<div class="section">
  <div class="section-label">Brightness</div>
  <div class="slider-container">
    <input type="range" id="brightness" min="0" max="255" value="128"
      oninput="document.getElementById('br-val').textContent=this.value; postDebounced('/brightness','brightness',this.value)">
    <span class="slider-val" id="br-val">128</span>
  </div>
</div>

<div class="section">
  <div class="section-label">Chroma</div>
  <div class="slider-container">
    <input type="range" id="chroma" min="0" max="255" value="255"
      oninput="document.getElementById('ch-val').textContent=this.value; postDebounced('/chroma','chroma',this.value)">
    <span class="slider-val" id="ch-val">255</span>
  </div>
</div>

<div class="section" id="sec-cycletime">
  <div class="section-label">Cycle Time</div>
  <div class="slider-container">
    <input type="range" id="cycletime" min="0" max="1000" step="1" value="71"
      oninput="document.getElementById('ct-val').textContent=fmtCycle(sliderToMs(this.value)); postDebounced('/cycletime','cycletime',sliderToMs(this.value))">
    <span class="slider-val" id="ct-val">8.0s</span>
  </div>
</div>

<div class="section" id="sec-palette">
  <div class="section-label">Palette</div>
  <div id="palette-container"></div>
</div>

<div class="section hidden" id="sec-solid">
  <div class="section-label">Color</div>
  <div class="color-row">
    <input type="color" id="colorpicker" value="#ff8c00"
      oninput="setSolid(this.value)">
    <span class="color-hex" id="color-hex">#ff8c00</span>
  </div>
</div>

<div class="status" id="status">Streaming...</div>

<script>
var GRADIENTS = [
  {id:'sap_flow', css:'linear-gradient(90deg, rgb(0,40,0), rgb(0,70,0), rgb(10,110,10), rgb(34,180,34), rgb(80,255,80))'},
  {id:'oklch_rainbow', css:'linear-gradient(90deg, rgb(251,3,64), rgb(188,82,1), rgb(2,155,72), rgb(3,129,252), rgb(184,1,169))'},
  {id:'red_blue', css:'linear-gradient(90deg, rgb(251,3,64), rgb(252,30,140), rgb(126,1,145), rgb(29,0,79), rgb(26,48,250))'},
  {id:'cyan_gold', css:'linear-gradient(90deg, rgb(2,144,145), rgb(29,118,252), rgb(53,0,97), rgb(150,1,7), rgb(166,100,1))'},
  {id:'green_purple', css:'linear-gradient(90deg, rgb(2,163,12), rgb(2,149,108), rgb(2,135,212), rgb(30,67,251), rgb(40,0,86))'},
  {id:'orange_teal', css:'linear-gradient(90deg, rgb(137,27,0), rgb(188,82,1), rgb(116,119,1), rgb(2,163,14), rgb(2,149,112))'},
  {id:'magenta_cyan', css:'linear-gradient(90deg, rgb(252,20,162), rgb(148,1,6), rgb(186,90,1), rgb(26,155,1), rgb(2,144,145))'},
  {id:'sunset_sky', css:'linear-gradient(90deg, rgb(50,40,130), rgb(125,50,115), rgb(210,60,55), rgb(175,60,5), rgb(210,110,15))'}
];
var container = document.getElementById('palette-container');
var allPaletteIds = [];
function buildGrid(items) {
  var grid = document.createElement('div');
  grid.className = 'palette-grid';
  items.forEach(function(p) {
    allPaletteIds.push(p.id);
    var btn = document.createElement('button');
    btn.className = 'palette-btn';
    btn.id = 'btn-' + p.id;
    btn.style.background = p.css;
    btn.onclick = function() { setPalette(p.id); };
    grid.appendChild(btn);
  });
  return grid;
}
container.appendChild(buildGrid(GRADIENTS));

function sliderToMs(pos) {
  pos = Number(pos);
  if (pos <= 600) return Math.round(1000 + pos * (59000 / 600));
  return Math.round(60000 * Math.pow(10, (pos - 600) / 400));
}
function fmtCycle(ms) {
  var s = ms / 1000;
  if (s < 60) return s.toFixed(1) + 's';
  var m = Math.floor(s / 60), sec = Math.round(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}
function post(path, key, val) {
  fetch(path, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: key+'='+val});
}
var _timers = {};
function postDebounced(path, key, val) {
  if (_timers[path]) clearTimeout(_timers[path]);
  _timers[path] = setTimeout(function() { post(path, key, val); }, 50);
}
function setMode(m) {
  post('/mode', 'mode', m);
  document.getElementById('btn-gradient').classList.toggle('active', m==='gradient');
  document.getElementById('btn-solid').classList.toggle('active', m==='solid');
  document.getElementById('sec-palette').classList.toggle('hidden', m!=='gradient');
  document.getElementById('sec-cycletime').classList.toggle('hidden', m!=='gradient');
  document.getElementById('sec-solid').classList.toggle('hidden', m!=='solid');
}
function setPalette(p) {
  post('/palette', 'palette', p);
  allPaletteIds.forEach(function(pid) {
    var el = document.getElementById('btn-' + pid);
    if (el) el.classList.toggle('active', pid === p);
  });
}
function setSolid(hex) {
  document.getElementById('color-hex').textContent = hex;
  post('/solid', 'color', hex);
}

function pollStatus() {
  fetch('/status').then(function(r){return r.json()}).then(function(s) {
    document.getElementById('btn-gradient').classList.toggle('active', s.mode==='gradient');
    document.getElementById('btn-solid').classList.toggle('active', s.mode==='solid');
    document.getElementById('sec-palette').classList.toggle('hidden', s.mode!=='gradient');
    document.getElementById('sec-cycletime').classList.toggle('hidden', s.mode!=='gradient');
    document.getElementById('sec-solid').classList.toggle('hidden', s.mode!=='solid');
    document.getElementById('brightness').value = s.brightness;
    document.getElementById('br-val').textContent = s.brightness;
    document.getElementById('chroma').value = s.chroma;
    document.getElementById('ch-val').textContent = s.chroma;
    allPaletteIds.forEach(function(pid) {
      var el = document.getElementById('btn-' + pid);
      if (el) el.classList.toggle('active', s.palette === pid);
    });
    document.getElementById('status').textContent = 'Streaming to ' + s.port + ' (' + s.leds + ' LEDs)';
  }).catch(function() {
    document.getElementById('status').textContent = 'Disconnected';
  });
}
pollStatus();
setInterval(pollStatus, 5000);
</script>
</body>
</html>'''


# ── HTTP handler ─────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence request logs

    def _parse_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return parse_qs(self.rfile.read(length).decode())

    def _ok(self, body=b'', content_type='text/plain'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self._ok(HTML.encode(), 'text/html')
        elif path == '/status':
            with state_lock:
                info = {
                    'mode': state['mode'],
                    'palette': state['palette'],
                    'brightness': state['brightness'],
                    'chroma': state['chroma'],
                    'cycleTime': state['cycle_ms'],
                    'port': self.server.serial_port,
                    'leds': self.server.num_leds,
                }
            self._ok(json.dumps(info).encode(), 'application/json')
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        params = self._parse_body()
        with state_lock:
            if path == '/palette':
                p = params.get('palette', ['oklch_rainbow'])[0]
                if p in PALETTES:
                    state['palette'] = p
                    state['mode'] = 'gradient'
            elif path == '/brightness':
                state['brightness'] = max(0, min(255, int(params.get('brightness', [128])[0])))
            elif path == '/chroma':
                state['chroma'] = max(0, min(255, int(params.get('chroma', [255])[0])))
            elif path == '/cycletime':
                state['cycle_ms'] = max(100, int(params.get('cycletime', [8000])[0]))
            elif path == '/mode':
                m = params.get('mode', ['gradient'])[0]
                if m in ('gradient', 'solid'):
                    state['mode'] = m
            elif path == '/solid':
                hex_color = params.get('color', ['#ffffff'])[0].lstrip('#')
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                state['solid_color'] = [r, g, b]
                state['mode'] = 'solid'
        self._ok()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Gradient picker → Nano serial streamer')
    parser.add_argument('--port', default='/dev/cu.usbserial-1240',
                        help='Serial port (default: /dev/cu.usbserial-1240)')
    parser.add_argument('--leds', type=int, default=300,
                        help='Number of LEDs (default: 300)')
    parser.add_argument('--web-port', type=int, default=8070,
                        help='Web server port (default: 8070)')
    parser.add_argument('--no-open', action='store_true',
                        help="Don't auto-open browser")
    args = parser.parse_args()

    print(f"Opening {args.port} at 1Mbps...")
    ser = serial.Serial(args.port, 1000000, timeout=1)
    time.sleep(2.0)  # bootloader timeout
    print("Serial ready.")

    # Start streaming thread
    t = threading.Thread(target=stream_loop, args=(ser, args.leds), daemon=True)
    t.start()

    # Start web server
    server = HTTPServer(('127.0.0.1', args.web_port), Handler)
    server.serial_port = args.port
    server.num_leds = args.leds
    url = f'http://localhost:{args.web_port}'
    print(f"Gradient picker: {url}")
    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        ser.close()


if __name__ == '__main__':
    main()
