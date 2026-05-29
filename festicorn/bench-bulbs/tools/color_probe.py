#!/usr/bin/env python3
"""
RGBW color probe — set all bench-bulbs LEDs to a raw RGBW value.
Opens a browser UI with sliders for RGBW, brightness, dithering, and FPS.

Serial commands:
  #RRGGBBWW  — set raw color (hex)
  !Bxxx      — set brightness 0-255
  !D0 / !D1  — dither off/on
  !Fxxx      — set dither FPS

Usage:
  python color_probe.py [serial_port]
"""

import sys
import serial
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = '/dev/cu.usbserial-0001'
BAUD = 460800
HTTP_PORT = 8321

ser = None

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>bench-bulbs color probe</title>
<style>
  body { background: #1a1a1a; color: #ccc; font-family: 'Menlo', monospace;
         max-width: 520px; margin: 40px auto; padding: 0 16px; }
  h2 { margin-bottom: 4px; }
  .section { border-top: 1px solid #333; padding-top: 12px; margin-top: 16px; }
  .ch { display: flex; align-items: center; margin: 10px 0; }
  .ch label { width: 30px; font-size: 18px; font-weight: bold; }
  .ch input[type=range] { flex: 1; margin: 0 12px; accent-color: var(--color); }
  .ch input[type=number] { width: 55px; background: #333; color: white; border: 1px solid #555;
    font-family: inherit; font-size: 14px; text-align: center; padding: 4px; }
  #preview { width: 100%; height: 50px; border: 1px solid #444; margin: 12px 0; }
  #hex { background: #333; color: white; border: 1px solid #555; font-family: inherit;
    font-size: 16px; padding: 6px 10px; width: 140px; }
  .ctrl { display: flex; align-items: center; margin: 8px 0; gap: 12px; }
  .ctrl label { width: 80px; font-size: 13px; color: #888; }
  .ctrl input[type=range] { flex: 1; }
  .ctrl input[type=number] { width: 55px; background: #333; color: white; border: 1px solid #555;
    font-family: inherit; font-size: 13px; text-align: center; padding: 3px; }
  .toggle { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
  .toggle button { background: #333; color: #ccc; border: 1px solid #555; padding: 6px 16px;
    font-family: inherit; font-size: 13px; cursor: pointer; }
  .toggle button.on { background: #2a6; color: white; border-color: #2a6; }
  #status { color: #666; font-size: 11px; margin-top: 12px; }
  .effective { color: #666; font-size: 11px; margin: 4px 0; }
</style></head><body>
<h2>color probe</h2>

<div class="ch" style="--color:#f44"><label>R</label>
  <input type="range" min="0" max="255" value="0" id="sR" oninput="sync('R')">
  <input type="number" min="0" max="255" value="0" id="nR" onchange="sync('R',true)"></div>
<div class="ch" style="--color:#4f4"><label>G</label>
  <input type="range" min="0" max="255" value="0" id="sG" oninput="sync('G')">
  <input type="number" min="0" max="255" value="0" id="nG" onchange="sync('G',true)"></div>
<div class="ch" style="--color:#48f"><label>B</label>
  <input type="range" min="0" max="255" value="0" id="sB" oninput="sync('B')">
  <input type="number" min="0" max="255" value="0" id="nB" onchange="sync('B',true)"></div>
<div class="ch" style="--color:#ccc"><label>W</label>
  <input type="range" min="0" max="255" value="0" id="sW" oninput="sync('W')">
  <input type="number" min="0" max="255" value="0" id="nW" onchange="sync('W',true)"></div>

<div style="margin:12px 0">Hex: <input id="hex" value="#00000000"
  onkeydown="if(event.key==='Enter')fromHex()"></div>
<div id="preview"></div>

<div class="section">
  <div class="ctrl">
    <label>Brightness</label>
    <input type="range" min="0" max="255" value="255" id="sBrt" oninput="syncBrt()">
    <input type="number" min="0" max="255" value="255" id="nBrt" onchange="syncBrt(true)">
  </div>
  <div class="effective" id="effValues"></div>

  <div class="toggle">
    <label style="width:80px;font-size:13px;color:#888">Dither</label>
    <button id="btnDither" onclick="toggleDither()">OFF</button>
  </div>

  <div class="ctrl" id="fpsRow" style="display:none">
    <label>FPS</label>
    <input type="range" min="1" max="500" value="150" id="sFps" oninput="syncFps()">
    <input type="number" min="1" max="500" value="150" id="nFps" onchange="syncFps(true)">
  </div>
</div>

<div id="status">ready</div>

<script>
let ditherOn = false;
function v(ch){ return parseInt(document.getElementById('s'+ch).value); }
function sync(ch, fromNum){
  if(fromNum) document.getElementById('s'+ch).value = document.getElementById('n'+ch).value;
  else document.getElementById('n'+ch).value = document.getElementById('s'+ch).value;
  sendColor();
}
function hex2(n){ return n.toString(16).padStart(2,'0').toUpperCase(); }

function sendColor(){
  let r=v('R'),g=v('G'),b=v('B'),w=v('W');
  let h='#'+hex2(r)+hex2(g)+hex2(b)+hex2(w);
  document.getElementById('hex').value=h;
  let brt=parseInt(document.getElementById('sBrt').value);
  let er=Math.round(r*brt/255), eg=Math.round(g*brt/255),
      eb=Math.round(b*brt/255), ew=Math.round(w*brt/255);
  let pr=Math.min(255,er+ew),pg=Math.min(255,eg+ew),pb=Math.min(255,eb+ew);
  document.getElementById('preview').style.background='rgb('+pr+','+pg+','+pb+')';
  document.getElementById('effValues').textContent=
    'Effective: R='+er+' G='+eg+' B='+eb+' W='+ew+
    (ditherOn?' (dithered, sub-byte)':er+eg+eb+ew>0&&brt<255?' (truncated, no dither)':'');
  fetch('/color?r='+r+'&g='+g+'&b='+b+'&w='+w);
}

function syncBrt(fromNum){
  if(fromNum) document.getElementById('sBrt').value = document.getElementById('nBrt').value;
  else document.getElementById('nBrt').value = document.getElementById('sBrt').value;
  let brt=parseInt(document.getElementById('sBrt').value);
  fetch('/brt?v='+brt);
  sendColor();
}

function toggleDither(){
  ditherOn = !ditherOn;
  let btn = document.getElementById('btnDither');
  btn.textContent = ditherOn ? 'ON' : 'OFF';
  btn.className = ditherOn ? 'on' : '';
  document.getElementById('fpsRow').style.display = ditherOn ? 'flex' : 'none';
  fetch('/dither?v='+(ditherOn?1:0));
  sendColor();
}

function syncFps(fromNum){
  if(fromNum) document.getElementById('sFps').value = document.getElementById('nFps').value;
  else document.getElementById('nFps').value = document.getElementById('sFps').value;
  let fps=parseInt(document.getElementById('sFps').value);
  fetch('/fps?v='+fps);
}

function fromHex(){
  let h=document.getElementById('hex').value.replace('#','');
  if(h.length>=6){
    try{
      document.getElementById('sR').value=document.getElementById('nR').value=parseInt(h.slice(0,2),16);
      document.getElementById('sG').value=document.getElementById('nG').value=parseInt(h.slice(2,4),16);
      document.getElementById('sB').value=document.getElementById('nB').value=parseInt(h.slice(4,6),16);
      if(h.length>=8) document.getElementById('sW').value=document.getElementById('nW').value=parseInt(h.slice(6,8),16);
      sendColor();
    }catch(e){}
  }
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._respond(200, 'text/html', HTML)
        elif self.path.startswith('/color?'):
            p = self._params()
            r, g, b, w = int(p['r']), int(p['g']), int(p['b']), int(p['w'])
            cmd = f'#{r:02X}{g:02X}{b:02X}{w:02X}\n'
            ser.write(cmd.encode())
            self._respond(200, 'text/plain', cmd.strip())
        elif self.path.startswith('/brt?'):
            v = int(self._params()['v'])
            ser.write(f'!B{v}\n'.encode())
            self._respond(200, 'text/plain', f'brightness={v}')
        elif self.path.startswith('/dither?'):
            v = int(self._params()['v'])
            ser.write(f'!D{v}\n'.encode())
            self._respond(200, 'text/plain', f'dither={v}')
        elif self.path.startswith('/fps?'):
            v = int(self._params()['v'])
            ser.write(f'!F{v}\n'.encode())
            self._respond(200, 'text/plain', f'fps={v}')
        else:
            self._respond(404, 'text/plain', 'not found')

    def _params(self):
        return dict(p.split('=') for p in self.path.split('?')[1].split('&'))

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def log_message(self, format, *args):
        pass


def main():
    global ser
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    ser = serial.Serial(port, BAUD, timeout=0.1)
    print(f"Serial: {port}")
    print(f"UI: http://localhost:{HTTP_PORT}")

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
