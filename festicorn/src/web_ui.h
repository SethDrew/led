#pragma once

// Minimal fallback UI served from PROGMEM when LittleFS has no index.html.
// Also serves as the /upload page for pushing new UI versions over the air.

const char UPLOAD_PAGE[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Festicorn — Upload</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    min-height: 100vh; padding: 20px;
    display: flex; flex-direction: column; align-items: center;
  }
  h1 { font-size: 1.8em; margin-bottom: 24px; color: #fff; }
  .card {
    width: 100%; max-width: 400px; padding: 20px;
    background: #16213e; border-radius: 12px; margin-bottom: 20px;
  }
  .card h2 { font-size: 1em; color: #888; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 12px; }
  input[type=file] { margin-bottom: 12px; width: 100%; }
  .btn {
    display: block; width: 100%; padding: 12px; border: 2px solid #333;
    border-radius: 8px; background: #e94560; color: #fff; font-size: 1em;
    cursor: pointer; text-align: center; transition: all 0.2s;
  }
  .btn:hover { background: #c73652; }
  .btn-ctrl {
    flex: 1; padding: 12px; border: 2px solid #333; border-radius: 8px;
    background: #16213e; color: #e0e0e0; font-size: 1em; cursor: pointer;
    transition: all 0.2s; text-align: center;
  }
  .btn-ctrl:hover { border-color: #555; }
  .btn-ctrl.active { border-color: #e94560; background: #e94560; color: #fff; }
  .btn-group { display: flex; gap: 8px; margin-bottom: 12px; }
  .slider-container { display: flex; align-items: center; gap: 12px; }
  .slider-container input[type=range] { flex: 1; accent-color: #e94560; }
  .slider-val { min-width: 40px; text-align: right; font-variant-numeric: tabular-nums; }
  .msg { margin-top: 12px; font-size: 0.9em; text-align: center; }
  .msg.ok { color: #4caf50; }
  .msg.err { color: #e94560; }
</style>
</head>
<body>
<h1>&#x1f984; Festicorn</h1>

<div class="card">
  <h2>Upload UI</h2>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" accept=".html,.htm" required>
    <button class="btn" type="submit">Upload</button>
  </form>
  <div class="msg" id="msg"></div>
</div>

<div class="card">
  <h2>Controls</h2>
  <div class="btn-group">
    <button class="btn-ctrl" id="btn-rainbow" onclick="setEffect('rainbow')">Rainbow</button>
    <button class="btn-ctrl" id="btn-gradient" onclick="setEffect('gradient')">Gradient</button>
  </div>
  <div class="slider-container">
    <span style="font-size:0.85em;color:#888">Brightness</span>
    <input type="range" id="brightness" min="0" max="255" value="128"
      oninput="document.getElementById('br-val').textContent=this.value"
      onchange="postVal('/brightness','brightness',this.value)">
    <span class="slider-val" id="br-val">128</span>
  </div>
</div>

<script>
function postVal(path, key, val) {
  fetch(path, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: key+'='+val});
}
function setEffect(e) {
  fetch('/effect', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: 'effect='+e}).then(()=>pollStatus());
}
function updateUI(s) {
  document.getElementById('btn-rainbow').classList.toggle('active', s.effect==='rainbow');
  document.getElementById('btn-gradient').classList.toggle('active', s.effect==='gradient');
  document.getElementById('brightness').value = s.brightness;
  document.getElementById('br-val').textContent = s.brightness;
}
function pollStatus() {
  fetch('/status').then(r=>r.json()).then(updateUI).catch(()=>{});
}
pollStatus();

// Show upload result from query param
const p = new URLSearchParams(location.search);
if (p.has('ok')) {
  document.getElementById('msg').className = 'msg ok';
  document.getElementById('msg').textContent = 'Upload successful! Redirecting...';
  setTimeout(() => location.href = '/', 1500);
}
if (p.has('err')) {
  document.getElementById('msg').className = 'msg err';
  document.getElementById('msg').textContent = 'Upload failed: ' + p.get('err');
}
</script>
</body>
</html>
)rawliteral";
