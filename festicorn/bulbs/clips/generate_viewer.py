#!/usr/bin/env python3
"""Generate a standalone HTML waveform viewer with embedded audio data."""

import wave
import numpy as np
import base64
import json
import os

CLIPS_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = ["hmUmmmmmm.wav", "humm.wav", "singing.wav", "speaking.wav"]
ENVELOPE_WINDOW = 512


def read_wav_float(path):
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        max_val = 2147483648.0
    elif sampwidth == 1:
        dtype = np.uint8
        max_val = 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if sampwidth == 1:
        samples = samples - 128.0
    samples /= max_val

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, framerate


def compute_envelope(samples, window=ENVELOPE_WINDOW):
    n_windows = len(samples) // window
    envelope = np.zeros(n_windows)
    for i in range(n_windows):
        chunk = samples[i * window : (i + 1) * window]
        envelope[i] = np.max(np.abs(chunk))
    return envelope


def get_base64_wav(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def generate_html():
    clips_data = []
    for fname in FILES:
        path = os.path.join(CLIPS_DIR, fname)
        samples, framerate = read_wav_float(path)
        envelope = compute_envelope(samples)
        duration = len(samples) / framerate
        b64 = get_base64_wav(path)

        clips_data.append({
            "name": fname,
            "duration": round(duration, 3),
            "sampleRate": framerate,
            "envelopeWindow": ENVELOPE_WINDOW,
            "envelope": [round(float(v), 4) for v in envelope],
            "audioBase64": b64,
        })

    clips_json = json.dumps(clips_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clip Waveform Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #1a1410;
    color: #e8dcc8;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    padding: 24px;
}}
h1 {{
    font-size: 18px;
    color: #f0a050;
    margin-bottom: 24px;
    letter-spacing: 1px;
}}
.clip {{
    background: #241e18;
    border: 1px solid #3a3028;
    border-radius: 8px;
    margin-bottom: 20px;
    padding: 16px;
}}
.clip-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}}
.clip-name {{
    font-size: 14px;
    color: #f0a050;
    font-weight: 600;
}}
.clip-duration {{
    font-size: 12px;
    color: #a09080;
}}
.clip-time {{
    font-size: 12px;
    color: #c0a880;
    margin-left: auto;
}}
.controls {{
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}}
button {{
    background: #3a2e20;
    color: #f0a050;
    border: 1px solid #5a4a38;
    border-radius: 4px;
    padding: 6px 14px;
    font-family: inherit;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s;
}}
button:hover {{
    background: #4a3e30;
}}
button.active {{
    background: #f0a050;
    color: #1a1410;
}}
canvas {{
    width: 100%;
    height: 100px;
    border-radius: 4px;
    cursor: pointer;
    display: block;
}}
</style>
</head>
<body>
<h1>festicorn / bulbs / clips</h1>
<div id="container"></div>

<script>
const CLIPS_DATA = {clips_json};

const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
const DPR = window.devicePixelRatio || 1;

const state = CLIPS_DATA.map(() => ({{
    playing: false,
    source: null,
    startTime: 0,
    startOffset: 0,
    buffer: null,
    animFrame: null,
}}));

function base64ToArrayBuffer(b64) {{
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
}}

function formatTime(sec) {{
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    return m > 0 ? m + ':' + s.padStart(4, '0') : s + 's';
}}

async function decodeAll() {{
    for (let i = 0; i < CLIPS_DATA.length; i++) {{
        const clip = CLIPS_DATA[i];
        const buf = base64ToArrayBuffer(clip.audioBase64);
        state[i].buffer = await audioCtx.decodeAudioData(buf);
    }}
}}

function drawWaveform(canvas, envelope, progress) {{
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const n = envelope.length;

    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = '#2a2218';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += h / 4) {{
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }}

    const mid = h / 2;
    const barW = Math.max(1, w / n);

    for (let i = 0; i < n; i++) {{
        const x = (i / n) * w;
        const amp = envelope[i];
        const barH = amp * mid * 0.95;

        const frac = i / n;
        if (progress !== null && frac <= progress) {{
            ctx.fillStyle = '#f0a050';
        }} else {{
            ctx.fillStyle = '#6a5a48';
        }}

        ctx.fillRect(x, mid - barH, Math.max(barW - 0.5, 0.5), barH * 2);
    }}

    // Cursor line
    if (progress !== null && progress >= 0 && progress <= 1) {{
        const cx = progress * w;
        ctx.strokeStyle = '#ff8020';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, h);
        ctx.stroke();
    }}
}}

function buildUI() {{
    const container = document.getElementById('container');

    CLIPS_DATA.forEach((clip, idx) => {{
        const div = document.createElement('div');
        div.className = 'clip';

        const header = document.createElement('div');
        header.className = 'clip-header';

        const name = document.createElement('span');
        name.className = 'clip-name';
        name.textContent = clip.name;

        const dur = document.createElement('span');
        dur.className = 'clip-duration';
        dur.textContent = clip.duration.toFixed(1) + 's';

        const timeDisp = document.createElement('span');
        timeDisp.className = 'clip-time';
        timeDisp.id = 'time-' + idx;
        timeDisp.textContent = '0.0s / ' + clip.duration.toFixed(1) + 's';

        header.appendChild(name);
        header.appendChild(dur);
        header.appendChild(timeDisp);

        const controls = document.createElement('div');
        controls.className = 'controls';

        const playBtn = document.createElement('button');
        playBtn.textContent = 'Play';
        playBtn.id = 'play-' + idx;
        playBtn.onclick = () => togglePlay(idx);

        const stopBtn = document.createElement('button');
        stopBtn.textContent = 'Stop';
        stopBtn.onclick = () => stopClip(idx);

        controls.appendChild(playBtn);
        controls.appendChild(stopBtn);

        const canvas = document.createElement('canvas');
        canvas.id = 'canvas-' + idx;

        div.appendChild(header);
        div.appendChild(controls);
        div.appendChild(canvas);
        container.appendChild(div);

        // Size canvas for HiDPI
        const rect = canvas.getBoundingClientRect();
        // Defer sizing to after layout
        requestAnimationFrame(() => {{
            const rect2 = canvas.getBoundingClientRect();
            canvas.width = rect2.width * DPR;
            canvas.height = 100 * DPR;
            canvas.getContext('2d').scale(DPR, DPR);
            canvas.style.height = '100px';
            drawWaveform(canvas, clip.envelope, null);
        }});

        // Click on canvas to seek
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const frac = (e.clientX - rect.left) / rect.width;
            seekClip(idx, frac * clip.duration);
        }});
    }});
}}

function togglePlay(idx) {{
    if (state[idx].playing) {{
        pauseClip(idx);
    }} else {{
        playClip(idx);
    }}
}}

function playClip(idx) {{
    if (!state[idx].buffer) return;
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const s = state[idx];
    const source = audioCtx.createBufferSource();
    source.buffer = s.buffer;
    source.connect(audioCtx.destination);

    s.source = source;
    s.startTime = audioCtx.currentTime;
    s.playing = true;

    source.start(0, s.startOffset);
    source.onended = () => {{
        if (s.playing) {{
            s.playing = false;
            s.startOffset = 0;
            document.getElementById('play-' + idx).textContent = 'Play';
            document.getElementById('play-' + idx).classList.remove('active');
            cancelAnimationFrame(s.animFrame);
            const canvas = document.getElementById('canvas-' + idx);
            drawWaveform(canvas, CLIPS_DATA[idx].envelope, null);
        }}
    }};

    document.getElementById('play-' + idx).textContent = 'Pause';
    document.getElementById('play-' + idx).classList.add('active');

    function animate() {{
        if (!s.playing) return;
        const elapsed = audioCtx.currentTime - s.startTime + s.startOffset;
        const progress = Math.min(elapsed / CLIPS_DATA[idx].duration, 1);
        const canvas = document.getElementById('canvas-' + idx);

        // Rescale if needed (e.g., window resize)
        const rect = canvas.getBoundingClientRect();
        if (Math.abs(canvas.width - rect.width * DPR) > 2) {{
            canvas.width = rect.width * DPR;
            canvas.height = 100 * DPR;
            canvas.getContext('2d').scale(DPR, DPR);
        }}

        drawWaveform(canvas, CLIPS_DATA[idx].envelope, progress);

        document.getElementById('time-' + idx).textContent =
            formatTime(elapsed) + ' / ' + CLIPS_DATA[idx].duration.toFixed(1) + 's';

        s.animFrame = requestAnimationFrame(animate);
    }}
    animate();
}}

function pauseClip(idx) {{
    const s = state[idx];
    if (!s.playing) return;
    s.playing = false;
    const elapsed = audioCtx.currentTime - s.startTime;
    s.startOffset += elapsed;
    try {{ s.source.onended = null; s.source.stop(); }} catch(e) {{}}
    document.getElementById('play-' + idx).textContent = 'Play';
    document.getElementById('play-' + idx).classList.remove('active');
    cancelAnimationFrame(s.animFrame);
}}

function stopClip(idx) {{
    const s = state[idx];
    s.playing = false;
    s.startOffset = 0;
    try {{ s.source.onended = null; s.source.stop(); }} catch(e) {{}}
    document.getElementById('play-' + idx).textContent = 'Play';
    document.getElementById('play-' + idx).classList.remove('active');
    cancelAnimationFrame(s.animFrame);
    const canvas = document.getElementById('canvas-' + idx);
    drawWaveform(canvas, CLIPS_DATA[idx].envelope, null);
    document.getElementById('time-' + idx).textContent =
        '0.0s / ' + CLIPS_DATA[idx].duration.toFixed(1) + 's';
}}

function seekClip(idx, time) {{
    const wasPlaying = state[idx].playing;
    if (wasPlaying) {{
        try {{ state[idx].source.onended = null; state[idx].source.stop(); }} catch(e) {{}}
        state[idx].playing = false;
        cancelAnimationFrame(state[idx].animFrame);
    }}
    state[idx].startOffset = Math.max(0, Math.min(time, CLIPS_DATA[idx].duration));
    if (wasPlaying) {{
        playClip(idx);
    }} else {{
        const progress = state[idx].startOffset / CLIPS_DATA[idx].duration;
        const canvas = document.getElementById('canvas-' + idx);
        drawWaveform(canvas, CLIPS_DATA[idx].envelope, progress);
        document.getElementById('time-' + idx).textContent =
            formatTime(state[idx].startOffset) + ' / ' + CLIPS_DATA[idx].duration.toFixed(1) + 's';
    }}
}}

// Handle window resize
window.addEventListener('resize', () => {{
    CLIPS_DATA.forEach((clip, idx) => {{
        const canvas = document.getElementById('canvas-' + idx);
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * DPR;
        canvas.height = 100 * DPR;
        canvas.getContext('2d').scale(DPR, DPR);
        const progress = state[idx].playing
            ? (audioCtx.currentTime - state[idx].startTime + state[idx].startOffset) / clip.duration
            : (state[idx].startOffset > 0 ? state[idx].startOffset / clip.duration : null);
        drawWaveform(canvas, clip.envelope, progress);
    }});
}});

buildUI();
decodeAll();
</script>
</body>
</html>"""

    output_path = os.path.join(CLIPS_DIR, "view.html")
    with open(output_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Generated {output_path}")
    print(f"File size: {size_mb:.1f} MB")
    for clip in clips_data:
        print(f"  {clip['name']}: {clip['duration']:.1f}s, {len(clip['envelope'])} envelope points")


if __name__ == "__main__":
    generate_html()
