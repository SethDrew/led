#!/usr/bin/env python3
"""
Web-based audio analysis viewer.

Browser-based replacement for the matplotlib viewer with perfect audio sync,
three view tabs (Analysis, Annotations, Stems), and a file picker.

Usage (via segment.py):
    python segment.py web              # Opens browser, explore all files
    python segment.py web --port 8080  # Fixed port

Architecture:
    Python server pre-renders analysis panels as PNGs using matplotlib Agg backend.
    Browser uses <audio> element for playback and requestAnimationFrame for cursor sync.
    audio.currentTime is sample-accurate — zero drift by definition.
"""

import io
import json
import os
import re
import subprocess
import sys
import threading
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from string import Template
from urllib.parse import urlparse, parse_qs, unquote

# Force Agg backend BEFORE any viewer imports (which import pyplot at module level)
import matplotlib
matplotlib.use('Agg')

SEGMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audio-segments')
DPI = 100

# ── Caches ────────────────────────────────────────────────────────────

_render_cache = {}      # (filepath, tab) -> (png_bytes, headers_dict)
_file_list_cache = None  # JSON-serializable list
_demucs_status = {}     # filepath -> bool (True = ready)
_demucs_locks = {}      # filepath -> threading.Lock
_recording = None       # {'stream': sd.InputStream, 'frames': list} or None


# ── File Discovery ────────────────────────────────────────────────────

def discover_files():
    """Scan audio-segments/ and harmonix/ for WAV files."""
    global _file_list_cache
    if _file_list_cache is not None:
        return _file_list_cache

    files = []
    segments_path = Path(SEGMENTS_DIR)

    # User clips
    for wav in sorted(segments_path.glob('*.wav')):
        ann_path = wav.with_suffix('.annotations.yaml')
        dur = _get_wav_duration(str(wav))
        files.append({
            'name': wav.name,
            'path': wav.name,
            'duration': dur,
            'has_annotations': ann_path.exists(),
            'group': 'user clips',
        })

    # Harmonix clips
    harmonix_dir = segments_path / 'harmonix'
    if harmonix_dir.exists():
        for wav in sorted(harmonix_dir.glob('*.wav')):
            ann_path = wav.with_suffix('.annotations.yaml')
            dur = _get_wav_duration(str(wav))
            files.append({
                'name': wav.name,
                'path': f'harmonix/{wav.name}',
                'duration': dur,
                'has_annotations': ann_path.exists(),
                'group': 'harmonix',
            })

    _file_list_cache = files
    return files


def _get_wav_duration(filepath):
    """Get WAV duration in seconds using stdlib wave."""
    try:
        with wave.open(filepath, 'rb') as wf:
            return round(wf.getnframes() / wf.getframerate(), 2)
    except Exception:
        return 0.0


def _resolve_audio_path(rel_path):
    """Resolve a relative path (from /api/render/<path>) to absolute filepath."""
    # Prevent path traversal
    clean = os.path.normpath(rel_path)
    if clean.startswith('..') or os.path.isabs(clean):
        return None
    full = os.path.join(SEGMENTS_DIR, clean)
    if os.path.exists(full) and full.endswith('.wav'):
        return full
    return None


# ── Rendering ─────────────────────────────────────────────────────────

def render_analysis(filepath, with_annotations=False, features=None):
    """Render analysis panels to PNG bytes. Returns (png_bytes, pixel_mapping).
    features: dict of {name: bool} for feature visibility, or None for defaults.
    """
    feat_key = frozenset(sorted(features.items())) if features else None
    cache_key = (filepath, 'annotations' if with_annotations else 'analysis', feat_key)
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    from viewer import SyncedVisualizer

    if with_annotations:
        viz = SyncedVisualizer(filepath, show_beats=True)
    else:
        viz = SyncedVisualizer(filepath, show_beats=True,
                               annotations_path='/dev/null/none.yaml')

    # Apply feature visibility overrides
    if features is not None and hasattr(viz, 'feature_toggle'):
        for name, feat in viz.feature_toggle.items():
            visible = features.get(name, feat['visible'])
            if visible != feat['visible']:
                feat['visible'] = visible
                for artist in feat['artists']:
                    artist.set_visible(visible)
        if hasattr(viz, '_update_features_title'):
            viz._update_features_title()

    # Remove matplotlib cursor lines — browser provides its own
    for line in viz.cursor_lines:
        line.remove()

    # Replace title
    filename = Path(filepath).name
    viz.fig.suptitle(filename, fontsize=14, fontweight='bold', y=0.995)

    # Required for Agg transform init
    viz.fig.canvas.draw()

    # Extract pixel mapping from the first axis
    ax = viz.axes[0]
    x_left = ax.transData.transform((0, 0))[0]
    x_right = ax.transData.transform((viz.duration, 0))[0]

    # Get PNG dimensions
    fig_width = viz.fig.get_figwidth() * DPI
    fig_height = viz.fig.get_figheight() * DPI

    buf = io.BytesIO()
    viz.fig.savefig(buf, format='png', dpi=DPI, facecolor=viz.fig.get_facecolor())
    matplotlib.pyplot.close(viz.fig)
    png_bytes = buf.getvalue()

    headers = {
        'X-Left-Px': str(x_left),
        'X-Right-Px': str(x_right),
        'X-Png-Width': str(fig_width),
        'X-Duration': str(viz.duration),
    }

    _render_cache[cache_key] = (png_bytes, headers)
    return png_bytes, headers


def render_stems(filepath):
    """Render stem spectrograms to PNG bytes. Returns (png_bytes, pixel_mapping)."""
    cache_key = (filepath, 'stems')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    import soundfile as sf
    import librosa
    from viewer import StemVisualizer

    separated_dir = os.path.join(SEGMENTS_DIR, 'separated')
    stem_name = Path(filepath).stem
    stem_dir = os.path.join(separated_dir, 'htdemucs', stem_name)

    stem_names = ['drums', 'bass', 'vocals', 'other']
    stems_playback = {}
    stems_mono = {}
    for name in stem_names:
        stem_path = os.path.join(stem_dir, f'{name}.wav')
        stems_playback[name] = sf.read(stem_path)
        stems_mono[name] = librosa.load(stem_path, sr=None, mono=True)

    viz = StemVisualizer(filepath, stem_names, stems_playback, stems_mono)

    # Remove matplotlib cursor lines
    for line in viz.cursor_lines:
        line.remove()

    filename = Path(filepath).name
    viz.fig.suptitle(f'{filename} — Stems', fontsize=14, fontweight='bold', y=0.995)
    viz.fig.canvas.draw()

    ax = viz.axes[0]
    x_left = ax.transData.transform((0, 0))[0]
    x_right = ax.transData.transform((viz.duration, 0))[0]

    fig_width = viz.fig.get_figwidth() * DPI
    fig_height = viz.fig.get_figheight() * DPI

    buf = io.BytesIO()
    viz.fig.savefig(buf, format='png', dpi=DPI, facecolor=viz.fig.get_facecolor())
    matplotlib.pyplot.close(viz.fig)
    png_bytes = buf.getvalue()

    headers = {
        'X-Left-Px': str(x_left),
        'X-Right-Px': str(x_right),
        'X-Png-Width': str(fig_width),
        'X-Duration': str(viz.duration),
    }

    _render_cache[cache_key] = (png_bytes, headers)
    return png_bytes, headers


def render_hpss(filepath):
    """Compute HPSS, save WAVs, render 2-row spectrogram PNG."""
    cache_key = (filepath, 'hpss')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    import numpy as np
    import soundfile as sf
    import librosa
    from viewer import StemVisualizer

    stem_name = Path(filepath).stem
    hpss_dir = os.path.join(SEGMENTS_DIR, 'separated', 'hpss', stem_name)
    h_path = os.path.join(hpss_dir, 'harmonic.wav')
    p_path = os.path.join(hpss_dir, 'percussive.wav')

    # Compute and save if not cached on disk
    if not (os.path.exists(h_path) and os.path.exists(p_path)):
        os.makedirs(hpss_dir, exist_ok=True)
        print(f"[hpss] Computing for {Path(filepath).name}...")

        y_play, sr_play = sf.read(filepath)
        y_mono, sr_mono = librosa.load(filepath, sr=None, mono=True)

        # HPSS per channel for stereo playback
        if y_play.ndim == 1:
            y_play = y_play[:, np.newaxis]
        n_ch = y_play.shape[1]
        y_h_play = np.zeros_like(y_play)
        y_p_play = np.zeros_like(y_play)
        for ch in range(n_ch):
            D_ch = librosa.stft(y_play[:, ch])
            H_ch, P_ch = librosa.decompose.hpss(D_ch)
            y_h_play[:, ch] = librosa.istft(H_ch, length=y_play.shape[0])
            y_p_play[:, ch] = librosa.istft(P_ch, length=y_play.shape[0])

        if n_ch == 1:
            y_h_play = y_h_play.squeeze()
            y_p_play = y_p_play.squeeze()

        sf.write(h_path, y_h_play, sr_play)
        sf.write(p_path, y_p_play, sr_play)
        print(f"[hpss] Done: {Path(filepath).name}")

    # Load stems from disk for StemVisualizer
    stem_names = ['harmonic', 'percussive']
    stems_playback = {}
    stems_mono = {}
    for name in stem_names:
        spath = os.path.join(hpss_dir, f'{name}.wav')
        stems_playback[name] = sf.read(spath)
        stems_mono[name] = librosa.load(spath, sr=None, mono=True)

    viz = StemVisualizer(filepath, stem_names, stems_playback, stems_mono)

    for line in viz.cursor_lines:
        line.remove()

    filename = Path(filepath).name
    viz.fig.suptitle(f'{filename} — HPSS', fontsize=14, fontweight='bold', y=0.995)
    viz.fig.canvas.draw()

    ax = viz.axes[0]
    x_left = ax.transData.transform((0, 0))[0]
    x_right = ax.transData.transform((viz.duration, 0))[0]

    fig_width = viz.fig.get_figwidth() * DPI

    buf = io.BytesIO()
    viz.fig.savefig(buf, format='png', dpi=DPI, facecolor=viz.fig.get_facecolor())
    matplotlib.pyplot.close(viz.fig)
    png_bytes = buf.getvalue()

    headers = {
        'X-Left-Px': str(x_left),
        'X-Right-Px': str(x_right),
        'X-Png-Width': str(fig_width),
        'X-Duration': str(viz.duration),
    }

    _render_cache[cache_key] = (png_bytes, headers)
    return png_bytes, headers


# ── Demucs ────────────────────────────────────────────────────────────

def _stems_ready(filepath):
    """Check if all 4 demucs stems exist for this file."""
    separated_dir = os.path.join(SEGMENTS_DIR, 'separated')
    stem_name = Path(filepath).stem
    stem_dir = os.path.join(separated_dir, 'htdemucs', stem_name)
    expected = ['drums.wav', 'bass.wav', 'vocals.wav', 'other.wav']
    return all(os.path.exists(os.path.join(stem_dir, f)) for f in expected)


def _run_demucs_background(filepath):
    """Run demucs in a background thread. Updates _demucs_status when done."""
    if filepath in _demucs_locks:
        return  # Already running

    _demucs_locks[filepath] = threading.Lock()
    _demucs_status[filepath] = False

    def run():
        separated_dir = os.path.join(SEGMENTS_DIR, 'separated')
        try:
            print(f"[demucs] Starting separation for {Path(filepath).name}...")
            subprocess.run(
                [sys.executable, '-m', 'demucs', '-n', 'htdemucs',
                 '-o', separated_dir, filepath],
                check=True
            )
            print(f"[demucs] Done: {Path(filepath).name}")
        except Exception as e:
            print(f"[demucs] Error: {e}")
        finally:
            _demucs_status[filepath] = _stems_ready(filepath)
            del _demucs_locks[filepath]

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ── Recording ────────────────────────────────────────────────────────

def start_recording():
    """Start recording from BlackHole. Returns {ok: bool, error: str}."""
    global _recording
    if _recording is not None:
        return {'ok': False, 'error': 'Already recording'}

    import sounddevice as sd
    import numpy as np

    # Find BlackHole device
    device_id = None
    for i, d in enumerate(sd.query_devices()):
        if 'blackhole' in d['name'].lower() and d['max_input_channels'] >= 2:
            device_id = i
            break

    if device_id is None:
        return {'ok': False, 'error': 'BlackHole device not found'}

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        device=device_id, channels=2, samplerate=44100, callback=callback
    )
    stream.start()
    _recording = {'stream': stream, 'frames': frames}
    print("[record] Started recording from BlackHole")
    return {'ok': True}


def stop_recording(name=''):
    """Stop recording, save WAV. Returns {ok, filename, duration} or {ok: false, error}."""
    global _recording, _file_list_cache
    if _recording is None:
        return {'ok': False, 'error': 'Not recording'}

    import numpy as np
    import soundfile as sf
    import yaml
    from datetime import datetime

    stream = _recording['stream']
    frames = _recording['frames']
    stream.stop()
    stream.close()
    _recording = None

    if not frames:
        return {'ok': False, 'error': 'No audio captured'}

    audio_data = np.concatenate(frames)
    sr = 44100
    duration = round(len(audio_data) / sr, 1)

    if not name:
        name = f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Sanitize filename
    name = re.sub(r'[^\w\-.]', '_', name)
    filename = f"{name}.wav"
    filepath = os.path.join(SEGMENTS_DIR, filename)

    sf.write(filepath, audio_data, sr)
    print(f"[record] Saved {filepath} ({duration}s)")

    # Update catalog
    catalog_path = os.path.join(SEGMENTS_DIR, 'catalog.yaml')
    catalog = []
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f) or []
    catalog.append({
        'id': name,
        'filename': filename,
        'duration_seconds': duration,
        'sample_rate': sr,
        'recorded': datetime.now().strftime('%Y-%m-%d %H:%M'),
    })
    with open(catalog_path, 'w') as f:
        yaml.dump(catalog, f, default_flow_style=False)

    # Clear file list cache
    _file_list_cache = None

    return {'ok': True, 'filename': filename, 'duration': duration}


# ── HTML Generation ───────────────────────────────────────────────────

def generate_html():
    """Generate the single-page app HTML."""
    # Using string.Template to avoid JS % escaping issues
    return Template(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="icon" type="image/svg+xml" href="/favicon.ico">
<title>Audio Explorer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, system-ui, sans-serif;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}
.header {
    display: flex; align-items: center; gap: 16px; padding: 10px 20px;
    background: #16213e; border-bottom: 1px solid #333;
}
.header h1 { font-size: 16px; font-weight: 600; white-space: nowrap; }
.header select {
    background: #0f3460; color: #e0e0e0; border: 1px solid #555; border-radius: 4px;
    padding: 6px 12px; font-size: 14px; min-width: 200px; cursor: pointer;
}
.header select optgroup { background: #0f3460; }
.header .time {
    font-family: 'SF Mono', 'Menlo', monospace; font-size: 14px; color: #aaa;
    min-width: 180px;
}
.tabs {
    display: flex; gap: 0; padding: 0 20px; background: #16213e;
    border-bottom: 1px solid #333;
}
.tab {
    padding: 8px 20px; cursor: pointer; border-bottom: 2px solid transparent;
    font-size: 13px; color: #888; transition: all 0.15s;
}
.tab:hover { color: #ccc; }
.tab.active { color: #e94560; border-bottom-color: #e94560; }
.tab.disabled { color: #444; cursor: default; pointer-events: none; }
.viewer {
    flex: 1; display: flex; justify-content: center; align-items: flex-start;
    overflow: auto; padding: 12px; position: relative;
}
.img-container {
    position: relative; display: inline-block; max-width: 100%;
}
.img-container img {
    display: block; max-width: 100%; height: auto; cursor: crosshair;
}
.cursor-line {
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: rgba(255, 255, 255, 0.9); pointer-events: none;
    z-index: 10; display: none;
}
.progress-bar {
    padding: 8px 20px; background: #16213e; border-top: 1px solid #333;
    display: flex; align-items: center; gap: 12px;
}
.progress-track {
    flex: 1; height: 6px; background: #333; border-radius: 3px; cursor: pointer;
    position: relative;
}
.progress-fill {
    height: 100%; background: #e94560; border-radius: 3px; width: 0%;
    transition: none;
}
.progress-thumb {
    position: absolute; top: -5px; width: 16px; height: 16px;
    background: #e94560; border-radius: 50%; left: 0%; transform: translateX(-50%);
    cursor: grab; transition: none;
}
.controls {
    font-size: 11px; color: #666; padding: 4px 20px 8px;
    background: #16213e; text-align: center;
}
.controls kbd {
    background: #333; padding: 1px 6px; border-radius: 3px;
    font-family: monospace; font-size: 11px; color: #aaa;
}
.overlay {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(26, 26, 46, 0.85); display: flex; align-items: center;
    justify-content: center; z-index: 20;
}
.overlay-text {
    font-size: 18px; color: #e94560; animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
.play-btn {
    background: none; border: 1px solid #555; color: #e0e0e0; border-radius: 4px;
    padding: 4px 12px; cursor: pointer; font-size: 18px; line-height: 1;
}
.play-btn:hover { border-color: #e94560; }
.stem-status {
    display: none; padding: 4px 20px; background: #16213e;
    text-align: center; font-size: 12px;
}
.stem-chip {
    display: inline-block; padding: 2px 10px; margin: 0 4px;
    border-radius: 3px; font-family: monospace; cursor: pointer;
    transition: opacity 0.15s;
}
.stem-chip.active { background: #e94560; color: #fff; opacity: 1; }
.stem-chip.muted { background: #333; color: #666; opacity: 0.5; }
.info-panel {
    display: none; max-width: 900px; padding: 24px 32px; line-height: 1.6;
    font-size: 14px; color: #ccc; overflow-y: auto; max-height: 100%;
}
.info-panel h2 { color: #e94560; font-size: 18px; margin: 24px 0 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }
.info-panel h2:first-child { margin-top: 0; }
.info-panel h3 { color: #e0e0e0; font-size: 15px; margin: 16px 0 6px; }
.info-panel p { margin: 6px 0; }
.info-panel .origin { color: #888; font-style: italic; }
.info-panel .verdict { color: #4fc3f7; font-weight: 600; }
.info-panel .tag {
    display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: 11px;
    font-weight: 600; margin-left: 8px; vertical-align: middle;
}
.info-panel .tag.essential { background: #1b5e20; color: #a5d6a7; }
.info-panel .tag.standard { background: #0d47a1; color: #90caf9; }
.info-panel .tag.expendable { background: #4e342e; color: #bcaaa4; }
.info-panel .tag.custom { background: #4a148c; color: #ce93d8; }
.info-panel .tag.gap { background: #b71c1c; color: #ef9a9a; }
.info-panel .missing-table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
.info-panel .missing-table th { text-align: left; color: #e94560; border-bottom: 1px solid #333; padding: 6px 8px; }
.info-panel .missing-table td { padding: 6px 8px; border-bottom: 1px solid #222; vertical-align: top; }
.info-panel .missing-table td:first-child { color: #e0e0e0; font-weight: 600; white-space: nowrap; }
.annotation-bar {
    display: none; padding: 6px 20px; background: #1a1a2e;
    border-bottom: 1px solid #333; align-items: center; gap: 12px; font-size: 13px;
}
.annotation-bar.visible { display: flex; }
.annotation-bar label { color: #888; }
.annotation-bar input {
    background: #0f3460; color: #e0e0e0; border: 1px solid #555; border-radius: 4px;
    padding: 4px 8px; font-size: 13px; font-family: monospace; width: 120px;
}
.annotation-bar .tap-info { color: #888; }
.annotation-bar .tap-count { color: #e94560; font-weight: 600; font-family: monospace; }
.annotation-bar button {
    background: #333; color: #e0e0e0; border: 1px solid #555; border-radius: 4px;
    padding: 4px 12px; cursor: pointer; font-size: 12px;
}
.annotation-bar button:hover:not(:disabled) { border-color: #e94560; }
.annotation-bar button:disabled { opacity: 0.3; cursor: default; }
.annotation-bar button.save { background: #1b5e20; border-color: #4caf50; }
.annotation-bar button.save:hover:not(:disabled) { background: #2e7d32; }
#tapCanvas { position: absolute; top: 0; left: 0; pointer-events: none; z-index: 5; }
.record-panel {
    display: none; flex-direction: column; align-items: center; justify-content: center;
    gap: 20px; padding: 40px; width: 100%; max-width: 500px;
}
.record-panel input[type="text"] {
    background: #0f3460; color: #e0e0e0; border: 1px solid #555; border-radius: 4px;
    padding: 8px 12px; font-size: 14px; font-family: monospace; width: 100%; text-align: center;
}
.record-btn {
    width: 80px; height: 80px; border-radius: 50%; border: 3px solid #555;
    background: #333; cursor: pointer; display: flex; align-items: center;
    justify-content: center; transition: all 0.2s;
}
.record-btn:hover { border-color: #e94560; }
.record-btn .dot {
    width: 32px; height: 32px; border-radius: 50%; background: #e94560; transition: all 0.2s;
}
.record-btn.recording .dot {
    width: 24px; height: 24px; border-radius: 4px; background: #e94560;
    animation: rec-pulse 1s ease-in-out infinite;
}
@keyframes rec-pulse { 0%,100% { opacity: 0.6; } 50% { opacity: 1; } }
.record-elapsed {
    font-family: 'SF Mono', 'Menlo', monospace; font-size: 28px; color: #e94560;
}
.record-status { color: #888; font-size: 13px; }
</style>
</head>
<body>

<div class="header">
    <h1>Audio Explorer</h1>
    <select id="filePicker"></select>
    <button class="play-btn" id="playBtn" title="Play/Pause">&#9654;</button>
    <span class="time" id="timeDisplay">0:00.000 / 0:00.000</span>
    <span style="display:flex;align-items:center;gap:6px;margin-left:auto;">
        <span style="font-size:14px;cursor:pointer;" id="volIcon" title="Mute/unmute">&#128266;</span>
        <input type="range" id="volSlider" min="0" max="1" step="0.01" value="0.8"
            style="width:80px;accent-color:#e94560;cursor:pointer;">
    </span>
</div>

<div class="tabs">
    <div class="tab active" data-tab="analysis">Analysis</div>
    <div class="tab" data-tab="annotations" id="annTab">Annotations</div>
    <div class="tab" data-tab="record">Record</div>
    <div class="tab" data-tab="stems">Stems</div>
    <div class="tab" data-tab="hpss">HPSS</div>
    <div class="tab" data-tab="metrics-info">Metrics Info</div>
</div>

<div class="annotation-bar" id="annotationBar">
    <label>Layer: <input type="text" id="layerInput" value="beat" spellcheck="false"></label>
    <span class="tap-info"><kbd>T</kbd> to tap &middot; <span class="tap-count" id="tapCount">0 taps</span></span>
    <button class="save" id="saveAnnBtn" disabled onclick="saveAnnotation()">Save (S)</button>
    <button id="discardAnnBtn" disabled onclick="discardAnnotation()">Discard</button>
</div>

<div class="viewer" id="viewer">
    <div class="img-container" id="imgContainer">
        <img id="panelImg" draggable="false">
        <div class="cursor-line" id="cursorLine"></div>
        <canvas id="tapCanvas"></canvas>
    </div>
    <div class="info-panel" id="infoPanel">
        <h2>First Class &mdash; Always Visible</h2>
        <p style="color:#888;margin-bottom:16px;">These panels are always rendered. They represent the core audio properties that directly drive LED mapping and musical understanding.</p>

        <h3>Waveform + RMS overlay <span class="tag essential">Essential</span></h3>
        <p>Raw audio samples (white) with optional RMS energy overlay (yellow, toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#FFD740;">E</kbd>). RMS is the root-mean-square of the waveform in each frame &mdash; smoothed loudness over time, scaled to match waveform amplitude.</p>
        <p class="origin">Origin: Waveform display dates to oscilloscopes in the 1940s. RMS as a power measure dates to 19th-century electrical engineering; standard in audio since VU meters in the 1930s. Every DAW has both.</p>
        <p>Waveform shows transient attacks, silence, macro structure. RMS reveals energy trends the raw waveform hides &mdash; our research found that derivatives of RMS matter more than absolute values (climax brightens 58x faster than build, despite identical static RMS).</p>
        <p class="verdict">Non-negotiable. RMS overlay hidden by default to reduce visual clutter &mdash; enable when analyzing energy trajectories.</p>

        <h3>Mel Spectrogram <span class="tag essential">Essential</span></h3>
        <p>Short-time Fourier Transform (STFT) converted to mel scale and displayed as a heatmap. Time on x-axis, frequency on y-axis (low=bottom, high=top), color=loudness.</p>
        <p class="origin">Origin: The mel scale comes from Stevens, Volkmann &amp; Newman (1937) &mdash; psychoacoustic research showing humans perceive pitch logarithmically (200Hz&rarr;400Hz <em>sounds</em> the same as 400Hz&rarr;800Hz). The spectrogram (STFT) dates to Gabor (1946). Mel spectrograms became standard input for audio ML in the 1980s.</p>
        <p>You can <em>see</em> bass hits (bright blobs at bottom), vocals (middle bands), hi-hats (top). Harmonic content = horizontal lines. Percussive content = vertical lines &mdash; this is why HPSS works (median filtering by orientation).</p>
        <p class="verdict">The single most informative audio visualization. Industry standard.</p>

        <h3>Band Energy <span class="tag essential">Essential</span></h3>
        <p>The mel spectrogram collapsed into 5 bands &mdash; Sub-bass (20&ndash;80Hz), Bass (80&ndash;250Hz), Mids (250&ndash;2kHz), High-mids (2&ndash;6kHz), Treble (6&ndash;8kHz) &mdash; each plotted as a line over time.</p>
        <p class="origin">Origin: Multi-band meters from mixing engineering. Band boundaries follow critical band theory (Fletcher, 1940s) and PA crossover points. &ldquo;Bass energy over time&rdquo; is the foundation of almost every audio-reactive LED system (WLED-SR&rsquo;s entire beat detection = threshold on the bass bin).</p>
        <p>Shows which frequency range dominates at each moment. Directly actionable for LED mapping (assign colors to bands). A bass drop = Sub-bass/Bass spike. A cymbal crash = treble spike.</p>
        <p class="verdict">The most directly useful panel for LED effects. Standard in audio-reactive systems.</p>

        <h3>Annotations <span class="tag custom">Custom</span></h3>
        <p>Your own tap data overlaid on the analysis &mdash; beat taps, section changes, airy moments, flourishes. Whatever layers exist in the <code>.annotations.yaml</code> file.</p>
        <p class="origin">Origin: Custom to this project. Our &ldquo;test set&rdquo; for evaluating audio features against human perception.</p>
        <p>Note: tap annotations exhibit <strong>tactus ambiguity</strong> &mdash; listeners lock onto different metrical layers (kick, snare, off-beat) per song, so taps may be phase-shifted from the &ldquo;metric beat&rdquo; by 100&ndash;250ms (Martens 2011, London 2004). LEDs could exploit this: by flashing a specific layer, we may be able to <em>entrain</em> the audience&rsquo;s tactus rather than follow it.</p>
        <p class="verdict">Essential for research. Only shown when annotation data exists.</p>

        <h2>Second Class &mdash; Hidden by Default</h2>
        <p style="color:#888;margin-bottom:16px;">These features are computed but hidden. Toggle them with keyboard shortcuts when first-class metrics fail to explain what you&rsquo;re hearing. They measure real audio properties but have proven less useful for LED mapping in our testing.</p>

        <h3>Onset Strength <span class="tag expendable">Weak</span></h3>
        <p>Spectral flux &mdash; how much the spectrum <em>changes</em> between adjacent frames. Peaks = &ldquo;something new happened.&rdquo; Toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#00E5FF;">O</kbd>.</p>
        <p class="origin">Origin: Onset detection is one of the oldest MIR problems (Bello et al., 2005). Spectral flux has been the workhorse since the 1990s. Librosa implementation follows B&ouml;ck &amp; Widmer (2013).</p>
        <p><strong>Why second class:</strong> Tested against Harmonix dataset (7 tracks). Mean F1@70ms = 0.435 (finds less than half the beats). Onset strength at beat times is actually <em>lower</em> than at non-beat times (0.89x ratio) &mdash; the signal is anti-correlated with beats on electronic music. Detects 2&ndash;3x more events than there are beats, over-triggering on string noise, vocal consonants, cymbal wash.</p>
        <p>Our annotation research also found user taps track bass peaks (19ms median), not onsets &mdash; only 48.5% of taps align with librosa onsets.</p>
        <p class="verdict">Measures something real (spectral novelty) but doesn&rsquo;t map to what humans perceive as beats. Band energy is more useful for LED triggering.</p>

        <h3>Spectral Centroid <span class="tag expendable">Weak</span></h3>
        <p>The &ldquo;center of mass&rdquo; of the spectrum &mdash; the frequency where half the energy is above and half below. Often called &ldquo;brightness.&rdquo; Toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#B388FF;">C</kbd>.</p>
        <p class="origin">Origin: One of the oldest timbral descriptors (Grey, 1977). A standard MPEG-7 audio descriptor. High centroid = bright/shimmery, low centroid = dark/boomy.</p>
        <p><strong>Why second class:</strong> A one-dimensional summary of a complex spectrum. Two very different spectra can have the same centroid (a tone at 2kHz vs flat noise centered at 2kHz). Band energy already gives richer timbral information across 5 bands.</p>
        <p class="verdict">Standard MIR feature but the least valuable for LED mapping. Most expendable metric we compute.</p>

        <h3>Librosa Beats <span class="tag expendable">Weak</span></h3>
        <p>Beat tracking via <code>librosa.beat.beat_track</code> &mdash; estimates tempo then snaps onset peaks to a grid. Toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#FF1744;">B</kbd>.</p>
        <p><strong>Why second class:</strong> Doubles tempo on syncopated rock (161.5 vs ~83 BPM on Tool&rsquo;s Opiate). Built on top of onset strength, which is itself a weak beat discriminator. Best F1=0.500 on dense rock.</p>
        <p class="verdict">Useful as a sanity check. Not reliable enough to drive LED effects directly.</p>

        <h2>Not Yet Implemented <span class="tag gap">Gap</span></h2>
        <table class="missing-table">
            <tr><th>Feature</th><th>What it measures</th><th>Why it might matter</th></tr>
            <tr><td>RMS derivative</td><td>Rate-of-change of loudness</td><td>Our most validated finding: derivatives &gt; absolutes. Shows how fast things get louder, not how loud they are. The biggest gap given our research.</td></tr>
            <tr><td>Spectral flatness</td><td>How noise-like vs tonal (0=pure tone, 1=white noise)</td><td>Could detect &ldquo;texture&rdquo; moments &mdash; washy cymbals, distortion, atmospheric sections. Relevant to &ldquo;airiness.&rdquo;</td></tr>
            <tr><td>Chromagram</td><td>Which notes/chords are present (12 pitch classes)</td><td>Shows harmonic progressions. &ldquo;Changes&rdquo; annotations might correlate with chroma shifts.</td></tr>
            <tr><td>Spectral contrast</td><td>Difference between spectral peaks and valleys per band</td><td>Harmonic richness vs noise content. Could be a better &ldquo;feeling&rdquo; feature than centroid.</td></tr>
            <tr><td>Zero crossing rate</td><td>How often the waveform crosses zero</td><td>High ZCR = percussive, low ZCR = tonal. Simple percussion discriminator.</td></tr>
        </table>
    </div>
    <div class="record-panel" id="recordPanel">
        <input type="text" id="recordName" placeholder="segment name (e.g. tool_lateralus_intro)" spellcheck="false">
        <button class="record-btn" id="recordBtn" onclick="toggleRecord()"><span class="dot"></span></button>
        <div class="record-elapsed" id="recordElapsed">0:00.0</div>
        <div class="record-status" id="recordStatus">Click to record from BlackHole</div>
    </div>
</div>

<div class="progress-bar">
    <div class="progress-track" id="progressTrack">
        <div class="progress-fill" id="progressFill"></div>
        <div class="progress-thumb" id="progressThumb"></div>
    </div>
</div>

<div class="stem-status" id="stemStatus"></div>

<div class="controls" id="controlsHint">
    <kbd>Space</kbd> play/pause &nbsp;
    <kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;
    Click panel to seek
</div>

<audio id="audio" preload="auto"></audio>

<script>
const audio = document.getElementById('audio');
const filePicker = document.getElementById('filePicker');
const panelImg = document.getElementById('panelImg');
const cursorLine = document.getElementById('cursorLine');
const imgContainer = document.getElementById('imgContainer');
const timeDisplay = document.getElementById('timeDisplay');
const playBtn = document.getElementById('playBtn');
const progressFill = document.getElementById('progressFill');
const progressThumb = document.getElementById('progressThumb');
const progressTrack = document.getElementById('progressTrack');
const annTab = document.getElementById('annTab');
const viewer = document.getElementById('viewer');
const infoPanel = document.getElementById('infoPanel');
const volSlider = document.getElementById('volSlider');
const volIcon = document.getElementById('volIcon');

let masterVolume = 0.8;
let mutedVolume = null;  // stashed volume when muted

audio.volume = masterVolume;

volSlider.addEventListener('input', () => {
    masterVolume = parseFloat(volSlider.value);
    mutedVolume = null;
    applyVolume();
    updateVolIcon();
});

volIcon.addEventListener('click', () => {
    if (mutedVolume !== null) {
        masterVolume = mutedVolume;
        mutedVolume = null;
    } else {
        mutedVolume = masterVolume;
        masterVolume = 0;
    }
    volSlider.value = masterVolume;
    applyVolume();
    updateVolIcon();
});

function applyVolume() {
    if (hasStemAudio()) {
        audio.volume = 0;
        Object.entries(stemAudios).forEach(([name, a]) => {
            a.volume = activeStems[name] ? masterVolume : 0;
        });
    } else {
        audio.volume = masterVolume;
    }
}

function updateVolIcon() {
    volIcon.textContent = masterVolume === 0 ? '\u{1F507}' : masterVolume < 0.5 ? '\u{1F509}' : '\u{1F50A}';
}

let currentFile = null;
let currentTab = 'analysis';
let pixelMapping = null;   // {xLeft, xRight, pngWidth, duration}
let files = [];
let stemsPollTimer = null;

// ── Stem audio ───────────────────────────────────────────────────

let currentStemNames = [];  // set dynamically for demucs or hpss
let stemAudios = {};   // name -> Audio element
let activeStems = {};  // name -> bool

function setupStemAudio(stemNames, basePath) {
    cleanupStemAudio();
    currentStemNames = stemNames;
    stemNames.forEach(name => {
        const a = new Audio(basePath + name + '.wav');
        a.preload = 'auto';
        a.volume = masterVolume;
        stemAudios[name] = a;
        activeStems[name] = true;
    });
    audio.volume = 0;  // mute original, use stems for sound
    updateStemUI();
}

function cleanupStemAudio() {
    Object.values(stemAudios).forEach(a => { a.pause(); a.src = ''; });
    stemAudios = {};
    activeStems = {};
    currentStemNames = [];
    audio.volume = masterVolume;
    document.getElementById('stemStatus').style.display = 'none';
    document.getElementById('controlsHint').innerHTML =
        '<kbd>Space</kbd> play/pause &nbsp;' +
        '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;' +
        'Click panel to seek';
}

function toggleStem(name) {
    if (!activeStems.hasOwnProperty(name)) return;
    activeStems[name] = !activeStems[name];
    if (stemAudios[name]) stemAudios[name].volume = activeStems[name] ? masterVolume : 0;
    updateStemUI();
}

function allStemsOn() {
    currentStemNames.forEach(name => {
        activeStems[name] = true;
        if (stemAudios[name]) stemAudios[name].volume = masterVolume;
    });
    updateStemUI();
}

function updateStemUI() {
    const status = document.getElementById('stemStatus');
    status.style.display = 'block';
    status.innerHTML = currentStemNames.map((name, i) =>
        '<span class="stem-chip ' + (activeStems[name] ? 'active' : 'muted') + '" ' +
        'onclick="toggleStem(\'' + name + '\')">' +
        (i + 1) + ':' + name + '</span>'
    ).join('');
    document.getElementById('controlsHint').innerHTML =
        '<kbd>Space</kbd> play/pause &nbsp;' +
        '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;' +
        '<kbd>1</kbd>-<kbd>' + currentStemNames.length + '</kbd> toggle stems &nbsp;' +
        '<kbd>A</kbd> all on';
}

function syncStemAudios() {
    const t = audio.currentTime;
    Object.values(stemAudios).forEach(a => {
        if (Math.abs(a.currentTime - t) > 0.05) a.currentTime = t;
    });
}

function stemPlay() {
    syncStemAudios();
    Object.values(stemAudios).forEach(a => a.play());
}

function stemPause() {
    Object.values(stemAudios).forEach(a => a.pause());
}

function stemSeek() {
    const t = audio.currentTime;
    const wasPlaying = Object.values(stemAudios).some(a => !a.paused);
    Object.values(stemAudios).forEach(a => { a.currentTime = t; });
}

function hasStemAudio() {
    return Object.keys(stemAudios).length > 0;
}

// ── Feature toggles (analysis/annotations tabs) ─────────────────

let featureState = {rms: false};

function toggleFeature(name) {
    featureState[name] = !featureState[name];
    updateFeatureUI();
    loadPanel();
}

function updateFeatureUI() {
    const status = document.getElementById('stemStatus');
    if (currentTab !== 'analysis' && currentTab !== 'annotations') return;
    status.style.display = 'block';
    status.innerHTML =
        '<span class="stem-chip ' + (featureState.rms ? 'active' : 'muted') + '" ' +
        'onclick="toggleFeature(\'rms\')">E:RMS overlay</span>';
    document.getElementById('controlsHint').innerHTML =
        '<kbd>Space</kbd> play/pause &nbsp;' +
        '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;' +
        'Click to seek &nbsp;' +
        '<kbd>E</kbd> toggle RMS overlay';
}

// ── Annotation recording ─────────────────────────────────────────

let annotationTaps = [];

function recordTap() {
    if (audio.paused || !pixelMapping) return;
    const t = parseFloat(audio.currentTime.toFixed(3));
    annotationTaps.push(t);
    updateAnnotationUI();
    drawTapMarkers();
}

function drawTapMarkers() {
    const canvas = document.getElementById('tapCanvas');
    if (!canvas || !pixelMapping) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (annotationTaps.length === 0) return;

    const scale = panelImg.clientWidth / pixelMapping.pngWidth;
    ctx.strokeStyle = 'rgba(233, 69, 96, 0.8)';
    ctx.lineWidth = 2;

    annotationTaps.forEach(t => {
        const frac = t / pixelMapping.duration;
        const px = (pixelMapping.xLeft + frac * (pixelMapping.xRight - pixelMapping.xLeft)) * scale;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, canvas.height);
        ctx.stroke();
    });
}

function resizeTapCanvas() {
    const canvas = document.getElementById('tapCanvas');
    if (!canvas || !panelImg.clientWidth) return;
    canvas.width = panelImg.clientWidth;
    canvas.height = panelImg.clientHeight;
    drawTapMarkers();
}

async function saveAnnotation() {
    if (annotationTaps.length === 0) return;
    const layer = document.getElementById('layerInput').value.trim();
    if (!layer) { alert('Enter a layer name'); return; }

    const resp = await fetch('/api/annotations/' + encodeURIComponent(currentFile), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ layer: layer, taps: annotationTaps.slice().sort((a, b) => a - b) })
    });

    if (resp.ok) {
        annotationTaps = [];
        updateAnnotationUI();
        drawTapMarkers();
        // Update file info to reflect new annotations
        const fileInfo = files.find(f => f.path === currentFile);
        if (fileInfo) fileInfo.has_annotations = true;
        // Re-render annotations tab to show saved data
        loadPanel();
    } else {
        alert('Save failed: ' + (await resp.text()));
    }
}

function discardAnnotation() {
    annotationTaps = [];
    updateAnnotationUI();
    drawTapMarkers();
}

function updateAnnotationUI() {
    const tapCount = document.getElementById('tapCount');
    const saveBtn = document.getElementById('saveAnnBtn');
    const discardBtn = document.getElementById('discardAnnBtn');
    if (!tapCount) return;
    tapCount.textContent = annotationTaps.length + ' taps';
    saveBtn.disabled = annotationTaps.length === 0;
    discardBtn.disabled = annotationTaps.length === 0;
}

panelImg.addEventListener('load', resizeTapCanvas);
window.addEventListener('resize', resizeTapCanvas);

// ── Recording ────────────────────────────────────────────────────

let isRecording = false;
let recordStartTime = null;
let recordTimer = null;

async function toggleRecord() {
    if (isRecording) {
        await stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    const resp = await fetch('/api/record/start', {method: 'POST'});
    const data = await resp.json();
    if (!data.ok) {
        document.getElementById('recordStatus').textContent = 'Error: ' + (data.error || 'failed');
        return;
    }
    isRecording = true;
    recordStartTime = Date.now();
    document.getElementById('recordBtn').classList.add('recording');
    document.getElementById('recordStatus').textContent = 'Recording... click to stop';
    recordTimer = setInterval(() => {
        const elapsed = (Date.now() - recordStartTime) / 1000;
        const m = Math.floor(elapsed / 60);
        const s = (elapsed - m * 60).toFixed(1).padStart(4, '0');
        document.getElementById('recordElapsed').textContent = m + ':' + s;
    }, 100);
}

async function stopRecording() {
    clearInterval(recordTimer);
    document.getElementById('recordBtn').classList.remove('recording');
    document.getElementById('recordStatus').textContent = 'Saving...';

    const name = document.getElementById('recordName').value.trim() || '';
    const resp = await fetch('/api/record/stop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name})
    });
    const data = await resp.json();
    isRecording = false;

    if (data.ok) {
        document.getElementById('recordStatus').textContent =
            'Saved: ' + data.filename + ' (' + data.duration + 's)';
        document.getElementById('recordElapsed').textContent = '0:00.0';
        document.getElementById('recordName').value = '';
        // Refresh file list and select the new file
        await loadFileList();
        selectFile(data.filename);
        // Switch to analysis tab
        currentTab = 'analysis';
        updateTabUI();
        loadPanel();
    } else {
        document.getElementById('recordStatus').textContent = 'Error: ' + (data.error || 'failed');
    }
}

// ── File picker ──────────────────────────────────────────────────

async function loadFileList() {
    const resp = await fetch('/api/files');
    files = await resp.json();

    // Group by group name
    const groups = {};
    files.forEach(f => {
        if (!groups[f.group]) groups[f.group] = [];
        groups[f.group].push(f);
    });

    filePicker.innerHTML = '';
    for (const [group, items] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = group;
        items.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.path;
            const dur = formatTime(f.duration);
            const ann = f.has_annotations ? ' [ann]' : '';
            opt.textContent = f.name + ' (' + dur + ')' + ann;
            optgroup.appendChild(opt);
        });
        filePicker.appendChild(optgroup);
    }

    if (files.length > 0) {
        selectFile(files[0].path);
    }
}

function selectFile(path) {
    cleanupStemAudio();
    audio.pause();
    playBtn.innerHTML = '&#9654;';
    audio.currentTime = 0;
    currentFile = path;
    filePicker.value = path;

    // Annotations tab always enabled (user can create new annotations)
    annTab.classList.remove('disabled');

    // Clear pending taps when switching files
    annotationTaps = [];
    updateAnnotationUI();

    // Set audio source
    audio.src = '/audio/' + encodeURIComponent(path);
    audio.load();

    loadPanel();
}

filePicker.addEventListener('change', () => selectFile(filePicker.value));

// ── Tabs ─────────────────────────────────────────────────────────

function updateTabUI() {
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === currentTab);
    });
}

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (tab.classList.contains('disabled')) return;
        const prev = currentTab;
        if ((prev === 'stems' || prev === 'hpss') && tab.dataset.tab !== prev) cleanupStemAudio();
        currentTab = tab.dataset.tab;
        updateTabUI();
        loadPanel();
    });
});

// ── Panel loading ────────────────────────────────────────────────

async function loadPanel() {
    // Show/hide annotation bar
    const annBar = document.getElementById('annotationBar');
    annBar.style.display = currentTab === 'annotations' ? 'flex' : 'none';

    const recordPanel = document.getElementById('recordPanel');

    if (currentTab === 'metrics-info') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'block';
        recordPanel.style.display = 'none';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML =
            'Reference information about the analysis panels and features';
        return;
    }
    if (currentTab === 'record') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'none';
        recordPanel.style.display = 'flex';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML =
            'Record audio from BlackHole loopback';
        return;
    }
    imgContainer.style.display = 'inline-block';
    infoPanel.style.display = 'none';
    recordPanel.style.display = 'none';

    if (!currentFile) return;

    if (currentTab === 'stems') {
        await loadStems();
        return;
    }
    if (currentTab === 'hpss') {
        await loadHPSS();
        return;
    }

    // Show feature toggle UI for analysis/annotations tabs
    updateFeatureUI();
    if (currentTab === 'annotations') {
        document.getElementById('controlsHint').innerHTML =
            '<kbd>Space</kbd> play/pause &nbsp;' +
            '<kbd>T</kbd> tap annotation &nbsp;' +
            '<kbd>S</kbd> save &nbsp;' +
            '<kbd>Esc</kbd> discard &nbsp;' +
            '<kbd>E</kbd> RMS overlay';
    }

    let url = '/api/render/' + encodeURIComponent(currentFile);
    const params = [];
    if (currentTab === 'annotations') params.push('annotations=1');
    const activeFeatures = Object.entries(featureState)
        .filter(([_, v]) => v).map(([k]) => k);
    if (activeFeatures.length < 4) {
        params.push('features=' + activeFeatures.join(','));
    }
    if (params.length) url += '?' + params.join('&');

    showOverlay('Rendering...');

    try {
        const resp = await fetch(url);
        if (!resp.ok) { showOverlay('Render failed'); return; }

        pixelMapping = {
            xLeft: parseFloat(resp.headers.get('X-Left-Px')),
            xRight: parseFloat(resp.headers.get('X-Right-Px')),
            pngWidth: parseFloat(resp.headers.get('X-Png-Width')),
            duration: parseFloat(resp.headers.get('X-Duration')),
        };

        const blob = await resp.blob();
        panelImg.src = URL.createObjectURL(blob);
        hideOverlay();
        cursorLine.style.display = 'block';
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadStems() {
    if (!currentFile) return;

    // Check if stems are ready
    const statusResp = await fetch('/api/stems/status/' + encodeURIComponent(currentFile));
    const status = await statusResp.json();

    if (!status.ready) {
        showOverlay('Running demucs...');
        // Trigger demucs and poll
        fetch('/api/stems/' + encodeURIComponent(currentFile));
        pollStemsReady();
        return;
    }

    showOverlay('Rendering stems...');

    try {
        const resp = await fetch('/api/stems/' + encodeURIComponent(currentFile));
        if (!resp.ok) { showOverlay('Stems render failed'); return; }

        pixelMapping = {
            xLeft: parseFloat(resp.headers.get('X-Left-Px')),
            xRight: parseFloat(resp.headers.get('X-Right-Px')),
            pngWidth: parseFloat(resp.headers.get('X-Png-Width')),
            duration: parseFloat(resp.headers.get('X-Duration')),
        };

        const blob = await resp.blob();
        panelImg.src = URL.createObjectURL(blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['drums', 'bass', 'vocals', 'other'],
                       '/audio/separated/htdemucs/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

function pollStemsReady() {
    if (stemsPollTimer) clearInterval(stemsPollTimer);
    stemsPollTimer = setInterval(async () => {
        const resp = await fetch('/api/stems/status/' + encodeURIComponent(currentFile));
        const status = await resp.json();
        if (status.ready) {
            clearInterval(stemsPollTimer);
            stemsPollTimer = null;
            if (currentTab === 'stems') loadStems();
        }
    }, 2000);
}

async function loadHPSS() {
    if (!currentFile) return;
    showOverlay('Computing HPSS...');

    try {
        const resp = await fetch('/api/hpss/' + encodeURIComponent(currentFile));
        if (!resp.ok) { showOverlay('HPSS render failed'); return; }

        pixelMapping = {
            xLeft: parseFloat(resp.headers.get('X-Left-Px')),
            xRight: parseFloat(resp.headers.get('X-Right-Px')),
            pngWidth: parseFloat(resp.headers.get('X-Png-Width')),
            duration: parseFloat(resp.headers.get('X-Duration')),
        };

        const blob = await resp.blob();
        panelImg.src = URL.createObjectURL(blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['harmonic', 'percussive'],
                       '/audio/separated/hpss/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

// ── Overlay ──────────────────────────────────────────────────────

function showOverlay(text) {
    let overlay = viewer.querySelector('.overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'overlay';
        overlay.innerHTML = '<span class="overlay-text"></span>';
        viewer.appendChild(overlay);
    }
    overlay.querySelector('.overlay-text').textContent = text;
    overlay.style.display = 'flex';
}

function hideOverlay() {
    const overlay = viewer.querySelector('.overlay');
    if (overlay) overlay.style.display = 'none';
}

// ── Cursor sync ──────────────────────────────────────────────────

function updateCursor() {
    if (pixelMapping && panelImg.naturalWidth > 0) {
        const t = audio.currentTime;
        const dur = pixelMapping.duration;
        const frac = dur > 0 ? t / dur : 0;
        const scale = panelImg.clientWidth / pixelMapping.pngWidth;
        const px = (pixelMapping.xLeft + frac * (pixelMapping.xRight - pixelMapping.xLeft)) * scale;
        cursorLine.style.left = px + 'px';

        // Progress bar
        const pct = dur > 0 ? (t / dur) * 100 : 0;
        progressFill.style.width = pct + '%';
        progressThumb.style.left = pct + '%';

        // Time display
        timeDisplay.textContent = formatTime(t) + ' / ' + formatTime(dur);
    }
    requestAnimationFrame(updateCursor);
}
requestAnimationFrame(updateCursor);

// ── Click to seek ────────────────────────────────────────────────

panelImg.addEventListener('click', (e) => {
    if (!pixelMapping) return;
    const rect = panelImg.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const scale = panelImg.clientWidth / pixelMapping.pngWidth;
    const xLeft = pixelMapping.xLeft * scale;
    const xRight = pixelMapping.xRight * scale;

    if (clickX < xLeft || clickX > xRight) return;

    const frac = (clickX - xLeft) / (xRight - xLeft);
    audio.currentTime = frac * pixelMapping.duration;
    if (hasStemAudio()) stemSeek();
});

// Progress bar seek
progressTrack.addEventListener('click', (e) => {
    if (!pixelMapping) return;
    const rect = progressTrack.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    audio.currentTime = Math.max(0, Math.min(frac * pixelMapping.duration, pixelMapping.duration));
    if (hasStemAudio()) stemSeek();
});

// ── Playback controls ────────────────────────────────────────────

function togglePlay() {
    if (audio.paused) {
        audio.play();
        if (hasStemAudio()) stemPlay();
        playBtn.innerHTML = '&#9646;&#9646;';
    } else {
        audio.pause();
        if (hasStemAudio()) stemPause();
        playBtn.innerHTML = '&#9654;';
    }
}

playBtn.addEventListener('click', togglePlay);

audio.addEventListener('ended', () => {
    playBtn.innerHTML = '&#9654;';
    if (hasStemAudio()) stemPause();
});

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT') return;

    if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
    } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        audio.currentTime = Math.max(0, audio.currentTime - 5);
        if (hasStemAudio()) stemSeek();
    } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        if (pixelMapping) {
            audio.currentTime = Math.min(pixelMapping.duration, audio.currentTime + 5);
            if (hasStemAudio()) stemSeek();
        }
    } else if (e.code === 'KeyT' && currentTab === 'annotations' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        recordTap();
    } else if (e.code === 'KeyS' && currentTab === 'annotations' && annotationTaps.length > 0 && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        saveAnnotation();
    } else if (e.code === 'Escape' && annotationTaps.length > 0) {
        discardAnnotation();
    } else if (!hasStemAudio() && (currentTab === 'analysis' || currentTab === 'annotations')) {
        if (e.code === 'KeyE') {
            toggleFeature('rms');
        }
    } else if (hasStemAudio()) {
        const digitMatch = e.code.match(/^Digit(\d)$/);
        if (digitMatch) {
            const idx = parseInt(digitMatch[1]) - 1;
            if (idx >= 0 && idx < currentStemNames.length) {
                toggleStem(currentStemNames[idx]);
            }
        } else if (e.code === 'KeyA') {
            allStemsOn();
        }
    }
});

// ── Helpers ──────────────────────────────────────────────────────

function formatTime(s) {
    if (!s || isNaN(s)) return '0:00.000';
    const m = Math.floor(s / 60);
    const sec = s - m * 60;
    return m + ':' + sec.toFixed(3).padStart(6, '0');
}

// ── Init ─────────────────────────────────────────────────────────

loadFileList();
</script>
</body>
</html>
''').safe_substitute()  # No substitutions needed, but safe_substitute avoids $-escaping


# ── HTTP Handler ──────────────────────────────────────────────────────

class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging — only errors
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == '/api/record/start':
            self._handle_record_start()
        elif path == '/api/record/stop':
            self._handle_record_stop()
        elif path.startswith('/api/annotations/'):
            rel_path = path[len('/api/annotations/'):]
            self._save_annotation(rel_path)
        else:
            self.send_error(404)

    def _handle_record_start(self):
        result = start_recording()
        data = json.dumps(result).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_record_stop(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            req = {}
        name = req.get('name', '')
        result = stop_recording(name)
        data = json.dumps(result).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _save_annotation(self, rel_path):
        """Save annotation taps to YAML file."""
        import yaml

        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404, 'File not found')
            return

        # Read POST body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return

        layer = data.get('layer', '').strip()
        taps = data.get('taps', [])
        if not layer or not isinstance(taps, list):
            self.send_error(400, 'Missing layer or taps')
            return

        # Load existing annotations
        ann_path = Path(filepath).with_suffix('.annotations.yaml')
        annotations = {}
        if ann_path.exists():
            with open(ann_path) as f:
                annotations = yaml.safe_load(f) or {}

        # Merge new layer
        annotations[layer] = [round(t, 3) for t in sorted(taps)]

        # Write back
        with open(ann_path, 'w') as f:
            yaml.dump(annotations, f, default_flow_style=False, sort_keys=False)

        print(f"[annotations] Saved {len(taps)} taps to layer '{layer}' in {ann_path.name}")
        print(f"  Layers: {list(annotations.keys())}")

        # Clear render caches for this file (annotations tab needs re-render)
        keys_to_clear = [k for k in _render_cache if k[0] == filepath]
        for k in keys_to_clear:
            del _render_cache[k]

        # Clear file list cache (has_annotations may have changed)
        global _file_list_cache
        _file_list_cache = None

        # Respond with success
        result = json.dumps({'ok': True, 'layer': layer, 'count': len(taps)}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(result)))
        self.end_headers()
        self.wfile.write(result)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path == '/':
            self._serve_html()
        elif path == '/favicon.ico':
            self._serve_favicon()
        elif path == '/api/files':
            self._serve_file_list()
        elif path.startswith('/api/render/'):
            rel_path = path[len('/api/render/'):]
            with_ann = 'annotations' in query
            features = None
            feat_str = query.get('features', [None])[0]
            if feat_str is not None:
                active = set(feat_str.split(',')) if feat_str else set()
                all_feat = ('rms',)
                features = {n: (n in active) for n in all_feat}
            self._serve_render(rel_path, with_ann, features)
        elif path.startswith('/api/stems/status/'):
            rel_path = path[len('/api/stems/status/'):]
            self._serve_stems_status(rel_path)
        elif path.startswith('/api/stems/'):
            rel_path = path[len('/api/stems/'):]
            self._serve_stems(rel_path)
        elif path.startswith('/api/hpss/'):
            rel_path = path[len('/api/hpss/'):]
            self._serve_hpss(rel_path)
        elif path.startswith('/audio/'):
            rel_path = path[len('/audio/'):]
            self._serve_audio(rel_path)
        else:
            self.send_error(404)

    def _serve_html(self):
        html = generate_html().encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_favicon(self):
        # 16x16 SVG waveform icon
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
            '<rect width="16" height="16" rx="3" fill="#1a1a2e"/>'
            '<path d="M2 8h1v-2h1v4h1v-6h1v8h1v-7h1v6h1v-4h1v3h1v-5h1v4h1v-2h1v1" '
            'stroke="#e94560" stroke-width="1" fill="none" stroke-linecap="round"/>'
            '</svg>'
        )
        data = svg.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'image/svg+xml')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'max-age=86400')
        self.end_headers()
        self.wfile.write(data)

    def _serve_file_list(self):
        data = json.dumps(discover_files()).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_render(self, rel_path, with_annotations, features=None):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404, 'File not found')
            return

        try:
            png_bytes, headers = render_analysis(filepath, with_annotations, features)
        except Exception as e:
            print(f"[render] Error: {e}")
            self.send_error(500, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(png_bytes)))
        # Expose custom headers to JS fetch
        exposed = ', '.join(headers.keys())
        self.send_header('Access-Control-Expose-Headers', exposed)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(png_bytes)

    def _serve_stems_status(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        ready = _stems_ready(filepath)
        data = json.dumps({'ready': ready}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_stems(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        if not _stems_ready(filepath):
            _run_demucs_background(filepath)
            data = json.dumps({'status': 'running'}).encode('utf-8')
            self.send_response(202)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        try:
            png_bytes, headers = render_stems(filepath)
        except Exception as e:
            print(f"[stems] Error: {e}")
            self.send_error(500, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(png_bytes)))
        exposed = ', '.join(headers.keys())
        self.send_header('Access-Control-Expose-Headers', exposed)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(png_bytes)

    def _serve_hpss(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        try:
            png_bytes, headers = render_hpss(filepath)
        except Exception as e:
            print(f"[hpss] Error: {e}")
            self.send_error(500, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(png_bytes)))
        exposed = ', '.join(headers.keys())
        self.send_header('Access-Control-Expose-Headers', exposed)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(png_bytes)

    def _serve_audio(self, rel_path):
        """Serve WAV with HTTP Range support (required for <audio> seeking)."""
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            self.send_error(404)
            return

        try:
            self._stream_audio(filepath, file_size)
        except BrokenPipeError:
            pass  # Browser cancelled request (e.g. switched files)

    def _stream_audio(self, filepath, file_size):
        """Stream WAV bytes, with Range support."""
        # Parse Range header
        range_header = self.headers.get('Range')
        if range_header:
            # Parse "bytes=start-end"
            m = re.match(r'bytes=(\d+)-(\d*)', range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(length))
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()

                with open(filepath, 'rb') as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return

        # Full file response
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(file_size))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()

        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)


# ── Server entry point ───────────────────────────────────────────────

def run_server(port=0):
    """Start the web viewer server and open the browser."""
    import webbrowser

    server = HTTPServer(('127.0.0.1', port), ViewerHandler)
    actual_port = server.server_address[1]

    url = f'http://127.0.0.1:{actual_port}'
    print(f"Audio Explorer running at {url}")
    print("Press Ctrl+C to stop\n")

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Web-based audio analysis viewer')
    parser.add_argument('--port', type=int, default=0, help='Server port (0=auto)')
    args = parser.parse_args()
    run_server(args.port)
