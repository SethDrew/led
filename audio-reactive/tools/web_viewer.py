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

import hashlib
import hmac
import http.cookies
import io
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from string import Template
from urllib.parse import urlparse, parse_qs, unquote

# Force Agg backend BEFORE any viewer imports (which import pyplot at module level)
import matplotlib
matplotlib.use('Agg')

SEGMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research', 'audio-segments')
DPI = 100

# ── Caches ────────────────────────────────────────────────────────────

_render_cache = {}      # (filepath, tab) -> (png_bytes, headers_dict)
_file_list_cache = None  # JSON-serializable list
_demucs_status = {}     # filepath -> bool (True = ready)
_demucs_locks = {}      # filepath -> threading.Lock
_recording = None       # {'stream': sd.InputStream, 'frames': list} or None

# ── Effects management ────────────────────────────────────────────────

_effect_process = None   # subprocess.Popen or None
_effects_cache = None    # list of effect names from runner.py --list

EFFECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'effects')

# ── Auth ─────────────────────────────────────────────────────────────

PUBLIC_MODE = os.environ.get('LED_VIEWER_PUBLIC', '') == '1'
_PASSCODE = os.environ.get('LED_VIEWER_PASSCODE', '')  # Set via env var on server
_AUTH_SECRET = os.environ.get('LED_VIEWER_SECRET', secrets.token_hex(32))
_AUTH_COOKIE = 'led_session'
_PROTECTED_PREFIXES = ('/api/stems', '/api/hpss/', '/api/lab-nmf/', '/api/lab-repet/', '/api/lab/')

def _make_auth_token():
    """Create a signed session token."""
    payload = f"authorized:{int(time.time())}"
    sig = hmac.new(_AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"

def _verify_auth_token(token):
    """Verify a signed session token. Returns True if valid."""
    if not token:
        return False
    parts = token.rsplit(':', 1)
    if len(parts) != 2:
        return False
    payload, sig = parts
    expected = hmac.new(_AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)

def _is_authenticated(handler):
    """Check if the request has a valid auth cookie."""
    if not PUBLIC_MODE:
        return True  # Local mode: everything unlocked
    cookie_header = handler.headers.get('Cookie', '')
    cookies = http.cookies.SimpleCookie()
    try:
        cookies.load(cookie_header)
    except http.cookies.CookieError:
        return False
    morsel = cookies.get(_AUTH_COOKIE)
    if morsel is None:
        return False
    return _verify_auth_token(morsel.value)

def _is_protected(path):
    """Check if a path requires authentication."""
    return any(path.startswith(p) for p in _PROTECTED_PREFIXES)
EFFECT_PREFS_PATH = os.path.join(EFFECTS_DIR, 'effect_prefs.json')


def _discover_effects():
    """Run runner.py --list and parse available effect names."""
    global _effects_cache
    if _effects_cache is not None:
        return _effects_cache

    try:
        result = subprocess.run(
            [sys.executable, 'runner.py', '--list'],
            cwd=EFFECTS_DIR, capture_output=True, text=True, timeout=10
        )
        names = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('Available') or line.startswith('('):
                continue
            # Lines like: "wled_volume          — WLED Volume Reactive"
            if '—' not in line:
                continue
            name = line.split('—')[0].strip()
            if name:
                names.append(name)
        _effects_cache = names
    except Exception as e:
        print(f"[effects] Discovery failed: {e}")
        _effects_cache = []

    return _effects_cache


def _load_effect_prefs():
    """Load effect preferences (ratings, order) from JSON file."""
    if os.path.exists(EFFECT_PREFS_PATH):
        try:
            with open(EFFECT_PREFS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_effect_prefs(prefs):
    """Save effect preferences to JSON file."""
    try:
        with open(EFFECT_PREFS_PATH, 'w') as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"[effects] Save prefs failed: {e}")


def _get_effects_list():
    """Get effects list merged with preferences (ratings, order)."""
    names = _discover_effects()
    prefs = _load_effect_prefs()

    effects = []
    for name in names:
        p = prefs.get(name, {})
        effects.append({
            'name': name,
            'order': p.get('order', 999),
            'rating': p.get('rating', 0),
        })

    effects.sort(key=lambda e: e['order'])
    return effects


def _start_effect(name):
    """Start an effect subprocess. Kills any running effect first."""
    global _effect_process
    _stop_effect()

    try:
        _effect_process = subprocess.Popen(
            [sys.executable, 'runner.py', name, '--no-leds'],
            cwd=EFFECTS_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        print(f"[effects] Started: {name} (pid {_effect_process.pid})")
    except Exception as e:
        print(f"[effects] Start failed: {e}")
        _effect_process = None


def _stop_effect():
    """Stop the running effect subprocess."""
    global _effect_process
    if _effect_process is not None:
        try:
            _effect_process.terminate()
            _effect_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _effect_process.kill()
        except Exception:
            pass
        print(f"[effects] Stopped (pid {_effect_process.pid})")
        _effect_process = None


def _get_running_effect_name():
    """Get the name of the currently running effect, or None."""
    global _effect_process
    if _effect_process is None:
        return None
    if _effect_process.poll() is not None:
        _effect_process = None
        return None
    # Extract from command line args
    args = _effect_process.args
    if len(args) >= 3:
        return args[2]
    return None


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

    # Uploaded files
    uploads_dir = segments_path / 'uploads'
    if uploads_dir.exists():
        for wav in sorted(uploads_dir.glob('*.wav')):
            dur = _get_wav_duration(str(wav))
            files.append({
                'name': wav.name,
                'path': f'uploads/{wav.name}',
                'duration': dur,
                'has_annotations': False,
                'group': 'uploads',
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


def render_lab_repet(filepath):
    """Compute REPET, save WAVs, render diagnostic PNG (beat spectrum + mask + separated specs)."""
    cache_key = (filepath, 'lab-repet')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    import numpy as np
    import soundfile as sf
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Add separation dir to path for import
    separation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'research', 'separation')
    if separation_dir not in sys.path:
        sys.path.insert(0, separation_dir)
    from repet import repet

    stem_name = Path(filepath).stem
    repet_dir = os.path.join(SEGMENTS_DIR, 'separated', 'repet', stem_name)
    rep_path = os.path.join(repet_dir, 'repeating.wav')
    nonrep_path = os.path.join(repet_dir, 'non-repeating.wav')
    info_path = os.path.join(repet_dir, 'info.npz')

    # Compute and save if not cached on disk
    # Version check: re-run if info.npz missing 'periods' (old format)
    needs_recompute = not (os.path.exists(rep_path) and os.path.exists(nonrep_path))
    if not needs_recompute and os.path.exists(info_path):
        try:
            check = np.load(info_path, allow_pickle=True)
            if 'periods' not in check:
                needs_recompute = True
        except Exception:
            needs_recompute = True

    if needs_recompute:
        os.makedirs(repet_dir, exist_ok=True)
        print(f"[lab-repet] Computing for {Path(filepath).name}...")

        y_play, sr_play = sf.read(filepath)
        y_mono, sr_mono = librosa.load(filepath, sr=None, mono=True)

        y_rep, y_nonrep, info = repet(y_mono, sr_mono)

        # Save info for later PNG rendering
        periods_arr = np.array(info['periods'])  # list of (frames, seconds) tuples
        np.savez(os.path.join(repet_dir, 'info.npz'),
                 beat_spec=info['beat_spec'],
                 mask=info['mask'],
                 magnitude=info['magnitude'],
                 periods=periods_arr,
                 n_segments=info['n_segments'],
                 hop_length=info['hop_length'],
                 sr=info['sr'],
                 sharpness=info['sharpness'],
                 percentile=info['percentile'])

        # REPET per channel for stereo playback
        if y_play.ndim == 1:
            y_play = y_play[:, np.newaxis]
        n_ch = y_play.shape[1]
        y_rep_play = np.zeros_like(y_play)
        y_nonrep_play = np.zeros_like(y_play)
        for ch in range(n_ch):
            D_ch = librosa.stft(y_play[:, ch], n_fft=2048, hop_length=512)
            mask = info['mask']
            min_frames = min(mask.shape[1], D_ch.shape[1])
            mask_ch = mask[:, :min_frames]
            D_rep_ch = np.zeros_like(D_ch)
            D_nonrep_ch = np.zeros_like(D_ch)
            D_rep_ch[:, :min_frames] = mask_ch * D_ch[:, :min_frames]
            D_nonrep_ch[:, :min_frames] = (1 - mask_ch) * D_ch[:, :min_frames]
            y_rep_play[:, ch] = librosa.istft(D_rep_ch, hop_length=512,
                                               length=y_play.shape[0])
            y_nonrep_play[:, ch] = librosa.istft(D_nonrep_ch, hop_length=512,
                                                   length=y_play.shape[0])

        if n_ch == 1:
            y_rep_play = y_rep_play.squeeze()
            y_nonrep_play = y_nonrep_play.squeeze()

        sf.write(rep_path, y_rep_play, sr_play)
        sf.write(nonrep_path, y_nonrep_play, sr_play)
        print(f"[lab-repet] Done: {Path(filepath).name}")

    # Load info from disk
    info_data = np.load(info_path, allow_pickle=True)
    bs = info_data['beat_spec']
    mask = info_data['mask']
    periods = info_data['periods']  # array of (frames, seconds) pairs
    n_segments = int(info_data['n_segments'])
    hop_length = int(info_data['hop_length'])
    sr = int(info_data['sr'])
    sharpness = float(info_data['sharpness'])
    percentile = int(info_data['percentile'])

    # Load separated mono for spectrograms
    y_rep_mono, sr_rep = librosa.load(rep_path, sr=None, mono=True)
    y_nonrep_mono, sr_nonrep = librosa.load(nonrep_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y_rep_mono, sr=sr_rep)

    # Build diagnostic figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1.5, 2, 2], hspace=0.35)

    # Panel 1: Beat spectrum with all detected periods marked
    ax1 = fig.add_subplot(gs[0])
    lag_times = np.arange(len(bs)) * hop_length / sr
    ax1.plot(lag_times, bs, color='#00E5FF', linewidth=1.5)
    period_colors = ['#FF4081', '#FFD740', '#69F0AE', '#40C4FF']
    max_period_sec = 0
    for i, (pf, ps) in enumerate(periods):
        pf, ps = int(pf), float(ps)
        color = period_colors[i % len(period_colors)]
        n_seg = len(bs) // pf if pf > 0 else 0
        ax1.axvline(x=ps, color=color, linewidth=2, linestyle='--',
                    label=f'{ps:.3f}s ({n_seg} seg)')
        max_period_sec = max(max_period_sec, ps)
    ax1.set_xlim([0, min(lag_times[-1], max_period_sec * 3)])
    ax1.set_ylabel('Autocorrelation')
    period_strs = [f'{float(ps):.2f}s' for _, ps in periods]
    ax1.set_title(f'Beat Spectrum — {len(periods)} periods detected: {", ".join(period_strs)} '
                  f'| sharpness={sharpness}, percentile={percentile}', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Sharpened mask
    ax2 = fig.add_subplot(gs[1])
    librosa.display.specshow(
        mask, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='linear',
        ax=ax2, cmap='magma', vmin=0, vmax=1
    )
    ax2.set_ylabel('Frequency (Hz)')
    # Show mask statistics
    pct_high = np.mean(mask > 0.8) * 100
    pct_low = np.mean(mask < 0.2) * 100
    ax2.set_title(f'Sharpened Mask (^{sharpness}) — '
                  f'{pct_high:.0f}% > 0.8 (repeating), '
                  f'{pct_low:.0f}% < 0.2 (non-repeating)', fontsize=11)

    # Panel 3: Repeating spectrogram
    ax3 = fig.add_subplot(gs[2])
    mel_rep = librosa.feature.melspectrogram(
        y=y_rep_mono, sr=sr_rep, n_fft=2048, hop_length=512,
        n_mels=64, fmin=20, fmax=None
    )
    mel_rep_db = librosa.power_to_db(mel_rep, ref=np.max)
    librosa.display.specshow(
        mel_rep_db, sr=sr_rep, hop_length=512,
        x_axis='time', y_axis='mel', fmin=20, fmax=None,
        ax=ax3, cmap='magma'
    )
    ax3.set_ylabel('Frequency (Hz)')
    rep_rms = np.sqrt(np.mean(y_rep_mono ** 2))
    orig_y, _ = librosa.load(filepath, sr=None, mono=True)
    orig_rms = np.sqrt(np.mean(orig_y ** 2))
    ax3.set_title(f'Repeating Layer — {rep_rms/orig_rms*100:.0f}% of original RMS', fontsize=11)

    # Panel 4: Non-repeating spectrogram
    ax4 = fig.add_subplot(gs[3])
    mel_nonrep = librosa.feature.melspectrogram(
        y=y_nonrep_mono, sr=sr_nonrep, n_fft=2048, hop_length=512,
        n_mels=64, fmin=20, fmax=None
    )
    mel_nonrep_db = librosa.power_to_db(mel_nonrep, ref=np.max)
    librosa.display.specshow(
        mel_nonrep_db, sr=sr_nonrep, hop_length=512,
        x_axis='time', y_axis='mel', fmin=20, fmax=None,
        ax=ax4, cmap='magma'
    )
    ax4.set_ylabel('Frequency (Hz)')
    nonrep_rms = np.sqrt(np.mean(y_nonrep_mono ** 2))
    ax4.set_title(f'Non-Repeating Layer — {nonrep_rms/orig_rms*100:.0f}% of original RMS', fontsize=11)

    filename = Path(filepath).name
    fig.suptitle(f'{filename} — lab-REPET', fontsize=14, fontweight='bold', y=0.995)
    fig.canvas.draw()

    # Cursor alignment from spectrogram panels
    x_left = ax3.transData.transform((0, 0))[0]
    x_right = ax3.transData.transform((duration, 0))[0]
    fig_width = fig.get_figwidth() * DPI

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor())
    matplotlib.pyplot.close(fig)
    png_bytes = buf.getvalue()

    headers = {
        'X-Left-Px': str(x_left),
        'X-Right-Px': str(x_right),
        'X-Png-Width': str(fig_width),
        'X-Duration': str(duration),
    }

    _render_cache[cache_key] = (png_bytes, headers)
    return png_bytes, headers


def render_lab_nmf(filepath):
    """Run online NMF source decomposition, render per-source activation/spectrogram PNG."""
    cache_key = (filepath, 'lab-nmf')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    import numpy as np
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    separation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'research', 'separation')
    if separation_dir not in sys.path:
        sys.path.insert(0, separation_dir)
    from nmf_separation import OnlineNMF

    dict_path = os.path.join(separation_dir, 'dictionaries.npz')
    if not os.path.exists(dict_path):
        raise FileNotFoundError(
            "NMF dictionaries not found. Run: cd research/separation && python train_nmf.py")

    print(f"[lab-nmf] Processing {Path(filepath).name}...")
    nmf = OnlineNMF.from_file(dict_path)

    stem_name = Path(filepath).stem
    nmf_dir = os.path.join(SEGMENTS_DIR, 'separated', 'nmf', stem_name)

    y, sr = librosa.load(filepath, sr=nmf.sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    result = nmf.process_audio_offline(y, sr)

    # Synthesize per-source audio via Wiener masking on STFT
    source_names_str = [str(n) for n in nmf.source_names]
    wavs_exist = all(
        os.path.exists(os.path.join(nmf_dir, f'{n}.wav'))
        for n in source_names_str
    )
    if not wavs_exist:
        import soundfile as sf
        os.makedirs(nmf_dir, exist_ok=True)

        # Load original audio for playback (may be stereo)
        y_play, sr_play = sf.read(filepath)
        if y_play.ndim == 1:
            y_play = y_play[:, np.newaxis]
        n_ch = y_play.shape[1]

        for name in source_names_str:
            mask_mel = result['masks'][name]  # (n_mels, n_frames)

            # Upsample mel mask to full STFT resolution via mel filterbank pseudoinverse
            mel_basis = nmf.mel_basis  # (n_mels, n_fft//2+1)
            # Pseudoinverse: (n_fft//2+1, n_mels) @ (n_mels, n_frames) -> (n_fft//2+1, n_frames)
            mel_pinv = np.linalg.pinv(mel_basis)
            mask_stft = np.clip(mel_pinv @ mask_mel, 0, 1)

            y_source = np.zeros_like(y_play)
            for ch in range(n_ch):
                D_ch = librosa.stft(y_play[:, ch], n_fft=nmf.n_fft,
                                     hop_length=nmf.hop_length)
                # Match frame count
                min_f = min(mask_stft.shape[1], D_ch.shape[1])
                D_masked = np.zeros_like(D_ch)
                D_masked[:, :min_f] = mask_stft[:, :min_f] * D_ch[:, :min_f]
                y_source[:, ch] = librosa.istft(D_masked, hop_length=nmf.hop_length,
                                                 length=y_play.shape[0])

            if n_ch == 1:
                y_source = y_source.squeeze()
            sf.write(os.path.join(nmf_dir, f'{name}.wav'), y_source, sr_play)

    times = result['times']
    source_names = nmf.source_names
    n_sources = len(source_names)

    # Compute relative energy for summary
    total_act = sum(result['activations'][n] for n in source_names)
    rel_energy = {}
    for name in source_names:
        ratio = result['activations'][name] / (total_act + 1e-10)
        rel_energy[name] = ratio.mean() * 100

    # Build figure: activation curves on top, then per-source masked spectrograms
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 3 + 2.5 * n_sources))
    gs = gridspec.GridSpec(1 + n_sources, 1,
                           height_ratios=[1.5] + [2] * n_sources,
                           hspace=0.35)

    source_colors = {
        'drums': '#FF4081',
        'bass': '#FF9100',
        'vocals': '#40C4FF',
        'other': '#69F0AE',
    }

    # Panel 1: Activation curves (all sources overlaid)
    ax_act = fig.add_subplot(gs[0])
    for name in source_names:
        act = result['activations'][name]
        # Normalize each source for visual comparison
        act_norm = act / (act.max() + 1e-10)
        color = source_colors.get(str(name), '#FFFFFF')
        ax_act.plot(times, act_norm, color=color, linewidth=1.5,
                    label=f'{name} ({rel_energy[name]:.0f}%)', alpha=0.9)
    ax_act.set_xlim([0, duration])
    ax_act.set_ylim([0, 1.1])
    ax_act.set_ylabel('Activation')
    ax_act.set_title(f'Source Activations (normalized) — 8 tracks, K=10/source, 0.07ms/frame',
                     fontsize=11)
    ax_act.legend(loc='upper right', fontsize=9)
    ax_act.grid(True, alpha=0.2)

    # Per-source masked spectrograms
    mel_db = librosa.power_to_db(result['mel_spec'], ref=np.max)
    axes = [ax_act]

    for i, name in enumerate(source_names):
        ax = fig.add_subplot(gs[1 + i])
        # Apply Wiener mask to mel spectrogram
        masked = result['masks'][name] * result['mel_spec']
        masked_db = librosa.power_to_db(masked, ref=np.max(result['mel_spec']))
        librosa.display.specshow(
            masked_db, sr=sr, hop_length=nmf.hop_length,
            x_axis='time', y_axis='mel', fmin=20, fmax=sr // 2,
            ax=ax, cmap='magma', vmin=mel_db.min(), vmax=mel_db.max()
        )
        color = source_colors.get(str(name), '#FFFFFF')
        ax.set_ylabel(f'{name}', fontsize=12, fontweight='bold', color=color)
        ax.set_title(f'{name} — {rel_energy[name]:.0f}% of energy', fontsize=11)
        axes.append(ax)

    filename = Path(filepath).name
    fig.suptitle(f'{filename} — lab-NMF (supervised, 4 sources)',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.canvas.draw()

    x_left = axes[1].transData.transform((0, 0))[0]
    x_right = axes[1].transData.transform((duration, 0))[0]
    fig_width = fig.get_figwidth() * DPI

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor())
    matplotlib.pyplot.close(fig)
    png_bytes = buf.getvalue()

    headers = {
        'X-Left-Px': str(x_left),
        'X-Right-Px': str(x_right),
        'X-Png-Width': str(fig_width),
        'X-Duration': str(duration),
    }

    _render_cache[cache_key] = (png_bytes, headers)
    print(f"[lab-nmf] Done: {filename}")
    return png_bytes, headers


def render_lab(filepath):
    """Render experimental features (spectral flatness, chromagram, spectral contrast, ZCR)."""
    cache_key = (filepath, 'lab')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    import numpy as np
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.style.use('dark_background')

    y, sr = librosa.load(filepath, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    n_fft = 2048
    hop_length = 512
    times = librosa.frames_to_time(
        np.arange(librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length).shape[1]),
        sr=sr, hop_length=hop_length
    )

    # Compute features
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 2, 2, 1], hspace=0.3)

    # Panel 1: Spectral Flatness
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, flatness, color='#80cbc4', linewidth=0.8)
    ax1.fill_between(times, flatness, alpha=0.3, color='#80cbc4')
    ax1.set_xlim([0, duration])
    ax1.set_ylim([0, max(0.01, np.max(flatness) * 1.1)])
    ax1.set_ylabel('Flatness')
    ax1.set_title('Spectral Flatness — 0 = pure tone, 1 = white noise', fontsize=11)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Chromagram
    ax2 = fig.add_subplot(gs[1])
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time',
                             sr=sr, hop_length=hop_length, ax=ax2, cmap='magma')
    ax2.set_xlim([0, duration])
    ax2.set_title('Chromagram — pitch class energy over time', fontsize=11)

    # Panel 3: Spectral Contrast
    ax3 = fig.add_subplot(gs[2])
    librosa.display.specshow(contrast, x_axis='time', sr=sr, hop_length=hop_length,
                             ax=ax3, cmap='inferno')
    ax3.set_xlim([0, duration])
    ax3.set_ylabel('Band')
    ax3.set_title('Spectral Contrast — peak-to-valley difference per frequency band', fontsize=11)

    # Panel 4: Zero Crossing Rate
    zcr_times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(zcr_times, zcr, color='#ffab91', linewidth=0.8)
    ax4.fill_between(zcr_times, zcr, alpha=0.3, color='#ffab91')
    ax4.set_xlim([0, duration])
    ax4.set_ylabel('ZCR')
    ax4.set_title('Zero Crossing Rate — high = percussive/noisy, low = tonal', fontsize=11)
    ax4.grid(True, alpha=0.2)

    filename = Path(filepath).name
    fig.suptitle(f'{filename} — Lab', fontsize=14, fontweight='bold', y=0.995)
    fig.canvas.draw()

    ax = ax1
    x_left = ax.transData.transform((0, 0))[0]
    x_right = ax.transData.transform((duration, 0))[0]
    fig_width = fig.get_figwidth() * DPI

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor())
    matplotlib.pyplot.close(fig)
    png_bytes = buf.getvalue()

    headers = {
        'X-Left-Px': str(x_left),
        'X-Right-Px': str(x_right),
        'X-Png-Width': str(fig_width),
        'X-Duration': str(duration),
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


def get_recording_level():
    """Return downsampled waveform + RMS of recent audio. ~100ms of data."""
    if _recording is None:
        return {'recording': False}

    import numpy as np

    frames = _recording['frames']
    if not frames:
        return {'recording': True, 'rms': 0, 'waveform': []}

    # Take last ~100ms of audio (4410 samples at 44100Hz)
    # Frames are arrays of shape (chunk_size, 2)
    target_samples = 4410
    recent = []
    total = 0
    for chunk in reversed(frames):
        recent.append(chunk)
        total += len(chunk)
        if total >= target_samples:
            break

    recent.reverse()
    block = np.concatenate(recent)[-target_samples:]

    # Mix to mono for display
    if block.ndim > 1:
        mono = block.mean(axis=1)
    else:
        mono = block

    # RMS
    rms = float(np.sqrt(np.mean(mono ** 2)))

    # Downsample to ~150 points for canvas drawing
    n_points = 150
    step = max(1, len(mono) // n_points)
    # Take max absolute value per chunk for a peak envelope view
    waveform = []
    for i in range(0, len(mono) - step + 1, step):
        chunk = mono[i:i + step]
        waveform.append(float(np.max(np.abs(chunk))))

    return {'recording': True, 'rms': rms, 'waveform': waveform}


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
.tab-dropdown { position: relative; }
.tab-dropdown-menu {
    display: none; position: absolute; top: 100%; left: 0; z-index: 100;
    background: #1a1a2e; border: 1px solid #333; border-top: none;
    min-width: 160px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.tab-dropdown.open .tab-dropdown-menu { display: block; }
.tab-dropdown-item {
    padding: 8px 20px; cursor: pointer; font-size: 13px; color: #888;
    transition: all 0.15s;
}
.tab-dropdown-item:hover { color: #ccc; background: #16213e; }
.tab-dropdown-item.active { color: #e94560; }

/* Auth UI */
.auth-area { position: relative; margin-left: 12px; }
.auth-link {
    color: #888; cursor: pointer; font-size: 13px; padding: 4px 10px;
    border: 1px solid #444; border-radius: 4px; transition: all 0.15s;
    user-select: none;
}
.auth-link:hover { color: #ccc; border-color: #666; }
.auth-link.authed { color: #4caf50; border-color: #4caf50; }
.auth-box {
    display: none; position: absolute; top: calc(100% + 8px); right: 0;
    background: #1a1a2e; border: 1px solid #444; border-radius: 6px;
    padding: 12px; z-index: 200; min-width: 200px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
.auth-box.open { display: flex; gap: 8px; flex-direction: column; }
.auth-box input {
    background: #0f0f23; border: 1px solid #444; color: #eee; padding: 6px 10px;
    border-radius: 4px; font-size: 13px; outline: none;
}
.auth-box input:focus { border-color: #e94560; }
.auth-box button {
    background: #e94560; color: #fff; border: none; padding: 6px 12px;
    border-radius: 4px; cursor: pointer; font-size: 13px;
}
.auth-box button:hover { background: #c73652; }
.auth-error { color: #ff6b6b; font-size: 12px; display: none; }

/* Locked tabs */
.tab.locked, .tab-dropdown-item.locked {
    opacity: 0.35; cursor: not-allowed !important;
}
.tab.locked:hover, .tab-dropdown-item.locked:hover {
    color: #888 !important; background: transparent !important;
}

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

/* Compute prompt */
.compute-prompt { text-align: center; }
.compute-btn {
    background: #e94560; color: #fff; border: none; padding: 12px 28px;
    border-radius: 6px; font-size: 16px; cursor: pointer; transition: background 0.15s;
}
.compute-btn:hover { background: #c73652; }
.compute-desc { color: #888; font-size: 13px; margin-top: 10px; }

/* Welcome panel */
.welcome-panel {
    display: none; align-items: center; justify-content: center;
    width: 100%; height: 100%; padding: 20px; box-sizing: border-box;
}
.welcome-card {
    max-width: 560px; width: 100%; color: #ccc;
    line-height: 1.7; font-size: 14px;
}
.welcome-card h1 { color: #e94560; font-size: 22px; margin: 0 0 6px 0; }
.welcome-card .subtitle { color: #888; font-size: 13px; margin-bottom: 20px; }
.welcome-card p { margin: 12px 0; }
.welcome-card .storage-note {
    background: #1a1a2e; border: 1px solid #333; border-radius: 6px;
    padding: 12px 14px; font-size: 12px; color: #aaa; margin: 16px 0;
}
.welcome-card .storage-note code { color: #e94560; }
.welcome-card .contact-link {
    color: #4fc3f7; text-decoration: none; cursor: pointer;
    border-bottom: 1px dotted #4fc3f7;
}
.welcome-card .contact-link:hover { color: #81d4fa; border-color: #81d4fa; }

/* Upload button */
.upload-btn {
    background: none; border: 1px solid #555; color: #888; border-radius: 4px;
    padding: 5px 12px; cursor: pointer; font-size: 13px; transition: all 0.15s;
}
.upload-btn:hover { border-color: #e94560; color: #ccc; }

/* Drag-and-drop overlay */
.drop-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(233, 69, 96, 0.1); border: 3px dashed #e94560;
    display: none; align-items: center; justify-content: center; z-index: 500;
}
.drop-overlay.active { display: flex; }
.drop-overlay-text { font-size: 24px; color: #e94560; pointer-events: none; }

/* Upload progress */
.upload-progress {
    position: fixed; bottom: 20px; right: 20px; background: #1a1a2e;
    border: 1px solid #444; border-radius: 8px; padding: 12px 20px;
    color: #ccc; font-size: 13px; z-index: 300; display: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}

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
.info-panel h2 { color: #90a4ae; font-size: 18px; margin: 24px 0 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }
.info-panel h2:first-child { margin-top: 0; }
.info-panel h3 { color: #e0e0e0; font-size: 15px; margin: 16px 0 6px; }
.info-panel p { margin: 6px 0; }
.info-panel .origin { color: #888; font-style: italic; }
.info-panel .verdict { color: #4fc3f7; font-weight: 600; }
.info-panel .tag {
    display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: 11px;
    font-weight: 600; margin-left: 8px; vertical-align: middle;
}
.info-panel .tag.core { background: #1b5e20; color: #a5d6a7; }
.info-panel .tag.common { background: #0d47a1; color: #90caf9; }
.info-panel .tag.deprecated { background: #4e342e; color: #bcaaa4; }
.info-panel .tag.custom { background: #4a148c; color: #ce93d8; }
.info-panel .tag.gap { background: #b71c1c; color: #ef9a9a; }
.info-panel .tag.experimental { background: #0d47a1; color: #64b5f6; }
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
    gap: 20px; padding: 40px; width: 100%; max-width: 600px;
}
.record-waveform {
    width: 100%; height: 120px; background: #111; border-radius: 4px;
    border: 1px solid #333;
}
.record-level {
    display: flex; align-items: center; gap: 10px; width: 100%; font-size: 12px;
}
.record-level-bar {
    flex: 1; height: 8px; background: #222; border-radius: 4px; overflow: hidden;
}
.record-level-fill {
    height: 100%; background: #e94560; width: 0%; transition: width 0.05s;
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
.file-item {
    display: flex; align-items: center; gap: 8px; padding: 8px 10px;
    background: #1a1a2e; border: 1px solid #333; border-radius: 4px; margin-bottom: 4px;
}
.file-item:hover { border-color: #555; }
.file-item .file-name {
    flex: 1; color: #ccc; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer;
}
.file-item .file-name:hover { color: #e94560; }
.file-item .file-dur { color: #666; font-size: 11px; font-family: monospace; min-width: 40px; text-align: right; }
.file-item button {
    background: none; border: 1px solid #444; color: #888; font-size: 11px; padding: 2px 8px;
    border-radius: 3px; cursor: pointer;
}
.file-item button:hover { color: #ccc; border-color: #888; }
.file-item button.del:hover { color: #e94560; border-color: #e94560; }
.effects-panel {
    display: none; flex-direction: column; gap: 8px; padding: 20px;
    width: 100%; max-width: 600px; overflow-y: auto; max-height: 100%;
}
.effects-panel.visible { display: flex; }
.effect-card {
    display: flex; align-items: center; gap: 12px; padding: 10px 14px;
    background: #16213e; border: 1px solid #333; border-radius: 6px;
    cursor: grab; transition: border-color 0.15s, background 0.15s;
}
.effect-card:hover { border-color: #555; }
.effect-card.dragging { opacity: 0.4; }
.effect-card.drag-over { border-color: #e94560; background: #1a2a4e; }
.effect-card.running { border-color: #e94560; }
.effect-drag {
    color: #555; font-size: 16px; cursor: grab; user-select: none;
    line-height: 1;
}
.effect-name {
    flex: 1; font-size: 14px; font-weight: 500; color: #e0e0e0;
}
.effect-name .running-dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: #e94560; margin-left: 6px; vertical-align: middle;
    animation: pulse 1.5s ease-in-out infinite;
}
.effect-stars {
    display: flex; gap: 2px;
}
.effect-star {
    width: 16px; height: 16px; border-radius: 50%; border: 1.5px solid #555;
    background: transparent; cursor: pointer; transition: all 0.1s;
    padding: 0;
}
.effect-star:hover { border-color: #e94560; }
.effect-star.filled { background: #e94560; border-color: #e94560; }
.effect-toggle {
    background: #333; color: #e0e0e0; border: 1px solid #555; border-radius: 4px;
    padding: 4px 12px; cursor: pointer; font-size: 12px; white-space: nowrap;
}
.effect-toggle:hover { border-color: #e94560; }
.effect-toggle.stop { background: #b71c1c; border-color: #e94560; }
</style>
</head>
<body>

<div class="header">
    <h1>Audio Explorer</h1>
    <select id="filePicker"></select>
    <label class="upload-btn" title="Upload WAV file">
        &#8679; Upload
        <input type="file" id="uploadInput" accept=".wav,audio/wav" style="display:none"
            onchange="handleFileUpload(this.files)">
    </label>
    <button class="play-btn" id="playBtn" title="Play/Pause">&#9654;</button>
    <button class="play-btn" id="replayBtn" title="Replay from start">&#8634;</button>
    <span class="time" id="timeDisplay">0:00.000 / 0:00.000</span>
    <span style="display:flex;align-items:center;gap:6px;margin-left:auto;">
        <span style="font-size:14px;cursor:pointer;" id="volIcon" title="Mute/unmute">&#128266;</span>
        <input type="range" id="volSlider" min="0" max="1" step="0.01" value="0.8"
            style="width:80px;accent-color:#e94560;cursor:pointer;">
    </span>
    <div class="auth-area" id="authArea">
        <span class="auth-link" id="authLink" onclick="toggleAuthBox()">Sign In</span>
        <div class="auth-box" id="authBox">
            <input type="password" id="passcodeInput" placeholder="Passcode"
                onkeydown="if(event.key==='Enter')submitPasscode()">
            <button onclick="submitPasscode()">Go</button>
            <div class="auth-error" id="authError"></div>
        </div>
    </div>
</div>

<div class="tabs">
    <div class="tab" data-tab="welcome" id="welcomeTab">Welcome</div>
    <div class="tab" data-tab="record">Record</div>
    <div class="tab active" data-tab="analysis">Analysis</div>
    <div class="tab" data-tab="annotations" id="annTab">Annotations</div>
    <div class="tab" data-tab="stems">Stems (Demucs)</div>
    <div class="tab" data-tab="hpss">Stems (HPSS)</div>
    <div class="tab-dropdown">
        <div class="tab" id="labDropdownToggle">Lab &#9662;</div>
        <div class="tab-dropdown-menu" id="labDropdownMenu">
            <div class="tab-dropdown-item" data-tab="lab-repet">REPET</div>
            <div class="tab-dropdown-item" data-tab="lab-nmf">NMF</div>
            <div class="tab-dropdown-item" data-tab="lab">Feature Sandbox</div>
        </div>
    </div>
    <div class="tab" data-tab="effects">Effects</div>
    <div class="tab" data-tab="reference">Reference</div>
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
        <h2>Core Analysis</h2>
        <p style="color:#888;margin-bottom:16px;">These panels are always rendered. They represent the core audio properties that directly drive LED mapping and musical understanding.</p>

        <h3>Waveform + RMS overlay <span class="tag core">Core</span></h3>
        <p>Raw audio samples (white) with optional RMS energy overlay (yellow, toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#FFD740;">E</kbd>). RMS is the root-mean-square of the waveform in each frame &mdash; smoothed loudness over time, scaled to match waveform amplitude.</p>
        <p class="origin">Origin: Waveform display dates to oscilloscopes in the 1940s. RMS as a power measure dates to 19th-century electrical engineering; standard in audio since VU meters in the 1930s. Every DAW has both.</p>
        <p>Waveform shows transient attacks, silence, macro structure. RMS reveals energy trends the raw waveform hides &mdash; our research found that derivatives of RMS matter more than absolute values (climax brightens 58x faster than build, despite identical static RMS).</p>
        <p class="verdict">Non-negotiable. RMS overlay hidden by default to reduce visual clutter &mdash; enable when analyzing energy trajectories.</p>

        <h3>Mel Spectrogram <span class="tag core">Core</span></h3>
        <p>Short-time Fourier Transform (STFT) converted to mel scale and displayed as a heatmap. Time on x-axis, frequency on y-axis (low=bottom, high=top), color=loudness.</p>
        <p class="origin">Origin: The mel scale comes from Stevens, Volkmann &amp; Newman (1937) &mdash; psychoacoustic research showing humans perceive pitch logarithmically (200Hz&rarr;400Hz <em>sounds</em> the same as 400Hz&rarr;800Hz). The spectrogram (STFT) dates to Gabor (1946). Mel spectrograms became standard input for audio ML in the 1980s.</p>
        <p>You can <em>see</em> bass hits (bright blobs at bottom), vocals (middle bands), hi-hats (top). Harmonic content = horizontal lines. Percussive content = vertical lines &mdash; this is why HPSS works (median filtering by orientation).</p>
        <p class="verdict">The single most informative audio visualization. Industry standard.</p>

        <h3>Band Energy <span class="tag common">Common</span></h3>
        <p>The mel spectrogram collapsed into 5 bands &mdash; Sub-bass (20&ndash;80Hz), Bass (80&ndash;250Hz), Mids (250&ndash;2kHz), High-mids (2&ndash;6kHz), Treble (6&ndash;8kHz) &mdash; each plotted as a line over time.</p>
        <p class="origin">Origin: Multi-band meters from mixing engineering. Band boundaries follow critical band theory (Fletcher, 1940s) and PA crossover points. &ldquo;Bass energy over time&rdquo; is the foundation of almost every audio-reactive LED system (WLED-SR&rsquo;s entire beat detection = threshold on the bass bin).</p>
        <p>Shows which frequency range dominates at each moment. A bass drop = Sub-bass/Bass spike. A cymbal crash = treble spike.</p>
        <p class="verdict">Standard in audio-reactive systems. Useful reference for understanding frequency content.</p>

        <h3>Annotations <span class="tag custom">Custom</span></h3>
        <p>Your own tap data overlaid on the analysis &mdash; beat taps, section changes, airy moments, flourishes. Whatever layers exist in the <code>.annotations.yaml</code> file.</p>
        <p class="origin">Origin: Custom to this project. Our &ldquo;test set&rdquo; for evaluating audio features against human perception.</p>
        <p>Note: tap annotations exhibit <strong>tactus ambiguity</strong> &mdash; listeners lock onto different metrical layers (kick, snare, off-beat) per song, so taps may be phase-shifted from the &ldquo;metric beat&rdquo; by 100&ndash;250ms (Martens 2011, London 2004). LEDs could exploit this: by flashing a specific layer, we may be able to <em>entrain</em> the audience&rsquo;s tactus rather than follow it.</p>
        <p class="verdict">Essential for research. Only shown when annotation data exists.</p>

        <h2>Exploratory Features</h2>
        <p style="color:#888;margin-bottom:16px;">Real audio properties, hidden by default. Not directly useful as raw indicators for LED mapping, but promising as inputs for <em>derived</em> features &mdash; running averages, deviation from context, rate-of-change, etc.</p>

        <h3>Onset Strength <span class="tag experimental">Experimental</span></h3>
        <p>Spectral flux &mdash; how much the spectrum <em>changes</em> between adjacent frames. Peaks = &ldquo;something new happened.&rdquo; Toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#00E5FF;">O</kbd>.</p>
        <p>Measures something real (spectral novelty) but raw values don&rsquo;t map to perceived beats &mdash; F1=0.435 on Harmonix, only 48.5% of user taps align. Potential as a derived feature (e.g. deviation from local average could signal section changes).</p>

        <h3>Spectral Centroid <span class="tag experimental">Experimental</span></h3>
        <p>The &ldquo;center of mass&rdquo; of the spectrum &mdash; the frequency where half the energy is above and half below. Often called &ldquo;brightness.&rdquo; Toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#B388FF;">C</kbd>.</p>
        <p>A standard timbral descriptor (Grey, 1977). Raw centroid isn&rsquo;t directly useful for LED mapping, but derived features (running average, deviation = &ldquo;airiness&rdquo;) could detect timbral shifts between sections.</p>

        <h3>Librosa Beats <span class="tag deprecated">Deprecated</span></h3>
        <p>Beat tracking via <code>librosa.beat.beat_track</code> &mdash; estimates tempo then snaps onset peaks to a grid. Toggle with <kbd style="background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;color:#FF1744;">B</kbd>.</p>
        <p><strong>Why second class:</strong> Doubles tempo on syncopated rock (161.5 vs ~83 BPM on Tool&rsquo;s Opiate). Built on top of onset strength, which is itself a weak beat discriminator. Best F1=0.500 on dense rock.</p>
        <p class="verdict">Useful as a sanity check. Not reliable enough to drive LED effects directly.</p>

        <h3>RMS Derivative <span class="tag core">Core</span></h3>
        <p>Rate-of-change of loudness (dRMS/dt). Red = getting louder, blue = getting quieter. Our most validated finding: a build and its climax can have identical RMS, but the climax brightens 58x faster.</p>
        <p class="verdict">Now on the Analysis panel. The signal that distinguishes builds from drops.</p>

        <h2>Lab &rsaquo; NMF</h2>
        <p><strong>Online Supervised NMF</strong>: pre-trained spectral dictionaries (10 components per source from 8 demucs-separated tracks) decompose each audio frame into drums/bass/vocals/other activations. 0.07ms/frame &mdash; ESP32-feasible.</p>
        <p>Top panel: per-source activation curves (normalized). Lower panels: Wiener-masked spectrograms per source. No stem audio toggle (NMF produces energy estimates, not separated audio).</p>
        <p class="verdict">The most promising approach for real-time LED source attribution on ESP32. Dictionary: 64 mel bins &times; 40 components = 10KB.</p>

        <h2>Lab &rsaquo; REPET</h2>
        <p><strong>REPET</strong> (REPeating Pattern Extraction Technique) separates audio into repeating (background) and non-repeating (foreground) layers by detecting cyclic patterns in the spectrogram. No ML &mdash; just autocorrelation + median filtering + soft masking. ESP32-feasible.</p>
        <p>Panels: beat spectrum (with detected period), soft mask, and spectrograms of each separated layer. Use 1/2 keys to solo/mute layers.</p>
        <p class="verdict">Based on Rafii &amp; Pardo 2012. Tests whether pattern repetition alone can usefully decompose music for LED mapping.</p>

        <h2>Lab &rsaquo; Feature Sandbox</h2>
        <p>Four experimental features: spectral flatness, chromagram, spectral contrast, and zero crossing rate. Use this tab to evaluate whether these are useful indicators for LED mapping.</p>
    </div>
    <div class="record-panel" id="recordPanel">
        <div id="recordLocal">
            <canvas class="record-waveform" id="recordWaveform"></canvas>
            <div class="record-level">
                <span style="color:#888;">Level</span>
                <div class="record-level-bar"><div class="record-level-fill" id="recordLevelFill"></div></div>
                <span style="color:#888;font-family:monospace;min-width:40px;" id="recordLevelDb">-∞ dB</span>
            </div>
            <input type="text" id="recordName" placeholder="segment name (e.g. tool_lateralus_intro)" spellcheck="false">
            <button class="record-btn" id="recordBtn" onclick="toggleRecord()"><span class="dot"></span></button>
            <div class="record-elapsed" id="recordElapsed">0:00.0</div>
            <div class="record-status" id="recordStatus">Click to record from BlackHole</div>
            <hr style="border-color:#333; margin:24px 0;">
            <h3 style="color:#ccc; font-size:14px; margin-bottom:8px;">Your Files</h3>
            <div id="fileManagerLocal" style="max-height:300px; overflow-y:auto;"></div>
        </div>
        <div id="recordPublic" style="display:none; max-width:600px; line-height:1.6;">
            <h3 style="color:#ccc;">Record Audio</h3>
            <p style="color:#aaa; margin-bottom:12px;">Record from your microphone or system audio (via <a href="https://existential.audio/blackhole/" target="_blank" style="color:#e94560;">BlackHole</a>).</p>
            <div style="margin-bottom:12px;">
                <label style="color:#888; font-size:12px; display:block; margin-bottom:4px;">Audio Input</label>
                <select id="audioDeviceSelect" style="background:#1a1a2e; color:#ccc; border:1px solid #333; padding:6px 10px; border-radius:4px; width:100%; font-size:13px;"></select>
            </div>
            <canvas class="record-waveform" id="browserRecordWaveform"></canvas>
            <div class="record-level">
                <span style="color:#888;">Level</span>
                <div class="record-level-bar"><div class="record-level-fill" id="browserLevelFill"></div></div>
                <span style="color:#888;font-family:monospace;min-width:40px;" id="browserLevelDb">&minus;&infin; dB</span>
            </div>
            <input type="text" id="browserRecordName" placeholder="segment name (optional)" spellcheck="false" style="background:#1a1a2e; color:#ccc; border:1px solid #333; padding:6px 10px; border-radius:4px; width:100%; box-sizing:border-box; margin-bottom:8px;">
            <button class="record-btn" id="browserRecordBtn" onclick="toggleBrowserRecord()"><span class="dot"></span></button>
            <div class="record-elapsed" id="browserRecordElapsed">0:00.0</div>
            <div class="record-status" id="browserRecordStatus">Select an audio input and click to record</div>
            <hr style="border-color:#333; margin:24px 0;">
            <details style="color:#aaa;">
                <summary style="color:#ccc; cursor:pointer;">Setup: Record system audio with BlackHole (macOS)</summary>
                <ol style="padding-left:20px; margin-top:8px;">
                    <li>Install BlackHole 2ch from <a href="https://existential.audio/blackhole/" target="_blank" style="color:#e94560;">existential.audio/blackhole</a></li>
                    <li>In System Settings &rarr; Sound &rarr; Output, select <strong>BlackHole 2ch</strong></li>
                    <li>Refresh this page &mdash; BlackHole will appear in the input dropdown above</li>
                </ol>
                <p style="color:#888; font-size:12px; margin-top:8px;"><strong>Note:</strong> With this setup you won't hear the audio while recording. To hear it too, open <strong>Audio MIDI Setup</strong>, click <strong>+</strong> &rarr; <strong>Create Multi-Output Device</strong>, check both your speakers and BlackHole, then set that as your system output instead.</p>
            </details>
            <hr style="border-color:#333; margin:24px 0;">
            <h3 style="color:#ccc;">Your Files</h3>
            <p style="color:#aaa; margin-bottom:12px;">Drag and drop a <code>.wav</code> file anywhere on the page, or use the <strong>Upload</strong> button. Files persist in your browser across sessions.</p>
            <div id="fileManager" style="max-height:300px; overflow-y:auto;"></div>
        </div>
    </div>
    <div class="effects-panel" id="effectsPanel"></div>
    <div class="welcome-panel" id="welcomePanel" style="display:none;">
        <div class="welcome-card">
            <h1>Audio Explorer</h1>
            <div class="subtitle">by Seth Drew</div>
            <p>This is an interactive audio visualization and audio interactivity research testbed. It's configured to require no user accounts. It allows you to upload or record and analyze any audio you would like.</p>
            <p>This audio is stored on your machine only for copyright reasons, which you can clean up using the <strong>Record</strong> tab or manually via:</p>
            <div class="storage-note">
                <code>Settings &rarr; Privacy &rarr; Site Data &rarr; audio.sethdrew.com</code>
            </div>
            <p>Glad to have you! If you have any questions feel free to reach out directly to me with questions or improvements &mdash; <span class="contact-link" id="contactLink" onclick="copyContact()">copy my email</span>
            <span class="contact-copied" id="contactCopied" style="display:none; color:#4caf50; font-size:12px;">Copied!</span></p>
            <p style="margin-top:20px; color:#888;">Select a file above or go to the <strong>Record</strong> tab to upload or record audio.</p>
        </div>
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

<div class="drop-overlay" id="dropOverlay">
    <span class="drop-overlay-text">Drop WAV file to upload</span>
</div>
<div class="upload-progress" id="uploadProgress">Uploading...</div>

<script>
// ── IndexedDB cache for analysis results ────────────────────────
const cacheDB = (() => {
    let db = null;
    const DB_NAME = 'led-viewer-cache';
    const DB_VERSION = 2;

    function open() {
        if (db) return Promise.resolve(db);
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            req.onupgradeneeded = (e) => {
                const d = req.result;
                if (!d.objectStoreNames.contains('panels')) d.createObjectStore('panels');
                if (!d.objectStoreNames.contains('audioFiles')) d.createObjectStore('audioFiles');
            };
            req.onsuccess = () => { db = req.result; resolve(db); };
            req.onerror = () => resolve(null);
        });
    }

    return {
        async get(key, store = 'panels') {
            try {
                const d = await open();
                if (!d) return null;
                return new Promise(resolve => {
                    const tx = d.transaction(store, 'readonly');
                    const req = tx.objectStore(store).get(key);
                    req.onsuccess = () => resolve(req.result || null);
                    req.onerror = () => resolve(null);
                });
            } catch { return null; }
        },
        async put(key, value, store = 'panels') {
            try {
                const d = await open();
                if (!d) return;
                const tx = d.transaction(store, 'readwrite');
                tx.objectStore(store).put(value, key);
            } catch {}
        },
        async getAll(store = 'panels') {
            try {
                const d = await open();
                if (!d) return [];
                return new Promise(resolve => {
                    const tx = d.transaction(store, 'readonly');
                    const req = tx.objectStore(store).getAll();
                    const keyReq = tx.objectStore(store).getAllKeys();
                    const results = {};
                    req.onsuccess = () => { results.values = req.result; };
                    keyReq.onsuccess = () => {
                        results.keys = keyReq.result;
                        const out = [];
                        for (let i = 0; i < results.keys.length; i++) {
                            out.push({ key: results.keys[i], value: results.values[i] });
                        }
                        resolve(out);
                    };
                    keyReq.onerror = () => resolve([]);
                });
            } catch { return []; }
        },
        async delete(key, store = 'panels') {
            try {
                const d = await open();
                if (!d) return;
                const tx = d.transaction(store, 'readwrite');
                tx.objectStore(store).delete(key);
            } catch {}
        }
    };
})();

async function cachedFetchPNG(url) {
    const cached = await cacheDB.get(url);
    if (cached) {
        return {
            blob: new Blob([cached.png], {type: 'image/png'}),
            pixelMapping: cached.pixelMapping
        };
    }

    const resp = await fetch(url);
    if (!resp.ok) return null;

    const pm = {
        xLeft: parseFloat(resp.headers.get('X-Left-Px')),
        xRight: parseFloat(resp.headers.get('X-Right-Px')),
        pngWidth: parseFloat(resp.headers.get('X-Png-Width')),
        duration: parseFloat(resp.headers.get('X-Duration')),
    };
    const blob = await resp.blob();
    const buf = await blob.arrayBuffer();

    await cacheDB.put(url, { png: buf, pixelMapping: pm });

    return { blob, pixelMapping: pm };
}

async function clearPanelCache(filePath) {
    // Clear all cached panels for a given file path
    const encoded = encodeURIComponent(filePath);
    const all = await cacheDB.getAll('panels');
    for (const { key } of all) {
        if (typeof key === 'string' && key.includes(encoded)) {
            await cacheDB.delete(key, 'panels');
        }
    }
}

// ── Auth ────────────────────────────────────────────────────────
let isAuthenticated = false;
let isPublicMode = false;
const LOCKED_TABS = new Set(['stems', 'hpss', 'lab-repet', 'lab-nmf', 'lab']);
const HIDDEN_TABS_PUBLIC = new Set(['effects']);
const HIDDEN_TABS_LOCAL = new Set(['welcome']);

async function checkAuth() {
    try {
        const resp = await fetch('/api/auth/status');
        const data = await resp.json();
        isAuthenticated = data.authenticated;
        isPublicMode = data.public;
        updateAuthUI();
        // Default to welcome tab on public mode (unless URL hash has a saved state)
        if (isPublicMode) {
            const saved = readHashState();
            if (!saved.tab && !saved.file) {
                currentTab = 'welcome';
                updateTabUI();
                loadPanel();
            }
        }
    } catch {}
}

function updateAuthUI() {
    const authLink = document.getElementById('authLink');
    const authArea = document.getElementById('authArea');
    if (!isPublicMode) {
        authArea.style.display = 'none';
        return;
    }
    authArea.style.display = 'block';
    if (isAuthenticated) {
        authLink.textContent = 'Signed In';
        authLink.classList.add('authed');
    } else {
        authLink.textContent = 'Sign In';
        authLink.classList.remove('authed');
    }
    // Toggle record panel content
    const recordLocal = document.getElementById('recordLocal');
    const recordPublic = document.getElementById('recordPublic');
    if (recordLocal && recordPublic) {
        recordLocal.style.display = isPublicMode ? 'none' : '';
        recordPublic.style.display = isPublicMode ? 'block' : 'none';
    }

    updateLockedTabs();
}

function updateLockedTabs() {
    // Lock/unlock tabs
    document.querySelectorAll('.tab[data-tab], .tab-dropdown-item[data-tab]').forEach(el => {
        const tab = el.dataset.tab;
        if (LOCKED_TABS.has(tab)) {
            if (isPublicMode && !isAuthenticated) {
                el.classList.add('locked');
                el.title = 'Sign in to unlock';
            } else {
                el.classList.remove('locked');
                el.title = '';
            }
        }
        if (HIDDEN_TABS_PUBLIC.has(tab)) {
            el.style.display = isPublicMode ? 'none' : '';
        }
        if (HIDDEN_TABS_LOCAL.has(tab)) {
            el.style.display = isPublicMode ? '' : 'none';
        }
    });
}

function toggleAuthBox() {
    if (isAuthenticated) {
        // Clicking "Signed In" logs out
        fetch('/api/auth/logout', {method: 'POST'}).then(() => {
            isAuthenticated = false;
            updateAuthUI();
        });
        return;
    }
    const box = document.getElementById('authBox');
    box.classList.toggle('open');
    if (box.classList.contains('open')) {
        document.getElementById('passcodeInput').focus();
    }
}

async function submitPasscode() {
    const input = document.getElementById('passcodeInput');
    const errEl = document.getElementById('authError');
    const passcode = input.value.trim();
    if (!passcode) return;

    const resp = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({passcode})
    });
    const data = await resp.json();
    if (data.ok) {
        isAuthenticated = true;
        input.value = '';
        errEl.style.display = 'none';
        document.getElementById('authBox').classList.remove('open');
        updateAuthUI();
    } else {
        errEl.textContent = data.error || 'Invalid passcode';
        errEl.style.display = 'block';
    }
}

// Close auth box on outside click
document.addEventListener('click', e => {
    const area = document.getElementById('authArea');
    if (area && !area.contains(e.target)) {
        document.getElementById('authBox').classList.remove('open');
    }
});

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
let currentTab = 'analysis'; // overridden to 'welcome' after checkAuth on public mode
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

    const sortedTaps = annotationTaps.slice().sort((a, b) => a - b);

    const resp = await fetch('/api/annotations/' + encodeURIComponent(currentFile), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ layer: layer, taps: sortedTaps })
    });

    if (resp.ok) {
        // Also cache annotations in IndexedDB for persistence
        const annKey = 'ann:' + currentFile;
        const existing = await cacheDB.get(annKey, 'audioFiles') || {};
        existing[layer] = sortedTaps;
        await cacheDB.put(annKey, existing, 'audioFiles');

        annotationTaps = [];
        updateAnnotationUI();
        drawTapMarkers();
        // Update file info to reflect new annotations
        const fileInfo = files.find(f => f.path === currentFile);
        if (fileInfo) fileInfo.has_annotations = true;
        // Clear cached panels so annotation render is fresh
        await clearPanelCache(currentFile);
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
let levelPollTimer = null;

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
    startLevelPolling();
}

async function stopRecording() {
    clearInterval(recordTimer);
    stopLevelPolling();
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
        await loadFileList();
        selectFile(data.filename);
        currentTab = 'analysis';
        updateTabUI();
        loadPanel();
    } else {
        document.getElementById('recordStatus').textContent = 'Error: ' + (data.error || 'failed');
    }
}

function startLevelPolling() {
    stopLevelPolling();
    levelPollTimer = setInterval(pollLevel, 70);
}

function stopLevelPolling() {
    if (levelPollTimer) { clearInterval(levelPollTimer); levelPollTimer = null; }
}

async function pollLevel() {
    try {
        const resp = await fetch('/api/record/level');
        const data = await resp.json();
        if (!data.recording || !data.waveform) return;
        drawRecordWaveform(data.waveform);
        // Update level bar + dB readout
        const rms = data.rms || 0;
        const pct = Math.min(100, rms * 300);  // scale for visibility (full scale ~0.33)
        document.getElementById('recordLevelFill').style.width = pct + '%';
        const db = rms > 0 ? (20 * Math.log10(rms)).toFixed(1) : '-Inf';
        document.getElementById('recordLevelDb').textContent = db + ' dB';
    } catch (e) {}
}

function drawRecordWaveform(waveform) {
    const canvas = document.getElementById('recordWaveform');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    // Center line
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (waveform.length === 0) return;

    // Draw mirrored peak envelope (bars)
    const barW = w / waveform.length;
    ctx.fillStyle = '#e94560';
    for (let i = 0; i < waveform.length; i++) {
        const amp = Math.min(1, waveform[i] * 3);  // scale for visibility
        const barH = amp * h * 0.45;
        const x = i * barW;
        ctx.fillRect(x, h / 2 - barH, Math.max(1, barW - 0.5), barH * 2);
    }
}

// ── Browser recording (public mode) ──────────────────────────────

let browserStream = null;
let browserAudioCtx = null;
let browserProcessor = null;
let browserAnalyser = null;
let browserChunks = [];
let browserRecording = false;
let browserRecordStart = null;
let browserRecordTimer = null;
let browserAnimFrame = null;

async function populateAudioDevices() {
    const select = document.getElementById('audioDeviceSelect');
    if (!select) return;
    try {
        // Request permission first so device labels are visible
        await navigator.mediaDevices.getUserMedia({ audio: true }).then(s => s.getTracks().forEach(t => t.stop()));
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(d => d.kind === 'audioinput');
        select.innerHTML = '';
        audioInputs.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            opt.textContent = d.label || ('Microphone ' + (select.options.length + 1));
            select.appendChild(opt);
        });
    } catch (e) {
        select.innerHTML = '<option>Microphone access denied</option>';
    }
}

async function toggleBrowserRecord() {
    if (browserRecording) {
        await stopBrowserRecord();
    } else {
        await startBrowserRecord();
    }
}

async function startBrowserRecord() {
    const select = document.getElementById('audioDeviceSelect');
    const deviceId = select ? select.value : undefined;
    const status = document.getElementById('browserRecordStatus');

    try {
        browserStream = await navigator.mediaDevices.getUserMedia({
            audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        });
    } catch (e) {
        status.textContent = 'Microphone access denied. Check browser permissions.';
        return;
    }

    browserAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
    const source = browserAudioCtx.createMediaStreamSource(browserStream);

    // Analyser for live waveform
    browserAnalyser = browserAudioCtx.createAnalyser();
    browserAnalyser.fftSize = 2048;
    source.connect(browserAnalyser);

    // ScriptProcessor to capture raw PCM
    browserProcessor = browserAudioCtx.createScriptProcessor(4096, 1, 1);
    browserChunks = [];
    browserProcessor.onaudioprocess = (e) => {
        if (browserRecording) {
            browserChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        }
    };
    source.connect(browserProcessor);
    browserProcessor.connect(browserAudioCtx.destination);

    browserRecording = true;
    browserRecordStart = Date.now();
    document.getElementById('browserRecordBtn').classList.add('recording');
    status.textContent = 'Recording... click to stop';

    browserRecordTimer = setInterval(() => {
        const elapsed = (Date.now() - browserRecordStart) / 1000;
        const m = Math.floor(elapsed / 60);
        const s = (elapsed - m * 60).toFixed(1).padStart(4, '0');
        document.getElementById('browserRecordElapsed').textContent = m + ':' + s;
    }, 100);

    // Animate waveform
    function drawLive() {
        if (!browserRecording) return;
        const canvas = document.getElementById('browserRecordWaveform');
        if (!canvas || !browserAnalyser) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        const bufLen = browserAnalyser.fftSize;
        const data = new Float32Array(bufLen);
        browserAnalyser.getFloatTimeDomainData(data);

        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = '#333';
        ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();

        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < bufLen; i++) {
            const x = (i / bufLen) * w;
            const y = (0.5 + data[i] * 0.45) * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Update level meter
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
        const rms = Math.sqrt(sum / data.length);
        const pct = Math.min(100, rms * 300);
        document.getElementById('browserLevelFill').style.width = pct + '%';
        const db = rms > 0 ? (20 * Math.log10(rms)).toFixed(1) : '-Inf';
        document.getElementById('browserLevelDb').textContent = db + ' dB';

        browserAnimFrame = requestAnimationFrame(drawLive);
    }
    drawLive();
}

async function stopBrowserRecord() {
    browserRecording = false;
    clearInterval(browserRecordTimer);
    if (browserAnimFrame) cancelAnimationFrame(browserAnimFrame);

    const status = document.getElementById('browserRecordStatus');
    const btn = document.getElementById('browserRecordBtn');
    btn.classList.remove('recording');
    status.textContent = 'Encoding WAV...';

    // Stop audio pipeline
    if (browserProcessor) { browserProcessor.disconnect(); browserProcessor = null; }
    if (browserAnalyser) { browserAnalyser.disconnect(); browserAnalyser = null; }
    if (browserAudioCtx) { browserAudioCtx.close(); browserAudioCtx = null; }
    if (browserStream) { browserStream.getTracks().forEach(t => t.stop()); browserStream = null; }

    // Encode WAV
    const wavBlob = encodeBrowserWAV(browserChunks, 44100);
    browserChunks = [];

    const nameInput = document.getElementById('browserRecordName');
    const name = (nameInput.value.trim() || 'recording_' + new Date().toISOString().slice(0,19).replace(/[:-]/g, '')) + '.wav';

    status.textContent = 'Uploading ' + name + '...';

    try {
        const data = await uploadWavBlob(wavBlob, name);
        if (data.ok) {
            const dur = (data.duration || 0).toFixed(1);
            status.textContent = 'Saved: ' + data.name + ' (' + dur + 's)';
            document.getElementById('browserRecordElapsed').textContent = '0:00.0';
            nameInput.value = '';
            await loadFileList(data.path);
        } else {
            status.textContent = 'Error: ' + (data.error || 'upload failed');
        }
    } catch (e) {
        status.textContent = 'Upload failed: ' + e.message;
    }
}

function encodeBrowserWAV(chunks, sampleRate) {
    let totalLength = 0;
    for (const c of chunks) totalLength += c.length;
    const samples = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) { samples.set(c, offset); offset += c.length; }

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeStr(off, str) { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); }

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

// Populate devices when record tab is shown
if (isPublicMode) populateAudioDevices();

// ── File picker ──────────────────────────────────────────────────

async function loadFileList(selectPath) {
    const resp = await fetch('/api/files');
    files = await resp.json();

    // Merge with locally cached files (IndexedDB) that may have been deleted from server
    const serverPaths = new Set(files.map(f => f.path));
    const serverNames = new Set(files.map(f => f.name));
    const localFiles = await cacheDB.getAll('audioFiles');
    for (const { key, value } of localFiles) {
        if (serverPaths.has(key)) continue;  // Already on server with same path
        // Check if server has this file under a sanitized name (stale IndexedDB entry)
        if (serverNames.has(value.name)) {
            cacheDB.delete(key, 'audioFiles');  // Clean up stale entry
            continue;
        }
        files.push({
            name: value.name,
            path: key,
            duration: 0,
            has_annotations: false,
            group: 'your files',
        });
    }

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
            const dur = f.duration ? formatTime(f.duration) : '?';
            const ann = f.has_annotations ? ' [ann]' : '';
            opt.textContent = f.name + ' (' + dur + ')' + ann;
            optgroup.appendChild(opt);
        });
        filePicker.appendChild(optgroup);
    }

    if (files.length > 0) {
        if (selectPath) {
            selectFile(selectPath);
        } else if (currentTab === 'welcome') {
            // Don't auto-select a file on the welcome tab — add placeholder option
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = 'Select a file...';
            placeholder.disabled = true;
            placeholder.selected = true;
            filePicker.insertBefore(placeholder, filePicker.firstChild);
        } else {
            const saved = readHashState();
            const savedFile = saved.file && files.find(f => f.path === saved.file);
            if (saved.tab) {
                currentTab = saved.tab;
                updateTabUI();
            }
            selectFile(savedFile ? savedFile.path : files[0].path);
        }
    }
}

async function selectFile(path) {
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

    // Ensure file exists on server (re-upload from IndexedDB if needed)
    if (path.startsWith('uploads/')) {
        await ensureFileOnServer(path);
    }

    // Sync annotations: fetch from server, cache in IndexedDB
    try {
        const annResp = await fetch('/api/annotations/' + encodeURIComponent(path));
        if (annResp.ok) {
            const annData = await annResp.json();
            if (Object.keys(annData).length > 0) {
                await cacheDB.put('ann:' + path, annData, 'audioFiles');
            }
        }
    } catch {}

    // Set audio source
    audio.src = '/audio/' + encodeURIComponent(path);
    audio.load();

    saveHashState();
    loadPanel();
}

filePicker.addEventListener('change', () => {
    if (currentTab === 'welcome') {
        currentTab = 'analysis';
        updateTabUI();
    }
    selectFile(filePicker.value);
});

// ── Tabs ─────────────────────────────────────────────────────────

const labTabs = new Set(['lab-repet', 'lab-nmf', 'lab']);
const labDropdown = document.querySelector('.tab-dropdown');
const labToggle = document.getElementById('labDropdownToggle');

function updateTabUI() {
    document.querySelectorAll('.tabs > .tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === currentTab);
    });
    // Lab dropdown: highlight toggle if a lab sub-tab is active
    labToggle.classList.toggle('active', labTabs.has(currentTab));
    document.querySelectorAll('.tab-dropdown-item').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === currentTab);
    });
}

function switchTab(tabId) {
    // Block locked tabs
    if (isPublicMode && !isAuthenticated && LOCKED_TABS.has(tabId)) return;
    const prev = currentTab;
    if ((prev === 'stems' || prev === 'hpss' || prev === 'lab-repet' || prev === 'lab-nmf') && tabId !== prev) cleanupStemAudio();
    currentTab = tabId;
    updateTabUI();
    saveHashState();
    loadPanel();
}

// Regular tab clicks
document.querySelectorAll('.tabs > .tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (tab.classList.contains('disabled')) return;
        if (tab.id === 'labDropdownToggle') return; // handled separately
        switchTab(tab.dataset.tab);
    });
});

// Lab dropdown toggle
labToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    labDropdown.classList.toggle('open');
});

// Lab dropdown item clicks
document.querySelectorAll('.tab-dropdown-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.stopPropagation();
        labDropdown.classList.remove('open');
        switchTab(item.dataset.tab);
    });
});

// Close dropdown on outside click
document.addEventListener('click', () => labDropdown.classList.remove('open'));

// ── Panel loading ────────────────────────────────────────────────

async function loadPanel() {
    // Show/hide annotation bar
    const annBar = document.getElementById('annotationBar');
    annBar.style.display = currentTab === 'annotations' ? 'flex' : 'none';

    const recordPanel = document.getElementById('recordPanel');
    const effectsPanel = document.getElementById('effectsPanel');
    const welcomePanel = document.getElementById('welcomePanel');

    if (currentTab === 'welcome') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'none';
        recordPanel.style.display = 'none';
        effectsPanel.style.display = 'none';
        welcomePanel.style.display = 'flex';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML = '';
        return;
    }
    welcomePanel.style.display = 'none';

    if (currentTab === 'reference') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'block';
        recordPanel.style.display = 'none';
        effectsPanel.style.display = 'none';
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
        effectsPanel.style.display = 'none';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML =
            isPublicMode ? 'Record from microphone or system audio' : 'Record audio from BlackHole loopback';
        if (isPublicMode) populateAudioDevices();
        renderFileManager();
        return;
    }
    if (currentTab === 'effects') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'none';
        recordPanel.style.display = 'none';
        effectsPanel.style.display = 'flex';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML =
            'Browse, start/stop, rate, and reorder audio-reactive LED effects';
        loadEffects();
        return;
    }
    imgContainer.style.display = 'inline-block';
    infoPanel.style.display = 'none';
    recordPanel.style.display = 'none';
    effectsPanel.style.display = 'none';

    if (!currentFile) return;

    const COMPUTE_TABS = {
        'stems': { label: 'Compute Stems (Demucs)', desc: 'Deep learning source separation — may take 30+ seconds', fn: loadStems },
        'hpss': { label: 'Compute HPSS', desc: 'Harmonic-percussive separation', fn: loadHPSS },
        'lab': { label: 'Compute Lab Features', desc: 'Spectral flatness, chromagram, contrast, ZCR', fn: loadLab },
        'lab-repet': { label: 'Compute REPET', desc: 'Repeating pattern extraction', fn: loadLabRepet },
        'lab-nmf': { label: 'Compute NMF', desc: 'Non-negative matrix factorization separation', fn: loadLabNMF },
    };

    if (COMPUTE_TABS[currentTab]) {
        // Demucs requires too much RAM for the public server
        if (currentTab === 'stems' && isPublicMode) {
            showDemucsUnavailable();
            return;
        }
        const info = COMPUTE_TABS[currentTab];
        showComputePrompt(info.label, info.desc, info.fn);
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
        const result = await cachedFetchPNG(url);
        if (!result) { showOverlay('Render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
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
        const stemUrl = '/api/stems/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(stemUrl);
        if (!result) { showOverlay('Stems render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
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
        const hpssUrl = '/api/hpss/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(hpssUrl);
        if (!result) { showOverlay('HPSS render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['harmonic', 'percussive'],
                       '/audio/separated/hpss/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadLab() {
    if (!currentFile) return;
    showOverlay('Computing lab features...');
    document.getElementById('controlsHint').innerHTML =
        '<kbd>Space</kbd> play/pause &nbsp; Click to seek &nbsp; Spectral Flatness &middot; Chromagram &middot; Spectral Contrast &middot; ZCR';
    document.getElementById('stemStatus').style.display = 'none';

    try {
        const labUrl = '/api/lab/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(labUrl);
        if (!result) { showOverlay('Lab render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadLabNMF() {
    if (!currentFile) return;
    showOverlay('Running NMF decomposition...');
    document.getElementById('stemStatus').style.display = 'none';

    try {
        const nmfUrl = '/api/lab-nmf/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(nmfUrl);
        if (!result) { showOverlay('NMF render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['drums', 'bass', 'vocals', 'other'],
                       '/audio/separated/nmf/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadLabRepet() {
    if (!currentFile) return;
    showOverlay('Computing REPET separation...');

    try {
        const repetUrl = '/api/lab-repet/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(repetUrl);
        if (!result) { showOverlay('REPET render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['repeating', 'non-repeating'],
                       '/audio/separated/repet/' + stemName + '/');
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

function showComputePrompt(label, desc, fn) {
    let overlay = viewer.querySelector('.overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'overlay';
        viewer.appendChild(overlay);
    }
    overlay.innerHTML = `
        <div class="compute-prompt">
            <button class="compute-btn" id="computeBtn">${label}</button>
            <p class="compute-desc">${desc}</p>
        </div>`;
    overlay.style.display = 'flex';
    document.getElementById('computeBtn').onclick = () => {
        overlay.innerHTML = '<span class="overlay-text"></span>';
        fn();
    };
}

function showDemucsUnavailable() {
    let overlay = viewer.querySelector('.overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'overlay';
        viewer.appendChild(overlay);
    }
    overlay.innerHTML = `
        <div class="compute-prompt">
            <p style="font-size:1.1em;margin-bottom:12px;">Demucs source separation requires ~4GB RAM — more than the demo server provides.</p>
            <p class="compute-desc">Run locally with Docker for full Demucs support:</p>
            <pre style="background:#1a1a2e;padding:12px;border-radius:6px;margin:12px 0;font-size:0.85em;text-align:left;overflow-x:auto;">docker run -p 8080:8080 -v ~/Music:/app/audio-reactive/research/audio-segments ghcr.io/sethdrew/led-viewer</pre>
            <a href="https://github.com/SethDrew/led#run-locally-with-docker" target="_blank"
               style="color:#4fc3f7;text-decoration:underline;">Setup instructions on GitHub</a>
        </div>`;
    overlay.style.display = 'flex';
}

// ── Cursor sync ──────────────────────────────────────────────────

let lastStemSync = 0;

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

        // Continuous stem sync — check every ~300ms during playback
        if (hasStemAudio() && !audio.paused) {
            const now = performance.now();
            if (now - lastStemSync > 300) {
                lastStemSync = now;
                syncStemAudios();
            }
        }
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

document.getElementById('replayBtn').addEventListener('click', () => {
    audio.currentTime = 0;
    if (hasStemAudio()) stemSeek();
    audio.play();
    if (hasStemAudio()) stemPlay();
    playBtn.innerHTML = '&#9646;&#9646;';
});

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

// ── URL hash state ───────────────────────────────────────────────

function saveHashState() {
    const params = new URLSearchParams();
    if (currentFile) params.set('file', currentFile);
    if (currentTab) params.set('tab', currentTab);
    history.replaceState(null, '', '#' + params.toString());
}

function readHashState() {
    const params = new URLSearchParams(location.hash.slice(1));
    return {
        file: params.get('file'),
        tab: params.get('tab'),
    };
}

// ── Effects ──────────────────────────────────────────────────────

let effectsList = [];
let effectsRunning = null;  // name of running effect or null
let effectsPollTimer = null;

async function loadEffects() {
    const panel = document.getElementById('effectsPanel');
    panel.innerHTML = '<span style="color:#888;">Loading effects...</span>';

    try {
        const resp = await fetch('/api/effects');
        if (!resp.ok) { panel.innerHTML = '<span style="color:#e94560;">Failed to load effects</span>'; return; }
        const data = await resp.json();
        effectsList = data.effects || [];
        effectsRunning = data.running;
        renderEffectsCards();
        startEffectsPoll();
    } catch (e) {
        panel.innerHTML = '<span style="color:#e94560;">Error: ' + e.message + '</span>';
    }
}

function renderEffectsCards() {
    const panel = document.getElementById('effectsPanel');
    if (effectsList.length === 0) {
        panel.innerHTML = '<span style="color:#888;">No effects found. Check runner.py --list</span>';
        return;
    }
    panel.innerHTML = '';
    effectsList.forEach((eff, idx) => {
        const card = document.createElement('div');
        card.className = 'effect-card' + (eff.name === effectsRunning ? ' running' : '');
        card.draggable = true;
        card.dataset.name = eff.name;
        card.dataset.idx = idx;

        // Drag handle
        const drag = document.createElement('span');
        drag.className = 'effect-drag';
        drag.textContent = '\u2261';
        card.appendChild(drag);

        // Name
        const nameEl = document.createElement('span');
        nameEl.className = 'effect-name';
        nameEl.textContent = eff.name;
        if (eff.name === effectsRunning) {
            const dot = document.createElement('span');
            dot.className = 'running-dot';
            nameEl.appendChild(dot);
        }
        card.appendChild(nameEl);

        // Stars
        const stars = document.createElement('span');
        stars.className = 'effect-stars';
        for (let s = 1; s <= 5; s++) {
            const star = document.createElement('button');
            star.className = 'effect-star' + (s <= (eff.rating || 0) ? ' filled' : '');
            star.title = s + ' star' + (s > 1 ? 's' : '');
            star.addEventListener('click', (e) => { e.stopPropagation(); rateEffect(eff.name, s); });
            stars.appendChild(star);
        }
        card.appendChild(stars);

        // Start/Stop button
        const btn = document.createElement('button');
        btn.className = 'effect-toggle' + (eff.name === effectsRunning ? ' stop' : '');
        btn.textContent = eff.name === effectsRunning ? 'Stop' : 'Start';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (eff.name === effectsRunning) stopEffect();
            else startEffect(eff.name);
        });
        card.appendChild(btn);

        // Drag events
        card.addEventListener('dragstart', (e) => {
            card.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', idx.toString());
        });
        card.addEventListener('dragend', () => card.classList.remove('dragging'));
        card.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            card.classList.add('drag-over');
        });
        card.addEventListener('dragleave', () => card.classList.remove('drag-over'));
        card.addEventListener('drop', (e) => {
            e.preventDefault();
            card.classList.remove('drag-over');
            const fromIdx = parseInt(e.dataTransfer.getData('text/plain'));
            const toIdx = idx;
            if (fromIdx !== toIdx) {
                const moved = effectsList.splice(fromIdx, 1)[0];
                effectsList.splice(toIdx, 0, moved);
                renderEffectsCards();
                reorderEffects(effectsList.map(ef => ef.name));
            }
        });

        panel.appendChild(card);
    });
}

async function startEffect(name) {
    try {
        await fetch('/api/effects/start/' + encodeURIComponent(name), { method: 'POST' });
        effectsRunning = name;
        renderEffectsCards();
    } catch (e) { console.error('startEffect:', e); }
}

async function stopEffect() {
    try {
        await fetch('/api/effects/stop', { method: 'POST' });
        effectsRunning = null;
        renderEffectsCards();
    } catch (e) { console.error('stopEffect:', e); }
}

async function rateEffect(name, rating) {
    try {
        await fetch('/api/effects/rate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, rating })
        });
        const eff = effectsList.find(e => e.name === name);
        if (eff) eff.rating = rating;
        renderEffectsCards();
    } catch (e) { console.error('rateEffect:', e); }
}

async function reorderEffects(order) {
    try {
        await fetch('/api/effects/reorder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ order })
        });
    } catch (e) { console.error('reorderEffects:', e); }
}

function startEffectsPoll() {
    stopEffectsPoll();
    effectsPollTimer = setInterval(async () => {
        if (currentTab !== 'effects') { stopEffectsPoll(); return; }
        try {
            const resp = await fetch('/api/effects');
            if (!resp.ok) return;
            const data = await resp.json();
            const newRunning = data.running;
            if (newRunning !== effectsRunning) {
                effectsRunning = newRunning;
                renderEffectsCards();
            }
        } catch (e) {}
    }, 2000);
}

function stopEffectsPoll() {
    if (effectsPollTimer) { clearInterval(effectsPollTimer); effectsPollTimer = null; }
}

// ── File upload ──────────────────────────────────────────────────

async function uploadWavBlob(blob, filename) {
    // Upload to server first to get canonical path
    const formData = new FormData();
    formData.append('file', blob, filename);
    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await resp.json();

    // Cache in IndexedDB using the server's canonical path (handles sanitization + dedup)
    if (data.ok && data.path) {
        const buf = await blob.arrayBuffer();
        await cacheDB.put(data.path, { name: data.name, wav: buf, savedAt: Date.now() }, 'audioFiles');
    }
    return data;
}

async function ensureFileOnServer(path) {
    // Check if the server has this file by trying to fetch its audio
    try {
        const resp = await fetch('/audio/' + encodeURIComponent(path), { method: 'HEAD' });
        if (resp.ok) return true;
    } catch {}

    // File missing on server — re-upload from IndexedDB
    const cached = await cacheDB.get(path, 'audioFiles');
    if (!cached) return false;

    const blob = new Blob([cached.wav], { type: 'audio/wav' });
    const progress = document.getElementById('uploadProgress');
    progress.textContent = 'Re-uploading ' + cached.name + '...';
    progress.style.display = 'block';

    try {
        const data = await uploadWavBlob(blob, cached.name);
        if (data.ok) {
            // Restore annotations from IndexedDB if any
            const annKey = 'ann:' + path;
            const annotations = await cacheDB.get(annKey, 'audioFiles');
            if (annotations && Object.keys(annotations).length > 0) {
                for (const [layer, taps] of Object.entries(annotations)) {
                    await fetch('/api/annotations/' + encodeURIComponent(data.path), {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ layer, taps })
                    });
                }
            }
            progress.textContent = 'Ready';
            setTimeout(() => progress.style.display = 'none', 1000);
            return true;
        }
    } catch {}
    progress.style.display = 'none';
    return false;
}

async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.name.toLowerCase().endsWith('.wav')) {
        alert('Only WAV files are supported');
        return;
    }

    const progress = document.getElementById('uploadProgress');
    progress.textContent = `Uploading ${file.name}...`;
    progress.style.display = 'block';

    try {
        const data = await uploadWavBlob(file, file.name);
        if (data.ok) {
            progress.textContent = `Uploaded ${data.name}`;
            setTimeout(() => progress.style.display = 'none', 2000);
            await loadFileList(data.path);
        } else {
            progress.textContent = `Error: ${data.error}`;
            setTimeout(() => progress.style.display = 'none', 3000);
        }
    } catch (e) {
        progress.textContent = `Upload failed: ${e.message}`;
        setTimeout(() => progress.style.display = 'none', 3000);
    }
    document.getElementById('uploadInput').value = '';
}

// ── File manager ─────────────────────────────────────────────────

async function renderFileManager() {
    // Render into whichever file manager container is visible
    const containers = [document.getElementById('fileManager'), document.getElementById('fileManagerLocal')].filter(Boolean);
    if (containers.length === 0) return;
    for (const container of containers) await _renderFileManagerInto(container);
}

async function _renderFileManagerInto(container) {

    // Get manageable files: uploads + recordings (user clips on local, uploads on public)
    const userFiles = files.filter(f =>
        f.path.startsWith('uploads/') || f.group === 'your files' ||
        (!isPublicMode && f.group === 'user clips')
    );

    if (userFiles.length === 0) {
        container.innerHTML = '<p style="color:#666; font-size:12px;">No uploaded files yet.</p>';
        return;
    }

    container.innerHTML = '';

    // Toolbar: select all + bulk delete
    const toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex; align-items:center; gap:8px; margin-bottom:8px;';
    toolbar.innerHTML = `
        <label style="color:#888; font-size:12px; display:flex; align-items:center; gap:4px; cursor:pointer;">
            <input type="checkbox" class="fm-select-all"> Select all
        </label>
        <button class="fm-bulk-delete" style="display:none; background:none; border:1px solid #e94560; color:#e94560; font-size:11px; padding:2px 10px; border-radius:3px; cursor:pointer; margin-left:auto;">Delete selected</button>`;
    container.appendChild(toolbar);

    const selectAllCb = toolbar.querySelector('.fm-select-all');
    const bulkDeleteBtn = toolbar.querySelector('.fm-bulk-delete');

    for (const f of userFiles) {
        const item = document.createElement('div');
        item.className = 'file-item';
        const dur = f.duration ? formatTime(f.duration) : '?';
        item.innerHTML = `
            <input type="checkbox" class="fm-cb" data-path="${f.path}" data-name="${f.name}">
            <span class="file-name" title="Click to select" data-path="${f.path}">${f.name}</span>
            <span class="file-dur">${dur}</span>
            <button class="ren" data-path="${f.path}" data-name="${f.name}">rename</button>`;
        container.appendChild(item);
    }

    function updateBulkUI() {
        const checked = container.querySelectorAll('.fm-cb:checked');
        bulkDeleteBtn.style.display = checked.length > 0 ? 'inline-block' : 'none';
        bulkDeleteBtn.textContent = 'Delete selected (' + checked.length + ')';
        const allCbs = container.querySelectorAll('.fm-cb');
        selectAllCb.checked = allCbs.length > 0 && checked.length === allCbs.length;
        selectAllCb.indeterminate = checked.length > 0 && checked.length < allCbs.length;
    }

    selectAllCb.onchange = () => {
        container.querySelectorAll('.fm-cb').forEach(cb => cb.checked = selectAllCb.checked);
        updateBulkUI();
    };

    container.querySelectorAll('.fm-cb').forEach(cb => cb.onchange = updateBulkUI);

    bulkDeleteBtn.onclick = async () => {
        const checked = container.querySelectorAll('.fm-cb:checked');
        const paths = Array.from(checked).map(cb => ({ path: cb.dataset.path, name: cb.dataset.name }));
        if (paths.length === 0) return;
        if (!confirm('Delete ' + paths.length + ' file(s)? This removes them from server and browser cache.')) return;
        for (const { path } of paths) {
            await _deleteFile(path);
        }
        await loadFileList();
        renderFileManager();
    };

    // Click handlers
    container.querySelectorAll('.file-name').forEach(el => {
        el.onclick = () => { selectFile(el.dataset.path); currentTab = 'analysis'; updateTabUI(); loadPanel(); };
    });
    container.querySelectorAll('.ren').forEach(el => {
        el.onclick = () => renameUserFile(el.dataset.path, el.dataset.name);
    });
}

async function _deleteFile(path) {
    // Delete from server
    try {
        await fetch('/api/files/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
    } catch {}
    // Delete from IndexedDB
    await cacheDB.delete(path, 'audioFiles');
    // Clear cached analysis panels
    const panelKeys = await cacheDB.getAll('panels');
    for (const { key } of panelKeys) {
        if (typeof key === 'string' && key.includes(encodeURIComponent(path))) {
            await cacheDB.delete(key, 'panels');
        }
    }
    if (currentFile === path) currentFile = null;
}

async function deleteUserFile(path, name) {
    if (!confirm('Delete "' + name + '"?')) return;
    await _deleteFile(path);
    await loadFileList();
    renderFileManager();
}

async function renameUserFile(path, oldName) {
    const newName = prompt('New name for "' + oldName + '":', oldName.replace('.wav', ''));
    if (!newName || newName === oldName.replace('.wav', '')) return;

    const finalName = newName.endsWith('.wav') ? newName : newName + '.wav';

    try {
        const resp = await fetch('/api/files/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, newName: finalName })
        });
        const data = await resp.json();

        if (data.ok) {
            // Update IndexedDB: copy to new key, delete old
            const cached = await cacheDB.get(path, 'audioFiles');
            if (cached) {
                await cacheDB.put(data.path, { ...cached, name: data.name }, 'audioFiles');
                await cacheDB.delete(path, 'audioFiles');
            }

            const wasSelected = currentFile === path;
            await loadFileList(wasSelected ? data.path : undefined);
            renderFileManager();
        } else {
            alert('Rename failed: ' + (data.error || 'unknown error'));
        }
    } catch (e) {
        alert('Rename failed: ' + e.message);
    }
}

// Drag and drop
let dragCounter = 0;
const dropOverlay = document.getElementById('dropOverlay');

document.addEventListener('dragenter', e => {
    e.preventDefault();
    dragCounter++;
    if (e.dataTransfer.types.includes('Files')) {
        dropOverlay.classList.add('active');
    }
});

document.addEventListener('dragleave', e => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
        dragCounter = 0;
        dropOverlay.classList.remove('active');
    }
});

document.addEventListener('dragover', e => e.preventDefault());

document.addEventListener('drop', e => {
    e.preventDefault();
    dragCounter = 0;
    dropOverlay.classList.remove('active');
    handleFileUpload(e.dataTransfer.files);
});

// ── Welcome ──────────────────────────────────────────────────────

function copyContact() {
    // Obfuscated email — assembled at runtime so scrapers can't find it in source
    const u = 'seth'; const d = 'sethdrew'; const t = 'com';
    const addr = u + '@' + d + '.' + t;
    navigator.clipboard.writeText(addr).then(() => {
        const el = document.getElementById('contactCopied');
        if (el) { el.style.display = 'inline'; setTimeout(() => el.style.display = 'none', 2000); }
    });
}

// ── Init ─────────────────────────────────────────────────────────

checkAuth();
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

        if path == '/api/auth/login':
            body = self._read_json_body()
            if body is None:
                return
            if body.get('passcode') == _PASSCODE:
                token = _make_auth_token()
                payload = json.dumps({'ok': True}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Set-Cookie',
                    f'{_AUTH_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000')
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._json_response({'ok': False, 'error': 'Invalid passcode'}, status=403)
            return
        elif path == '/api/auth/logout':
            payload = json.dumps({'ok': True}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Set-Cookie',
                f'{_AUTH_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0')
            self.end_headers()
            self.wfile.write(payload)
            return

        # Block record/effects in public mode
        if PUBLIC_MODE and (path.startswith('/api/record') or path.startswith('/api/effects')):
            self._json_response({'error': 'Not available'}, status=403)
            return

        if path == '/api/upload':
            self._handle_upload()
            return
        elif path == '/api/files/delete':
            self._handle_file_delete()
            return
        elif path == '/api/files/rename':
            self._handle_file_rename()
            return

        if path == '/api/record/start':
            self._handle_record_start()
        elif path == '/api/record/stop':
            self._handle_record_stop()
        elif path.startswith('/api/effects/start/'):
            name = unquote(path[len('/api/effects/start/'):])
            self._handle_effect_start(name)
        elif path == '/api/effects/stop':
            self._handle_effect_stop()
        elif path == '/api/effects/rate':
            self._handle_effect_rate()
        elif path == '/api/effects/reorder':
            self._handle_effect_reorder()
        elif path.startswith('/api/annotations/'):
            rel_path = path[len('/api/annotations/'):]
            self._save_annotation(rel_path)
        else:
            self.send_error(404)

    def _handle_effect_start(self, name):
        _start_effect(name)
        self._json_response({'ok': True})

    def _handle_effect_stop(self):
        _stop_effect()
        self._json_response({'ok': True})

    def _handle_effect_rate(self):
        body = self._read_json_body()
        if body is None:
            return
        name = body.get('name', '')
        rating = body.get('rating', 0)
        if not name or not isinstance(rating, int) or rating < 0 or rating > 5:
            self.send_error(400, 'Invalid name or rating')
            return
        prefs = _load_effect_prefs()
        if name not in prefs:
            prefs[name] = {}
        prefs[name]['rating'] = rating
        _save_effect_prefs(prefs)
        self._json_response({'ok': True})

    def _handle_effect_reorder(self):
        body = self._read_json_body()
        if body is None:
            return
        order = body.get('order', [])
        if not isinstance(order, list):
            self.send_error(400, 'Invalid order')
            return
        prefs = _load_effect_prefs()
        for i, name in enumerate(order):
            if name not in prefs:
                prefs[name] = {}
            prefs[name]['order'] = i
        _save_effect_prefs(prefs)
        self._json_response({'ok': True})

    def _read_json_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return None

    def _handle_upload(self):
        """Handle WAV file upload. Saves to uploads/ subdirectory."""
        content_length = int(self.headers.get('Content-Length', 0))
        max_size = 100 * 1024 * 1024  # 100MB
        if content_length > max_size:
            self._json_response({'ok': False, 'error': 'File too large (max 100MB)'}, status=413)
            return
        if content_length == 0:
            self._json_response({'ok': False, 'error': 'No file data'}, status=400)
            return

        # Parse multipart form data
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self._json_response({'ok': False, 'error': 'Expected multipart/form-data'}, status=400)
            return

        import cgi
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': content_type})

        file_item = form['file'] if 'file' in form else None
        if file_item is None or not file_item.filename:
            self._json_response({'ok': False, 'error': 'No file provided'}, status=400)
            return

        # Sanitize filename
        filename = os.path.basename(file_item.filename)
        filename = re.sub(r'[^\w\s\-.]', '_', filename)
        if not filename.lower().endswith('.wav'):
            self._json_response({'ok': False, 'error': 'Only WAV files are supported'}, status=400)
            return

        # Save to uploads directory
        uploads_dir = os.path.join(SEGMENTS_DIR, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        filepath = os.path.join(uploads_dir, filename)

        # Avoid overwriting — add suffix if exists
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(uploads_dir, f"{base}_{counter}{ext}")
            counter += 1

        with open(filepath, 'wb') as f:
            f.write(file_item.file.read())

        # Validate it's actually a WAV file
        dur = _get_wav_duration(filepath)
        if dur == 0.0:
            os.remove(filepath)
            self._json_response({'ok': False, 'error': 'Invalid WAV file'}, status=400)
            return

        # Clear file list cache so it gets re-scanned
        global _file_list_cache
        _file_list_cache = None

        saved_name = os.path.basename(filepath)
        print(f"[upload] Saved {saved_name} ({dur:.1f}s)")
        self._json_response({
            'ok': True,
            'name': saved_name,
            'path': f'uploads/{saved_name}',
            'duration': dur,
        })

    def _handle_file_delete(self):
        body = self._read_json_body()
        if body is None:
            return
        path = body.get('path', '')
        if '..' in path or '/' in path.replace('uploads/', '', 1).replace('harmonix/', '', 1):
            self._json_response({'ok': False, 'error': 'Invalid path'}, status=400)
            return
        # Public mode: only allow deleting uploads
        if PUBLIC_MODE and not path.startswith('uploads/'):
            self._json_response({'ok': False, 'error': 'Cannot delete built-in files'}, status=403)
            return
        filepath = os.path.join(SEGMENTS_DIR, path)
        if os.path.exists(filepath):
            os.remove(filepath)
            # Also remove annotations if they exist
            ann_path = filepath.replace('.wav', '.annotations.yaml')
            if os.path.exists(ann_path):
                os.remove(ann_path)
            print(f"[delete] Removed {path}")
        global _file_list_cache
        _file_list_cache = None
        self._json_response({'ok': True})

    def _handle_file_rename(self):
        body = self._read_json_body()
        if body is None:
            return
        path = body.get('path', '')
        new_name = body.get('newName', '')
        if '..' in path:
            self._json_response({'ok': False, 'error': 'Invalid path'}, status=400)
            return
        # Public mode: only allow renaming uploads
        if PUBLIC_MODE and not path.startswith('uploads/'):
            self._json_response({'ok': False, 'error': 'Cannot rename built-in files'}, status=403)
            return
        new_name = re.sub(r'[^\w\s\-.]', '_', os.path.basename(new_name))
        if not new_name.lower().endswith('.wav'):
            self._json_response({'ok': False, 'error': 'Name must end with .wav'}, status=400)
            return
        old_filepath = os.path.join(SEGMENTS_DIR, path)
        # Keep in same directory
        parent = os.path.dirname(old_filepath)
        new_filepath = os.path.join(parent, new_name)
        if not os.path.exists(old_filepath):
            self._json_response({'ok': False, 'error': 'File not found'}, status=404)
            return
        if os.path.exists(new_filepath) and old_filepath != new_filepath:
            self._json_response({'ok': False, 'error': 'A file with that name already exists'}, status=409)
            return
        os.rename(old_filepath, new_filepath)
        # Also rename annotations if they exist
        old_ann = old_filepath.replace('.wav', '.annotations.yaml')
        new_ann = new_filepath.replace('.wav', '.annotations.yaml')
        if os.path.exists(old_ann):
            os.rename(old_ann, new_ann)
        global _file_list_cache
        _file_list_cache = None
        new_path = path.rsplit('/', 1)
        new_path = (new_path[0] + '/' + new_name) if len(new_path) > 1 else new_name
        print(f"[rename] {path} -> {new_path}")
        self._json_response({'ok': True, 'name': new_name, 'path': new_path})

    def _json_response(self, data, status=200):
        payload = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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

    def _serve_annotations(self, rel_path):
        """Serve annotation YAML as JSON for a given audio file."""
        import yaml
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self._json_response({})
            return
        ann_path = Path(filepath).with_suffix('.annotations.yaml')
        if ann_path.exists():
            with open(ann_path) as f:
                annotations = yaml.safe_load(f) or {}
            self._json_response(annotations)
        else:
            self._json_response({})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        # Auth gate for protected endpoints
        if _is_protected(path) and not _is_authenticated(self):
            self._json_response({'error': 'Sign in to access this feature'}, status=401)
            return

        if path == '/':
            self._serve_html()
        elif path == '/favicon.ico':
            self._serve_favicon()
        elif path == '/api/auth/status':
            self._json_response({'authenticated': _is_authenticated(self), 'public': PUBLIC_MODE})
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
        elif path.startswith('/api/lab-nmf/'):
            rel_path = path[len('/api/lab-nmf/'):]
            self._serve_lab_nmf(rel_path)
        elif path.startswith('/api/lab-repet/'):
            rel_path = path[len('/api/lab-repet/'):]
            self._serve_lab_repet(rel_path)
        elif path.startswith('/api/lab/'):
            rel_path = path[len('/api/lab/'):]
            self._serve_lab(rel_path)
        elif path.startswith('/api/annotations/'):
            rel_path = path[len('/api/annotations/'):]
            self._serve_annotations(rel_path)
        elif path == '/api/effects':
            self._json_response({
                'effects': _get_effects_list(),
                'running': _get_running_effect_name(),
            })
        elif path == '/api/record/level':
            data = json.dumps(get_recording_level()).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
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
        self.send_header('Cache-Control', 'max-age=3600')
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
        self.send_header('Cache-Control', 'max-age=3600')
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
        self.send_header('Cache-Control', 'max-age=3600')
        exposed = ', '.join(headers.keys())
        self.send_header('Access-Control-Expose-Headers', exposed)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(png_bytes)

    def _serve_lab_nmf(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        try:
            png_bytes, headers = render_lab_nmf(filepath)
        except Exception as e:
            print(f"[lab-nmf] Error: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(png_bytes)))
        self.send_header('Cache-Control', 'max-age=3600')
        exposed = ', '.join(headers.keys())
        self.send_header('Access-Control-Expose-Headers', exposed)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(png_bytes)

    def _serve_lab_repet(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        try:
            png_bytes, headers = render_lab_repet(filepath)
        except Exception as e:
            print(f"[lab-repet] Error: {e}")
            self.send_error(500, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(png_bytes)))
        self.send_header('Cache-Control', 'max-age=3600')
        exposed = ', '.join(headers.keys())
        self.send_header('Access-Control-Expose-Headers', exposed)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(png_bytes)

    def _serve_lab(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        try:
            png_bytes, headers = render_lab(filepath)
        except Exception as e:
            print(f"[lab] Error: {e}")
            self.send_error(500, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(png_bytes)))
        self.send_header('Cache-Control', 'max-age=3600')
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

def run_server(port=0, host='127.0.0.1', no_browser=False):
    """Start the web viewer server and open the browser."""
    server = HTTPServer((host, port), ViewerHandler)
    actual_port = server.server_address[1]

    url = f'http://{host}:{actual_port}'
    print(f"Audio Explorer running at {url}")
    print("Press Ctrl+C to stop\n")

    if not no_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{actual_port}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _stop_effect()
        print("\nServer stopped.")
        server.server_close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Web-based audio analysis viewer')
    parser.add_argument('--port', type=int, default=0, help='Server port (0=auto)')
    args = parser.parse_args()
    run_server(args.port)
