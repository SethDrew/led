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
_active_controller = None  # id of the controller the running effect is using, or None

EFFECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'effects')

# ── Controllers ──────────────────────────────────────────────────────

def _load_controllers():
    """Load LED controller definitions from controllers.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'controllers.json')
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[controllers] Could not load controllers.json: {e}")
        return []

def _resolve_controller_ports(controllers):
    """Match controllers to connected serial ports by USB VID/PID.

    Each controller in controllers.json specifies vid/pid (hex strings).
    We scan connected ports and assign the 'port' field at runtime.
    If multiple devices share the same VID/PID, assignment is arbitrary.
    """
    try:
        from serial.tools import list_ports
        connected = list(list_ports.comports())
    except ImportError:
        print("[controllers] pyserial not installed — cannot auto-detect ports")
        return

    # Build a lookup: (vid, pid) -> list of device paths
    by_vidpid = {}
    for p in connected:
        if p.vid is not None and p.pid is not None:
            key = (f"{p.vid:04x}", f"{p.pid:04x}")
            by_vidpid.setdefault(key, []).append(p.device)

    for c in controllers:
        vid = c.get('vid')
        pid = c.get('pid')
        if not vid or not pid:
            continue
        key = (vid.lower(), pid.lower())
        ports = by_vidpid.get(key, [])
        if ports:
            c['port'] = ports.pop(0)  # consume one; arbitrary if >1
            print(f"[controllers] {c['name']}: {c['port']} (vid={vid} pid={pid})")
        else:
            c['port'] = None
            print(f"[controllers] {c['name']}: not connected (vid={vid} pid={pid})")

_controllers = _load_controllers()
_resolve_controller_ports(_controllers)

# ── Sculptures ──────────────────────────────────────────────────

def _load_sculptures():
    """Load sculpture definitions from hardware/sculptures.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hardware', 'sculptures.json')
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[sculptures] Could not load sculptures.json: {e}")
        return []


def _build_output_targets(sculptures, controllers):
    """Build merged output list: sculptures first, then controllers without any sculpture.

    Each entry: {id, name, type, controller_id, controller_name, leds, port}
    """
    targets = []
    controllers_with_sculpture = set()

    for s in sculptures:
        ctrl = next((c for c in controllers if c['id'] == s['controller']), None)
        logical_leds = sum(b['count'] for b in s['branches'])
        targets.append({
            'id': s['id'],
            'name': s['name'],
            'type': 'sculpture',
            'controller_id': s['controller'],
            'controller_name': ctrl['name'] if ctrl else s['controller'],
            'leds': logical_leds,
            'port': ctrl.get('port') if ctrl else None,
        })
        controllers_with_sculpture.add(s['controller'])

    for c in controllers:
        if c['id'] not in controllers_with_sculpture:
            targets.append({
                'id': c['id'],
                'name': c['name'],
                'type': 'controller',
                'controller_id': c['id'],
                'controller_name': c['name'],
                'leds': c['leds'],
                'port': c.get('port'),
            })

    return targets


_sculptures = _load_sculptures()

# ── Live reload (local dev only) ──────────────────────────────────────

def _source_hash():
    """Hash web_viewer.py + static files + all effect files to detect changes."""
    h = hashlib.md5()
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    static_files = sorted(
        str(p) for p in Path(static_dir).glob('*') if p.is_file()
    ) if os.path.isdir(static_dir) else []
    controllers_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'controllers.json')
    sculptures_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hardware', 'sculptures.json')
    for path in [__file__, controllers_file, sculptures_file] + static_files + sorted(
        str(p) for p in Path(EFFECTS_DIR).glob('*.py') if p.is_file()
    ):
        try:
            h.update(Path(path).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]

_startup_hash = _source_hash()

# ── Auth ─────────────────────────────────────────────────────────────

PUBLIC_MODE = os.environ.get('LED_VIEWER_PUBLIC', '') == '1'
_PASSCODE = os.environ.get('LED_VIEWER_PASSCODE', '')  # Set via env var on server
_AUTH_SECRET = os.environ.get('LED_VIEWER_SECRET', secrets.token_hex(32))
_AUTH_COOKIE = 'led_session'
_PROTECTED_PREFIXES = ('/api/stems',)  # Only Demucs is gated (can't run on public server)

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
    """Run runner.py --list-json and parse structured effect data."""
    try:
        result = subprocess.run(
            [sys.executable, 'runner.py', '--list-json'],
            cwd=EFFECTS_DIR, capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        entries = []
        for sig in data.get('signals', []):
            entries.append({
                'name': sig['name'],
                'description': sig.get('description', ''),
                'is_signal': True,
                'default_palette': sig.get('default_palette', 'amber'),
            })
        for eff in data.get('effects', []):
            entries.append({
                'name': eff['name'],
                'description': eff.get('description', ''),
                'is_signal': False,
            })
        return entries, data.get('palettes', [])
    except Exception as e:
        print(f"[effects] Discovery failed: {e}")
        return [], []


_EFFECT_RENAME_MAP = {
    # Legacy short names
    'absint_pred': 'impulse_predict',
    'absint_down': 'impulse_downbeat',
    'absint_reds': 'impulse_glow',
    'absint_red_palette': 'impulse_glow',
    'absint_sec': 'impulse_sections',
    'absint_band': 'impulse_bands',
    'longint_sec': 'longint_sections',
    'three_voices': 'hpss_voices',
    # absint → impulse rename
    'absint_pulse': 'impulse',
    'absint_prop': 'impulse_glow',
    'absint_predict': 'impulse_predict',
    'absint_downbeat': 'impulse_downbeat',
    'absint_breathe': 'impulse_breathe',
    'absint_sections': 'impulse_sections',
    'absint_meter': 'impulse_meter',
    'absint_snake': 'impulse_snake',
    'absint_bands': 'impulse_bands',
}

def _load_effect_prefs():
    """Load effect preferences (ratings, order) from JSON file."""
    if os.path.exists(EFFECT_PREFS_PATH):
        try:
            with open(EFFECT_PREFS_PATH) as f:
                prefs = json.load(f)
            # One-time migration for renamed effects
            migrated = False
            for old_name, new_name in _EFFECT_RENAME_MAP.items():
                if old_name in prefs and new_name not in prefs:
                    prefs[new_name] = prefs.pop(old_name)
                    migrated = True
            if migrated:
                _save_effect_prefs(prefs)
            return prefs
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
    """Get effects list merged with preferences (ratings, order).

    Returns (active_effects, deprecated_effects, palettes).
    Deprecated effects are separated from active ones.
    """
    entries, palettes = _discover_effects()
    prefs = _load_effect_prefs()

    effects = []
    deprecated = []
    for entry in entries:
        name = entry['name'] if isinstance(entry, dict) else entry
        desc = entry.get('description', '') if isinstance(entry, dict) else ''
        p = prefs.get(name, {})
        e = {
            'name': name,
            'description': desc,
            'order': p.get('order', 999),
            'rating': p.get('rating', 0),
        }
        if p.get('display_name'):
            e['display_name'] = p['display_name']
        if p.get('notes'):
            e['notes'] = p['notes']
        if isinstance(entry, dict):
            e['is_signal'] = entry.get('is_signal', False)
            if entry.get('default_palette'):
                e['default_palette'] = entry['default_palette']
        if p.get('deprecated'):
            e['deprecated'] = True
            e['deprecated_reason'] = p.get('deprecated_reason', '')
            deprecated.append(e)
        else:
            effects.append(e)

    effects.sort(key=lambda e: e['order'])
    deprecated.sort(key=lambda e: e['name'])
    return effects, deprecated, palettes


def _start_effect(name, controller_id=None, sculpture_id=None, palette_name=None,
                  brightness=None):
    """Start an effect subprocess. Kills any running effect first.

    Args:
        name: Effect registry name
        controller_id: Controller ID (direct controller, no topology)
        sculpture_id: Sculpture ID (topology mapping, overrides controller_id)
        palette_name: Palette override for signal effects
        brightness: Brightness cap (0-1)
    """
    global _effect_process, _active_controller
    _stop_effect()

    # Re-resolve ports in case devices were unplugged/replugged
    _resolve_controller_ports(_controllers)

    cmd = [sys.executable, 'runner.py', name]

    if palette_name:
        cmd += ['--chroma', palette_name]

    if brightness is not None:
        cmd += ['--brightness', str(brightness)]

    # Sculpture mode: --sculpture passes topology to runner.py which resolves
    # controller port internally. Controller mode: --port/--leds as before.
    target_id = None
    if sculpture_id:
        sculpture = next((s for s in _sculptures if s['id'] == sculpture_id), None)
        if sculpture:
            controller = next((c for c in _controllers if c['id'] == sculpture['controller']), None)
            if controller and controller.get('port'):
                cmd += ['--sculpture', sculpture_id, '--port', controller['port']]
                target_id = sculpture_id
            else:
                cmd.append('--no-leds')
        else:
            cmd.append('--no-leds')
    elif controller_id:
        controller = next((c for c in _controllers if c['id'] == controller_id), None)
        if controller and controller.get('port'):
            cmd += ['--port', controller['port'], '--leds', str(controller['leds'])]
            target_id = controller_id
        else:
            cmd.append('--no-leds')
    else:
        cmd.append('--no-leds')

    try:
        _effect_process = subprocess.Popen(
            cmd,
            cwd=EFFECTS_DIR
        )
        _active_controller = target_id
        print(f"[effects] Started: {name} (pid {_effect_process.pid}) cmd: {' '.join(cmd)}")
    except Exception as e:
        print(f"[effects] Start failed: {e}")
        _effect_process = None
        _active_controller = None


def _stop_effect():
    """Stop the running effect subprocess."""
    global _effect_process, _active_controller
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
        _active_controller = None


def _get_running_effect_name():
    """Get the name of the currently running effect, or None."""
    global _effect_process, _active_controller
    if _effect_process is None:
        return None
    if _effect_process.poll() is not None:
        _effect_process = None
        _active_controller = None
        return None
    # Extract from command line args
    args = _effect_process.args
    if len(args) >= 3:
        return args[2]
    return None


# ── User Palettes ────────────────────────────────────────────────────

def _palette_module():
    """Import palette module from effects directory."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('palette', os.path.join(EFFECTS_DIR, 'palette.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_all_palettes_list():
    """Return list of all palettes (builtin + user) with metadata."""
    mod = _palette_module()
    result = []
    for name, palette in mod.PALETTE_PRESETS.items():
        d = mod.palette_to_dict(palette)
        d['name'] = name
        d['is_builtin'] = True
        result.append(d)
    for name, palette in mod.load_user_palettes().items():
        if name not in mod.PALETTE_PRESETS:
            d = mod.palette_to_dict(palette)
            d['name'] = name
            d['is_builtin'] = False
            result.append(d)
    return result


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
            ann_path = wav.with_suffix('.annotations.yaml')
            files.append({
                'name': wav.name,
                'path': f'uploads/{wav.name}',
                'duration': dur,
                'has_annotations': ann_path.exists(),
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
        viz = SyncedVisualizer(filepath)
    else:
        viz = SyncedVisualizer(filepath, annotations_path='/dev/null/none.yaml')

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

    # Replace title and tighten top margin (default ~0.88 wastes 12% as blank space)
    filename = Path(filepath).name
    viz.fig.suptitle(filename, fontsize=14, fontweight='bold', y=0.995)
    viz.fig.subplots_adjust(top=0.975, bottom=0.02)

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


def render_annotate(filepath):
    """Render annotation view: waveform + spectrogram + annotation lanes only."""
    cache_key = (filepath, 'annotate')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    from viewer import SyncedVisualizer

    viz = SyncedVisualizer(filepath, panels=['waveform', 'spectrogram'])

    for line in viz.cursor_lines:
        line.remove()

    filename = Path(filepath).name
    viz.fig.suptitle(filename, fontsize=14, fontweight='bold', y=0.99)
    viz.fig.subplots_adjust(top=0.97, bottom=0.04)

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


ANALYSIS_PANELS = ('waveform', 'spectrogram', 'bands', 'rms-derivative',
                    'centroid', 'centroid-derivative', 'band-derivative',
                    'mfcc', 'novelty', 'band-deviation', 'annotations',
                    'band-rt', 'band-share', 'band-context-deviation',
                    'band-rt-derivative', 'band-integral')

BAND_ANALYSIS_PANELS = ('band-rt', 'band-share', 'band-context-deviation',
                        'band-rt-derivative', 'band-integral')


def render_band_analysis(filepath):
    """Render band analysis view: 5 panels showing band energy through multiple lenses."""
    cache_key = (filepath, 'band-analysis')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    from viewer import SyncedVisualizer

    viz = SyncedVisualizer(filepath,
                           panels=list(BAND_ANALYSIS_PANELS),
                           annotations_path='/dev/null/none.yaml')

    for line in viz.cursor_lines:
        line.remove()

    filename = Path(filepath).name
    viz.fig.suptitle(filename, fontsize=14, fontweight='bold', y=0.995)
    viz.fig.subplots_adjust(top=0.97, bottom=0.04)

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


def render_single_panel(filepath, panel_name):
    """Render a single focused analysis panel to PNG bytes."""
    if panel_name not in ANALYSIS_PANELS:
        raise ValueError(f"Unknown panel: {panel_name}")

    cache_key = (filepath, 'panel', panel_name)
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    from viewer import SyncedVisualizer

    # Annotations panel needs annotations loaded; others suppress them
    ann_path = None if panel_name == 'annotations' else '/dev/null/none.yaml'
    viz = SyncedVisualizer(filepath, focus_panel=panel_name,
                           annotations_path=ann_path)

    if panel_name == 'annotations' and not viz.annotations:
        raise ValueError("No annotations found for this file")

    for line in viz.cursor_lines:
        line.remove()

    viz.fig.suptitle('', fontsize=1)
    viz.fig.set_figheight(4)
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
    viz.fig.subplots_adjust(top=0.975, bottom=0.02)
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
    viz.fig.subplots_adjust(top=0.975, bottom=0.02)
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
    fig.subplots_adjust(top=0.975, bottom=0.02)
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
    fig.subplots_adjust(top=0.975, bottom=0.02)
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


def render_lab(filepath, variant='timbral'):
    """Lab page dispatcher."""
    if variant == 'misc':
        return render_lab_misc(filepath)
    return render_lab_timbral(filepath)


def render_lab_misc(filepath):
    """Misc lab features (spectral flatness, chromagram, spectral contrast, ZCR)."""
    cache_key = (filepath, 'lab-misc')
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

    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 2, 2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, flatness, color='#80cbc4', linewidth=0.8)
    ax1.fill_between(times, flatness, alpha=0.3, color='#80cbc4')
    ax1.set_xlim([0, duration])
    ax1.set_ylim([0, max(0.01, np.max(flatness) * 1.1)])
    ax1.set_ylabel('Flatness')
    ax1.set_title('Spectral Flatness — 0 = pure tone, 1 = white noise', fontsize=11)
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1])
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time',
                             sr=sr, hop_length=hop_length, ax=ax2, cmap='magma')
    ax2.set_xlim([0, duration])
    ax2.set_title('Chromagram — pitch class energy over time', fontsize=11)

    ax3 = fig.add_subplot(gs[2])
    librosa.display.specshow(contrast, x_axis='time', sr=sr, hop_length=hop_length,
                             ax=ax3, cmap='inferno')
    ax3.set_xlim([0, duration])
    ax3.set_ylabel('Band')
    ax3.set_title('Spectral Contrast — peak-to-valley difference per frequency band', fontsize=11)

    zcr_times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(zcr_times, zcr, color='#ffab91', linewidth=0.8)
    ax4.fill_between(zcr_times, zcr, alpha=0.3, color='#ffab91')
    ax4.set_xlim([0, duration])
    ax4.set_ylabel('ZCR')
    ax4.set_title('Zero Crossing Rate — high = percussive/noisy, low = tonal', fontsize=11)
    ax4.grid(True, alpha=0.2)

    filename = Path(filepath).name
    fig.suptitle(f'{filename} — Misc Lab', fontsize=14, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.975, bottom=0.02)
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


def render_lab_timbral(filepath):
    """Timbral Shape Lab — MFCC coefficients broken out with explanations."""
    cache_key = (filepath, 'lab-timbral')
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    import numpy as np
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    plt.style.use('dark_background')

    y, sr = librosa.load(filepath, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    n_fft = 2048
    hop_length = 512

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr, hop_length=hop_length)

    # Foote's checkerboard novelty on MFCC similarity
    fps = sr / hop_length
    norms = np.linalg.norm(mfccs, axis=0, keepdims=True)
    norms[norms == 0] = 1
    mfcc_norm = mfccs / norms
    S = (mfcc_norm.T @ mfcc_norm).astype(np.float32)

    from viewer import _checkerboard_novelty
    kernel_seconds = 3
    kernel_size = max(16, int(kernel_seconds * fps))
    if kernel_size % 2 != 0:
        kernel_size += 1
    novelty = _checkerboard_novelty(S, kernel_size)
    del S

    # Layout: heatmap + 4 individual coefficients + fine texture heatmap + novelty
    # Total 7 panels
    fig = plt.figure(figsize=(18, 28))
    gs = gridspec.GridSpec(7, 1, height_ratios=[1.5, 1, 1, 1, 1, 1.5, 1.2], hspace=0.35)

    sigma = 5  # smoothing for individual coefficient plots

    # ── Panel 1: Full MFCC heatmap ──
    ax = fig.add_subplot(gs[0])
    ax.imshow(mfccs, aspect='auto', origin='lower', cmap='magma',
              extent=[0, duration, 0, 13])
    ax.set_xlim([0, duration])
    ax.set_ylabel('Coefficient')
    ax.set_yticks(np.arange(13) + 0.5)
    ax.set_yticklabels([str(i) for i in range(13)])
    ax.set_title('All 13 MFCC Coefficients — the complete timbral fingerprint', fontsize=12)

    # ── Panel 2: MFCC 0 — Overall Energy ──
    ax = fig.add_subplot(gs[1])
    raw = mfccs[0]
    smoothed = gaussian_filter1d(raw, sigma=sigma)
    ax.plot(times, raw, color='#FF8A65', linewidth=0.4, alpha=0.3)
    ax.plot(times, smoothed, color='#FF8A65', linewidth=2)
    ax.set_xlim([0, duration])
    ax.set_ylabel('MFCC 0')
    ax.set_title('MFCC 0: Overall Energy Level — correlates with loudness (like RMS)', fontsize=11)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: MFCC 1 — Spectral Tilt ──
    ax = fig.add_subplot(gs[2])
    raw = mfccs[1]
    smoothed = gaussian_filter1d(raw, sigma=sigma)
    ax.plot(times, raw, color='#4DD0E1', linewidth=0.4, alpha=0.3)
    ax.plot(times, smoothed, color='#4DD0E1', linewidth=2)
    ax.axhline(y=0, color='#666', linewidth=0.5)
    ax.set_xlim([0, duration])
    ax.set_ylabel('MFCC 1')
    ax.set_title('MFCC 1: Spectral Tilt — positive = dark/bassy, negative = bright/trebly',
                 fontsize=11)
    ax.grid(True, alpha=0.2)

    # ── Panel 4: MFCC 2 — Spectral Curvature ──
    ax = fig.add_subplot(gs[3])
    raw = mfccs[2]
    smoothed = gaussian_filter1d(raw, sigma=sigma)
    ax.plot(times, raw, color='#B388FF', linewidth=0.4, alpha=0.3)
    ax.plot(times, smoothed, color='#B388FF', linewidth=2)
    ax.axhline(y=0, color='#666', linewidth=0.5)
    ax.set_xlim([0, duration])
    ax.set_ylabel('MFCC 2')
    ax.set_title('MFCC 2: Spectral Curvature — shape of the energy distribution '
                 '(single peak vs spread out)', fontsize=11)
    ax.grid(True, alpha=0.2)

    # ── Panel 5: MFCC 3 — Spectral Detail ──
    ax = fig.add_subplot(gs[4])
    raw = mfccs[3]
    smoothed = gaussian_filter1d(raw, sigma=sigma)
    ax.plot(times, raw, color='#69F0AE', linewidth=0.4, alpha=0.3)
    ax.plot(times, smoothed, color='#69F0AE', linewidth=2)
    ax.axhline(y=0, color='#666', linewidth=0.5)
    ax.set_xlim([0, duration])
    ax.set_ylabel('MFCC 3')
    ax.set_title('MFCC 3: Spectral Asymmetry — where the energy humps and dips are',
                 fontsize=11)
    ax.grid(True, alpha=0.2)

    # ── Panel 6: MFCC 4-12 heatmap — Fine Timbral Texture ──
    ax = fig.add_subplot(gs[5])
    ax.imshow(mfccs[4:], aspect='auto', origin='lower', cmap='magma',
              extent=[0, duration, 4, 13])
    ax.set_xlim([0, duration])
    ax.set_ylabel('Coefficient')
    ax.set_yticks(np.arange(4, 13) + 0.5)
    ax.set_yticklabels([str(i) for i in range(4, 13)])
    ax.set_title('MFCC 4-12: Fine Timbral Texture — subtle details '
                 '(buzzy vs smooth, nasal vs hollow, etc.)', fontsize=11)

    # ── Panel 7: Timbral Shift (MFCC Novelty) ──
    ax = fig.add_subplot(gs[6])
    n = min(len(times), len(novelty))
    ax.plot(times[:n], novelty[:n], color='#FF8A65', linewidth=1.5)
    ax.fill_between(times[:n], novelty[:n], alpha=0.3, color='#FF8A65')
    peak_dist = max(1, int(fps * 1.5))
    peaks, _ = find_peaks(novelty[:n], prominence=0.15, distance=peak_dist)
    ax.scatter(times[peaks], novelty[peaks], color='#FF8A65', s=30, zorder=5, marker='v')
    ax.set_xlim([0, duration])
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Novelty')
    ax.set_xlabel('Time (s)')
    ax.set_title('Timbral Shift — peaks where the overall sound character changes '
                 '(Foote checkerboard on MFCC similarity)', fontsize=11)
    ax.grid(True, alpha=0.2)

    filename = Path(filepath).name
    fig.suptitle(f'{filename} — Timbral Shape (MFCC) Lab', fontsize=14,
                 fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.975, bottom=0.02)
    fig.canvas.draw()

    ax0 = fig.axes[0]
    x_left = ax0.transData.transform((0, 0))[0]
    x_right = ax0.transData.transform((duration, 0))[0]
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

_server_start_iso = None

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


def _read_static(filename):
    """Read a file from the static/ directory."""
    with open(os.path.join(_STATIC_DIR, filename), 'r') as f:
        return f.read()


def generate_html():
    """Generate the single-page app HTML by assembling static/ files.

    CSS, HTML body, and JS are in separate files for easier editing.
    They're inlined into a single response (no extra HTTP requests).
    """
    css = _read_static('style.css')
    body_html = _read_static('body.html')
    js = _read_static('app.js')
    iso = _server_start_iso or ''
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<link rel="icon" type="image/svg+xml" href="/favicon.ico">\n'
        '<title>Audio Explorer</title>\n'
        f'<style>\n{css}\n</style>\n'
        '</head>\n'
        f'{body_html}\n'
        f'<script>window.__SERVER_START_ISO = "{iso}";</script>\n'
        f'<script>\n{js}\n</script>\n'
        '</body>\n</html>\n'
    )


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

        # Block record/effects/annotation-save in public mode
        if PUBLIC_MODE and (path.startswith('/api/record') or path.startswith('/api/effects')
                           or path.startswith('/api/annotations/') or path.startswith('/api/palettes/')):
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
        elif path == '/api/effects/rename':
            self._handle_effect_rename()
        elif path == '/api/effects/notes':
            self._handle_effect_notes()
        elif path == '/api/effects/deprecate':
            self._handle_effect_deprecate()
        elif path == '/api/effects/analyze':
            self._handle_effect_analyze()
        elif path == '/api/effects/controller':
            body = self._read_json_body()
            prefs = _load_effect_prefs()
            # Support both sculpture and controller selection
            if body and 'sculpture' in body:
                prefs['__sculpture__'] = body['sculpture']
                prefs.pop('__controller__', None)
            elif body and 'controller' in body:
                prefs['__controller__'] = body['controller']
                prefs.pop('__sculpture__', None)
            else:
                prefs.pop('__controller__', None)
                prefs.pop('__sculpture__', None)
            _save_effect_prefs(prefs)
            self._json_response({'ok': True})
        elif path == '/api/palettes/save':
            self._handle_palette_save()
        elif path == '/api/palettes/delete':
            self._handle_palette_delete()
        elif path.startswith('/api/annotations/'):
            rel_path = path[len('/api/annotations/'):]
            self._save_annotation(rel_path)
        else:
            self.send_error(404)

    def _handle_effect_start(self, name):
        body = self._read_json_body()
        controller_id = body.get('controller') if body else None
        sculpture_id = body.get('sculpture') if body else None
        palette_name = body.get('palette') if body else None
        brightness = body.get('brightness') if body else None
        _start_effect(name, controller_id=controller_id, sculpture_id=sculpture_id,
                      palette_name=palette_name, brightness=brightness)
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

    def _handle_effect_rename(self):
        body = self._read_json_body()
        if body is None:
            return
        name = body.get('name', '')
        display_name = body.get('display_name', '').strip()
        if not name:
            self.send_error(400, 'Missing effect name')
            return
        prefs = _load_effect_prefs()
        if name not in prefs:
            prefs[name] = {}
        if display_name:
            prefs[name]['display_name'] = display_name
        else:
            prefs[name].pop('display_name', None)
        _save_effect_prefs(prefs)
        self._json_response({'ok': True})

    def _handle_effect_notes(self):
        body = self._read_json_body()
        if body is None:
            return
        name = body.get('name', '')
        notes = body.get('notes', '').strip()
        if not name:
            self.send_error(400, 'Missing effect name')
            return
        prefs = _load_effect_prefs()
        if name not in prefs:
            prefs[name] = {}
        if notes:
            prefs[name]['notes'] = notes
        else:
            prefs[name].pop('notes', None)
        _save_effect_prefs(prefs)
        self._json_response({'ok': True})

    def _handle_effect_deprecate(self):
        body = self._read_json_body()
        if body is None:
            return
        name = body.get('name', '')
        deprecated = body.get('deprecated', True)
        reason = body.get('reason', '').strip()
        if not name:
            self.send_error(400, 'Missing effect name')
            return
        prefs = _load_effect_prefs()
        if name not in prefs:
            prefs[name] = {}
        if deprecated:
            prefs[name]['deprecated'] = True
            if reason:
                prefs[name]['deprecated_reason'] = reason
        else:
            prefs[name].pop('deprecated', None)
            prefs[name].pop('deprecated_reason', None)
        _save_effect_prefs(prefs)
        self._json_response({'ok': True})

    def _handle_effect_analyze(self):
        body = self._read_json_body()
        if body is None:
            return
        effect_name = body.get('effect', '')
        file_path = body.get('file', '')
        palette_name = body.get('palette', '')
        if not effect_name or not file_path:
            self._json_response({'error': 'Missing effect or file'}, status=400)
            return
        wav_path = _resolve_audio_path(file_path)
        if not wav_path:
            self._json_response({'error': 'Invalid audio file'}, status=400)
            return
        try:
            cmd = [sys.executable, 'runner.py', '--analyze', effect_name, '--wav', wav_path]
            if palette_name:
                cmd += ['--chroma', palette_name]
            result = subprocess.run(
                cmd,
                cwd=EFFECTS_DIR, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                self._json_response({'error': result.stderr[:500]}, status=500)
                return
            data = json.loads(result.stdout)
            self._json_response(data)
        except subprocess.TimeoutExpired:
            self._json_response({'error': 'Analysis timed out'}, status=504)
        except json.JSONDecodeError:
            self._json_response({'error': 'Invalid JSON from analyzer'}, status=500)

    def _handle_palette_save(self):
        body = self._read_json_body()
        if body is None:
            return
        name = body.get('name', '').strip()
        if not name:
            self._json_response({'error': 'Name required'}, status=400)
            return
        mod = _palette_module()
        if name in mod.PALETTE_PRESETS:
            self._json_response({'error': 'Cannot overwrite built-in palette'}, status=400)
            return
        spec = {
            'colors': body.get('colors', body.get('palette', [[255, 255, 255]])),
            'gamma': float(body.get('gamma', 0.7)),
            'brightness_cap': float(body.get('brightness_cap', 1.0)),
            'spatial_mode': body.get('spatial_mode', 'uniform'),
            'fill_from': body.get('fill_from', 'start'),
        }
        mod.save_user_palette(name, spec)
        self._json_response({'ok': True})

    def _handle_palette_delete(self):
        body = self._read_json_body()
        if body is None:
            return
        name = body.get('name', '').strip()
        if not name:
            self._json_response({'error': 'Name required'}, status=400)
            return
        mod = _palette_module()
        if name in mod.PALETTE_PRESETS:
            self._json_response({'error': 'Cannot delete built-in palette'}, status=400)
            return
        if mod.delete_user_palette(name):
            self._json_response({'ok': True})
        else:
            self._json_response({'error': 'Palette not found'}, status=404)

    def _read_json_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return None

    def _handle_upload(self):
        """Handle audio file upload. Converts non-WAV formats to WAV via ffmpeg."""
        ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm'}

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
        _, ext = os.path.splitext(filename.lower())
        if ext not in ALLOWED_EXTENSIONS:
            self._json_response({'ok': False, 'error': f'Unsupported format. Accepted: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}, status=400)
            return

        # Save to uploads directory
        uploads_dir = os.path.join(SEGMENTS_DIR, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        needs_conversion = ext != '.wav'
        if needs_conversion:
            # Save original to temp file, convert to WAV
            import tempfile
            import subprocess
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            try:
                with os.fdopen(tmp_fd, 'wb') as f:
                    f.write(file_item.file.read())

                wav_filename = os.path.splitext(filename)[0] + '.wav'
                filepath = os.path.join(uploads_dir, wav_filename)

                # Avoid overwriting
                base_name = os.path.splitext(wav_filename)[0]
                counter = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(uploads_dir, f"{base_name}_{counter}.wav")
                    counter += 1

                result = subprocess.run(
                    ['ffmpeg', '-i', tmp_path, '-ar', '44100', '-ac', '2', filepath, '-y'],
                    capture_output=True, timeout=120)
                if result.returncode != 0:
                    self._json_response({'ok': False, 'error': 'Conversion failed: ' + result.stderr.decode(errors='replace')[-200:]}, status=400)
                    return
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            filepath = os.path.join(uploads_dir, filename)

            # Avoid overwriting
            base, wav_ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filepath = os.path.join(uploads_dir, f"{base}_{counter}{wav_ext}")
                counter += 1

            with open(filepath, 'wb') as f:
                f.write(file_item.file.read())

        # Validate it's actually a WAV file
        dur = _get_wav_duration(filepath)
        if dur == 0.0:
            os.remove(filepath)
            self._json_response({'ok': False, 'error': 'Invalid or unreadable audio file'}, status=400)
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

    def do_HEAD(self):
        """Handle HEAD requests — used by ensureFileOnServer to check if a file exists."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith('/audio/'):
            rel_path = path[len('/audio/'):]
            filepath = _resolve_audio_path(rel_path)
            if filepath and os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(os.path.getsize(filepath)))
                self.end_headers()
            else:
                self.send_error(404)
        else:
            self.send_error(404)

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
            self._serve_file_list(query)
        elif path.startswith('/api/render/'):
            rel_path = path[len('/api/render/'):]
            with_ann = 'annotations' in query
            features = None
            feat_str = query.get('features', [None])[0]
            if feat_str is not None:
                active = set(feat_str.split(',')) if feat_str else set()
                all_feat = ('rms', 'events')
                features = {n: (n in active) for n in all_feat}
            self._serve_render(rel_path, with_ann, features)
        elif path.startswith('/api/render-annotate/'):
            rel_path = path[len('/api/render-annotate/'):]
            self._serve_render_annotate(rel_path)
        elif path.startswith('/api/render-band-analysis/'):
            rel_path = path[len('/api/render-band-analysis/'):]
            self._serve_render_band_analysis(rel_path)
        elif path.startswith('/api/render-panel/'):
            rel_path = path[len('/api/render-panel/'):]
            panel = query.get('panel', [None])[0]
            if not panel:
                self._json_response({'error': 'panel parameter required'}, status=400)
            else:
                self._serve_render_panel(rel_path, panel)
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
            variant = query.get('variant', ['timbral'])[0]
            self._serve_lab(rel_path, variant)
        elif path.startswith('/api/annotations/'):
            rel_path = path[len('/api/annotations/'):]
            self._serve_annotations(rel_path)
        elif path == '/api/controllers':
            global _sculptures
            _sculptures = _load_sculptures()
            _resolve_controller_ports(_controllers)
            targets = _build_output_targets(_sculptures, _controllers)
            self._json_response(targets)
        elif path == '/api/effects':
            effects, deprecated, palettes = _get_effects_list()
            # Return active target, or fall back to persisted preference
            prefs = _load_effect_prefs()
            target = _active_controller
            target_type = None
            if target is None:
                if '__sculpture__' in prefs:
                    target = prefs['__sculpture__']
                    target_type = 'sculpture'
                elif '__controller__' in prefs:
                    target = prefs['__controller__']
                    target_type = 'controller'
            else:
                # Determine type from active target
                if any(s['id'] == target for s in _sculptures):
                    target_type = 'sculpture'
                else:
                    target_type = 'controller'
            self._json_response({
                'effects': effects,
                'deprecated': deprecated,
                'palettes': palettes,
                'running': _get_running_effect_name(),
                'controller': target,
                'controller_type': target_type,
            })
        elif path == '/api/palettes':
            self._json_response(_get_all_palettes_list())
        elif path == '/api/livereload':
            current = _source_hash()
            self._json_response({'hash': current, 'changed': current != _startup_hash})
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

    def _serve_file_list(self, query=None):
        files = discover_files()
        # In public mode, only return demo files + files the client claims to own
        if PUBLIC_MODE:
            demo = {'cinematic_drums.wav', 'freq_sweep.wav'}
            allowed = set()
            for p in (query or {}).get('paths', []):
                allowed.update(p.split(','))
            files = [f for f in files if f['path'] in allowed or f['path'] in demo]
        data = json.dumps(files).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
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

    def _serve_render_annotate(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404, 'File not found')
            return

        try:
            png_bytes, headers = render_annotate(filepath)
        except Exception as e:
            print(f"[render-annotate] Error: {e}")
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

    def _serve_render_band_analysis(self, rel_path):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404, 'File not found')
            return

        try:
            png_bytes, headers = render_band_analysis(filepath)
        except Exception as e:
            print(f"[render-band-analysis] Error: {e}")
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

    def _serve_render_panel(self, rel_path, panel_name):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404, 'File not found')
            return

        try:
            png_bytes, headers = render_single_panel(filepath, panel_name)
        except ValueError as e:
            self._json_response({'error': str(e)}, status=400)
            return
        except Exception as e:
            print(f"[render-panel] Error: {e}")
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

    def _serve_lab(self, rel_path, variant='timbral'):
        filepath = _resolve_audio_path(rel_path)
        if filepath is None:
            self.send_error(404)
            return

        try:
            png_bytes, headers = render_lab(filepath, variant=variant)
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
    global _server_start_iso
    from datetime import datetime, timezone
    _server_start_iso = datetime.now(timezone.utc).isoformat()

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
