#!/usr/bin/env python3
"""
Audio-Reactive Effect Runner

Feeds audio to any AudioReactiveEffect and drives LEDs via serial.
Supports live BlackHole capture and WAV file playback.

Usage:
    # List available effects
    python runner.py --list

    # Structured JSON output (for web_viewer)
    python runner.py --list-json

    # Run an effect on live audio
    python runner.py impulse
    python runner.py impulse_glow

    # Signal effect with palette override
    python runner.py impulse_glow --palette reds

    # Terminal-only (no LEDs)
    python runner.py impulse --no-leds

    # WAV file instead of live audio
    python runner.py impulse --wav ../research/audio-segments/fa_br_drop1.wav

    # Custom LED count / serial port
    python runner.py impulse --leds 150 --port /dev/cu.usbserial-11230

Controls:
    Ctrl+C  - Quit
"""

import sys
import os
import time
import argparse
import threading
import glob
import json
import base64

import numpy as np
import sounddevice as sd

from base import AudioReactiveEffect, ScalarSignalEffect
from palette import PALETTE_PRESETS, all_palettes, palette_to_dict, resolve_palette_name
from composed import ComposedEffect

# ── Settings ───────────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024          # ~23ms per chunk
LED_FPS = 30
NUM_LEDS = 197             # Tree default
BAUD_RATE = 1000000
BRIGHTNESS_CAP = 0.30      # 30%
START_BYTE_1 = 0xFF
START_BYTE_2 = 0xAA


def find_blackhole_device():
    """Auto-detect BlackHole audio device."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if 'blackhole' in d['name'].lower() and d['max_input_channels'] >= 2:
            return i
    return None


CMD_IDENTIFY = 0xFE
CMD_SET_ID = 0xFD


def identify_serial_port(port, baud=1000000, timeout=0.5):
    """Query a serial port for its EEPROM device_id.

    Returns device_id (int) or None if no response / not a streaming receiver.
    Opening the port triggers Arduino reset (DTR), so we wait for the
    bootloader to hand off to our firmware (~2s).
    """
    import serial
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)  # bootloader timeout
        ser.reset_input_buffer()
        ser.write(bytes([CMD_IDENTIFY]))
        resp = ser.read(2)
        ser.close()
        if len(resp) == 2 and resp[0] == CMD_IDENTIFY:
            return resp[1]
    except Exception:
        pass
    return None


def find_serial_port(controller=None):
    """Auto-detect Arduino serial port.

    If controller has a device_id, queries all CH340 ports to find the match.
    Falls back to port_hint, then first available port.
    """
    candidates = glob.glob('/dev/cu.usbserial-*')
    if not candidates:
        return None

    # If controller has EEPROM device_id, query each port
    if controller and 'device_id' in controller:
        target_id = controller['device_id']
        for port in candidates:
            dev_id = identify_serial_port(port)
            if dev_id == target_id:
                return port
        # EEPROM query failed — fall through to port_hint

    # Fallback: port_hint (USB location ID match)
    if controller:
        hint = controller.get('port_hint')
        if hint:
            matches = [p for p in candidates if hint in p]
            if matches:
                return matches[0]
        # Specific controller requested but not found
        return None

    # No controller specified — return first available
    return candidates[0]


def load_sculpture(sculpture_id):
    """Load a sculpture definition and its controller from JSON files.

    Returns (sculpture_def, controller_def) where:
        sculpture_def has 'branches', 'logical_leds', 'physical_leds'
        controller_def has 'port', 'leds', etc from controllers.json
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    with open(os.path.join(base_dir, 'hardware', 'sculptures.json')) as f:
        sculptures = json.load(f)
    with open(os.path.join(base_dir, 'hardware', 'controllers.json')) as f:
        controllers = json.load(f)

    sculpture = next((s for s in sculptures if s['id'] == sculpture_id), None)
    if not sculpture:
        raise ValueError(f"Unknown sculpture: {sculpture_id}")

    controller = next((c for c in controllers if c['id'] == sculpture['controller']), None)
    if not controller:
        raise ValueError(f"Unknown controller '{sculpture['controller']}' for sculpture {sculpture_id}")

    mode = sculpture.get('mode', 'branch')
    if mode == 'height':
        logical_leds = max(b['count'] for b in sculpture['branches'])
    else:
        logical_leds = sum(b['count'] for b in sculpture['branches'])
    physical_leds = sculpture.get('physical_leds', controller['leds'])

    sculpture['logical_leds'] = logical_leds
    sculpture['physical_leds'] = physical_leds

    return sculpture, controller


def apply_topology(logical_frame, sculpture_def):
    """Map logical LED frame to physical LED positions.

    Two modes:
      'branch' (default): Each branch gets its own slice of the logical array.
      'height': Logical array is a single height axis (0=base, 1=peak).
                Each branch samples it at its own resolution, rounding to
                nearest index. All branches share base and peak colors.

    Args:
        logical_frame: (logical_leds, 3) uint8 array
        sculpture_def: dict with 'branches', 'physical_leds', optional 'mode'

    Returns:
        (physical_leds, 3) uint8 array
    """
    physical = np.zeros((sculpture_def['physical_leds'], 3), dtype=logical_frame.dtype)
    mode = sculpture_def.get('mode', 'branch')
    max_idx = len(logical_frame) - 1

    if mode == 'height':
        for branch in sculpture_def['branches']:
            count = branch['count']
            start = branch['start']
            for j in range(count):
                t = j / max(count - 1, 1)  # 0=base, 1=peak
                gamma = branch.get('gamma', 1.0)
                height = t ** gamma  # gamma<1: fast rise at base, slower at top
                logical_idx = round(height * max_idx)
                phys = start + (count - 1 - j) if branch.get('reverse') else start + j
                physical[phys] = logical_frame[logical_idx]
    else:
        logical_offset = 0
        for branch in sculpture_def['branches']:
            count = branch['count']
            start = branch['start']
            logical_slice = logical_frame[logical_offset:logical_offset + count]
            if branch.get('reverse'):
                physical[start:start + count] = logical_slice[::-1]
            else:
                physical[start:start + count] = logical_slice
            logical_offset += count

    return physical


def get_effect_registry():
    """Auto-discover all available effects by scanning .py files.

    Finds all AudioReactiveEffect/ScalarSignalEffect subclasses that have
    a `registry_name` class attribute. New effects are auto-registered
    just by adding a .py file — no need to edit runner.py.

    Returns dict with two keys:
        'signals': {name: class} — ScalarSignalEffect subclasses (composable with palette)
        'effects': {name: class} — full AudioReactiveEffect subclasses (own RGB rendering)
    """
    import importlib.util
    import inspect

    signals = {}
    effects = {}

    # Infrastructure files to skip
    _SKIP = {'base', 'signals', 'composed', 'palette', 'runner', 'feature_computer', '__init__'}

    effects_dir = os.path.dirname(os.path.abspath(__file__))

    # Scan top-level .py files and wled_sr/ subdirectory
    py_files = glob.glob(os.path.join(effects_dir, '*.py'))
    py_files += glob.glob(os.path.join(effects_dir, 'wled_sr', '*.py'))

    for filepath in py_files:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        if module_name in _SKIP:
            continue

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            continue

        for _, cls in inspect.getmembers(mod, inspect.isclass):
            name = getattr(cls, 'registry_name', None)
            if not name:
                continue
            if not issubclass(cls, AudioReactiveEffect):
                continue
            if cls is AudioReactiveEffect or cls is ScalarSignalEffect:
                continue

            if issubclass(cls, ScalarSignalEffect):
                signals[name] = cls
            else:
                effects[name] = cls

    return {'signals': signals, 'effects': effects}


def create_effect(name, num_leds, sample_rate, palette_name=None):
    """Create an effect by name, composing with palette if it's a signal effect.

    Args:
        name: Effect registry name
        num_leds: Number of LEDs
        sample_rate: Audio sample rate
        palette_name: Optional palette override (signal effects only)

    Returns:
        AudioReactiveEffect instance
    """
    registry = get_effect_registry()
    signals = registry['signals']
    effects = registry['effects']

    if name in signals:
        signal = signals[name](num_leds=num_leds, sample_rate=sample_rate)
        preset = palette_name or signal.default_palette
        preset = resolve_palette_name(preset)
        available = all_palettes()
        if preset not in available:
            raise ValueError(f"Unknown palette: {preset}. Available: {', '.join(available.keys())}")
        import copy
        palette = copy.deepcopy(available[preset])
        return ComposedEffect(signal, palette)
    elif name in effects:
        return effects[name](num_leds=num_leds, sample_rate=sample_rate)
    else:
        raise ValueError(f"Unknown effect: {name}")


class MultiEffect(AudioReactiveEffect):
    """Runs multiple effects in parallel, concatenates their LED output."""

    def __init__(self, effects):
        total_leds = sum(e.num_leds for e in effects)
        super().__init__(total_leds)
        self.effects = effects

    @property
    def name(self):
        return ' + '.join(e.name for e in self.effects)

    def process_audio(self, mono_chunk):
        for effect in self.effects:
            effect.process_audio(mono_chunk)

    def render(self, dt):
        frames = [effect.render(dt) for effect in self.effects]
        return np.vstack(frames)

    def get_diagnostics(self):
        diag = {}
        for effect in self.effects:
            d = effect.get_diagnostics()
            tag = effect.name.split()[-1].lower()[:4]
            for k, v in d.items():
                diag[f'{tag}.{k}'] = v
        return diag


class SerialLEDOutput:
    """Sends RGB frames to Arduino via serial."""

    def __init__(self, port, num_leds, baud_rate=BAUD_RATE):
        self.num_leds = num_leds
        self.ser = None

        if port:
            try:
                import serial
                if port.startswith('rfc2217://'):
                    self.ser = serial.serial_for_url(port, baudrate=baud_rate, timeout=1)
                else:
                    self.ser = serial.Serial(port, baud_rate, timeout=1)
                    time.sleep(2)  # Arduino reset
                while self.ser.in_waiting:
                    self.ser.readline()
                print(f"  LED output: {port} ({num_leds} LEDs)")
            except Exception as e:
                print(f"  Serial connection failed: {e}")
                self.ser = None

    def send_frame(self, frame):
        """Send RGB frame. frame shape: (num_leds, 3), dtype uint8.

        Protocol: [0xFF][0xAA][RGB bytes][XOR checksum]
        Checksum = XOR of all RGB bytes. Firmware validates and holds
        last good frame on mismatch (prevents glitch flashes).
        """
        if self.ser:
            try:
                # Pad or truncate to match expected LED count
                if len(frame) < self.num_leds:
                    padded = np.zeros((self.num_leds, 3), dtype=np.uint8)
                    padded[:len(frame)] = frame
                    frame = padded
                elif len(frame) > self.num_leds:
                    frame = frame[:self.num_leds]
                rgb = frame.flatten()
                checksum = int(np.bitwise_xor.reduce(rgb))
                packet = bytearray([START_BYTE_1, START_BYTE_2])
                packet.extend(rgb.tobytes())
                packet.append(checksum)
                self.ser.write(packet)
                self.ser.flush()
                # Drain any Arduino responses (FPS stats, ready signals)
                # to prevent RX buffer buildup
                if self.ser.in_waiting:
                    self.ser.read(self.ser.in_waiting)
            except Exception:
                pass

    def close(self):
        if self.ser:
            black = np.zeros((self.num_leds, 3), dtype=np.uint8)
            self.send_frame(black)
            self.ser.close()


def print_diagnostics(effect, frame_num):
    """Print effect diagnostics to terminal."""
    diag = effect.get_diagnostics()
    parts = [f"  [{effect.name}]"]

    for key, val in diag.items():
        if isinstance(val, float):
            parts.append(f"{key}:{val:.2f}")
        elif isinstance(val, bool):
            parts.append(f"{key}:{'Y' if val else 'n'}")
        else:
            parts.append(f"{key}:{val}")

    sys.stdout.write('\r' + ' '.join(parts) + '   ')
    sys.stdout.flush()


def run_live(effect, led_output, device_id, brightness_cap=BRIGHTNESS_CAP,
             sculpture_def=None, stream_features=False):
    """Run effect on live BlackHole audio.

    Architecture (required by WS2812B timing):
      Audio callback thread: process_audio() — stores analysis results
      Main loop: render() + send_frame() at FIXED rate — no FFT here
    """
    # Determine what to stream: effect's own source features or generic FeatureComputer
    has_source_features = stream_features and hasattr(effect, 'source_features') and effect.source_features
    feat_computer = None
    if stream_features and not has_source_features:
        from feature_computer import FeatureComputer
        feat_computer = FeatureComputer(sample_rate=SAMPLE_RATE)
    # Peak-decay normalization per source feature for live display
    # Standard algorithm: instant attack, ~46s half-life at 30fps
    DISPLAY_PEAK_DECAY = 0.9995
    source_peak = {} if has_source_features else None

    # Write metadata line so consumer knows what features to expect
    if stream_features:
        if has_source_features:
            meta = {'_meta': True, 'features': effect.source_features}
        else:
            meta = {'_meta': True, 'features': [
                {'id': k, 'label': k, 'color': c} for k, c in [
                    ('abs_integral', '#e94560'), ('rms', '#ffd740'),
                    ('centroid', '#4ca5ff'), ('autocorr_conf', '#9c27b0'),
                ]
            ]}
        sys.stderr.write(json.dumps(meta) + '\n')
        sys.stderr.flush()

    def audio_callback(indata, frames, time_info, status):
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        # Process audio in callback thread — effect handles its own locking
        effect.process_audio(mono)
        if feat_computer:
            feat_computer.process_audio(mono)

    stream = sd.InputStream(
        device=device_id,
        channels=2,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )

    print("  Listening... Play music through BlackHole. Ctrl+C to stop.\n")

    try:
        stream.start()
        frame_interval = 1.0 / LED_FPS
        next_frame_time = time.time()
        frame_num = 0

        while True:
            # Render LED frame (reads pre-computed state from process_audio)
            frame = effect.render(frame_interval)

            # Apply brightness cap
            frame = (frame.astype(np.float32) * brightness_cap).astype(np.uint8)

            # Apply sculpture topology mapping (logical → physical)
            if sculpture_def:
                frame = apply_topology(frame, sculpture_def)

            # Send to hardware at fixed rate
            led_output.send_frame(frame)

            # Terminal display
            print_diagnostics(effect, frame_num)

            # Stream source features as JSONL to stderr
            if has_source_features:
                raw = effect.get_source_values()
                # Peak-decay normalization (standard algorithm)
                for k, v in raw.items():
                    prev = source_peak.get(k, 1e-10)
                    source_peak[k] = max(v, prev * DISPLAY_PEAK_DECAY)
                    raw[k] = v / source_peak[k] if source_peak[k] > 1e-10 else 0.0
                sys.stderr.write(json.dumps(raw) + '\n')
                sys.stderr.flush()
            elif feat_computer:
                features = feat_computer.get_features()
                sys.stderr.write(json.dumps(features) + '\n')
                sys.stderr.flush()

            frame_num += 1

            # Fixed-rate timing (absolute time targets to prevent drift)
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Behind schedule — reset to prevent backlog
                next_frame_time = time.time()

    except KeyboardInterrupt:
        print("\n\n  Stopping...")
    finally:
        stream.stop()
        stream.close()


def run_wav(effect, led_output, wav_path, brightness_cap=BRIGHTNESS_CAP,
            sculpture_def=None):
    """Run effect on a WAV file (simulates real-time playback)."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype='float32')
    if sr != SAMPLE_RATE:
        print(f"  Warning: WAV sample rate {sr} != {SAMPLE_RATE}, resampling not implemented")

    # Mix to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr
    print(f"  Playing {os.path.basename(wav_path)} ({duration:.1f}s)")
    print(f"  Simulating real-time playback... Ctrl+C to stop.\n")

    # Also play audio through speakers so you can hear it
    try:
        play_stream = sd.OutputStream(samplerate=sr, channels=1)
        play_stream.start()
    except Exception:
        play_stream = None

    try:
        frame_interval = 1.0 / LED_FPS
        next_frame_time = time.time()
        chunk_idx = 0
        frame_num = 0

        while chunk_idx < len(audio):
            # Feed audio chunk
            chunk_end = min(chunk_idx + CHUNK_SIZE, len(audio))
            chunk = audio[chunk_idx:chunk_end]
            effect.process_audio(chunk)
            chunk_idx = chunk_end

            # Play audio
            if play_stream:
                try:
                    play_stream.write(chunk.reshape(-1, 1))
                except Exception:
                    pass

            # Render at LED frame rate (not every audio chunk)
            now = time.time()
            if now >= next_frame_time:
                frame = effect.render(frame_interval)
                frame = (frame.astype(np.float32) * brightness_cap).astype(np.uint8)
                if sculpture_def:
                    frame = apply_topology(frame, sculpture_def)
                led_output.send_frame(frame)
                print_diagnostics(effect, frame_num)
                frame_num += 1
                next_frame_time += frame_interval

            # Pace to real-time
            elapsed_audio = chunk_end / sr
            elapsed_wall = time.time() - (next_frame_time - frame_interval * frame_num)
            # Simple pacing — not perfect but good enough for comparison
            time.sleep(max(0, CHUNK_SIZE / sr * 0.8))

    except KeyboardInterrupt:
        print("\n\n  Stopping...")
    finally:
        if play_stream:
            play_stream.stop()
            play_stream.close()

    print(f"\n  Finished. {frame_num} frames rendered.")


def analyze_effect(effect_name, wav_path, num_leds=NUM_LEDS, sample_rate=SAMPLE_RATE,
                   palette_name=None):
    """Run effect offline at full speed, collecting LED frames and diagnostics.

    Returns dict with led_data (base64), diagnostics, waveform_peaks, etc.
    """
    import soundfile as sf

    try:
        effect = create_effect(effect_name, num_leds, sample_rate, palette_name)
    except ValueError as e:
        return {'error': str(e)}

    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr
    frame_interval = 1.0 / LED_FPS

    # Determine feature source: effect's own source_features or generic FeatureComputer
    has_source = hasattr(effect, 'source_features') and effect.source_features
    feat_computer = None
    if not has_source:
        from feature_computer import FeatureComputer
        feat_computer = FeatureComputer(sample_rate=sr)

    # Collect results
    led_frames = []
    diag_list = []
    feat_list = []
    waveform_peaks = []

    # Track time in audio samples to decide when to render
    samples_per_frame = int(sr / LED_FPS)
    chunk_idx = 0
    next_render_sample = samples_per_frame  # render after first frame's worth of audio

    while chunk_idx < len(audio):
        chunk_end = min(chunk_idx + CHUNK_SIZE, len(audio))
        chunk = audio[chunk_idx:chunk_end]
        effect.process_audio(chunk)
        if feat_computer:
            feat_computer.process_audio(chunk)

        # Waveform peak for this chunk
        waveform_peaks.append(float(np.max(np.abs(chunk))))

        chunk_idx = chunk_end

        # Render at 30fps boundaries
        while chunk_idx >= next_render_sample and next_render_sample <= len(audio):
            frame = effect.render(frame_interval)
            # Clamp to uint8
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            led_frames.append(frame)

            # Collect diagnostics
            diag = effect.get_diagnostics()
            normalized = {}
            for k, v in diag.items():
                if isinstance(v, bool):
                    normalized[k] = 1.0 if v else 0.0
                elif isinstance(v, (int, float)):
                    normalized[k] = float(v)
                elif isinstance(v, str):
                    # Strings like "BEAT!" → 1.0, empty/quiet → 0.0
                    normalized[k] = 0.0 if v in ('', '-', 'n', 'no', 'off') else 1.0
                else:
                    normalized[k] = 0.0
            diag_list.append(normalized)

            # Collect features at same rate as LED frames
            if has_source:
                feat_list.append(effect.get_source_values())
            else:
                feat_list.append(feat_computer.get_features())

            next_render_sample += samples_per_frame

    # Build LED data as flat uint8 array, base64 encoded
    if led_frames:
        led_array = np.stack(led_frames)  # (N, num_leds, 3)
        led_b64 = base64.b64encode(led_array.tobytes()).decode('ascii')
    else:
        led_array = np.zeros((0, num_leds, 3), dtype=np.uint8)
        led_b64 = ''

    # Collect all diagnostic keys across all frames
    all_keys = []
    seen = set()
    for d in diag_list:
        for k in d:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Feature metadata
    if has_source:
        source_meta = list(effect.source_features)
        feature_keys = [f['id'] for f in source_meta]
        # Normalize raw source values to 0-1 by global max per feature
        # Skip features already marked as normalized (e.g. live_intensity)
        pre_normalized = {fm['id'] for fm in source_meta if fm.get('normalized')}
        for key in feature_keys:
            if key in pre_normalized:
                continue
            global_max = max((f[key] for f in feat_list), default=0.0)
            if global_max > 1e-10:
                for f in feat_list:
                    f[key] = f[key] / global_max
    else:
        source_meta = [
            {'id': 'abs_integral', 'label': 'Abs-Integral', 'color': '#e94560'},
            {'id': 'rms', 'label': 'RMS', 'color': '#ffd740'},
            {'id': 'centroid', 'label': 'Centroid', 'color': '#4ca5ff'},
            {'id': 'autocorr_conf', 'label': 'Autocorr', 'color': '#9c27b0'},
        ]
        feature_keys = [f['id'] for f in source_meta]

    return {
        'num_frames': len(led_frames),
        'num_leds': num_leds,
        'fps': LED_FPS,
        'duration': duration,
        'led_data': led_b64,
        'diagnostics': diag_list,
        'diag_keys': all_keys,
        'waveform_peaks': waveform_peaks,
        'features': feat_list,
        'feature_keys': feature_keys,
        'source_features': source_meta,
    }


def list_json():
    """Output structured JSON of all effects, signals, and palette presets."""
    registry = get_effect_registry()
    signals = []
    for name, cls in registry['signals'].items():
        try:
            e = cls(num_leds=1)
            signals.append({
                'name': name,
                'display_name': e.name,
                'description': e.description,
                'default_palette': e.default_palette,
                'is_signal': True,
            })
        except Exception:
            pass

    effects = []
    for name, cls in registry['effects'].items():
        try:
            e = cls(num_leds=1)
            effects.append({
                'name': name,
                'display_name': e.name,
                'description': e.description,
                'is_signal': False,
            })
        except Exception:
            pass

    palettes = []
    for name, pal in PALETTE_PRESETS.items():
        d = palette_to_dict(pal)
        d['name'] = name
        d['is_builtin'] = True
        palettes.append(d)
    from palette import load_user_palettes
    for name, pal in load_user_palettes().items():
        if name not in PALETTE_PRESETS:
            d = palette_to_dict(pal)
            d['name'] = name
            d['is_builtin'] = False
            palettes.append(d)

    return {
        'signals': signals,
        'effects': effects,
        'palettes': palettes,
    }


def main():
    parser = argparse.ArgumentParser(description='Audio-reactive effect runner')
    parser.add_argument('effect', nargs='?', help='Effect name (use --list to see options)')
    parser.add_argument('--list', action='store_true', help='List available effects')
    parser.add_argument('--list-json', action='store_true', help='Structured JSON output')
    parser.add_argument('--analyze', action='store_true', help='Offline analysis mode — outputs JSON')
    parser.add_argument('--chroma', '--palette', default=None, dest='palette',
                        help='Palette override (signal effects only)')
    parser.add_argument('--wav', help='WAV file to play (instead of live audio)')
    parser.add_argument('--no-leds', action='store_true', help='Terminal visualization only')
    parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
    parser.add_argument('--leds', type=int, default=NUM_LEDS, help='Number of LEDs')
    parser.add_argument('--sculpture', default=None, help='Sculpture ID (overrides --port/--leds)')
    parser.add_argument('--stream-features', action='store_true',
                        help='Stream FeatureComputer output as JSONL to stderr')
    parser.add_argument('--brightness', type=float, default=BRIGHTNESS_CAP,
                        help='Brightness cap (0-1, default 0.03)')
    args = parser.parse_args()

    # Structured JSON output
    if args.list_json:
        json.dump(list_json(), sys.stdout)
        sys.exit(0)

    # Analyze mode — run offline, output JSON to stdout
    if args.analyze:
        if not args.effect:
            print(json.dumps({'error': 'Effect name required'}))
            sys.exit(1)
        if not args.wav:
            print(json.dumps({'error': '--wav required for --analyze'}))
            sys.exit(1)
        result = analyze_effect(args.effect, args.wav, num_leds=args.leds,
                                palette_name=args.palette)
        json.dump(result, sys.stdout)
        sys.exit(0)

    brightness_cap = args.brightness

    registry = get_effect_registry()
    all_effects = {**registry['signals'], **registry['effects']}

    if args.list or not args.effect:
        print("\n  Signal effects (composable with --palette):")
        print(f"  {'='*55}")
        for name, cls in registry['signals'].items():
            try:
                e = cls(num_leds=1)
                palette_tag = f" [{e.default_palette}]"
                desc = f" | {e.description}" if e.description else ""
                print(f"  {name:20s} — {e.name}{palette_tag}{desc}")
            except Exception as ex:
                print(f"  {name:20s} — (error: {ex})")

        print(f"\n  Full effects (own color rendering):")
        print(f"  {'='*55}")
        for name, cls in registry['effects'].items():
            try:
                e = cls(num_leds=1)
                desc = f" | {e.description}" if e.description else ""
                print(f"  {name:20s} — {e.name}{desc}")
            except Exception as ex:
                print(f"  {name:20s} — (error: {ex})")

        print(f"\n  Palettes:")
        print(f"  {'='*55}")
        for name, pal in all_palettes().items():
            tag = '' if name in PALETTE_PRESETS else ' [user]'
            print(f"  {name:25s} — {pal.spatial_mode}, cap={pal.brightness_cap:.0%}, gamma={pal.gamma}{tag}")

        print()
        return

    if args.effect not in all_effects:
        print(f"  Unknown effect: {args.effect}")
        print(f"  Available: {', '.join(all_effects.keys())}")
        sys.exit(1)

    # Resolve sculpture topology (if specified)
    sculpture_def = None
    if args.sculpture:
        try:
            sculpture_def, controller = load_sculpture(args.sculpture)
        except ValueError as e:
            print(f"  Error: {e}")
            sys.exit(1)
        logical_leds = sculpture_def['logical_leds']
        physical_leds = sculpture_def['physical_leds']
        # Use controller's port unless explicitly overridden
        if not args.port:
            args.port = find_serial_port(controller)
        args.leds = physical_leds  # for SerialLEDOutput packet size
    else:
        logical_leds = args.leds
        physical_leds = args.leds

    # Create effect (with palette composition if signal)
    # Effect renders to LOGICAL LED count (unless it handles topology itself)
    try:
        effect = create_effect(args.effect, logical_leds, SAMPLE_RATE, args.palette)
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    # Effects with handles_topology render directly in physical space —
    # skip runner's topology mapping but keep physical_leds for serial packet size
    if getattr(effect, 'handles_topology', False) and sculpture_def:
        sculpture_def = None  # don't apply topology again

    print(f"\n  Audio-Reactive Effect Runner")
    print(f"  {'='*40}")
    print(f"  Effect: {effect.name}")
    if args.palette:
        print(f"  Palette: {args.palette}")
    if sculpture_def:
        print(f"  Sculpture: {args.sculpture} ({logical_leds} logical → {physical_leds} physical LEDs)")
    else:
        print(f"  LEDs: {logical_leds}")
    print(f"  Brightness cap: {brightness_cap*100:.0f}%")

    # LED output — initialized with PHYSICAL LED count
    serial_port = None
    if not args.no_leds:
        serial_port = args.port or find_serial_port()
        if serial_port is None:
            print("  No serial port found — terminal-only mode")

    led_output = SerialLEDOutput(serial_port, physical_leds)

    # Run
    try:
        if args.wav:
            run_wav(effect, led_output, args.wav, brightness_cap,
                    sculpture_def=sculpture_def)
        else:
            device_id = find_blackhole_device()
            if device_id is None:
                print("  Error: BlackHole not found.")
                print("  Available devices:")
                print(sd.query_devices())
                sys.exit(1)
            device_info = sd.query_devices(device_id)
            print(f"  Audio: {device_info['name']} (#{device_id})")
            run_live(effect, led_output, device_id, brightness_cap,
                     sculpture_def=sculpture_def,
                     stream_features=args.stream_features)
    finally:
        led_output.close()
        diag = effect.get_diagnostics()
        if diag:
            print(f"  Final: {diag}")
        print("  Done!")


if __name__ == '__main__':
    main()
