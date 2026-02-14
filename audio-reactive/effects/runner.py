#!/usr/bin/env python3
"""
Audio-Reactive Effect Runner

Feeds audio to any AudioReactiveEffect and drives LEDs via serial.
Supports live BlackHole capture and WAV file playback.

Usage:
    # List available effects
    python runner.py --list

    # Run an effect on live audio
    python runner.py wled_volume
    python runner.py wled_geq
    python runner.py wled_beat

    # Terminal-only (no LEDs)
    python runner.py wled_volume --no-leds

    # WAV file instead of live audio
    python runner.py wled_volume --wav ../research/audio-segments/fa_br_drop1.wav

    # Custom LED count / serial port
    python runner.py wled_volume --leds 150 --port /dev/cu.usbserial-11230

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

from base import AudioReactiveEffect

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


def find_serial_port():
    """Auto-detect Arduino serial port."""
    candidates = glob.glob('/dev/cu.usbserial-*')
    if candidates:
        return candidates[0]
    return None


def get_effect_registry():
    """Discover all available effects."""
    effects = {}

    # WLED-SR effects
    try:
        from wled_sr.volume_reactive import WLEDVolumeReactive
        effects['wled_volume'] = WLEDVolumeReactive
    except ImportError:
        pass

    try:
        from wled_sr.frequency_reactive import WLEDFrequencyReactive
        effects['wled_geq'] = WLEDFrequencyReactive
    except ImportError:
        pass

    try:
        from wled_sr.beat_reactive import WLEDBeatReactive
        effects['wled_beat'] = WLEDBeatReactive
    except ImportError:
        pass

    # Our custom effects
    try:
        from three_voices import ThreeVoicesEffect
        effects['hpss_voices'] = ThreeVoicesEffect
    except ImportError:
        pass

    try:
        from bass_pulse import BassPulseEffect
        effects['bass_pulse'] = BassPulseEffect
    except ImportError:
        pass

    try:
        from absint_pulse import AbsIntPulseEffect
        effects['absint_pulse'] = AbsIntPulseEffect
    except ImportError:
        pass

    try:
        from absint_proportional import AbsIntProportionalEffect
        effects['absint_prop'] = AbsIntProportionalEffect
    except ImportError:
        pass

    try:
        from absint_predictive import AbsIntPredictiveEffect
        effects['absint_predict'] = AbsIntPredictiveEffect
    except ImportError:
        pass

    try:
        from band_prop import BandProportionalEffect
        effects['band_prop'] = BandProportionalEffect
    except ImportError:
        pass

    try:
        from absint_reds import AbsIntRedsEffect
        effects['absint_red_palette'] = AbsIntRedsEffect
    except ImportError:
        pass

    try:
        from absint_downbeat import AbsIntDownbeatEffect
        effects['absint_downbeat'] = AbsIntDownbeatEffect
    except ImportError:
        pass

    try:
        from absint_sections import AbsIntSectionsEffect
        effects['absint_sections'] = AbsIntSectionsEffect
    except ImportError:
        pass

    try:
        from longint_sections import LongIntSectionsEffect
        effects['longint_sections'] = LongIntSectionsEffect
    except ImportError:
        pass

    try:
        from absint_breathe import AbsIntBreatheEffect
        effects['absint_breathe'] = AbsIntBreatheEffect
    except ImportError:
        pass

    try:
        from tempo_pulse import TempoPulseEffect
        effects['tempo_pulse'] = TempoPulseEffect
    except ImportError:
        pass

    try:
        from absint_snake import AbsIntSnakeEffect
        effects['absint_snake'] = AbsIntSnakeEffect
    except ImportError:
        pass

    try:
        from absint_band_pulse import AbsIntBandPulseEffect
        effects['absint_bands'] = AbsIntBandPulseEffect
    except ImportError:
        pass

    try:
        from absint_meter import AbsIntMeterEffect
        effects['absint_meter'] = AbsIntMeterEffect
    except ImportError:
        pass

    try:
        from rms_meter import RMSMeterEffect
        effects['rms_meter'] = RMSMeterEffect
    except ImportError:
        pass

    try:
        from basic_sparkles import BasicSparklesEffect
        effects['basic_sparkles'] = BasicSparklesEffect
    except ImportError:
        pass

    try:
        from band_sparkles import BandSparklesEffect
        effects['band_sparkles'] = BandSparklesEffect
    except ImportError:
        pass

    try:
        from band_tempo_sparkles import BandTempoSparklesEffect
        effects['band_tempo_sparkles'] = BandTempoSparklesEffect
    except ImportError:
        pass

    # All 12 WLED 1D effects side by side
    try:
        from wled_sr.all_effects import WLEDAllEffects
        effects['wled_all'] = WLEDAllEffects
    except ImportError:
        pass

    return effects


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
        """Send RGB frame. frame shape: (num_leds, 3), dtype uint8."""
        if self.ser:
            try:
                packet = bytearray([START_BYTE_1, START_BYTE_2])
                packet.extend(frame.flatten().tobytes())
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


def run_live(effect, led_output, device_id, brightness_cap=BRIGHTNESS_CAP):
    """Run effect on live BlackHole audio.

    Architecture (required by WS2812B timing):
      Audio callback thread: process_audio() — stores analysis results
      Main loop: render() + send_frame() at FIXED rate — no FFT here
    """

    def audio_callback(indata, frames, time_info, status):
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        # Process audio in callback thread — effect handles its own locking
        effect.process_audio(mono)

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

            # Send to hardware at fixed rate
            led_output.send_frame(frame)

            # Terminal display
            print_diagnostics(effect, frame_num)
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


def run_wav(effect, led_output, wav_path, brightness_cap=BRIGHTNESS_CAP):
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


def analyze_effect(effect_name, wav_path, num_leds=NUM_LEDS, sample_rate=SAMPLE_RATE):
    """Run effect offline at full speed, collecting LED frames and diagnostics.

    Returns dict with led_data (base64), diagnostics, waveform_peaks, etc.
    """
    import soundfile as sf

    effects = get_effect_registry()
    if effect_name not in effects:
        return {'error': f'Unknown effect: {effect_name}'}

    effect = effects[effect_name](num_leds=num_leds, sample_rate=sample_rate)

    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr
    frame_interval = 1.0 / LED_FPS

    # Collect results
    led_frames = []
    diag_list = []
    waveform_peaks = []

    # Track time in audio samples to decide when to render
    samples_per_frame = int(sr / LED_FPS)
    chunk_idx = 0
    next_render_sample = samples_per_frame  # render after first frame's worth of audio

    while chunk_idx < len(audio):
        chunk_end = min(chunk_idx + CHUNK_SIZE, len(audio))
        chunk = audio[chunk_idx:chunk_end]
        effect.process_audio(chunk)

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

    return {
        'num_frames': len(led_frames),
        'num_leds': num_leds,
        'fps': LED_FPS,
        'duration': duration,
        'led_data': led_b64,
        'diagnostics': diag_list,
        'diag_keys': all_keys,
        'waveform_peaks': waveform_peaks,
    }


def main():
    parser = argparse.ArgumentParser(description='Audio-reactive effect runner')
    parser.add_argument('effect', nargs='?', help='Effect name (use --list to see options)')
    parser.add_argument('--list', action='store_true', help='List available effects')
    parser.add_argument('--analyze', action='store_true', help='Offline analysis mode — outputs JSON')
    parser.add_argument('--wav', help='WAV file to play (instead of live audio)')
    parser.add_argument('--no-leds', action='store_true', help='Terminal visualization only')
    parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
    parser.add_argument('--leds', type=int, default=NUM_LEDS, help='Number of LEDs')
    parser.add_argument('--brightness', type=float, default=BRIGHTNESS_CAP,
                        help='Brightness cap (0-1, default 0.03)')
    args = parser.parse_args()

    # Analyze mode — run offline, output JSON to stdout
    if args.analyze:
        if not args.effect:
            print(json.dumps({'error': 'Effect name required'}))
            sys.exit(1)
        if not args.wav:
            print(json.dumps({'error': '--wav required for --analyze'}))
            sys.exit(1)
        result = analyze_effect(args.effect, args.wav, num_leds=args.leds)
        json.dump(result, sys.stdout)
        sys.exit(0)

    brightness_cap = args.brightness

    effects = get_effect_registry()

    if args.list or not args.effect:
        print("\n  Available effects:")
        print(f"  {'='*40}")
        if not effects:
            print("  (none found — implement effects in wled_sr/ or ours/)")
        for name, cls in effects.items():
            # Instantiate briefly to get the name and description
            try:
                e = cls(num_leds=1)
                desc = f" | {e.description}" if e.description else ""
                print(f"  {name:20s} — {e.name}{desc}")
            except Exception as ex:
                print(f"  {name:20s} — (error: {ex})")
        print()
        return

    if args.effect not in effects:
        print(f"  Unknown effect: {args.effect}")
        print(f"  Available: {', '.join(effects.keys())}")
        sys.exit(1)

    # Create effect
    effect = effects[args.effect](num_leds=args.leds, sample_rate=SAMPLE_RATE)

    print(f"\n  Audio-Reactive Effect Runner")
    print(f"  {'='*40}")
    print(f"  Effect: {effect.name}")
    print(f"  LEDs: {args.leds}")
    print(f"  Brightness cap: {brightness_cap*100:.0f}%")

    # LED output
    serial_port = None
    if not args.no_leds:
        serial_port = args.port or find_serial_port()
        if serial_port is None:
            print("  No serial port found — terminal-only mode")

    led_output = SerialLEDOutput(serial_port, args.leds)

    # Run
    try:
        if args.wav:
            run_wav(effect, led_output, args.wav, brightness_cap)
        else:
            device_id = find_blackhole_device()
            if device_id is None:
                print("  Error: BlackHole not found.")
                print("  Available devices:")
                print(sd.query_devices())
                sys.exit(1)
            device_info = sd.query_devices(device_id)
            print(f"  Audio: {device_info['name']} (#{device_id})")
            run_live(effect, led_output, device_id, brightness_cap)
    finally:
        led_output.close()
        diag = effect.get_diagnostics()
        if diag:
            print(f"  Final: {diag}")
        print("  Done!")


if __name__ == '__main__':
    main()
