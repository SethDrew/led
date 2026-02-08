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
    python runner.py wled_volume --wav ../audio-segments/fa_br_drop1.wav

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

import numpy as np
import sounddevice as sd

from base import AudioReactiveEffect

# ── Settings ───────────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024          # ~23ms per chunk
LED_FPS = 30
NUM_LEDS = 197             # Tree default
BAUD_RATE = 1000000
BRIGHTNESS_CAP = 0.03      # 3% for testing
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

    return effects


class SerialLEDOutput:
    """Sends RGB frames to Arduino via serial."""

    def __init__(self, port, num_leds, baud_rate=BAUD_RATE):
        self.num_leds = num_leds
        self.ser = None

        if port:
            try:
                import serial
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

    effect_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        # Process audio in callback thread — keeps main loop timing clean
        with effect_lock:
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
            with effect_lock:
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


def main():
    parser = argparse.ArgumentParser(description='Audio-reactive effect runner')
    parser.add_argument('effect', nargs='?', help='Effect name (use --list to see options)')
    parser.add_argument('--list', action='store_true', help='List available effects')
    parser.add_argument('--wav', help='WAV file to play (instead of live audio)')
    parser.add_argument('--no-leds', action='store_true', help='Terminal visualization only')
    parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
    parser.add_argument('--leds', type=int, default=NUM_LEDS, help='Number of LEDs')
    parser.add_argument('--brightness', type=float, default=BRIGHTNESS_CAP,
                        help='Brightness cap (0-1, default 0.03)')
    args = parser.parse_args()

    brightness_cap = args.brightness

    effects = get_effect_registry()

    if args.list or not args.effect:
        print("\n  Available effects:")
        print(f"  {'='*40}")
        if not effects:
            print("  (none found — implement effects in wled_sr/ or ours/)")
        for name, cls in effects.items():
            # Instantiate briefly to get the name
            try:
                e = cls(num_leds=1)
                print(f"  {name:20s} — {e.name}")
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
