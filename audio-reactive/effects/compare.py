#!/usr/bin/env python3
"""
A/B Effect Comparison — run two effects on one dual-strip RGBW serial output.

Renders effect A to LEDs 0-49 and effect B to LEDs 50-99 on a single
serial connection. The stream_recv firmware splits these across two
physical SK6812 RGBW strips (GPIO 10 and GPIO 21).

Effects return RGB (N,3) or RGBW (N,4). RGB frames get a smart conversion:
below a brightness threshold, all light goes to the W channel (avoids RGB
die strobe at low PWM). Above, RGB carries the color with W=0.

Usage:
    # Live mic
    python compare.py sparkle_burst flicker_flame_warmth \\
        --mic --port /dev/cu.usbmodem112201

    # WAV file
    python compare.py sparkle_burst flicker_flame_warmth \\
        --wav ../clips/speaking.wav --port /dev/cu.usbmodem112201

    # No hardware (terminal diagnostics only)
    python compare.py sparkle_burst flicker_flame_warmth \\
        --wav ../clips/speaking.wav --no-leds
"""

import sys
import os
import time
import argparse
import numpy as np
import sounddevice as sd

# Reuse runner infrastructure
from runner import (
    SerialLEDOutput, get_effect_registry, SAMPLE_RATE, CHUNK_SIZE,
    LED_FPS, BRIGHTNESS_CAP, find_blackhole_device, NUM_LEDS,
    START_BYTE_1, START_BYTE_2,
)
from base import AudioReactiveEffect


def rgb_to_rgbw(frame):
    """Convert (N,3) RGB to (N,4) RGBW with pure-W at low brightness.

    Below PURE_W_THRESHOLD brightness, output equal-channel values so the
    firmware puts all light on the dedicated warm white LED (avoids RGB
    die strobe from near-zero PWM toggling). Above, smoothly blend in
    the actual RGB color with W=0.
    """
    r = frame[:, 0].astype(np.float32)
    g = frame[:, 1].astype(np.float32)
    b = frame[:, 2].astype(np.float32)

    brightness = np.maximum(r, np.maximum(g, b)) / 255.0

    # Below this threshold, shift everything to W channel
    PURE_W_THRESHOLD = 0.25
    DEADBAND = 0.02  # below this, snap to full black

    # Blend factor: 0 at low brightness (pure W), 1 above threshold (full RGB)
    blend = np.clip((brightness - DEADBAND) / (PURE_W_THRESHOLD - DEADBAND), 0.0, 1.0)

    # Average brightness for W channel (perceptual, weighted toward max channel)
    avg = (r + g + b) / 3.0

    out = np.zeros((len(frame), 4), dtype=np.uint8)
    # RGB channels fade in above threshold
    out[:, 0] = np.clip(r * blend, 0, 255).astype(np.uint8)
    out[:, 1] = np.clip(g * blend, 0, 255).astype(np.uint8)
    out[:, 2] = np.clip(b * blend, 0, 255).astype(np.uint8)
    # W channel: carries all light at low brightness, fades out above threshold
    out[:, 3] = np.clip(avg * (1.0 - blend), 0, 255).astype(np.uint8)

    return out


def ensure_rgbw(frame):
    """Ensure frame is (N, 4) RGBW. Convert from RGB if needed."""
    if frame.ndim == 2 and frame.shape[1] == 4:
        return frame
    return rgb_to_rgbw(frame)


class ChunkedRGBWOutput:
    """Serial output for RGBW frames with chunked writes for USB-CDC."""
    CHUNK_SIZE = 200

    def __init__(self, port, num_leds):
        self.num_leds = num_leds
        self.ser = None
        if port:
            import serial
            self.ser = serial.Serial(port, 1000000, timeout=0.01)
            time.sleep(0.1)
            self.ser.reset_input_buffer()

    def send_frame(self, frame):
        """Send RGBW frame. frame shape: (num_leds, 4), dtype uint8."""
        if not self.ser:
            return
        try:
            if len(frame) < self.num_leds:
                padded = np.zeros((self.num_leds, 4), dtype=np.uint8)
                padded[:len(frame)] = frame
                frame = padded
            elif len(frame) > self.num_leds:
                frame = frame[:self.num_leds]
            data = frame.flatten()
            checksum = int(np.bitwise_xor.reduce(data))
            packet = bytearray([START_BYTE_1, START_BYTE_2])
            packet.extend(data.tobytes())
            packet.append(checksum)
            for start in range(0, len(packet), self.CHUNK_SIZE):
                self.ser.write(packet[start:start + self.CHUNK_SIZE])
                self.ser.flush()
                if start + self.CHUNK_SIZE < len(packet):
                    time.sleep(0.001)
            # Drain incoming stats
            while self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
        except Exception:
            pass

    def close(self):
        if self.ser:
            black = np.zeros((self.num_leds, 4), dtype=np.uint8)
            self.send_frame(black)
            self.ser.close()


def create_effect(spec, num_leds, sample_rate, registry):
    """Create effect from spec. Supports 'name' or 'name:kwarg=val' syntax.

    Examples: 'sparkle_burst', 'flicker_flame_warmth:flicker_scale=2.0'
    """
    signals = registry['signals']
    effects = registry['effects']

    # Parse name:key=val,key=val
    kwargs = {}
    if ':' in spec:
        name, params = spec.split(':', 1)
        for p in params.split(','):
            k, v = p.split('=')
            # Auto-convert numeric values
            try:
                v = float(v)
                if v == int(v):
                    v = int(v)
            except ValueError:
                pass
            kwargs[k] = v
    else:
        name = spec

    if name in effects:
        return effects[name](num_leds, sample_rate, **kwargs)
    elif name in signals:
        return signals[name](num_leds, sample_rate, **kwargs)
    else:
        all_names = sorted(set(list(signals.keys()) + list(effects.keys())))
        raise ValueError(f"Unknown effect '{name}'. Available: {', '.join(all_names)}")


def render_rgbw(effect, dt, brightness_cap):
    """Render an effect and return (N, 4) RGBW uint8 with brightness cap."""
    frame = effect.render(dt)
    frame = (frame.astype(np.float32) * brightness_cap).astype(np.uint8)
    return ensure_rgbw(frame)


def run_compare_wav(effect_a, effect_b, led_output, wav_path, brightness_cap):
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr
    print(f"  Playing {os.path.basename(wav_path)} ({duration:.1f}s)")
    print(f"  Strip A: {effect_a.name}  |  Strip B: {effect_b.name}\n")

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
            chunk_end = min(chunk_idx + CHUNK_SIZE, len(audio))
            chunk = audio[chunk_idx:chunk_end]

            effect_a.process_audio(chunk)
            effect_b.process_audio(chunk)
            chunk_idx = chunk_end

            if play_stream:
                try:
                    play_stream.write(chunk.reshape(-1, 1))
                except Exception:
                    pass

            now = time.time()
            if now >= next_frame_time:
                frame_a = render_rgbw(effect_a, frame_interval, brightness_cap)
                frame_b = render_rgbw(effect_b, frame_interval, brightness_cap)

                combined = np.vstack([frame_a, frame_b])
                led_output.send_frame(combined)

                # Diagnostics
                diag_a = effect_a.get_diagnostics()
                diag_b = effect_b.get_diagnostics()
                e_a = diag_a.get('energy', diag_a.get('bright', 0))
                e_b = diag_b.get('energy', diag_b.get('bright', 0))
                if isinstance(e_a, float):
                    e_a = f"{e_a:.2f}"
                if isinstance(e_b, float):
                    e_b = f"{e_b:.2f}"
                sys.stdout.write(f"\r  A:{effect_a.name}={e_a}  B:{effect_b.name}={e_b}  frame:{frame_num}   ")
                sys.stdout.flush()

                frame_num += 1
                next_frame_time += frame_interval

            time.sleep(max(0, CHUNK_SIZE / sr * 0.8))

    except KeyboardInterrupt:
        print("\n\n  Stopping...")
    finally:
        if play_stream:
            play_stream.stop()
            play_stream.close()

    print(f"\n  Finished. {frame_num} frames rendered.")


def run_compare_live(effect_a, effect_b, led_output, device_id, brightness_cap):
    dev_info = sd.query_devices(device_id if device_id is not None else sd.default.device[0])
    channels = min(int(dev_info['max_input_channels']), 2)

    def audio_callback(indata, frames, time_info, status):
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        effect_a.process_audio(mono)
        effect_b.process_audio(mono)

    stream = sd.InputStream(
        device=device_id,
        channels=channels,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )

    print(f"  Strip A: {effect_a.name}  |  Strip B: {effect_b.name}")
    print(f"  Audio device: {dev_info['name']} ({channels}ch)")
    print("  Listening... Ctrl+C to stop.\n")

    try:
        stream.start()
        frame_interval = 1.0 / LED_FPS
        next_frame_time = time.time()
        frame_num = 0

        while True:
            frame_a = render_rgbw(effect_a, frame_interval, brightness_cap)
            frame_b = render_rgbw(effect_b, frame_interval, brightness_cap)

            combined = np.vstack([frame_a, frame_b])
            led_output.send_frame(combined)

            frame_num += 1
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.time()

    except KeyboardInterrupt:
        print("\n\n  Stopping...")
    finally:
        stream.stop()
        stream.close()


def main():
    parser = argparse.ArgumentParser(description='A/B effect comparison on dual RGBW strips')
    parser.add_argument('effect_a', help='Effect name for strip A (GPIO 10)')
    parser.add_argument('effect_b', help='Effect name for strip B (GPIO 21)')
    parser.add_argument('--wav', help='WAV file (omit for live audio)')
    parser.add_argument('--port', help='Serial port for dual-strip receiver')
    parser.add_argument('--no-leds', action='store_true', help='Terminal only, no serial')
    parser.add_argument('--leds', type=int, default=50, help='LEDs per strip')
    parser.add_argument('--mic', action='store_true', help='Use default mic instead of BlackHole')
    parser.add_argument('--brightness', type=float, default=BRIGHTNESS_CAP)
    args = parser.parse_args()

    registry = get_effect_registry()
    leds_per = args.leds
    total_leds = leds_per * 2

    effect_a = create_effect(args.effect_a, leds_per, SAMPLE_RATE, registry)
    effect_b = create_effect(args.effect_b, leds_per, SAMPLE_RATE, registry)

    print(f"\n  === A/B Comparison (RGBW) ===")
    print(f"  Strip A (GPIO 10): {effect_a.name}")
    print(f"  Strip B (GPIO 21): {effect_b.name}")

    serial_port = None if args.no_leds else args.port
    led_output = ChunkedRGBWOutput(serial_port, total_leds)

    try:
        if args.wav:
            run_compare_wav(effect_a, effect_b, led_output, args.wav, args.brightness)
        else:
            if args.mic:
                device_id = None
                print(f"  Audio: system default mic")
            else:
                device_id = find_blackhole_device()
                if device_id is None:
                    print("  No BlackHole found, falling back to default mic")
                    device_id = None
            run_compare_live(effect_a, effect_b, led_output, device_id, args.brightness)
    finally:
        led_output.close()


if __name__ == '__main__':
    main()
