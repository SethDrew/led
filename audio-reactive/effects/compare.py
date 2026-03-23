#!/usr/bin/env python3
"""
A/B Effect Comparison — run two effects on one dual-strip serial output.

Renders effect A to LEDs 0-49 and effect B to LEDs 50-99 on a single
serial connection. The stream_recv firmware splits these across two
physical strips (GPIO 1 and GPIO 0).

Usage:
    # WAV file
    python compare.py syllable_pulse sparkle_burst \\
        --wav ../research/audio-segments/poem_reading.wav \\
        --port /dev/cu.usbmodem11401

    # Live audio (BlackHole)
    python compare.py syllable_pulse sparkle_burst \\
        --port /dev/cu.usbmodem11401

    # No hardware (terminal diagnostics only)
    python compare.py syllable_pulse sparkle_burst \\
        --wav ../research/audio-segments/poem_reading.wav --no-leds
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


class ChunkedSerialOutput(SerialLEDOutput):
    """SerialLEDOutput with chunked writes for ESP32-C3 USB-CDC.

    The ESP32-C3 USB-CDC RX buffer is ~256 bytes. Frames larger than
    that need to be sent in chunks with small delays between them.
    """
    CHUNK_SIZE = 200

    def send_frame(self, frame):
        if self.ser:
            try:
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
                # Send in chunks to avoid USB-CDC buffer overflow
                for start in range(0, len(packet), self.CHUNK_SIZE):
                    self.ser.write(packet[start:start + self.CHUNK_SIZE])
                    self.ser.flush()
                    if start + self.CHUNK_SIZE < len(packet):
                        time.sleep(0.001)
                self._drain_and_parse()
            except Exception:
                pass


def create_effect(name, num_leds, sample_rate, registry):
    signals = registry['signals']
    effects = registry['effects']
    if name in effects:
        return effects[name](num_leds, sample_rate)
    elif name in signals:
        return signals[name](num_leds, sample_rate)
    else:
        all_names = sorted(set(list(signals.keys()) + list(effects.keys())))
        raise ValueError(f"Unknown effect '{name}'. Available: {', '.join(all_names)}")


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

            # Feed same audio to both effects
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
                frame_a = effect_a.render(frame_interval)
                frame_b = effect_b.render(frame_interval)

                frame_a = (frame_a.astype(np.float32) * brightness_cap).astype(np.uint8)
                frame_b = (frame_b.astype(np.float32) * brightness_cap).astype(np.uint8)

                # Concatenate: A (0-49) + B (50-99)
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
    # Query device to get its channel count
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
            frame_a = effect_a.render(frame_interval)
            frame_b = effect_b.render(frame_interval)

            frame_a = (frame_a.astype(np.float32) * brightness_cap).astype(np.uint8)
            frame_b = (frame_b.astype(np.float32) * brightness_cap).astype(np.uint8)

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
    parser = argparse.ArgumentParser(description='A/B effect comparison on dual strips')
    parser.add_argument('effect_a', help='Effect name for strip A (GPIO 1)')
    parser.add_argument('effect_b', help='Effect name for strip B (GPIO 0)')
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

    print(f"\n  === A/B Comparison ===")
    print(f"  Strip A (GPIO 1): {effect_a.name}")
    print(f"  Strip B (GPIO 0): {effect_b.name}")

    serial_port = None if args.no_leds else args.port
    led_output = ChunkedSerialOutput(serial_port, total_leds)

    try:
        if args.wav:
            run_compare_wav(effect_a, effect_b, led_output, args.wav, args.brightness)
        else:
            if args.mic:
                device_id = None  # use system default input (built-in mic)
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
