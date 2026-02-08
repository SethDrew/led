#!/usr/bin/env python3
"""
Real-Time Beat-Reactive LED Controller

Captures audio from BlackHole, detects beats using bass-band spectral flux,
and drives LED strip with pulse/decay effect.

This uses ALGORITHMIC beat detection (not user taps) so we can validate
whether the algorithm produces compelling LED effects.

Usage:
    python realtime_beat_led.py                    # Auto-detect audio + serial (tree: 197 LEDs)
    python realtime_beat_led.py --no-leds          # Terminal visualization only
    python realtime_beat_led.py --port /dev/cu.usbserial-11240  # Tree
    python realtime_beat_led.py --port /dev/cu.usbserial-11230 --leds 150  # Nebula strip

Controls:
    q / Ctrl+C  - Quit
"""

import sys
import os
import time
import argparse
import threading
import numpy as np
import sounddevice as sd

# ── Audio Settings ──────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024          # ~23ms per chunk — low latency
N_FFT = 2048               # FFT window size
HOP_SIZE = CHUNK_SIZE      # Process every chunk

# ── Beat Detection Settings ─────────────────────────────────────────
BASS_LOW_HZ = 20
BASS_HIGH_HZ = 250
FLUX_HISTORY_SEC = 3.0     # Seconds of spectral flux history for threshold
MIN_BEAT_INTERVAL_SEC = 0.3  # Minimum time between beats (~200 BPM max)
THRESHOLD_MULTIPLIER = 1.5  # How many std above mean to trigger

# ── LED Settings ────────────────────────────────────────────────────
NUM_LEDS = 197             # Tree: 197 nodes, Nebula strip: 150
BAUD_RATE = 1000000
START_BYTE_1 = 0xFF
START_BYTE_2 = 0xAA

# ── Visual Settings ─────────────────────────────────────────────────
ATTACK_ALPHA = 0.95        # Fast attack (0=instant, 1=slow)
DECAY_ALPHA = 0.12         # Slow decay (lower = slower fade)
GAMMA = 2.2                # LED brightness correction
LED_FPS = 30               # Frame rate for LED output (reduced to prevent serial overflow)

# Colors (RGB, 0-255)
BEAT_COLOR = np.array([255, 0, 0], dtype=np.float64)  # Solid red
BASE_COLOR = np.array([0, 0, 0], dtype=np.float64)    # Black (off)


def find_blackhole_device():
    """Auto-detect BlackHole audio device."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if 'blackhole' in d['name'].lower() and d['max_input_channels'] >= 2:
            return i
    return None


def find_serial_port():
    """Auto-detect Arduino serial port."""
    import glob
    candidates = glob.glob('/dev/cu.usbserial-*')
    if candidates:
        return candidates[0]
    return None


class BeatDetector:
    """Real-time bass-band spectral flux beat detection."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT):
        self.sr = sample_rate
        self.n_fft = n_fft

        # Frequency bin indices for bass range
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self.bass_bins = np.where((freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ))[0]

        # State
        self.prev_spectrum = None
        self.flux_history = []
        self.max_history = int(FLUX_HISTORY_SEC * sample_rate / CHUNK_SIZE)
        self.last_beat_time = 0
        self.beat_count = 0

        # Windowing
        self.window = np.hanning(n_fft)

        # Audio buffer for overlapping FFT
        self.audio_buffer = np.zeros(n_fft)

    def process_chunk(self, audio_chunk):
        """Process an audio chunk. Returns (is_beat, flux_value, threshold)."""
        # Shift buffer and add new audio
        chunk_len = len(audio_chunk)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = audio_chunk

        # Windowed FFT
        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Extract bass band
        bass_spectrum = spectrum[self.bass_bins]

        if self.prev_spectrum is None:
            self.prev_spectrum = bass_spectrum
            return False, 0.0, 0.0

        # Spectral flux: sum of positive differences (half-wave rectified)
        diff = bass_spectrum - self.prev_spectrum
        flux = np.sum(np.maximum(diff, 0))
        self.prev_spectrum = bass_spectrum

        # Adaptive threshold
        self.flux_history.append(flux)
        if len(self.flux_history) > self.max_history:
            self.flux_history.pop(0)

        if len(self.flux_history) < 10:
            return False, flux, 0.0

        mean_flux = np.mean(self.flux_history)
        std_flux = np.std(self.flux_history)
        threshold = mean_flux + THRESHOLD_MULTIPLIER * std_flux

        # Beat detection with minimum interval
        now = time.time()
        is_beat = (flux > threshold and
                   (now - self.last_beat_time) > MIN_BEAT_INTERVAL_SEC)

        if is_beat:
            self.last_beat_time = now
            self.beat_count += 1

        return is_beat, flux, threshold


class LEDController:
    """Manages LED state with pulse/decay effect."""

    def __init__(self, num_leds=NUM_LEDS, serial_port=None):
        self.num_leds = num_leds
        self.brightness = 0.0  # Current brightness (0-1)
        self.ser = None

        if serial_port:
            try:
                import serial
                self.ser = serial.Serial(serial_port, BAUD_RATE, timeout=1)
                time.sleep(2)  # Arduino reset
                # Drain startup messages
                while self.ser.in_waiting:
                    self.ser.readline()
                print(f"  LED strip connected on {serial_port}")
            except Exception as e:
                print(f"  Serial connection failed: {e}")
                self.ser = None

    def trigger_beat(self, intensity=1.0):
        """Trigger a beat flash."""
        self.brightness = max(self.brightness, intensity)

    def update(self, dt):
        """Update brightness with exponential decay. Returns current frame."""
        # Decay
        decay = DECAY_ALPHA ** (dt * LED_FPS)  # Time-based decay
        self.brightness *= (1.0 - decay)
        self.brightness = max(self.brightness, 0.0)

        # Gamma-corrected brightness
        gamma_brightness = self.brightness ** (1.0 / GAMMA)

        # Interpolate color
        color = BASE_COLOR + (BEAT_COLOR - BASE_COLOR) * gamma_brightness

        # Cap brightness at 3%
        color = color * 0.03

        color = np.clip(color, 0, 255).astype(np.uint8)

        # Build frame (all LEDs same color for now)
        frame = np.tile(color, (self.num_leds, 1))

        return frame

    def send_frame(self, frame):
        """Send RGB frame to Arduino (matches nebula_stream protocol)"""
        if self.ser:
            try:
                # Build packet: [START1] [START2] [RGB data...]
                packet = bytearray([START_BYTE_1, START_BYTE_2])
                packet.extend(frame.flatten().tobytes())
                self.ser.write(packet)
                self.ser.flush()  # Ensure immediate transmission
            except Exception as e:
                # Don't let serial errors crash the timing loop
                pass

    def close(self):
        if self.ser:
            # Send black frame
            black = np.zeros((self.num_leds, 3), dtype=np.uint8)
            self.send_frame(black)
            self.ser.close()


def print_meter(flux, threshold, is_beat, brightness, beat_count):
    """Print a terminal beat meter."""
    bar_width = 50
    flux_bar = int(min(flux / max(threshold * 2, 1), 1.0) * bar_width)
    thresh_pos = int(min(threshold / max(flux * 2, threshold * 2, 1), 1.0) * bar_width)

    bar = list('.' * bar_width)
    for i in range(min(flux_bar, bar_width)):
        bar[i] = '|'
    if 0 <= thresh_pos < bar_width:
        bar[thresh_pos] = 'T'

    beat_indicator = ' BEAT!' if is_beat else ''
    bright_bar = '#' * int(brightness * 20)

    sys.stdout.write(
        f"\r  [{''.join(bar)}] "
        f"LED [{bright_bar:<20s}] "
        f"beats:{beat_count:4d}{beat_indicator}   "
    )
    sys.stdout.flush()


def main():
    global THRESHOLD_MULTIPLIER, MIN_BEAT_INTERVAL_SEC, DECAY_ALPHA

    parser = argparse.ArgumentParser(description='Real-time beat-reactive LEDs')
    parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
    parser.add_argument('--no-leds', action='store_true', help='Terminal visualization only')
    parser.add_argument('--leds', type=int, default=NUM_LEDS, help='Number of LEDs')
    parser.add_argument('--threshold', type=float, default=THRESHOLD_MULTIPLIER,
                        help='Beat threshold multiplier (higher = fewer beats)')
    parser.add_argument('--min-interval', type=float, default=MIN_BEAT_INTERVAL_SEC,
                        help='Minimum seconds between beats')
    parser.add_argument('--decay', type=float, default=DECAY_ALPHA,
                        help='LED decay rate (lower = slower fade, 0.05-0.3)')
    args = parser.parse_args()
    THRESHOLD_MULTIPLIER = args.threshold
    MIN_BEAT_INTERVAL_SEC = args.min_interval
    DECAY_ALPHA = args.decay

    # Find audio device
    device_id = find_blackhole_device()
    if device_id is None:
        print("Error: BlackHole audio device not found.")
        print("Available devices:")
        print(sd.query_devices())
        sys.exit(1)

    device_info = sd.query_devices(device_id)
    print(f"\n  Audio-Reactive Beat LED Controller")
    print(f"  {'='*40}")
    print(f"  Audio device: {device_info['name']} (#{device_id})")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chunk size: {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLE_RATE*1000:.0f}ms)")
    print(f"  Beat threshold: {THRESHOLD_MULTIPLIER}x std")
    print(f"  Min beat interval: {MIN_BEAT_INTERVAL_SEC}s ({60/MIN_BEAT_INTERVAL_SEC:.0f} BPM max)")
    print(f"  Decay rate: {DECAY_ALPHA}")

    # Find LED port
    serial_port = None
    if not args.no_leds:
        serial_port = args.port or find_serial_port()
        if serial_port is None:
            print("  No serial port found — running in terminal-only mode")
        else:
            print(f"  Serial port: {serial_port}")

    print()

    # Initialize
    detector = BeatDetector()
    leds = LEDController(num_leds=args.leds, serial_port=serial_port)

    # Audio callback
    audio_lock = threading.Lock()
    shared_state = {
        'is_beat': False,
        'flux': 0.0,
        'threshold': 0.0,
        'beat_intensity': 0.0
    }

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass  # Ignore overflow warnings
        # Mix to mono
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()

        is_beat, flux, threshold = detector.process_chunk(mono)

        # ONLY update shared state — no LED mutations from audio thread
        with audio_lock:
            shared_state['is_beat'] = is_beat
            shared_state['flux'] = flux
            shared_state['threshold'] = threshold
            if is_beat:
                shared_state['beat_intensity'] = 1.0

    # Start audio stream
    stream = sd.InputStream(
        device=device_id,
        channels=2,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )

    print("  Listening... Play music through BlackHole. Press Ctrl+C to stop.\n")

    try:
        stream.start()

        # Fixed-rate timing loop
        frame_interval = 1.0 / LED_FPS
        next_frame_time = time.time()

        while True:
            # Read audio analysis + beat trigger from shared state
            with audio_lock:
                is_beat = shared_state['is_beat']
                flux = shared_state['flux']
                threshold = shared_state['threshold']
                beat_intensity = shared_state['beat_intensity']
                shared_state['is_beat'] = False  # Reset beat flag
                shared_state['beat_intensity'] = 0.0

            # Trigger beat in main thread (safe LED state mutation)
            if beat_intensity > 0:
                leds.trigger_beat(beat_intensity)

            # Update LED state
            frame = leds.update(frame_interval)

            # Send to hardware at fixed rate
            leds.send_frame(frame)

            # Terminal display
            print_meter(flux, threshold, is_beat, leds.brightness, detector.beat_count)

            # Fixed-rate timing (absolute time targets to prevent drift)
            next_frame_time += frame_interval
            now = time.time()
            sleep_time = next_frame_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're behind — reset target to avoid backlog
                next_frame_time = now

    except KeyboardInterrupt:
        print("\n\n  Stopping...")
    finally:
        stream.stop()
        stream.close()
        leds.close()
        print(f"  Total beats detected: {detector.beat_count}")
        print("  Done!")


if __name__ == '__main__':
    main()
