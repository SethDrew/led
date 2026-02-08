#!/usr/bin/env python3
"""
Real-Time Onset Detection LED Controller with A/B Testing

Compares two beat detection algorithms:
1. BassFluxDetector — Bass-band spectral flux (20-250 Hz, detects kick drums)
2. OnsetDetector — Full-spectrum onset strength (detects all transients: kick, snare, hi-hat, synth)

This lets us visually compare which detector fires when, and validate against
electronic music where bass is continuous (sub-bass fails, onsets succeed).

Usage:
    python realtime_onset_led.py --detector bass     # Original bass flux (red)
    python realtime_onset_led.py --detector onset    # New onset detector (blue)
    python realtime_onset_led.py --detector both     # A/B test (red=bass, blue=onset, white=both)
    python realtime_onset_led.py --no-leds           # Terminal visualization only

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

# ── Bass Flux Detection Settings ────────────────────────────────────
BASS_LOW_HZ = 20
BASS_HIGH_HZ = 250
BASS_FLUX_HISTORY_SEC = 3.0
BASS_MIN_INTERVAL_SEC = 0.3
BASS_THRESHOLD_MULT = 1.5

# ── Onset Detection Settings ────────────────────────────────────────
ONSET_N_MELS = 40          # Mel bands for perceptual weighting
ONSET_FMIN = 20            # Low frequency cutoff
ONSET_FMAX = 8000          # High frequency cutoff (LED-relevant)
ONSET_HISTORY_SEC = 3.0    # Seconds of onset strength history
ONSET_MIN_INTERVAL_SEC = 0.1  # Faster minimum (can detect hi-hats)
ONSET_THRESHOLD_MULT = 2.0    # Higher threshold (more onsets to filter)

# ── LED Settings ────────────────────────────────────────────────────
NUM_LEDS = 197             # Tree: 197 nodes, Nebula strip: 150
BAUD_RATE = 1000000
START_BYTE_1 = 0xFF
START_BYTE_2 = 0xAA

# ── Visual Settings ─────────────────────────────────────────────────
ATTACK_ALPHA = 0.95        # Fast attack
DECAY_ALPHA = 0.12         # Slow decay
GAMMA = 2.2                # LED brightness correction
LED_FPS = 30               # Frame rate

# Colors (RGB, 0-255)
BASS_COLOR = np.array([255, 0, 0], dtype=np.float64)   # Red
ONSET_COLOR = np.array([0, 0, 255], dtype=np.float64)  # Blue
BOTH_COLOR = np.array([255, 255, 255], dtype=np.float64)  # White
BASE_COLOR = np.array([0, 0, 0], dtype=np.float64)     # Black


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


def mel_frequencies(n_mels, fmin, fmax, sample_rate):
    """Generate mel-spaced frequency bins."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    return mel_to_hz(mels)


def create_mel_filterbank(n_fft, n_mels, fmin, fmax, sample_rate):
    """Create triangular mel filterbank."""
    mel_freqs = mel_frequencies(n_mels, fmin, fmax, sample_rate)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    # Convert mel frequencies to FFT bin indices
    mel_bins = np.floor((n_fft + 1) * mel_freqs / sample_rate).astype(int)

    # Build triangular filters
    filterbank = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        left = mel_bins[i]
        center = mel_bins[i + 1]
        right = mel_bins[i + 2]

        # Rising slope
        if center > left:
            filterbank[i, left:center] = np.linspace(0, 1, center - left)

        # Falling slope
        if right > center:
            filterbank[i, center:right] = np.linspace(1, 0, right - center)

    return filterbank


class BassFluxDetector:
    """Bass-band spectral flux beat detection (original algorithm)."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT):
        self.sr = sample_rate
        self.n_fft = n_fft

        # Frequency bin indices for bass range
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self.bass_bins = np.where((freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ))[0]

        # State
        self.prev_spectrum = None
        self.flux_history = []
        self.max_history = int(BASS_FLUX_HISTORY_SEC * sample_rate / CHUNK_SIZE)
        self.last_beat_time = 0
        self.last_beat_frame = -999  # For offline processing
        self.beat_count = 0
        self.frame_count = 0

        # Windowing
        self.window = np.hanning(n_fft)

        # Audio buffer for overlapping FFT
        self.audio_buffer = np.zeros(n_fft)

    def process_chunk(self, audio_chunk, use_realtime=True):
        """Process an audio chunk. Returns (is_beat, strength, threshold)."""
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
            self.frame_count += 1
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
            self.frame_count += 1
            return False, flux, 0.0

        mean_flux = np.mean(self.flux_history)
        std_flux = np.std(self.flux_history)
        threshold = mean_flux + BASS_THRESHOLD_MULT * std_flux

        # Beat detection with minimum interval
        if use_realtime:
            now = time.time()
            is_beat = (flux > threshold and
                       (now - self.last_beat_time) > BASS_MIN_INTERVAL_SEC)
            if is_beat:
                self.last_beat_time = now
        else:
            # Offline mode: use frame counter
            min_frames = int(BASS_MIN_INTERVAL_SEC * self.sr / CHUNK_SIZE)
            is_beat = (flux > threshold and
                       (self.frame_count - self.last_beat_frame) > min_frames)
            if is_beat:
                self.last_beat_frame = self.frame_count

        if is_beat:
            self.beat_count += 1

        self.frame_count += 1
        return is_beat, flux, threshold


class OnsetDetector:
    """Full-spectrum onset strength detection (librosa-style algorithm)."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=ONSET_N_MELS):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels

        # Build mel filterbank
        self.mel_fb = create_mel_filterbank(n_fft, n_mels, ONSET_FMIN, ONSET_FMAX, sample_rate)

        # State
        self.prev_mel_spectrum = None
        self.onset_history = []
        self.max_history = int(ONSET_HISTORY_SEC * sample_rate / CHUNK_SIZE)
        self.last_beat_time = 0
        self.last_beat_frame = -999  # For offline processing
        self.beat_count = 0
        self.frame_count = 0

        # Windowing
        self.window = np.hanning(n_fft)

        # Audio buffer for overlapping FFT
        self.audio_buffer = np.zeros(n_fft)

    def process_chunk(self, audio_chunk, use_realtime=True):
        """Process an audio chunk. Returns (is_beat, strength, threshold)."""
        # Shift buffer and add new audio
        chunk_len = len(audio_chunk)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = audio_chunk

        # Windowed FFT
        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Apply mel filterbank (weighted sum across frequency bins)
        mel_spectrum = np.dot(self.mel_fb, spectrum)

        # Log compression for perceptual scaling
        mel_spectrum = np.log1p(mel_spectrum)

        if self.prev_mel_spectrum is None:
            self.prev_mel_spectrum = mel_spectrum
            self.frame_count += 1
            return False, 0.0, 0.0

        # Onset strength: sum of positive spectral differences across ALL mel bands
        diff = mel_spectrum - self.prev_mel_spectrum
        onset_strength = np.sum(np.maximum(diff, 0))
        self.prev_mel_spectrum = mel_spectrum

        # Adaptive threshold
        self.onset_history.append(onset_strength)
        if len(self.onset_history) > self.max_history:
            self.onset_history.pop(0)

        if len(self.onset_history) < 10:
            self.frame_count += 1
            return False, onset_strength, 0.0

        mean_onset = np.mean(self.onset_history)
        std_onset = np.std(self.onset_history)
        threshold = mean_onset + ONSET_THRESHOLD_MULT * std_onset

        # Beat detection with minimum interval
        if use_realtime:
            now = time.time()
            is_beat = (onset_strength > threshold and
                       (now - self.last_beat_time) > ONSET_MIN_INTERVAL_SEC)
            if is_beat:
                self.last_beat_time = now
        else:
            # Offline mode: use frame counter
            min_frames = int(ONSET_MIN_INTERVAL_SEC * self.sr / CHUNK_SIZE)
            is_beat = (onset_strength > threshold and
                       (self.frame_count - self.last_beat_frame) > min_frames)
            if is_beat:
                self.last_beat_frame = self.frame_count

        if is_beat:
            self.beat_count += 1

        self.frame_count += 1
        return is_beat, onset_strength, threshold


class LEDController:
    """Manages LED state with pulse/decay effect (supports multi-detector mode)."""

    def __init__(self, num_leds=NUM_LEDS, serial_port=None, mode='bass'):
        self.num_leds = num_leds
        self.mode = mode  # 'bass', 'onset', or 'both'

        # Separate brightness channels for A/B testing
        self.bass_brightness = 0.0
        self.onset_brightness = 0.0

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

    def trigger_bass_beat(self, intensity=1.0):
        """Trigger a bass beat flash."""
        self.bass_brightness = max(self.bass_brightness, intensity)

    def trigger_onset_beat(self, intensity=1.0):
        """Trigger an onset beat flash."""
        self.onset_brightness = max(self.onset_brightness, intensity)

    def update(self, dt):
        """Update brightness with exponential decay. Returns current frame."""
        # Decay both channels
        decay = DECAY_ALPHA ** (dt * LED_FPS)
        self.bass_brightness *= (1.0 - decay)
        self.onset_brightness *= (1.0 - decay)
        self.bass_brightness = max(self.bass_brightness, 0.0)
        self.onset_brightness = max(self.onset_brightness, 0.0)

        # Gamma correction
        bass_gamma = self.bass_brightness ** (1.0 / GAMMA)
        onset_gamma = self.onset_brightness ** (1.0 / GAMMA)

        # Blend colors based on mode
        if self.mode == 'bass':
            color = BASE_COLOR + (BASS_COLOR - BASE_COLOR) * bass_gamma
        elif self.mode == 'onset':
            color = BASE_COLOR + (ONSET_COLOR - BASE_COLOR) * onset_gamma
        else:  # both
            # If both fire, show white; otherwise red or blue
            if bass_gamma > 0.05 and onset_gamma > 0.05:
                # Both active — blend to white
                both_brightness = max(bass_gamma, onset_gamma)
                color = BASE_COLOR + (BOTH_COLOR - BASE_COLOR) * both_brightness
            else:
                # Only one active — show respective color
                bass_component = BASS_COLOR * bass_gamma
                onset_component = ONSET_COLOR * onset_gamma
                color = BASE_COLOR + bass_component + onset_component

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
                pass  # Don't let serial errors crash timing loop

    def close(self):
        if self.ser:
            # Send black frame
            black = np.zeros((self.num_leds, 3), dtype=np.uint8)
            self.send_frame(black)
            self.ser.close()


def print_meter(mode, bass_val, bass_thresh, bass_beat, onset_val, onset_thresh, onset_beat,
                bass_brightness, onset_brightness, bass_count, onset_count):
    """Print a terminal beat meter."""
    bar_width = 40

    # Bass meter
    bass_bar_val = int(min(bass_val / max(bass_thresh * 2, 1), 1.0) * bar_width)
    bass_thresh_pos = int(min(bass_thresh / max(bass_val * 2, bass_thresh * 2, 1), 1.0) * bar_width)
    bass_bar = list('.' * bar_width)
    for i in range(min(bass_bar_val, bar_width)):
        bass_bar[i] = '|'
    if 0 <= bass_thresh_pos < bar_width:
        bass_bar[bass_thresh_pos] = 'T'
    bass_indicator = ' BEAT!' if bass_beat else ''

    # Onset meter
    onset_bar_val = int(min(onset_val / max(onset_thresh * 2, 1), 1.0) * bar_width)
    onset_thresh_pos = int(min(onset_thresh / max(onset_val * 2, onset_thresh * 2, 1), 1.0) * bar_width)
    onset_bar = list('.' * bar_width)
    for i in range(min(onset_bar_val, bar_width)):
        onset_bar[i] = '|'
    if 0 <= onset_thresh_pos < bar_width:
        onset_bar[onset_thresh_pos] = 'T'
    onset_indicator = ' BEAT!' if onset_beat else ''

    # Brightness bars
    bass_bright_bar = '#' * int(bass_brightness * 15)
    onset_bright_bar = '#' * int(onset_brightness * 15)

    if mode == 'bass':
        sys.stdout.write(
            f"\r  BASS: [{''.join(bass_bar)}] "
            f"LED [{bass_bright_bar:<15s}] "
            f"beats:{bass_count:4d}{bass_indicator}   "
        )
    elif mode == 'onset':
        sys.stdout.write(
            f"\r  ONSET: [{''.join(onset_bar)}] "
            f"LED [{onset_bright_bar:<15s}] "
            f"beats:{onset_count:4d}{onset_indicator}   "
        )
    else:  # both
        sys.stdout.write(
            f"\r  BASS: [{''.join(bass_bar)}] {bass_count:3d}{bass_indicator:6s} | "
            f"ONSET: [{''.join(onset_bar)}] {onset_count:3d}{onset_indicator:6s}   "
        )

    sys.stdout.flush()


def main():
    global BASS_THRESHOLD_MULT, ONSET_THRESHOLD_MULT, BASS_MIN_INTERVAL_SEC, ONSET_MIN_INTERVAL_SEC, DECAY_ALPHA

    parser = argparse.ArgumentParser(description='Real-time onset detection LED controller')
    parser.add_argument('--detector', choices=['bass', 'onset', 'both'], default='both',
                        help='Detector mode: bass (red), onset (blue), or both (A/B test)')
    parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
    parser.add_argument('--no-leds', action='store_true', help='Terminal visualization only')
    parser.add_argument('--leds', type=int, default=NUM_LEDS, help='Number of LEDs')
    parser.add_argument('--bass-threshold', type=float, default=BASS_THRESHOLD_MULT,
                        help='Bass threshold multiplier')
    parser.add_argument('--onset-threshold', type=float, default=ONSET_THRESHOLD_MULT,
                        help='Onset threshold multiplier')
    parser.add_argument('--decay', type=float, default=DECAY_ALPHA,
                        help='LED decay rate (0.05-0.3)')
    args = parser.parse_args()

    BASS_THRESHOLD_MULT = args.bass_threshold
    ONSET_THRESHOLD_MULT = args.onset_threshold
    DECAY_ALPHA = args.decay

    # Find audio device
    device_id = find_blackhole_device()
    if device_id is None:
        print("Error: BlackHole audio device not found.")
        print("Available devices:")
        print(sd.query_devices())
        sys.exit(1)

    device_info = sd.query_devices(device_id)
    print(f"\n  Onset Detection LED Controller (A/B Test Mode)")
    print(f"  {'='*50}")
    print(f"  Audio device: {device_info['name']} (#{device_id})")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Detector mode: {args.detector.upper()}")

    if args.detector in ['bass', 'both']:
        print(f"  Bass flux: {BASS_LOW_HZ}-{BASS_HIGH_HZ} Hz, threshold={BASS_THRESHOLD_MULT}x, min_interval={BASS_MIN_INTERVAL_SEC}s")
    if args.detector in ['onset', 'both']:
        print(f"  Onset: {ONSET_FMIN}-{ONSET_FMAX} Hz, {ONSET_N_MELS} mel bands, threshold={ONSET_THRESHOLD_MULT}x, min_interval={ONSET_MIN_INTERVAL_SEC}s")

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

    # Initialize detectors based on mode
    bass_detector = BassFluxDetector() if args.detector in ['bass', 'both'] else None
    onset_detector = OnsetDetector() if args.detector in ['onset', 'both'] else None

    leds = LEDController(num_leds=args.leds, serial_port=serial_port, mode=args.detector)

    # Audio callback
    audio_lock = threading.Lock()
    shared_state = {
        'bass_beat': False,
        'bass_val': 0.0,
        'bass_thresh': 0.0,
        'onset_beat': False,
        'onset_val': 0.0,
        'onset_thresh': 0.0,
        'bass_intensity': 0.0,
        'onset_intensity': 0.0
    }

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass  # Ignore overflow warnings

        # Mix to mono
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()

        # Process with active detectors
        if bass_detector:
            bass_beat, bass_val, bass_thresh = bass_detector.process_chunk(mono)
        else:
            bass_beat, bass_val, bass_thresh = False, 0.0, 0.0

        if onset_detector:
            onset_beat, onset_val, onset_thresh = onset_detector.process_chunk(mono)
        else:
            onset_beat, onset_val, onset_thresh = False, 0.0, 0.0

        # ONLY update shared state — no LED mutations from audio thread
        with audio_lock:
            shared_state['bass_beat'] = bass_beat
            shared_state['bass_val'] = bass_val
            shared_state['bass_thresh'] = bass_thresh
            shared_state['onset_beat'] = onset_beat
            shared_state['onset_val'] = onset_val
            shared_state['onset_thresh'] = onset_thresh

            if bass_beat:
                shared_state['bass_intensity'] = 1.0
            if onset_beat:
                shared_state['onset_intensity'] = 1.0

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
            # Read audio analysis + beat triggers from shared state
            with audio_lock:
                bass_beat = shared_state['bass_beat']
                bass_val = shared_state['bass_val']
                bass_thresh = shared_state['bass_thresh']
                onset_beat = shared_state['onset_beat']
                onset_val = shared_state['onset_val']
                onset_thresh = shared_state['onset_thresh']
                bass_intensity = shared_state['bass_intensity']
                onset_intensity = shared_state['onset_intensity']

                # Reset beat flags
                shared_state['bass_beat'] = False
                shared_state['onset_beat'] = False
                shared_state['bass_intensity'] = 0.0
                shared_state['onset_intensity'] = 0.0

            # Trigger beats in main thread (safe LED state mutation)
            if bass_intensity > 0:
                leds.trigger_bass_beat(bass_intensity)
            if onset_intensity > 0:
                leds.trigger_onset_beat(onset_intensity)

            # Update LED state
            frame = leds.update(frame_interval)

            # Send to hardware at fixed rate
            leds.send_frame(frame)

            # Terminal display
            bass_count = bass_detector.beat_count if bass_detector else 0
            onset_count = onset_detector.beat_count if onset_detector else 0
            print_meter(args.detector, bass_val, bass_thresh, bass_beat,
                        onset_val, onset_thresh, onset_beat,
                        leds.bass_brightness, leds.onset_brightness,
                        bass_count, onset_count)

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

        if bass_detector:
            print(f"  Bass beats detected: {bass_detector.beat_count}")
        if onset_detector:
            print(f"  Onset beats detected: {onset_detector.beat_count}")
        print("  Done!")


if __name__ == '__main__':
    main()
