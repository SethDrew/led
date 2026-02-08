#!/usr/bin/env python3
"""
Spectrum Audio Reactive LED Streaming
Inspired by scottlawsonbc/audio-reactive-led-strip

Multiple visualization modes:
- spectrum: Frequency spectrum visualization
- energy: Bass energy reactive
- scroll: Scrolling spectrum effect

Usage:
    python spectrum_stream.py [mode]
    Modes: spectrum, energy, scroll (default: energy)
"""

import numpy as np
import sounddevice as sd
import serial
import time
import sys
from scipy.ndimage import gaussian_filter1d

# Configuration
NUM_LEDS = 197
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
FPS = 60  # Higher FPS for smoother animations

# FFT Configuration
MIN_FREQUENCY = 20
MAX_FREQUENCY = 8000
N_FFT_BINS = 24  # Number of frequency bins

# Energy mode threshold (0.0-1.0)
# Only trigger LEDs when bass energy exceeds this value
# 0.80 = top ~10% of bass hits (conservative)
# 0.85 = top ~5% of bass hits (moderate)
BASS_THRESHOLD = 0.80

# Serial protocol
START_BYTES = bytes([0xFF, 0xAA])

class SpectrumStreamer:
    def __init__(self, serial_port, mode='energy', baud_rate=1000000):
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)

        self.mode = mode
        self.frame_count = 0

        # FFT setup
        self.fft_bins = np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, N_FFT_BINS)
        self.freq_bin_edges = self._create_mel_bins(N_FFT_BINS)

        # State for smoothing
        self.prev_spectrum = np.zeros(N_FFT_BINS)
        self.prev_energy = 0.0

        # Shared state between callback and main loop
        self.latest_spectrum = np.zeros(N_FFT_BINS)
        self.latest_bass_energy = 0.0

        # Debug: Track bass energy statistics
        self.bass_history = []
        self.bass_min = float('inf')
        self.bass_max = 0.0
        self.trigger_count = 0

        print(f"Connected to {serial_port}")
        print(f"Mode: {mode}")
        print(f"Streaming {NUM_LEDS} LEDs at {FPS} FPS")
        if mode == 'energy':
            print(f"Bass Threshold: {BASS_THRESHOLD:.2f} (LEDs trigger above this)")
        print("Press Ctrl+C to stop\n")

    def _create_mel_bins(self, n_bins):
        """Create mel-scale frequency bins for perceptually uniform distribution"""
        # Mel scale: more bins in bass, fewer in treble (matches human hearing)
        mel_min = self._hz_to_mel(MIN_FREQUENCY)
        mel_max = self._hz_to_mel(MAX_FREQUENCY)
        mels = np.linspace(mel_min, mel_max, n_bins + 1)
        return self._mel_to_hz(mels)

    def _hz_to_mel(self, hz):
        """Convert Hz to mel scale"""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        """Convert mel scale to Hz"""
        return 700 * (10**(mel / 2595) - 1)

    def analyze_audio(self, audio_chunk):
        """
        Analyze audio and extract frequency spectrum
        Returns: (spectrum, bass_energy)
        """
        # Apply Hann window
        windowed = audio_chunk * np.hanning(len(audio_chunk))

        # Compute FFT
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), 1/SAMPLE_RATE)

        # Bin frequencies into mel-scale bins
        spectrum = np.zeros(N_FFT_BINS)
        for i in range(N_FFT_BINS):
            freq_mask = (freqs >= self.freq_bin_edges[i]) & (freqs < self.freq_bin_edges[i+1])
            spectrum[i] = np.mean(fft[freq_mask]) if freq_mask.any() else 0

        # Normalize spectrum
        spectrum = spectrum / (np.max(spectrum) + 1e-6)

        # Bass energy (first 3 bins)
        bass_energy = np.mean(spectrum[:3])

        return spectrum, bass_energy

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback - just analyze, don't send frames"""
        if status:
            print(f"Audio: {status}")

        audio_mono = indata[:, 0] if indata.ndim > 1 else indata
        spectrum, bass_energy = self.analyze_audio(audio_mono)

        # Store for main loop
        self.latest_spectrum = spectrum
        self.latest_bass_energy = bass_energy

    def generate_spectrum_frame(self, spectrum):
        """Generate spectrum analyzer frame"""
        # Map spectrum to LED positions
        led_colors = np.zeros((NUM_LEDS, 3))

        # Distribute spectrum across LEDs
        for i, led_idx in enumerate(np.linspace(0, NUM_LEDS-1, N_FFT_BINS, dtype=int)):
            intensity = spectrum[i]

            # Color based on frequency (bass=red, mid=green, high=blue)
            freq_ratio = i / N_FFT_BINS
            if freq_ratio < 0.33:  # Bass
                color = np.array([255, 50, 0]) * intensity
            elif freq_ratio < 0.66:  # Mid
                color = np.array([0, 255, 50]) * intensity
            else:  # High
                color = np.array([0, 50, 255]) * intensity

            # Apply to LED with some spread
            spread = 3
            for j in range(max(0, led_idx-spread), min(NUM_LEDS, led_idx+spread)):
                led_colors[j] = np.maximum(led_colors[j], color * (1 - abs(j-led_idx)/spread))

        return led_colors.astype(np.uint8)

    def generate_energy_frame(self, bass_energy):
        """Generate bass energy reactive frame (simple pulse)"""
        # Apply threshold - only respond to bass above threshold
        if bass_energy > BASS_THRESHOLD:
            # Map energy above threshold to 0-1 range
            # threshold -> 0.0, max (1.0) -> 1.0
            triggered_energy = (bass_energy - BASS_THRESHOLD) / (1.0 - BASS_THRESHOLD)
        else:
            triggered_energy = 0.0

        # Smooth energy transitions
        self.prev_energy = self.prev_energy * 0.5 + triggered_energy * 0.5

        # Create gradient from bottom (bright) to top (dim) based on energy
        led_colors = np.zeros((NUM_LEDS, 3))

        if self.prev_energy > 0.01:  # Only light up if energy is significant
            for i in range(NUM_LEDS):
                # Height-based intensity (bottom is brightest)
                height_ratio = 1.0 - (i / NUM_LEDS)
                intensity = self.prev_energy * height_ratio

                # Green color
                color = np.array([0, 255, 100]) * intensity
                led_colors[i] = color

        return np.clip(led_colors, 0, 255).astype(np.uint8)

    def generate_scroll_frame(self, spectrum):
        """Generate scrolling spectrum effect"""
        # TODO: Implement scrolling buffer
        return self.generate_spectrum_frame(spectrum)

    def generate_frame(self):
        """Generate frame based on current mode"""
        # Smooth spectrum
        self.prev_spectrum = self.prev_spectrum * 0.2 + self.latest_spectrum * 0.8

        if self.mode == 'spectrum':
            return self.generate_spectrum_frame(self.prev_spectrum)
        elif self.mode == 'energy':
            return self.generate_energy_frame(self.latest_bass_energy)
        elif self.mode == 'scroll':
            return self.generate_scroll_frame(self.prev_spectrum)
        else:
            return np.zeros((NUM_LEDS, 3), dtype=np.uint8)

    def send_frame(self, frame):
        """Send frame to Arduino"""
        frame_bytes = frame.flatten().tobytes()
        self.ser.write(START_BYTES + frame_bytes)

    def run(self):
        """Main loop with fixed-rate frame sending"""
        frame_time = 1.0 / FPS

        try:
            with sd.InputStream(
                device=5,  # BlackHole 2ch
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback
            ):
                print("Streaming started...\n")
                while True:
                    start = time.time()

                    # Track bass energy statistics
                    bass = self.latest_bass_energy
                    self.bass_history.append(bass)
                    self.bass_min = min(self.bass_min, bass)
                    self.bass_max = max(self.bass_max, bass)

                    # Track triggers (for energy mode)
                    triggered = bass > BASS_THRESHOLD if self.mode == 'energy' else False
                    if triggered:
                        self.trigger_count += 1

                    # Generate and send frame
                    frame = self.generate_frame()
                    self.send_frame(frame)
                    self.frame_count += 1

                    # Debug: Print when bass triggers above threshold
                    if self.mode == 'energy' and triggered:
                        print(f"🎵 BASS HIT! Frame {self.frame_count} | Energy: {bass:.4f} | Smoothed: {self.prev_energy:.4f}")
                    elif self.frame_count % FPS == 0:
                        avg = np.mean(self.bass_history[-FPS*10:])  # Average of last 10 seconds
                        trigger_rate = (self.trigger_count / self.frame_count * 100) if self.frame_count > 0 else 0
                        print(f"Frames: {self.frame_count} | Bass: {bass:.4f} | Avg: {avg:.4f} | Triggers: {trigger_rate:.1f}%")

                    # Maintain frame rate
                    elapsed = time.time() - start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            if self.mode == 'energy':
                print("\n=== ENERGY MODE STATISTICS ===")
                if self.bass_history:
                    arr = np.array(self.bass_history)
                    trigger_rate = (self.trigger_count / len(self.bass_history) * 100)
                    print(f"Total frames: {len(self.bass_history)}")
                    print(f"Bass hits (>{BASS_THRESHOLD:.2f}): {self.trigger_count} ({trigger_rate:.1f}%)")
                    print(f"\nBass energy range:")
                    print(f"  Min: {self.bass_min:.4f}")
                    print(f"  Max: {self.bass_max:.4f}")
                    print(f"  Mean: {np.mean(arr):.4f}")
                    print(f"  Median: {np.median(arr):.4f}")
                    print(f"\nTo adjust sensitivity, edit BASS_THRESHOLD in the code:")
                    print(f"  Lower (0.70-0.75): More sensitive, triggers more often")
                    print(f"  Current: {BASS_THRESHOLD:.2f}")
                    print(f"  Higher (0.85-0.90): Less sensitive, only strong bass")
        finally:
            # Clear LEDs
            black = np.zeros((NUM_LEDS, 3), dtype=np.uint8)
            self.send_frame(black)
            self.ser.close()
            print("Disconnected")

def main():
    port = "/dev/cu.usbserial-1240"
    mode = sys.argv[1] if len(sys.argv) > 1 else 'energy'

    if mode not in ['spectrum', 'energy', 'scroll']:
        print(f"Unknown mode: {mode}")
        print("Available modes: spectrum, energy, scroll")
        sys.exit(1)

    print("="*60)
    print("Spectrum Audio Reactive LED Streaming")
    print("Inspired by scottlawsonbc/audio-reactive-led-strip")
    print("="*60)

    try:
        streamer = SpectrumStreamer(port, mode=mode)
        streamer.run()
    except serial.SerialException as e:
        print(f"\nError: Could not open {port}")
        print(f"Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
