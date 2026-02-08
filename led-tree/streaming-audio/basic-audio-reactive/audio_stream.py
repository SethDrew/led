#!/usr/bin/env python3
"""
Audio-Reactive LED Streaming

Captures audio from computer microphone, detects bass hits,
and streams animated frames to Arduino.

Background: Pulsating nebula green
Bass hits: Quick white flash

Usage:
    python audio_stream.py [serial_port]

Dependencies:
    pip install sounddevice numpy pyserial
"""

import numpy as np
import sounddevice as sd
import serial
import time
import sys
from collections import deque

# Configuration
NUM_LEDS = 197
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048  # Audio samples per chunk
FPS = 25           # Target frames per second

# Bass detection
BASS_LOW_FREQ = 20    # Hz
BASS_HIGH_FREQ = 250  # Hz
BASS_THRESHOLD = 2.0  # 2x average = significant bass hit

# Colors
NEBULA_GREEN = np.array([5, 40, 10])     # Very dark green
NEBULA_BRIGHT = np.array([15, 90, 20])   # Darker bright green
WHITE_FLASH = np.array([80, 180, 90])    # Greenish flash instead of white

# Serial protocol
START_BYTES = bytes([0xFF, 0xAA])

class AudioReactiveLEDs:
    def __init__(self, serial_port, baud_rate=1000000):
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize

        # Animation state
        self.pulse_phase = 0.0
        self.bass_flash = 0.0  # 0-1, decays over time
        self.frame_count = 0   # For smooth pulsation test

        # Bass detection
        self.bass_history = deque(maxlen=10)  # Track recent bass energy

        # Shared state between audio callback and main loop
        self.latest_bass_energy = 0.0

        print(f"Connected to {serial_port} at {baud_rate} baud")
        print(f"Streaming {NUM_LEDS} LEDs at ~{FPS} FPS")
        print(f"Bass detection: {BASS_LOW_FREQ}-{BASS_HIGH_FREQ} Hz")
        print("\nListening for audio... (Ctrl+C to stop)")

    def detect_bass(self, audio_chunk):
        """
        Detect bass energy in audio chunk using FFT
        Returns: bass energy (0.0 - 1.0+)
        """
        # Apply window to reduce spectral leakage
        windowed = audio_chunk * np.hanning(len(audio_chunk))

        # FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1/SAMPLE_RATE)
        magnitudes = np.abs(fft)

        # Extract bass range
        bass_mask = (freqs >= BASS_LOW_FREQ) & (freqs <= BASS_HIGH_FREQ)
        bass_energy_raw = magnitudes[bass_mask].mean()

        # Normalize against history
        if len(self.bass_history) > 0:
            avg_energy = np.mean(self.bass_history)
            if avg_energy > 0:
                bass_energy_normalized = bass_energy_raw / avg_energy
            else:
                bass_energy_normalized = 0
        else:
            bass_energy_normalized = 0

        # IMPORTANT: Append raw value to history, not normalized!
        self.bass_history.append(bass_energy_raw)

        return bass_energy_normalized

    def generate_frame(self, bass_energy):
        """
        Generate LED frame based on current state and bass energy
        Returns: numpy array of shape (NUM_LEDS, 3)
        """
        # Detect bass hit
        if bass_energy > BASS_THRESHOLD:
            self.bass_flash = min(1.0, self.bass_flash + 0.4)
            print(f"BASS! Energy: {bass_energy:.2f}")

        # Decay bass flash
        self.bass_flash *= 0.80

        # Black background
        base_color = np.array([0, 0, 0])

        # Green flash on bass
        flash_color = np.array([80, 180, 90])
        color = base_color + flash_color * self.bass_flash
        color = np.clip(color, 0, 255).astype(np.uint8)

        # Create frame (all LEDs same color for now)
        frame = np.tile(color, (NUM_LEDS, 1))

        return frame

    def send_frame(self, frame):
        """
        Send frame to Arduino via serial
        frame: numpy array of shape (NUM_LEDS, 3) with RGB values
        """
        # Flatten frame to bytes: [R0, G0, B0, R1, G1, B1, ...]
        frame_bytes = frame.flatten().tobytes()

        # Send: [START_BYTE_1] [START_BYTE_2] [frame_bytes]
        self.ser.write(START_BYTES + frame_bytes)

    def audio_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice for each audio chunk
        Just analyze audio, don't send frames (timing decoupling)
        """
        if status:
            print(f"Audio status: {status}")

        # Convert to mono
        audio_mono = indata[:, 0] if indata.ndim > 1 else indata

        # Detect bass and store it
        self.latest_bass_energy = self.detect_bass(audio_mono)

    def run(self):
        """
        Start audio stream and run until interrupted
        Fixed-rate frame sending (decoupled from audio callback timing)
        """
        print("Using device 5: BlackHole 2ch (System Audio)")
        print("Fixed-rate frame sending at {} FPS".format(FPS))

        frame_time = 1.0 / FPS

        try:
            with sd.InputStream(
                device=5,  # BlackHole 2ch
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback
            ):
                while True:
                    start = time.time()

                    # Read latest bass energy from audio callback
                    bass_energy = self.latest_bass_energy

                    # Generate and send frame at fixed rate
                    frame = self.generate_frame(bass_energy)
                    self.send_frame(frame)

                    # Maintain fixed frame rate
                    elapsed = time.time() - start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            # Clear LEDs on exit
            black_frame = np.zeros((NUM_LEDS, 3), dtype=np.uint8)
            self.send_frame(black_frame)
            self.ser.close()
            print("Disconnected")

def main():
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = "/dev/cu.usbserial-1240"

    print("="*60)
    print("Audio-Reactive LED Streaming")
    print("="*60)

    try:
        streamer = AudioReactiveLEDs(port)
        streamer.run()
    except serial.SerialException as e:
        print(f"\nError: Could not open serial port {port}")
        print(f"Details: {e}")
        print("\nAvailable ports:")
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}")
        sys.exit(1)

if __name__ == "__main__":
    main()
