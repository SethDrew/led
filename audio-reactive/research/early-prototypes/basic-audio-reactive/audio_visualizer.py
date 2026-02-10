#!/usr/bin/env python3
"""
Real-time Audio Visualization

Displays live plots of:
- Total audio level (RMS)
- Bass energy (20-250 Hz)
- Mid energy (250-2000 Hz)
- High energy (2000-8000 Hz)
- Bass threshold line

Usage:
    python audio_visualizer.py

Dependencies:
    pip install sounddevice numpy matplotlib
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
HISTORY_LENGTH = 200  # Number of data points to show

# Frequency ranges
BASS_LOW = 20
BASS_HIGH = 250
MID_LOW = 250
MID_HIGH = 2000
HIGH_LOW = 2000
HIGH_HIGH = 8000

# Bass detection threshold (from audio_stream.py)
BASS_THRESHOLD = 2.0  # 2x average = significant bass hit

class AudioVisualizer:
    def __init__(self):
        # Data buffers
        self.time_data = deque(maxlen=HISTORY_LENGTH)
        self.total_level = deque(maxlen=HISTORY_LENGTH)
        self.bass_energy = deque(maxlen=HISTORY_LENGTH)
        self.mid_energy = deque(maxlen=HISTORY_LENGTH)
        self.high_energy = deque(maxlen=HISTORY_LENGTH)
        self.bass_detected = deque(maxlen=HISTORY_LENGTH)

        self.bass_history = deque(maxlen=10)  # For normalization
        self.frame_count = 0
        self.detection_hold = 0  # Hold detection display for visibility

        # Setup plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Real-time Audio Analysis', fontsize=16)

        # Top plot: Dual Y-axis for RMS (left) and Energy (right)

        # Left Y-axis: RMS Total Level
        self.line_total, = self.ax1.plot([], [], 'w-', label='Total RMS', linewidth=2, alpha=0.7)
        self.ax1.set_xlim(0, HISTORY_LENGTH)
        self.ax1.set_ylim(0, 100)
        self.ax1.set_ylabel('RMS Level (amplitude)', color='white')
        self.ax1.tick_params(axis='y', labelcolor='white')
        self.ax1.set_title('Audio Analysis - Dual Axis')
        self.ax1.grid(True, alpha=0.3)

        # Right Y-axis: Normalized energy bands
        self.ax1_right = self.ax1.twinx()
        self.line_bass, = self.ax1_right.plot([], [], 'r-', label='Bass Energy', linewidth=2)
        self.line_mid, = self.ax1_right.plot([], [], 'g-', label='Mid Energy', linewidth=1)
        self.line_high, = self.ax1_right.plot([], [], 'b-', label='High Energy', linewidth=1)

        self.ax1_right.set_ylim(0, 5)
        self.ax1_right.set_ylabel('Normalized Energy (ratio to avg)', color='red')
        self.ax1_right.tick_params(axis='y', labelcolor='red')

        # Threshold line on right axis
        self.threshold_line = self.ax1_right.axhline(y=BASS_THRESHOLD, color='yellow',
                                                      linestyle='--', linewidth=2,
                                                      label=f'Threshold ({BASS_THRESHOLD})')

        # Combined legend
        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax1_right.get_legend_handles_labels()
        self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Bottom plot: Bass detection events
        self.line_detect, = self.ax2.plot([], [], 'r-', linewidth=3)
        self.fill_detect = self.ax2.fill_between([], [], 0, alpha=0.5, color='red')
        self.ax2.set_xlim(0, HISTORY_LENGTH)
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlabel('Time (frames)')
        self.ax2.set_ylabel('Bass Detected')
        self.ax2.set_title('Bass Detection Events')
        self.ax2.grid(True, alpha=0.3)

        # Text for current values
        self.text = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
                                  verticalalignment='top', fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        plt.tight_layout()

        print("Audio Visualizer Starting...")
        print(f"Sample Rate: {SAMPLE_RATE} Hz")
        print(f"Chunk Size: {CHUNK_SIZE} samples")
        print(f"Bass Range: {BASS_LOW}-{BASS_HIGH} Hz")
        print(f"Bass Threshold: {BASS_THRESHOLD}")
        print("\nClose the plot window to stop.")

    def analyze_audio(self, audio_chunk):
        """Analyze audio chunk and extract features"""
        # Total level (RMS)
        total = np.sqrt(np.mean(audio_chunk ** 2))

        # Apply window
        windowed = audio_chunk * np.hanning(len(audio_chunk))

        # FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1/SAMPLE_RATE)
        magnitudes = np.abs(fft)

        # Bass energy (raw)
        bass_mask = (freqs >= BASS_LOW) & (freqs <= BASS_HIGH)
        bass_raw = magnitudes[bass_mask].mean() if bass_mask.any() else 0

        # Mid energy
        mid_mask = (freqs >= MID_LOW) & (freqs <= MID_HIGH)
        mid = magnitudes[mid_mask].mean() if mid_mask.any() else 0

        # High energy
        high_mask = (freqs >= HIGH_LOW) & (freqs <= HIGH_HIGH)
        high = magnitudes[high_mask].mean() if high_mask.any() else 0

        # Normalize bass against history (SAME ORDER as audio_stream.py)
        if len(self.bass_history) > 0:
            avg_bass = np.mean(self.bass_history)
            if avg_bass > 0:
                bass_normalized = bass_raw / avg_bass
            else:
                bass_normalized = 0
        else:
            bass_normalized = 0

        # IMPORTANT: Append raw value AFTER normalization
        self.bass_history.append(bass_raw)

        # Normalize mid and high similar to bass for better visualization
        self.mid_history = getattr(self, 'mid_history', deque(maxlen=10))
        self.high_history = getattr(self, 'high_history', deque(maxlen=10))

        if len(self.mid_history) > 0 and np.mean(self.mid_history) > 0:
            mid_normalized = mid / np.mean(self.mid_history)
        else:
            mid_normalized = 0
        self.mid_history.append(mid)

        if len(self.high_history) > 0 and np.mean(self.high_history) > 0:
            high_normalized = high / np.mean(self.high_history)
        else:
            high_normalized = 0
        self.high_history.append(high)

        # Detect bass hit (returns 1.0 if above threshold, 0.0 otherwise)
        detected = 1.0 if bass_normalized > BASS_THRESHOLD else 0.0

        # Return normalized values for all bands
        return total * 100, bass_normalized, mid_normalized, high_normalized, detected

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk"""
        if status:
            print(f"Audio status: {status}")

        # Convert to mono
        audio_mono = indata[:, 0] if indata.ndim > 1 else indata

        # Analyze
        total, bass, mid, high, detected = self.analyze_audio(audio_mono)

        # Debug: Print values every 20 frames
        if self.frame_count % 20 == 0:
            print(f"RMS: {total:.1f} | Bass: {bass:.2f} | Mid: {mid:.2f} | High: {high:.2f}")

        # Make detection "stick" for visibility (hold for 3 frames = ~150ms)
        if detected > 0:
            self.detection_hold = 3
            print(f"🎵 BASS DETECTED! Energy: {bass:.2f}")

        display_detected = 1.0 if self.detection_hold > 0 else 0.0
        if self.detection_hold > 0:
            self.detection_hold -= 1

        # Update buffers
        self.time_data.append(self.frame_count)
        self.total_level.append(total)
        self.bass_energy.append(bass)
        self.mid_energy.append(mid)
        self.high_energy.append(high)
        self.bass_detected.append(display_detected)  # Use held value for display

        self.frame_count += 1

    def update_plot(self, frame):
        """Update plot with latest data"""
        if len(self.time_data) < 2:
            return self.line_total, self.line_bass, self.line_mid, self.line_high, self.line_detect, self.fill_detect

        x = list(range(len(self.time_data)))

        # Update lines (total on left axis, others on right axis)
        self.line_total.set_data(x, list(self.total_level))
        self.line_bass.set_data(x, list(self.bass_energy))
        self.line_mid.set_data(x, list(self.mid_energy))
        self.line_high.set_data(x, list(self.high_energy))

        # Update bass detection
        detect_list = list(self.bass_detected)
        self.line_detect.set_data(x, detect_list)

        # Update fill
        self.fill_detect.remove()
        self.fill_detect = self.ax2.fill_between(x, detect_list, 0, alpha=0.5, color='red')

        # Update text with current values
        if len(self.bass_energy) > 0:
            current_bass = self.bass_energy[-1]
            current_total = self.total_level[-1]
            status = "DETECTED!" if self.bass_detected[-1] > 0 else "---"

            self.text.set_text(
                f"Total: {current_total:.2f}\n"
                f"Bass: {current_bass:.2f}\n"
                f"Status: {status}"
            )

        # Auto-scale Y axes independently
        if len(self.total_level) > 0:
            max_rms = max(self.total_level)
            self.ax1.set_ylim(0, max(100, max_rms * 1.1))

        if len(self.bass_energy) > 0:
            max_energy = max(max(self.bass_energy), BASS_THRESHOLD * 1.2)
            self.ax1_right.set_ylim(0, max(5, max_energy * 1.1))

        return self.line_total, self.line_bass, self.line_mid, self.line_high, self.line_detect, self.fill_detect

    def run(self):
        """Start audio stream and visualization"""
        # Use BlackHole for system audio capture (loopback)
        print("\nUsing device 5: BlackHole 2ch (System Audio)")

        # Start audio stream
        stream = sd.InputStream(
            device=5,  # BlackHole 2ch
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=self.audio_callback
        )

        with stream:
            # Start animation (20ms interval = 50 FPS for better responsiveness)
            ani = FuncAnimation(self.fig, self.update_plot, interval=20, blit=True)
            plt.show()

def main():
    visualizer = AudioVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
