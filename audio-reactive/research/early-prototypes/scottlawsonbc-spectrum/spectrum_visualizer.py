#!/usr/bin/env python3
"""
Spectrum Visualizer

Real-time visualization for spectrum_stream.py
Shows frequency spectrum analysis and energy levels

Usage:
    python spectrum_visualizer.py
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
HISTORY_LENGTH = 100

# FFT Configuration (match spectrum_stream.py)
MIN_FREQUENCY = 20
MAX_FREQUENCY = 8000
N_FFT_BINS = 24

class SpectrumVisualizer:
    def __init__(self):
        # Data buffers
        self.time_data = deque(maxlen=HISTORY_LENGTH)
        self.spectrum_history = deque(maxlen=HISTORY_LENGTH)
        self.bass_energy_history = deque(maxlen=HISTORY_LENGTH)
        self.mid_energy_history = deque(maxlen=HISTORY_LENGTH)
        self.high_energy_history = deque(maxlen=HISTORY_LENGTH)

        self.frame_count = 0

        # FFT setup
        self.freq_bin_edges = self._create_mel_bins(N_FFT_BINS)
        self.freq_labels = [f"{int(f)}Hz" for f in self.freq_bin_edges[:-1]]

        # Setup plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.suptitle('Spectrum Audio Visualizer', fontsize=16)

        # Plot 1: Current frequency spectrum (bar chart)
        self.bars = self.ax1.bar(range(N_FFT_BINS), np.zeros(N_FFT_BINS),
                                  color=['red']*8 + ['green']*8 + ['blue']*8,
                                  alpha=0.7, edgecolor='white', linewidth=0.5)
        self.ax1.set_xlim(-0.5, N_FFT_BINS-0.5)
        self.ax1.set_ylim(0, 1.0)
        self.ax1.set_ylabel('Intensity')
        self.ax1.set_title('Current Frequency Spectrum (24 Mel-Scale Bins)')
        self.ax1.set_xticks([0, 8, 16, 23])
        self.ax1.set_xticklabels(['Bass', 'Mid', 'High', ''])
        self.ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Energy bands over time
        self.line_bass, = self.ax2.plot([], [], 'r-', label='Bass Energy', linewidth=2)
        self.line_mid, = self.ax2.plot([], [], 'g-', label='Mid Energy', linewidth=2)
        self.line_high, = self.ax2.plot([], [], 'b-', label='High Energy', linewidth=2)

        self.ax2.set_xlim(0, HISTORY_LENGTH)
        self.ax2.set_ylim(0, 1.0)
        self.ax2.set_ylabel('Energy Level')
        self.ax2.set_xlabel('Time (frames)')
        self.ax2.set_title('Energy Bands Over Time')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)

        # Plot 3: Spectrum waterfall (spectrogram)
        self.spectrogram_data = np.zeros((HISTORY_LENGTH, N_FFT_BINS))
        self.spectrogram = self.ax3.imshow(
            self.spectrogram_data.T,
            aspect='auto',
            cmap='viridis',
            origin='lower',
            interpolation='nearest',
            extent=[0, HISTORY_LENGTH, 0, N_FFT_BINS]
        )
        self.ax3.set_ylabel('Frequency Bin')
        self.ax3.set_xlabel('Time (frames)')
        self.ax3.set_title('Spectrogram (Waterfall)')
        self.ax3.set_yticks([0, 8, 16, 23])
        self.ax3.set_yticklabels(['Bass', 'Mid', 'High', 'Top'])

        # Colorbar
        plt.colorbar(self.spectrogram, ax=self.ax3, label='Intensity')

        # Stats text
        self.text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes,
                                  verticalalignment='top', fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        plt.tight_layout()

        print("Spectrum Visualizer Starting...")
        print(f"Sample Rate: {SAMPLE_RATE} Hz")
        print(f"FFT Bins: {N_FFT_BINS} (mel-scale)")
        print(f"Frequency Range: {MIN_FREQUENCY}-{MAX_FREQUENCY} Hz")
        print("\nClose window to stop.")

    def _create_mel_bins(self, n_bins):
        """Create mel-scale frequency bins"""
        mel_min = self._hz_to_mel(MIN_FREQUENCY)
        mel_max = self._hz_to_mel(MAX_FREQUENCY)
        mels = np.linspace(mel_min, mel_max, n_bins + 1)
        return self._mel_to_hz(mels)

    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def analyze_audio(self, audio_chunk):
        """Analyze audio and extract spectrum"""
        # Hann window
        windowed = audio_chunk * np.hanning(len(audio_chunk))

        # FFT
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), 1/SAMPLE_RATE)

        # Bin into mel-scale bins (same as spectrum_stream.py)
        spectrum = np.zeros(N_FFT_BINS)
        for i in range(N_FFT_BINS):
            freq_mask = (freqs >= self.freq_bin_edges[i]) & (freqs < self.freq_bin_edges[i+1])
            spectrum[i] = np.mean(fft[freq_mask]) if freq_mask.any() else 0

        # Normalize
        spectrum = spectrum / (np.max(spectrum) + 1e-6)

        # Energy by bands
        bass_energy = np.mean(spectrum[:8])   # First third
        mid_energy = np.mean(spectrum[8:16])  # Middle third
        high_energy = np.mean(spectrum[16:])  # Last third

        return spectrum, bass_energy, mid_energy, high_energy

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback"""
        if status:
            print(f"Audio: {status}")

        audio_mono = indata[:, 0] if indata.ndim > 1 else indata
        spectrum, bass, mid, high = self.analyze_audio(audio_mono)

        # Store
        self.time_data.append(self.frame_count)
        self.spectrum_history.append(spectrum)
        self.bass_energy_history.append(bass)
        self.mid_energy_history.append(mid)
        self.high_energy_history.append(high)

        self.frame_count += 1

    def update_plot(self, frame):
        """Update visualization"""
        if len(self.spectrum_history) < 1:
            return

        # Get latest spectrum
        latest_spectrum = self.spectrum_history[-1]

        # Update bar chart
        for bar, height in zip(self.bars, latest_spectrum):
            bar.set_height(height)

        # Update energy lines
        x = list(range(len(self.bass_energy_history)))
        self.line_bass.set_data(x, list(self.bass_energy_history))
        self.line_mid.set_data(x, list(self.mid_energy_history))
        self.line_high.set_data(x, list(self.high_energy_history))

        # Update spectrogram (shift and add new)
        if len(self.spectrum_history) > 0:
            self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=0)
            self.spectrogram_data[-1, :] = latest_spectrum
            self.spectrogram.set_data(self.spectrogram_data.T)

        # Update stats text
        if len(self.bass_energy_history) > 0:
            bass = self.bass_energy_history[-1]
            mid = self.mid_energy_history[-1]
            high = self.high_energy_history[-1]

            self.text.set_text(
                f"Bass: {bass:.3f} | Mid: {mid:.3f} | High: {high:.3f}\n"
                f"Frames: {self.frame_count}"
            )

        return self.bars, self.line_bass, self.line_mid, self.line_high, self.spectrogram

    def run(self):
        """Start visualization"""
        print("\nUsing device 5: BlackHole 2ch")

        stream = sd.InputStream(
            device=5,  # BlackHole
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=self.audio_callback
        )

        with stream:
            ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)
            plt.show()

def main():
    visualizer = SpectrumVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
