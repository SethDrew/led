"""
WLED Sound Reactive: Beat/Peak Reactive effect.

Reimplements WLED's peak detection algorithm:
  - Compute FFT, find the dominant frequency bin (FFT_MajorPeak)
  - Track volume peak: if the dominant bin exceeds maxVol threshold,
    and 100ms has passed since last peak, trigger samplePeak
  - On peak: flash LEDs bright, then decay

This is WLED's beat detection — notably simpler than our spectral flux
approach. It's energy-threshold on a single bin, not cross-bin flux.

WLED source: usermods/audioreactive/audio_reactive.cpp (getSample peak detection)
"""

import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import AudioReactiveEffect


class WLEDBeatReactive(AudioReactiveEffect):
    """WLED's peak detection: dominant FFT bin threshold → LED pulse."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 1024
        self.window = np.hanning(self.n_fft)

        # Audio buffer
        self.audio_buffer = np.zeros(self.n_fft)

        # WLED peak detection state
        self.sample_peak = False
        self.time_of_peak = 0.0
        self.peak_delay = 0.1          # 100ms minimum between peaks (WLED default)
        self.max_vol = 0.0             # adaptive max volume for peak bin
        self.bin_num = 0               # index of dominant frequency bin
        self.fft_major_peak = 0.0      # frequency of dominant bin

        # Volume tracking (for threshold)
        self.sample_avg = 0.0
        self.sample_max = 1.0

        # Adaptive maxVol tracking
        self.max_vol_history = []
        self.max_vol_smooth = 0.0

        # LED state
        self.brightness = 0.0
        self.beat_count = 0

        # Colors
        self.beat_color = np.array([255, 100, 0], dtype=np.float64)  # orange flash
        self.peak_decay = 0.15  # decay rate per frame

    @property
    def name(self):
        return "WLED Beat"

    def process_audio(self, mono_chunk: np.ndarray):
        # Fill buffer
        chunk_len = min(len(mono_chunk), self.n_fft)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = mono_chunk[:chunk_len]

        # RMS for volume
        rms = np.sqrt(np.mean(self.audio_buffer ** 2))
        sample = rms * 32768.0

        # Volume smoothing (WLED's 16-tap EMA)
        self.sample_avg = (self.sample_avg * 15.0 + sample) / 16.0

        # FFT
        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))
        spectrum[0] = 0  # remove DC

        # Find dominant bin (WLED's FFT.majorPeak)
        # Only consider bins above ~80 Hz (bin 2 at 43 Hz resolution)
        valid_start = 2
        valid_end = min(len(spectrum), 256)  # up to ~11 kHz
        if valid_end > valid_start:
            self.bin_num = valid_start + np.argmax(spectrum[valid_start:valid_end])
            self.fft_major_peak = self.bin_num * self.sample_rate / self.n_fft
            peak_magnitude = spectrum[self.bin_num] / 16.0  # WLED scaling
        else:
            peak_magnitude = 0.0

        # Adaptive maxVol tracking
        # WLED uses a user-adjustable maxVol; we auto-adapt
        self.max_vol_history.append(peak_magnitude)
        if len(self.max_vol_history) > 200:  # ~5 seconds of history
            self.max_vol_history.pop(0)

        if len(self.max_vol_history) > 10:
            # Set threshold at ~60th percentile of recent peak magnitudes
            sorted_hist = sorted(self.max_vol_history)
            self.max_vol = sorted_hist[int(len(sorted_hist) * 0.6)]

        # WLED peak detection logic (from getSample):
        # if (sampleAvg > 1) && (maxVol > 0) && (binNum > 4) &&
        #    (vReal[binNum] > maxVol) && ((millis() - timeOfPeak) > 100)
        now = time.time()
        self.sample_peak = False

        if (self.sample_avg > 1.0 and
                self.max_vol > 0.0 and
                self.bin_num > 4 and
                peak_magnitude > self.max_vol and
                (now - self.time_of_peak) > self.peak_delay):
            self.sample_peak = True
            self.time_of_peak = now
            self.beat_count += 1
            self.brightness = 1.0

    def render(self, dt: float) -> np.ndarray:
        # Decay
        if not self.sample_peak:
            self.brightness *= (1.0 - self.peak_decay)
            self.brightness = max(0.0, self.brightness)

        # Gamma correction
        gamma_bright = self.brightness ** (1.0 / 2.2)

        # Build frame
        color = (self.beat_color * gamma_bright).astype(np.uint8)
        frame = np.tile(color, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.beat_count,
            'peak': 'BEAT!' if self.sample_peak else '',
            'freq': f'{self.fft_major_peak:.0f}Hz',
            'vol': f'{self.sample_avg:.0f}',
            'thresh': f'{self.max_vol:.0f}',
            'bright': f'{self.brightness:.2f}',
        }
