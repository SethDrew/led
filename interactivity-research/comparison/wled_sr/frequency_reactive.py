"""
WLED Sound Reactive: Frequency Reactive (GEQ) effect.

Reimplements WLED's GEQ (Graphic EQualizer) mode:
  - 1024-pt FFT at 44100 Hz (= WLED's 512-pt at 22050, same 43 Hz resolution)
  - 16 frequency bands with WLED's exact bin boundaries
  - Pink noise compensation (WLED's fftResultPink[] array)
  - Asymmetric smoothing (fast attack 0.75, slow decay 0.22)
  - Map bands to LED positions, height = amplitude, color = palette

For 1D strip/tree: maps 16 bands across the LED strip, each band gets
a section of LEDs, bar height = brightness of that section.

WLED source: usermods/audioreactive/audio_reactive.cpp (FFTcode, postProcessFFTResults)
WLED source: wled00/FX.cpp (mode_2DGEQ adapted to 1D)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import AudioReactiveEffect

# WLED's 16-band bin boundaries (from FFTcode at 22050/512 = 43.07 Hz/bin)
# At 44100/1024 we get the same 43.07 Hz resolution, so indices are identical
WLED_BAND_BINS = [
    (1, 2),      # Band 0:  43-86 Hz   (sub-bass)
    (2, 3),      # Band 1:  86-129 Hz  (bass)
    (3, 5),      # Band 2:  129-216 Hz (bass)
    (5, 7),      # Band 3:  216-301 Hz
    (7, 10),     # Band 4:  301-430 Hz
    (10, 13),    # Band 5:  430-560 Hz
    (13, 19),    # Band 6:  560-818 Hz
    (19, 26),    # Band 7:  818-1120 Hz
    (26, 33),    # Band 8:  1120-1421 Hz
    (33, 44),    # Band 9:  1421-1895 Hz
    (44, 56),    # Band 10: 1895-2412 Hz
    (56, 70),    # Band 11: 2412-3015 Hz
    (70, 86),    # Band 12: 3015-3704 Hz
    (86, 104),   # Band 13: 3704-4479 Hz
    (104, 165),  # Band 14: 4479-7106 Hz
    (165, 215),  # Band 15: 7106-9259 Hz
]

# WLED's pink noise compensation factors
WLED_PINK_NOISE = np.array([
    1.70, 1.71, 1.73, 1.78, 1.68, 1.56, 1.55, 1.63,
    1.79, 1.62, 1.80, 2.06, 2.47, 3.35, 6.83, 9.55
])

# High-band scaling factors from WLED
WLED_HIGH_BAND_SCALE = [1.0] * 14 + [0.88, 0.70]

# GEQ colors: rainbow across 16 bands (WLED uses palette, we use fixed rainbow)
GEQ_COLORS = np.array([
    [255, 0, 0],      # Band 0: red (sub-bass)
    [255, 40, 0],     # Band 1: red-orange
    [255, 80, 0],     # Band 2: orange
    [255, 140, 0],    # Band 3: orange-yellow
    [255, 200, 0],    # Band 4: yellow
    [200, 255, 0],    # Band 5: yellow-green
    [100, 255, 0],    # Band 6: green
    [0, 255, 50],     # Band 7: green-cyan
    [0, 255, 150],    # Band 8: cyan
    [0, 200, 255],    # Band 9: cyan-blue
    [0, 100, 255],    # Band 10: blue
    [50, 0, 255],     # Band 11: blue-purple
    [130, 0, 255],    # Band 12: purple
    [200, 0, 255],    # Band 13: magenta
    [255, 0, 200],    # Band 14: pink
    [255, 0, 100],    # Band 15: hot pink
], dtype=np.float64)


class WLEDFrequencyReactive(AudioReactiveEffect):
    """WLED's GEQ mode: 16 FFT bands → colored bars across strip."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 1024
        self.window = np.hanning(self.n_fft)

        # FFT result state (matches WLED's fftCalc/fftAvg/fftResult)
        self.fft_calc = np.zeros(16)
        self.fft_avg = np.zeros(16)
        self.fft_result = np.zeros(16, dtype=np.uint8)

        # Audio buffer for FFT
        self.audio_buffer = np.zeros(self.n_fft)

        # AGC-like gain
        self.sample_max = 1.0
        self.gain = 1.0

    @property
    def name(self):
        return "WLED GEQ"

    def process_audio(self, mono_chunk: np.ndarray):
        # Fill buffer
        chunk_len = min(len(mono_chunk), self.n_fft)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = mono_chunk[:chunk_len]

        # Windowed FFT
        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Compute 16 bands using WLED's exact bin boundaries
        for i, (lo, hi) in enumerate(WLED_BAND_BINS):
            hi = min(hi, len(spectrum))
            if lo < len(spectrum):
                self.fft_calc[i] = np.mean(spectrum[lo:hi]) if hi > lo else 0.0
            else:
                self.fft_calc[i] = 0.0

        # Apply high-band scaling (WLED applies 0.88 and 0.70 to top two bands)
        for i in range(16):
            self.fft_calc[i] *= WLED_HIGH_BAND_SCALE[i]

        # Pink noise compensation (WLED's frequency response equalization)
        self.fft_calc *= WLED_PINK_NOISE

        # Auto-gain: track peak across all bands and normalize
        # This replaces WLED's AGC which is tuned for their mic's int16 range
        band_max = np.max(self.fft_calc)
        if band_max > self.sample_max:
            self.sample_max = band_max
        else:
            self.sample_max = self.sample_max * 0.998 + band_max * 0.002

        if self.sample_max > 1e-6:
            self.gain = 1.0 / self.sample_max
        else:
            self.gain = 1.0

        # Normalize bands to 0-1 range
        normalized = self.fft_calc * self.gain

        # Asymmetric smoothing (WLED's postProcessFFTResults — fast attack, slow decay)
        for i in range(16):
            if normalized[i] > self.fft_avg[i]:
                self.fft_avg[i] = normalized[i] * 0.75 + 0.25 * self.fft_avg[i]
            else:
                self.fft_avg[i] = normalized[i] * 0.22 + 0.78 * self.fft_avg[i]

        # Map to 0-255 with sqrt compression (WLED's mode 3 behavior)
        for i in range(16):
            val = np.sqrt(max(self.fft_avg[i], 0.0)) * 255.0
            self.fft_result[i] = int(np.clip(val, 0, 255))

    def render(self, dt: float) -> np.ndarray:
        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)

        # Map 16 bands across the LED strip
        band_width = self.num_leds / 16.0

        for band in range(16):
            start_led = int(band * band_width)
            end_led = int((band + 1) * band_width)
            end_led = min(end_led, self.num_leds)

            # Band amplitude as brightness (0-1)
            amp = self.fft_result[band] / 255.0

            # Color for this band
            color = (GEQ_COLORS[band] * amp).astype(np.uint8)

            for led in range(start_led, end_led):
                frame[led] = color

        return frame

    def get_diagnostics(self) -> dict:
        # Show a mini-GEQ in the diagnostics
        bars = ''
        for i in range(16):
            level = self.fft_result[i]
            if level > 200:
                bars += '#'
            elif level > 150:
                bars += '='
            elif level > 100:
                bars += '-'
            elif level > 50:
                bars += '.'
            else:
                bars += ' '
        return {
            'geq': f'[{bars}]',
            'gain': round(self.gain, 1),
        }
