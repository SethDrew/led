"""
Centroid Color — color temperature tracks spectral centroid.

Maps spectral centroid (timbral brightness) to color temperature:
low centroid (bass-heavy) → warm red/orange, high centroid (bright/airy) →
cool blue. Brightness from smoothed RMS.

VJ prior art: spectral centroid → color temperature is the most validated
audio-visual correspondence across scottlawsonbc, LedFx, and WLED.
(ARCHITECTURE.md Axis 7)

Architecture placement:
  Pattern:   Ambient Effect + Proportional Mapping
  Axis 3:    Spectral centroid (color), RMS (brightness)
  Axis 4:    Gradient shift (background behavior)
  Axis 5:    Phrase temporal scope (~2s centroid smoothing)
  Axis 7:    Centroid → color temperature (proven correspondence)
"""

import numpy as np
import threading
from base import AudioReactiveEffect


# Warm (low centroid) → neutral → cool (high centroid)
# Red/orange → warm white → blue
WARM_COOL_COLORS = np.array([
    [255, 80,  0],    # deep warm (low centroid)
    [255, 160, 40],   # orange
    [255, 220, 140],  # warm white
    [140, 180, 255],  # cool white
    [0,   100, 255],  # blue (high centroid)
], dtype=np.float32)


def _lerp_palette(colors, t):
    """Sample color from palette at position t (0-1)."""
    t = np.clip(t, 0.0, 1.0)
    n = len(colors) - 1
    idx = t * n
    lo = int(idx)
    hi = min(lo + 1, n)
    frac = idx - lo
    return colors[lo] * (1 - frac) + colors[hi] * frac


class CentroidColorEffect(AudioReactiveEffect):
    """Color temperature shifts with spectral centroid; brightness from RMS."""

    registry_name = 'centroid_color'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # FFT
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)

        # Audio accumulation
        self.audio_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.audio_buf_pos = 0

        # Centroid: context-normalized via slow-decay EMA
        # Typical centroid range for music: ~500-5000 Hz
        self.centroid_ema = 2000.0   # starting estimate
        self.centroid_min_ema = 800.0
        self.centroid_max_ema = 4000.0
        self.centroid_adapt = 0.005  # ~200 frames / ~7s to adapt

        # RMS: peak-decay normalization
        self.rms_peak = 1e-6
        self.rms_peak_decay = 0.998  # ~5s half-life at 30fps

        # Smoothed outputs for render thread
        self.smooth_centroid_t = 0.5  # 0=warm, 1=cool
        self.smooth_brightness = 0.0

        self._lock = threading.Lock()
        self._raw_centroid = 0.0
        self._raw_rms = 0.0

    @property
    def name(self):
        return "Centroid Color"

    @property
    def description(self):
        return "Color temperature drifts warm/cool with spectral centroid; brightness from RMS."

    def process_audio(self, mono_chunk: np.ndarray):
        n = len(mono_chunk)
        pos = self.audio_buf_pos

        while n > 0:
            space = self.n_fft - pos
            take = min(n, space)
            self.audio_buf[pos:pos + take] = mono_chunk[:take]
            mono_chunk = mono_chunk[take:]
            pos += take
            n -= take

            if pos >= self.n_fft:
                self._process_frame(self.audio_buf.copy())
                pos = 0

        self.audio_buf_pos = pos

    def _process_frame(self, frame):
        spec = np.abs(np.fft.rfft(frame * self.window))
        power = spec ** 2

        # Spectral centroid: weighted mean of frequencies
        total_power = np.sum(power)
        if total_power > 1e-10:
            centroid = np.sum(self.freq_bins * power) / total_power
        else:
            centroid = self.centroid_ema

        # RMS
        rms = np.sqrt(np.mean(frame ** 2))

        # Adapt centroid range via slow EMA
        self.centroid_ema += (centroid - self.centroid_ema) * self.centroid_adapt
        self.centroid_min_ema += (centroid - self.centroid_min_ema) * self.centroid_adapt * 0.5
        self.centroid_max_ema += (centroid - self.centroid_max_ema) * self.centroid_adapt * 0.5
        # Keep min/max from collapsing
        if self.centroid_max_ema - self.centroid_min_ema < 200:
            self.centroid_min_ema = self.centroid_ema - 100
            self.centroid_max_ema = self.centroid_ema + 100

        # Normalize centroid to 0-1 within adapted range
        span = self.centroid_max_ema - self.centroid_min_ema
        centroid_t = (centroid - self.centroid_min_ema) / span if span > 0 else 0.5

        # Normalize RMS via peak-decay
        self.rms_peak = max(rms, self.rms_peak * self.rms_peak_decay)
        rms_norm = rms / self.rms_peak if self.rms_peak > 1e-10 else 0.0

        with self._lock:
            self.smooth_centroid_t = centroid_t
            self.smooth_brightness = rms_norm
            self._raw_centroid = centroid
            self._raw_rms = rms

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            centroid_t = self.smooth_centroid_t
            brightness = self.smooth_brightness

        # Gamma correction for perceived brightness linearity
        display_b = max(0.0, min(brightness, 1.0)) ** 0.7

        # Minimum glow so LEDs aren't fully dark in quiet passages
        display_b = max(display_b, 0.03)

        # Sample color from warm-cool palette
        color = _lerp_palette(WARM_COOL_COLORS, centroid_t)

        pixel = (color * display_b).clip(0, 255).astype(np.uint8)
        return np.tile(pixel, (self.num_leds, 1))

    def get_diagnostics(self) -> dict:
        with self._lock:
            centroid = self._raw_centroid
            rms = self._raw_rms
            ct = self.smooth_centroid_t
            b = self.smooth_brightness
        return {
            'centroid': f'{centroid:.0f} Hz',
            'range': f'{self.centroid_min_ema:.0f}-{self.centroid_max_ema:.0f}',
            't': f'{ct:.2f}',
            'rms': f'{rms:.3f}',
            'brightness': f'{b:.2f}',
        }
