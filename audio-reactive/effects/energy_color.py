"""
Energy Color — color vibrancy tracks rolling integral of RMS energy.

The rolling integral sums RMS over a 10-second window. During builds it
grows, during breakdowns it shrinks — capturing the song's energy arc
at phrase-level temporal scope.

Normalized using peak-decay (standard algorithm): instant attack,
slow exponential decay with ~1 minute effective memory.

High sustained energy → vivid saturated color, bright.
Low/declining energy → muted dusty color, dim.

Architecture placement:
  Pattern:   Proportional Mapping + Ambient Effect
  Axis 3:    Rolling integral of RMS (calculus feature, phrase scope)
  Axis 4:    Gradient shift (background behavior)
  Axis 5:    Phrase (~10s rolling window)
  Axis 7:    Energy → brightness (proven) + energy → saturation (novel)
"""

import numpy as np
import threading
from base import ScalarSignalEffect


class EnergyColorEffect(ScalarSignalEffect):
    """Color vibrancy tracks 5s rolling integral of RMS energy."""

    registry_name = 'energy_color'
    default_palette = 'energy_bloom'

    source_features = [
        {'id': 'rms_integral', 'label': 'RMS Integral (10s)', 'color': '#e94560'},
    ]

    # Rolling integral window
    WINDOW_SECONDS = 10.0

    # Peak-decay normalization (~1 minute effective memory)
    PEAK_DECAY = 0.9995  # ~46s half-life at 30fps

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Rolling integral: ring buffer of RMS values, sum over window
        # Audio chunks arrive at ~43 Hz (1024 samples @ 44100)
        self.chunk_rate = sample_rate / 1024
        self.ring_size = int(self.WINDOW_SECONDS * self.chunk_rate)
        self.ring = np.zeros(self.ring_size, dtype=np.float32)
        self.ring_pos = 0
        self.ring_sum = 0.0

        # Peak-decay reference
        self.peak = 1e-10

        # Smoothed output for render
        self.normalized = 0.0
        self.intensity = 0.0

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Energy Color"

    @property
    def description(self):
        return "Color vibrancy tracks 10s rolling integral of RMS; builds glow vivid, breakdowns fade muted."

    def process_audio(self, mono_chunk: np.ndarray):
        rms = float(np.sqrt(np.mean(mono_chunk ** 2)))

        # Update rolling sum: subtract old value, add new
        idx = self.ring_pos % self.ring_size
        self.ring_sum -= self.ring[idx]
        self.ring[idx] = rms
        self.ring_sum += rms
        self.ring_pos += 1

        integral = max(self.ring_sum, 0.0)

        # Peak-decay normalization
        self.peak = max(integral, self.peak * self.PEAK_DECAY)
        normalized = integral / self.peak if self.peak > 1e-10 else 0.0

        with self._lock:
            self.normalized = normalized

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            target = self.normalized

        # Gentle smoothing (~0.5s transition for visual continuity)
        alpha = 1.0 - 0.85 ** (dt * 30)
        self.intensity += (target - self.intensity) * alpha

        return self.intensity

    def get_source_values(self) -> dict:
        # Raw integral — caller normalizes for display
        return {'rms_integral': float(max(self.ring_sum, 0.0))}

    def get_diagnostics(self) -> dict:
        return {
            'intensity': f'{self.intensity:.2f}',
            'integral': f'{self.ring_sum:.3f}',
            'peak': f'{self.peak:.3f}',
        }
