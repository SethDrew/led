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
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'rolling RMS integral 10s'

    source_features = [
        {'id': 'rms_integral', 'label': 'RMS Integral (10s)', 'color': '#e94560'},
        {'id': 'live_intensity', 'label': 'Live Intensity (palette input)', 'color': '#ffd740', 'normalized': True},
    ]

    # Rolling integral window
    WINDOW_SECONDS = 10.0

    # Min-max decay normalization (~1 minute effective memory)
    PEAK_DECAY = 0.9995   # ceiling: instant attack, slow decay (~46s half-life at 43Hz)
    FLOOR_RISE = 0.9995   # floor: instant drop, slow rise (same time constant)

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Rolling integral: ring buffer of RMS values, sum over window
        # Audio chunks arrive at ~43 Hz (1024 samples @ 44100)
        self.chunk_rate = sample_rate / 1024
        self.ring_size = int(self.WINDOW_SECONDS * self.chunk_rate)
        self.ring = np.zeros(self.ring_size, dtype=np.float32)
        self.ring_pos = 0
        self.ring_sum = 0.0

        # Min-max references (ceiling + floor)
        self.ceiling = 1e-10
        self.floor = 0.0

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

        # Ceiling: instant attack, slow decay (tracks recent max)
        self.ceiling = max(integral, self.ceiling * self.PEAK_DECAY)
        # Floor: instant drop, slow rise (tracks recent min)
        if integral < self.floor:
            self.floor = integral
        else:
            self.floor += (integral - self.floor) * (1 - self.FLOOR_RISE)

        # Normalize to full [0, 1] using dynamic range
        span = self.ceiling - self.floor
        normalized = (integral - self.floor) / span if span > 1e-10 else 0.0

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
        return {
            'rms_integral': float(max(self.ring_sum, 0.0)),
            'live_intensity': float(self.intensity),
        }

    def get_diagnostics(self) -> dict:
        return {
            'intensity': f'{self.intensity:.2f}',
            'integral': f'{self.ring_sum:.3f}',
            'ceiling': f'{self.ceiling:.3f}',
            'floor': f'{self.floor:.3f}',
        }
