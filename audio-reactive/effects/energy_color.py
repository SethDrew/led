"""
Energy Color — color vibrancy tracks total energy (RMS).

High energy → vivid saturated color, bright.
Low energy → muted dusty color, dim.

Uses ScalarSignalEffect: RMS drives both palette position (muted→vibrant)
and brightness simultaneously. This double-reinforcement makes quiet
passages feel genuinely quiet and loud passages feel alive.

Architecture placement:
  Pattern:   Proportional Mapping + Ambient Effect
  Axis 3:    RMS (standard feature)
  Axis 4:    Gradient shift (background behavior)
  Axis 5:    Beat-to-phrase (~1s smoothing)
  Axis 7:    Loudness → brightness (proven) + loudness → saturation (novel)
"""

import numpy as np
import threading
from base import ScalarSignalEffect


class EnergyColorEffect(ScalarSignalEffect):
    """Color vibrancy and brightness both track RMS energy."""

    registry_name = 'energy_color'
    default_palette = 'energy_bloom'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # RMS computation
        self.rms = 0.0
        self.rms_peak = 1e-6
        self.rms_peak_decay = 0.998  # ~5s half-life at 30fps

        # Smoothed output
        self.intensity = 0.0
        self.attack = 0.4      # fast attack for transient response
        self.decay_rate = 0.92  # slower decay for visual sustain

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Energy Color"

    @property
    def description(self):
        return "Color shifts from muted to vibrant with total energy; quiet = dusty dim, loud = vivid bright."

    def process_audio(self, mono_chunk: np.ndarray):
        rms = float(np.sqrt(np.mean(mono_chunk ** 2)))

        # Peak-decay normalization (context-relative)
        self.rms_peak = max(rms, self.rms_peak * self.rms_peak_decay)
        normalized = rms / self.rms_peak if self.rms_peak > 1e-10 else 0.0

        with self._lock:
            self.rms = normalized

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            target = self.rms

        # Asymmetric smoothing
        if target > self.intensity:
            self.intensity += (target - self.intensity) * self.attack
        else:
            self.intensity *= self.decay_rate ** (dt * 30)

        return self.intensity

    def get_diagnostics(self) -> dict:
        return {
            'intensity': f'{self.intensity:.2f}',
            'rms_peak': f'{self.rms_peak:.4f}',
        }
