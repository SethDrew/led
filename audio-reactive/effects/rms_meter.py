"""
RMS Meter — volume meter driven by raw waveform amplitude (RMS).

No derivatives, no integrals — just how loud the music is right now.
Number of lit LEDs = current RMS normalized against a slow-decay peak.

Fast attack, slow decay.
Color is handled by the reds palette preset.
"""

import numpy as np
import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator


class RMSMeterEffect(ScalarSignalEffect):
    """Volume meter: lit LED count proportional to RMS amplitude."""

    registry_name = 'rms_meter'
    default_palette = 'reds'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()

        # Signal state
        self.rms_peak = 1e-10
        self.peak_decay = 0.9998

        # Visual state
        self.target_level = 0.0
        self.level = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85

        self._lock = threading.Lock()

    @property
    def name(self):
        return "RMS Meter"

    @property
    def description(self):
        return "Simple volume meter: lit LED count equals current RMS normalized against slow-decay peak; no derivatives or integrals."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            rms = np.sqrt(np.mean(frame ** 2))
            self.rms_peak = max(rms, self.rms_peak * self.peak_decay)
            normalized = rms / self.rms_peak if self.rms_peak > 0 else 0

            with self._lock:
                self.target_level = normalized

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            target = self.target_level

        if target > self.level:
            self.level += (target - self.level) * self.attack_rate
        else:
            self.level *= self.decay_rate ** (dt * 30)

        return self.level

    def get_diagnostics(self) -> dict:
        lit = int(min(self.level, 1.0) * self.num_leds)
        return {
            'level': f'{self.level:.2f}',
            'leds': f'{lit}/{self.num_leds}',
        }
