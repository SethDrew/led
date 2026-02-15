"""
AbsInt Meter — volume meter style. Number of lit LEDs = abs-integral magnitude.

Low signal → only a few LEDs lit from the base.
High signal → LEDs fill toward the end of the strip.

Proportional mapping (no threshold/detection), fast attack, slow decay.
Color is handled by the reds palette preset.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class AbsIntMeterEffect(ScalarSignalEffect):
    """Volume meter: lit LED count proportional to abs-integral."""

    default_palette = 'reds'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        # Visual state
        self.target_level = 0.0
        self.level = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Meter"

    @property
    def description(self):
        return "Volume meter where lit LED count is proportional to abs-integral magnitude; red-to-magenta gradient from base to tip."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)
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
