"""
Abs-Integral Proportional — no beat detection, just proportional brightness.

Instead of detecting beats (threshold -> binary), map the abs-integral signal
directly to LED brightness. Big energy change = bright flash. Small change =
dim flash. Steady-state = dark.

This sidesteps the false positive problem entirely: a false positive from a
small energy change produces a proportionally small flash that humans won't
notice. The brightness IS the signal.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class AbsIntProportionalEffect(ScalarSignalEffect):
    """Whole-tree brightness directly mapped to abs-integral of RMS derivative."""

    default_palette = 'amber'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        self.target_brightness = 0.0
        self.brightness = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Glow"

    @property
    def description(self):
        return "Maps abs-integral of RMS derivative directly to brightness without beat detection; false positives stay proportionally dim."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)
            with self._lock:
                self.target_brightness = normalized

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            target = self.target_brightness

        if target > self.brightness:
            self.brightness += (target - self.brightness) * self.attack_rate
        else:
            self.brightness *= self.decay_rate ** (dt * 30)

        return self.brightness

    def get_diagnostics(self) -> dict:
        return {
            'brightness': f'{self.brightness:.2f}',
            'integral': f'{self.absint.raw:.3f}',
            'peak': f'{self.absint.peak:.3f}',
        }
