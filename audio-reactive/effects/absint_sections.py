"""
AbsInt Sections — Fibonacci-sized sections with orange→purple gradient.

Sections are Fibonacci-sized (1, 2, 3, 5, 8, 13, 21, 34, 55, ...)
starting from the END of the strip. The smallest section (1 LED) is
at the tip, growing toward the start. The largest section absorbs
any remaining LEDs.

Color is handled by the fib_orange_purple palette preset.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class AbsIntSectionsEffect(ScalarSignalEffect):
    """Fibonacci sections, abs-integral brightness."""

    default_palette = 'fib_orange_purple'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        # Visual state
        self.target_brightness = 0.0
        self.brightness = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Sections"

    @property
    def description(self):
        return "Fibonacci-sized strip sections with proportional abs-integral brightness; orange-to-purple gradient from tip to base."

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
        }
