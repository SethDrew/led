"""
AbsInt Meter — volume meter style. Number of lit LEDs = abs-integral magnitude.

Low signal → only a few LEDs lit from the base.
High signal → LEDs fill toward the end of the strip.

Proportional mapping (no threshold/detection), fast attack, slow decay.
Red-to-magenta gradient from base to tip.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class AbsIntMeterEffect(AudioReactiveEffect):
    """Volume meter: lit LED count proportional to abs-integral."""

    registry_name = 'impulse_meter'

    # Gradient colors from base to tip
    COLORS = np.array([
        [40,  5,  0],
        [160, 50, 0],
        [200, 20, 0],
        [180, 0,  60],
    ], dtype=np.float32)

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        # Visual state
        self.target_level = 0.0
        self.level = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85

        # Precompute per-LED gradient
        self._led_colors = np.zeros((num_leds, 3), dtype=np.float32)
        for i in range(num_leds):
            t = i / max(num_leds - 1, 1)
            idx = t * (len(self.COLORS) - 1)
            lo, hi = int(idx), min(int(idx) + 1, len(self.COLORS) - 1)
            frac = idx - lo
            self._led_colors[i] = self.COLORS[lo] * (1 - frac) + self.COLORS[hi] * frac

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

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            target = self.target_level

        if target > self.level:
            self.level += (target - self.level) * self.attack_rate
        else:
            self.level *= self.decay_rate ** (dt * 30)

        lit = int(min(self.level, 1.0) * self.num_leds)
        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        if lit > 0:
            frame[:lit] = self._led_colors[:lit].clip(0, 255).astype(np.uint8)
        return frame

    def get_diagnostics(self) -> dict:
        lit = int(min(self.level, 1.0) * self.num_leds)
        return {
            'level': f'{self.level:.2f}',
            'leds': f'{lit}/{self.num_leds}',
        }
