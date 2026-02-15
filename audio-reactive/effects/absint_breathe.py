"""
AbsInt Breathe — symmetric fade-on and fade-off pulse.

Same abs-integral signal as absint_proportional, but both rise AND fall are
smoothed equally. The LED breathes up and down with the music rather than
snapping on and fading off. Good for ambient and slower music.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class AbsIntBreatheEffect(ScalarSignalEffect):
    """Symmetric fade-on/fade-off using abs-integral of RMS derivative."""

    registry_name = 'impulse_breathe'
    default_palette = 'reds'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        self.target_brightness = 0.0
        self.brightness = 0.0
        self.smooth_rate = 0.74

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Breathe"

    @property
    def description(self):
        return "Symmetric fade-on and fade-off pulse from abs-integral signal; gentler and more organic than snap-on/slow-off effects."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)
            with self._lock:
                self.target_brightness = normalized

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            target = self.target_brightness

        # Symmetric smoothing: same rate for both rise and fall
        alpha = 1.0 - self.smooth_rate ** (dt * 30)
        self.brightness += (target - self.brightness) * alpha

        return self.brightness

    def get_diagnostics(self) -> dict:
        return {
            'brightness': f'{self.brightness:.2f}',
            'target': f'{self.target_brightness:.2f}',
            'integral': f'{self.absint.raw:.3f}',
        }
