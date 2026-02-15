"""
Abs-Integral Pulse — beat detection via absolute-value integral of RMS derivative.

Observation: RMS derivative pulses positive (energy arriving) then negative
(energy leaving) on each beat. The absolute integral over a short window
captures this "perturbation" — high when a beat just happened, low during
steady-state. This gives F1 scores 20-50% better than bass-band spectral flux.

Visual: whole tree pulses on each detected beat, exponential decay.
This is the "late detection" version — fires when it confirms a beat happened.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class AbsIntPulseEffect(ScalarSignalEffect):
    """Whole-tree pulse using abs-integral of RMS derivative."""

    registry_name = 'impulse'
    default_palette = 'amber'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        # Beat detection
        self.threshold = 0.30
        self.cooldown = 0.25
        self.last_beat_time = -1.0
        self.beat_count = 0

        # Visual state
        self.brightness = 0.0
        self.decay_rate = 0.82

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse"

    @property
    def description(self):
        return "Beat detection via absolute-integral of RMS derivative; whole tree pulses on each beat with exponential decay."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)

            # Beat detection
            time_since_beat = self.absint.time_acc - self.last_beat_time

            if normalized > self.threshold and time_since_beat > self.cooldown:
                with self._lock:
                    self.brightness = min(1.0, normalized)
                    self.last_beat_time = self.absint.time_acc
                    self.beat_count += 1

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)
        return b

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.beat_count,
            'brightness': f'{self.brightness:.2f}',
            'integral': f'{self.absint.raw:.3f}',
        }
