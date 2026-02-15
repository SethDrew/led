"""
AbsInt Downbeat — pulses only every 4th detected beat.

Uses the same abs-integral late detection as absint_pulse, but counts
beats internally and only fires a visible pulse on every 4th one.
This creates a slow, hypnotic pulse at the bar/downbeat level.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class AbsIntDownbeatEffect(ScalarSignalEffect):
    """Pulse every 4th detected beat."""

    registry_name = 'impulse_downbeat'
    default_palette = 'night_dim'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        # Beat detection
        self.threshold = 0.30
        self.cooldown = 0.20
        self.last_beat_time = -1.0

        # Downbeat counter: fire on every 4th beat
        self.beat_counter = 0
        self.beats_per_bar = 4
        self.total_beats = 0
        self.downbeat_count = 0

        # Visual state
        self.brightness = 0.0
        self.decay_rate = 0.78  # slower decay for downbeat — more sustained glow

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Downbeat"

    @property
    def description(self):
        return "Pulses only every 4th detected beat (downbeat); sub-beats show as dim ticks."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)

            # Beat detection
            time_since_beat = self.absint.time_acc - self.last_beat_time

            if normalized > self.threshold and time_since_beat > self.cooldown:
                self.last_beat_time = self.absint.time_acc
                self.total_beats += 1
                self.beat_counter += 1

                if self.beat_counter >= self.beats_per_bar:
                    # Downbeat — full pulse
                    self.beat_counter = 0
                    self.downbeat_count += 1
                    with self._lock:
                        self.brightness = min(1.0, normalized)
                else:
                    # Sub-beat — tiny tick (barely visible)
                    with self._lock:
                        self.brightness = max(self.brightness, 0.08)

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)
        return b

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.total_beats,
            'downbeats': self.downbeat_count,
            'counter': f'{self.beat_counter}/{self.beats_per_bar}',
            'brightness': f'{self.brightness:.2f}',
        }
