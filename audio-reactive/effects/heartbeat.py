"""
Heartbeat — spatial pulse that radiates across branches, synced to tempo.

Pulse origin walks the outer perimeter (left + right branches),
skipping tip self-loops so it moves smoothly around the sculpture.
Reach uses spatial (Euclidean) distance from topology coordinates,
so the pulse crosses branches when they're physically close.

Tempo: uses OnsetTempoTracker — best false-positive resilience for
electronic/hip-hop (honest low confidence when unsure). Picks the
octave that puts the period closest to ~1 second. Falls back to
fixed 1.0s period when no lock.

Usage:
    python runner.py heartbeat --sculpture cob_diamond --no-leds
    python runner.py heartbeat --sculpture cob_diamond
"""

import math
import threading
import numpy as np
from base import AudioReactiveEffect
from topology import SculptureTopology
from signals import OverlapFrameAccumulator, OnsetTempoTracker


TARGET_PERIOD = 1.0  # desired seconds between pulses


def octave_nearest(period, target):
    """Shift period by octaves (2x/0.5x) to land closest to target."""
    if period <= 0:
        return target
    # Find the octave shift that minimizes |log2(period * 2^n / target)|
    log_ratio = math.log2(target / period)
    n = round(log_ratio)
    return period * (2 ** n)


class HeartbeatEffect(AudioReactiveEffect):
    registry_name = 'heartbeat'

    FADE_ON = 0.15     # seconds to reach full brightness
    FADE_OFF = 0.85    # seconds to fade to black (total = ~1s at target)
    REACH = 0.15       # spatial radius in coordinate units
    COLOR = np.array([140, 0, 0], dtype=np.float64)

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.topo = SculptureTopology('cob_diamond')

        self.accum = OverlapFrameAccumulator()
        self.tracker = OnsetTempoTracker(sample_rate=sample_rate)

        # Beat phase accumulator (0→1 per pulse period, wraps)
        self._beat_phase = 0.0
        self._pulse_count = 0
        self._period = TARGET_PERIOD  # current octave-adjusted period

        self._lock = threading.Lock()
        # Snapshot for render thread
        self._snap_phase = 0.0
        self._snap_pulse = 0
        self._snap_period = TARGET_PERIOD

    @property
    def name(self):
        return "Heartbeat"

    @property
    def description(self):
        return "Spatial pulse on tempo, radiating across branches."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self.tracker.feed_frame(frame)

            raw_period = self.tracker.estimated_period
            if raw_period > 0 and self.tracker.confidence > 0.3:
                period = octave_nearest(raw_period, TARGET_PERIOD)
            else:
                period = TARGET_PERIOD

            dt_step = self.tracker.rms_dt
            self._beat_phase += dt_step / period
            if self._beat_phase >= 1.0:
                self._beat_phase -= 1.0
                self._pulse_count += 1

            with self._lock:
                self._snap_phase = self._beat_phase
                self._snap_pulse = self._pulse_count
                self._snap_period = period

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            phase = self._snap_phase
            pulse_count = self._snap_pulse
            period = self._snap_period

        # Fade envelope: quick on, slow off
        fade_on_frac = self.FADE_ON / (self.FADE_ON + self.FADE_OFF)
        if phase < fade_on_frac:
            brightness = phase / fade_on_frac
        else:
            brightness = 1.0 - (phase - fade_on_frac) / (1.0 - fade_on_frac)

        # Origin walks the outer perimeter (left + right, skipping middle)
        origin = self.topo.rotation_path[pulse_count % len(self.topo.rotation_path)]
        distances = self.topo.distances_from(origin)

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        for i in range(self.topo.num_leds):
            if distances[i] < self.REACH:
                spatial = 1.0 - distances[i] / self.REACH
                frame[i] = (self.COLOR * brightness * spatial).astype(np.uint8)
        return frame

    def get_diagnostics(self) -> dict:
        with self._lock:
            phase = self._snap_phase
            pulse_count = self._snap_pulse
            period = self._snap_period

        fade_on_frac = self.FADE_ON / (self.FADE_ON + self.FADE_OFF)
        ph_label = 'on' if phase < fade_on_frac else 'off'
        origin = self.topo.rotation_path[pulse_count % len(self.topo.rotation_path)]
        xy = self.topo.coords[origin]
        raw = self.tracker.estimated_period
        return {
            'phase': ph_label,
            'origin': f'{origin}',
            'xy': f'({xy[0]:.2f}, {xy[1]:.2f})',
            'bpm': f'{self.tracker.bpm:.0f}',
            'period': f'{period:.2f}s',
            'raw': f'{raw:.2f}s' if raw > 0 else '-',
            'conf': f'{self.tracker.confidence:.2f}',
        }
