"""
Rap Pulse — slow background pulse at 1/6th of detected tempo.

Uses OnsetTempoTracker (multi-band onset envelope autocorrelation) for
tempo estimation. Better than BeatPredictor for dense percussive content
like rap, where constant hi-hats and 808s make the abs-integral noisy.

Free-runs a 6-beat cycle: 3 beats fade on, 3 beats fade off.
The fade shape is quadratic (t²), so it accelerates toward each turnaround
point (peak and trough) — like a pendulum at the bottom of its swing.

May lock to half-tempo (sub-harmonic) — that's fine for slow fades,
it just makes the breathing cycle twice as long.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, OnsetTempoTracker


class RapPulseEffect(AudioReactiveEffect):
    """Slow background pulse locked to 1/6th of detected tempo."""

    registry_name = 'rap_pulse'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.tracker = OnsetTempoTracker(sample_rate=sample_rate)

        # One full cycle = 4 beats (2 up + 2 down)
        self.cycle_beats = 4
        self.phase = 0.0  # 0→1 over one cycle

        self.brightness = 0.0

        # Dev: beat-rate phase accumulator for debug LED
        self.beat_phase = 0.0
        self._beat_phase_snap = 0.0

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Rap Pulse"

    @property
    def description(self):
        return ("Slow background pulse at 1/6th tempo; "
                "3-beat fade on, 3-beat fade off with acceleration markers.")

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            self.tracker.feed_frame(frame)

            # Advance phase based on estimated tempo.
            # The period is conservatively smoothed (80/20 blend).
            if self.tracker.estimated_period > 0:
                dt_step = self.tracker.rms_dt
                cycle_period = self.tracker.estimated_period * self.cycle_beats
                self.phase += dt_step / cycle_period
                if self.phase >= 1.0:
                    self.phase -= 1.0

                # Beat-rate phase for debug LED (same accumulator approach)
                self.beat_phase += dt_step / self.tracker.estimated_period
                if self.beat_phase >= 1.0:
                    self.beat_phase -= 1.0

                # Shaped brightness: quadratic ease into each turnaround
                #   Rise (phase 0→0.5): t²     — slow departure from trough,
                #                                  fast arrival at peak
                #   Fall (phase 0.5→1): 1-t²   — slow departure from peak,
                #                                  fast arrival at trough
                if self.phase < 0.5:
                    t = self.phase / 0.5
                    brightness = t * t
                else:
                    t = (self.phase - 0.5) / 0.5
                    brightness = 1.0 - t * t
            else:
                brightness = 0.0

            with self._lock:
                self.brightness = brightness
                self._beat_phase_snap = self.beat_phase

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            brightness = self.brightness
            beat_phase = self._beat_phase_snap

        # Background: warm amber scaled by pulse brightness
        r = int(255 * brightness)
        g = int(180 * brightness)
        b = int(60 * brightness)
        frame = np.full((self.num_leds, 3), [r, g, b], dtype=np.uint8)

        # Dev: top LED flashes blue at beat rate
        flash = max(0.0, 1.0 - beat_phase * 4)  # sharp decay over first 25%
        frame[-1] = [0, 0, int(255 * flash)]

        return frame

    def get_diagnostics(self) -> dict:
        period = self.tracker.estimated_period
        cycle_s = period * self.cycle_beats if period > 0 else 0
        return {
            'brightness': f'{self.brightness:.2f}',
            'bpm': f'{self.tracker.bpm:.1f}',
            'conf': f'{self.tracker.confidence:.2f}',
            'phase': f'{self.phase:.2f}',
            'cycle': f'{cycle_s:.1f}s' if cycle_s > 0 else '-',
        }
