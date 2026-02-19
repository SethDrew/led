"""
Rap Pulse — slow background pulse at 1/6th of detected tempo.

Uses BeatPredictor's autocorrelation tempo estimation to lock to the beat,
then free-runs a 6-beat cycle: 3 beats fade on, 3 beats fade off.

The fade shape is quadratic (t²), so it accelerates toward each turnaround
point (peak and trough). This creates a natural visual "snap" marker where
the brightness is changing fastest right as it hits the extreme, then
instantly reverses direction slowly — like a pendulum at the bottom of
its swing.

Dev mode: top LED flashes pure blue on each detected beat.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, AbsIntegral, BeatPredictor


class RapPulseEffect(AudioReactiveEffect):
    """Slow background pulse locked to 1/6th of detected tempo."""

    registry_name = 'rap_pulse'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)
        self.predictor = BeatPredictor(rms_fps=self.absint.rms_fps)

        # One full cycle = 6 beats (3 up + 3 down)
        self.cycle_beats = 6
        self.phase = 0.0  # 0→1 over one cycle

        self.brightness = 0.0

        # Dev: beat flash on top LED (exponential decay)
        self.beat_flash = 0.0

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
            normalized = self.absint.update(frame)
            beats = self.predictor.feed(
                self.absint.raw, normalized, self.absint.time_acc
            )

            # Beat flash trigger
            beat_flash = self.beat_flash
            for b in beats:
                if b['type'] == 'confirmed':
                    beat_flash = 1.0
            # Exponential decay (~100ms visible)
            beat_flash *= 0.92

            # Advance phase based on estimated tempo.
            # Once estimated_period is set, keep using it — the period
            # itself is already conservatively smoothed (80/20 blend,
            # only updates on high-confidence beats). Gating on
            # per-frame confidence causes flicker during complex sections.
            if self.predictor.estimated_period > 0:
                cycle_period = self.predictor.estimated_period * self.cycle_beats
                self.phase += self.absint.rms_dt / cycle_period
                if self.phase >= 1.0:
                    self.phase -= 1.0

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
                self.beat_flash = beat_flash

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            brightness = self.brightness
            beat_flash = self.beat_flash

        # Background: warm amber scaled by pulse brightness
        r = int(255 * brightness)
        g = int(180 * brightness)
        b = int(60 * brightness)
        frame = np.full((self.num_leds, 3), [r, g, b], dtype=np.uint8)

        # Dev: top LED = pure blue beat indicator
        frame[-1] = [0, 0, int(255 * beat_flash)]

        return frame

    def get_diagnostics(self) -> dict:
        period = self.predictor.estimated_period
        cycle_s = period * self.cycle_beats if period > 0 else 0
        return {
            'brightness': f'{self.brightness:.2f}',
            'bpm': f'{self.predictor.bpm:.1f}',
            'conf': f'{self.predictor.confidence:.2f}',
            'phase': f'{self.phase:.2f}',
            'cycle': f'{cycle_s:.1f}s' if cycle_s > 0 else '-',
        }
