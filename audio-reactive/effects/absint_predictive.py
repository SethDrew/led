"""
Impulse Predict — beat prediction via onset-envelope tempo estimation.

Uses abs-integral threshold crossing for confirmed beat detection, and
OnsetTempoTracker (onset-envelope autocorrelation) for tempo estimation.
Confirmed beats fire at 100% brightness, predicted beats fire on-time at 80%.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral, OnsetTempoTracker


class AbsIntPredictiveEffect(ScalarSignalEffect):
    """Whole-tree pulse with tempo prediction via multi-onset autocorrelation."""

    registry_name = 'impulse_predict'
    default_palette = 'reds'
    ref_pattern = 'accent'
    ref_scope = 'beat'
    ref_input = 'abs-integral + onset autocorr'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)
        self.tempo = OnsetTempoTracker(sample_rate=sample_rate)

        # Beat detection (threshold on abs-integral)
        self.threshold = 0.30
        self.cooldown = 0.25  # seconds
        self.last_beat_time = -1.0
        self.beat_count = 0

        # Prediction state
        self.predicted_strength = 0.80
        self.next_predicted_beat = 0.0
        self.prediction_active = False
        self.predicted_beat_count = 0
        self.max_missed = 4
        self.missed_count = 0

        # Visual state
        self.brightness = 0.0
        self.is_predicted = False
        self.decay_rate = 0.82

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Predict"

    @property
    def description(self):
        return ("Combines abs-integral beat detection with multi-onset tempo "
                "estimation to fire predicted beats on-time; confirmed at 100%, "
                "predicted at 80%.")

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)
            self.tempo.feed_frame(frame)
            t = self.absint.time_acc

            confirmed = False

            # Confirmed beat: threshold crossing with cooldown
            if normalized > self.threshold and (t - self.last_beat_time) > self.cooldown:
                self.last_beat_time = t
                self.beat_count += 1
                self.missed_count = 0
                confirmed = True

                with self._lock:
                    self.brightness = normalized
                    self.is_predicted = False

                # Phase-lock prediction to confirmed beat
                if self.tempo.estimated_period > 0 and self.tempo.confidence >= 0.3:
                    self.next_predicted_beat = t + self.tempo.estimated_period
                    self.prediction_active = True

            # Predicted beat: fire on-time between confirmed beats
            if not confirmed and self.prediction_active and self.tempo.estimated_period > 0:
                if t >= self.next_predicted_beat:
                    self.predicted_beat_count += 1
                    self.missed_count += 1
                    self.next_predicted_beat += self.tempo.estimated_period

                    with self._lock:
                        self.brightness = max(self.brightness, self.predicted_strength)
                        self.is_predicted = True

                    if self.missed_count >= self.max_missed:
                        self.prediction_active = False

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)
        return b

    def get_diagnostics(self) -> dict:
        period_ms = self.tempo.estimated_period * 1000 if self.tempo.estimated_period > 0 else 0
        return {
            'confirmed': self.beat_count,
            'predicted': self.predicted_beat_count,
            'brightness': f'{self.brightness:.2f}',
            'period_ms': f'{period_ms:.0f}',
            'bpm': f'{self.tempo.bpm:.1f}',
            'ac_conf': f'{self.tempo.confidence:.2f}',
            'predicting': self.prediction_active,
        }
