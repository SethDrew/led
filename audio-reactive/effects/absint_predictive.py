"""
Abs-Integral Predictive — beat prediction via autocorrelation of abs-integral signal.

Builds on absint_pulse (late detection of beats via abs-integral threshold) and adds
tempo estimation via autocorrelation + forward prediction. The late detection confirms
beats, autocorrelation finds the period, and we predict the NEXT beat to fire on-time.

Key insight: autocorrelation of the abs-integral signal over a ~5s window is far more
robust for tempo estimation than measuring intervals between noisy threshold crossings.
The autocorrelation naturally averages over many beats and handles occasional missed/extra
detections gracefully.

Visual: whole tree pulses on each predicted beat (80% brightness) and
confirmed beat (100% brightness). Exponential decay. Falls back to late-only detection
if autocorrelation can't find a confident period.
"""

import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral, BeatPredictor


class AbsIntPredictiveEffect(ScalarSignalEffect):
    """Whole-tree pulse with tempo prediction via autocorrelation of abs-integral."""

    registry_name = 'impulse_predict'
    default_palette = 'reds'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)
        self.predictor = BeatPredictor(rms_fps=self.absint.rms_fps)

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
        return "Combines abs-integral beat detection with autocorrelation tempo estimation to fire predicted beats on-time; confirmed at 100%, predicted at 80%."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)
            beats = self.predictor.feed(self.absint.raw, normalized, self.absint.time_acc)

            for beat in beats:
                if beat['type'] == 'confirmed':
                    with self._lock:
                        self.brightness = beat['strength']
                        self.is_predicted = False
                elif beat['type'] == 'predicted':
                    with self._lock:
                        self.brightness = max(self.brightness, beat['strength'])
                        self.is_predicted = True

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)
        return b

    def get_diagnostics(self) -> dict:
        period_ms = self.predictor.estimated_period * 1000 if self.predictor.estimated_period > 0 else 0
        return {
            'confirmed': self.predictor.beat_count,
            'predicted': self.predictor.predicted_beat_count,
            'brightness': f'{self.brightness:.2f}',
            'period_ms': f'{period_ms:.0f}',
            'bpm': f'{self.predictor.bpm:.1f}',
            'ac_conf': f'{self.predictor.confidence:.2f}',
            'predicting': self.predictor.prediction_active,
        }
