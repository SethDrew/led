"""
Tempo Pulse — onset-envelope tempo estimation drives a free-running pulse,
brightness scaled by current amplitude.

How it works:
  1. OnsetTempoTracker estimates tempo from onset-envelope autocorrelation.
  2. Free-run a pulse oscillator at that period (no late detection needed).
  3. Each pulse's brightness = current RMS, so loud moments flash bright
     and quiet moments flash dim — but the TIMING is always on the grid.

The pulse shape is a raised cosine: smooth fade-on, peak, smooth fade-off.
This avoids the hard-edge snap of threshold-based detection.

Falls back to proportional mode (no pulse timing) until it has enough data
to estimate tempo confidently (~5-10 seconds).
"""

import numpy as np
import threading
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, OnsetTempoTracker


class TempoPulseEffect(ScalarSignalEffect):
    """Free-running tempo pulse, brightness scaled by current amplitude."""

    registry_name = 'tempo_pulse'
    default_palette = 'reds'
    ref_pattern = 'groove'
    ref_scope = 'phrase'
    ref_input = 'RMS + onset autocorr'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.tempo = OnsetTempoTracker(sample_rate=sample_rate)

        # RMS state (for amplitude scaling)
        self.current_rms = 0.0
        self.rms_peak = 1e-10
        self.rms_peak_decay = 0.9998

        # Timing
        self.rms_dt = 512.0 / sample_rate  # time per hop

        # Pulse oscillator
        self.phase = 0.0  # 0-1 within current beat period
        self.pulse_width = 0.80  # fraction of period for pulse on-time

        # Visual state
        self.brightness = 0.0

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Tempo Pulse"

    @property
    def description(self):
        return ("Free-running pulse oscillator at multi-onset tempo estimate; "
                "brightness scaled by current RMS; raised-cosine pulse shape.")

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            # Current RMS (for amplitude scaling)
            rms = np.sqrt(np.mean(frame ** 2))
            self.rms_peak = max(rms, self.rms_peak * self.rms_peak_decay)
            rms_normalized = rms / self.rms_peak if self.rms_peak > 0 else 0

            # Update tempo tracker
            self.tempo.feed_frame(frame)

            # Advance pulse phase
            if self.tempo.estimated_period > 0 and self.tempo.confidence >= 0.25:
                self.phase += self.rms_dt / self.tempo.estimated_period
                if self.phase >= 1.0:
                    self.phase -= 1.0

                # Raised cosine pulse shape within pulse_width window
                if self.phase < self.pulse_width:
                    t = self.phase / self.pulse_width
                    pulse_envelope = 0.5 * (1.0 - np.cos(2 * np.pi * t))
                else:
                    pulse_envelope = 0.0

                # Scale by current amplitude
                brightness = pulse_envelope * rms_normalized
            else:
                # No tempo estimate — fall back to proportional
                brightness = rms_normalized * 0.5

            with self._lock:
                self.current_rms = rms_normalized
                self.brightness = brightness

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            return self.brightness

    def get_diagnostics(self) -> dict:
        return {
            'brightness': f'{self.brightness:.2f}',
            'rms': f'{self.current_rms:.2f}',
            'bpm': f'{self.tempo.bpm:.1f}',
            'conf': f'{self.tempo.confidence:.2f}',
            'phase': f'{self.phase:.2f}',
        }
