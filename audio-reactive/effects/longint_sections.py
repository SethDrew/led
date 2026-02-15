"""
LongInt Sections — smooth long-horizon brightness with bass-reactive sparkle.

Brightness is a blend of two signals:
  80% — long integral: rolling RMS average over ~10 seconds. This is the
         overall energy envelope. Changes slowly, follows the song's arc.
  20% — bass abs-integral: abs-integral of bass-band RMS derivative over
         ~150ms. This catches kick drums and bass transients as quick flashes.

The result is a smooth, breathing base that reacts to the song's energy level,
with bass hits adding a sharp 20% brightness kick on top.

Color is handled by the fib_orange_purple palette preset (Fibonacci sections).
"""

import numpy as np
import threading
from scipy.signal import butter, sosfilt
from base import ScalarSignalEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class LongIntSectionsEffect(ScalarSignalEffect):
    """Smooth long-horizon brightness + bass-reactive top layer."""

    registry_name = 'longint_sections'
    default_palette = 'fib_orange_purple'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()

        # Long integral: rolling RMS over ~10 seconds
        self.long_window_sec = 10.0
        dt_per_frame = 512 / sample_rate  # rms_hop / sample_rate
        self.long_window_frames = max(1, int(self.long_window_sec / dt_per_frame))
        self.rms_ring = np.zeros(self.long_window_frames, dtype=np.float32)
        self.rms_ring_pos = 0
        self.long_rms = 0.0
        self.long_rms_peak = 1e-10
        self.long_peak_decay = 0.9999

        # Bass abs-integral: short reactive signal
        # Bass bandpass filter (20-250 Hz)
        nyq = sample_rate / 2
        low_n = max(20 / nyq, 0.001)
        high_n = min(250 / nyq, 0.999)
        self.bass_sos = butter(4, [low_n, high_n], btype='band', output='sos')
        self.bass_filter_state = np.zeros((self.bass_sos.shape[0], 2))

        # Use AbsIntegral for bass derivative computation
        self.bass_absint = AbsIntegral(sample_rate=sample_rate, peak_decay=0.998)

        # Blend ratio
        self.long_weight = 0.80
        self.bass_weight = 0.20

        # Visual state
        self.target_brightness = 0.0
        self.brightness = 0.0
        self.attack_rate = 0.5
        self.decay_rate = 0.88

        self._lock = threading.Lock()

    @property
    def name(self):
        return "LongInt Sections"

    @property
    def description(self):
        return "Blends 80% long-horizon RMS (10s energy envelope) with 20% bass abs-integral (kick transients)."

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            # Long integral: rolling RMS average
            rms = np.sqrt(np.mean(frame ** 2))
            self.rms_ring[self.rms_ring_pos % self.long_window_frames] = rms
            self.rms_ring_pos += 1
            filled = min(self.rms_ring_pos, self.long_window_frames)
            self.long_rms = np.mean(self.rms_ring[:filled])

            self.long_rms_peak = max(self.long_rms, self.long_rms_peak * self.long_peak_decay)
            long_normalized = self.long_rms / self.long_rms_peak if self.long_rms_peak > 0 else 0

            # Bass abs-integral: short reactive signal
            filtered, self.bass_filter_state = sosfilt(
                self.bass_sos, frame, zi=self.bass_filter_state)
            # Create a frame from filtered bass audio
            bass_frame = filtered
            bass_normalized = self.bass_absint.update(bass_frame)

            # Blend
            blended = self.long_weight * long_normalized + self.bass_weight * bass_normalized

            with self._lock:
                self.target_brightness = blended
                self._diag_long = long_normalized
                self._diag_bass = bass_normalized

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            target = self.target_brightness

        if target > self.brightness:
            self.brightness += (target - self.brightness) * self.attack_rate
        else:
            self.brightness *= self.decay_rate ** (dt * 30)

        return self.brightness

    def get_diagnostics(self) -> dict:
        long_v = getattr(self, '_diag_long', 0)
        bass_v = getattr(self, '_diag_bass', 0)
        return {
            'brightness': f'{self.brightness:.2f}',
            'long': f'{long_v:.2f}',
            'bass': f'{bass_v:.2f}',
        }
