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


class LongIntSectionsEffect(ScalarSignalEffect):
    """Smooth long-horizon brightness + bass-reactive top layer."""

    default_palette = 'fib_orange_purple'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # RMS computation
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0

        # ── Long integral: rolling RMS over ~10 seconds ──
        self.long_window_sec = 10.0
        dt_per_frame = self.rms_hop / sample_rate  # time per RMS frame
        self.long_window_frames = max(1, int(self.long_window_sec / dt_per_frame))
        self.rms_ring = np.zeros(self.long_window_frames, dtype=np.float32)
        self.rms_ring_pos = 0
        self.long_rms = 0.0
        self.long_rms_peak = 1e-10
        self.long_peak_decay = 0.9999  # very slow decay for 10s window

        # ── Bass abs-integral: short reactive signal ──
        # Bass bandpass filter (20-250 Hz)
        nyq = sample_rate / 2
        low_n = max(20 / nyq, 0.001)
        high_n = min(250 / nyq, 0.999)
        self.bass_sos = butter(4, [low_n, high_n], btype='band', output='sos')
        self.bass_filter_state = np.zeros((self.bass_sos.shape[0], 2))

        self.prev_bass_rms = 0.0
        self.bass_window_sec = 0.15
        self.bass_window_frames = max(1, int(self.bass_window_sec / dt_per_frame))
        self.bass_deriv_buf = np.zeros(self.bass_window_frames, dtype=np.float32)
        self.bass_deriv_pos = 0
        self.bass_abs_integral = 0.0
        self.bass_integral_peak = 1e-10
        self.bass_peak_decay = 0.998

        # ── Blend ratio ──
        self.long_weight = 0.80
        self.bass_weight = 0.20

        # ── Visual state ──
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

    def process_audio(self, mono_chunk: np.ndarray):
        n = len(mono_chunk)
        pos = self.audio_buf_pos
        while n > 0:
            space = self.rms_frame_len - pos
            take = min(n, space)
            self.audio_buf[pos:pos + take] = mono_chunk[:take]
            mono_chunk = mono_chunk[take:]
            pos += take
            n -= take
            if pos >= self.rms_frame_len:
                self._process_frame(self.audio_buf.copy())
                self.audio_buf[:self.rms_frame_len - self.rms_hop] = \
                    self.audio_buf[self.rms_hop:]
                pos = self.rms_frame_len - self.rms_hop
        self.audio_buf_pos = pos

    def _process_frame(self, frame):
        dt = self.rms_hop / self.sample_rate

        # ── Long integral: rolling RMS average ──
        rms = np.sqrt(np.mean(frame ** 2))
        self.rms_ring[self.rms_ring_pos % self.long_window_frames] = rms
        self.rms_ring_pos += 1
        filled = min(self.rms_ring_pos, self.long_window_frames)
        self.long_rms = np.mean(self.rms_ring[:filled])

        self.long_rms_peak = max(self.long_rms, self.long_rms_peak * self.long_peak_decay)
        long_normalized = self.long_rms / self.long_rms_peak if self.long_rms_peak > 0 else 0

        # ── Bass abs-integral: short reactive signal ──
        filtered, self.bass_filter_state = sosfilt(
            self.bass_sos, frame, zi=self.bass_filter_state)
        bass_rms = np.sqrt(np.mean(filtered ** 2))
        bass_deriv = (bass_rms - self.prev_bass_rms) / dt
        self.prev_bass_rms = bass_rms

        buf_idx = self.bass_deriv_pos % self.bass_window_frames
        self.bass_deriv_buf[buf_idx] = abs(bass_deriv)
        self.bass_deriv_pos += 1
        self.bass_abs_integral = np.sum(self.bass_deriv_buf) * dt

        self.bass_integral_peak = max(
            self.bass_abs_integral,
            self.bass_integral_peak * self.bass_peak_decay)
        bass_normalized = (self.bass_abs_integral / self.bass_integral_peak
                           if self.bass_integral_peak > 0 else 0)

        # ── Blend ──
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
