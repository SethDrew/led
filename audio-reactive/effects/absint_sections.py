"""
AbsInt Sections — Fibonacci-sized sections with orange→purple gradient.

Sections are Fibonacci-sized (1, 2, 3, 5, 8, 13, 21, 34, 55, ...)
starting from the END of the strip. The smallest section (1 LED) is
at the tip, growing toward the start. The largest section absorbs
any remaining LEDs.

Color is handled by the fib_orange_purple palette preset.
"""

import numpy as np
import threading
from base import ScalarSignalEffect


class AbsIntSectionsEffect(ScalarSignalEffect):
    """Fibonacci sections, abs-integral brightness."""

    default_palette = 'fib_orange_purple'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # RMS computation
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0
        self.prev_rms = 0.0

        # Abs-integral ring buffer
        self.window_sec = 0.15
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        # Signal state
        self.abs_integral = 0.0
        self.integral_peak = 1e-10
        self.peak_decay = 0.998

        # Visual state
        self.target_brightness = 0.0
        self.brightness = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Sections"

    @property
    def description(self):
        return "Fibonacci-sized strip sections with proportional abs-integral brightness; orange-to-purple gradient from tip to base."

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
                self._process_rms_frame(self.audio_buf.copy())
                self.audio_buf[:self.rms_frame_len - self.rms_hop] = \
                    self.audio_buf[self.rms_hop:]
                pos = self.rms_frame_len - self.rms_hop
        self.audio_buf_pos = pos

    def _process_rms_frame(self, frame):
        rms = np.sqrt(np.mean(frame ** 2))
        dt = self.rms_frame_len / self.sample_rate
        rms_deriv = (rms - self.prev_rms) / dt
        self.prev_rms = rms

        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1
        self.abs_integral = np.sum(self.deriv_buf) * dt

        self.integral_peak = max(self.abs_integral, self.integral_peak * self.peak_decay)
        normalized = self.abs_integral / self.integral_peak if self.integral_peak > 0 else 0

        with self._lock:
            self.target_brightness = normalized

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            target = self.target_brightness

        if target > self.brightness:
            self.brightness += (target - self.brightness) * self.attack_rate
        else:
            self.brightness *= self.decay_rate ** (dt * 30)

        return self.brightness

    def get_diagnostics(self) -> dict:
        return {
            'brightness': f'{self.brightness:.2f}',
            'integral': f'{self.abs_integral:.3f}',
        }
