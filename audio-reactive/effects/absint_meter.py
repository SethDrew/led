"""
AbsInt Meter — volume meter style. Number of lit LEDs = abs-integral magnitude.

Low signal → only a few LEDs lit from the base.
High signal → LEDs fill toward the end of the strip.
Color gradient red → magenta along the strip (same as snake).

Proportional mapping (no threshold/detection), fast attack, slow decay.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class AbsIntMeterEffect(AudioReactiveEffect):
    """Volume meter: lit LED count proportional to abs-integral."""

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
        self.target_level = 0.0
        self.level = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85
        self.max_brightness = 0.80

        # Color gradient: red (base) → magenta (tip)
        self.color_start = np.array([200, 20, 0], dtype=np.float32)
        self.color_end = np.array([180, 0, 160], dtype=np.float32)

        # Precompute per-LED colors
        self.led_colors = np.zeros((num_leds, 3), dtype=np.float32)
        for i in range(num_leds):
            t = i / max(num_leds - 1, 1)
            self.led_colors[i] = self.color_start * (1 - t) + self.color_end * t

        self._lock = threading.Lock()

    @property
    def name(self):
        return "AbsInt Meter"

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
            self.target_level = normalized

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            target = self.target_level

        if target > self.level:
            self.level += (target - self.level) * self.attack_rate
        else:
            self.level *= self.decay_rate ** (dt * 30)

        # Number of lit LEDs
        lit = int(min(self.level, 1.0) * self.num_leds)

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        if lit > 0:
            frame[:lit] = (self.led_colors[:lit] * self.max_brightness).clip(0, 255).astype(np.uint8)

        return frame

    def get_diagnostics(self) -> dict:
        lit = int(min(self.level, 1.0) * self.num_leds)
        return {
            'level': f'{self.level:.2f}',
            'leds': f'{lit}/{self.num_leds}',
        }
