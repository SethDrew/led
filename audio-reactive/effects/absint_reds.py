"""
AbsInt Reds — proportional brightness in warm red/purple/magenta palette.

Low brightness (0-20%) night mode. Color shifts with signal intensity:
  Low energy change  → deep red
  Medium             → magenta
  High               → bright purple/pink

Uses the same abs-integral of RMS derivative as absint_proportional.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class AbsIntRedsEffect(AudioReactiveEffect):
    """Proportional abs-integral mapped to red/purple/magenta palette, night mode."""

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

        # Brightness cap: 80% max
        self.max_brightness = 0.80

        # Color palette: deep red → orange → red → magenta → purple
        # Indexed by normalized intensity (0-1)
        self.palette = np.array([
            [40,  5,  0],     # 0.0 — deep dark red
            [160, 50, 0],     # 0.25 — orange
            [200, 20, 0],     # 0.50 — red-orange
            [180, 0,  60],    # 0.75 — red-magenta
            [160, 20, 180],   # 1.0 — purple/pink
        ], dtype=np.float32)

        self._lock = threading.Lock()

    @property
    def name(self):
        return "AbsInt Reds"

    @property
    def description(self):
        return "Proportional abs-integral brightness in night mode (80% cap) with deep red to orange to magenta color gradient."

    def _sample_palette(self, t):
        """Sample color from palette at position t (0-1)."""
        t = np.clip(t, 0, 1)
        n = len(self.palette) - 1
        idx = t * n
        lo = int(idx)
        hi = min(lo + 1, n)
        frac = idx - lo
        return self.palette[lo] * (1 - frac) + self.palette[hi] * frac

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

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            target = self.target_brightness

        if target > self.brightness:
            self.brightness += (target - self.brightness) * self.attack_rate
        else:
            self.brightness *= self.decay_rate ** (dt * 30)

        # Cap brightness
        b = min(self.brightness, 1.0) * self.max_brightness

        # Sample color from palette based on intensity
        color = self._sample_palette(self.brightness)

        # Apply brightness
        pixel = (color * b).clip(0, 255).astype(np.uint8)
        frame = np.tile(pixel, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'brightness': f'{self.brightness:.2f}',
            'integral': f'{self.abs_integral:.3f}',
        }
