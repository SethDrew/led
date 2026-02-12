"""
Abs-Integral Proportional — no beat detection, just proportional brightness.

Instead of detecting beats (threshold → binary), map the abs-integral signal
directly to LED brightness. Big energy change = bright flash. Small change =
dim flash. Steady-state = dark.

This sidesteps the false positive problem entirely: a false positive from a
small energy change produces a proportionally small flash that humans won't
notice. The brightness IS the signal.

Visual: whole tree pulses warm white, brightness proportional to recent
absolute energy change.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class AbsIntProportionalEffect(AudioReactiveEffect):
    """Whole-tree brightness directly mapped to abs-integral of RMS derivative."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # RMS computation
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0

        # RMS state
        self.prev_rms = 0.0

        # Abs-integral: ring buffer of |d(RMS)/dt| values
        self.window_sec = 0.15  # 150ms trailing window
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        # Signal state
        self.abs_integral = 0.0
        self.integral_peak = 1e-10  # slow-decay peak for normalization
        self.peak_decay = 0.998     # slower decay — keeps dynamic range wider

        # Visual state — NO threshold, NO cooldown, NO beat detection
        self.target_brightness = 0.0  # what audio says brightness should be
        self.brightness = 0.0         # smoothed for display
        self.attack_rate = 0.6        # how fast brightness rises (0-1, per render)
        self.decay_rate = 0.85        # how fast brightness falls (per frame at 30 FPS)

        # Color: warm white
        self.color = np.array([255, 200, 100], dtype=np.float32)

        self._lock = threading.Lock()

    @property
    def name(self):
        return "AbsInt Proportional"

    def process_audio(self, mono_chunk: np.ndarray):
        """Accumulate audio into RMS frames."""
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
        """Compute abs-integral and map directly to brightness."""
        rms = np.sqrt(np.mean(frame ** 2))
        dt = self.rms_frame_len / self.sample_rate

        # RMS derivative
        rms_deriv = (rms - self.prev_rms) / dt
        self.prev_rms = rms

        # Store |derivative| in ring buffer
        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1

        # Abs-integral
        self.abs_integral = np.sum(self.deriv_buf) * dt

        # Slow-decay peak normalization
        self.integral_peak = max(self.abs_integral, self.integral_peak * self.peak_decay)
        normalized = self.abs_integral / self.integral_peak if self.integral_peak > 0 else 0

        with self._lock:
            self.target_brightness = normalized

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            target = self.target_brightness

        # Asymmetric smoothing: fast attack, slow decay
        if target > self.brightness:
            # Rising: jump toward target quickly
            self.brightness += (target - self.brightness) * self.attack_rate
        else:
            # Falling: exponential decay
            self.brightness *= self.decay_rate ** (dt * 30)

        # Gamma correction for visual pop (lower = more contrast)
        display_b = self.brightness ** 0.6

        pixel = (self.color * display_b).clip(0, 255).astype(np.uint8)
        frame = np.tile(pixel, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'brightness': f'{self.brightness:.2f}',
            'integral': f'{self.abs_integral:.3f}',
            'peak': f'{self.integral_peak:.3f}',
        }
