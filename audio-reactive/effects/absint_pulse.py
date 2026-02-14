"""
Abs-Integral Pulse — beat detection via absolute-value integral of RMS derivative.

Observation: RMS derivative pulses positive (energy arriving) then negative
(energy leaving) on each beat. The absolute integral over a short window
captures this "perturbation" — high when a beat just happened, low during
steady-state. This gives F1 scores 20-50% better than bass-band spectral flux.

Visual: whole tree pulses warm white on each detected beat, exponential decay.
This is the "late detection" version — fires when it confirms a beat happened.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class AbsIntPulseEffect(AudioReactiveEffect):
    """Whole-tree pulse using abs-integral of RMS derivative."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # RMS computation
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0

        # RMS state
        self.prev_rms = 0.0
        self.dt = self.rms_hop / sample_rate  # time per RMS frame

        # Abs-integral: ring buffer of |d(RMS)/dt| values
        self.window_sec = 0.15  # 150ms trailing window (best from analysis)
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        # Beat detection
        self.abs_integral = 0.0
        self.integral_peak = 1e-10  # slow-decay peak for normalization
        self.peak_decay = 0.997     # ~15s half-life
        self.threshold = 0.30       # normalized threshold
        self.cooldown = 0.25        # 250ms between beats
        self.last_beat_time = -1.0
        self.time_acc = 0.0
        self.beat_count = 0

        # Visual state
        self.brightness = 0.0
        self.decay_rate = 0.82       # per-frame at 30 FPS
        self.color = np.array([255, 200, 100], dtype=np.float32)  # warm white

        self._lock = threading.Lock()

    @property
    def name(self):
        return "AbsInt Pulse"

    @property
    def description(self):
        return "Beat detection via absolute-integral of RMS derivative; whole tree pulses warm white on each beat with exponential decay."

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
                # Overlap: shift by hop
                self.audio_buf[:self.rms_frame_len - self.rms_hop] = \
                    self.audio_buf[self.rms_hop:]
                pos = self.rms_frame_len - self.rms_hop

        self.audio_buf_pos = pos

    def _process_rms_frame(self, frame):
        """Compute RMS, derivative, abs-integral, and detect beats."""
        rms = np.sqrt(np.mean(frame ** 2))
        dt = self.rms_frame_len / self.sample_rate

        # RMS derivative
        rms_deriv = (rms - self.prev_rms) / dt
        self.prev_rms = rms

        # Store |derivative| in ring buffer
        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1

        # Abs-integral: sum of ring buffer * dt
        self.abs_integral = np.sum(self.deriv_buf) * dt

        # Slow-decay peak normalization
        self.integral_peak = max(self.abs_integral, self.integral_peak * self.peak_decay)
        normalized = self.abs_integral / self.integral_peak if self.integral_peak > 0 else 0

        # Beat detection
        self.time_acc += dt
        time_since_beat = self.time_acc - self.last_beat_time

        if normalized > self.threshold and time_since_beat > self.cooldown:
            with self._lock:
                self.brightness = min(1.0, normalized)
                self.last_beat_time = self.time_acc
                self.beat_count += 1

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)

        # Gamma correction
        display_b = b ** 0.7

        pixel = (self.color * display_b).clip(0, 255).astype(np.uint8)
        frame = np.tile(pixel, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.beat_count,
            'brightness': f'{self.brightness:.2f}',
            'integral': f'{self.abs_integral:.3f}',
        }
