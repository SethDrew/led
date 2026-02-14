"""
AbsInt Downbeat — pulses only every 4th detected beat.

Uses the same abs-integral late detection as absint_pulse, but counts
beats internally and only fires a visible pulse on every 4th one.
This creates a slow, hypnotic pulse at the bar/downbeat level.

Night mode: 0-20% brightness, red/purple palette.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class AbsIntDownbeatEffect(AudioReactiveEffect):
    """Pulse every 4th detected beat. Night mode reds."""

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
        self.peak_decay = 0.997

        # Beat detection (same as absint_pulse)
        self.threshold = 0.30
        self.cooldown = 0.20
        self.last_beat_time = -1.0
        self.time_acc = 0.0

        # Downbeat counter: fire on every 4th beat
        self.beat_counter = 0
        self.beats_per_bar = 4
        self.total_beats = 0
        self.downbeat_count = 0

        # Visual state
        self.brightness = 0.0
        self.decay_rate = 0.78  # slower decay for downbeat — more sustained glow
        self.max_brightness = 0.20  # night mode cap

        # Palette: deep red → magenta on downbeat
        self.color_downbeat = np.array([180, 0, 100], dtype=np.float32)  # magenta
        self.color_sub = np.array([30, 0, 5], dtype=np.float32)  # very dim red tick

        self._lock = threading.Lock()

    @property
    def name(self):
        return "AbsInt Downbeat"

    @property
    def description(self):
        return "Pulses only every 4th detected beat (downbeat); sub-beats show as dim ticks; night mode red/magenta palette."

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

        # Beat detection
        self.time_acc += dt
        time_since_beat = self.time_acc - self.last_beat_time

        if normalized > self.threshold and time_since_beat > self.cooldown:
            self.last_beat_time = self.time_acc
            self.total_beats += 1
            self.beat_counter += 1

            if self.beat_counter >= self.beats_per_bar:
                # Downbeat — full pulse
                self.beat_counter = 0
                self.downbeat_count += 1
                with self._lock:
                    self.brightness = min(1.0, normalized)
            else:
                # Sub-beat — tiny tick (barely visible)
                with self._lock:
                    self.brightness = max(self.brightness, 0.08)

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)

        # Cap brightness for night mode
        display_b = min(b ** 0.7, 1.0) * self.max_brightness

        # Color: brighter = more magenta, dimmer = deep red
        t = min(b, 1.0)
        color = self.color_sub * (1 - t) + self.color_downbeat * t

        pixel = (color * display_b).clip(0, 255).astype(np.uint8)
        frame = np.tile(pixel, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.total_beats,
            'downbeats': self.downbeat_count,
            'counter': f'{self.beat_counter}/{self.beats_per_bar}',
            'brightness': f'{self.brightness:.2f}',
        }
