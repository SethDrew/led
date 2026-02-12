"""
RMS Meter — volume meter driven by raw waveform amplitude (RMS).

No derivatives, no integrals — just how loud the music is right now.
Number of lit LEDs = current RMS normalized against a slow-decay peak.
Color gradient red → magenta along the strip.

Fast attack, slow decay.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class RMSMeterEffect(AudioReactiveEffect):
    """Volume meter: lit LED count proportional to RMS amplitude."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Signal state
        self.rms_peak = 1e-10
        self.peak_decay = 0.9998

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
        return "RMS Meter"

    def process_audio(self, mono_chunk: np.ndarray):
        rms = np.sqrt(np.mean(mono_chunk ** 2))
        self.rms_peak = max(rms, self.rms_peak * self.peak_decay)
        normalized = rms / self.rms_peak if self.rms_peak > 0 else 0

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
            start = self.num_leds - lit
            frame[start:] = (self.led_colors[start:] * self.max_brightness).clip(0, 255).astype(np.uint8)

        return frame

    def get_diagnostics(self) -> dict:
        lit = int(min(self.level, 1.0) * self.num_leds)
        return {
            'level': f'{self.level:.2f}',
            'leds': f'{lit}/{self.num_leds}',
        }
