"""
Polycule Gust — three colored snakes with RMS-driven speed.

Same snakes as polycule (yellow, cyan, red, 4px each) but no collision
logic. Each snake gets a random base speed and bounces off strip ends.
Audio energy (RMS, EMA-normalized) continuously modulates ALL snake speeds:
quiet = lazy drift, loud = surge forward. Same mapping as leaf_gust.

Usage:
    python runner.py polycule_gust --no-leds
    python runner.py polycule_gust
"""

import random
import threading
import numpy as np
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


COLORS = [
    (128, 100, 0),   # yellow
    (0, 100, 128),   # cyan
    (128, 20, 20),   # red
]

SNAKE_SIZE = 4
TRAVERSE_PERIOD = 2.0
SPEED_VARIATION = 0.5
SPEED_BOOST = 3.0
BRIGHTNESS_PULSE = 0.30

# Energy bar: slow charge / slow drain
BAR_ATTACK_TC = 7.0    # seconds to charge from 0→1 under sustained bass
BAR_RELEASE_TC = 3.0   # seconds to drain from 1→0 when bass drops
PEAK_DECAY_TC = 20.0   # very slow peak tracker for normalization (half-life ~14s)


class Snake:
    def __init__(self, pos, direction, speed, color, size, num_leds):
        self.pos = float(pos)
        self.dir = direction
        self.speed = speed
        self.color = color
        self.size = size
        self.num_leds = num_leds

    def pixels(self):
        head = max(0, min(self.num_leds - 1, int(round(self.pos))))
        trail_dir = -self.dir
        pxs = set()
        for i in range(self.size):
            idx = head + trail_dir * i
            if 0 <= idx < self.num_leds:
                pxs.add(idx)
        return pxs

    def advance(self, dt, speed_mult):
        self.pos += self.dir * self.speed * speed_mult * dt
        if self.pos > self.num_leds - 1:
            self.pos = float(self.num_leds - 1) - (self.pos - (self.num_leds - 1))
            self.dir = -1
        elif self.pos < 0:
            self.pos = -self.pos
            self.dir = 1


class PolyculeGustEffect(AudioReactiveEffect):
    """Three colored snakes — speed surges with audio energy."""

    registry_name = 'polycule_gust'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'RMS energy (EMA-normalized, continuous speed)'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        base_speed = (num_leds - 1) / TRAVERSE_PERIOD
        self._snakes = []
        for i, color in enumerate(COLORS):
            pos = int(num_leds * (i + 0.5) / len(COLORS))
            direction = 1 if i % 2 == 0 else -1
            speed = base_speed * random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
            self._snakes.append(Snake(pos, direction, speed, color, SNAKE_SIZE, num_leds))

        # Audio: bass RMS with slow peak normalization + energy bar
        self._n_fft = 2048
        self._hop = 512
        self._accum = OverlapFrameAccumulator(frame_len=self._n_fft, hop=self._hop)
        fps = sample_rate / self._hop

        # Bass band: ~60-200Hz
        self._bass_lo = int(60 * self._n_fft / sample_rate)
        self._bass_hi = int(200 * self._n_fft / sample_rate)

        # Very slow peak tracker for volume normalization
        self._peak = 1e-6
        self._peak_decay = 1.0 - 1.0 / (PEAK_DECAY_TC * fps)

        # Energy bar (0-1): charges slowly under sustained bass, drains when quiet
        self._bar = 0.0
        self._bar_attack_rate = 1.0 / (BAR_ATTACK_TC * fps)
        self._bar_release_rate = 1.0 / (BAR_RELEASE_TC * fps)

        self._energy = 0.0
        self._lock = threading.Lock()

    @property
    def name(self):
        return "Polycule Gust"

    @property
    def description(self):
        return "Three colored snakes surge with the music — no collisions, pure speed."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self._accum.feed(mono_chunk):
            spectrum = np.abs(np.fft.rfft(frame))
            bass_rms = float(np.sqrt(np.mean(spectrum[self._bass_lo:self._bass_hi] ** 2)))

            # Slow peak decay for normalization (adapts to venue over ~20s)
            if bass_rms > self._peak:
                self._peak = bass_rms
            else:
                self._peak *= self._peak_decay

            # Normalized bass level (0-1)
            level = bass_rms / self._peak if self._peak > 1e-10 else 0.0

            # Energy bar: charges when bass present, drains when absent
            if level > self._bar:
                self._bar += (level - self._bar) * self._bar_attack_rate
            else:
                self._bar -= (self._bar - level) * self._bar_release_rate
            self._bar = max(0.0, min(1.0, self._bar))

            with self._lock:
                self._energy = self._bar

    def render(self, dt: float) -> np.ndarray:
        dt = min(dt, 0.1)

        with self._lock:
            energy = self._energy

        speed_mult = 1.0 + energy * SPEED_BOOST
        bright_mult = (1.0 - BRIGHTNESS_PULSE) + BRIGHTNESS_PULSE * energy

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)

        for s in self._snakes:
            s.advance(dt, speed_mult)
            r = min(255, int(s.color[0] * bright_mult))
            g = min(255, int(s.color[1] * bright_mult))
            b = min(255, int(s.color[2] * bright_mult))
            for idx in s.pixels():
                frame[idx] = (r, g, b)

        return frame

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = self._energy
        speed = (self.num_leds - 1) / TRAVERSE_PERIOD * (1 + energy * SPEED_BOOST)
        return {
            'energy': f'{energy:.2f}',
            'speed': f'{speed:.1f}',
            'snakes': ' '.join(
                f'{s.pos:.0f}{"R" if s.dir == 1 else "L"}'
                for s in self._snakes),
        }
