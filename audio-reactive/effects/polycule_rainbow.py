"""
Polycule Rainbow — five soft particles drifting with rainbow gradient tails.

Each particle is a short rainbow fade (cycling hue across its glow).
Particles bounce off strip ends and exchange momentum on collision.
Brightness pulses with RMS energy.
"""

import math
import random
import threading
import numpy as np
import colorsys
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


TRAVERSE_PERIOD = 3.0
SPEED_VARIATION = 0.5
PULSE_DEPTH = 0.40
NUM_PARTICLES = 5
COLLISION_RADIUS = 6.0
RESTITUTION = 0.85


class Particle:
    def __init__(self, pos, vel, hue_offset, num_leds):
        self.pos = float(pos)
        self.vel = vel
        self.hue_offset = hue_offset
        self.num_leds = num_leds

    def advance(self, dt):
        self.pos += self.vel * dt
        if self.pos > self.num_leds - 1:
            self.pos = 2.0 * (self.num_leds - 1) - self.pos
            self.vel *= -1
        elif self.pos < 0:
            self.pos = -self.pos
            self.vel *= -1


class PolyculeRainbowEffect(AudioReactiveEffect):

    registry_name = 'polycule_rainbow'
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'RMS energy (brightness pulse)'
    ref_inputs_required = ['audio']
    input_roles = {'audio': 'RMS energy pulses rainbow particle brightness'}

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        base_speed = (num_leds - 1) / TRAVERSE_PERIOD
        self._particles = []
        for i in range(NUM_PARTICLES):
            pos = num_leds * (i + 0.5) / NUM_PARTICLES
            speed = base_speed * random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
            vel = speed * (1 if i % 2 == 0 else -1)
            hue_offset = i / NUM_PARTICLES
            self._particles.append(Particle(pos, vel, hue_offset, num_leds))

        self._accum = OverlapFrameAccumulator()
        fps = sample_rate / 512
        self._slow_ema = 1e-6
        self._slow_alpha = 2.0 / (3.0 * fps + 1.0)
        self._fast_ema = 0.0
        self._fast_alpha = 2.0 / (0.15 * fps + 1.0)
        self._energy = 0.0
        self._lock = threading.Lock()
        self._time = 0.0

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self._accum.feed(mono_chunk):
            rms = float(np.sqrt(np.mean(frame ** 2)))
            self._slow_ema += self._slow_alpha * (rms - self._slow_ema)
            ratio = rms / self._slow_ema if self._slow_ema > 1e-10 else 0.0
            self._fast_ema += self._fast_alpha * (ratio - self._fast_ema)
            energy = np.clip((self._fast_ema - 0.5) / 1.5, 0.0, 1.0)
            with self._lock:
                self._energy = float(energy)

    def _resolve_collisions(self):
        ps = sorted(self._particles, key=lambda p: p.pos)
        for i in range(len(ps) - 1):
            a, b = ps[i], ps[i + 1]
            gap = b.pos - a.pos
            if gap < COLLISION_RADIUS:
                a.vel = -abs(a.vel)
                b.vel = abs(b.vel)
                push = (COLLISION_RADIUS - gap) / 2.0 + 0.5
                a.pos -= push
                b.pos += push

    def render(self, dt: float) -> np.ndarray:
        self._time += dt
        with self._lock:
            energy = self._energy
        mult = 1.0 - PULSE_DEPTH + 2.0 * PULSE_DEPTH * energy

        frame = np.zeros((self.num_leds, 3), dtype=np.float32)
        glow = 2.5

        for p in self._particles:
            p.advance(dt)

        self._resolve_collisions()

        for p in self._particles:
            p.pos = max(0.0, min(float(self.num_leds - 1), p.pos))

            for i in range(self.num_leds):
                dist = abs(i - p.pos)
                brightness = math.exp(-(dist ** 2) / (2 * glow ** 2))
                if brightness < 0.01:
                    continue
                hue = (p.hue_offset + dist * 0.06 + self._time * 0.1) % 1.0
                r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)
                b_scaled = brightness * mult * 200
                frame[i, 0] += r * b_scaled
                frame[i, 1] += g * b_scaled
                frame[i, 2] += b * b_scaled

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = self._energy
        return {
            'energy': f'{energy:.2f}',
            'particles': ' '.join(
                f'{p.pos:.0f}{"R" if p.vel > 0 else "L"}'
                for p in self._particles),
        }
