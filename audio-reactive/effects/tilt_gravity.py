"""
Tilt Gravity — multiple particles roll like balls in a tube based on board tilt.

Accelerometer tilt angle from duck sender drives gravity.
Pot spawns new particles (turn right) or erases them (turn left).
Each particle gets a random damping (stickiness) and hue on spawn.
"""

import math
import random
import numpy as np
import colorsys
from base import AudioReactiveEffect
from inputs import accel_shake


class TiltGravityEffect(AudioReactiveEffect):

    registry_name = 'tilt_gravity'
    ref_pattern = 'ambient'
    ref_scope = 'beat'
    ref_input = 'accel (tilt) + pot (spawn/erase)'
    ref_interactivity = 'sensor'
    ref_inputs_required = ['accel_shake', 'pot_position']
    input_roles = {
        'accel_shake': 'x-axis motion drives gravity force on particles',
        'pot_position': 'turning the knob spawns (+) or pops (-) particles by delta',
    }

    MAX_PARTICLES = 20

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.accel_x = 0.0
        self.pot_raw = 512.0
        self.prev_pot = 512.0
        self.glow_width = 1.5
        self.particles = []
        self._ax_state = {}
        self._spawn_particle()

    def _spawn_particle(self):
        if len(self.particles) >= self.MAX_PARTICLES:
            return
        self.particles.append({
            'pos': self.num_leds * random.random(),
            'vel': 0.0,
            'damping': random.uniform(0.90, 0.998),
            'hue': random.random(),
        })

    def set_pot_value(self, raw):
        self.prev_pot = self.pot_raw
        self.pot_raw = float(raw)

    def set_imu_data(self, data):
        shake = accel_shake((data.get('ax', 0) / 16384.0,
                             data.get('ay', 0) / 16384.0,
                             data.get('az', 0) / 16384.0),
                            self._ax_state)
        self.accel_x = shake[0]

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def render(self, dt: float) -> np.ndarray:
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        # Pot delta for spawn/erase (with deadzone)
        delta = self.pot_raw - self.prev_pot
        if delta > 15:
            self._spawn_particle()
        elif delta < -15 and len(self.particles) > 1:
            self.particles.pop()

        gravity = -self.accel_x * 80.0

        for p in self.particles:
            p['vel'] += gravity * dt
            p['vel'] *= p['damping']
            p['pos'] += p['vel'] * dt

            if p['pos'] < 0:
                p['pos'] = 0
                p['vel'] *= -0.3
            elif p['pos'] > self.num_leds - 1:
                p['pos'] = self.num_leds - 1
                p['vel'] *= -0.3

            r, g, b = colorsys.hls_to_rgb(p['hue'], 0.5, 1.0)

            for i in range(self.num_leds):
                dist = abs(i - p['pos'])
                brightness = math.exp(-(dist ** 2) / (2 * self.glow_width ** 2))
                if brightness > 0.01:
                    frame[i, 0] += r * brightness * 255
                    frame[i, 1] += g * brightness * 255
                    frame[i, 2] += b * brightness * 255

        return np.clip(frame, 0, 255).astype(np.uint8)
