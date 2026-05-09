"""
Pot Particle — single particle positioned by pot, colored by gyro rotation speed.

Pot value (0–1023) maps to LED position (smoothed to suppress ADC jitter).
Gyro magnitude from duck sender determines hue: still = warm (red), fast spin = cool (blue).
Particle has a soft Gaussian glow around it.

Press 1: explosion — shrapnel flies outward from particle position, fades and dies.
"""

import math
import random
import numpy as np
import colorsys
from base import AudioReactiveEffect


class PotParticleEffect(AudioReactiveEffect):

    registry_name = 'pot_particle'
    ref_pattern = 'ambient'
    ref_scope = 'beat'
    ref_input = 'pot (position) + gyro (rotation speed) + keyboard (explosion)'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.pot_raw = 512.0
        self.pot_smoothed = 512.0
        self.gyro_dps = 0.0
        self.hue = 0.0
        self.glow_width = 3.0
        self.pot_alpha = 0.2
        self.hue_alpha = 0.05
        self.pot_deadzone = 6.0
        self.keys = set()
        self.prev_keys = set()
        self.shrapnel = []

    def set_pot_value(self, raw):
        self.pot_raw = float(raw)

    def set_duck_data(self, duck_data):
        gx = duck_data.get('gx', 0)
        gy = duck_data.get('gy', 0)
        gz = duck_data.get('gz', 0)
        mag_raw = math.sqrt(gx * gx + gy * gy + gz * gz)
        self.gyro_dps = mag_raw / 131.0

    def set_keys(self, keys):
        self.prev_keys = self.keys
        self.keys = keys

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def _explode(self, pos):
        num_pieces = random.randint(8, 14)
        for _ in range(num_pieces):
            speed = random.uniform(15, 80)
            direction = random.choice([-1, 1])
            self.shrapnel.append({
                'pos': pos,
                'vel': speed * direction,
                'life': 1.0,
                'decay': random.uniform(1.5, 3.5),
                'hue': self.hue + random.uniform(-0.15, 0.15),
            })

    def render(self, dt: float) -> np.ndarray:
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        # Pot smoothing
        if abs(self.pot_raw - self.pot_smoothed) > self.pot_deadzone:
            self.pot_smoothed += (self.pot_raw - self.pot_smoothed) * self.pot_alpha

        position = (self.pot_smoothed / 1023.0) * (self.num_leds - 1)

        # Gyro → hue
        rotation_speed = min(self.gyro_dps / 300.0, 1.0)
        target_hue = rotation_speed * 0.65
        self.hue += (target_hue - self.hue) * self.hue_alpha

        # Trigger explosion on key press (not hold)
        if 'Digit1' in self.keys and 'Digit1' not in self.prev_keys:
            self._explode(position)

        # Draw main particle
        r, g, b = colorsys.hls_to_rgb(self.hue, 0.5, 1.0)
        for i in range(self.num_leds):
            dist = abs(i - position)
            brightness = math.exp(-(dist ** 2) / (2 * self.glow_width ** 2))
            if brightness > 0.01:
                frame[i, 0] += r * brightness * 255
                frame[i, 1] += g * brightness * 255
                frame[i, 2] += b * brightness * 255

        # Draw and update shrapnel
        alive = []
        for s in self.shrapnel:
            s['pos'] += s['vel'] * dt
            s['vel'] *= 0.96
            s['life'] -= s['decay'] * dt

            if s['life'] <= 0 or s['pos'] < -5 or s['pos'] > self.num_leds + 5:
                continue

            alive.append(s)
            sr, sg, sb = colorsys.hls_to_rgb(s['hue'] % 1.0, 0.5, 1.0)
            glow = 0.8 + s['life'] * 1.2
            bright_scale = s['life'] * 255

            for i in range(self.num_leds):
                dist = abs(i - s['pos'])
                if dist > glow * 3:
                    continue
                brightness = math.exp(-(dist ** 2) / (2 * glow ** 2))
                if brightness > 0.01:
                    frame[i, 0] += sr * brightness * bright_scale
                    frame[i, 1] += sg * brightness * bright_scale
                    frame[i, 2] += sb * brightness * bright_scale

        self.shrapnel = alive

        return np.clip(frame, 0, 255).astype(np.uint8)
