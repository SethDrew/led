"""
Nebula Explosions — nebula background + tap-triggered particle bursts.

Breathing blue-magenta waves with drifting warm-white orbs (from nebula).
When the v1 telemetry sender detects a hard tap (amag_max above threshold),
a random orb explodes into pot_particle-style shrapnel.
"""

import math
import random
import numpy as np
import colorsys
from base import AudioReactiveEffect
from inputs import tap_event, AMAG_TAP_THRESH


class NebulaExplosionsEffect(AudioReactiveEffect):

    registry_name = 'nebula_explosions'
    ref_pattern = 'ambient'
    ref_scope = 'global'
    ref_input = 'v1 telemetry IMU (tap → explosion)'
    ref_interactivity = 'sensor'
    ref_inputs_required = ['tap_event']
    input_roles = {
        'tap_event': 'each tap on the v1 sender explodes a random orb into shrapnel',
    }

    TAP_COOLDOWN_S = 0.15  # fast cooldown — users may tap many times/sec

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # --- Nebula background ---
        self.elapsed = 0.0
        self.speed = 0.4

        self.BREATH_FREQ = 0.0105 * self.speed
        self.BREATH_CENTER = 51
        self.BREATH_AMP = 38
        self.SPATIAL_AMP = 51
        self.SPATIAL_SPEED = 0.006 * self.speed
        self.BG_MAX = 153

        self.positions = np.arange(num_leds, dtype=np.float32) / num_leds

        # --- Orbs ---
        self.max_orbs = 5
        self.orb_size = 30.0
        self.orb_base_speed = 0.45
        self.orbs = []
        self.orb_brightness = np.zeros(num_leds, dtype=np.float32)

        # --- Tap detection (v1 telemetry) ---
        self.amag_max = 0
        self.last_tap_time = -1.0
        self._tap_state = {}
        self._v1_count = 0

        # --- Shrapnel ---
        self.shrapnel = []

    @property
    def name(self):
        return "Nebula Explosions"

    @property
    def description(self):
        return "Nebula with orbs that explode into shrapnel on tap."

    def set_v1_data(self, data):
        self.amag_max = data['amag_max']
        self._v1_count += 1
        if self._v1_count % 25 == 0:
            print(f"[nebula_explosions] v1 #{self._v1_count}: "
                  f"amag={self.amag_max} (thresh={AMAG_TAP_THRESH})  "
                  f"orbs={len(self.orbs)}")

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def _explode_orb(self, orb):
        pos = orb['pos']
        num_pieces = random.randint(8, 14)
        for _ in range(num_pieces):
            speed = random.uniform(15, 80)
            direction = random.choice([-1, 1])
            self.shrapnel.append({
                'pos': pos,
                'vel': speed * direction,
                'life': 1.0,
                'decay': random.uniform(1.5, 3.5),
                'hue': random.uniform(0.7, 1.1) % 1.0,
            })

    def render(self, dt: float) -> np.ndarray:
        self.elapsed += dt
        t = self.elapsed * 60.0

        # === Background breathing ===
        breathing = self.BREATH_CENTER + self.BREATH_AMP * np.sin(t * self.BREATH_FREQ)
        phases = self.positions + t * self.SPATIAL_SPEED
        spatial = self.SPATIAL_AMP * (0.5 + 0.5 * np.cos(2.0 * np.pi * phases))
        bg = np.clip(breathing + spatial, 0, self.BG_MAX).astype(np.uint16)

        cp = self.positions * 2.0 * np.pi + t * 0.009
        cs = 0.5 + 0.5 * np.sin(cp)
        bg_r = (20 + cs * 235).astype(np.uint16)
        bg_g = (30 - cs * 20).astype(np.uint16)
        bg_b = (255 - cs * 125).astype(np.uint16)

        frame = np.zeros((self.num_leds, 3), dtype=np.float32)
        frame[:, 0] = (bg_r * bg) >> 8
        frame[:, 1] = (bg_g * bg) >> 8
        frame[:, 2] = (bg_b * bg) >> 8

        # === Orb decay ===
        decay_per_frame = 1.0 - (1.0 / self.orb_size)
        decay_per_sec = decay_per_frame ** 60.0
        self.orb_brightness *= decay_per_sec ** dt
        self.orb_brightness[self.orb_brightness < 0.01] = 0.0

        # Spawn orbs (2x rate for 5s after an explosion)
        respawn_boost = 2.0 if (self.elapsed - self.last_tap_time < 5.0) else 1.0
        if len(self.orbs) < self.max_orbs and np.random.rand() < 0.03 * respawn_boost:
            self.orbs.append({
                'pos': float(np.random.randint(0, self.num_leds)),
                'vel': np.random.choice([-1, 1]) * self.orb_base_speed * self.speed * np.random.uniform(0.7, 1.3),
                'age': 0,
                'lifetime': np.random.randint(200, 300),
            })

        # === Tap detection → explode random orb ===
        if tap_event(self.amag_max, self._tap_state,
                     cooldown_s=self.TAP_COOLDOWN_S, now=self.elapsed) and len(self.orbs) > 0:
            victim = random.choice(self.orbs)
            self._explode_orb(victim)
            self.orbs.remove(victim)
            self.last_tap_time = self.elapsed
            print(f"[nebula_explosions] TAP! amag={self.amag_max} → exploded orb at {victim['pos']:.0f}")

        self.amag_max = 0

        # Update orbs
        dead = []
        for orb in self.orbs:
            orb['age'] += 1
            orb['pos'] = (orb['pos'] + orb['vel']) % self.num_leds
            if orb['age'] >= orb['lifetime']:
                dead.append(orb)
                continue

            lc = orb['age'] / orb['lifetime']
            if lc < 0.4:
                tf = lc / 0.4
                bright = tf * tf * (3.0 - 2.0 * tf)
            elif lc > 0.6:
                tf = (1.0 - lc) / 0.4
                bright = tf * tf * (3.0 - 2.0 * tf)
            else:
                bright = 1.0

            px = int(orb['pos'])
            pn = (px + 1) % self.num_leds
            frac = orb['pos'] - px
            self.orb_brightness[px] = min(1.0, self.orb_brightness[px] + bright * 0.6 * (1.0 - frac))
            self.orb_brightness[pn] = min(1.0, self.orb_brightness[pn] + bright * 0.6 * frac)

        for orb in dead:
            self.orbs.remove(orb)

        # Render orb layer (warm white)
        mask = self.orb_brightness > 0.01
        ob = self.orb_brightness[mask]
        frame[mask, 0] = np.clip(frame[mask, 0] + 255 * ob, 0, 255)
        frame[mask, 1] = np.clip(frame[mask, 1] + 240 * ob, 0, 255)
        frame[mask, 2] = np.clip(frame[mask, 2] + 200 * ob, 0, 255)

        # === Shrapnel ===
        alive = []
        for s in self.shrapnel:
            s['pos'] += s['vel'] * dt
            s['vel'] *= 0.96
            s['life'] -= s['decay'] * dt

            if s['life'] <= 0 or s['pos'] < -5 or s['pos'] > self.num_leds + 5:
                continue

            alive.append(s)
            sr, sg, sb = colorsys.hls_to_rgb(s['hue'], 0.5, 1.0)
            glow = 0.8 + s['life'] * 1.2
            bright_scale = s['life'] * 255
            cutoff = glow * 3

            for i in range(self.num_leds):
                dist = abs(i - s['pos'])
                if dist > cutoff:
                    continue
                brightness = math.exp(-(dist ** 2) / (2 * glow ** 2))
                if brightness > 0.01:
                    frame[i, 0] += sr * brightness * bright_scale
                    frame[i, 1] += sg * brightness * bright_scale
                    frame[i, 2] += sb * brightness * bright_scale

        self.shrapnel = alive

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        return {
            'orbs': len(self.orbs),
            'shrapnel': len(self.shrapnel),
            'amag': self.amag_max,
        }
