"""
Nebula Explosions — nebula background + tap-triggered particle bursts.

Breathing blue-magenta waves with drifting warm-white orbs (from nebula).
When the duck sender detects a hard tap (peak AC accel > threshold),
a random orb explodes into pot_particle-style shrapnel.
"""

import math
import random
import numpy as np
import colorsys
from base import AudioReactiveEffect


class NebulaExplosionsEffect(AudioReactiveEffect):

    registry_name = 'nebula_explosions'
    ref_pattern = 'ambient'
    ref_scope = 'global'
    ref_input = 'accel (tap → explosion)'

    TAP_THRESH_G = 0.50
    TAP_COOLDOWN_S = 0.3

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

        # --- Tap detection ---
        self.ax_baseline = 0.0
        self.ay_baseline = 0.0
        self.az_baseline = 0.0
        self.baseline_ready = False
        self.peak_ac_g = 0.0
        self.last_tap_time = -1.0
        self._duck_call_count = 0

        # --- Shrapnel ---
        self.shrapnel = []

    @property
    def name(self):
        return "Nebula Explosions"

    @property
    def description(self):
        return "Nebula with orbs that explode into shrapnel on tap."

    def set_duck_data(self, data):
        raw_ax = data.get('ax', 0) / 16384.0
        raw_ay = data.get('ay', 0) / 16384.0
        raw_az = data.get('az', 0) / 16384.0

        if not self.baseline_ready:
            self.ax_baseline = raw_ax
            self.ay_baseline = raw_ay
            self.az_baseline = raw_az
            self.baseline_ready = True

        alpha = 0.008
        self.ax_baseline += (raw_ax - self.ax_baseline) * alpha
        self.ay_baseline += (raw_ay - self.ay_baseline) * alpha
        self.az_baseline += (raw_az - self.az_baseline) * alpha

        ac_x = abs(raw_ax - self.ax_baseline)
        ac_y = abs(raw_ay - self.ay_baseline)
        ac_z = abs(raw_az - self.az_baseline)
        self.peak_ac_g = max(ac_x, ac_y, ac_z)
        self._duck_call_count += 1
        if self._duck_call_count % 25 == 0:
            print(f"[nebula_explosions] duck #{self._duck_call_count}: "
                  f"raw=({raw_ax:.2f}, {raw_ay:.2f}, {raw_az:.2f})g  "
                  f"peak_ac={self.peak_ac_g:.3f}g  orbs={len(self.orbs)}")

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
                'hue': random.uniform(0.7, 1.1) % 1.0,  # blue-magenta range
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

        # Spawn orbs
        if len(self.orbs) < self.max_orbs and np.random.rand() < 0.03:
            self.orbs.append({
                'pos': float(np.random.randint(0, self.num_leds)),
                'vel': np.random.choice([-1, 1]) * self.orb_base_speed * self.speed * np.random.uniform(0.7, 1.3),
                'age': 0,
                'lifetime': np.random.randint(200, 300),
            })

        # === Tap detection → explode random orb ===
        if (self.peak_ac_g >= self.TAP_THRESH_G
                and self.elapsed - self.last_tap_time > self.TAP_COOLDOWN_S
                and len(self.orbs) > 0):
            victim = random.choice(self.orbs)
            self._explode_orb(victim)
            self.orbs.remove(victim)
            self.last_tap_time = self.elapsed

        self.peak_ac_g = 0.0

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
            'peak_g': round(self.peak_ac_g, 2),
        }
