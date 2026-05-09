"""
Nebula — non-audio-reactive breathing waves with glowing orbs.

Blue-to-magenta breathing background with warm-white orbs that drift
and leave decay trails. Ported from static-animations/nebula/.
"""

import numpy as np
from base import AudioReactiveEffect


class NebulaEffect(AudioReactiveEffect):
    """Breathing color waves with drifting, trailing orbs."""

    registry_name = 'nebula'
    ref_pattern = 'ambient'
    ref_scope = 'global'
    ref_input = 'none (visual only)'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.elapsed = 0.0
        self.speed = 0.4

        self.BREATH_FREQ = 0.0105 * self.speed
        self.BREATH_CENTER = 51
        self.BREATH_AMP = 38
        self.SPATIAL_AMP = 51
        self.SPATIAL_SPEED = 0.006 * self.speed
        self.BG_MAX = 153

        self.max_orbs = 5
        self.orb_size = 30.0
        self.orb_base_speed = 0.45
        self.orbs = []
        self.orb_brightness = np.zeros(num_leds, dtype=np.float32)

        self.positions = np.arange(num_leds, dtype=np.float32) / num_leds

    @property
    def name(self):
        return "Nebula"

    @property
    def description(self):
        return "Breathing blue-magenta waves with drifting warm-white orbs. Not audio-reactive."

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def render(self, dt: float) -> np.ndarray:
        self.elapsed += dt
        t = self.elapsed * 60.0

        # Background breathing
        breathing = self.BREATH_CENTER + self.BREATH_AMP * np.sin(t * self.BREATH_FREQ)
        phases = self.positions + t * self.SPATIAL_SPEED
        spatial = self.SPATIAL_AMP * (0.5 + 0.5 * np.cos(2.0 * np.pi * phases))
        bg = np.clip(breathing + spatial, 0, self.BG_MAX).astype(np.uint16)

        # Color variation (blue to magenta)
        cp = self.positions * 2.0 * np.pi + t * 0.009
        cs = 0.5 + 0.5 * np.sin(cp)
        bg_r = (20 + cs * 235).astype(np.uint16)
        bg_g = (30 - cs * 20).astype(np.uint16)
        bg_b = (255 - cs * 125).astype(np.uint16)

        frame = np.zeros((self.num_leds, 3), dtype=np.uint16)
        frame[:, 0] = (bg_r * bg) >> 8
        frame[:, 1] = (bg_g * bg) >> 8
        frame[:, 2] = (bg_b * bg) >> 8

        # Orb decay
        decay_per_frame = 1.0 - (1.0 / self.orb_size)
        decay_per_sec = decay_per_frame ** 60.0
        self.orb_brightness *= decay_per_sec ** dt
        self.orb_brightness[self.orb_brightness < 0.01] = 0.0

        # Spawn
        if len(self.orbs) < self.max_orbs and np.random.rand() < 0.03:
            self.orbs.append({
                'pos': float(np.random.randint(0, self.num_leds)),
                'vel': np.random.choice([-1, 1]) * self.orb_base_speed * self.speed * np.random.uniform(0.7, 1.3),
                'age': 0,
                'lifetime': np.random.randint(200, 300),
            })

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

        # Render orbs (warm white) onto frame
        mask = self.orb_brightness > 0.01
        ob = self.orb_brightness[mask]
        frame[mask, 0] = np.clip(frame[mask, 0] + (255 * ob).astype(np.uint16), 0, 255)
        frame[mask, 1] = np.clip(frame[mask, 1] + (240 * ob).astype(np.uint16), 0, 255)
        frame[mask, 2] = np.clip(frame[mask, 2] + (200 * ob).astype(np.uint16), 0, 255)

        return frame.astype(np.uint8)

    def get_diagnostics(self) -> dict:
        return {
            'orbs': len(self.orbs),
            'elapsed': round(self.elapsed, 1),
        }
