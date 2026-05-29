"""
Drift bloom — bioluminescent creatures that fade in and out as a whole.

Each creature is an ~8-LED-wide entity that drifts slowly along the strip.
Periodically, the entire creature body fades up then back down — no expanding
ring, just a smooth whole-body pulse. Similar drift behavior to drift_crawl_bloom.

Non-audio-reactive: pure ambient animation.

Usage:
    python runner.py drift_bloom --leds 150
    python runner.py drift_bloom --leds 150 --port /dev/cu.usbserial-11120
"""

import math
import random
import numpy as np
from base import AudioReactiveEffect

# --- Tuning knobs ---
NUM_CREATURES = 7
PULSE_EXPANSION_SPEED = 3.3  # px/s — shared anchor with drift_crawl_bloom
PULSE_TO_DRIFT_RATIO = 1.9   # pulse is Nx faster than avg drift
_AVG_DRIFT = PULSE_EXPANSION_SPEED / PULSE_TO_DRIFT_RATIO
_DRIFT_SPREAD = 0.33
DRIFT_SPEED_RANGE = (_AVG_DRIFT * (1 - _DRIFT_SPREAD), _AVG_DRIFT * (1 + _DRIFT_SPREAD))
DRIFT_VEL_MAX = _AVG_DRIFT * 1.7
CREATURE_RADIUS = 3.0     # half-width in pixels
EMIT_INTERVAL = (1.2, 2.8)

BLOOM_RISE = 2.0           # seconds to fade up
BLOOM_HOLD = 1.5           # seconds at peak
BLOOM_FALL = 5.0           # seconds to fade down
BLOOM_TOTAL = BLOOM_RISE + BLOOM_HOLD + BLOOM_FALL
EDGE_SOFTNESS = 0.8        # pixels of soft falloff at creature edge

COLOR = np.array([0, 180, 220], dtype=np.float64)

BOUNCE_MARGIN = 8


class Creature:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.emit_timer = random.uniform(*EMIT_INTERVAL)
        self.blooms = []

    def update(self, dt, strip_len):
        self.pos += self.vel * dt

        if random.random() < 0.03:
            self.vel += random.choice([-0.25, 0.25])
            self.vel = max(-DRIFT_VEL_MAX, min(DRIFT_VEL_MAX, self.vel))

        if self.pos < BOUNCE_MARGIN:
            self.vel = abs(self.vel)
        if self.pos > strip_len - BOUNCE_MARGIN:
            self.vel = -abs(self.vel)

        self.emit_timer -= dt
        if self.emit_timer <= 0:
            self.blooms.append(Bloom())
            base_refract = BLOOM_TOTAL + random.uniform(*EMIT_INTERVAL)
            self.emit_timer = base_refract * random.uniform(0.33, 2.0)

        for b in self.blooms:
            b.age += dt
        self.blooms = [b for b in self.blooms if b.age < BLOOM_TOTAL]

    def render_into(self, buf, strip_len):
        center = self.pos

        lo = max(0, int(center - CREATURE_RADIUS - EDGE_SOFTNESS - 1))
        hi = min(strip_len - 1, int(center + CREATURE_RADIUS + EDGE_SOFTNESS + 1))

        for b in self.blooms:
            age = b.age

            # Envelope: rise → hold → fall
            if age < BLOOM_RISE:
                envelope = age / BLOOM_RISE
            elif age < BLOOM_RISE + BLOOM_HOLD:
                envelope = 1.0
            else:
                fall_t = (age - BLOOM_RISE - BLOOM_HOLD) / BLOOM_FALL
                envelope = max(0.0, 1.0 - fall_t)
                envelope *= envelope  # quadratic fall

            if envelope < 0.003:
                continue

            for i in range(lo, hi + 1):
                dist = abs(i - center)

                # Soft edge: full brightness inside radius, smooth falloff outside
                if dist <= CREATURE_RADIUS:
                    spatial = 1.0
                else:
                    overshoot = dist - CREATURE_RADIUS
                    spatial = math.exp(-overshoot / EDGE_SOFTNESS)

                brightness = envelope * spatial
                if brightness > 0.003:
                    buf[i] = buf[i] + brightness - buf[i] * brightness


class Bloom:
    def __init__(self):
        self.age = 0.0


class DriftBloomEffect(AudioReactiveEffect):
    """Bioluminescent jellyfish — ambient drifting creatures with whole-body bloom."""

    registry_name = 'drift_bloom'
    ref_pattern = 'ambient'
    ref_scope = 'song'
    ref_input = 'none (standalone animation)'
    ref_interactivity = 'visual'

    def __init__(self, num_leds: int, sample_rate: int = 44100, **kwargs):
        super().__init__(num_leds, sample_rate)

        spacing = num_leds / (NUM_CREATURES + 1)
        self.creatures = [
            Creature(
                pos=spacing * (i + 1) + random.uniform(-10, 10),
                vel=random.choice([-1, 1]) * random.uniform(*DRIFT_SPEED_RANGE),
            )
            for i in range(NUM_CREATURES)
        ]

    @property
    def name(self):
        return "Drift Bloom"

    @property
    def description(self):
        return "Bioluminescent jellyfish drift along strip with whole-body fade pulses."

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def render(self, dt: float) -> np.ndarray:
        buf = np.zeros(self.num_leds, dtype=np.float64)

        for c in self.creatures:
            c.update(dt, self.num_leds)
            c.render_into(buf, self.num_leds)

        np.clip(buf, 0.0, 1.0, out=buf)
        buf = buf * buf  # gamma

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        for ch in range(3):
            frame[:, ch] = np.clip(buf * COLOR[ch], 0, 255).astype(np.uint8)

        return frame
