"""
Drift crawl bloom — bioluminescent creatures with expanding pulse rings.

Each creature is an ~8-LED-wide entity that drifts slowly along the strip.
Periodically, expanding pulse rings animate within the creature's local
coordinate space. The whole creature (body + active pulses) moves as a unit.

Non-audio-reactive: pure ambient animation.

Usage:
    python runner.py drift_crawl_bloom --leds 150
    python runner.py drift_crawl_bloom --leds 150 --port /dev/cu.usbserial-11120
"""

import math
import random
import numpy as np
from base import AudioReactiveEffect

# --- Tuning knobs ---
NUM_CREATURES = 3
PULSE_EXPANSION_SPEED = 3.3  # px/s — fixed anchor
PULSE_TO_DRIFT_RATIO = 1.9  # pulse is Nx faster than avg drift
_AVG_DRIFT = PULSE_EXPANSION_SPEED / PULSE_TO_DRIFT_RATIO
_DRIFT_SPREAD = 0.33  # ± fraction around avg
DRIFT_SPEED_RANGE = (_AVG_DRIFT * (1 - _DRIFT_SPREAD), _AVG_DRIFT * (1 + _DRIFT_SPREAD))
DRIFT_VEL_MAX = _AVG_DRIFT * 1.7
CREATURE_RADIUS = 8.0    # half-width in pixels
EMIT_INTERVAL = (1.2, 2.8)  # seconds between pulses, avg ~0.5Hz

PULSE_LIFETIME = CREATURE_RADIUS / PULSE_EXPANSION_SPEED
PULSE_FADE = 1.4         # seconds for tail to fade after front stops
TAIL_DECAY = 4.0         # exponential decay distance in pixels

# Color: bioluminescent cyan-teal
COLOR = np.array([0, 180, 220], dtype=np.float64)

BOUNCE_MARGIN = 8


class Creature:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.emit_timer = random.uniform(*EMIT_INTERVAL)
        self.pulses = []

    def update(self, dt, strip_len):
        self.pos += self.vel * dt

        # Random walk on velocity
        if random.random() < 0.03:
            self.vel += random.choice([-0.25, 0.25])
            self.vel = max(-DRIFT_VEL_MAX, min(DRIFT_VEL_MAX, self.vel))

        # Soft bounce
        if self.pos < BOUNCE_MARGIN:
            self.vel = abs(self.vel)
        if self.pos > strip_len - BOUNCE_MARGIN:
            self.vel = -abs(self.vel)

        # Emit new pulse?
        self.emit_timer -= dt
        if self.emit_timer <= 0:
            self.pulses.append(Pulse())
            self.emit_timer = PULSE_LIFETIME + PULSE_FADE + random.uniform(*EMIT_INTERVAL)

        # Age pulses, remove dead ones
        for p in self.pulses:
            p.age += dt
        self.pulses = [p for p in self.pulses if p.age < PULSE_LIFETIME + PULSE_FADE]

    def render_into(self, buf, strip_len):
        """Render creature's pulses into a float brightness buffer."""
        center = self.pos

        lo = max(0, int(center - CREATURE_RADIUS - TAIL_DECAY - 2))
        hi = min(strip_len - 1, int(center + CREATURE_RADIUS + TAIL_DECAY + 2))

        for p in self.pulses:
            t = min(p.age / PULSE_LIFETIME, 1.0)
            radius = t * CREATURE_RADIUS

            # After front stops expanding, apply fade
            fade = 1.0
            if p.age > PULSE_LIFETIME:
                fade_t = (p.age - PULSE_LIFETIME) / PULSE_FADE
                fade = max(0.0, 1.0 - fade_t)
                fade *= fade  # quadratic

            for i in range(lo, hi + 1):
                dist_from_center = abs(i - center)
                if dist_from_center > radius:
                    continue

                behind = radius - dist_from_center
                brightness = math.exp(-behind / TAIL_DECAY)

                # Leading edge fade-in (1px)
                if behind < 1.0:
                    brightness *= behind

                brightness *= fade

                if brightness > 0.003:
                    buf[i] = max(buf[i], brightness)


class Pulse:
    def __init__(self):
        self.age = 0.0


class DriftCrawlBloomEffect(AudioReactiveEffect):
    """Bioluminescent jellyfish — ambient drifting creatures with expanding pulses."""

    registry_name = 'drift_crawl_bloom'
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
        return "Drift Crawl Bloom"

    @property
    def description(self):
        return "Bioluminescent jellyfish drift along strip with expanding pulse rings."

    def process_audio(self, mono_chunk: np.ndarray):
        pass  # non-audio-reactive

    def render(self, dt: float) -> np.ndarray:
        buf = np.zeros(self.num_leds, dtype=np.float64)

        for c in self.creatures:
            c.update(dt, self.num_leds)
            c.render_into(buf, self.num_leds)

        # Clamp and apply gamma + color
        np.clip(buf, 0.0, 1.0, out=buf)
        buf = buf * buf  # gamma

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        for ch in range(3):
            frame[:, ch] = np.clip(buf * COLOR[ch], 0, 255).astype(np.uint8)

        return frame
