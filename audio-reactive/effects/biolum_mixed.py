"""
Biolum mixed — mixed bloom and crawl creatures spawning from edges.

Creatures randomly spawn at either end of the strip, drift across, and
despawn when they reach the other end. Each creature is randomly either
a bloom type (whole-body fade) or a crawl type (expanding ring). 7 total
creatures maintained — as one despawns, a new one spawns to replace it.

Non-audio-reactive: pure ambient animation.

Usage:
    python runner.py biolum_mixed --leds 150
    python runner.py biolum_mixed --leds 150 --port /dev/cu.usbserial-11120
"""

import math
import random
import numpy as np
from base import AudioReactiveEffect

# --- Shared tuning ---
MAX_CREATURES = 7
PULSE_EXPANSION_SPEED = 3.3
PULSE_TO_DRIFT_RATIO = 1.9
_AVG_DRIFT = PULSE_EXPANSION_SPEED / PULSE_TO_DRIFT_RATIO
_DRIFT_SPREAD = 0.33
DRIFT_SPEED_RANGE = (_AVG_DRIFT * (1 - _DRIFT_SPREAD), _AVG_DRIFT * (1 + _DRIFT_SPREAD))

SPAWN_MARGIN = 5       # spawn within this many px of strip edge
DESPAWN_MARGIN = -5    # despawn this many px past strip edge

COLOR = np.array([0, 180, 220], dtype=np.float64)

# --- Bloom params ---
BLOOM_RADIUS = 3.0
BLOOM_EDGE_SOFTNESS = 0.8
BLOOM_RISE = 2.0
BLOOM_HOLD = 1.5
BLOOM_FALL = 5.0
BLOOM_TOTAL = BLOOM_RISE + BLOOM_HOLD + BLOOM_FALL
BLOOM_EMIT_INTERVAL = (1.2, 2.8)

# --- Crawl params ---
CRAWL_RADIUS = 8.0
CRAWL_PULSE_LIFETIME = CRAWL_RADIUS / PULSE_EXPANSION_SPEED
CRAWL_PULSE_FADE = 1.4
CRAWL_TAIL_DECAY = 4.0
CRAWL_EMIT_INTERVAL = (1.2, 2.8)


class BloomAnim:
    def __init__(self):
        self.age = 0.0


class CrawlAnim:
    def __init__(self):
        self.age = 0.0


class Creature:
    def __init__(self, pos, vel, kind):
        self.pos = pos
        self.vel = vel
        self.kind = kind  # 'bloom' or 'crawl'
        self.anims = []
        self.alive = True

        if kind == 'bloom':
            self.radius = BLOOM_RADIUS
            self.emit_timer = random.uniform(*BLOOM_EMIT_INTERVAL)
        else:
            self.radius = CRAWL_RADIUS
            self.emit_timer = random.uniform(*CRAWL_EMIT_INTERVAL)

    def update(self, dt, strip_len):
        self.pos += self.vel * dt

        # Despawn past edges
        if self.pos < DESPAWN_MARGIN or self.pos > strip_len - DESPAWN_MARGIN:
            self.alive = False
            return

        # Emit
        self.emit_timer -= dt
        if self.emit_timer <= 0:
            if self.kind == 'bloom':
                self.anims.append(BloomAnim())
                base = BLOOM_TOTAL + random.uniform(*BLOOM_EMIT_INTERVAL)
                self.emit_timer = base * random.uniform(0.33, 2.0)
            else:
                self.anims.append(CrawlAnim())
                self.emit_timer = (CRAWL_PULSE_LIFETIME + CRAWL_PULSE_FADE
                                   + random.uniform(*CRAWL_EMIT_INTERVAL))

        # Age and prune anims
        for a in self.anims:
            a.age += dt

        if self.kind == 'bloom':
            self.anims = [a for a in self.anims if a.age < BLOOM_TOTAL]
        else:
            self.anims = [a for a in self.anims
                          if a.age < CRAWL_PULSE_LIFETIME + CRAWL_PULSE_FADE]

    def render_into(self, buf, strip_len):
        if self.kind == 'bloom':
            self._render_bloom(buf, strip_len)
        else:
            self._render_crawl(buf, strip_len)

    def _render_bloom(self, buf, strip_len):
        center = self.pos
        lo = max(0, int(center - BLOOM_RADIUS - BLOOM_EDGE_SOFTNESS - 1))
        hi = min(strip_len - 1, int(center + BLOOM_RADIUS + BLOOM_EDGE_SOFTNESS + 1))

        for a in self.anims:
            age = a.age
            if age < BLOOM_RISE:
                envelope = age / BLOOM_RISE
            elif age < BLOOM_RISE + BLOOM_HOLD:
                envelope = 1.0
            else:
                fall_t = (age - BLOOM_RISE - BLOOM_HOLD) / BLOOM_FALL
                envelope = max(0.0, 1.0 - fall_t)
                envelope *= envelope

            if envelope < 0.003:
                continue

            for i in range(lo, hi + 1):
                dist = abs(i - center)
                if dist <= BLOOM_RADIUS:
                    spatial = 1.0
                else:
                    spatial = math.exp(-(dist - BLOOM_RADIUS) / BLOOM_EDGE_SOFTNESS)

                brightness = envelope * spatial
                if brightness > 0.003:
                    buf[i] = buf[i] + brightness - buf[i] * brightness

    def _render_crawl(self, buf, strip_len):
        center = self.pos
        lo = max(0, int(center - CRAWL_RADIUS - CRAWL_TAIL_DECAY - 2))
        hi = min(strip_len - 1, int(center + CRAWL_RADIUS + CRAWL_TAIL_DECAY + 2))

        for a in self.anims:
            t = min(a.age / CRAWL_PULSE_LIFETIME, 1.0)
            radius = t * CRAWL_RADIUS

            fade = 1.0
            if a.age > CRAWL_PULSE_LIFETIME:
                fade_t = (a.age - CRAWL_PULSE_LIFETIME) / CRAWL_PULSE_FADE
                fade = max(0.0, 1.0 - fade_t)
                fade *= fade

            for i in range(lo, hi + 1):
                dist_from_center = abs(i - center)
                if dist_from_center > radius:
                    continue

                behind = radius - dist_from_center
                brightness = math.exp(-behind / CRAWL_TAIL_DECAY)
                if behind < 1.0:
                    brightness *= behind
                brightness *= fade

                if brightness > 0.003:
                    buf[i] = buf[i] + brightness - buf[i] * brightness


def spawn_creature(strip_len):
    """Spawn a creature at a random position, drifting in a random direction."""
    kind = random.choice(['bloom', 'crawl'])
    speed = random.uniform(*DRIFT_SPEED_RANGE)
    pos = random.uniform(SPAWN_MARGIN, strip_len - SPAWN_MARGIN)
    vel = random.choice([-1, 1]) * speed

    return Creature(pos, vel, kind)


class BiolumMixedEffect(AudioReactiveEffect):
    """Mixed bioluminescent creatures — bloom and crawl — spawning from edges."""

    registry_name = 'biolum_mixed'
    ref_pattern = 'ambient'
    ref_scope = 'song'
    ref_input = 'none (standalone animation)'
    ref_interactivity = 'visual'

    def __init__(self, num_leds: int, sample_rate: int = 44100, **kwargs):
        super().__init__(num_leds, sample_rate)
        self.creatures = [spawn_creature(num_leds) for _ in range(MAX_CREATURES)]

    @property
    def name(self):
        return "Biolum Mixed"

    @property
    def description(self):
        return "Mixed bloom and crawl creatures spawn from edges, drift across, despawn."

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def render(self, dt: float) -> np.ndarray:
        buf = np.zeros(self.num_leds, dtype=np.float64)

        for c in self.creatures:
            c.update(dt, self.num_leds)
            c.render_into(buf, self.num_leds)

        # Remove dead, respawn to maintain count
        self.creatures = [c for c in self.creatures if c.alive]
        while len(self.creatures) < MAX_CREATURES:
            self.creatures.append(spawn_creature(self.num_leds))

        np.clip(buf, 0.0, 1.0, out=buf)
        buf = buf * buf

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        for ch in range(3):
            frame[:, ch] = np.clip(buf * COLOR[ch], 0, 255).astype(np.uint8)

        return frame
