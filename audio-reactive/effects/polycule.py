"""
Polycule — three colored snakes bouncing on a dark strip.

Yellow, cyan, and red, 4 pixels each, randomized speeds. Snakes collide
with each other (stun + reverse) and bounce off strip ends. Brightness
pulses ±40% with RMS energy (EMA-normalized).

When all 3 touch simultaneously, 10% chance they merge into one
multi-colored snake for 8 seconds, then split with new random speeds.

Usage:
    python runner.py polycule --no-leds
    python runner.py polycule
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

SNAKE_PULSE_DEPTH = 0.40
SNAKE_SIZE = 4
TRAVERSE_PERIOD = 2.0
SPEED_VARIATION = 0.5  # ±50% of base speed
STUN_TIME = 0.4        # seconds to pause after collision before reversing
MERGE_CHANCE = 0.10    # 10% chance of merge on triple collision
MERGE_DURATION = 8.0   # seconds merged snakes travel together


class Snake:
    def __init__(self, pos, direction, speed, color, size, num_leds):
        self.pos = float(pos)
        self.dir = direction
        self.speed = speed
        self.color = color
        self.size = size
        self.num_leds = num_leds
        self.stun = 0.0
        self._pending_reverse = False
        self.immune = 0.0

    def pixels(self):
        head = max(0, min(self.num_leds - 1, int(round(self.pos))))
        trail_dir = -self.dir
        pxs = set()
        for i in range(self.size):
            idx = head + trail_dir * i
            if 0 <= idx < self.num_leds:
                pxs.add(idx)
        return pxs

    @property
    def stunned(self):
        return self.stun > 0

    @property
    def collidable(self):
        return not self.stunned and self.immune <= 0

    def advance(self, dt):
        if self.immune > 0:
            self.immune -= dt
        if self.stunned:
            self.stun -= dt
            if self.stun <= 0:
                self.stun = 0.0
                if self._pending_reverse:
                    self.dir *= -1
                    self._pending_reverse = False
                self.immune = 1.0
            return
        self.pos += self.dir * self.speed * dt

    def hit(self):
        self.stun = STUN_TIME
        self._pending_reverse = True

    def check_strip_ends(self):
        if self.stunned:
            return
        if self.pos > self.num_leds - 1:
            self.pos = float(self.num_leds - 1) - (self.pos - (self.num_leds - 1))
            self.dir = -1
        elif self.pos < 0:
            self.pos = -self.pos
            self.dir = 1

    def clamp(self):
        self.pos = max(0.0, min(float(self.num_leds - 1), self.pos))


class PolyculeEffect(AudioReactiveEffect):
    """Three colored snakes bouncing — brightness pulses with audio."""

    registry_name = 'polycule'
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'RMS energy (EMA-normalized, snake brightness pulse)'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        base_speed = (num_leds - 1) / TRAVERSE_PERIOD
        self._snakes = []
        for i, color in enumerate(COLORS):
            pos = int(num_leds * (i + 0.5) / len(COLORS))
            direction = 1 if i % 2 == 0 else -1
            speed = base_speed * random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
            self._snakes.append(Snake(pos, direction, speed, color, SNAKE_SIZE, num_leds))

        # Merge state
        self._merged = False
        self._merge_timer = 0.0
        self._merge_snake = None

        # Audio: RMS with dual EMA
        self._accum = OverlapFrameAccumulator()
        fps = sample_rate / 512
        self._slow_ema = 1e-6
        self._slow_alpha = 2.0 / (3.0 * fps + 1.0)
        self._fast_ema = 0.0
        self._fast_alpha = 2.0 / (0.15 * fps + 1.0)
        self._energy = 0.0
        self._lock = threading.Lock()

    @property
    def name(self):
        return "Polycule"

    @property
    def description(self):
        return "Three colored snakes — collide, stun, and occasionally merge."

    def _start_merge(self):
        self._merged = True
        self._merge_timer = MERGE_DURATION
        avg_pos = sum(s.pos for s in self._snakes) / len(self._snakes)
        base_speed = (self.num_leds - 1) / TRAVERSE_PERIOD
        speed = base_speed * random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
        combined_size = sum(s.size for s in self._snakes)
        self._merge_snake = Snake(
            avg_pos, random.choice([-1, 1]), speed,
            COLORS[0], combined_size, self.num_leds)
        self._merge_snake.clamp()

    def _end_merge(self):
        self._merged = False
        merge_pos = self._merge_snake.pos
        self._merge_snake = None
        base_speed = (self.num_leds - 1) / TRAVERSE_PERIOD
        for i, s in enumerate(self._snakes):
            offset = (i - 1) * (SNAKE_SIZE + 2)
            s.pos = merge_pos + offset
            s.dir = random.choice([-1, 1])
            s.speed = base_speed * random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
            s.stun = 0.0
            s._pending_reverse = False
            s.immune = 1.0
            s.clamp()

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self._accum.feed(mono_chunk):
            rms = float(np.sqrt(np.mean(frame ** 2)))
            self._slow_ema += self._slow_alpha * (rms - self._slow_ema)
            ratio = rms / self._slow_ema if self._slow_ema > 1e-10 else 0.0
            self._fast_ema += self._fast_alpha * (ratio - self._fast_ema)
            energy = np.clip((self._fast_ema - 0.5) / 1.5, 0.0, 1.0)
            with self._lock:
                self._energy = float(energy)

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = self._energy
        mult = 1.0 - SNAKE_PULSE_DEPTH + 2.0 * SNAKE_PULSE_DEPTH * energy

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)

        if self._merged:
            self._merge_timer -= dt
            ms = self._merge_snake
            ms.advance(dt)
            ms.check_strip_ends()

            pxs = sorted(ms.pixels())
            for k, idx in enumerate(pxs):
                color = COLORS[k % len(COLORS)]
                r = min(255, int(color[0] * mult))
                g = min(255, int(color[1] * mult))
                b = min(255, int(color[2] * mult))
                frame[idx] = (r, g, b)

            if self._merge_timer <= 0:
                for s in self._snakes:
                    s.pos = ms.pos
                self._end_merge()

            return frame

        # --- Normal mode ---

        for s in self._snakes:
            s.advance(dt)
            s.check_strip_ends()

        # Triple collision: each snake touches at least one other
        p0, p1, p2 = [s.pixels() for s in self._snakes]
        s0_touches = bool(p0 & p1) or bool(p0 & p2)
        s1_touches = bool(p1 & p0) or bool(p1 & p2)
        s2_touches = bool(p2 & p0) or bool(p2 & p1)
        if s0_touches and s1_touches and s2_touches:
            if random.random() < MERGE_CHANCE:
                self._start_merge()
                pxs = sorted(self._merge_snake.pixels())
                for k, idx in enumerate(pxs):
                    color = COLORS[k % len(COLORS)]
                    r = min(255, int(color[0] * mult))
                    g = min(255, int(color[1] * mult))
                    b = min(255, int(color[2] * mult))
                    frame[idx] = (r, g, b)
                return frame

        # Pairwise collision: push apart, stun, reverse after delay
        for i in range(len(self._snakes)):
            for j in range(i + 1, len(self._snakes)):
                si, sj = self._snakes[i], self._snakes[j]
                if not si.collidable or not sj.collidable:
                    continue
                overlap = si.pixels() & sj.pixels()
                if not overlap:
                    continue
                if si.pos <= sj.pos:
                    left, right = si, sj
                else:
                    left, right = sj, si
                push = (len(overlap) + 1) / 2.0
                left.pos -= push
                right.pos += push
                left.clamp()
                right.clamp()
                si.hit()
                sj.hit()

        # Draw snakes
        for s in self._snakes:
            r = min(255, int(s.color[0] * mult))
            g = min(255, int(s.color[1] * mult))
            b = min(255, int(s.color[2] * mult))
            for idx in s.pixels():
                frame[idx] = (r, g, b)

        return frame

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = self._energy
        d = {'energy': f'{energy:.2f}'}
        if self._merged:
            ms = self._merge_snake
            d['merged'] = f'{ms.pos:.0f} {ms.size}px {self._merge_timer:.1f}s'
        else:
            d['snakes'] = ' '.join(
                f'{s.pos:.0f}{"R" if s.dir == 1 else "L"}'
                + ('*' if s.stunned else '')
                + ('i' if s.immune > 0 else '')
                for s in self._snakes)
        return d
