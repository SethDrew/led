"""
Two Snakes — yellow and orange snakes on a dim purple background.

Each starts at 1 pixel from opposite ends. They grow by 1 pixel each
time they hit a wall. Tail becomes head on bounce. Period ~2 seconds.
Reset to 1 pixel when filling the strip.

Non-audio-reactive.

Usage:
    python runner.py two_snakes --no-leds
    python runner.py two_snakes
"""

import numpy as np
from base import AudioReactiveEffect


# Background: dim purple
BG = (20, 0, 30)

# Snake colors at 50% brightness
YELLOW = (128, 100, 0)
ORANGE = (128, 50, 0)

TRAVERSE_PERIOD = 2.0
FADE_PIXELS = 2  # how many pixels the fade extends beyond the snake body


class Snake:
    def __init__(self, pos, direction, color, speed, num_leds):
        self.pos = float(pos)
        self.dir = direction
        self.color = color
        self.speed = speed
        self.size = 1
        self.num_leds = num_leds

    def brightness_at(self, idx):
        """Return 0.0-1.0 brightness for a given pixel index.

        Full brightness inside the snake body, fading out at head and tail.
        Uses fractional head position for sub-pixel smooth movement.
        """
        head = self.pos
        if self.dir == 1:
            # Body: [head - (size-1), head], head is rightmost
            body_lo = head - (self.size - 1)
            body_hi = head
        else:
            # Body: [head, head + (size-1)], head is leftmost
            body_lo = head
            body_hi = head + (self.size - 1)

        if body_lo <= idx <= body_hi:
            # Inside body — fade at edges
            dist_from_head = abs(idx - head)
            dist_from_tail = abs(idx - (body_lo if self.dir == 1 else body_hi))
            edge_dist = min(dist_from_head, dist_from_tail)
            # Fade the outermost pixel of head/tail smoothly
            if dist_from_head < 1.0:
                return max(0.0, min(1.0, 1.0 - (1.0 - dist_from_head) * 0.3))
            if dist_from_tail < 1.0:
                return max(0.0, min(1.0, 0.5 + dist_from_tail * 0.5))
            return 1.0
        else:
            # Outside body — fade region
            if idx < body_lo:
                dist = body_lo - idx
            else:
                dist = idx - body_hi
            if dist <= FADE_PIXELS:
                return max(0.0, 1.0 - dist / FADE_PIXELS) ** 2
            return 0.0

    def advance(self, dt):
        self.pos += self.dir * self.speed * dt

    def bounce(self):
        """Tail becomes head, grow by 1."""
        tail_offset = (self.size - 1) * (-self.dir)
        self.pos = float(int(round(self.pos)) + tail_offset)
        self.dir *= -1
        self.size += 1
        # Clamp range to strip
        lo, hi = self._range()
        if lo < 0:
            self.pos += float(-lo)
        lo, hi = self._range()
        if hi > self.num_leds - 1:
            self.pos -= float(hi - (self.num_leds - 1))

    def _range(self):
        head = int(round(self.pos))
        if self.dir == 1:
            return head - (self.size - 1), head
        else:
            return head, head + (self.size - 1)

    def check_strip_ends(self):
        if self.pos > self.num_leds - 1:
            self.pos = float(self.num_leds - 1)
            self.bounce()
        elif self.pos < 0:
            self.pos = 0.0
            self.bounce()


class TwoSnakesEffect(AudioReactiveEffect):
    """Yellow and orange snakes on purple — grow on each bounce."""

    registry_name = 'two_snakes'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        speed = (num_leds - 1) / TRAVERSE_PERIOD
        self._snakes = [
            Snake(0, 1, YELLOW, speed, num_leds),
            Snake(num_leds - 1, -1, ORANGE, speed, num_leds),
        ]

    @property
    def name(self):
        return "Two Snakes"

    @property
    def description(self):
        return "Yellow and orange snakes on purple — grow 1px each bounce."

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def render(self, dt: float) -> np.ndarray:
        for s in self._snakes:
            s.advance(dt)
            s.check_strip_ends()

        # Reset when snake fills the strip
        for s in self._snakes:
            if s.size >= self.num_leds:
                s.size = 1

        bg = np.array(BG, dtype=np.float64)
        frame = np.full((self.num_leds, 3), BG, dtype=np.float64)
        for s in self._snakes:
            color = np.array(s.color, dtype=np.float64)
            head = int(round(s.pos))
            # Only check pixels in range of snake + fade
            lo = max(0, head - s.size - FADE_PIXELS - 1)
            hi = min(self.num_leds, head + s.size + FADE_PIXELS + 2)
            for idx in range(lo, hi):
                b = s.brightness_at(idx)
                if b > 0:
                    frame[idx] = frame[idx] * (1.0 - b) + color * b
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def get_diagnostics(self) -> dict:
        a, b = self._snakes
        return {
            'yellow': f'{a.pos:.0f} {a.size}px {"R" if a.dir == 1 else "L"}',
            'orange': f'{b.pos:.0f} {b.size}px {"R" if b.dir == 1 else "L"}',
        }
