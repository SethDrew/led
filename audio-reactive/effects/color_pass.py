"""
Color Pass — three primary-colored parents spawn mix-color children.

Red, yellow, and blue parent blobs (8 LEDs each) glide back and forth on
a dark strip. When two parents first touch, a child blob in the
combination color is born at the meeting point and drifts off in a random
direction at a random speed until it exits the strip. Triple overlaps
spawn a brown child. Parents are unaffected and keep bouncing.

    red    + yellow         → orange
    yellow + blue           → green
    red    + blue           → purple
    red    + yellow + blue  → brown

Brightness pulses ±40% with RMS energy (EMA-normalized).

Usage:
    python runner.py color_pass --no-leds
    python runner.py color_pass
"""

import random
import threading
import numpy as np
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator
from color import NAMED_HUES, swatch


# Six swatches pulled from the validated OKLCH variable-L rainbow LUT
# (256-entry, hue-indexed). Brown isn't representable from the LUT alone —
# it needs low chroma at low L, which the current 1D LUT doesn't expose —
# so it stays hand-tuned. See ledger: oklch-perceptual-rainbow,
# oklch-color-solid-coverage.
RED    = swatch(NAMED_HUES["red"])       # idx 0
ORANGE = swatch(NAMED_HUES["orange"])    # idx 21   (R + Y)
YELLOW = swatch(NAMED_HUES["yellow"])    # idx 43
GREEN  = swatch(NAMED_HUES["green"])     # idx 85   (Y + B)
BLUE   = swatch(NAMED_HUES["blue"])      # idx 171
PURPLE = swatch(NAMED_HUES["purple"])    # idx 192  (R + B)
BROWN  = (42, 16, 13)                    # hand-tuned: low-L, low-C orange (R + Y + B)

PRIMARIES = [RED, YELLOW, BLUE]

BLOB_SIZE = 8
TRAVERSE_PERIOD = 12.0     # seconds for one end-to-end traversal
SPEED_VARIATION = 0.2      # ±20% so crossing points drift over time
PULSE_DEPTH = 0.40         # ±40% brightness with audio energy
EDGE_FADE = 1.5            # pixels of soft falloff at head/tail

# Spawned children
CHILD_SIZE = 4                      # half the parent length
CHILD_SPEED_RANGE = (0.5, 1.5)      # multiplier on parent base speed


class Blob:
    def __init__(self, pos, direction, speed, color, size, num_leds, bounces=True):
        self.pos = float(pos)
        self.dir = direction
        self.speed = speed
        self.color = np.array(color, dtype=np.float64)
        self.size = size
        self.num_leds = num_leds
        self.bounces = bounces

    def advance(self, dt):
        self.pos += self.dir * self.speed * dt

    def check_strip_ends(self):
        if not self.bounces:
            return
        half = (self.size - 1) / 2.0
        lo, hi = half, (self.num_leds - 1) - half
        if self.pos > hi:
            self.pos = hi - (self.pos - hi)
            self.dir = -1
        elif self.pos < lo:
            self.pos = lo + (lo - self.pos)
            self.dir = 1

    def is_off_strip(self):
        """For non-bouncing blobs: whether the blob has fully exited the strip."""
        half = (self.size - 1) / 2.0
        body_lo = self.pos - half - EDGE_FADE
        body_hi = self.pos + half + EDGE_FADE
        return body_hi < 0 or body_lo > self.num_leds - 1

    def weights_into(self, buf):
        """Write per-pixel coverage weights (0..1) for this blob into buf."""
        half = (self.size - 1) / 2.0
        body_lo = self.pos - half
        body_hi = self.pos + half
        lo = max(0, int(np.floor(body_lo - EDGE_FADE)))
        hi = min(self.num_leds - 1, int(np.ceil(body_hi + EDGE_FADE)))
        for idx in range(lo, hi + 1):
            if body_lo <= idx <= body_hi:
                buf[idx] = 1.0
            elif idx < body_lo:
                buf[idx] = max(0.0, 1.0 - (body_lo - idx) / EDGE_FADE) ** 2
            else:
                buf[idx] = max(0.0, 1.0 - (idx - body_hi) / EDGE_FADE) ** 2

    def body_overlaps(self, other):
        """True if the body ranges (no edge fade) of self and other overlap."""
        half_a = (self.size - 1) / 2.0
        half_b = (other.size - 1) / 2.0
        return abs(self.pos - other.pos) < (half_a + half_b)


class ColorPassEffect(AudioReactiveEffect):
    """Three primary-colored blobs pass through each other with hardcoded mix colors."""

    registry_name = 'color_pass'
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'RMS energy (EMA-normalized, blob brightness pulse)'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        base_speed = (num_leds - 1) / TRAVERSE_PERIOD
        # Three blobs spaced across the strip, alternating directions, slight speed variation.
        starts = [num_leds * 1/6, num_leds * 3/6, num_leds * 5/6]
        dirs = [1, -1, 1]
        self._blobs = []
        for pos, d, color in zip(starts, dirs, PRIMARIES):
            speed = base_speed * random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
            self._blobs.append(Blob(pos, d, speed, color, BLOB_SIZE, num_leds))

        # Precompute color arrays for the render path.
        self._c_primary = [np.array(c, dtype=np.float64) for c in PRIMARIES]
        self._c_ry = np.array(ORANGE, dtype=np.float64)
        self._c_yb = np.array(GREEN, dtype=np.float64)
        self._c_rb = np.array(PURPLE, dtype=np.float64)
        self._c_ryb = np.array(BROWN, dtype=np.float64)

        # Children: drifting mix-color blobs spawned on parent contact.
        self._children = []
        # Rising-edge state per pair and triple — only spawn on transition False→True.
        self._pair_overlapping = {(0, 1): False, (1, 2): False, (0, 2): False}
        self._triple_overlapping = False
        self._base_speed = base_speed

        # Audio: RMS with dual EMA (same as polycule)
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
        return "Color Pass"

    @property
    def description(self):
        return "Red, yellow, blue blobs pass through each other; overlaps paint orange/green/purple/brown."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self._accum.feed(mono_chunk):
            rms = float(np.sqrt(np.mean(frame ** 2)))
            self._slow_ema += self._slow_alpha * (rms - self._slow_ema)
            ratio = rms / self._slow_ema if self._slow_ema > 1e-10 else 0.0
            self._fast_ema += self._fast_alpha * (ratio - self._fast_ema)
            energy = np.clip((self._fast_ema - 0.5) / 1.5, 0.0, 1.0)
            with self._lock:
                self._energy = float(energy)

    def _spawn_child(self, indices, color):
        """Spawn a mix-color child at the centroid of the listed parents.

        Skips spawning if a child of this color is already alive on the strip
        — only one orange / green / purple / brown at a time.
        """
        color_arr = np.array(color, dtype=np.float64)
        for c in self._children:
            if np.array_equal(c.color, color_arr):
                return
        positions = [self._blobs[i].pos for i in indices]
        pos = sum(positions) / len(positions)
        direction = random.choice([-1, 1])
        speed = self._base_speed * random.uniform(*CHILD_SPEED_RANGE)
        self._children.append(Blob(
            pos, direction, speed, color, CHILD_SIZE,
            self.num_leds, bounces=False,
        ))

    def _detect_and_spawn(self):
        """Edge-triggered spawning: fire only on overlap-state rising edges."""
        a, b, c = self._blobs
        ab = a.body_overlaps(b)
        bc = b.body_overlaps(c)
        ac = a.body_overlaps(c)
        triple_now = ab and bc and ac

        if triple_now and not self._triple_overlapping:
            self._spawn_child([0, 1, 2], BROWN)
        # Pairwise spawns. Triple state covers all three pairs simultaneously,
        # so only spawn pair children when NOT already in a triple.
        elif ab and not self._pair_overlapping[(0, 1)]:
            self._spawn_child([0, 1], ORANGE)
        elif bc and not self._pair_overlapping[(1, 2)]:
            self._spawn_child([1, 2], GREEN)
        elif ac and not self._pair_overlapping[(0, 2)]:
            self._spawn_child([0, 2], PURPLE)

        self._pair_overlapping[(0, 1)] = ab
        self._pair_overlapping[(1, 2)] = bc
        self._pair_overlapping[(0, 2)] = ac
        self._triple_overlapping = triple_now

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = self._energy
        mult = 1.0 - PULSE_DEPTH + 2.0 * PULSE_DEPTH * energy

        # Advance parents (bouncing).
        for b in self._blobs:
            b.advance(dt)
            b.check_strip_ends()

        # Spawn new children on rising-edge contact.
        self._detect_and_spawn()

        # Advance children, drop ones that have fully exited.
        for c in self._children:
            c.advance(dt)
        self._children = [c for c in self._children if not c.is_off_strip()]

        # --- Render parents via overlap decomposition (frame stays float64) ---
        w_r = np.zeros(self.num_leds, dtype=np.float64)
        w_y = np.zeros(self.num_leds, dtype=np.float64)
        w_b = np.zeros(self.num_leds, dtype=np.float64)
        self._blobs[0].weights_into(w_r)
        self._blobs[1].weights_into(w_y)
        self._blobs[2].weights_into(w_b)

        triple = np.minimum(np.minimum(w_r, w_y), w_b)
        rr = w_r - triple
        yy = w_y - triple
        bb = w_b - triple
        ry = np.minimum(rr, yy)
        yb = np.minimum(yy, bb)
        rb = np.minimum(rr, bb)
        r_only = rr - ry - rb
        y_only = yy - ry - yb
        b_only = bb - yb - rb

        frame = (
            r_only[:, None] * self._c_primary[0]
            + y_only[:, None] * self._c_primary[1]
            + b_only[:, None] * self._c_primary[2]
            + ry[:, None] * self._c_ry
            + yb[:, None] * self._c_yb
            + rb[:, None] * self._c_rb
            + triple[:, None] * self._c_ryb
        ) * mult

        # --- Blend children on top (alpha blend by coverage weight) ---
        for child in self._children:
            cw = np.zeros(self.num_leds, dtype=np.float64)
            child.weights_into(cw)
            cw_col = cw[:, None]
            frame = (1.0 - cw_col) * frame + cw_col * child.color * mult

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = self._energy
        labels = ['red', 'yel', 'blu']
        d = {'energy': f'{energy:.2f}', 'kids': str(len(self._children))}
        for label, b in zip(labels, self._blobs):
            d[label] = f'{b.pos:.0f}{"R" if b.dir == 1 else "L"}'
        return d
