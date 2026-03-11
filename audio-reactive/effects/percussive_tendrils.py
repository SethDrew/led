"""
Percussive Tendrils — beat-reactive tentacles crawling through the diamond.

Each percussive hit spawns a tendril that expands along the wire path from
a chosen origin LED. Tendril length scales with hit amplitude. Origins
drift slowly so all LEDs get used over time.

Colors live within a palette that rotates through the hue wheel over ~60s,
with spatial hue offsets along each tendril to create a rotation illusion.

Uses streaming HPSS (from band_zone_pulse pattern) to isolate percussive
transients, then spawns tendrils on confirmed hits.

Usage:
    python runner.py percussive_tendrils --sculpture cob_diamond --no-leds
    python runner.py percussive_tendrils --sculpture cob_diamond
"""

import math
import threading
import numpy as np
from base import AudioReactiveEffect
from topology import SculptureTopology
from signals import OverlapFrameAccumulator


# ── Tendril constants ──
MAX_TENDRILS = 8
TENDRIL_MIN_LENGTH = 3       # LEDs for weakest hit
TENDRIL_MAX_LENGTH = 25      # LEDs for strongest hit
TENDRIL_EXPAND_SPEED = 40.0  # LEDs per second expansion rate
TENDRIL_DECAY = 0.82         # per-frame (at 30fps) brightness decay — fast fade
TENDRIL_TRAIL_FALLOFF = 0.80 # brightness multiplier per LED behind head
TENDRIL_MAX_BRIGHTNESS = 0.6 # cap peak brightness to reduce flashiness

# ── Color rotation ──
HUE_CYCLE_PERIOD = 20.0      # seconds for full color wheel rotation
PALETTE_WIDTH = 0.15          # hue spread within active palette (0-1)
SPATIAL_HUE_SPREAD = 0.08    # hue offset along tendril length

# ── Origin drift ──
ORIGIN_DRIFT_INTERVAL = 8.0  # seconds between origin changes


def hsv_to_rgb(h, s, v):
    """Convert HSV (all 0-1) to RGB (0-255) as uint8 array."""
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return np.array([r * 255, g * 255, b * 255], dtype=np.float32)


class PercussiveTendrilsEffect(AudioReactiveEffect):
    """Percussive hits spawn expanding tendrils along the diamond's wire paths."""

    registry_name = 'percussive_tendrils'
    handles_topology = True

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.topo = SculptureTopology('cob_diamond')
        self.num_leds = self.topo.num_leds

        # ── Build wire-path adjacency ──
        self.adjacency = self._build_adjacency()

        # ── Audio processing ──
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        self.accum = OverlapFrameAccumulator(frame_len=self.n_fft, hop=512)

        # Streaming HPSS buffer
        self.hpss_buf_size = 17
        self.spec_buf = np.zeros((self.hpss_buf_size, self.n_fft // 2 + 1),
                                 dtype=np.float32)
        self.spec_buf_idx = 0
        self.spec_buf_filled = 0

        # Percussive energy tracking
        self.perc_history_len = 8
        self.perc_history = np.zeros(self.perc_history_len, dtype=np.float32)
        self.perc_hist_idx = 0
        self.perc_hist_filled = 0
        self.perc_peak = 1e-10
        self.perc_peak_decay = 0.998
        self.perc_cooldown = 0  # frames remaining
        self.perc_cooldown_frames = 5

        # ── Shared state (audio → render) ──
        self._pending_hits = []  # list of float strengths 0-1
        self._lock = threading.Lock()

        # ── Tendril state (render thread only) ──
        self.tendrils = []  # list of tendril dicts

        # ── Origin management ──
        # Pre-compute all possible origins (every LED)
        self._all_origins = list(range(self.num_leds))
        self._origin_pool = list(self._all_origins)
        np.random.shuffle(self._origin_pool)
        self._origin_idx = 0
        self._origin_timer = 0.0

        # ── Color state ──
        self._hue_phase = np.random.random()  # start at random hue
        self._time_acc = 0.0

        # ── Pot control ──
        self._pot_scale = 1.0
        self._pot_decay = TENDRIL_DECAY

        # Per-LED brightness buffer (render thread)
        self._led_brightness = np.zeros(self.num_leds, dtype=np.float32)
        self._led_color = np.zeros((self.num_leds, 3), dtype=np.float32)

    @property
    def name(self):
        return "Percussive Tendrils"

    @property
    def description(self):
        return ("Percussive hits spawn expanding tendrils along diamond wire "
                "paths; palette rotates through color wheel.")

    def set_pot_value(self, raw):
        """Map pot 0-1023 to tendril length scale 0.5x-8.0x and decay 0.78-0.96."""
        t = raw / 1023.0
        self._pot_scale = 0.5 + t * 7.5
        # Higher pot = slower decay (tendrils linger longer)
        self._pot_decay = 0.78 + t * 0.18  # 0.78 (fast fade) → 0.96 (slow fade)

    # ── Wire-path adjacency ─────────────────────────────────────────

    def _build_adjacency(self):
        """Build adjacency list from branch wiring + cross-branch connections.

        Each LED connects to its neighbors within the same branch (i-1, i+1),
        plus cross-branch connections at physical junctions.
        """
        adj = [set() for _ in range(self.num_leds)]

        # Within-branch adjacency
        branches = self.topo.branches  # {'left': (0, 41), 'right': (42, 61), ...}
        for name, (start, end) in branches.items():
            for i in range(start, end + 1):
                if i > start:
                    adj[i].add(i - 1)
                if i < end:
                    adj[i].add(i + 1)

        # Cross-branch connections at physical junctions
        lm = self.topo.landmarks

        # Base convergence: LEDs 0, 61, 62 are all at the base
        base_leds = [lm.get('base_start', 0),
                     lm.get('base_junction', 61),
                     62]  # middle branch start
        for a in base_leds:
            for b in base_leds:
                if a != b:
                    adj[a].add(b)

        # Apex: left branch end (41) meets right branch start (42)
        # and middle branch end (71)
        apex_leds = [lm.get('apex_left', 41),
                     lm.get('apex_right', 42),
                     71]  # middle end
        for a in apex_leds:
            for b in apex_leds:
                if a != b:
                    adj[a].add(b)

        # Right branch crosses middle at crossover (46/66)
        cross_a = lm.get('crossover', 46)
        adj[cross_a].add(66)
        adj[66].add(cross_a)

        # Right branch self-loop (52/57 same physical position)
        adj[52].add(57)
        adj[57].add(52)

        # Near-neighbors: LEDs 4 and 63
        adj[4].add(63)
        adj[63].add(4)

        # Convert to sorted lists for deterministic traversal
        return [sorted(s) for s in adj]

    # ── BFS path expansion ──────────────────────────────────────────

    def _bfs_path(self, origin, max_length):
        """BFS from origin, returning LEDs in visit order up to max_length.

        Returns list of (led_index, distance_in_hops) pairs.
        Randomizes neighbor order slightly for organic feel.
        """
        visited = set()
        visited.add(origin)
        queue = [(origin, 0)]
        result = [(origin, 0)]

        i = 0
        while i < len(queue) and len(result) < max_length:
            led, dist = queue[i]
            i += 1

            neighbors = list(self.adjacency[led])
            # Slight shuffle for organic branching
            if len(neighbors) > 1:
                np.random.shuffle(neighbors)

            for n in neighbors:
                if n not in visited and len(result) < max_length:
                    visited.add(n)
                    queue.append((n, dist + 1))
                    result.append((n, dist + 1))

        return result

    # ── Origin selection ────────────────────────────────────────────

    def _next_origin(self):
        """Pick next origin, cycling through all LEDs."""
        if self._origin_idx >= len(self._origin_pool):
            self._origin_pool = list(self._all_origins)
            np.random.shuffle(self._origin_pool)
            self._origin_idx = 0
        origin = self._origin_pool[self._origin_idx]
        self._origin_idx += 1
        return origin

    # ── Audio processing ────────────────────────────────────────────

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame):
        spec = np.abs(np.fft.rfft(frame * self.window))

        # Streaming HPSS
        self.spec_buf[self.spec_buf_idx] = spec
        self.spec_buf_idx = (self.spec_buf_idx + 1) % self.hpss_buf_size
        self.spec_buf_filled = min(self.spec_buf_filled + 1, self.hpss_buf_size)

        if self.spec_buf_filled >= 3:
            buf_slice = self.spec_buf[:self.spec_buf_filled]
            harmonic_mask = np.median(buf_slice, axis=0)
            percussive = np.maximum(spec - harmonic_mask, 0)
        else:
            percussive = spec * 0.5

        # Total percussive energy
        perc_energy = float(np.sum(percussive ** 2))

        # Track peak
        self.perc_peak = max(perc_energy, self.perc_peak * self.perc_peak_decay)

        # History ring for adaptive threshold
        h_idx = self.perc_hist_idx % self.perc_history_len
        self.perc_history[h_idx] = perc_energy
        self.perc_hist_idx += 1
        self.perc_hist_filled = min(self.perc_hist_filled + 1,
                                     self.perc_history_len)

        # Cooldown
        self.perc_cooldown = max(0, self.perc_cooldown - 1)

        if self.perc_hist_filled >= 3 and self.perc_cooldown == 0:
            hist = self.perc_history[:self.perc_hist_filled]
            mean = np.mean(hist)
            std = np.std(hist)
            threshold = mean + 2.5 * std
            floor = self.perc_peak * 0.05

            if perc_energy > threshold and perc_energy > floor:
                excess = perc_energy - threshold
                headroom = self.perc_peak - threshold
                strength = float(np.clip(excess / (headroom + 1e-10), 0.3, 1.0))
                with self._lock:
                    self._pending_hits.append(strength)
                self.perc_cooldown = self.perc_cooldown_frames

    # ── Rendering ───────────────────────────────────────────────────

    def render(self, dt: float) -> np.ndarray:
        step = dt * 30  # normalize to 30fps

        # Advance time
        self._time_acc += dt
        self._hue_phase = (self._time_acc / HUE_CYCLE_PERIOD) % 1.0

        # Advance origin drift timer
        self._origin_timer += dt

        # Drain pending hits → spawn tendrils
        with self._lock:
            hits = self._pending_hits[:]
            self._pending_hits.clear()

        for strength in hits:
            self._spawn_tendril(strength)

        # Update tendrils
        alive = []
        for t in self.tendrils:
            # Expand head
            t['current_length'] += TENDRIL_EXPAND_SPEED * dt
            t['current_length'] = min(t['current_length'], t['max_length'])

            # Decay overall brightness (pot controls fade speed)
            t['brightness'] *= self._pot_decay ** step

            # Kill if too dim
            if t['brightness'] < 0.03:
                continue
            alive.append(t)
        self.tendrils = alive

        # Render frame
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        for t in self.tendrils:
            path = t['path']  # list of (led, hop_distance)
            n_lit = int(t['current_length'])
            base_hue = t['hue']

            for i in range(min(n_lit, len(path))):
                led, hop = path[i]

                # Brightness: decays along tendril from head
                # Head is the most recently reached LED
                dist_from_head = n_lit - 1 - i
                spatial_fade = TENDRIL_TRAIL_FALLOFF ** dist_from_head
                brightness = t['brightness'] * spatial_fade * t['strength']

                # Color: base hue + spatial offset along tendril
                tendril_frac = i / max(len(path) - 1, 1)
                hue = (base_hue + tendril_frac * SPATIAL_HUE_SPREAD) % 1.0

                # Saturation varies slightly with position
                sat = 0.85 + 0.15 * (1.0 - tendril_frac)

                color = hsv_to_rgb(hue, sat, min(brightness, TENDRIL_MAX_BRIGHTNESS))

                # Additive blend
                frame[led] += color

        return np.clip(frame, 0, 255).astype(np.uint8)

    def _spawn_tendril(self, strength):
        """Spawn a new tendril from the current origin."""
        # Pick origin: use next from pool, or advance if timer elapsed
        if self._origin_timer >= ORIGIN_DRIFT_INTERVAL:
            self._origin_timer = 0.0
        origin = self._next_origin()

        # Tendril length scales with strength × pot multiplier
        base_len = TENDRIL_MIN_LENGTH + strength * (TENDRIL_MAX_LENGTH - TENDRIL_MIN_LENGTH)
        max_len = int(base_len * self._pot_scale)
        max_len = max(2, min(max_len, self.num_leds))  # clamp to sane range

        # Pre-compute BFS path
        path = self._bfs_path(origin, max_len)

        # Hue: current palette center + small random offset
        hue = (self._hue_phase +
               np.random.uniform(-PALETTE_WIDTH / 2, PALETTE_WIDTH / 2)) % 1.0

        tendril = {
            'origin': origin,
            'path': path,
            'max_length': max_len,
            'current_length': 1.0,  # starts at origin, expands
            'brightness': 1.0,
            'strength': strength,
            'hue': hue,
        }

        self.tendrils.append(tendril)
        if len(self.tendrils) > MAX_TENDRILS:
            # Remove oldest (dimmest first would be better but oldest is simpler)
            self.tendrils.pop(0)

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        n = len(self.tendrils)
        hue_deg = int(self._hue_phase * 360)
        origins = [t['origin'] for t in self.tendrils[:4]]
        return {
            'tendrils': n,
            'hue': f'{hue_deg}°',
            'pot': f'{self._pot_scale:.1f}x',
            'origins': str(origins),
        }
