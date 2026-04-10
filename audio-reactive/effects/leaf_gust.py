"""
Leaf Gust — continuously drifting leaves with energy-driven speed and brightness.

A fixed pool of leaves drifts lazily along the sculpture branches. Audio
energy (RMS, EMA-normalized) continuously modulates the wind speed and
brightness of ALL leaves: quiet = gentle drift at 80% brightness, loud =
leaves surge forward with a 20% brightness pulse. Each onset feels like
pressing the gas pedal.

Same wind projection and topology as leaf_wind, different audio mapping:
proportional/continuous instead of accent/onset-driven.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from topology import SculptureTopology
from signals import OverlapFrameAccumulator


def _noise1d(pos, t, seed=0):
    """Cheap 1D time-varying noise. Returns roughly [-1, 1]."""
    return (np.sin(pos * 0.4 + t * 0.3 + seed * 7.3)
            * np.cos(pos * 0.17 - t * 0.19 + seed * 3.1)
            + np.sin(pos * 0.09 + t * 0.13 + seed * 1.7) * 0.5) / 1.5


class _Branch:
    __slots__ = ('start', 'end', 'length', 'spawn_end', 'exit_end')

    def __init__(self, start, end, spawn_end, exit_end):
        self.start = start
        self.end = end
        self.length = end - start + 1
        self.spawn_end = spawn_end
        self.exit_end = exit_end


class Leaf:
    __slots__ = ('pos', 'vel', 'color', 'brightness', 'age',
                 'branch_idx', 'ref_pos', 'ref_time')

    def __init__(self, pos, color, branch_idx):
        self.pos = pos
        self.vel = 0.0
        self.color = color
        self.brightness = 0.0
        self.age = 0.0
        self.branch_idx = branch_idx
        self.ref_pos = pos
        self.ref_time = 0.0


class LeafGustEffect(AudioReactiveEffect):
    """Continuously drifting leaves with energy-driven speed and brightness."""

    registry_name = 'leaf_gust'
    handles_topology = True
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'RMS energy (EMA-normalized, continuous)'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 sculpture_id: str = None,
                 # Wind
                 wind_speed: float = 5.0,        # base speed (quiet) in LEDs/s
                 wind_angle_deg: float = 112.0,
                 turbulence: float = 0.5,
                 noise_scale: float = 1.0,
                 damping: float = 0.92,
                 # Energy → speed/brightness
                 speed_boost: float = 3.0,        # at max energy: speed * (1 + boost)
                 brightness_pulse: float = 0.30,   # brightness range: (1-pulse) to 1.0
                 energy_ema_tc: float = 2.0,       # slow EMA for normalization
                 energy_smooth_tc: float = 0.1,    # fast EMA for responsiveness
                 # Leaves
                 num_leaves: int = 6,
                 leaf_sigma: float = 1.5,
                 fade_in_time: float = 1.0,
                 fade_out_leds: float = 10.0,
                 max_brightness: float = 0.85,
                 stall_radius: float = 3.0,
                 stall_timeout: float = 3.0,
                 ):
        super().__init__(num_leds, sample_rate)

        sid = sculpture_id or 'cob_diamond'
        topo = SculptureTopology(sid)
        self.num_leds = topo.num_leds
        n = self.num_leds
        coords = topo.coords

        branch_ranges = list(topo.branches.values())

        # Per-LED tangent
        tangents = np.zeros((n, 2))
        for bstart, bend in branch_ranges:
            for i in range(bstart, bend + 1):
                if i == bstart:
                    tangents[i] = coords[min(i + 1, bend)] - coords[i]
                elif i == bend:
                    tangents[i] = coords[i] - coords[max(i - 1, bstart)]
                else:
                    tangents[i] = coords[i + 1] - coords[i - 1]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        tangents /= norms

        # Wind projection + edge-safe smoothing + floor
        angle_rad = np.radians(wind_angle_deg)
        wind_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        self._wind_proj = (tangents @ wind_dir).astype(np.float64)

        min_proj = 0.25
        for bstart, bend in branch_ranges:
            seg = self._wind_proj[bstart:bend + 1].copy()
            k = min(7, len(seg))
            if k >= 3:
                pad = k // 2
                padded = np.pad(seg, pad, mode='edge')
                seg = np.convolve(padded, np.ones(k) / k, mode='valid')
            avg = np.mean(seg)
            if avg >= 0:
                seg = np.maximum(seg, min_proj)
            else:
                seg = np.minimum(seg, -min_proj)
            self._wind_proj[bstart:bend + 1] = seg

        # Build branches
        self._branches: list[_Branch] = []
        for bstart, bend in branch_ranges:
            avg = np.mean(self._wind_proj[bstart:bend + 1])
            if avg >= 0:
                self._branches.append(_Branch(bstart, bend, bstart, bend))
            else:
                self._branches.append(_Branch(bstart, bend, bend, bstart))

        nb = len(self._branches)
        self._branch_weights = np.ones(nb) / nb

        # Wind params
        self._wind_speed = wind_speed
        self._speed_boost = speed_boost
        self._brightness_pulse = brightness_pulse
        self._turbulence = turbulence
        self._noise_scale = noise_scale
        self._damping = damping

        # Rendering params
        self._num_leaves = num_leaves
        self._leaf_sigma = leaf_sigma
        self._sigma_sq2 = 2.0 * leaf_sigma ** 2
        self._fade_in_time = fade_in_time
        self._fade_out_leds = fade_out_leds
        self._max_brightness = max_brightness
        self._stall_radius = stall_radius
        self._stall_timeout = stall_timeout

        self._led_indices = np.arange(n, dtype=np.float64)

        # Autumn palette
        self._palette = np.array([
            [255, 140, 20],
            [240, 100, 10],
            [220, 60,  5],
            [200, 40,  10],
            [180, 30,  5],
            [255, 180, 40],
            [160, 25,  5],
        ], dtype=np.float64)

        # Audio analysis — RMS with dual EMA (slow for normalization, fast for response)
        self._n_fft = 2048
        self._hop = 512
        self._accum = OverlapFrameAccumulator(frame_len=self._n_fft, hop=self._hop)
        fps = sample_rate / self._hop
        self._slow_ema = 1e-6          # tracks average RMS level
        self._slow_alpha = 2.0 / (energy_ema_tc * fps + 1.0)
        self._fast_ema = 0.0           # tracks recent RMS for responsiveness
        self._fast_alpha = 2.0 / (energy_smooth_tc * fps + 1.0)

        # Energy value shared between threads (0-1, EMA-normalized)
        self._energy = 0.0
        self._lock = threading.Lock()

        # Init leaves staggered across branches
        self._leaves: list[Leaf] = []
        self._time = 0.0
        self._leaf_counter = 0
        for i in range(num_leaves):
            self._leaves.append(self._spawn_leaf(stagger=True))

        self._frame_buf = np.zeros((n, 3), dtype=np.uint8)

    def _spawn_leaf(self, stagger=False) -> Leaf:
        """Spawn a leaf on a random branch."""
        bi = int(np.random.choice(len(self._branches), p=self._branch_weights))
        branch = self._branches[bi]
        color = self._palette[np.random.randint(len(self._palette))]

        if stagger:
            pos = np.random.uniform(branch.start, branch.end)
            leaf = Leaf(pos, color, bi)
            leaf.brightness = 1.0
            leaf.age = self._fade_in_time + 1.0
        else:
            direction = 1 if branch.spawn_end == branch.start else -1
            pos = branch.spawn_end - direction * np.random.uniform(0, 2)
            leaf = Leaf(pos, color, bi)

        self._leaf_counter += 1
        return leaf

    @property
    def name(self):
        return "Leaf Gust"

    @property
    def description(self):
        return ("Autumn leaves drift lazily, surging forward and brightening "
                "with audio energy — each onset presses the gas.")

    # ── Audio thread ──

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self._accum.feed(mono_chunk):
            rms = float(np.sqrt(np.mean(frame ** 2)))

            # Slow EMA: tracks the average level for normalization
            self._slow_ema += self._slow_alpha * (rms - self._slow_ema)

            # Normalize: ratio of current RMS to average
            # >1 when louder than average, <1 when quieter
            ratio = rms / self._slow_ema if self._slow_ema > 1e-10 else 0.0

            # Fast EMA: smooth the ratio for responsive but non-jittery output
            self._fast_ema += self._fast_alpha * (ratio - self._fast_ema)

            # Map to 0-1: ratio of 0.5 → 0, ratio of 2.0 → 1
            energy = np.clip((self._fast_ema - 0.5) / 1.5, 0.0, 1.0)

            with self._lock:
                self._energy = float(energy)

    # ── Render thread ──

    def render(self, dt: float) -> np.ndarray:
        dt = min(dt, 0.1)
        self._time += dt
        t = self._time
        ns = self._noise_scale
        n = self.num_leds

        with self._lock:
            energy = self._energy

        # Energy-driven speed and brightness
        speed_mult_global = 1.0 + energy * self._speed_boost
        bright_mult = (1.0 - self._brightness_pulse) + self._brightness_pulse * energy

        # Update leaves
        dead = []
        for i, leaf in enumerate(self._leaves):
            branch = self._branches[leaf.branch_idx]

            # Uniform speed — all leaves move at the same rate,
            # direction determined by branch orientation
            direction = 1.0 if branch.spawn_end == branch.start else -1.0
            leaf.vel = self._wind_speed * speed_mult_global * direction
            leaf.pos += leaf.vel * dt
            leaf.age += dt

            # Stall detection
            elapsed = leaf.age - leaf.ref_time
            if elapsed > self._stall_timeout:
                if abs(leaf.pos - leaf.ref_pos) < self._stall_radius:
                    dead.append(i)
                    continue
                leaf.ref_pos = leaf.pos
                leaf.ref_time = leaf.age

            # Fade in
            if leaf.age < self._fade_in_time:
                leaf.brightness = leaf.age / self._fade_in_time
            else:
                leaf.brightness = 1.0

            # Fade out near exit
            if self._fade_out_leds > 0:
                if branch.exit_end >= branch.spawn_end:
                    dist_to_exit = branch.exit_end - leaf.pos
                else:
                    dist_to_exit = leaf.pos - branch.exit_end
                if dist_to_exit < self._fade_out_leds:
                    leaf.brightness *= max(dist_to_exit / self._fade_out_leds, 0.0)

            # Past branch bounds — respawn
            margin = self._leaf_sigma * 3
            if leaf.pos > branch.end + margin or leaf.pos < branch.start - margin:
                dead.append(i)

        # Replace dead leaves (maintain constant pool)
        for i in sorted(dead, reverse=True):
            self._leaves[i] = self._spawn_leaf()

        # Render
        glow = np.zeros(n, dtype=np.float64)
        colors = np.zeros((n, 3), dtype=np.float64)

        for leaf in self._leaves:
            dist_sq = (self._led_indices - leaf.pos) ** 2
            intensity = np.exp(-dist_sq / self._sigma_sq2) * leaf.brightness
            glow += intensity
            colors += intensity[:, np.newaxis] * leaf.color[np.newaxis, :]

        mask = glow > 1e-6
        colors[mask] /= glow[mask, np.newaxis]
        colors[~mask] = 0

        brightness = np.clip(glow, 0, 1) * self._max_brightness * bright_mult
        result = colors * brightness[:, np.newaxis]

        self._frame_buf[:] = np.clip(result, 0, 255).astype(np.uint8)
        return self._frame_buf.copy()

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = self._energy
        info = []
        names = ['L', 'R', 'M']
        for leaf in self._leaves:
            bname = names[leaf.branch_idx] if leaf.branch_idx < 3 else '?'
            branch = self._branches[leaf.branch_idx]
            total = abs(branch.exit_end - branch.spawn_end)
            pct = int(abs(leaf.pos - branch.spawn_end) / max(total, 1) * 100)
            info.append(f'{bname}{pct}%')
        speed = self._wind_speed * (1 + energy * self._speed_boost)
        return {
            'leaves': ' '.join(info),
            'energy': f'{energy:.2f}',
            'speed': f'{speed:.1f}',
            'count': str(len(self._leaves)),
        }
