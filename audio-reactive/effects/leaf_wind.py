"""
Leaf Wind — onset-driven particles drifting along the physical LED strip.

Audio onsets feed an energy reservoir. When it crosses a rising threshold
(reluctant eagerness), a leaf spawns on an available branch — one that
has no leaf still in the first half. The leaf bursts on fast (proportional
to accumulated energy) then decelerates to the prevailing wind pace.

Wind direction is projected onto each LED's local strip tangent so
leaves always follow the wire path in the correct physical direction.
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
    __slots__ = ('pos', 'vel', 'boost', 'color', 'brightness', 'age',
                 'branch_idx', 'ref_pos', 'ref_time')

    def __init__(self, pos, color, branch_idx, boost=0.0):
        self.pos = pos
        self.vel = 0.0
        self.boost = boost
        self.color = color
        self.brightness = 0.0
        self.age = 0.0
        self.branch_idx = branch_idx
        self.ref_pos = pos
        self.ref_time = 0.0


class LeafWindEffect(AudioReactiveEffect):
    """Onset-driven leaves drifting along sculpture branches."""

    registry_name = 'leaf_wind'
    handles_topology = True
    ref_pattern = 'accent'
    ref_scope = 'beat'
    ref_input = 'RMS delta onsets (EMA-normalized, reluctant eagerness)'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 sculpture_id: str = None,
                 # Wind
                 wind_speed: float = 9.0,
                 wind_angle_deg: float = 112.0,
                 turbulence: float = 0.5,
                 noise_scale: float = 1.0,
                 damping: float = 0.92,
                 # Onset detection
                 onset_ratio: float = 3.0,
                 onset_ema_tc: float = 0.5,
                 onset_min_gap_s: float = 0.08,
                 # Spawning (reluctant eagerness)
                 max_leaves: int = 15,
                 spawn_threshold: float = 0.15,     # base reservoir level to spawn
                 reluctance_bump: float = 1.3,       # threshold multiplier after each spawn
                 reluctance_tc: float = 1.0,         # seconds for reluctance to decay back
                 reservoir_tc: float = 3.0,          # seconds for unused energy to fade
                 branch_clear_pct: float = 0.4,      # leaf must be this far before re-spawn
                 # Boost
                 boost_speed: float = 25.0,
                 boost_tc: float = 2.0,
                 # Rendering
                 leaf_sigma: float = 1.5,
                 fade_in_time: float = 0.3,
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

        # Branch ranges from topology
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
                # Edge-pad before convolving so short branches don't lose
                # projection strength at their ends
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

        total_len = sum(b.length for b in self._branches)
        self._branch_weights = np.array([b.length / total_len
                                         for b in self._branches])

        # Wind params
        self._wind_speed = wind_speed
        self._turbulence = turbulence
        self._noise_scale = noise_scale
        self._damping = damping

        # Spawning params
        self._max_leaves = max_leaves
        self._spawn_threshold = spawn_threshold
        self._reluctance_bump = reluctance_bump
        self._reluctance_tc = reluctance_tc
        self._reservoir_tc = reservoir_tc
        self._branch_clear_pct = branch_clear_pct
        self._boost_speed = boost_speed
        self._boost_tc = boost_tc

        # Rendering params
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
            [255, 140, 20],   # golden amber
            [240, 100, 10],   # warm orange
            [220, 60,  5],    # burnt orange
            [200, 40,  10],   # deep orange-red
            [180, 30,  5],    # rust
            [255, 180, 40],   # light gold
            [160, 25,  5],    # dark rust
        ], dtype=np.float64)

        # Audio analysis — EMA-based onset detection
        self._n_fft = 2048
        self._hop = 512
        self._accum = OverlapFrameAccumulator(frame_len=self._n_fft, hop=self._hop)
        self._prev_rms = 0.0
        self._delta_ema = 1e-6
        fps = sample_rate / self._hop
        self._delta_ema_alpha = 2.0 / (onset_ema_tc * fps + 1.0)
        self._onset_ratio = onset_ratio
        self._onset_min_gap_frames = max(1, int(onset_min_gap_s * fps))
        self._frames_since_onset = self._onset_min_gap_frames

        # Shared between audio and render threads
        self._pending_energy = 0.0  # audio thread writes, render drains
        self._lock = threading.Lock()

        # Reluctant eagerness state (render thread only)
        self._reservoir = 0.0
        self._reluctance = spawn_threshold  # current threshold, decays back to base

        # Dynamic leaf list
        self._leaves: list[Leaf] = []
        self._time = 0.0
        self._leaf_counter = 0

        self._frame_buf = np.zeros((n, 3), dtype=np.uint8)

    def _branch_available(self, bi: int) -> bool:
        """True if no leaf on this branch is still in the first portion."""
        branch = self._branches[bi]
        total = abs(branch.exit_end - branch.spawn_end)
        if total < 1:
            return True
        for leaf in self._leaves:
            if leaf.branch_idx != bi:
                continue
            traveled = abs(leaf.pos - branch.spawn_end)
            if traveled / total < self._branch_clear_pct:
                return False
        return True

    def _spawn_leaf_on(self, bi: int, strength: float) -> Leaf:
        """Spawn a leaf on a specific branch."""
        branch = self._branches[bi]
        color = self._palette[np.random.randint(len(self._palette))]

        direction = 1 if branch.spawn_end == branch.start else -1
        pos = branch.spawn_end - direction * np.random.uniform(0, 2)

        boost = min(strength, 1.0) * self._boost_speed * direction
        leaf = Leaf(pos, color, bi, boost=boost)
        self._leaf_counter += 1
        return leaf

    @property
    def name(self):
        return "Leaf Wind"

    @property
    def description(self):
        return ("Onset-driven autumn leaves — energy builds, reluctantly "
                "spawns a leaf that bursts fast then settles to wind pace.")

    # ── Audio thread ──

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self._accum.feed(mono_chunk):
            rms = float(np.sqrt(np.mean(frame ** 2)))
            delta = abs(rms - self._prev_rms)
            self._prev_rms = rms

            self._delta_ema += self._delta_ema_alpha * (delta - self._delta_ema)
            self._frames_since_onset += 1

            ratio = delta / self._delta_ema if self._delta_ema > 1e-10 else 0.0
            if (ratio > self._onset_ratio
                    and self._frames_since_onset >= self._onset_min_gap_frames):
                self._frames_since_onset = 0
                strength = min((ratio - self._onset_ratio)
                               / self._onset_ratio, 1.0)
                with self._lock:
                    self._pending_energy += strength

    # ── Render thread ──

    def render(self, dt: float) -> np.ndarray:
        dt = min(dt, 0.1)
        self._time += dt
        t = self._time
        ns = self._noise_scale
        n = self.num_leds

        # Drain pending energy from audio thread into reservoir
        with self._lock:
            pending = self._pending_energy
            self._pending_energy = 0.0
        self._reservoir += pending

        # Decay reservoir (unused energy fades) and reluctance (back to base)
        self._reservoir *= np.exp(-dt / self._reservoir_tc)
        self._reluctance += (self._spawn_threshold - self._reluctance) * (
            1.0 - np.exp(-dt / self._reluctance_tc))

        # Reluctant eagerness: spawn count proportional to how far
        # reservoir exceeds reluctance.  1x → 1 leaf, 2x → 2, 3x → 3.
        if self._reservoir > self._reluctance:
            ratio = self._reservoir / self._reluctance
            num_to_spawn = min(int(ratio), 3)  # cap at 3 per burst

            available = [i for i in range(len(self._branches))
                         if self._branch_available(i)]
            np.random.shuffle(available)

            spawned = 0
            for bi in available:
                if spawned >= num_to_spawn:
                    break
                if len(self._leaves) >= self._max_leaves:
                    break
                strength = min(self._reservoir, 1.0)
                self._leaves.append(self._spawn_leaf_on(bi, strength))
                spawned += 1

            if spawned > 0:
                self._reservoir -= self._reluctance
                self._reluctance *= self._reluctance_bump ** spawned

        # Boost decay factor for this frame
        boost_decay = np.exp(-dt / self._boost_tc)

        # Update leaves
        dead = []
        for i, leaf in enumerate(self._leaves):
            branch = self._branches[leaf.branch_idx]

            idx = int(np.clip(leaf.pos, branch.start, branch.end))
            local_proj = self._wind_proj[idx]

            noise_val = _noise1d(leaf.pos * ns, t,
                                 seed=self._leaf_counter + i)
            speed_mult = 1.0 + noise_val * self._turbulence
            speed_mult = max(speed_mult, 0.05)

            force = local_proj * self._wind_speed * speed_mult
            leaf.vel = leaf.vel * self._damping + force * (1.0 - self._damping)

            leaf.boost *= boost_decay
            leaf.pos += (leaf.vel + leaf.boost) * dt
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

            # Fade out near exit (if enabled)
            if self._fade_out_leds > 0:
                if branch.exit_end >= branch.spawn_end:
                    dist_to_exit = branch.exit_end - leaf.pos
                else:
                    dist_to_exit = leaf.pos - branch.exit_end
                if dist_to_exit < self._fade_out_leds:
                    leaf.brightness *= max(dist_to_exit / self._fade_out_leds, 0.0)

            # Past branch bounds — let leaf drift off before cleanup (~3σ)
            margin = self._leaf_sigma * 3
            if leaf.pos > branch.end + margin or leaf.pos < branch.start - margin:
                dead.append(i)

        for i in sorted(dead, reverse=True):
            self._leaves.pop(i)

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

        brightness = np.clip(glow, 0, 1) * self._max_brightness
        result = colors * brightness[:, np.newaxis]

        self._frame_buf[:] = np.clip(result, 0, 255).astype(np.uint8)
        return self._frame_buf.copy()

    def get_diagnostics(self) -> dict:
        info = []
        names = ['L', 'R', 'M']
        for leaf in self._leaves:
            bname = names[leaf.branch_idx] if leaf.branch_idx < 3 else '?'
            branch = self._branches[leaf.branch_idx]
            total = abs(branch.exit_end - branch.spawn_end)
            pct = int(abs(leaf.pos - branch.spawn_end) / max(total, 1) * 100)
            info.append(f'{bname}{pct}%')
        return {
            'leaves': ' '.join(info) if info else '—',
            'reservoir': f'{self._reservoir:.2f}',
            'reluctance': f'{self._reluctance:.2f}',
            'count': str(len(self._leaves)),
        }
