"""
Heat Diffusion — 1D/graph thermal simulation driven by audio energy.

Audio energy (via sticky floor normalization) continuously adds heat at
a rotating injection point. The heat equation spreads it to neighbors.
A sharp energy spike creates a bright hot spot that softens and widens as
it diffuses outward, getting dimmer as it spreads.

Supports two modes:
  - Strip mode (default): 1D Laplacian, injection rotates along LED indices.
  - Sculpture mode (--sculpture): graph Laplacian from physical distance
    matrix, injection follows the sculpture's perimeter rotation path.

Physics: ∂T/∂t = α ∇²T - cooling * T + source(x)

Color: black body inspired — black → deep red → orange → white at max heat.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, StickyFloorRMS, OnsetTempoTracker


class HeatDiffusionEffect(AudioReactiveEffect):
    """Heat equation PDE with rotating source, optional topology awareness."""

    registry_name = 'heat_diffusion'
    handles_topology = True  # render in physical space — rotation follows strip order
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'RMS amplitude (sticky floor)'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 # --- Thermal physics ---
                 diffusivity: float = 40.0,
                 cooling: float = 4.0,
                 substeps: int = 6,
                 # --- Rotating source ---
                 rotation_period: float = 10.0,
                 inj_sigma: float = 3.0,
                 # --- Sculpture ---
                 sculpture_id: str = None,
                 neighbor_radius: float = 0.12,
                 # --- Appearance ---
                 heat_gain: float = 15.0,
                 gamma: float = 1.0,
                 max_brightness: float = 0.95,
                 debug_gradient: bool = False,
                 ):
        """
        Args:
            diffusivity:     Thermal diffusivity α. Higher = faster spread.
            cooling:         Radiative cooling rate (1/s). Prevents saturation.
            substeps:        PDE substeps per frame (stability).
            rotation_period: Seconds for the injection point to complete one
                             full cycle around the strip/sculpture perimeter.
            sculpture_id:    If set, loads topology from sculptures.json and
                             uses graph Laplacian + perimeter rotation path.
                             E.g. 'cob_diamond'. None = plain 1D strip.
            neighbor_radius: For graph mode: physical distance threshold for
                             neighbor connectivity. Smaller = sparser graph.
            heat_gain:       Multiplier for heat injection per unit energy.
            gamma:           Brightness curve for LEDs.
            max_brightness:  Brightness cap (0-1). 0.7 = 70% max.
        """
        super().__init__(num_leds, sample_rate)

        self._use_graph = False

        # --- Audio analysis ---
        self.n_fft = 2048
        self.hop_length = 512
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self._sticky = StickyFloorRMS(fps=sample_rate / self.hop_length,
                                      floor_tc=20.0, warm_start=True)
        self._tempo = OnsetTempoTracker(sample_rate=sample_rate)

        # Peak-decay normalization: instant attack, 5s decay
        fps = sample_rate / self.hop_length
        self._peak = 1e-6
        self._peak_alpha = 2.0 / (5.0 * fps + 1.0)

        self._energy = np.float32(0.0)
        self._tempo_period = 0.0  # seconds per beat, 0 = unknown
        self._lock = threading.Lock()

        # Load sculpture topology if provided (runner passes this automatically)
        if sculpture_id:
            self._setup_graph_topology(sculpture_id, neighbor_radius)
            num_leds = self.num_leds

        # --- Heat PDE state ---
        self._T = np.zeros(num_leds, dtype=np.float64)
        self._diffusivity = diffusivity
        self._cooling = cooling
        self._substeps = substeps
        self._heat_gain = heat_gain
        self._gamma = gamma
        self._max_brightness = max_brightness
        self._debug_gradient = debug_gradient

        # Hot wand: thermal reservoir between audio and sculpture.
        # Audio energy heats the wand instantly. The wand transfers heat
        # into the sculpture at a limited rate — smooths jarring brightness
        # spikes while preserving total energy over time.
        self._wand_heat = 0.0
        self._wand_transfer_rate = 16.0  # 1/s — wand TC ≈ 0.063s

        # Sticky ceiling: adaptive max for temperature-to-color mapping.
        # Fast up (4x) so peaks don't blow out. Slow down (0.1x) so quiet
        # sections stay visible but ceiling eventually adapts.
        # Same asymmetric EMA principle as the sticky floor, mirrored.
        self._ceiling = 1e-6
        self._ceiling_alpha = 2.0 / (10.0 * 30.0 + 1.0)  # ~10s base TC at 30fps

        # --- Rotation (ping-pong: 0→N-1→0→...) ---
        self._rotation_period = rotation_period
        self._rotation_phase = 0.0  # 0-1, fraction of cycle

        if not self._use_graph:
            # Bounce: forward then reverse, so injection sweeps back and forth
            fwd = list(range(num_leds))
            self._rotation_path = fwd + fwd[-2:0:-1]  # 0,1,...,N-1,N-2,...,1

        # Black body color ramp
        self._color_ramp = np.array([
            [0,     0,   0],     # 0.00 — black
            [180,  10,   0],     # 0.25 — deep red
            [255,  80,   0],     # 0.50 — orange
            [255, 200,  40],     # 0.75 — yellow
            [255, 255, 200],     # 1.00 — white-hot
        ], dtype=np.float64)

        # Gaussian injection kernel (strip mode only).
        # Precompute unnormalized weights; we shift and normalize per frame.
        self._inj_sigma = inj_sigma
        kernel_hw = int(np.ceil(inj_sigma * 3))  # half-width: 3 sigma
        offsets = np.arange(-kernel_hw, kernel_hw + 1, dtype=np.float64)
        self._inj_kernel = np.exp(-0.5 * (offsets / inj_sigma) ** 2)
        self._inj_kernel *= inj_sigma / self._inj_kernel.sum()  # sum to sigma: wider spread = proportionally more heat
        self._inj_kernel_hw = kernel_hw

        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    def _setup_graph_topology(self, sculpture_id, neighbor_radius):
        """Build graph Laplacian from sculpture distance matrix.

        Instead of T[i+1] - 2*T[i] + T[i-1] (1D strip), the graph Laplacian
        uses physical distances so heat at a branch junction diffuses into
        both branches simultaneously, and physically close LEDs on different
        branches exchange heat correctly.
        """
        from topology import SculptureTopology
        topo = SculptureTopology(sculpture_id)
        self._use_graph = True
        self.num_leds = topo.num_leds

        # Build rotation path at constant physical speed via arc-length
        # parameterization. Resample the LED index path at uniform arc-length
        # intervals so the hot spot moves at the same physical speed everywhere.
        n = topo.num_leds
        fwd = list(range(n))

        # Cumulative arc length along the forward path
        arc = [0.0]
        for i in range(len(fwd) - 1):
            d = topo.distances[fwd[i], fwd[i + 1]]
            arc.append(arc[-1] + max(d, 1e-6))  # clamp junction zeros
        total_arc = arc[-1]

        # Resample: N steps spaced evenly in arc length, map back to LED index
        # Use same step count as LED count so timing is similar to unweighted
        num_steps = n
        step_arc = total_arc / num_steps
        path_fwd = []
        j = 0
        for s in range(num_steps):
            target = s * step_arc
            while j < len(arc) - 1 and arc[j + 1] < target:
                j += 1
            path_fwd.append(fwd[j])

        # Ping-pong
        path_rev = path_fwd[-2:0:-1]
        self._rotation_path = path_fwd + path_rev

        # Build sparse neighbor lists + weights from distance matrix.
        # Weight = 1/distance² (inverse square, like the 1D stencil where
        # dx=1 gives weight=1). Normalized so sum of weights per LED
        # matches the 1D case (2 neighbors with weight 1 each).
        n = topo.num_leds
        self._neighbors = []  # list of (indices_array, weights_array) per LED
        for i in range(n):
            dists = topo.distances[i]
            mask = (dists > 0) & (dists < neighbor_radius)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                self._neighbors.append((np.array([], dtype=int),
                                        np.array([], dtype=np.float64)))
                continue
            weights = 1.0 / (dists[indices] ** 2)
            # Normalize: mean weight = 1.0 (so diffusivity has same meaning)
            weights *= 2.0 / (np.mean(weights) * len(weights) / 2.0) if len(weights) > 0 else 1.0
            self._neighbors.append((indices, weights))

    @property
    def name(self):
        return "Heat Diffusion"

    @property
    def description(self):
        return ("Heat equation with rotating source — audio adds heat, "
                "diffusion spreads organically. Black body color ramp.")

    # ------------------------------------------------------------------ #
    #  Audio thread                                                        #
    # ------------------------------------------------------------------ #

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            val = self._sticky.update(frame)
            raw_rms = np.sqrt(np.mean(frame ** 2))
            self._tempo.feed_frame(frame)
            with self._lock:
                # Blend: 70% sticky (transient-focused) + 30% raw RMS (sustain warmth)
                self._energy = np.float32(val * 0.7 + raw_rms * 0.3)
                self._tempo_period = self._tempo.estimated_period

    # ------------------------------------------------------------------ #
    #  Render thread                                                       #
    # ------------------------------------------------------------------ #

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = float(self._energy)

        n = self.num_leds
        T = self._T
        alpha = self._diffusivity
        cool = self._cooling
        gain = self._heat_gain

        # Tempo-driven rotation speed.
        # Pick the power-of-2 multiple of beat period that falls in [10s, 30s].
        # 10s = fastest (current feel), 30s = 3x slower.
        # No tempo lock → fall back to fixed rotation_period.
        with self._lock:
            beat_period = self._tempo_period

        MIN_ROT, MAX_ROT = 10.0, 30.0
        rot_period = self._rotation_period  # fallback
        if beat_period > 0.1:
            # Start with the raw beat period and double until in range
            p = beat_period
            while p < MIN_ROT:
                p *= 2.0
            # If we overshot, halve once
            while p > MAX_ROT:
                p /= 2.0
            # Clamp to guardrails
            rot_period = max(MIN_ROT, min(MAX_ROT, p))

        self._rotation_phase += dt / rot_period
        self._rotation_phase %= 1.0
        path = self._rotation_path
        inj = path[int(self._rotation_phase * len(path)) % len(path)]

        # --- Hot wand: absorb audio energy, transfer to sculpture ---
        self._wand_heat += energy * gain * dt
        transfer = self._wand_heat * self._wand_transfer_rate * dt
        self._wand_heat -= transfer

        # --- PDE substeps ---
        dt_sub = dt / self._substeps
        heat_per_sub = transfer / self._substeps

        for _ in range(self._substeps):
            if self._use_graph:
                # Graph mode: inject into inj + physical neighbors (wider heat source)
                T[inj] += heat_per_sub
                nbr_idx, nbr_w = self._neighbors[inj]
                if len(nbr_idx) > 0:
                    w_norm = nbr_w / nbr_w.sum() * 2.0  # neighbors add 2x, center adds 1x = 3x total
                    T[nbr_idx] += heat_per_sub * w_norm
            else:
                # Strip mode: gaussian kernel injection
                hw = self._inj_kernel_hw
                for k in range(-hw, hw + 1):
                    idx = inj + k
                    if 0 <= idx < n:
                        T[idx] += heat_per_sub * self._inj_kernel[k + hw]

            if self._use_graph:
                # Graph Laplacian: weighted sum of neighbor temp differences
                lap = np.zeros_like(T)
                for i in range(n):
                    indices, weights = self._neighbors[i]
                    if len(indices) > 0:
                        lap[i] = np.sum(weights * (T[indices] - T[i]))
            else:
                # 1D strip Laplacian
                lap = np.zeros_like(T)
                lap[1:-1] = T[2:] - 2.0 * T[1:-1] + T[:-2]
                # Insulated boundaries (Neumann: no heat flow out the ends)
                lap[0] = T[1] - T[0]
                lap[-1] = T[-2] - T[-1]

            T += (alpha * lap - cool * T) * dt_sub
            np.clip(T, 0.0, None, out=T)

        # --- Color mapping (black body ramp, sticky ceiling) ---
        # Sticky ceiling: fast up (catches peaks), slow down (adapts to quiet).
        # Ensures the hottest pixel always maps to the top of the color ramp,
        # so peaks read as white-hot regardless of the current energy level.
        t_max = np.max(T)
        if t_max > self._ceiling:
            self._ceiling += self._ceiling_alpha * 4.0 * (t_max - self._ceiling)
        else:
            self._ceiling += self._ceiling_alpha * 1.0 * (t_max - self._ceiling)
        self._ceiling = max(self._ceiling, 1e-6)

        t_norm = T / self._ceiling
        t_display = np.clip(t_norm, 0.0, 1.0) ** self._gamma

        ramp_pos = t_display * (len(self._color_ramp) - 1)
        ramp_idx = np.clip(ramp_pos.astype(int), 0, len(self._color_ramp) - 2)
        ramp_frac = ramp_pos - ramp_idx

        lower = self._color_ramp[ramp_idx]
        upper = self._color_ramp[ramp_idx + 1]
        colors = lower + (upper - lower) * ramp_frac[:, np.newaxis]

        # Apply brightness cap
        colors *= self._max_brightness

        self._frame_buf[:] = np.clip(colors, 0, 255).astype(np.uint8)

        # Debug: overwrite first 10 pixels with white-hot gradient (no brightness cap)
        # Shows what the physical LEDs look like from warm-orange to full white-hot
        if self._debug_gradient:
            for i in range(min(10, n)):
                t = i / 9.0  # 0.0 to 1.0
                rp = 0.5 + t * 0.5  # ramp position 0.5 (orange) to 1.0 (white-hot)
                ri = min(int(rp * (len(self._color_ramp) - 1)), len(self._color_ramp) - 2)
                rf = rp * (len(self._color_ramp) - 1) - ri
                c = self._color_ramp[ri] + (self._color_ramp[ri + 1] - self._color_ramp[ri]) * rf
                self._frame_buf[i] = np.clip(c, 0, 255).astype(np.uint8)

        return self._frame_buf.copy()

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = float(self._energy)

        path = self._rotation_path
        inj = path[int(self._rotation_phase * len(path)) % len(path)]

        bpm = self._tempo.bpm
        return {
            'energy': f'{energy:.3f}',
            'T_max': f'{np.max(self._T):.4f}',
            'inj_led': str(inj),
            'rotation': f'{self._rotation_phase:.2f}',
            'bpm': f'{bpm:.0f}' if bpm > 0 else '—',
            'mode': 'graph' if self._use_graph else '1D',
        }
