"""
Heat Burst — static snapshot of an ecstatic heat bloom.

A single frozen frame: maximum heat sustained at the base anchor LEDs,
diffused through the sculpture by the heat-equation algorithm until it
reaches steady state. The resulting spatial profile (white-hot base
fading to orange/red toward the apex) is computed once at construction
and rendered identically every frame.

Companion to heat_diffusion. Same physics, same color ramp, but no
audio input and no time evolution — just the silhouette of the bloom.

Steady state of: ∂T/∂t = α ∇²T - cooling * T + source(x_base)
"""

import numpy as np
from base import AudioReactiveEffect


class HeatBurstEffect(AudioReactiveEffect):
    """Static, non-audio bloom from the sculpture base."""

    registry_name = 'heat_burst'
    handles_topology = True
    ref_pattern = ''
    ref_scope = ''
    ref_input = 'none (static snapshot)'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 # --- Thermal physics ---
                 # Same diffusivity as heat_diffusion. Cooling is much
                 # lower because there's no audio source to keep replacing
                 # heat — at heat_diffusion's cooling=4 the steady-state
                 # gradient is so steep the apex stays dark.
                 diffusivity: float = 40.0,
                 cooling: float = 4.0,
                 substeps: int = 6,
                 # --- Source ---
                 source_strength: float = 1.0,
                 base_height: float = 0.005,
                 # --- Topology ---
                 sculpture_id: str = None,
                 neighbor_radius: float = 0.12,
                 # --- Convergence ---
                 sim_seconds: float = 20.0,
                 sim_fps: float = 30.0,
                 # --- Post-source fade ---
                 # After steady state, run physics with no source for
                 # this long. Heat keeps diffusing up while base dims —
                 # captures the "music just stopped" moment.
                 fade_seconds: float = 0.0,
                 # --- Appearance ---
                 gamma: float = 1.0,
                 max_brightness: float = 0.35,
                 ):
        """
        Args:
            cooling:        Radiative cooling rate. Sets the gradient
                            sharpness — higher = bloom dies near the
                            base, lower = whole sculpture glows.
            source_strength: Sustained heat injection at base anchors.
                            Absolute value doesn't matter (frame is
                            normalized) but sets the convergence scale.
            base_height:    Height threshold for what counts as the base.
                            LEDs with topology y < threshold are anchors.
            sim_seconds:    Simulated time to run before freezing.
                            Long enough to reach steady state at the
                            chosen cooling rate.
        """
        super().__init__(num_leds, sample_rate)

        self._use_graph = False
        if sculpture_id:
            self._setup_graph_topology(sculpture_id, neighbor_radius)
            num_leds = self.num_leds

        self._color_ramp = np.array([
            [0,     0,   0],
            [180,  10,   0],
            [255,  80,   0],
            [255, 200,  40],
            [255, 255, 200],
        ], dtype=np.float64)

        anchors = self._find_base_anchors(sculpture_id, base_height)

        # Phase 1: source on, run to steady state.
        T = np.zeros(num_leds, dtype=np.float64)
        self._run_physics(T, anchors, diffusivity, cooling, source_strength,
                          substeps, sim_seconds, sim_fps)
        ceiling = float(np.max(T))  # color reference — fade dims against this

        # Phase 2: source off, run fade so heat keeps diffusing up while base
        # cools. Rendered against the pre-fade ceiling so the dim is visible.
        if fade_seconds > 0:
            self._run_physics(T, anchors, diffusivity, cooling, 0.0,
                              substeps, fade_seconds, sim_fps)

        self._cached_frame = self._temp_to_colors(T, gamma, max_brightness,
                                                  t_max=ceiling)

        # Diagnostics-only state
        self._anchors = anchors
        self._T_ceiling = ceiling
        self._T_max_final = float(np.max(T))

    def _setup_graph_topology(self, sculpture_id, neighbor_radius):
        from topology import SculptureTopology
        topo = SculptureTopology(sculpture_id)
        self._use_graph = True
        self.num_leds = topo.num_leds

        n = topo.num_leds
        self._neighbors = []
        for i in range(n):
            dists = topo.distances[i]
            mask = (dists > 0) & (dists < neighbor_radius)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                self._neighbors.append((np.array([], dtype=int),
                                        np.array([], dtype=np.float64)))
                continue
            weights = 1.0 / (dists[indices] ** 2)
            weights *= 2.0 / (np.mean(weights) * len(weights) / 2.0)
            self._neighbors.append((indices, weights))

    def _find_base_anchors(self, sculpture_id, base_height):
        if not self._use_graph or sculpture_id is None:
            return [0]
        from topology import SculptureTopology
        topo = SculptureTopology(sculpture_id)
        ys = topo.coords[:, 1]
        base = [int(i) for i in np.where(ys < base_height)[0]]
        return base if base else [0]

    def _run_physics(self, T, anchors, alpha, cool, source,
                     substeps, sim_t, fps):
        """Advance the diffusion PDE in-place. source=0 disables injection.

        Same kernel as heat_diffusion: per-anchor center+graph-neighbor
        injection (graph mode) or 1D stencil (strip mode), then weighted
        Laplacian, then T += (α·∇²T − c·T)·dt.
        """
        n = len(T)
        dt_sub = (1.0 / fps) / substeps
        n_steps = int(sim_t * fps)

        for _ in range(n_steps):
            for _ in range(substeps):
                if source > 0:
                    if self._use_graph:
                        for c in anchors:
                            nbr_idx, nbr_w = self._neighbors[c]
                            if len(nbr_idx) > 0:
                                w_norm = nbr_w / nbr_w.sum() * 2.0
                                T[c] += source / 3.0 * dt_sub
                                T[nbr_idx] += (source / 3.0) * w_norm * dt_sub
                            else:
                                T[c] += source * dt_sub
                    else:
                        for c in anchors:
                            T[c] += source * dt_sub

                if self._use_graph:
                    lap = np.zeros_like(T)
                    for i in range(n):
                        indices, weights = self._neighbors[i]
                        if len(indices) > 0:
                            lap[i] = np.sum(weights * (T[indices] - T[i]))
                else:
                    lap = np.zeros_like(T)
                    if n >= 2:
                        lap[1:-1] = T[2:] - 2.0 * T[1:-1] + T[:-2]
                        lap[0] = T[1] - T[0]
                        lap[-1] = T[-2] - T[-1]

                T += (alpha * lap - cool * T) * dt_sub
                np.clip(T, 0.0, None, out=T)

    def _temp_to_colors(self, T, gamma, max_brightness, t_max=None):
        """Map a temperature array to an RGB frame via the black-body ramp.

        t_max is the normalization reference. Pass the steady-state ceiling
        so post-fade frames render dimmer than the steady-state frame would.
        Defaults to the array's own max (full saturation).
        """
        if t_max is None:
            t_max = float(np.max(T))
        t_max = max(t_max, 1e-9)
        t_norm = np.clip(T / t_max, 0.0, 1.0) ** gamma

        ramp_pos = t_norm * (len(self._color_ramp) - 1)
        ramp_idx = np.clip(ramp_pos.astype(int), 0, len(self._color_ramp) - 2)
        ramp_frac = ramp_pos - ramp_idx
        lower = self._color_ramp[ramp_idx]
        upper = self._color_ramp[ramp_idx + 1]
        colors = lower + (upper - lower) * ramp_frac[:, np.newaxis]
        colors *= max_brightness
        return np.clip(colors, 0, 255).astype(np.uint8)

    @property
    def name(self):
        return "Heat Burst"

    @property
    def description(self):
        return ("Static snapshot of a heat bloom from the base — "
                "non-audio. Same physics as heat_diffusion, frozen "
                "at steady state.")

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def render(self, dt: float) -> np.ndarray:
        return self._cached_frame.copy()

    def get_diagnostics(self) -> dict:
        return {
            'anchors': str(self._anchors),
            'T_ceiling': f'{self._T_ceiling:.3f}',
            'T_final': f'{self._T_max_final:.3f}',
            'mode': 'graph' if self._use_graph else '1D',
        }
