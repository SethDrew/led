"""
Diamond Voices — topology-aware three-voice effect for the diamond sculpture.

Three audio layers mapped to the diamond's physical (x, y) spatial layout:

  Voice 1 (Bass Foundation): Bass energy radiates outward from the hub (base
      junction where all three branches meet). Warm red/orange pulses spread
      along all branches, fading with physical distance from the hub.

  Voice 2 (Harmonic Body): Harmonic content mapped per-branch using the
      derivative of harmonic energy (rate of change) for visible fluctuation.
      Sub-bands are split across branches so each one responds to different
      harmonic content.

  Voice 3 (Percussive Flash): White spatial ripple originating from a random
      LED and expanding outward using the distance matrix. Each flash is a
      wavefront, not a depth-band highlight.

Uses SculptureTopology('cob_diamond') for all spatial calculations.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from topology import SculptureTopology


class DiamondVoicesEffect(AudioReactiveEffect):
    """Three-voice topology-mapped effect for the diamond sculpture."""

    registry_name = 'diamond_voices'
    handles_topology = True
    ref_pattern = 'composite'
    ref_scope = 'beat'
    ref_input = 'streaming HPSS 3-voice (topology-mapped)'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 sculpture_id: str = 'cob_diamond'):
        super().__init__(num_leds, sample_rate)

        # ── Topology ───────────────────────────────────────────────
        self.topo = SculptureTopology(sculpture_id)
        self.num_leds = self.topo.num_leds  # 72 for diamond

        # Hub = base junction where all branches converge (LED 0 / 61 / 62)
        hub_led = self.topo.landmarks.get('base_start', 0)
        self.hub_distances = self.topo.distances_from(hub_led)
        # Normalize to 0-1 range for spatial falloff
        max_hub_dist = self.hub_distances.max() + 1e-10
        self.hub_dist_norm = self.hub_distances / max_hub_dist

        # Branch masks (boolean arrays for per-branch rendering)
        self.branch_masks = {}
        for name, (start, end) in self.topo.branches.items():
            mask = np.zeros(self.num_leds, dtype=bool)
            mask[start:end + 1] = True
            self.branch_masks[name] = mask

        # ── FFT parameters ─────────────────────────────────────────
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)

        # Frequency band masks
        self.bass_mask = (self.freq_bins >= 20) & (self.freq_bins <= 250)

        # Harmonic sub-bands — split mids into three regions for branch mapping
        self.harm_low_mask = (self.freq_bins >= 250) & (self.freq_bins <= 800)
        self.harm_mid_mask = (self.freq_bins >= 800) & (self.freq_bins <= 2500)
        self.harm_hi_mask = (self.freq_bins >= 2500) & (self.freq_bins <= 6000)

        # Audio accumulation buffer
        self.audio_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.audio_buf_pos = 0

        # Streaming HPSS: circular buffer of magnitude spectra
        self.hpss_buf_size = 17  # ~400ms at 23ms/frame
        self.spec_buf = np.zeros((self.hpss_buf_size, self.n_fft // 2 + 1),
                                 dtype=np.float32)
        self.spec_buf_idx = 0
        self.spec_buf_filled = 0

        # ── EMA normalization ──────────────────────────────────────
        self._fft_fps = sample_rate / self.n_fft
        self._ema_tc = 5.0  # seconds
        self._ema_alpha = 2.0 / (self._ema_tc * self._fft_fps + 1.0)

        # Voice 1: Bass energy
        self.bass_energy = 0.0
        self.bass_smooth = 0.0
        self.bass_ema = 1e-10

        # Voice 2: Per-branch harmonic energies + derivatives
        self.harm_energies = [0.0, 0.0, 0.0]  # left, right, middle
        self.harm_prev = [0.0, 0.0, 0.0]
        self.harm_deltas = [0.0, 0.0, 0.0]  # derivative (rate of change)
        self.harm_smooth = [0.0, 0.0, 0.0]
        self.harm_delta_smooth = [0.0, 0.0, 0.0]
        self.harm_emas = [1e-10, 1e-10, 1e-10]
        self.harm_delta_emas = [1e-10, 1e-10, 1e-10]

        # Voice 3: Percussive flash state
        self.perc_energy = 0.0
        self.perc_ema = 1e-10
        self.perc_history = np.zeros(32, dtype=np.float32)
        self.perc_hist_idx = 0
        self.flash_brightness = 0.0
        self.flash_origin = 0  # LED index where flash originates
        self.flash_dists = np.zeros(self.num_leds, dtype=np.float64)
        self.last_flash_time = 0.0
        self.time_acc = 0.0
        self.flash_cooldown = 0.05  # 50ms

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Diamond Voices"

    @property
    def description(self):
        return ("Three topology-mapped voices for diamond sculpture: "
                "bass hub radiation, harmonic branch mapping, percussive spatial ripples.")

    def process_audio(self, mono_chunk: np.ndarray):
        """Accumulate audio samples and process full FFT frames."""
        n = len(mono_chunk)
        space = self.n_fft - self.audio_buf_pos
        if n < space:
            self.audio_buf[self.audio_buf_pos:self.audio_buf_pos + n] = mono_chunk
            self.audio_buf_pos += n
            return
        # Fill remaining space and process
        self.audio_buf[self.audio_buf_pos:] = mono_chunk[:space]
        self._process_frame(self.audio_buf.copy())
        # Start next buffer with leftover
        leftover = n - space
        self.audio_buf[:leftover] = mono_chunk[space:]
        self.audio_buf_pos = leftover

    def _process_frame(self, frame):
        """Process one full FFT frame: HPSS separation + voice extraction."""
        windowed = frame * self.window
        spec = np.abs(np.fft.rfft(windowed))

        # Add to HPSS circular buffer
        self.spec_buf[self.spec_buf_idx] = spec
        self.spec_buf_idx = (self.spec_buf_idx + 1) % self.hpss_buf_size
        self.spec_buf_filled = min(self.spec_buf_filled + 1, self.hpss_buf_size)

        # Streaming HPSS
        if self.spec_buf_filled >= 3:
            buf_slice = self.spec_buf[:self.spec_buf_filled]
            harmonic_mask = np.median(buf_slice, axis=0)
            percussive = np.maximum(spec - harmonic_mask, 0)
        else:
            harmonic_mask = spec * 0.5
            percussive = spec * 0.5

        # ── Voice 1: Bass energy ───────────────────────────────────
        bass_e = np.sum(spec[self.bass_mask] ** 2)

        # ── Voice 2: Per-branch harmonic sub-bands ─────────────────
        harm_low = np.sum(harmonic_mask[self.harm_low_mask] ** 2)
        harm_mid = np.sum(harmonic_mask[self.harm_mid_mask] ** 2)
        harm_hi = np.sum(harmonic_mask[self.harm_hi_mask] ** 2)
        harm_bands = [harm_low, harm_mid, harm_hi]

        # ── Voice 3: Percussive total energy ───────────────────────
        perc_e = np.sum(percussive ** 2)

        # Update EMA trackers
        alpha = self._ema_alpha
        self.bass_ema += alpha * (bass_e - self.bass_ema)
        self.perc_ema += alpha * (perc_e - self.perc_ema)

        # Percussive peak detection
        self.perc_history[self.perc_hist_idx] = perc_e
        self.perc_hist_idx = (self.perc_hist_idx + 1) % len(self.perc_history)
        perc_mean = np.mean(self.perc_history)
        perc_std = np.std(self.perc_history)

        with self._lock:
            # Voice 1: normalized bass
            self.bass_energy = min(1.0, bass_e / (self.bass_ema + 1e-10))

            # Voice 2: per-branch harmonic + derivative
            for i in range(3):
                self.harm_emas[i] += alpha * (harm_bands[i] - self.harm_emas[i])
                norm_e = min(1.0, harm_bands[i] / (self.harm_emas[i] + 1e-10))

                # Derivative: how fast is this band changing?
                delta = abs(norm_e - self.harm_prev[i])
                self.harm_delta_emas[i] += alpha * (delta - self.harm_delta_emas[i])
                norm_delta = min(1.0, delta / (self.harm_delta_emas[i] + 1e-10))

                self.harm_prev[i] = norm_e
                self.harm_energies[i] = norm_e
                self.harm_deltas[i] = norm_delta

            # Voice 3: flash on percussive peaks
            perc_threshold = perc_mean + 1.0 * perc_std
            if perc_e > perc_threshold and perc_threshold > 0:
                self.perc_energy = min(1.0, perc_e / (self.perc_ema + 1e-10))

    def render(self, dt: float) -> np.ndarray:
        self.time_acc += dt
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        with self._lock:
            bass = self.bass_energy
            harm_e = list(self.harm_energies)
            harm_d = list(self.harm_deltas)
            perc = self.perc_energy
            self.perc_energy = 0.0  # consume the trigger

        # ── Smooth bass (fast attack, slow decay) ──────────────────
        attack_bass = 0.3
        decay_bass = 0.05
        alpha_b = attack_bass if bass > self.bass_smooth else decay_bass
        self.bass_smooth += alpha_b * (bass - self.bass_smooth)

        # ── Smooth harmonics (per-branch) ──────────────────────────
        for i in range(3):
            # Level: moderate smoothing
            a_h = 0.2 if harm_e[i] > self.harm_smooth[i] else 0.04
            self.harm_smooth[i] += a_h * (harm_e[i] - self.harm_smooth[i])
            # Delta: fast attack, moderate decay for visible flicker
            a_d = 0.4 if harm_d[i] > self.harm_delta_smooth[i] else 0.08
            self.harm_delta_smooth[i] += a_d * (harm_d[i] - self.harm_delta_smooth[i])

        # ── Voice 1: Bass Foundation ───────────────────────────────
        # Warm red/orange pulse radiating from hub, fading with distance
        bass_brightness = self.bass_smooth ** 0.7
        # Spatial falloff: Gaussian centered at hub, sigma ~0.3 of max distance
        bass_spatial = np.exp(-2.5 * self.hub_dist_norm ** 2)
        bass_val = bass_spatial * bass_brightness
        frame[:, 0] += bass_val * 255        # red
        frame[:, 1] += bass_val * 0.3 * 255  # orange tint

        # ── Voice 2: Harmonic Body ─────────────────────────────────
        # Each branch driven by a different harmonic sub-band.
        # Brightness = blend of level and derivative for visible fluctuation.
        # Branch mapping: left=low mids, middle=mid mids, right=high mids
        branch_order = ['left', 'middle', 'right']
        # Cool palette per branch: purple, teal, blue
        branch_colors = [
            np.array([60, 15, 160], dtype=np.float32),   # left: purple
            np.array([15, 140, 120], dtype=np.float32),   # middle: teal
            np.array([30, 30, 170], dtype=np.float32),    # right: blue
        ]

        for i, bname in enumerate(branch_order):
            mask = self.branch_masks.get(bname)
            if mask is None:
                continue

            level = self.harm_smooth[i] ** 0.8
            delta = self.harm_delta_smooth[i] ** 0.6

            # Blend: 40% base level + 60% derivative for dynamic response
            brightness = 0.4 * level + 0.6 * delta

            # Apply with distance-from-hub gradient (brighter further from hub)
            # This creates visual separation from the bass voice at the hub
            harm_spatial = np.clip(self.hub_dist_norm * 1.5 - 0.15, 0, 1)

            color = branch_colors[i]
            for ch in range(3):
                frame[mask, ch] += harm_spatial[mask] * brightness * color[ch]

        # ── Voice 3: Percussive Flash ──────────────────────────────
        # Spatial ripple from random origin using distance matrix
        if perc > 0 and (self.time_acc - self.last_flash_time) > self.flash_cooldown:
            self.flash_brightness = min(1.0, perc)
            # Pick a random LED as flash origin
            self.flash_origin = np.random.randint(0, self.num_leds)
            # Pre-fetch distances from this origin
            self.flash_dists = self.topo.distances_from(self.flash_origin).copy()
            self.last_flash_time = self.time_acc

        if self.flash_brightness > 0.01:
            # Expanding wavefront: radius grows as brightness decays
            max_dist = self.flash_dists.max() + 1e-10
            norm_dists = self.flash_dists / max_dist

            # Wavefront radius expands from 0 to ~0.6 as flash fades
            elapsed_frac = 1.0 - self.flash_brightness  # 0 at start, ~1 at end
            wave_center = elapsed_frac * 0.6  # expanding ring radius
            wave_width = 0.08 + elapsed_frac * 0.12  # ring thickens as it expands

            # Ring shape: Gaussian around the wavefront radius
            ring = np.exp(-0.5 * ((norm_dists - wave_center) / wave_width) ** 2)
            # Also keep a core glow at the origin
            core = np.exp(-8.0 * norm_dists ** 2)
            flash_spatial = np.maximum(ring, core * self.flash_brightness)

            flash_val = flash_spatial * self.flash_brightness * 255
            frame[:, 0] += flash_val
            frame[:, 1] += flash_val
            frame[:, 2] += flash_val

            # Fast exponential decay
            self.flash_brightness *= 0.85 ** (dt * 30)  # ~150ms decay

        # Clamp and convert
        np.clip(frame, 0, 255, out=frame)
        return frame.astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            return {
                'bass': f'{self.bass_smooth:.2f}',
                'harm_L': f'{self.harm_smooth[0]:.2f}',
                'harm_M': f'{self.harm_smooth[1]:.2f}',
                'harm_R': f'{self.harm_smooth[2]:.2f}',
                'delta_L': f'{self.harm_delta_smooth[0]:.2f}',
                'delta_M': f'{self.harm_delta_smooth[1]:.2f}',
                'delta_R': f'{self.harm_delta_smooth[2]:.2f}',
                'flash': f'{self.flash_brightness:.2f}',
            }
