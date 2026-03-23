"""
Band Tendrils v2 — three-metric hybrid onset detection.

Evolved from band_tendrils (v1) based on analysis of five failure modes:

1. TRIGGER: Log-energy onset strength (not raw spectral flux).
   Per-band: FFT → band energy → log1p → first-difference → half-wave rectify.
   Log compression naturally favors transients over gradual harmonic changes,
   reducing the ~30% harmonic false positives seen with raw spectral flux
   (see ledger: hpss-vs-flux-empirical). EMA-normalized per band with global
   EMA, per-band cooldown.

2. LENGTH: Per-band RMS at the trigger frame.
   Separate metric from trigger. Maps to tendril max_length through a wider
   dynamic range than v1's clipped 0.35-1.0 strength. A ghost kick spawns a
   short tendril; a full kick fills the branch.

3. DECAY: Per-band AbsInt sampled continuously after trigger.
   Each tendril stores its source band. Each render frame, the band's current
   AbsInt level interpolates decay between fast (0.75 for percussive) and slow
   (0.85 for sustained). A kick tendril fades in ~150ms; a cymbal tendril
   sustains for ~500ms.

Everything else from band_tendrils v1: BFS topology, origin drift, color
rotation with 120-degree band offsets, global EMA, per-band cooldown.

Usage:
    python runner.py band_tendrils_v2 --sculpture cob_diamond --no-leds
    python runner.py band_tendrils_v2 --sculpture cob_diamond
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
TENDRIL_MAX_LENGTH = 20      # LEDs for strongest hit
TENDRIL_EXPAND_SPEED = 40.0  # LEDs per second expansion rate
TENDRIL_TRAIL_FALLOFF = 0.80 # brightness multiplier per LED behind head
TENDRIL_MAX_BRIGHTNESS = 0.6 # cap peak brightness to reduce flashiness

# ── Decay modulation (v2: AbsInt-driven) ──
DECAY_FAST = 0.75            # per-frame decay for sharp percussive (kick, snare)
DECAY_SLOW = 0.85            # per-frame decay for sustained (cymbal, pad)
DECAY_DEFAULT = 0.82         # fallback before AbsInt has data

# ── Color rotation ──
HUE_CYCLE_PERIOD = 20.0      # seconds for full color wheel rotation
PALETTE_WIDTH = 0.15          # hue spread within active palette (0-1)
SPATIAL_HUE_SPREAD = 0.08    # hue offset along tendril length

# ── Origin drift ──
ORIGIN_DRIFT_INTERVAL = 8.0  # seconds between origin changes

# ── Onset detection ──
ONSET_EMA_TIME_CONSTANT = 5.0  # seconds — stable baseline
ONSET_THRESHOLD = 3.3           # ratio above EMA mean to trigger
ONSET_COOLDOWN_S = 0.45         # per-band cooldown in seconds
RMS_FLOOR = 0.005               # minimum RMS to allow any onset detection

# ── Frequency bands for per-band onset detection ──
BAND_EDGES_HZ = [0, 250, 2000]  # low: 0-250, mid: 250-2000, high: 2000+
BAND_HUE_OFFSETS = [0.0, 0.333, 0.667]  # 120 degrees apart

# ── Per-band AbsInt constants ──
ABSINT_WINDOW_SEC = 0.15     # 150ms integration window (matches signals.AbsIntegral)
# AbsInt EMA for normalization (per-band, for decay modulation)
ABSINT_EMA_TIME_CONSTANT = 2.0  # shorter than onset EMA — tracks energy envelope

# ── RMS-to-length mapping (v2: wider dynamic range) ──
# Instead of v1's 0.35-1.0 clip on flux ratio, we use per-band RMS
# with a log mapping to preserve dynamic range across 40+ dB
RMS_LENGTH_FLOOR = 0.003     # below this RMS, minimum length
RMS_LENGTH_CEIL = 0.20       # above this RMS, maximum length


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


class BandTendrilsEffect(AudioReactiveEffect):
    """Three-metric hybrid: onset strength trigger, RMS length, AbsInt decay.

    v2 improvements over v1 (band_tendrils):
    - Trigger: log-energy onset strength instead of raw spectral flux
    - Length: per-band RMS with log mapping instead of clipped flux ratio
    - Decay: per-band AbsInt modulates tendril fade speed per-sound
    """

    registry_name = 'band_tendrils'
    handles_topology = True

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 sculpture_id: str = 'cob_diamond'):
        super().__init__(num_leds, sample_rate)
        self.topo = SculptureTopology(sculpture_id)
        self.num_leds = self.topo.num_leds

        # ── Build wire-path adjacency ──
        self.adjacency = self._build_adjacency()

        # ── Audio processing ──
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.accum = OverlapFrameAccumulator(frame_len=self.n_fft, hop=512)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        frames_per_sec = sample_rate / 512.0

        # ── Per-band onset detection (log-energy onset strength) ──
        self.n_bands = len(BAND_EDGES_HZ)  # 3
        self.band_slices = []
        for b in range(self.n_bands):
            lo_hz = BAND_EDGES_HZ[b]
            hi_hz = BAND_EDGES_HZ[b + 1] if b + 1 < self.n_bands else sample_rate / 2
            lo_bin = int(np.searchsorted(self.freq_bins, lo_hz))
            hi_bin = int(np.searchsorted(self.freq_bins, hi_hz))
            self.band_slices.append(slice(lo_bin, hi_bin))

        # Previous frame's per-band log energy (for first-difference)
        self._prev_band_log_energy = np.zeros(self.n_bands, dtype=np.float64)
        self._prev_frame_valid = False

        # Global EMA for onset strength normalization
        self.onset_ema_alpha = 1.0 / (ONSET_EMA_TIME_CONSTANT * frames_per_sec)
        self.global_onset_ema = 0.0
        self.global_ema_initialized = False

        # Per-band cooldown
        self._cooldown_frames = max(1, int(ONSET_COOLDOWN_S * frames_per_sec))
        self.band_cooldown = np.zeros(self.n_bands, dtype=np.int32)

        # ── Per-band AbsInt (for decay modulation) ──
        # Each band tracks |d(band_rms)/dt| integrated over ABSINT_WINDOW_SEC
        absint_window_frames = max(1, int(ABSINT_WINDOW_SEC / (1.0 / frames_per_sec)))
        self._absint_window_frames = absint_window_frames
        self._absint_deriv_buf = np.zeros((self.n_bands, absint_window_frames),
                                          dtype=np.float32)
        self._absint_buf_pos = 0
        self._absint_prev_band_rms = np.zeros(self.n_bands, dtype=np.float32)
        self._absint_raw = np.zeros(self.n_bands, dtype=np.float32)
        self._absint_dt = 1.0 / frames_per_sec  # seconds per frame

        # Per-band AbsInt EMA for normalization (so we can get a 0-1 signal)
        self._absint_ema_alpha = 1.0 / (ABSINT_EMA_TIME_CONSTANT * frames_per_sec)
        self._absint_ema = np.zeros(self.n_bands, dtype=np.float32)
        self._absint_ema_initialized = np.zeros(self.n_bands, dtype=bool)
        # Normalized per-band absint (0-1 ish), shared with render thread
        self._absint_normalized = np.zeros(self.n_bands, dtype=np.float32)

        # ── Per-band RMS (for tendril length) ──
        self._band_rms = np.zeros(self.n_bands, dtype=np.float32)

        # ── RMS energy gate ──
        self.rms_ema_alpha = 1.0 / (1.0 * frames_per_sec)  # 1s time constant
        self.rms_ema = 0.0
        self.rms_ema_initialized = False

        # ── Shared state (audio -> render) ──
        # hits carry (band_rms, band_index) — length comes from RMS, not flux ratio
        self._pending_hits = []
        self._lock = threading.Lock()

        # ── Tendril state (render thread only) ──
        self.tendrils = []

        # ── Origin management ──
        self._all_origins = list(range(self.num_leds))
        self._origin_pool = list(self._all_origins)
        np.random.shuffle(self._origin_pool)
        self._origin_idx = 0
        self._origin_timer = 0.0

        # ── Color state ──
        self._hue_phase = np.random.random()
        self._time_acc = 0.0

        # ── Pot control ──
        self._pot_scale = 1.0

        # Per-LED buffers (render thread)
        self._led_brightness = np.zeros(self.num_leds, dtype=np.float32)
        self._led_color = np.zeros((self.num_leds, 3), dtype=np.float32)

    @property
    def name(self):
        return "Band Tendrils v2"

    @property
    def description(self):
        return ("Three-metric hybrid: onset strength trigger, RMS-driven length, "
                "AbsInt-modulated decay. Per-band with 120-degree color offsets.")

    def set_pot_value(self, raw):
        """Map pot 0-1023 to tendril length scale 0.5x-8.0x."""
        t = raw / 1023.0
        self._pot_scale = 0.5 + t * 7.5

    # ── Wire-path adjacency ─────────────────────────────────────────

    def _build_adjacency(self):
        """Build adjacency list from branch wiring + cross-branch connections."""
        adj = [set() for _ in range(self.num_leds)]

        branches = self.topo.branches
        for name, (start, end) in branches.items():
            for i in range(start, end + 1):
                if i > start:
                    adj[i].add(i - 1)
                if i < end:
                    adj[i].add(i + 1)

        lm = self.topo.landmarks

        base_leds = [lm.get('base_start', 0),
                     lm.get('base_junction', 61),
                     62]
        for a in base_leds:
            for b in base_leds:
                if a != b:
                    adj[a].add(b)

        apex_leds = [lm.get('apex_left', 41),
                     lm.get('apex_right', 42),
                     71]
        for a in apex_leds:
            for b in apex_leds:
                if a != b:
                    adj[a].add(b)

        cross_a = lm.get('crossover', 46)
        adj[cross_a].add(66)
        adj[66].add(cross_a)

        adj[52].add(57)
        adj[57].add(52)

        adj[4].add(63)
        adj[63].add(4)

        return [sorted(s) for s in adj]

    # ── BFS path expansion ──────────────────────────────────────────

    def _bfs_path(self, origin, max_length):
        """BFS from origin, returning LEDs in visit order up to max_length."""
        visited = set()
        visited.add(origin)
        queue = [(origin, 0)]
        result = [(origin, 0)]

        i = 0
        while i < len(queue) and len(result) < max_length:
            led, dist = queue[i]
            i += 1

            neighbors = list(self.adjacency[led])
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

        # ── RMS energy gate ──
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if not self.rms_ema_initialized:
            self.rms_ema = rms
            self.rms_ema_initialized = True
        else:
            self.rms_ema += self.rms_ema_alpha * (rms - self.rms_ema)

        # ── Per-band computations ──
        band_log_energy = np.zeros(self.n_bands, dtype=np.float64)
        band_rms = np.zeros(self.n_bands, dtype=np.float32)

        for b in range(self.n_bands):
            sl = self.band_slices[b]
            band_spec = spec[sl]
            # Log-energy for onset detection (matches OnsetTempoTracker approach)
            band_log_energy[b] = np.log1p(np.sum(band_spec ** 2))
            # Linear RMS for tendril length mapping
            band_rms[b] = float(np.sqrt(np.mean(band_spec ** 2))) if len(band_spec) > 0 else 0.0

        self._band_rms = band_rms

        # ── Per-band AbsInt update ──
        # |d(band_rms)/dt| integrated over 150ms window
        buf_idx = self._absint_buf_pos % self._absint_window_frames
        for b in range(self.n_bands):
            deriv = (band_rms[b] - self._absint_prev_band_rms[b]) / self._absint_dt
            self._absint_deriv_buf[b, buf_idx] = abs(deriv)
            raw = float(np.sum(self._absint_deriv_buf[b])) * self._absint_dt
            self._absint_raw[b] = raw

            # EMA normalization of absint (for 0-1 signal)
            if not self._absint_ema_initialized[b]:
                self._absint_ema[b] = raw
                self._absint_ema_initialized[b] = True
            else:
                self._absint_ema[b] += self._absint_ema_alpha * (raw - self._absint_ema[b])

            # Ratio: raw / ema gives ~1.0 at average activity, >1 during transients
            # We want 0-1 for decay interpolation, so use sigmoid-like mapping:
            # ratio / (ratio + 1) maps [0, inf) -> [0, 1)
            ema_val = self._absint_ema[b]
            if ema_val > 1e-10:
                ratio = raw / ema_val
            else:
                ratio = 0.0
            self._absint_normalized[b] = ratio / (ratio + 1.0)

        self._absint_prev_band_rms = band_rms.copy()
        self._absint_buf_pos += 1

        # ── Log-energy onset strength (trigger metric) ──
        if self._prev_frame_valid:
            # First-difference of log energy per band
            flux = band_log_energy - self._prev_band_log_energy
            # Half-wave rectify: only energy increases are onsets
            onset_per_band = np.maximum(flux, 0)

            # Tick all band cooldowns
            self.band_cooldown = np.maximum(self.band_cooldown - 1, 0)

            # Global EMA: mean onset strength across all bands
            total_onset = float(np.mean(onset_per_band))
            if not self.global_ema_initialized:
                self.global_onset_ema = total_onset
                self.global_ema_initialized = True
            else:
                self.global_onset_ema += self.onset_ema_alpha * (
                    total_onset - self.global_onset_ema)

            hits_this_frame = []

            for b in range(self.n_bands):
                band_onset = float(onset_per_band[b])

                # Normalize against global EMA (cross-band suppression)
                band_share = self.global_onset_ema  # each band compared to global mean
                normalized = band_onset / (band_share + 1e-10)

                # Three-gate onset detection
                energy_ok = rms > RMS_FLOOR
                onset_ok = normalized > ONSET_THRESHOLD
                cooldown_ok = self.band_cooldown[b] == 0

                if energy_ok and onset_ok and cooldown_ok:
                    # v2: pass band RMS for length, not flux-derived strength
                    hits_this_frame.append((band_rms[b], b))
                    self.band_cooldown[b] = self._cooldown_frames

            if hits_this_frame:
                # Take the strongest hit if multiple bands fire simultaneously
                best_rms, best_band = max(hits_this_frame, key=lambda x: x[0])
                with self._lock:
                    self._pending_hits.append((best_rms, best_band))

        self._prev_band_log_energy = band_log_energy.copy()
        self._prev_frame_valid = True

    # ── Rendering ───────────────────────────────────────────────────

    def render(self, dt: float) -> np.ndarray:
        step = dt * 30  # normalize to 30fps

        # Advance time
        self._time_acc += dt
        self._hue_phase = (self._time_acc / HUE_CYCLE_PERIOD) % 1.0

        # Advance origin drift timer
        self._origin_timer += dt

        # Drain pending hits -> spawn tendrils
        with self._lock:
            hits = self._pending_hits[:]
            self._pending_hits.clear()
            # Snapshot per-band absint for decay modulation
            absint_snapshot = self._absint_normalized.copy()

        for band_rms_val, band in hits:
            self._spawn_tendril(band_rms_val, band)

        # Update tendrils
        alive = []
        for t in self.tendrils:
            # Expand head
            t['current_length'] += TENDRIL_EXPAND_SPEED * dt
            t['current_length'] = min(t['current_length'], t['max_length'])

            # v2: AbsInt-modulated decay
            # High absint (sustained sound) -> slow decay
            # Low absint (percussive transient has ended) -> fast decay
            band_idx = t['band']
            absint_level = absint_snapshot[band_idx]

            # Interpolate: absint_level=0 -> DECAY_FAST, absint_level=1 -> DECAY_SLOW
            # But absint_normalized is ratio/(ratio+1), so it centers around 0.5
            # at average activity. We want sustained sounds (above average) to
            # decay slowly and percussive transients (below average after the hit)
            # to decay quickly.
            decay = DECAY_FAST + absint_level * (DECAY_SLOW - DECAY_FAST)

            t['brightness'] *= decay ** step

            # Kill if too dim
            if t['brightness'] < 0.03:
                continue
            alive.append(t)
        self.tendrils = alive

        # Render frame
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        for t in self.tendrils:
            path = t['path']
            n_lit = int(t['current_length'])
            base_hue = t['hue']

            for i in range(min(n_lit, len(path))):
                led, hop = path[i]

                dist_from_head = n_lit - 1 - i
                spatial_fade = TENDRIL_TRAIL_FALLOFF ** dist_from_head
                brightness = t['brightness'] * spatial_fade

                tendril_frac = i / max(len(path) - 1, 1)
                hue = (base_hue + tendril_frac * SPATIAL_HUE_SPREAD) % 1.0

                sat = 0.85 + 0.15 * (1.0 - tendril_frac)

                color = hsv_to_rgb(hue, sat, min(brightness, TENDRIL_MAX_BRIGHTNESS))

                frame[led] += color

        return np.clip(frame, 0, 255).astype(np.uint8)

    def _spawn_tendril(self, band_rms_val, band):
        """Spawn a new tendril. Length from band RMS, not flux ratio."""
        if self._origin_timer >= ORIGIN_DRIFT_INTERVAL:
            self._origin_timer = 0.0
        origin = self._next_origin()

        # v2: Log-mapped RMS to length (wider dynamic range)
        # log mapping: a 20dB range of RMS maps linearly to tendril length
        if band_rms_val < RMS_LENGTH_FLOOR:
            length_frac = 0.0
        elif band_rms_val > RMS_LENGTH_CEIL:
            length_frac = 1.0
        else:
            # Log-scale mapping: equal perceptual steps across dynamic range
            log_floor = math.log(RMS_LENGTH_FLOOR)
            log_ceil = math.log(RMS_LENGTH_CEIL)
            length_frac = (math.log(band_rms_val) - log_floor) / (log_ceil - log_floor)

        base_len = TENDRIL_MIN_LENGTH + length_frac * (TENDRIL_MAX_LENGTH - TENDRIL_MIN_LENGTH)
        max_len = int(base_len * self._pot_scale)
        max_len = max(2, min(max_len, self.num_leds))

        path = self._bfs_path(origin, max_len)

        # Hue: band offset + rotating phase + small random jitter
        hue = (self._hue_phase + BAND_HUE_OFFSETS[band] +
               np.random.uniform(-PALETTE_WIDTH / 2, PALETTE_WIDTH / 2)) % 1.0

        tendril = {
            'origin': origin,
            'path': path,
            'max_length': max_len,
            'current_length': 1.0,
            'brightness': 1.0,
            'band': band,           # v2: store band for AbsInt decay lookup
            'hue': hue,
        }

        self.tendrils.append(tendril)
        if len(self.tendrils) > MAX_TENDRILS:
            self.tendrils.pop(0)

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        n = len(self.tendrils)
        hue_deg = int(self._hue_phase * 360)
        origins = [t['origin'] for t in self.tendrils[:4]]
        band_absint = [f'{v:.2f}' for v in self._absint_normalized]
        return {
            'tendrils': n,
            'hue': f'{hue_deg}\u00b0',
            'pot': f'{self._pot_scale:.1f}x',
            'origins': str(origins),
            'global_ema': f'{self.global_onset_ema:.3f}',
            'rms_ema': f'{self.rms_ema:.4f}',
            'absint': str(band_absint),
        }
