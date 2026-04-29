"""
Band Tendrils — AbsInt-driven percussion detection.

Per-band trigger uses |d(band_rms)/dt| integrated over a 150ms window (AbsInt).
A trigger requires BOTH:
  - Absolute floor: AbsInt exceeds a per-band threshold (rejects ambient texture)
  - Ratio gate: AbsInt exceeds k × per-band EMA (rejects sustained loud signals)

The earlier log-energy onset trigger fired on ambient music because its
ratio-only gate adapted its baseline DOWN during quiet sections. AbsInt is
linear (not log-compressed), so absolute floors have physical meaning.

Length and best-of-frame pick are also driven by AbsInt magnitude, so loud
percussion produces longer tendrils than quiet hits. Decay continues to use
the per-band normalized AbsInt for fast/slow modulation (kick fades quickly,
cymbal sustains).

Topology, origin drift, BFS path expansion, and 120-degree band hue offsets
are unchanged from earlier versions.

Usage:
    python runner.py band_tendrils --sculpture cob_diamond --no-leds
    python runner.py band_tendrils --sculpture cob_diamond
"""

import threading
import numpy as np
from base import AudioReactiveEffect
from topology import SculptureTopology
from signals import OverlapFrameAccumulator


# ── Tendril constants ──
MAX_TENDRILS = 2
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

# ── Onset detection (AbsInt-based) ──
# Trigger requires BOTH:
#   1. Absolute floor: AbsInt above per-band level (rejects quiet ambient)
#   2. Ratio gate: AbsInt > k × recent per-band EMA (rejects sustained loud signals)
#
# Calibration methodology (one-off, ran in audio-reactive/research/audio-segments):
#   Reference tracks chosen for perceptual goals —
#     ambient.wav         (Fred Again "tayla")  — must NOT fire (quiet pads)
#     opiate_intro.wav    (Tool, rock beat)     — must fire ~1-3/s
#     electronic_beat.wav (Fred Again "Tanya")  — must fire ~1-3/s
#     fa_br_drop1.wav     (build → drop)        — silent in build, many in drop
#   Per-band AbsInt percentile sweep with config: FFT=2048, hop=512, AbsInt
#   window=0.15s, AbsInt EMA tau=2.0s, single-best-per-frame with 0.45s cooldown.
#   Floors picked between ambient p100 and percussive p95; ratio chosen so
#   sustained loud signals (steady drone, sustained drop) don't keep retriggering.
ABSINT_FLOOR_PER_BAND = [20.0, 7.0, 1.4]    # LOW, MID, HIGH absolute thresholds
ABSINT_RATIO_THRESHOLD = 2.0   # AbsInt must be 2x its own recent baseline
ONSET_COOLDOWN_S = 0.45        # per-band cooldown in seconds

# ── Frequency bands for per-band onset detection ──
BAND_EDGES_HZ = [0, 250, 2000]  # low: 0-250, mid: 250-2000, high: 2000+
BAND_HUE_OFFSETS = [0.0, 0.333, 0.667]  # 120 degrees apart

# ── Per-band AbsInt ──
ABSINT_WINDOW_SEC = 0.15       # 150ms integration window
ABSINT_EMA_TIME_CONSTANT = 2.0 # tracks per-band energy envelope (for ratio + decay)

# ── AbsInt-to-length mapping ──
# Length scales by how much the AbsInt exceeds the trigger floor.
# At floor → MIN length, at LENGTH_CEIL × floor → MAX length.
ABSINT_LENGTH_CEIL_MULT = 2.0  # raw AbsInt at this multiple of floor → max length


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

        # ── Per-band frequency slices ──
        self.n_bands = len(BAND_EDGES_HZ)  # 3
        self.band_slices = []
        for b in range(self.n_bands):
            lo_hz = BAND_EDGES_HZ[b]
            hi_hz = BAND_EDGES_HZ[b + 1] if b + 1 < self.n_bands else sample_rate / 2
            lo_bin = int(np.searchsorted(self.freq_bins, lo_hz))
            hi_bin = int(np.searchsorted(self.freq_bins, hi_hz))
            self.band_slices.append(slice(lo_bin, hi_bin))

        # Per-band cooldown
        self._cooldown_frames = max(1, int(ONSET_COOLDOWN_S * frames_per_sec))
        self.band_cooldown = np.zeros(self.n_bands, dtype=np.int32)

        # ── Per-band AbsInt: trigger metric, length input, decay modulator ──
        # AbsInt = integral of |d(band_rms)/dt| over ABSINT_WINDOW_SEC.
        # Sustained signals have small derivatives → low AbsInt; transients spike.
        absint_window_frames = max(1, int(ABSINT_WINDOW_SEC * frames_per_sec))
        self._absint_window_frames = absint_window_frames
        self._absint_deriv_buf = np.zeros((self.n_bands, absint_window_frames),
                                          dtype=np.float32)
        self._absint_buf_pos = 0
        self._absint_prev_band_rms = np.zeros(self.n_bands, dtype=np.float32)
        self._absint_raw = np.zeros(self.n_bands, dtype=np.float32)
        self._absint_dt = 1.0 / frames_per_sec  # seconds per frame

        # Per-band AbsInt EMA — used for both the trigger ratio gate and decay.
        self._absint_ema_alpha = 1.0 / (ABSINT_EMA_TIME_CONSTANT * frames_per_sec)
        self._absint_ema = np.zeros(self.n_bands, dtype=np.float32)
        self._absint_ema_initialized = np.zeros(self.n_bands, dtype=bool)
        # Normalized per-band absint (ratio/(ratio+1), in [0,1)) — drives decay.
        self._absint_normalized = np.zeros(self.n_bands, dtype=np.float32)

        # ── Shared state (audio -> render) ──
        # hits carry (absint_raw, band_index)
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

        # ── Per-band RMS (input to AbsInt) ──
        band_rms = np.zeros(self.n_bands, dtype=np.float32)
        for b in range(self.n_bands):
            band_spec = spec[self.band_slices[b]]
            band_rms[b] = float(np.sqrt(np.mean(band_spec ** 2))) if len(band_spec) > 0 else 0.0

        # ── Per-band AbsInt update + EMA ──
        buf_idx = self._absint_buf_pos % self._absint_window_frames
        for b in range(self.n_bands):
            deriv = (band_rms[b] - self._absint_prev_band_rms[b]) / self._absint_dt
            self._absint_deriv_buf[b, buf_idx] = abs(deriv)
            raw = float(np.sum(self._absint_deriv_buf[b])) * self._absint_dt
            self._absint_raw[b] = raw

            if not self._absint_ema_initialized[b]:
                self._absint_ema[b] = raw
                self._absint_ema_initialized[b] = True
            else:
                self._absint_ema[b] += self._absint_ema_alpha * (raw - self._absint_ema[b])

            ema_val = self._absint_ema[b]
            ratio = raw / ema_val if ema_val > 1e-10 else 0.0
            # Normalized form for decay (saturates smoothly into [0, 1))
            self._absint_normalized[b] = ratio / (ratio + 1.0)

        self._absint_prev_band_rms = band_rms.copy()
        self._absint_buf_pos += 1

        # ── Trigger gate: absolute floor AND ratio>k, per band ──
        self.band_cooldown = np.maximum(self.band_cooldown - 1, 0)
        hits_this_frame = []
        for b in range(self.n_bands):
            raw = self._absint_raw[b]
            ema_val = self._absint_ema[b]
            ratio = raw / ema_val if ema_val > 1e-10 else 0.0
            floor_ok = raw > ABSINT_FLOOR_PER_BAND[b]
            ratio_ok = ratio > ABSINT_RATIO_THRESHOLD
            cooldown_ok = self.band_cooldown[b] == 0
            if floor_ok and ratio_ok and cooldown_ok:
                hits_this_frame.append((raw, b))
                self.band_cooldown[b] = self._cooldown_frames

        if hits_this_frame:
            # Pick the band whose AbsInt is largest relative to its own floor
            # (comparable across bands with very different absolute scales).
            best = max(hits_this_frame,
                       key=lambda x: x[0] / ABSINT_FLOOR_PER_BAND[x[1]])
            with self._lock:
                self._pending_hits.append(best)

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

        for absint_val, band in hits:
            self._spawn_tendril(absint_val, band)

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

    def _spawn_tendril(self, absint_val, band):
        """Spawn a new tendril. Length scales with AbsInt above the band's floor."""
        if self._origin_timer >= ORIGIN_DRIFT_INTERVAL:
            self._origin_timer = 0.0
        origin = self._next_origin()

        # AbsInt at the floor → MIN length; at LENGTH_CEIL_MULT × floor → MAX length.
        # Per-band scaling makes the mapping comparable across bands of different
        # absolute magnitudes (LOW band is ~20× larger than HIGH band).
        floor = ABSINT_FLOOR_PER_BAND[band]
        ceil = floor * ABSINT_LENGTH_CEIL_MULT
        length_frac = (absint_val - floor) / (ceil - floor)
        length_frac = max(0.0, min(1.0, length_frac))

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
        absint_raw = [f'{v:.1f}' for v in self._absint_raw]
        absint_norm = [f'{v:.2f}' for v in self._absint_normalized]
        return {
            'tendrils': n,
            'hue': f'{hue_deg}\u00b0',
            'pot': f'{self._pot_scale:.1f}x',
            'origins': str(origins),
            'absint_raw': str(absint_raw),
            'absint_norm': str(absint_norm),
        }
