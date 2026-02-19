"""
Band Zone Pulse — frequency-zoned percussive pulses via streaming HPSS.

Splits the LED strip into 5 frequency zones (low→high = bottom→top).
Percussive hits pulse in their zone at the exact frequency position:
  Sub-bass (20-80 Hz)   → dark red    kick drums
  Bass (80-250 Hz)      → orange      bass guitar, toms
  Mids (250-2kHz)       → orange      snare body, vocals
  High-mids (2-6kHz)    → green       snare crack, guitar attack
  Treble (6-8kHz)       → blue        hi-hats, cymbals

Same instrument re-triggers in place; different instruments avoid overlap.
Zone positions drift slowly (~10s) to keep the display fresh.
Optional harmonic-dominant background glow (currently off for testing).
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


BANDS = [
    ('Sub-bass', 20, 80),
    ('Bass', 80, 250),
    ('Mids', 250, 2000),
    ('High-mids', 2000, 6000),
    ('Treble', 6000, 8000),
]

# Vibrant pulse colors
BAND_COLORS = np.array([
    [200, 0, 0],       # Sub-bass: deep red
    [255, 100, 0],     # Bass:     orange
    [255, 140, 0],     # Mids:     orange (tuned for COB)
    [0, 230, 118],     # High-mids: green
    [0, 176, 255],     # Treble:   blue
], dtype=np.float32)

# Muted background variants — desaturated, shifted slightly from pulse colors
BG_COLORS = np.array([
    [90, 15, 25],      # Sub-bass: dusty maroon
    [160, 80, 30],     # Bass:     muted amber
    [170, 110, 30],    # Mids:     warm tan
    [20, 140, 90],     # High-mids: muted teal
    [30, 110, 160],    # Treble:   muted steel blue
], dtype=np.float32)

MAX_SPARKLES = 40


class BandZonePulseEffect(AudioReactiveEffect):
    """Frequency-zoned percussive pulses via streaming HPSS."""

    registry_name = 'band_zone_pulse'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)

        # Frame accumulator (2048 frame, 512 hop)
        self.accum = OverlapFrameAccumulator(frame_len=self.n_fft, hop=512)

        # Band masks
        self.n_bands = len(BANDS)
        self.band_masks = []
        for _, lo, hi in BANDS:
            self.band_masks.append((self.freq_bins >= lo) & (self.freq_bins < hi))

        # ── Streaming HPSS ──
        self.hpss_buf_size = 17  # ~400ms context
        self.spec_buf = np.zeros((self.hpss_buf_size, self.n_fft // 2 + 1), dtype=np.float32)
        self.spec_buf_idx = 0
        self.spec_buf_filled = 0

        # ── Harmonic band energy (drives background color) ──
        self.band_peaks = np.full(self.n_bands, 1e-10, dtype=np.float32)
        self.band_peak_decay = 0.9995

        # Vote ring: which group won each frame (0=low, 1=mid, 2=high)
        self.vote_window_len = int(5 * sample_rate / self.n_fft)
        self.vote_ring = np.zeros(self.vote_window_len, dtype=np.int32)
        self.vote_ring_pos = 0
        self.vote_ring_filled = 0

        # ── Percussive per-band peak detection ──
        self.perc_history_len = 8  # ~180ms at ~46ms/frame
        self.perc_history = np.zeros((self.perc_history_len, self.n_bands), dtype=np.float32)
        self.perc_hist_idx = 0
        self.perc_hist_filled = 0
        self.perc_cooldown_frames = 3  # ~140ms between hits per band
        self.perc_cooldown_counters = np.zeros(self.n_bands, dtype=np.int32)
        self.perc_band_peaks = np.full(self.n_bands, 1e-10, dtype=np.float32)
        self.perc_peak_decay = 0.998  # slow-decay peak per band

        # ── Shared state (audio → render) ──
        self.dominant_color = np.array([15, 1, 6], dtype=np.float32)
        self.dominant_color_target = self.dominant_color.copy()
        self.dominant_idx = 0
        self._pending_hits = []  # list of (band_index, strength)
        self._lock = threading.Lock()

        # ── Sparkle state (render thread only) ──
        self.sparkle_pos = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_brightness = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_width = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_color = np.zeros((MAX_SPARKLES, 3), dtype=np.float32)
        self.sparkle_active = np.zeros(MAX_SPARKLES, dtype=bool)
        self.sparkle_band = np.full(MAX_SPARKLES, -1, dtype=np.int32)
        self.sparkle_next = 0

        # Per-pixel background fade-in opacity (0 = suppressed by pulse, 1 = full bg)
        self.bg_opacity = np.ones(num_leds, dtype=np.float32)

        # Per-band zone offset: drifts smoothly, new target every ~10s
        self.zone_offset = np.zeros(self.n_bands, dtype=np.float32)
        self.zone_offset_target = np.random.uniform(-0.3, 0.3, self.n_bands).astype(np.float32)
        self.zone_offset_timer = 0.0
        self.zone_offset_interval = 10.0  # seconds between new targets

    @property
    def name(self):
        return "Band Zone Pulse"

    @property
    def description(self):
        return "Frequency-zoned percussive pulses: each band lights its own strip zone via streaming HPSS."

    # ── Audio processing ─────────────────────────────────────────────

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame):
        spec = np.abs(np.fft.rfft(frame * self.window))

        # ── Streaming HPSS ──
        self.spec_buf[self.spec_buf_idx] = spec
        self.spec_buf_idx = (self.spec_buf_idx + 1) % self.hpss_buf_size
        self.spec_buf_filled = min(self.spec_buf_filled + 1, self.hpss_buf_size)

        if self.spec_buf_filled >= 3:
            buf_slice = self.spec_buf[:self.spec_buf_filled]
            harmonic_mask = np.median(buf_slice, axis=0)
            percussive = np.maximum(spec - harmonic_mask, 0)
        else:
            harmonic_mask = spec * 0.5
            percussive = spec * 0.5

        # ── Full-spectrum band energy → background color ──
        # (uses full spectrum, not harmonic-only, to match what the ear hears)
        band_energies = np.array(
            [np.sum(spec[m] ** 2) for m in self.band_masks],
            dtype=np.float32)

        for i in range(self.n_bands):
            self.band_peaks[i] = max(
                band_energies[i], self.band_peaks[i] * self.band_peak_decay)
        normalized = band_energies / self.band_peaks

        # This frame's winner (3 groups, averaged)
        group_energy = np.array([
            (normalized[0] + normalized[1]) / 2,  # low
            normalized[2],                         # mid
            (normalized[3] + normalized[4]) / 2,  # high
        ])
        frame_winner = int(np.argmax(group_energy))

        # Store vote in ring buffer
        v_idx = self.vote_ring_pos % self.vote_window_len
        self.vote_ring[v_idx] = frame_winner
        self.vote_ring_pos += 1
        self.vote_ring_filled = min(self.vote_ring_filled + 1, self.vote_window_len)

        # Count votes: which group won the most frames in the window
        votes = self.vote_ring[:self.vote_ring_filled]
        vote_counts = np.array([np.sum(votes == g) for g in range(3)])
        dominant_group = int(np.argmax(vote_counts))

        bg_palette = np.array([
            [180, 40, 80],    # pink
            [255, 60, 0],     # orange
            [40, 100, 180],   # blue
        ], dtype=np.float32)
        color = bg_palette[dominant_group].copy()

        # ── Percussive per-band peak detection ──
        perc_energies = np.array(
            [np.sum(percussive[m] ** 2) for m in self.band_masks],
            dtype=np.float32)

        # Store in history ring
        h_idx = self.perc_hist_idx % self.perc_history_len
        self.perc_history[h_idx] = perc_energies
        self.perc_hist_idx += 1
        self.perc_hist_filled = min(self.perc_hist_filled + 1, self.perc_history_len)

        # Tick cooldowns
        self.perc_cooldown_counters = np.maximum(self.perc_cooldown_counters - 1, 0)

        hits = []
        # Update per-band slow-decay peaks (for absolute floor)
        for i in range(self.n_bands):
            self.perc_band_peaks[i] = max(
                perc_energies[i], self.perc_band_peaks[i] * self.perc_peak_decay)

        # Band's share of total percussive energy — dominant band gate
        total_perc = np.sum(perc_energies)
        if total_perc > 0:
            band_shares = perc_energies / total_perc
        else:
            band_shares = np.zeros(self.n_bands)

        if self.perc_hist_filled >= 3:
            hist = self.perc_history[:self.perc_hist_filled]
            means = np.mean(hist, axis=0)
            stds = np.std(hist, axis=0)

            for i in range(self.n_bands):
                if self.perc_cooldown_counters[i] > 0:
                    continue
                # Must have significant share of total percussive energy
                if band_shares[i] < 0.15:
                    continue
                threshold = means[i] + 2.5 * stds[i]
                # Floor: must exceed 5% of the band's own peak to ignore quiet noise
                floor = self.perc_band_peaks[i] * 0.05
                if perc_energies[i] > threshold and perc_energies[i] > floor:
                    # Brightness: band share scales the strength
                    excess = perc_energies[i] - threshold
                    headroom = self.perc_band_peaks[i] - threshold
                    raw_strength = min(1.0, excess / (headroom + 1e-10))
                    strength = raw_strength * band_shares[i] / max(band_shares)
                    hits.append((i, max(0.3, strength)))
                    self.perc_cooldown_counters[i] = self.perc_cooldown_frames

        with self._lock:
            self.dominant_color_target = color
            self.dominant_idx = dominant_group
            if hits:
                self._pending_hits.extend(hits)

    # ── Sparkle management ───────────────────────────────────────────

    def _spawn_sparkle(self, band_idx, intensity):
        """Spawn or re-trigger a sparkle. Same band = re-trigger, else new slot."""
        zone_size = self.num_leds / self.n_bands
        zone_start = band_idx * zone_size

        # Check for existing sparkle in same band → re-trigger in place
        for j in range(MAX_SPARKLES):
            if self.sparkle_active[j] and self.sparkle_band[j] == band_idx:
                self.sparkle_brightness[j] = max(self.sparkle_brightness[j], intensity)
                return

        # Pick random position within zone, with drifting offset
        offset_frac = np.clip(np.random.uniform(0, 1) + self.zone_offset[band_idx], 0.0, 1.0)
        target_pos = zone_start + offset_frac * zone_size

        # Find a free slot
        best = -1
        for j in range(MAX_SPARKLES):
            if not self.sparkle_active[j]:
                best = j
                break
        if best == -1:
            best = self.sparkle_next % MAX_SPARKLES
            self.sparkle_next += 1

        i = best
        self.sparkle_active[i] = True
        self.sparkle_band[i] = band_idx
        self.sparkle_pos[i] = target_pos
        self.sparkle_width[i] = np.random.uniform(3.0, 5.0)
        self.sparkle_brightness[i] = intensity

        # Color from the band that produced the hit, with ±15% variation
        variation = 1.0 + np.random.uniform(-0.15, 0.15, 3)
        self.sparkle_color[i] = np.clip(
            BAND_COLORS[band_idx] * variation, 0, 255)

    # ── Render ───────────────────────────────────────────────────────

    def render(self, dt: float) -> np.ndarray:
        step = dt * 30

        # Drain pending hits
        with self._lock:
            target = self.dominant_color_target.copy()
            hits = self._pending_hits[:]
            self._pending_hits.clear()

        # Smooth background color (~2-3s crossfade)
        alpha = 1.0 - 0.98 ** step
        self.dominant_color += (target - self.dominant_color) * alpha

        # Drift zone offsets toward targets, pick new targets every ~10s
        self.zone_offset_timer += dt
        if self.zone_offset_timer >= self.zone_offset_interval:
            self.zone_offset_target = np.random.uniform(-0.3, 0.3, self.n_bands).astype(np.float32)
            self.zone_offset_timer = 0.0
        drift_alpha = 1.0 - 0.97 ** step  # smooth ~1s transition
        self.zone_offset += (self.zone_offset_target - self.zone_offset) * drift_alpha

        # Spawn sparkles for each percussive hit
        for band_idx, strength in hits:
            count = max(1, int(strength * 3 + 0.5))
            for _ in range(count):
                self._spawn_sparkle(band_idx, strength)

        # Decay existing sparkles
        decay = 0.88 ** step
        for i in range(MAX_SPARKLES):
            if not self.sparkle_active[i]:
                continue
            self.sparkle_brightness[i] *= decay
            if self.sparkle_brightness[i] < 0.04:
                self.sparkle_active[i] = False
                self.sparkle_band[i] = -1

        # Build pulse layer (black base — pulses fade to black)
        pulse_frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        for i in range(MAX_SPARKLES):
            if not self.sparkle_active[i]:
                continue

            b = self.sparkle_brightness[i]
            half_w = self.sparkle_width[i] / 2.0
            center = self.sparkle_pos[i]
            start = int(center - half_w) - 1
            end = int(center + half_w) + 1

            for p in range(start, end + 1):
                if p < 0 or p >= self.num_leds:
                    continue
                pixel = p

                dist = abs(center - p)
                if dist >= half_w:
                    continue

                # Flat brightness across the pulse (no spatial falloff)
                contribution = self.sparkle_color[i] * b
                nonzero = self.sparkle_color[i] > 0
                contribution[nonzero] = np.maximum(contribution[nonzero], 1.0)
                if np.max(contribution) > np.max(pulse_frame[pixel]):
                    pulse_frame[pixel] = contribution

        # Update per-pixel background opacity
        has_pulse = np.max(pulse_frame, axis=1) > 0
        # Pulse active → suppress background instantly
        self.bg_opacity[has_pulse] = 0.0
        # No pulse → fade background back in (~0.5s)
        fade_in = ~has_pulse
        self.bg_opacity[fade_in] += 0.07 * step
        np.clip(self.bg_opacity, 0.0, 1.0, out=self.bg_opacity)

        # Composite: background at per-pixel opacity, pulse replaces where active
        base_color = self.dominant_color * 0.10
        frame = np.outer(self.bg_opacity, base_color).reshape(self.num_leds, 3).astype(np.float32)
        frame[has_pulse] = pulse_frame[has_pulse]

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            idx = self.dominant_idx
        n_active = int(np.sum(self.sparkle_active))
        # Count active sparkles per zone
        zone_size = self.num_leds / self.n_bands
        zones = ['_'] * self.n_bands
        zone_labels = ['Sub', 'Bas', 'Mid', 'HiM', 'Tre']
        for i in range(MAX_SPARKLES):
            if self.sparkle_active[i]:
                z = min(int(self.sparkle_pos[i] / zone_size), self.n_bands - 1)
                zones[z] = zone_labels[z]
        bg_names = ['pink', 'orng', 'blue']
        return {
            'bg': bg_names[idx],
            'n': n_active,
            'zones': ' '.join(zones),
        }
