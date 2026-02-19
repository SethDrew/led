"""
Band Sparkle Flux — spectral flux variant for A/B comparison.

Same visual effect as band_sparkle_band_fade_bg, but replaces streaming HPSS
with per-band half-wave rectified spectral flux for transient detection,
and uses raw band energy (no harmonic filtering) for background color.

Compare visually against band_sparkle_band_fade_bg to see which transient
detection approach produces more appealing sparkle patterns.
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

BAND_COLORS = np.array([
    [180, 0, 0],       # Sub-bass: dark red
    [255, 100, 0],     # Bass:     orange
    [255, 140, 0],     # Mids:     orange (tuned for COB)
    [0, 230, 118],     # High-mids: green
    [0, 176, 255],     # Treble:   blue
], dtype=np.float32)

MAX_SPARKLES = 40


class BandSparkleFluxEffect(AudioReactiveEffect):
    """Spectral flux variant: raw band energy for background, per-band flux for sparkles."""

    registry_name = 'band_sparkle_flux'

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

        # ── Background: raw band energy (no HPSS) ──
        self.band_peaks = np.full(self.n_bands, 1e-10, dtype=np.float32)
        self.band_peak_decay = 0.9995

        self.band_window_len = int(5 * sample_rate / self.n_fft)
        self.band_ring = np.zeros((self.band_window_len, self.n_bands), dtype=np.float32)
        self.band_ring_pos = 0
        self.band_ring_filled = 0

        # ── Spectral flux state (4-frame running average) ──
        self.flux_avg_len = 4
        self.spec_rings = [
            np.zeros((self.flux_avg_len, int(np.sum(m))), dtype=np.float32)
            for m in self.band_masks
        ]
        self.spec_ring_idx = 0
        self.spec_ring_filled = 0

        # Per-band flux peak detection (same threshold logic as HPSS version)
        self.flux_history_len = 8
        self.flux_history = np.zeros((self.flux_history_len, self.n_bands), dtype=np.float32)
        self.flux_hist_idx = 0
        self.flux_hist_filled = 0
        self.flux_cooldown_frames = 3
        self.flux_cooldown_counters = np.zeros(self.n_bands, dtype=np.int32)
        self.flux_band_peaks = np.full(self.n_bands, 1e-10, dtype=np.float32)
        self.flux_peak_decay = 0.998

        # ── Per-band centroid range tracking ──
        self.centroid_min = np.array([lo for _, lo, _ in BANDS], dtype=np.float32)
        self.centroid_max = np.array([hi for _, _, hi in BANDS], dtype=np.float32)
        self.centroid_decay = 0.9998

        # ── Shared state (audio → render) ──
        self.dominant_color = np.array([15, 1, 6], dtype=np.float32)
        self.dominant_color_target = self.dominant_color.copy()
        self.dominant_idx = 0
        self._pending_hits = []
        self._lock = threading.Lock()

        # Debug state
        self._debug_weights = np.zeros(self.n_bands, dtype=np.float32)
        self._debug_proportions = np.zeros(self.n_bands, dtype=np.float32)
        self._debug_color = np.zeros(3, dtype=np.float32)

        # ── Sparkle state (render thread only) ──
        self.sparkle_pos = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_brightness = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_width = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_color = np.zeros((MAX_SPARKLES, 3), dtype=np.float32)
        self.sparkle_active = np.zeros(MAX_SPARKLES, dtype=bool)
        self.sparkle_next = 0

    @property
    def name(self):
        return "Band Sparkle Flux"

    @property
    def description(self):
        return ("Spectral flux variant: raw band energy for background, "
                "per-band half-wave spectral flux for sparkle triggers.")

    # ── Audio processing ─────────────────────────────────────────────

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame):
        spec = np.abs(np.fft.rfft(frame * self.window))

        # ── Background: raw band energy → dominant color ──
        raw_energies = np.array(
            [np.sum(spec[m] ** 2) for m in self.band_masks],
            dtype=np.float32)

        for i in range(self.n_bands):
            self.band_peaks[i] = max(
                raw_energies[i], self.band_peaks[i] * self.band_peak_decay)
        normalized = raw_energies / self.band_peaks

        idx = self.band_ring_pos % self.band_window_len
        self.band_ring[idx] = normalized
        self.band_ring_pos += 1
        self.band_ring_filled = min(self.band_ring_filled + 1, self.band_window_len)

        filled = self.band_ring[:self.band_ring_filled]
        integrals = np.sum(filled, axis=0)

        total = np.sum(integrals)
        if total > 0:
            proportions = integrals / total
            sharpened = proportions ** 3
            s_total = np.sum(sharpened)
            weights = sharpened / s_total if s_total > 0 else proportions
        else:
            proportions = np.ones(self.n_bands) / self.n_bands
            weights = proportions.copy()

        dominant = int(np.argmax(weights))
        color = BAND_COLORS[dominant].copy()

        # ── Per-band half-wave rectified spectral flux (vs 4-frame avg) ──
        ring_idx = self.spec_ring_idx % self.flux_avg_len
        flux_values = np.zeros(self.n_bands, dtype=np.float32)
        for i, m in enumerate(self.band_masks):
            band_spec = spec[m]
            if self.spec_ring_filled >= 1:
                avg = np.mean(self.spec_rings[i][:self.spec_ring_filled], axis=0)
                diff = band_spec - avg
                flux_values[i] = np.sum(np.maximum(diff, 0) ** 2)
            self.spec_rings[i][ring_idx] = band_spec
        self.spec_ring_idx += 1
        self.spec_ring_filled = min(self.spec_ring_filled + 1, self.flux_avg_len)

        # Store in history ring
        h_idx = self.flux_hist_idx % self.flux_history_len
        self.flux_history[h_idx] = flux_values
        self.flux_hist_idx += 1
        self.flux_hist_filled = min(self.flux_hist_filled + 1, self.flux_history_len)

        # Tick cooldowns
        self.flux_cooldown_counters = np.maximum(self.flux_cooldown_counters - 1, 0)

        hits = []
        for i in range(self.n_bands):
            self.flux_band_peaks[i] = max(
                flux_values[i], self.flux_band_peaks[i] * self.flux_peak_decay)

        # Band's share of total flux — dominant band gate
        total_flux = np.sum(flux_values)
        if total_flux > 0:
            band_shares = flux_values / total_flux
        else:
            band_shares = np.zeros(self.n_bands)

        if self.flux_hist_filled >= 3:
            hist = self.flux_history[:self.flux_hist_filled]
            means = np.mean(hist, axis=0)
            stds = np.std(hist, axis=0)

            for i in range(self.n_bands):
                if self.flux_cooldown_counters[i] > 0:
                    continue
                if band_shares[i] < 0.15:
                    continue
                threshold = means[i] + 2.5 * stds[i]
                floor = self.flux_band_peaks[i] * 0.05
                if flux_values[i] > threshold and flux_values[i] > floor:
                    excess = flux_values[i] - threshold
                    headroom = self.flux_band_peaks[i] - threshold
                    raw_strength = min(1.0, excess / (headroom + 1e-10))
                    strength = raw_strength * band_shares[i] / max(band_shares)
                    # Spectral centroid within band → position
                    mask = self.band_masks[i]
                    band_spec = spec[mask] ** 2
                    band_total = np.sum(band_spec)
                    if band_total > 0:
                        freq_pos = np.sum(self.freq_bins[mask] * band_spec) / band_total
                        lo_obs = self.centroid_min[i]
                        hi_obs = self.centroid_max[i]
                        band_mid = (BANDS[i][1] + BANDS[i][2]) * 0.5
                        self.centroid_min[i] = min(freq_pos, lo_obs + (band_mid - lo_obs) * (1 - self.centroid_decay))
                        self.centroid_max[i] = max(freq_pos, hi_obs - (hi_obs - band_mid) * (1 - self.centroid_decay))
                        span = self.centroid_max[i] - self.centroid_min[i]
                        if span > 0:
                            zone_frac = (freq_pos - self.centroid_min[i]) / span
                        else:
                            zone_frac = 0.5
                        zone_frac = np.clip(zone_frac, 0.0, 1.0)
                    else:
                        zone_frac = 0.5
                    hits.append((i, max(0.3, strength), zone_frac))
                    self.flux_cooldown_counters[i] = self.flux_cooldown_frames

        with self._lock:
            self.dominant_color_target = color
            self.dominant_idx = dominant
            self._debug_weights = weights.copy()
            self._debug_proportions = proportions.copy()
            self._debug_color = color.copy()
            if hits:
                self._pending_hits.extend(hits)

    # ── Sparkle management ───────────────────────────────────────────

    def _spawn_sparkle(self, band_idx, intensity, zone_frac):
        """Spawn a wide sparkle colored by the hit's band, positioned by frequency."""
        i = self.sparkle_next % MAX_SPARKLES
        self.sparkle_next += 1

        self.sparkle_active[i] = True
        zone_size = self.num_leds / self.n_bands
        zone_start = band_idx * zone_size
        self.sparkle_pos[i] = zone_start + zone_frac * zone_size
        self.sparkle_width[i] = np.random.uniform(3.0, 5.0)
        self.sparkle_brightness[i] = intensity

        variation = 1.0 + np.random.uniform(-0.15, 0.15, 3)
        self.sparkle_color[i] = np.clip(
            BAND_COLORS[band_idx] * variation, 0, 255)

    # ── Render ───────────────────────────────────────────────────────

    def render(self, dt: float) -> np.ndarray:
        step = dt * 30

        with self._lock:
            target = self.dominant_color_target.copy()
            hits = self._pending_hits[:]
            self._pending_hits.clear()

        alpha = 1.0 - 0.98 ** step
        self.dominant_color += (target - self.dominant_color) * alpha

        for band_idx, strength, zone_frac in hits:
            count = max(1, int(strength * 3 + 0.5))
            for _ in range(count):
                self._spawn_sparkle(band_idx, strength, zone_frac)

        decay = 0.88 ** step
        for i in range(MAX_SPARKLES):
            if not self.sparkle_active[i]:
                continue
            self.sparkle_brightness[i] *= decay
            if self.sparkle_brightness[i] < 0.04:
                self.sparkle_active[i] = False

        base_color = self.dominant_color * 0.0  # background off for testing
        frame = np.tile(base_color, (self.num_leds, 1)).astype(np.float32)

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

                norm = dist / half_w
                intensity = (1 - norm * norm)
                intensity = intensity * intensity
                intensity *= b

                frame[pixel] += self.sparkle_color[i] * intensity

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            idx = self.dominant_idx
        n_active = int(np.sum(self.sparkle_active))
        zone_size = self.num_leds / self.n_bands
        zones = ['_'] * self.n_bands
        zone_labels = ['Sub', 'Bas', 'Mid', 'HiM', 'Tre']
        for i in range(MAX_SPARKLES):
            if self.sparkle_active[i]:
                z = min(int(self.sparkle_pos[i] / zone_size), self.n_bands - 1)
                zones[z] = zone_labels[z]
        return {
            'bg': BANDS[idx][0][:3],
            'n': n_active,
            'zones': ' '.join(zones),
        }
