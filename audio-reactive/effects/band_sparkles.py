"""
Band Sparkles — twinkling LEDs over an audio-reactive base color.

The base color of the whole strip shifts slowly based on which frequency
band has been dominant over the past ~5 seconds of music. Band colors
match the band energy display in the analysis tab:

  Sub-bass (20-80 Hz)   → red        #FF1744
  Bass (80-250 Hz)      → orange     #FF9100
  Mids (250-2kHz)       → yellow     #FFEA00
  High-mids (2-6kHz)    → green      #00E676
  Treble (6-8kHz)       → blue       #00B0FF

Individual LEDs twinkle on top in the same color family, averaging
~20 active at any time. The twinkle color tracks the dominant band.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


# Band definitions matching viewer.py exactly
BANDS = [
    ('Sub-bass', 20, 80),
    ('Bass', 80, 250),
    ('Mids', 250, 2000),
    ('High-mids', 2000, 6000),
    ('Treble', 6000, 8000),
]

# RGB colors matching the band energy display (viewer.py BAND_COLORS)
BAND_COLORS = np.array([
    [255, 23, 68],     # Sub-bass: #FF1744
    [255, 145, 0],     # Bass:     #FF9100
    [255, 234, 0],     # Mids:     #FFEA00
    [0, 230, 118],     # High-mids:#00E676
    [0, 176, 255],     # Treble:   #00B0FF
], dtype=np.float32)


class BandSparklesEffect(AudioReactiveEffect):
    """Twinkles over a slowly-shifting band-dominant base color."""

    registry_name = 'band_sparkles'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # FFT setup
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)

        # Precompute band masks
        self.n_bands = len(BANDS)
        self.band_masks = []
        for _, lo, hi in BANDS:
            self.band_masks.append((self.freq_bins >= lo) & (self.freq_bins < hi))

        # Audio accumulation
        self.audio_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.audio_buf_pos = 0

        # Per-band normalization: slow-decay peak so each band is 0-1
        # relative to its own history (bass has 100x more raw energy than
        # treble, so without this bass always wins)
        self.band_peaks = np.full(self.n_bands, 1e-10, dtype=np.float32)
        self.band_peak_decay = 0.9995  # very slow decay (~30s half-life)

        # Rolling normalized energy integral: ~5 seconds at ~22 frames/sec
        self.window_len = int(5 * sample_rate / self.n_fft)
        self.energy_ring = np.zeros((self.window_len, self.n_bands), dtype=np.float32)
        self.ring_pos = 0
        self.ring_filled = 0

        # Dominant color (smoothed for display)
        self.dominant_color = np.array([15, 1, 6], dtype=np.float32)  # start dim red
        self.dominant_idx = 0
        self._lock = threading.Lock()

        # Twinkle state
        self.twinkle_brightness = np.zeros(num_leds, dtype=np.float32)
        self.twinkle_target = np.zeros(num_leds, dtype=np.float32)
        self.twinkle_speed = np.zeros(num_leds, dtype=np.float32)
        self.twinkle_colors = np.zeros((num_leds, 3), dtype=np.float32)
        self.target_active = 20

    @property
    def name(self):
        return "Band Sparkles"

    @property
    def description(self):
        return "Twinkles over a base color that shifts with the dominant frequency band (5s rolling integral)."

    def process_audio(self, mono_chunk: np.ndarray):
        n = len(mono_chunk)
        pos = self.audio_buf_pos

        while n > 0:
            space = self.n_fft - pos
            take = min(n, space)
            self.audio_buf[pos:pos + take] = mono_chunk[:take]
            mono_chunk = mono_chunk[take:]
            pos += take
            n -= take

            if pos >= self.n_fft:
                self._process_frame(self.audio_buf.copy())
                pos = 0

        self.audio_buf_pos = pos

    def _process_frame(self, frame):
        spec = np.abs(np.fft.rfft(frame * self.window))
        # Energy per band (sum of squared magnitudes)
        energies = np.array([np.sum(spec[mask] ** 2) for mask in self.band_masks],
                            dtype=np.float32)

        # Normalize each band by its own slow-decay peak so all bands
        # compete on a 0-1 scale (raw bass energy is 100x treble)
        for i in range(self.n_bands):
            self.band_peaks[i] = max(energies[i], self.band_peaks[i] * self.band_peak_decay)
        normalized = energies / self.band_peaks

        # Store normalized values in rolling buffer
        idx = self.ring_pos % self.window_len
        self.energy_ring[idx] = normalized
        self.ring_pos += 1
        self.ring_filled = min(self.ring_filled + 1, self.window_len)

        # Compute 20-second integral of normalized energy per band
        filled = self.energy_ring[:self.ring_filled]
        band_integrals = np.sum(filled, axis=0)

        # Sharpen: raise proportions to a power so the winner dominates
        total = np.sum(band_integrals)
        if total > 0:
            proportions = band_integrals / total
            sharpened = proportions ** 3
            sharp_total = np.sum(sharpened)
            if sharp_total > 0:
                weights = sharpened / sharp_total
            else:
                weights = proportions
        else:
            weights = np.ones(self.n_bands) / self.n_bands

        # Pick the single dominant band's color (no blending)
        dominant = int(np.argmax(weights))
        color = BAND_COLORS[dominant].copy()

        with self._lock:
            self.dominant_color_target = color
            self.dominant_idx = dominant
            self._debug_weights = weights.copy()
            self._debug_color = color.copy()
            self._debug_proportions = proportions.copy() if total > 0 else weights.copy()

    def _pick_twinkle_color(self):
        """Twinkle color: the current dominant color with slight random variation."""
        with self._lock:
            base = self.dominant_color.copy()
        # Add ±15% variation per channel for organic feel
        variation = 1.0 + np.random.uniform(-0.15, 0.15, 3)
        return np.clip(base * variation, 0, 255).astype(np.float32)

    def render(self, dt: float) -> np.ndarray:
        step = dt * 30

        # Smoothly move displayed color toward target
        with self._lock:
            target = getattr(self, 'dominant_color_target', self.dominant_color)
        # Slow smoothing (~2-3 second transition)
        alpha = 1.0 - 0.98 ** step
        self.dominant_color += (target - self.dominant_color) * alpha

        # Base glow: dim version of dominant color (~6% brightness)
        base_color = self.dominant_color * 0.06

        # Update twinkles toward their targets
        for i in range(self.num_leds):
            if self.twinkle_brightness[i] == 0 and self.twinkle_target[i] == 0:
                continue

            diff = self.twinkle_target[i] - self.twinkle_brightness[i]
            move = self.twinkle_speed[i] * step

            if abs(diff) <= move:
                self.twinkle_brightness[i] = self.twinkle_target[i]
                if self.twinkle_target[i] > 0:
                    self.twinkle_target[i] = 0
                    self.twinkle_speed[i] = np.random.uniform(0.008, 0.025)
            else:
                self.twinkle_brightness[i] += move if diff > 0 else -move

        # Spawn new twinkles
        active = np.sum(self.twinkle_brightness > 0)
        deficit = self.target_active - active
        if deficit > 0:
            spawns = min(int(deficit * 0.3) + 1, 3)
            inactive = np.where(self.twinkle_brightness == 0)[0]
            if len(inactive) > 0:
                chosen = np.random.choice(inactive, size=min(spawns, len(inactive)), replace=False)
                for idx in chosen:
                    self.twinkle_target[idx] = np.random.uniform(0.4, 1.0)
                    self.twinkle_speed[idx] = np.random.uniform(0.01, 0.03)
                    self.twinkle_brightness[idx] = 0.001
                    self.twinkle_colors[idx] = self._pick_twinkle_color()

        # Build frame
        frame = np.tile(base_color, (self.num_leds, 1))
        mask = self.twinkle_brightness > 0
        if np.any(mask):
            intensities = self.twinkle_brightness[mask, np.newaxis]
            frame[mask] += self.twinkle_colors[mask] * intensities

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            idx = self.dominant_idx
            w = getattr(self, '_debug_weights', np.zeros(self.n_bands))
            c = getattr(self, '_debug_color', np.zeros(3))
            p = getattr(self, '_debug_proportions', np.zeros(self.n_bands))
        # Show pre-sharpened proportions, post-sharpened weights, and resulting RGB
        band_names = ['Sub', 'Bas', 'Mid', 'HiM', 'Tre']
        prop_str = ' '.join(f'{band_names[i]}={p[i]:.2f}' for i in range(self.n_bands))
        wt_str = ' '.join(f'{band_names[i]}={w[i]:.2f}' for i in range(self.n_bands))
        return {
            'band': BANDS[idx][0],
            'props': prop_str,
            'weights': wt_str,
            'rgb': f'{int(c[0])},{int(c[1])},{int(c[2])}',
            'twinkles': int(np.sum(self.twinkle_brightness > 0)),
        }
