"""
Band Tempo Sparkles — beat-triggered wide sparkles in band-dominant color.

Combines:
  - Band-dominant color (5-band FFT, 5s rolling integral, per-band peak
    normalization) — same as band_sparkles
  - Tempo estimation via autocorrelation of abs-integral signal — same
    algorithm as absint_predictive (the best beat tracker we have)
  - Wide sparkles (3-5 LEDs) that spawn at random positions on each
    detected/predicted beat, then fade out with soft edges

The base glow is a dim version of the dominant band color. Sparkles
flash on beats in the same color family.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, AbsIntegral, BeatPredictor


# Band definitions matching viewer.py exactly
BANDS = [
    ('Sub-bass', 20, 80),
    ('Bass', 80, 250),
    ('Mids', 250, 2000),
    ('High-mids', 2000, 6000),
    ('Treble', 6000, 8000),
]

BAND_COLORS = np.array([
    [255, 23, 68],     # Sub-bass: #FF1744
    [255, 145, 0],     # Bass:     #FF9100
    [255, 234, 0],     # Mids:     #FFEA00
    [0, 230, 118],     # High-mids:#00E676
    [0, 176, 255],     # Treble:   #00B0FF
], dtype=np.float32)

MAX_SPARKLES = 30


class BandTempoSparklesEffect(AudioReactiveEffect):
    """Beat-triggered wide sparkles colored by the dominant frequency band."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Band energy (same as band_sparkles)
        self.n_fft_band = 2048
        self.band_window = np.hanning(self.n_fft_band).astype(np.float32)
        self.band_freq_bins = np.fft.rfftfreq(self.n_fft_band, 1.0 / sample_rate)

        self.n_bands = len(BANDS)
        self.band_masks = []
        for _, lo, hi in BANDS:
            self.band_masks.append(
                (self.band_freq_bins >= lo) & (self.band_freq_bins < hi))

        self.band_peaks = np.full(self.n_bands, 1e-10, dtype=np.float32)
        self.band_peak_decay = 0.9995

        self.band_window_len = int(5 * sample_rate / self.n_fft_band)
        self.band_ring = np.zeros(
            (self.band_window_len, self.n_bands), dtype=np.float32)
        self.band_ring_pos = 0
        self.band_ring_filled = 0

        self.dominant_color = np.array([15, 1, 6], dtype=np.float32)
        self.dominant_color_target = self.dominant_color.copy()
        self.dominant_idx = 0

        # Beat detection (using signal primitives)
        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)
        self.predictor = BeatPredictor(rms_fps=self.absint.rms_fps)

        # Sparkle state
        self.sparkle_pos = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_brightness = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_width = np.zeros(MAX_SPARKLES, dtype=np.float32)
        self.sparkle_color = np.zeros((MAX_SPARKLES, 3), dtype=np.float32)
        self.sparkle_active = np.zeros(MAX_SPARKLES, dtype=bool)
        self.sparkle_next = 0  # round-robin index

        # Pending beat spawns (set in audio thread, consumed in render)
        self._pending_beats = []
        self._lock = threading.Lock()

    @property
    def name(self):
        return "Band Tempo Sparkles"

    @property
    def description(self):
        return ("Wide sparkles triggered by predicted beats, colored by "
                "the dominant frequency band (5s rolling integral).")

    # ── Audio processing ─────────────────────────────────────────────

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            # Process band energy
            self._process_bands(frame)

            # Process beat detection
            normalized = self.absint.update(frame)
            beats = self.predictor.feed(self.absint.raw, normalized, self.absint.time_acc)

            for beat in beats:
                with self._lock:
                    self._pending_beats.append(beat['strength'])

    def _process_bands(self, frame):
        """Compute per-band energy and update dominant color target."""
        spec = np.abs(np.fft.rfft(frame * self.band_window))
        energies = np.array(
            [np.sum(spec[m] ** 2) for m in self.band_masks], dtype=np.float32)

        for i in range(self.n_bands):
            self.band_peaks[i] = max(
                energies[i], self.band_peaks[i] * self.band_peak_decay)
        normalized = energies / self.band_peaks

        idx = self.band_ring_pos % self.band_window_len
        self.band_ring[idx] = normalized
        self.band_ring_pos += 1
        self.band_ring_filled = min(
            self.band_ring_filled + 1, self.band_window_len)

        filled = self.band_ring[:self.band_ring_filled]
        integrals = np.sum(filled, axis=0)

        total = np.sum(integrals)
        if total > 0:
            props = integrals / total
            sharpened = props ** 3
            s_total = np.sum(sharpened)
            weights = sharpened / s_total if s_total > 0 else props
        else:
            weights = np.ones(self.n_bands) / self.n_bands

        color = np.zeros(3, dtype=np.float32)
        for i in range(self.n_bands):
            color += BAND_COLORS[i] * weights[i]

        with self._lock:
            self.dominant_color_target = color
            self.dominant_idx = int(np.argmax(weights))

    # ── Sparkle management ───────────────────────────────────────────

    def _spawn_sparkle(self, intensity):
        """Spawn a wide sparkle at a random position."""
        i = self.sparkle_next % MAX_SPARKLES
        self.sparkle_next += 1

        self.sparkle_active[i] = True
        self.sparkle_pos[i] = np.random.uniform(0, self.num_leds)
        self.sparkle_width[i] = np.random.uniform(3.0, 5.0)
        self.sparkle_brightness[i] = intensity

        # Color from current dominant color with slight variation
        variation = 1.0 + np.random.uniform(-0.15, 0.15, 3)
        self.sparkle_color[i] = np.clip(
            self.dominant_color * variation, 0, 255)

    # ── Render ───────────────────────────────────────────────────────

    def render(self, dt: float) -> np.ndarray:
        step = dt * 30

        # Smooth dominant color
        with self._lock:
            target = self.dominant_color_target.copy()
            beats = self._pending_beats[:]
            self._pending_beats.clear()

        alpha = 1.0 - 0.98 ** step
        self.dominant_color += (target - self.dominant_color) * alpha

        # Spawn sparkles for pending beats
        for intensity in beats:
            # 2-4 sparkles per beat depending on intensity
            count = max(2, int(intensity * 4 + 0.5))
            for _ in range(count):
                self._spawn_sparkle(intensity)

        # Decay existing sparkles
        decay = 0.88 ** step
        for i in range(MAX_SPARKLES):
            if not self.sparkle_active[i]:
                continue
            self.sparkle_brightness[i] *= decay
            if self.sparkle_brightness[i] < 0.02:
                self.sparkle_active[i] = False

        # Build frame: dim base + sparkles
        base_color = self.dominant_color * 0.06
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
                pixel = int(p) % self.num_leds

                dist = abs(center - p)
                if dist > self.num_leds / 2:
                    dist = self.num_leds - dist
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
        return {
            'band': BANDS[idx][0],
            'bpm': f'{self.predictor.bpm:.1f}',
            'ac_conf': f'{self.predictor.confidence:.2f}',
            'beats': self.predictor.beat_count,
            'predicted': self.predictor.predicted_beat_count,
            'sparkles': int(np.sum(self.sparkle_active)),
        }
