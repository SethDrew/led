"""
Band Proportional — three-band abs-integral mapped to RGB color.

Computes abs-integral of RMS derivative in three frequency bands independently:
  Bass  (20-250 Hz)  → Red channel
  Mid   (250-2000 Hz) → Green channel
  High  (2000-10kHz)  → Blue channel

The color tells you what's changing:
  Red    = kick drum / bass transient
  Green  = snare / guitar / vocal onset
  Blue   = hi-hat / cymbal
  White  = full-kit accent (all bands)
  Yellow = kick + snare (bass + mid)
  Magenta = kick + hi-hat (bass + high)

No beat detection or thresholds — proportional mapping with fast attack,
slow decay. Works on any LED count (strip or tree).
"""

import numpy as np
import threading
from scipy.signal import butter, sosfilt
from base import AudioReactiveEffect


class BandProportionalEffect(AudioReactiveEffect):
    """Three-band abs-integral → RGB color mapping."""

    registry_name = 'band_prop'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Band definitions: (low_hz, high_hz)
        self.bands = [
            (20, 250),      # bass → red
            (250, 2000),    # mid → green
            (2000, 10000),  # high → blue
        ]
        self.n_bands = len(self.bands)

        # Build bandpass filters (Butterworth, 4th order)
        self.filters = []
        nyq = sample_rate / 2
        for low, high in self.bands:
            low_n = max(low / nyq, 0.001)
            high_n = min(high / nyq, 0.999)
            sos = butter(4, [low_n, high_n], btype='band', output='sos')
            self.filters.append(sos)

        # Filter state (for streaming — maintains continuity between chunks)
        self.filter_states = [np.zeros((sos.shape[0], 2)) for sos in self.filters]

        # RMS computation per band
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0

        # Per-band state
        self.prev_rms = np.zeros(self.n_bands, dtype=np.float32)

        # Abs-integral ring buffers (one per band)
        self.window_sec = 0.15
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_bufs = [np.zeros(self.window_frames, dtype=np.float32)
                           for _ in range(self.n_bands)]
        self.deriv_buf_pos = 0

        # Per-band abs-integral and normalization
        self.abs_integrals = np.zeros(self.n_bands, dtype=np.float32)
        self.integral_peaks = np.full(self.n_bands, 1e-10, dtype=np.float32)
        self.peak_decay = 0.998

        # Visual state
        self.target_rgb = np.zeros(3, dtype=np.float32)  # normalized 0-1
        self.display_rgb = np.zeros(3, dtype=np.float32)
        self.attack_rate = 0.6
        self.decay_rate = 0.85

        # Color mapping: band → RGB contribution
        # Bass → warm red, Mid → warm green/yellow, High → cool blue
        self.band_colors = np.array([
            [1.0, 0.2, 0.0],    # bass: red with a touch of warmth
            [0.3, 1.0, 0.1],    # mid: green
            [0.0, 0.3, 1.0],    # high: blue with slight cyan
        ], dtype=np.float32)

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Band Proportional"

    @property
    def description(self):
        return "Three independent abs-integral signals mapped to RGB (bass=red, mid=green, high=blue); color shows which frequency bands are changing."

    def process_audio(self, mono_chunk: np.ndarray):
        """Accumulate audio into frames."""
        n = len(mono_chunk)
        pos = self.audio_buf_pos

        while n > 0:
            space = self.rms_frame_len - pos
            take = min(n, space)
            self.audio_buf[pos:pos + take] = mono_chunk[:take]
            mono_chunk = mono_chunk[take:]
            pos += take
            n -= take

            if pos >= self.rms_frame_len:
                self._process_frame(self.audio_buf.copy())
                self.audio_buf[:self.rms_frame_len - self.rms_hop] = \
                    self.audio_buf[self.rms_hop:]
                pos = self.rms_frame_len - self.rms_hop

        self.audio_buf_pos = pos

    def _process_frame(self, frame):
        """Compute per-band abs-integrals and update color target."""
        dt = self.rms_frame_len / self.sample_rate
        buf_idx = self.deriv_buf_pos % self.window_frames

        normalized = np.zeros(self.n_bands, dtype=np.float32)

        for i in range(self.n_bands):
            # Bandpass filter
            filtered, self.filter_states[i] = sosfilt(
                self.filters[i], frame, zi=self.filter_states[i])

            # RMS of filtered band
            rms = np.sqrt(np.mean(filtered ** 2))

            # Derivative
            rms_deriv = (rms - self.prev_rms[i]) / dt
            self.prev_rms[i] = rms

            # Store |derivative| in ring buffer
            self.deriv_bufs[i][buf_idx] = abs(rms_deriv)

            # Abs-integral
            self.abs_integrals[i] = np.sum(self.deriv_bufs[i]) * dt

            # Slow-decay peak normalization
            self.integral_peaks[i] = max(
                self.abs_integrals[i],
                self.integral_peaks[i] * self.peak_decay)
            normalized[i] = (self.abs_integrals[i] / self.integral_peaks[i]
                             if self.integral_peaks[i] > 0 else 0)

        self.deriv_buf_pos += 1

        # Map normalized band values to RGB via color matrix
        # Each band contributes its color weighted by its normalized intensity
        rgb = np.zeros(3, dtype=np.float32)
        for i in range(self.n_bands):
            rgb += self.band_colors[i] * normalized[i]

        # Normalize so max channel is the strongest band's value
        rgb_max = np.max(rgb)
        if rgb_max > 0:
            rgb = rgb / rgb_max * np.max(normalized)

        with self._lock:
            self.target_rgb = rgb

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            target = self.target_rgb.copy()

        # Per-channel asymmetric smoothing
        for c in range(3):
            if target[c] > self.display_rgb[c]:
                self.display_rgb[c] += (target[c] - self.display_rgb[c]) * self.attack_rate
            else:
                self.display_rgb[c] *= self.decay_rate ** (dt * 30)

        # Gamma correction
        display = self.display_rgb ** 0.6

        # Scale to 0-255
        pixel = (display * 255).clip(0, 255).astype(np.uint8)
        frame = np.tile(pixel, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        n = self.abs_integrals / (self.integral_peaks + 1e-10)
        r, g, b = self.display_rgb
        return {
            'bass': f'{n[0]:.2f}',
            'mid': f'{n[1]:.2f}',
            'high': f'{n[2]:.2f}',
            'rgb': f'{r:.2f}/{g:.2f}/{b:.2f}',
        }
