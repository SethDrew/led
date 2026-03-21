"""
150 Combined Spectro Fall — split-strip: spectro chroma + energy waterfall.

Two halves on a 150-LED strip:
  LEDs 0-74  (Spectro Chroma): Mel spectrogram with RMS-driven brightness
             and per-bin gray→red chroma mapping.
  LEDs 75-149 (Energy Waterfall): Scrolling RMS energy pulses — raw waveform
             amplitude creates bright/dim bands traveling down the strip.

Color: gray → deep red → bright red (no blues/purples).
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


class CombinedSpectroFallEffect(AudioReactiveEffect):
    """Split-strip: spectro chroma + energy waterfall."""

    registry_name = '150_combined_spectro_fall'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = '64-bin mel spectrogram'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Strip geometry
        self.n_spec = min(75, num_leds)
        self.n_waterfall = max(0, num_leds - self.n_spec)

        # Mel filterbank parameters
        self.n_mels = 64
        self.n_fft = 2048
        self.hop_length = 512

        # Pre-compute mel filterbank matrix (n_mels, n_fft//2 + 1)
        self.mel_fb = self._mel_filterbank(
            sr=sample_rate, n_fft=self.n_fft,
            n_mels=self.n_mels, fmin=20.0, fmax=8000.0,
        )

        # Streaming FFT accumulator
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self.window = np.hanning(self.n_fft).astype(np.float32)

        # Pre-compute mel-to-LED interpolation for spectrogram half
        # Map 75 LEDs linearly across 64 mel bins
        fractional = np.linspace(0, self.n_mels - 1, self.n_spec)
        self._spec_mel_idx = np.clip(fractional.astype(np.int32), 0, self.n_mels - 2)
        self._spec_mel_weight = (fractional - self._spec_mel_idx).astype(np.float32)

        # ── Shared state (audio thread → render thread) ──────────────────
        self._mel_frame = np.zeros(self.n_mels, dtype=np.float32)
        self._rms = np.float32(0.0)
        self._lock = threading.Lock()

        # ── Peak-decay normalization for spectrogram (audio thread) ──────
        self._mel_peaks = np.full(self.n_mels, 1e-10, dtype=np.float32)
        self._peak_decay = 0.9995
        self._rms_peak = np.float32(1e-10)

        # ── Temporal smoothing for spectrogram (render thread) ───────────
        self._smoothed_chroma = np.zeros(self.n_spec, dtype=np.float32)
        self._smoothed_bright = np.float32(0.0)
        self._smooth_alpha = 0.3

        # ── Waterfall buffer ─────────────────────────────────────────────
        self._wf_buffer = np.zeros((self.n_waterfall, 3), dtype=np.uint8)

    @property
    def name(self):
        return "150 Combined Spectro Fall"

    @property
    def description(self):
        return "Split-strip: spectro chroma (0-74) + energy waterfall (75-149)."

    # ── Audio processing (audio thread) ───────────────────────────────────

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame):
        # RMS of the time-domain frame (total amplitude)
        rms = np.float32(np.sqrt(np.mean(frame ** 2)))

        # Peak-decay normalization for RMS
        self._rms_peak = max(rms, self._rms_peak * self._peak_decay)
        rms_norm = rms / self._rms_peak if self._rms_peak > 1e-10 else 0.0

        # FFT magnitude spectrum
        spec = np.abs(np.fft.rfft(frame * self.window))

        # Apply mel filterbank → (n_mels,) energy per bin
        mel_energy = self.mel_fb @ (spec ** 2)

        # Convert to dB scale for per-bin chroma mapping (floor at -80 dB)
        mel_db = 10.0 * np.log10(np.maximum(mel_energy, 1e-10))
        mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
        mel_db -= mel_db.min()

        # Peak-decay normalization per bin → [0, 1] chroma control
        self._mel_peaks = np.maximum(mel_db, self._mel_peaks * self._peak_decay)
        normalized = np.where(
            self._mel_peaks > 1e-10,
            mel_db / self._mel_peaks,
            0.0,
        ).astype(np.float32)

        with self._lock:
            self._mel_frame = normalized
            self._rms = np.float32(rms_norm)

    # ── Render (main loop thread) ─────────────────────────────────────────

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            mel = self._mel_frame.copy()
            rms = float(self._rms)

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)

        # ── Left half: spectrogram (LEDs 0-74) ────────────────────────
        # Brightness = total amplitude (RMS), same for all LEDs
        # Color = gray → full red, scaled by per-bin frequency energy

        # Interpolate per-bin energy (chroma control) for each LED
        lower = mel[self._spec_mel_idx]
        upper = mel[np.minimum(self._spec_mel_idx + 1, self.n_mels - 1)]
        led_chroma = lower * (1.0 - self._spec_mel_weight) + upper * self._spec_mel_weight

        # Temporal smoothing on both axes
        self._smoothed_chroma += self._smooth_alpha * (led_chroma - self._smoothed_chroma)
        self._smoothed_bright += self._smooth_alpha * (rms - self._smoothed_bright)

        brightness = np.clip(self._smoothed_bright, 0.0, 1.0)
        chroma = np.clip(self._smoothed_chroma, 0.0, 1.0)  # (n_spec,)

        # Full red = [255, 40, 0] (from our red LUT top end)
        # Gray = [v, v, v] where v matches red luminance
        # Blend: gray*(1-chroma) + red*chroma, then scale by brightness
        full_red = np.array([255.0, 40.0, 0.0])
        gray_value = 90.0  # neutral gray that matches red perceived brightness
        gray = np.array([gray_value, gray_value, gray_value])

        # Per-LED color: lerp gray → red by chroma
        colors = gray[np.newaxis, :] * (1.0 - chroma[:, np.newaxis]) + \
                 full_red[np.newaxis, :] * chroma[:, np.newaxis]

        # Apply global brightness
        spec_rgb = (colors * brightness).astype(np.uint8)
        frame[:self.n_spec] = spec_rgb

        # ── Right half: energy pulse waterfall (LEDs 75-149) ─────────
        # Each frame pushes current RMS as brightness into LED 75.
        # The buffer scrolls naturally — short bursts = narrow bright
        # pulses traveling down, sustained energy = wide bright bands.
        # No smoothing on the waterfall input — raw RMS for crisp edges.

        self._wf_buffer[1:] = self._wf_buffer[:-1]

        # Raw RMS → brightness, full red color
        wf_bright = np.clip(rms, 0.0, 1.0)
        full_red = np.array([255.0, 40.0, 0.0])
        new_pixel = (full_red * wf_bright).astype(np.uint8)
        self._wf_buffer[0] = new_pixel

        frame[self.n_spec:self.n_spec + self.n_waterfall] = self._wf_buffer

        return frame

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        with self._lock:
            mel = self._mel_frame.copy()

        # Dominant third
        third = self.n_mels // 3
        band_energy = np.array([
            np.mean(mel[:third]),
            np.mean(mel[third:2 * third]),
            np.mean(mel[2 * third:]),
        ])
        band_names = ['lows', 'mids', 'highs']
        peak_band = band_names[int(np.argmax(band_energy))]

        active = int(np.sum(self._smoothed_chroma > 0.08))

        return {
            'peak_band': peak_band,
            'rms': f'{self._smoothed_bright:.2f}',
            'active': active,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        """Compute a mel filterbank matrix (n_mels, n_fft//2 + 1)."""
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        n_freqs = n_fft // 2 + 1
        fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            lo, center, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            up = (fft_freqs - lo) / (center - lo + 1e-10)
            down = (hi - fft_freqs) / (hi - center + 1e-10)
            fb[i] = np.maximum(0, np.minimum(up, down))

        return fb
