"""
Spectro Chroma — mel spectrogram with RMS brightness and per-bin chroma.

Global brightness tracks total waveform amplitude (RMS). Per-LED color
goes from gray (no energy at that frequency) to full red (high energy).
64 mel bins (20-8000 Hz) interpolated across all LEDs.

Color: gray → deep red → bright red (no blues/purples).
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


class SpectroChromaEffect(AudioReactiveEffect):
    """Mel spectrogram: RMS brightness, per-bin gray→red chroma."""

    registry_name = 'spectro_chroma'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = '64-bin mel spectrogram + RMS'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_mels = 64
        self.n_fft = 2048
        self.hop_length = 512

        self.mel_fb = self._mel_filterbank(
            sr=sample_rate, n_fft=self.n_fft,
            n_mels=self.n_mels, fmin=20.0, fmax=8000.0,
        )

        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self.window = np.hanning(self.n_fft).astype(np.float32)

        # Mel-to-LED interpolation
        fractional = np.linspace(0, self.n_mels - 1, num_leds)
        self._mel_idx = np.clip(fractional.astype(np.int32), 0, self.n_mels - 2)
        self._mel_weight = (fractional - self._mel_idx).astype(np.float32)

        # Shared state
        self._mel_frame = np.zeros(self.n_mels, dtype=np.float32)
        self._rms = np.float32(0.0)
        self._lock = threading.Lock()

        # Peak-decay normalization
        self._mel_peaks = np.full(self.n_mels, 1e-10, dtype=np.float32)
        self._peak_decay = 0.9995
        self._rms_peak = np.float32(1e-10)

        # Temporal smoothing (render thread)
        self._smoothed_chroma = np.zeros(num_leds, dtype=np.float32)
        self._smoothed_bright = np.float32(0.0)
        self._smooth_alpha = 0.3

    @property
    def name(self):
        return "Spectro Chroma"

    @property
    def description(self):
        return "Mel spectrogram: RMS brightness, per-bin gray-to-red chroma."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame):
        rms = np.float32(np.sqrt(np.mean(frame ** 2)))
        self._rms_peak = max(rms, self._rms_peak * self._peak_decay)
        rms_norm = rms / self._rms_peak if self._rms_peak > 1e-10 else 0.0

        spec = np.abs(np.fft.rfft(frame * self.window))
        mel_energy = self.mel_fb @ (spec ** 2)

        mel_db = 10.0 * np.log10(np.maximum(mel_energy, 1e-10))
        mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
        mel_db -= mel_db.min()

        self._mel_peaks = np.maximum(mel_db, self._mel_peaks * self._peak_decay)
        normalized = np.where(
            self._mel_peaks > 1e-10,
            mel_db / self._mel_peaks,
            0.0,
        ).astype(np.float32)

        with self._lock:
            self._mel_frame = normalized
            self._rms = np.float32(rms_norm)

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            mel = self._mel_frame.copy()
            rms = float(self._rms)

        # Interpolate per-bin chroma
        lower = mel[self._mel_idx]
        upper = mel[np.minimum(self._mel_idx + 1, self.n_mels - 1)]
        led_chroma = lower * (1.0 - self._mel_weight) + upper * self._mel_weight

        self._smoothed_chroma += self._smooth_alpha * (led_chroma - self._smoothed_chroma)
        self._smoothed_bright += self._smooth_alpha * (rms - self._smoothed_bright)

        brightness = np.clip(self._smoothed_bright, 0.0, 1.0)
        chroma = np.clip(self._smoothed_chroma, 0.0, 1.0)

        full_red = np.array([255.0, 40.0, 0.0])
        gray = np.array([90.0, 90.0, 90.0])

        colors = gray[np.newaxis, :] * (1.0 - chroma[:, np.newaxis]) + \
                 full_red[np.newaxis, :] * chroma[:, np.newaxis]

        return (colors * brightness).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            mel = self._mel_frame.copy()

        third = self.n_mels // 3
        band_energy = np.array([
            np.mean(mel[:third]),
            np.mean(mel[third:2 * third]),
            np.mean(mel[2 * third:]),
        ])
        peak_band = ['lows', 'mids', 'highs'][int(np.argmax(band_energy))]
        active = int(np.sum(self._smoothed_chroma > 0.08))

        return {
            'peak_band': peak_band,
            'rms': f'{self._smoothed_bright:.2f}',
            'active': active,
        }

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
