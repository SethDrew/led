"""
Full-spectrum onset strength detector as an AudioReactiveEffect.

Algorithm:
  FFT (2048-sample window) -> 40 mel bands (20-8000 Hz) ->
  log compression -> half-wave rectified spectral flux ->
  adaptive threshold (mean + 2.0x std over 3s history) ->
  beat flag with 0.1s cooldown.

Render: blue pulse with exponential decay (alpha=0.12, gamma=2.2).

See also: analysis/algorithms/beat_detector_validation.md
"""

import time
import threading
import numpy as np

from base import AudioReactiveEffect

# Detection parameters
N_FFT = 2048
CHUNK_SIZE = 1024
N_MELS = 40
ONSET_FMIN = 20
ONSET_FMAX = 8000
ONSET_HISTORY_SEC = 3.0
ONSET_MIN_INTERVAL_SEC = 0.1
ONSET_THRESHOLD_MULT = 2.0

# Visual parameters
DECAY_ALPHA = 0.12
GAMMA = 2.2
LED_FPS = 30

ONSET_COLOR = np.array([0, 0, 255], dtype=np.float64)


def _mel_frequencies(n_mels, fmin, fmax):
    """Generate mel-spaced frequency bins."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    return mel_to_hz(mels)


def _create_mel_filterbank(n_fft, n_mels, fmin, fmax, sample_rate):
    """Create triangular mel filterbank."""
    mel_freqs = _mel_frequencies(n_mels, fmin, fmax)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    mel_bins = np.floor((n_fft + 1) * mel_freqs / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        left = mel_bins[i]
        center = mel_bins[i + 1]
        right = mel_bins[i + 2]

        if center > left:
            filterbank[i, left:center] = np.linspace(0, 1, center - left)
        if right > center:
            filterbank[i, center:right] = np.linspace(1, 0, right - center)

    return filterbank


class OnsetStrengthDetector(AudioReactiveEffect):
    """Full-spectrum onset strength detection with blue pulse/decay."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Build mel filterbank
        self.mel_fb = _create_mel_filterbank(N_FFT, N_MELS, ONSET_FMIN, ONSET_FMAX, sample_rate)

        # State (protected by lock for audio thread safety)
        self._lock = threading.Lock()
        self.prev_mel_spectrum = None
        self.onset_history = []
        self.max_history = int(ONSET_HISTORY_SEC * sample_rate / CHUNK_SIZE)
        self.last_beat_time = 0
        self.beat_count = 0
        self.brightness = 0.0

        # Windowing + audio buffer
        self.window = np.hanning(N_FFT)
        self.audio_buffer = np.zeros(N_FFT)

    @property
    def name(self) -> str:
        return "Onset Strength Detector"

    def process_audio(self, mono_chunk: np.ndarray):
        """FFT -> mel bands -> log compression -> spectral flux -> threshold."""
        chunk_len = len(mono_chunk)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = mono_chunk

        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Apply mel filterbank + log compression
        mel_spectrum = np.dot(self.mel_fb, spectrum)
        mel_spectrum = np.log1p(mel_spectrum)

        if self.prev_mel_spectrum is None:
            self.prev_mel_spectrum = mel_spectrum
            return

        # Onset strength: sum of positive spectral differences
        diff = mel_spectrum - self.prev_mel_spectrum
        onset_strength = np.sum(np.maximum(diff, 0))
        self.prev_mel_spectrum = mel_spectrum

        # Adaptive threshold
        self.onset_history.append(onset_strength)
        if len(self.onset_history) > self.max_history:
            self.onset_history.pop(0)

        if len(self.onset_history) < 10:
            return

        mean_onset = np.mean(self.onset_history)
        std_onset = np.std(self.onset_history)
        threshold = mean_onset + ONSET_THRESHOLD_MULT * std_onset

        now = time.time()
        is_beat = (onset_strength > threshold and
                   (now - self.last_beat_time) > ONSET_MIN_INTERVAL_SEC)

        if is_beat:
            self.last_beat_time = now
            self.beat_count += 1
            with self._lock:
                self.brightness = max(self.brightness, 1.0)

    def render(self, dt: float) -> np.ndarray:
        """Blue pulse with exponential decay."""
        with self._lock:
            decay = DECAY_ALPHA ** (dt * LED_FPS)
            self.brightness *= (1.0 - decay)
            self.brightness = max(self.brightness, 0.0)
            b = self.brightness

        gamma_b = b ** (1.0 / GAMMA)

        color = (ONSET_COLOR * gamma_b).astype(np.uint8)
        frame = np.tile(color, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.beat_count,
            'brightness': round(self.brightness, 2),
        }
