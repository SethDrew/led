"""
Bass-band spectral flux beat detector as an AudioReactiveEffect.

Algorithm:
  FFT (2048-sample window) -> extract bass bins (20-250 Hz) ->
  half-wave rectified spectral flux -> adaptive threshold
  (mean + 1.5x std over 3s history) -> beat flag with 0.3s cooldown.

Render: red pulse with exponential decay (alpha=0.12, gamma=2.2).

See also: analysis/algorithms/beat_detector_validation.md
"""

import time
import threading
import numpy as np

from base import AudioReactiveEffect

# Detection parameters
BASS_LOW_HZ = 20
BASS_HIGH_HZ = 250
N_FFT = 2048
CHUNK_SIZE = 1024
FLUX_HISTORY_SEC = 3.0
MIN_BEAT_INTERVAL_SEC = 0.3
THRESHOLD_MULTIPLIER = 1.5

# Visual parameters
DECAY_ALPHA = 0.12
GAMMA = 2.2
LED_FPS = 30

BEAT_COLOR = np.array([255, 0, 0], dtype=np.float64)


class BassFluxBeatDetector(AudioReactiveEffect):
    """Bass-band spectral flux beat detection with red pulse/decay."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Frequency bin indices for bass range
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
        self.bass_bins = np.where((freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ))[0]

        # State (protected by lock for audio thread safety)
        self._lock = threading.Lock()
        self.prev_spectrum = None
        self.flux_history = []
        self.max_history = int(FLUX_HISTORY_SEC * sample_rate / CHUNK_SIZE)
        self.last_beat_time = 0
        self.beat_count = 0
        self.brightness = 0.0

        # Windowing + audio buffer
        self.window = np.hanning(N_FFT)
        self.audio_buffer = np.zeros(N_FFT)

    @property
    def name(self) -> str:
        return "Bass Flux Beat Detector"

    def process_audio(self, mono_chunk: np.ndarray):
        """FFT -> bass bins -> spectral flux -> adaptive threshold -> beat flag."""
        chunk_len = len(mono_chunk)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = mono_chunk

        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))
        bass_spectrum = spectrum[self.bass_bins]

        if self.prev_spectrum is None:
            self.prev_spectrum = bass_spectrum
            return

        # Half-wave rectified spectral flux
        diff = bass_spectrum - self.prev_spectrum
        flux = np.sum(np.maximum(diff, 0))
        self.prev_spectrum = bass_spectrum

        # Adaptive threshold
        self.flux_history.append(flux)
        if len(self.flux_history) > self.max_history:
            self.flux_history.pop(0)

        if len(self.flux_history) < 10:
            return

        mean_flux = np.mean(self.flux_history)
        std_flux = np.std(self.flux_history)
        threshold = mean_flux + THRESHOLD_MULTIPLIER * std_flux

        now = time.time()
        is_beat = (flux > threshold and
                   (now - self.last_beat_time) > MIN_BEAT_INTERVAL_SEC)

        if is_beat:
            self.last_beat_time = now
            self.beat_count += 1
            with self._lock:
                self.brightness = max(self.brightness, 1.0)

    def render(self, dt: float) -> np.ndarray:
        """Red pulse with exponential decay."""
        with self._lock:
            decay = DECAY_ALPHA ** (dt * LED_FPS)
            self.brightness *= (1.0 - decay)
            self.brightness = max(self.brightness, 0.0)
            b = self.brightness

        # Gamma-corrected brightness
        gamma_b = b ** (1.0 / GAMMA)

        color = (BEAT_COLOR * gamma_b).astype(np.uint8)
        frame = np.tile(color, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.beat_count,
            'brightness': round(self.brightness, 2),
        }
