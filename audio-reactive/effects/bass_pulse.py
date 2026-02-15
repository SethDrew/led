"""
Bass Pulse — simple beat detection via bass-band spectral flux.

Informed by failures of other approaches:
  - Full-spectrum onset is anti-correlated with beats on electronic music (F1=0.435)
  - Raw bass energy fails on continuous sub-bass (F1=0.06, self-normalization kills signal)
  - librosa beat_track doubles tempo on syncopated rock
  - User taps track bass peaks (19ms median), not onsets

Fix: half-wave-rectified spectral flux in the bass band (20-250 Hz).
  - Continuous sub-bass = near-zero flux (spectrum doesn't change)
  - Kick drum = big positive flux spike (new energy arrives)
  - Slow-decay peak normalization preserves absolute dynamics
  - Simple adaptive threshold + cooldown
"""

import numpy as np
import threading
from base import ScalarSignalEffect


class BassPulseEffect(ScalarSignalEffect):
    """Whole-tree pulse on bass transients. Simple and robust."""

    registry_name = 'bass_pulse'
    default_palette = 'amber'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # FFT
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)

        # Bass band: 20-250 Hz (kick drums, bass notes)
        self.bass_mask = (self.freq_bins >= 20) & (self.freq_bins <= 250)

        # Audio accumulation
        self.audio_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.audio_buf_pos = 0

        # Previous bass spectrum (for spectral flux)
        self.prev_bass_spec = None

        # Beat detection state
        self.flux_peak = 1e-10         # slow-decay peak for normalization
        self.peak_decay = 0.997        # ~15s half-life at ~22 frames/sec
        self.threshold = 0.55          # normalized flux threshold (0-1)
        self.cooldown = 0.18           # 180ms min between beats (~333 BPM cap)
        self.last_beat_time = -1.0
        self.time_acc = 0.0
        self.beat_count = 0

        # Visual state
        self.brightness = 0.0
        self.decay_rate = 0.82         # per-frame multiplier at 30 FPS (~150ms decay)

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Bass Pulse"

    @property
    def description(self):
        return "Beat detection via half-wave-rectified spectral flux in bass band (20-250 Hz); kick drums spike, continuous sub-bass stays silent."

    def process_audio(self, mono_chunk: np.ndarray):
        n = len(mono_chunk)
        space = self.n_fft - self.audio_buf_pos
        if n < space:
            self.audio_buf[self.audio_buf_pos:self.audio_buf_pos + n] = mono_chunk
            self.audio_buf_pos += n
            return

        self.audio_buf[self.audio_buf_pos:] = mono_chunk[:space]
        self._process_frame(self.audio_buf.copy())

        leftover = n - space
        self.audio_buf[:leftover] = mono_chunk[space:]
        self.audio_buf_pos = leftover

    def _process_frame(self, frame):
        windowed = frame * self.window
        spec = np.abs(np.fft.rfft(windowed))
        bass_spec = spec[self.bass_mask]

        if self.prev_bass_spec is not None:
            # Half-wave rectified spectral flux: only positive changes
            diff = bass_spec - self.prev_bass_spec
            flux = np.sum(np.maximum(diff, 0))

            # Slow-decay peak normalization
            self.flux_peak = max(flux, self.flux_peak * self.peak_decay)
            normalized = flux / self.flux_peak if self.flux_peak > 0 else 0

            # Beat detection
            self.time_acc += self.n_fft / self.sample_rate
            time_since_beat = self.time_acc - self.last_beat_time

            if normalized > self.threshold and time_since_beat > self.cooldown:
                with self._lock:
                    self.brightness = min(1.0, normalized)
                    self.last_beat_time = self.time_acc
                    self.beat_count += 1

        self.prev_bass_spec = bass_spec.copy()

    def get_intensity(self, dt: float) -> float:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)
        return b

    def get_diagnostics(self) -> dict:
        return {
            'beats': self.beat_count,
            'brightness': f'{self.brightness:.2f}',
        }
