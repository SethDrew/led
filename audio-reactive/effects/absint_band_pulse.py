"""
AbsInt Band Pulse — multi-band Gaussian glow pulses.

Each frequency band (bass/mid/high) runs its own beat detector using
RMS derivative → abs-integral (same proven approach as absint_snake).
Detected beats spawn smooth Gaussian-shaped pulses that travel down
the strip, spreading and fading as they go.

Band colors:
  Bass:  warm red/orange  (255, 80, 10)
  Mid:   cool blue/cyan   (20, 100, 255)
  High:  white/gold       (255, 220, 150)

Pulses blend additively when overlapping.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


# Band definitions: (name, color_rgb, freq_lo_hz, freq_hi_hz)
BANDS = [
    ('bass', np.array([255, 80, 10], dtype=np.float32), 20, 250),
    ('mid',  np.array([20, 100, 255], dtype=np.float32), 250, 2000),
    ('high', np.array([255, 220, 150], dtype=np.float32), 2000, 10000),
]


class BandState:
    """Per-band beat detection state."""

    def __init__(self, window_frames):
        self.prev_rms = 0.0
        self.deriv_buf = np.zeros(window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0
        self.abs_integral = 0.0
        self.integral_peak = 1e-10
        self.last_beat_time = -1.0
        self.beat_count = 0


class AbsIntBandPulseEffect(AudioReactiveEffect):
    """Multi-band Gaussian glow pulses triggered by per-band beat detection."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # ── FFT / audio buffering ──
        self.fft_len = 2048
        self.hop = 512
        self.audio_buf = np.zeros(self.fft_len, dtype=np.float32)
        self.audio_buf_pos = 0
        self.fft_dt = self.hop / sample_rate

        # Hann window for FFT
        self.window = np.hanning(self.fft_len).astype(np.float32)

        # Precompute bin ranges for each band
        bin_hz = sample_rate / self.fft_len
        self.band_slices = []
        for _, _, lo, hi in BANDS:
            lo_bin = max(1, int(lo / bin_hz))
            hi_bin = min(self.fft_len // 2, int(hi / bin_hz) + 1)
            self.band_slices.append((lo_bin, hi_bin))

        # ── Per-band beat detection ──
        self.window_sec = 0.15
        self.window_frames = max(1, int(self.window_sec / (self.fft_len / sample_rate)))

        self.bands = [BandState(self.window_frames) for _ in BANDS]
        self.peak_decay = 0.997
        self.threshold = 0.30
        self.cooldown = 0.20
        self.time_acc = 0.0

        # ── Pulse parameters ──
        self.speed = 0.3            # strip-lengths per second
        self.initial_sigma = 3.0    # LEDs
        self.max_sigma = 8.0        # LEDs at end of travel
        self.max_pulses = 15
        self.pulses = []            # list of pulse dicts

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Bands"

    @property
    def description(self):
        return "Multi-band beat detection (bass/mid/high) spawning Gaussian pulses that travel across the strip; warm red, blue, and white."

    def _spawn_pulse(self, band_idx, strength):
        """Spawn a Gaussian pulse for the given band."""
        _, color, _, _ = BANDS[band_idx]
        pulse = {
            'band': band_idx,
            'pos': 0.0,
            'strength': np.clip(strength, 0, 1),
            'sigma': self.initial_sigma,
            'color': color,
        }
        with self._lock:
            self.pulses.append(pulse)
            if len(self.pulses) > self.max_pulses:
                self.pulses.pop(0)

    def process_audio(self, mono_chunk: np.ndarray):
        """Buffer audio and process overlapping FFT frames."""
        n = len(mono_chunk)
        pos = self.audio_buf_pos
        while n > 0:
            space = self.fft_len - pos
            take = min(n, space)
            self.audio_buf[pos:pos + take] = mono_chunk[:take]
            mono_chunk = mono_chunk[take:]
            pos += take
            n -= take
            if pos >= self.fft_len:
                self._process_fft_frame(self.audio_buf.copy())
                # Slide buffer by hop
                self.audio_buf[:self.fft_len - self.hop] = \
                    self.audio_buf[self.hop:]
                pos = self.fft_len - self.hop
        self.audio_buf_pos = pos

    def _process_fft_frame(self, frame):
        """Run FFT, compute per-band RMS derivative, detect beats."""
        windowed = frame * self.window
        spectrum = np.abs(np.fft.rfft(windowed))
        dt = self.fft_len / self.sample_rate
        self.time_acc += self.fft_dt

        for i, (band, (lo_bin, hi_bin)) in enumerate(zip(self.bands, self.band_slices)):
            # Band RMS from FFT bins
            band_bins = spectrum[lo_bin:hi_bin]
            band_rms = np.sqrt(np.mean(band_bins ** 2)) if len(band_bins) > 0 else 0.0

            # Derivative
            rms_deriv = (band_rms - band.prev_rms) / dt
            band.prev_rms = band_rms

            # Abs-integral over window
            band.deriv_buf[band.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
            band.deriv_buf_pos += 1
            band.abs_integral = np.sum(band.deriv_buf) * dt

            # Slow-decay peak normalization
            band.integral_peak = max(band.abs_integral, band.integral_peak * self.peak_decay)
            normalized = band.abs_integral / band.integral_peak if band.integral_peak > 0 else 0

            # Beat detection with cooldown
            time_since_beat = self.time_acc - band.last_beat_time
            if normalized > self.threshold and time_since_beat > self.cooldown:
                band.last_beat_time = self.time_acc
                band.beat_count += 1
                self._spawn_pulse(i, normalized)

    def render(self, dt: float) -> np.ndarray:
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)
        step = self.speed * self.num_leds * dt
        led_indices = np.arange(self.num_leds, dtype=np.float32)

        with self._lock:
            alive = []
            for pulse in self.pulses:
                pulse['pos'] += step

                # Progress through strip (0 to 1)
                progress = pulse['pos'] / max(self.num_leds - 1, 1)

                # Dead once center is well past the end
                if pulse['pos'] - 3 * pulse['sigma'] > self.num_leds:
                    continue
                alive.append(pulse)

                # Sigma grows from initial to max over travel
                t = min(progress, 1.0)
                pulse['sigma'] = self.initial_sigma + t * (self.max_sigma - self.initial_sigma)

                # Fade out in last 30%
                if progress > 0.7:
                    fade = max(0.0, 1.0 - (progress - 0.7) / 0.3)
                else:
                    fade = 1.0

                # Gaussian shape: brightness per LED
                sigma = pulse['sigma']
                gauss = np.exp(-0.5 * ((led_indices - pulse['pos']) / sigma) ** 2)
                brightness = pulse['strength'] * fade * gauss

                # Additive blend
                frame += np.outer(brightness, pulse['color'])

            self.pulses = alive

        return frame.clip(0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        return {
            'bass': self.bands[0].beat_count,
            'mid': self.bands[1].beat_count,
            'high': self.bands[2].beat_count,
            'pulses': len(self.pulses),
        }
