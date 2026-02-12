"""
LongInt Sections — smooth long-horizon brightness with bass-reactive sparkle.

Fibonacci-sized sections with orange→purple gradient (same as absint_sections).

Brightness is a blend of two signals:
  80% — long integral: rolling RMS average over ~10 seconds. This is the
         overall energy envelope. Changes slowly, follows the song's arc.
  20% — bass abs-integral: abs-integral of bass-band RMS derivative over
         ~150ms. This catches kick drums and bass transients as quick flashes.

The result is a smooth, breathing base that reacts to the song's energy level,
with bass hits adding a sharp 20% brightness kick on top.
"""

import numpy as np
import threading
from scipy.signal import butter, sosfilt
from base import AudioReactiveEffect


def fibonacci_sections(total_leds):
    """Generate Fibonacci-sized sections that fit in total_leds.
    Returns list of (start, end, section_index) from start of strip."""
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= total_leds:
        fibs.append(fibs[-1] + fibs[-2])

    sections = []
    pos = total_leds
    for i, size in enumerate(fibs):
        if pos <= 0:
            break
        start = max(0, pos - size)
        sections.append((start, pos, i))
        pos = start

    if pos > 0 and sections:
        last_start, last_end, last_idx = sections[-1]
        sections[-1] = (0, last_end, last_idx)

    return sections


class LongIntSectionsEffect(AudioReactiveEffect):
    """Smooth long-horizon brightness + bass-reactive top layer, Fibonacci sections."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # RMS computation
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0

        # ── Long integral: rolling RMS over ~10 seconds ──
        self.long_window_sec = 10.0
        dt_per_frame = self.rms_hop / sample_rate  # time per RMS frame
        self.long_window_frames = max(1, int(self.long_window_sec / dt_per_frame))
        self.rms_ring = np.zeros(self.long_window_frames, dtype=np.float32)
        self.rms_ring_pos = 0
        self.long_rms = 0.0
        self.long_rms_peak = 1e-10
        self.long_peak_decay = 0.9999  # very slow decay for 10s window

        # ── Bass abs-integral: short reactive signal ──
        # Bass bandpass filter (20-250 Hz)
        nyq = sample_rate / 2
        low_n = max(20 / nyq, 0.001)
        high_n = min(250 / nyq, 0.999)
        self.bass_sos = butter(4, [low_n, high_n], btype='band', output='sos')
        self.bass_filter_state = np.zeros((self.bass_sos.shape[0], 2))

        self.prev_bass_rms = 0.0
        self.bass_window_sec = 0.15
        self.bass_window_frames = max(1, int(self.bass_window_sec / dt_per_frame))
        self.bass_deriv_buf = np.zeros(self.bass_window_frames, dtype=np.float32)
        self.bass_deriv_pos = 0
        self.bass_abs_integral = 0.0
        self.bass_integral_peak = 1e-10
        self.bass_peak_decay = 0.998

        # ── Blend ratio ──
        self.long_weight = 0.80
        self.bass_weight = 0.20

        # ── Visual state ──
        self.target_brightness = 0.0
        self.brightness = 0.0
        self.attack_rate = 0.5
        self.decay_rate = 0.88
        self.max_brightness = 0.80

        # ── Color: orange (tip) → purple (base), same palette as absint_sections ──
        self.palette = np.array([
            [220, 80,  0],     # orange (tip, smallest)
            [200, 30,  0],     # red-orange
            [180, 10,  20],    # deep red
            [170, 0,   80],    # red-magenta
            [150, 0,  140],    # magenta
            [120, 10, 200],    # purple (base, largest)
        ], dtype=np.float32)

        # Build Fibonacci sections and assign colors
        self.sections = fibonacci_sections(num_leds)
        self.n_groups = len(self.sections)
        self.led_groups = np.zeros(num_leds, dtype=np.int32)
        self.led_colors = np.zeros((num_leds, 3), dtype=np.float32)

        for start, end, idx in self.sections:
            t = idx / max(self.n_groups - 1, 1)
            color = self._sample_palette(t)
            self.led_groups[start:end] = idx
            self.led_colors[start:end] = color

        self._lock = threading.Lock()

    def _sample_palette(self, t):
        """Interpolate through the palette at position t (0=orange/tip, 1=purple/base)."""
        t = np.clip(t, 0, 1)
        n = len(self.palette) - 1
        idx = t * n
        lo = int(idx)
        hi = min(lo + 1, n)
        frac = idx - lo
        return self.palette[lo] * (1 - frac) + self.palette[hi] * frac

    @property
    def name(self):
        return "LongInt Sections"

    def process_audio(self, mono_chunk: np.ndarray):
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
        dt = self.rms_hop / self.sample_rate

        # ── Long integral: rolling RMS average ──
        rms = np.sqrt(np.mean(frame ** 2))
        self.rms_ring[self.rms_ring_pos % self.long_window_frames] = rms
        self.rms_ring_pos += 1
        # Average RMS over the filled portion of the ring
        filled = min(self.rms_ring_pos, self.long_window_frames)
        self.long_rms = np.mean(self.rms_ring[:filled])

        # Normalize against slow-decay peak
        self.long_rms_peak = max(self.long_rms, self.long_rms_peak * self.long_peak_decay)
        long_normalized = self.long_rms / self.long_rms_peak if self.long_rms_peak > 0 else 0

        # ── Bass abs-integral: short reactive signal ──
        filtered, self.bass_filter_state = sosfilt(
            self.bass_sos, frame, zi=self.bass_filter_state)
        bass_rms = np.sqrt(np.mean(filtered ** 2))
        bass_deriv = (bass_rms - self.prev_bass_rms) / dt
        self.prev_bass_rms = bass_rms

        buf_idx = self.bass_deriv_pos % self.bass_window_frames
        self.bass_deriv_buf[buf_idx] = abs(bass_deriv)
        self.bass_deriv_pos += 1
        self.bass_abs_integral = np.sum(self.bass_deriv_buf) * dt

        self.bass_integral_peak = max(
            self.bass_abs_integral,
            self.bass_integral_peak * self.bass_peak_decay)
        bass_normalized = (self.bass_abs_integral / self.bass_integral_peak
                           if self.bass_integral_peak > 0 else 0)

        # ── Blend ──
        blended = self.long_weight * long_normalized + self.bass_weight * bass_normalized

        with self._lock:
            self.target_brightness = blended
            self._diag_long = long_normalized
            self._diag_bass = bass_normalized

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            target = self.target_brightness

        if target > self.brightness:
            self.brightness += (target - self.brightness) * self.attack_rate
        else:
            self.brightness *= self.decay_rate ** (dt * 30)

        b = min(self.brightness, 1.0) * self.max_brightness
        display_b = b ** 0.6

        frame = (self.led_colors * display_b).clip(0, 255).astype(np.uint8)
        return frame

    def get_diagnostics(self) -> dict:
        long_v = getattr(self, '_diag_long', 0)
        bass_v = getattr(self, '_diag_bass', 0)
        return {
            'brightness': f'{self.brightness:.2f}',
            'long': f'{long_v:.2f}',
            'bass': f'{bass_v:.2f}',
        }
