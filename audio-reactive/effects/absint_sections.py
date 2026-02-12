"""
AbsInt Sections — Fibonacci-sized sections with orange→purple gradient.

Sections are Fibonacci-sized (1, 2, 3, 5, 8, 13, 21, 34, 55, ...)
starting from the END of the strip. The smallest section (1 LED) is
at the tip, growing toward the start. The largest section absorbs
any remaining LEDs.

Color gradients from orange (smallest/tip) through red and magenta
to purple (largest/base).

Each LED has a group index in self.led_groups for future per-section
algorithm routing.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


def fibonacci_sections(total_leds):
    """Generate Fibonacci-sized sections that fit in total_leds.
    Returns list of (start, end, section_index) from start of strip."""
    # Generate Fibonacci sequence: 1, 2, 3, 5, 8, 13, 21, 34, 55, ...
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= total_leds:
        fibs.append(fibs[-1] + fibs[-2])

    # Build sections from the END of the strip (smallest at tip)
    sections = []
    pos = total_leds  # start from end
    for i, size in enumerate(fibs):
        if pos <= 0:
            break
        start = max(0, pos - size)
        sections.append((start, pos, i))
        pos = start

    # If LEDs remain at the start, extend the last (largest) section
    if pos > 0 and sections:
        last_start, last_end, last_idx = sections[-1]
        sections[-1] = (0, last_end, last_idx)

    return sections


class AbsIntSectionsEffect(AudioReactiveEffect):
    """Fibonacci sections, orange→purple gradient, abs-integral brightness."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # RMS computation
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0
        self.prev_rms = 0.0

        # Abs-integral ring buffer
        self.window_sec = 0.15
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        # Signal state
        self.abs_integral = 0.0
        self.integral_peak = 1e-10
        self.peak_decay = 0.998

        # Visual state
        self.target_brightness = 0.0
        self.brightness = 0.0
        self.attack_rate = 0.6
        self.decay_rate = 0.85
        self.max_brightness = 0.80

        # Color gradient endpoints: orange (tip/small) → purple (base/large)
        self.color_start = np.array([220, 80, 0], dtype=np.float32)    # orange
        self.color_end = np.array([120, 10, 200], dtype=np.float32)    # purple
        # Midpoint waypoints for richer gradient
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
            # Map section index to palette position (0 = largest/base, max = smallest/tip)
            t = idx / max(self.n_groups - 1, 1)  # 0 at base, 1 at tip
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
        return "AbsInt Sections"

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
                self._process_rms_frame(self.audio_buf.copy())
                self.audio_buf[:self.rms_frame_len - self.rms_hop] = \
                    self.audio_buf[self.rms_hop:]
                pos = self.rms_frame_len - self.rms_hop
        self.audio_buf_pos = pos

    def _process_rms_frame(self, frame):
        rms = np.sqrt(np.mean(frame ** 2))
        dt = self.rms_frame_len / self.sample_rate
        rms_deriv = (rms - self.prev_rms) / dt
        self.prev_rms = rms

        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1
        self.abs_integral = np.sum(self.deriv_buf) * dt

        self.integral_peak = max(self.abs_integral, self.integral_peak * self.peak_decay)
        normalized = self.abs_integral / self.integral_peak if self.integral_peak > 0 else 0

        with self._lock:
            self.target_brightness = normalized

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
        return {
            'brightness': f'{self.brightness:.2f}',
            'integral': f'{self.abs_integral:.3f}',
        }
