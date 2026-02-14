"""
Tempo Pulse — autocorrelation-estimated BPM drives a free-running pulse,
brightness scaled by current amplitude.

How it works:
  1. Accumulate abs-integral signal over a ~30s rolling window.
  2. Run autocorrelation to find the dominant beat period.
  3. Free-run a pulse oscillator at that period (no late detection needed).
  4. Each pulse's brightness = current RMS, so loud moments flash bright
     and quiet moments flash dim — but the TIMING is always on the grid.

The pulse shape is a raised cosine: smooth fade-on, peak, smooth fade-off.
This avoids the hard-edge snap of threshold-based detection.

Falls back to proportional mode (no pulse timing) until it has enough data
to estimate tempo confidently (~5-10 seconds).
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class TempoPulseEffect(AudioReactiveEffect):
    """Free-running tempo pulse, brightness scaled by current amplitude."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # ── RMS computation ──
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0
        self.prev_rms = 0.0
        self.rms_dt = self.rms_hop / sample_rate
        self.rms_fps = sample_rate / self.rms_hop
        self.time_acc = 0.0

        # ── Current RMS (for amplitude scaling) ──
        self.current_rms = 0.0
        self.rms_peak = 1e-10
        self.rms_peak_decay = 0.9998  # slow decay for amplitude envelope

        # ── Abs-integral (for autocorrelation input) ──
        self.window_sec = 0.15
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0
        self.abs_integral = 0.0

        # ── Autocorrelation tempo estimation (30s window) ──
        self.ac_window_sec = 30.0
        self.ac_window_frames = int(self.ac_window_sec * self.rms_fps)
        self.ac_buf = np.zeros(self.ac_window_frames, dtype=np.float32)
        self.ac_buf_pos = 0
        self.ac_buf_filled = 0

        # Period bounds (40-200 BPM)
        self.min_period_sec = 0.300   # 200 BPM
        self.max_period_sec = 1.500   # 40 BPM
        self.min_period_frames = max(1, int(self.min_period_sec * self.rms_fps))
        self.max_period_frames = int(self.max_period_sec * self.rms_fps)

        # Autocorrelation state
        self.ac_confidence = 0.0
        self.ac_min_confidence = 0.25
        self.estimated_period = 0.0  # seconds, 0 = unknown
        self.ac_update_interval = 1.0  # recompute every 1 second
        self.last_ac_update = 0.0

        # ── Pulse oscillator ──
        self.phase = 0.0  # 0-1 within current beat period
        self.pulse_width = 0.80  # fraction of period for pulse on-time

        # ── Visual state ──
        self.brightness = 0.0
        self.max_brightness = 0.80

        # Color palette: deep red → orange → red → magenta → purple
        self.palette = np.array([
            [40,  5,  0],     # 0.0 — deep dark red
            [160, 50, 0],     # 0.25 — orange
            [200, 20, 0],     # 0.50 — red-orange
            [180, 0,  60],    # 0.75 — red-magenta
            [160, 20, 180],   # 1.0 — purple/pink
        ], dtype=np.float32)

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Tempo Pulse"

    @property
    def description(self):
        return "Free-running pulse oscillator at autocorrelation-estimated tempo; brightness scaled by current RMS; raised-cosine pulse shape."

    def _sample_palette(self, t):
        """Sample color from palette at position t (0-1)."""
        t = np.clip(t, 0, 1)
        n = len(self.palette) - 1
        idx = t * n
        lo = int(idx)
        hi = min(lo + 1, n)
        frac = idx - lo
        return self.palette[lo] * (1 - frac) + self.palette[hi] * frac

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
        dt = self.rms_frame_len / self.sample_rate

        # Current RMS (for amplitude scaling)
        rms = np.sqrt(np.mean(frame ** 2))
        self.rms_peak = max(rms, self.rms_peak * self.rms_peak_decay)
        rms_normalized = rms / self.rms_peak if self.rms_peak > 0 else 0

        # RMS derivative → abs-integral (for autocorrelation input)
        rms_deriv = (rms - self.prev_rms) / dt
        self.prev_rms = rms
        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1
        self.abs_integral = np.sum(self.deriv_buf) * dt

        # Store in long buffer for autocorrelation
        self.ac_buf[self.ac_buf_pos % self.ac_window_frames] = self.abs_integral
        self.ac_buf_pos += 1
        self.ac_buf_filled = min(self.ac_buf_filled + 1, self.ac_window_frames)

        self.time_acc += self.rms_dt

        # Periodically update autocorrelation
        if self.time_acc - self.last_ac_update > self.ac_update_interval:
            self._update_autocorrelation()
            self.last_ac_update = self.time_acc

        # Advance pulse phase
        if self.estimated_period > 0 and self.ac_confidence >= self.ac_min_confidence:
            self.phase += self.rms_dt / self.estimated_period
            if self.phase >= 1.0:
                self.phase -= 1.0

            # Raised cosine pulse shape within pulse_width window
            if self.phase < self.pulse_width:
                # Map [0, pulse_width] → [0, pi] for a half-cosine bump
                t = self.phase / self.pulse_width
                pulse_envelope = 0.5 * (1.0 - np.cos(2 * np.pi * t))
            else:
                pulse_envelope = 0.0

            # Scale by current amplitude
            brightness = pulse_envelope * rms_normalized
        else:
            # No tempo estimate — fall back to proportional
            # (abs-integral normalized, like absint_prop)
            brightness = rms_normalized * 0.5

        with self._lock:
            self.current_rms = rms_normalized
            self.brightness = brightness

    def _update_autocorrelation(self):
        """Estimate beat period from autocorrelation of abs-integral signal."""
        if self.ac_buf_filled < self.min_period_frames * 4:
            return

        n = self.ac_buf_filled
        if n >= self.ac_window_frames:
            start = self.ac_buf_pos % self.ac_window_frames
            signal = np.concatenate([self.ac_buf[start:], self.ac_buf[:start]])
        else:
            signal = self.ac_buf[:n].copy()

        signal = signal - np.mean(signal)
        norm = np.dot(signal, signal)
        if norm < 1e-20:
            return

        min_lag = self.min_period_frames
        max_lag = min(self.max_period_frames, len(signal) // 2)
        if min_lag >= max_lag:
            return

        autocorr = np.zeros(max_lag - min_lag, dtype=np.float64)
        for i, lag in enumerate(range(min_lag, max_lag)):
            autocorr[i] = np.dot(signal[:-lag], signal[lag:]) / norm

        # Find first prominent peak
        best_lag = -1
        best_corr = 0.0
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > best_corr:
                    best_corr = autocorr[i]
                    best_lag = min_lag + i
                    if best_corr > self.ac_min_confidence:
                        break

        self.ac_confidence = best_corr

        if best_corr > self.ac_min_confidence and best_lag > 0:
            new_period = best_lag / self.rms_fps

            # Smooth period changes to avoid jarring tempo jumps
            if self.estimated_period > 0:
                # Only update if within 20% of current estimate or first estimate
                ratio = new_period / self.estimated_period
                if 0.8 < ratio < 1.2:
                    # Exponential smoothing
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * new_period
                elif 0.45 < ratio < 0.55:
                    # Detected double-time — use it
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * (new_period * 2)
                elif 1.8 < ratio < 2.2:
                    # Detected half-time — use it
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * (new_period / 2)
                # Otherwise ignore the outlier
            else:
                self.estimated_period = new_period

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            b = self.brightness

        b = min(b, 1.0) * self.max_brightness
        display_b = b ** 0.6

        color = self._sample_palette(b / self.max_brightness if self.max_brightness > 0 else 0)
        pixel = (color * display_b).clip(0, 255).astype(np.uint8)
        frame = np.tile(pixel, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        bpm = 60.0 / self.estimated_period if self.estimated_period > 0 else 0
        return {
            'brightness': f'{self.brightness:.2f}',
            'rms': f'{self.current_rms:.2f}',
            'bpm': f'{bpm:.1f}',
            'conf': f'{self.ac_confidence:.2f}',
            'phase': f'{self.phase:.2f}',
        }
