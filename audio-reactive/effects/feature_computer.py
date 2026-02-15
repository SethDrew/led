"""
FeatureComputer — computes common audio features from raw audio chunks.

Standalone feature extraction, independent of any specific effect.
Thread-safe: process_audio() called from audio callback, get_features() from main loop.

Features:
  - abs_integral: |dRMS/dt| integrated over 150ms, peak-normalized (decay 0.998)
  - rms: Raw RMS, peak-normalized (decay 0.9998)
  - centroid: Spectral centroid mapped to 0-1 (log scale, 200Hz-10kHz)
  - autocorr_conf: Autocorrelation peak confidence from 5s buffer (0-1)
"""

import numpy as np
import threading


class FeatureComputer:
    """Computes common audio features from raw audio chunks."""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

        # ── RMS computation ──
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0

        # RMS state
        self.prev_rms = 0.0
        self.rms_dt = self.rms_hop / sample_rate

        # ── Abs-integral (150ms window) ──
        self.window_sec = 0.15
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0
        self.abs_integral = 0.0
        self.integral_peak = 1e-10
        self.integral_peak_decay = 0.998

        # ── RMS peak normalization ──
        self.current_rms = 0.0
        self.rms_peak = 1e-10
        self.rms_peak_decay = 0.9998

        # ── Spectral centroid ──
        self.fft_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.centroid_hz = 0.0
        self.centroid_min_hz = 200.0
        self.centroid_max_hz = 10000.0
        self.log_min = np.log(self.centroid_min_hz)
        self.log_max = np.log(self.centroid_max_hz)

        # ── Autocorrelation (5s buffer of abs-integral values) ──
        self.rms_fps = sample_rate / self.rms_hop
        self.ac_window_sec = 5.0
        self.ac_window_frames = int(self.ac_window_sec * self.rms_fps)
        self.ac_buf = np.zeros(self.ac_window_frames, dtype=np.float32)
        self.ac_buf_pos = 0
        self.ac_buf_filled = 0
        self.ac_confidence = 0.0
        self.min_period_frames = max(1, int(0.200 * self.rms_fps))  # 300 BPM
        self.max_period_frames = int(1.500 * self.rms_fps)           # 40 BPM

        # ── Thread safety ──
        self._lock = threading.Lock()
        self._features = {
            'abs_integral': 0.0,
            'rms': 0.0,
            'centroid': 0.0,
            'autocorr_conf': 0.0,
        }

    def process_audio(self, mono_chunk):
        """Thread-safe. Called from audio callback."""
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
        """Process one RMS frame: compute all features."""
        dt = self.rms_frame_len / self.sample_rate

        # ── RMS ──
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self.current_rms = rms
        self.rms_peak = max(rms, self.rms_peak * self.rms_peak_decay)
        norm_rms = rms / self.rms_peak if self.rms_peak > 0 else 0.0

        # ── RMS derivative → abs-integral ──
        rms_deriv = (rms - self.prev_rms) / dt
        self.prev_rms = rms

        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1
        self.abs_integral = float(np.sum(self.deriv_buf) * dt)

        self.integral_peak = max(self.abs_integral, self.integral_peak * self.integral_peak_decay)
        norm_integral = self.abs_integral / self.integral_peak if self.integral_peak > 0 else 0.0

        # ── Spectral centroid ──
        windowed = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)
        total = spectrum.sum()
        if total > 1e-10:
            self.centroid_hz = float(np.sum(freqs * spectrum) / total)
        centroid_clamped = np.clip(self.centroid_hz, self.centroid_min_hz, self.centroid_max_hz)
        norm_centroid = (np.log(centroid_clamped) - self.log_min) / (self.log_max - self.log_min)
        norm_centroid = float(np.clip(norm_centroid, 0.0, 1.0))

        # ── Autocorrelation buffer ──
        self.ac_buf[self.ac_buf_pos % self.ac_window_frames] = self.abs_integral
        self.ac_buf_pos += 1
        self.ac_buf_filled = min(self.ac_buf_filled + 1, self.ac_window_frames)

        # Recompute autocorrelation every ~0.5s (every 43 frames)
        if self.ac_buf_pos % 43 == 0:
            self._update_autocorrelation()

        with self._lock:
            self._features = {
                'abs_integral': float(np.clip(norm_integral, 0.0, 1.0)),
                'rms': float(np.clip(norm_rms, 0.0, 1.0)),
                'centroid': norm_centroid,
                'autocorr_conf': float(np.clip(self.ac_confidence, 0.0, 1.0)),
            }

    def _update_autocorrelation(self):
        """Compute autocorrelation peak confidence."""
        if self.ac_buf_filled < self.min_period_frames * 3:
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

        best_corr = 0.0
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > best_corr:
                    best_corr = autocorr[i]
                    if best_corr > 0.3:
                        break

        self.ac_confidence = best_corr

    def get_features(self):
        """Returns dict of normalized (0-1) feature values. Thread-safe."""
        with self._lock:
            return dict(self._features)
