"""
Abs-Integral Predictive — beat prediction via autocorrelation of abs-integral signal.

Builds on absint_pulse (late detection of beats via abs-integral threshold) and adds
tempo estimation via autocorrelation + forward prediction. The late detection confirms
beats, autocorrelation finds the period, and we predict the NEXT beat to fire on-time.

Key insight: autocorrelation of the abs-integral signal over a ~5s window is far more
robust for tempo estimation than measuring intervals between noisy threshold crossings.
The autocorrelation naturally averages over many beats and handles occasional missed/extra
detections gracefully.

Visual: whole tree pulses warm white on each predicted beat (80% brightness) and
confirmed beat (100% brightness). Exponential decay. Falls back to late-only detection
if autocorrelation can't find a confident period.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class AbsIntPredictiveEffect(AudioReactiveEffect):
    """Whole-tree pulse with tempo prediction via autocorrelation of abs-integral."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # ── RMS computation ──────────────────────────────────────────
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0

        # RMS state
        self.prev_rms = 0.0

        # Time per RMS hop (how often _process_rms_frame fires)
        self.rms_dt = self.rms_hop / sample_rate  # ~11.6ms at 44100/512

        # Frames per second of the abs-integral signal
        self.rms_fps = sample_rate / self.rms_hop  # ~86.13

        # ── Abs-integral: short window for beat detection ────────────
        self.window_sec = 0.15  # 150ms trailing window (best from analysis)
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        # Beat detection (same as absint_pulse)
        self.abs_integral = 0.0
        self.integral_peak = 1e-10
        self.peak_decay = 0.997
        self.threshold = 0.30
        self.cooldown = 0.25  # 250ms between late detections
        self.last_beat_time = -1.0
        self.time_acc = 0.0
        self.beat_count = 0  # confirmed (late) beats

        # ── Autocorrelation tempo estimation ─────────────────────────
        # Ring buffer of abs-integral values over last ~5 seconds
        self.ac_window_sec = 5.0
        self.ac_window_frames = int(self.ac_window_sec * self.rms_fps)
        self.ac_buf = np.zeros(self.ac_window_frames, dtype=np.float32)
        self.ac_buf_pos = 0
        self.ac_buf_filled = 0  # how many frames we've written total

        # Period bounds in RMS frames
        self.min_period_sec = 0.200   # 300 BPM cap
        self.max_period_sec = 1.500   # 40 BPM floor
        self.min_period_frames = max(1, int(self.min_period_sec * self.rms_fps))
        self.max_period_frames = int(self.max_period_sec * self.rms_fps)

        # Autocorrelation result
        self.ac_confidence = 0.0       # peak correlation strength (0-1)
        self.ac_min_confidence = 0.3   # minimum to trust autocorrelation
        self.estimated_period = 0.0    # in seconds, 0 means unknown

        # ── Prediction state ─────────────────────────────────────────
        self.next_predicted_beat = 0.0      # time_acc when next beat expected
        self.prediction_active = False       # are we currently predicting?
        self.last_detection_time = -1.0      # time_acc of most recent late detection
        self.phase_correction_ms = 0.050     # 50ms threshold
        self.missed_beats = 0                # consecutive missed expected beats
        self.max_missed_beats = 4            # kill prediction after this many
        self.predicted_beat_count = 0        # beats fired by prediction

        # ── Visual state (shared between threads) ────────────────────
        self.brightness = 0.0
        self.is_predicted = False  # was the current flash from prediction?
        self.decay_rate = 0.82
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
        return "AbsInt Predictive"

    @property
    def description(self):
        return "Combines abs-integral beat detection with autocorrelation tempo estimation to fire predicted beats on-time; confirmed at 100%, predicted at 80%."

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
        """Accumulate audio into RMS frames."""
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
                # Overlap: shift by hop
                self.audio_buf[:self.rms_frame_len - self.rms_hop] = \
                    self.audio_buf[self.rms_hop:]
                pos = self.rms_frame_len - self.rms_hop

        self.audio_buf_pos = pos

    def _process_rms_frame(self, frame):
        """Compute RMS, derivative, abs-integral, detect beats, and update prediction."""
        rms = np.sqrt(np.mean(frame ** 2))
        dt = self.rms_frame_len / self.sample_rate

        # RMS derivative
        rms_deriv = (rms - self.prev_rms) / dt
        self.prev_rms = rms

        # Store |derivative| in short ring buffer
        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1

        # Abs-integral: sum of ring buffer * dt
        self.abs_integral = np.sum(self.deriv_buf) * dt

        # Slow-decay peak normalization
        self.integral_peak = max(self.abs_integral, self.integral_peak * self.peak_decay)
        normalized = self.abs_integral / self.integral_peak if self.integral_peak > 0 else 0

        # Store abs-integral in long ring buffer for autocorrelation
        self.ac_buf[self.ac_buf_pos % self.ac_window_frames] = self.abs_integral
        self.ac_buf_pos += 1
        self.ac_buf_filled = min(self.ac_buf_filled + 1, self.ac_window_frames)

        # Advance time
        self.time_acc += self.rms_dt

        # ── Late beat detection (same as absint_pulse) ───────────────
        time_since_beat = self.time_acc - self.last_beat_time
        beat_detected = False

        if normalized > self.threshold and time_since_beat > self.cooldown:
            beat_detected = True
            self.last_beat_time = self.time_acc
            self.last_detection_time = self.time_acc
            self.beat_count += 1

            # Recompute autocorrelation on each late detection
            self._update_autocorrelation()

            # Phase-lock prediction to this detection
            self._update_prediction_phase()

            # Fire confirmed beat at full brightness
            with self._lock:
                self.brightness = min(1.0, normalized)
                self.is_predicted = False

        # ── Check if a predicted beat should fire ────────────────────
        if not beat_detected:
            self._check_predicted_beat()

    def _update_autocorrelation(self):
        """Compute autocorrelation of abs-integral signal to estimate beat period."""
        if self.ac_buf_filled < self.min_period_frames * 3:
            # Need at least 3 beat periods of data
            return

        # Extract the filled portion of the ring buffer in chronological order
        n = self.ac_buf_filled
        if n >= self.ac_window_frames:
            # Buffer is full — unwrap from current write position
            start = self.ac_buf_pos % self.ac_window_frames
            signal = np.concatenate([
                self.ac_buf[start:],
                self.ac_buf[:start]
            ])
        else:
            signal = self.ac_buf[:n].copy()

        # Remove mean to focus on oscillation
        signal = signal - np.mean(signal)

        # Normalize
        norm = np.dot(signal, signal)
        if norm < 1e-20:
            return

        # Compute autocorrelation for lags in [min_period, max_period]
        min_lag = self.min_period_frames
        max_lag = min(self.max_period_frames, len(signal) // 2)

        if min_lag >= max_lag:
            return

        autocorr = np.zeros(max_lag - min_lag, dtype=np.float64)
        for i, lag in enumerate(range(min_lag, max_lag)):
            autocorr[i] = np.dot(signal[:-lag], signal[lag:]) / norm

        # Find the first prominent peak
        # Look for local maxima
        best_lag = -1
        best_corr = 0.0

        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > best_corr:
                    best_corr = autocorr[i]
                    best_lag = min_lag + i
                    # Take the first prominent peak, not the global max
                    # (first peak = fundamental period, later peaks = harmonics)
                    if best_corr > self.ac_min_confidence:
                        break

        self.ac_confidence = best_corr

        if best_corr > self.ac_min_confidence and best_lag > 0:
            self.estimated_period = best_lag / self.rms_fps
        # If confidence is low, keep the old period (don't reset to 0)
        # This provides hysteresis — we only update when we have strong evidence

    def _update_prediction_phase(self):
        """Phase-lock prediction to the most recent late detection."""
        if self.estimated_period <= 0 or self.ac_confidence < self.ac_min_confidence:
            # No confident period — can't predict
            self.prediction_active = False
            return

        # Set next predicted beat from the most recent detection
        self.next_predicted_beat = self.last_detection_time + self.estimated_period
        self.prediction_active = True
        self.missed_beats = 0

    def _check_predicted_beat(self):
        """Check if a predicted beat should fire now."""
        if not self.prediction_active:
            return

        if self.estimated_period <= 0:
            return

        # Has the predicted beat time arrived?
        if self.time_acc >= self.next_predicted_beat:
            # Check if a late detection is close enough to this prediction
            # (if so, the late detection already fired — don't double-fire)
            time_since_detection = self.time_acc - self.last_detection_time
            if time_since_detection < self.estimated_period * 0.3:
                # Recent detection was close — it already fired, just advance
                self.next_predicted_beat += self.estimated_period
                self.missed_beats = 0
                return

            # No recent detection — fire predicted beat at 80% brightness
            with self._lock:
                self.brightness = max(self.brightness, 0.8)
                self.is_predicted = True
            self.predicted_beat_count += 1

            # Advance to next expected beat
            self.next_predicted_beat += self.estimated_period
            self.missed_beats += 1

            # Kill prediction if too many misses
            if self.missed_beats >= self.max_missed_beats:
                self.prediction_active = False
                self.missed_beats = 0

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            b = self.brightness

        self.brightness *= self.decay_rate ** (dt * 30)

        # Gamma correction
        display_b = b ** 0.7

        color = self._sample_palette(b)
        pixel = (color * display_b).clip(0, 255).astype(np.uint8)
        frame = np.tile(pixel, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        period_ms = self.estimated_period * 1000 if self.estimated_period > 0 else 0
        bpm = 60.0 / self.estimated_period if self.estimated_period > 0 else 0
        return {
            'confirmed': self.beat_count,
            'predicted': self.predicted_beat_count,
            'brightness': f'{self.brightness:.2f}',
            'period_ms': f'{period_ms:.0f}',
            'bpm': f'{bpm:.1f}',
            'ac_conf': f'{self.ac_confidence:.2f}',
            'predicting': self.prediction_active,
        }
