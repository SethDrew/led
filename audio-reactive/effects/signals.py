"""
Reusable signal processing primitives for audio-reactive effects.

Three building blocks that effects compose via has-a:

  OverlapFrameAccumulator — feeds audio chunks, yields overlapped frames
  AbsIntegral             — computes normalized abs-integral of RMS derivative
  BeatPredictor           — autocorrelation tempo + predicted/confirmed beats

Usage:
    accum = OverlapFrameAccumulator()
    absint = AbsIntegral(sample_rate=44100)
    predictor = BeatPredictor(rms_fps=absint.rms_fps)

    def process_audio(self, chunk):
        for frame in accum.feed(chunk):
            normalized = absint.update(frame)
            beats = predictor.feed(normalized, absint.time_acc)
            ...
"""

import numpy as np


class OverlapFrameAccumulator:
    """Accumulates audio chunks into overlapped analysis frames.

    Handles the bookkeeping of filling a buffer, yielding complete frames,
    and shifting by hop_size for overlap. Every effect that processes RMS
    or FFT frames uses this same loop.
    """

    def __init__(self, frame_len: int = 2048, hop: int = 512):
        self.frame_len = frame_len
        self.hop = hop
        self.buf = np.zeros(frame_len, dtype=np.float32)
        self.pos = 0

    def feed(self, chunk: np.ndarray) -> list:
        """Feed an audio chunk, return list of complete overlapped frames."""
        frames = []
        n = len(chunk)
        pos = self.pos
        offset = 0

        while n > 0:
            space = self.frame_len - pos
            take = min(n, space)
            self.buf[pos:pos + take] = chunk[offset:offset + take]
            offset += take
            pos += take
            n -= take

            if pos >= self.frame_len:
                frames.append(self.buf.copy())
                self.buf[:self.frame_len - self.hop] = self.buf[self.hop:]
                pos = self.frame_len - self.hop

        self.pos = pos
        return frames


class AbsIntegral:
    """Computes normalized absolute integral of RMS derivative.

    The core signal used by most absint_* effects. Tracks how much the
    loudness is *changing* over a short window (default 150ms).

    Returns a normalized 0-1 value via update(). Also exposes raw values
    for effects that need them (e.g. for autocorrelation input).
    """

    def __init__(self, frame_len: int = 2048, sample_rate: int = 44100,
                 window_sec: float = 0.15, peak_decay: float = 0.998):
        self.frame_len = frame_len
        self.sample_rate = sample_rate
        self.dt = frame_len / sample_rate
        self.rms_fps = sample_rate / 512  # hop-based FPS (for autocorrelation)

        self.prev_rms = 0.0

        # Ring buffer of |d(RMS)/dt| values
        self.window_frames = max(1, int(window_sec / self.dt))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        # Signal state
        self.raw = 0.0              # raw abs-integral value
        self.peak = 1e-10           # slow-decay peak for normalization
        self.peak_decay = peak_decay
        self.normalized = 0.0       # 0-1 normalized value

        # Time tracking
        self.time_acc = 0.0
        self.rms_dt = 512 / sample_rate  # time per hop

    def update(self, frame: np.ndarray) -> float:
        """Process one frame, return normalized 0-1 value."""
        rms = np.sqrt(np.mean(frame ** 2))

        # RMS derivative
        rms_deriv = (rms - self.prev_rms) / self.dt
        self.prev_rms = rms

        # Store |derivative| in ring buffer
        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1

        # Abs-integral: sum of ring buffer * dt
        self.raw = np.sum(self.deriv_buf) * self.dt

        # Slow-decay peak normalization
        self.peak = max(self.raw, self.peak * self.peak_decay)
        self.normalized = self.raw / self.peak if self.peak > 0 else 0.0

        # Advance time
        self.time_acc += self.rms_dt

        return self.normalized

    @property
    def current_rms(self):
        """Last computed RMS value."""
        return self.prev_rms


class BeatPredictor:
    """Autocorrelation-based tempo estimation and beat prediction.

    Feed it the abs-integral signal value each frame. It detects confirmed
    beats (threshold crossing), estimates tempo via autocorrelation, and
    predicts future beats.

    Returns a list of beat events per frame:
      [{'type': 'confirmed', 'strength': 0.85}, ...]
      [{'type': 'predicted', 'strength': 0.80}, ...]
      []  # no beat this frame
    """

    def __init__(self, rms_fps: float, threshold: float = 0.30,
                 cooldown: float = 0.25, ac_window_sec: float = 5.0,
                 min_bpm: float = 40, max_bpm: float = 300,
                 confidence_threshold: float = 0.3,
                 predicted_strength: float = 0.8,
                 max_missed: int = 4):
        self.rms_fps = rms_fps
        self.threshold = threshold
        self.cooldown = cooldown
        self.predicted_strength = predicted_strength
        self.max_missed = max_missed

        # Beat detection state
        self.last_beat_time = -1.0
        self.beat_count = 0
        self.predicted_beat_count = 0

        # Autocorrelation buffer
        self.ac_window_frames = int(ac_window_sec * rms_fps)
        self.ac_buf = np.zeros(self.ac_window_frames, dtype=np.float32)
        self.ac_buf_pos = 0
        self.ac_buf_filled = 0

        # Period bounds
        self.min_period_frames = max(1, int(rms_fps * 60.0 / max_bpm))
        self.max_period_frames = int(rms_fps * 60.0 / min_bpm)

        # Autocorrelation result
        self.confidence = 0.0
        self.min_confidence = confidence_threshold
        self.estimated_period = 0.0  # seconds

        # Prediction state
        self.next_predicted_beat = 0.0
        self.prediction_active = False
        self.last_detection_time = -1.0
        self.missed_beats = 0

    def feed(self, abs_integral_value: float, normalized: float,
             time_acc: float) -> list:
        """Feed one frame of signal. Returns list of beat events."""
        events = []

        # Store for autocorrelation
        self.ac_buf[self.ac_buf_pos % self.ac_window_frames] = abs_integral_value
        self.ac_buf_pos += 1
        self.ac_buf_filled = min(self.ac_buf_filled + 1, self.ac_window_frames)

        # Late beat detection
        time_since_beat = time_acc - self.last_beat_time
        beat_detected = False

        if normalized > self.threshold and time_since_beat > self.cooldown:
            beat_detected = True
            self.last_beat_time = time_acc
            self.last_detection_time = time_acc
            self.beat_count += 1

            self._update_autocorrelation()
            self._update_prediction_phase()

            events.append({
                'type': 'confirmed',
                'strength': min(1.0, normalized),
            })

        # Check predicted beats
        if not beat_detected:
            pred = self._check_predicted_beat(time_acc)
            if pred:
                events.append(pred)

        return events

    def _update_autocorrelation(self):
        """Compute autocorrelation of abs-integral signal to estimate period."""
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

        best_lag = -1
        best_corr = 0.0
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > best_corr:
                    best_corr = autocorr[i]
                    best_lag = min_lag + i
                    if best_corr > self.min_confidence:
                        break

        self.confidence = best_corr

        if best_corr > self.min_confidence and best_lag > 0:
            new_period = best_lag / self.rms_fps

            # Octave correction + smoothing (from tempo_pulse's mature version)
            if self.estimated_period > 0:
                ratio = new_period / self.estimated_period
                if 0.8 < ratio < 1.2:
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * new_period
                elif 0.45 < ratio < 0.55:
                    # Detected half-period — double it
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * (new_period * 2)
                elif 1.8 < ratio < 2.2:
                    # Detected double-period — halve it
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * (new_period / 2)
                # else: too far off, ignore this estimate
            else:
                self.estimated_period = new_period

    def _update_prediction_phase(self):
        """Phase-lock prediction to the most recent late detection."""
        if self.estimated_period <= 0 or self.confidence < self.min_confidence:
            self.prediction_active = False
            return
        self.next_predicted_beat = self.last_detection_time + self.estimated_period
        self.prediction_active = True
        self.missed_beats = 0

    def _check_predicted_beat(self, time_acc: float):
        """Check if a predicted beat should fire now."""
        if not self.prediction_active or self.estimated_period <= 0:
            return None

        if time_acc >= self.next_predicted_beat:
            time_since_detection = time_acc - self.last_detection_time
            if time_since_detection < self.estimated_period * 0.3:
                # Too close to a real detection — skip
                self.next_predicted_beat += self.estimated_period
                self.missed_beats = 0
                return None

            self.predicted_beat_count += 1
            self.next_predicted_beat += self.estimated_period
            self.missed_beats += 1

            if self.missed_beats >= self.max_missed:
                self.prediction_active = False
                self.missed_beats = 0

            return {
                'type': 'predicted',
                'strength': self.predicted_strength,
            }
        return None

    @property
    def bpm(self) -> float:
        """Current estimated BPM, or 0 if unknown."""
        return 60.0 / self.estimated_period if self.estimated_period > 0 else 0.0
