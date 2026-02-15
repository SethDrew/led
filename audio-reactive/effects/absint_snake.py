"""
AbsInt Snake — each detected beat spawns a traveling pulse ("snake").

Snake properties scale with beat strength (abs-integral magnitude):
  - Length: 1-10 LEDs
  - Travel distance: 20%-100% of strip
  - Speed: constant (~2 strip-lengths/sec), so bigger beats live longer

Color shifts from red → magenta as the snake travels toward the end of
the strip. Multiple snakes can overlap (additive blending).

Uses the same late detection + autocorrelation prediction from absint_pred.
"""

import numpy as np
import threading
from base import AudioReactiveEffect


class AbsIntSnakeEffect(AudioReactiveEffect):
    """Beat-triggered traveling pulses with size proportional to beat strength."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # ── RMS / abs-integral (same as absint_pred) ──
        self.rms_frame_len = 2048
        self.rms_hop = 512
        self.audio_buf = np.zeros(self.rms_frame_len, dtype=np.float32)
        self.audio_buf_pos = 0
        self.prev_rms = 0.0
        self.rms_dt = self.rms_hop / sample_rate
        self.rms_fps = sample_rate / self.rms_hop

        self.window_sec = 0.15
        self.window_frames = max(1, int(self.window_sec / (self.rms_frame_len / sample_rate)))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        self.abs_integral = 0.0
        self.integral_peak = 1e-10
        self.peak_decay = 0.997
        self.threshold = 0.30
        self.cooldown = 0.25
        self.last_beat_time = -1.0
        self.time_acc = 0.0
        self.beat_count = 0

        # ── Autocorrelation (from absint_pred) ──
        self.ac_window_sec = 5.0
        self.ac_window_frames = int(self.ac_window_sec * self.rms_fps)
        self.ac_buf = np.zeros(self.ac_window_frames, dtype=np.float32)
        self.ac_buf_pos = 0
        self.ac_buf_filled = 0

        self.min_period_sec = 0.200
        self.max_period_sec = 1.500
        self.min_period_frames = max(1, int(self.min_period_sec * self.rms_fps))
        self.max_period_frames = int(self.max_period_sec * self.rms_fps)

        self.ac_confidence = 0.0
        self.ac_min_confidence = 0.3
        self.estimated_period = 0.0
        self.next_predicted_beat = 0.0
        self.prediction_active = False
        self.last_detection_time = -1.0
        self.missed_beats = 0
        self.max_missed_beats = 4
        self.predicted_beat_count = 0

        # ── Snake parameters ──
        self.min_length = 1
        self.max_length = 10
        self.min_travel_frac = 0.20   # weakest beat travels 20% of strip
        self.max_travel_frac = 1.00   # strongest beat travels full strip
        self.speed = 0.25             # strip-lengths per second (4s to traverse full strip)

        # Active snakes: list of dicts
        self.snakes = []
        self.max_snakes = 12

        # Colors: red at start → magenta at end
        self.color_start = np.array([200, 20, 0], dtype=np.float32)    # red
        self.color_end = np.array([180, 0, 160], dtype=np.float32)     # magenta

        self.max_brightness = 0.80

        # Base pulse: first 20 LEDs flash on each beat
        self.base_pulse_leds = 20
        self.base_pulse_brightness = 0.0
        self.base_pulse_decay = 0.82

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Impulse Snake"

    @property
    def description(self):
        return "Beats spawn traveling pulses whose size and travel distance scale with beat strength; autocorrelation tempo prediction; red-to-magenta gradient."

    def _spawn_snake(self, strength):
        """Spawn a new snake with properties scaled by beat strength (0-1)."""
        s = np.clip(strength, 0, 1)
        length = int(self.min_length + s * (self.max_length - self.min_length))
        travel_dist = self.num_leds - self.base_pulse_leds

        snake = {
            'pos': float(self.base_pulse_leds),  # start from end of base pulse
            'length': length,
            'max_dist': travel_dist,
            'strength': s,
        }

        with self._lock:
            self.snakes.append(snake)
            if len(self.snakes) > self.max_snakes:
                self.snakes.pop(0)
            self.base_pulse_brightness = s

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

        # Store for autocorrelation
        self.ac_buf[self.ac_buf_pos % self.ac_window_frames] = self.abs_integral
        self.ac_buf_pos += 1
        self.ac_buf_filled = min(self.ac_buf_filled + 1, self.ac_window_frames)

        self.time_acc += self.rms_dt

        # Late detection
        time_since_beat = self.time_acc - self.last_beat_time
        beat_detected = False

        if normalized > self.threshold and time_since_beat > self.cooldown:
            beat_detected = True
            self.last_beat_time = self.time_acc
            self.last_detection_time = self.time_acc
            self.beat_count += 1
            self._update_autocorrelation()
            self._update_prediction_phase()
            self._spawn_snake(normalized)

        if not beat_detected:
            self._check_predicted_beat()

    def _update_autocorrelation(self):
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
                    if best_corr > self.ac_min_confidence:
                        break
        self.ac_confidence = best_corr
        if best_corr > self.ac_min_confidence and best_lag > 0:
            self.estimated_period = best_lag / self.rms_fps

    def _update_prediction_phase(self):
        if self.estimated_period <= 0 or self.ac_confidence < self.ac_min_confidence:
            self.prediction_active = False
            return
        self.next_predicted_beat = self.last_detection_time + self.estimated_period
        self.prediction_active = True
        self.missed_beats = 0

    def _check_predicted_beat(self):
        if not self.prediction_active or self.estimated_period <= 0:
            return
        if self.time_acc >= self.next_predicted_beat:
            time_since_detection = self.time_acc - self.last_detection_time
            if time_since_detection < self.estimated_period * 0.3:
                self.next_predicted_beat += self.estimated_period
                self.missed_beats = 0
                return
            # Predicted beat — spawn at 80% strength
            self._spawn_snake(0.8)
            self.predicted_beat_count += 1
            self.next_predicted_beat += self.estimated_period
            self.missed_beats += 1
            if self.missed_beats >= self.max_missed_beats:
                self.prediction_active = False
                self.missed_beats = 0

    def render(self, dt: float) -> np.ndarray:
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)
        step = self.speed * self.num_leds * dt  # LEDs to advance this frame

        with self._lock:
            # Base pulse: first 20 LEDs flash on beat, exponential decay
            bp = self.base_pulse_brightness
            self.base_pulse_brightness *= self.base_pulse_decay ** (dt * 30)

            alive = []
            for snake in self.snakes:
                # Advance position
                snake['pos'] += step

                # Dead if traveled past max distance
                if snake['pos'] - snake['length'] > snake['max_dist']:
                    continue
                alive.append(snake)

                # Draw snake
                head = int(snake['pos'])
                tail = max(0, head - snake['length'])
                head = min(head, self.num_leds)

                if tail >= self.num_leds or head <= 0:
                    continue

                for led in range(tail, head):
                    # Position along strip (0-1) for color
                    t = led / max(self.num_leds - 1, 1)
                    color = self.color_start * (1 - t) + self.color_end * t

                    # Fade based on how far into travel distance
                    travel_progress = snake['pos'] / snake['max_dist']
                    # Fade out in last 30% of travel
                    if travel_progress > 0.7:
                        fade = 1.0 - (travel_progress - 0.7) / 0.3
                    else:
                        fade = 1.0

                    brightness = snake['strength'] * fade * self.max_brightness
                    # Additive blend
                    frame[led] += color * brightness

            self.snakes = alive

        # Draw base pulse on first N LEDs
        if bp > 0.01:
            pulse_b = min(bp, 1.0) * self.max_brightness
            n = min(self.base_pulse_leds, self.num_leds)
            for led in range(n):
                t = led / max(self.num_leds - 1, 1)
                color = self.color_start * (1 - t) + self.color_end * t
                frame[led] += color * pulse_b

        return frame.clip(0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        bpm = 60.0 / self.estimated_period if self.estimated_period > 0 else 0
        return {
            'beats': self.beat_count,
            'pred': self.predicted_beat_count,
            'snakes': len(self.snakes),
            'bpm': f'{bpm:.1f}',
            'conf': f'{self.ac_confidence:.2f}',
        }
