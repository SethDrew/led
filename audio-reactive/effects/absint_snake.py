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
from signals import OverlapFrameAccumulator, AbsIntegral, BeatPredictor


class AbsIntSnakeEffect(AudioReactiveEffect):
    """Beat-triggered traveling pulses with size proportional to beat strength."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)
        self.predictor = BeatPredictor(rms_fps=self.absint.rms_fps)

        # Snake parameters
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

    def process_audio(self, mono_chunk):
        for frame in self.accum.feed(mono_chunk):
            normalized = self.absint.update(frame)
            beats = self.predictor.feed(self.absint.raw, normalized, self.absint.time_acc)

            for beat in beats:
                if beat['type'] == 'confirmed':
                    self._spawn_snake(beat['strength'])
                elif beat['type'] == 'predicted':
                    # Predicted beat — spawn at 80% strength
                    self._spawn_snake(beat['strength'])

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
        return {
            'beats': self.predictor.beat_count,
            'pred': self.predictor.predicted_beat_count,
            'snakes': len(self.snakes),
            'bpm': f'{self.predictor.bpm:.1f}',
            'conf': f'{self.predictor.confidence:.2f}',
        }
