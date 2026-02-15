"""
Basic Sparkles — non-audio-reactive red-magenta twinkling over a dim base.

The whole strip glows at very low brightness in a deep red-magenta.
Individual LEDs randomly fade in and out (twinkle) on top, averaging
~20 active at any time. Single-LED twinkles, no movement.
"""

import numpy as np
from base import AudioReactiveEffect


class BasicSparklesEffect(AudioReactiveEffect):
    """Dim red-magenta base with single-LED twinkles fading in and out."""

    registry_name = 'basic_sparkles'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # Dim base color (deep red-magenta, very low)
        self.base_color = np.array([15, 1, 6], dtype=np.float32)

        # Twinkle state: one slot per LED, brightness 0 = inactive
        self.twinkle_brightness = np.zeros(num_leds, dtype=np.float32)
        self.twinkle_target = np.zeros(num_leds, dtype=np.float32)  # 0 = fading out, >0 = fading in
        self.twinkle_speed = np.zeros(num_leds, dtype=np.float32)   # fade speed per frame
        # Color per LED (assigned on spawn)
        self.twinkle_colors = np.zeros((num_leds, 3), dtype=np.float32)

        self.target_active = 20

    @property
    def name(self):
        return "Basic Sparkles"

    @property
    def description(self):
        return "Dim red-magenta base with single-LED twinkles fading in and out. Not audio-reactive."

    def _pick_color(self):
        """Random color in the red-magenta range."""
        hue = np.random.uniform(0, 1)
        if hue < 0.5:
            t = hue / 0.5
            return np.array([255, 0, t * 130], dtype=np.float32)
        else:
            t = (hue - 0.5) / 0.5
            return np.array([255, t * 30, 130 - t * 50], dtype=np.float32)

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def render(self, dt: float) -> np.ndarray:
        step = dt * 30  # normalize to ~30fps

        # Update all twinkles toward their targets
        for i in range(self.num_leds):
            if self.twinkle_brightness[i] == 0 and self.twinkle_target[i] == 0:
                continue  # inactive

            diff = self.twinkle_target[i] - self.twinkle_brightness[i]
            move = self.twinkle_speed[i] * step

            if abs(diff) <= move:
                self.twinkle_brightness[i] = self.twinkle_target[i]
                if self.twinkle_target[i] > 0:
                    # Reached peak — start fading out
                    self.twinkle_target[i] = 0
                    self.twinkle_speed[i] = np.random.uniform(0.008, 0.025)
                # else: reached 0, now fully inactive
            else:
                self.twinkle_brightness[i] += move if diff > 0 else -move

        # Count active (brightness > 0)
        active = np.sum(self.twinkle_brightness > 0)

        # Spawn new twinkles to maintain target count
        # Spawn a few per frame to stay near target smoothly
        deficit = self.target_active - active
        if deficit > 0:
            spawns = min(int(deficit * 0.3) + 1, 3)
            inactive = np.where(self.twinkle_brightness == 0)[0]
            if len(inactive) > 0:
                chosen = np.random.choice(inactive, size=min(spawns, len(inactive)), replace=False)
                for idx in chosen:
                    self.twinkle_target[idx] = np.random.uniform(0.4, 1.0)
                    self.twinkle_speed[idx] = np.random.uniform(0.01, 0.03)
                    self.twinkle_brightness[idx] = 0.001  # mark as active
                    self.twinkle_colors[idx] = self._pick_color()

        # Build frame: base + twinkles
        frame = np.tile(self.base_color, (self.num_leds, 1))
        mask = self.twinkle_brightness > 0
        if np.any(mask):
            intensities = self.twinkle_brightness[mask, np.newaxis]
            frame[mask] += self.twinkle_colors[mask] * intensities

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        return {
            'active': int(np.sum(self.twinkle_brightness > 0)),
        }
