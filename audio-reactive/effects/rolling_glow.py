"""
Rolling Glow — phrase-level energy arc with warmth-shifting color.

All LEDs show the same color/brightness. A 1.5-second rolling integral
of RMS smooths out individual syllables into phrase-level arcs — the
brightness rises and falls with the cadence of speech or musical phrases
rather than individual beats.

Color shifts based on the energy derivative: building energy warms toward
orange, decaying energy cools toward blue-white, steady energy stays warm
white. This gives an organic, breathing quality that responds to the
emotional arc of the audio.

Silence fades naturally as the rolling window empties (~1.5s decay).
"""

import numpy as np
import threading
import math
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, RollingIntegral


class RollingGlowEffect(AudioReactiveEffect):
    """Phrase-level energy arc with warmth-shifting color."""

    registry_name = 'rolling_glow'
    ref_pattern = 'section'
    ref_scope = 'phrase'
    ref_input = 'Rolling RMS integral'

    source_features = [
        {'id': 'energy', 'label': 'Energy', 'color': '#ff8800'},
        {'id': 'warmth', 'label': 'Warmth', 'color': '#ff6600'},
    ]

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # --- Audio analysis ---
        self.n_fft = 2048
        self.hop_length = 512
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self._integral = RollingIntegral(
            window_sec=1.5, fps=sample_rate / self.hop_length,
        )

        self._energy = np.float32(0.0)
        self._lock = threading.Lock()

        # --- Render state ---
        self._peak = 1e-6          # slow-adapting peak for normalization
        self._prev_integral = 0.0  # for derivative computation
        self._deriv_smooth = 0.0   # EMA-smoothed derivative
        self._warmth = 0.0         # mapped derivative, -1 to 1

        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Rolling Glow"

    @property
    def description(self):
        return "Phrase-level energy arc with warmth-shifting color."

    # ------------------------------------------------------------------ #
    #  Audio thread                                                        #
    # ------------------------------------------------------------------ #

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            val = self._integral.update(frame)
            with self._lock:
                self._energy = np.float32(val)

    # ------------------------------------------------------------------ #
    #  Render thread                                                       #
    # ------------------------------------------------------------------ #

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            integral = float(self._energy)

        # Slow-adapting peak for auto-scaling (decay 0.9998 per frame)
        self._peak = max(integral, self._peak * 0.9998)
        self._peak = max(self._peak, 1e-6)
        brightness = min(1.0, integral / self._peak)

        # Energy derivative → warmth shift
        if dt > 0:
            raw_deriv = (integral - self._prev_integral) / dt
        else:
            raw_deriv = 0.0
        self._prev_integral = integral

        # EMA smooth: ~500ms TC at 30fps → alpha ≈ 0.065
        deriv_alpha = min(1.0, dt / 0.500)
        self._deriv_smooth += deriv_alpha * (raw_deriv - self._deriv_smooth)

        # Map through tanh: scale so typical speech deltas (~0.5/s) → ~0.5
        self._warmth = math.tanh(self._deriv_smooth * 2.0)

        # Color blending based on warmth [-1, 1]
        #   -1 (quieting) → cool blue-white (200, 210, 255)
        #    0 (neutral)  → warm white (255, 220, 180)
        #   +1 (building) → warm orange (255, 140, 40)
        w = self._warmth
        if w >= 0:
            # Neutral → warm orange
            r = 255.0 + (255.0 - 255.0) * w
            g = 220.0 + (140.0 - 220.0) * w
            b = 180.0 + (40.0 - 180.0) * w
        else:
            # Neutral → cool blue-white
            t = -w
            r = 255.0 + (200.0 - 255.0) * t
            g = 220.0 + (210.0 - 220.0) * t
            b = 180.0 + (255.0 - 180.0) * t

        # Apply brightness
        pixel = np.array([
            int(max(0, min(255, r * brightness))),
            int(max(0, min(255, g * brightness))),
            int(max(0, min(255, b * brightness))),
        ], dtype=np.uint8)

        self._frame_buf[:] = pixel
        return self._frame_buf.copy()

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = float(self._energy)

        return {
            'integral': f'{energy:.4f}',
            'peak': f'{self._peak:.4f}',
            'warmth': f'{self._warmth:.3f}',
            'deriv': f'{self._deriv_smooth:.4f}',
        }

    def get_source_values(self) -> dict:
        with self._lock:
            integral = float(self._energy)

        brightness = min(1.0, integral / self._peak) if self._peak > 1e-6 else 0.0
        # Map warmth from [-1,1] to [0,1] for source value
        warmth_01 = (self._warmth + 1.0) / 2.0

        return {
            'energy': brightness,
            'warmth': warmth_01,
        }
