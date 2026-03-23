"""
Flicker Flame — per-LED organic flame flicker driven by audio.

Each LED flickers independently like a candle. Audio energy controls overall
brightness while energy delta modulates flicker turbulence. Colors follow a
black-body curve from deep red embers through orange to warm yellow.

Looks great in a bunch — every LED dances on its own timeline.
"""

import numpy as np
import math
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, FixedRangeRMS, EnergyDelta


class FlickerFlameEffect(AudioReactiveEffect):
    """Per-LED flame flicker driven by audio energy and dynamics."""

    registry_name = 'flicker_flame'
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'RMS energy + energy delta'

    source_features = [
        {'id': 'energy', 'label': 'Energy', 'color': '#ff4400'},
        {'id': 'flicker', 'label': 'Flicker', 'color': '#ffaa00'},
    ]

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # --- Audio analysis ---
        self.n_fft = 2048
        self.hop_length = 512
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self._mapper = FixedRangeRMS(
            floor_rms=0.005, ceiling_rms=0.06,
            peak_decay=0.9999,
            fps=sample_rate / self.hop_length,
        )
        self._delta = EnergyDelta()

        self._energy = np.float32(0.0)
        self._energy_delta = np.float32(0.0)
        self._lock = threading.Lock()

        # --- Render state ---
        self._time = 0.0
        self._base_brightness = 0.0
        self._flicker_intensity = 0.0

        # Sustain tracking for accents
        self._sustain_time = 0.0  # how long energy has been above threshold
        self._warmth_drift = 0.0  # 0-1 warmth accumulator

        # Pre-compute LED index arrays for vectorized noise
        self._led_indices = np.arange(num_leds, dtype=np.float64)

        # Black-body color anchors (RGB)
        # dim < 0.3: deep red/ember
        # medium ~ 0.5: orange
        # bright > 0.8: warm yellow
        self._color_points = np.array([
            [180.0, 30.0, 0.0],    # dim — ember
            [255.0, 120.0, 20.0],  # medium — orange
            [255.0, 220.0, 80.0],  # bright — warm yellow
        ])
        self._color_breaks = np.array([0.3, 0.8])  # boundaries between regions

        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Flicker Flame"

    @property
    def description(self):
        return "Organic per-LED flame flicker driven by audio energy and dynamics."

    # ------------------------------------------------------------------ #
    #  Audio thread                                                        #
    # ------------------------------------------------------------------ #

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            energy = self._mapper.update(frame)
            delta = self._delta.update(frame)
            with self._lock:
                self._energy = np.float32(energy)
                self._energy_delta = np.float32(delta)

    # ------------------------------------------------------------------ #
    #  Render thread                                                       #
    # ------------------------------------------------------------------ #

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = float(self._energy)
            energy_delta = float(self._energy_delta)

        self._time += dt

        # FixedRangeRMS returns 0 when gated
        is_silent = energy < 0.001

        # Asymmetric EMA for base brightness: ~50ms attack, ~1s release
        attack_alpha = min(1.0, dt / 0.050)
        decay_alpha = min(1.0, dt / 1.0)

        target = 0.0 if is_silent else energy
        if target > self._base_brightness:
            self._base_brightness += attack_alpha * (target - self._base_brightness)
        else:
            self._base_brightness += decay_alpha * (target - self._base_brightness)

        # Flicker intensity: EMA-smoothed energy delta (~200ms TC)
        flicker_alpha = min(1.0, dt / 0.200)
        delta_target = 0.0 if is_silent else energy_delta
        self._flicker_intensity += flicker_alpha * (delta_target - self._flicker_intensity)

        # --- Sustain accents ---
        # Track how long energy has been sustained
        if self._base_brightness > 0.15:
            self._sustain_time += dt
        else:
            self._sustain_time = max(0.0, self._sustain_time - dt * 2)

        # 1. Slow shimmer wave: kicks in after 0.5s sustain
        shimmer = 0.0
        if self._sustain_time > 0.5:
            shimmer_amount = min(1.0, (self._sustain_time - 0.5) / 2.0)  # ramps over 2s
            shimmer = shimmer_amount * 0.15  # up to 15% brightness modulation

        # 2. Warmth drift: sustained notes gradually shift warmer
        if self._sustain_time > 0.3:
            self._warmth_drift = min(1.0, self._warmth_drift + dt * 0.3)  # ~3s to full
        else:
            self._warmth_drift = max(0.0, self._warmth_drift - dt * 1.0)  # ~1s to reset

        # Vectorized per-LED noise: two incommensurate sine waves
        t = self._time
        idx = self._led_indices
        noise = (np.sin(idx * 7.3 + t * 4.1) *
                 np.sin(idx * 3.7 + t * 2.3) * 0.5 + 0.5)

        # Slow shimmer: a separate low-frequency wave across LEDs
        shimmer_wave = np.sin(idx * 0.4 + t * 1.2) * 0.5 + 0.5  # slow, broad

        # Per-LED brightness
        base = self._base_brightness
        bright = (base * (0.7 + 0.3 * noise)
                  + self._flicker_intensity * (noise - 0.5) * 0.4
                  + shimmer * shimmer_wave)

        # Deadband: snap to black below threshold, no hue shift during fade
        DEADBAND = 0.08
        bright = np.where(bright < DEADBAND, 0.0, bright)
        bright = np.clip(bright, 0.0, 1.0)

        # Black-body color interpolation based on per-LED brightness
        # Warmth drift shifts the palette: embers get oranger, orange gets yellower
        w = self._warmth_drift
        c0 = self._color_points[0] + w * np.array([30.0, 40.0, 10.0])   # ember → warmer ember
        c1 = self._color_points[1] + w * np.array([0.0, 30.0, 20.0])    # orange → richer orange
        c2 = self._color_points[2] + w * np.array([0.0, 10.0, 30.0])    # yellow → brighter yellow
        c0 = np.clip(c0, 0, 255)
        c1 = np.clip(c1, 0, 255)
        c2 = np.clip(c2, 0, 255)
        b0, b1 = self._color_breaks

        # Compute interpolation for each LED
        colors = np.empty((self.num_leds, 3), dtype=np.float64)

        # Lower region: ember -> orange
        low_mask = bright <= b0
        low_t = np.clip(bright / b0, 0.0, 1.0)
        for ch in range(3):
            colors[:, ch] = c0[ch] + (c1[ch] - c0[ch]) * low_t

        # Upper region: orange -> yellow
        high_mask = bright > b0
        high_t = np.clip((bright - b0) / (b1 - b0), 0.0, 1.0)
        for ch in range(3):
            colors[high_mask, ch] = c1[ch] + (c2[ch] - c1[ch]) * high_t[high_mask]

        # Apply brightness to color
        colors *= bright[:, np.newaxis]
        np.clip(colors, 0, 255, out=colors)

        self._frame_buf[:] = colors.astype(np.uint8)
        return self._frame_buf.copy()

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = float(self._energy)
            delta = float(self._energy_delta)

        return {
            'energy': f'{energy:.3f}',
            'energy_delta': f'{delta:.3f}',
            'base_brightness': f'{self._base_brightness:.3f}',
            'flicker_intensity': f'{self._flicker_intensity:.3f}',
        }

    def get_source_values(self) -> dict:
        with self._lock:
            return {
                'energy': float(self._energy),
                'flicker': float(self._energy_delta),
            }
