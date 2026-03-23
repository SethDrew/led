"""
Flicker Flame — per-LED organic flame flicker driven by audio.

Each LED flickers independently like a candle. Audio energy controls color:
more energy pushes toward deep saturated red, while silence fades to a soft
warm white at low brightness (near-equal RGB so firmware RGBW conversion
puts most light on the dedicated warm white LED).

Color journey: warm amber (idle) → deep red (loud) → soft warm white (silence/decay)
"""

import numpy as np
import math
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, FixedRangeRMS, EnergyDelta


class FlickerFlameWarmthEffect(AudioReactiveEffect):
    """Flicker flame with energy-driven red shift and warm white decay."""

    registry_name = 'flicker_flame_warmth'
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

        # Smoothed energy for color mapping — tracks sustained input
        self._color_energy = 0.0

        # Pre-compute LED index arrays for vectorized noise
        self._led_indices = np.arange(num_leds, dtype=np.float64)

        # Color anchors
        self._color_amber = np.array([255.0, 140.0, 30.0])       # warm amber (default)
        self._color_deep_red = np.array([200.0, 20.0, 0.0])      # high energy
        self._color_warm_white = np.array([180.0, 170.0, 160.0])  # silence decay (RGBW → W)

        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Flicker Flame"

    @property
    def description(self):
        return "Organic per-LED flame flicker: energy → deep red, silence → soft warm white."

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

        # Sustained energy vs percussive-only detection:
        # High energy_delta relative to energy means transient-only (no sustain).
        # We want color to respond to sustained energy, not just transients.
        is_percussive_only = (not is_silent and energy < 0.15 and energy_delta > 0.5)

        # --- Base brightness envelope ---
        # Asymmetric EMA: ~50ms attack, ~2s release (slower release for warm white hold)
        attack_alpha = min(1.0, dt / 0.050)
        decay_alpha = min(1.0, dt / 2.0)

        if is_silent:
            target_brightness = 0.25  # hold at 25% during silence (above deadband even with noise)
        else:
            target_brightness = max(0.25, energy)  # at least 25%, up to full energy

        if target_brightness > self._base_brightness:
            self._base_brightness += attack_alpha * (target_brightness - self._base_brightness)
        else:
            self._base_brightness += decay_alpha * (target_brightness - self._base_brightness)

        # --- Flicker intensity: EMA-smoothed energy delta (~200ms TC) ---
        flicker_alpha = min(1.0, dt / 0.200)
        delta_target = 0.0 if is_silent else energy_delta
        self._flicker_intensity += flicker_alpha * (delta_target - self._flicker_intensity)

        # --- Color energy: smooth tracking of sustained energy for color mapping ---
        # Fast attack so color responds quickly to voice, slow release (~2s) for
        # smooth fade back to warm white
        color_attack = min(1.0, dt / 0.080)
        color_decay = min(1.0, dt / 2.0)

        if is_percussive_only:
            # Percussive-only: treat as silence for color purposes
            color_target = 0.0
        elif is_silent:
            color_target = 0.0
        else:
            color_target = energy

        if color_target > self._color_energy:
            self._color_energy += color_attack * (color_target - self._color_energy)
        else:
            self._color_energy += color_decay * (color_target - self._color_energy)

        # --- Per-LED flicker noise ---
        t = self._time
        idx = self._led_indices
        noise = (np.sin(idx * 7.3 + t * 4.1) *
                 np.sin(idx * 3.7 + t * 2.3) * 0.5 + 0.5)

        # Per-LED brightness
        base = self._base_brightness

        # Deadband: snap base to zero to prevent hue shift during fade.
        # Applied to base (not per-LED), so if base is above deadband all LEDs
        # stay lit with their flicker variation intact.
        DEADBAND = 0.08
        if base < DEADBAND:
            base = 0.0

        # Noise amplitude scales inversely with brightness: wide swing at idle
        # (~5%-20% range at 15% base) narrowing at high energy.
        noise_amp = max(0.3, 0.15 / max(base, 0.1))
        bright = (base * (1.0 + noise_amp * (noise - 0.5))
                  + self._flicker_intensity * (noise - 0.5) * 0.4)
        bright = np.clip(bright, 0.0, 1.0)

        # --- Color mapping: energy drives amber → red, silence drives → warm white ---
        ce = self._color_energy

        # Two-phase color:
        #   ce near 0 → warm white (silence/decay)
        #   ce > 0    → blend from amber toward deep red proportional to energy
        #
        # When ce is low (fading out), we blend from amber toward warm white.
        # When ce is high, we blend from amber toward deep red.

        # Threshold below which we start blending toward warm white
        WHITE_BLEND_THRESHOLD = 0.15

        if ce < WHITE_BLEND_THRESHOLD:
            # Blend amber → warm white as ce approaches 0
            t_white = 1.0 - (ce / WHITE_BLEND_THRESHOLD)  # 1.0 at silence, 0.0 at threshold
            base_color = self._color_amber * (1.0 - t_white) + self._color_warm_white * t_white
        else:
            # Blend amber → deep red. Full red by ce=0.5 (not 1.0)
            # so moderate sustained input reaches deep red within 4s
            RED_FULL = 0.5
            t_red = (ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD)
            t_red = min(1.0, t_red)
            base_color = self._color_amber * (1.0 - t_red) + self._color_deep_red * t_red

        # At low brightness, use pure equal-channel (→ all goes to W LED, no RGB strobe)
        # At higher brightness, blend in the actual color
        PURE_W_THRESHOLD = 0.25
        equal_white = np.array([170.0, 170.0, 170.0])  # equal channels → pure W after firmware conversion

        # Per-LED: blend between equal white and base_color based on brightness
        color_blend = np.clip((bright - 0.08) / (PURE_W_THRESHOLD - 0.08), 0.0, 1.0)
        # color_blend is 0 at low brightness (pure W), 1 above threshold (full color)
        per_led_color_r = equal_white[0] * (1.0 - color_blend) + base_color[0] * color_blend
        per_led_color_g = equal_white[1] * (1.0 - color_blend) + base_color[1] * color_blend
        per_led_color_b = equal_white[2] * (1.0 - color_blend) + base_color[2] * color_blend

        r = per_led_color_r * bright
        g = per_led_color_g * bright
        b = per_led_color_b * bright

        self._frame_buf[:, 0] = np.clip(r, 0, 255).astype(np.uint8)
        self._frame_buf[:, 1] = np.clip(g, 0, 255).astype(np.uint8)
        self._frame_buf[:, 2] = np.clip(b, 0, 255).astype(np.uint8)

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
            'color_energy': f'{self._color_energy:.3f}',
        }

    def get_source_values(self) -> dict:
        with self._lock:
            return {
                'energy': float(self._energy),
                'flicker': float(self._energy_delta),
            }
