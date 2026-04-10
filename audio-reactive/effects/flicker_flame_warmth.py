"""
Fire Meld / Fire Flicker — per-LED organic flame flicker driven by audio.

Two bulb effects built from the same engine:

  Fire Meld (registry: fire_meld)
    Each LED flickers independently like a candle. Audio energy controls color:
    amber at rest, deep red when loud, soft warm white during silence/decay.
    No sustain dropout — pure flickering fire.

  Fire Flicker (registry: fire_flicker)
    Same base flicker, plus a sustain-triggered "blown fire" effect: after ~1s
    of sustained energy, LEDs gradually shift to ember red then dim out one by
    one over ~3s, like blowing on a campfire. Resets when energy changes.

Color journey: warm amber (idle) → deep red (loud) → soft warm white (silence/decay)
"""

import numpy as np
import math
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, FixedRangeRMS, EnergyDelta


class FlickerFlameWarmthEffect(AudioReactiveEffect):
    """Per-LED flame flicker with energy-driven color shift. Base class for fire effects."""

    registry_name = 'fire_meld'
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'RMS energy + energy delta'

    source_features = [
        {'id': 'energy', 'label': 'Energy', 'color': '#ff4400'},
        {'id': 'flicker', 'label': 'Flicker', 'color': '#ffaa00'},
    ]

    def __init__(self, num_leds: int, sample_rate: int = 44100, flicker_scale: float = 3.0, dropout: float = 0.0):
        super().__init__(num_leds, sample_rate)
        self._flicker_scale = flicker_scale
        self._dropout_depth = dropout  # 0-1, max depth of sustain-triggered dropout

        # Sustain-triggered dropout state
        self._prev_energy_for_deriv = 0.0
        self._energy_deriv_smooth = 0.0
        self._dropout_amount = 0.0  # 0-1 current dropout level (ramps up after 500ms sustain)

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
        return "Fire Meld"

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

        # --- Sustain-triggered dropout ---
        # Detect sustained energy via smoothed derivative (same as shimmer variant)
        if self._dropout_depth > 0:
            energy_deriv = (energy - self._prev_energy_for_deriv) / max(dt, 0.001)
            self._prev_energy_for_deriv = energy
            deriv_alpha = min(1.0, dt / 0.200)
            self._energy_deriv_smooth += deriv_alpha * (energy_deriv - self._energy_deriv_smooth)

            # Ramp up after sustained energy: requires actual audio (not silence idle),
            # and derivative must be below onset level. Real onsets are 1.0+,
            # natural vocal wobble is 0.2-0.3, so 0.5 threshold rejects only true attacks.
            is_sustaining = (not is_silent and energy > 0.05
                             and abs(self._energy_deriv_smooth) <= 0.5)
            if is_sustaining:
                self._dropout_amount = min(1.0, self._dropout_amount + dt * 0.35)  # full in ~3s
            else:
                self._dropout_amount = max(0.0, self._dropout_amount - dt * 1.0)  # off in 1s

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

        # --- Per-LED flicker noise (slowed for subtlety) ---
        t = self._time
        idx = self._led_indices
        noise = (np.sin(idx * 7.3 + t * 2.5) *
                 np.sin(idx * 3.7 + t * 1.4) * 0.5 + 0.5)

        # Per-LED brightness
        base = self._base_brightness

        # Deadband: snap base to zero to prevent hue shift during fade.
        DEADBAND = 0.08
        if base < DEADBAND:
            base = 0.0

        # Noise amplitude scaled by flicker_scale
        s = self._flicker_scale
        noise_amp = max(0.15 * s, 0.10 * s / max(base, 0.1))
        bright = (base * (1.0 + noise_amp * (noise - 0.5))
                  + self._flicker_intensity * (noise - 0.5) * 0.25 * s)

        # --- Fire-blown dropout: sustain → redden → dim → go out ---
        # Each LED has a slow-drifting "resilience" (0-1). Low-resilience LEDs
        # start dimming first, creating a staggered wave of embers going out.
        # Color shifts to deep red BEFORE brightness drops (embers glow red
        # before dying). At full dropout (~4s sustain), ~60% of LEDs are out.
        per_led_dim = np.zeros(self.num_leds)
        if self._dropout_depth > 0 and self._dropout_amount > 0:
            # Slow-drifting resilience per LED (changes over ~5-10s)
            resilience = (np.sin(idx * 13.7 + t * 0.3) *
                          np.sin(idx * 9.1 + t * 0.2) * 0.5 + 0.5)
            # resilience is 0-1. LED starts dimming when dropout_amount
            # exceeds its resilience * 0.7 (so low-resilience LEDs go first).
            # The /0.3 controls how quickly each LED ramps once started.
            per_led_dim = np.clip(
                (self._dropout_amount - resilience * 0.7) / 0.3, 0.0, 1.0
            ) * self._dropout_depth

        # Apply brightness dropout
        bright *= (1.0 - per_led_dim)
        bright = np.clip(bright, 0.0, 1.0)

        # --- Color mapping: energy drives amber → red, silence drives → warm white ---
        # Dropout shifts color toward deep red/ember (color leads brightness)
        ce = self._color_energy

        WHITE_BLEND_THRESHOLD = 0.15

        if ce < WHITE_BLEND_THRESHOLD:
            t_white = 1.0 - (ce / WHITE_BLEND_THRESHOLD)
            base_color = self._color_amber * (1.0 - t_white) + self._color_warm_white * t_white
        else:
            RED_FULL = 0.5
            t_red = (ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD)
            t_red = min(1.0, t_red)
            base_color = self._color_amber * (1.0 - t_red) + self._color_deep_red * t_red

        # Per-LED color: dropout shifts toward ember red (color leads brightness)
        # At per_led_dim=0.3 the LED is already fully red, keeps dimming after that
        color_red_shift = np.clip(per_led_dim / 0.3, 0.0, 1.0)
        ember_color = self._color_deep_red  # [200, 20, 0]

        r = (base_color[0] * (1.0 - color_red_shift) + ember_color[0] * color_red_shift) * bright
        g = (base_color[1] * (1.0 - color_red_shift) + ember_color[1] * color_red_shift) * bright
        b = (base_color[2] * (1.0 - color_red_shift) + ember_color[2] * color_red_shift) * bright

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


class FireFlickerEffect(FlickerFlameWarmthEffect):
    """Fire Flicker — flame flicker with sustain-triggered blown-fire dropout."""

    registry_name = 'fire_flicker'

    def __init__(self, num_leds: int, sample_rate: int = 44100, **kwargs):
        kwargs.setdefault('flicker_scale', 3.0)
        kwargs.setdefault('dropout', 0.85)
        super().__init__(num_leds, sample_rate, **kwargs)

    @property
    def name(self):
        return "Fire Flicker"

    @property
    def description(self):
        return "Flame flicker with sustain-triggered blown-fire dropout: redden then dim."
