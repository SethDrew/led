"""
Sparkle Burst — onset-driven per-LED sparkle bursts above a capped base glow.

Per-LED effect: sustained energy sets a warm amber base capped at ~50%
brightness. Onsets (claps, percussive hits) ignite random LEDs to full
brightness with a whiter color, then each decays back toward the base level.

Audio: FixedRangeRMS for energy envelope, EnergyDelta for onset detection.
Color: warm amber base (255, 180, 80), sparkle flash white (255, 240, 200).
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, FixedRangeRMS, EnergyDelta


class SparkleBurstEffect(AudioReactiveEffect):
    """Onset-driven per-LED sparkle bursts above a capped base glow."""

    registry_name = 'sparkle_burst'
    ref_pattern = 'accent'
    ref_scope = 'beat'
    ref_input = 'RMS energy + onset'

    source_features = [
        {'id': 'energy', 'label': 'Energy', 'color': '#ff8800'},
        {'id': 'onset', 'label': 'Onset', 'color': '#ffcc00'},
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
        self._onset = EnergyDelta()

        self._energy = np.float32(0.0)
        self._onset_val = np.float32(0.0)
        self._lock = threading.Lock()

        # --- Render state ---
        self._envelope = 0.0
        self._rng = np.random.default_rng(seed=42)

        # Per-LED sparkle state (0-1, where 1 = full sparkle above base)
        self.sparkle = np.zeros(num_leds)
        self._decay_rates = np.full(num_leds, 0.90)

        # Onset cooldown
        self._cooldown_remaining = 0.0

        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Sparkle Burst"

    @property
    def description(self):
        return "Onset-driven per-LED sparkle bursts above a capped base glow."

    # ------------------------------------------------------------------ #
    #  Audio thread                                                        #
    # ------------------------------------------------------------------ #

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            energy = self._mapper.update(frame)
            onset = self._onset.update(frame)
            with self._lock:
                self._energy = np.float32(energy)
                self._onset_val = np.float32(onset)

    # ------------------------------------------------------------------ #
    #  Render thread                                                       #
    # ------------------------------------------------------------------ #

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = float(self._energy)
            onset = float(self._onset_val)

        is_silent = energy < 0.001

        # Asymmetric envelope: moderate attack, 400ms decay
        attack_alpha = min(1.0, dt / 0.030)   # ~30ms attack
        decay_alpha = min(1.0, dt / 0.400)    # ~400ms decay

        if energy > self._envelope:
            self._envelope += attack_alpha * (energy - self._envelope)
        else:
            self._envelope += decay_alpha * (energy - self._envelope)

        # Cooldown tick
        self._cooldown_remaining = max(0.0, self._cooldown_remaining - dt)

        # Onset rate proportional to energy: higher energy = more frequent sparkles
        # Lower the onset threshold when energy is high
        onset_threshold = max(0.15, 0.4 - self._envelope * 0.3)  # 0.4 at silence → 0.15 at full energy

        # Onset detection: ignite random subset of LEDs
        if onset > onset_threshold and self._cooldown_remaining <= 0.0 and not is_silent:
            # Cooldown shorter at higher energy = more frequent sparkles
            self._cooldown_remaining = max(0.050, 0.150 - self._envelope * 0.10)

            onset_strength = min(1.0, max(0.0, onset))
            n_ignite = int(self.num_leds * (0.3 + 0.2 * onset_strength))
            indices = self._rng.choice(
                self.num_leds, size=n_ignite, replace=False,
            )

            spark_val = 0.7 + 0.3 * onset_strength
            self.sparkle[indices] = spark_val

            # Slower decay rates: 0.92-0.97 (was 0.85-0.95) — sparkles linger longer
            self._decay_rates[indices] = (
                0.92 + self._rng.random(n_ignite) * 0.05
            )

        # Per-LED decay toward 0 (frame-rate independent: rate^(dt*30))
        decay = self._decay_rates ** (dt * 30.0)
        self.sparkle *= decay

        # Subtle shimmer jitter between onsets (only when not silent)
        if not is_silent:
            jitter = self._rng.uniform(-0.01, 0.01, size=self.num_leds)
            self.sparkle = np.clip(self.sparkle + jitter, 0.0, self.sparkle)

        # Base brightness: envelope capped at 50%
        base = min(self._envelope * 1.0, 0.5)

        # Final per-LED brightness: base + sparkle fills the gap up to 1.0
        brightness = base + self.sparkle * (1.0 - base)
        brightness = np.clip(brightness, 0.0, 1.0)

        # Color: warm amber base, sparkle interpolates toward whiter
        base_r, base_g, base_b = 255.0, 180.0, 80.0
        white_r, white_g, white_b = 255.0, 240.0, 200.0

        lerp = self.sparkle  # 0-1 interpolation toward sparkle color
        r = base_r + (white_r - base_r) * lerp
        g = base_g + (white_g - base_g) * lerp
        b = base_b + (white_b - base_b) * lerp

        # Deadband snap-to-zero: below threshold, snap all to black
        DEADBAND = 0.08
        brightness = np.where(brightness < DEADBAND, 0.0, brightness)

        self._frame_buf[:, 0] = np.clip(r * brightness, 0, 255).astype(np.uint8)
        self._frame_buf[:, 1] = np.clip(g * brightness, 0, 255).astype(np.uint8)
        self._frame_buf[:, 2] = np.clip(b * brightness, 0, 255).astype(np.uint8)

        return self._frame_buf.copy()

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = float(self._energy)
            onset = float(self._onset_val)

        active = int(np.sum(self.sparkle > 0.05))
        return {
            'energy': f'{energy:.3f}',
            'onset': f'{onset:.3f}',
            'envelope': f'{self._envelope:.3f}',
            'active_sparkles': str(active),
            'cooldown': f'{self._cooldown_remaining:.3f}',
        }

    def get_source_values(self) -> dict:
        with self._lock:
            return {
                'energy': float(self._energy),
                'onset': float(self._onset_val),
            }
