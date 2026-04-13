"""
Firefly Synchronization — percussive flash response.

Fireflies scattered across the strip. Each one probabilistically flashes
in response to detected percussion (AbsIntegral signal). Harder hits
recruit more fireflies. No tempo tracking, no coupling — just raw
percussive response as a foundation to build on.

Uses EMA normalization on the raw absint signal (not built-in peak-decay).
Signal = raw / ema(raw), centered around 1.0 during steady sections.
Transients spike well above 1.0 (section boundary contrast preserved).

Probability curve: squared, anchored at 1.0 (the running mean).
  Below 0.8: essentially silent (1% floor).
  1.0 (mean): ~7% — occasional single firefly on typical beats.
  1.5 (moderate transient): ~40% — cluster response.
  2.0+ (hard transient/drop): 70%+ — full strip recruitment.
"""

import numpy as np
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, AbsIntegral


class FireflySyncEffect(AudioReactiveEffect):
    """Fireflies that flash in response to percussion."""

    registry_name = 'firefly_sync'
    ref_pattern = 'groove'
    ref_scope = 'phrase'
    ref_input = 'absint'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 n_fireflies: int = 28):
        super().__init__(num_leds, sample_rate)

        self.n_fireflies = n_fireflies

        # Audio processing
        self.accum = OverlapFrameAccumulator()
        self.absint = AbsIntegral(sample_rate=sample_rate)

        # EMA normalization of raw absint (replaces built-in peak-decay)
        # 20s time constant — tracks section-level energy context
        self._ema_tc = 20.0
        self._absint_ema = 0.0
        self._absint_normed = 0.0
        self._warmup_frames = 0
        self._warmup_target = int(2.0 / (2048 / sample_rate))  # ~2s of frames

        # Firefly positions: spread evenly with jitter
        base_positions = np.linspace(0, num_leds - 1, n_fireflies)
        jitter = np.random.uniform(-num_leds / n_fireflies * 0.3,
                                    num_leds / n_fireflies * 0.3,
                                    n_fireflies)
        self.positions = np.clip(base_positions + jitter, 0, num_leds - 1)

        # Flash state: brightness per firefly (decays after flash)
        self.flash_brightness = np.zeros(n_fireflies, dtype=np.float32)

        # Position drift: slow random walk (LEDs/second)
        self.drift_speed = 3.0

        # Spatial glow radius — sigma 0.65 gives ~1-2 LED tight points
        self.glow_radius = 0.65

        # Warm amber palette with per-firefly variation
        self.colors = np.zeros((n_fireflies, 3), dtype=np.float32)
        for i in range(n_fireflies):
            warmth = np.random.uniform(0.85, 1.0)
            self.colors[i] = [255 * warmth, 180 * warmth, 60 * warmth]

    @property
    def name(self):
        return "Firefly Sync"

    @property
    def description(self):
        return "Fireflies flash on percussion — harder hits recruit more."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self.absint.update(frame)
            raw = self.absint.raw

            # During warmup (~2s), use fast TC to seed the EMA quickly
            if self._warmup_frames < self._warmup_target:
                self._warmup_frames += 1
                tc = 0.5  # fast warmup TC
            else:
                tc = self._ema_tc

            alpha = 1.0 - np.exp(-self.absint.dt / tc)
            self._absint_ema += alpha * (raw - self._absint_ema)

            # Normalized: raw / ema. >1.0 on transients, <1.0 in lulls.
            if self._absint_ema > 1e-10:
                self._absint_normed = raw / self._absint_ema
            else:
                self._absint_normed = 0.0

    def render(self, dt: float) -> np.ndarray:
        absint = self._absint_normed

        # Probability curve: squared, anchored at the running mean (1.0).
        # Floor at 0.8 — below-mean energy is essentially silent.
        # Coefficient 3.0 lets hard transients (2.0+) saturate.
        absint_clean = max(0.0, absint - 0.8)
        prob = min(1.0, 0.01 + 3.0 * (absint_clean / 1.2) ** 2)

        # Each firefly independently rolls against the probability
        rolls = np.random.random(self.n_fireflies)
        triggered = rolls < prob

        # Flash brightness: scales with how far above mean
        brightness = min(1.0, 0.3 + 0.5 * absint)
        self.flash_brightness[triggered] = np.maximum(
            self.flash_brightness[triggered], brightness)

        # Drift positions: slow random walk
        self.positions += np.random.normal(0, self.drift_speed,
                                           self.n_fireflies) * dt
        self.positions = np.clip(self.positions, 0, self.num_leds - 1)

        # Decay flashes — 250ms time constant
        decay = np.exp(-dt / 0.25)
        self.flash_brightness *= decay

        # Render LED frame
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)
        frame[:] = [4, 2, 1]  # dim warm base (embers)

        led_indices = np.arange(self.num_leds)
        for i in range(self.n_fireflies):
            if self.flash_brightness[i] < 0.01:
                continue
            pos = self.positions[i]
            b = self.flash_brightness[i]
            color = self.colors[i] * b
            distances = np.abs(led_indices - pos)
            falloff = np.exp(-0.5 * (distances / self.glow_radius) ** 2)
            frame[:, 0] += color[0] * falloff
            frame[:, 1] += color[1] * falloff
            frame[:, 2] += color[2] * falloff

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        absint = self._absint_normed
        absint_clean = max(0.0, absint - 0.8)
        prob = min(1.0, 0.01 + 3.0 * (absint_clean / 1.2) ** 2)
        active = int(np.sum(self.flash_brightness > 0.05))
        return {
            'absint': f'{absint:.2f}',
            'ema': f'{self._absint_ema:.4f}',
            'prob': f'{prob:.1%}',
            'active': f'{active}/{self.n_fireflies}',
        }

    source_features = [
        {'id': 'absint', 'label': 'AbsIntegral', 'color': '#FF6600'},
        {'id': 'flash_prob', 'label': 'Flash Probability', 'color': '#FFAA00'},
    ]

    def get_source_values(self) -> dict:
        absint = self._absint_normed
        absint_clean = max(0.0, absint - 0.8)
        prob = min(1.0, 0.01 + 3.0 * (absint_clean / 1.2) ** 2)
        return {
            'absint': float(absint),
            'flash_prob': float(prob),
        }
