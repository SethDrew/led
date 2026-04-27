"""
Worley Collision — bass-driven Voronoi boundary flares.

Seeds spread evenly across the strip, connected by springs to rest positions.
Bass kicks push seeds outward from center; springs pull them back.
Brightness comes from F2-F1 (boundary proximity): dark inside cells,
bright where cell territories collide. OKLCH rainbow colors per cell.

Visual: dark strip with dim boundary lines. Kick → seeds compress outward →
boundary lines rush toward edges and flare bright. Decay → boundaries
drift back to rest. Sustained bass keeps boundaries compressed and lit.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from color.oklch import RAINBOW_LUT


# ── Worley Collision effect ──────────────────────────────────────────

class WorleyCollisionEffect(AudioReactiveEffect):
    """Bass-driven Voronoi boundary collisions with spring physics."""

    registry_name = 'worley_collision'
    ref_pattern = 'accent'
    ref_scope = 'beat'
    ref_input = 'bass flux 20-250Hz'

    source_features = [
        {'id': 'bass_flux', 'label': 'Bass Flux', 'color': '#FF4400'},
    ]

    # Physics tuning
    NUM_SEEDS = 5
    SPRING_K = 12.0             # spring stiffness
    DAMPING = 7.0               # ~critically damped (ζ ≈ 1.01)
    IMPULSE_STRENGTH = 80.0     # px/s per unit flux → ~8-10px peak displacement

    # Visual tuning
    BOUNDARY_WIDTH = 5.0        # exponential falloff width for boundary lines
    AMBIENT_BRIGHTNESS = 0.12   # dim boundary glow with no audio
    CELL_DIM = 0.03             # barely visible cell interior

    # Brightness envelope
    ATTACK_RATE = 20.0          # instant attack (per second, multiplicative)
    DECAY_RATE = 0.88           # per-frame at 30fps → ~200ms half-life

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.center = num_leds / 2.0

        # Evenly spaced rest positions
        spacing = num_leds / (self.NUM_SEEDS + 1)
        self.rest_positions = np.array(
            [spacing * (i + 1) for i in range(self.NUM_SEEDS)],
            dtype=np.float64,
        )
        self.positions = self.rest_positions.copy()
        self.velocities = np.zeros(self.NUM_SEEDS, dtype=np.float64)

        # Rainbow hues spread across seeds
        self.hues = np.linspace(0, 255, self.NUM_SEEDS, endpoint=False).astype(np.uint8)

        # Precompute LED positions
        self._led_pos = np.arange(num_leds, dtype=np.float64) + 0.5

        # Brightness envelope (0-1, driven by kicks)
        self._brightness_env = 0.0

        # ── Bass detection (from bass_pulse pattern) ──
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        self.bass_mask = (self.freq_bins >= 20) & (self.freq_bins <= 250)
        self.audio_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.audio_buf_pos = 0
        self.prev_bass_spec = None
        self._flux_peak = 1e-10
        self._peak_decay = 0.997
        self._threshold = 0.55
        self._cooldown = 0.15
        self._last_beat_time = -1.0
        self._time_acc = 0.0
        self._pending_impulse = 0.0
        self._beat_count = 0
        self._lock = threading.Lock()

    @property
    def name(self):
        return "Worley Collision"

    @property
    def description(self):
        return "Bass kicks push Voronoi seeds outward from center; boundaries flare on collision."

    # ── Audio processing ─────────────────────────────────────────────

    def process_audio(self, mono_chunk: np.ndarray):
        n = len(mono_chunk)
        space = self.n_fft - self.audio_buf_pos
        if n < space:
            self.audio_buf[self.audio_buf_pos:self.audio_buf_pos + n] = mono_chunk
            self.audio_buf_pos += n
            return

        self.audio_buf[self.audio_buf_pos:] = mono_chunk[:space]
        self._process_frame(self.audio_buf.copy())

        leftover = n - space
        self.audio_buf[:leftover] = mono_chunk[space:]
        self.audio_buf_pos = leftover

    def _process_frame(self, frame):
        windowed = frame * self.window
        spec = np.abs(np.fft.rfft(windowed))
        bass_spec = spec[self.bass_mask]

        if self.prev_bass_spec is not None:
            diff = bass_spec - self.prev_bass_spec
            flux = np.sum(np.maximum(diff, 0))

            self._flux_peak = max(flux, self._flux_peak * self._peak_decay)
            normalized = flux / self._flux_peak if self._flux_peak > 0 else 0.0

            self._time_acc += self.n_fft / self.sample_rate
            time_since_beat = self._time_acc - self._last_beat_time

            with self._lock:
                if normalized > self._threshold and time_since_beat > self._cooldown:
                    self._pending_impulse = max(self._pending_impulse, normalized)
                    self._last_beat_time = self._time_acc
                    self._beat_count += 1

        self.prev_bass_spec = bass_spec.copy()

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, dt: float) -> np.ndarray:
        # Consume beat impulse
        with self._lock:
            impulse = self._pending_impulse
            self._pending_impulse = 0.0

        # ── Physics: spring-damper with outward kick impulse ──

        if impulse > 0:
            directions = np.sign(self.positions - self.center)
            at_center = directions == 0
            directions[at_center] = np.random.choice([-1, 1], size=np.sum(at_center))
            self.velocities += directions * impulse * self.IMPULSE_STRENGTH

            # Flash brightness envelope
            self._brightness_env = max(self._brightness_env, impulse)

        # Spring-damper: a = -k*x - c*v
        displacement = self.positions - self.rest_positions
        acceleration = -self.SPRING_K * displacement - self.DAMPING * self.velocities
        self.velocities += acceleration * dt
        self.positions += self.velocities * dt

        # Wall bounce at strip edges
        for i in range(self.NUM_SEEDS):
            if self.positions[i] < 0.5:
                self.positions[i] = 0.5
                self.velocities[i] *= -0.3
            elif self.positions[i] > self.num_leds - 0.5:
                self.positions[i] = self.num_leds - 0.5
                self.velocities[i] *= -0.3

        # Decay brightness envelope
        self._brightness_env *= self.DECAY_RATE ** (dt * 30)

        # ── Voronoi distance field ──

        distances = np.abs(self._led_pos[:, np.newaxis] - self.positions[np.newaxis, :])

        # F1 (nearest) and F2 (second nearest)
        idx_sorted = np.argpartition(distances, 2, axis=1)[:, :2]
        rows = np.arange(self.num_leds)
        d0 = distances[rows, idx_sorted[:, 0]]
        d1 = distances[rows, idx_sorted[:, 1]]
        f1 = np.minimum(d0, d1)
        f2 = np.maximum(d0, d1)
        nearest = np.where(d0 <= d1, idx_sorted[:, 0], idx_sorted[:, 1])

        # Boundary proximity → brightness via exponential falloff
        boundary = f2 - f1
        boundary_brightness = np.exp(-boundary / self.BOUNDARY_WIDTH)

        # Combine: ambient floor + envelope-driven brightness
        brightness = (self.AMBIENT_BRIGHTNESS +
                      (1.0 - self.AMBIENT_BRIGHTNESS) * self._brightness_env
                      ) * boundary_brightness

        # Faint cell interior glow (always on)
        brightness = np.maximum(brightness, self.CELL_DIM)

        # Color from OKLCH rainbow
        base_colors = RAINBOW_LUT[self.hues[nearest]].astype(np.float32)
        frame = (base_colors * brightness[:, np.newaxis]).clip(0, 255).astype(np.uint8)

        return frame

    def get_source_values(self) -> dict:
        return {'bass_flux': float(self._brightness_env)}

    def get_diagnostics(self) -> dict:
        max_disp = np.max(np.abs(self.positions - self.rest_positions))
        return {
            'beats': self._beat_count,
            'envelope': f'{self._brightness_env:.2f}',
            'max_disp': f'{max_disp:.1f}px',
        }
