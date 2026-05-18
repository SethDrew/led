"""
Tilt Pendulum — spring particle with pot-driven oscillation.

Pot sweeps the rest position across the strip. Tilt adds gravity force
like tilt_gravity. Audio RMS from BlackHole pulses the particle width.
"""

import math
import numpy as np
import colorsys
from base import AudioReactiveEffect
from inputs import pot_position, accel_shake


class TiltPendulumEffect(AudioReactiveEffect):

    registry_name = 'tilt_pendulum'
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'pot (oscillation) + accel (gravity) + RMS (particle width)'
    ref_interactivity = 'hybrid'
    ref_inputs_required = ['pot_position', 'accel_shake', 'audio']
    input_roles = {
        'pot_position': 'sweeps the spring rest position across the strip',
        'accel_shake': 'x-axis motion adds gravity force to the particle',
        'audio': 'RMS modulates the particle glow width',
    }

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.pos = num_leds / 2.0
        self.vel = 0.0
        self.accel_x = 0.0
        self.rms = 0.0
        self.rms_smooth = 0.0
        self.pot_norm = 0.5
        self.base_glow = 1.5
        self._pot_state = {'smoothed': 512.0}
        self._ax_state = {}

    def set_pot_value(self, raw):
        # No smoothing — spring physics absorbs jitter downstream.
        self.pot_norm = pot_position(raw, self._pot_state, alpha=1.0, deadzone_raw=0.0)

    def set_imu_data(self, data):
        shake = accel_shake((data.get('ax', 0) / 16384.0,
                             data.get('ay', 0) / 16384.0,
                             data.get('az', 0) / 16384.0),
                            self._ax_state)
        self.accel_x = shake[0]

    def process_audio(self, mono_chunk: np.ndarray):
        self.rms = float(np.sqrt(np.mean(mono_chunk ** 2)))

    def render(self, dt: float) -> np.ndarray:
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        # Pot -> rest position across strip
        rest = self.pot_norm * (self.num_leds - 1)

        # Spring toward pot rest position
        stiffness = 80.0
        damping_coeff = 2.0 * math.sqrt(stiffness) * 0.2
        displacement = self.pos - rest
        spring_force = -stiffness * displacement * dt
        damping_force = -damping_coeff * self.vel * dt

        # Tilt -> gravity force (same as tilt_gravity)
        gravity = -self.accel_x * 400.0

        self.vel += spring_force + damping_force + gravity * dt
        self.pos += self.vel * dt

        if self.pos < 0:
            self.pos = 0; self.vel *= -0.2
        elif self.pos > self.num_leds - 1:
            self.pos = self.num_leds - 1; self.vel *= -0.2

        # RMS -> glow width
        self.rms_smooth += (self.rms - self.rms_smooth) * 0.3
        glow = self.base_glow + self.rms_smooth * 60.0

        # Color: speed -> hue
        speed = min(abs(self.vel) / 40.0, 1.0)
        hue = 0.08 + speed * 0.5
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)

        for i in range(self.num_leds):
            dist = abs(i - self.pos)
            brightness = math.exp(-(dist ** 2) / (2 * glow ** 2))
            if brightness > 0.01:
                frame[i, 0] += r * brightness * 255
                frame[i, 1] += g * brightness * 255
                frame[i, 2] += b * brightness * 255

        return np.clip(frame, 0, 255).astype(np.uint8)
