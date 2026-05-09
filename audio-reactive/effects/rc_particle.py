"""
RC Particle — tilt-steered particle with tap-triggered explosions.

Two v1 senders:
  - Controller (first sender seen): tilt steers particle position,
    gyro magnitude drives color (still=warm, spin=cool).
  - Trigger (second sender seen): tap causes particle to explode
    into pot_particle-style shrapnel.

Senders auto-assign by arrival order. Power-cycle to swap roles.
"""

import math
import random
import numpy as np
import colorsys
from base import AudioReactiveEffect

AMAG_TAP_THRESH = 130


class RcParticleEffect(AudioReactiveEffect):

    registry_name = 'rc_particle'
    ref_pattern = 'ambient'
    ref_scope = 'beat'
    ref_input = 'v1 accel (tilt → position) + v1 tap (explosion)'
    ref_interactivity = 'sensor'

    TAP_COOLDOWN_S = 0.15

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.elapsed = 0.0

        # --- Particle state ---
        self.tilt = 0.0
        self.tilt_smooth = 0.0
        self.position = num_leds / 2.0
        self.gmag = 0.0
        self.hue = 0.0
        self.glow_width = 3.0
        self.tilt_alpha = 0.15
        self.hue_alpha = 0.05
        self.tilt_baseline = 0.0
        self.baseline_ready = False

        # --- Sender roles (auto-assigned) ---
        self.controller_mac = None
        self.trigger_mac = None

        # --- Tap / explosion ---
        self.tap_amag = 0
        self.last_tap_time = -1.0
        self.shrapnel = []

    def set_v1_data(self, data):
        mac = data.get('mac', '')

        # Auto-assign roles
        if self.controller_mac is None:
            self.controller_mac = mac
            print(f"[rc_particle] controller assigned: {mac}")
        elif mac != self.controller_mac and self.trigger_mac is None:
            self.trigger_mac = mac
            print(f"[rc_particle] trigger assigned: {mac}")

        if mac == self.controller_mac:
            ax = data.get('ax_mean', 0)
            az = data.get('az_mean', 0)
            raw_tilt = math.atan2(ax, az)

            if not self.baseline_ready:
                self.tilt_baseline = raw_tilt
                self.tilt_smooth = 0.0
                self.baseline_ready = True
            self.tilt_baseline += (raw_tilt - self.tilt_baseline) * 0.008
            self.tilt = raw_tilt - self.tilt_baseline

            self.gmag = data.get('gmag_mean', 0)

        elif mac == self.trigger_mac:
            amag = data.get('amag_max', 0)
            if amag > self.tap_amag:
                self.tap_amag = amag

    def process_audio(self, mono_chunk: np.ndarray):
        pass

    def _explode(self, pos):
        num_pieces = random.randint(8, 14)
        for _ in range(num_pieces):
            speed = random.uniform(15, 80)
            direction = random.choice([-1, 1])
            self.shrapnel.append({
                'pos': pos,
                'vel': speed * direction,
                'life': 1.0,
                'decay': random.uniform(1.5, 3.5),
                'hue': self.hue + random.uniform(-0.15, 0.15),
            })

    def render(self, dt: float) -> np.ndarray:
        self.elapsed += dt
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        # Smooth tilt → position
        self.tilt_smooth += (self.tilt - self.tilt_smooth) * self.tilt_alpha
        norm = max(min(self.tilt_smooth / (math.pi / 4), 1.0), -1.0)
        self.position = ((norm + 1.0) / 2.0) * (self.num_leds - 1)

        # Gyro → hue
        rotation = min(self.gmag / 120.0, 1.0)
        target_hue = rotation * 0.65
        self.hue += (target_hue - self.hue) * self.hue_alpha

        # Tap → explode
        if (self.tap_amag >= AMAG_TAP_THRESH
                and self.elapsed - self.last_tap_time > self.TAP_COOLDOWN_S):
            self._explode(self.position)
            self.last_tap_time = self.elapsed
            print(f"[rc_particle] TAP! amag={self.tap_amag} → explode at {self.position:.0f}")
        self.tap_amag = 0

        # Draw main particle
        r, g, b = colorsys.hls_to_rgb(self.hue, 0.5, 1.0)
        for i in range(self.num_leds):
            dist = abs(i - self.position)
            brightness = math.exp(-(dist ** 2) / (2 * self.glow_width ** 2))
            if brightness > 0.01:
                frame[i, 0] += r * brightness * 255
                frame[i, 1] += g * brightness * 255
                frame[i, 2] += b * brightness * 255

        # Draw shrapnel
        alive = []
        for s in self.shrapnel:
            s['pos'] += s['vel'] * dt
            s['vel'] *= 0.96
            s['life'] -= s['decay'] * dt

            if s['life'] <= 0 or s['pos'] < -5 or s['pos'] > self.num_leds + 5:
                continue

            alive.append(s)
            sr, sg, sb = colorsys.hls_to_rgb(s['hue'] % 1.0, 0.5, 1.0)
            glow = 0.8 + s['life'] * 1.2
            bright_scale = s['life'] * 255
            cutoff = glow * 3

            for i in range(self.num_leds):
                dist = abs(i - s['pos'])
                if dist > cutoff:
                    continue
                brightness = math.exp(-(dist ** 2) / (2 * glow ** 2))
                if brightness > 0.01:
                    frame[i, 0] += sr * brightness * bright_scale
                    frame[i, 1] += sg * brightness * bright_scale
                    frame[i, 2] += sb * brightness * bright_scale

        self.shrapnel = alive

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        return {
            'pos': round(self.position, 1),
            'shrapnel': len(self.shrapnel),
            'ctrl': (self.controller_mac or '?')[-5:],
            'trig': (self.trigger_mac or '?')[-5:],
        }
