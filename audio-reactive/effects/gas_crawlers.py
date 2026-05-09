"""
Gas Crawlers — small white particles drifting lazily, agitated by pot rotation.

Idle: soft white crawlers at ~0.1 LED/s, very slow direction changes.
Spinning the pot stirs them up: speed increases to 5 LED/s, colors emerge,
motion becomes erratic. Agitation decays back to calm when pot stops.
Tilt provides gravity. Audio RMS adds gentle brightness pulse.
"""

import math
import random
import numpy as np
import colorsys
from base import AudioReactiveEffect


class GasCrawlersEffect(AudioReactiveEffect):

    registry_name = 'gas_crawlers'
    ref_pattern = 'ambient'
    ref_scope = 'phrase'
    ref_input = 'pot rotation (agitation) + accel (gravity)'
    ref_interactivity = 'sensor'

    NUM_CRAWLERS = 15

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)
        self.rms = 0.0
        self.rms_smooth = 0.0
        self.pot_raw = 0.0
        self.prev_pot = 0.0
        self.agitation = 0.0
        self.color_agitation = 0.0
        self.accel_x = 0.0
        self.ax_baseline = 0.0
        self.baseline_ready = False
        self.crawlers = []
        for _ in range(self.NUM_CRAWLERS):
            self.crawlers.append({
                'pos': random.random() * (num_leds - 1),
                'vel': random.gauss(0, 1.0),
                'hue': random.random(),
                'jitter_timer': 0.0,
            })

    def set_pot_value(self, raw):
        self.prev_pot = self.pot_raw
        self.pot_raw = float(raw)
        delta = abs(self.pot_raw - self.prev_pot)
        if delta > 8:
            kick = min((delta - 8) / 20.0, 1.0)
            self.agitation = min(self.agitation + kick, 1.0)
            for c in self.crawlers:
                c['vel'] += random.gauss(0, kick * 8.0)
                c['jitter_timer'] = min(c['jitter_timer'], 0.1)

    def set_imu_data(self, data):
        raw_ax = data.get('ax', 0) / 16384.0
        if not self.baseline_ready:
            self.ax_baseline = raw_ax
            self.baseline_ready = True
        self.ax_baseline += (raw_ax - self.ax_baseline) * 0.008
        self.accel_x = raw_ax - self.ax_baseline

    def process_audio(self, mono_chunk: np.ndarray):
        self.rms = float(np.sqrt(np.mean(mono_chunk ** 2)))

    def render(self, dt: float) -> np.ndarray:
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        # Agitation decays toward zero (fast decay)
        self.agitation *= 0.97
        a = self.agitation
        # Color ramps up fast, fades slow
        alpha = 0.3 if a > self.color_agitation else 0.08
        self.color_agitation += (a - self.color_agitation) * alpha
        ca = self.color_agitation

        target_speed = 2.5 + a * 13.0
        # Jitter frequency: very rare idle (~every 20s), frequent when stirred (~every 0.5s)
        jitter_rate = 0.3 + a * 2.0

        gravity = -self.accel_x * 60.0

        brightness_boost = 1.0

        for c in self.crawlers:
            c['jitter_timer'] -= dt
            if c['jitter_timer'] <= 0:
                nudge = random.gauss(0, target_speed * 0.5)
                c['vel'] += nudge
                # Ensure minimum drift speed
                if abs(c['vel']) < target_speed * 0.3:
                    c['vel'] = math.copysign(target_speed * 0.5, c['vel'] if c['vel'] != 0 else nudge)
                c['vel'] = max(-target_speed, min(target_speed, c['vel']))
                c['jitter_timer'] = max(0.3, random.expovariate(jitter_rate))

            c['vel'] += gravity * dt
            c['vel'] *= 0.998
            c['pos'] += c['vel'] * dt

            if c['pos'] < 0:
                c['pos'] += self.num_leds
            elif c['pos'] >= self.num_leds:
                c['pos'] -= self.num_leds

            r, g, b = colorsys.hls_to_rgb(c['hue'], 0.5, ca)
            white_mix = 1.0 - ca
            r = r * ca + white_mix
            g = g * ca + white_mix
            b = b * ca + white_mix

            glow = 0.6 + a * 0.8

            for i in range(self.num_leds):
                dist = abs(i - c['pos'])
                dist = min(dist, self.num_leds - dist)
                brightness = math.exp(-(dist ** 2) / (2 * glow ** 2))
                if brightness > 0.01:
                    b_scaled = brightness * brightness_boost * 200
                    frame[i, 0] += r * b_scaled
                    frame[i, 1] += g * b_scaled
                    frame[i, 2] += b * b_scaled

        return np.clip(frame, 0, 255).astype(np.uint8)
