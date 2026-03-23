"""
Jellyfish — bioluminescent bell pulsations on the diamond sculpture.

The diamond's apex is the top of the jellyfish bell; left_tip and right_tip
are trailing tendrils. Each beat drives a contraction wavefront that sweeps
from apex downward through the bell, then bleeds into the tendrils as a
gentle trailing glow that fades before reaching the tips.

Potentiometer controls hue — the current hue enters at the apex and flows
downward over time, so continuously turning the pot paints a rainbow from
top to bottom through the sculpture.

Audio-reactive: tempo from OnsetTempoTracker, octave-folded to target
~1.5s period for a calm, deliberate rhythm.

Usage:
    python runner.py jellyfish --sculpture cob_diamond --no-leds
    python runner.py jellyfish --sculpture cob_diamond
"""

import math
import threading
import numpy as np
from base import AudioReactiveEffect
from topology import SculptureTopology
from signals import OverlapFrameAccumulator, OnsetTempoTracker


TARGET_PERIOD = 1.5  # desired seconds per bell cycle — calm jellyfish

# Phase boundaries
CONTRACTION_END = 0.35   # wavefront reaches rim
HOLD_END = 0.42          # brief peak brightness hold
# 0.42–1.0 = relaxation + tendril trailing (longer dark gap)

# Wavefront geometry
BELL_TOP = 1.0           # normalized height of apex
BELL_RIM = 0.35          # height where bell ends and tendrils begin
WAVEFRONT_SIGMA = 0.06   # Gaussian softness — tight edge for visible sweep on 72 LEDs
TENDRIL_DECAY = 4.0      # exponential decay rate for tendril trailing glow
PEAK_BRIGHTNESS = 0.65   # bioluminescent cap — never full white

# Ambient base
AMBIENT = 0.01           # barely-visible glow — jellyfish should breathe from darkness

# Hue trail: how many seconds of pot history are visible top-to-bottom
TRAIL_SECONDS = 2.5      # time for a color to flow from apex to tips
TRAIL_SATURATION = 0.65  # soft bioluminescent saturation (not full rainbow blast)

# Autonomous hue drift: slow color rotation when pot is not being turned
AUTO_HUE_CYCLE = 60.0    # seconds for full color wheel rotation


def octave_nearest(period, target):
    """Shift period by octaves (2x/0.5x) to land closest to target."""
    if period <= 0:
        return target
    log_ratio = math.log2(target / period)
    n = round(log_ratio)
    return period * (2 ** n)


def hsv_to_rgb(h, s, v):
    """Convert HSV (all 0-1) to RGB (0-1) float tuple."""
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    elif i == 1:
        return q, v, p
    elif i == 2:
        return p, v, t
    elif i == 3:
        return p, q, v
    elif i == 4:
        return t, p, v
    else:
        return v, p, q


def bell_brightness(height, phase):
    """Compute brightness (0-1) for an LED at given height and beat phase.

    Contraction (0–0.35): wavefront descends from apex to rim.
    Hold (0.35–0.50): full bell lit, slight pulsing.
    Relaxation (0.50–1.0): bell dims; tendril wavefront continues descending.
    """
    if phase < CONTRACTION_END:
        # Wavefront descends from apex (1.0) to rim (BELL_RIM)
        frac = phase / CONTRACTION_END
        wavefront_h = BELL_TOP - frac * (BELL_TOP - BELL_RIM)

        # Only the bell region (above rim) lights during contraction
        if height < BELL_RIM:
            return AMBIENT

        # Gaussian peak centered ON the wavefront — narrow bright band
        # LEDs well above the wavefront (already passed) fade back down
        dist = height - wavefront_h
        # Forward edge: bright ahead of wavefront (standard erf)
        forward = 0.5 * (1.0 + math.erf(dist / (WAVEFRONT_SIGMA * math.sqrt(2))))
        # Trailing fade: LEDs far above the wavefront dim back
        trail_fade = math.exp(-3.0 * max(0, dist) ** 2)
        # Blend: the wavefront is a bright band, not a permanent fill
        envelope = forward * (0.15 + 0.85 * trail_fade)

        return AMBIENT + (1.0 - AMBIENT) * envelope

    elif phase < HOLD_END:
        # Hold: brief peak, dimmer than before (0.7 peak instead of 0.9-1.0)
        hold_frac = (phase - CONTRACTION_END) / (HOLD_END - CONTRACTION_END)
        pulse = 0.4 + 0.1 * math.cos(hold_frac * math.pi)  # brief dim throb

        if height >= BELL_RIM:
            return AMBIENT + (1.0 - AMBIENT) * pulse
        else:
            # Tendril wavefront just starting to bleed below rim
            dist_below = BELL_RIM - height
            trail = math.exp(-TENDRIL_DECAY * dist_below) * pulse * 0.3
            return AMBIENT + trail

    else:
        # Relaxation: bell dims aggressively, tendril wavefront descends
        relax_frac = (phase - HOLD_END) / (1.0 - HOLD_END)  # 0-1

        # Aggressive power-curve dimming: reaches near-zero by 40% through relaxation
        # so the last 60% of the cycle is near-dark (breathing room before next pulse)
        dim_curve = max(0.0, 1.0 - relax_frac ** 0.3) ** 2.5

        if height >= BELL_RIM:
            # Bell fades smoothly -- top fades first (inverse of contraction)
            fade_wavefront = BELL_TOP - relax_frac * (BELL_TOP - BELL_RIM)
            dist = height - fade_wavefront
            fade_envelope = 0.5 * (1.0 - math.erf(dist / (WAVEFRONT_SIGMA * math.sqrt(2))))
            bell_dim = dim_curve * fade_envelope
            return AMBIENT + (1.0 - AMBIENT) * bell_dim
        else:
            # Tendril trailing glow: wavefront continues below rim
            tendril_wavefront = BELL_RIM - relax_frac * BELL_RIM
            dist_below = tendril_wavefront - height
            if dist_below > 0:
                trail = math.exp(-TENDRIL_DECAY * dist_below)
            else:
                trail = math.exp(-TENDRIL_DECAY * abs(dist_below) * 0.5)
            # Aggressive overall fade matching the bell
            return AMBIENT + 0.4 * trail * dim_curve


class JellyfishEffect(AudioReactiveEffect):
    """Bioluminescent jellyfish bell pulsations on the diamond sculpture."""

    registry_name = 'jellyfish'
    handles_topology = True
    ref_pattern = 'groove'
    ref_scope = 'beat'
    ref_input = 'tempo tracker'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 sculpture_id: str = 'cob_diamond'):
        super().__init__(num_leds, sample_rate)
        self.topo = SculptureTopology(sculpture_id)
        self.num_leds = self.topo.num_leds

        self.accum = OverlapFrameAccumulator()
        self.tracker = OnsetTempoTracker(sample_rate=sample_rate)

        # Beat phase accumulator
        self._beat_phase = 0.0
        self._period = TARGET_PERIOD

        self._lock = threading.Lock()
        self._snap_phase = 0.0
        self._snap_period = TARGET_PERIOD

        # Pre-compute normalized heights for each LED from y-coordinates
        ys = self.topo.coords[:, 1]
        y_min, y_max = ys.min(), ys.max()
        span = y_max - y_min if y_max > y_min else 1.0
        self._heights = (ys - y_min) / span  # 0 = lowest, 1 = highest

        # Hue trail: ring buffer of hue values sampled each frame.
        # Apex reads the newest entry, tips read the oldest.
        self._trail_len = int(TRAIL_SECONDS * 30)  # ~30fps
        self._hue_trail = np.zeros(self._trail_len, dtype=np.float64)
        self._trail_head = 0  # index of newest entry

        # Pot state: hue 0-1 mapped from pot 0-1023
        self._pot_hue = 0.75  # default: start in the blue-magenta range
        self._pot_active = False  # True when pot has been turned at least once
        self._auto_hue_phase = 0.75  # autonomous drift starting hue
        self._render_time = 0.0  # accumulated render time for auto-drift

    @property
    def name(self):
        return "Jellyfish"

    @property
    def description(self):
        return "Bioluminescent bell pulsations — pot paints rainbow top to bottom."

    def set_pot_value(self, raw):
        """Map pot 0-1023 to hue 0-1 (full color wheel)."""
        self._pot_hue = raw / 1023.0
        self._pot_active = True

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self.tracker.feed_frame(frame)

            raw_period = self.tracker.estimated_period
            if raw_period > 0 and self.tracker.confidence > 0.3:
                period = octave_nearest(raw_period, TARGET_PERIOD)
            else:
                period = TARGET_PERIOD

            dt_step = self.tracker.rms_dt
            self._beat_phase += dt_step / period
            if self._beat_phase >= 1.0:
                self._beat_phase -= 1.0

            with self._lock:
                self._snap_phase = self._beat_phase
                self._snap_period = period

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            phase = self._snap_phase

        # Determine current hue: pot if active, otherwise slow autonomous drift
        self._render_time += dt
        if self._pot_active:
            current_hue = self._pot_hue
        else:
            self._auto_hue_phase = (self._render_time / AUTO_HUE_CYCLE) % 1.0
            current_hue = self._auto_hue_phase

        # Advance hue trail: push current hue into the ring buffer
        self._trail_head = (self._trail_head + 1) % self._trail_len
        self._hue_trail[self._trail_head] = current_hue

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        for i in range(self.num_leds):
            b = bell_brightness(self._heights[i], phase)
            b = max(0.0, min(PEAK_BRIGHTNESS, b))
            if b > 0:
                # Look up hue from trail: height=1 (apex) → newest,
                # height=0 (tips) → oldest
                age = (1.0 - self._heights[i]) * (self._trail_len - 1)
                trail_idx = (self._trail_head - int(age)) % self._trail_len
                hue = self._hue_trail[trail_idx]

                r, g, bl = hsv_to_rgb(hue, TRAIL_SATURATION, b)
                frame[i, 0] = min(int(r * 255), 255)
                frame[i, 1] = min(int(g * 255), 255)
                frame[i, 2] = min(int(bl * 255), 255)
        return frame

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        with self._lock:
            phase = self._snap_phase
            period = self._snap_period

        if phase < CONTRACTION_END:
            ph_label = 'contract'
        elif phase < HOLD_END:
            ph_label = 'hold'
        else:
            ph_label = 'relax'

        raw = self.tracker.estimated_period
        return {
            'phase': ph_label,
            'beat_phase': f'{phase:.2f}',
            'hue': f'{int(self._pot_hue * 360)}°',
            'bpm': f'{self.tracker.bpm:.0f}',
            'period': f'{period:.2f}s',
            'raw': f'{raw:.2f}s' if raw > 0 else '-',
            'conf': f'{self.tracker.confidence:.2f}',
        }
