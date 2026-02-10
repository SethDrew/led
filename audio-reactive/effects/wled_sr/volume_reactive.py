"""
WLED Sound Reactive: Volume Reactive effect.

Reimplements WLED's simplest audio-reactive mode:
  - Compute RMS volume from audio chunk
  - Apply exponential smoothing (16-sample EMA, matching WLED's getSample())
  - Apply squelch (noise gate)
  - Map smoothed volume to LED brightness

This is the baseline — if this already looks good, complex algorithms
need to beat it clearly to justify their existence.

WLED source: usermods/audioreactive/audio_reactive.cpp (getSample function)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import AudioReactiveEffect


class WLEDVolumeReactive(AudioReactiveEffect):
    """WLED's volume-reactive mode: RMS → brightness with smoothing."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # WLED parameters (from getSample())
        self.squelch = 10            # noise gate threshold (WLED default)
        self.sample_avg = 0.0        # smoothed sample (WLED: sampleAvg)
        self.sample_peak = False     # peak detected this frame
        self.mult_agc = 1.0          # AGC gain multiplier

        # AGC state (simplified from WLED's agcAvg)
        self.agc_gain = 1.0
        self.sample_max = 1.0

        # Rendering state
        self.brightness = 0.0
        self.peak_brightness = 0.0

        # WLED color: default is palette-based, we'll use warm white
        self.color = np.array([255, 180, 50], dtype=np.float64)  # warm

    @property
    def name(self):
        return "WLED Volume"

    def process_audio(self, mono_chunk: np.ndarray):
        # RMS volume (WLED computes sample magnitude differently but RMS is equivalent)
        rms = np.sqrt(np.mean(mono_chunk ** 2))

        # Scale to roughly match WLED's amplitude range (they use int16 internally)
        sample = rms * 32768.0

        # Squelch / noise gate
        if sample < self.squelch:
            sample = 0.0

        # Simple AGC: track max and normalize
        if sample > self.sample_max:
            self.sample_max = sample
        else:
            # Slow decay of max tracker
            self.sample_max = self.sample_max * 0.9995 + sample * 0.0005

        if self.sample_max > self.squelch:
            sample_adj = sample / self.sample_max * 255.0
        else:
            sample_adj = 0.0

        # WLED's 16-sample EMA: sampleAvg = (sampleAvg * 15 + sampleAdj) / 16
        self.sample_avg = (self.sample_avg * 15.0 + sample_adj) / 16.0

    def render(self, dt: float) -> np.ndarray:
        # Map smoothed volume (0-255) to brightness (0-1)
        vol_frac = min(self.sample_avg / 255.0, 1.0)

        # Apply gamma for perceived linearity (WLED uses gamma 2.8 internally)
        gamma_vol = vol_frac ** 2.8

        # Build frame: all LEDs same brightness
        color = (self.color * gamma_vol).astype(np.uint8)
        frame = np.tile(color, (self.num_leds, 1))
        return frame

    def get_diagnostics(self) -> dict:
        return {
            'vol': self.sample_avg,
            'max': self.sample_max,
            'agc': self.mult_agc,
        }
