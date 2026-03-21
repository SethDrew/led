"""
Energy Waterfall — scrolling RMS energy pulses.

Each frame pushes the current waveform RMS as brightness into LED 0.
The buffer scrolls naturally — short bursts create narrow bright pulses
traveling down the strip, sustained energy creates wide bright bands.
No beat detection or smoothing — raw waveform energy for crisp edges.

Color: full red, brightness-modulated.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, StickyFloorRMS


class EnergyWaterfallEffect(AudioReactiveEffect):
    """Scrolling RMS energy pulses — raw waveform amplitude waterfall."""

    registry_name = 'energy_waterfall'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'RMS amplitude (sticky floor)'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.hop_length = 512

        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )

        # Sticky floor normalization: constant energy decays over time.
        self._sticky = StickyFloorRMS(
            fps=sample_rate / self.hop_length,
        )

        # Shared state
        self._energy = np.float32(0.0)
        self._lock = threading.Lock()

        # Waterfall buffer
        self._wf_buffer = np.zeros((num_leds, 3), dtype=np.uint8)

        # Full red color
        self._full_red = np.array([255.0, 40.0, 0.0])

    @property
    def name(self):
        return "Energy Waterfall"

    @property
    def description(self):
        return "Scrolling RMS energy pulses — sticky floor adaptive normalization."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            val = self._sticky.update(frame)
            with self._lock:
                self._energy = np.float32(val)

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = float(self._energy)

        # Scroll buffer
        self._wf_buffer[1:] = self._wf_buffer[:-1]

        # New entry: sticky floor energy → brightness on full red
        wf_bright = np.clip(energy, 0.0, 1.0)
        self._wf_buffer[0] = (self._full_red * wf_bright).astype(np.uint8)

        return self._wf_buffer.copy()

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = float(self._energy)

        return {
            'energy': f'{energy:.2f}',
            'floor': f'{self._sticky._floor:.6f}',
        }
