"""
Energy Waterfall — scrolling RMS energy pulses with phosphor background.

Each frame pushes the current waveform RMS as brightness into LED 0.
The buffer scrolls naturally — short bursts create narrow bright pulses
traveling down the strip, sustained energy creates wide bright bands.

Background: each pulse deposits a tiny brightness residue on every pixel
it passes through. Each pixel's residue decays independently. Areas with
heavy pulse traffic glow warm; areas with no traffic go dark.

Color: full red, brightness-modulated.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, StickyFloorRMS


class EnergyWaterfallEffect(AudioReactiveEffect):
    """Scrolling RMS energy pulses with phosphor-decay background."""

    registry_name = 'energy_waterfall'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'RMS amplitude'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 # How much brightness each pulse deposits per pixel per frame.
                 # 0.008 = a full-brightness pulse adds ~0.8% per frame it sits on a pixel.
                 pulse_deposit: float = 0.008,
                 # Per-frame decay multiplier for background brightness.
                 # 0.993 at 30fps → ~4.5s half-life.
                 bg_decay: float = 0.985,
                 ):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.hop_length = 512

        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )

        # Sticky floor: constant energy decays over time.
        # up_mult=0.5 → floor adapts to new energy levels in ~2-4s
        # (vs default 0.1 which takes ~50s — too slow for the waterfall).
        self._sticky = StickyFloorRMS(
            fps=sample_rate / self.hop_length,
            up_mult=0.5,
        )

        # Peak-decay for normalization
        self._rms_peak = np.float32(1e-10)
        self._peak_decay = 0.9995

        # Shared state
        self._energy = np.float32(0.0)
        self._lock = threading.Lock()

        # Two half-buffers: pulses enter from both ends, travel to midpoint
        self._mid = num_leds // 2
        self._left_buf = np.zeros(self._mid, dtype=np.float32)   # LED 0 → mid
        self._right_buf = np.zeros(num_leds - self._mid, dtype=np.float32)  # LED N-1 → mid

        # Per-pixel background brightness (phosphor residue)
        self._bg = np.zeros(num_leds, dtype=np.float32)
        self._pulse_deposit = pulse_deposit
        self._bg_decay = bg_decay

        # Full red color
        self._full_red = np.array([255.0, 40.0, 0.0])

        # Output buffer
        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Energy Waterfall"

    @property
    def description(self):
        return "Scrolling RMS pulses with phosphor-decay background glow."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            rms = np.float32(np.sqrt(np.mean(frame ** 2)))

            # Peak-decay
            self._rms_peak = max(rms, self._rms_peak * self._peak_decay)
            pd_val = rms / self._rms_peak if self._rms_peak > 1e-10 else 0.0

            # Sticky floor
            sticky_val = self._sticky.update(frame)

            # Blend: sticky for dynamics, peak-decay as baseline
            output = sticky_val * 0.85 + pd_val * 0.15

            with self._lock:
                self._energy = np.float32(output)

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = float(self._energy)

        val = np.clip(energy, 0.0, 1.0)

        # Left half: inject at index 0, scroll toward midpoint (higher index)
        self._left_buf[1:] = self._left_buf[:-1]
        self._left_buf[0] = val

        # Right half: inject at index 0 (maps to LED N-1), scroll toward midpoint
        self._right_buf[1:] = self._right_buf[:-1]
        self._right_buf[0] = val

        # Combine into full strip: left half normal, right half reversed
        n = self.num_leds
        mid = self._mid
        wf_combined = np.empty(n, dtype=np.float32)
        wf_combined[:mid] = self._left_buf
        wf_combined[mid:] = self._right_buf[::-1]

        # Decay all background pixels
        self._bg *= self._bg_decay

        # Each pulse pixel deposits brightness proportional to its intensity
        self._bg += wf_combined * self._pulse_deposit

        # Clamp background
        np.clip(self._bg, 0.0, 0.4, out=self._bg)

        # Final brightness: max of pulse and background
        brightness = np.maximum(wf_combined, self._bg)

        # Map to color
        self._frame_buf[:] = (self._full_red * brightness[:, np.newaxis]).astype(np.uint8)

        return self._frame_buf.copy()

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = float(self._energy)

        return {
            'energy': f'{energy:.2f}',
            'bg_mean': f'{self._bg.mean():.3f}',
            'bg_max': f'{self._bg.max():.3f}',
        }
