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
                 bg_decay: float = 0.96,
                 # Breathing dead band: meeting points oscillate outward from center.
                 breathe_amplitude: int = 25,    # max LEDs each boundary moves from center
                 breathe_period: float = 60.0,   # seconds for full expand-hold-contract cycle
                 breathe_hold: float = 10.0,     # seconds to hold at widest point
                 breathe_fade: int = 4,          # pixels of soft fade at dead band edges
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

        # Center source buffers (scroll from center toward dead band boundaries)
        self._center_left = np.zeros(breathe_amplitude, dtype=np.float32)
        self._center_right = np.zeros(breathe_amplitude, dtype=np.float32)

        # Breathing dead band
        self._breathe_amplitude = breathe_amplitude
        self._breathe_period = breathe_period
        self._breathe_hold = breathe_hold
        self._breathe_fade = breathe_fade
        self._breathe_time = 0.0

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

        # Three-phase breathing: expand → hold → contract
        self._breathe_time += dt
        cycle_t = self._breathe_time % self._breathe_period
        move_time = (self._breathe_period - self._breathe_hold) / 2.0

        if cycle_t < move_time:
            # Expanding outward (smooth ease-in-out)
            t = cycle_t / move_time
            raw_offset = self._breathe_amplitude * 0.5 * (1.0 - np.cos(np.pi * t))
        elif cycle_t < move_time + self._breathe_hold:
            # Holding at maximum
            raw_offset = float(self._breathe_amplitude)
        else:
            # Contracting back (smooth ease-in-out)
            t = (cycle_t - move_time - self._breathe_hold) / move_time
            raw_offset = self._breathe_amplitude * 0.5 * (1.0 + np.cos(np.pi * t))

        offset = int(raw_offset)

        # Outer waterfall: scroll from ends toward center
        self._left_buf[1:] = self._left_buf[:-1]
        self._left_buf[0] = val
        self._right_buf[1:] = self._right_buf[:-1]
        self._right_buf[0] = val

        # Center source: always scroll so it's ready when dead band opens
        self._center_left[1:] = self._center_left[:-1]
        self._center_left[0] = val
        self._center_right[1:] = self._center_right[:-1]
        self._center_right[0] = val

        # Combine outer waterfall into full strip
        n = self.num_leds
        mid = self._mid
        wf_combined = np.empty(n, dtype=np.float32)
        wf_combined[:mid] = self._left_buf
        wf_combined[mid:] = self._right_buf[::-1]

        # Replace dead band region with center source
        dead_start = mid - offset
        dead_end = mid + offset
        if offset > 0:
            # Center-left: newest at center (mid-1), oldest at boundary (dead_start)
            wf_combined[dead_start:mid] = self._center_left[:offset][::-1]
            # Center-right: newest at center (mid), oldest at boundary (dead_end-1)
            wf_combined[mid:dead_end] = self._center_right[:offset]

        # Background phosphor
        self._bg *= self._bg_decay
        self._bg += wf_combined * self._pulse_deposit
        np.clip(self._bg, 0.0, 0.4, out=self._bg)

        # Final brightness: max of pulse and background
        brightness = np.maximum(wf_combined, self._bg)

        # Soft fade at boundaries between outer and center sources
        if offset > 0:
            fw = self._breathe_fade
            for i in range(fw):
                alpha = (i + 1) / (fw + 1)
                # Left boundary — outer side
                idx = dead_start - 1 - i
                if 0 <= idx:
                    brightness[idx] *= alpha
                # Left boundary — center side
                idx = dead_start + i
                if idx < mid:
                    brightness[idx] *= alpha
                # Right boundary — center side
                idx = dead_end - 1 - i
                if idx >= mid:
                    brightness[idx] *= alpha
                # Right boundary — outer side
                idx = dead_end + i
                if idx < n:
                    brightness[idx] *= alpha

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
