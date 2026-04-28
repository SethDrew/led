"""
Energy Waterfall 3-Band — three independent waterfalls, max-merged into one stream.

Each band (bass/mids/treble) has its own dual-scroll waterfall and phosphor
background, driven by per-band EMA-ratio normalized RMS. Per pixel, the final
brightness is the max across the three bands — bass no longer dominates the
brightness budget, and mids/treble pulses cut through cleanly when they're
the loudest band locally. Color is single full-red (same palette as basic
energy_waterfall); the per-band split lives entirely in the signal layer.

A pulse-emission overlay rides on top of the continuous waterfall: per-band
absint detects transients, max-of-three drives the unified pulse, and a
brightness boost is injected on the dominant band so it wins the max
locally.

See library/arch-impl/axis3-audio-features/PER_BAND_NORMALIZATION.md.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import (
    OverlapFrameAccumulator,
    PerBandEMANormalize,
    PerBandAbsIntegral,
    PulseDriver,
)


BANDS = ('bass', 'mids', 'treble')
BAND_RANGES = {
    'bass':   (20, 250),
    'mids':   (250, 2000),
    'treble': (2000, 8000),
}


class EnergyWaterfall3BandEffect(AudioReactiveEffect):
    """Three per-band waterfalls, color-stacked, with pulse-emission overlay."""

    registry_name = 'energy_waterfall_3band'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'per-band EMA-normalized RMS + per-band absint pulses'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 pulse_deposit: float = 0.008,
                 bg_decay: float = 0.96,
                 breathe_amplitude: int = 25,
                 breathe_period: float = 60.0,
                 breathe_hold: float = 10.0,
                 # Per-band normalization
                 ema_tc: float = 30.0,
                 noise_floor_rms: float = 0.0,
                 # Pulse overlay
                 pulse_threshold: float = 0.5,
                 pulse_cooldown_sec: float = 0.1,
                 pulse_decay: float = 0.5,
                 ):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.hop_length = 512
        self.window = np.hanning(self.n_fft).astype(np.float32)

        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )

        # FFT bin masks per band
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        self._mask_array = np.stack(
            [(freq_bins >= BAND_RANGES[b][0]) & (freq_bins < BAND_RANGES[b][1])
             for b in BANDS], axis=0)

        fps = sample_rate / self.hop_length

        # Per-band signal pipeline
        self._normalizer = PerBandEMANormalize(
            num_bands=len(BANDS), fps=fps,
            ema_tc=ema_tc, noise_floor_rms=noise_floor_rms,
        )
        self._absint = PerBandAbsIntegral(
            num_bands=len(BANDS), fps=fps,
        )
        self._pulse = PulseDriver(
            threshold=pulse_threshold, cooldown_sec=pulse_cooldown_sec,
        )

        self._pulse_decay = float(pulse_decay)

        # Audio-thread time accumulator (for PulseDriver cooldown)
        self._hop_dt = self.hop_length / sample_rate
        self._audio_time = 0.0

        # Shared state across audio + render threads
        self._vals = np.zeros(len(BANDS), dtype=np.float32)
        # Pulse strength per band, drained each render
        self._pending_pulse = np.zeros(len(BANDS), dtype=np.float32)
        self._lock = threading.Lock()

        # Per-band dual-scroll buffers
        self._mid = num_leds // 2
        self._left_bufs = {b: np.zeros(self._mid, dtype=np.float32) for b in BANDS}
        self._right_bufs = {b: np.zeros(num_leds - self._mid, dtype=np.float32) for b in BANDS}
        self._center_left = {b: np.zeros(breathe_amplitude, dtype=np.float32) for b in BANDS}
        self._center_right = {b: np.zeros(breathe_amplitude, dtype=np.float32) for b in BANDS}

        # Per-band phosphor residue
        self._bgs = {b: np.zeros(num_leds, dtype=np.float32) for b in BANDS}
        self._pulse_deposit = pulse_deposit
        self._bg_decay = bg_decay

        # Per-band pulse boost (decays each render frame, max-injected at source)
        self._pulse_boost = np.zeros(len(BANDS), dtype=np.float32)

        # Breathing dead band
        self._breathe_amplitude = breathe_amplitude
        self._breathe_period = breathe_period
        self._breathe_hold = breathe_hold
        self._breathe_time = 0.0

        # Single full-red palette (matches basic energy_waterfall)
        self._full_red = np.array([255.0, 40.0, 0.0])

        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

        self._pulse_count = 0

    @property
    def name(self):
        return "Energy Waterfall 3-Band"

    @property
    def description(self):
        return "Per-band waterfalls max-merged into a single red stream."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            spectrum = np.abs(np.fft.rfft(frame * self.window)).astype(np.float32)
            power = spectrum ** 2

            # Per-band RMS via vectorized mask multiply
            band_rms = np.sqrt(
                (self._mask_array * power[np.newaxis, :]).sum(axis=1)
                / np.maximum(self._mask_array.sum(axis=1), 1)
            ).astype(np.float32)

            normalized = self._normalizer.update(band_rms)
            absint = self._absint.update(normalized)

            self._audio_time += self._hop_dt
            fired = self._pulse.update(absint, self._audio_time)

            with self._lock:
                self._vals = normalized.copy()
                if fired:
                    self._pending_pulse[self._pulse.dominant] = max(
                        self._pending_pulse[self._pulse.dominant],
                        self._pulse.value,
                    )
                    self._pulse_count += 1

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            vals = self._vals.copy()
            pending = self._pending_pulse.copy()
            self._pending_pulse[:] = 0.0

        # Decay pulse boost, then absorb pending pulses
        self._pulse_boost *= self._pulse_decay
        np.maximum(self._pulse_boost, pending, out=self._pulse_boost)

        # Three-phase breathing: expand → hold → contract
        self._breathe_time += dt
        cycle_t = self._breathe_time % self._breathe_period
        move_time = (self._breathe_period - self._breathe_hold) / 2.0

        if cycle_t < move_time:
            t = cycle_t / move_time
            raw_offset = self._breathe_amplitude * 0.5 * (1.0 - np.cos(np.pi * t))
        elif cycle_t < move_time + self._breathe_hold:
            raw_offset = float(self._breathe_amplitude)
        else:
            t = (cycle_t - move_time - self._breathe_hold) / move_time
            raw_offset = self._breathe_amplitude * 0.5 * (1.0 + np.cos(np.pi * t))

        offset = max(int(raw_offset), 5)
        n = self.num_leds
        mid = self._mid
        dead_start = mid - offset
        dead_end = mid + offset

        band_brightness = np.zeros((len(BANDS), n), dtype=np.float32)

        for i, b in enumerate(BANDS):
            # Continuous brightness max'd with pulse boost at injection point
            inject_val = float(max(np.clip(vals[i], 0.0, 1.0),
                                   self._pulse_boost[i]))

            # Scroll outer + center buffers, inject newest sample at index 0
            self._left_bufs[b][1:] = self._left_bufs[b][:-1]
            self._left_bufs[b][0] = inject_val
            self._right_bufs[b][1:] = self._right_bufs[b][:-1]
            self._right_bufs[b][0] = inject_val
            self._center_left[b][1:] = self._center_left[b][:-1]
            self._center_left[b][0] = inject_val
            self._center_right[b][1:] = self._center_right[b][:-1]
            self._center_right[b][0] = inject_val

            # Combine outer waterfall into full strip
            wf = np.empty(n, dtype=np.float32)
            wf[:mid] = self._left_bufs[b]
            wf[mid:] = self._right_bufs[b][::-1]

            # Replace dead band with center source (with 1-px overlap blend)
            outer_left = wf[dead_start]
            outer_right = wf[dead_end - 1]
            wf[dead_start:mid] = self._center_left[b][:offset][::-1]
            wf[mid:dead_end] = self._center_right[b][:offset]
            wf[dead_start] = max(wf[dead_start], outer_left)
            wf[dead_end - 1] = max(wf[dead_end - 1], outer_right)

            # Per-band phosphor
            self._bgs[b] *= self._bg_decay
            self._bgs[b] += wf * self._pulse_deposit
            np.clip(self._bgs[b], 0.0, 0.4, out=self._bgs[b])

            band_brightness[i] = np.maximum(wf, self._bgs[b])

        # Collapse bands: per-pixel max brightness, single red color
        brightness = band_brightness.max(axis=0)  # (n,)
        rgb = self._full_red * brightness[:, np.newaxis]

        np.clip(rgb, 0.0, 255.0, out=rgb)
        self._frame_buf[:] = rgb.astype(np.uint8)

        return self._frame_buf.copy()

    def get_diagnostics(self) -> dict:
        with self._lock:
            vals = self._vals.copy()
            pulses = int(self._pulse_count)

        return {
            'bass': f'{float(vals[0]):.2f}',
            'mids': f'{float(vals[1]):.2f}',
            'treble': f'{float(vals[2]):.2f}',
            'pulses': str(pulses),
        }
