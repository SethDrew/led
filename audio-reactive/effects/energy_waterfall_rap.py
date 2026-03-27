"""
Energy Waterfall Rap — bass/vocal balanced waterfall.

Same dual-scroll structure as energy_waterfall, but splits audio into
two independently-normalized frequency bands then combines them:
  - Bass (20-250 Hz)  — normalized independently
  - Vocal (600-3 kHz) — normalized independently

Both bands emit from both sources (edges and center). Independent
normalization prevents bass from drowning out vocal rhythm.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


BASS_BAND = (20, 250)
VOCAL_BAND = (600, 3000)


class EnergyWaterfallRapEffect(AudioReactiveEffect):
    """Bass/vocal balanced waterfall — both bands emit from both sources."""

    registry_name = 'energy_waterfall_rap'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'bass + vocal band energy'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 pulse_deposit: float = 0.008,
                 bg_decay: float = 0.96,
                 breathe_amplitude: int = 25,
                 breathe_period: float = 60.0,
                 breathe_hold: float = 10.0,
                 ):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.hop_length = 512
        self.window = np.hanning(self.n_fft).astype(np.float32)

        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )

        # FFT frequency bins for band masks
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        self._bass_mask = (freq_bins >= BASS_BAND[0]) & (freq_bins < BASS_BAND[1])
        self._vocal_mask = (freq_bins >= VOCAL_BAND[0]) & (freq_bins < VOCAL_BAND[1])

        # Per-band sticky floor + peak-decay normalization
        fps = sample_rate / self.hop_length
        self._bass_peak = np.float32(1e-10)
        self._vocal_peak = np.float32(1e-10)
        self._peak_decay = 0.9995

        self._bass_floor = np.float32(1e-10)
        self._vocal_floor = np.float32(1e-10)
        self._floor_alpha = 2.0 / (10.0 * fps + 1)
        self._floor_up_mult = 1.0
        self._floor_down_mult = 4.0
        self._db_window = 15.0

        # Shared state: two energy channels
        self._bass_energy = np.float32(0.0)
        self._vocal_energy = np.float32(0.0)
        self._lock = threading.Lock()

        # Two half-buffers for outer waterfall
        self._mid = num_leds // 2
        self._left_buf = np.zeros(self._mid, dtype=np.float32)
        self._right_buf = np.zeros(num_leds - self._mid, dtype=np.float32)

        # Per-pixel background brightness (phosphor residue)
        self._bg = np.zeros(num_leds, dtype=np.float32)
        self._pulse_deposit = pulse_deposit
        self._bg_decay = bg_decay

        # Center source buffers
        self._center_left = np.zeros(breathe_amplitude, dtype=np.float32)
        self._center_right = np.zeros(breathe_amplitude, dtype=np.float32)

        # Breathing dead band
        self._breathe_amplitude = breathe_amplitude
        self._breathe_period = breathe_period
        self._breathe_hold = breathe_hold
        self._breathe_time = 0.0

        # Full red color
        self._full_red = np.array([255.0, 40.0, 0.0])

        # Output buffer
        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Energy Waterfall Rap"

    @property
    def description(self):
        return "Bass/vocal balanced waterfall for rap."

    def _sticky_floor_db(self, rms, floor, peak):
        """Sticky floor + dB normalization for a single band.

        Returns (normalized_value, updated_floor, updated_peak).
        """
        if rms > floor:
            alpha = self._floor_alpha * self._floor_up_mult
        else:
            alpha = self._floor_alpha * self._floor_down_mult
        floor = floor + alpha * (rms - floor)
        floor = max(floor, 1e-10)

        ratio = max(rms / floor, 1.0)
        db_above = 20.0 * np.log10(ratio)
        val = np.clip(db_above / self._db_window, 0.0, 1.0)

        peak = max(val, peak * self._peak_decay)
        if peak > 1e-10:
            val = val / peak

        return np.float32(val), np.float32(floor), np.float32(peak)

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            spectrum = np.abs(np.fft.rfft(frame * self.window))

            bass_rms = np.sqrt(np.mean(spectrum[self._bass_mask] ** 2))
            vocal_rms = np.sqrt(np.mean(spectrum[self._vocal_mask] ** 2))

            bass_val, self._bass_floor, self._bass_peak = self._sticky_floor_db(
                bass_rms, self._bass_floor, self._bass_peak)
            vocal_val, self._vocal_floor, self._vocal_peak = self._sticky_floor_db(
                vocal_rms, self._vocal_floor, self._vocal_peak)

            with self._lock:
                self._bass_energy = bass_val
                self._vocal_energy = vocal_val

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            bass = float(self._bass_energy)
            vocal = float(self._vocal_energy)

        bass = np.clip(bass, 0.0, 1.0)
        vocal = np.clip(vocal, 0.0, 1.0)

        # Combine: both bands drive both sources
        val = max(bass, vocal)

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

        # Outer waterfall: edges → center
        self._left_buf[1:] = self._left_buf[:-1]
        self._left_buf[0] = val
        self._right_buf[1:] = self._right_buf[:-1]
        self._right_buf[0] = val

        # Center source: center → outward
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
            # Save outer values at boundary before overwriting
            outer_left = wf_combined[dead_start]
            outer_right = wf_combined[dead_end - 1]

            wf_combined[dead_start:mid] = self._center_left[:offset][::-1]
            wf_combined[mid:dead_end] = self._center_right[:offset]

            # 1-pixel overlap at boundary: max of outer and center
            wf_combined[dead_start] = max(wf_combined[dead_start], outer_left)
            wf_combined[dead_end - 1] = max(wf_combined[dead_end - 1], outer_right)

        # Background phosphor
        self._bg *= self._bg_decay
        self._bg += wf_combined * self._pulse_deposit
        np.clip(self._bg, 0.0, 0.4, out=self._bg)

        # Final brightness: max of pulse and background
        brightness = np.maximum(wf_combined, self._bg)

        # Map to color
        self._frame_buf[:] = (self._full_red * brightness[:, np.newaxis]).astype(np.uint8)

        return self._frame_buf.copy()

    def get_diagnostics(self) -> dict:
        with self._lock:
            bass = float(self._bass_energy)
            vocal = float(self._vocal_energy)

        return {
            'bass': f'{bass:.2f}',
            'vocal': f'{vocal:.2f}',
            'bg_mean': f'{self._bg.mean():.3f}',
            'bg_max': f'{self._bg.max():.3f}',
        }
