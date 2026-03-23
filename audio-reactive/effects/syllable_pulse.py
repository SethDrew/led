"""
Syllable Pulse — uniform brightness pulsing driven by speech rhythm.

All LEDs show the same color/brightness. Energy from sticky-floor RMS
drives a smoothed brightness envelope with asymmetric attack/decay.
EnergyDelta detects syllable onsets, adding a brief white-shifted flash.

Designed for the duck's primary mode: a single rubber duck where every
LED pulses together in response to spoken word audio.

Color: warm amber RGB(255, 180, 80), shifting toward white on onsets.
Silence: gentle sinusoidal breathing animation.
"""

import numpy as np
import threading
import math
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, FixedRangeRMS, EnergyDelta


class SyllablePulseEffect(AudioReactiveEffect):
    """Speech rhythm drives uniform brightness pulsing with onset flash."""

    registry_name = 'syllable_pulse'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'RMS energy + onset'

    source_features = [
        {'id': 'energy', 'label': 'Energy', 'color': '#ff8800'},
        {'id': 'onset', 'label': 'Onset', 'color': '#ffcc00'},
    ]

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # --- Audio analysis ---
        self.n_fft = 2048
        self.hop_length = 512
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self._mapper = FixedRangeRMS(
            floor_rms=0.005, ceiling_rms=0.06,
            peak_decay=0.9999,
            fps=sample_rate / self.hop_length,
        )
        self._onset = EnergyDelta()

        self._energy = np.float32(0.0)
        self._onset_val = np.float32(0.0)
        self._lock = threading.Lock()

        # --- Render state ---
        self._envelope = 0.0
        self._onset_pulse = 0.0
        self._silence_frames = 0
        self._time = 0.0

        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Syllable Pulse"

    @property
    def description(self):
        return "Speech rhythm drives uniform brightness pulsing with onset flash."

    # ------------------------------------------------------------------ #
    #  Audio thread                                                        #
    # ------------------------------------------------------------------ #

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            energy = self._mapper.update(frame)
            onset = self._onset.update(frame)
            with self._lock:
                self._energy = np.float32(energy)
                self._onset_val = np.float32(onset)

    # ------------------------------------------------------------------ #
    #  Render thread                                                       #
    # ------------------------------------------------------------------ #

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = float(self._energy)
            onset = float(self._onset_val)

        self._time += dt

        # FixedRangeRMS returns 0.0 when below the gate — use that as silence
        is_silent = energy < 0.001

        # Asymmetric envelope: fast attack (~20ms), medium decay (~300ms)
        attack_alpha = min(1.0, dt / 0.020)   # ~20ms attack
        decay_alpha = min(1.0, dt / 0.300)    # ~300ms decay

        if energy > self._envelope:
            self._envelope += attack_alpha * (energy - self._envelope)
        else:
            self._envelope += decay_alpha * (energy - self._envelope)

        # Onset pulse: trigger on threshold, exponential decay
        if onset > 0.3:
            self._onset_pulse = 1.0
        # Decay: 0.9^(dt*30) per frame ≈ ~100ms half-life
        self._onset_pulse *= 0.9 ** (dt * 30.0)

        # Final brightness
        brightness = min(1.0, max(0.0,
            self._envelope * (1.0 + 0.3 * self._onset_pulse)))

        # Silence detection: breathing animation
        if is_silent:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        # 500ms at ~30fps = 15 frames
        if self._silence_frames > 15:
            breathing = 0.08 * (0.5 + 0.5 * math.sin(
                2.0 * math.pi * self._time / 4.0))
            brightness = max(brightness, breathing)

        # Color: warm amber, onset shifts toward white
        base_r, base_g, base_b = 255.0, 180.0, 80.0
        white_r, white_g, white_b = 255.0, 240.0, 220.0
        lerp = self._onset_pulse * 0.5
        r = base_r + (white_r - base_r) * lerp
        g = base_g + (white_g - base_g) * lerp
        b = base_b + (white_b - base_b) * lerp

        # Deadband snap-to-zero: hold minimum brightness where all channels
        # stay represented, then snap to black. Prevents hue shift during fade
        # (blue channel would hit 0 first, shifting amber toward red).
        DEADBAND = 0.08
        if brightness < DEADBAND:
            brightness = 0.0

        # Apply brightness
        pixel = np.array([
            int(r * brightness),
            int(g * brightness),
            int(b * brightness),
        ], dtype=np.uint8)

        self._frame_buf[:] = pixel
        return self._frame_buf.copy()

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self) -> dict:
        with self._lock:
            energy = float(self._energy)
            onset = float(self._onset_val)

        return {
            'energy': f'{energy:.3f}',
            'onset': f'{onset:.3f}',
            'envelope': f'{self._envelope:.3f}',
            'onset_pulse': f'{self._onset_pulse:.3f}',
            'silence': str(self._silence_frames),
        }

    def get_source_values(self) -> dict:
        with self._lock:
            return {
                'energy': float(self._energy),
                'onset': float(self._onset_val),
            }
