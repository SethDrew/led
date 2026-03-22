# WIP: MFCC Chroma Rainbow
# Preserved from timbral_chroma_split right-side rendering.
# The gradient itself looks great — each MFCC coefficient maps to a
# distinct hue (12-color rainbow), interpolated across LED positions.
# RMS controls saturation (loud = vivid colors, quiet = gray).
#
# STATUS: The gradient/rendering is excellent. The animation responsiveness
# (how the MFCC values drive the gradient) needs iteration — it can feel
# static or under-reactive depending on the track. Consider tuning the
# EMA alpha for different genres.

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


# Per-MFCC saturated colors for chroma encoding (12 coefficients).
# Each coefficient gets a distinct hue so you can see which ones are active.
MFCC_HUES = np.array([
    [255,  50,  50],     # MFCC 1  — red
    [255, 120,  20],     # MFCC 2  — orange
    [240, 200,  30],     # MFCC 3  — yellow
    [140, 230,  40],     # MFCC 4  — lime
    [ 40, 220,  80],     # MFCC 5  — green
    [ 30, 200, 180],     # MFCC 6  — teal
    [ 30, 150, 240],     # MFCC 7  — sky blue
    [ 60,  80, 255],     # MFCC 8  — blue
    [120,  50, 255],     # MFCC 9  — indigo
    [180,  40, 230],     # MFCC 10 — purple
    [230,  40, 160],     # MFCC 11 — magenta
    [255,  50, 100],     # MFCC 12 — pink
], dtype=np.float32)

GRAY = np.array([50.0, 50.0, 50.0], dtype=np.float32)


class WipMfccChromaRainbowEffect(AudioReactiveEffect):
    """MFCC chroma rainbow across the full strip.

    Each LED position maps to an interpolated MFCC coefficient (1-12).
    The 12-color rainbow hue encodes which coefficient is active.
    RMS drives saturation: loud = vivid, quiet = desaturated gray.
    Silence dims and slows the animation via a log-scaled factor.
    """

    registry_name = 'wip_mfcc_chroma_rainbow'
    ref_pattern = 'proportional'
    ref_scope = 'phrase'
    ref_input = 'MFCCs 1-12 + RMS'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 13  # compute 13, display 1-12

        # Frame accumulator
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self.window = np.hanning(self.n_fft).astype(np.float32)

        # Mel filterbank (40 mel bands → DCT → 13 coefficients)
        self.n_mels = 40
        self.mel_fb = self._mel_filterbank(
            sr=sample_rate, n_fft=self.n_fft,
            n_mels=self.n_mels, fmin=20.0, fmax=8000.0,
        )
        # DCT matrix for MFCCs
        self.dct_matrix = self._dct_matrix(self.n_mfcc, self.n_mels)

        # Interpolation: map all LED positions to MFCC coefficients 1-12
        self.n_display = 12
        frac = np.linspace(0, self.n_display - 1, num_leds)
        self._led_idx = np.clip(frac.astype(np.int32), 0, self.n_display - 2)
        self._led_weight = (frac - self._led_idx).astype(np.float32)

        # Pre-compute per-LED saturated colors (interpolated between MFCC hues)
        self._led_hues = np.zeros((num_leds, 3), dtype=np.float32)
        for i in range(num_leds):
            idx = self._led_idx[i]
            idx2 = min(idx + 1, self.n_display - 1)
            w = self._led_weight[i]
            self._led_hues[i] = MFCC_HUES[idx] * (1 - w) + MFCC_HUES[idx2] * w

        # Shared state (audio thread → render thread)
        self._mfcc_norm = np.zeros(self.n_display, dtype=np.float32)
        self._rms = np.float32(0.0)
        self._lock = threading.Lock()

        # Adaptive min/max per MFCC coefficient (sticky expand, slow contract)
        self._mfcc_min = np.zeros(self.n_mfcc, dtype=np.float32)
        self._mfcc_max = np.ones(self.n_mfcc, dtype=np.float32)
        self._adapt_count = 0

        # RMS normalization
        self._rms_peak = np.float32(1e-10)
        self._peak_decay = 0.9995

        # Temporal smoothing (render thread)
        self._smooth_vals = np.zeros(num_leds, dtype=np.float32)
        self._smooth_bright = np.float32(0.0)

        # Logarithmic silence response.
        # Computes a 0-1 factor from RMS energy using a dB-scaled curve:
        #   normal/loud -> factor ~0.95-1.0 (imperceptible)
        #   quiet breakdown -> factor ~0.6-0.8 (visible dimming + slowdown)
        #   near silence -> factor ~0.0-0.2 (dark, nearly frozen)
        self._silence_factor = np.float32(1.0)
        self._silence_alpha = np.float32(0.08)  # ~0.4s settle at 30fps

    @property
    def name(self):
        return "WIP MFCC Chroma Rainbow"

    @property
    def description(self):
        return "MFCC chroma rainbow: each coefficient maps to a hue, RMS drives saturation."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame):
        # RMS
        rms = np.float32(np.sqrt(np.mean(frame ** 2)))
        self._rms_peak = max(rms, self._rms_peak * self._peak_decay)
        rms_norm = rms / self._rms_peak if self._rms_peak > 1e-10 else 0.0

        # FFT → Mel → MFCC
        spec = np.abs(np.fft.rfft(frame * self.window))
        mel_energy = self.mel_fb @ (spec ** 2)
        mel_log = np.log(np.maximum(mel_energy, 1e-10))
        mfcc = self.dct_matrix @ mel_log

        # Adaptive min/max per coefficient (sticky expand, slow contract)
        self._adapt_count += 1
        alpha = 0.05 if self._adapt_count < 60 else 0.005
        for i in range(self.n_mfcc):
            if mfcc[i] < self._mfcc_min[i]:
                self._mfcc_min[i] += 0.1 * (mfcc[i] - self._mfcc_min[i])
            else:
                self._mfcc_min[i] += alpha * (mfcc[i] - self._mfcc_min[i])
            if mfcc[i] > self._mfcc_max[i]:
                self._mfcc_max[i] += 0.1 * (mfcc[i] - self._mfcc_max[i])
            else:
                self._mfcc_max[i] += alpha * (mfcc[i] - self._mfcc_max[i])

        # Normalize MFCCs 1-12 to 0-1
        span = self._mfcc_max - self._mfcc_min
        span = np.maximum(span, 0.1)
        all_norm = np.clip((mfcc - self._mfcc_min) / span, 0, 1).astype(np.float32)
        mfcc_display = all_norm[1:13]  # skip MFCC0 (overall energy)

        with self._lock:
            self._mfcc_norm = mfcc_display
            self._rms = np.float32(rms_norm)

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            mfcc = self._mfcc_norm.copy()
            rms = float(self._rms)

        # ── Logarithmic silence factor ──
        # Convert linear RMS (0-1) to dB, then map through a log curve.
        db = np.float32(20.0 * np.log10(max(rms, 1e-10)))
        # Map dB to 0-1: -40dB -> 0, -6dB -> 1, above -6dB -> 1
        silence_raw = np.clip((db + 40.0) / 34.0, 0.0, 1.0)
        # Power curve: compresses the top so normal levels are ~1.0
        # and the response accelerates near silence
        silence_target = np.float32(silence_raw ** 0.3)
        # Smooth to prevent flicker (~0.4s settle at 30fps)
        self._silence_factor += self._silence_alpha * (silence_target - self._silence_factor)

        # Interpolate 12 MFCC coefficients across all LEDs
        lower = mfcc[self._led_idx]
        upper = mfcc[np.minimum(self._led_idx + 1, self.n_display - 1)]
        led_vals = lower * (1.0 - self._led_weight) + upper * self._led_weight

        # Animation alpha scales with silence factor: near silence -> nearly frozen
        alpha = 0.25 * max(self._silence_factor, 0.02)
        self._smooth_vals += alpha * (led_vals - self._smooth_vals)
        # (smooth_vals drives animation speed; not used for color here —
        # the rainbow hue is purely positional, not value-driven)

        # ── Chroma encoding: MFCC position → hue, RMS → saturation + brightness ──
        self._smooth_bright += 0.3 * (rms - self._smooth_bright)
        sat = np.clip(self._smooth_bright, 0.0, 1.0)
        colors = (GRAY[np.newaxis, :] * (1.0 - sat) +
                  self._led_hues * sat)

        # Brightness from RMS, further scaled by silence factor
        brightness = np.clip(self._smooth_bright, 0.02, 1.0) ** 0.7
        brightness *= self._silence_factor

        return (colors * brightness).clip(0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            mfcc = self._mfcc_norm.copy()

        thirds = [np.mean(mfcc[0:4]), np.mean(mfcc[4:8]), np.mean(mfcc[8:12])]
        shape = ['broad', 'mid-texture', 'fine-detail'][int(np.argmax(thirds))]

        return {
            'timbral': shape,
            'rms': f'{self._smooth_bright:.2f}',
            'silence': f'{self._silence_factor:.2f}',
        }

    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        """Compute a mel filterbank matrix (n_mels, n_fft//2 + 1)."""
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        n_freqs = n_fft // 2 + 1
        fft_freqs = np.linspace(0, sr / 2.0, n_freqs)
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            lo, center, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            up = (fft_freqs - lo) / (center - lo + 1e-10)
            down = (hi - fft_freqs) / (hi - center + 1e-10)
            fb[i] = np.maximum(0, np.minimum(up, down))
        return fb

    @staticmethod
    def _dct_matrix(n_mfcc, n_mels):
        """Type-II DCT matrix for MFCC computation."""
        dct = np.zeros((n_mfcc, n_mels), dtype=np.float32)
        for k in range(n_mfcc):
            for n in range(n_mels):
                dct[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
        dct[0] *= 1.0 / np.sqrt(n_mels)
        dct[1:] *= np.sqrt(2.0 / n_mels)
        return dct
