# MFCC Chroma Rainbow — timbral texture mapped as a rainbow gradient.
#
# Each LED position maps to an interpolated MFCC coefficient (1-12).
# Saturation is driven by MFCC spectral contrast (variance across
# normalized coefficients), NOT by RMS — tonal sounds (instruments,
# vocals) produce vivid colors, flat spectra (noise) go gray.
# RMS drives overall brightness via an asymmetric envelope.

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


# Per-MFCC saturated colors for chroma encoding (12 coefficients).
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

GRAY = 80.0  # neutral gray level per channel


class MfccChromaRainbowEffect(AudioReactiveEffect):
    """MFCC chroma rainbow across the full strip.

    Each LED position maps to an interpolated MFCC coefficient (1-12).
    Spectral contrast (MFCC variance) drives saturation — tonal content
    produces vivid colors, flat spectra go gray. RMS drives overall
    brightness with asymmetric attack/decay.
    """

    registry_name = 'mfcc_chroma_rainbow'
    ref_pattern = 'proportional'
    ref_scope = 'phrase'
    ref_input = 'MFCCs 1-12 + RMS'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 13   # compute 13, display 1-12
        self.n_display = 12

        # Frame accumulator (overlapped frames for FFT)
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self.window = np.hanning(self.n_fft).astype(np.float32)

        # Audio-thread dt: fixed, derived from hop size
        self._audio_dt = self.hop_length / sample_rate

        # Mel filterbank (40 bands, 20-8000 Hz) → DCT → 13 MFCCs
        self.n_mels = 40
        self.mel_fb = self._mel_filterbank(
            sr=sample_rate, n_fft=self.n_fft,
            n_mels=self.n_mels, fmin=20.0, fmax=8000.0,
        )
        self.dct_matrix = self._dct_matrix(self.n_mfcc, self.n_mels)

        # LED → MFCC interpolation mapping
        frac = np.linspace(0, self.n_display - 1, num_leds)
        self._led_idx = np.clip(frac.astype(np.int32), 0, self.n_display - 2)
        self._led_weight = (frac - self._led_idx).astype(np.float32)

        # Pre-compute per-LED hue colors (interpolated between MFCC hues)
        self._led_hues = np.zeros((num_leds, 3), dtype=np.float32)
        for i in range(num_leds):
            idx = self._led_idx[i]
            idx2 = min(idx + 1, self.n_display - 1)
            w = self._led_weight[i]
            self._led_hues[i] = MFCC_HUES[idx] * (1 - w) + MFCC_HUES[idx2] * w

        # ── Audio-thread state (updated in process_audio) ──

        # Adaptive min/max per MFCC (sticky expand, slow contract)
        self._mfcc_min = np.zeros(self.n_mfcc, dtype=np.float32)
        self._mfcc_max = np.ones(self.n_mfcc, dtype=np.float32)
        self._adapt_count = 0

        # RMS normalization
        self._rms_peak = np.float32(1e-10)

        # ── Shared state (audio → render, protected by lock) ──
        self._shared_mfcc_norm = np.zeros(self.n_display, dtype=np.float32)
        self._shared_rms_norm = np.float32(0.0)
        self._shared_mfcc_contrast = np.float32(0.0)
        self._lock = threading.Lock()

        # ── Render-thread state ──
        self._smooth_br = np.zeros(num_leds, dtype=np.float32)
        self._smooth_bright = np.float32(0.0)     # RMS envelope
        self._smooth_sat = np.float32(1.0)         # start chromatic
        self._silence_factor = np.float32(1.0)

    @property
    def name(self):
        return "MFCC Chroma Rainbow"

    @property
    def description(self):
        return ("MFCC chroma rainbow: spectral contrast drives saturation, "
                "RMS drives brightness.")

    # ════════════════════════════════════════════
    # Audio thread
    # ════════════════════════════════════════════

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray):
        dt = self._audio_dt

        # RMS
        rms = np.float32(np.sqrt(np.mean(frame ** 2)))
        self._rms_peak = max(rms, self._rms_peak * 0.9995)
        rms_norm = float(rms / self._rms_peak) if self._rms_peak > 1e-10 else 0.0

        # FFT → power spectrum → mel energies → log → DCT → MFCCs
        spec = np.abs(np.fft.rfft(frame * self.window))
        mel_energy = self.mel_fb @ (spec ** 2)
        mel_log = np.log(np.maximum(mel_energy, 1e-10))
        mfcc = self.dct_matrix @ mel_log

        # Adaptive min/max per coefficient (dt-based, sticky expand / slow contract)
        self._adapt_count += 1
        EXPAND_RATE = 0.6       # new extremes absorbed ~1.2s half-life
        CONTRACT_RATE = 0.3     # range shrinks ~2.3s half-life
        WARMUP_CONTRACT = 3.0   # faster contraction during warmup
        expand_alpha = 1.0 - np.exp(-EXPAND_RATE * dt)
        contract_rate = WARMUP_CONTRACT if self._adapt_count < 60 else CONTRACT_RATE
        contract_alpha = 1.0 - np.exp(-contract_rate * dt)

        for i in range(self.n_mfcc):
            if mfcc[i] < self._mfcc_min[i]:
                self._mfcc_min[i] += expand_alpha * (mfcc[i] - self._mfcc_min[i])
            else:
                self._mfcc_min[i] += contract_alpha * (mfcc[i] - self._mfcc_min[i])
            if mfcc[i] > self._mfcc_max[i]:
                self._mfcc_max[i] += expand_alpha * (mfcc[i] - self._mfcc_max[i])
            else:
                self._mfcc_max[i] += contract_alpha * (mfcc[i] - self._mfcc_max[i])

        # Normalize MFCCs 1-12 to 0-1 (skip MFCC0 = overall energy)
        mfcc_norm = np.zeros(self.n_display, dtype=np.float32)
        for i in range(self.n_display):
            span = max(self._mfcc_max[i + 1] - self._mfcc_min[i + 1], 0.1)
            mfcc_norm[i] = np.clip(
                (mfcc[i + 1] - self._mfcc_min[i + 1]) / span, 0.0, 1.0
            )

        # MFCC spectral contrast: std dev of normalized coefficients
        mfcc_mean = np.mean(mfcc_norm)
        mfcc_var = np.mean((mfcc_norm - mfcc_mean) ** 2)
        mfcc_contrast = float(np.sqrt(mfcc_var))

        with self._lock:
            self._shared_mfcc_norm = mfcc_norm
            self._shared_rms_norm = np.float32(rms_norm)
            self._shared_mfcc_contrast = np.float32(mfcc_contrast)

    # ════════════════════════════════════════════
    # Render thread
    # ════════════════════════════════════════════

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            mfcc = self._shared_mfcc_norm.copy()
            rms = float(self._shared_rms_norm)
            contrast = float(self._shared_mfcc_contrast)

        # ── Silence factor (dB-scaled curve, dt-based EMA ~5/s) ──
        db = 20.0 * np.log10(max(rms, 1e-10))
        sil_raw = np.clip((db + 40.0) / 34.0, 0.0, 1.0)
        sil_alpha = 1.0 - np.exp(-5.0 * dt)
        self._silence_factor += sil_alpha * (sil_raw ** 0.3 - self._silence_factor)

        # ── Per-LED brightness: fast attack, multiplicative decay toward zero ──
        ATTACK = 55.0           # ~instant onset
        BR_DECAY = 6.93         # brightness half-life ~100ms

        a_attack = 1.0 - np.exp(-ATTACK * dt)
        br_decay = np.exp(-BR_DECAY * dt)

        # Interpolate MFCC values across LED positions
        lower = mfcc[self._led_idx]
        upper = mfcc[np.minimum(self._led_idx + 1, self.n_display - 1)]
        led_targets = lower * (1.0 - self._led_weight) + upper * self._led_weight

        # Asymmetric: attack toward target, decay toward zero
        rising = led_targets > self._smooth_br
        self._smooth_br = np.where(
            rising,
            self._smooth_br + a_attack * (led_targets - self._smooth_br),
            self._smooth_br * br_decay,
        )

        # ── Spectral contrast → saturation ──
        # Exponential curve: chromatic for music, gray for noise
        sat_target = 1.0 - np.exp(-contrast * 16.0)
        SAT_ATTACK = 6.0        # snap to color ~115ms
        SAT_DECAY_RATE = 3.0    # hold color ~230ms
        sat_rate = SAT_ATTACK if sat_target > self._smooth_sat else SAT_DECAY_RATE
        sat_alpha = 1.0 - np.exp(-sat_rate * dt)
        self._smooth_sat += sat_alpha * (sat_target - self._smooth_sat)

        # ── Overall brightness from RMS (asymmetric: fast attack / slow decay) ──
        if rms > self._smooth_bright:
            br_env_alpha = 1.0 - np.exp(-21.0 * dt)
            self._smooth_bright += br_env_alpha * (rms - self._smooth_bright)
        else:
            self._smooth_bright *= np.exp(-1.0 * dt)
        base_br = float(min(self._smooth_bright, 1.0) ** 0.7) * float(self._silence_factor)

        # ── Render: (hue * sat + GRAY * (1-sat)) * brightness ──
        sat = float(self._smooth_sat)
        br_per_led = base_br * self._smooth_br

        # Brightness cutoff: br < 0.12 → pure black (prevents dim residue)
        colors = np.zeros((self.num_leds, 3), dtype=np.float32)
        active = br_per_led >= 0.12
        if np.any(active):
            hue_part = self._led_hues[active] * sat + GRAY * (1.0 - sat)
            colors[active] = hue_part * br_per_led[active, np.newaxis]

        return np.clip(colors, 0, 255).astype(np.uint8)

    # ════════════════════════════════════════════
    # Diagnostics
    # ════════════════════════════════════════════

    def get_diagnostics(self) -> dict:
        with self._lock:
            mfcc = self._shared_mfcc_norm.copy()
            contrast = float(self._shared_mfcc_contrast)

        thirds = [np.mean(mfcc[0:4]), np.mean(mfcc[4:8]), np.mean(mfcc[8:12])]
        shape = ['broad', 'mid-texture', 'fine-detail'][int(np.argmax(thirds))]

        return {
            'timbral': shape,
            'contrast': f'{contrast:.3f}',
            'saturation': f'{float(self._smooth_sat):.2f}',
            'brightness': f'{float(self._smooth_bright):.2f}',
            'silence': f'{float(self._silence_factor):.2f}',
        }

    # ════════════════════════════════════════════
    # Static helpers
    # ════════════════════════════════════════════

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
