"""
Timbral Split — same MFCC timbral shape, two color encodings side by side.

Both halves show the same data: MFCCs 1-12 interpolated across LEDs.
Each LED position = a timbral coefficient (low = broad spectral shape,
high = fine texture). The current frame is a vertical slice through the
MFCC heatmap — literally the vibes timbral panel, but on physical LEDs.

First 75 LEDs:  Palette heatmap encoding (palette cycles on timbral shift)
  MFCC value → palette colormap.
  Palette switches when a timbral shift is detected.

Last 75 LEDs:   Chroma (saturation) encoding
  MFCC position → hue, RMS → saturation and brightness.
  Quiet = dim gray, loud = bright saturated color.

A/B comparison: which color encoding makes timbral shifts more readable
on physical LEDs?
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


# Palette collection for left half — cycles on timbral section change.
# Each palette must be visually distinct from its neighbors in the list,
# with meaningful hue variation across the gradient (not monochromatic).
GOOD_PALETTES = [
    # magma (purple → red → orange → yellow)
    np.array([[0,0,4], [20,10,50], [70,10,100], [140,20,80], [200,50,40], [240,110,10], [250,190,60], [255,255,180]], dtype=np.float32),
    # ocean (deep blue → teal → aqua → seafoam)
    np.array([[0,0,4], [0,10,50], [0,40,120], [0,100,160], [20,180,180], [80,230,200], [180,255,220]], dtype=np.float32),
    # ember (dark red → crimson → orange → gold)
    np.array([[0,0,4], [40,5,0], [120,10,0], [200,20,0], [240,80,0], [255,160,20], [255,220,80]], dtype=np.float32),
    # aurora (deep violet → blue → green → lime)
    np.array([[0,0,4], [30,0,60], [40,0,140], [0,60,200], [0,160,120], [40,220,60], [160,255,80]], dtype=np.float32),
    # rose (deep pink → magenta → salmon → peach)
    np.array([[0,0,4], [40,0,30], [100,0,60], [180,20,80], [230,60,100], [255,120,140], [255,200,180]], dtype=np.float32),
    # frost (dark blue → ice blue → lavender → white)
    np.array([[0,0,4], [10,10,60], [30,40,140], [80,100,200], [140,160,240], [200,210,255], [240,240,255]], dtype=np.float32),
]

# Per-MFCC saturated colors for chroma encoding (12 coefficients)
# Each coefficient gets a distinct hue so you can see which ones are active
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


def _sample_palette(t, anchors):
    """Sample a palette at position t (0-1). Works with arrays."""
    t = np.clip(t, 0.0, 1.0)
    n = len(anchors) - 1
    idx = t * n
    lo = np.clip(idx.astype(np.int32), 0, n - 1)
    hi = np.minimum(lo + 1, n)
    frac = (idx - lo)[:, np.newaxis]
    return anchors[lo] * (1 - frac) + anchors[hi] * frac


class TimbralChromaSplitEffect(AudioReactiveEffect):
    """Split strip: MFCC timbral shape with magma (left) vs chroma (right)."""

    registry_name = 'timbral_chroma_split'
    ref_pattern = 'proportional'
    ref_scope = 'phrase'
    ref_input = 'MFCCs 1-12 + RMS'

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 13  # compute 13, display 1-12

        # Split point
        self.split = num_leds // 2
        self.n_left = self.split
        self.n_right = num_leds - self.split

        # Frame accumulator
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )
        self.window = np.hanning(self.n_fft).astype(np.float32)

        # Mel filterbank for MFCCs (40 mel bands → DCT → 13 coefficients)
        self.n_mels = 40
        self.mel_fb = self._mel_filterbank(
            sr=sample_rate, n_fft=self.n_fft,
            n_mels=self.n_mels, fmin=20.0, fmax=8000.0,
        )
        # DCT matrix for MFCCs
        self.dct_matrix = self._dct_matrix(self.n_mfcc, self.n_mels)

        # Interpolation: map LED positions to MFCC coefficients 1-12
        # Same mapping for both halves (same data, different color encoding)
        self.n_display = 12  # MFCCs 1-12
        frac = np.linspace(0, self.n_display - 1, self.n_left)
        self._left_idx = np.clip(frac.astype(np.int32), 0, self.n_display - 2)
        self._left_weight = (frac - self._left_idx).astype(np.float32)

        frac_r = np.linspace(0, self.n_display - 1, self.n_right)
        self._right_idx = np.clip(frac_r.astype(np.int32), 0, self.n_display - 2)
        self._right_weight = (frac_r - self._right_idx).astype(np.float32)

        # Pre-compute per-LED saturated colors for right half (chroma encoding)
        self._right_hues = np.zeros((self.n_right, 3), dtype=np.float32)
        for i in range(self.n_right):
            idx = self._right_idx[i]
            idx2 = min(idx + 1, self.n_display - 1)
            w = self._right_weight[i]
            self._right_hues[i] = MFCC_HUES[idx] * (1 - w) + MFCC_HUES[idx2] * w

        # Shared state
        self._mfcc_norm = np.zeros(self.n_display, dtype=np.float32)
        self._rms = np.float32(0.0)
        self._lock = threading.Lock()

        # Adaptive min/max per MFCC coefficient
        self._mfcc_min = np.zeros(self.n_mfcc, dtype=np.float32)
        self._mfcc_max = np.ones(self.n_mfcc, dtype=np.float32)
        self._adapt_count = 0

        # RMS normalization
        self._rms_peak = np.float32(1e-10)
        self._peak_decay = 0.9995

        # Temporal smoothing (render thread)
        self._smooth_left = np.zeros(self.n_left, dtype=np.float32)
        self._smooth_right = np.zeros(self.n_right, dtype=np.float32)
        self._smooth_bright = np.float32(0.0)

        # Logarithmic silence response (right side only).
        # Computes a 0-1 factor from RMS energy using a dB-scaled curve:
        #   normal/loud -> factor ~0.95-1.0 (imperceptible)
        #   quiet breakdown -> factor ~0.6-0.8 (visible dimming + slowdown)
        #   near silence -> factor ~0.0-0.2 (dark, nearly frozen)
        self._silence_factor = np.float32(1.0)
        self._silence_alpha = np.float32(0.08)  # ~0.4s settle at 30fps

        # Palette switching state
        self._palette_idx = len(GOOD_PALETTES) - 1  # start on magma
        self._prev_palette_idx = self._palette_idx
        self._palette_blend = np.float32(0.0)  # 0 = prev palette, 1 = current

        # Timbral section-change detection (anchored reference with rollback)
        #
        # Operates on L2-normalized MFCCs 0-12. Including MFCC 0 (overall
        # loudness) means energy-driven boundaries (EDM drops) are captured
        # alongside pure shape changes. L2 normalization makes MFCC 0 just
        # one of 13 dimensions, not the dominant signal.
        #
        # Two EMAs at very different timescales:
        #   anchor = "what section we're in" (240s TC, re-snapped on commit)
        #   probe  = "what is happening now" (20s TC)
        # The anchor barely moves within a section, so accumulated drift
        # from gradual timbral evolution builds up over 30-60s until it
        # crosses the threshold. On commit, anchor snaps to probe (re-anchors
        # to the new section's timbral signature).
        self._n_detect = self.n_mfcc  # 13 coefficients (MFCCs 0-12)
        fps_audio = sample_rate / self.hop_length
        self._fps_audio = np.float32(fps_audio)
        self._anchor_sig = np.zeros(self._n_detect, dtype=np.float32)
        self._probe_sig = np.zeros(self._n_detect, dtype=np.float32)
        self._anchor_alpha = np.float32(1.0 / (240.0 * fps_audio))  # very slow
        self._probe_alpha = np.float32(1.0 / (20.0 * fps_audio))    # moderate
        self._shift_blend = np.float32(0.0)
        self._blend_fade_in = np.float32(0.03 * (30.0 / fps_audio))   # ~1.7s to commit
        self._blend_fade_out = np.float32(0.05 * (30.0 / fps_audio))  # ~1.0s to roll back
        self._eagerness_floor = np.float32(1.05)
        self._eagerness_ceiling = np.float32(2.0)
        self._eagerness_tau = np.float32(30.0)
        self._dist_ema = np.float32(0.0)
        self._dist_ema_rise_alpha = np.float32(1.0 / (120.0 * fps_audio))
        self._dist_ema_fall_alpha = np.float32(1.0 / (15.0 * fps_audio))
        self._warmup_dist_sum = np.float32(0.0)
        self._warmup_dist_count = 0
        self._dead_zone_frames = int(15.0 * fps_audio)
        self._warmup_frames = int(25.0 * fps_audio)
        self._shift_frames = 0
        self._frames_since_commit = 0  # start in dead zone so warmup EMA can stabilize
        self._need_ema_reseed = False

    @property
    def name(self):
        return "Timbral Split"

    @property
    def description(self):
        return "MFCC timbral shape: magma heatmap (left) vs chroma saturation (right)."

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

        # Timbral section-change detection (anchored reference with rollback)
        # Operates on L2-normalized raw MFCCs 0-12 (13 coefficients).
        # Including MFCC 0 captures energy-driven boundaries; L2 normalization
        # prevents it from dominating the distance metric.
        mfcc_raw = mfcc[0:self._n_detect].astype(np.float32)
        raw_norm = np.float32(np.sqrt(np.dot(mfcc_raw, mfcc_raw)))
        mfcc_unit = mfcc_raw / (raw_norm + 1e-10)

        self._shift_frames += 1
        self._frames_since_commit += 1

        # Update both EMAs (on unit-length MFCC vectors)
        self._anchor_sig += self._anchor_alpha * (mfcc_unit - self._anchor_sig)
        self._probe_sig += self._probe_alpha * (mfcc_unit - self._probe_sig)

        # Euclidean distance between anchor and probe on the unit sphere.
        # This measures how far the current timbral shape has drifted from
        # the section reference, independent of overall energy level.
        diff = self._anchor_sig - self._probe_sig
        shape_dist = np.float32(np.sqrt(np.dot(diff, diff)))

        # Blend management: fade up when diverged, roll back when not
        in_dead_zone = self._frames_since_commit < self._dead_zone_frames

        # Accumulate warmup distances to seed EMA
        in_warmup = self._shift_frames < self._warmup_frames
        if self._shift_frames > 1 and in_warmup:
            self._warmup_dist_sum += shape_dist
            self._warmup_dist_count += 1
        if self._shift_frames == self._warmup_frames and self._warmup_dist_count > 0:
            self._dist_ema = np.float32(self._warmup_dist_sum / self._warmup_dist_count)
            self._frames_since_commit = 0  # enter dead zone at warmup end

        # Re-seed EMA when dead zone ends after a commit, so the EMA
        # reflects the new section's baseline rather than the old one.
        if self._need_ema_reseed and not in_dead_zone:
            self._dist_ema = np.float32(shape_dist)
            self._need_ema_reseed = False

        # Asymmetric EMA normalization: slow rise (spike-resistant),
        # fast fall (tracks baseline). Frozen during dead zone and warmup.
        if not in_dead_zone and not in_warmup:
            alpha = self._dist_ema_rise_alpha if shape_dist > self._dist_ema else self._dist_ema_fall_alpha
            self._dist_ema += alpha * (shape_dist - self._dist_ema)
        shape_dist_norm = shape_dist / (self._dist_ema + 1e-10)

        # Eagerness curve: threshold decays from ceiling to floor
        time_since_commit = self._frames_since_commit / self._fps_audio
        threshold = self._eagerness_floor + (self._eagerness_ceiling - self._eagerness_floor) * np.exp(-time_since_commit / self._eagerness_tau)
        if not in_dead_zone and self._shift_frames >= self._warmup_frames:
            if shape_dist_norm > threshold:
                self._shift_blend += self._blend_fade_in
            else:
                self._shift_blend -= self._blend_fade_out
            self._shift_blend = np.clip(self._shift_blend, 0.0, 1.0)
        else:
            # During warmup/dead zone, only allow rollback
            self._shift_blend = max(np.float32(0.0),
                                    self._shift_blend - self._blend_fade_out)

        with self._lock:
            self._mfcc_norm = mfcc_display
            self._rms = np.float32(rms_norm)
            # Commit: sustained divergence confirmed
            if self._shift_blend > 0.9:
                self._prev_palette_idx = self._palette_idx
                self._palette_idx = (self._palette_idx + 1) % len(GOOD_PALETTES)
                self._palette_blend = np.float32(0.0)  # start blending in render
                # Re-anchor: snap anchor to current probe (new section baseline)
                self._anchor_sig[:] = self._probe_sig
                self._shift_blend = np.float32(0.0)
                self._frames_since_commit = 0
                self._need_ema_reseed = True

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            mfcc = self._mfcc_norm.copy()
            rms = float(self._rms)
            pal_idx = self._palette_idx
            prev_pal_idx = self._prev_palette_idx
            pal_blend = float(self._palette_blend)

        # Advance palette blend toward 1.0 (smooth transition over ~1s @30fps)
        pal_blend = min(1.0, pal_blend + 0.033)
        with self._lock:
            self._palette_blend = np.float32(pal_blend)

        # Interpolate 12 MFCC coefficients across LEDs
        # Left half
        lower = mfcc[self._left_idx]
        upper = mfcc[np.minimum(self._left_idx + 1, self.n_display - 1)]
        led_left = lower * (1.0 - self._left_weight) + upper * self._left_weight
        # EMA smoothing: alpha=0.35 gives ~0.2s settle time at 30fps.
        # Responsive enough to track timbral changes within a beat,
        # smooth enough to prevent single-frame flicker. Preserves full
        # dynamic range (unlike the old slew-rate limiter).
        self._smooth_left += 0.35 * (led_left - self._smooth_left)
        left_val = np.clip(self._smooth_left, 0.0, 1.0)

        # ── Logarithmic silence factor (right side only) ──
        # Convert linear RMS (0-1) to dB, then map through a log curve.
        # rms here is rms_norm (0-1 relative to adaptive peak).
        db = np.float32(20.0 * np.log10(max(rms, 1e-10)))
        # Map dB to 0-1: -40dB -> 0, -6dB -> 1, above -6dB -> 1
        silence_raw = np.clip((db + 40.0) / 34.0, 0.0, 1.0)
        # Power curve: compresses the top so normal levels are ~1.0
        # and the response accelerates near silence
        silence_target = np.float32(silence_raw ** 0.3)
        # Smooth to prevent flicker (~0.4s settle at 30fps)
        self._silence_factor += self._silence_alpha * (silence_target - self._silence_factor)

        # Right half (same data, animation speed modulated by silence)
        lower_r = mfcc[self._right_idx]
        upper_r = mfcc[np.minimum(self._right_idx + 1, self.n_display - 1)]
        led_right = lower_r * (1.0 - self._right_weight) + upper_r * self._right_weight
        # Animation alpha scales with silence factor: near silence -> nearly frozen
        right_alpha = 0.25 * max(self._silence_factor, 0.02)
        self._smooth_right += right_alpha * (led_right - self._smooth_right)
        right_val = np.clip(self._smooth_right, 0.0, 1.0)

        # ── Left: palette heatmap (value → color) ──
        # Smooth cross-fade between previous and current palette
        if pal_blend >= 1.0 or prev_pal_idx == pal_idx:
            left_colors = _sample_palette(left_val, GOOD_PALETTES[pal_idx])
        else:
            old_colors = _sample_palette(left_val, GOOD_PALETTES[prev_pal_idx])
            new_colors = _sample_palette(left_val, GOOD_PALETTES[pal_idx])
            left_colors = old_colors * (1.0 - pal_blend) + new_colors * pal_blend

        # ── Right: chroma encoding (MFCC position → hue, RMS → saturation + brightness) ──
        self._smooth_bright += 0.3 * (rms - self._smooth_bright)
        sat = np.clip(self._smooth_bright, 0.0, 1.0)
        right_colors = (GRAY[np.newaxis, :] * (1.0 - sat) +
                        self._right_hues * sat)

        # Brightness from RMS, further scaled by silence factor
        brightness = np.clip(self._smooth_bright, 0.02, 1.0) ** 0.7
        brightness *= self._silence_factor

        frame = np.zeros((self.num_leds, 3), dtype=np.float32)
        frame[:self.split] = left_colors
        frame[self.split:] = right_colors * brightness

        return frame.clip(0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            mfcc = self._mfcc_norm.copy()

        thirds = [np.mean(mfcc[0:4]), np.mean(mfcc[4:8]), np.mean(mfcc[8:12])]
        shape = ['broad', 'mid-texture', 'fine-detail'][int(np.argmax(thirds))]

        return {
            'timbral': shape,
            'rms': f'{self._smooth_bright:.2f}',
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
