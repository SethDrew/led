"""
All 12 WLED Sound Reactive 1D effects.

Shares a single audio processor (volume, 16 FFT bands, peak detection,
dominant frequency) across 12 visual effect renderers. Each renderer
produces a strip of LEDs — stack them vertically for side-by-side comparison.

Effect list matches WLED source (wled00/FX.cpp audio-reactive 1D effects):
  0  Juggles      — volumeSmth
  1  Midnoise     — volumeSmth + FFT_MajorPeak
  2  Noisemeter   — volumeSmth + samplePeak
  3  Plasmoid     — volumeSmth + samplePeak
  4  Blurz        — fftResult[16]
  5  DJLight      — fftResult[16]
  6  Freqmap      — FFT_MajorPeak + volumeSmth
  7  Freqmatrix   — FFT_MajorPeak + volumeSmth
  8  Freqpixels   — FFT_MajorPeak + volumeSmth
  9  Freqwave     — FFT_MajorPeak + volumeSmth
  10 Noisemove    — fftResult[16]
  11 Rocktaves    — FFT_MajorPeak + magnitude
"""

import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import AudioReactiveEffect

# ── WLED frequency band definitions (from frequency_reactive.py) ─────

WLED_BAND_BINS = [
    (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 13), (13, 19), (19, 26),
    (26, 33), (33, 44), (44, 56), (56, 70), (70, 86), (86, 104), (104, 165), (165, 215),
]

WLED_PINK_NOISE = np.array([
    1.70, 1.71, 1.73, 1.78, 1.68, 1.56, 1.55, 1.63,
    1.79, 1.62, 1.80, 2.06, 2.47, 3.35, 6.83, 9.55
])

# ── Helpers ──────────────────────────────────────────────────────────

def _hsv(h, s, v):
    """HSV to RGB. h: 0-1 (wraps), s: 0-1, v: 0-1. Returns (r,g,b) ints 0-255."""
    if s < 0.001:
        vi = int(v * 255)
        return (vi, vi, vi)
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = int(v * (1 - s) * 255)
    q = int(v * (1 - s * f) * 255)
    t = int(v * (1 - s * (1 - f)) * 255)
    vi = int(v * 255)
    if i == 0:   return (vi, t, p)
    elif i == 1: return (q, vi, p)
    elif i == 2: return (p, vi, t)
    elif i == 3: return (p, q, vi)
    elif i == 4: return (t, p, vi)
    else:        return (vi, p, q)

# Permutation table for value noise
_PERM = list(range(256))
_PERM_RNG = __import__('random').Random(42)
_PERM_RNG.shuffle(_PERM)
_PERM = _PERM + _PERM

def _noise1d(x):
    """1D value noise, returns 0.0-1.0."""
    xi = int(np.floor(x)) & 255
    xf = x - np.floor(x)
    u = xf * xf * (3 - 2 * xf)
    return (_PERM[xi] * (1 - u) + _PERM[xi + 1] * u) / 255.0

def _freq_to_hue(freq):
    """Map frequency (Hz) to hue 0-1 on a log scale."""
    freq = max(freq, 43)
    return np.clip(np.log2(freq / 43.0) / np.log2(10000 / 43.0), 0, 1)


# ── Shared Audio State ───────────────────────────────────────────────

class WLEDAudioState:
    """Computes WLED's audio features once per chunk for all effects."""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.n_fft = 1024
        self.window = np.hanning(self.n_fft)
        self.audio_buffer = np.zeros(self.n_fft)

        # Volume
        self.volume_smth = 0.0
        self.volume_raw = 0.0
        self.sample_max = 1.0
        self.volume_norm = 0.0  # 0-255

        # FFT 16 bands
        self.fft_result = np.zeros(16)  # 0-255
        self._fft_avg = np.zeros(16)
        self._band_max = 1.0

        # Peak / major peak
        self.sample_peak = False
        self.fft_major_peak = 0.0
        self.fft_magnitude = 0.0
        self.bin_num = 0
        self._peak_history = []
        self._peak_threshold = 0.0
        self._last_peak_time = 0.0

    def process(self, mono_chunk):
        chunk_len = min(len(mono_chunk), self.n_fft)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = mono_chunk[:chunk_len]

        # ── Volume ──
        rms = np.sqrt(np.mean(self.audio_buffer ** 2))
        sample = rms * 32768.0
        self.volume_raw = sample
        self.volume_smth = (self.volume_smth * 15.0 + sample) / 16.0
        if sample > self.sample_max:
            self.sample_max = sample
        else:
            self.sample_max = self.sample_max * 0.9995 + sample * 0.0005
        self.volume_norm = min(sample / self.sample_max * 255, 255) if self.sample_max > 10 else 0

        # ── FFT ──
        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))
        spectrum[0] = 0

        # 16 bands
        fft_calc = np.zeros(16)
        for i, (lo, hi) in enumerate(WLED_BAND_BINS):
            hi = min(hi, len(spectrum))
            if lo < len(spectrum) and hi > lo:
                fft_calc[i] = np.mean(spectrum[lo:hi])
        fft_calc *= WLED_PINK_NOISE

        # Auto-gain
        bm = np.max(fft_calc)
        if bm > self._band_max:
            self._band_max = bm
        else:
            self._band_max = self._band_max * 0.998 + bm * 0.002
        gain = 1.0 / self._band_max if self._band_max > 1e-6 else 1.0
        normed = fft_calc * gain

        # Asymmetric smoothing
        for i in range(16):
            if normed[i] > self._fft_avg[i]:
                self._fft_avg[i] = normed[i] * 0.75 + 0.25 * self._fft_avg[i]
            else:
                self._fft_avg[i] = normed[i] * 0.22 + 0.78 * self._fft_avg[i]
            self.fft_result[i] = min(np.sqrt(max(self._fft_avg[i], 0)) * 255, 255)

        # ── Major peak ──
        vs, ve = 2, min(len(spectrum), 256)
        if ve > vs:
            self.bin_num = vs + np.argmax(spectrum[vs:ve])
            self.fft_major_peak = self.bin_num * self.sample_rate / self.n_fft
            peak_mag = spectrum[self.bin_num] / 16.0
            self.fft_magnitude = peak_mag
        else:
            peak_mag = 0.0

        # ── Peak detection ──
        self._peak_history.append(peak_mag)
        if len(self._peak_history) > 200:
            self._peak_history.pop(0)

        self.sample_peak = False
        if len(self._peak_history) > 10:
            sh = sorted(self._peak_history)
            self._peak_threshold = sh[int(len(sh) * 0.6)]

        now = time.time()
        if (self.volume_smth > 1 and self._peak_threshold > 0 and
                self.bin_num > 4 and peak_mag > self._peak_threshold and
                (now - self._last_peak_time) > 0.1):
            self.sample_peak = True
            self._last_peak_time = now


# ── 12 Visual Effects ────────────────────────────────────────────────

class _Juggles:
    """Colored dots juggling, brightness = volume."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.t = 0
    def render(self, dt):
        self.t += dt
        frame = np.zeros((self.n, 3), dtype=np.float64)
        vol = self.audio.volume_norm / 255.0
        for i in range(8):
            speed = 1 + i * 0.7
            pos = int((np.sin(self.t * speed + i * 1.5) * 0.5 + 0.5) * (self.n - 1))
            pos = np.clip(pos, 0, self.n - 1)
            r, g, b = _hsv(i / 8.0, 1.0, vol)
            frame[pos] = np.minimum(frame[pos] + [r, g, b], 255)
        return frame.astype(np.uint8)


class _Midnoise:
    """Noise pattern colored by dominant frequency."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.t = 0
    def render(self, dt):
        self.t += dt
        frame = np.zeros((self.n, 3), dtype=np.uint8)
        vol = self.audio.volume_norm / 255.0
        hue = _freq_to_hue(self.audio.fft_major_peak)
        for i in range(self.n):
            nv = _noise1d(i * 0.15 + self.t * 2)
            r, g, b = _hsv(hue + nv * 0.2, 1.0, nv * vol)
            frame[i] = [r, g, b]
        return frame


class _Noisemeter:
    """VU bar + peak flash."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.flash = 0.0
    def render(self, dt):
        frame = np.zeros((self.n, 3), dtype=np.uint8)
        vol = self.audio.volume_norm / 255.0
        fill = int(vol * self.n)
        for i in range(min(fill, self.n)):
            r, g, b = _hsv(0.33 - (i / self.n) * 0.33, 1.0, 1.0)
            frame[i] = [r, g, b]
        if self.audio.sample_peak:
            self.flash = 1.0
        if self.flash > 0.01:
            f = int(self.flash * 255)
            frame[:] = np.maximum(frame, [f, f, f])
            self.flash *= 0.82
        return frame


class _Plasmoid:
    """Plasma blobs modulated by volume, flash on peak."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.t = 0
    def render(self, dt):
        vol = self.audio.volume_norm / 255.0
        self.t += dt * (1 + vol * 3)
        frame = np.zeros((self.n, 3), dtype=np.uint8)
        for i in range(self.n):
            x = i / self.n
            p = (np.sin(x * 10 + self.t * 3) + np.sin(x * 7 - self.t * 2) +
                 np.sin(x * 3 + self.t * 5)) / 3.0 * 0.5 + 0.5
            r, g, b = _hsv(p * 0.8, 1.0, vol * p)
            frame[i] = [r, g, b]
        if self.audio.sample_peak:
            frame[:] = np.clip(frame.astype(int) + 80, 0, 255).astype(np.uint8)
        return frame


class _Blurz:
    """Blurred frequency bands."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.prev = np.zeros((n, 3), dtype=np.float64)
    def render(self, dt):
        frame = np.zeros((self.n, 3), dtype=np.float64)
        bw = self.n / 16.0
        for b in range(16):
            s, e = int(b * bw), min(int((b + 1) * bw), self.n)
            amp = self.audio.fft_result[b] / 255.0
            r, g, b_ = _hsv(b / 16.0, 1.0, amp)
            frame[s:e] = [r, g, b_]
        # Spatial blur
        blurred = frame.copy()
        blurred[1:-1] = (frame[:-2] + frame[1:-1] * 2 + frame[2:]) / 4.0
        # Temporal blend
        blurred = blurred * 0.7 + self.prev * 0.3
        self.prev = blurred.copy()
        return np.clip(blurred, 0, 255).astype(np.uint8)


class _DJLight:
    """All LEDs colored by dominant frequency band."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.hue_smooth = 0.0
    def render(self, dt):
        dom = int(np.argmax(self.audio.fft_result))
        target_hue = dom / 16.0
        self.hue_smooth = self.hue_smooth * 0.8 + target_hue * 0.2
        vol = self.audio.volume_norm / 255.0
        frame = np.zeros((self.n, 3), dtype=np.uint8)
        for i in range(self.n):
            pv = _noise1d(i * 0.3) * 0.3 + 0.7
            r, g, b = _hsv(self.hue_smooth, 1.0, vol * pv)
            frame[i] = [r, g, b]
        return frame


class _Freqmap:
    """Dot at position mapped from dominant frequency."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.pos_sm = 0.5
    def render(self, dt):
        pos = _freq_to_hue(self.audio.fft_major_peak)
        self.pos_sm = self.pos_sm * 0.7 + pos * 0.3
        led = int(self.pos_sm * (self.n - 1))
        vol = self.audio.volume_norm / 255.0
        frame = np.zeros((self.n, 3), dtype=np.uint8)
        for i in range(self.n):
            d = abs(i - led)
            if d < 3:
                br = vol * max(0, 1 - d / 3.0)
                r, g, b = _hsv(self.pos_sm, 1.0, br)
                frame[i] = [r, g, b]
        return frame


class _Freqmatrix:
    """Scrolling trail colored by dominant frequency."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.buf = np.zeros((n, 3), dtype=np.float64)
    def render(self, dt):
        self.buf[1:] = self.buf[:-1]
        hue = _freq_to_hue(self.audio.fft_major_peak)
        vol = self.audio.volume_norm / 255.0
        r, g, b = _hsv(hue, 1.0, vol)
        self.buf[0] = [r, g, b]
        return self.buf.astype(np.uint8)


class _Freqpixels:
    """Random sparkle pixels colored by dominant frequency."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.px = np.zeros((n, 3), dtype=np.float64)
        self.rng = np.random.RandomState(42)
    def render(self, dt):
        self.px *= 0.90
        vol = self.audio.volume_norm / 255.0
        hue = _freq_to_hue(self.audio.fft_major_peak)
        for _ in range(max(1, int(vol * 3))):
            p = self.rng.randint(0, self.n)
            r, g, b = _hsv(hue + self.rng.uniform(-0.05, 0.05), 1.0, vol)
            self.px[p] = np.maximum(self.px[p], [r, g, b])
        return np.clip(self.px, 0, 255).astype(np.uint8)


class _Freqwave:
    """Center-outward ripples colored by frequency."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.waves = []  # [radius, (r,g,b), intensity]
        self.last_vol = 0.0
    def render(self, dt):
        frame = np.zeros((self.n, 3), dtype=np.float64)
        center = self.n // 2
        vol = self.audio.volume_norm / 255.0
        # Spawn wave on rising volume
        if vol > 0.3 and vol > self.last_vol + 0.05 and (
                not self.waves or self.waves[-1][0] > 2):
            hue = _freq_to_hue(self.audio.fft_major_peak)
            r, g, b = _hsv(hue, 1.0, 1.0)
            self.waves.append([0.0, np.array([r, g, b], dtype=np.float64), vol])
        self.last_vol = vol
        alive = []
        for w in self.waves:
            w[0] += dt * 25
            fade = w[2] * max(0, 1 - w[0] / (self.n / 2))
            if fade > 0.01:
                r = int(w[0])
                for off in range(-1, 2):
                    for side in [center - r + off, center + r + off]:
                        if 0 <= side < self.n:
                            frame[side] = np.maximum(frame[side], w[1] * fade)
                alive.append(w)
        self.waves = alive[-12:]
        return np.clip(frame, 0, 255).astype(np.uint8)


class _Noisemove:
    """Noise texture shifted by FFT band energies."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
        self.t = 0
        self.offsets = np.zeros(16)
    def render(self, dt):
        self.t += dt
        for i in range(16):
            self.offsets[i] += self.audio.fft_result[i] / 255.0 * dt * 5
        frame = np.zeros((self.n, 3), dtype=np.uint8)
        for i in range(self.n):
            b = min(int(i / self.n * 16), 15)
            nv = _noise1d(i * 0.2 + self.offsets[b] + self.t)
            amp = self.audio.fft_result[b] / 255.0
            r, g, b_ = _hsv(b / 16.0 + nv * 0.1, 1.0, nv * amp)
            frame[i] = [r, g, b_]
        return frame


OCTAVE_COLORS = [
    (255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0),
    (0, 255, 0), (0, 255, 128), (0, 255, 255), (0, 128, 255),
    (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128),
]

class _Rocktaves:
    """Color from musical octave of dominant frequency."""
    def __init__(self, n, audio):
        self.n, self.audio = n, audio
    def render(self, dt):
        freq = max(self.audio.fft_major_peak, 20)
        note = 12 * np.log2(freq / 440.0) + 69
        color = np.array(OCTAVE_COLORS[int(note) % 12], dtype=np.float64)
        vol = self.audio.volume_norm / 255.0
        frame = np.zeros((self.n, 3), dtype=np.uint8)
        frame[:] = (color * vol).astype(np.uint8)
        return frame


# ── Combined Effect ──────────────────────────────────────────────────

EFFECT_ORDER = [
    ('Juggles',    _Juggles),
    ('Midnoise',   _Midnoise),
    ('Noisemeter', _Noisemeter),
    ('Plasmoid',   _Plasmoid),
    ('Blurz',      _Blurz),
    ('DJLight',    _DJLight),
    ('Freqmap',    _Freqmap),
    ('Freqmatrix', _Freqmatrix),
    ('Freqpixels', _Freqpixels),
    ('Freqwave',   _Freqwave),
    ('Noisemove',  _Noisemove),
    ('Rocktaves',  _Rocktaves),
]


class WLEDAllEffects(AudioReactiveEffect):
    """All 12 WLED 1D audio effects, stacked vertically."""

    def __init__(self, num_leds, sample_rate=44100):
        super().__init__(num_leds, sample_rate)
        self.leds_per = num_leds // 12
        self.audio_state = WLEDAudioState(sample_rate)
        self.effects = [(name, cls(self.leds_per, self.audio_state))
                        for name, cls in EFFECT_ORDER]

    @property
    def name(self):
        return "WLED All 12 Effects"

    @property
    def description(self):
        return "All 12 WLED 1D audio effects stacked vertically: Juggles, Midnoise, Noisemeter, Plasmoid, Blurz, DJLight, Freqmap, Freqmatrix, Freqpixels, Freqwave, Noisemove, Rocktaves."

    def process_audio(self, mono_chunk):
        self.audio_state.process(mono_chunk)

    def render(self, dt):
        frames = [fx.render(dt) for _, fx in self.effects]
        return np.vstack(frames)

    def get_diagnostics(self):
        return {
            'vol': f'{self.audio_state.volume_norm:.0f}',
            'freq': f'{self.audio_state.fft_major_peak:.0f}Hz',
            'peak': 'BEAT!' if self.audio_state.sample_peak else '',
            'bin': self.audio_state.bin_num,
        }
