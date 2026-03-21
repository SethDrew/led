#!/usr/bin/env python3
"""
Centroid Blue Dual — simultaneous effects on diamond + 150-LED strip.

Diamond (port 11230, 80 physical LEDs, height-mode topology):
  Spectral centroid → grayscale (dark sound) to saturated blue (bright sound).
  Height gradient: bottom more gray, top more blue, shifted by centroid.
  Auto-adaptive range learns centroid center and spread (sticky min/max).

Strip (port 11240, 150 LEDs):
  50% chroma blue, brightness 10-60 from audio energy.

Both use standard streaming serial protocol:
  [0xFF][0xAA][N×3 RGB bytes][XOR checksum] at 1 Mbps.

No gamma correction here — LEDs have native ~gamma 2.2 response.
Raw byte values are sent directly.
"""

import sys
import os
import time
import threading
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import (SerialLEDOutput, find_blackhole_device,
                    load_sculpture, apply_topology)

# ── Audio ────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
N_FFT = 2048
LED_FPS = 30

# ── Hardware ─────────────────────────────────────────────────────────
DIAMOND_PORT = '/dev/cu.usbserial-11230'
DIAMOND_PHYSICAL = 80

STRIP_PORT = '/dev/cu.usbserial-11240'
STRIP_LEDS = 150

# ── Color definitions (raw byte values at max brightness 255) ────────
# Pure blue from OKLCH LUT index ~200 (deep blue, minimal green)
FULL_BLUE = np.array([25.0, 1.0, 248.0])
# Gray at matched OKLCH lightness (~L=0.38 for this blue)
GRAY = np.array([50.0, 50.0, 50.0])
# 50% chroma blue: halfway between full blue and matched gray
HALF_CHROMA_BLUE = (FULL_BLUE + GRAY) / 2.0  # ≈ (37, 25, 149)

# Brightness range (raw byte scale, 0-255)
BRIGHT_MIN = 2
BRIGHT_MAX = 25

# RMS gate: below this, centroid is unreliable (silence/noise)
RMS_GATE = 0.0005

# Energy peak decay: 0.998 at 30fps ≈ 11.5s half-life
PEAK_DECAY = 0.998


class AdaptiveRange:
    """Auto-adaptive signal normalization (gyro_color_stick.cpp pattern).

    Warmup (~2s): fast tracking to learn the signal's natural range.
    Steady-state: sticky min/max (fast expand, slow contract).
    """

    WARMUP_FRAMES = 60

    def __init__(self, tc_range=20.0, fps=30.0):
        self.lo = None
        self.hi = None
        self.count = 0
        self.alpha_expand = 2.0 / (tc_range * 0.3 * fps + 1.0)
        self.alpha_contract = 2.0 / (tc_range * fps + 1.0)

    def update(self, value):
        self.count += 1

        if self.lo is None:
            self.lo = value * 0.7
            self.hi = value * 1.3 + 1.0
            return 0.5

        if self.count < self.WARMUP_FRAMES:
            alpha = 0.15
            if value < self.lo:
                self.lo += alpha * (value - self.lo)
            else:
                self.lo += 0.02 * (value - self.lo)
            if value > self.hi:
                self.hi += alpha * (value - self.hi)
            else:
                self.hi += 0.02 * (value - self.hi)
        else:
            if value < self.lo:
                self.lo += self.alpha_expand * (value - self.lo)
            else:
                self.lo += self.alpha_contract * (value - self.lo)
            if value > self.hi:
                self.hi += self.alpha_expand * (value - self.hi)
            else:
                self.hi += self.alpha_contract * (value - self.hi)

        span = self.hi - self.lo
        if span < 1e-10:
            return 0.5
        return np.clip((value - self.lo) / span, 0.0, 1.0)


def color_at_brightness(color_255, brightness):
    """Scale a color (defined at max=255) to a target brightness level.

    brightness: target max-channel value (e.g., 10-60).
    Preserves hue/chroma ratios. No gamma — raw bytes for WS2812B.
    """
    max_ch = np.max(color_255)
    if max_ch < 1:
        return np.zeros(3, dtype=np.uint8)
    scaled = color_255 * (brightness / max_ch)
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def main():
    sculpture_def, _ = load_sculpture('cob_diamond')
    diamond_logical = sculpture_def['logical_leds']
    print(f"  Diamond: {diamond_logical} logical → {DIAMOND_PHYSICAL} physical LEDs")
    print(f"  Strip:   {STRIP_LEDS} LEDs")

    print(f"\n  Connecting...")
    diamond_out = SerialLEDOutput(DIAMOND_PORT, DIAMOND_PHYSICAL)
    strip_out = SerialLEDOutput(STRIP_PORT, STRIP_LEDS)

    bh = find_blackhole_device()
    if bh is None:
        print("  Error: BlackHole audio device not found")
        sys.exit(1)
    print(f"  Audio: {sd.query_devices(bh)['name']}")

    # ── Audio state ──────────────────────────────────────────────────
    window = np.hanning(N_FFT).astype(np.float32)
    freq_bins = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    audio_buf = np.zeros(N_FFT, dtype=np.float32)
    buf_pos = [0]

    lock = threading.Lock()
    features = {'centroid': 0.0, 'rms': 0.0}

    def audio_callback(indata, frames, time_info, status):
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        rms = float(np.sqrt(np.mean(mono ** 2)))

        n = len(mono)
        pos = buf_pos[0]
        space = N_FFT - pos

        if n < space:
            audio_buf[pos:pos + n] = mono
            buf_pos[0] = pos + n
            with lock:
                features['rms'] = rms
            return

        audio_buf[pos:] = mono[:space]
        windowed = audio_buf * window
        spec = np.abs(np.fft.rfft(windowed))

        total = np.sum(spec)
        centroid = float(np.sum(freq_bins * spec) / total) if total > 1e-10 else 0.0

        leftover = n - space
        if leftover > 0:
            audio_buf[:leftover] = mono[space:]
        buf_pos[0] = leftover

        with lock:
            features['centroid'] = centroid
            features['rms'] = rms

    # ── Signal processing state ──────────────────────────────────────
    centroid_ar = AdaptiveRange(tc_range=20.0, fps=LED_FPS)
    energy_peak = 1e-8

    centroid_smooth = 0.5
    energy_smooth = 0.0

    height = np.linspace(0.0, 1.0, diamond_logical)

    stream = sd.InputStream(
        device=bh, channels=2, samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE, callback=audio_callback
    )

    print("  Listening... Play music through BlackHole. Ctrl+C to stop.\n")

    try:
        stream.start()
        dt = 1.0 / LED_FPS
        next_time = time.time()
        frame_num = 0

        while True:
            with lock:
                raw_centroid = features['centroid']
                raw_rms = features['rms']

            # ── Centroid: gate by RMS ────────────────────────────────
            if raw_rms > RMS_GATE:
                cn = centroid_ar.update(raw_centroid)
            else:
                cn = 0.5

            # ── Energy: peak-decay + sqrt compression ────────────────
            energy_peak = max(raw_rms, energy_peak * PEAK_DECAY)
            ratio = raw_rms / max(energy_peak, 1e-8)
            en = ratio ** 0.7  # power compression: 0.1 → 0.20, 0.5 → 0.62

            # Asymmetric EMA
            a = 0.25 if cn > centroid_smooth else 0.04
            centroid_smooth += a * (cn - centroid_smooth)

            a = 0.3 if en > energy_smooth else 0.05
            energy_smooth += a * (en - energy_smooth)

            # Brightness in raw byte range
            bright = BRIGHT_MIN + energy_smooth * (BRIGHT_MAX - BRIGHT_MIN)

            # ── DIAMOND ──────────────────────────────────────────────
            # Height gradient: sat = centroid * (0.3 + 0.7*height)
            #   centroid=0: all gray
            #   centroid=1: bottom 30% sat, top 100% sat
            sat = centroid_smooth * (0.3 + 0.7 * height)

            # Per-pixel color at target brightness (no gamma)
            frame = np.zeros((diamond_logical, 3), dtype=np.uint8)
            for i in range(diamond_logical):
                s = sat[i]
                color = GRAY * (1.0 - s) + FULL_BLUE * s
                frame[i] = color_at_brightness(color, bright)

            physical = apply_topology(frame, sculpture_def)
            diamond_out.send_frame(physical)

            # ── STRIP ────────────────────────────────────────────────
            strip_pixel = color_at_brightness(HALF_CHROMA_BLUE, bright)
            strip_frame = np.tile(strip_pixel, (STRIP_LEDS, 1))
            strip_out.send_frame(strip_frame)

            # ── Diagnostics ──────────────────────────────────────────
            if frame_num % 10 == 0:
                lo = centroid_ar.lo or 0
                hi = centroid_ar.hi or 0
                gate = '*' if raw_rms > RMS_GATE else '.'
                # Show actual pixel values being sent
                sp = strip_pixel
                sys.stdout.write(
                    f'\r  {gate} cent={raw_centroid:5.0f} '
                    f'[{lo:.0f}-{hi:.0f}] '
                    f'sat={centroid_smooth:.2f}  '
                    f'en={energy_smooth:.2f} bright={bright:.0f} '
                    f'strip=({sp[0]},{sp[1]},{sp[2]})   '
                )
                sys.stdout.flush()

            frame_num += 1
            next_time += dt
            sleep = next_time - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_time = time.time()

    except KeyboardInterrupt:
        print("\n\n  Stopping...")
    finally:
        stream.stop()
        stream.close()
        diamond_out.close()
        strip_out.close()
        print("  Done!")


if __name__ == '__main__':
    main()
