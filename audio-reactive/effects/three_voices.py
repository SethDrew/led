"""
Three Voices — depth-mapped audio-reactive LED effect for the LED tree.

Three independent audio layers composited additively:
  Voice 1 (Bass Foundation): sub-bass + bass energy → bottom of tree, warm red/orange
  Voice 2 (Harmonic Body):   streaming HPSS harmonic → full tree, cool blue/purple
  Voice 3 (Percussive Flash): HPSS percussive peaks → white flashes with upward decay

Designed for fa_br_drop1.wav (Fred again..) but generalizes to any music with
clear bass/harmonic/percussive separation.

The key insight: no section detection needed. The three voices respond independently
to the audio, and section changes emerge naturally because the music itself changes.
During a build (bass drops out, mids rise), the bottom of the tree goes dark while
the upper tree brightens — no special logic required.
"""

import numpy as np
import threading
from base import AudioReactiveEffect

# ── Tree depth map ──────────────────────────────────────────────────
# Reproduces TreeTopology.h node depths for 197 LEDs.
# Node index → depth (0-70). This is the spatial dimension for effects.

def _build_depth_map():
    """Reproduce TreeTopology.h node layout. Firmware declares nodes[197]
    so we generate exactly 197 entries (indices 0-196)."""
    depths = []
    # Strip 1: Lower trunk (0-38), Branch A (38-61), Branch B (62-91)
    for i in range(39):          # 39 nodes: depth 0-38
        depths.append(i)
    for i in range(38, 62):      # 24 nodes: Branch A, depth 38-61
        depths.append(i)
    for i in range(62, 92):      # 30 nodes: Branch B, depth 38-67
        depths.append(38 + (i - 62))
    # Strip 2: Branch C (6 LEDs, depth 25-30)
    for i in range(6):           # 6 nodes
        depths.append(25 + i)
    # Strip 3: Upper trunk (0-70), Branch D (71-72), Branch E (73-98)
    for i in range(71):          # 71 nodes: depth 0-70
        depths.append(i)
    for i in range(71, 73):      # 2 nodes: Branch D, depth 43-44
        depths.append(43 + (i - 71))
    for i in range(73, 99):      # 26 nodes: Branch E, depth 43-68
        depths.append(43 + (i - 73))
    # Firmware nodes[197] only fits 0-196, trim to match serial protocol
    return np.array(depths[:197], dtype=np.float32)

TREE_DEPTHS = _build_depth_map()  # 197 elements, values 0-70
MAX_DEPTH = 70.0


class ThreeVoicesEffect(AudioReactiveEffect):
    """Three-layer depth-mapped effect: bass foundation + harmonic body + percussive flash."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        # FFT parameters
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)

        # Frequency band masks
        self.bass_mask = (self.freq_bins >= 20) & (self.freq_bins <= 250)
        self.mid_mask = (self.freq_bins >= 250) & (self.freq_bins <= 6000)

        # Audio accumulation buffer (collect samples until we have n_fft)
        self.audio_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.audio_buf_pos = 0

        # Streaming HPSS: circular buffer of magnitude spectra
        self.hpss_buf_size = 17  # ~400ms at 23ms/frame
        self.spec_buf = np.zeros((self.hpss_buf_size, self.n_fft // 2 + 1), dtype=np.float32)
        self.spec_buf_idx = 0
        self.spec_buf_filled = 0

        # Slow-decay peak trackers for normalization.
        # Fast attack (instant), slow release (~10s half-life).
        # This preserves absolute energy relationships: when bass drops out,
        # the peak stays high so normalized value → 0.
        self.peak_decay = 0.998  # per FFT frame (~46ms), half-life ~16s

        # Voice 1: Bass energy
        self.bass_energy = 0.0
        self.bass_smooth = 0.0
        self.bass_peak = 1e-10  # slow-decay peak tracker

        # Voice 2: Harmonic energy
        self.harmonic_energy = 0.0
        self.harmonic_smooth = 0.0
        self.harmonic_peak = 1e-10

        # Voice 3: Percussive flash state
        self.perc_energy = 0.0
        self.perc_peak = 1e-10
        self.perc_history = np.zeros(64, dtype=np.float32)  # short window for threshold
        self.perc_hist_idx = 0
        self.flash_brightness = 0.0
        self.flash_depth = 35.0  # where the flash originates
        self.last_flash_time = 0.0
        self.time_acc = 0.0
        self.flash_cooldown = 0.08  # 80ms between flashes

        # Depth map (use tree depths if 197 LEDs, otherwise linear)
        if num_leds == 197:
            self.depths = TREE_DEPTHS.copy()
        else:
            self.depths = np.linspace(0, MAX_DEPTH, num_leds).astype(np.float32)
        self.norm_depths = self.depths / MAX_DEPTH  # 0-1 range

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Three Voices"

    def process_audio(self, mono_chunk: np.ndarray):
        # Accumulate samples
        n = len(mono_chunk)
        space = self.n_fft - self.audio_buf_pos
        if n < space:
            self.audio_buf[self.audio_buf_pos:self.audio_buf_pos + n] = mono_chunk
            self.audio_buf_pos += n
            return
        # Fill remaining space and process
        self.audio_buf[self.audio_buf_pos:] = mono_chunk[:space]
        self._process_frame(self.audio_buf.copy())
        # Start next buffer with leftover
        leftover = n - space
        self.audio_buf[:leftover] = mono_chunk[space:]
        self.audio_buf_pos = leftover

    def _process_frame(self, frame):
        """Process one full FFT frame."""
        windowed = frame * self.window
        spec = np.abs(np.fft.rfft(windowed))

        # Add to HPSS circular buffer
        self.spec_buf[self.spec_buf_idx] = spec
        self.spec_buf_idx = (self.spec_buf_idx + 1) % self.hpss_buf_size
        self.spec_buf_filled = min(self.spec_buf_filled + 1, self.hpss_buf_size)

        # Streaming HPSS: harmonic = median across time at each freq bin
        if self.spec_buf_filled >= 3:
            buf_slice = self.spec_buf[:self.spec_buf_filled]
            harmonic_mask = np.median(buf_slice, axis=0)
            percussive = np.maximum(spec - harmonic_mask, 0)
        else:
            harmonic_mask = spec * 0.5
            percussive = spec * 0.5

        # Voice 1: Bass energy (sub-bass + bass bands)
        bass_e = np.sum(spec[self.bass_mask] ** 2)

        # Voice 2: Harmonic mid energy (from HPSS harmonic component)
        harm_e = np.sum(harmonic_mask[self.mid_mask] ** 2)

        # Voice 3: Percussive total energy
        perc_e = np.sum(percussive ** 2)

        # Update slow-decay peak trackers (fast attack, slow release)
        self.bass_peak = max(bass_e, self.bass_peak * self.peak_decay)
        self.harmonic_peak = max(harm_e, self.harmonic_peak * self.peak_decay)
        self.perc_peak = max(perc_e, self.perc_peak * self.peak_decay)

        # Percussive peak detection: short-window stats for threshold
        self.perc_history[self.perc_hist_idx] = perc_e
        self.perc_hist_idx = (self.perc_hist_idx + 1) % len(self.perc_history)
        perc_mean = np.mean(self.perc_history)
        perc_std = np.std(self.perc_history)

        with self._lock:
            # Voice 1: normalized bass — when bass drops out, peak stays high → ratio → 0
            self.bass_energy = min(1.0, bass_e / self.bass_peak)

            # Voice 2: normalized harmonic
            self.harmonic_energy = min(1.0, harm_e / self.harmonic_peak)

            # Voice 3: flash on percussive peaks exceeding threshold
            perc_threshold = perc_mean + 1.5 * perc_std
            if perc_e > perc_threshold and perc_threshold > 0:
                self.perc_energy = min(1.0, perc_e / self.perc_peak)

    def render(self, dt: float) -> np.ndarray:
        self.time_acc += dt
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        with self._lock:
            bass = self.bass_energy
            harmonic = self.harmonic_energy
            perc = self.perc_energy
            self.perc_energy = 0.0  # consume the trigger

        # Smooth bass and harmonic (attack fast, decay slow)
        attack_bass = 0.3
        decay_bass = 0.05
        alpha_bass = attack_bass if bass > self.bass_smooth else decay_bass
        self.bass_smooth += alpha_bass * (bass - self.bass_smooth)

        attack_harm = 0.2
        decay_harm = 0.03
        alpha_harm = attack_harm if harmonic > self.harmonic_smooth else decay_harm
        self.harmonic_smooth += alpha_harm * (harmonic - self.harmonic_smooth)

        # ── Voice 1: Bass Foundation ──
        # Bottom of tree glows warm red/orange, intensity = bass energy
        # Spatial falloff: full brightness at depth 0, fades to zero at depth ~30
        bass_spatial = np.clip(1.0 - self.norm_depths * 3.0, 0, 1)  # depth 0-23 = lit
        bass_brightness = self.bass_smooth ** 0.7  # slight gamma for visual pop
        bass_r = bass_spatial * bass_brightness
        bass_g = bass_spatial * bass_brightness * 0.3  # warm orange tint
        frame[:, 0] += bass_r * 255
        frame[:, 1] += bass_g * 255

        # ── Voice 2: Harmonic Body ──
        # Full tree, weighted toward upper half, cool blue/purple
        harm_spatial = np.clip(self.norm_depths * 1.5 - 0.15, 0, 1)  # fades in from depth ~10
        harm_brightness = self.harmonic_smooth ** 0.8
        frame[:, 0] += harm_spatial * harm_brightness * 60   # slight purple warmth
        frame[:, 1] += harm_spatial * harm_brightness * 20
        frame[:, 2] += harm_spatial * harm_brightness * 200  # dominant blue

        # ── Voice 3: Percussive Flash ──
        # Trigger new flash on percussive peak
        if perc > 0 and (self.time_acc - self.last_flash_time) > self.flash_cooldown:
            self.flash_brightness = min(1.0, perc)
            self.flash_depth = np.random.uniform(10, 60)  # random origin point
            self.last_flash_time = self.time_acc

        # Decay existing flash
        if self.flash_brightness > 0.01:
            # Flash: white, centered at flash_depth, Gaussian spatial falloff
            dist = np.abs(self.depths - self.flash_depth)
            flash_radius = 8.0 + (1.0 - self.flash_brightness) * 15.0  # expands as it fades
            flash_spatial = np.exp(-0.5 * (dist / flash_radius) ** 2)
            flash_val = flash_spatial * self.flash_brightness * 255

            frame[:, 0] += flash_val
            frame[:, 1] += flash_val
            frame[:, 2] += flash_val

            # Fast exponential decay
            self.flash_brightness *= 0.85 ** (dt * 30)  # ~150ms decay

        # Clamp and convert
        np.clip(frame, 0, 255, out=frame)
        return frame.astype(np.uint8)

    def get_diagnostics(self) -> dict:
        with self._lock:
            return {
                'bass': f'{self.bass_smooth:.2f}',
                'harmonic': f'{self.harmonic_smooth:.2f}',
                'flash': f'{self.flash_brightness:.2f}',
            }
