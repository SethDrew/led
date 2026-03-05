"""
Auto Lightshow — section-aware effect for whole-song playback.

Detects energy trajectory (building/sustaining/breaking down) and kick
presence to classify the current section, then switches between visual
behaviors on the diamond sculpture's height-mapped topology.

Section detection (simple heuristics, no ML):
  - Rolling integral of RMS over 10s window → energy level
  - Slope of that integral over 3s → building vs declining
  - Low-freq energy ratio (40-200Hz / total) → kick presence
  - Onset density over 1s → rhythmic activity

Visual behaviors:
  BREAKDOWN  — slow breathing, dim, cool purple/blue, gentle bottom-up glow
  BUILDING   — rising brightness, warm amber climbing from base, pulse quickens
  DROP       — full brightness, fast kick-reactive pulses, vivid magenta/white,
               energy radiates from peak downward

Architecture placement:
  Axis 3:   Rolling integral slope + band energy ratio
  Axis 4:   Section-driven behavior switching
  Axis 5:   Phrase-level (~3-10s section detection window)
  Axis 7:   Energy → brightness + section → color palette
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


# ── Color palettes per section ──────────────────────────────────────

# Each palette: list of (R, G, B) color stops, sampled by height position
PALETTE_BREAKDOWN = np.array([
    [15,  5,  40],    # deep indigo at base
    [30,  10, 80],    # dark purple
    [50,  20, 120],   # muted violet at peak
], dtype=np.float32)

PALETTE_BUILDING = np.array([
    [40,  10, 5],     # dark ember at base
    [180, 60, 0],     # warm orange
    [255, 140, 20],   # bright amber at peak
], dtype=np.float32)

PALETTE_DROP = np.array([
    [200, 0,  80],    # magenta at base
    [255, 20, 120],   # hot pink
    [255, 200, 255],  # near-white at peak
], dtype=np.float32)


def _sample_gradient(palette, t):
    """Sample color from Nx3 palette at position t (0-1)."""
    t = np.clip(t, 0.0, 1.0)
    n = len(palette) - 1
    idx = t * n
    lo = int(idx)
    hi = min(lo + 1, n)
    frac = idx - lo
    return palette[lo] * (1 - frac) + palette[hi] * frac


class AutoLightshowEffect(AudioReactiveEffect):
    """Section-aware lightshow for whole-song playback on diamond sculpture."""

    registry_name = 'auto_lightshow'

    # ── Section states ──
    BREAKDOWN = 0
    BUILDING = 1
    DROP = 2

    SECTION_NAMES = ['BREAKDOWN', 'BUILDING', 'DROP']

    source_features = [
        {'id': 'energy_level', 'label': 'Energy Level', 'color': '#e94560'},
        {'id': 'energy_slope', 'label': 'Energy Slope', 'color': '#ffd740'},
        {'id': 'bass_ratio', 'label': 'Bass Ratio', 'color': '#4ca5ff'},
        {'id': 'onset_density', 'label': 'Onset Density', 'color': '#9c27b0'},
    ]

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        super().__init__(num_leds, sample_rate)

        self.accum = OverlapFrameAccumulator()

        # ── Energy level: rolling RMS integral (10s window) ──
        self.chunk_rate = sample_rate / 512  # ~86 Hz (hop-based)
        self.energy_window_sec = 10.0
        self.energy_ring_size = int(self.energy_window_sec * self.chunk_rate)
        self.energy_ring = np.zeros(self.energy_ring_size, dtype=np.float32)
        self.energy_ring_pos = 0
        self.energy_sum = 0.0

        # Peak-decay normalization for energy
        self.energy_ceiling = 1e-10
        self.energy_floor = 0.0
        self.ceiling_decay = 0.9995
        self.floor_rise = 0.9995

        # ── Energy slope: derivative of rolling integral (3s window) ──
        self.slope_window_sec = 3.0
        self.slope_ring_size = int(self.slope_window_sec * self.chunk_rate)
        self.slope_ring = np.zeros(self.slope_ring_size, dtype=np.float32)
        self.slope_ring_pos = 0

        # ── Bass energy ratio (40-200 Hz vs total) ──
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        self.bass_mask = (freqs >= 40) & (freqs <= 200)
        self.bass_ratio_ema = 0.0
        self.bass_ema_alpha = 0.05  # ~20 frames to converge

        # ── Onset density (1s window) ──
        self.onset_window_sec = 1.0
        self.onset_ring_size = int(self.onset_window_sec * self.chunk_rate)
        self.onset_ring = np.zeros(self.onset_ring_size, dtype=np.float32)
        self.onset_ring_pos = 0
        self.prev_rms = 0.0

        # ── Section classification state ──
        self.current_section = self.BREAKDOWN
        self.section_confidence = 0.0
        self.section_hold_frames = 0  # minimum hold before switching
        self.section_hold_min = int(2.0 * self.chunk_rate)  # 2s minimum

        # ── Visual state (read by render thread) ──
        self._lock = threading.Lock()
        self._energy_norm = 0.0
        self._energy_slope = 0.0
        self._bass_ratio = 0.0
        self._onset_density = 0.0
        self._section = self.BREAKDOWN
        self._kick_flash = 0.0  # short-lived kick flash intensity

        # ── Render state (main thread only) ──
        self.smooth_brightness = 0.0
        self.smooth_section_t = 0.0  # 0=breakdown, 0.5=building, 1=drop
        self.pulse_phase = 0.0
        self.kick_decay = 0.0
        self.height_offset = 0.0  # animated height sweep

    @property
    def name(self):
        return "Auto Lightshow"

    @property
    def description(self):
        return "Section-aware lightshow: detects builds/drops/breakdowns and switches visual behavior."

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame):
        rms = float(np.sqrt(np.mean(frame ** 2)))

        # ── Update rolling energy integral ──
        idx = self.energy_ring_pos % self.energy_ring_size
        self.energy_sum -= self.energy_ring[idx]
        self.energy_ring[idx] = rms
        self.energy_sum += rms
        self.energy_ring_pos += 1

        energy = max(self.energy_sum, 0.0)

        # Min-max normalization with slow adaptation
        self.energy_ceiling = max(energy, self.energy_ceiling * self.ceiling_decay)
        if energy < self.energy_floor:
            self.energy_floor = energy
        else:
            self.energy_floor += (energy - self.energy_floor) * (1 - self.floor_rise)

        span = self.energy_ceiling - self.energy_floor
        energy_norm = (energy - self.energy_floor) / span if span > 1e-10 else 0.0
        energy_norm = float(np.clip(energy_norm, 0.0, 1.0))

        # ── Update slope ring (stores energy_norm history) ──
        slope_idx = self.slope_ring_pos % self.slope_ring_size
        self.slope_ring[slope_idx] = energy_norm
        self.slope_ring_pos += 1

        # Compute slope: difference between recent and old energy
        filled = min(self.slope_ring_pos, self.slope_ring_size)
        if filled >= 2:
            # Compare most recent quarter vs oldest quarter of the window
            quarter = max(1, filled // 4)
            recent_start = (self.slope_ring_pos - quarter) % self.slope_ring_size
            old_start = (self.slope_ring_pos - filled) % self.slope_ring_size

            # Get recent and old averages
            recent_vals = []
            old_vals = []
            for i in range(quarter):
                recent_vals.append(self.slope_ring[(recent_start + i) % self.slope_ring_size])
                old_vals.append(self.slope_ring[(old_start + i) % self.slope_ring_size])
            energy_slope = np.mean(recent_vals) - np.mean(old_vals)
        else:
            energy_slope = 0.0

        # ── Bass energy ratio ──
        spectrum = np.abs(np.fft.rfft(frame * self.window))
        power = spectrum ** 2
        total_power = np.sum(power)
        bass_power = np.sum(power[self.bass_mask])
        if total_power > 1e-10:
            bass_ratio_raw = bass_power / total_power
        else:
            bass_ratio_raw = 0.0
        self.bass_ratio_ema += (bass_ratio_raw - self.bass_ratio_ema) * self.bass_ema_alpha

        # ── Onset density (count of positive RMS jumps) ──
        rms_delta = rms - self.prev_rms
        self.prev_rms = rms
        # Store onset strength (half-wave rectified RMS delta)
        onset_val = max(0.0, rms_delta)
        onset_idx = self.onset_ring_pos % self.onset_ring_size
        self.onset_ring[onset_idx] = onset_val
        self.onset_ring_pos += 1
        onset_density = float(np.mean(self.onset_ring[:min(self.onset_ring_pos, self.onset_ring_size)]))

        # ── Kick flash: instantaneous bass transient ──
        kick_flash = 0.0
        if rms_delta > 0 and self.bass_ratio_ema > 0.15:
            kick_flash = min(1.0, rms_delta * 20.0)

        # ── Section classification ──
        new_section = self._classify_section(energy_norm, energy_slope,
                                              self.bass_ratio_ema, onset_density)

        # Hold current section for minimum duration to avoid flicker
        self.section_hold_frames += 1
        if new_section != self.current_section:
            if self.section_hold_frames >= self.section_hold_min:
                self.current_section = new_section
                self.section_hold_frames = 0

        with self._lock:
            self._energy_norm = energy_norm
            self._energy_slope = float(energy_slope)
            self._bass_ratio = float(self.bass_ratio_ema)
            self._onset_density = float(onset_density)
            self._section = self.current_section
            self._kick_flash = kick_flash

    def _classify_section(self, energy, slope, bass_ratio, onset_density):
        """Simple heuristic section classifier.

        Returns BREAKDOWN, BUILDING, or DROP based on current features.
        """
        # DROP: high energy + strong bass + active onsets
        if energy > 0.55 and bass_ratio > 0.12 and onset_density > 0.0005:
            return self.DROP

        # BUILDING: positive slope (energy rising), moderate level
        if slope > 0.08 and energy > 0.25:
            return self.BUILDING

        # Also building if energy is moderate-high and rising slightly
        if slope > 0.03 and energy > 0.40:
            return self.BUILDING

        # BREAKDOWN: low energy or declining energy
        if energy < 0.35 or slope < -0.05:
            return self.BREAKDOWN

        # Default: maintain current if ambiguous, or use energy level
        if energy > 0.50:
            return self.DROP
        elif energy > 0.30:
            return self.BUILDING
        return self.BREAKDOWN

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            energy = self._energy_norm
            section = self._section
            kick = self._kick_flash
            slope = self._energy_slope

        # ── Section-dependent visual parameters ──
        if section == self.BREAKDOWN:
            target_brightness = 0.15 + 0.15 * energy
            pulse_speed = 0.3   # slow breathing
            palette = PALETTE_BREAKDOWN
            kick_strength = 0.1  # minimal kick reaction
            height_sweep_speed = 0.05
        elif section == self.BUILDING:
            target_brightness = 0.3 + 0.4 * energy
            pulse_speed = 0.6 + 0.4 * energy  # accelerating
            palette = PALETTE_BUILDING
            kick_strength = 0.3
            height_sweep_speed = 0.15
        else:  # DROP
            target_brightness = 0.6 + 0.4 * energy
            pulse_speed = 1.5   # fast
            palette = PALETTE_DROP
            kick_strength = 0.6  # strong kick reaction
            height_sweep_speed = 0.3

        # ── Smooth brightness transitions ──
        alpha = 1.0 - 0.92 ** (dt * 30)
        self.smooth_brightness += (target_brightness - self.smooth_brightness) * alpha

        # ── Pulse animation (breathing effect) ──
        self.pulse_phase += dt * pulse_speed * 2 * np.pi
        if self.pulse_phase > 2 * np.pi:
            self.pulse_phase -= 2 * np.pi
        pulse = 0.5 + 0.5 * np.sin(self.pulse_phase)  # 0-1

        # ── Kick decay (fast attack, medium decay) ──
        if kick > self.kick_decay:
            self.kick_decay = kick
        else:
            self.kick_decay *= 0.85 ** (dt * 30)

        # ── Height sweep animation ──
        self.height_offset += dt * height_sweep_speed
        if self.height_offset > 1.0:
            self.height_offset -= 1.0

        # ── Render LED frame ──
        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)

        for i in range(self.num_leds):
            # Height position: 0 = base, 1 = peak
            height = i / max(self.num_leds - 1, 1)

            # Base color from section palette
            color = _sample_gradient(palette, height)

            # Brightness: base level + pulse modulation + kick flash
            # Pulse affects more at peak in DROP, more at base in BUILDING
            if section == self.DROP:
                # Energy radiates from peak
                pulse_at_height = pulse * (0.5 + 0.5 * height)
                kick_at_height = self.kick_decay * kick_strength * (0.3 + 0.7 * height)
            elif section == self.BUILDING:
                # Energy rises from base
                rise_pos = min(1.0, self.smooth_brightness * 1.5)
                if height <= rise_pos:
                    pulse_at_height = pulse * (0.3 + 0.7 * (1.0 - height))
                else:
                    pulse_at_height = pulse * 0.1
                kick_at_height = self.kick_decay * kick_strength * (0.5 + 0.5 * (1.0 - height))
            else:
                # Breakdown: gentle uniform breathing
                pulse_at_height = pulse * 0.3
                kick_at_height = self.kick_decay * kick_strength

            brightness = self.smooth_brightness * (0.6 + 0.4 * pulse_at_height) + kick_at_height
            brightness = min(1.0, brightness)

            # Apply brightness with gamma correction
            display_b = brightness ** 0.7

            pixel = (color * display_b).clip(0, 255)
            frame[i] = pixel.astype(np.uint8)

        return frame

    def get_source_values(self) -> dict:
        with self._lock:
            return {
                'energy_level': self._energy_norm,
                'energy_slope': self._energy_slope,
                'bass_ratio': self._bass_ratio,
                'onset_density': self._onset_density,
            }

    def get_diagnostics(self) -> dict:
        with self._lock:
            section = self._section
            energy = self._energy_norm
            slope = self._energy_slope
            bass = self._bass_ratio
        return {
            'section': self.SECTION_NAMES[section],
            'energy': round(energy, 2),
            'slope': round(slope, 3),
            'bass': round(bass, 2),
            'bright': round(self.smooth_brightness, 2),
            'kick': round(self.kick_decay, 2),
        }
