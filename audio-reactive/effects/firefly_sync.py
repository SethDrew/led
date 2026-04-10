"""
Firefly Synchronization — Kuramoto model driven by bass energy.

Independent fireflies scattered across the strip, each oscillating near
the detected tempo. They flash independently at first, then gradually
phase-lock via Kuramoto coupling. Stronger bass = stronger coupling =
faster synchronization.

The Kuramoto model:
  dθ_i/dt = ω_i + (K/N) * Σ sin(θ_j - θ_i)

Each firefly has a natural frequency ω_i (randomized around detected tempo).
K is the coupling constant, modulated by bass energy. When K is high,
fireflies pull each other's phases together. When K drops, they drift
apart at their natural frequencies.

Flash mechanism: when a firefly's phase crosses 0 (the "flash point"),
it emits a warm amber glow at its position that spreads to nearby LEDs
and decays over time.

Designed for reggae: steady ~70 BPM groove, bass-dominant.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator, OnsetTempoTracker


class FireflySyncEffect(AudioReactiveEffect):
    """Fireflies that synchronize to the beat via Kuramoto coupling."""

    registry_name = 'firefly_sync'
    ref_pattern = 'groove'
    ref_scope = 'phrase'
    ref_input = 'bass flux + onset tempo'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 n_fireflies: int = 28):
        super().__init__(num_leds, sample_rate)

        self.n_fireflies = n_fireflies

        # Audio processing
        self.accum = OverlapFrameAccumulator()
        self.tempo = OnsetTempoTracker(sample_rate=sample_rate)

        # Bass flux detection (same approach as bass_pulse)
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        self.bass_mask = (self.freq_bins >= 20) & (self.freq_bins <= 250)
        self.prev_bass_spec = None
        self.bass_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.bass_buf_pos = 0
        self.flux_peak = 1e-10
        self.flux_peak_decay = 0.997
        self.bass_energy = 0.0       # normalized 0-1 (instantaneous)
        self.bass_energy_slow = 0.0  # EMA-smoothed for coupling (section-level)
        # EMA alpha for ~2s time constant at ~21 bass frames/sec (44100/2048)
        self._bass_ema_alpha = 1.0 / (2.0 * sample_rate / self.n_fft)

        # Firefly state
        # Positions: spread evenly with some jitter
        base_positions = np.linspace(0, num_leds - 1, n_fireflies)
        jitter = np.random.uniform(-num_leds / n_fireflies * 0.3,
                                    num_leds / n_fireflies * 0.3,
                                    n_fireflies)
        self.positions = np.clip(base_positions + jitter, 0, num_leds - 1)

        # Phases: random initial, range [0, 2π)
        self.phases = np.random.uniform(0, 2 * np.pi, n_fireflies)

        # Natural frequency offsets: small random deviations from detected tempo
        # These are multipliers on the detected angular frequency
        self.freq_offsets = np.random.normal(1.0, 0.08, n_fireflies)

        # Flash state: brightness per firefly (decays after flash)
        self.flash_brightness = np.zeros(n_fireflies, dtype=np.float32)

        # Coupling — no floor: without bass, fireflies drift freely.
        # Bass drives coupling above critical threshold via positive feedback.
        # Tuned for ~10s convergence, ~7s divergence.
        self.K_base = 0.0
        self.K_bass_mult = 8.0

        # Phase noise: chaotic divergence when coupling is weak
        self.noise_sigma = 2.5  # base noise intensity (rad/s)

        # Position drift: slow random walk (LEDs/second)
        self.drift_speed = 3.0

        # Spatial glow radius — sigma 0.65 gives ~1-2 LED tight points
        self.glow_radius = 0.65

        # Warm amber palette: base color for firefly flash
        # Slight variation per firefly for organic feel
        self.colors = np.zeros((n_fireflies, 3), dtype=np.float32)
        for i in range(n_fireflies):
            warmth = np.random.uniform(0.85, 1.0)
            self.colors[i] = [255 * warmth, 180 * warmth, 60 * warmth]

        self._lock = threading.Lock()

    @property
    def name(self):
        return "Firefly Sync"

    @property
    def description(self):
        return ("Kuramoto-coupled fireflies sync to detected tempo; "
                "bass energy modulates coupling strength.")

    def process_audio(self, mono_chunk: np.ndarray):
        # Feed tempo tracker through overlap accumulator
        for frame in self.accum.feed(mono_chunk):
            self.tempo.feed_frame(frame)

        # Bass flux detection (separate buffer, full FFT frames)
        n = len(mono_chunk)
        space = self.n_fft - self.bass_buf_pos
        if n < space:
            self.bass_buf[self.bass_buf_pos:self.bass_buf_pos + n] = mono_chunk
            self.bass_buf_pos += n
            return

        self.bass_buf[self.bass_buf_pos:] = mono_chunk[:space]
        self._process_bass_frame(self.bass_buf.copy())

        leftover = n - space
        self.bass_buf[:leftover] = mono_chunk[space:]
        self.bass_buf_pos = leftover

    def _process_bass_frame(self, frame):
        windowed = frame * self.window
        spec = np.abs(np.fft.rfft(windowed))
        bass_spec = spec[self.bass_mask]

        if self.prev_bass_spec is not None:
            diff = bass_spec - self.prev_bass_spec
            flux = np.sum(np.maximum(diff, 0))

            self.flux_peak = max(flux, self.flux_peak * self.flux_peak_decay)
            normalized = flux / self.flux_peak if self.flux_peak > 0 else 0

            # Slow EMA for section-level coupling drive
            a = self._bass_ema_alpha
            self.bass_energy_slow = a * normalized + (1 - a) * self.bass_energy_slow

            with self._lock:
                self.bass_energy = self.bass_energy_slow

        self.prev_bass_spec = bass_spec.copy()

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            bass = self.bass_energy

        # Determine angular frequency from tempo
        if self.tempo.estimated_period > 0 and self.tempo.confidence >= 0.2:
            # Fold detected BPM into groove range [50, 140]
            period = self.tempo.estimated_period
            bpm = 60.0 / period
            while bpm > 140:
                period *= 2
                bpm /= 2
            while bpm < 40:
                period /= 2
                bpm *= 2
            base_omega = 2 * np.pi / period
        else:
            # Fallback: assume ~70 BPM (reggae default)
            base_omega = 2 * np.pi * (70.0 / 60.0)

        # Kuramoto coupling: bass-modulated with r-dependent positive feedback
        # K_floor prevents total collapse; K_bass * r drives beat-locked sync
        K_floor = self.K_base
        K_driven = bass * self.K_bass_mult

        # Compute mean phase and order parameter r (circular mean)
        mean_sin = np.mean(np.sin(self.phases))
        mean_cos = np.mean(np.cos(self.phases))
        mean_phase = np.arctan2(mean_sin, mean_cos)
        r = np.sqrt(mean_sin**2 + mean_cos**2)

        # Effective coupling: floor (constant) + driven (r-scaled positive feedback)
        K_eff = K_floor + K_driven * r

        # Phase noise: chaotic scatter when coupling is weak
        # Noise scales inversely with K_eff — strong coupling suppresses noise
        noise_scale = self.noise_sigma / (1.0 + K_eff * 5.0)

        # Update each firefly's phase
        for i in range(self.n_fireflies):
            omega_i = base_omega * self.freq_offsets[i]
            coupling = K_eff * np.sin(mean_phase - self.phases[i])
            noise = np.random.normal(0, noise_scale) * dt
            self.phases[i] += (omega_i + coupling) * dt + noise

        # Drift positions: slow random walk
        self.positions += np.random.normal(0, self.drift_speed, self.n_fireflies) * dt
        self.positions = np.clip(self.positions, 0, self.num_leds - 1)

        # Detect flashes: phase crossing 0 (mod 2π)
        wrapped = self.phases % (2 * np.pi)
        for i in range(self.n_fireflies):
            # Flash when phase wraps around (crosses 2π → 0)
            if wrapped[i] < 0.3 and self.flash_brightness[i] < 0.1:
                self.flash_brightness[i] = 1.0
        self.phases = wrapped

        # Decay flashes — 250ms time constant for warm reggae glow
        decay = np.exp(-dt / 0.25)
        self.flash_brightness *= decay

        # Render LED frame
        frame = np.zeros((self.num_leds, 3), dtype=np.float32)

        # Very dim warm base (like embers)
        frame[:] = [4, 2, 1]

        # Add each firefly's glow
        for i in range(self.n_fireflies):
            if self.flash_brightness[i] < 0.01:
                continue

            pos = self.positions[i]
            brightness = self.flash_brightness[i]
            color = self.colors[i] * brightness

            # Gaussian spatial falloff
            led_indices = np.arange(self.num_leds)
            distances = np.abs(led_indices - pos)
            falloff = np.exp(-0.5 * (distances / self.glow_radius) ** 2)

            frame[:, 0] += color[0] * falloff
            frame[:, 1] += color[1] * falloff
            frame[:, 2] += color[2] * falloff

        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        # Phase coherence: 0 = fully desynchronized, 1 = fully synchronized
        r = np.abs(np.mean(np.exp(1j * self.phases)))
        return {
            'coherence': f'{r:.2f}',
            'bass': f'{self.bass_energy:.2f}',
            'bpm': f'{self.tempo.bpm:.1f}',
            'K': f'{self.K_base + self.bass_energy * self.K_bass_mult * r:.1f}',
        }

    source_features = [
        {'id': 'bass_flux', 'label': 'Bass Flux', 'color': '#FF6600'},
        {'id': 'coherence', 'label': 'Phase Coherence', 'color': '#FFAA00'},
    ]

    def get_source_values(self) -> dict:
        r = np.abs(np.mean(np.exp(1j * self.phases)))
        return {
            'bass_flux': float(self.bass_energy),
            'coherence': float(r),
        }
