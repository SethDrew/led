"""
Wave String — 1D wave equation PDE driven by continuous audio energy.

A vibrating string simulation where waveform RMS energy continuously displaces
LED 0 (or a configurable injection point). Waves propagate outward, reflect off
the far end, and interfere to create standing wave patterns. This is NOT
beat-triggered — the continuous RMS energy IS the displacement boundary condition.

Physics: ∂²y/∂t² = c² ∂²y/∂x² - damping * ∂y/∂t

Color mapping: positive displacement → warm (red/amber), negative → cool
(blue/cyan), zero → black. Absolute value controls brightness, sign controls hue.
Standing wave patterns appear as alternating warm/cool zones.
"""

import numpy as np
import threading
from base import AudioReactiveEffect
from signals import OverlapFrameAccumulator


class WaveStringEffect(AudioReactiveEffect):
    """1D wave equation PDE driven by continuous audio RMS energy."""

    registry_name = 'wave_string'
    ref_pattern = 'proportional'
    ref_scope = 'beat'
    ref_input = 'RMS amplitude'

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 # --- Wave physics ---
                 wave_speed: float = 60.0,
                 damping: float = 1.0,
                 substeps: int = 6,
                 # --- Injection ---
                 injection_idx: int = 0,
                 # --- Color ---
                 warm_color: tuple = (255, 100, 10),
                 cool_color: tuple = (10, 80, 255),
                 gamma: float = 0.7,
                 ):
        """
        Args:
            num_leds:      Number of LEDs in the strip.
            sample_rate:   Audio sample rate.
            wave_speed:    Wave propagation speed in LEDs/second. Controls how fast
                           waves travel along the strip. Higher = faster propagation.
                           A value of 60 means a wave crosses 60 LEDs per second,
                           i.e. ~1 second to traverse a 60-LED strip.
                           Visually: low c → slow, sloshy waves; high c → fast,
                           tight ripples.
            damping:       Velocity-proportional damping coefficient (1/s). Controls
                           how quickly waves decay. 0 = no damping (energy builds
                           forever, will explode). 0.5 = gentle decay, long-lived
                           reflections. 2.0 = heavy damping, waves die within one
                           traversal. Sweet spot is 0.5-2.0.
            substeps:      Number of PDE integration substeps per render frame.
                           Needed for numerical stability (CFL condition: c*dt/dx < 1).
                           At 30fps with c=60 LEDs/s: dt_sub = 1/(30*6) ≈ 0.0056s,
                           c*dt_sub = 60*0.0056 = 0.33 < 1. Stable.
                           If you increase wave_speed, you may need more substeps.
                           Rule of thumb: substeps >= wave_speed / (fps * 0.8).
            injection_idx: Which LED receives the audio displacement. 0 = base of
                           strip (waves travel away from base). Can be set to
                           num_leds//2 for center injection (waves travel both ways).
            warm_color:    RGB tuple for positive displacement (default: red/amber).
            cool_color:    RGB tuple for negative displacement (default: blue/cyan).
            gamma:         Gamma curve for brightness mapping. < 1.0 compresses
                           dynamic range (makes dim values brighter). 0.7 is a good
                           default for LEDs which have poor low-end response.
        """
        super().__init__(num_leds, sample_rate)

        # --- Audio analysis (same pattern as energy_waterfall) ---
        self.n_fft = 2048
        self.hop_length = 512
        self.accum = OverlapFrameAccumulator(
            frame_len=self.n_fft, hop=self.hop_length,
        )

        # Shared state: audio thread writes, render thread reads
        self._rms = np.float32(0.0)
        self._lock = threading.Lock()

        # Peak-decay normalization for RMS.
        # Tracks the running maximum RMS and slowly decays it so that the
        # normalized output adapts to changing volume levels.
        # 0.9995 at ~86 fps (44100/512) → ~8 second half-life.
        self._rms_peak = np.float32(1e-10)
        self._peak_decay = 0.9995

        # --- Wave PDE state ---
        # y[i] = displacement of LED i. Starts at zero (flat string).
        self._y = np.zeros(num_leds, dtype=np.float64)
        # velocity[i] = ∂y/∂t at LED i.
        self._vel = np.zeros(num_leds, dtype=np.float64)

        # Physics parameters
        self._wave_speed = wave_speed
        # c² used in the finite difference equation. dx=1 (one LED spacing),
        # so c² is in units of LEDs²/s².
        self._c_sq = wave_speed ** 2
        self._damping = damping
        self._substeps = substeps
        self._injection_idx = injection_idx

        # Bipolar force: subtract a running baseline from RMS so that
        # rising energy → positive force, falling energy → negative force.
        # Without this, RMS is always positive and the string just bows
        # in one direction (no oscillation, no wave patterns).
        self._rms_baseline = 0.0

        # --- Color mapping ---
        self._warm = np.array(warm_color, dtype=np.float64)
        self._cool = np.array(cool_color, dtype=np.float64)
        self._gamma = gamma

        # Peak-decay normalization for displacement.
        # Tracks the running max |y| to auto-scale brightness so the effect
        # uses the full 0-255 range regardless of energy level.
        # Slightly faster decay than RMS peak so brightness adapts quickly
        # when energy drops.
        self._disp_peak = 1e-10
        self._disp_peak_decay = 0.999

        # Output buffer
        self._frame_buf = np.zeros((num_leds, 3), dtype=np.uint8)

    @property
    def name(self):
        return "Wave String"

    @property
    def description(self):
        return ("1D wave equation PDE — continuous audio RMS drives displacement, "
                "creating propagating and reflecting waves along the LED strip.")

    # ------------------------------------------------------------------ #
    #  Audio thread                                                        #
    # ------------------------------------------------------------------ #

    def process_audio(self, mono_chunk: np.ndarray):
        """Process audio chunk → normalized RMS. Called from audio thread."""
        for frame in self.accum.feed(mono_chunk):
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray):
        """Compute RMS with peak-decay normalization."""
        rms = np.float32(np.sqrt(np.mean(frame ** 2)))

        # Peak-decay: slowly lower the ceiling so quiet sections still register.
        self._rms_peak = max(rms, self._rms_peak * self._peak_decay)
        rms_norm = rms / self._rms_peak if self._rms_peak > 1e-10 else 0.0

        with self._lock:
            self._rms = np.float32(rms_norm)

    # ------------------------------------------------------------------ #
    #  Render thread                                                       #
    # ------------------------------------------------------------------ #

    def render(self, dt: float) -> np.ndarray:
        """Step the wave PDE and map displacement to RGB.

        Called from main loop at ~30fps. dt is seconds since last call.
        """
        # Read the latest normalized RMS from the audio thread.
        with self._lock:
            rms = float(self._rms)

        n = self.num_leds
        inj = self._injection_idx

        # --- PDE substeps ---
        #
        # The CFL (Courant-Friedrichs-Lewy) stability condition for the 1D
        # wave equation with explicit finite differences requires:
        #
        #     c * dt_sub / dx < 1
        #
        # where dx = 1 (one LED spacing). If this is violated, the simulation
        # explodes exponentially. Substeps divide the render dt into smaller
        # steps to satisfy this condition.
        #
        # With wave_speed=60, substeps=6, fps=30:
        #   dt_sub = (1/30) / 6 ≈ 0.0056s
        #   CFL number = 60 * 0.0056 / 1 = 0.33 → stable (< 1)

        dt_sub = dt / self._substeps
        c_sq = self._c_sq
        damp = self._damping
        y = self._y
        vel = self._vel

        # Bipolar force: deviation from running baseline.
        # Rising energy → positive force, falling → negative force.
        # Alpha=0.3 tracks fast enough that the signal oscillates around zero.
        self._rms_baseline += 0.3 * (rms - self._rms_baseline)
        force = (rms - self._rms_baseline)

        for _ in range(self._substeps):
            # Injection: clamp displacement to bipolar signal.
            # Positive when energy rises above baseline, negative when it dips.
            # The PDE propagates these oscillations outward as waves.
            y[inj] = force
            vel[inj] = 0.0

            # Free boundary at far end: waves pass through and exit.
            # No reflection, no interference buildup — energy flows one
            # direction and dissipates via damping along the way.
            #
            # Alternative boundary conditions:
            #
            # FIXED END (reflection with phase inversion):
            #   y[-1] = 0.0; vel[-1] = 0.0
            #   Incoming positive wave returns as negative, creating standing
            #   wave nodes. Visually rich but can overload the strip with
            #   overlapping reflections when energy is continuous.
            #
            # PERIODIC (wrap-around, string is a loop):
            #   Treat y[0] and y[-1] as neighbors. Waves wrap around endlessly.
            #   Implementation: use modular indexing in the curvature calc.
            y[-1] = y[-2]
            vel[-1] = vel[-2]

            # Compute acceleration from the wave equation:
            #   a[i] = c² * (y[i+1] - 2*y[i] + y[i-1]) - damping * vel[i]
            #
            # The first term is the discrete Laplacian (curvature) — points
            # pulled toward the average of their neighbors. The second term
            # is velocity-proportional drag that dissipates energy over time.

            # Interior points: 1..n-2 (excluding boundaries)
            curvature = y[2:] - 2.0 * y[1:-1] + y[:-2]
            accel = c_sq * curvature - damp * vel[1:-1]

            # Symplectic Euler integration (update velocity, then position).
            # More stable than standard Euler for oscillatory systems.
            vel[1:-1] += accel * dt_sub
            y[1:-1] += vel[1:-1] * dt_sub

        # --- Displacement normalization (waterfall-style peak-decay) ---
        # Same approach as energy_waterfall: track running peak, normalize,
        # clip to [0, 1]. Fast attack (instant), slow decay via _disp_peak_decay.
        max_disp = np.max(np.abs(y))
        self._disp_peak = max(max_disp, self._disp_peak * self._disp_peak_decay)

        if self._disp_peak > 1e-10:
            y_norm = y / self._disp_peak
        else:
            y_norm = y

        # --- Color mapping ---
        # Positive displacement → warm color (red/amber)
        # Negative displacement → cool color (blue/cyan)
        # Magnitude → brightness (with gamma correction)

        brightness = np.clip(np.abs(y_norm), 0.0, 1.0) ** self._gamma

        # Vectorized color mapping: warm where positive, cool where negative
        pos = (y_norm >= 0)
        colors = np.where(pos[:, np.newaxis], self._warm, self._cool)
        self._frame_buf[:] = np.clip(colors * brightness[:, np.newaxis], 0, 255).astype(np.uint8)

        return self._frame_buf.copy()

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self) -> dict:
        """Return debug info for terminal display."""
        with self._lock:
            rms = float(self._rms)

        max_disp = np.max(np.abs(self._y))
        max_vel = np.max(np.abs(self._vel))

        # CFL number for current settings (should be < 1 for stability).
        # This is computed at the actual render dt, not a fixed 1/30.
        # Shown so the user can verify stability.
        cfl = self._wave_speed * (1.0 / 30.0) / self._substeps

        return {
            'rms': f'{rms:.3f}',
            'max_disp': f'{max_disp:.4f}',
            'max_vel': f'{max_vel:.4f}',
            'disp_peak': f'{self._disp_peak:.4f}',
            'cfl': f'{cfl:.3f}',
            'wave_speed': f'{self._wave_speed:.1f}',
            'damping': f'{self._damping:.2f}',
            'substeps': str(self._substeps),
        }


# ====================================================================== #
#  Extension ideas (not implemented — notes for future development)        #
#                                                                          #
#  1. Multi-point injection: Split audio into frequency bands (bass,       #
#     mids, treble) and inject each at different positions along the       #
#     strip. Bass at LED 0, treble at LED n-1, mids in the middle.        #
#     Creates frequency-dependent wave interference.                       #
#                                                                          #
#  2. Frequency-dependent wave speed: Instead of a single c, make wave    #
#     speed depend on the dominant frequency of the audio. Low-frequency   #
#     content → slow waves, high-frequency → fast. This creates           #
#     dispersion (wave packets spread out over time), like light through   #
#     a prism but for sound→LED.                                          #
#                                                                          #
#  3. Nonlinear wave equation: Add a cubic term to the restoring force    #
#     (y³) to get amplitude-dependent wave speed. Loud passages create    #
#     shock-wave-like steepening, quiet passages behave linearly.         #
#     Models real vibrating strings more accurately.                       #
#                                                                          #
#  4. Coupled strings: Run multiple wave equations in parallel with       #
#     weak coupling between them (neighboring strings exchange energy).    #
#     Map each string to a different LED strip or different color channel. #
#     Creates complex interference patterns from simple audio input.       #
#                                                                          #
#  5. Variable damping: Make damping position-dependent (heavy damping     #
#     near the ends, light in the middle) to create a "resonance body"    #
#     effect where the center of the strip rings longer.                   #
#                                                                          #
#  6. Velocity-based coloring: Instead of (or in addition to) displacement,#
#     use velocity for color mapping. Velocity peaks where displacement    #
#     crosses zero, creating a complementary visual pattern.               #
# ====================================================================== #
