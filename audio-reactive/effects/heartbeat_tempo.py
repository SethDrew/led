"""
Heartbeat Tempo — blood-flow pulse synced to onset tempo tracker.

Single expanding wavefront per beat from two opposed origins.

Multi-beat cycle:
  Beat 0 = active pulse (blood-flow wavefront expands from both origins)
  Beats 1..N = trailing glow fades out (frozen wavefront, linear fade)
  N = max(1, round(period * FADE_RATE))

Two pulse origins orbit the outer perimeter via the same 4-anchor
parameterization as heartbeat.py, always geometrically opposed.
Orbit advances once per CYCLE (not per beat).

Tempo: uses OnsetTempoTracker — octave-nearest to ~1s period.
Falls back to fixed 1.0s when no lock.

Usage:
    python runner.py heartbeat_tempo --sculpture cob_diamond --no-leds
    python runner.py heartbeat_tempo --sculpture cob_diamond
"""

import math
import threading
import numpy as np
from base import AudioReactiveEffect
from topology import SculptureTopology
from signals import OverlapFrameAccumulator, OnsetTempoTracker


SPEED = 2.0           # global speed scalar (1.0 = base tempo)

# Base constants (scaled by SPEED)
_BASE_PERIOD = 1.0    # seconds between pulses at SPEED=1
_BASE_FADE_RATE = 2.0 # fade_beats multiplier at SPEED=1
TARGET_PERIOD = _BASE_PERIOD / SPEED
FADE_RATE = _BASE_FADE_RATE / SPEED
WAVE_REACH = 0.25     # max wavefront radius in coordinate units (not scaled)
PULSE_DECAY = 6.0     # exponential decay rate for trailing glow (not scaled)


def octave_nearest(period, target):
    """Shift period by octaves (2x/0.5x) to land closest to target."""
    if period <= 0:
        return target
    log_ratio = math.log2(target / period)
    n = round(log_ratio)
    return period * (2 ** n)


class HeartbeatTempoEffect(AudioReactiveEffect):
    registry_name = 'heartbeat_tempo'
    handles_topology = True  # renders in physical LED space via its own topology
    ref_pattern = 'groove'
    ref_scope = 'phrase'
    ref_input = 'tempo-locked heartbeat'

    COLOR = np.array([140, 0, 0], dtype=np.float64)

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 sculpture_id: str = 'cob_diamond'):
        super().__init__(num_leds, sample_rate)
        self.topo = SculptureTopology(sculpture_id)
        self.num_leds = self.topo.num_leds

        self.accum = OverlapFrameAccumulator()
        self.tracker = OnsetTempoTracker(sample_rate=sample_rate)

        # Cycle phase accumulator: goes 0 -> cycle_beats, wraps
        self._cycle_phase = 0.0
        self._pulse_count = 0
        self._period = TARGET_PERIOD
        self._fade_beats = max(1, round(TARGET_PERIOD * FADE_RATE))
        self._cycle_beats = 1 + self._fade_beats  # active beat + fade beats

        # Orbit parameter: advances by 1/len(perimeter) per cycle
        self._orbit_t = 0.0

        self._lock = threading.Lock()
        # Snapshot for render thread
        self._snap_cycle_phase = 0.0
        self._snap_pulse = 0
        self._snap_period = TARGET_PERIOD
        self._snap_orbit_t = 0.0
        self._snap_fade_beats = self._fade_beats
        self._snap_cycle_beats = self._cycle_beats

        # Build perimeter parameterization (same as heartbeat.py)
        self._perimeter_leds, self._perimeter_params = self._build_perimeter_params()
        self._orbit_step = 1.0 / len(self._perimeter_leds)

        # Cache for frozen wavefront at end of active beat, used during fade
        self._snap_frozen = np.zeros(self.num_leds, dtype=np.float64)

    @property
    def name(self):
        return "Heartbeat Tempo"

    @property
    def description(self):
        return "Blood-flow heartbeat synced to tempo, multi-beat fade cycle."

    # ------------------------------------------------------------------
    # 4-anchor perimeter parameterization (identical to heartbeat.py)
    # ------------------------------------------------------------------

    def _build_perimeter_params(self):
        """Reorder rotation_path starting from apex, assign parameters via 4 anchors.

        Anchors at equal parameter spacing:
            t=0.00  apex (top)       -- nearest path LED to apex_right (42)
            t=0.25  right_tip        -- nearest path LED to right_tip (55)
            t=0.50  base (bottom)    -- nearest path LED to base_start (0)
            t=0.75  left_tip         -- nearest path LED to left_tip (26)

        Tip LEDs may be skipped by self-loop detection, so we find the
        nearest path LED to each landmark's physical position.

        LEDs between anchors get linearly interpolated parameters.
        Returns (perimeter_leds, perimeter_params) -- parallel arrays.
        """
        raw_path = list(self.topo.rotation_path)
        path_set = set(raw_path)

        def nearest_in_path(landmark_led):
            """Find the path LED closest to landmark_led's physical position."""
            if landmark_led in path_set:
                return landmark_led
            target_xy = self.topo.coords[landmark_led]
            best_led, best_d = raw_path[0], float('inf')
            for p in raw_path:
                d = float(np.sqrt(((self.topo.coords[p] - target_xy) ** 2).sum()))
                if d < best_d:
                    best_d = d
                    best_led = p
            return best_led

        # Resolve landmarks to actual path LEDs
        apex_led = nearest_in_path(self.topo.landmarks.get('apex_right', 42))
        right_tip_led = nearest_in_path(self.topo.landmarks.get('right_tip', 55))
        base_led = nearest_in_path(self.topo.landmarks.get('base_start', 0))
        left_tip_led = nearest_in_path(self.topo.landmarks.get('left_tip', 26))

        # Reorder path to start from apex
        apex_idx = raw_path.index(apex_led)
        path = raw_path[apex_idx:] + raw_path[:apex_idx]

        # Build anchor map: path_index -> target parameter
        anchor_leds = [
            (apex_led, 0.00),
            (right_tip_led, 0.25),
            (base_led, 0.50),
            (left_tip_led, 0.75),
        ]
        anchor_positions = []
        for led, t_val in anchor_leds:
            idx = path.index(led)
            anchor_positions.append((idx, t_val))

        # Sort by path index
        anchor_positions.sort(key=lambda x: x[0])

        # Assign parameters via linear interpolation between consecutive anchors
        params = np.zeros(len(path), dtype=np.float64)
        for seg_idx in range(len(anchor_positions)):
            i0, t0 = anchor_positions[seg_idx]
            i1, t1 = anchor_positions[(seg_idx + 1) % len(anchor_positions)]

            # Handle wrap-around for last segment (left_tip -> apex)
            if i1 <= i0:
                i1 += len(path)
            if t1 <= t0:
                t1 += 1.0

            n = i1 - i0
            for j in range(n):
                idx = (i0 + j) % len(path)
                frac = j / max(n, 1)
                params[idx] = (t0 + frac * (t1 - t0)) % 1.0

        return path, params

    def _origin_at_param(self, t):
        """Find the perimeter LED nearest to parameter t (0-1, wraps)."""
        t = t % 1.0
        best_idx = 0
        best_dist = 1.0
        for i, p in enumerate(self._perimeter_params):
            d = abs(p - t)
            d = min(d, 1.0 - d)  # wrap-around distance
            if d < best_dist:
                best_dist = d
                best_idx = i
        return self._perimeter_leds[best_idx]

    # ------------------------------------------------------------------
    # Blood-flow brightness for sub-pulses
    # ------------------------------------------------------------------

    @staticmethod
    def _blood_flow_brightness(dist, phase):
        """Expanding wavefront envelope with trailing exponential decay.

        Pure expansion over the full beat — no diastolic fade here;
        the multi-beat fade cycle handles decay after the active beat.

        Args:
            dist: Euclidean distance from origin to LED
            phase: 0-1 phase within the active beat

        Returns:
            brightness 0-1
        """
        wave_r = phase * WAVE_REACH

        if dist > wave_r:
            return 0.0

        behind = wave_r - dist
        trail = math.exp(-PULSE_DECAY * behind)
        pressure = max(0.0, 1.0 - dist / WAVE_REACH)
        return trail * pressure

    @staticmethod
    def _blood_flow_brightness_frozen(dist):
        """Brightness at maximum expansion (end of systole, no diastole fade).

        Used to compute the frozen wavefront shape for fade beats.
        """
        if dist > WAVE_REACH:
            return 0.0
        behind = WAVE_REACH - dist
        trail = math.exp(-PULSE_DECAY * behind)
        pressure = max(0.0, 1.0 - dist / WAVE_REACH)
        return trail * pressure

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def process_audio(self, mono_chunk: np.ndarray):
        for frame in self.accum.feed(mono_chunk):
            self.tracker.feed_frame(frame)

            raw_period = self.tracker.estimated_period
            if raw_period > 0 and self.tracker.confidence > 0.3:
                period = octave_nearest(raw_period, TARGET_PERIOD)
            else:
                period = TARGET_PERIOD

            # Recalculate cycle structure for current period
            fade_beats = max(1, round(period * FADE_RATE))
            cycle_beats = 1 + fade_beats

            dt_step = self.tracker.rms_dt
            self._cycle_phase += dt_step / period
            if self._cycle_phase >= cycle_beats:
                self._cycle_phase -= cycle_beats
                self._pulse_count += 1
                # Advance orbit by one step per CYCLE (not per beat)
                self._orbit_t = (self._orbit_t + self._orbit_step) % 1.0

            self._period = period
            self._fade_beats = fade_beats
            self._cycle_beats = cycle_beats

            with self._lock:
                self._snap_cycle_phase = self._cycle_phase
                self._snap_pulse = self._pulse_count
                self._snap_period = period
                self._snap_orbit_t = self._orbit_t
                self._snap_fade_beats = fade_beats
                self._snap_cycle_beats = cycle_beats

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            cycle_phase = self._snap_cycle_phase
            orbit_t = self._snap_orbit_t
            fade_beats = self._snap_fade_beats
            cycle_beats = self._snap_cycle_beats

        beat_in_cycle = int(cycle_phase)
        sub_phase = cycle_phase - beat_in_cycle

        # Clamp beat_in_cycle to valid range
        if beat_in_cycle >= cycle_beats:
            beat_in_cycle = cycle_beats - 1
            sub_phase = 1.0

        # Both origins pulse simultaneously (geometrically opposed)
        origin_a = self._origin_at_param(orbit_t)
        origin_b = self._origin_at_param(orbit_t + 0.5)

        dists_a = self.topo.distances_from(origin_a)
        dists_b = self.topo.distances_from(origin_b)

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)

        if beat_in_cycle == 0:
            # Active beat: single wavefront from both origins
            for i in range(self.num_leds):
                ba = self._blood_flow_brightness(dists_a[i], sub_phase)
                bb = self._blood_flow_brightness(dists_b[i], sub_phase)
                brightness = max(ba, bb)
                if brightness > 0:
                    frame[i] = (self.COLOR * brightness).astype(np.uint8)

            # Snapshot frozen shape for fade beats
            frozen = np.zeros(self.num_leds, dtype=np.float64)
            for i in range(self.num_leds):
                frozen[i] = max(
                    self._blood_flow_brightness_frozen(dists_a[i]),
                    self._blood_flow_brightness_frozen(dists_b[i]),
                )
            with self._lock:
                self._snap_frozen = frozen

        else:
            # Fade beats: frozen wavefront fading linearly to black
            fade_progress = (beat_in_cycle - 1 + sub_phase) / fade_beats
            fade_factor = max(0.0, 1.0 - fade_progress)

            with self._lock:
                frozen = self._snap_frozen

            for i in range(self.num_leds):
                brightness = frozen[i] * fade_factor
                if brightness > 0:
                    frame[i] = (self.COLOR * brightness).astype(np.uint8)

        return frame

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        with self._lock:
            cycle_phase = self._snap_cycle_phase
            orbit_t = self._snap_orbit_t
            period = self._snap_period
            fade_beats = self._snap_fade_beats
            cycle_beats = self._snap_cycle_beats

        beat_in_cycle = int(cycle_phase)
        if beat_in_cycle >= cycle_beats:
            beat_in_cycle = cycle_beats - 1

        # Phase label
        if beat_in_cycle == 0:
            ph_label = 'pulse'
        else:
            ph_label = 'fade'

        # Origins
        origin_a = self._origin_at_param(orbit_t)
        origin_b = self._origin_at_param(orbit_t + 0.5)
        xy_a = self.topo.coords[origin_a]
        xy_b = self.topo.coords[origin_b]

        raw = self.tracker.estimated_period
        conf = self.tracker.confidence
        locked = conf > 0.3

        return {
            'phase': ph_label,
            'beat': f'{beat_in_cycle}/{cycle_beats}',
            'fade_beats': f'{fade_beats}',
            'orbit_t': f'{orbit_t:.3f}',
            'origin_a': f'{origin_a}',
            'origin_b': f'{origin_b}',
            'xy_a': f'({xy_a[0]:.2f}, {xy_a[1]:.2f})',
            'xy_b': f'({xy_b[0]:.2f}, {xy_b[1]:.2f})',
            'bpm': f'{self.tracker.bpm:.0f}',
            'period': f'{period:.2f}s',
            'raw': f'{raw:.2f}s' if raw > 0 else '-',
            'conf': f'{conf:.2f}',
            'tempo': 'locked' if locked else 'searching',
        }
