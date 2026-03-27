"""
Heartbeat — two blood-flow pressure waves orbiting the diamond, synced to tempo.

Two pulse origins orbit the outer perimeter, always geometrically opposed
via a 4-anchor parameterization that compensates for the diamond's asymmetric
branch lengths (left=42 LEDs, right=20 LEDs).

Each pulse models blood flowing outward from the origin — an expanding
wavefront with exponential trailing decay, not a static blob. Systolic
phase (first 70%) expands the wave; diastolic phase (last 30%) fades
everything to black.

Middle branch (interior) lights up by spatial proximity when origins
pass near apex or base.

Tempo: uses OnsetTempoTracker — best false-positive resilience for
electronic/hip-hop (honest low confidence when unsure). Picks the
octave that puts the period closest to ~1 second. Falls back to
fixed 1.0s period when no lock.

Usage:
    python runner.py heartbeat --sculpture cob_diamond --no-leds
    python runner.py heartbeat --sculpture cob_diamond
"""

import math
import threading
import numpy as np
from base import AudioReactiveEffect
from topology import SculptureTopology
from signals import OverlapFrameAccumulator, OnsetTempoTracker


TARGET_PERIOD = 1.0  # desired seconds between pulses

# Blood-flow wavefront constants
WAVE_REACH = 0.25     # max wavefront radius in coordinate units
PULSE_DECAY = 6.0     # exponential decay rate for trailing glow
SYSTOLE_FRAC = 0.7    # fraction of beat that is systolic (expanding) phase


def octave_nearest(period, target):
    """Shift period by octaves (2x/0.5x) to land closest to target."""
    if period <= 0:
        return target
    log_ratio = math.log2(target / period)
    n = round(log_ratio)
    return period * (2 ** n)


class HeartbeatEffect(AudioReactiveEffect):
    registry_name = 'heartbeat'
    handles_topology = True  # renders in physical LED space via its own topology
    ref_pattern = 'groove'
    ref_scope = 'beat'
    ref_input = 'pulse pattern generator'

    COLOR = np.array([140, 0, 0], dtype=np.float64)

    def __init__(self, num_leds: int, sample_rate: int = 44100,
                 sculpture_id: str = 'cob_diamond'):
        super().__init__(num_leds, sample_rate)
        self.topo = SculptureTopology(sculpture_id)
        self.num_leds = self.topo.num_leds

        self.accum = OverlapFrameAccumulator()
        self.tracker = OnsetTempoTracker(sample_rate=sample_rate)

        # Beat phase accumulator (0->1 per pulse period, wraps)
        self._beat_phase = 0.0
        self._pulse_count = 0
        self._period = TARGET_PERIOD

        # Orbit parameter: advances by 1/len(perimeter) per beat
        self._orbit_t = 0.0

        self._lock = threading.Lock()
        # Snapshot for render thread
        self._snap_phase = 0.0
        self._snap_pulse = 0
        self._snap_period = TARGET_PERIOD
        self._snap_orbit_t = 0.0

        # Build perimeter parameterization
        self._perimeter_leds, self._perimeter_params = self._build_perimeter_params()
        self._orbit_step = 1.0 / len(self._perimeter_leds)

    @property
    def name(self):
        return "Heartbeat"

    @property
    def description(self):
        return "Twin blood-flow pulses orbiting opposed across branches."

    # ------------------------------------------------------------------
    # 4-anchor perimeter parameterization
    # ------------------------------------------------------------------

    def _build_perimeter_params(self):
        """Reorder rotation_path starting from apex, assign parameters via 4 anchors.

        Anchors at equal parameter spacing:
            t=0.00  apex (top)       — nearest path LED to apex_right (42)
            t=0.25  right_tip        — nearest path LED to right_tip (55)
            t=0.50  base (bottom)    — nearest path LED to base_start (0)
            t=0.75  left_tip         — nearest path LED to left_tip (26)

        Tip LEDs may be skipped by self-loop detection, so we find the
        nearest path LED to each landmark's physical position.

        LEDs between anchors get linearly interpolated parameters.
        Returns (perimeter_leds, perimeter_params) — parallel arrays.
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

        # Sort by path index (should already be sorted given reorder from apex)
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
        # Find the LED whose parameter is closest to t (accounting for wrap)
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
    # Blood-flow brightness
    # ------------------------------------------------------------------

    @staticmethod
    def _blood_flow_brightness(dist, phase):
        """Expanding wavefront envelope with trailing exponential decay.

        Args:
            dist: Euclidean distance from origin to LED
            phase: beat phase 0-1

        Returns:
            brightness 0-1
        """
        if phase < SYSTOLE_FRAC:
            # Systolic phase: wave expands outward
            systole_phase = phase / SYSTOLE_FRAC  # 0-1 within systole
            wave_r = systole_phase * WAVE_REACH    # wavefront radius

            if dist > wave_r:
                # Wave hasn't reached this LED yet
                return 0.0

            # Distance behind the wavefront
            behind = wave_r - dist
            # Exponential trailing decay — bright at wavefront, dim behind
            trail = math.exp(-PULSE_DECAY * behind)
            # Arterial pressure drop: weaken with distance from origin
            pressure = max(0.0, 1.0 - dist / WAVE_REACH)
            return trail * pressure
        else:
            # Diastolic phase: everything fades linearly to black
            diastole_phase = (phase - SYSTOLE_FRAC) / (1.0 - SYSTOLE_FRAC)  # 0-1
            wave_r = WAVE_REACH  # wave fully expanded

            if dist > wave_r:
                return 0.0

            # Same spatial shape as end of systole, but fading out
            behind = wave_r - dist
            trail = math.exp(-PULSE_DECAY * behind)
            pressure = max(0.0, 1.0 - dist / WAVE_REACH)
            fade = 1.0 - diastole_phase  # linear fade to black
            return trail * pressure * fade

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

            dt_step = self.tracker.rms_dt
            self._beat_phase += dt_step / period
            if self._beat_phase >= 1.0:
                self._beat_phase -= 1.0
                self._pulse_count += 1
                # Advance orbit by one step per beat
                self._orbit_t = (self._orbit_t + self._orbit_step) % 1.0

            with self._lock:
                self._snap_phase = self._beat_phase
                self._snap_pulse = self._pulse_count
                self._snap_period = period
                self._snap_orbit_t = self._orbit_t

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, dt: float) -> np.ndarray:
        with self._lock:
            phase = self._snap_phase
            orbit_t = self._snap_orbit_t

        # Find leader + follower origins (geometrically opposed)
        origin_a = self._origin_at_param(orbit_t)
        origin_b = self._origin_at_param(orbit_t + 0.5)

        # Pre-fetch distance arrays from both origins
        dists_a = self.topo.distances_from(origin_a)
        dists_b = self.topo.distances_from(origin_b)

        frame = np.zeros((self.num_leds, 3), dtype=np.uint8)
        for i in range(self.topo.num_leds):
            # Blood-flow brightness from each origin, take max
            ba = self._blood_flow_brightness(dists_a[i], phase)
            bb = self._blood_flow_brightness(dists_b[i], phase)
            brightness = max(ba, bb)
            if brightness > 0:
                frame[i] = (self.COLOR * brightness).astype(np.uint8)
        return frame

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        with self._lock:
            phase = self._snap_phase
            orbit_t = self._snap_orbit_t
            period = self._snap_period

        # Phase label
        if phase < SYSTOLE_FRAC:
            ph_label = 'systole'
        else:
            ph_label = 'diastole'

        # Origins
        origin_a = self._origin_at_param(orbit_t)
        origin_b = self._origin_at_param(orbit_t + 0.5)
        xy_a = self.topo.coords[origin_a]
        xy_b = self.topo.coords[origin_b]

        raw = self.tracker.estimated_period
        return {
            'phase': ph_label,
            'orbit_t': f'{orbit_t:.3f}',
            'origin_a': f'{origin_a}',
            'origin_b': f'{origin_b}',
            'xy_a': f'({xy_a[0]:.2f}, {xy_a[1]:.2f})',
            'xy_b': f'({xy_b[0]:.2f}, {xy_b[1]:.2f})',
            'bpm': f'{self.tracker.bpm:.0f}',
            'period': f'{period:.2f}s',
            'raw': f'{raw:.2f}s' if raw > 0 else '-',
            'conf': f'{self.tracker.confidence:.2f}',
        }
