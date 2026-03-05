"""
Spatial topology for LED sculptures.

Loads xy_keyframes from sculptures.json, interpolates (x, y) coordinates
for every LED, and pre-computes a distance matrix. Any effect can import
this to get spatial awareness across branches.

Usage:
    from topology import SculptureTopology
    topo = SculptureTopology('cob_diamond')
    dists = topo.distances_from(0)        # distances from LED 0
    near = topo.leds_within(0, 0.15)      # [(idx, dist), ...] within radius
"""

import json
import os
import numpy as np


class SculptureTopology:
    """Spatial coordinate system and distance matrix for a sculpture."""

    def __init__(self, sculpture_id='cob_diamond'):
        sculpture = self._load_sculpture(sculpture_id)
        branches = sculpture['branches']
        self.num_leds = sculpture.get('num_leds',
                                      sum(b['count'] for b in branches))

        # Branch metadata
        self.branches = {}
        for b in branches:
            self.branches[b['name']] = (b['start'], b['start'] + b['count'] - 1)

        # Landmarks
        self.landmarks = sculpture.get('landmarks', {})

        # Interpolate (x, y) for every LED from keyframes
        self.coords = np.zeros((self.num_leds, 2), dtype=np.float64)
        for b in branches:
            kf = b.get('xy_keyframes')
            if not kf:
                # Fallback: no spatial data, place LEDs along x=0
                for i in range(b['count']):
                    idx = b['start'] + i
                    self.coords[idx] = [0.0, i / max(b['count'] - 1, 1)]
                continue

            # Sort keyframes by LED index
            kf = sorted(kf, key=lambda k: k[0])
            for seg in range(len(kf) - 1):
                i0, x0, y0 = kf[seg]
                i1, x1, y1 = kf[seg + 1]
                n = i1 - i0
                for j in range(n + (1 if seg == len(kf) - 2 else 0)):
                    t = j / max(n, 1)
                    idx = i0 + j
                    if idx < self.num_leds:
                        self.coords[idx] = [
                            x0 + (x1 - x0) * t,
                            y0 + (y1 - y0) * t,
                        ]

        # Pre-compute distance matrix (symmetric)
        # For 72 LEDs this is 72x72 = 5184 floats, ~40KB — trivial.
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        self.distances = np.sqrt((diff ** 2).sum(axis=2))

        # Rotation path: outer perimeter (left + right branches)
        # Middle branch is interior, not part of the perimeter.
        # Self-loops at tips are skipped per-branch — the origin jumps
        # across the connection point instead of tracing around the tip.
        left_start, left_end = self.branches.get('left', (0, 0))
        right_start, right_end = self.branches.get('right', (0, 0))
        left_path = self._skip_self_loops(list(range(left_start, left_end + 1)))
        right_path = self._skip_self_loops(list(range(right_start, right_end + 1)))
        self.rotation_path = left_path + right_path

    def distances_from(self, led: int) -> np.ndarray:
        """Return distance from `led` to every other LED."""
        return self.distances[led]

    def leds_within(self, led: int, radius: float) -> list:
        """Return [(index, distance), ...] for LEDs within `radius` of `led`.

        Sorted by distance. Excludes `led` itself.
        """
        dists = self.distances[led]
        mask = (dists < radius) & (np.arange(self.num_leds) != led)
        indices = np.where(mask)[0]
        pairs = [(int(i), float(dists[i])) for i in indices]
        pairs.sort(key=lambda p: p[1])
        return pairs

    def _skip_self_loops(self, raw):
        """Remove tip excursions from rotation path.

        Detects where the path crosses itself (two LEDs far apart in
        index but at the same physical position) and jumps across,
        keeping the origin moving smoothly around the perimeter.
        """
        THRESHOLD = 0.02
        path = []
        i = 0
        while i < len(raw):
            path.append(raw[i])
            skip_to = -1
            for j in range(i + 3, min(i + 10, len(raw))):
                if self.distances[raw[i], raw[j]] < THRESHOLD:
                    skip_to = j
                    break
            if skip_to >= 0:
                i = skip_to
            else:
                i += 1
        return path

    @staticmethod
    def _load_sculpture(sculpture_id):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        path = os.path.join(base_dir, 'hardware', 'sculptures.json')
        with open(path) as f:
            sculptures = json.load(f)
        sculpture = next((s for s in sculptures if s['id'] == sculpture_id), None)
        if not sculpture:
            raise ValueError(f"Unknown sculpture: {sculpture_id}")
        return sculpture
