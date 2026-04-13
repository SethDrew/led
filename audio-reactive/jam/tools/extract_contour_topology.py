"""
Extract LED positions from the Yggdrasil tree mask using distance-transform
iso-contours.  Produces ~3000 evenly-distributed points that trace the shape
of the tree from its outer edge inward.

Approach:
  1. Load alpha mask  →  binary tree region
  2. Distance transform (each tree pixel → distance to nearest background pixel)
  3. Choose evenly-spaced iso-levels across the distance range
  4. Extract iso-contours at each level with skimage.measure.find_contours
  5. Distribute points evenly along each contour (arc-length resampling)
  6. Map pixel coords → normalized topology coords

Usage:
    /path/to/.venv/bin/python tools/extract_contour_topology.py [--num-leds 3000] [--output path]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_mask(path: Path) -> np.ndarray:
    """Load tree mask, return binary array (True = tree)."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        alpha = np.array(img)[:, :, 3]
        return alpha > 128
    else:
        gray = np.array(img.convert('L'))
        return gray > 128


# ---------------------------------------------------------------------------
# Coordinate mapping  (must match extract_topology.py exactly)
# ---------------------------------------------------------------------------

def pixel_to_topology(
    row: float, col: float, img_height: int, img_width: int
) -> tuple[float, float]:
    """Convert pixel coords to topology coordinate space."""
    min_x, max_x = -0.85, 0.85
    min_y, max_y = -0.55, 0.95

    u = col / img_width
    v = row / img_height

    tx = u * (max_x - min_x) + min_x
    ty = (1 - v) * (max_y - min_y) + min_y

    return (round(tx, 4), round(ty, 4))


# ---------------------------------------------------------------------------
# Contour resampling
# ---------------------------------------------------------------------------

def contour_arc_lengths(contour: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a contour (N×2 array)."""
    diffs = np.diff(contour, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def resample_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    """Evenly resample a contour to `num_points` positions."""
    arc = contour_arc_lengths(contour)
    total = arc[-1]
    if total < 1e-6 or num_points < 1:
        return contour[:1]
    targets = np.linspace(0, total, num_points, endpoint=False)
    # Interpolate row and col independently along arc length
    rows = np.interp(targets, arc, contour[:, 0])
    cols = np.interp(targets, arc, contour[:, 1])
    return np.column_stack([rows, cols])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract LED topology from tree mask via distance-transform contours')
    parser.add_argument('--num-leds', type=int, default=3000,
                        help='Target number of LEDs (default: 3000)')
    parser.add_argument('--num-levels', type=int, default=0,
                        help='Number of iso-contour levels (0 = auto)')
    parser.add_argument('--min-contour-px', type=float, default=20.0,
                        help='Minimum contour arc length in pixels (skip tiny fragments)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    jam_root = Path(__file__).parent.parent
    mask_path = jam_root / 'client' / 'public' / 'yggdrasil-mask.png'

    # 1. Load mask
    print(f'Loading mask from {mask_path}')
    mask = load_mask(mask_path)
    img_h, img_w = mask.shape
    tree_pixels = mask.sum()
    print(f'Mask size: {img_w}×{img_h}, tree pixels: {tree_pixels}')

    # 2. Distance transform
    print('Computing distance transform...')
    dist = distance_transform_edt(mask)
    max_dist = dist.max()
    print(f'Max distance from edge: {max_dist:.1f} px')

    # 3. Choose iso-levels
    # Start just inside the edge (level 0.5) up to near the medial axis.
    # More levels near the edge (where contours are longer) means more even
    # LED density across levels.
    num_levels = args.num_levels if args.num_levels > 0 else max(10, int(max_dist / 3))
    # Linear spacing from just-inside-edge to 90% of max distance
    # (beyond ~90% contours get very tiny and fragmented)
    levels = np.linspace(0.5, max_dist * 0.90, num_levels)
    print(f'Extracting contours at {num_levels} iso-levels '
          f'(d = {levels[0]:.1f} … {levels[-1]:.1f} px)')

    # 4. Extract contours at each level
    all_contours = []  # list of (contour_array, arc_length)
    for level in levels:
        contours = find_contours(dist, level)
        for c in contours:
            arc = contour_arc_lengths(c)
            length = arc[-1]
            if length >= args.min_contour_px:
                all_contours.append((c, length))

    total_arc = sum(l for _, l in all_contours)
    print(f'Found {len(all_contours)} contours, total arc length: {total_arc:.0f} px')

    if len(all_contours) == 0:
        print('ERROR: No contours found. Check mask image.')
        return

    # 5. Distribute LEDs proportionally to arc length
    target = args.num_leds
    # Allocate at least 1 LED per contour, then distribute the rest by length
    base_alloc = len(all_contours)  # 1 per contour
    remaining = max(0, target - base_alloc)
    led_counts = []
    for _, length in all_contours:
        n = 1 + int(round(remaining * length / total_arc))
        led_counts.append(n)

    # Adjust to hit the target exactly
    actual = sum(led_counts)
    diff = target - actual
    if diff != 0:
        # Add/remove from the longest contours first
        sorted_idx = sorted(range(len(all_contours)),
                            key=lambda i: all_contours[i][1], reverse=True)
        for i in sorted_idx:
            if diff == 0:
                break
            if diff > 0:
                led_counts[i] += 1
                diff -= 1
            elif led_counts[i] > 1:
                led_counts[i] -= 1
                diff += 1

    print(f'LED allocation: {sum(led_counts)} total across {len(all_contours)} contours')

    # 6. Resample each contour and collect points
    all_points = []  # list of (row, col)
    for (contour, _), n in zip(all_contours, led_counts):
        pts = resample_contour(contour, n)
        all_points.extend(pts.tolist())

    print(f'Total points: {len(all_points)}')

    # 7. Sort points for spatial coherence
    # Use angle from trunk base (bottom-center of tree), matching extract_topology.py
    points_arr = np.array(all_points)
    trunk_base = np.array([img_h * 0.55, img_w * 0.5])
    offsets = points_arr - trunk_base
    angles = np.arctan2(offsets[:, 1], -offsets[:, 0])
    sort_idx = np.argsort(angles)
    points_arr = points_arr[sort_idx]

    # 8. Convert to topology coordinates
    path_coords = [
        pixel_to_topology(r, c, img_h, img_w)
        for r, c in points_arr
    ]

    # 9. Build and write topology JSON
    topology = {
        "name": "yggdrasil",
        "numLeds": len(path_coords),
        "branches": [
            {
                "id": "contours",
                "parent": None,
                "path": path_coords,
                "ledCount": len(path_coords)
            }
        ]
    }

    output_path = args.output or str(
        jam_root / 'client' / 'src' / 'topology' / 'yggdrasil-contours.json')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f'Saved topology ({len(path_coords)} LEDs) to {output_path}')

    # Quick stats
    xs = [p[0] for p in path_coords]
    ys = [p[1] for p in path_coords]
    print(f'Coordinate ranges — X: [{min(xs):.3f}, {max(xs):.3f}]  '
          f'Y: [{min(ys):.3f}, {max(ys):.3f}]')


if __name__ == '__main__':
    main()
