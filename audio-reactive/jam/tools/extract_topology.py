"""
Extract LED positions from the Yggdrasil tree mask image using skeletonization
and farthest-point sampling for even spatial distribution.

Usage:
    /path/to/.venv/bin/python tools/extract_topology.py [--num-leds 400] [--output path]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial import KDTree


def load_mask(path: Path) -> np.ndarray:
    """Load tree mask, return binary array (True = tree)."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        alpha = np.array(img)[:, :, 3]
        return alpha > 128
    else:
        gray = np.array(img.convert('L'))
        return gray > 128


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """
    Select n points from `points` using farthest-point sampling.
    Guarantees even spatial spread.
    """
    if len(points) <= n:
        return points

    # Start from the point closest to the centroid (trunk-ish)
    centroid = points.mean(axis=0)
    dists_to_center = np.linalg.norm(points - centroid, axis=1)
    first_idx = np.argmin(dists_to_center)

    selected = [first_idx]
    min_dists = np.full(len(points), np.inf)

    for _ in range(n - 1):
        # Update min distances to selected set
        last = points[selected[-1]]
        d = np.linalg.norm(points - last, axis=1)
        min_dists = np.minimum(min_dists, d)
        # Pick the farthest point from all selected
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

    return points[selected]


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


def main():
    parser = argparse.ArgumentParser(description='Extract LED topology from tree mask')
    parser.add_argument('--num-leds', type=int, default=400,
                        help='Number of LEDs to distribute (default: 400)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--dilate', type=int, default=0,
                        help='Dilate mask by N pixels before skeletonizing')
    args = parser.parse_args()

    jam_root = Path(__file__).parent.parent
    mask_path = jam_root / 'client' / 'public' / 'yggdrasil-mask.png'

    print(f'Loading mask from {mask_path}')
    mask = load_mask(mask_path)
    img_h, img_w = mask.shape
    print(f'Mask size: {img_w}x{img_h}, tree pixels: {mask.sum()}')

    if args.dilate > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        mask = ndimage.binary_dilation(mask, structure=struct, iterations=args.dilate)
        print(f'After dilation ({args.dilate}px): tree pixels: {mask.sum()}')

    # Skeletonize
    print('Computing skeleton...')
    skel = skeletonize(mask)
    skel_count = np.sum(skel)
    print(f'Skeleton pixels: {skel_count}')

    # Get skeleton coordinates as (row, col)
    rows, cols = np.where(skel)
    skel_points = np.column_stack([rows, cols]).astype(np.float64)

    # Farthest-point sampling for even spatial spread
    print(f'Sampling {args.num_leds} LEDs via farthest-point sampling...')
    sampled = farthest_point_sample(skel_points, args.num_leds)
    print(f'Sampled {len(sampled)} LED positions')

    # Sort by a spatial ordering for smooth rainbow color assignment.
    # Use angle from trunk base (bottom center of tree) — this gives a
    # natural radial sweep: roots left, trunk, canopy left→top→right, roots right
    # Trunk base is approximately at the center-bottom of the mask
    trunk_base = np.array([img_h * 0.55, img_w * 0.5])  # roughly where trunk meets roots
    offsets = sampled - trunk_base
    # Angle from trunk base, starting at bottom-left going clockwise
    angles = np.arctan2(offsets[:, 1], -offsets[:, 0])  # negate row so up = positive
    sort_idx = np.argsort(angles)
    sampled = sampled[sort_idx]

    # Convert to topology coordinates
    path_coords = [
        pixel_to_topology(r, c, img_h, img_w)
        for r, c in sampled
    ]

    topology = {
        "name": "yggdrasil",
        "numLeds": len(path_coords),
        "branches": [
            {
                "id": "skeleton",
                "parent": None,
                "path": path_coords,
                "ledCount": len(path_coords)
            }
        ]
    }

    output_path = args.output or str(jam_root / 'tools' / 'extracted_topology.json')
    with open(output_path, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f'Saved topology to {output_path}')


if __name__ == '__main__':
    main()
