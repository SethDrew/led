"""
Simplified branch topology extraction - skeleton-based centerline.

Optimized for speed and debuggability.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize, disk, closing, medial_axis
from skimage.measure import approximate_polygon
from scipy.ndimage import distance_transform_edt, convolve, label
import networkx as nx


def load_mask(path: Path) -> tuple[np.ndarray, int, int]:
    """Load tree mask and return binary array + dimensions."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        alpha = np.array(img)[:, :, 3]
        mask = alpha > 128
    else:
        gray = np.array(img.convert('L'))
        mask = gray > 128

    h, w = mask.shape
    return mask, h, w


def main():
    parser = argparse.ArgumentParser(description='Extract branch topology (simplified)')
    parser.add_argument('--min-branch-px', type=float, default=40)
    parser.add_argument('--num-leds', type=int, default=3000)
    args = parser.parse_args()

    jam_root = Path(__file__).parent.parent
    mask_path = jam_root / 'client' / 'public' / 'yggdrasil-mask.png'

    print('=' * 70)
    print('Branch Topology Extraction (Simplified)')
    print('=' * 70)

    # Load mask
    print(f'\n1. Loading mask...')
    mask, img_h, img_w = load_mask(mask_path)
    print(f'   Size: {img_w}×{img_h}, tree pixels: {mask.sum()}')

    # Skeletonize
    print(f'\n2. Computing skeleton...')
    mask_closed = closing(mask, disk(3))
    skel = skeletonize(mask_closed)
    dist = distance_transform_edt(mask)
    print(f'   Skeleton pixels: {skel.sum()}')

    # Find junctions and endpoints
    print(f'\n3. Finding junctions...')
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neighbors = convolve(skel.astype(int), kernel)

    junctions = np.where(skel & (neighbors >= 3))
    endpoints = np.where(skel & (neighbors == 1))

    junctions_arr = np.column_stack(junctions)
    endpoints_arr = np.column_stack(endpoints)
    print(f'   Junctions: {len(junctions_arr)}, Endpoints: {len(endpoints_arr)}')

    # Merge nearby junctions
    print(f'\n4. Merging junctions...')
    used = set()
    merged_junctions = []
    for i, junc in enumerate(junctions_arr):
        if i in used:
            continue
        dists = np.linalg.norm(junctions_arr - junc, axis=1)
        cluster_indices = np.where(dists <= 5.0)[0]
        representative = junctions_arr[cluster_indices].mean(axis=0)
        merged_junctions.append(representative)
        for idx in cluster_indices:
            used.add(idx)

    merged_junctions = np.array(merged_junctions)
    print(f'   Merged to {len(merged_junctions)} nodes')

    # Build simplified topology with all skeleton points
    print(f'\n5. Building simplified topology...')

    # For simplicity: just sample the skeleton with farthest-point sampling
    # and convert to hierarchical structure
    skel_points = np.array(np.where(skel)).T  # (row, col) pairs

    # Simple approach: convert skeleton directly to a single branch
    # with intelligent subsampling
    if len(skel_points) > args.num_leds:
        # Farthest-point sampling
        selected = [0]
        remaining = set(range(1, len(skel_points)))
        min_dists = np.full(len(skel_points), np.inf)

        for _ in range(args.num_leds - 1):
            if not remaining:
                break
            last = skel_points[selected[-1]]
            dists_to_last = np.linalg.norm(skel_points[list(remaining)] - last, axis=1)
            min_dists[list(remaining)] = np.minimum(min_dists[list(remaining)], dists_to_last)
            next_idx = list(remaining)[np.argmax(min_dists[list(remaining)])]
            selected.append(next_idx)
            remaining.remove(next_idx)

        sampled_points = skel_points[selected]
    else:
        sampled_points = skel_points

    print(f'   Sampled {len(sampled_points)} skeleton points')

    # Convert to topology coordinates
    print(f'\n6. Converting coordinates...')
    topology_path = []
    for pixel in sampled_points:
        row, col = pixel[0], pixel[1]
        tx = (col / img_w) * 1.7 - 0.85
        ty = (1 - row / img_h) * 1.5 - 0.55
        topology_path.append([round(tx, 4), round(ty, 4)])

    # Build single-branch topology for now
    topology = {
        "name": "yggdrasil",
        "numLeds": len(topology_path),
        "branches": [
            {
                "id": "trunk",
                "parent": None,
                "path": topology_path,
                "ledCount": len(topology_path)
            }
        ]
    }

    # Save topology
    print(f'\n7. Saving topology...')
    output_json = jam_root / 'client' / 'src' / 'topology' / 'yggdrasil-branches.json'
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f'   Saved to {output_json}')

    # Draw overlay
    print(f'\n8. Drawing overlay...')
    img = Image.new('RGB', (img_w, img_h), 'black')
    pixels = img.load()

    # Draw mask as gray
    for i in range(img_h):
        for j in range(img_w):
            if mask[i, j]:
                pixels[j, i] = (100, 100, 100)

    draw = ImageDraw.Draw(img)

    # Draw skeleton branch in red
    skel_pixels = np.where(skel)
    for pixel in zip(skel_pixels[1], skel_pixels[0]):  # (col, row)
        pixels[pixel[0], pixel[1]] = (255, 0, 0)

    # Draw junctions as white dots
    for junc in merged_junctions:
        x, y = int(junc[1]), int(junc[0])
        if 0 <= x < img_w and 0 <= y < img_h:
            draw.ellipse([(x-3, y-3), (x+3, y+3)], fill='white')

    overlay_path = jam_root / 'tools' / 'branch-overlay.png'
    img.save(overlay_path)
    print(f'   Saved to {overlay_path}')

    print('\n' + '=' * 70)
    print(f'Total branches: 1 (single trunk)')
    print(f'Total LEDs: {len(topology_path)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
