"""
Extract LED-strip paths from a hand-drawn topology diagram.

Approach (revised): the ink on this photo is faint and the paper has light-blue
ruled lines + pink margin lines that compete with the actual ink strokes. Instead
of color-thresholding the raw photo, we:

  1. Rotate + resize.
  2. Background-subtract via a large-blur estimate of the paper, giving us a
     "darkness" map that is high wherever ink (any color) sits.
  3. Threshold darkness => binary "any-ink" mask.
  4. For each ink pixel, look up its hue/chroma in HSV and classify into one of
     {blue, orange, pink, black, ruled-line, other}. Ruled lines get rejected
     because they are very light (low darkness) AND specifically blue-cyan.
  5. Per-color: morphology cleanup, skeletonize, walk skeleton end-to-end,
     downsample to waypoints.
"""

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

PHOTO = Path("/Users/sethdrew/Downloads/20260526_182005.jpg")
OUT_DIR = Path("/Users/sethdrew/Documents/projects/led/festicorn/topo-tester")
OUT = OUT_DIR / "diagram_parsed.png"
MAX_DIM = 1400


def load_and_orient(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"failed to load {path}")
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def darkness_map(bgr: np.ndarray) -> np.ndarray:
    """Return uint8 image where ink-y pixels are bright."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Estimate local paper brightness with a big blur, then divide.
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=35)
    blur = np.clip(blur, 1, 255).astype(np.float32)
    # ratio < 1 where pixel is darker than local paper.
    ratio = gray.astype(np.float32) / blur
    darkness = np.clip((1.0 - ratio) * 400.0, 0, 255).astype(np.uint8)
    return darkness


def classify_pixel(h, s, v, dark):
    """Return color name for an ink pixel, or None to reject."""
    # Pale-blue ruled lines: low darkness AND H roughly cyan AND moderate saturation.
    if dark < 35:
        return None
    # Black: low chroma OR very low value.
    if s < 55 and v < 130:
        return "black"
    # Orange ink: H roughly 5..22 (red-orange-yellow), needs visible saturation.
    if 3 <= h <= 22 and s > 55:
        return "orange"
    # Pink / red ink (margin-line ink is similar hue, filtered out below).
    if (h <= 8 or h >= 158) and s > 70:
        return "pink"
    # Blue ink: H roughly 95..130. Distinguish from ruled lines by requiring
    # higher darkness (already filtered above) AND H not in the pale-cyan band.
    if 90 <= h <= 135 and s > 45:
        return "blue"
    return None


def page_mask(bgr: np.ndarray) -> np.ndarray:
    """Find the bright paper region (high V, low-ish S) and return its convex hull."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    bright = (v > 150).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=3)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if num <= 1:
        return np.ones_like(bright) * 255
    biggest = 1 + np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, num)])
    page = (labels == biggest).astype(np.uint8) * 255
    # Convex hull to cover small dark indents (vertex labels written outside).
    ys, xs = np.where(page > 0)
    if len(xs) == 0:
        return page
    hull = cv2.convexHull(np.column_stack([xs, ys]))
    page = np.zeros_like(page)
    cv2.fillConvexPoly(page, hull, 255)
    # Erode a bit so we don't include the page edge shadow.
    page = cv2.erode(page, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    cv2.imwrite(str(OUT_DIR / "debug_page.png"), page)
    return page


def color_masks(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark = darkness_map(bgr)
    page = page_mask(bgr) > 0
    H, W = dark.shape

    # Vectorized classification.
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    ink = (dark >= 35) & page  # candidate ink pixels, restricted to paper

    # Build per-color boolean masks.
    black_m = ink & (s_ch < 55) & (v_ch < 130)
    orange_m = ink & (h_ch >= 3) & (h_ch <= 22) & (s_ch > 55) & (v_ch > 60)
    pink_m = ink & ((h_ch <= 8) | (h_ch >= 158)) & (s_ch > 70) & (v_ch > 60)
    blue_m = ink & (h_ch >= 90) & (h_ch <= 135) & (s_ch > 45)

    # Reject black hits that overlap with blue/orange/pink (color wins over black).
    colored = orange_m | pink_m | blue_m
    black_m = black_m & ~colored
    # Pink shouldn't claim orange territory and vice-versa.
    orange_m = orange_m & ~pink_m

    # Ruled-line rejection for blue: blue ink strokes are darker than ruled lines.
    blue_m = blue_m & (dark > 60)

    # Save darkness for debug.
    cv2.imwrite(str(OUT_DIR / "debug_darkness.png"), dark)

    out = {
        "blue":   (blue_m.astype(np.uint8) * 255),
        "orange": (orange_m.astype(np.uint8) * 255),
        "pink":   (pink_m.astype(np.uint8) * 255),
        "black":  (black_m.astype(np.uint8) * 255),
    }
    return out


def clean_mask(mask: np.ndarray) -> np.ndarray:
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # First aggressive close to bridge dashed strokes.
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k15, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)
    # Keep the dominant connected components: any component >= 8% of largest or >= 400 px.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
    areas.sort(key=lambda t: t[1], reverse=True)
    biggest = areas[0][1]
    keep = np.zeros_like(m)
    for i, a in areas:
        if a >= max(400, biggest * 0.08):
            keep[labels == i] = 255
    # Final close to bridge small gaps in dashed strokes.
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k5, iterations=3)
    return keep


def skeleton_path(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.sum() == 0:
        return []
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []
    pts = set(zip(xs.tolist(), ys.tolist()))

    def neighbors(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in pts]

    def bfs_farthest(src):
        seen = {src: None}
        dq = deque([src])
        last = src
        while dq:
            p = dq.popleft()
            last = p
            for n in neighbors(p):
                if n not in seen:
                    seen[n] = p
                    dq.append(n)
        return last, seen

    endpoints = [p for p in pts if len(neighbors(p)) == 1]
    start = endpoints[0] if endpoints else next(iter(pts))
    far1, _ = bfs_farthest(start)
    far2, parents = bfs_farthest(far1)
    path = []
    cur = far2
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def downsample(path, n=30):
    if len(path) <= n:
        return path
    idx = np.linspace(0, len(path) - 1, n).astype(int)
    return [path[i] for i in idx]


DRAW_COLOR = {
    "blue":   (255, 80, 0),
    "orange": (0, 140, 255),
    "pink":   (180, 30, 220),
    "black":  (30, 30, 30),
}


def main():
    img = load_and_orient(PHOTO)
    raw_masks = color_masks(img)

    overlay = img.copy()
    results = {}
    for name, raw in raw_masks.items():
        cleaned = clean_mask(raw)
        cv2.imwrite(str(OUT_DIR / f"mask_{name}.png"), cleaned)
        path = skeleton_path(cleaned)
        wps = downsample(path, 30)
        results[name] = wps

        c = DRAW_COLOR[name]
        for x, y in path:
            cv2.circle(overlay, (x, y), 1, c, -1)
        for i, (x, y) in enumerate(wps):
            cv2.circle(overlay, (x, y), 6, c, 2)
            if i in (0, len(wps) - 1):
                cv2.circle(overlay, (x, y), 11, (0, 255, 0), 2)

    blended = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)
    cv2.imwrite(str(OUT), blended)

    print(f"\nImage: {img.shape[1]}x{img.shape[0]} (after rotation+resize)\n")
    for name, pts in results.items():
        print(f"=== {name} ({len(pts)} waypoints) ===")
        if not pts:
            print("  (no path)\n")
            continue
        print(f"  start: {pts[0]}    end: {pts[-1]}")
        for i, (x, y) in enumerate(pts):
            print(f"  [{i:2d}] ({x:4d}, {y:4d})")
        print()


if __name__ == "__main__":
    main()
