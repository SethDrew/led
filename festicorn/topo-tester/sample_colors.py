"""Sample HSV/BGR values at hand-picked points on each ink stroke to calibrate thresholds."""
import cv2
import numpy as np

img = cv2.imread("/Users/sethdrew/Downloads/20260526_182005.jpg")
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
h, w = img.shape[:2]
scale = 1400 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(f"shape: {img.shape}")

# Save a numbered grid image so we can pick points by looking.
grid = img.copy()
H, W = img.shape[:2]
for x in range(0, W, 50):
    cv2.line(grid, (x, 0), (x, H), (0, 255, 0), 1)
    cv2.putText(grid, str(x), (x+2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
for y in range(0, H, 50):
    cv2.line(grid, (0, y), (W, y), (0, 255, 0), 1)
    cv2.putText(grid, str(y), (2, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
cv2.imwrite("/Users/sethdrew/Documents/projects/led/festicorn/topo-tester/grid.png", grid)

# Sample a swath of regions and report dominant non-white HSV.
# Region tuples: name, (x0,y0,x1,y1)
regions = {
    "blue_1to2":     (590, 180, 640, 230),
    "blue_1to6":     (330, 480, 380, 530),
    "orange_1to5":   (540, 580, 590, 630),
    "pink_horiz_L":  (430, 430, 480, 480),
    "pink_horiz_R":  (760, 480, 810, 530),
    "black_2to3":    (900, 160, 950, 210),
    "black_3to4":    (1080, 450, 1130, 500),
    "paper_white":   (820, 320, 870, 370),
    "ruled_line":    (1300, 400, 1340, 500),
}
for name, (x0, y0, x1, y1) in regions.items():
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
    # Filter out near-white (V high, S low) to find ink-ish pixels.
    # find darker / more saturated pixels (ink)
    nonwhite = patch[(patch[:, 1] > 40) | (patch[:, 2] < 160)]
    if len(nonwhite) == 0:
        print(f"{name:25s}  all whiteish")
        continue
    h_med, s_med, v_med = np.median(nonwhite, axis=0)
    print(f"{name:25s}  N={len(nonwhite):4d}  H={h_med:3.0f} S={s_med:3.0f} V={v_med:3.0f}")
