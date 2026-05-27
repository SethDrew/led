#!/usr/bin/env python3
"""Visualize LED positions from topology.h on a 2D plane."""
import math

# Vertex positions (from topology.h)
VERTICES = [
    (0.00, 0.50),  # 0 — origin
    (0.25, 0.85),  # 1
    (0.50, 0.95),  # 2
    (0.78, 0.80),  # 3
    (0.95, 0.50),  # 4
    (0.55, 0.10),  # 5
    (0.25, 0.15),  # 6
]

# Strip paths: (waypoints, color, label)
STRIPS = {
    0: {"waypoints": [(0.00,0.50), (0.25,0.15), (0.50,0.95), (0.95,0.50)],
        "color": "blue", "label": "strip 0 (blue)"},
    2: {"waypoints": [(0.00,0.50), (0.25,0.85), (0.55,0.10), (0.72,0.70)],
        "color": "orange", "label": "strip 2 (orange)"},
    4: {"waypoints": [(0.00,0.50), (0.78,0.80), (0.55,0.10), (0.40,0.12)],
        "color": "black", "label": "strip 4 (black)"},
    5: {"waypoints": [(0.00,0.50), (0.95,0.50), (0.25,0.85)],
        "color": "deeppink", "label": "strip 5 (pink)"},
}

LEDS_PER_STRIP = 100

def interpolate_path(waypoints, n):
    """Interpolate n equidistant points along a polyline."""
    # Compute cumulative arc lengths
    cum = [0.0]
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i-1][0]
        dy = waypoints[i][1] - waypoints[i-1][1]
        cum.append(cum[-1] + math.sqrt(dx*dx + dy*dy))
    total = cum[-1]

    points = []
    for i in range(n):
        t = i / (n - 1)
        dist = t * total
        # Find segment
        seg = 0
        for w in range(1, len(waypoints)):
            if dist <= cum[w] or w == len(waypoints) - 1:
                seg = w - 1
                break
        seg_len = cum[seg+1] - cum[seg]
        frac = (dist - cum[seg]) / seg_len if seg_len > 1e-6 else 0.0
        frac = min(frac, 1.0)
        x = waypoints[seg][0] + frac * (waypoints[seg+1][0] - waypoints[seg][0])
        y = waypoints[seg][1] + frac * (waypoints[seg+1][1] - waypoints[seg][1])
        points.append((x, y))
    return points


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f8f8')
    ax.set_title('Bulb-Fleet LED Topology — 2D Positions', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Draw vertices
    for i, (vx, vy) in enumerate(VERTICES):
        ax.plot(vx, vy, 'ko', markersize=10, zorder=10)
        ax.annotate(f'  v{i}', (vx, vy), fontsize=11, fontweight='bold', zorder=11)

    # Draw each strip
    for sid, info in STRIPS.items():
        pts = interpolate_path(info["waypoints"], LEDS_PER_STRIP)
        color = info["color"]
        label = info["label"]

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Draw path line
        ax.plot(xs, ys, '-', color=color, alpha=0.3, linewidth=2)

        # Draw LED dots with index labels
        for idx, (px, py) in enumerate(pts):
            size = 4
            if idx == 0:
                size = 8
            elif idx == 99:
                size = 8
            ax.plot(px, py, 'o', color=color, markersize=size, zorder=5)

            # Label every 10th LED and first/last
            if idx % 10 == 0 or idx == 99:
                ax.annotate(f'{idx}', (px, py), fontsize=6, color=color,
                           ha='center', va='bottom', xytext=(0, 3),
                           textcoords='offset points', zorder=6)

        # Legend entry
        ax.plot([], [], 'o-', color=color, label=label, markersize=5)

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.2)

    out = '/Users/sethdrew/Documents/projects/led/festicorn/topo-tester/topology_map.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

except ImportError:
    print("matplotlib not available, printing text positions instead")
    for sid, info in STRIPS.items():
        pts = interpolate_path(info["waypoints"], LEDS_PER_STRIP)
        print(f"\n--- {info['label']} ---")
        for i, (x, y) in enumerate(pts):
            if i % 10 == 0 or i == 99:
                print(f"  LED {i:3d}: ({x:.3f}, {y:.3f})")
