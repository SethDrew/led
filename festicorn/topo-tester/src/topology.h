/*
 * topology.h — 2D pixel map for bulb-fleet installation.
 *
 * 6 strips × 100 LEDs, all originating from vertex 0.
 * Coordinates normalized so vertex 0 is near the center-left and the
 * overall shape fits roughly in a 1.0 × 1.0 box.
 *
 * FIRST DRAFT — adjust vertex positions after testing against physical
 * installation. All tunable constants are at the top.
 */

#pragma once
#include <math.h>

struct Vec2 { float x, y; };

// ── Vertex coordinates (easy to adjust) ─────────────────────────
// Rough hex layout from the hand-drawn diagram, viewed from above.
// Origin = vertex 0 on the left. Y+ is "up" in the overhead view.
//
//        1           2
//         \         / \
//          \       /   \
//   0 ------+-----+     3
//          /       \   /
//         /         \ /
//        6           5         4
//
// Normalize so 0 is at ~(0.0, 0.5) and the shape spans roughly 0–1.

static const Vec2 VERTICES[] = {
    { 0.00f, 0.50f },   // 0 — origin (left)
    { 0.25f, 0.85f },   // 1 — upper-left
    { 0.50f, 0.95f },   // 2 — top-center
    { 0.78f, 0.80f },   // 3 — upper-right
    { 0.95f, 0.50f },   // 4 — far right
    { 0.55f, 0.10f },   // 5 — bottom-center
    { 0.25f, 0.15f },   // 6 — lower-left
};

// ── Strip path definitions ──────────────────────────────────────
// Each strip is a polyline from vertex 0 through waypoints to an endpoint.
// Endpoint can be a vertex or an arbitrary coordinate (orange strip).
//
// Strip-to-color mapping (from diagram + hue offsets in firmware):
//   strip 0 (hue   0) — black #1:  0 → 1 → 2 → 3
//   strip 1 (hue  42) — blue  #1:  0 → 6 → 5 → 2
//   strip 2 (hue  85) — pink  #1:  0 → 5 → 4 → 1
//   strip 3 (hue 128) — orange:    0 → 1 → 6 → (no-man's land)
//   strip 4 (hue 170) — blue  #2:  0 → 2 → 4
//   strip 5 (hue 213) — black #2:  0 → 3 → between 5&6
//
// NOTE: strip↔color mapping is a best guess from the diagram.
// Swap definitions here if a strip maps to the wrong physical path.

static const int MAX_WAYPOINTS = 5;

struct StripPath {
    int       nWaypoints;
    Vec2      waypoints[MAX_WAYPOINTS];
};

static const StripPath STRIP_PATHS[6] = {
    // strip 0 — black #1: 0 → 1 → 2 → 3
    { 4, { {0.00f,0.50f}, {0.25f,0.85f}, {0.50f,0.95f}, {0.78f,0.80f} } },

    // strip 1 — blue #1: 0 → 6 → 5 → 2
    { 4, { {0.00f,0.50f}, {0.25f,0.15f}, {0.55f,0.10f}, {0.50f,0.95f} } },

    // strip 2 — pink: 0 → 5 → 4 → 1
    { 4, { {0.00f,0.50f}, {0.55f,0.10f}, {0.95f,0.50f}, {0.25f,0.85f} } },

    // strip 3 — orange: 0 → 1 → 6 → no-man's land (midpoint below 5-6 line)
    { 4, { {0.00f,0.50f}, {0.25f,0.85f}, {0.25f,0.15f}, {0.40f,0.02f} } },

    // strip 4 — blue #2: 0 → 2 → 4
    { 3, { {0.00f,0.50f}, {0.50f,0.95f}, {0.95f,0.50f} } },

    // strip 5 — black #2: 0 → 3 → between 5 & 6
    { 3, { {0.00f,0.50f}, {0.78f,0.80f}, {0.40f,0.12f} } },
};

// ── Precomputed LED positions ───────────────────────────────────

#define TOPO_STRIPS     6
#define TOPO_LEDS       100

static Vec2 ledPos[TOPO_STRIPS][TOPO_LEDS];

// Linearly interpolate TOPO_LEDS equidistant points along a polyline.
static void initTopology() {
    for (int s = 0; s < TOPO_STRIPS; s++) {
        const StripPath &path = STRIP_PATHS[s];

        // 1. Compute cumulative arc-length at each waypoint.
        float segLen[MAX_WAYPOINTS]; // segLen[i] = length of segment i→i+1
        float cumLen[MAX_WAYPOINTS]; // cumLen[i] = total length up to waypoint i
        cumLen[0] = 0.0f;
        for (int w = 1; w < path.nWaypoints; w++) {
            float dx = path.waypoints[w].x - path.waypoints[w-1].x;
            float dy = path.waypoints[w].y - path.waypoints[w-1].y;
            segLen[w-1] = sqrtf(dx*dx + dy*dy);
            cumLen[w] = cumLen[w-1] + segLen[w-1];
        }
        float totalLen = cumLen[path.nWaypoints - 1];

        // 2. Place 100 LEDs equidistant along the polyline.
        for (int i = 0; i < TOPO_LEDS; i++) {
            float t = (float)i / (float)(TOPO_LEDS - 1); // 0.0 … 1.0
            float dist = t * totalLen;

            // Find which segment this distance falls in.
            int seg = 0;
            for (int w = 1; w < path.nWaypoints; w++) {
                if (dist <= cumLen[w] || w == path.nWaypoints - 1) {
                    seg = w - 1;
                    break;
                }
            }

            float segStart = cumLen[seg];
            float frac = (segLen[seg] > 1e-6f)
                       ? (dist - segStart) / segLen[seg]
                       : 0.0f;
            if (frac > 1.0f) frac = 1.0f;

            ledPos[s][i].x = path.waypoints[seg].x
                           + frac * (path.waypoints[seg+1].x - path.waypoints[seg].x);
            ledPos[s][i].y = path.waypoints[seg].y
                           + frac * (path.waypoints[seg+1].y - path.waypoints[seg].y);
        }
    }
}
