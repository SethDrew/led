// effects.h — helix-canopy render functions (hardware-independent).
//
// THE source of truth for effect math. Both the desktop sim
// (tools/sim/effect_sim.cpp) and any future device firmware #include THIS file
// directly — never a copy. So there is no second copy to drift (the ledsim
// doctrine, enforced structurally instead of by discipline).
//
// A render function fills out[HC_NUM_LEDS] for absolute time t (seconds), using
// each LED's world position from LED_POS[]. Functions of absolute t are
// frame-rate independent by construction (no per-frame decay constants needed).
//
// No Arduino, no FastLED, no globals: pure (positions, t) -> colors. The device
// main supplies the pixel sink (NeoPixelBus/FastLED); the sim writes a CSV.

#pragma once
#include <math.h>
#include <stdint.h>
#include "topology.h"   // LED_POS, HC_NUM_LEDS, HC_BOUND_*, HC_CANOPY_Z

struct RGBf { float r, g, b; };   // linear 0..1; the sink applies gamma/dither

static inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// ── Effect: "pour" ──────────────────────────────────────────────────────
// Light ignites across the heptagon canopy, then a wavefront falls straight
// down the helix to the floor, leaving a fading comet trail above it.
// REQUIRES real z — this is the cross-form effect index-space can't express.
static inline void fx_pour(RGBf* out, float t) {
    const float z_top = HC_CANOPY_Z;
    const float z_bot = HC_BOUND_MIN.z;
    const float period = 3.0f;           // seconds per fall
    const float sigma  = 0.18f;          // leading-edge thickness (m)
    const float tail   = 0.55f;          // trail length above the front (m)

    float phase = fmodf(t, period) / period;         // 0..1
    float zf = z_top - phase * (z_top - z_bot);      // front descends

    for (int i = 0; i < HC_NUM_LEDS; i++) {
        float dz = LED_POS[i].z - zf;                // >0 = above the front
        float b;
        if (dz >= 0.0f) {
            // leading gaussian + exponential comet trail above
            b = expf(-(dz * dz) / (2.0f * sigma * sigma));
            b = fmaxf(b, expf(-dz / tail) * 0.6f);
        } else {
            // just below the front: sharp fade (light has passed)
            b = expf(-(dz * dz) / (2.0f * (sigma * 0.5f) * (sigma * 0.5f)));
        }
        b = clamp01(b);
        // cool leading edge -> deeper blue in the trail
        out[i].r = clamp01(b * b * 0.6f);
        out[i].g = clamp01(b * 0.85f);
        out[i].b = clamp01(0.25f * b + 0.75f * b * b);
    }
}

// ── Effect: "axis_radial" ───────────────────────────────────────────────
// A ring expands outward from the central (helix) axis: it lights the helix
// when the ring radius ~ helix radius, then sweeps out across the canopy.
// Demonstrates horizontal-distance-from-axis as a coordinate.
static inline void fx_axis_radial(RGBf* out, float t) {
    const float maxR  = fmaxf(HC_BOUND_MAX.x, HC_BOUND_MAX.y);
    const float sigma = 0.20f;
    const float speed = 0.9f;            // rings/sec
    float rf = (0.5f + 0.5f * sinf(t * speed * 2.0f * (float)M_PI)) * maxR;

    for (int i = 0; i < HC_NUM_LEDS; i++) {
        float r = hypotf(LED_POS[i].x, LED_POS[i].y);
        float d = r - rf;
        float b = clamp01(expf(-(d * d) / (2.0f * sigma * sigma)));
        // warm amber ring
        out[i].r = clamp01(b);
        out[i].g = clamp01(b * 0.55f);
        out[i].b = clamp01(b * 0.12f);
    }
}

// ── Registry (add effects here; the sim exposes them by name) ────────────
typedef void (*HcEffectFn)(RGBf*, float);
struct HcEffect { const char* name; HcEffectFn fn; };
static const HcEffect HC_EFFECTS[] = {
    { "pour",        fx_pour },
    { "axis_radial", fx_axis_radial },
};
static const int HC_NUM_EFFECTS = sizeof(HC_EFFECTS) / sizeof(HC_EFFECTS[0]);
