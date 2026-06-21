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

// ── "nebula" — the composed two-layer effect, INDEX vs TOPOLOGY ─────────
//
// The real nebula (static-animations/effects/) is NebulaBackground (breathing
// color waves, REPLACE) + CrawlingStarsForeground (drifting warm-white orbs with
// trails, ADD). We provide BOTH layers in two forms:
//
//   fx_nebula        — faithful INDEX-BASED port. Background wave runs along
//                      i/N; orbs crawl along pixel index. On this sculpture the
//                      wave/orbs travel WIRE ORDER (strand A, then strand B, then
//                      canopy serpentine) — two LEDs inches apart in 3D can be at
//                      opposite phases, and an orb climbing strand A then
//                      "teleports" to the bottom of strand B.
//
//   fx_nebula_native — the SAME effect rebuilt in 3D space. Background wave runs
//                      vertically through z (identical on both strands at equal
//                      height); orbs are glowing spheres that rise through space,
//                      lighting whatever LED is physically near them — so they
//                      cross both helix strands together and bloom on the canopy.
//
// These two are the sharpest statement of the whole thesis: identical artistic
// intent, one oblivious to the geometry and one expressed through it.
//
// Both layers are STATEFUL (orbs + decaying trail buffer). The functions hold
// function-local state stepped once per call; the sim calls each frame once in
// order (fresh process per run), so this reproduces the original frame-based
// integration. This is a faithful legacy port — frame-based, not dt-correct;
// the topology-native production effects (fx_pour etc.) are written dt-correct.

static inline float smoothstep01(float t) {
    if (t < 0) t = 0; if (t > 1) t = 1;
    return t * t * (3.0f - 2.0f * t);
}
static inline uint32_t xs32(uint32_t& s) {  // deterministic PRNG (reproducible sims)
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
static inline float xsf(uint32_t& s) { return (xs32(s) & 0xFFFFFF) / (float)0x1000000; }

// Shared background: breathing color wave. phase01 selects the spatial axis —
// pass i/N for index-native, or normalized height for topology-native.
static inline void nebula_bg_pixel(float phase01, float frame, RGBf& o) {
    const float BREATH_FREQUENCY  = 0.035f;
    const float BREATH_CENTER     = 51.0f;
    const float BREATH_AMPLITUDE  = 38.0f;
    const float SPATIAL_AMPLITUDE = 51.0f;
    const float SPATIAL_SPEED     = 0.02f;
    const float BACKGROUND_MAX    = 153.0f;

    float breathing = BREATH_CENTER + BREATH_AMPLITUDE * sinf(frame * BREATH_FREQUENCY);
    float phase = phase01 + frame * SPATIAL_SPEED;
    float spatial = SPATIAL_AMPLITUDE * (0.5f + 0.5f * cosf(2.0f * (float)M_PI * phase));
    float combined = breathing + spatial;
    if (combined < 0) combined = 0;
    if (combined > BACKGROUND_MAX) combined = BACKGROUND_MAX;
    float v = combined / 255.0f;

    float colorPhase = phase01 * (float)M_PI * 2.0f + frame * BREATH_FREQUENCY;
    float colorShift = 0.5f + 0.5f * sinf(colorPhase);
    o.r = clamp01((20.0f  + colorShift * 235.0f) / 255.0f * v);
    o.g = clamp01((30.0f  - colorShift *  20.0f) / 255.0f * v);
    o.b = clamp01((255.0f - colorShift * 125.0f) / 255.0f * v);
}

static inline void star_add(RGBf& o, float v8) {  // v8 in 0..255, warm white, ADD
    float v = v8 / 255.0f;
    o.r = clamp01(o.r + (255.0f / 255.0f) * v);
    o.g = clamp01(o.g + (240.0f / 255.0f) * v);
    o.b = clamp01(o.b + (200.0f / 255.0f) * v);
}

// The original was tuned on a 25-LED strip with 5 orbs (~0.2 orbs/pixel — dense).
// Scale orb count to LED count so the particle-to-background ratio is preserved
// on this much larger sculpture, instead of reading sparse.
#define NEB_REF_PIXELS 25
#define NEB_REF_ORBS   5
#define NEB_MAXSLOTS   (NEB_REF_ORBS * HC_NUM_LEDS / NEB_REF_PIXELS + 8)
struct NebOrb { bool active; float pos; float vel; float age; float life; };       // index space
struct NebOrb3 { bool active; float x, y, z, vx, vy, vz, age, life; };             // 3D space

// ── fx_nebula — faithful index-based background + crawling stars ────────
static inline void fx_nebula(RGBf* out, float t) {
    const float NEBULA_FPS = 30.0f;
    const float N = (float)HC_NUM_LEDS;
    const float speed = 0.3f, orbSize = 7.0f;
    const int   maxOrbs = NEB_REF_ORBS * HC_NUM_LEDS / NEB_REF_PIXELS;  // density-matched
    float frame = t * NEBULA_FPS;

    static bool init = false;
    static NebOrb orbs[NEB_MAXSLOTS];
    static float led[HC_NUM_LEDS];
    static uint32_t rng = 0xC0FFEEu;
    if (!init) { for (int i = 0; i < NEB_MAXSLOTS; i++) orbs[i].active = false;
                 for (int i = 0; i < HC_NUM_LEDS; i++) led[i] = 0; init = true; }

    // background
    for (int i = 0; i < HC_NUM_LEDS; i++)
        nebula_bg_pixel((float)i / N, frame, out[i]);

    // ── crawling stars (one frame step) ──
    float decay = 1.0f - 1.0f / orbSize;
    for (int i = 0; i < HC_NUM_LEDS; i++) { led[i] *= decay; if (led[i] < 3) led[i] = 0; }

    int active = 0;
    for (int i = 0; i < NEB_MAXSLOTS; i++) if (orbs[i].active) active++;
    // Maintain ~maxOrbs population (refill inactive slots gradually for organic
    // turnover); the original's single-spawn-at-3% can't fill a large count.
    for (int i = 0; i < NEB_MAXSLOTS && active < maxOrbs; i++) {
        if (orbs[i].active) continue;
        if (xsf(rng) > 0.08f) continue;
        orbs[i].active = true;
        orbs[i].pos = xsf(rng) * N;
        orbs[i].vel = (xsf(rng) < 0.5f ? 1.f : -1.f) * speed * (0.5f + xsf(rng) * 0.5f);
        orbs[i].age = 0; orbs[i].life = (40.0f + xsf(rng) * 100.0f) * (0.3f / speed);
        active++;
    }
    for (int i = 0; i < NEB_MAXSLOTS; i++) {
        if (!orbs[i].active) continue;
        orbs[i].age += 1.0f;
        if (orbs[i].age >= orbs[i].life) { orbs[i].active = false; continue; }
        orbs[i].pos += orbs[i].vel;
        if (orbs[i].pos < 0) orbs[i].pos += N;
        if (orbs[i].pos >= N) orbs[i].pos -= N;
        float lc = orbs[i].age / orbs[i].life;
        float lb = (lc < 0.4f) ? smoothstep01(lc / 0.4f)
                 : (lc > 0.6f) ? smoothstep01((1.0f - lc) / 0.4f) : 1.0f;
        int p = (int)orbs[i].pos;
        if (p >= 0 && p < HC_NUM_LEDS) { float nv = led[p] + lb * 153.0f; led[p] = nv > 255 ? 255 : nv; }
    }
    for (int i = 0; i < HC_NUM_LEDS; i++) if (led[i] > 2) star_add(out[i], led[i]);
}

// ── fx_nebula_native — same effect, expressed through 3D space ──────────
static inline void fx_nebula_native(RGBf* out, float t) {
    const float NEBULA_FPS = 30.0f;
    // Fewer than the index version: each 3D orb glows a ~0.22m sphere covering
    // many LEDs, so a smaller count yields comparable particle coverage.
    const int   maxOrbs = NEB_REF_ORBS * HC_NUM_LEDS / NEB_REF_PIXELS / 6;
    const float orbSize = 7.0f;
    const float sigma = 0.22f;               // orb glow radius (m)
    const float rise  = 0.030f;              // m per frame upward (~0.9 m/s)
    float frame = t * NEBULA_FPS;
    float zspan = HC_BOUND_MAX.z - HC_BOUND_MIN.z;

    static bool init = false;
    static NebOrb3 orbs[NEB_MAXSLOTS];
    static float led[HC_NUM_LEDS];
    static uint32_t rng = 0xBADCAFEu;
    if (!init) { for (int i = 0; i < NEB_MAXSLOTS; i++) orbs[i].active = false;
                 for (int i = 0; i < HC_NUM_LEDS; i++) led[i] = 0; init = true; }

    // background: wave travels vertically through z (identical on both strands)
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        float zn = (LED_POS[i].z - HC_BOUND_MIN.z) / (zspan > 1e-6f ? zspan : 1.0f);
        nebula_bg_pixel(zn, frame, out[i]);
    }

    // ── rising 3D orbs (one frame step) ──
    float decay = 1.0f - 1.0f / orbSize;
    for (int i = 0; i < HC_NUM_LEDS; i++) { led[i] *= decay; if (led[i] < 3) led[i] = 0; }

    int active = 0;
    for (int i = 0; i < NEB_MAXSLOTS; i++) if (orbs[i].active) active++;
    for (int i = 0; i < NEB_MAXSLOTS && active < maxOrbs; i++) {
        if (orbs[i].active) continue;
        if (xsf(rng) > 0.08f) continue;
        // spawn on the helix near the floor, rise through the structure
        orbs[i].active = true;
        float ang = xsf(rng) * 6.2832f;
        float rad = 0.30f;               // helix radius
        orbs[i].x = rad * cosf(ang); orbs[i].y = rad * sinf(ang);
        orbs[i].z = HC_BOUND_MIN.z + xsf(rng) * 0.3f;
        orbs[i].vx = (xsf(rng) - 0.5f) * 0.01f;
        orbs[i].vy = (xsf(rng) - 0.5f) * 0.01f;
        orbs[i].vz = rise * (0.7f + xsf(rng) * 0.6f);
        orbs[i].age = 0; orbs[i].life = 60.0f + xsf(rng) * 80.0f;
        active++;
    }
    for (int o = 0; o < NEB_MAXSLOTS; o++) {
        if (!orbs[o].active) continue;
        orbs[o].age += 1.0f;
        if (orbs[o].age >= orbs[o].life || orbs[o].z > HC_BOUND_MAX.z + 0.2f) { orbs[o].active = false; continue; }
        orbs[o].x += orbs[o].vx; orbs[o].y += orbs[o].vy; orbs[o].z += orbs[o].vz;
        float lc = orbs[o].age / orbs[o].life;
        float lb = (lc < 0.4f) ? smoothstep01(lc / 0.4f)
                 : (lc > 0.6f) ? smoothstep01((1.0f - lc) / 0.4f) : 1.0f;
        // deposit into every LED near the orb in 3D
        for (int i = 0; i < HC_NUM_LEDS; i++) {
            float dx = LED_POS[i].x - orbs[o].x, dy = LED_POS[i].y - orbs[o].y, dz = LED_POS[i].z - orbs[o].z;
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > (3.0f * sigma) * (3.0f * sigma)) continue;
            float g = expf(-d2 / (2.0f * sigma * sigma));
            float nv = led[i] + lb * 153.0f * g;
            led[i] = nv > 255 ? 255 : nv;
        }
    }
    for (int i = 0; i < HC_NUM_LEDS; i++) if (led[i] > 2) star_add(out[i], led[i]);
}

// ── Registry (add effects here; the sim exposes them by name) ────────────
typedef void (*HcEffectFn)(RGBf*, float);
struct HcEffect { const char* name; HcEffectFn fn; };
static const HcEffect HC_EFFECTS[] = {
    { "pour",        fx_pour },
    { "axis_radial", fx_axis_radial },
    { "nebula",        fx_nebula },
    { "nebula_native", fx_nebula_native },
};
static const int HC_NUM_EFFECTS = sizeof(HC_EFFECTS) / sizeof(HC_EFFECTS[0]);
