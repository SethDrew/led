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

// ── Live tunables ─────────────────────────────────────────────────────────
// Mutable globals the effects read, exposed to the IDE via the HC_PARAMS
// registry at the bottom of this file (the one source the WASM param ABI reads).
// Defaults reproduce the faithful-port / original-tuning behaviour exactly.
static float POUR_period      = 3.0f;    // seconds per fall (lower = faster)
static float AXIS_speed       = 0.9f;    // rings/sec
static float NEB_orb_speed    = 0.3f;    // index units / 30Hz-frame (the "speed" lever)
static float NEB_density      = 1.0f;    // x baseline orb count (index nebula)
static float NEB_orb_size     = 7.0f;    // trail persistence; decay = 1 - 1/orb_size
static float NEB_rise_speed   = 0.030f;  // m / 30Hz-frame, rising orbs (native nebula)
static float NEB_density_nat  = 1.0f;    // x baseline orb count (native nebula)
static float PULSE_speed      = 2.5f;    // m/s the button pulse climbs root->trunk->tip
static float PULSE_width      = 0.22f;   // pulse head thickness (m)

// ── Control inputs (bespoke per-installation widget bus) ───────────────────
// The interaction layer specific to THIS installation. On the real sculpture
// physical controls write these fields; in the IDE the on-screen widgets write
// the same. Effects read them. This is where a generic IDE stops and an
// installation begins. Three control kinds, one per panel:
//   • 3 momentary buttons -> launch a pulse that climbs roots -> trunk -> tip
//   • a slider            -> trunk brightness (applied at the output stage)
//   • an infinite dial    -> scene spin (a view transform; lives in the viewer)
//
// Pulses are a small ring buffer of (launch time, source button). A press
// appends one; fx_pulse renders every still-climbing pulse additively.
#define HC_MAX_PULSES 24
struct HcPulse { float t0; int btn; };
static HcPulse HC_PULSES[HC_MAX_PULSES];
static int     HC_PULSE_HEAD = 0;
static bool    HC_PULSES_READY = false;
static inline void hc_pulses_init() {
    if (HC_PULSES_READY) return;
    for (int i = 0; i < HC_MAX_PULSES; i++) { HC_PULSES[i].t0 = -1.0f; HC_PULSES[i].btn = 0; }
    HC_PULSES_READY = true;
}
// Launch a pulse from button `btn` (0..2) at absolute time `t0` (seconds).
static inline void hc_launch_pulse(int btn, float t0) {
    hc_pulses_init();
    HC_PULSES[HC_PULSE_HEAD].t0  = t0;
    HC_PULSES[HC_PULSE_HEAD].btn = (btn < 0 ? 0 : (btn > 2 ? 2 : btn));
    HC_PULSE_HEAD = (HC_PULSE_HEAD + 1) % HC_MAX_PULSES;
}

// Spin (dial): global pattern azimuth in radians. An effect that samples its
// spatial field at hc_pos(i) instead of LED_POS[i] renders the pattern rotated
// by +HC_SPIN about the vertical axis — the FIELD-CORRECT spin: the pixels stay
// bolted in place and the pattern turns over them (what a real installation must
// do), as opposed to rotating the whole model in the viewer. Because hc_pos only
// ever lands on real LED positions, the angular quantization is honest: features
// snap to where LEDs exist (roots ~51.4°, canopy ~25.7°, trunk 2-fold). Rotation
// about z preserves z, so height-based structure (e.g. the nebula background
// wave) is correctly unaffected.
static float HC_SPIN = 0.0f;
struct HcVec3 { float x, y, z; };
static inline HcVec3 hc_pos(int i) {
    float c = cosf(HC_SPIN), s = sinf(HC_SPIN);          // R(-spin): pattern turns +spin
    HcVec3 p = {  c * LED_POS[i].x + s * LED_POS[i].y,
                 -s * LED_POS[i].x + c * LED_POS[i].y,
                  LED_POS[i].z };
    return p;
}

// ── Effect: "pour" ──────────────────────────────────────────────────────
// Light ignites at the TOP of the trunk and flows down the helix. When the
// wavefront reaches the canopy it radiates outward along the canopy strands
// *while continuing down the trunk*; when it reaches the floor it radiates out
// along the roots. Driven by a per-LED travel distance from the top source:
// trunk = vertical descent, canopy/root = (descent to the junction) + along-
// strand arc length from that junction. So co-planar canopy/root LEDs light in
// sequence (a spreading wave), not all at once. REQUIRES real (x,y,z) + the
// strip topology — index space can't express this.
static inline void fx_pour(RGBf* out, float t) {
    const float z_top  = HC_BOUND_MAX.z;   // top of the poking helix = the source
    const float z_can  = HC_CANOPY_Z;      // canopy plane (trunk->canopy junction)
    const float z_bot  = HC_BOUND_MIN.z;   // floor (trunk->root junction)
    const float period = POUR_period;      // seconds per pour
    const float sigma  = 0.18f;            // leading-edge thickness (m)
    const float tail   = 0.55f;            // comet trail length behind the front (m)

    // Per-LED travel distance from the source, built once (geometry is static).
    static bool  init = false;
    static float dist[HC_NUM_LEDS];
    static float Dmax = 1.0f;
    if (!init) {
        Dmax = 0.0f;
        for (int s = 0; s < HC_NUM_STRIPS; s++) {
            int   k   = HC_STRIPS[s].kind;
            int   st  = HC_STRIPS[s].start;
            int   cnt = HC_STRIPS[s].count;
            float arc = 0.0f;                       // along-strand length from the junction
            for (int j = 0; j < cnt; j++) {
                int i = st + j;
                if (j > 0) {
                    float dx = LED_POS[i].x - LED_POS[i-1].x;
                    float dy = LED_POS[i].y - LED_POS[i-1].y;
                    float dz = LED_POS[i].z - LED_POS[i-1].z;
                    arc += sqrtf(dx*dx + dy*dy + dz*dz);
                }
                float d;
                if (k == 1)      d = (z_top - z_can) + arc;   // canopy: reach canopy, then spread
                else if (k == 2) d = (z_top - z_bot) + arc;   // roots: reach floor, then spread
                else             d = (z_top - LED_POS[i].z);  // helix: descend the trunk
                dist[i] = d;
                if (d > Dmax) Dmax = d;
            }
        }
        init = true;
    }

    float front = fmodf(t, period) / period * (Dmax + tail * 3.0f);  // 0 -> past the far end
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        float dd = front - dist[i];                  // >0 front has passed (trail), <0 still ahead
        float b;
        if (dd >= 0.0f) {
            b = expf(-(dd * dd) / (2.0f * sigma * sigma));
            b = fmaxf(b, expf(-dd / tail) * 0.6f);   // comet trail behind the front
        } else {
            float a = -dd;
            b = expf(-(a * a) / (2.0f * (sigma * 0.5f) * (sigma * 0.5f)));
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
    const float speed = AXIS_speed;      // rings/sec
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

// ── Effect: "pulse" — button-launched wave, roots -> trunk -> tip ───────
// Interactive: each of the 3 panel buttons appends a pulse (hc_launch_pulse).
// A pulse ignites at the root TIPS and climbs inward along the roots, converges
// at the floor junction, then rises up the double-helix trunk to the poking tip
// — the reverse path of "pour", and event-driven instead of looping. Each
// button drives a DIFFERENT non-adjacent pair of roots (PULSE_ROOTS); the shared
// trunk always carries the pulse on up to the tip. Multiple pulses coexist
// (additive); the source button picks the hue. Canopy LEDs are off this path and
// stay dark. Built on a per-LED path distance from the tips: roots are
// normalized per-strand so every root's base lands at the same distance
// (root_ref) and hands off to the trunk together regardless of its own length;
// the trunk then continues by height above floor.
//
// Button -> root pair. Roots are numbered 0..6 around the heptagon in strip
// order; each pair sits 3 spokes (~154°) apart so the three buttons read as
// spatially distinct. Three disjoint pairs cover 6 of 7 roots — root 6 is the
// odd one out and intentionally idle. Retune freely.
static const int PULSE_ROOTS[3][2] = { {0, 3}, {1, 4}, {2, 5} };

static inline void fx_pulse(RGBf* out, float t) {
    hc_pulses_init();
    const float z_bot = HC_BOUND_MIN.z;
    const float speed = PULSE_speed;     // m/s the wavefront climbs
    const float sigma = PULSE_width;     // pulse head thickness (m)
    const float tail  = 0.40f;           // comet trail behind the head (m)

    // per-LED path distance from the root tips (-1 marks off-path canopy) and
    // per-LED root index 0..6 (-1 for trunk/canopy), so a pulse can light just
    // its button's pair of roots while the trunk stays shared.
    static bool  init = false;
    static float dist[HC_NUM_LEDS];
    static int   root_id[HC_NUM_LEDS];
    static float Dmax = 1.0f;
    if (!init) {
        for (int i = 0; i < HC_NUM_LEDS; i++) root_id[i] = -1;
        int rcount = 0;
        // representative root length sets where every root hands off to the trunk
        float root_ref = 0.0f;
        for (int s = 0; s < HC_NUM_STRIPS; s++) {
            if (HC_STRIPS[s].kind != 2) continue;
            int st = HC_STRIPS[s].start, cnt = HC_STRIPS[s].count;
            float arc = 0.0f;
            for (int j = 1; j < cnt; j++) {
                int i = st + j;
                float dx = LED_POS[i].x - LED_POS[i-1].x, dy = LED_POS[i].y - LED_POS[i-1].y, dz = LED_POS[i].z - LED_POS[i-1].z;
                arc += sqrtf(dx*dx + dy*dy + dz*dz);
            }
            if (arc > root_ref) root_ref = arc;
        }
        if (root_ref < 1e-3f) root_ref = 1.0f;

        Dmax = 0.0f;
        for (int s = 0; s < HC_NUM_STRIPS; s++) {
            int k = HC_STRIPS[s].kind, st = HC_STRIPS[s].start, cnt = HC_STRIPS[s].count;
            if (k == 1) { for (int j = 0; j < cnt; j++) dist[st+j] = -1.0f; continue; }  // canopy off-path
            if (k == 2) {
                float total = 0.0f;                          // this root's full arc length
                for (int j = 1; j < cnt; j++) {
                    int i = st + j;
                    float dx = LED_POS[i].x - LED_POS[i-1].x, dy = LED_POS[i].y - LED_POS[i-1].y, dz = LED_POS[i].z - LED_POS[i-1].z;
                    total += sqrtf(dx*dx + dy*dy + dz*dz);
                }
                if (total < 1e-3f) total = 1.0f;
                float arc = 0.0f;
                for (int j = 0; j < cnt; j++) {
                    int i = st + j;
                    if (j > 0) {
                        float dx = LED_POS[i].x - LED_POS[i-1].x, dy = LED_POS[i].y - LED_POS[i-1].y, dz = LED_POS[i].z - LED_POS[i-1].z;
                        arc += sqrtf(dx*dx + dy*dy + dz*dz);
                    }
                    float d = (1.0f - arc / total) * root_ref;   // tip(arc=total)->0, base(arc=0)->root_ref
                    dist[i] = d; if (d > Dmax) Dmax = d;
                    root_id[i] = rcount;                          // this root's index 0..6
                }
                rcount++;
            } else {                                          // helix/trunk: continue above the floor
                for (int j = 0; j < cnt; j++) {
                    int i = st + j;
                    float d = root_ref + (LED_POS[i].z - z_bot);
                    dist[i] = d; if (d > Dmax) Dmax = d;
                }
            }
        }
        init = true;
    }

    static const float BTN_COL[3][3] = {
        { 1.00f, 0.80f, 0.45f },   // button 0 — warm gold
        { 0.30f, 0.85f, 1.00f },   // button 1 — cyan
        { 1.00f, 0.40f, 0.85f },   // button 2 — magenta
    };
    const float life_d = Dmax + tail * 4.0f;   // distance a pulse travels before it's spent

    for (int i = 0; i < HC_NUM_LEDS; i++) { out[i].r = 0; out[i].g = 0; out[i].b = 0; }
    for (int s = 0; s < HC_MAX_PULSES; s++) {
        if (HC_PULSES[s].t0 < 0.0f) continue;               // empty slot
        float age = t - HC_PULSES[s].t0;
        if (age < 0.0f) continue;                           // not launched yet (scrubbed back)
        float front = age * speed;
        if (front > life_d) continue;                       // spent
        int btn = HC_PULSES[s].btn;
        const float* col = BTN_COL[btn];
        int ra = PULSE_ROOTS[btn][0], rb = PULSE_ROOTS[btn][1];
        for (int i = 0; i < HC_NUM_LEDS; i++) {
            float di = dist[i];
            if (di < 0.0f) continue;                        // canopy: off-path
            int rid = root_id[i];
            if (rid >= 0 && rid != ra && rid != rb) continue;   // root not in this button's pair
            float dd = front - di;
            float b;
            if (dd >= 0.0f) {
                b = expf(-(dd * dd) / (2.0f * sigma * sigma));
                b = fmaxf(b, expf(-dd / tail) * 0.5f);      // comet trail
            } else {
                float a = -dd;                              // crisp leading edge
                b = expf(-(a * a) / (2.0f * (sigma * 0.5f) * (sigma * 0.5f)));
            }
            out[i].r = clamp01(out[i].r + b * col[0]);
            out[i].g = clamp01(out[i].g + b * col[1]);
            out[i].b = clamp01(out[i].b + b * col[2]);
        }
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
// Both layers are STATEFUL (orbs + decaying trail buffer). The original's math
// is per-frame; we keep it identical but drive it with a FIXED-TIMESTEP
// accumulator (neb_steps) keyed off absolute t, so the simulation advances at a
// constant 30 Hz regardless of how often the host calls us. The desktop sim
// calls at exactly 30 fps (one step per call, byte-identical to the legacy
// integration); the live browser viewer calls at ~120 fps (one step every ~4th
// call) — without this the stars would crawl and churn ~4x too fast on screen.

static inline float smoothstep01(float t) {
    if (t < 0) t = 0; if (t > 1) t = 1;
    return t * t * (3.0f - 2.0f * t);
}
static inline uint32_t xs32(uint32_t& s) {  // deterministic PRNG (reproducible sims)
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
static inline float xsf(uint32_t& s) { return (xs32(s) & 0xFFFFFF) / (float)0x1000000; }

// Fixed-timestep accumulator: how many whole 30Hz frames have elapsed since the
// last call. Caller holds the per-effect last_t/acc statics. A backward scrub
// (dt<0) yields 0 (state holds, no rewind); a large forward seek is clamped so
// we don't spiral catching up.
static inline int neb_steps(float t, float fps, float& last_t, float& acc) {
    if (last_t < -0.5f) last_t = t;
    acc += (t - last_t) * fps;
    last_t = t;
    int steps = (int)(acc + 1e-4f);
    if (steps < 0) steps = 0;
    if (steps > 8) steps = 8;
    acc -= (float)steps;
    return steps;
}

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
// x2 baseline headroom so the density slider can go up to ~2.0 without overflow.
#define NEB_MAXSLOTS   (NEB_REF_ORBS * HC_NUM_LEDS / NEB_REF_PIXELS * 2 + 8)
struct NebOrb { bool active; float pos; float vel; float age; float life; };       // index space
struct NebOrb3 { bool active; float x, y, z, vx, vy, vz, age, life; };             // 3D space

// ── fx_nebula — faithful index-based background + crawling stars ────────
static inline void fx_nebula(RGBf* out, float t) {
    const float NEBULA_FPS = 30.0f;
    const float N = (float)HC_NUM_LEDS;
    const float speed = NEB_orb_speed, orbSize = NEB_orb_size;
    int maxOrbs = (int)((NEB_REF_ORBS * HC_NUM_LEDS / NEB_REF_PIXELS) * NEB_density);
    if (maxOrbs < 1) maxOrbs = 1;
    if (maxOrbs > NEB_MAXSLOTS) maxOrbs = NEB_MAXSLOTS;
    float frame = t * NEBULA_FPS;

    static bool init = false;
    static NebOrb orbs[NEB_MAXSLOTS];
    static float led[HC_NUM_LEDS];
    static uint32_t rng = 0xC0FFEEu;
    static float last_t = -1.0f, acc = 0.0f;
    if (!init) { for (int i = 0; i < NEB_MAXSLOTS; i++) orbs[i].active = false;
                 for (int i = 0; i < HC_NUM_LEDS; i++) led[i] = 0; init = true; }

    // background (continuous in t)
    for (int i = 0; i < HC_NUM_LEDS; i++)
        nebula_bg_pixel((float)i / N, frame, out[i]);

    // ── crawling stars: advance the sim in fixed 30Hz steps ──
    float decay = 1.0f - 1.0f / orbSize;
    int nsteps = neb_steps(t, NEBULA_FPS, last_t, acc);
    for (int s = 0; s < nsteps; s++) {
        for (int i = 0; i < HC_NUM_LEDS; i++) { led[i] *= decay; if (led[i] < 3) led[i] = 0; }

        int active = 0;
        for (int i = 0; i < NEB_MAXSLOTS; i++) if (orbs[i].active) active++;
        // Maintain ~maxOrbs population (refill inactive slots gradually for
        // organic turnover); the original's single-spawn-at-3% can't fill a
        // large count.
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
    }
    for (int i = 0; i < HC_NUM_LEDS; i++) if (led[i] > 2) star_add(out[i], led[i]);
}

// ── fx_nebula_native — same effect, expressed through 3D space ──────────
static inline void fx_nebula_native(RGBf* out, float t) {
    const float NEBULA_FPS = 30.0f;
    // Fewer than the index version: each 3D orb glows a ~0.22m sphere covering
    // many LEDs, so a smaller count yields comparable particle coverage.
    int maxOrbs = (int)((NEB_REF_ORBS * HC_NUM_LEDS / NEB_REF_PIXELS / 6) * NEB_density_nat);
    if (maxOrbs < 1) maxOrbs = 1;
    if (maxOrbs > NEB_MAXSLOTS) maxOrbs = NEB_MAXSLOTS;
    const float orbSize = NEB_orb_size;
    const float sigma = 0.22f;               // orb glow radius (m)
    const float rise  = NEB_rise_speed;      // m per 30Hz-frame upward
    float frame = t * NEBULA_FPS;
    float zspan = HC_BOUND_MAX.z - HC_BOUND_MIN.z;

    static bool init = false;
    static NebOrb3 orbs[NEB_MAXSLOTS];
    static float led[HC_NUM_LEDS];
    static uint32_t rng = 0xBADCAFEu;
    static float last_t = -1.0f, acc = 0.0f;
    if (!init) { for (int i = 0; i < NEB_MAXSLOTS; i++) orbs[i].active = false;
                 for (int i = 0; i < HC_NUM_LEDS; i++) led[i] = 0; init = true; }

    // background: wave travels vertically through z (identical on both strands).
    // z-based, so the spin (rotation about z) leaves the breathing wave fixed —
    // correctly, it has no angular structure to turn.
    const float BG_GAIN = 0.55f;   // dimmed so the spin-visible orb layer reads over it
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        float zn = (LED_POS[i].z - HC_BOUND_MIN.z) / (zspan > 1e-6f ? zspan : 1.0f);
        nebula_bg_pixel(zn, frame, out[i]);
        out[i].r *= BG_GAIN; out[i].g *= BG_GAIN; out[i].b *= BG_GAIN;
    }

    // Spin-rotated LED positions (field-correct spin): the orbs live in a fixed
    // frame and we sample each LED's glow at its rotated position, so the orb
    // constellation appears to turn about the trunk axis. Computed once per
    // frame (HC_SPIN is constant within a render call).
    static float rx[HC_NUM_LEDS], ry[HC_NUM_LEDS];
    for (int i = 0; i < HC_NUM_LEDS; i++) { HcVec3 p = hc_pos(i); rx[i] = p.x; ry[i] = p.y; }

    // ── rising 3D orbs: advance the sim in fixed 30Hz steps ──
    float decay = 1.0f - 1.0f / orbSize;
    int nsteps = neb_steps(t, NEBULA_FPS, last_t, acc);
    for (int s = 0; s < nsteps; s++) {
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
            // fan outward as they rise: orbs spiral from the axis toward the canopy
            // rim, so by the top they carry real angular structure and the dial
            // (spin) visibly sweeps the constellation across the rim. Near the axis
            // rotation barely moves a point — out at the rim it sweeps a wide arc.
            float orr = hypotf(orbs[o].x, orbs[o].y);
            if (orr > 1e-3f) { const float oacc = 0.0005f; orbs[o].vx += oacc * orbs[o].x / orr; orbs[o].vy += oacc * orbs[o].y / orr; }
            orbs[o].x += orbs[o].vx; orbs[o].y += orbs[o].vy; orbs[o].z += orbs[o].vz;
            float lc = orbs[o].age / orbs[o].life;
            float lb = (lc < 0.4f) ? smoothstep01(lc / 0.4f)
                     : (lc > 0.6f) ? smoothstep01((1.0f - lc) / 0.4f) : 1.0f;
            // deposit into every LED near the orb in 3D, sampling at the LED's
            // spin-rotated position (rx/ry) so the constellation turns with HC_SPIN
            for (int i = 0; i < HC_NUM_LEDS; i++) {
                float dx = rx[i] - orbs[o].x, dy = ry[i] - orbs[o].y, dz = LED_POS[i].z - orbs[o].z;
                float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > (3.0f * sigma) * (3.0f * sigma)) continue;
                float g = expf(-d2 / (2.0f * sigma * sigma));
                float nv = led[i] + lb * 153.0f * g;
                led[i] = nv > 255 ? 255 : nv;
            }
        }
    }
    for (int i = 0; i < HC_NUM_LEDS; i++) if (led[i] > 2) star_add(out[i], led[i]);
}

// ── Effect: "line" (dev) — a vertical meridian blade that spins ─────────
// Development/test pattern. A thin vertical sheet at ONE azimuth, lighting every
// LED near that bearing — the root at that bearing, up through the trunk, and out
// the canopy spoke: a line from the bottom up through the canopy. Its azimuth IS
// the spin angle (HC_SPIN), so the dial (pattern mode) sweeps it around. This is
// the clean way to SEE the field-correct spin and feel its quantization: as the
// blade turns, it snaps onto the 7 roots / 14 canopy spokes that actually exist.
static float LINE_width_deg = 6.0f;     // angular half-width of the blade (degrees)
static inline void fx_line(RGBf* out, float t) {
    (void)t;
    const float sigma = LINE_width_deg * (float)M_PI / 180.0f;
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        float az = atan2f(LED_POS[i].y, LED_POS[i].x);
        float d = az - HC_SPIN;
        while (d >  (float)M_PI) d -= 2.0f * (float)M_PI;     // wrap to [-pi,pi]
        while (d < -(float)M_PI) d += 2.0f * (float)M_PI;
        float b = clamp01(expf(-(d * d) / (2.0f * sigma * sigma)));
        out[i].r = clamp01(b * 0.55f);                        // cool white-cyan
        out[i].g = clamp01(b * 0.95f);
        out[i].b = clamp01(b);
    }
}

// ── Effect: "creatures" (dev) — bioluminescent bloom+crawl in 3D space ──
// Reworked from the wire-order port (audio-reactive/effects/biolum_mixed.py).
// Creatures no longer crawl the LED INDEX — they live in WORLD space at an
// (azimuth, height), drifting in azimuth. Each is a soft teal body that either
// BLOOMs (whole-body fade in/out) or CRAWLs (an expanding ANGULAR ring). Render
// deposits only onto the strip LEDs near the creature's (azimuth, height), so the
// light always reads as groups crawling ALONG the strands — a wedge of canopy rim,
// a root at one bearing, a chunk of helix — never a free-floating ball in space.
// Position is sampled through hc_pos, so the dial (HC_SPIN) sweeps the whole
// population around the trunk: spin for real, on the same footing as the line.
// As a creature drifts past the roots it CLICKS between their 7 discrete bearings;
// across the canopy rim it glides — the topology's quantization, made visible.
// Stateful; advanced with the fixed-timestep accumulator at a constant rate.
#define CR_MAX   16
#define CR_ANIMS 12
struct CrAnim   { bool active; float age; };
struct Creature { bool alive; int kind; float th, z, vth, vz, emit_timer; CrAnim anims[CR_ANIMS]; };
static float CREAT_count    = 7.0f;     // creatures maintained
static float CREAT_speed    = 1.0f;     // azimuthal drift-speed multiplier
static float CREAT_size_deg = 7.0f;     // angular half-size (sigma) of a creature body
static float CREAT_thick_m  = 0.16f;    // vertical half-size (sigma, m) of a creature body

static inline void cr_spawn(Creature& c, uint32_t& rng) {
    c.kind = (xsf(rng) < 0.5f) ? 0 : 1;                       // 0 bloom, 1 crawl
    // spawn ON a strand: take a real LED's (azimuth, height) so the creature sits
    // on the geometry like the wire-order original, not floating in empty space.
    // Section coverage then follows LED density — matching the original's spawn.
    int li = (int)(xsf(rng) * (float)HC_NUM_LEDS); if (li >= HC_NUM_LEDS) li = HC_NUM_LEDS - 1;
    c.th = atan2f(LED_POS[li].y, LED_POS[li].x);
    c.z  = LED_POS[li].z;
    float sp = (0.18f + xsf(rng) * 0.34f) * CREAT_speed;      // rad/s azimuthal drift
    c.vth  = (xsf(rng) < 0.5f ? -1.0f : 1.0f) * sp;
    c.vz   = (xsf(rng) - 0.5f) * 0.04f;                       // gentle vertical wander (m/s)
    c.emit_timer = 0.4f + xsf(rng) * 1.6f;
    for (int k = 0; k < CR_ANIMS; k++) c.anims[k].active = false;
    c.alive = true;
}

static inline void fx_creatures(RGBf* out, float t) {
    const float FPS = 30.0f;
    const float dt  = 1.0f / FPS;
    const float TAU = 6.2831853f;
    int count = (int)(CREAT_count + 0.5f); if (count < 1) count = 1; if (count > CR_MAX) count = CR_MAX;

    const float zlo = HC_BOUND_MIN.z, zhi = HC_BOUND_MAX.z;
    const float sig_th = CREAT_size_deg * (float)M_PI / 180.0f;   // angular body sigma
    const float sig_z  = CREAT_thick_m;                          // vertical body sigma (m)
    const float BLOOM_RISE = 2.0f, BLOOM_HOLD = 1.5f, BLOOM_FALL = 5.0f, BLOOM_TOTAL = 8.5f;
    const float crawl_ang  = 55.0f * (float)M_PI / 180.0f;        // max ring half-angle
    const float crawl_tail = 16.0f * (float)M_PI / 180.0f;        // ring softness (angular)
    const float crawl_edge = 5.0f  * (float)M_PI / 180.0f;        // leading-edge ramp
    const float crawl_life = 2.4f, CRAWL_FADE = 1.4f;

    static bool init = false;
    static Creature cr[CR_MAX];
    static uint32_t rng = 0x5EED1234u;
    static float last_t = -1.0f, acc = 0.0f;
    if (!init) { for (int c = 0; c < CR_MAX; c++) cr_spawn(cr[c], rng); init = true; }

    int nsteps = neb_steps(t, FPS, last_t, acc);
    for (int s = 0; s < nsteps; s++) {
        for (int c = 0; c < count; c++) {
            Creature& cc = cr[c];
            if (!cc.alive) cr_spawn(cc, rng);
            cc.th += cc.vth * dt;
            if (cc.th >= TAU) cc.th -= TAU;
            if (cc.th <  0.0f) cc.th += TAU;
            cc.z += cc.vz * dt;
            if (cc.z < zlo) { cc.z = zlo; cc.vz = -cc.vz; }       // bounce off the floor/ceiling
            if (cc.z > zhi) { cc.z = zhi; cc.vz = -cc.vz; }
            cc.emit_timer -= dt;
            if (cc.emit_timer <= 0.0f) {
                for (int k = 0; k < CR_ANIMS; k++) if (!cc.anims[k].active) { cc.anims[k].active = true; cc.anims[k].age = 0.0f; break; }
                if (cc.kind == 0) { float base = BLOOM_TOTAL + (1.2f + xsf(rng) * 1.6f); cc.emit_timer = base * (0.33f + xsf(rng) * 1.0f); }
                else              { cc.emit_timer = crawl_life + CRAWL_FADE + (1.0f + xsf(rng) * 1.6f); }
            }
            float lifelim = (cc.kind == 0) ? BLOOM_TOTAL : (crawl_life + CRAWL_FADE);
            for (int k = 0; k < CR_ANIMS; k++) { if (!cc.anims[k].active) continue; cc.anims[k].age += dt; if (cc.anims[k].age >= lifelim) cc.anims[k].active = false; }
        }
    }

    // Per-LED rotated (azimuth, height) — sampled through hc_pos so HC_SPIN turns
    // the whole population. Computed once per frame (spin is constant within it).
    static float Laz[HC_NUM_LEDS], Lz[HC_NUM_LEDS];
    for (int i = 0; i < HC_NUM_LEDS; i++) { HcVec3 p = hc_pos(i); Laz[i] = atan2f(p.y, p.x); Lz[i] = p.z; }

    static float buf[HC_NUM_LEDS];
    for (int i = 0; i < HC_NUM_LEDS; i++) buf[i] = 0.0f;

    for (int c = 0; c < count; c++) {
        Creature& cc = cr[c];
        if (!cc.alive) continue;
        for (int k = 0; k < CR_ANIMS; k++) {
            if (!cc.anims[k].active) continue;
            float age = cc.anims[k].age;
            // envelope (bloom) / ring geometry (crawl), computed once per anim
            float env = 1.0f, radius = 0.0f, fade = 1.0f;
            if (cc.kind == 0) {
                if (age < BLOOM_RISE) env = age / BLOOM_RISE;
                else if (age < BLOOM_RISE + BLOOM_HOLD) env = 1.0f;
                else { float ft = (age - BLOOM_RISE - BLOOM_HOLD) / BLOOM_FALL; env = fmaxf(0.0f, 1.0f - ft); env *= env; }
                if (env < 0.003f) continue;
            } else {
                float tt = age / crawl_life; if (tt > 1.0f) tt = 1.0f;
                radius = tt * crawl_ang;
                if (age > crawl_life) { float ft = (age - crawl_life) / CRAWL_FADE; fade = fmaxf(0.0f, 1.0f - ft); fade *= fade; }
            }
            for (int i = 0; i < HC_NUM_LEDS; i++) {
                float dz = Lz[i] - cc.z;
                if (fabsf(dz) > 3.0f * sig_z) continue;          // outside the body's height band
                float dth = Laz[i] - cc.th;
                while (dth >  (float)M_PI) dth -= TAU;            // shortest angular gap
                while (dth < -(float)M_PI) dth += TAU;
                float adth = fabsf(dth);
                float zg = expf(-(dz * dz) / (2.0f * sig_z * sig_z));
                float b;
                if (cc.kind == 0) {                              // bloom: flat-top body, soft edges
                    // solid core out to sig_th (like the original's bloom_radius),
                    // then a soft angular falloff — brighter & more "bodied" than a
                    // bare gaussian, matching the linear reference's solid blooms.
                    float soft = sig_th * 0.8f;
                    float spatial = (adth <= sig_th) ? 1.0f
                                  : expf(-((adth - sig_th) * (adth - sig_th)) / (2.0f * soft * soft));
                    b = env * spatial * zg;
                } else {                                         // crawl: expanding angular ring
                    if (adth > radius) continue;
                    float behind = radius - adth;
                    b = expf(-behind / crawl_tail);
                    if (behind < crawl_edge) b *= behind / crawl_edge;
                    b *= fade * zg;
                }
                if (b > 0.003f) buf[i] = buf[i] + b - buf[i] * b;   // screen blend
            }
        }
    }
    for (int i = 0; i < HC_NUM_LEDS; i++) {                    // teal, squared response
        float v = clamp01(buf[i]); v = v * v;
        out[i].r = 0.0f;
        out[i].g = clamp01(v * (180.0f / 255.0f));
        out[i].b = clamp01(v * (220.0f / 255.0f));
    }
}

// ── Effect: "creatures_lin" (reference) — the ORIGINAL wire-order creatures ──
// Kept verbatim from audio-reactive/effects/biolum_mixed.py as the NON-rotation-
// aware reference: blooms/crawls crawling the LED INDEX (along strands, jumping at
// wire boundaries). NOT spinnable (index-native). Its only job is to be the target
// the angular "creatures" is tuned against — render it and the angular version
// side by side and match density/size/brightness. Fixed tuning (no live params) so
// it stays a stable yardstick.
struct CrLin { bool alive; int kind; float pos, vel, emit_timer; CrAnim anims[CR_ANIMS]; };
static inline void crlin_spawn(CrLin& c, float N, float scale, uint32_t& rng) {
    c.kind = (xsf(rng) < 0.5f) ? 0 : 1;
    const float avg = 3.3f / 1.9f, spread = 0.33f;
    float speed = (avg * (1.0f - spread) + xsf(rng) * (avg * 2.0f * spread)) * scale;
    c.pos = 5.0f * scale + xsf(rng) * (N - 10.0f * scale);
    c.vel = (xsf(rng) < 0.5f ? -1.0f : 1.0f) * speed;
    c.emit_timer = 1.2f + xsf(rng) * 1.6f;
    for (int k = 0; k < CR_ANIMS; k++) c.anims[k].active = false;
    c.alive = true;
}
static inline void fx_creatures_lin(RGBf* out, float t) {
    const float FPS = 30.0f;
    const float N = (float)HC_NUM_LEDS;
    const float scale = N / 150.0f;
    const float dt = 1.0f / FPS;
    const int count = 7;
    const float bloom_radius = 3.0f * scale, bloom_soft = 0.8f * scale;
    const float BLOOM_RISE = 2.0f, BLOOM_HOLD = 1.5f, BLOOM_FALL = 5.0f, BLOOM_TOTAL = 8.5f;
    const float crawl_radius = 8.0f * scale, crawl_tail = 4.0f * scale;
    const float crawl_life = 8.0f / 3.3f, CRAWL_FADE = 1.4f, DESPAWN = 5.0f * scale;

    static bool init = false;
    static CrLin cr[CR_MAX];
    static uint32_t rng = 0x5EED1234u;
    static float last_t = -1.0f, acc = 0.0f;
    if (!init) { for (int c = 0; c < CR_MAX; c++) crlin_spawn(cr[c], N, scale, rng); init = true; }

    int nsteps = neb_steps(t, FPS, last_t, acc);
    for (int s = 0; s < nsteps; s++) {
        for (int c = 0; c < count; c++) {
            CrLin& cc = cr[c];
            if (!cc.alive) crlin_spawn(cc, N, scale, rng);
            cc.pos += cc.vel * dt;
            if (cc.pos < -DESPAWN || cc.pos > N + DESPAWN) { crlin_spawn(cc, N, scale, rng); continue; }
            cc.emit_timer -= dt;
            if (cc.emit_timer <= 0.0f) {
                for (int k = 0; k < CR_ANIMS; k++) if (!cc.anims[k].active) { cc.anims[k].active = true; cc.anims[k].age = 0.0f; break; }
                if (cc.kind == 0) { float base = BLOOM_TOTAL + (1.2f + xsf(rng) * 1.6f); cc.emit_timer = base * (0.33f + xsf(rng) * 1.67f); }
                else              { cc.emit_timer = crawl_life + CRAWL_FADE + (1.2f + xsf(rng) * 1.6f); }
            }
            float lifelim = (cc.kind == 0) ? BLOOM_TOTAL : (crawl_life + CRAWL_FADE);
            for (int k = 0; k < CR_ANIMS; k++) { if (!cc.anims[k].active) continue; cc.anims[k].age += dt; if (cc.anims[k].age >= lifelim) cc.anims[k].active = false; }
        }
    }

    static float buf[HC_NUM_LEDS];
    for (int i = 0; i < HC_NUM_LEDS; i++) buf[i] = 0.0f;
    for (int c = 0; c < count; c++) {
        CrLin& cc = cr[c];
        if (!cc.alive) continue;
        float center = cc.pos;
        if (cc.kind == 0) {
            int lo = (int)(center - bloom_radius - bloom_soft - 1); if (lo < 0) lo = 0;
            int hi = (int)(center + bloom_radius + bloom_soft + 1); if (hi > HC_NUM_LEDS - 1) hi = HC_NUM_LEDS - 1;
            for (int k = 0; k < CR_ANIMS; k++) {
                if (!cc.anims[k].active) continue;
                float age = cc.anims[k].age, env;
                if (age < BLOOM_RISE) env = age / BLOOM_RISE;
                else if (age < BLOOM_RISE + BLOOM_HOLD) env = 1.0f;
                else { float ft = (age - BLOOM_RISE - BLOOM_HOLD) / BLOOM_FALL; env = fmaxf(0.0f, 1.0f - ft); env *= env; }
                if (env < 0.003f) continue;
                for (int i = lo; i <= hi; i++) {
                    float dist = fabsf((float)i - center);
                    float spatial = (dist <= bloom_radius) ? 1.0f : expf(-(dist - bloom_radius) / bloom_soft);
                    float b = env * spatial;
                    if (b > 0.003f) buf[i] = buf[i] + b - buf[i] * b;
                }
            }
        } else {
            int lo = (int)(center - crawl_radius - crawl_tail - 2); if (lo < 0) lo = 0;
            int hi = (int)(center + crawl_radius + crawl_tail + 2); if (hi > HC_NUM_LEDS - 1) hi = HC_NUM_LEDS - 1;
            for (int k = 0; k < CR_ANIMS; k++) {
                if (!cc.anims[k].active) continue;
                float tt = cc.anims[k].age / crawl_life; if (tt > 1.0f) tt = 1.0f;
                float radius = tt * crawl_radius;
                float fade = 1.0f;
                if (cc.anims[k].age > crawl_life) { float ft = (cc.anims[k].age - crawl_life) / CRAWL_FADE; fade = fmaxf(0.0f, 1.0f - ft); fade *= fade; }
                for (int i = lo; i <= hi; i++) {
                    float dc = fabsf((float)i - center);
                    if (dc > radius) continue;
                    float behind = radius - dc;
                    float b = expf(-behind / crawl_tail);
                    if (behind < 1.0f) b *= behind;
                    b *= fade;
                    if (b > 0.003f) buf[i] = buf[i] + b - buf[i] * b;
                }
            }
        }
    }
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        float v = clamp01(buf[i]); v = v * v;
        out[i].r = 0.0f;
        out[i].g = clamp01(v * (180.0f / 255.0f));
        out[i].b = clamp01(v * (220.0f / 255.0f));
    }
}

#include "axismundi_fx.h"   // the Axis Mundi installation layer (console-driven)

// ── Registry (add effects here; the sim exposes them by name) ────────────
typedef void (*HcEffectFn)(RGBf*, float);
struct HcEffect { const char* name; HcEffectFn fn; };
static const HcEffect HC_EFFECTS[] = {
    { "pour",        fx_pour },
    { "pulse",       fx_pulse },
    { "axis_radial", fx_axis_radial },
    { "nebula",        fx_nebula },          // original look · spins via B (output resample)
    { "nebula_az",     fx_nebula_native },   // azimuthal port · spins via A (field-correct)
    { "line",          fx_line },
    { "creatures",     fx_creatures_lin },   // original look · spins via B (output resample)
    { "creatures_az",  fx_creatures },       // azimuthal port · spins via A (field-correct)
    { "axismundi",     fx_axismundi },       // clicky-pots particle field, live console
};
static const int HC_NUM_EFFECTS = sizeof(HC_EFFECTS) / sizeof(HC_EFFECTS[0]);

// ── Live tunables registry ────────────────────────────────────────────────
// The single source the WASM param ABI exposes to the IDE. `effect` scopes a
// param to one effect's UI (the viewer shows only the active effect's params);
// `lo`/`hi` drive the slider range. Edit here to add a knob — no glue elsewhere.
struct HcParamDef { const char* effect; const char* name; float* ptr; float lo, hi; };
static HcParamDef HC_PARAMS[] = {
    { "pour",          "fall_secs",  &POUR_period,     1.0f,  8.0f  },
    { "pulse",         "climb_mps",  &PULSE_speed,     0.5f,  8.0f  },
    { "pulse",         "width_m",    &PULSE_width,     0.05f, 0.6f  },
    { "axis_radial",   "speed",      &AXIS_speed,      0.1f,  3.0f  },
    { "nebula",        "orb_speed",  &NEB_orb_speed,   0.02f, 1.0f  },
    { "nebula",        "density",    &NEB_density,     0.1f,  2.0f  },
    { "nebula",        "orb_size",   &NEB_orb_size,    2.0f,  20.0f },
    { "nebula_az",     "rise_speed", &NEB_rise_speed,  0.005f, 0.08f },
    { "nebula_az",     "density",    &NEB_density_nat, 0.1f,  3.0f  },
    { "nebula_az",     "orb_size",   &NEB_orb_size,    2.0f,  20.0f },
    { "line",          "width_deg",  &LINE_width_deg,  1.0f,  30.0f },
    { "creatures_az",  "count",      &CREAT_count,     1.0f,  16.0f },
    { "creatures_az",  "speed",      &CREAT_speed,     0.2f,  3.0f  },
    { "creatures_az",  "size_deg",   &CREAT_size_deg,  1.0f,  20.0f },
    { "creatures_az",  "thick_m",    &CREAT_thick_m,   0.05f, 0.6f  },
    { "axismundi",     "armature",    &AXK_armature,    0.0f, 1.0f  },
};
static const int HC_NUM_PARAMS = sizeof(HC_PARAMS) / sizeof(HC_PARAMS[0]);
