// axismundi_fx.h — the Axis Mundi installation effects, in helix-canopy space.
//
// Ported from festicorn/axismundi/src/axismundi.cpp. That firmware renders a
// particle field in normalized [0,1] space and rasterizes it onto each strip at
// that strip's own length (constant-PROPORTION policy) — which is exactly what
// makes it portable here: helix-canopy's 23 strips (297 / 50 / 55-60 LEDs) are
// just more "strips of arbitrary length". The kettle layer's math is not copied
// but shared: both this and the installation include lib/kettle_energy, so the
// tunings cannot drift apart.
//
// Differences from the firmware, all output-stage: no delta-sigma floor policy
// (an anti-flicker measure for real LEDs), and colors leave here as linear 0..1
// because the helix-canopy sink owns the display transform.
//
// Console state arrives from the physical desks via AX_CLICKY / AX_KETTLE,
// which mirror ClickyPacketV1 / KettlePacketV1 field for field.

#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <kettle_energy.h>
#include <duck_energy.h>
#include <hue_arc.h>
#include "topology.h"

#define AX_NUM_POTS 6
#define AX_NUM_BTNS 5

struct AxClickyState { uint8_t pullBits, btnBits; uint16_t pos[AX_NUM_POTS]; };
struct AxKettleState { uint8_t btnBits; int32_t enc; uint32_t upMs; };

static AxClickyState AX_CLICKY = { 0, 0, { 5000, 5000, 5000, 5000, 5000, 10000 } };
static AxKettleState AX_KETTLE = { 0, 0, 0 };
static DuckPacketV1  AX_DUCK = { DUCK_MAGIC, DUCK_VER, 0, 0, 0, 0, 0, 0, DUCK_FLAG_NO_IMU, 0 };
static bool AX_CLICKY_SEEN = false;
static bool AX_KETTLE_SEEN = false;
static bool AX_DUCK_SEEN = false;
static float AX_DUCK_AGE = 1e9f;
// Seconds since the host last pushed kettle state. The console bus stops
// pushing entirely once a desk goes stale, so this is what stops a console
// unplugged mid-press from acting forever.
static float AX_KETTLE_AGE = 1e9f;


#define AX_REF_LEN    100.0f
#define AX_MAX_STRIP  297

#define AX_BRIGHT_POT  5
#define AX_BRIGHT_FLOOR 0.0f

#define AX_SIGMA_NORM    0.025f
#define AX_PULL_NORM     0.04f
#define AX_POS_TAU       0.05f
#define AX_RUFFLE_AMP    0.45f
#define AX_RUFFLE_FREQ   1.4f
#define AX_RUFFLE_SPEED  7.0f

#define AX_BTN_DESAT   0
#define AX_BTN_PUMP    1
#define AX_BTN_HUEROT  2
#define AX_BTN_GATHER  3
#define AX_BTN_FADE    4
#define AX_BTN_TRAIL   5

#define AX_TRAIL_K     1.4f
#define AX_PAINT_S     0.6f
#define AX_PAINT_STOP_S 0.3f
#define AX_PAINT_CAP   160.0f
#define AX_FADE_S      1.5f
#define AX_RECOVER_S   0.7f
#define AX_CRACKLE_S   0.5f
#define AX_PUMP_MAX    3.0f
#define AX_PUMP_TAU    0.6f
#define AX_PUMP_RISE_S 1.0f
#define AX_HUEROT_DEG_S  90.0f
#define AX_RAINBOW_BURST 10
#define AX_EMIT_LIFE     0.45f
#define AX_ABSORB_LIFE   0.50f
#define AX_SPLIT_S       0.10f
#define AX_MOVE_EPS    0.0015f
#define AX_GATHER_S    0.25f
#define AX_DESAT_MIN   0.5f
#define AX_DESAT_S     0.3f
#define AX_NUM_SHARDS  48

static uint32_t ax_prng = 0x1234567u;
static inline uint32_t ax_xorshift32() {
    ax_prng ^= ax_prng << 13; ax_prng ^= ax_prng >> 17; ax_prng ^= ax_prng << 5;
    return ax_prng;
}
static inline float ax_rand() { return (float)(ax_xorshift32() & 0xFFFFFF) / 16777216.0f; }
static inline float ax_lerp(float a, float b, float t) { return a + (b - a) * t; }

static void ax_hsv(float h, float s, float v, float &r, float &g, float &b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r1, g1, b1;
    if      (h < 60)  { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else              { r1 = c; g1 = 0; b1 = x; }
    r = (r1 + m) * 255.0f; g = (g1 + m) * 255.0f; b = (b1 + m) * 255.0f;
}

// Must match CLICKY_HUES in axismundi/src/axismundi.cpp, where the 70 deg
// span that keeps red and blue apart is explained.
static const float AX_HUES[AX_NUM_POTS] = { 0, 60, 120, 180, 272, 300 };
static float ax_hueRotDeg = 0.0f;

struct AxShard { float pos, vel, hue, life, maxLife; };
static AxShard ax_shards[AX_NUM_SHARDS];

static float ax_pos[AX_NUM_POTS];
static bool  ax_posInit = false;
static float ax_rufflePhase = 0.0f;
// Level + LOGICAL hue, never rendered RGB: paint outlives the rotation that
// made it, so a baked color would strand old strokes outside the current arc.
static float ax_trailLvl[HC_NUM_LEDS];
static float ax_trailHue[HC_NUM_LEDS];
static float ax_fade = 1.0f;
static float ax_crackleT = 0.0f;
static float ax_pump[AX_NUM_POTS];
static int   ax_shardNext = 0;
static float ax_gather = 0.0f;
static float ax_movedAgo[AX_NUM_POTS];
static float ax_desat = 1.0f;
static uint8_t ax_prevPull = 0;
static float ax_split[AX_NUM_POTS];
static float ax_splitEase[AX_NUM_POTS];
static float ax_brightness = 1.0f;
static bool  ax_ready = false;

static void ax_reset() {
    ax_posInit = false;
    memset(ax_trailLvl, 0, sizeof(ax_trailLvl));
    memset(ax_trailHue, 0, sizeof(ax_trailHue));
    memset(ax_shards, 0, sizeof(ax_shards));
    ax_hueRotDeg = 0.0f;
    memset(ax_split, 0, sizeof(ax_split));
    memset(ax_splitEase, 0, sizeof(ax_splitEase));
    for (int k = 0; k < AX_NUM_POTS; k++) { ax_movedAgo[k] = 1e9f; ax_pump[k] = 1.0f; }
    ax_fade = 1.0f; ax_crackleT = 0.0f; ax_gather = 0.0f;
    ax_desat = 1.0f; ax_prevPull = 0; ax_shardNext = 0;
    ax_ready = true;
}

// ── Kettle carousel (spin-driven) ───────────────────────────────
// The spin drives ax_kettle.rot — the carousel phase every strip's composed
// output rotates by (each strip is a closed ring, one knob revolution per
// ring trip). Math from lib/kettle_energy.
#define AXK_STALE_S   1.0f
#define AXK_CRK_LEVEL 0.863f   // = axismundi.cpp KETTLE_CRK_LEVEL 220/255

static KettleState ax_kettle;

static void ax_kettle_reset() {
    kettleReset(ax_kettle);
    // The viewer has 23 strips but the sculpture has 4 rings; strips fan into
    // the same four phase groups via (s & 3) at the call sites.
    kettleSetRings(ax_kettle, 4);
}

static void ax_kettle_update(float dt) {
    AX_KETTLE_AGE += dt;
    if (AX_KETTLE_SEEN && AX_KETTLE_AGE < AXK_STALE_S)
        kettleInput(ax_kettle, AX_KETTLE.enc, AX_KETTLE.btnBits, AX_KETTLE.upMs);
    kettleRotStep(ax_kettle, dt);
}

// ── Duck energy waterfall ───────────────────────────────────────
// Features and waterfall math come from lib/duck_energy, shared with the
// installation. Cell state is flat over all LEDs, indexed by strip base like
// ax_trail. The viewer's clock is the effect time, so it supplies its own
// milliseconds for the rest-vector calibration window.
static DuckFeatures ax_duck;
static float ax_wf_lvl[HC_NUM_LEDS];
static float ax_wf_hue[HC_NUM_LEDS];
static float ax_wf_tlt[HC_NUM_LEDS];
static float ax_duck_inject = 0.0f;
static float ax_now_ms = 0.0f;

static void ax_duck_reset() {
    duckFeaturesReset(ax_duck);
    memset(ax_wf_lvl, 0, sizeof(ax_wf_lvl));
    memset(ax_wf_hue, 0, sizeof(ax_wf_hue));
    memset(ax_wf_tlt, 0, sizeof(ax_wf_tlt));
    ax_duck_inject = 0.0f;
}

static void ax_duck_update(float dt) {
    AX_DUCK_AGE += dt;
    ax_now_ms += dt * 1000.0f;
    bool live = AX_DUCK_SEEN && AX_DUCK_AGE < AXK_STALE_S;
    duckFeaturesUpdate(ax_duck, AX_DUCK, dt, (uint32_t)ax_now_ms, live);
    ax_duck_inject = duckWaterfallInject(ax_duck, live);
}

// Carousel output. comp is the clicky field in LOGICAL ring space and gets
// rotated by this ring's phase and scaled by the pot-5 fader; duck is the
// waterfall in PHYSICAL space and the crackle is splatted at fixed physical
// positions, so neither moves when the carousel spins.
static void ax_write_ring(RGBf* out, int base, int len, int ring,
                          const float (*comp)[3], const float (*duck)[3]) {
    static float crk[AX_MAX_STRIP];
    const float off = kettleRot(ax_kettle, ring) * (float)len;
    memset(crk, 0, sizeof(float) * len);
    kettleCrackleSplat(ax_kettle, ring, len, crk);
    const float s = ax_brightness / 255.0f;
    for (int i = 0; i < len; i++) {
        float srcF = fmodf((float)i - off, (float)len);
        if (srcF < 0.0f) srcF += (float)len;
        int i0 = (int)srcF;
        if (i0 >= len) i0 = len - 1;
        int i1 = (i0 + 1) % len;
        float fr = srcF - (float)i0;
        const float c = crk[i] * AXK_CRK_LEVEL;
        float pr = ax_lerp(comp[i0][0], comp[i1][0], fr) * s + duck[i][0] / 255.0f + c;
        float pg = ax_lerp(comp[i0][1], comp[i1][1], fr) * s + duck[i][1] / 255.0f + c;
        float pb = ax_lerp(comp[i0][2], comp[i1][2], fr) * s + duck[i][2] / 255.0f + c;
        hueArcRolloff(pr, pg, pb, 1.0f);
        out[base+i].r = pr;
        out[base+i].g = pg;
        out[base+i].b = pb;
    }
}

static void ax_raster_strip(RGBf* out, int base, int len, int ring,
                            float gatherT, float meanPos, float dt) {
    static float buf[AX_MAX_STRIP][3];
    static float cov[AX_MAX_STRIP];
    static float bufHue[AX_MAX_STRIP];
    static float bufW[AX_MAX_STRIP];
    static int8_t bufPot[AX_MAX_STRIP];
    memset(bufW, 0, sizeof(float) * len);
    memset(buf, 0, sizeof(float) * len * 3);
    memset(cov, 0, sizeof(float) * len);
    memset(bufPot, 0xFF, sizeof(int8_t) * len);

    const float sigScale = (float)len / AX_REF_LEN;

    for (int k = 0; k < AX_NUM_POTS; k++) {
        if (k == AX_BRIGHT_POT) continue;
        float centerN = ax_lerp(ax_pos[k], meanPos, gatherT);
        float center = centerN * (float)(len - 1);
        float sigma = fmaxf(0.5f, AX_SIGMA_NORM * AX_REF_LEN * sigScale * ax_pump[k]);

        if (ax_crackleT > 0.0f) {
            float env = ax_crackleT / AX_CRACKLE_S;
            for (int n = 0; n < 4; n++) {
                if (ax_rand() > env) continue;
                int i = (int)lroundf(center + (ax_rand() - 0.5f) * 4.0f * sigma);
                i = ((i % len) + len) % len;
                float v = (8.0f + ax_rand() * 30.0f) * env;
                buf[i][0] = buf[i][1] = buf[i][2] = v;
                cov[i] = 1.0f;
            }
            continue;
        }
        if (ax_fade <= 0.0f) continue;

        float r, g, b;
        ax_hsv(hueWrap360(AX_HUES[k] + ax_hueRotDeg),
               sqrtf(ax_fade), 1.0f, r, g, b);

        float fullSpacing = fmaxf(1.0f, AX_PULL_NORM * AX_REF_LEN * sigScale);
        float spacingF = ax_lerp(1.0f, fullSpacing, ax_splitEase[k]);
        int reach = (int)(4.0f * sigma);
        for (int j = -reach; j <= reach; j++) {
            float tc = center + (float)j * spacingF;
            float d = (float)j / sigma;
            float w = expf(-d * d) * ax_fade;
            float tcs = (ax_splitEase[k] <= 0.001f || ax_splitEase[k] >= 0.999f)
                ? (float)lroundf(tc) : tc;
            tcs -= floorf(tcs / (float)len) * (float)len;
            int i0 = (int)tcs;
            float fr = tcs - (float)i0;
            i0 %= len;
            int i1 = (i0 + 1) % len;
            float w0 = w * (1.0f - fr);
            buf[i0][0] += r * w0; buf[i0][1] += g * w0; buf[i0][2] += b * w0;
            cov[i0] += w0;
            if (w0 > bufW[i0]) { bufW[i0] = w0; bufHue[i0] = AX_HUES[k]; bufPot[i0] = (int8_t)k; }
            float w1 = w * fr;
            buf[i1][0] += r * w1; buf[i1][1] += g * w1; buf[i1][2] += b * w1;
            cov[i1] += w1;
            if (w1 > bufW[i1]) { bufW[i1] = w1; bufHue[i1] = AX_HUES[k]; bufPot[i1] = (int8_t)k; }
        }
    }

    for (int n = 0; n < AX_NUM_SHARDS; n++) {
        AxShard &sh = ax_shards[n];
        if (sh.life <= 0.0f) continue;
        float fadeF = fmaxf(0.0f, sh.life / sh.maxLife);
        float r, g, b;
        ax_hsv(hueWrap360(sh.hue + ax_hueRotDeg), 1.0f, 1.0f, r, g, b);
        int i = (int)lroundf(sh.pos * (float)(len - 1));
        for (int o = -1; o <= 1; o++) {
            int ii = ((i + o) % len + len) % len;
            float w = fadeF * (o == 0 ? 0.4f : 0.125f);
            float sr = r * w, sg = g * w, sb = b * w;
            if (fmaxf(sr, fmaxf(sg, sb)) >
                fmaxf(buf[ii][0], fmaxf(buf[ii][1], buf[ii][2]))) {
                buf[ii][0] = sr; buf[ii][1] = sg; buf[ii][2] = sb;
                bufW[ii] = w; bufHue[ii] = sh.hue; bufPot[ii] = -1;
            }
            cov[ii] = fmaxf(cov[ii], w);
        }
    }

    bool trailHeld = AX_CLICKY.btnBits & (1 << AX_BTN_TRAIL);
    if (trailHeld) {
        float rate = dt * (AX_PAINT_CAP / 255.0f) / AX_PAINT_S;
        for (int i = 0; i < len; i++) {
            float add = fmaxf(fmaxf(buf[i][0], buf[i][1]), buf[i][2]) * rate;
            if (add <= 0.0f) continue;
            int8_t p = bufPot[i];
            if (p >= 0) {
                float gate = 1.0f - ax_movedAgo[p] / AX_PAINT_STOP_S;
                if (gate <= 0.0f) continue;
                add *= gate;
            }
            float lvl = ax_trailLvl[base+i];
            float dh = fmodf(bufHue[i] - ax_trailHue[base+i] + 540.0f, 360.0f)
                       - 180.0f;
            ax_trailHue[base+i] = hueWrap360(ax_trailHue[base+i]
                                             + dh * (add / (add + lvl + 1e-6f)));
            ax_trailLvl[base+i] = fminf(AX_PAINT_CAP, lvl + add);
        }
    } else {
        float decay = expf(-AX_TRAIL_K * dt);
        for (int i = 0; i < len; i++)
            ax_trailLvl[base+i] *= decay;
    }

    static float comp[AX_MAX_STRIP][3];
    for (int i = 0; i < len; i++) {
        float ei = (float)i / (float)len * AX_REF_LEN;
        float ruffle = 1.0f + AX_RUFFLE_AMP
            * sinf(ei * AX_RUFFLE_FREQ + ax_rufflePhase);
        float bg = ax_trailLvl[base+i] * (1.0f / 255.0f);
        float tr = 0.0f, tg = 0.0f, tb = 0.0f;
        if (bg > 0.0f)
            ax_hsv(hueWrap360(ax_trailHue[base+i] + ax_hueRotDeg), 1.0f, 1.0f, tr, tg, tb);
        float cr = buf[i][0] + tr * bg;
        float cg = buf[i][1] + tg * bg;
        float cb = buf[i][2] + tb * bg;
        if (ax_desat < 0.999f) {
            float luma = 0.299f * cr + 0.587f * cg + 0.114f * cb;
            cr = luma + (cr - luma) * ax_desat;
            cg = luma + (cg - luma) * ax_desat;
            cb = luma + (cb - luma) * ax_desat;
        }
        comp[i][0] = cr * ruffle;
        comp[i][1] = cg * ruffle;
        comp[i][2] = cb * ruffle;
    }

    static float dbuf[AX_MAX_STRIP][3];
    memset(dbuf, 0, sizeof(float) * len * 3);
    duckWaterfallStep(ax_wf_lvl + base, ax_wf_hue + base, ax_wf_tlt + base, len,
                      ax_duck, ax_duck_inject, dt, ax_rand, dbuf, false);

    ax_write_ring(out, base, len, ring, comp, dbuf);
}

static void ax_spawn_burst(float centerN, bool outward) {
    for (int m = 0; m < AX_RAINBOW_BURST; m++) {
        AxShard &sh = ax_shards[ax_shardNext];
        ax_shardNext = (ax_shardNext + 1) % AX_NUM_SHARDS;
        int j = (m / 2 + 1) * ((m & 1) ? 1 : -1);
        float toothN = (float)j * AX_PULL_NORM;
        sh.hue = 360.0f * (float)m / (float)AX_RAINBOW_BURST
                 + (ax_rand() - 0.5f) * 20.0f;
        if (outward) {
            sh.maxLife = AX_EMIT_LIFE * (0.8f + ax_rand() * 0.4f);
            sh.pos = centerN;
            sh.vel = toothN / sh.maxLife;
        } else {
            sh.maxLife = AX_ABSORB_LIFE * (0.8f + ax_rand() * 0.4f);
            sh.pos = centerN + toothN;
            sh.vel = -toothN / sh.maxLife;
        }
        sh.life = sh.maxLife;
    }
}

// Scrubbing the viewer timeline backwards yields a negative delta; a stateful
// effect can only ever advance, so clamp rather than integrate the jump.
static inline float ax_dt(float t) {
    static float last = -1.0f;
    float dt = (last < 0.0f) ? (1.0f / 60.0f) : (t - last);
    last = t;
    if (dt < 0.0f) dt = 0.0f;
    if (dt > 0.1f) dt = 0.1f;
    return dt;
}

// Duck alone: the firmware's clicky-absent branch (zeroed frame, waterfall,
// carousel) with no particle layer to compose against.
static void ax_raster_duck_only(RGBf* out, int base, int len, int ring, float dt) {
    static float comp[AX_MAX_STRIP][3];
    static float dbuf[AX_MAX_STRIP][3];
    memset(comp, 0, sizeof(float) * len * 3);
    memset(dbuf, 0, sizeof(float) * len * 3);
    duckWaterfallStep(ax_wf_lvl + base, ax_wf_hue + base, ax_wf_tlt + base, len,
                      ax_duck, ax_duck_inject, dt, ax_rand, dbuf, false);
    ax_write_ring(out, base, len, ring, comp, dbuf);
}

static void ax_render_clicky(RGBf* out, float dt) {
    ax_rufflePhase += dt * AX_RUFFLE_SPEED;
    if (ax_rufflePhase >= 2.0f * (float)M_PI) ax_rufflePhase -= 2.0f * (float)M_PI;
    float alpha = fminf(1.0f, dt / AX_POS_TAU);

    bool fadeHeld   = AX_CLICKY.btnBits & (1 << AX_BTN_FADE);
    bool gatherHeld = AX_CLICKY.btnBits & (1 << AX_BTN_GATHER);
    bool desatHeld  = AX_CLICKY.btnBits & (1 << AX_BTN_DESAT);
    bool pumpHeld   = AX_CLICKY.btnBits & (1 << AX_BTN_PUMP);
    bool hueRotHeld = AX_CLICKY.btnBits & (1 << AX_BTN_HUEROT);

    float prevFade = ax_fade;
    if (fadeHeld) ax_fade = fmaxf(0.0f, ax_fade - dt / AX_FADE_S);
    else          ax_fade = fminf(1.0f, ax_fade + dt / AX_RECOVER_S);
    if (prevFade > 0.0f && ax_fade <= 0.0f) ax_crackleT = AX_CRACKLE_S;

    if (gatherHeld) ax_gather = fminf(1.0f, ax_gather + dt / AX_GATHER_S);
    else            ax_gather = fmaxf(0.0f, ax_gather - dt / AX_GATHER_S);
    float gatherT = ax_gather * ax_gather * (3.0f - 2.0f * ax_gather);

    float meanPos = 0.0f;
    for (int k = 0; k < AX_NUM_POTS; k++) {
        float target = (float)AX_CLICKY.pos[k] / 10000.0f;
        if (!ax_posInit) ax_pos[k] = target;
        float before = ax_pos[k];
        ax_pos[k] += alpha * (target - ax_pos[k]);
        if (fabsf(ax_pos[k] - before) > AX_MOVE_EPS) ax_movedAgo[k] = 0.0f;
        else ax_movedAgo[k] += dt;
        if (k != AX_BRIGHT_POT) meanPos += ax_pos[k];
    }
    meanPos /= (float)(AX_NUM_POTS - 1);
    ax_posInit = true;

    ax_brightness = AX_BRIGHT_FLOOR + (1.0f - AX_BRIGHT_FLOOR) * ax_pos[AX_BRIGHT_POT];

    if (hueRotHeld)
        ax_hueRotDeg = hueWrap360(ax_hueRotDeg + AX_HUEROT_DEG_S * dt);

    if (pumpHeld) {
        float rise = dt * (AX_PUMP_MAX - 1.0f) / AX_PUMP_RISE_S;
        for (int k = 0; k < AX_NUM_POTS; k++)
            ax_pump[k] = fminf(AX_PUMP_MAX, ax_pump[k] + rise);
    } else {
        for (int k = 0; k < AX_NUM_POTS; k++)
            ax_pump[k] = 1.0f + (ax_pump[k] - 1.0f) * expf(-dt / AX_PUMP_TAU);
    }

    uint8_t pulledEdge = AX_CLICKY.pullBits & ~ax_prevPull;
    uint8_t pushedEdge = ~AX_CLICKY.pullBits & ax_prevPull;
    ax_prevPull = AX_CLICKY.pullBits;
    for (int k = 0; k < AX_NUM_POTS; k++) {
        if (k == AX_BRIGHT_POT) continue;
        if (pulledEdge & (1 << k)) ax_spawn_burst(ax_pos[k], true);
        if (pushedEdge & (1 << k)) ax_spawn_burst(ax_pos[k], false);
        if (AX_CLICKY.pullBits & (1 << k))
            ax_split[k] = fminf(1.0f, ax_split[k] + dt / AX_SPLIT_S);
        else
            ax_split[k] = fmaxf(0.0f, ax_split[k] - dt / AX_SPLIT_S);
        float s = ax_split[k];
        ax_splitEase[k] = s * s * (3.0f - 2.0f * s);
    }

    {
        float target = desatHeld ? AX_DESAT_MIN : 1.0f;
        float da = fminf(1.0f, dt / AX_DESAT_S);
        ax_desat += da * (target - ax_desat);
    }

    for (int n = 0; n < AX_NUM_SHARDS; n++) {
        AxShard &sh = ax_shards[n];
        if (sh.life <= 0.0f) continue;
        sh.life -= dt;
        sh.pos += sh.vel * dt;
        sh.pos -= floorf(sh.pos);
    }

    for (int s = 0; s < HC_NUM_STRIPS; s++)
        ax_raster_strip(out, HC_STRIPS[s].start, HC_STRIPS[s].count, s & 3,
                        gatherT, meanPos, dt);

    if (ax_crackleT > 0.0f) ax_crackleT -= dt;
}

// ── Resting armature (viewer-only) ──────────────────────────────────────────
// The installation has no equivalent and must not: real strips are physically
// visible when unlit, but a 3D panel showing pure black reads as a dead
// renderer rather than a sculpture at rest. So trace every strip at a luminance
// far below any real output, in a cool grey that cannot be mistaken for the
// amber palette. It is a per-pixel FLOOR, not a mode: a lit strip covers its own
// pixels and leaves its unlit neighbours traced, which is what the physical
// sculpture does. Gating it globally instead made the panel show less of the
// sculpture awake than at rest.
static float AXK_armature = 1.0f;
#define AXK_ARM_LEVEL  0.015f

static inline void ax_apply_armature(RGBf* out, float dt) {
    (void)dt;
    if (AXK_armature < 0.5f) return;
    const float v = AXK_ARM_LEVEL * AXK_armature;
    // Whole-pixel replacement, never a per-channel max: the trace grey is
    // tinted, so maxing it into a dim LIT pixel drags that pixel off its hue.
    // Measured with every layer requesting one hue, per-channel maxing smeared
    // the panel 21.8 deg where this form smears 4.1, and that smear is charged
    // against the red/blue gap the chord is solved into.
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        if (fmaxf(out[i].r, fmaxf(out[i].g, out[i].b)) >= v) continue;
        out[i].r = v * 0.50f;
        out[i].g = v * 0.62f;
        out[i].b = v;
    }
}

// Layer composition, mirroring the firmware's loop(): the kettle layer always
// runs, and either the clicky raster max-blends it in, or it renders alone.
// Either console may be absent; with neither the sculpture holds dark.
static inline void fx_axismundi(RGBf* out, float t) {
    if (!ax_ready) { ax_reset(); ax_kettle_reset(); ax_duck_reset(); }
    const float dt = ax_dt(t);

    if (!AX_CLICKY_SEEN && !AX_KETTLE_SEEN && !AX_DUCK_SEEN) {
        for (int i = 0; i < HC_NUM_LEDS; i++) out[i].r = out[i].g = out[i].b = 0.0f;
        ax_apply_armature(out, dt);
        return;
    }

    ax_kettle_update(dt);
    ax_duck_update(dt);

    if (AX_CLICKY_SEEN) {
        ax_render_clicky(out, dt);
    } else {
        for (int i = 0; i < HC_NUM_LEDS; i++) out[i].r = out[i].g = out[i].b = 0.0f;
        ax_brightness = 1.0f;
        for (int s = 0; s < HC_NUM_STRIPS; s++)
            ax_raster_duck_only(out, HC_STRIPS[s].start, HC_STRIPS[s].count,
                                s & 3, dt);
    }
    ax_apply_armature(out, dt);
}
