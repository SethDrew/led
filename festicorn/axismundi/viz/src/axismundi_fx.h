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
#include <stdlib.h>
#include <string.h>
#define KETTLE_SPIN_GAIN 0.5f
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

#define AXFX_NS     HC_NUM_STRIPS
#define AXFX_MAXLEN AX_MAX_STRIP
#define AXFX_LEN(s) HC_STRIPS[s].count
#include <axfx_ambient.h>

// Ambient effect triggers (axfx). Three freed kettle buttons each summon one
// overlay while held; the knob spin is diverted from the carousel into that
// effect's drive. Mirrors axismundi_sandbox.cpp.
#define AXFX_BTN_BLOOM     KETTLE_BTN_FLYWHEEL
#define AXFX_BTN_FIRE      KETTLE_BTN_COUNTER
#define AXFX_BTN_LEAFWIND  KETTLE_BTN_CRACKLE
#define AXFX_CEIL_POT      0
static AxfxDrive ax_axfxDrive;
static int       ax_axfxEffect = AXFX_NONE;
static int32_t   ax_axfxPrevEnc = 0;
static bool      ax_axfxEncInit = false;
static float     ax_axfxCeiling = 1.0f;

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

// ── Ghost recorder (sandbox experiment, mirrors axismundi_sandbox.cpp) ──────
// AX_GHOST 1 replaces the six button holds: hold button k to record particle
// k's motion, release to leave a dim ghost looping the gesture ping-pong,
// starting from the release pose (backward first) after press/release-lag
// stillness is trimmed. Tap (< GHOST_MIN samples) clears. 0 restores
// production button behavior.
#define AX_GHOST 1
#define AXG_CAP_HZ       10.0f
#define AXG_MAX_S        90
#define AXG_MAX_SAMPLES  ((int)(AXG_CAP_HZ * AXG_MAX_S))
#define AXG_MIN_SAMPLES  3
#define AXG_BTN_LANES    5
#define AXG_LEVEL        0.45f
#define AXG_FADE_S       30.0f
#define AXG_TRIM_MAX_S   1.0f
#define AXG_TRIM_EPS     0.003f

// Button -> pots recorded, many-to-many; recordings are keyed by BUTTON so
// two buttons can hold two gestures of the same particles. Default: every
// recording button captures the WHOLE FIELD. Mirrors GHOST_BTN_POTS in
// axismundi_sandbox.cpp — keep the two tables identical.
static const uint8_t AXG_BTN_POTS[6] = { 0x1F, 0x1F, 0x1F, 0x1F, 0, 0 };

// Buttons 4 (GPIO 16) and 5 (GPIO 13) keep their production functions.
#define AXG_BTN_TRAIL  4
#define AXG_BTN_HUEROT 5

// Heap-allocated and quantized like the firmware (its DRAM constraint,
// kept identical here): pos u16, split u8.
typedef uint16_t AxgPosTrack[AXG_BTN_LANES][AXG_MAX_SAMPLES];
typedef uint8_t  AxgSplitTrack[AXG_BTN_LANES][AXG_MAX_SAMPLES];
static AxgPosTrack*   axg_track = nullptr;
static AxgSplitTrack* axg_splitTrack = nullptr;
static inline uint16_t axg_q(float x) {
    return (uint16_t)(fminf(1.0f, fmaxf(0.0f, x)) * 65535.0f + 0.5f);
}
static inline uint8_t axg_q8(float x) {
    return (uint8_t)(fminf(1.0f, fmaxf(0.0f, x)) * 255.0f + 0.5f);
}
#define AXG_DQ  (1.0f / 65535.0f)
#define AXG_DQ8 (1.0f / 255.0f)
// Per-sample console context, replayed with the gesture: global hue offset
// (ghost = self-contained color memory) and the trail bit.
typedef uint16_t AxgHueTrack[AXG_MAX_SAMPLES];
typedef uint8_t  AxgFlagTrack[AXG_MAX_SAMPLES];
static AxgHueTrack*  axg_hueTrack = nullptr;
static AxgFlagTrack* axg_flagTrack = nullptr;
static uint8_t  axg_lanePot[6][AXG_BTN_LANES];
static uint8_t  axg_lanes[6];
static uint16_t axg_count[6];
static bool     axg_rec[6];
static float    axg_playPos[6];
static float    axg_playDir[6];
static float    axg_capAccum[6];
static uint8_t  axg_prevBtn = 0;
static float    axg_renderPos[6][AXG_BTN_LANES];
static float    axg_renderSplit[6][AXG_BTN_LANES];
static float    axg_renderLvl[6];
static float    axg_age[6];
static bool     axg_renderOn[6];

// Overwritten (or tapped-away) recording dissolves instead of vanishing.
#define AXG_OUT_S 3.0f
static float   axg_outPos[6][AXG_BTN_LANES];
static float   axg_outSplit[6][AXG_BTN_LANES];
static uint8_t axg_outPot[6][AXG_BTN_LANES];
static uint8_t axg_outLanes[6];
static float   axg_outLvl[6];
static float   axg_outT[6];
static float   axg_outHueRot[6];

static float axg_renderHueRot[6];
static bool  axg_renderTrail[6];
static float axg_laneMovedAgo[6][AXG_BTN_LANES];

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

static inline void axg_paint(int idx, float add, float hue) {
    float lvl = ax_trailLvl[idx];
    float dh = fmodf(hue - ax_trailHue[idx] + 540.0f, 360.0f) - 180.0f;
    ax_trailHue[idx] = hueWrap360(ax_trailHue[idx]
                                  + dh * (add / (add + lvl + 1e-6f)));
    ax_trailLvl[idx] = fminf(AX_PAINT_CAP, lvl + add);
}
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
    axfxDriveInit(ax_axfxDrive);
}

#define AX_LOOP_RECORD 0
#if AX_LOOP_RECORD
#include "loop_record_fx.h"
#endif

// Sandbox: knob press = return to known-good state — quadratic ease-out home
// along each ring's shortest path, not a snap. Only twist (btn B) reaches the
// kettle math; flywheel, counter-rotate, and crackle are masked off,
// reserved. Mirrors axismundi_sandbox.cpp.
#define AXK_RESET_S 1.2f
static float axk_resetFrom[4];
static float axk_resetT = -1.0f;

// Sandbox v2: twist is the only surviving spin modifier. Bare spin = carousel
// (leaks home at rest); hold twist and spin fans the rings into a helix. The
// other kettle buttons are freed for the ambient-effect triggers (axfx). Keep
// identical to axismundi_sandbox.cpp.
#define AXK_LEAK_TAU       6.0f
#define AXK_LEAK_DELAY_S   1.0f
#define AXK_LEAK_RAMP_S    2.0f
#define AXK_LEAK_EPS_REVS  0.02f

static void ax_kettle_update(float dt) {
    AX_KETTLE_AGE += dt;
    if (AX_KETTLE_SEEN && AX_KETTLE_AGE < AXK_STALE_S) {
        static uint8_t prevHold = 0;
        uint8_t hold = (AX_KETTLE.btnBits >> KETTLE_BTN_HOLD) & 1;
        if (hold && !prevHold) {
            ax_kettle.rotPending = 0.0f;
            ax_kettle.vel = 0.0f;
            for (int r = 0; r < 4; r++) {
                float p = ax_kettle.ringRot[r];
                axk_resetFrom[r] = p - roundf(p);
            }
            axk_resetT = 0.0f;
        }
        prevHold = hold;
        // Twist implies alternate-direction rings: feed the lib's counter
        // bit alongside twist so every other ring runs opposite.
        uint8_t tw = AX_KETTLE.btnBits & (uint8_t)(1 << KETTLE_BTN_TWIST);
        kettleInput(ax_kettle, AX_KETTLE.enc,
                    tw ? (uint8_t)(tw | (1 << KETTLE_BTN_COUNTER)) : 0,
                    AX_KETTLE.upMs);

        int held = AXFX_NONE;
        if ((AX_KETTLE.btnBits >> AXFX_BTN_BLOOM) & 1)         held = AXFX_BLOOM;
        else if ((AX_KETTLE.btnBits >> AXFX_BTN_FIRE) & 1)     held = AXFX_FIRE;
        else if ((AX_KETTLE.btnBits >> AXFX_BTN_LEAFWIND) & 1) held = AXFX_LEAFWIND;

        if (!ax_axfxEncInit) { ax_axfxPrevEnc = AX_KETTLE.enc; ax_axfxEncInit = true; }
        int32_t encDelta = AX_KETTLE.enc - ax_axfxPrevEnc;
        ax_axfxPrevEnc = AX_KETTLE.enc;
        float encRev = (float)encDelta
            / (float)(KETTLE_COUNTS_PER_DETENT * KETTLE_DETENTS_PER_REV);

        float driveSpin = 0.0f;
        if (held != AXFX_NONE) {
            ax_kettle.rotPending -= encRev * KETTLE_SPIN_GAIN;
            driveSpin = (dt > 0.0f) ? encRev / dt : 0.0f;
            if (held != ax_axfxEffect) { ax_axfxEffect = held; axfxReset(held); }
            ax_axfxDrive.floor = AXFX_FLOOR;
        } else if (ax_axfxEffect != AXFX_NONE) {
            ax_axfxDrive.floor = 0.0f;
            if (ax_axfxDrive.level < 0.006f && ax_axfxDrive.energy < 0.01f)
                ax_axfxEffect = AXFX_NONE;
        }
        ax_axfxDrive.ceiling = ax_axfxCeiling;
        axfxDriveStep(ax_axfxDrive, driveSpin, dt);
    }
    float rotBefore = ax_kettle.rot;
    kettleRotStep(ax_kettle, dt);
    float adv = ax_kettle.rot - rotBefore;
    adv -= roundf(adv);
    float speed = (dt > 0.0f) ? adv / dt : 0.0f;

    // Drift-home waits out a deadband after the spin stops, then ramps in.
    static float axk_idleT = 0.0f;
    if (fabsf(speed) >= AXK_LEAK_EPS_REVS) axk_idleT = 0.0f;
    else axk_idleT += dt;
    if (axk_resetT < 0.0f && axk_idleT > AXK_LEAK_DELAY_S) {
        float u = fminf(1.0f, (axk_idleT - AXK_LEAK_DELAY_S) / AXK_LEAK_RAMP_S);
        float f = expf(-dt * u / AXK_LEAK_TAU);
        for (int r = 0; r < 4; r++) {
            float d = ax_kettle.ringRot[r];
            d -= roundf(d);
            ax_kettle.ringRot[r] = kettleWrap01(d * f);
        }
        float d = ax_kettle.rot;
        d -= roundf(d);
        ax_kettle.rot = kettleWrap01(d * f);
    }

    if (axk_resetT >= 0.0f) {
        axk_resetT += dt;
        float u = axk_resetT / AXK_RESET_S;
        if (u >= 1.0f) {
            for (int r = 0; r < 4; r++) ax_kettle.ringRot[r] = 0.0f;
            ax_kettle.rot = 0.0f;
            axk_resetT = -1.0f;
        } else {
            float f = (1.0f - u) * (1.0f - u);
            for (int r = 0; r < 4; r++)
                ax_kettle.ringRot[r] = kettleWrap01(axk_resetFrom[r] * f);
            ax_kettle.rot = 0.0f;
        }
    }
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
static void ax_write_ring(RGBf* out, int sIdx, int base, int len, int ring,
                          const float (*comp)[3], const float (*duck)[3]) {
    static float crk[AX_MAX_STRIP];
    const float off = (ring < 0) ? 0.0f
        : kettleRot(ax_kettle, ring) * (float)len;
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
        float pr = ax_lerp(comp[i0][0], comp[i1][0], fr) * s + duck[i][0] / 255.0f + axfxBuf[sIdx][i][0] / 255.0f + c;
        float pg = ax_lerp(comp[i0][1], comp[i1][1], fr) * s + duck[i][1] / 255.0f + axfxBuf[sIdx][i][1] / 255.0f + c;
        float pb = ax_lerp(comp[i0][2], comp[i1][2], fr) * s + duck[i][2] / 255.0f + axfxBuf[sIdx][i][2] / 255.0f + c;
        hueArcRolloff(pr, pg, pb, 1.0f);
        out[base+i].r = pr;
        out[base+i].g = pg;
        out[base+i].b = pb;
    }
}

static inline bool axg_sampleNear(int b, int i, int j, int epsQ, int epsQ8) {
    for (int L = 0; L < axg_lanes[b]; L++) {
        int dp = (int)axg_track[b][L][i] - (int)axg_track[b][L][j];
        int ds = (int)axg_splitTrack[b][L][i] - (int)axg_splitTrack[b][L][j];
        if (dp > epsQ || -dp > epsQ || ds > epsQ8 || -ds > epsQ8) return false;
    }
    return true;
}

static void axg_trimStill(int b) {
    int n = axg_count[b];
    int cap = (int)(AXG_CAP_HZ * AXG_TRIM_MAX_S);
    int epsQ = (int)(AXG_TRIM_EPS * 65535.0f + 0.5f);
    int head = 0;
    while (head < cap && head + 1 < n
           && axg_sampleNear(b, head + 1, 0, epsQ, 2))
        head++;
    int tail = 0;
    while (tail < cap && head + tail + 2 < n
           && axg_sampleNear(b, n - 2 - tail, n - 1, epsQ, 2))
        tail++;
    int m = n - head - tail;
    if (head > 0) {
        for (int L = 0; L < axg_lanes[b]; L++) {
            memmove(axg_track[b][L], axg_track[b][L] + head,
                    m * sizeof(uint16_t));
            memmove(axg_splitTrack[b][L], axg_splitTrack[b][L] + head,
                    m * sizeof(uint8_t));
        }
        memmove(axg_hueTrack[b], axg_hueTrack[b] + head, m * sizeof(uint16_t));
        memmove(axg_flagTrack[b], axg_flagTrack[b] + head, m * sizeof(uint8_t));
    }
    axg_count[b] = (uint16_t)m;
}

static void axg_update(uint8_t btnBits, float dt) {
    if (!axg_track) {
        axg_track = (AxgPosTrack*)malloc(sizeof(AxgPosTrack) * 6);
        axg_splitTrack = (AxgSplitTrack*)malloc(sizeof(AxgSplitTrack) * 6);
        axg_hueTrack = (AxgHueTrack*)malloc(sizeof(AxgHueTrack) * 6);
        axg_flagTrack = (AxgFlagTrack*)malloc(sizeof(AxgFlagTrack) * 6);
    }
    if (!axg_track || !axg_splitTrack || !axg_hueTrack || !axg_flagTrack)
        return;
    uint8_t trailBit = (btnBits >> AXG_BTN_TRAIL) & 1;
    uint8_t edgeDown = btnBits & ~axg_prevBtn;
    uint8_t edgeUp   = ~btnBits & axg_prevBtn;
    axg_prevBtn = btnBits;
    for (int b = 0; b < 6; b++) {
        axg_renderOn[b] = false;
        if (AXG_BTN_POTS[b] == 0) continue;
        if (axg_outLvl[b] > 0.0f) {
            axg_outT[b] += dt;
            if (axg_outT[b] >= AXG_OUT_S) axg_outLvl[b] = 0.0f;
        }
        if (edgeDown & (1 << b)) {
            if (axg_count[b] > 0 && !axg_rec[b]) {
                axg_outLanes[b] = axg_lanes[b];
                for (int L = 0; L < axg_lanes[b]; L++) {
                    axg_outPot[b][L] = axg_lanePot[b][L];
                    axg_outPos[b][L] = axg_renderPos[b][L];
                    axg_outSplit[b][L] = axg_renderSplit[b][L];
                }
                axg_outLvl[b] = axg_renderLvl[b];
                axg_outHueRot[b] = axg_renderHueRot[b];
                axg_outT[b] = 0.0f;
            }
            axg_lanes[b] = 0;
            for (int k = 0; k < AX_NUM_POTS && axg_lanes[b] < AXG_BTN_LANES; k++)
                if ((AXG_BTN_POTS[b] & (1 << k)) && k != AX_BRIGHT_POT)
                    axg_lanePot[b][axg_lanes[b]++] = (uint8_t)k;
            axg_count[b] = 0;
            axg_rec[b] = true;
            axg_capAccum[b] = 1e9f;
        }
        if (axg_rec[b]) {
            axg_capAccum[b] += dt;
            if (axg_capAccum[b] >= 1.0f / AXG_CAP_HZ) {
                axg_capAccum[b] = 0.0f;
                if (axg_count[b] < AXG_MAX_SAMPLES) {
                    for (int L = 0; L < axg_lanes[b]; L++) {
                        int k = axg_lanePot[b][L];
                        axg_track[b][L][axg_count[b]] = axg_q(ax_pos[k]);
                        axg_splitTrack[b][L][axg_count[b]] = axg_q8(ax_splitEase[k]);
                    }
                    axg_hueTrack[b][axg_count[b]] = (uint16_t)
                        (hueWrap360(ax_hueRotDeg) * (65535.0f / 360.0f));
                    axg_flagTrack[b][axg_count[b]] = trailBit;
                    axg_count[b]++;
                }
            }
            if (edgeUp & (1 << b)) {
                axg_rec[b] = false;
                if (axg_count[b] < AXG_MIN_SAMPLES) axg_count[b] = 0;
                if (axg_count[b] > 0) axg_trimStill(b);
                axg_playPos[b] = (float)(axg_count[b] > 0 ? axg_count[b] - 1 : 0);
                axg_playDir[b] = -1.0f;
                axg_age[b] = 0.0f;
            }
            continue;
        }
        if (axg_count[b] == 0) continue;
        axg_age[b] += dt;
        if (axg_age[b] >= AXG_FADE_S) { axg_count[b] = 0; continue; }
        axg_renderLvl[b] = AXG_LEVEL * (1.0f - axg_age[b] / AXG_FADE_S);
        float span = (float)axg_count[b] - 1.0f;
        axg_playPos[b] += axg_playDir[b] * AXG_CAP_HZ * dt;
        if (span <= 0.0f) {
            axg_playPos[b] = 0.0f;
        } else {
            while (axg_playPos[b] > span || axg_playPos[b] < 0.0f) {
                if (axg_playPos[b] > span) {
                    axg_playPos[b] = 2.0f * span - axg_playPos[b];
                    axg_playDir[b] = -1.0f;
                }
                if (axg_playPos[b] < 0.0f) {
                    axg_playPos[b] = -axg_playPos[b];
                    axg_playDir[b] = 1.0f;
                }
            }
        }
        uint16_t i0 = (uint16_t)axg_playPos[b];
        if (i0 >= axg_count[b]) i0 = axg_count[b] - 1;
        uint16_t i1 = (uint16_t)((i0 + 1 < axg_count[b]) ? i0 + 1 : i0);
        float fr = axg_playPos[b] - (float)i0;
        for (int L = 0; L < axg_lanes[b]; L++) {
            float npos = AXG_DQ *
                ax_lerp(axg_track[b][L][i0], axg_track[b][L][i1], fr);
            if (fabsf(npos - axg_renderPos[b][L]) > AX_MOVE_EPS)
                axg_laneMovedAgo[b][L] = 0.0f;
            else
                axg_laneMovedAgo[b][L] += dt;
            axg_renderPos[b][L] = npos;
            axg_renderSplit[b][L] = AXG_DQ8 *
                ax_lerp(axg_splitTrack[b][L][i0], axg_splitTrack[b][L][i1], fr);
        }
        float h0 = axg_hueTrack[b][i0] * (360.0f / 65535.0f);
        float h1 = axg_hueTrack[b][i1] * (360.0f / 65535.0f);
        float dh = h1 - h0;
        dh -= 360.0f * roundf(dh / 360.0f);
        axg_renderHueRot[b] = hueWrap360(h0 + dh * fr);
        axg_renderTrail[b] = axg_flagTrack[b][i0] & 1;
        axg_renderOn[b] = true;
    }
}

static float ax_axfrac[HC_NUM_LEDS];
// Canopy azimuth per LED: the rim samples the field by wheel angle.
static float ax_azim[HC_NUM_LEDS];
static bool  ax_axfrac_ready = false;

// Canopy sampling: the field is rasterized at the spoke's RADIAL length
// (lenR = axn). Spoke LEDs (i < axn) read it by index — knobs own this axis,
// and travel ENDS at the spoke tip. Rim LEDs read it by wheel azimuth,
// triangle-folded so the whole particle list appears TWICE, mirror-reflected
// around the wheel — and the kettle rotation spins that layer only.
static inline float ax_coordF(bool canopy, int base, int i, int lenR) {
    if (!canopy)
        return ax_axfrac[base + i] * (float)(lenR - 1);
    if (i < lenR)
        return (float)i;
    float u = ax_azim[base + i] * (0.5f / (float)M_PI) - ax_kettle.rot;
    u -= floorf(u);
    // the particle list spans half the wheel and repeats point-reflected
    // through the center: every particle's twin sits 180 deg across
    if (u >= 0.5f) u -= 0.5f;
    return 2.0f * u * (float)(lenR - 1);
}

// Per-LED raster coordinate in [0,1]. Index-space for normal strips; for
// roots (kind 2, axn set) it is the LED's position along the POLE AXIS,
// measured as normalized radial distance from the sculpture center — valid
// for any wire path on the pole (spiral, serpentine, out-and-back, meander).

static void ax_build_axfrac() {
    for (int s = 0; s < HC_NUM_STRIPS; s++) {
        int b = HC_STRIPS[s].start, n = HC_STRIPS[s].count;
        if (HC_STRIPS[s].kind == 1) {
            for (int i = 0; i < n; i++) {
                ax_azim[b + i] = atan2f(LED_POS[b + i].y, LED_POS[b + i].x);
                ax_axfrac[b + i] = (n > 1) ? (float)i / (float)(n - 1) : 0.0f;
            }
            continue;
        }
        if (HC_STRIPS[s].axn == 0) {
            for (int i = 0; i < n; i++)
                ax_axfrac[b + i] = (n > 1) ? (float)i / (float)(n - 1) : 0.0f;
            continue;
        }
        float rmin = 1e9f, rmax = -1e9f;
        for (int i = 0; i < n; i++) {
            float r = hypotf(LED_POS[b + i].x, LED_POS[b + i].y);
            if (r < rmin) rmin = r;
            if (r > rmax) rmax = r;
        }
        float span = fmaxf(1e-6f, rmax - rmin);
        for (int i = 0; i < n; i++)
            ax_axfrac[b + i] =
                (hypotf(LED_POS[b + i].x, LED_POS[b + i].y) - rmin) / span;
    }
    ax_axfrac_ready = true;
}

// The particle field is splatted at lenR (RASTER resolution) and mapped onto
// the strip's len physical LEDs through ax_axfrac in the compose loop. For
// most strips lenR == len and the mapping is the identity. For roots lenR =
// axn (the pole's axial resolution), so a particle reads as a circumferential
// band sliding down the root, whatever shape the wire takes.
static void ax_raster_strip(RGBf* out, int sIdx, int base, int len, int lenR, int ring,
                            int kind, float gatherT, float meanPos, float dt) {
    const bool canopy = (kind == 1);
    static float buf[AX_MAX_STRIP][3];
    static float cov[AX_MAX_STRIP];
    static float bufHue[AX_MAX_STRIP];
    static float bufW[AX_MAX_STRIP];
    static int8_t bufPot[AX_MAX_STRIP];
    memset(bufW, 0, sizeof(float) * lenR);
    memset(buf, 0, sizeof(float) * lenR * 3);
    memset(cov, 0, sizeof(float) * lenR);
    memset(bufPot, 0xFF, sizeof(int8_t) * lenR);
    static float gpW[AX_MAX_STRIP];
    static float gpBest[AX_MAX_STRIP];
    static float gpHue[AX_MAX_STRIP];
    memset(gpW, 0, sizeof(float) * lenR);
    memset(gpBest, 0, sizeof(float) * lenR);

    const float sigScale = (float)lenR / AX_REF_LEN;

    for (int k = 0; k < AX_NUM_POTS; k++) {
        if (k == AX_BRIGHT_POT) continue;
        float centerN = ax_lerp(ax_pos[k], meanPos, gatherT);
        float center = centerN * (float)(lenR - 1);
        // the END position renders: peak may sit on the last LED, only
        // tails/teeth clip
        center = fminf(fmaxf(center, 0.0f), (float)(lenR - 1));
        float sigma = fmaxf(0.5f, AX_SIGMA_NORM * AX_REF_LEN * sigScale * ax_pump[k]);

        if (ax_crackleT > 0.0f) {
            float env = ax_crackleT / AX_CRACKLE_S;
            for (int n = 0; n < 4; n++) {
                if (ax_rand() > env) continue;
                int i = (int)lroundf(center + (ax_rand() - 0.5f) * 4.0f * sigma);
                if (i < 0 || i >= lenR) continue;
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

        float easeK = ax_splitEase[k];
        float fullSpacing = fmaxf(1.0f, AX_PULL_NORM * AX_REF_LEN * sigScale);
        float spacingF = ax_lerp(1.0f, fullSpacing, easeK);
        int reach = (int)(4.0f * sigma);
        for (int j = -reach; j <= reach; j++) {
            float tc = center + (float)j * spacingF;
            float d = (float)j / sigma;
            float w = expf(-d * d) * ax_fade;
            float tcs = (easeK <= 0.001f || easeK >= 0.999f)
                ? (float)lroundf(tc) : tc;
            // Clip at the strip ends — the field is a LINE, not a ring; only
            // kettle displacement (dup layer, carousel stage) bridges the seam.
            if (tcs < 0.0f || tcs > (float)(lenR - 1)) continue;
            int i0 = (int)tcs;
            float fr = tcs - (float)i0;
            int i1 = (i0 + 1 < lenR) ? i0 + 1 : i0;
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

#if AX_GHOST
    for (int gb = 0; gb < 6; gb++) {
        if (!axg_renderOn[gb]) continue;
        for (int L = 0; L < axg_lanes[gb]; L++) {
        int k = axg_lanePot[gb][L];
        float center = fminf(axg_renderPos[gb][L], 1.0f) * (float)(lenR - 1);
        float sigma = fmaxf(0.5f, AX_SIGMA_NORM * AX_REF_LEN * sigScale);
        float r, g, b;
        ax_hsv(hueWrap360(AX_HUES[k] + axg_renderHueRot[gb]), 1.0f, 1.0f, r, g, b);
        float ease = axg_renderSplit[gb][L];
        float fullSpacing = fmaxf(1.0f, AX_PULL_NORM * AX_REF_LEN * sigScale);
        float spacingF = ax_lerp(1.0f, fullSpacing, ease);
        int reach = (int)(4.0f * sigma);
        bool gpaint = axg_renderTrail[gb];
        float prate = 0.0f, ghue = 0.0f;
        if (gpaint) {
            float gate = 1.0f - axg_laneMovedAgo[gb][L] / AX_PAINT_STOP_S;
            if (gate <= 0.0f) {
                gpaint = false;
            } else {
                prate = dt * (AX_PAINT_CAP / 255.0f) / AX_PAINT_S * gate;
                // stored hue is LOGICAL: compose adds the live rotation, so
                // subtract it out to land exactly on the ghost's color now
                ghue = hueWrap360(AX_HUES[k] + axg_renderHueRot[gb]
                                  - ax_hueRotDeg);
            }
        }
        for (int j = -reach; j <= reach; j++) {
            float tc = center + (float)j * spacingF;
            float d = (float)j / sigma;
            float w = expf(-d * d) * axg_renderLvl[gb];
            float tcs = (ease <= 0.001f || ease >= 0.999f)
                ? (float)lroundf(tc) : tc;
            if (tcs < 0.0f || tcs > (float)(lenR - 1)) continue;
            int i0 = (int)tcs;
            float fr = tcs - (float)i0;
            int i1 = (i0 + 1 < lenR) ? i0 + 1 : i0;
            float w0 = w * (1.0f - fr);
            buf[i0][0] += r * w0; buf[i0][1] += g * w0; buf[i0][2] += b * w0;
            cov[i0] += w0;
            float w1 = w * fr;
            buf[i1][0] += r * w1; buf[i1][1] += g * w1; buf[i1][2] += b * w1;
            cov[i1] += w1;
            if (gpaint) {
                float bmax = fmaxf(r, fmaxf(g, b));
                float a0 = bmax * w0 * prate;
                gpW[i0] += a0;
                if (a0 > gpBest[i0]) { gpBest[i0] = a0; gpHue[i0] = ghue; }
                float a1 = bmax * w1 * prate;
                gpW[i1] += a1;
                if (a1 > gpBest[i1]) { gpBest[i1] = a1; gpHue[i1] = ghue; }
            }
        }
        }
    }

    for (int gb = 0; gb < 6; gb++) {
        if (axg_outLvl[gb] <= 0.001f) continue;
        float lvl = axg_outLvl[gb]
            * fmaxf(0.0f, 1.0f - axg_outT[gb] / AXG_OUT_S);
        for (int L = 0; L < axg_outLanes[gb]; L++) {
        int k = axg_outPot[gb][L];
        float center = fminf(axg_outPos[gb][L], 1.0f) * (float)(lenR - 1);
        float sigma = fmaxf(0.5f, AX_SIGMA_NORM * AX_REF_LEN * sigScale);
        float r, g, b;
        ax_hsv(hueWrap360(AX_HUES[k] + axg_outHueRot[gb]), 1.0f, 1.0f, r, g, b);
        float ease = axg_outSplit[gb][L];
        float fullSpacing = fmaxf(1.0f, AX_PULL_NORM * AX_REF_LEN * sigScale);
        float spacingF = ax_lerp(1.0f, fullSpacing, ease);
        int reach = (int)(4.0f * sigma);
        for (int j = -reach; j <= reach; j++) {
            float tc = center + (float)j * spacingF;
            float d = (float)j / sigma;
            float w = expf(-d * d) * lvl;
            float tcs = (ease <= 0.001f || ease >= 0.999f)
                ? (float)lroundf(tc) : tc;
            if (tcs < 0.0f || tcs > (float)(lenR - 1)) continue;
            int i0 = (int)tcs;
            float fr = tcs - (float)i0;
            int i1 = (i0 + 1 < lenR) ? i0 + 1 : i0;
            float w0 = w * (1.0f - fr);
            buf[i0][0] += r * w0; buf[i0][1] += g * w0; buf[i0][2] += b * w0;
            cov[i0] += w0;
            float w1 = w * fr;
            buf[i1][0] += r * w1; buf[i1][1] += g * w1; buf[i1][2] += b * w1;
            cov[i1] += w1;
        }
        }
    }
#endif

    for (int i = 0; i < len; i++) {
        int ai = (int)(ax_coordF(canopy, base, i, lenR) + 0.5f);
        if (ai >= lenR) ai = lenR - 1;
        if (gpW[ai] > 0.0f) axg_paint(base + i, gpW[ai], gpHue[ai]);
    }

    for (int n = 0; n < AX_NUM_SHARDS; n++) {
        AxShard &sh = ax_shards[n];
        if (sh.life <= 0.0f) continue;
        float fadeF = fmaxf(0.0f, sh.life / sh.maxLife);
        float r, g, b;
        ax_hsv(hueWrap360(sh.hue + ax_hueRotDeg), 1.0f, 1.0f, r, g, b);
        int i = (int)lroundf(sh.pos * (float)(lenR - 1));
        for (int o = -1; o <= 1; o++) {
            int ii = i + o;
            if (ii < 0 || ii >= lenR) continue;
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

#if AX_GHOST
    bool trailHeld = AX_CLICKY.btnBits & (1 << AXG_BTN_TRAIL);
#else
    bool trailHeld = AX_CLICKY.btnBits & (1 << AX_BTN_TRAIL);
#endif
    if (trailHeld) {
        float rate = dt * (AX_PAINT_CAP / 255.0f) / AX_PAINT_S;
        for (int i = 0; i < len; i++) {
            int ai = (int)(ax_coordF(canopy, base, i, lenR) + 0.5f);
            if (ai >= lenR) ai = lenR - 1;
            float add = fmaxf(fmaxf(buf[ai][0], buf[ai][1]), buf[ai][2]) * rate;
            if (add <= 0.0f) continue;
            int8_t p = bufPot[ai];
            if (p >= 0) {
                float gate = 1.0f - ax_movedAgo[p] / AX_PAINT_STOP_S;
                if (gate <= 0.0f) continue;
                add *= gate;
            }
            float lvl = ax_trailLvl[base+i];
            float dh = fmodf(bufHue[ai] - ax_trailHue[base+i] + 540.0f, 360.0f)
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
        float srcF = ax_coordF(canopy, base, i, lenR);
        int a0 = (int)srcF;
        float afr = srcF - (float)a0;
        a0 %= lenR;
        int a1 = (a0 + 1) % lenR;
        float cr = ax_lerp(buf[a0][0], buf[a1][0], afr) + tr * bg;
        float cg = ax_lerp(buf[a0][1], buf[a1][1], afr) + tg * bg;
        float cb = ax_lerp(buf[a0][2], buf[a1][2], afr) + tb * bg;
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

    ax_write_ring(out, sIdx, base, len, canopy ? -1 : ring, comp, dbuf);
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
static void ax_raster_duck_only(RGBf* out, int sIdx, int base, int len, int ring, float dt) {
    static float comp[AX_MAX_STRIP][3];
    static float dbuf[AX_MAX_STRIP][3];
    memset(comp, 0, sizeof(float) * len * 3);
    memset(dbuf, 0, sizeof(float) * len * 3);
    duckWaterfallStep(ax_wf_lvl + base, ax_wf_hue + base, ax_wf_tlt + base, len,
                      ax_duck, ax_duck_inject, dt, ax_rand, dbuf, false);
    ax_write_ring(out, sIdx, base, len, ring, comp, dbuf);
}

static void ax_render_clicky(RGBf* out, float dt) {
    ax_rufflePhase += dt * AX_RUFFLE_SPEED;
    if (ax_rufflePhase >= 2.0f * (float)M_PI) ax_rufflePhase -= 2.0f * (float)M_PI;
    float alpha = fminf(1.0f, dt / AX_POS_TAU);

#if AX_GHOST
    bool fadeHeld   = false;
    bool gatherHeld = false;
    bool desatHeld  = false;
    bool pumpHeld   = false;
    bool hueRotHeld = AX_CLICKY.btnBits & (1 << AXG_BTN_HUEROT);
#else
    bool fadeHeld   = AX_CLICKY.btnBits & (1 << AX_BTN_FADE);
    bool gatherHeld = AX_CLICKY.btnBits & (1 << AX_BTN_GATHER);
    bool desatHeld  = AX_CLICKY.btnBits & (1 << AX_BTN_DESAT);
    bool pumpHeld   = AX_CLICKY.btnBits & (1 << AX_BTN_PUMP);
    bool hueRotHeld = AX_CLICKY.btnBits & (1 << AX_BTN_HUEROT);
#endif

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
    ax_axfxCeiling = ax_pos[AXFX_CEIL_POT];

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
    }

#if AX_GHOST
    axg_update(AX_CLICKY.btnBits, dt);
#endif

    for (int s = 0; s < HC_NUM_STRIPS; s++) {
        int count = HC_STRIPS[s].count;
        int lenR = HC_STRIPS[s].axn ? HC_STRIPS[s].axn : count;
        if (lenR < 2) lenR = 2;
        ax_raster_strip(out, s, HC_STRIPS[s].start, count, lenR, s & 3,
                        HC_STRIPS[s].kind, gatherT, meanPos, dt);
    }

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
    if (!ax_axfrac_ready) ax_build_axfrac();
    const float dt = ax_dt(t);

    if (!AX_CLICKY_SEEN && !AX_KETTLE_SEEN && !AX_DUCK_SEEN) {
        for (int i = 0; i < HC_NUM_LEDS; i++) out[i].r = out[i].g = out[i].b = 0.0f;
        ax_apply_armature(out, dt);
        return;
    }

    ax_kettle_update(dt);
#if AX_LOOP_RECORD
    axlr_tick(dt);
#endif
    ax_duck_update(dt);
    axfxRender(ax_axfxEffect, ax_axfxDrive, dt);

    if (AX_CLICKY_SEEN) {
        ax_render_clicky(out, dt);
    } else {
        for (int i = 0; i < HC_NUM_LEDS; i++) out[i].r = out[i].g = out[i].b = 0.0f;
        ax_brightness = 1.0f;
        for (int s = 0; s < HC_NUM_STRIPS; s++)
            ax_raster_duck_only(out, s, HC_STRIPS[s].start, HC_STRIPS[s].count,
                                s & 3, dt);
    }
#if AX_LOOP_RECORD
    axlr_apply(out);
#endif
    ax_apply_armature(out, dt);
}
