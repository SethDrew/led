// Kettle carousel — the spin-driven rotation math, hardware-independent.
//
// Shared by festicorn/axismundi/src/axismundi.cpp (the installation) and
// festicorn/axismundi/viz/src/axismundi_fx.h (the 3D visualizer), so the two
// cannot drift.
//
// Model: every strip is a closed ring, and the COMPOSED output frame (all
// layers) rotates around it, signed with spin direction. One knob revolution =
// one full trip around the ring. Rotation is a host raster transform: render
// everything in logical ring space, then read output pixel i from ring
// position i - kettleRot(k, ring)*len (wrapped, interpolated).
//
// Rings are host-defined phase groups, not strips. A host with more strips
// than rings maps strip -> ring slot itself.
//
// Four panel buttons (packet bits 1..4) drive four motion modes, all
// hold-to-actuate; the knob press (bit 0) is unassigned here.
//
// The btn-D crackle layer is deliberately PHYSICAL-space: events are stamped
// at a fixed output-pixel position and composited AFTER the carousel rotation.
// The 2026-07-28 removal of the old sparkle system was because logical-space
// spawns get scattered to arbitrary physical positions by rotation, which
// reads as glitches. Do not composite crackle before the rotation transform.

#ifndef KETTLE_ENERGY_H
#define KETTLE_ENERGY_H

#include <math.h>
#include <stdint.h>
#include <kettle_packet_v1.h>

#define KETTLE_ROT_TAU 0.12f   // detent steps glide instead of snapping

#define KETTLE_BTN_FLYWHEEL 1
#define KETTLE_BTN_TWIST    2
#define KETTLE_BTN_COUNTER  3
#define KETTLE_BTN_CRACKLE  4

#define KETTLE_MAX_RINGS   32
#define KETTLE_MAX_CRACKLE 48

#define KETTLE_FLY_GAIN_PER_REV 1.5f   // flywheel rev/s gained per rev of hand spin
#define KETTLE_FLY_TAU          3.5f   // friction time constant, seconds
#define KETTLE_FLY_VEL_MAX      2.5f   // rev/s

#define KETTLE_TWIST_GAIN 0.5f         // ring i advances (1 + GAIN*i) x base

#define KETTLE_CRK_RATE_BASE     25.0f // events/s at standstill while held
#define KETTLE_CRK_RATE_PER_REVS 40.0f // extra events/s per rev/s of spin
#define KETTLE_CRK_RATE_MAX      90.0f
#define KETTLE_CRK_LIFE_MIN      0.14f
#define KETTLE_CRK_LIFE_MAX      0.40f
#define KETTLE_CRK_SIGMA         0.006f // normalized ring space, ~1 LED at 150

struct KettleCrackle {
    float   pos;    // physical ring position, [0,1)
    float   t;
    float   life;
    float   amp;
    uint8_t ring;
    bool    active;
};

struct KettleState {
    float    rot;         // base ring phase (untwisted, uncountered), [0,1)
    float    rotPending;  // spin not yet eased into rot
    float    ringRot[KETTLE_MAX_RINGS];
    float    vel;         // flywheel angular velocity, rev/s
    int32_t  lastEnc;
    uint32_t lastUp;
    bool     encInit;

    uint8_t  btnBits;

    KettleCrackle crk[KETTLE_MAX_CRACKLE];
    float    crkAcc;
    uint32_t rng;
    uint8_t  numRings;
};

static inline uint32_t kettleRand(KettleState &k) {
    k.rng ^= k.rng << 13;
    k.rng ^= k.rng >> 17;
    k.rng ^= k.rng << 5;
    return k.rng;
}

static inline float kettleRandF(KettleState &k) {
    return (float)(kettleRand(k) & 0xFFFFFF) / 16777216.0f;
}

static inline void kettleSetRings(KettleState &k, int n) {
    if (n < 1) n = 1;
    if (n > KETTLE_MAX_RINGS) n = KETTLE_MAX_RINGS;
    k.numRings = (uint8_t)n;
}

static inline void kettleReset(KettleState &k) {
    k.rot = 0.0f;
    k.rotPending = 0.0f;
    for (int i = 0; i < KETTLE_MAX_RINGS; i++) k.ringRot[i] = 0.0f;
    k.vel = 0.0f;
    k.lastEnc = 0;
    k.lastUp = 0;
    k.encInit = false;
    k.btnBits = 0;
    for (int i = 0; i < KETTLE_MAX_CRACKLE; i++) k.crk[i].active = false;
    k.crkAcc = 0.0f;
    k.rng = 0x1EC11u;
    k.numRings = 4;
}

// Console state in: signed spin queues rotation, btnBits latched for the next
// step. upMs regression = console reboot; rebase instead of integrating the
// jump. Safe to call at packet rate or frame rate — it consumes deltas.
static inline void kettleInput(KettleState &k, int32_t enc, uint8_t btnBits,
                               uint32_t upMs) {
    if (!k.encInit || upMs < k.lastUp) {
        k.lastEnc = enc;
        k.encInit = true;
    }
    int32_t delta = enc - k.lastEnc;
    k.lastEnc = enc;
    k.lastUp = upMs;
    k.rotPending += (float)delta
        / (float)(KETTLE_COUNTS_PER_DETENT * KETTLE_DETENTS_PER_REV);
    k.btnBits = btnBits;
}

static inline bool kettleHeld(const KettleState &k, int bit) {
    return (k.btnBits >> bit) & 1;
}

static inline float kettleWrap01(float x) {
    x = fmodf(x, 1.0f);
    if (x < 0.0f) x += 1.0f;
    return x;
}

static inline void kettleCrackleStep(KettleState &k, float speedRevS, float dt) {
    for (int i = 0; i < KETTLE_MAX_CRACKLE; i++) {
        KettleCrackle &e = k.crk[i];
        if (!e.active) continue;
        e.t += dt;
        if (e.t >= e.life) e.active = false;
    }
    if (!kettleHeld(k, KETTLE_BTN_CRACKLE)) {
        k.crkAcc = 0.0f;
        return;
    }
    float rate = KETTLE_CRK_RATE_BASE + fabsf(speedRevS) * KETTLE_CRK_RATE_PER_REVS;
    if (rate > KETTLE_CRK_RATE_MAX) rate = KETTLE_CRK_RATE_MAX;
    k.crkAcc += rate * dt;
    if (k.crkAcc > 8.0f) k.crkAcc = 8.0f;
    while (k.crkAcc >= 1.0f) {
        k.crkAcc -= 1.0f;
        for (int i = 0; i < KETTLE_MAX_CRACKLE; i++) {
            KettleCrackle &e = k.crk[i];
            if (e.active) continue;
            e.pos = kettleRandF(k);
            e.t = 0.0f;
            e.life = KETTLE_CRK_LIFE_MIN
                + kettleRandF(k) * (KETTLE_CRK_LIFE_MAX - KETTLE_CRK_LIFE_MIN);
            e.amp = 0.55f + 0.45f * kettleRandF(k);
            e.ring = (uint8_t)(kettleRand(k) % k.numRings);
            e.active = true;
            break;
        }
    }
}

// Ease queued spin into the ring phases. Pending is drained rather than kept
// as an absolute target so float precision never degrades with total
// revolutions; per-ring phases accumulate incrementally so toggling twist or
// counter-rotation never jumps the sculpture.
static inline void kettleRotStep(KettleState &k, float dt) {
    if (dt <= 0.0f) return;

    float alpha = fminf(1.0f, dt / KETTLE_ROT_TAU);
    float step = k.rotPending * alpha;
    k.rotPending -= step;

    if (kettleHeld(k, KETTLE_BTN_FLYWHEEL)) {
        k.vel += step * KETTLE_FLY_GAIN_PER_REV;
        if (k.vel > KETTLE_FLY_VEL_MAX) k.vel = KETTLE_FLY_VEL_MAX;
        if (k.vel < -KETTLE_FLY_VEL_MAX) k.vel = -KETTLE_FLY_VEL_MAX;
    }
    float adv = step + k.vel * dt;
    k.vel *= expf(-dt / KETTLE_FLY_TAU);

    k.rot = kettleWrap01(k.rot + adv);

    bool twist = kettleHeld(k, KETTLE_BTN_TWIST);
    bool counter = kettleHeld(k, KETTLE_BTN_COUNTER);
    for (int i = 0; i < k.numRings; i++) {
        float ringAdv = adv;
        if (twist) ringAdv *= 1.0f + KETTLE_TWIST_GAIN * (float)i;
        if (counter && (i & 1)) ringAdv = -ringAdv;
        k.ringRot[i] = kettleWrap01(k.ringRot[i] + ringAdv);
    }

    kettleCrackleStep(k, adv / dt, dt);
}

static inline float kettleRot(const KettleState &k, int ring) {
    if (ring < 0 || ring >= k.numRings) return k.rot;
    return k.ringRot[ring];
}

static inline float kettleCrackleEnv(const KettleCrackle &e) {
    if (!e.active) return 0.0f;
    float u = e.t / e.life;
    if (u >= 1.0f) return 0.0f;
    return e.amp * expf(-4.0f * u) * (1.0f - u);
}

// Accumulate this ring's crackle into a physical-space scalar buffer of len
// pixels. The host adds colour and composites AFTER the carousel rotation.
//
// The host's tint should be achromatic or share the palette's hue. A fixed
// off-palette tint (this was warm white until 2026-07-28) partially cancels the
// chroma of the pixels it lands on, and near cancellation the residual hue
// swings anywhere.
static inline void kettleCrackleSplat(const KettleState &k, int ring, int len,
                                      float *dst) {
    if (len <= 0) return;
    const float sigma = KETTLE_CRK_SIGMA * (float)len;
    const int radius = (int)(3.0f * sigma) + 1;
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int i = 0; i < KETTLE_MAX_CRACKLE; i++) {
        const KettleCrackle &e = k.crk[i];
        if (!e.active || (int)e.ring != ring) continue;
        float env = kettleCrackleEnv(e);
        if (env <= 0.0f) continue;
        float c = e.pos * (float)len;
        int c0 = (int)c;
        for (int d = -radius; d <= radius; d++) {
            int idx = c0 + d;
            float x = (float)idx - c;
            idx %= len;
            if (idx < 0) idx += len;
            dst[idx] += env * expf(-x * x * inv2s2);
        }
    }
}

#endif
