// Ambient effect overlays (bloom / fire / leaf-wind), ported from
// tree-of-record. Shared verbatim by the firmware (src/axismundi_sandbox.cpp)
// and its 3D twin (viz/src/axismundi_fx.h) so the two cannot drift — the twin
// pattern loop_record.h could not use because its capture format differs, but
// these effects are byte-identical on both.
//
// Host contract, defined before the include:
//   AXFX_NS         number of strips
//   AXFX_MAXLEN     max LEDs on any strip (sizes the state arrays)
//   AXFX_LEN(s)     LED count of strip s
// The host memsets nothing here — it reads axfxRow(s)[i][3] (linear, 0..255,
// PHYSICAL space) and adds it alongside the duck waterfall AFTER the carousel
// rotation, so the overlay stays pinned to the sculpture while the field spins.
// The sap layer (axfxSapRow) composites the same way and additionally needs
// each segment's journey span declared once via axfxSapSetSeg/SetSegJ.

#ifndef AXFX_AMBIENT_H
#define AXFX_AMBIENT_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fast_math.h>

enum AxfxEffect : int8_t { AXFX_NONE = -1, AXFX_BLOOM = 0, AXFX_FIRE = 1,
                           AXFX_LEAFWIND = 2 };

// Heap-allocated lazily, like the ghost tracks: the static DRAM segment
// overflows on the 10-segment mockup long before the runtime heap does.
typedef float AxfxStripBuf[AXFX_MAXLEN][3];
static AxfxStripBuf* axfxBuf = nullptr;

static inline float axfxClamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline float axfxLerp(float a, float b, float t) { return a + (b - a) * t; }

static uint32_t axfxRng = 0x9E3779B9u;
static inline uint32_t axfxRand32() {
    axfxRng ^= axfxRng << 13; axfxRng ^= axfxRng >> 17; axfxRng ^= axfxRng << 5;
    return axfxRng;
}
static inline float axfxRandF() {
    return (float)(axfxRand32() & 0xFFFFFF) / 16777216.0f;
}

// Sub-LSB channels dither a single die on/off at the ~12% floor (additive
// mixed-channel colour). Snap each channel that never reaches one 8-bit step to
// black so the overlay parks cleanly. See ledger per-channel-noise-gate.
#define AXFX_CH_SNAP 0.75f

// ── Drive: kettle spin → effect energy/onset + brightness envelope ──────
// Hold an effect button: the effect comes up at AXFX_FLOOR. Spinning the knob
// injects energy (internal liveliness) and pushes the brightness envelope from
// the floor up toward the clicky-pot ceiling; a spin impulse fires onset.
#define AXFX_FLOOR          0.12f
#define AXFX_ENERGY_TAU     0.35f   // spin-speed EMA
#define AXFX_ENERGY_REF     1.2f    // rev/s mapping to energy = 1
#define AXFX_ONSET_ACCEL    6.0f    // rev/s^2 mapping to onset = 1
#define AXFX_ONSET_TAU      0.25f
#define AXFX_LEVEL_RISE     2.5f    // envelope climb rate toward target
#define AXFX_LEVEL_FALL_TAU 1.5f    // envelope leak back to floor
#define AXFX_SPEED_TAU      0.12f   // fast, near-instantaneous spin magnitude

struct AxfxDrive {
    float floor, ceiling;
    float energy, onset, level, prevSpeed, speed;
};

static inline void axfxDriveInit(AxfxDrive &d) {
    d.floor = AXFX_FLOOR;
    d.ceiling = 1.0f;
    d.energy = d.onset = d.prevSpeed = d.speed = 0.0f;
    d.level = AXFX_FLOOR;
}

static inline void axfxDriveStep(AxfxDrive &d, float spinRevS, float dt) {
    if (dt <= 0.0f) return;
    float sp = fabsf(spinRevS);
    // Lightly-smoothed current spin magnitude (rev/s): collapses within a few
    // hundred ms of stopping, so effects that read it break on a pause instead
    // of coasting on the buffered energy EMA.
    float sa = fminf(1.0f, dt / AXFX_SPEED_TAU);
    d.speed += sa * (sp - d.speed);
    float e = fminf(1.0f, sp / AXFX_ENERGY_REF);
    float ea = fminf(1.0f, dt / AXFX_ENERGY_TAU);
    d.energy += ea * (e - d.energy);

    float accel = (sp - d.prevSpeed) / fmaxf(dt, 0.001f);
    d.prevSpeed = sp;
    float impulse = axfxClamp(accel / AXFX_ONSET_ACCEL, 0.0f, 1.0f);
    if (impulse > d.onset) d.onset = impulse;
    d.onset *= expf(-dt / AXFX_ONSET_TAU);

    float target = d.floor + (d.ceiling - d.floor) * d.energy;
    if (target > d.level) d.level += fminf(1.0f, dt * AXFX_LEVEL_RISE) * (target - d.level);
    else                  d.level += fminf(1.0f, dt / AXFX_LEVEL_FALL_TAU) * (target - d.level);
    if (d.level < d.floor) d.level = d.floor;
}

// ── Bloom ───────────────────────────────────────────────────────────────
#define AXFX_BLOOM_BREATH_MIN_PERIOD 3.0f
#define AXFX_BLOOM_BREATH_MAX_PERIOD 8.0f
#define AXFX_BLOOM_BREATH_MIN_PEAK   0.75f
#define AXFX_BLOOM_BREATH_MAX_PEAK   1.00f
#define AXFX_BLOOM_BREATH_FLOOR      0.28f
#define AXFX_BLOOM_CONTRAST          1.6f
#define AXFX_BLOOM_STRETCH_GAIN      2.5f
#define AXFX_BLOOM_AUDIO_CEIL        3.0f
#define AXFX_BLOOM_HUE_DRIFT_MIN     (1.0f / 45.0f)
#define AXFX_BLOOM_HUE_DRIFT_MAX     (1.0f / 15.0f)
#define AXFX_BLOOM_FLASH_R           200.0f
#define AXFX_BLOOM_FLASH_G           120.0f
#define AXFX_BLOOM_FLASH_B           255.0f
#define AXFX_BLOOM_FLASH_DECAY       3.0f
#define AXFX_BLOOM_A_R                 0.0f
#define AXFX_BLOOM_A_G               180.0f
#define AXFX_BLOOM_A_B               120.0f
#define AXFX_BLOOM_B_R               140.0f
#define AXFX_BLOOM_B_G                20.0f
#define AXFX_BLOOM_B_B               255.0f

static float axfxBreathPhase[AXFX_NS][AXFX_MAXLEN];
static float axfxBreathPeriod[AXFX_NS][AXFX_MAXLEN];
static float axfxBreathPeak[AXFX_NS][AXFX_MAXLEN];
static float axfxHueT[AXFX_NS][AXFX_MAXLEN];
static float axfxHueDrift[AXFX_NS][AXFX_MAXLEN];
static float axfxFlash;

static void axfxResetBloom() {
    axfxFlash = 0.0f;
    for (int s = 0; s < AXFX_NS; s++)
        for (int i = 0; i < AXFX_MAXLEN; i++) {
            axfxBreathPhase[s][i]  = axfxRandF();
            axfxBreathPeriod[s][i] = AXFX_BLOOM_BREATH_MIN_PERIOD
                + axfxRandF() * (AXFX_BLOOM_BREATH_MAX_PERIOD - AXFX_BLOOM_BREATH_MIN_PERIOD);
            axfxBreathPeak[s][i]   = AXFX_BLOOM_BREATH_MIN_PEAK
                + axfxRandF() * (AXFX_BLOOM_BREATH_MAX_PEAK - AXFX_BLOOM_BREATH_MIN_PEAK);
            axfxHueT[s][i]         = axfxRandF();
            float rate = AXFX_BLOOM_HUE_DRIFT_MIN
                + axfxRandF() * (AXFX_BLOOM_HUE_DRIFT_MAX - AXFX_BLOOM_HUE_DRIFT_MIN);
            axfxHueDrift[s][i]     = (axfxRandF() > 0.5f) ? rate : -rate;
        }
}

static void axfxRenderBloom(const AxfxDrive &d, float dt) {
    if (d.onset > axfxFlash) axfxFlash = d.onset;
    axfxFlash *= expf(-AXFX_BLOOM_FLASH_DECAY * dt);
    if (axfxFlash < 0.005f) axfxFlash = 0.0f;
    float flashLin = axfxFlash;
    float flashFrac = (axfxFlash > 0.1f) ? axfxClamp(axfxFlash / 0.3f, 0.0f, 1.0f) : 0.0f;
    float oneMinusFF = 1.0f - flashFrac;
    float audioDrive = AXFX_BLOOM_STRETCH_GAIN * d.energy;

    for (int s = 0; s < AXFX_NS; s++) {
        int len = AXFX_LEN(s);
        float glowSum = 0.0f;
        for (int i = 0; i < len; i++) {
            float breath = fastSinPhase(axfxBreathPhase[s][i]) * 0.5f + 0.5f;
            glowSum += AXFX_BLOOM_BREATH_FLOOR
                + breath * (axfxBreathPeak[s][i] - AXFX_BLOOM_BREATH_FLOOR);
        }
        float glowMean = glowSum / (float)len;
        for (int i = 0; i < len; i++) {
            float breath = fastSinPhase(axfxBreathPhase[s][i]) * 0.5f + 0.5f;
            float glow = AXFX_BLOOM_BREATH_FLOOR
                + breath * (axfxBreathPeak[s][i] - AXFX_BLOOM_BREATH_FLOOR);
            float w = axfxClamp((glow - glowMean) * 4.0f, -1.0f, 1.5f);
            float stretched = axfxClamp(glowMean + (glow - glowMean) * AXFX_BLOOM_CONTRAST, 0.0f, 1.0f);
            float ambient = fastGamma24(stretched) * d.level;
            float breathLin = axfxClamp(ambient + audioDrive * w * d.level, 0.0f, AXFX_BLOOM_AUDIO_CEIL);

            axfxBreathPhase[s][i] += dt / axfxBreathPeriod[s][i];
            if (axfxBreathPhase[s][i] >= 1.0f) axfxBreathPhase[s][i] -= 1.0f;
            axfxHueT[s][i] += axfxHueDrift[s][i] * dt;
            if (axfxHueT[s][i] > 1.0f) axfxHueT[s][i] -= 1.0f;
            else if (axfxHueT[s][i] < 0.0f) axfxHueT[s][i] += 1.0f;

            float h = axfxHueT[s][i];
            float baseR = axfxLerp(AXFX_BLOOM_A_R, AXFX_BLOOM_B_R, h);
            float baseG = axfxLerp(AXFX_BLOOM_A_G, AXFX_BLOOM_B_G, h);
            float baseB = axfxLerp(AXFX_BLOOM_A_B, AXFX_BLOOM_B_B, h);
            float fl = flashLin * d.level;

            float oR = baseR * breathLin + (baseR * oneMinusFF + AXFX_BLOOM_FLASH_R * flashFrac) * fl;
            float oG = baseG * breathLin + (baseG * oneMinusFF + AXFX_BLOOM_FLASH_G * flashFrac) * fl;
            float oB = baseB * breathLin + (baseB * oneMinusFF + AXFX_BLOOM_FLASH_B * flashFrac) * fl;
            if (oR < AXFX_CH_SNAP) oR = 0.0f;
            if (oG < AXFX_CH_SNAP) oG = 0.0f;
            if (oB < AXFX_CH_SNAP) oB = 0.0f;
            axfxBuf[s][i][0] = oR;
            axfxBuf[s][i][1] = oG;
            axfxBuf[s][i][2] = oB;
        }
    }
}

// ── Fire ────────────────────────────────────────────────────────────────
#define AXFX_FIRE_FLICKER_SCALE 3.0f
#define AXFX_FIRE_EMBER_FLOOR   0.78f
#define AXFX_FIRE_DEADBAND      0.08f
// Resting fire never dims below this brightness envelope (idle output lands at
// ~EMBER_FLOOR*this ^2.4 ≈ 10%), so a held-but-still fire keeps a live glow.
#define AXFX_FIRE_LEVEL_FLOOR   0.50f

static float axfxFireTime[AXFX_NS];
static float axfxFireBase = 0.0f;
static float axfxFireFlicker = 0.0f;
static float axfxFireColor = 0.0f;

static void axfxResetFire() {
    for (int s = 0; s < AXFX_NS; s++) axfxFireTime[s] = (float)s * 1.37f;
    axfxFireBase = axfxFireFlicker = axfxFireColor = 0.0f;
}

static void axfxRenderFire(const AxfxDrive &d, float dt) {
    float energy = d.energy;
    float onset  = d.onset;
    bool isSilent = energy < 0.001f;
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);

    float attackAlpha = fminf(1.0f, dt / 0.050f);
    float decayAlpha  = fminf(1.0f, dt / 2.0f);
    float targetBrightness = isSilent ? AXFX_FIRE_EMBER_FLOOR
                                      : fmaxf(AXFX_FIRE_EMBER_FLOOR, energy);
    if (targetBrightness > axfxFireBase) axfxFireBase += attackAlpha * (targetBrightness - axfxFireBase);
    else                                 axfxFireBase += decayAlpha  * (targetBrightness - axfxFireBase);

    float flickerAlpha = fminf(1.0f, dt / 0.200f);
    float deltaTarget = isSilent ? 0.0f : onset;
    axfxFireFlicker += flickerAlpha * (deltaTarget - axfxFireFlicker);

    float colorAttack = fminf(1.0f, dt / 0.080f);
    float colorDecay  = fminf(1.0f, dt / 2.0f);
    float colorTarget = isPercussiveOnly ? 0.0f : (isSilent ? 0.3f : fmaxf(0.3f, energy));
    if (colorTarget > axfxFireColor) axfxFireColor += colorAttack * (colorTarget - axfxFireColor);
    else                             axfxFireColor += colorDecay  * (colorTarget - axfxFireColor);

    float ce = axfxFireColor;
    const float WHITE_BLEND = 0.15f, RED_FULL = 0.5f;
    const float amberR = 255.0f, amberG = 75.0f, amberB = 5.0f;
    const float redR = 200.0f, redG = 15.0f, redB = 0.0f;
    const float whiteR = 180.0f, whiteG = 85.0f, whiteB = 160.0f;
    float baseColR, baseColG, baseColB;
    if (ce < WHITE_BLEND) {
        float tw = 1.0f - (ce / WHITE_BLEND);
        baseColR = amberR * (1.0f - tw) + whiteR * tw;
        baseColG = amberG * (1.0f - tw) + whiteG * tw;
        baseColB = amberB * (1.0f - tw) + whiteB * tw;
    } else {
        float tr = fminf(1.0f, (ce - WHITE_BLEND) / (RED_FULL - WHITE_BLEND));
        baseColR = amberR * (1.0f - tr) + redR * tr;
        baseColG = amberG * (1.0f - tr) + redG * tr;
        baseColB = amberB * (1.0f - tr) + redB * tr;
    }

    float base = axfxFireBase;
    if (base < AXFX_FIRE_DEADBAND) base = 0.0f;
    float sScl = AXFX_FIRE_FLICKER_SCALE;

    for (int s = 0; s < AXFX_NS; s++) {
        axfxFireTime[s] = fmodf(axfxFireTime[s] + dt, 6283.1853f);
        float t = axfxFireTime[s];
        float sOff = (float)s * 17.0f;
        int len = AXFX_LEN(s);
        for (int i = 0; i < len; i++) {
            float fi = (float)i + sOff;
            float noise = fastSin(fi * 7.3f + t * 2.5f) * fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;
            float noiseAmp = fmaxf(0.15f * sScl, 0.10f * sScl / fmaxf(base, 0.1f));
            float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                         + axfxFireFlicker * (noise - 0.5f) * 0.25f * sScl;

            bright = axfxClamp(bright, 0.0f, 1.0f) * fmaxf(d.level, AXFX_FIRE_LEVEL_FLOOR);

            float colR = baseColR;
            float colG = baseColG;
            float colB = baseColB;

            float linBright = fastGamma24(bright);
            float oR = axfxClamp(colR * linBright, 0.0f, 255.0f);
            float oG = axfxClamp(colG * linBright, 0.0f, 255.0f);
            float oB = axfxClamp(colB * linBright, 0.0f, 255.0f);
            if (oR < AXFX_CH_SNAP) oR = 0.0f;
            if (oG < AXFX_CH_SNAP) oG = 0.0f;
            if (oB < AXFX_CH_SNAP) oB = 0.0f;
            axfxBuf[s][i][0] = oR;
            axfxBuf[s][i][1] = oG;
            axfxBuf[s][i][2] = oB;
        }
    }
}

// ── Leaf wind ─────────────────────────────────────────────────────────────
#define AXFX_LW_MAX_LEAVES  8
#define AXFX_LW_LEAF_VALUE   1.25f
#define AXFX_LW_GLOW_RADIUS  0.05f
#define AXFX_LW_GLOW_SQ2     (2.0f * AXFX_LW_GLOW_RADIUS * AXFX_LW_GLOW_RADIUS)
#define AXFX_LW_WIND_SPEED   0.35f
#define AXFX_LW_SPAWN_INT    0.9f
#define AXFX_LW_FADE_IN      0.4f
#define AXFX_LW_VEL_TAU      0.22f
#define AXFX_LW_TURBULENCE   0.3f
#define AXFX_LW_BOOST_SPEED  0.157f
#define AXFX_LW_BOOST_TC     20.0f
// Wind now tracks the CURRENT knob spin (d.speed), not the buffered energy EMA:
// spinning emits leaves, faster spin emits denser/brighter/faster, and pausing
// past EMIT_MIN for a few hundred ms stops new spawns so the stream breaks.
#define AXFX_LW_SPEED_REF    1.2f    // rev/s mapping to full wind
#define AXFX_LW_EMIT_MIN     0.06f   // rev/s below which the stream stops spawning

static const uint8_t AXFX_LW_PALETTE[][3] = {
    {255, 140, 20}, {240, 100, 10}, {220, 60, 5}, {200, 40, 10},
    {180, 30, 5},   {255, 180, 40}, {160, 25, 5},
};
#define AXFX_LW_PALETTE_SIZE (sizeof(AXFX_LW_PALETTE) / sizeof(AXFX_LW_PALETTE[0]))

struct AxfxLeaf { float pos, vel, boost, age, brightness, bmax; uint8_t r, g, b; bool active; };
static AxfxLeaf axfxLeaves[AXFX_NS][AXFX_LW_MAX_LEAVES];
static float axfxLwTime = 0.0f;
static float axfxLwSpawnTimer[AXFX_NS];

static inline float axfxLwNoise1d(float pos, float t, int seed) {
    return (fastSin(pos * 0.4f + t * 0.3f + seed * 7.3f)
          * fastSin(pos * 0.17f - t * 0.19f + seed * 3.1f + (float)M_PI * 0.5f)
          + fastSin(pos * 0.09f + t * 0.13f + seed * 1.7f) * 0.5f) / 1.5f;
}

static void axfxResetLeafWind() {
    axfxLwTime = 0.0f;
    for (int s = 0; s < AXFX_NS; s++) {
        axfxLwSpawnTimer[s] = axfxRandF() * AXFX_LW_SPAWN_INT;
        for (int i = 0; i < AXFX_LW_MAX_LEAVES; i++) axfxLeaves[s][i].active = false;
    }
}

static void axfxLwSpawnLeaf(int s, float boostGain, float wind) {
    for (int i = 0; i < AXFX_LW_MAX_LEAVES; i++) {
        if (axfxLeaves[s][i].active) continue;
        AxfxLeaf &lf = axfxLeaves[s][i];
        lf.active = true;
        lf.pos = -0.05f;
        lf.boost = AXFX_LW_BOOST_SPEED * boostGain * (0.5f + axfxRandF() * 0.5f);
        lf.vel = 0.0f;
        lf.age = 0.0f;
        lf.brightness = 0.0f;
        lf.bmax = 0.4f + 0.6f * wind;
        int ci = (int)(axfxRand32() % AXFX_LW_PALETTE_SIZE);
        lf.r = AXFX_LW_PALETTE[ci][0];
        lf.g = AXFX_LW_PALETTE[ci][1];
        lf.b = AXFX_LW_PALETTE[ci][2];
        return;
    }
}

static void axfxRenderLeafWind(const AxfxDrive &d, float dt) {
    axfxLwTime += dt;
    float boostDecay = expf(-dt / AXFX_LW_BOOST_TC);
    float lwAlpha = fminf(1.0f, dt / AXFX_LW_VEL_TAU);
    float wind = fminf(1.0f, d.speed / AXFX_LW_SPEED_REF);
    float windMult = 1.0f + wind * 1.5f;
    float turb = AXFX_LW_TURBULENCE * (1.0f + wind);
    bool emitting = d.speed > AXFX_LW_EMIT_MIN;
    float spawnInt = AXFX_LW_SPAWN_INT / (0.3f + wind * 3.0f);
    float boostGain = 1.0f + wind * 2.0f;

    for (int s = 0; s < AXFX_NS; s++) {
        int len = AXFX_LEN(s);
        if (emitting) {
            axfxLwSpawnTimer[s] += dt;
            // A primed idle timer (set huge below) overshoots one interval:
            // clamp so the first spin emits exactly one leaf, not a burst.
            if (axfxLwSpawnTimer[s] > spawnInt + dt) axfxLwSpawnTimer[s] = spawnInt;
            while (axfxLwSpawnTimer[s] >= spawnInt) {
                axfxLwSpawnTimer[s] -= spawnInt;
                axfxLwSpawnLeaf(s, boostGain, wind);
            }
        } else {
            axfxLwSpawnTimer[s] = 1.0e9f;
        }
        for (int i = 0; i < AXFX_LW_MAX_LEAVES; i++) {
            if (!axfxLeaves[s][i].active) continue;
            AxfxLeaf &lf = axfxLeaves[s][i];
            float noise = axfxLwNoise1d(lf.pos * 5.0f + (float)s * 3.0f, axfxLwTime, i);
            float speedMult = fmaxf(0.1f, 1.0f + noise * turb);
            float force = AXFX_LW_WIND_SPEED * windMult * speedMult;
            lf.vel = fmaxf(AXFX_LW_WIND_SPEED * 0.4f, lf.vel + lwAlpha * (force - lf.vel));
            lf.boost *= boostDecay;
            lf.pos += (lf.vel + lf.boost) * dt;
            lf.age += dt;
            lf.brightness = (lf.age < AXFX_LW_FADE_IN) ? (lf.age / AXFX_LW_FADE_IN) : 1.0f;
            if (lf.pos > 1.05f) lf.active = false;
        }
        for (int i = 0; i < len; i++) {
            float lpos = (float)i / (float)(len - 1);
            float totalGlow = 0.0f, cr = 0.0f, cg = 0.0f, cb = 0.0f;
            for (int li = 0; li < AXFX_LW_MAX_LEAVES; li++) {
                if (!axfxLeaves[s][li].active) continue;
                AxfxLeaf &lf = axfxLeaves[s][li];
                float dd = lpos - lf.pos;
                float intensity = expf(-(dd * dd) / AXFX_LW_GLOW_SQ2) * lf.brightness * lf.bmax;
                if (intensity < 0.005f) continue;
                totalGlow += intensity;
                cr += intensity * lf.r;
                cg += intensity * lf.g;
                cb += intensity * lf.b;
            }
            float oR = 0.0f, oG = 0.0f, oB = 0.0f;
            if (totalGlow > 0.01f) {
                cr /= totalGlow; cg /= totalGlow; cb /= totalGlow;
                float bright = fminf(totalGlow * AXFX_LW_LEAF_VALUE, 1.0f);
                float linBright = fastGamma24(bright);
                oR = cr * linBright; oG = cg * linBright; oB = cb * linBright;
                if (oR < AXFX_CH_SNAP) oR = 0.0f;
                if (oG < AXFX_CH_SNAP) oG = 0.0f;
                if (oB < AXFX_CH_SNAP) oB = 0.0f;
            }
            axfxBuf[s][i][0] = oR;
            axfxBuf[s][i][1] = oG;
            axfxBuf[s][i][2] = oB;
        }
    }
}

// ── Sap flow ──────────────────────────────────────────────────────────────
// Ported from bench-tree/src/foregrounds/SapFlowForeground.h (the original
// LED-tree sap flow), re-aimed from that tree's integer depth onto a GLOBAL
// journey coordinate j in [0,1] spanning the whole sculpture: root tip 0 ->
// trunk -> helix -> canopy tip 1. The field is one scalar function of j, so a
// particle sitting on a span boundary lights both adjoining segments.
// Depth constants divide by the original's MAX_DEPTH of 70.
#define AXFX_SAP_MAX_PARTICLES 12
#define AXFX_SAP_VEL_J       (7.25f / 70.0f)
#define AXFX_SAP_RADIUS_J    (3.0f / 70.0f)
#define AXFX_SAP_END_J       (75.0f / 70.0f)
#define AXFX_SAP_MIN_INT_S   1.91f
#define AXFX_SAP_MAX_INT_S   12.73f
#define AXFX_SAP_RATE_HZ     1.68f
#define AXFX_SAP_BRIGHT_MIN  80.0f
#define AXFX_SAP_BRIGHT_SPAN 70.0f
#define AXFX_SAP_R            34.0f
#define AXFX_SAP_G           139.0f
#define AXFX_SAP_B            34.0f
#define AXFX_SAP_BED_R  1.0f
#define AXFX_SAP_BED_G  6.0f
#define AXFX_SAP_BED_B  1.0f
// A segment spanning little journey over many LEDs would render the blob
// thinner than a pixel: floor the radius at this many LEDs, the CLICKY_SIGMA
// pattern expressed in journey units.
#define AXFX_SAP_MIN_LEDS 1.5f

// Journey boundaries, shared so the firmware's per-segment table and the
// twin's per-kind mapping cannot drift apart.
#define AXFX_SAP_J_ROOTS 0.35f
#define AXFX_SAP_J_HELIX 0.75f

// KettlePacketV1 btnBits bit (== KETTLE_BTN_KEEPTEMP): the keep-warm latched
// toggle — the wire bit is switch position, so every receiver board converges
// on it via the heartbeat regardless of loss.
#define AXFX_SAP_BTN 6

// Takeover boundary sweeping the journey. The console's indicator bar wipes on
// in ~1.6 s; the sculpture is deliberately ~2x statelier, and releases at 2:1.
#define AXFX_WIPE_ON_S   3.0f
#define AXFX_WIPE_OFF_S  1.5f
#define AXFX_WIPE_BAND   0.08f

struct AxfxSapParticle { float j, vel, bright; bool active; };
static AxfxSapParticle axfxSapP[AXFX_SAP_MAX_PARTICLES];
static float axfxSapSince = 0.0f;
static float axfxWipeP = 0.0f;

typedef float AxfxSapJBuf[AXFX_MAXLEN];
static AxfxSapJBuf*  axfxSapJ = nullptr;
static AxfxStripBuf* axfxSapBuf = nullptr;
static uint16_t axfxSapLen[AXFX_NS];
static float axfxSapRad[AXFX_NS];
static bool axfxSapLit = false;

// Stand-in row for a layer whose lazy calloc failed: the host composites it
// unconditionally, so degrade to black rather than dereference null.
static float axfxZeroRow[AXFX_MAXLEN][3];
static inline const float (*axfxRow(int s))[3] {
    return axfxBuf ? axfxBuf[s] : axfxZeroRow;
}
static inline const float (*axfxSapRow(int s))[3] {
    return axfxSapBuf ? axfxSapBuf[s] : axfxZeroRow;
}

static bool axfxSapAlloc() {
    if (!axfxSapJ)   axfxSapJ = (AxfxSapJBuf*)calloc(AXFX_NS, sizeof(AxfxSapJBuf));
    if (!axfxSapBuf) axfxSapBuf = (AxfxStripBuf*)calloc(AXFX_NS, sizeof(AxfxStripBuf));
    return axfxSapJ && axfxSapBuf;
}

// Host declares each segment's journey coordinate per LED. j may run either
// way along the wire and need not be monotonic (folded canopy chains).
static void axfxSapSetSegJ(int s, int len, const float *j) {
    if (s < 0 || s >= AXFX_NS) return;
    if (len <= 0 || !j || !axfxSapAlloc()) { axfxSapLen[s] = 0; return; }
    if (len > AXFX_MAXLEN) len = AXFX_MAXLEN;
    axfxSapLen[s] = (uint16_t)len;
    float lo = j[0], hi = j[0];
    for (int i = 0; i < len; i++) {
        axfxSapJ[s][i] = j[i];
        if (j[i] < lo) lo = j[i];
        if (j[i] > hi) hi = j[i];
    }
    float perLed = (len > 1) ? (hi - lo) / (float)(len - 1) : 0.0f;
    axfxSapRad[s] = fmaxf(AXFX_SAP_RADIUS_J, AXFX_SAP_MIN_LEDS * perLed);
}

// Index-space form: u runs 0->1 along the wire, or 1->0 when the wire head
// sits at the far end of the flow (the wfRev segments).
static void axfxSapSetSeg(int s, int len, float j0, float j1, bool rev) {
    static float tmp[AXFX_MAXLEN];
    if (len <= 0) { axfxSapSetSegJ(s, 0, nullptr); return; }
    if (len > AXFX_MAXLEN) len = AXFX_MAXLEN;
    float d = (len > 1) ? 1.0f / (float)(len - 1) : 0.0f;
    for (int i = 0; i < len; i++) {
        float u = (float)i * d;
        if (rev) u = 1.0f - u;
        tmp[i] = j0 + (j1 - j0) * u;
    }
    axfxSapSetSegJ(s, len, tmp);
}

static void axfxSapReset() {
    for (int i = 0; i < AXFX_SAP_MAX_PARTICLES; i++) axfxSapP[i].active = false;
    axfxSapSince = 0.0f;
    axfxWipeP = 0.0f;
    axfxSapLit = false;
}

// Two-stage gate: guaranteed arrival at MAX_INT, probabilistic at RATE_HZ once
// past MIN_INT — irregular but never dead.
static void axfxSapStep(float dt) {
    axfxSapSince += dt;
    bool spawn = false;
    if (axfxSapSince >= AXFX_SAP_MAX_INT_S) spawn = true;
    else if (axfxSapSince >= AXFX_SAP_MIN_INT_S)
        spawn = axfxRandF() < AXFX_SAP_RATE_HZ * dt;
    if (spawn) {
        for (int i = 0; i < AXFX_SAP_MAX_PARTICLES; i++) {
            if (axfxSapP[i].active) continue;
            axfxSapP[i].active = true;
            axfxSapP[i].j = 0.0f;
            axfxSapP[i].vel = AXFX_SAP_VEL_J;
            axfxSapP[i].bright = AXFX_SAP_BRIGHT_MIN
                + axfxRandF() * AXFX_SAP_BRIGHT_SPAN;
            axfxSapSince = 0.0f;
            break;
        }
    }
    for (int i = 0; i < AXFX_SAP_MAX_PARTICLES; i++) {
        if (!axfxSapP[i].active) continue;
        axfxSapP[i].j += axfxSapP[i].vel * dt;
        if (axfxSapP[i].j > AXFX_SAP_END_J) axfxSapP[i].active = false;
    }
}

static void axfxWipeStep(bool on, float dt) {
    float rate = on ? (dt / AXFX_WIPE_ON_S) : -(dt / AXFX_WIPE_OFF_S);
    axfxWipeP = axfxClamp(axfxWipeP + rate, 0.0f, 1.0f);
}

// Sap's share of a point in journey space. The band is swept a half-width past
// each end of the journey so the wipe parks fully off at p=0 and fully on at
// p=1 instead of straddling the tips.
static inline float axfxWipeAtJ(float j) {
    float lo = axfxWipeP * (1.0f + AXFX_WIPE_BAND) - AXFX_WIPE_BAND;
    float t = axfxClamp((j - lo) / AXFX_WIPE_BAND, 0.0f, 1.0f);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}

static inline float axfxWipeAt(int s, int i) {
    return axfxSapJ ? axfxWipeAtJ(axfxSapJ[s][i]) : 0.0f;
}

// Fills axfxSapBuf with the wipe already applied; the host scales its clicky
// particle layer by (1 - axfxWipeAt) and adds this on top.
static void axfxSapFrame(bool on, float dt) {
    axfxWipeStep(on, dt);
    if (!axfxSapAlloc()) return;
    axfxSapStep(dt);
    if (axfxWipeP <= 0.0f) {
        if (axfxSapLit) {
            memset(axfxSapBuf, 0, sizeof(AxfxStripBuf) * AXFX_NS);
            axfxSapLit = false;
        }
        return;
    }
    axfxSapLit = true;
    for (int s = 0; s < AXFX_NS; s++) {
        int len = axfxSapLen[s];
        float rad = axfxSapRad[s];
        float inv = (rad > 0.0f) ? 1.0f / rad : 0.0f;
        for (int i = 0; i < len; i++) {
            float j = axfxSapJ[s][i];
            float w = axfxWipeAtJ(j);
            float oR = AXFX_SAP_BED_R, oG = AXFX_SAP_BED_G, oB = AXFX_SAP_BED_B;
            for (int p = 0; p < AXFX_SAP_MAX_PARTICLES; p++) {
                if (!axfxSapP[p].active) continue;
                float d = fabsf(j - axfxSapP[p].j);
                if (d >= rad) continue;
                float f = 1.0f - d * inv;
                f *= f * axfxSapP[p].bright * (1.0f / 255.0f);
                oR += AXFX_SAP_R * f;
                oG += AXFX_SAP_G * f;
                oB += AXFX_SAP_B * f;
            }
            oR = fminf(255.0f, oR) * w;
            oG = fminf(255.0f, oG) * w;
            oB = fminf(255.0f, oB) * w;
            if (oR < AXFX_CH_SNAP) oR = 0.0f;
            if (oG < AXFX_CH_SNAP) oG = 0.0f;
            if (oB < AXFX_CH_SNAP) oB = 0.0f;
            axfxSapBuf[s][i][0] = oR;
            axfxSapBuf[s][i][1] = oG;
            axfxSapBuf[s][i][2] = oB;
        }
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────
static void axfxReset(int effect) {
    if (effect == AXFX_BLOOM)         axfxResetBloom();
    else if (effect == AXFX_FIRE)     axfxResetFire();
    else if (effect == AXFX_LEAFWIND) axfxResetLeafWind();
}

// Fills axfxBuf for the active effect; zeros it when none is active.
static void axfxRender(int effect, const AxfxDrive &d, float dt) {
    if (!axfxBuf) {
        axfxBuf = (AxfxStripBuf*)calloc(AXFX_NS, sizeof(AxfxStripBuf));
        if (!axfxBuf) return;
    }
    if (effect == AXFX_BLOOM)         axfxRenderBloom(d, dt);
    else if (effect == AXFX_FIRE)     axfxRenderFire(d, dt);
    else if (effect == AXFX_LEAFWIND) axfxRenderLeafWind(d, dt);
    else memset(axfxBuf, 0, sizeof(AxfxStripBuf) * AXFX_NS);
}

#endif
