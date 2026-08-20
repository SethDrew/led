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
// The host memsets nothing here — it reads axfxBuf[s][i][3] (linear, 0..255,
// PHYSICAL space) and adds it alongside the duck waterfall AFTER the carousel
// rotation, so the overlay stays pinned to the sculpture while the field spins.

#ifndef AXFX_AMBIENT_H
#define AXFX_AMBIENT_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <fast_math.h>

enum AxfxEffect : int8_t { AXFX_NONE = -1, AXFX_BLOOM = 0, AXFX_FIRE = 1,
                           AXFX_LEAFWIND = 2 };

static float axfxBuf[AXFX_NS][AXFX_MAXLEN][3];

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

struct AxfxDrive {
    float floor, ceiling;
    float energy, onset, level, prevSpeed;
};

static inline void axfxDriveInit(AxfxDrive &d) {
    d.floor = AXFX_FLOOR;
    d.ceiling = 1.0f;
    d.energy = d.onset = d.prevSpeed = 0.0f;
    d.level = AXFX_FLOOR;
}

static inline void axfxDriveStep(AxfxDrive &d, float spinRevS, float dt) {
    if (dt <= 0.0f) return;
    float sp = fabsf(spinRevS);
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
#define AXFX_FIRE_DROPOUT_DEPTH 0.85f

static float axfxFireTime[AXFX_NS];
static float axfxFireBase = 0.0f;
static float axfxFireFlicker = 0.0f;
static float axfxFireColor = 0.0f;
static float axfxFirePrevE = 0.0f;
static float axfxFireDerivS = 0.0f;
static float axfxFireDropout = 0.0f;

static void axfxResetFire() {
    for (int s = 0; s < AXFX_NS; s++) axfxFireTime[s] = (float)s * 1.37f;
    axfxFireBase = axfxFireFlicker = axfxFireColor = 0.0f;
    axfxFirePrevE = axfxFireDerivS = axfxFireDropout = 0.0f;
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

    float energyDeriv = (energy - axfxFirePrevE) / fmaxf(dt, 0.001f);
    axfxFirePrevE = energy;
    float derivAlpha = fminf(1.0f, dt / 0.200f);
    axfxFireDerivS += derivAlpha * (energyDeriv - axfxFireDerivS);
    bool isSustaining = (!isSilent && energy > 0.05f && fabsf(axfxFireDerivS) <= 0.5f);
    if (isSustaining) axfxFireDropout = fminf(1.0f, axfxFireDropout + dt * 0.35f);
    else              axfxFireDropout = fmaxf(0.0f, axfxFireDropout - dt * 1.0f);
    float dropoutAmount = axfxFireDropout;

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

            float perLedDim = 0.0f, colorRedShift = 0.0f;
            if (dropoutAmount > 0.0f) {
                float resilience = fastSin(fi * 13.7f + t * 0.3f) * fastSin(fi * 9.1f + t * 0.2f) * 0.5f + 0.5f;
                perLedDim = axfxClamp((dropoutAmount - resilience * 0.7f) / 0.3f, 0.0f, 1.0f) * AXFX_FIRE_DROPOUT_DEPTH;
                colorRedShift = axfxClamp(perLedDim / 0.3f, 0.0f, 1.0f);
            }
            bright *= (1.0f - perLedDim);
            bright = axfxClamp(bright, 0.0f, 1.0f) * d.level;

            float colR = baseColR * (1.0f - colorRedShift) + redR * colorRedShift;
            float colG = baseColG * (1.0f - colorRedShift) + redG * colorRedShift;
            float colB = baseColB * (1.0f - colorRedShift) + redB * colorRedShift;

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

static const uint8_t AXFX_LW_PALETTE[][3] = {
    {255, 140, 20}, {240, 100, 10}, {220, 60, 5}, {200, 40, 10},
    {180, 30, 5},   {255, 180, 40}, {160, 25, 5},
};
#define AXFX_LW_PALETTE_SIZE (sizeof(AXFX_LW_PALETTE) / sizeof(AXFX_LW_PALETTE[0]))

struct AxfxLeaf { float pos, vel, boost, age, brightness; uint8_t r, g, b; bool active; };
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

static void axfxLwSpawnLeaf(int s, float boostGain) {
    for (int i = 0; i < AXFX_LW_MAX_LEAVES; i++) {
        if (axfxLeaves[s][i].active) continue;
        AxfxLeaf &lf = axfxLeaves[s][i];
        lf.active = true;
        lf.pos = -0.05f;
        lf.boost = AXFX_LW_BOOST_SPEED * boostGain * (0.5f + axfxRandF() * 0.5f);
        lf.vel = 0.0f;
        lf.age = 0.0f;
        lf.brightness = 0.0f;
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
    float windMult = 1.0f + d.energy * 1.5f;
    float turb = AXFX_LW_TURBULENCE * (1.0f + d.energy);
    float spawnInt = AXFX_LW_SPAWN_INT / (1.0f + d.energy * 2.0f);
    float boostGain = 1.0f + d.onset * 2.0f;

    for (int s = 0; s < AXFX_NS; s++) {
        int len = AXFX_LEN(s);
        axfxLwSpawnTimer[s] += dt;
        while (axfxLwSpawnTimer[s] >= spawnInt) {
            axfxLwSpawnTimer[s] -= spawnInt;
            axfxLwSpawnLeaf(s, boostGain);
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
                float intensity = expf(-(dd * dd) / AXFX_LW_GLOW_SQ2) * lf.brightness;
                if (intensity < 0.005f) continue;
                totalGlow += intensity;
                cr += intensity * lf.r;
                cg += intensity * lf.g;
                cb += intensity * lf.b;
            }
            float oR = 0.0f, oG = 0.0f, oB = 0.0f;
            if (totalGlow > 0.01f) {
                cr /= totalGlow; cg /= totalGlow; cb /= totalGlow;
                float bright = fminf(totalGlow * AXFX_LW_LEAF_VALUE, 1.0f) * d.level;
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

// ── Dispatch ──────────────────────────────────────────────────────────────
static void axfxReset(int effect) {
    if (effect == AXFX_BLOOM)         axfxResetBloom();
    else if (effect == AXFX_FIRE)     axfxResetFire();
    else if (effect == AXFX_LEAFWIND) axfxResetLeafWind();
}

// Fills axfxBuf for the active effect; zeros it when none is active.
static void axfxRender(int effect, const AxfxDrive &d, float dt) {
    if (effect == AXFX_BLOOM)         axfxRenderBloom(d, dt);
    else if (effect == AXFX_FIRE)     axfxRenderFire(d, dt);
    else if (effect == AXFX_LEAFWIND) axfxRenderLeafWind(d, dt);
    else memset(axfxBuf, 0, sizeof(axfxBuf));
}

#endif
