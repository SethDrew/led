// Desktop simulation of road-bulbs effects.
// All math copied verbatim from receiver.cpp. Only hardware removed.

#include "effects.h"

// ── Include the shared libs directly (header-only or inline) ────
// We paste the LUTs and functions inline to avoid build-system complexity.

// -- fast_math (from festicorn/lib/fast_math/fast_math.h) --
static const int16_t sinLUT[256] = {
         0,   804,  1608,  2410,  3212,  4011,  4808,  5602,  6393,  7179,  7962,  8739,  9512, 10278, 11039, 11793,
     12539, 13279, 14010, 14732, 15446, 16151, 16846, 17530, 18204, 18868, 19519, 20159, 20787, 21403, 22005, 22594,
     23170, 23731, 24279, 24811, 25329, 25832, 26319, 26790, 27245, 27683, 28105, 28510, 28898, 29268, 29621, 29956,
     30273, 30571, 30852, 31113, 31356, 31580, 31785, 31971, 32137, 32285, 32412, 32521, 32609, 32678, 32728, 32757,
     32767, 32757, 32728, 32678, 32609, 32521, 32412, 32285, 32137, 31971, 31785, 31580, 31356, 31113, 30852, 30571,
     30273, 29956, 29621, 29268, 28898, 28510, 28105, 27683, 27245, 26790, 26319, 25832, 25329, 24811, 24279, 23731,
     23170, 22594, 22005, 21403, 20787, 20159, 19519, 18868, 18204, 17530, 16846, 16151, 15446, 14732, 14010, 13279,
     12539, 11793, 11039, 10278,  9512,  8739,  7962,  7179,  6393,  5602,  4808,  4011,  3212,  2410,  1608,   804,
         0,  -804, -1608, -2410, -3212, -4011, -4808, -5602, -6393, -7179, -7962, -8739, -9512,-10278,-11039,-11793,
    -12539,-13279,-14010,-14732,-15446,-16151,-16846,-17530,-18204,-18868,-19519,-20159,-20787,-21403,-22005,-22594,
    -23170,-23731,-24279,-24811,-25329,-25832,-26319,-26790,-27245,-27683,-28105,-28510,-28898,-29268,-29621,-29956,
    -30273,-30571,-30852,-31113,-31356,-31580,-31785,-31971,-32137,-32285,-32412,-32521,-32609,-32678,-32728,-32757,
    -32767,-32757,-32728,-32678,-32609,-32521,-32412,-32285,-32137,-31971,-31785,-31580,-31356,-31113,-30852,-30571,
    -30273,-29956,-29621,-29268,-28898,-28510,-28105,-27683,-27245,-26790,-26319,-25832,-25329,-24811,-24279,-23731,
    -23170,-22594,-22005,-21403,-20787,-20159,-19519,-18868,-18204,-17530,-16846,-16151,-15446,-14732,-14010,-13279,
    -12539,-11793,-11039,-10278, -9512, -8739, -7962, -7179, -6393, -5602, -4808, -4011, -3212, -2410, -1608,  -804,
};

static const uint16_t gammaLUT[256] = {
        0,    0,    1,    2,    3,    5,    8,   12,   16,   21,   28,   35,   43,   52,   62,   73,
       85,   99,  113,  129,  146,  164,  183,  204,  226,  249,  273,  299,  327,  355,  385,  417,
      450,  484,  520,  558,  597,  637,  680,  723,  769,  816,  864,  914,  966, 1020, 1075, 1132,
     1191, 1251, 1313, 1377, 1443, 1510, 1580, 1651, 1724, 1798, 1875, 1954, 2034, 2116, 2200, 2287,
     2375, 2465, 2557, 2651, 2747, 2845, 2945, 3046, 3150, 3257, 3365, 3475, 3587, 3701, 3818, 3936,
     4057, 4180, 4305, 4432, 4561, 4692, 4826, 4962, 5100, 5240, 5382, 5527, 5674, 5823, 5974, 6128,
     6284, 6442, 6603, 6766, 6931, 7098, 7268, 7440, 7615, 7792, 7971, 8153, 8337, 8523, 8712, 8903,
     9097, 9293, 9492, 9693, 9896,10102,10311,10522,10735,10951,11170,11391,11614,11840,12069,12300,
    12534,12770,13009,13250,13494,13741,13990,14242,14497,14754,15014,15276,15541,15809,16079,16352,
    16628,16907,17188,17472,17758,18048,18340,18635,18932,19233,19536,19841,20150,20461,20776,21093,
    21412,21735,22060,22389,22720,23054,23390,23730,24072,24418,24766,25117,25471,25828,26188,26550,
    26916,27284,27656,28030,28407,28788,29171,29557,29946,30338,30733,31131,31532,31936,32343,32753,
    33167,33583,34002,34424,34849,35277,35709,36143,36580,37021,37465,37911,38361,38814,39270,39729,
    40191,40656,41125,41596,42071,42549,43030,43514,44001,44492,44985,45482,45982,46486,46992,47502,
    48014,48531,49050,49572,50098,50627,51159,51695,52233,52775,53321,53869,54421,54976,55534,56096,
    56661,57229,57801,58376,58954,59535,60120,60709,61300,61895,62493,63095,63700,64308,64920,65535,
};

static inline float fastSin(float radians) {
    int32_t idx = (int32_t)(radians * 40.7436654f);
    return sinLUT[idx & 0xFF] * (1.0f / 32767.0f);
}

static inline float fastSinPhase(float phase01) {
    uint8_t idx = (uint8_t)(phase01 * 256.0f);
    return sinLUT[idx] * (1.0f / 32767.0f);
}

static inline float fastGamma24(float x) {
    if (x <= 0.0f) return 0.0f;
    int32_t idx = (int32_t)(x * 255.0f);
    if (idx > 255) idx = 255;
    return gammaLUT[idx] * (1.0f / 65535.0f);
}

static inline float fastDecay(float base, float exponent) {
    return 1.0f + exponent * (base - 1.0f);
}

// -- delta_sigma --
static inline uint8_t deltaSigma(uint16_t &accum, uint16_t target16) {
    accum += target16;
    uint8_t out = accum >> 8;
    accum &= 0xFF;
    return out;
}

// -- oklch_lut (variable-L only, that's what gravity particle uses) --
extern const uint8_t oklchVarL[256][3];

// ── Constants (copied from receiver.cpp) ────────────────────────

#define GAMMA 2.4f
#define BRIGHTNESS_CAP 0.10f
#define SPARKLE_BRIGHTNESS_CAP 0.50f
#define PURE_W_CEIL    0.10f
#define PURE_W_BLEND   0.15f

#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f
#define BLEND_RANGE_DEG 40.0f
#define SENSOR_HZ       25.0f

#define RMS_CEILING     5000.0f

// ── Adaptive floor (P4-guarded) ──────────────────────────────────
#define MIC_NOISE_FLOOR_MIN    50.0f
#define FLOOR_LEAK_RATE        0.005f
#define FLOOR_SNAP_EPSILON     0.05f
#define FLOOR_SNAP_CONSECUTIVE 3
#define FLOOR_SNAP_MIN_RATIO   0.4f
#define FLOOR_SOFT_SIGMA       0.6f
#define FLOOR_LONG_DRIFT       0.001f
#define FLOOR_HEADROOM         1.4f

#define SPARKLE_DEADBAND 0.08f

#define FIRE_FLICKER_SCALE  3.0f
#define FIRE_DEADBAND       0.08f
#define FIRE_DROPOUT_DEPTH  0.85f

#define BLOOM_BRIGHTNESS_CAP   0.25f
#define BLOOM_NOISE_GATE       256

#define SURPRISE_RATIO         3.0f
#define DRAIN_SCALE            100.0f
#define DRAIN_ENVELOPE_DECAY   0.85f
#define FLASH_MOTION_SCALE     300.0f
#define ENERGY_MULTIPLIER      1.4f
#define MOTION_SETTLE_MS       300

#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.65f
#define BLOOM_BREATH_MAX_PEAK   1.00f
#define BLOOM_BREATH_FLOOR      0.15f

#define BLOOM_FLASH_DECAY_LO   0.96f
#define BLOOM_FLASH_DECAY_HI   0.985f

#define BLOOM_RECOVERY_RAMP    0.033f
#define BLOOM_RECOVERY_SPREAD  0.70f

#define BLOOM_HUE_A_G   20.0f
#define BLOOM_HUE_A_B  100.0f
#define BLOOM_HUE_B_G   70.0f
#define BLOOM_HUE_B_B  110.0f
#define BLOOM_FLASH_G  150.0f
#define BLOOM_FLASH_B  170.0f
#define BLOOM_W_ONSET    0.5f

#define TIMEOUT_MS     500

#define GS_PARTICLE_COUNT     7
#define GS_GRAVITY_SCALE      40.0f
#define GS_VELOCITY_DAMP      0.92f
#define GS_BOUNCE_REBOUND     0.5f
#define GS_SPLAT_RADIUS       2.5f
#define GS_BRIGHTNESS_CAP     0.45f

// sparkle_simple uses the adaptive floor too

// ── Framebuffer (replaces Adafruit_NeoPixel) ────────────────────

static RgbwPixel framebuf[LED_COUNT];

static inline void setPixel(uint16_t i, uint8_t r, uint8_t g, uint8_t b, uint8_t w) {
    framebuf[i] = {r, g, b, w};
}

// ── Helpers ─────────────────────────────────────────────────────

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float vecLen(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}

static void vecNormalize(float &x, float &y, float &z) {
    float len = vecLen(x, y, z);
    if (len > 0) { x /= len; y /= len; z /= len; }
}

static uint32_t prngState;

static inline uint32_t xorshift32() {
    prngState ^= prngState << 13;
    prngState ^= prngState >> 17;
    prngState ^= prngState << 5;
    return prngState;
}

static inline float randFloat() {
    return (float)(xorshift32() & 0xFFFFFF) / 16777216.0f;
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// ── Signal processing ───────────────────────────────────────────

static float frrCeiling = RMS_CEILING;
static float adaptiveFloor = 0.0f;
static float longMin = 0.0f;
static int belowFloorCount = 0;
static float energy = 0.0f;
static float prevRms = 0.0f;
static float deltaPeak = 1e-6f;
static float onset = 0.0f;

static float computeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx*fx + fy*fy + fz*fz) / 131.0f;
}

static float computeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx*fx + fy*fy + fz*fz);
    return fabsf(mag - 16384.0f) / 16384.0f;
}

static float updateAdaptiveFloor(float rms, float dt) {
    if (adaptiveFloor < 1.0f) {
        adaptiveFloor = fmaxf(rms, MIC_NOISE_FLOOR_MIN);
        longMin = adaptiveFloor;
        return adaptiveFloor;
    }

    if (rms < longMin) {
        longMin = fmaxf(rms, MIC_NOISE_FLOOR_MIN);
    } else {
        longMin *= (1.0f + FLOOR_LONG_DRIFT * dt);
    }

    if (rms < adaptiveFloor * (1.0f + FLOOR_SNAP_EPSILON)) {
        belowFloorCount++;
        if (belowFloorCount >= FLOOR_SNAP_CONSECUTIVE) {
            float target = fmaxf(rms, FLOOR_SNAP_MIN_RATIO * longMin);
            target = fmaxf(target, MIC_NOISE_FLOOR_MIN);
            float snapAlpha = fminf(1.0f, dt / 0.11f);
            adaptiveFloor += snapAlpha * (target - adaptiveFloor);
        }
    } else {
        belowFloorCount = 0;
        float ratio = rms / fmaxf(adaptiveFloor, 1.0f);
        float d = (ratio - 1.0f) / FLOOR_SOFT_SIGMA;
        float weight = expf(-(d * d));
        adaptiveFloor *= (1.0f + FLOOR_LEAK_RATE * dt * weight);
    }

    adaptiveFloor = fmaxf(adaptiveFloor, MIC_NOISE_FLOOR_MIN);
    return adaptiveFloor;
}

static float computeEnergy(uint16_t rawRms, float dt) {
    float rms = (float)rawRms;
    frrCeiling = fmaxf(RMS_CEILING, frrCeiling * expf(-0.0025f * dt));
    if (rms > frrCeiling) frrCeiling = rms;
    updateAdaptiveFloor(rms, dt);
    float effectiveFloor = adaptiveFloor * FLOOR_HEADROOM;
    if (rms < effectiveFloor) return 0.0f;
    float db = 20.0f * log10f(rms / effectiveFloor);
    float dbRange = 20.0f * log10f(frrCeiling / effectiveFloor);
    if (dbRange < 1.0f) dbRange = 1.0f;
    return clampf(db / dbRange, 0.0f, 1.0f);
}

static float computeOnset(uint16_t rawRms, float dt) {
    float rms = (float)rawRms;
    float delta = fabsf(rms - prevRms);
    prevRms = rms;
    float decay = expf(-1.3f * dt);
    deltaPeak = fmaxf(delta, deltaPeak * decay);
    return (deltaPeak > 1e-6f) ? (delta / deltaPeak) : 0.0f;
}

static float motionRms = 0.0f;
static float motionRmsEma = 0.0f;
static float prevMotionRms = 0.0f;
static float motionDeltaPeak = 0.0f;

static void computeMotionEnergy(int16_t ax, int16_t ay, int16_t az,
                                 int16_t gx, int16_t gy, int16_t gz,
                                 float dt,
                                 float &outEnergy, float &outOnset) {
    float gyroRate = computeGyroRate(gx, gy, gz);
    float accelJolt = computeAccelJolt(ax, ay, az);
    float raw = fmaxf(gyroRate * 200.0f, accelJolt * 40000.0f);
    motionRms = raw;
    float emaAlpha = (raw > motionRmsEma) ? fminf(1.0f, dt / 0.11f)
                                          : fminf(1.0f, dt / 0.77f);
    motionRmsEma += emaAlpha * (raw - motionRmsEma);
    float ceiling = fmaxf(10000.0f, motionRmsEma * 3.0f);
    outEnergy = clampf(raw / ceiling, 0.0f, 1.0f);
    float delta = fabsf(raw - prevMotionRms);
    prevMotionRms = raw;
    motionDeltaPeak = fmaxf(delta, motionDeltaPeak * expf(-0.51f * dt));
    outOnset = (motionDeltaPeak > 1e-6f) ? (delta / motionDeltaPeak) : 0.0f;
}

// ── Effect state ────────────────────────────────────────────────

static Algorithm currentAlg = ALG_GRAVITY_PARTICLE;

// Delta-sigma accumulators
static uint16_t dsR[LED_COUNT], dsG[LED_COUNT], dsB[LED_COUNT], dsW[LED_COUNT];

// Calibration
static float restAx = 0, restAy = 0, restAz = 1.0f;
static bool calibrated = false;
static float calSumAx = 0, calSumAy = 0, calSumAz = 0;
static uint32_t calSamples = 0;
static uint32_t calStartMs = 0;
#define CAL_DURATION_MS 2000

// Gravity particle
struct GsParticle {
    float pos, vel, bright, hue;
};
static GsParticle gsParticles[GS_PARTICLE_COUNT];

// Sparkle
static float sparkle[LED_COUNT];
static float decayRates[LED_COUNT];
static float envelope = 0.0f;
static float cooldownRemaining = 0.0f;

// Fire
static float fireTime = 0.0f;
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

// Bloom
static float bloomBreathPhase[LED_COUNT];
static float bloomBreathPeriod[LED_COUNT];
static float bloomBreathPeak[LED_COUNT];
static float bloomHueT[LED_COUNT];
static float bloomFlashGlow[LED_COUNT];
static float bloomFlashDecay[LED_COUNT];
static float bloomColonyEnergy = 1.0f;
static uint32_t bloomLastMotionMs = 0;
static float bloomMotionRate = 0.0f;
static float motionEMA = 0.0f;
static float drainEnvelope = 0.0f;
static float bloomHitIntensity = 0.0f;
static uint32_t bloomPrevPktCount = 0;
static uint32_t pktCount = 0;

static SensorPacket latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

static uint8_t engageHueOffset = 0;

// ── Reset functions ─────────────────────────────────────────────

static void resetGravitySparkle() {
    for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
        gsParticles[i].pos = (float)(LED_COUNT - 1) * (float)i / (float)(GS_PARTICLE_COUNT - 1);
        gsParticles[i].vel = 0.0f;
        gsParticles[i].bright = 1.0f;
        gsParticles[i].hue = 256.0f * (float)i / (float)GS_PARTICLE_COUNT;
    }
}

static void resetSparkleState() {
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] = 0.0f;
        decayRates[i] = 0.92f + randFloat() * 0.05f;
    }
    envelope = 0.0f;
    cooldownRemaining = 0.0f;
}

static void resetFireState() {
    fireTime = 0.0f;
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

static void resetBloomState() {
    bloomColonyEnergy = 1.0f;
    bloomLastMotionMs = 0;
    bloomMotionRate = 0.0f;
    motionEMA = 0.0f;
    drainEnvelope = 0.0f;
    bloomHitIntensity = 0.0f;
    bloomPrevPktCount = 0;
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        bloomBreathPhase[i] = randFloat();
        bloomBreathPeriod[i] = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bloomBreathPeak[i] = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bloomHueT[i] = randFloat();
        bloomFlashGlow[i] = 0.0f;
        bloomFlashDecay[i] = BLOOM_FLASH_DECAY_LO
            + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
    }
}

// ── Calibration ─────────────────────────────────────────────────

static void updateCalibration(float ax, float ay, float az, uint32_t now) {
    if (calStartMs == 0) {
        calibrated = false;
        calSumAx = 0; calSumAy = 0; calSumAz = 0;
        calSamples = 0;
        calStartMs = now;
    }
    if (!calibrated) {
        calSumAx += ax; calSumAy += ay; calSumAz += az;
        calSamples++;
        if (now - calStartMs >= CAL_DURATION_MS && calSamples > 0) {
            restAx = calSumAx / calSamples;
            restAy = calSumAy / calSamples;
            restAz = calSumAz / calSamples;
            vecNormalize(restAx, restAy, restAz);
            calibrated = true;
        }
    }
}

// ── Effect renders (verbatim from receiver.cpp) ─────────────────

static void renderGravitySparkle(float dt) {
    float gravG = clampf((float)latestPacket.ax / 16384.0f, -1.5f, 1.5f);
    float accel = gravG * GS_GRAVITY_SCALE;
    float damp = fastDecay(GS_VELOCITY_DAMP, dt * 30.0f);

    static float accR[LED_COUNT], accG[LED_COUNT], accB[LED_COUNT];
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        accR[i] = 0; accG[i] = 0; accB[i] = 0;
    }

    const float maxPos = (float)(LED_COUNT - 1);
    const float invTwoSigSq = 1.0f / (2.0f * GS_SPLAT_RADIUS * GS_SPLAT_RADIUS);

    for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
        GsParticle &p = gsParticles[i];
        p.vel = p.vel * damp + accel * dt;
        p.pos += p.vel * dt;
        if (p.pos < 0.0f) {
            p.pos = 0.0f;
            if (p.vel < 0.0f) p.vel = -p.vel * GS_BOUNCE_REBOUND;
        } else if (p.pos > maxPos) {
            p.pos = maxPos;
            if (p.vel > 0.0f) p.vel = -p.vel * GS_BOUNCE_REBOUND;
        }
        uint8_t hueIdx = (uint8_t)((uint32_t)p.hue & 0xFF);
        float colR = (float)oklchVarL[hueIdx][0];
        float colG = (float)oklchVarL[hueIdx][1];
        float colB = (float)oklchVarL[hueIdx][2];
        int center = (int)(p.pos + 0.5f);
        int lo = center - 3; if (lo < 0) lo = 0;
        int hi = center + 3; if (hi > (int)(LED_COUNT - 1)) hi = LED_COUNT - 1;
        for (int j = lo; j <= hi; j++) {
            float d = (float)j - p.pos;
            float w = expf(-(d * d) * invTwoSigSq);
            accR[j] += colR * w;
            accG[j] += colG * w;
            accB[j] += colB * w;
        }
    }

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float r = accR[i];
        float g = accG[i];
        float b = accB[i];
        float maxCh = fmaxf(r, fmaxf(g, b));
        float bright = clampf(maxCh / 255.0f, 0.0f, 1.0f);
        float linBright = fastGamma24(bright) * GS_BRIGHTNESS_CAP;
        float norm = (bright > 0.001f) ? (linBright / bright) : 0.0f;
        uint8_t rr = (uint8_t)clampf(r * norm + 0.5f, 0, 255);
        uint8_t gg = (uint8_t)clampf(g * norm + 0.5f, 0, 255);
        uint8_t bb = (uint8_t)clampf(b * norm + 0.5f, 0, 255);
        setPixel(i, rr, gg, bb, 0);
    }
}

static void renderSparkleBurst(float dt, float angleDeg, float tiltBlend,
                                float tiltR, float tiltG, float tiltB) {
    bool isSilent = energy < 0.001f;
    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > envelope)
        envelope += attackAlpha * (energy - envelope);
    else
        envelope += decayAlpha * (energy - envelope);
    cooldownRemaining = fmaxf(0.0f, cooldownRemaining - dt);
    float onsetThreshold = fmaxf(0.15f, 0.4f - envelope * 0.3f);
    if (onset > onsetThreshold && cooldownRemaining <= 0.0f && !isSilent) {
        cooldownRemaining = fmaxf(0.050f, 0.150f - envelope * 0.10f);
        float onsetStrength = clampf(onset, 0.0f, 1.0f);
        int nIgnite = (int)(LED_COUNT * (0.3f + 0.2f * onsetStrength));
        static uint8_t indices[LED_COUNT];
        for (uint8_t i = 0; i < LED_COUNT; i++) indices[i] = i;
        for (int i = 0; i < nIgnite; i++) {
            int j = i + (int)(xorshift32() % (LED_COUNT - i));
            uint8_t tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
        float sparkVal = 0.7f + 0.3f * onsetStrength;
        for (int i = 0; i < nIgnite; i++) {
            sparkle[indices[i]] = sparkVal;
            decayRates[indices[i]] = 0.92f + randFloat() * 0.05f;
        }
    }
    for (uint16_t i = 0; i < LED_COUNT; i++)
        sparkle[i] *= fastDecay(decayRates[i], dt * 30.0f);
    if (!isSilent) {
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float jitter = (randFloat() - 0.5f) * 0.02f;
            float newVal = sparkle[i] + jitter;
            if (newVal < 0.0f) newVal = 0.0f;
            if (newVal > sparkle[i] && jitter > 0) newVal = sparkle[i];
            sparkle[i] = newVal;
        }
    }
    float base = fminf(envelope, 0.2f);
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float s = sparkle[i];
        float bright = base + s * (1.0f - base);
        if (bright < SPARKLE_DEADBAND) bright = 0.0f;
        float colR = 255.0f;
        float colG = 180.0f + (240.0f - 180.0f) * s;
        float colB =  80.0f + (200.0f -  80.0f) * s;
        if (tiltBlend > 0.0f) {
            colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
            colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
            colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
        }
        float linBright = fastGamma24(bright) * SPARKLE_BRIGHTNESS_CAP;
        float colW = 255.0f * (1.0f - tiltBlend);
        float fR = colR * linBright;
        float fG = colG * linBright;
        float fB = colB * linBright;
        float fW = colW * linBright;
        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);
        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;
        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        setPixel(i, r, g, b, w);
    }
}

// ── Sparkle syllable (onset injected via simInjectOnset) ─────────

static float syllSparkle[LED_COUNT];
static float syllDecay[LED_COUNT];
static float syllEnvelope = 0.0f;
static float syllCooldown = 0.0f;
static uint8_t syllPendingStrength = 0;
static bool syllHasOnset = false;

static void resetSyllableState() {
    memset(syllSparkle, 0, sizeof(syllSparkle));
    syllEnvelope = 0.0f;
    syllCooldown = 0.0f;
    syllPendingStrength = 0;
    syllHasOnset = false;
}

void simInjectOnset(uint8_t strength, uint8_t /*band*/) {
    syllPendingStrength = strength;
    syllHasOnset = true;
}

static void renderSparkleSyllable(float dt, float angleDeg, float tiltBlend,
                                   float tiltR, float tiltG, float tiltB) {
    float onsetNorm = syllHasOnset ? (syllPendingStrength / 255.0f) : 0.0f;
    syllHasOnset = false;

    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > syllEnvelope)
        syllEnvelope += attackAlpha * (energy - syllEnvelope);
    else
        syllEnvelope += decayAlpha * (energy - syllEnvelope);

    syllCooldown = fmaxf(0.0f, syllCooldown - dt);

    if (onsetNorm > 0.1f && syllCooldown <= 0.0f) {
        syllCooldown = 0.060f;
        int nIgnite = (int)(LED_COUNT * (0.25f + 0.25f * onsetNorm));
        static uint8_t indices[LED_COUNT];
        for (uint8_t i = 0; i < LED_COUNT; i++) indices[i] = i;
        for (int i = 0; i < nIgnite; i++) {
            int j = i + (int)(xorshift32() % (LED_COUNT - i));
            uint8_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        float sparkVal = 0.6f + 0.4f * onsetNorm;
        for (int i = 0; i < nIgnite; i++) {
            syllSparkle[indices[i]] = sparkVal;
            syllDecay[indices[i]] = 0.92f + randFloat() * 0.05f;
        }
    }

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        syllSparkle[i] *= fastDecay(syllDecay[i], dt * 30.0f);
    }

    float base = fminf(syllEnvelope, 0.15f);

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float s = syllSparkle[i];
        float bright = base + s * (1.0f - base);
        if (bright < SPARKLE_DEADBAND) bright = 0.0f;

        float colR = 255.0f;
        float colG = 180.0f + (240.0f - 180.0f) * s;
        float colB =  80.0f + (200.0f -  80.0f) * s;

        if (tiltBlend > 0.0f) {
            colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
            colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
            colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
        }

        float linBright = fastGamma24(bright) * SPARKLE_BRIGHTNESS_CAP;
        float colW = 255.0f * (1.0f - tiltBlend);
        float fR = colR * linBright;
        float fG = colG * linBright;
        float fB = colB * linBright;
        float fW = colW * linBright;

        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);

        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        setPixel(i, r, g, b, w);
    }
}

static void renderFire(float dt, bool withDropout, float tiltBlend) {
    fireTime = fmodf(fireTime + dt, 6283.1853f);
    float t = fireTime;
    bool isSilent = energy < 0.001f;
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);
    float attackAlpha = fminf(1.0f, dt / 0.050f);
    float decayAlpha  = fminf(1.0f, dt / 2.0f);
    float targetBrightness;
    if (isSilent) targetBrightness = 0.25f;
    else targetBrightness = fmaxf(0.25f, energy);
    if (targetBrightness > fireBaseBrightness)
        fireBaseBrightness += attackAlpha * (targetBrightness - fireBaseBrightness);
    else
        fireBaseBrightness += decayAlpha * (targetBrightness - fireBaseBrightness);
    float flickerAlpha = fminf(1.0f, dt / 0.200f);
    float deltaTarget = isSilent ? 0.0f : onset;
    fireFlickerIntensity += flickerAlpha * (deltaTarget - fireFlickerIntensity);
    float dropoutAmount = 0.0f;
    if (withDropout) {
        float energyDeriv = (energy - firePrevEnergyForDeriv) / fmaxf(dt, 0.001f);
        firePrevEnergyForDeriv = energy;
        float derivAlpha = fminf(1.0f, dt / 0.200f);
        fireEnergyDerivSmooth += derivAlpha * (energyDeriv - fireEnergyDerivSmooth);
        bool isSustaining = (!isSilent && energy > 0.05f
                             && fabsf(fireEnergyDerivSmooth) <= 0.5f);
        if (isSustaining)
            fireDropoutAmount = fminf(1.0f, fireDropoutAmount + dt * 0.35f);
        else
            fireDropoutAmount = fmaxf(0.0f, fireDropoutAmount - dt * 1.0f);
        dropoutAmount = fireDropoutAmount;
    }
    float colorAttack = fminf(1.0f, dt / 0.080f);
    float colorDecay  = fminf(1.0f, dt / 2.0f);
    float colorTarget;
    if (isPercussiveOnly || isSilent) colorTarget = 0.0f;
    else colorTarget = energy;
    if (colorTarget > fireColorEnergy)
        fireColorEnergy += colorAttack * (colorTarget - fireColorEnergy);
    else
        fireColorEnergy += colorDecay * (colorTarget - fireColorEnergy);
    float ce = fireColorEnergy;
    float baseColR, baseColG, baseColB;
    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;
    const float amberR = 255.0f, amberG = 140.0f, amberB = 30.0f;
    const float redR   = 200.0f, redG   =  20.0f, redB   =  0.0f;
    const float whiteR = 180.0f, whiteG = 170.0f, whiteB = 160.0f;
    if (ce < WHITE_BLEND_THRESHOLD) {
        float tw = 1.0f - (ce / WHITE_BLEND_THRESHOLD);
        baseColR = amberR * (1.0f - tw) + whiteR * tw;
        baseColG = amberG * (1.0f - tw) + whiteG * tw;
        baseColB = amberB * (1.0f - tw) + whiteB * tw;
    } else {
        float tr = (ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD);
        if (tr > 1.0f) tr = 1.0f;
        baseColR = amberR * (1.0f - tr) + redR * tr;
        baseColG = amberG * (1.0f - tr) + redG * tr;
        baseColB = amberB * (1.0f - tr) + redB * tr;
    }
    float baseBr = fireBaseBrightness;
    if (baseBr < FIRE_DEADBAND) baseBr = 0.0f;
    float s = FIRE_FLICKER_SCALE;
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float fi = (float)i;
        float noise = fastSin(fi * 7.3f + t * 2.5f) *
                       fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;
        float noiseAmp = fmaxf(0.15f * s, 0.10f * s / fmaxf(baseBr, 0.1f));
        float bright = baseBr * (1.0f + noiseAmp * (noise - 0.5f))
                        + fireFlickerIntensity * (noise - 0.5f) * 0.25f * s;
        float perLedDim = 0.0f;
        float colorRedShift = 0.0f;
        if (withDropout && dropoutAmount > 0.0f) {
            float resilience = fastSin(fi * 13.7f + t * 0.3f) *
                                fastSin(fi * 9.1f + t * 0.2f) * 0.5f + 0.5f;
            perLedDim = clampf(
                (dropoutAmount - resilience * 0.7f) / 0.3f, 0.0f, 1.0f
            ) * FIRE_DROPOUT_DEPTH;
            colorRedShift = clampf(perLedDim / 0.3f, 0.0f, 1.0f);
        }
        bright *= (1.0f - perLedDim);
        bright = clampf(bright, 0.0f, 1.0f);
        float colR = baseColR * (1.0f - colorRedShift) + redR * colorRedShift;
        float colG = baseColG * (1.0f - colorRedShift) + redG * colorRedShift;
        float colB = baseColB * (1.0f - colorRedShift) + redB * colorRedShift;
        float linBright = fastGamma24(bright) * BRIGHTNESS_CAP;
        float oR = colR * linBright;
        float oG = colG * linBright;
        float oB = colB * linBright;
        float maxCh_f = fmaxf(oR, fmaxf(oG, oB));
        float bFrac = maxCh_f / 255.0f;
        float rgbBlend = clampf((bFrac - PURE_W_CEIL) / PURE_W_BLEND, 0.0f, 1.0f);
        float avgRGB = (oR + oG + oB) / 3.0f;
        float fR = oR * rgbBlend;
        float fG = oG * rgbBlend;
        float fB = oB * rgbBlend;
        float fW = avgRGB * (1.0f - rgbBlend) * (1.0f - tiltBlend);
        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);
        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;
        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        setPixel(i, r, g, b, w);
    }
}

static void bloomProcessMotion(float pktDt, uint32_t now) {
    float gyroRate = computeGyroRate(latestPacket.gx, latestPacket.gy, latestPacket.gz);
    float accelJolt = computeAccelJolt(latestPacket.ax, latestPacket.ay, latestPacket.az);
    bloomMotionRate = fmaxf(gyroRate, accelJolt * 300.0f);
    float surprise = fmaxf(0.0f, bloomMotionRate - motionEMA * SURPRISE_RATIO);
    float alpha = (bloomMotionRate > motionEMA) ? fminf(1.0f, pktDt / 0.77f)
                                                : fminf(1.0f, pktDt / 0.16f);
    motionEMA += alpha * (bloomMotionRate - motionEMA);
    if (surprise > 1.0f) {
        bloomLastMotionMs = now;
        float hitIntensity = clampf(
            log2f(1.0f + surprise) / log2f(1.0f + FLASH_MOTION_SCALE),
            0.0f, 1.0f);
        if (hitIntensity > bloomHitIntensity) bloomHitIntensity = hitIntensity;
        float normMotion = surprise / DRAIN_SCALE;
        float newDrain = normMotion * normMotion * normMotion
                       * (1.0f - DRAIN_ENVELOPE_DECAY);
        if (newDrain > drainEnvelope) drainEnvelope = newDrain;
    }
}

static void renderQuietBloom(float dt, uint32_t now) {
    bool draining = drainEnvelope > 0.001f;
    if (!draining) bloomHitIntensity = 0.0f;
    if (now - bloomLastMotionMs > MOTION_SETTLE_MS)
        bloomColonyEnergy = fminf(1.0f, bloomColonyEnergy + BLOOM_RECOVERY_RAMP * dt);
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        if (!draining) {
            bloomFlashGlow[i] *= fastDecay(bloomFlashDecay[i], dt * 30.0f);
            if (bloomFlashGlow[i] < 0.005f) bloomFlashGlow[i] = 0.0f;
        }
        bloomBreathPhase[i] += dt / bloomBreathPeriod[i];
        if (bloomBreathPhase[i] >= 1.0f) bloomBreathPhase[i] -= 1.0f;
        float wakeThresh = bloomHueT[i] * BLOOM_RECOVERY_SPREAD;
        float ledRecovery = clampf(
            (bloomColonyEnergy - wakeThresh) / 0.30f, 0.0f, 1.0f);
        float breath = (fastSinPhase(bloomBreathPhase[i]) * 0.5f + 0.5f);
        float breathGlow = BLOOM_BREATH_FLOOR
            + breath * (bloomBreathPeak[i] - BLOOM_BREATH_FLOOR);
        breathGlow *= ledRecovery;
        if (draining) {
            float target = breathGlow * bloomHitIntensity * ENERGY_MULTIPLIER;
            if (target > bloomFlashGlow[i]) {
                bloomFlashGlow[i] = target;
                bloomFlashDecay[i] = BLOOM_FLASH_DECAY_LO
                    + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
            }
        }
        float g = fmaxf(breathGlow, bloomFlashGlow[i]);
        float flashFrac = (bloomFlashGlow[i] > breathGlow) ? 1.0f : 0.0f;
        float h = bloomHueT[i];
        float colG = lerpf(lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h),
                           BLOOM_FLASH_G, flashFrac);
        float colB = lerpf(lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h),
                           BLOOM_FLASH_B, flashFrac);
        float linBright = fastGamma24(g) * BLOOM_BRIGHTNESS_CAP;
        float oG = colG * linBright;
        float oB = colB * linBright;
        float wFrac = clampf((g - BLOOM_W_ONSET) / (1.0f - BLOOM_W_ONSET),
                             0.0f, 1.0f);
        float energyGate = clampf((bloomColonyEnergy - 0.7f) / 0.3f, 0.0f, 1.0f);
        float wGate = fmaxf(energyGate, flashFrac);
        float oW = wFrac * wGate * linBright * 200.0f;
        uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);
        if (tG16 < BLOOM_NOISE_GATE) tG16 = 0;
        if (tB16 < BLOOM_NOISE_GATE) tB16 = 0;
        if (tW16 < BLOOM_NOISE_GATE) tW16 = 0;
        uint8_t gc = deltaSigma(dsG[i], tG16);
        uint8_t b  = deltaSigma(dsB[i], tB16);
        uint8_t w  = deltaSigma(dsW[i], tW16);
        setPixel(i, 0, gc, b, w);
    }
    if (draining) {
        float drain = drainEnvelope * dt;
        drain = fminf(drain, bloomColonyEnergy);
        bloomColonyEnergy -= drain;
        drainEnvelope *= expf(-4.07f * dt);
        if (drainEnvelope <= 0.001f) drainEnvelope = 0.0f;
    }
}

// ── Public API ──────────────────────────────────────────────────

void simInit(uint32_t seed) {
    prngState = seed ? seed : 1;
    memset(framebuf, 0, sizeof(framebuf));
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint16_t s = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
        dsR[i] = s;
        dsG[i] = (s + 64) & 0xFF;
        dsB[i] = (s + 128) & 0xFF;
        dsW[i] = (s + 192) & 0xFF;
    }
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] = 0.0f;
        decayRates[i] = 0.92f + randFloat() * 0.05f;
    }
    resetBloomState();
    resetGravitySparkle();
    calStartMs = 0;
    calibrated = false;
    pktCount = 0;
    latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};
}

void simSetAlgorithm(Algorithm alg) {
    currentAlg = alg;
    switch (alg) {
        case ALG_SPARKLE_BURST:
            resetSparkleState(); break;
        case ALG_FIRE_MELD:
        case ALG_FIRE_FLICKER:
            resetFireState(); break;
        case ALG_QUIET_BLOOM:
            resetBloomState(); break;
        case ALG_GRAVITY_PARTICLE:
            resetGravitySparkle(); break;
        case ALG_SPARKLE_SYLLABLE:
            resetSyllableState(); break;
    }
}

void simStep(const SensorPacket &pkt, float dt, uint32_t nowMs) {
    latestPacket = pkt;
    pktCount++;

    float ax = (float)pkt.ax;
    float ay = (float)pkt.ay;
    float az = (float)pkt.az;
    vecNormalize(ax, ay, az);
    uint16_t rawRms = pkt.rawRms;
    bool micOn = pkt.micEnabled != 0;

    updateCalibration(ax, ay, az, nowMs);

    float angleDeg = 0;
    float dot = restAx * ax + restAy * ay + restAz * az;
    if (calibrated) {
        angleDeg = acosf(clampf(dot, -1.0f, 1.0f)) * (180.0f / M_PI);
    }

    if (currentAlg != ALG_QUIET_BLOOM) {
        if (micOn) {
            energy = computeEnergy(rawRms, dt);
            onset = computeOnset(rawRms, dt);
        } else {
            computeMotionEnergy(pkt.ax, pkt.ay, pkt.az,
                                pkt.gx, pkt.gy, pkt.gz,
                                dt, energy, onset);
        }
    }

    if (currentAlg == ALG_QUIET_BLOOM) {
        bloomProcessMotion(dt, nowMs);
    }

    float tiltR = 0, tiltG = 0, tiltB = 0;
    float tiltBlend = 0.0f;
    if (angleDeg > DEADZONE_DEG) {
        float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
        if (hueFrac > 1.0f) hueFrac = 1.0f;
        uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255) + engageHueOffset) & 0xFF);
        tiltR = (float)oklchVarL[hueIdx][0];
        tiltG = (float)oklchVarL[hueIdx][1];
        tiltB = (float)oklchVarL[hueIdx][2];
        tiltBlend = (angleDeg - DEADZONE_DEG) / BLEND_RANGE_DEG;
        if (tiltBlend > 1.0f) tiltBlend = 1.0f;
    }

    switch (currentAlg) {
        case ALG_SPARKLE_BURST:
            renderSparkleBurst(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB); break;
        case ALG_FIRE_MELD:
            renderFire(dt, false, tiltBlend); break;
        case ALG_FIRE_FLICKER:
            renderFire(dt, true, tiltBlend); break;
        case ALG_QUIET_BLOOM:
            renderQuietBloom(dt, nowMs); break;
        case ALG_GRAVITY_PARTICLE:
            renderGravitySparkle(dt); break;
        case ALG_SPARKLE_SYLLABLE:
            renderSparkleSyllable(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB); break;
    }
}

Algorithm simGetAlgorithm() { return currentAlg; }
const RgbwPixel* simGetFramebuffer() { return framebuf; }
