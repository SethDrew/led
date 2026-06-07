// Host-side simulation of the tree-of-record bloom render + delta-sigma
// dither pipeline. Reproduces renderBloomStrip() exactly (ambient-only,
// uniform brightness cap) and emits per-frame, per-pixel CSV telemetry so
// the low-brightness flicker behaviour can be analysed off-hardware.
//
// Source of truth: festicorn/tree-of-record/src/main.cpp (renderBloomStrip).
// Shared libs included directly from festicorn/lib/.
//
// Build:  make
// Run:    ./bloom_dither_sim --brightness 0.15 --policy current > out.csv

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

#include "fast_math.h"     // fastSinPhase, fastGamma24  (../../../lib/fast_math)
#include "delta_sigma.h"   // deltaSigma                 (../../../lib/delta_sigma)

// ── Geometry / timing ────────────────────────────────────────────
#define LEDS_PER_STRIP 100
static const float SIM_DT  = 1.0f / 200.0f;   // 200 fps
static int   SIM_FRAMES    = 2000;            // 10 s
static uint32_t SIM_SEED   = 0x1234567u;

// ── Gate policies ────────────────────────────────────────────────
enum GatePolicy { POLICY_CURRENT, POLICY_PER_CHANNEL_SNAP, POLICY_RAISED_FLOOR };
static GatePolicy gPolicy = POLICY_CURRENT;

// ── Bloom constants (verbatim from main.cpp) ─────────────────────
static float bloomBrightnessCap   = 0.15f;
static float bloomBufferDrain     = 15.0f;
static float bloomFlashDecayRate  = 3.0f;
static float bloomBreathFloor     = 0.15f;

#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.65f
#define BLOOM_BREATH_MAX_PEAK   1.00f

#define BLOOM_FLASH_R  200.0f
#define BLOOM_FLASH_G  120.0f
#define BLOOM_FLASH_B  255.0f

#define BLOOM_HUE_DRIFT_MIN  (1.0f / 45.0f)
#define BLOOM_HUE_DRIFT_MAX  (1.0f / 15.0f)

static float bloomDormancyMin  = 3.0f;
static float bloomDormancyMax  = 7.0f;
static float bloomDormancyFrac = 0.0f;

#define GATE_ON_THRESH  0.7f
#define GATE_OFF_THRESH 1.2f

#define BLOOM_HUE_A_R    0.0f
#define BLOOM_HUE_A_G  180.0f
#define BLOOM_HUE_A_B  120.0f
#define BLOOM_HUE_B_R  140.0f
#define BLOOM_HUE_B_G   20.0f
#define BLOOM_HUE_B_B  255.0f

static float renderBrightness = 1.0f;   // global brightness scale (the sweep knob)

// ── PRNG (verbatim) ──────────────────────────────────────────────
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
static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// ── Strip state (verbatim) ───────────────────────────────────────
struct BloomStrip {
    float breathPhase[LEDS_PER_STRIP];
    float breathPeriod[LEDS_PER_STRIP];
    float breathPeak[LEDS_PER_STRIP];
    float hueT[LEDS_PER_STRIP];
    float hueDrift[LEDS_PER_STRIP];
    float blackoutTimer[LEDS_PER_STRIP];
    float dormancyDur[LEDS_PER_STRIP];
    float dormancyRoll[LEDS_PER_STRIP];
    bool  gateOff[LEDS_PER_STRIP];
    uint16_t ditherR[LEDS_PER_STRIP];
    uint16_t ditherG[LEDS_PER_STRIP];
    uint16_t ditherB[LEDS_PER_STRIP];
    float flash;
    float energyBuffer;
};

static BloomStrip bloom;

static void resetBloomStrip(BloomStrip &bs) {
    bs.flash = 0.0f;
    bs.energyBuffer = 0.0f;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        bs.breathPhase[i]  = randFloat();
        bs.breathPeriod[i] = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bs.breathPeak[i]   = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bs.hueT[i]         = randFloat();
        float rate = BLOOM_HUE_DRIFT_MIN
            + randFloat() * (BLOOM_HUE_DRIFT_MAX - BLOOM_HUE_DRIFT_MIN);
        bs.hueDrift[i]     = (randFloat() > 0.5f) ? rate : -rate;
        bs.blackoutTimer[i] = 0.0f;
        bs.dormancyDur[i]   = bloomDormancyMin + randFloat() * (bloomDormancyMax - bloomDormancyMin);
        bs.dormancyRoll[i]  = randFloat();
        bs.gateOff[i]      = false;
        bs.ditherR[i] = 0;
        bs.ditherG[i] = 0;
        bs.ditherB[i] = 0;
    }
}

// Apply the selected gate policy's sub-LSB handling to one channel target.
static inline uint16_t applyPolicy(uint16_t t16) {
    switch (gPolicy) {
        case POLICY_PER_CHANNEL_SNAP:
            return (t16 < 256) ? 0 : t16;
        case POLICY_RAISED_FLOOR:
            return (t16 > 0 && t16 < 256) ? 256 : t16;
        case POLICY_CURRENT:
        default:
            return t16;
    }
}

// Render one frame, emitting a CSV row per pixel.
static void renderBloomFrame(BloomStrip &bs, int frame, float dt) {
    float drain = bs.energyBuffer * bloomBufferDrain * dt;
    if (bs.energyBuffer < 0.05f) drain += 0.1f * dt;
    drain = fminf(drain, bs.energyBuffer);
    bs.energyBuffer -= drain;
    bs.flash = fminf(1.0f, bs.flash + drain);

    bs.flash *= expf(-bloomFlashDecayRate * dt);
    if (bs.flash < 0.005f) bs.flash = 0.0f;

    float flashLin = bs.flash * bloomBrightnessCap;
    float flashFrac = (bs.flash > 0.1f) ? clampf(bs.flash / 0.3f, 0.0f, 1.0f) : 0.0f;
    float oneMinusFF = 1.0f - flashFrac;

    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        float breath = fastSinPhase(bs.breathPhase[i]) * 0.5f + 0.5f;
        float breathGlow = bloomBreathFloor + breath * (bs.breathPeak[i] - bloomBreathFloor);
        float breathLin = fastGamma24(breathGlow) * bloomBrightnessCap;

        bs.breathPhase[i] += dt / bs.breathPeriod[i];
        if (bs.breathPhase[i] >= 1.0f) bs.breathPhase[i] -= 1.0f;
        bs.hueT[i] += bs.hueDrift[i] * dt;
        if (bs.hueT[i] > 1.0f) bs.hueT[i] -= 1.0f;
        else if (bs.hueT[i] < 0.0f) bs.hueT[i] += 1.0f;

        float h = bs.hueT[i];
        float baseR = lerpf(BLOOM_HUE_A_R, BLOOM_HUE_B_R, h);
        float baseG = lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h);
        float baseB = lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h);

        float oR = baseR * breathLin + (baseR * oneMinusFF + BLOOM_FLASH_R * flashFrac) * flashLin;
        float oG = baseG * breathLin + (baseG * oneMinusFF + BLOOM_FLASH_G * flashFrac) * flashLin;
        float oB = baseB * breathLin + (baseB * oneMinusFF + BLOOM_FLASH_B * flashFrac) * flashLin;

        // Hysteresis noise gate with dormancy hold
        float maxCh = fmaxf(fmaxf(oR, oG), oB);
        float thresh = bs.gateOff[i] ? GATE_OFF_THRESH : GATE_ON_THRESH;
        if (maxCh < thresh) {
            oR = oG = oB = 0.0f;
            if (!bs.gateOff[i] && bs.dormancyRoll[i] < bloomDormancyFrac) {
                bs.gateOff[i] = true;
                bs.blackoutTimer[i] = 0.0f;
                bs.dormancyDur[i] = bloomDormancyMin
                    + randFloat() * (bloomDormancyMax - bloomDormancyMin);
            }
        } else if (bs.gateOff[i]) {
            bs.blackoutTimer[i] += dt;
            if (bs.blackoutTimer[i] < bs.dormancyDur[i]) {
                oR = oG = oB = 0.0f;
            } else {
                bs.gateOff[i] = false;
                bs.blackoutTimer[i] = 0.0f;
            }
        }

        oR *= renderBrightness;
        oG *= renderBrightness;
        oB *= renderBrightness;

        uint16_t t16R = (uint16_t)fminf(oR * 256.0f, 65535.0f);
        uint16_t t16G = (uint16_t)fminf(oG * 256.0f, 65535.0f);
        uint16_t t16B = (uint16_t)fminf(oB * 256.0f, 65535.0f);

        t16R = applyPolicy(t16R);
        t16G = applyPolicy(t16G);
        t16B = applyPolicy(t16B);

        if ((t16R | t16G | t16B) == 0) {
            bs.ditherR[i] = bs.ditherG[i] = bs.ditherB[i] = 0;
        }
        uint8_t r8 = deltaSigma(bs.ditherR[i], t16R);
        uint8_t g8 = deltaSigma(bs.ditherG[i], t16G);
        uint8_t b8 = deltaSigma(bs.ditherB[i], t16B);

        printf("%d,%u,%.5f,%.5f,%.5f,%d,%u,%u,%u,%u,%u,%u\n",
               frame, i, oR, oG, oB, bs.gateOff[i] ? 1 : 0,
               t16R, t16G, t16B, r8, g8, b8);
    }
}

int main(int argc, char **argv) {
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--brightness") && a + 1 < argc) {
            renderBrightness = atof(argv[++a]);
        } else if (!strcmp(argv[a], "--policy") && a + 1 < argc) {
            const char *p = argv[++a];
            if      (!strcmp(p, "current"))     gPolicy = POLICY_CURRENT;
            else if (!strcmp(p, "snap"))        gPolicy = POLICY_PER_CHANNEL_SNAP;
            else if (!strcmp(p, "floor"))       gPolicy = POLICY_RAISED_FLOOR;
            else { fprintf(stderr, "unknown policy: %s\n", p); return 1; }
        } else if (!strcmp(argv[a], "--frames") && a + 1 < argc) {
            SIM_FRAMES = atoi(argv[++a]);
        } else if (!strcmp(argv[a], "--seed") && a + 1 < argc) {
            SIM_SEED = (uint32_t)strtoul(argv[++a], nullptr, 0);
        } else {
            fprintf(stderr, "usage: %s [--brightness F] [--policy current|snap|floor] "
                            "[--frames N] [--seed S]\n", argv[0]);
            return 1;
        }
    }

    prngState = SIM_SEED;
    if (prngState == 0) prngState = 1;
    resetBloomStrip(bloom);

    printf("frame,pixel,oR,oG,oB,gateOff,t16R,t16G,t16B,r8,g8,b8\n");
    for (int f = 0; f < SIM_FRAMES; f++) {
        renderBloomFrame(bloom, f, SIM_DT);
    }
    return 0;
}
