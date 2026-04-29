/*
 * BIOLUM BLOOM RECEIVER — classic ESP32, 6 WS2812B strips, 12 colonies
 *
 * Bioluminescence/quiet_bloom + tap-driven wave_pulse, ported from
 * rgb_bulbs/bulb_receiver.cpp (v1 receiver) onto biolum's 6-strip hardware.
 * Each strip is split in half → 2 colonies per pin × 6 pins = 12 colonies total.
 * All colonies receive the same motion input; each has independent drain/recovery.
 *
 * Receives v1 16-byte TelemetryPacketV1 via ESP-NOW from the festicorn handheld
 * sender (festicorn/sender_rnd/src/sender.cpp). Wire schema documented there
 * and in V1_SCHEMA_DERIVATION.md. Inverse companding:
 *   counts = (byte/255)² × 57000
 *   g      = counts / 8192   (±4g rail)
 *   dps    = counts / 32.8   (±1000 dps rail)
 *
 * Algorithms (selected via serial command):
 *   'b' — quiet_bloom (default)
 *   'w' — wave_pulse  (taps spawn leaves at LED 0 on every strip in parallel)
 *   'c' — recalibrate rest vector
 *
 * Wiring (NEO_RGB):
 *   GPIO 4, 16, 17, 5, 18, 19 — 100 LEDs each (first 50 = colony 2s, next 50 = colony 2s+1)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <esp_now.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <delta_sigma.h>
#include <fast_math.h>
#include <oklch_lut.h>
#include "wifi_credentials_local.h"

// ── Strip layout ─────────────────────────────────────────────────
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 30
#endif

static const uint8_t  STRIP_PINS[]       = { 4, 16, 17, 5, 18, 19 };
static const uint8_t  NUM_STRIPS         = sizeof(STRIP_PINS) / sizeof(STRIP_PINS[0]);
static const uint8_t  COLONIES_PER_STRIP = 2;
static const uint8_t  NUM_COLONIES       = NUM_STRIPS * COLONIES_PER_STRIP;
static const uint16_t LEDS_PER_COLONY    = LEDS_PER_STRIP / COLONIES_PER_STRIP;
static const uint16_t TOTAL_LEDS         = (uint16_t)NUM_STRIPS * LEDS_PER_STRIP;

// LED → colony mapping: strip s, local li → colony (s * 2) + (li >= LEDS_PER_COLONY ? 1 : 0)
static inline uint8_t ledColony(uint8_t s, uint16_t li) {
    return (uint8_t)(s * COLONIES_PER_STRIP + (li / LEDS_PER_COLONY));
}

// ── Rendering / tilt ─────────────────────────────────────────────
#define BRIGHTNESS_CAP    0.30f
#define DEADZONE_DEG     10.0f
#define MAX_ANGLE_DEG   180.0f
#define SENSOR_HZ        25.0f

// ── v1 telemetry decode constants ────────────────────────────────
#define MAG_FS              57000.0f
#define COUNTS_PER_G         8192.0f
#define COUNTS_PER_DPS         32.8f
#define COUNTS_PER_INT8       256.0f

// ── Bloom parameters (lifted from bulb_receiver) ────────────────
#define BLOOM_BRIGHTNESS_CAP   0.70f
#define BLOOM_NOISE_GATE       256

#define SURPRISE_EMA_UP        0.05f
#define SURPRISE_EMA_DOWN      0.2f
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
#define BLOOM_PURPLE_MAX  60.0f
#define BLOOM_PURPLE_RATE 0.15f

// ── Wave pulse parameters (per-strip; identical math to rgb_bulbs) ──
#define WP_MAX_LEAVES       16            // per strip
#define WP_WIND_SPEED       6.0f
#define WP_TURBULENCE       0.5f
#define WP_DAMPING          0.92f
#define WP_BOOST_SPEED      28.0f
#define WP_BOOST_TC         1.5f
#define WP_LEAF_SIGMA       1.5f
#define WP_FADE_IN_TIME     0.20f
#define WP_FADE_OUT_LEDS    8.0f
#define WP_MAX_BRIGHTNESS   0.85f
#define WP_STALL_RADIUS     2.5f
#define WP_STALL_TIMEOUT    3.0f
#define WP_DEFAULT_R       50.0f
#define WP_DEFAULT_G      200.0f
#define WP_DEFAULT_B       80.0f

// Hanging-mode tap threshold (0.50 g leaves the light pendulum silent while
// medium taps fire reliably). No DUCK_MODE — biolum is a fixed installation.
#define WP_TAP_THRESH_G   0.50f
#define WP_COOLDOWN_MS    120

// ── ESP-NOW v1 packet ────────────────────────────────────────────
// TelemetryPacketV1 lives in lib/v1_telemetry/v1_packet.h (shared with sender).
#include "v1_packet.h"

#define TIMEOUT_MS     500

static volatile uint32_t pktCount = 0;
static volatile uint32_t lastPacketMs = 0;
static TelemetryPacketV1 latestPacket = {0};

void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    pktCount++;
    if (len == sizeof(TelemetryPacketV1)) {
        memcpy((void*)&latestPacket, data, sizeof(TelemetryPacketV1));
        lastPacketMs = millis();
    }
}

// ── Algorithm enum ───────────────────────────────────────────────
enum Algorithm {
    ALG_QUIET_BLOOM,
    ALG_WAVE_PULSE,
};
static volatile Algorithm currentAlg = ALG_QUIET_BLOOM;

// ── LED driver: 6 strips via RMT, one channel per strip ──
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 16);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, 19);

// Refresh editing-buffer pointers each frame (NeoPixelBus swaps buffers on Show()).
static RgbColor* stripPixels[NUM_STRIPS];
static inline void refreshStripPointers() {
    stripPixels[0] = (RgbColor*)strip0.Pixels(); stripPixels[1] = (RgbColor*)strip1.Pixels();
    stripPixels[2] = (RgbColor*)strip2.Pixels(); stripPixels[3] = (RgbColor*)strip3.Pixels();
    stripPixels[4] = (RgbColor*)strip4.Pixels(); stripPixels[5] = (RgbColor*)strip5.Pixels();
}

static inline void clearAllStrips() {
    RgbColor off(0, 0, 0);
    strip0.ClearTo(off); strip1.ClearTo(off); strip2.ClearTo(off);
    strip3.ClearTo(off); strip4.ClearTo(off); strip5.ClearTo(off);
}

static inline void fillAllStrips(uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    strip0.ClearTo(c); strip1.ClearTo(c); strip2.ClearTo(c);
    strip3.ClearTo(c); strip4.ClearTo(c); strip5.ClearTo(c);
}

static volatile uint32_t g_stripShowUs[NUM_STRIPS] = {0};

static inline void showAllStrips() {
    strip0.Dirty(); strip1.Dirty(); strip2.Dirty();
    strip3.Dirty(); strip4.Dirty(); strip5.Dirty();
    uint32_t t;
    t = micros(); strip0.Show(false); g_stripShowUs[0] = micros() - t;
    t = micros(); strip1.Show(false); g_stripShowUs[1] = micros() - t;
    t = micros(); strip2.Show(false); g_stripShowUs[2] = micros() - t;
    t = micros(); strip3.Show(false); g_stripShowUs[3] = micros() - t;
    t = micros(); strip4.Show(false); g_stripShowUs[4] = micros() - t;
    t = micros(); strip5.Show(false); g_stripShowUs[5] = micros() - t;
}

// ── Per-LED state ────────────────────────────────────────────────
static uint16_t dsR[TOTAL_LEDS], dsG[TOTAL_LEDS], dsB[TOTAL_LEDS];
static float bloomBreathPhase[TOTAL_LEDS];
static float bloomBreathRate[TOTAL_LEDS];
static float bloomBreathPeak[TOTAL_LEDS];
static float bloomHueT[TOTAL_LEDS];
static float bloomFlashGlow[TOTAL_LEDS];
static float bloomFlashDecay[TOTAL_LEDS];

// ── Per-colony state ─────────────────────────────────────────────
static float    bloomColonyEnergy[NUM_COLONIES];
static uint32_t bloomLastMotionMs[NUM_COLONIES];
static float    drainEnvelope[NUM_COLONIES];
static float    bloomHitIntensity[NUM_COLONIES];

// ── Shared motion state ──────────────────────────────────────────
static float    bloomMotionRate = 0.0f;
static float    motionEMA = 0.0f;
static uint32_t bloomPrevPktCount = 0;

// ── Calibration (rest vector from per-axis means) ────────────────
static float restAx = 0, restAy = 0, restAz = 1.0f;
static bool  calibrated = false;
static float calSumAx = 0, calSumAy = 0, calSumAz = 0;
static uint32_t calSamples = 0;
static uint32_t calStartMs = 0;
#define CAL_DURATION_MS 2000

// ── PRNG ────────────────────────────────────────────────────────
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
static inline float vecLen(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}
static void vecNormalize(float &x, float &y, float &z) {
    float len = vecLen(x, y, z);
    if (len > 0) { x /= len; y /= len; z /= len; }
}

// ── v1 decode helpers ────────────────────────────────────────────
static inline float magCountsFromByte(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * MAG_FS;
}
static inline float amagGFromByte(uint8_t b) {
    return magCountsFromByte(b) / COUNTS_PER_G;
}
static inline float gmagDpsFromByte(uint8_t b) {
    return magCountsFromByte(b) / COUNTS_PER_DPS;
}
static inline float axisMeanG(int8_t m) {
    return ((float)m * COUNTS_PER_INT8) / COUNTS_PER_G;
}

// ── Calibration ──────────────────────────────────────────────────
static void startCalibration() {
    calibrated = false;
    calSumAx = 0; calSumAy = 0; calSumAz = 0;
    calSamples = 0;
    calStartMs = millis();
    Serial.println("Calibrating — keep sender still for 2 seconds...");
}

static void updateCalibration(float ax, float ay, float az) {
    if (calStartMs == 0) startCalibration();
    if (!calibrated) {
        calSumAx += ax;
        calSumAy += ay;
        calSumAz += az;
        calSamples++;
        if (millis() - calStartMs >= CAL_DURATION_MS && calSamples > 0) {
            restAx = calSumAx / calSamples;
            restAy = calSumAy / calSamples;
            restAz = calSumAz / calSamples;
            vecNormalize(restAx, restAy, restAz);
            calibrated = true;
            Serial.printf("Calibrated! Rest vector: (%.4f, %.4f, %.4f)\n",
                restAx, restAy, restAz);
        }
    }
}

// ── Bloom motion processing ──────────────────────────────────────
static void bloomProcessMotion(uint32_t now) {
    float gyroRate = gmagDpsFromByte(latestPacket.gmag_max);
    float amagG = amagGFromByte(latestPacket.amag_max);
    float accelJolt = fabsf(amagG - 1.0f);
    bloomMotionRate = fmaxf(gyroRate, accelJolt * 300.0f);

    float surprise = fmaxf(0.0f, bloomMotionRate - motionEMA * SURPRISE_RATIO);

    float alpha = (bloomMotionRate > motionEMA) ? SURPRISE_EMA_UP : SURPRISE_EMA_DOWN;
    motionEMA += alpha * (bloomMotionRate - motionEMA);

    if (surprise > 1.0f) {
        float hitIntensity = clampf(
            log2f(1.0f + surprise) / log2f(1.0f + FLASH_MOTION_SCALE),
            0.0f, 1.0f);
        float normMotion = surprise / DRAIN_SCALE;
        float newDrain = normMotion * normMotion * normMotion
                       * (1.0f - DRAIN_ENVELOPE_DECAY);

        for (uint8_t c = 0; c < NUM_COLONIES; c++) {
            bloomLastMotionMs[c] = now;
            if (hitIntensity > bloomHitIntensity[c]) bloomHitIntensity[c] = hitIntensity;
            if (newDrain > drainEnvelope[c]) drainEnvelope[c] = newDrain;
        }
    }
}

// ── Bloom render ─────────────────────────────────────────────────
static void renderQuietBloom(float dt, uint32_t now) {
    bool colonyDraining[NUM_COLONIES];
    for (uint8_t c = 0; c < NUM_COLONIES; c++) {
        colonyDraining[c] = drainEnvelope[c] > 0.001f;
        if (!colonyDraining[c]) bloomHitIntensity[c] = 0.0f;
        if (now - bloomLastMotionMs[c] > MOTION_SETTLE_MS) {
            bloomColonyEnergy[c] = fminf(
                1.0f, bloomColonyEnergy[c] + BLOOM_RECOVERY_RAMP * dt);
        }
    }

    const float purpleTimeBase = (float)now * (0.001f * BLOOM_PURPLE_RATE);
    const float decayExp = dt * 30.0f;

    refreshStripPointers();

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        RgbColor* px = stripPixels[s];
        for (uint8_t seg = 0; seg < COLONIES_PER_STRIP; seg++) {
            uint8_t c = s * COLONIES_PER_STRIP + seg;
            bool draining = colonyDraining[c];
            float colonyEnergy = bloomColonyEnergy[c];
            float hitIntensity = bloomHitIntensity[c];
            uint16_t segStart = seg * LEDS_PER_COLONY;
            uint16_t segEnd   = segStart + LEDS_PER_COLONY;

            for (uint16_t li = segStart; li < segEnd; li++) {
                uint16_t i = (uint16_t)s * LEDS_PER_STRIP + li;

                if (!draining) {
                    bloomFlashGlow[i] *= fastDecay(bloomFlashDecay[i], decayExp);
                    if (bloomFlashGlow[i] < 0.005f) bloomFlashGlow[i] = 0.0f;
                }

                bloomBreathPhase[i] += dt * bloomBreathRate[i];
                if (bloomBreathPhase[i] >= 1.0f) bloomBreathPhase[i] -= 1.0f;

                float h = bloomHueT[i];
                float wakeThresh = h * BLOOM_RECOVERY_SPREAD;
                float ledRecovery = clampf(
                    (colonyEnergy - wakeThresh) * (1.0f / 0.30f), 0.0f, 1.0f);

                float breath = (fastSinPhase(bloomBreathPhase[i]) * 0.5f + 0.5f);
                float breathGlow = BLOOM_BREATH_FLOOR
                    + breath * (bloomBreathPeak[i] - BLOOM_BREATH_FLOOR);
                breathGlow *= ledRecovery;

                if (draining) {
                    float target = breathGlow * hitIntensity * ENERGY_MULTIPLIER;
                    if (target > bloomFlashGlow[i]) {
                        bloomFlashGlow[i] = target;
                        bloomFlashDecay[i] = BLOOM_FLASH_DECAY_LO
                            + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
                    }
                }

                float g = fmaxf(breathGlow, bloomFlashGlow[i]);

                float flashFrac = (bloomFlashGlow[i] > breathGlow) ? 1.0f : 0.0f;
                float colG = lerpf(lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h),
                                   BLOOM_FLASH_G, flashFrac);
                float colB = lerpf(lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h),
                                   BLOOM_FLASH_B, flashFrac);

                float purplePhase = purpleTimeBase + h * 4.0f;
                float purpleMix = (fastSin(purplePhase) + 1.0f) * 0.5f;
                float colR = purpleMix * BLOOM_PURPLE_MAX;

                float linBright = fastGamma24(g) * BLOOM_BRIGHTNESS_CAP;
                float oR = colR * linBright;
                float oG = colG * linBright;
                float oB = colB * linBright;

                uint16_t tR16 = (uint16_t)clampf(oR * 256.0f, 0, 65535);
                uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
                uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);

                if (tR16 < BLOOM_NOISE_GATE) tR16 = 0;
                if (tG16 < BLOOM_NOISE_GATE) tG16 = 0;
                if (tB16 < BLOOM_NOISE_GATE) tB16 = 0;

                px[li].R = deltaSigma(dsR[i], tR16);
                px[li].G = deltaSigma(dsG[i], tG16);
                px[li].B = deltaSigma(dsB[i], tB16);
            }
        }
    }

    for (uint8_t c = 0; c < NUM_COLONIES; c++) {
        if (drainEnvelope[c] > 0.001f) {
            float drain = drainEnvelope[c] * dt;
            drain = fminf(drain, bloomColonyEnergy[c]);
            bloomColonyEnergy[c] -= drain;
            drainEnvelope[c] *= DRAIN_ENVELOPE_DECAY;
            if (drainEnvelope[c] <= 0.001f) drainEnvelope[c] = 0.0f;
        }
    }
}

// ── Wave pulse: per-strip leaves; tap fires on every strip ──────
struct WPLeaf {
    float pos;
    float vel;
    float boost;
    float colR, colG, colB;
    float brightness;
    float age;
    float refPos;
    float refTime;
    bool  active;
};

static WPLeaf   wpLeaves[NUM_STRIPS][WP_MAX_LEAVES];
static float    wpTime         = 0.0f;
static uint32_t wpLastHitMs    = 0;
static uint32_t wpPrevPktCount = 0;
static float    wpLastAmagG    = 0.0f;
static float    wpLastAngleDeg = 0.0f;

static void resetWavePulseState() {
    wpTime         = 0.0f;
    wpLastHitMs    = 0;
    wpPrevPktCount = 0;
    wpLastAmagG    = 0.0f;
    wpLastAngleDeg = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint8_t i = 0; i < WP_MAX_LEAVES; i++)
            wpLeaves[s][i].active = false;
}

static void resetBloomDynamicState() {
    bloomMotionRate = 0.0f;
    motionEMA = 0.0f;
    bloomPrevPktCount = 0;
    for (uint8_t c = 0; c < NUM_COLONIES; c++) {
        bloomColonyEnergy[c] = 1.0f;
        bloomLastMotionMs[c] = 0;
        drainEnvelope[c]     = 0.0f;
        bloomHitIntensity[c] = 0.0f;
    }
    for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
        bloomFlashGlow[i] = 0.0f;
    }
}

static void wpPickColor(float angleDeg, float &cR, float &cG, float &cB) {
    if (angleDeg <= DEADZONE_DEG) {
        cR = WP_DEFAULT_R; cG = WP_DEFAULT_G; cB = WP_DEFAULT_B;
        return;
    }
    float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
    if (hueFrac > 1.0f) hueFrac = 1.0f;
    uint8_t hueIdx = (uint8_t)(hueFrac * 255.0f);
    cR = (float)oklchVarL[hueIdx][0];
    cG = (float)oklchVarL[hueIdx][1];
    cB = (float)oklchVarL[hueIdx][2];
}

// Per-packet tap test: peak per-axis AC excursion in g (NOT crest factor —
// that gate was structurally unreachable under Option A and was dropped).
// Sender encodes (rawMax - rawMean) >> 8 per axis → 256 raw counts/step.
// At ±4g, 256 / 8192 ≈ 31.25 mg per int8 step.
static void wavePulseProcessMotion(uint32_t now, float angleDeg) {
    constexpr float COUNTS_PER_G_4G = 8192.0f;
    float ax_pk = fmaxf(fabsf((float)latestPacket.ax_max),
                        fabsf((float)latestPacket.ax_min));
    float ay_pk = fmaxf(fabsf((float)latestPacket.ay_max),
                        fabsf((float)latestPacket.ay_min));
    float az_pk = fmaxf(fabsf((float)latestPacket.az_max),
                        fabsf((float)latestPacket.az_min));
    float peak_axis_ac_g = fmaxf(ax_pk, fmaxf(ay_pk, az_pk))
                         * (256.0f / COUNTS_PER_G_4G);

    wpLastAmagG    = peak_axis_ac_g;
    wpLastAngleDeg = angleDeg;

    bool tap = (peak_axis_ac_g >= WP_TAP_THRESH_G);
    if (!tap) return;
    if ((now - wpLastHitMs) < WP_COOLDOWN_MS) return;

    wpLastHitMs = now;

    float overshoot = (peak_axis_ac_g - WP_TAP_THRESH_G) / 2.5f;
    float hitIntensity = clampf(overshoot, 0.0f, 1.0f);
    int nLeaves = 1 + (int)(hitIntensity * 3.0f + 0.5f);
    if (nLeaves > 4) nLeaves = 4;

    float baseR, baseG, baseB;
    wpPickColor(angleDeg, baseR, baseG, baseB);

    // Spawn the same burst on every strip simultaneously. Per-strip jitter
    // keeps the strips from looking like a perfect chorus line.
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (int n = 0; n < nLeaves; n++) {
            int8_t slot = -1;
            for (uint8_t j = 0; j < WP_MAX_LEAVES; j++) {
                if (!wpLeaves[s][j].active) { slot = j; break; }
            }
            if (slot < 0) break;

            WPLeaf &lf = wpLeaves[s][slot];
            lf.pos       = -(float)n * 0.7f - randFloat() * 0.5f;
            lf.vel       = 0.0f;
            lf.boost     = WP_BOOST_SPEED
                         * (0.5f + 0.5f * randFloat())
                         * (0.4f + hitIntensity * 0.8f);
            float jitterR = 0.9f + 0.2f * randFloat();
            float jitterG = 0.9f + 0.2f * randFloat();
            float jitterB = 0.9f + 0.2f * randFloat();
            lf.colR       = clampf(baseR * jitterR, 0.0f, 255.0f);
            lf.colG       = clampf(baseG * jitterG, 0.0f, 255.0f);
            lf.colB       = clampf(baseB * jitterB, 0.0f, 255.0f);
            lf.brightness = 0.0f;
            lf.age        = 0.0f;
            lf.refPos     = lf.pos;
            lf.refTime    = 0.0f;
            lf.active     = true;
        }
    }
}

static void renderWavePulse(float dt) {
    wpTime += dt;
    float t = wpTime;

    float boostDecay = expf(-dt / WP_BOOST_TC);
    float sigma_sq2  = 2.0f * WP_LEAF_SIGMA * WP_LEAF_SIGMA;

    refreshStripPointers();

    // Step 1: advance every active leaf on every strip.
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t i = 0; i < WP_MAX_LEAVES; i++) {
            WPLeaf &lf = wpLeaves[s][i];
            if (!lf.active) continue;

            float noiseVal = fastSin(lf.pos * 0.4f + t * 0.3f
                                     + (float)(s * 13 + i * 7))
                           * fastSin(lf.pos * 0.17f - t * 0.19f
                                     + (float)(s * 5 + i * 3)) * 0.5f
                           + fastSin(lf.pos * 0.09f + t * 0.13f
                                     + (float)(s * 11 + i * 1)) * 0.33f;
            float speedMult = fmaxf(1.0f + noiseVal * WP_TURBULENCE, 0.05f);

            float force = WP_WIND_SPEED * speedMult;
            lf.vel = lf.vel * WP_DAMPING + force * (1.0f - WP_DAMPING);
            lf.boost *= boostDecay;
            lf.pos  += (lf.vel + lf.boost) * dt;
            lf.age  += dt;

            float elapsed = lf.age - lf.refTime;
            if (elapsed > WP_STALL_TIMEOUT) {
                if (fabsf(lf.pos - lf.refPos) < WP_STALL_RADIUS) {
                    lf.active = false; continue;
                }
                lf.refPos  = lf.pos;
                lf.refTime = lf.age;
            }

            lf.brightness = (lf.age < WP_FADE_IN_TIME)
                            ? (lf.age / WP_FADE_IN_TIME) : 1.0f;

            float distToEnd = (float)(LEDS_PER_STRIP - 1) - lf.pos;
            if (distToEnd < WP_FADE_OUT_LEDS && distToEnd >= 0.0f)
                lf.brightness *= distToEnd / WP_FADE_OUT_LEDS;

            if (lf.pos > (float)(LEDS_PER_STRIP - 1) + WP_LEAF_SIGMA * 3.0f)
                lf.active = false;
        }
    }

    // Step 2: render each strip's pixels from its own leaf array.
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        RgbColor* px = stripPixels[s];
        for (uint16_t li = 0; li < LEDS_PER_STRIP; li++) {
            uint16_t i = (uint16_t)s * LEDS_PER_STRIP + li;
            float glow = 0.0f, oR = 0.0f, oG = 0.0f, oB = 0.0f;

            for (uint8_t j = 0; j < WP_MAX_LEAVES; j++) {
                const WPLeaf &lf = wpLeaves[s][j];
                if (!lf.active) continue;
                float d = (float)li - lf.pos;
                float w = expf(-(d * d) / sigma_sq2) * lf.brightness;
                glow += w;
                oR   += w * lf.colR;
                oG   += w * lf.colG;
                oB   += w * lf.colB;
            }

            float fR = 0.0f, fG = 0.0f, fB = 0.0f;
            if (glow > 1e-6f) {
                float bright = clampf(glow, 0.0f, 1.0f) * WP_MAX_BRIGHTNESS;
                float linBright = fastGamma24(bright) * BRIGHTNESS_CAP;
                fR = (oR / glow) * linBright;
                fG = (oG / glow) * linBright;
                fB = (oB / glow) * linBright;
            }

            uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
            uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
            uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
            if (tR16 < 256) tR16 = 0;
            if (tG16 < 256) tG16 = 0;
            if (tB16 < 256) tB16 = 0;

            px[li].R = deltaSigma(dsR[i], tR16);
            px[li].G = deltaSigma(dsG[i], tG16);
            px[li].B = deltaSigma(dsB[i], tB16);
        }
    }
}

// ── Dual-core handshake ──────────────────────────────────────────
static TaskHandle_t showTaskHandle   = nullptr;
static TaskHandle_t renderTaskHandle = nullptr;
static volatile uint32_t g_renderUsLast = 0;
static volatile uint32_t g_showUsLast   = 0;

static void renderTask(void* /*param*/) {
    uint32_t lastRenderMs = 0;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        uint32_t now = millis();
        float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
        if (dt > 0.1f) dt = 0.1f;
        lastRenderMs = now;

        // Tilt + calibration from per-axis means (gravity vector @ 25 Hz).
        float ax = axisMeanG(latestPacket.ax_mean);
        float ay = axisMeanG(latestPacket.ay_mean);
        float az = axisMeanG(latestPacket.az_mean);
        vecNormalize(ax, ay, az);
        bool connected = (lastPacketMs > 0) && (now - lastPacketMs < TIMEOUT_MS);
        if (connected) updateCalibration(ax, ay, az);

        float angleDeg = 0.0f;
        if (calibrated && connected) {
            float cosAngle = clampf(restAx * ax + restAy * ay + restAz * az,
                                    -1.0f, 1.0f);
            angleDeg = acosf(cosAngle) * (180.0f / M_PI);
        }

        if (currentAlg == ALG_QUIET_BLOOM && connected) {
            if (pktCount != bloomPrevPktCount) {
                bloomProcessMotion(now);
                bloomPrevPktCount = pktCount;
            }
        } else if (currentAlg == ALG_WAVE_PULSE && connected) {
            if (pktCount != wpPrevPktCount) {
                wavePulseProcessMotion(now, angleDeg);
                wpPrevPktCount = pktCount;
            }
        }

        uint32_t t0 = micros();
        if (currentAlg == ALG_QUIET_BLOOM) {
            renderQuietBloom(dt, now);
        } else {
            renderWavePulse(dt);
        }
        g_renderUsLast = micros() - t0;

        xTaskNotifyGive(showTaskHandle);
    }
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.printf("biolum bloom (v1) — %u strips × %u LEDs = %u total\n",
                  NUM_STRIPS, LEDS_PER_STRIP, TOTAL_LEDS);

    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
    clearAllStrips();
    showAllStrips();

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    for (uint8_t c = 0; c < NUM_COLONIES; c++) {
        bloomColonyEnergy[c] = 1.0f;
        bloomLastMotionMs[c] = 0;
        drainEnvelope[c]     = 0.0f;
        bloomHitIntensity[c] = 0.0f;
    }

    for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / TOTAL_LEDS);
        dsR[i] = seed;
        dsG[i] = (seed + 85)  & 0xFF;
        dsB[i] = (seed + 170) & 0xFF;

        bloomBreathPhase[i]  = randFloat();
        float period = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bloomBreathRate[i]   = 1.0f / period;
        bloomBreathPeak[i]   = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bloomHueT[i]         = randFloat();
        bloomFlashGlow[i]    = 0.0f;
        bloomFlashDecay[i]   = BLOOM_FLASH_DECAY_LO
            + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
    }

    resetWavePulseState();

    // Boot flash
    fillAllStrips(0, 0, 40);
    showAllStrips();
    delay(200);
    clearAllStrips();
    showAllStrips();

    WiFi.mode(WIFI_STA);
    WiFi.setHostname("biolum");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("[wifi] Connecting to '%s'...", WIFI_SSID);
    uint32_t wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 15000) {
        delay(250);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[wifi] connected, ch=%d ip=%s\n",
                      WiFi.channel(), WiFi.localIP().toString().c_str());
        ArduinoOTA.setHostname("biolum");
        ArduinoOTA.begin();
        Serial.println("[ota] ready — `biolum.local` or upload_port = "
                       + WiFi.localIP().toString());
    } else {
        Serial.println("\n[wifi] FAILED — running ESP-NOW only on default channel");
    }

    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.printf("WiFi ch=%d MAC=%s — waiting for v1 sender...\n",
                  WiFi.channel(), WiFi.macAddress().c_str());
    Serial.println("Commands: 'c' recal, 'b' bloom, 'w' wave_pulse");

    showTaskHandle = xTaskGetCurrentTaskHandle();
    xTaskCreatePinnedToCore(renderTask, "render", 8192, nullptr, 1,
                            &renderTaskHandle, 0);
    xTaskNotifyGive(renderTaskHandle);
}

// ── Main loop (show task on core 1) ──────────────────────────────
void loop() {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    uint32_t t0 = micros();
    showAllStrips();
    g_showUsLast = micros() - t0;

    xTaskNotifyGive(renderTaskHandle);

    ArduinoOTA.handle();

    // Serial commands (algorithm switch + recalibrate)
    if (Serial.available()) {
        char ch = Serial.read();
        if (ch == 'c' || ch == 'C') {
            startCalibration();
        } else if (ch == 'b' || ch == 'B') {
            currentAlg = ALG_QUIET_BLOOM;
            resetBloomDynamicState();
            Serial.println("Algorithm: quiet_bloom");
        } else if (ch == 'w' || ch == 'W') {
            currentAlg = ALG_WAVE_PULSE;
            resetWavePulseState();
            Serial.println("Algorithm: wave_pulse");
        }
    }

    static uint32_t frameCount = 0;
    frameCount++;
    static uint64_t renderUsAccum = 0, showUsAccum = 0;
    renderUsAccum += g_renderUsLast;
    showUsAccum   += g_showUsLast;

    static uint32_t lastTelemetryMs = 0;
    static uint32_t lastTelemetryFrames = 0;
    uint32_t now = millis();
    bool connected = (lastPacketMs > 0) && (now - lastPacketMs < TIMEOUT_MS);
    if (now - lastTelemetryMs >= 1000) {
        uint32_t fps = frameCount - lastTelemetryFrames;
        uint32_t avgRender = fps ? (uint32_t)(renderUsAccum / fps) : 0;
        uint32_t avgShow   = fps ? (uint32_t)(showUsAccum / fps) : 0;
        lastTelemetryFrames = frameCount;
        lastTelemetryMs = now;
        renderUsAccum = 0;
        showUsAccum = 0;
        if (currentAlg == ALG_QUIET_BLOOM) {
            float meanE = 0.0f;
            uint8_t draining = 0;
            for (uint8_t c = 0; c < NUM_COLONIES; c++) {
                meanE += bloomColonyEnergy[c];
                if (drainEnvelope[c] > 0.001f) draining++;
            }
            meanE /= NUM_COLONIES;
            Serial.printf("[bloom] %ufps r=%uus s=%uus rate=%.1f meanE=%.3f drain=%u %s\n",
                          fps, avgRender, avgShow,
                          bloomMotionRate, meanE, draining,
                          connected ? "connected" : "no-sender");
        } else {
            uint8_t nActive = 0;
            for (uint8_t s = 0; s < NUM_STRIPS; s++)
                for (uint8_t i = 0; i < WP_MAX_LEAVES; i++)
                    if (wpLeaves[s][i].active) nActive++;
            Serial.printf("[wave] %ufps r=%uus s=%uus peak=%.2fg leaves=%u angle=%.1f thresh=%.2fg %s\n",
                          fps, avgRender, avgShow,
                          wpLastAmagG, nActive, wpLastAngleDeg, WP_TAP_THRESH_G,
                          connected ? "connected" : "no-sender");
        }
    }
}
