/*
 * BENCH-BULBS — A/B test rig for both festicorn animation pipelines on
 * one ESP32-C3, driven by two simultaneous ESP-NOW senders.
 *
 * Receiver: ESP32-C3 super-mini, MAC 10:00:3B:B0:6E:04, /dev/cu.usbmodem114401
 *
 *   Strip A: 50× SK6812 RGBW on GPIO 10 — driven by the original-duck v0.1
 *            sender (mic + gyro, 15-byte SensorPacket). Animations:
 *            sparkle_burst, fire_meld, fire_flicker, quiet_bloom. Effect
 *            cycling is automatic from tilt + audio onset.
 *
 *   Strip B: 50× WS2812B RGB on GPIO 21 — driven by the gyro-sense v1
 *            sender (gyro only, 16-byte TelemetryPacketV1). Animation:
 *            bloom (verbatim per-LED semantics, collapsed from 6 strips
 *            to one 50-LED strip with 2 colonies).
 *
 * The ESP-NOW receive callback dispatches by packet length:
 *     len == 15 → SensorPacket        → duck state
 *     len == 16 → TelemetryPacketV1   → biolum state
 *
 * NOT PRODUCTION. The biolum/ project remains the production target.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <oklch_lut.h>
#include <delta_sigma.h>
#include <fast_math.h>

// ════════════════════════════════════════════════════════════════════
//   SHARED HARDWARE / WIFI / HELPERS
// ════════════════════════════════════════════════════════════════════

// ── Strip A: SK6812 RGBW (duck) ─────────────────────────────────
#define DUCK_LED_PIN    10
#define DUCK_LED_COUNT  50

// ── Strip B: WS2812B RGB (biolum) ───────────────────────────────
#define BIO_LED_PIN     21
#define BIO_LED_COUNT   50

// NeoPixelBus drivers — RMT0 → SK6812 RGBW, RMT1 → WS2812B RGB.
static NeoPixelBus<NeoGrbwFeature, NeoEsp32Rmt0Sk6812Method>  stripDuck(DUCK_LED_COUNT, DUCK_LED_PIN);
static NeoPixelBus<NeoGrbFeature,  NeoEsp32Rmt1Ws2812xMethod> stripBio (BIO_LED_COUNT,  BIO_LED_PIN);

// ── ESP-NOW timeouts / Wi-Fi channel discovery ─────────────────
#define TIMEOUT_MS         500
#define CHANNEL_FALLBACK   1
#define CHANNEL_RESCAN_MS  (5UL * 60UL * 1000UL)
static const char* WIFI_SSID_TARGET = "cuteplant";

static uint8_t  currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs     = 0;

// ── Shared PRNG (xorshift32) ───────────────────────────────────
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

// ── Shared math helpers ────────────────────────────────────────
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

// ════════════════════════════════════════════════════════════════════
//   PACKET STRUCTS + ESP-NOW DISPATCH
// ════════════════════════════════════════════════════════════════════

// LEGACY: 15-byte v0.1 SensorPacket from the original-duck sender.
struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;     // 6 bytes: raw accelerometer (±2g, 16384 = 1g)
    int16_t gx, gy, gz;     // 6 bytes: raw gyroscope (±250°/s, /131 = deg/s)
    uint16_t rawRms;        // 2 bytes: raw audio RMS
    uint8_t micEnabled;     // 1 byte: shake toggle state
};                          // 15 bytes total
static_assert(sizeof(SensorPacket) == 15, "duck v0.1 packet must be 15 bytes");

// 16-byte v1 TelemetryPacketV1 — defined in lib/v1_telemetry/v1_packet.h.
#include "v1_packet.h"

// Two independent latest-packet stores. Each pipeline only reads its own.
static volatile uint32_t duckPktCount   = 0;
static volatile uint32_t duckLastPktMs  = 0;
static SensorPacket      latestDuckPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

static volatile uint32_t bioPktCount    = 0;
static volatile uint32_t bioLastPktMs   = 0;
static TelemetryPacketV1 latestV1Packet = {0};

// Dispatch by length. Wrong-length packets are dropped silently — strict
// length filtering is what makes simultaneous dual-sender broadcasts safe.
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len == sizeof(SensorPacket)) {
        memcpy((void*)&latestDuckPacket, data, sizeof(SensorPacket));
        duckLastPktMs = millis();
        duckPktCount++;
    } else if (len == sizeof(TelemetryPacketV1)) {
        memcpy((void*)&latestV1Packet, data, sizeof(TelemetryPacketV1));
        bioLastPktMs = millis();
        bioPktCount++;
    }
}

// ── SSID-based channel discovery (sender pattern, adapted for receiver) ──
// Receiver doesn't join the AP — it just locks the radio to whatever channel
// the AP advertises on, so the senders (which use the same scan) and the
// receiver agree without any router-side configuration.
static uint8_t scanForSsidChannel(const char* ssid) {
    int n = WiFi.scanNetworks(/*async=*/false, /*show_hidden=*/false,
                              /*passive=*/true, /*max_ms_per_chan=*/120);
    uint8_t found = 0;
    int8_t bestRssi = -127;
    for (int i = 0; i < n; i++) {
        if (WiFi.SSID(i) == ssid) {
            int8_t rssi = WiFi.RSSI(i);
            if (rssi > bestRssi) {
                bestRssi = rssi;
                found = WiFi.channel(i);
            }
        }
    }
    WiFi.scanDelete();
    return found;
}

static void applyChannel(uint8_t ch) {
    if (ch == 0) ch = CHANNEL_FALLBACK;
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    currentChannel = ch;
}

// ════════════════════════════════════════════════════════════════════
// === DUCK PIPELINE ===
//
//   Verbatim semantics from festicorn/original-duck/src/bulb_receiver.cpp.
//   Drives the SK6812 RGBW strip on GPIO 10. Reads latestDuckPacket only.
//   All globals here are prefixed `duck`.
// ════════════════════════════════════════════════════════════════════

// ── Rendering ────────────────────────────────────────────────────
#define DUCK_BRIGHTNESS_CAP 0.10f
#define DUCK_PURE_W_CEIL    0.10f  // below this: pure W, no RGB dies
#define DUCK_PURE_W_BLEND   0.15f  // blend range above CEIL to full RGB

// ── Color / tilt mapping ─────────────────────────────────────────
#define DUCK_DEADZONE_DEG    10.0f
#define DUCK_MAX_ANGLE_DEG   180.0f
#define DUCK_BLEND_RANGE_DEG 40.0f
#define DUCK_SENSOR_HZ       25.0f

// ── FixedRangeRMS parameters ─────────────────────────────────────
// Floor=10000 picked empirically: bench INMP441 ambient peaks ~12k
// (1Hz×35s sample, max=12324, p99=10170). 10k cleanly clears the noise
// distribution while still admitting close-talked / shouted voice and any
// musical environment (music sits 20k+, fully into uint16 saturation
// territory). Quiet conversational voice at arm's length will not trigger —
// expected behavior; user can move closer or speak up to engage the duck.
// See library/test-vectors/inmp441-validation/session_20260428_171620 for
// per-condition RMS distributions used to pick this number.
#define DUCK_RMS_FLOOR       10000.0f
#define DUCK_RMS_CEILING     50000.0f
#define DUCK_RMS_PEAK_DECAY  0.9999f

// ── EnergyDelta parameters ───────────────────────────────────────
#define DUCK_DELTA_PEAK_DECAY 0.998f

// ── Sparkle / Fire parameters ────────────────────────────────────
#define DUCK_SPARKLE_DEADBAND   0.08f
#define DUCK_FIRE_FLICKER_SCALE 3.0f
#define DUCK_FIRE_DEADBAND      0.08f
#define DUCK_FIRE_DROPOUT_DEPTH 0.85f

// ── Quiet bloom parameters ───────────────────────────────────────
#define DUCK_BLOOM_BRIGHTNESS_CAP   0.70f
#define DUCK_BLOOM_NOISE_GATE       256
#define DUCK_SURPRISE_EMA_UP        0.05f
#define DUCK_SURPRISE_EMA_DOWN      0.2f
#define DUCK_SURPRISE_RATIO         3.0f
#define DUCK_DRAIN_SCALE            100.0f
#define DUCK_DRAIN_ENVELOPE_DECAY   0.85f
#define DUCK_FLASH_MOTION_SCALE     300.0f
#define DUCK_ENERGY_MULTIPLIER      1.4f
#define DUCK_MOTION_SETTLE_MS       300
#define DUCK_BLOOM_BREATH_MIN_PERIOD 3.0f
#define DUCK_BLOOM_BREATH_MAX_PERIOD 8.0f
#define DUCK_BLOOM_BREATH_MIN_PEAK   0.65f
#define DUCK_BLOOM_BREATH_MAX_PEAK   1.00f
#define DUCK_BLOOM_BREATH_FLOOR      0.15f
#define DUCK_BLOOM_FLASH_DECAY_LO   0.96f
#define DUCK_BLOOM_FLASH_DECAY_HI   0.985f
#define DUCK_BLOOM_RECOVERY_RAMP    0.033f
#define DUCK_BLOOM_RECOVERY_SPREAD  0.70f
#define DUCK_BLOOM_HUE_A_G   20.0f
#define DUCK_BLOOM_HUE_A_B  100.0f
#define DUCK_BLOOM_HUE_B_G   70.0f
#define DUCK_BLOOM_HUE_B_B  110.0f
#define DUCK_BLOOM_FLASH_G  150.0f
#define DUCK_BLOOM_FLASH_B  170.0f
#define DUCK_BLOOM_W_ONSET    0.5f

// ── Algorithm enum ───────────────────────────────────────────────
enum DuckAlgorithm {
    DUCK_ALG_SPARKLE_BURST,
    DUCK_ALG_FIRE_MELD,
    DUCK_ALG_FIRE_FLICKER,
    DUCK_ALG_QUIET_BLOOM,
};
static DuckAlgorithm duckCurrentAlg = DUCK_ALG_SPARKLE_BURST;

// ── Per-channel delta-sigma accumulators ─────────────────────────
static uint16_t duckDsR[DUCK_LED_COUNT], duckDsG[DUCK_LED_COUNT],
                duckDsB[DUCK_LED_COUNT], duckDsW[DUCK_LED_COUNT];

// ── Calibration — rest vector ───────────────────────────────────
static float duckRestAx = 0, duckRestAy = 0, duckRestAz = 1.0f;
static bool  duckCalibrated = false;
static float duckCalSumAx = 0, duckCalSumAy = 0, duckCalSumAz = 0;
static uint32_t duckCalSamples = 0;
static uint32_t duckCalStartMs = 0;
#define DUCK_CAL_DURATION_MS 2000

// ── FixedRangeRMS state ──────────────────────────────────────────
static float duckFrrCeiling = DUCK_RMS_CEILING;
static float duckEnergy = 0.0f;

// ── EnergyDelta state ────────────────────────────────────────────
static float duckPrevRms = 0.0f;
static float duckDeltaPeak = 1e-6f;
static float duckOnset = 0.0f;

// ── Sparkle render state ─────────────────────────────────────────
static float duckSparkle[DUCK_LED_COUNT];
static float duckDecayRates[DUCK_LED_COUNT];
static float duckEnvelope = 0.0f;
static float duckCooldownRemaining = 0.0f;

// ── Fire render state ────────────────────────────────────────────
static float duckFireTime = 0.0f;
static float duckFireBaseBrightness = 0.0f;
static float duckFireFlickerIntensity = 0.0f;
static float duckFireColorEnergy = 0.0f;
static float duckFirePrevEnergyForDeriv = 0.0f;
static float duckFireEnergyDerivSmooth = 0.0f;
static float duckFireDropoutAmount = 0.0f;

// ── Quiet bloom render state ─────────────────────────────────────
static float duckBloomBreathPhase[DUCK_LED_COUNT];
static float duckBloomBreathPeriod[DUCK_LED_COUNT];
static float duckBloomBreathPeak[DUCK_LED_COUNT];
static float duckBloomHueT[DUCK_LED_COUNT];
static float duckBloomFlashGlow[DUCK_LED_COUNT];
static float duckBloomFlashDecay[DUCK_LED_COUNT];
static float duckBloomColonyEnergy = 1.0f;
static uint32_t duckBloomLastMotionMs = 0;
static float duckBloomMotionRate = 0.0f;
static float duckMotionEMA = 0.0f;
static float duckDrainEnvelope = 0.0f;
static float duckBloomHitIntensity = 0.0f;
static uint32_t duckBloomPrevPktCount = 0;

static uint8_t duckSparkPeakR = 0, duckSparkPeakG = 0,
               duckSparkPeakB = 0, duckSparkPeakW = 0;
static float   duckSparkPeakBright = 0.0f;
static float   duckSparkPeakLin = 0.0f;

// ── Motion helpers (duck) ─────────────────────────────────────────
static float duckComputeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx * fx + fy * fy + fz * fz) / 131.0f;
}

static float duckComputeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx * fx + fy * fy + fz * fz);
    return fabsf(mag - 16384.0f) / 16384.0f;
}

// ── FixedRangeRMS: raw uint16 RMS → 0-1 energy ───────────────────
static float duckComputeEnergy(uint16_t rawRms) {
    float rms = (float)rawRms;
    duckFrrCeiling = fmaxf(DUCK_RMS_CEILING, duckFrrCeiling * DUCK_RMS_PEAK_DECAY);
    if (rms > duckFrrCeiling) duckFrrCeiling = rms;
    if (rms < DUCK_RMS_FLOOR) return 0.0f;
    float db = 20.0f * log10f(rms / DUCK_RMS_FLOOR);
    float dbRange = 20.0f * log10f(duckFrrCeiling / DUCK_RMS_FLOOR);
    return clampf(db / dbRange, 0.0f, 1.0f);
}

// ── EnergyDelta: frame-to-frame RMS change → 0-1 onset ───────────
static float duckComputeOnset(uint16_t rawRms) {
    float rms = (float)rawRms;
    float delta = fabsf(rms - duckPrevRms);
    duckPrevRms = rms;
    duckDeltaPeak = fmaxf(delta, duckDeltaPeak * DUCK_DELTA_PEAK_DECAY);
    return (duckDeltaPeak > 1e-6f) ? (delta / duckDeltaPeak) : 0.0f;
}

// ── Calibration ──────────────────────────────────────────────────
static void duckStartCalibration() {
    duckCalibrated = false;
    duckCalSumAx = 0; duckCalSumAy = 0; duckCalSumAz = 0;
    duckCalSamples = 0;
    duckCalStartMs = millis();
}

static void duckUpdateCalibration(float ax, float ay, float az) {
    if (duckCalStartMs == 0) duckStartCalibration();
    if (!duckCalibrated) {
        duckCalSumAx += ax; duckCalSumAy += ay; duckCalSumAz += az;
        duckCalSamples++;
        if (millis() - duckCalStartMs >= DUCK_CAL_DURATION_MS && duckCalSamples > 0) {
            duckRestAx = duckCalSumAx / duckCalSamples;
            duckRestAy = duckCalSumAy / duckCalSamples;
            duckRestAz = duckCalSumAz / duckCalSamples;
            vecNormalize(duckRestAx, duckRestAy, duckRestAz);
            duckCalibrated = true;
        }
    }
}

// ── Reset helpers ────────────────────────────────────────────────
static void duckResetFireState() {
    duckFireTime = 0.0f;
    duckFireBaseBrightness = 0.0f;
    duckFireFlickerIntensity = 0.0f;
    duckFireColorEnergy = 0.0f;
    duckFirePrevEnergyForDeriv = 0.0f;
    duckFireEnergyDerivSmooth = 0.0f;
    duckFireDropoutAmount = 0.0f;
}

static void duckResetSparkleState() {
    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        duckSparkle[i] = 0.0f;
        duckDecayRates[i] = 0.92f + randFloat() * 0.05f;
    }
    duckEnvelope = 0.0f;
    duckCooldownRemaining = 0.0f;
}

static void duckResetBloomState() {
    duckBloomColonyEnergy = 1.0f;
    duckBloomLastMotionMs = 0;
    duckBloomMotionRate = 0.0f;
    duckMotionEMA = 0.0f;
    duckDrainEnvelope = 0.0f;
    duckBloomHitIntensity = 0.0f;
    duckBloomPrevPktCount = 0;
    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        duckBloomBreathPhase[i] = randFloat();
        duckBloomBreathPeriod[i] = DUCK_BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (DUCK_BLOOM_BREATH_MAX_PERIOD - DUCK_BLOOM_BREATH_MIN_PERIOD);
        duckBloomBreathPeak[i] = DUCK_BLOOM_BREATH_MIN_PEAK
            + randFloat() * (DUCK_BLOOM_BREATH_MAX_PEAK - DUCK_BLOOM_BREATH_MIN_PEAK);
        duckBloomHueT[i] = randFloat();
        duckBloomFlashGlow[i] = 0.0f;
        duckBloomFlashDecay[i] = DUCK_BLOOM_FLASH_DECAY_LO
            + randFloat() * (DUCK_BLOOM_FLASH_DECAY_HI - DUCK_BLOOM_FLASH_DECAY_LO);
    }
}

// ── Quiet bloom motion / render (duck side, single colony) ──────
static void duckBloomProcessMotion(uint32_t now) {
    float gyroRate  = duckComputeGyroRate(latestDuckPacket.gx,
                                          latestDuckPacket.gy,
                                          latestDuckPacket.gz);
    float accelJolt = duckComputeAccelJolt(latestDuckPacket.ax,
                                           latestDuckPacket.ay,
                                           latestDuckPacket.az);
    duckBloomMotionRate = fmaxf(gyroRate, accelJolt * 300.0f);

    float surprise = fmaxf(0.0f, duckBloomMotionRate - duckMotionEMA * DUCK_SURPRISE_RATIO);

    float alpha = (duckBloomMotionRate > duckMotionEMA)
        ? DUCK_SURPRISE_EMA_UP : DUCK_SURPRISE_EMA_DOWN;
    duckMotionEMA += alpha * (duckBloomMotionRate - duckMotionEMA);

    if (surprise > 1.0f) {
        duckBloomLastMotionMs = now;
        float hitIntensity = clampf(
            log2f(1.0f + surprise) / log2f(1.0f + DUCK_FLASH_MOTION_SCALE),
            0.0f, 1.0f);
        if (hitIntensity > duckBloomHitIntensity) duckBloomHitIntensity = hitIntensity;

        float normMotion = surprise / DUCK_DRAIN_SCALE;
        float newDrain = normMotion * normMotion * normMotion
                       * (1.0f - DUCK_DRAIN_ENVELOPE_DECAY);
        if (newDrain > duckDrainEnvelope) duckDrainEnvelope = newDrain;
    }
}

static void duckRenderQuietBloom(float dt, uint32_t now) {
    bool draining = duckDrainEnvelope > 0.001f;
    if (!draining) duckBloomHitIntensity = 0.0f;

    if (now - duckBloomLastMotionMs > DUCK_MOTION_SETTLE_MS) {
        duckBloomColonyEnergy = fminf(1.0f, duckBloomColonyEnergy
                                            + DUCK_BLOOM_RECOVERY_RAMP * dt);
    }

    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        if (!draining) {
            duckBloomFlashGlow[i] *= fastDecay(duckBloomFlashDecay[i], dt * 30.0f);
            if (duckBloomFlashGlow[i] < 0.005f) duckBloomFlashGlow[i] = 0.0f;
        }

        duckBloomBreathPhase[i] += dt / duckBloomBreathPeriod[i];
        if (duckBloomBreathPhase[i] >= 1.0f) duckBloomBreathPhase[i] -= 1.0f;

        float wakeThresh = duckBloomHueT[i] * DUCK_BLOOM_RECOVERY_SPREAD;
        float ledRecovery = clampf(
            (duckBloomColonyEnergy - wakeThresh) / 0.30f, 0.0f, 1.0f);

        float breath = (fastSinPhase(duckBloomBreathPhase[i]) * 0.5f + 0.5f);
        float breathGlow = DUCK_BLOOM_BREATH_FLOOR
            + breath * (duckBloomBreathPeak[i] - DUCK_BLOOM_BREATH_FLOOR);
        breathGlow *= ledRecovery;

        if (draining) {
            float target = breathGlow * duckBloomHitIntensity * DUCK_ENERGY_MULTIPLIER;
            if (target > duckBloomFlashGlow[i]) {
                duckBloomFlashGlow[i] = target;
                duckBloomFlashDecay[i] = DUCK_BLOOM_FLASH_DECAY_LO
                    + randFloat() * (DUCK_BLOOM_FLASH_DECAY_HI - DUCK_BLOOM_FLASH_DECAY_LO);
            }
        }

        float g = fmaxf(breathGlow, duckBloomFlashGlow[i]);

        float flashFrac = (duckBloomFlashGlow[i] > breathGlow) ? 1.0f : 0.0f;
        float h = duckBloomHueT[i];
        float colG = lerpf(lerpf(DUCK_BLOOM_HUE_A_G, DUCK_BLOOM_HUE_B_G, h),
                           DUCK_BLOOM_FLASH_G, flashFrac);
        float colB = lerpf(lerpf(DUCK_BLOOM_HUE_A_B, DUCK_BLOOM_HUE_B_B, h),
                           DUCK_BLOOM_FLASH_B, flashFrac);

        float linBright = fastGamma24(g) * DUCK_BLOOM_BRIGHTNESS_CAP;
        float oG = colG * linBright;
        float oB = colB * linBright;

        float wFrac = clampf((g - DUCK_BLOOM_W_ONSET) / (1.0f - DUCK_BLOOM_W_ONSET),
                             0.0f, 1.0f);
        float energyGate = clampf((duckBloomColonyEnergy - 0.7f) / 0.3f, 0.0f, 1.0f);
        float wGate = fmaxf(energyGate, flashFrac);
        float oW = wFrac * wGate * linBright * 200.0f;

        uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

        if (tG16 < DUCK_BLOOM_NOISE_GATE) tG16 = 0;
        if (tB16 < DUCK_BLOOM_NOISE_GATE) tB16 = 0;
        if (tW16 < DUCK_BLOOM_NOISE_GATE) tW16 = 0;

        uint8_t gc = deltaSigma(duckDsG[i], tG16);
        uint8_t b  = deltaSigma(duckDsB[i], tB16);
        uint8_t w  = deltaSigma(duckDsW[i], tW16);
        stripDuck.SetPixelColor(i, RgbwColor(0, gc, b, w));
    }

    if (draining) {
        float drain = duckDrainEnvelope * dt;
        drain = fminf(drain, duckBloomColonyEnergy);
        duckBloomColonyEnergy -= drain;
        duckDrainEnvelope *= DUCK_DRAIN_ENVELOPE_DECAY;
        if (duckDrainEnvelope <= 0.001f) duckDrainEnvelope = 0.0f;
    }
}

// ── Sparkle burst render ─────────────────────────────────────────
static void duckRenderSparkleBurst(float dt, float angleDeg, float tiltBlend,
                                    float tiltR, float tiltG, float tiltB) {
    bool isSilent = duckEnergy < 0.001f;

    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (duckEnergy > duckEnvelope)
        duckEnvelope += attackAlpha * (duckEnergy - duckEnvelope);
    else
        duckEnvelope += decayAlpha * (duckEnergy - duckEnvelope);

    duckCooldownRemaining = fmaxf(0.0f, duckCooldownRemaining - dt);

    float onsetThreshold = fmaxf(0.15f, 0.4f - duckEnvelope * 0.3f);

    if (duckOnset > onsetThreshold && duckCooldownRemaining <= 0.0f && !isSilent) {
        duckCooldownRemaining = fmaxf(0.050f, 0.150f - duckEnvelope * 0.10f);

        float onsetStrength = clampf(duckOnset, 0.0f, 1.0f);
        int nIgnite = (int)(DUCK_LED_COUNT * (0.3f + 0.2f * onsetStrength));

        static uint8_t indices[DUCK_LED_COUNT];
        for (uint8_t i = 0; i < DUCK_LED_COUNT; i++) indices[i] = i;
        for (int i = 0; i < nIgnite; i++) {
            int j = i + (int)(xorshift32() % (DUCK_LED_COUNT - i));
            uint8_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        float sparkVal = 0.7f + 0.3f * onsetStrength;
        for (int i = 0; i < nIgnite; i++) {
            duckSparkle[indices[i]] = sparkVal;
            duckDecayRates[indices[i]] = 0.92f + randFloat() * 0.05f;
        }
    }

    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        duckSparkle[i] *= fastDecay(duckDecayRates[i], dt * 30.0f);
    }

    if (!isSilent) {
        for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
            float jitter = (randFloat() - 0.5f) * 0.02f;
            float newVal = duckSparkle[i] + jitter;
            if (newVal < 0.0f) newVal = 0.0f;
            if (newVal > duckSparkle[i] && jitter > 0) newVal = duckSparkle[i];
            duckSparkle[i] = newVal;
        }
    }

    float base = fminf(duckEnvelope, 0.2f);

    duckSparkPeakR = duckSparkPeakG = duckSparkPeakB = duckSparkPeakW = 0;
    duckSparkPeakBright = 0.0f;
    duckSparkPeakLin = 0.0f;

    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        float s = duckSparkle[i];

        float bright = base + s * (1.0f - base);
        if (bright < DUCK_SPARKLE_DEADBAND) bright = 0.0f;

        float colR = 255.0f;
        float colG = 180.0f + (240.0f - 180.0f) * s;
        float colB =  80.0f + (200.0f -  80.0f) * s;

        if (tiltBlend > 0.0f) {
            colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
            colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
            colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
        }

        float linBright = fastGamma24(bright) * DUCK_BRIGHTNESS_CAP;
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

        uint8_t r = deltaSigma(duckDsR[i], tR16);
        uint8_t g = deltaSigma(duckDsG[i], tG16);
        uint8_t b = deltaSigma(duckDsB[i], tB16);
        uint8_t w = deltaSigma(duckDsW[i], tW16);
        stripDuck.SetPixelColor(i, RgbwColor(r, g, b, w));

        if (r > duckSparkPeakR) duckSparkPeakR = r;
        if (g > duckSparkPeakG) duckSparkPeakG = g;
        if (b > duckSparkPeakB) duckSparkPeakB = b;
        if (w > duckSparkPeakW) duckSparkPeakW = w;
        if (bright > duckSparkPeakBright) duckSparkPeakBright = bright;
        if (linBright > duckSparkPeakLin) duckSparkPeakLin = linBright;
    }
}

// ── Fire render (shared by fire_meld and fire_flicker) ───────────
static void duckRenderFire(float dt, bool withDropout, float tiltBlend) {
    duckFireTime += dt;
    float t = duckFireTime;

    bool isSilent = duckEnergy < 0.001f;
    bool isPercussiveOnly = (!isSilent && duckEnergy < 0.15f && duckOnset > 0.5f);

    float attackAlpha = fminf(1.0f, dt / 0.050f);
    float decayAlpha  = fminf(1.0f, dt / 2.0f);

    float targetBrightness;
    if (isSilent) {
        targetBrightness = 0.25f;
    } else {
        targetBrightness = fmaxf(0.25f, duckEnergy);
    }

    if (targetBrightness > duckFireBaseBrightness)
        duckFireBaseBrightness += attackAlpha * (targetBrightness - duckFireBaseBrightness);
    else
        duckFireBaseBrightness += decayAlpha * (targetBrightness - duckFireBaseBrightness);

    float flickerAlpha = fminf(1.0f, dt / 0.200f);
    float deltaTarget = isSilent ? 0.0f : duckOnset;
    duckFireFlickerIntensity += flickerAlpha * (deltaTarget - duckFireFlickerIntensity);

    float dropoutAmount = 0.0f;
    if (withDropout) {
        float energyDeriv = (duckEnergy - duckFirePrevEnergyForDeriv) / fmaxf(dt, 0.001f);
        duckFirePrevEnergyForDeriv = duckEnergy;
        float derivAlpha = fminf(1.0f, dt / 0.200f);
        duckFireEnergyDerivSmooth += derivAlpha
            * (energyDeriv - duckFireEnergyDerivSmooth);

        bool isSustaining = (!isSilent && duckEnergy > 0.05f
                             && fabsf(duckFireEnergyDerivSmooth) <= 0.5f);
        if (isSustaining)
            duckFireDropoutAmount = fminf(1.0f, duckFireDropoutAmount + dt * 0.35f);
        else
            duckFireDropoutAmount = fmaxf(0.0f, duckFireDropoutAmount - dt * 1.0f);

        dropoutAmount = duckFireDropoutAmount;
    }

    float colorAttack = fminf(1.0f, dt / 0.080f);
    float colorDecay  = fminf(1.0f, dt / 2.0f);

    float colorTarget;
    if (isPercussiveOnly || isSilent)
        colorTarget = 0.0f;
    else
        colorTarget = duckEnergy;

    if (colorTarget > duckFireColorEnergy)
        duckFireColorEnergy += colorAttack * (colorTarget - duckFireColorEnergy);
    else
        duckFireColorEnergy += colorDecay * (colorTarget - duckFireColorEnergy);

    float ce = duckFireColorEnergy;
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

    float base = duckFireBaseBrightness;
    if (base < DUCK_FIRE_DEADBAND) base = 0.0f;

    float s = DUCK_FIRE_FLICKER_SCALE;

    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        float fi = (float)i;

        float noise = fastSin(fi * 7.3f + t * 2.5f) *
                       fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

        float noiseAmp = fmaxf(0.15f * s, 0.10f * s / fmaxf(base, 0.1f));
        float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                        + duckFireFlickerIntensity * (noise - 0.5f) * 0.25f * s;

        float perLedDim = 0.0f;
        float colorRedShift = 0.0f;
        if (withDropout && dropoutAmount > 0.0f) {
            float resilience = fastSin(fi * 13.7f + t * 0.3f) *
                                fastSin(fi * 9.1f + t * 0.2f) * 0.5f + 0.5f;
            perLedDim = clampf(
                (dropoutAmount - resilience * 0.7f) / 0.3f, 0.0f, 1.0f
            ) * DUCK_FIRE_DROPOUT_DEPTH;

            colorRedShift = clampf(perLedDim / 0.3f, 0.0f, 1.0f);
        }

        bright *= (1.0f - perLedDim);
        bright = clampf(bright, 0.0f, 1.0f);

        float colR = baseColR * (1.0f - colorRedShift) + redR * colorRedShift;
        float colG = baseColG * (1.0f - colorRedShift) + redG * colorRedShift;
        float colB = baseColB * (1.0f - colorRedShift) + redB * colorRedShift;

        float linBright = fastGamma24(bright) * DUCK_BRIGHTNESS_CAP;
        float oR = colR * linBright;
        float oG = colG * linBright;
        float oB = colB * linBright;

        float maxCh_f = fmaxf(oR, fmaxf(oG, oB));
        float bFrac = maxCh_f / 255.0f;
        float rgbBlend = clampf((bFrac - DUCK_PURE_W_CEIL) / DUCK_PURE_W_BLEND, 0.0f, 1.0f);
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

        uint8_t r = deltaSigma(duckDsR[i], tR16);
        uint8_t g = deltaSigma(duckDsG[i], tG16);
        uint8_t b = deltaSigma(duckDsB[i], tB16);
        uint8_t w = deltaSigma(duckDsW[i], tW16);
        stripDuck.SetPixelColor(i, RgbwColor(r, g, b, w));
    }
}

// ── Duck pipeline frame driver ───────────────────────────────────
//
// Reads latestDuckPacket, derives angle/energy/onset, dispatches to the
// appropriate render fn. Writes RGBW pixels into stripDuck (Show() is
// called from the top-level render loop, AFTER both pipelines have run).
static void duckRenderFrame(float dt, uint32_t now) {
    bool connected = (duckLastPktMs > 0) && (now - duckLastPktMs < TIMEOUT_MS);

    float ax = (float)latestDuckPacket.ax;
    float ay = (float)latestDuckPacket.ay;
    float az = (float)latestDuckPacket.az;
    vecNormalize(ax, ay, az);
    uint16_t rawRms = latestDuckPacket.rawRms;
    bool micOn = latestDuckPacket.micEnabled != 0;

    if (connected) duckUpdateCalibration(ax, ay, az);

    float angleDeg = 0.0f;
    if (duckCalibrated && connected) {
        float cosAngle = clampf(
            duckRestAx * ax + duckRestAy * ay + duckRestAz * az,
            -1.0f, 1.0f);
        angleDeg = acosf(cosAngle) * (180.0f / M_PI);
    }

    if (duckCurrentAlg != DUCK_ALG_QUIET_BLOOM) {
        if (connected && micOn) {
            duckEnergy = duckComputeEnergy(rawRms);
            duckOnset  = duckComputeOnset(rawRms);
        } else if (connected && !micOn) {
            duckEnergy = 0.15f;
            duckOnset = 0.0f;
        }
    }

    if (duckCurrentAlg == DUCK_ALG_QUIET_BLOOM && connected) {
        if (duckPktCount != duckBloomPrevPktCount) {
            duckBloomProcessMotion(now);
            duckBloomPrevPktCount = duckPktCount;
        }
    }

    if (!connected) {
        // Breathing idle on warm white W-only.
        float breath = (fastSin(now / 1000.0f * M_PI) + 1.0f) / 2.0f;
        uint16_t bW = (uint16_t)(fastGamma24(breath * 0.12f) * 65535.0f);
        for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
            uint8_t w = deltaSigma(duckDsW[i], bW >> 8);
            stripDuck.SetPixelColor(i, RgbwColor(0, 0, 0, w));
        }
        return;
    }

    // Tilt → OKLCH hue (shared by all duck algorithms)
    float tiltR = 0, tiltG = 0, tiltB = 0;
    float tiltBlend = 0.0f;
    if (angleDeg > DUCK_DEADZONE_DEG) {
        float hueFrac = (angleDeg - DUCK_DEADZONE_DEG)
                        / (DUCK_MAX_ANGLE_DEG - DUCK_DEADZONE_DEG);
        if (hueFrac > 1.0f) hueFrac = 1.0f;
        uint8_t hueIdx = (uint8_t)(hueFrac * 255) % 256;
        tiltR = (float)oklchVarL[hueIdx][0];
        tiltG = (float)oklchVarL[hueIdx][1];
        tiltB = (float)oklchVarL[hueIdx][2];
        tiltBlend = (angleDeg - DUCK_DEADZONE_DEG) / DUCK_BLEND_RANGE_DEG;
        if (tiltBlend > 1.0f) tiltBlend = 1.0f;
    }

    switch (duckCurrentAlg) {
        case DUCK_ALG_SPARKLE_BURST:
            duckRenderSparkleBurst(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB);
            break;
        case DUCK_ALG_FIRE_MELD:
            duckRenderFire(dt, false, tiltBlend);
            break;
        case DUCK_ALG_FIRE_FLICKER:
            duckRenderFire(dt, true, tiltBlend);
            break;
        case DUCK_ALG_QUIET_BLOOM:
            duckRenderQuietBloom(dt, now);
            break;
    }
}

// ════════════════════════════════════════════════════════════════════
// === BIOLUM PIPELINE ===
//
//   Verbatim semantics from festicorn/biolum/src/bloom.cpp, collapsed
//   from 6 strips × 100 LEDs (12 colonies) to 1 strip × 50 LEDs
//   (2 colonies). Drives the WS2812B RGB strip on GPIO 21. Reads
//   latestV1Packet only. All globals here are prefixed `bio`.
// ════════════════════════════════════════════════════════════════════

// Strip layout (collapsed):
//   1 strip × 50 LEDs, 2 colonies × 25 LEDs each.
static const uint8_t  BIO_NUM_STRIPS         = 1;
static const uint8_t  BIO_COLONIES_PER_STRIP = 2;
static const uint8_t  BIO_NUM_COLONIES       = BIO_NUM_STRIPS * BIO_COLONIES_PER_STRIP;
static const uint16_t BIO_LEDS_PER_STRIP     = BIO_LED_COUNT;
static const uint16_t BIO_LEDS_PER_COLONY    = BIO_LEDS_PER_STRIP / BIO_COLONIES_PER_STRIP;
static const uint16_t BIO_TOTAL_LEDS         = BIO_NUM_STRIPS * BIO_LEDS_PER_STRIP;

// ── Rendering / tilt ─────────────────────────────────────────────
#define BIO_BRIGHTNESS_CAP    0.30f
#define BIO_DEADZONE_DEG     10.0f
#define BIO_MAX_ANGLE_DEG   180.0f
#define BIO_SENSOR_HZ        25.0f

// ── v1 telemetry decode constants ────────────────────────────────
#define BIO_MAG_FS              57000.0f
#define BIO_COUNTS_PER_G         8192.0f
#define BIO_COUNTS_PER_DPS         32.8f
#define BIO_COUNTS_PER_INT8       256.0f

// ── Bloom parameters (lifted verbatim from biolum/bloom.cpp) ────
#define BIO_BLOOM_BRIGHTNESS_CAP   0.70f
#define BIO_BLOOM_NOISE_GATE       256

#define BIO_SURPRISE_EMA_UP        0.05f
#define BIO_SURPRISE_EMA_DOWN      0.2f
#define BIO_SURPRISE_RATIO         3.0f
#define BIO_DRAIN_SCALE            100.0f
#define BIO_DRAIN_ENVELOPE_DECAY   0.85f
#define BIO_FLASH_MOTION_SCALE     300.0f
#define BIO_ENERGY_MULTIPLIER      1.4f
#define BIO_MOTION_SETTLE_MS       300

#define BIO_BLOOM_BREATH_MIN_PERIOD 3.0f
#define BIO_BLOOM_BREATH_MAX_PERIOD 8.0f
#define BIO_BLOOM_BREATH_MIN_PEAK   0.65f
#define BIO_BLOOM_BREATH_MAX_PEAK   1.00f
#define BIO_BLOOM_BREATH_FLOOR      0.15f

#define BIO_BLOOM_FLASH_DECAY_LO   0.96f
#define BIO_BLOOM_FLASH_DECAY_HI   0.985f

#define BIO_BLOOM_RECOVERY_RAMP    0.033f
#define BIO_BLOOM_RECOVERY_SPREAD  0.70f

#define BIO_BLOOM_HUE_A_G   20.0f
#define BIO_BLOOM_HUE_A_B  100.0f
#define BIO_BLOOM_HUE_B_G   70.0f
#define BIO_BLOOM_HUE_B_B  110.0f
#define BIO_BLOOM_FLASH_G  150.0f
#define BIO_BLOOM_FLASH_B  170.0f
#define BIO_BLOOM_PURPLE_MAX  60.0f
#define BIO_BLOOM_PURPLE_RATE 0.15f

// ── Per-LED state (50 entries) ───────────────────────────────────
static uint16_t bioDsR[BIO_TOTAL_LEDS], bioDsG[BIO_TOTAL_LEDS], bioDsB[BIO_TOTAL_LEDS];
static float    bioBloomBreathPhase[BIO_TOTAL_LEDS];
static float    bioBloomBreathRate[BIO_TOTAL_LEDS];
static float    bioBloomBreathPeak[BIO_TOTAL_LEDS];
static float    bioBloomHueT[BIO_TOTAL_LEDS];
static float    bioBloomFlashGlow[BIO_TOTAL_LEDS];
static float    bioBloomFlashDecay[BIO_TOTAL_LEDS];

// ── Per-colony state (2 entries) ─────────────────────────────────
static float    bioBloomColonyEnergy[BIO_NUM_COLONIES];
static uint32_t bioBloomLastMotionMs[BIO_NUM_COLONIES];
static float    bioDrainEnvelope[BIO_NUM_COLONIES];
static float    bioBloomHitIntensity[BIO_NUM_COLONIES];

// ── Shared motion state ──────────────────────────────────────────
static float    bioBloomMotionRate = 0.0f;
static float    bioMotionEMA = 0.0f;
static uint32_t bioBloomPrevPktCount = 0;

// ── Calibration ──────────────────────────────────────────────────
static float bioRestAx = 0, bioRestAy = 0, bioRestAz = 1.0f;
static bool  bioCalibrated = false;
static float bioCalSumAx = 0, bioCalSumAy = 0, bioCalSumAz = 0;
static uint32_t bioCalSamples = 0;
static uint32_t bioCalStartMs = 0;
#define BIO_CAL_DURATION_MS 2000

// ── v1 decode helpers ────────────────────────────────────────────
static inline float bioMagCountsFromByte(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * BIO_MAG_FS;
}
static inline float bioAmagGFromByte(uint8_t b) {
    return bioMagCountsFromByte(b) / BIO_COUNTS_PER_G;
}
static inline float bioGmagDpsFromByte(uint8_t b) {
    return bioMagCountsFromByte(b) / BIO_COUNTS_PER_DPS;
}
static inline float bioAxisMeanG(int8_t m) {
    return ((float)m * BIO_COUNTS_PER_INT8) / BIO_COUNTS_PER_G;
}

// ── Calibration ──────────────────────────────────────────────────
static void bioStartCalibration() {
    bioCalibrated = false;
    bioCalSumAx = 0; bioCalSumAy = 0; bioCalSumAz = 0;
    bioCalSamples = 0;
    bioCalStartMs = millis();
}

static void bioUpdateCalibration(float ax, float ay, float az) {
    if (bioCalStartMs == 0) bioStartCalibration();
    if (!bioCalibrated) {
        bioCalSumAx += ax; bioCalSumAy += ay; bioCalSumAz += az;
        bioCalSamples++;
        if (millis() - bioCalStartMs >= BIO_CAL_DURATION_MS && bioCalSamples > 0) {
            bioRestAx = bioCalSumAx / bioCalSamples;
            bioRestAy = bioCalSumAy / bioCalSamples;
            bioRestAz = bioCalSumAz / bioCalSamples;
            vecNormalize(bioRestAx, bioRestAy, bioRestAz);
            bioCalibrated = true;
        }
    }
}

// ── Bloom motion processing ──────────────────────────────────────
static void bioBloomProcessMotion(uint32_t now) {
    float gyroRate = bioGmagDpsFromByte(latestV1Packet.gmag_max);
    float amagG    = bioAmagGFromByte(latestV1Packet.amag_max);
    float accelJolt = fabsf(amagG - 1.0f);
    bioBloomMotionRate = fmaxf(gyroRate, accelJolt * 300.0f);

    float surprise = fmaxf(0.0f, bioBloomMotionRate - bioMotionEMA * BIO_SURPRISE_RATIO);

    float alpha = (bioBloomMotionRate > bioMotionEMA)
        ? BIO_SURPRISE_EMA_UP : BIO_SURPRISE_EMA_DOWN;
    bioMotionEMA += alpha * (bioBloomMotionRate - bioMotionEMA);

    if (surprise > 1.0f) {
        float hitIntensity = clampf(
            log2f(1.0f + surprise) / log2f(1.0f + BIO_FLASH_MOTION_SCALE),
            0.0f, 1.0f);
        float normMotion = surprise / BIO_DRAIN_SCALE;
        float newDrain = normMotion * normMotion * normMotion
                       * (1.0f - BIO_DRAIN_ENVELOPE_DECAY);

        for (uint8_t c = 0; c < BIO_NUM_COLONIES; c++) {
            bioBloomLastMotionMs[c] = now;
            if (hitIntensity > bioBloomHitIntensity[c]) bioBloomHitIntensity[c] = hitIntensity;
            if (newDrain > bioDrainEnvelope[c]) bioDrainEnvelope[c] = newDrain;
        }
    }
}

// ── Wave pulse: tap spawns leaves at LED 0, travel toward LED N ──
// Verbatim semantics from biolum/bloom.cpp's renderWavePulse, collapsed
// from per-strip arrays (NUM_STRIPS × WP_MAX_LEAVES) to one strip.
#define BIO_WP_MAX_LEAVES       16
#define BIO_WP_WIND_SPEED        6.0f
#define BIO_WP_TURBULENCE        0.5f
#define BIO_WP_DAMPING           0.92f
#define BIO_WP_BOOST_SPEED      28.0f
#define BIO_WP_BOOST_TC          1.5f
#define BIO_WP_LEAF_SIGMA        1.5f
#define BIO_WP_FADE_IN_TIME      0.20f
#define BIO_WP_FADE_OUT_LEDS     8.0f
#define BIO_WP_MAX_BRIGHTNESS    0.85f
#define BIO_WP_STALL_RADIUS      2.5f
#define BIO_WP_STALL_TIMEOUT     3.0f
#define BIO_WP_DEFAULT_R        50.0f
#define BIO_WP_DEFAULT_G       200.0f
#define BIO_WP_DEFAULT_B        80.0f
#define BIO_WP_TAP_THRESH_G      0.50f
#define BIO_WP_COOLDOWN_MS       120

struct BioWPLeaf {
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

static BioWPLeaf bioWpLeaves[BIO_WP_MAX_LEAVES];
static float     bioWpTime         = 0.0f;
static uint32_t  bioWpLastHitMs    = 0;
static uint32_t  bioWpPrevPktCount = 0;

static void bioWpPickColor(float angleDeg, float &cR, float &cG, float &cB) {
    if (angleDeg <= BIO_DEADZONE_DEG) {
        cR = BIO_WP_DEFAULT_R; cG = BIO_WP_DEFAULT_G; cB = BIO_WP_DEFAULT_B;
        return;
    }
    float hueFrac = (angleDeg - BIO_DEADZONE_DEG) / (BIO_MAX_ANGLE_DEG - BIO_DEADZONE_DEG);
    if (hueFrac > 1.0f) hueFrac = 1.0f;
    uint8_t hueIdx = (uint8_t)(hueFrac * 255.0f);
    cR = (float)oklchVarL[hueIdx][0];
    cG = (float)oklchVarL[hueIdx][1];
    cB = (float)oklchVarL[hueIdx][2];
}

// Per-packet tap test: peak per-axis AC excursion in g.
// Sender encodes (rawMax - rawMean) >> 8 per axis → 256 raw counts/step.
// At ±4g, 256 / 8192 ≈ 31.25 mg per int8 step.
static void bioWavePulseProcessMotion(uint32_t now, float angleDeg) {
    float ax_pk = fmaxf(fabsf((float)latestV1Packet.ax_max),
                        fabsf((float)latestV1Packet.ax_min));
    float ay_pk = fmaxf(fabsf((float)latestV1Packet.ay_max),
                        fabsf((float)latestV1Packet.ay_min));
    float az_pk = fmaxf(fabsf((float)latestV1Packet.az_max),
                        fabsf((float)latestV1Packet.az_min));
    float peak_axis_ac_g = fmaxf(ax_pk, fmaxf(ay_pk, az_pk))
                         * (BIO_COUNTS_PER_INT8 / BIO_COUNTS_PER_G);

    bool tap = (peak_axis_ac_g >= BIO_WP_TAP_THRESH_G);
    if (!tap) return;
    if ((now - bioWpLastHitMs) < BIO_WP_COOLDOWN_MS) return;

    bioWpLastHitMs = now;

    Serial.printf("[tap] peak=%.2fg  ax=%+d/%+d ay=%+d/%+d az=%+d/%+d  "
                  "grav=(%+d,%+d,%+d)  angle=%.0f\n",
                  peak_axis_ac_g,
                  latestV1Packet.ax_max, latestV1Packet.ax_min,
                  latestV1Packet.ay_max, latestV1Packet.ay_min,
                  latestV1Packet.az_max, latestV1Packet.az_min,
                  latestV1Packet.ax_mean, latestV1Packet.ay_mean,
                  latestV1Packet.az_mean, angleDeg);

    float overshoot = (peak_axis_ac_g - BIO_WP_TAP_THRESH_G) / 2.5f;
    float hitIntensity = clampf(overshoot, 0.0f, 1.0f);
    int nLeaves = 1 + (int)(hitIntensity * 3.0f + 0.5f);
    if (nLeaves > 4) nLeaves = 4;

    float baseR, baseG, baseB;
    bioWpPickColor(angleDeg, baseR, baseG, baseB);

    for (int n = 0; n < nLeaves; n++) {
        int8_t slot = -1;
        for (uint8_t j = 0; j < BIO_WP_MAX_LEAVES; j++) {
            if (!bioWpLeaves[j].active) { slot = j; break; }
        }
        if (slot < 0) break;

        BioWPLeaf &lf = bioWpLeaves[slot];
        lf.pos       = -(float)n * 0.7f - randFloat() * 0.5f;
        lf.vel       = 0.0f;
        lf.boost     = BIO_WP_BOOST_SPEED
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

static void bioRenderWavePulse(float dt) {
    // Wrap at 1000·2π so float32 precision never collapses dt into a no-op
    // (would freeze noise after ~1 day). t feeds only fastSin(); phase jump
    // at wrap is invisible in chaotic noise.
    bioWpTime = fmodf(bioWpTime + dt, 6283.1853f);
    float t = bioWpTime;

    float boostDecay = expf(-dt / BIO_WP_BOOST_TC);
    float sigma_sq2  = 2.0f * BIO_WP_LEAF_SIGMA * BIO_WP_LEAF_SIGMA;

    // Step 1: advance every active leaf.
    for (uint8_t i = 0; i < BIO_WP_MAX_LEAVES; i++) {
        BioWPLeaf &lf = bioWpLeaves[i];
        if (!lf.active) continue;

        float noiseVal = fastSin(lf.pos * 0.4f  + t * 0.3f  + (float)(i * 7))
                       * fastSin(lf.pos * 0.17f - t * 0.19f + (float)(i * 3)) * 0.5f
                       + fastSin(lf.pos * 0.09f + t * 0.13f + (float)(i * 1)) * 0.33f;
        float speedMult = fmaxf(1.0f + noiseVal * BIO_WP_TURBULENCE, 0.05f);

        float force = BIO_WP_WIND_SPEED * speedMult;
        lf.vel = lf.vel * BIO_WP_DAMPING + force * (1.0f - BIO_WP_DAMPING);
        lf.boost *= boostDecay;
        lf.pos  += (lf.vel + lf.boost) * dt;
        lf.age  += dt;

        float elapsed = lf.age - lf.refTime;
        if (elapsed > BIO_WP_STALL_TIMEOUT) {
            if (fabsf(lf.pos - lf.refPos) < BIO_WP_STALL_RADIUS) {
                lf.active = false; continue;
            }
            lf.refPos  = lf.pos;
            lf.refTime = lf.age;
        }

        lf.brightness = (lf.age < BIO_WP_FADE_IN_TIME)
                        ? (lf.age / BIO_WP_FADE_IN_TIME) : 1.0f;

        float distToEnd = (float)(BIO_LEDS_PER_STRIP - 1) - lf.pos;
        if (distToEnd < BIO_WP_FADE_OUT_LEDS && distToEnd >= 0.0f)
            lf.brightness *= distToEnd / BIO_WP_FADE_OUT_LEDS;

        // Deactivate immediately upon reaching the end — rendering past
        // the end with brightness=1.0 lets the gaussian tail bleed back
        // into the last LEDs at high weight.
        if (lf.pos >= (float)(BIO_LEDS_PER_STRIP - 1))
            lf.active = false;
    }

    // Step 2: render pixels from leaf array.
    RgbColor* px = (RgbColor*)stripBio.Pixels();
    for (uint16_t li = 0; li < BIO_LEDS_PER_STRIP; li++) {
        float glow = 0.0f, oR = 0.0f, oG = 0.0f, oB = 0.0f;

        for (uint8_t j = 0; j < BIO_WP_MAX_LEAVES; j++) {
            const BioWPLeaf &lf = bioWpLeaves[j];
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
            float bright = clampf(glow, 0.0f, 1.0f) * BIO_WP_MAX_BRIGHTNESS;
            float linBright = fastGamma24(bright) * BIO_BRIGHTNESS_CAP;
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

        px[li].R = deltaSigma(bioDsR[li], tR16);
        px[li].G = deltaSigma(bioDsG[li], tG16);
        px[li].B = deltaSigma(bioDsB[li], tB16);
    }
    stripBio.Dirty();
}

// ── Bloom render (single strip, 2 colonies) ─────────────────────
static void bioRenderQuietBloom(float dt, uint32_t now) {
    bool colonyDraining[BIO_NUM_COLONIES];
    for (uint8_t c = 0; c < BIO_NUM_COLONIES; c++) {
        colonyDraining[c] = bioDrainEnvelope[c] > 0.001f;
        if (!colonyDraining[c]) bioBloomHitIntensity[c] = 0.0f;
        if (now - bioBloomLastMotionMs[c] > BIO_MOTION_SETTLE_MS) {
            bioBloomColonyEnergy[c] = fminf(
                1.0f, bioBloomColonyEnergy[c] + BIO_BLOOM_RECOVERY_RAMP * dt);
        }
    }

    const float purpleTimeBase = (float)now * (0.001f * BIO_BLOOM_PURPLE_RATE);
    const float decayExp = dt * 30.0f;

    // Single strip — biolum's per-strip phase jitter loop is collapsed.
    RgbColor* px = (RgbColor*)stripBio.Pixels();

    for (uint8_t seg = 0; seg < BIO_COLONIES_PER_STRIP; seg++) {
        uint8_t  c = seg;  // single strip → colony index = segment index
        bool     draining = colonyDraining[c];
        float    colonyEnergy = bioBloomColonyEnergy[c];
        float    hitIntensity = bioBloomHitIntensity[c];
        uint16_t segStart = seg * BIO_LEDS_PER_COLONY;
        uint16_t segEnd   = segStart + BIO_LEDS_PER_COLONY;

        for (uint16_t li = segStart; li < segEnd; li++) {
            uint16_t i = li;  // single strip → LED index = local index

            if (!draining) {
                bioBloomFlashGlow[i] *= fastDecay(bioBloomFlashDecay[i], decayExp);
                if (bioBloomFlashGlow[i] < 0.005f) bioBloomFlashGlow[i] = 0.0f;
            }

            bioBloomBreathPhase[i] += dt * bioBloomBreathRate[i];
            if (bioBloomBreathPhase[i] >= 1.0f) bioBloomBreathPhase[i] -= 1.0f;

            float h = bioBloomHueT[i];
            float wakeThresh = h * BIO_BLOOM_RECOVERY_SPREAD;
            float ledRecovery = clampf(
                (colonyEnergy - wakeThresh) * (1.0f / 0.30f), 0.0f, 1.0f);

            float breath = (fastSinPhase(bioBloomBreathPhase[i]) * 0.5f + 0.5f);
            float breathGlow = BIO_BLOOM_BREATH_FLOOR
                + breath * (bioBloomBreathPeak[i] - BIO_BLOOM_BREATH_FLOOR);
            breathGlow *= ledRecovery;

            if (draining) {
                float target = breathGlow * hitIntensity * BIO_ENERGY_MULTIPLIER;
                if (target > bioBloomFlashGlow[i]) {
                    bioBloomFlashGlow[i] = target;
                    bioBloomFlashDecay[i] = BIO_BLOOM_FLASH_DECAY_LO
                        + randFloat() * (BIO_BLOOM_FLASH_DECAY_HI - BIO_BLOOM_FLASH_DECAY_LO);
                }
            }

            float g = fmaxf(breathGlow, bioBloomFlashGlow[i]);

            float flashFrac = (bioBloomFlashGlow[i] > breathGlow) ? 1.0f : 0.0f;
            float colG = lerpf(lerpf(BIO_BLOOM_HUE_A_G, BIO_BLOOM_HUE_B_G, h),
                               BIO_BLOOM_FLASH_G, flashFrac);
            float colB = lerpf(lerpf(BIO_BLOOM_HUE_A_B, BIO_BLOOM_HUE_B_B, h),
                               BIO_BLOOM_FLASH_B, flashFrac);

            float purplePhase = purpleTimeBase + h * 4.0f;
            float purpleMix = (fastSin(purplePhase) + 1.0f) * 0.5f;
            float colR = purpleMix * BIO_BLOOM_PURPLE_MAX;

            float linBright = fastGamma24(g) * BIO_BLOOM_BRIGHTNESS_CAP;
            float oR = colR * linBright;
            float oG = colG * linBright;
            float oB = colB * linBright;

            uint16_t tR16 = (uint16_t)clampf(oR * 256.0f, 0, 65535);
            uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
            uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);

            if (tR16 < BIO_BLOOM_NOISE_GATE) tR16 = 0;
            if (tG16 < BIO_BLOOM_NOISE_GATE) tG16 = 0;
            if (tB16 < BIO_BLOOM_NOISE_GATE) tB16 = 0;

            // NeoGrbFeature.RgbColor: R/G/B fields wired to GRB on the wire.
            px[li].R = deltaSigma(bioDsR[i], tR16);
            px[li].G = deltaSigma(bioDsG[i], tG16);
            px[li].B = deltaSigma(bioDsB[i], tB16);
        }
    }

    stripBio.Dirty();

    for (uint8_t c = 0; c < BIO_NUM_COLONIES; c++) {
        if (bioDrainEnvelope[c] > 0.001f) {
            float drain = bioDrainEnvelope[c] * dt;
            drain = fminf(drain, bioBloomColonyEnergy[c]);
            bioBloomColonyEnergy[c] -= drain;
            bioDrainEnvelope[c] *= BIO_DRAIN_ENVELOPE_DECAY;
            if (bioDrainEnvelope[c] <= 0.001f) bioDrainEnvelope[c] = 0.0f;
        }
    }
}

// ── Biolum pipeline frame driver ─────────────────────────────────
static void bioRenderFrame(float dt, uint32_t now) {
    bool connected = (bioLastPktMs > 0) && (now - bioLastPktMs < TIMEOUT_MS);

    float ax = bioAxisMeanG(latestV1Packet.ax_mean);
    float ay = bioAxisMeanG(latestV1Packet.ay_mean);
    float az = bioAxisMeanG(latestV1Packet.az_mean);
    vecNormalize(ax, ay, az);
    if (connected) bioUpdateCalibration(ax, ay, az);

    float angleDeg = 0.0f;
    if (bioCalibrated) {
        float cosAngle = clampf(
            bioRestAx*ax + bioRestAy*ay + bioRestAz*az, -1.0f, 1.0f);
        angleDeg = acosf(cosAngle) * (180.0f / M_PI);
    }

    if (connected && bioPktCount != bioWpPrevPktCount) {
        bioWavePulseProcessMotion(now, angleDeg);
        bioWpPrevPktCount = bioPktCount;
    }

    if (!connected) {
        // Faint blue breathing idle on the bio strip when the v1 sender is silent.
        float breath = (fastSin(now / 1000.0f * M_PI) + 1.0f) / 2.0f;
        uint16_t bB = (uint16_t)(fastGamma24(breath * 0.10f) * 65535.0f);
        RgbColor* px = (RgbColor*)stripBio.Pixels();
        for (uint16_t i = 0; i < BIO_TOTAL_LEDS; i++) {
            px[i].R = 0;
            px[i].G = 0;
            px[i].B = deltaSigma(bioDsB[i], bB >> 8);
        }
        stripBio.Dirty();
        return;
    }

    bioRenderWavePulse(dt);
}

// ════════════════════════════════════════════════════════════════════
//   SETUP + MAIN RENDER LOOP
// ════════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(460800);
    delay(300);
    Serial.println();
    Serial.println("bench-bulbs — dual-pipeline lab firmware");

    // ── PRNG ─────────────────────────────────────────────────────
    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    // ── Strip init ───────────────────────────────────────────────
    stripDuck.Begin();
    stripBio.Begin();
    stripDuck.ClearTo(RgbwColor(0, 0, 0, 0));
    stripBio.ClearTo(RgbColor(0, 0, 0));
    stripDuck.Show();
    stripBio.Show();

    // Seed delta-sigma accumulators (duck side, RGBW).
    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / DUCK_LED_COUNT);
        duckDsR[i] = seed;
        duckDsG[i] = (seed +  64) & 0xFF;
        duckDsB[i] = (seed + 128) & 0xFF;
        duckDsW[i] = (seed + 192) & 0xFF;
    }
    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++) {
        duckSparkle[i]    = 0.0f;
        duckDecayRates[i] = 0.92f + randFloat() * 0.05f;
    }
    duckResetBloomState();

    // Seed delta-sigma + bloom state (bio side, RGB).
    for (uint16_t i = 0; i < BIO_TOTAL_LEDS; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / BIO_TOTAL_LEDS);
        bioDsR[i] = seed;
        bioDsG[i] = (seed +  85) & 0xFF;
        bioDsB[i] = (seed + 170) & 0xFF;

        bioBloomBreathPhase[i] = randFloat();
        float period = BIO_BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BIO_BLOOM_BREATH_MAX_PERIOD - BIO_BLOOM_BREATH_MIN_PERIOD);
        bioBloomBreathRate[i]  = 1.0f / period;
        bioBloomBreathPeak[i]  = BIO_BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BIO_BLOOM_BREATH_MAX_PEAK - BIO_BLOOM_BREATH_MIN_PEAK);
        bioBloomHueT[i]        = randFloat();
        bioBloomFlashGlow[i]   = 0.0f;
        bioBloomFlashDecay[i]  = BIO_BLOOM_FLASH_DECAY_LO
            + randFloat() * (BIO_BLOOM_FLASH_DECAY_HI - BIO_BLOOM_FLASH_DECAY_LO);
    }
    for (uint8_t c = 0; c < BIO_NUM_COLONIES; c++) {
        bioBloomColonyEnergy[c] = 1.0f;
        bioBloomLastMotionMs[c] = 0;
        bioDrainEnvelope[c]     = 0.0f;
        bioBloomHitIntensity[c] = 0.0f;
    }

    // Boot flash — green on duck, blue on bio so you can tell at a glance
    // both strips are wired correctly.
    for (uint16_t i = 0; i < DUCK_LED_COUNT; i++)
        stripDuck.SetPixelColor(i, RgbwColor(0, 40, 0, 0));
    stripBio.ClearTo(RgbColor(0, 0, 40));
    stripDuck.Show();
    stripBio.Show();
    delay(200);
    stripDuck.ClearTo(RgbwColor(0, 0, 0, 0));
    stripBio.ClearTo(RgbColor(0, 0, 0));
    stripDuck.Show();
    stripBio.Show();

    // ── Wi-Fi: STA mode + SSID-based channel discovery (no AP join) ──
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);  // C3 super-mini regulator brownout fix

    uint8_t ch = scanForSsidChannel(WIFI_SSID_TARGET);
    if (ch) {
        Serial.printf("Found '%s' on ch=%u\n", WIFI_SSID_TARGET, ch);
    } else {
        Serial.printf("'%s' not visible — falling back to ch=%u\n",
                      WIFI_SSID_TARGET, CHANNEL_FALLBACK);
    }
    applyChannel(ch ? ch : CHANNEL_FALLBACK);
    lastScanMs = millis();

    // ── ESP-NOW init (single callback dispatches by length) ──────
    esp_err_t initResult = esp_now_init();
    esp_err_t cbResult   = esp_now_register_recv_cb(onReceive);
    Serial.printf("ESP-NOW init=%d cb=%d ch=%u MAC=%s\n",
        initResult, cbResult, currentChannel, WiFi.macAddress().c_str());
    Serial.printf("Strip A (RGBW, GPIO %u): %u LEDs — duck pipeline\n",
        DUCK_LED_PIN, DUCK_LED_COUNT);
    Serial.printf("Strip B (RGB,  GPIO %u): %u LEDs — biolum pipeline\n",
        BIO_LED_PIN, BIO_LED_COUNT);
    Serial.println("Awaiting both senders...");
}

// ── Render loop (target ~200 FPS) ────────────────────────────────
//
// Single-core C3: render both pipelines back-to-back, then queue both
// Show()s to RMT. NeoPixelBus's Show(false) is non-blocking — once the
// data is in the RMT FIFO the strips transmit in parallel hardware-side.
//
// Frame budget @ 200 Hz: 5.0 ms.
//   parallel RMT TX: ~2.0 ms (RGBW dominates: 50 × 32 bits × 1.25 µs)
//   render budget:   ~3.0 ms for both pipelines
//
// We don't busy-wait — the loop falls through and FreeRTOS gets the slack
// (critical on single-core C3 to keep WiFi healthy).
static const uint32_t FRAME_INTERVAL_US = 5000;  // 200 FPS target

void loop() {
    static uint32_t lastFrameUs = 0;
    static uint32_t lastRenderMs = 0;

    uint32_t nowUs = micros();
    if (lastFrameUs && (nowUs - lastFrameUs) < FRAME_INTERVAL_US) {
        delay(1);
        return;
    }
    lastFrameUs = nowUs;

    uint32_t now = millis();
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f
                                  : (1.0f / DUCK_SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    // === DUCK PIPELINE ===
    duckRenderFrame(dt, now);

    // === BIOLUM PIPELINE ===
    bioRenderFrame(dt, now);

    // Queue both transmissions; RMT hardware overlaps them.
    stripDuck.Show();
    stripBio.Show();

    // Periodic SSID rescan — heals if the AP moves channel.
    if (now - lastScanMs > CHANNEL_RESCAN_MS) {
        uint8_t newCh = scanForSsidChannel(WIFI_SSID_TARGET);
        if (newCh && newCh != currentChannel) {
            Serial.printf("[heal] channel drift %u -> %u\n", currentChannel, newCh);
            applyChannel(newCh);
        }
        lastScanMs = millis();
    }

    // ── Telemetry every ~1 s ─────────────────────────────────────
    static uint32_t lastTelemetryMs = 0;
    static uint32_t frameCount = 0;
    static uint32_t lastTelemetryFrames = 0;
    frameCount++;
    if (now - lastTelemetryMs >= 1000) {
        uint32_t fps = frameCount - lastTelemetryFrames;
        lastTelemetryFrames = frameCount;
        lastTelemetryMs = now;

        bool duckConn = (duckLastPktMs > 0) && (now - duckLastPktMs < TIMEOUT_MS);
        bool bioConn  = (bioLastPktMs  > 0) && (now - bioLastPktMs  < TIMEOUT_MS);
        Serial.printf("[bench] %ufps duck=%s(%u) bio=%s(%u) ch=%u\n",
            fps,
            duckConn ? "ok" : "--", (unsigned)duckPktCount,
            bioConn  ? "ok" : "--", (unsigned)bioPktCount,
            currentChannel);
    }
}
