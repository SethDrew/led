/*
 * BENCH-BULBS — phone WiFi UDP architecture (adapted from road-bulbs).
 *
 * Phone (Sensor Logger app) → WiFi UDP → ESP32 listener. No ESP-NOW.
 * Connects to phone hotspot SSID "cuteplant".
 *
 * Strip A: 50× SK6812 RGBW on GPIO 4   (main output, NeoPixelBus, RMT0)
 * Strip B: 50× WS2812  RGB  on GPIO 18 (mirror of A, W dropped, RMT1)
 * Stick A: 12× WS2812  RGB  on GPIO 32 (green breathe, RMT2)
 * Stick B: 11× WS2812  RGB  on GPIO 12 (green breathe, RMT3)
 *
 * UDP ports: 4210 sensor, 4211 cmd, 4212 discovery, 4213 onset.
 * Effects ported verbatim from road-bulbs/src/receiver.cpp.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <ESPmDNS.h>
#include <AsyncUDP.h>
#include <esp_now.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <esp_random.h>
#include <esp_system.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <oklch_lut.h>
#include <delta_sigma.h>
#include <fast_math.h>
#include "v1_packet.h"

// ── WiFi credentials (phone hotspot) ────────────────────────────
#ifndef WIFI_SSID
#define WIFI_SSID "cuteplant"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS "bigboiredwood"
#endif

static AsyncUDP udp;
static AsyncUDP cmdUdp;
static AsyncUDP discoverUdp;
static AsyncUDP onsetUdp;
#define UDP_PORT 4210
#define CMD_PORT 4211
#define DISCOVER_PORT 4212
#define ONSET_PORT 4213
static portMUX_TYPE pktMux = portMUX_INITIALIZER_UNLOCKED;

// ── LED config ───────────────────────────────────────────────────
#define LED_COUNT  50

// ── Global sliders (set via UDP cmd) ────────────────────────────
static float globalBrightness  = 1.0f;   // 0.0–2.0, midpoint=1.0 unchanged
static float globalSensitivity = 1.0f;   // 0.05–2.0, scales RMS_CEILING

#define DUCK_LED_PIN   4
#define BIO_LED_PIN    18
#define STICK_A_PIN    32
#define STICK_A_COUNT  12
#define STICK_B_PIN    12
#define STICK_B_COUNT  11

static NeoPixelBus<NeoGrbwFeature, NeoEsp32Rmt0Sk6812Method>  stripA (LED_COUNT,      DUCK_LED_PIN);
static NeoPixelBus<NeoRgbFeature,  NeoEsp32Rmt1Ws2812xMethod> stripB (LED_COUNT,      BIO_LED_PIN);
static NeoPixelBus<NeoGrbFeature,  NeoEsp32Rmt2Ws2812xMethod> stripStkA (STICK_A_COUNT, STICK_A_PIN);
static NeoPixelBus<NeoGrbFeature,  NeoEsp32Rmt3Ws2812xMethod> stripStkB (STICK_B_COUNT, STICK_B_PIN);

// Low-level raw writer: bytes straight to both strips, no processing.
// Used by ALG_RAW_COLOR and twinkle's pure-white asymmetric path.
static inline void setBothPixel(uint16_t i, uint8_t r, uint8_t g, uint8_t b, uint8_t w) {
    if (globalBrightness < 1.0f) {
        uint8_t ir = r, ig = g, ib = b, iw = w;
        r = (uint8_t)(r * globalBrightness);
        g = (uint8_t)(g * globalBrightness);
        b = (uint8_t)(b * globalBrightness);
        w = (uint8_t)(w * globalBrightness);
        if (r | g | b | w) {
            if (ir && !r) r = 1;
            if (ig && !g) g = 1;
            if (ib && !b) b = 1;
            if (iw && !w) w = 1;
        }
    }
    stripA.SetPixelColor(i, RgbwColor(r, g, b, w));
    stripB.SetPixelColor(i, RgbColor(r, g, b));
}

// outputPixel — unified output stage — defined after clampf / dsR
// later in the file. Effects produce floats in 0..255 (post-gamma,
// post per-effect cap); outputPixel handles brightness scale,
// coordinated noise gate, optional dither, proportional floor.
static int outputDitherEnabled = 0;
#define OUTPUT_NOISE_GATE_F 1.0f
static void outputPixel(uint16_t i, float r, float g, float b, float w);

// ── Rendering ────────────────────────────────────────────────────
#define GAMMA 2.4f
#define BRIGHTNESS_CAP 0.25f
#define SPARKLE_BRIGHTNESS_CAP_DEF 0.05f
static float sparkleBrightnessCap = SPARKLE_BRIGHTNESS_CAP_DEF;
#define PURE_W_CEIL    0.10f
#define PURE_W_BLEND   0.15f

// ── Color / tilt mapping ─────────────────────────────────────────
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f
#define BLEND_RANGE_DEG 40.0f
#define SENSOR_HZ       25.0f

// ── FixedRangeRMS parameters ─────────────────────────────────────
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

// ── Sparkle ──────────────────────────────────────────────────────
#define SPARKLE_DEADBAND 0.08f

// ── Sparkle twinkle (variant B: crisp snap-on / fast snap-off) ───
// Free-running ambient sparkle. Independent of audio/motion.
#define TWINKLE_SPAWN_RATE_DEF   60.0f
#define TWINKLE_ATTACK_S_DEF     0.067f
#define TWINKLE_TAU_S_DEF        0.10f
#define TWINKLE_PEAK_MIN_DEF     0.6f
#define TWINKLE_PEAK_MAX_DEF     1.0f
static float twinkleSpawnRate = TWINKLE_SPAWN_RATE_DEF;
static float twinkleAttackS   = TWINKLE_ATTACK_S_DEF;
static float twinkleTauS      = TWINKLE_TAU_S_DEF;
static float twinklePeakMin   = TWINKLE_PEAK_MIN_DEF;
static float twinklePeakMax   = TWINKLE_PEAK_MAX_DEF;
// Color: clean white. Easy to tune — these are the per-channel 0..255
// weights applied before brightness. On RGBW Strip A the white rides the
// W channel; on RGB Strip B setBothPixel drops W so RGB carries it.
#define TWINKLE_COL_R          0.0f
#define TWINKLE_COL_G          0.0f
#define TWINKLE_COL_B          0.0f
#define TWINKLE_COL_W        255.0f
// RGB fallback (Strip B has no W) — drive r=g=b so it still shows white.
#define TWINKLE_RGB_FALLBACK 255.0f

// ── Fire ─────────────────────────────────────────────────────────
#define FIRE_FLICKER_SCALE  3.0f
#define FIRE_DEADBAND       0.08f
#define FIRE_DROPOUT_DEPTH  0.85f

// ── Quiet bloom ──────────────────────────────────────────────────
// Output stage — single brightness control, applied after gamma.
// Effect code works in 0.0–1.0 intensity; color palette in 0–255.
#define BLOOM_BRIGHTNESS_CAP_DEF   0.50f
#define BLOOM_FLASH_BUMP_DEF       0.06f
#define BLOOM_ACCEL_THRESH_DEF     1.5f
#define BLOOM_FLASH_DECAY_RATE_DEF 1.5f
static float bloomBrightnessCap  = BLOOM_BRIGHTNESS_CAP_DEF;
static float bloomFlashBump      = BLOOM_FLASH_BUMP_DEF;
static float bloomAccelThresh    = BLOOM_ACCEL_THRESH_DEF;
static float bloomFlashDecayRate = BLOOM_FLASH_DECAY_RATE_DEF;
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
#define BLOOM_BREATH_FLOOR_DEF      0.15f
static float bloomBreathFloor = BLOOM_BREATH_FLOOR_DEF;

#define BLOOM_FLASH_DECAY_LO   0.96f
#define BLOOM_FLASH_DECAY_HI   0.985f

#define BLOOM_RECOVERY_RAMP    0.033f
#define BLOOM_RECOVERY_SPREAD  0.70f

// Bloom palette — full-range 0–255 colors. Brightness is NOT baked in here;
// the output stage applies BLOOM_BRIGHTNESS_CAP separately after gamma.
//
// Per-LED hue position h ∈ [0,1] interpolates between A and B endpoints.
// Sweep: deep teal (h=0) → blue (h≈0.5) → violet (h=1).
// Flash color: bright lavender (high-energy hit response).
#define BLOOM_HUE_A_R    0.0f
#define BLOOM_HUE_A_G  180.0f
#define BLOOM_HUE_A_B  120.0f

#define BLOOM_HUE_B_R  140.0f
#define BLOOM_HUE_B_G   20.0f
#define BLOOM_HUE_B_B  255.0f

#define BLOOM_FLASH_R  200.0f
#define BLOOM_FLASH_G  120.0f
#define BLOOM_FLASH_B  255.0f

#define BLOOM_W_SCALE  255.0f

// Hue drift — each LED slowly wanders through the palette.
// Full cycle takes 15–45 seconds. Direction randomized per LED.
#define BLOOM_HUE_DRIFT_MIN  (1.0f / 45.0f)
#define BLOOM_HUE_DRIFT_MAX  (1.0f / 15.0f)

// ── Timeouts ─────────────────────────────────────────────────────
#define TIMEOUT_MS     500

// ── v1 telemetry decode ─────────────────────────────────────────
#define MAG_FS              57000.0f
#define COUNTS_PER_G         8192.0f
#define COUNTS_PER_DPS         32.8f
#define COUNTS_PER_INT8       256.0f

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

// ── ESP-NOW v1 packet state ─────────────────────────────────────
static volatile uint32_t espnowPktCount = 0;
static volatile uint32_t espnowLastMs = 0;
static TelemetryPacketV1 espnowPacket = {0};
// Track max motion across all senders since last consume
static volatile uint8_t espnowPeakAmag = 0;
static volatile uint8_t espnowPeakGmag = 0;

void onEspNowReceive(const uint8_t *mac, const uint8_t *data, int len) {
    espnowPktCount++;
    if (len == sizeof(TelemetryPacketV1)) {
        TelemetryPacketV1 pkt;
        memcpy(&pkt, data, sizeof(TelemetryPacketV1));
        if (pkt.amag_max > espnowPeakAmag) espnowPeakAmag = pkt.amag_max;
        if (pkt.gmag_max > espnowPeakGmag) espnowPeakGmag = pkt.gmag_max;
        memcpy((void*)&espnowPacket, &pkt, sizeof(TelemetryPacketV1));
        espnowLastMs = millis();
    }
}

struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    uint16_t rawRms;
    uint8_t micEnabled;
};  // 15 bytes

enum Algorithm {
    ALG_OFF,
    ALG_SPARKLE_BURST,
    ALG_FIRE_MELD,
    ALG_FIRE_FLICKER,
    ALG_QUIET_BLOOM,
    ALG_GRAVITY_PARTICLE,
    ALG_SPARKLE_SYLLABLE,
    ALG_SPARKLE_TWINKLE,
    ALG_IDLE,
    ALG_WHITE_STRESS,
    ALG_RAW_COLOR,
};

static uint8_t rawR = 0, rawG = 0, rawB = 0, rawW = 0;
static uint8_t rawBrightness = 255;
static bool rawDither = false;
static uint16_t rawTargetFps = 150;

static Algorithm currentAlg = ALG_GRAVITY_PARTICLE;

// ── Gravity sparkle ──────────────────────────────────────────────
#define GS_PARTICLE_COUNT_DEF 7
#define GS_PARTICLE_COUNT_MAX 32
#define GS_GRAVITY_SCALE_DEF  40.0f
#define GS_VELOCITY_DAMP_DEF  0.92f
#define GS_BOUNCE_REBOUND_DEF 0.5f
#define GS_SPLAT_RADIUS_DEF   2.5f
#define GS_BRIGHTNESS_CAP_DEF 0.45f
static int   gsParticleCount  = GS_PARTICLE_COUNT_DEF;
static float gsGravityScale   = GS_GRAVITY_SCALE_DEF;
static float gsVelocityDamp   = GS_VELOCITY_DAMP_DEF;
static float gsBounceRebound  = GS_BOUNCE_REBOUND_DEF;
static float gsSplatRadius    = GS_SPLAT_RADIUS_DEF;
static float gsBrightnessCap  = GS_BRIGHTNESS_CAP_DEF;

struct GsParticle {
    float pos;
    float vel;
    float bright;
    float hue;
};
static GsParticle gsParticles[GS_PARTICLE_COUNT_MAX];

// ── Global state ─────────────────────────────────────────────────

static uint16_t dsR[LED_COUNT], dsG[LED_COUNT], dsB[LED_COUNT], dsW[LED_COUNT];

static volatile uint32_t lastPacketMs = 0;
static volatile uint32_t pktCount = 0;
static SensorPacket latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

// Calibration
static float restAx = 0, restAy = 0, restAz = 1.0f;
static bool calibrated = false;
static float calSumAx = 0, calSumAy = 0, calSumAz = 0;
static uint32_t calSamples = 0;
static uint32_t calStartMs = 0;
#define CAL_DURATION_MS 2000

// FixedRangeRMS
static float frrCeiling = RMS_CEILING;
static float energy = 0.0f;

// Adaptive floor state
static float adaptiveFloor = 0.0f;
static float longMin = 0.0f;
static int belowFloorCount = 0;

static float prevRms = 0.0f;
static float deltaPeak = 1e-6f;
static float onset = 0.0f;

// Syllable onset from phone (port 4213)
static volatile uint8_t syllOnsetStrength = 0;
static volatile uint32_t syllOnsetMs = 0;

// Sparkle (burst state removed — syllable has its own arrays)

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
static float bloomHueDrift[LED_COUNT];
static float bloomFlashGlow[LED_COUNT];
static float bloomFlashDecay[LED_COUNT];
static float bloomColonyEnergy = 1.0f;
static uint32_t bloomLastMotionMs = 0;
static float bloomMotionRate = 0.0f;
static float motionEMA = 0.0f;
static float drainEnvelope = 0.0f;
static float bloomHitIntensity = 0.0f;
static uint32_t bloomPrevPktCount = 0;

static uint32_t prngState;

static uint8_t engageHueOffset = 0;

// ── MAC for identification ───────────────────────────────────────
static char macStr[18] = {0};
static void initMacStr() {
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

// ── Helpers ──────────────────────────────────────────────────────

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

// Unified output stage. See forward decl near setBothPixel.
// Inputs are 0..255 float per channel (already gamma'd / capped).
static void outputPixel(uint16_t i, float r, float g, float b, float w) {
    float sr = r * globalBrightness;
    float sg = g * globalBrightness;
    float sb = b * globalBrightness;
    float sw = w * globalBrightness;

    if (sr < OUTPUT_NOISE_GATE_F && sg < OUTPUT_NOISE_GATE_F &&
        sb < OUTPUT_NOISE_GATE_F && sw < OUTPUT_NOISE_GATE_F) {
        sr = sg = sb = sw = 0.0f;
    }

    uint8_t oR, oG, oB, oW;
    if (outputDitherEnabled) {
        uint16_t tR16 = (uint16_t)clampf(sr * 256.0f, 0.0f, 65535.0f);
        uint16_t tG16 = (uint16_t)clampf(sg * 256.0f, 0.0f, 65535.0f);
        uint16_t tB16 = (uint16_t)clampf(sb * 256.0f, 0.0f, 65535.0f);
        uint16_t tW16 = (uint16_t)clampf(sw * 256.0f, 0.0f, 65535.0f);
        oR = deltaSigma(dsR[i], tR16);
        oG = deltaSigma(dsG[i], tG16);
        oB = deltaSigma(dsB[i], tB16);
        oW = deltaSigma(dsW[i], tW16);
    } else {
        oR = (uint8_t)clampf(sr, 0.0f, 255.0f);
        oG = (uint8_t)clampf(sg, 0.0f, 255.0f);
        oB = (uint8_t)clampf(sb, 0.0f, 255.0f);
        oW = (uint8_t)clampf(sw, 0.0f, 255.0f);
    }

    // Proportional floor: protect originally-nonzero channels from rounding to zero.
    if (oR | oG | oB | oW) {
        if (r > 0.0f && oR == 0) oR = 1;
        if (g > 0.0f && oG == 0) oG = 1;
        if (b > 0.0f && oB == 0) oB = 1;
        if (w > 0.0f && oW == 0) oW = 1;
    }

    stripA.SetPixelColor(i, RgbwColor(oR, oG, oB, oW));
    stripB.SetPixelColor(i, RgbColor(oR, oG, oB));
}

static float computeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx * fx + fy * fy + fz * fz) / 131.0f;
}

static float computeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx * fx + fy * fy + fz * fz);
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
    float scaledCeiling = RMS_CEILING / globalSensitivity;
    frrCeiling = fmaxf(scaledCeiling, frrCeiling * expf(-0.0025f * dt));
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

// Motion-derived energy/onset
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

static void resetBloomState();
static void resetFireState();
static void resetGravitySparkle();
static void resetSyllableState();
static void resetTwinkleState();
void startCalibration();
void handleSerialCommand(char c);
static void handleSliderCommand(const uint8_t *data, size_t len);

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    initMacStr();

    stripA.Begin();
    stripB.Begin();
    stripStkA.Begin();
    stripStkB.Begin();

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
        dsR[i] = seed;
        dsG[i] = (seed + 64) & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;
    }
    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    resetSyllableState();
    resetBloomState();
    resetGravitySparkle();
    resetTwinkleState();

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Scan for "cuteplant" SSID to match gyro-sense channel
    {
        int n = WiFi.scanNetworks(false, false, true, 120);
        uint8_t bestCh = 1;
        int8_t bestRssi = -127;
        for (int i = 0; i < n; i++) {
            if (WiFi.SSID(i) == WIFI_SSID) {
                int8_t rssi = WiFi.RSSI(i);
                if (rssi > bestRssi) { bestRssi = rssi; bestCh = WiFi.channel(i); }
            }
        }
        WiFi.scanDelete();
        esp_wifi_set_promiscuous(true);
        esp_wifi_set_channel(bestCh, WIFI_SECOND_CHAN_NONE);
        esp_wifi_set_promiscuous(false);
        Serial.printf("[BOOT] channel=%u (ssid %s)\n", bestCh, bestCh > 1 ? "found" : "fallback");
    }

    if (esp_now_init() != ESP_OK) {
        Serial.println("[ESPNOW] init FAILED");
    } else {
        esp_now_register_recv_cb(onEspNowReceive);
        Serial.println("[ESPNOW] listening for v1 telemetry");
    }

    stripA.ClearTo(RgbwColor(0, 0, 0, 0));
    stripB.ClearTo(RgbColor(0, 0, 0));
    stripA.Show();
    stripB.Show();

    Serial.println("Commands: 'c' recal, 's' sparkle, 'm' fire_meld, 'f' fire_flicker, 'b' bloom, 'g' gravity_particle, 'y' syllable, 't' twinkle, '?' identify");
    Serial.printf("[BOOT] role=bench-bulbs MAC=%s fw=bench_bulbs_espnow\n", macStr);
}

// ── Calibration ──────────────────────────────────────────────────

void startCalibration() {
    calibrated = false;
    calSumAx = 0; calSumAy = 0; calSumAz = 0;
    calSamples = 0;
    calStartMs = millis();
    Serial.println("Calibrating — keep rig still for 2 seconds...");
}

void updateCalibration(float ax, float ay, float az) {
    if (calStartMs == 0) {
        startCalibration();
    }

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
        float rate = BLOOM_HUE_DRIFT_MIN
            + randFloat() * (BLOOM_HUE_DRIFT_MAX - BLOOM_HUE_DRIFT_MIN);
        bloomHueDrift[i] = (randFloat() > 0.5f) ? rate : -rate;
        bloomFlashGlow[i] = 0.0f;
        bloomFlashDecay[i] = BLOOM_FLASH_DECAY_LO
            + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
    }
}

static void resetGravitySparkle() {
    if (gsParticleCount < 1) gsParticleCount = 1;
    if (gsParticleCount > GS_PARTICLE_COUNT_MAX) gsParticleCount = GS_PARTICLE_COUNT_MAX;
    int n = gsParticleCount;
    float denom = (n > 1) ? (float)(n - 1) : 1.0f;
    for (int i = 0; i < n; i++) {
        gsParticles[i].pos = (float)(LED_COUNT - 1) * (float)i / denom;
        gsParticles[i].vel = 0.0f;
        gsParticles[i].bright = 1.0f;
        gsParticles[i].hue = 256.0f * (float)i / (float)n;
    }
}

// ── Gravity sparkle render ───────────────────────────────────────

static void renderGravitySparkle(float dt) {
    float gravG = clampf((float)latestPacket.ax / 16384.0f, -1.5f, 1.5f);
    float accel = gravG * gsGravityScale;
    float damp = fastDecay(gsVelocityDamp, dt * 30.0f);

    static float accR[LED_COUNT], accG[LED_COUNT], accB[LED_COUNT];
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        accR[i] = 0; accG[i] = 0; accB[i] = 0;
    }

    const float maxPos = (float)(LED_COUNT - 1);
    const float invTwoSigSq = 1.0f / (2.0f * gsSplatRadius * gsSplatRadius);

    for (int i = 0; i < gsParticleCount; i++) {
        GsParticle &p = gsParticles[i];

        p.vel = p.vel * damp + accel * dt;
        p.pos += p.vel * dt;

        if (p.pos < 0.0f) {
            p.pos = 0.0f;
            if (p.vel < 0.0f) p.vel = -p.vel * gsBounceRebound;
        } else if (p.pos > maxPos) {
            p.pos = maxPos;
            if (p.vel > 0.0f) p.vel = -p.vel * gsBounceRebound;
        }

        uint8_t hueIdx = (uint8_t)((uint32_t)p.hue & 0xFF);
        float colR = (float)oklchVarL[hueIdx][0];
        float colG = (float)oklchVarL[hueIdx][1];
        float colB = (float)oklchVarL[hueIdx][2];

        int center = (int)(p.pos + 0.5f);
        int half = (int)(gsSplatRadius * 2.0f + 1.0f);
        int lo = center - half; if (lo < 0) lo = 0;
        int hi = center + half; if (hi > (int)(LED_COUNT - 1)) hi = LED_COUNT - 1;
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
        float linBright = fastGamma24(bright) * gsBrightnessCap;
        float norm = (bright > 0.001f) ? (linBright / bright) : 0.0f;
        outputPixel(i, r * norm, g * norm, b * norm, 0.0f);
    }
}

// ── Serial dispatch ──────────────────────────────────────────────

static float syllSparkle[LED_COUNT];
static float syllDecay[LED_COUNT];
static float syllEnvelope = 0.0f;
static float syllCooldown = 0.0f;

static void resetSyllableState() {
    memset(syllSparkle, 0, sizeof(syllSparkle));
    syllEnvelope = 0.0f;
    syllCooldown = 0.0f;
}

// ── Idle rainbow wash ───────────────────────────────────────────
static float idlePhase = 0.0f;
#define IDLE_BRIGHTNESS_DEF 0.30f
#define IDLE_SPEED_DEF      0.03f
static float idleBrightness = IDLE_BRIGHTNESS_DEF;
static float idleSpeed      = IDLE_SPEED_DEF;

static void renderIdle(float dt) {
    idlePhase = fmodf(idlePhase + idleSpeed * dt, 1.0f);
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float pos = (float)i / (float)LED_COUNT;
        float hue = fmodf(pos + idlePhase, 1.0f);
        uint8_t idx = (uint8_t)(hue * 255.0f);
        float bright = fastGamma24(idleBrightness);
        outputPixel(i,
            (float)oklchVarL[idx][0] * bright,
            (float)oklchVarL[idx][1] * bright,
            (float)oklchVarL[idx][2] * bright, 0.0f);
    }
}

static void handleSliderCommand(const uint8_t *data, size_t len) {
    if (len < 2) return;
    char cmd = (char)data[0];
    uint8_t val = data[1];
    if (cmd == 'B') {
        globalBrightness = val / 128.0f;
        Serial.printf("[SLIDER] brightness=%.0f%%\n", globalBrightness * 100.0f);
    } else if (cmd == 'S') {
        globalSensitivity = fmaxf(0.05f, val / 128.0f);
        Serial.printf("[SLIDER] sensitivity=%.2f\n", globalSensitivity);
    }
}

void handleSerialCommand(char c) {
    if (c == 'c' || c == 'C') {
        startCalibration();
    } else if (c == 's') {
        currentAlg = ALG_SPARKLE_SYLLABLE;
        resetSyllableState();
        Serial.println("Algorithm: sparkle_syllable");
    } else if (c == 'm' || c == 'M') {
        currentAlg = ALG_FIRE_MELD;
        resetFireState();
        Serial.println("Algorithm: fire_meld");
    } else if (c == 'f' || c == 'F') {
        currentAlg = ALG_FIRE_FLICKER;
        resetFireState();
        Serial.println("Algorithm: fire_flicker");
    } else if (c == 'b' || c == 'B') {
        currentAlg = ALG_QUIET_BLOOM;
        resetBloomState();
        Serial.println("Algorithm: quiet_bloom");
    } else if (c == 'g' || c == 'G') {
        currentAlg = ALG_GRAVITY_PARTICLE;
        resetGravitySparkle();
        Serial.println("Algorithm: gravity_particle");
    } else if (c == 'y' || c == 'Y') {
        currentAlg = ALG_SPARKLE_SYLLABLE;
        resetSyllableState();
        Serial.println("Algorithm: sparkle_syllable (alias)");
    } else if (c == 't' || c == 'T') {
        currentAlg = ALG_SPARKLE_TWINKLE;
        resetTwinkleState();
        Serial.println("Algorithm: sparkle_twinkle");
    } else if (c == 'i' || c == 'I') {
        currentAlg = ALG_IDLE;
        idlePhase = 0.0f;
        Serial.println("Algorithm: idle");
    } else if (c == 'w' || c == 'W') {
        currentAlg = ALG_WHITE_STRESS;
        Serial.println("Algorithm: white_stress (ALL LEDs full white)");
    } else if (c == 'x' || c == 'X') {
        currentAlg = ALG_OFF;
        stripA.ClearTo(RgbwColor(0, 0, 0, 0));
        stripB.ClearTo(RgbColor(0, 0, 0));
        stripA.Show();
        stripB.Show();
        Serial.println("Algorithm: off");
    } else if (c == '?') {
        Serial.printf("[BOOT] role=bench-bulbs MAC=%s fw=bench_bulbs_wifi\n", macStr);
    }
}

static char serialBuf[64];
static uint8_t serialBufLen = 0;

static bool parseHexByte(const char *s, uint8_t &out) {
    uint8_t v = 0;
    for (int i = 0; i < 2; i++) {
        char c = s[i];
        if (c >= '0' && c <= '9') v = v * 16 + (c - '0');
        else if (c >= 'a' && c <= 'f') v = v * 16 + (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v = v * 16 + (c - 'A' + 10);
        else return false;
    }
    out = v;
    return true;
}

static void applyRawColor() {
    if (rawDither) {
        memset(dsR, 0, sizeof(dsR));
        memset(dsG, 0, sizeof(dsG));
        memset(dsB, 0, sizeof(dsB));
        memset(dsW, 0, sizeof(dsW));
    } else {
        uint8_t r = (uint8_t)((uint16_t)rawR * rawBrightness / 255);
        uint8_t g = (uint8_t)((uint16_t)rawG * rawBrightness / 255);
        uint8_t b = (uint8_t)((uint16_t)rawB * rawBrightness / 255);
        uint8_t w = (uint8_t)((uint16_t)rawW * rawBrightness / 255);
        for (uint16_t i = 0; i < LED_COUNT; i++) setBothPixel(i, r, g, b, w);
        stripA.Show(); stripB.Show();
    }
}

// ── Runtime parameter table ──────────────────────────────────────
struct Param {
    const char *name;
    enum Type { F32, I32 } type;
    void *ptr;
    float lo, hi;
    void (*onChange)();
};

static const Param PARAMS[] = {
    // bloom
    { "BLOOM_FLASH_BUMP",       Param::F32, &bloomFlashBump,       0.0f,  0.30f, nullptr },
    { "BLOOM_ACCEL_THRESH",     Param::F32, &bloomAccelThresh,     1.05f, 3.0f,  nullptr },
    { "BLOOM_BRIGHTNESS_CAP",   Param::F32, &bloomBrightnessCap,   0.05f, 1.0f,  nullptr },
    { "BLOOM_FLASH_DECAY_RATE", Param::F32, &bloomFlashDecayRate,  0.5f,  5.0f,  nullptr },
    { "BLOOM_BREATH_FLOOR",     Param::F32, &bloomBreathFloor,     0.0f,  0.5f,  nullptr },
    // gravity
    { "GS_PARTICLE_COUNT",      Param::I32, &gsParticleCount,      1.0f,  (float)GS_PARTICLE_COUNT_MAX, resetGravitySparkle },
    { "GS_GRAVITY_SCALE",       Param::F32, &gsGravityScale,       5.0f,  200.0f, nullptr },
    { "GS_VELOCITY_DAMP",       Param::F32, &gsVelocityDamp,       0.5f,  0.999f, nullptr },
    { "GS_BOUNCE_REBOUND",      Param::F32, &gsBounceRebound,      0.0f,  1.0f,   nullptr },
    { "GS_SPLAT_RADIUS",        Param::F32, &gsSplatRadius,        0.5f,  8.0f,   nullptr },
    { "GS_BRIGHTNESS_CAP",      Param::F32, &gsBrightnessCap,      0.05f, 1.0f,   nullptr },
    // twinkle
    { "TWINKLE_SPAWN_RATE",     Param::F32, &twinkleSpawnRate,     1.0f,  300.0f, nullptr },
    { "TWINKLE_ATTACK_S",       Param::F32, &twinkleAttackS,       0.005f,0.5f,   nullptr },
    { "TWINKLE_TAU_S",          Param::F32, &twinkleTauS,          0.02f, 2.0f,   nullptr },
    { "TWINKLE_PEAK_MIN",       Param::F32, &twinklePeakMin,       0.0f,  1.0f,   nullptr },
    { "TWINKLE_PEAK_MAX",       Param::F32, &twinklePeakMax,       0.0f,  1.0f,   nullptr },
    { "SPARKLE_BRIGHTNESS_CAP", Param::F32, &sparkleBrightnessCap, 0.01f, 0.5f,   nullptr },
    // idle
    { "IDLE_BRIGHTNESS",        Param::F32, &idleBrightness,       0.0f,  1.0f,   nullptr },
    { "IDLE_SPEED",             Param::F32, &idleSpeed,            0.0f,  0.5f,   nullptr },
    // global
    { "GLOBAL_BRIGHTNESS",      Param::F32, &globalBrightness,     0.0f,  2.0f,   nullptr },
    { "GLOBAL_SENSITIVITY",     Param::F32, &globalSensitivity,    0.05f, 2.0f,   nullptr },
    // output stage
    { "OUTPUT_DITHER",          Param::I32, &outputDitherEnabled,  0.0f,  1.0f,  nullptr },
};
static const size_t PARAM_COUNT = sizeof(PARAMS) / sizeof(PARAMS[0]);

static void dumpParam(const Param &p) {
    if (p.type == Param::F32) {
        Serial.printf("[PARAM] %s=%.4f\n", p.name, *(float*)p.ptr);
    } else {
        Serial.printf("[PARAM] %s=%d\n", p.name, *(int*)p.ptr);
    }
}

static void dumpAllParams() {
    for (size_t i = 0; i < PARAM_COUNT; i++) dumpParam(PARAMS[i]);
}

static void setParamFromLine(const char *kv) {
    // kv looks like "KEY=value" (leading spaces tolerated)
    while (*kv == ' ') kv++;
    const char *eq = strchr(kv, '=');
    if (!eq) { Serial.println("[PARAM] bad syntax"); return; }
    size_t nameLen = (size_t)(eq - kv);
    const char *valStr = eq + 1;
    for (size_t i = 0; i < PARAM_COUNT; i++) {
        const Param &p = PARAMS[i];
        if (strlen(p.name) == nameLen && strncmp(p.name, kv, nameLen) == 0) {
            float v = atof(valStr);
            if (v < p.lo) v = p.lo;
            if (v > p.hi) v = p.hi;
            if (p.type == Param::F32) *(float*)p.ptr = v;
            else                       *(int*)p.ptr   = (int)(v + 0.5f);
            if (p.onChange) p.onChange();
            dumpParam(p);
            return;
        }
    }
    Serial.printf("[PARAM] unknown key (len=%u)\n", (unsigned)nameLen);
}

static void processSerialLine(const char *line, uint8_t len) {
    if (len >= 7 && line[0] == '#') {
        uint8_t r, g, b, w = 0;
        if (parseHexByte(line + 1, r) && parseHexByte(line + 3, g) && parseHexByte(line + 5, b)) {
            if (len >= 9) parseHexByte(line + 7, w);
            rawR = r; rawG = g; rawB = b; rawW = w;
            currentAlg = ALG_RAW_COLOR;
            applyRawColor();
            Serial.printf("[RAW] #%02X%02X%02X%02X brt=%u dith=%d fps=%u\n",
                          r, g, b, w, rawBrightness, rawDither, rawTargetFps);
            return;
        }
    }
    if (len >= 2 && line[0] == '!') {
        if (line[1] == 'B' && len >= 3) {
            rawBrightness = (uint8_t)atoi(line + 2);
            if (currentAlg == ALG_RAW_COLOR) applyRawColor();
            Serial.printf("[RAW] brightness=%u\n", rawBrightness);
            return;
        }
        if (line[1] == 'D' && len >= 3) {
            rawDither = (line[2] == '1');
            if (currentAlg == ALG_RAW_COLOR) applyRawColor();
            Serial.printf("[RAW] dither=%d\n", rawDither);
            return;
        }
        if (line[1] == 'F' && len >= 3) {
            rawTargetFps = (uint16_t)atoi(line + 2);
            if (rawTargetFps > 500) rawTargetFps = 500;
            if (rawTargetFps < 1) rawTargetFps = 1;
            Serial.printf("[RAW] fps=%u\n", rawTargetFps);
            return;
        }
        if (line[1] == 'P') {
            if (len >= 3 && line[2] == '?') { dumpAllParams(); return; }
            if (len >= 4) { setParamFromLine(line + 2); return; }
            return;
        }
    }
    if (len == 1) handleSerialCommand(line[0]);
}

static void parseSerialCommands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBufLen > 0) {
                serialBuf[serialBufLen] = '\0';
                processSerialLine(serialBuf, serialBufLen);
                serialBufLen = 0;
            }
        } else if (c >= 32 && c < 127 && serialBufLen < sizeof(serialBuf) - 1) {
            if (serialBufLen == 0 && c != '#' && c != '!') {
                handleSerialCommand(c);
            } else {
                serialBuf[serialBufLen++] = c;
            }
        }
    }
}

// ── Bloom motion ─────────────────────────────────────────────────

static float bloomFlash = 0.0f;

static void bloomProcessMotion(float pktDt, uint32_t now) {
    uint8_t peakAmag = espnowPeakAmag;
    espnowPeakAmag = 0;
    espnowPeakGmag = 0;

    float amagG = amagGFromByte(peakAmag);

    static uint32_t lastMotionLog = 0;
    if (amagG > 1.2f || now - lastMotionLog > 2000) {
        Serial.printf("[MOTION] amag=%.2fg flash=%.2f pkts=%lu\n",
                      amagG, bloomFlash, (unsigned long)espnowPktCount);
        lastMotionLog = now;
    }

    if (amagG > bloomAccelThresh) {
        bloomFlash = fminf(1.0f, bloomFlash + bloomFlashBump);
    }
}

static void renderQuietBloom(float dt, uint32_t now) {
    // Decay flash toward zero
    bloomFlash *= expf(-bloomFlashDecayRate * dt);
    if (bloomFlash < 0.005f) bloomFlash = 0.0f;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        bloomBreathPhase[i] += dt / bloomBreathPeriod[i];
        if (bloomBreathPhase[i] >= 1.0f) bloomBreathPhase[i] -= 1.0f;

        bloomHueT[i] += bloomHueDrift[i] * dt;
        if (bloomHueT[i] > 1.0f) bloomHueT[i] -= 1.0f;
        else if (bloomHueT[i] < 0.0f) bloomHueT[i] += 1.0f;

        float breath = (fastSinPhase(bloomBreathPhase[i]) * 0.5f + 0.5f);
        float breathGlow = bloomBreathFloor
            + breath * (bloomBreathPeak[i] - bloomBreathFloor);

        // ── Layer 1: intensity = breath + flash bump ──
        float intensity = clampf(breathGlow + bloomFlash, 0.0f, 1.0f);
        float flashFrac = (bloomFlash > 0.1f) ? clampf(bloomFlash / 0.3f, 0.0f, 1.0f) : 0.0f;

        // ── Layer 2: color lookup (full-range 0–255) ──
        float h = bloomHueT[i];
        float colR = lerpf(lerpf(BLOOM_HUE_A_R, BLOOM_HUE_B_R, h),
                           BLOOM_FLASH_R, flashFrac);
        float colG = lerpf(lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h),
                           BLOOM_FLASH_G, flashFrac);
        float colB = lerpf(lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h),
                           BLOOM_FLASH_B, flashFrac);

        // W channel: ramps in at high intensity
        float wFrac = clampf((intensity - 0.5f) * 2.0f, 0.0f, 1.0f);
        float colW = wFrac * flashFrac * BLOOM_W_SCALE;

        // ── Layer 3: output stage ──
        float linBright = fastGamma24(intensity) * bloomBrightnessCap;
        outputPixel(i,
            colR * linBright,
            colG * linBright,
            colB * linBright,
            colW * linBright);
    }

}

// ── Sparkle syllable ─────────────────────────────────────────────

static void renderSparkleSyllable(float dt, float angleDeg, float tiltBlend,
                                   float tiltR, float tiltG, float tiltB) {
    uint32_t now = millis();
    uint32_t onsetAge = now - syllOnsetMs;
    uint8_t strength = syllOnsetStrength;

    bool gotOnset = (syllOnsetMs > 0 && onsetAge < 100);
    float onsetNorm = gotOnset ? (strength / 255.0f) : 0.0f;

    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > syllEnvelope)
        syllEnvelope += attackAlpha * (energy - syllEnvelope);
    else
        syllEnvelope += decayAlpha * (energy - syllEnvelope);

    syllCooldown = fmaxf(0.0f, syllCooldown - dt);

    if (gotOnset && onsetNorm > 0.1f && syllCooldown <= 0.0f) {
        syllOnsetMs = 0;
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

        float linBright = fastGamma24(bright) * sparkleBrightnessCap;
        float colW = 255.0f * (1.0f - tiltBlend);
        outputPixel(i,
            colR * linBright,
            colG * linBright,
            colB * linBright,
            colW * linBright);
    }
}

// ── Sparkle twinkle render (variant B) ──────────────────────────
//
// 50 independent LEDs. New sparkles ignite at random positions as a
// Poisson process (TWINKLE_SPAWN_RATE expected ignitions/sec across the
// whole strip). Each sparkle does a short linear attack to a random peak,
// then exponential decay in place. No motion, no base glow, no beams.
// Collisions keep the brighter sparkle and restart its attack.
//
// Per LED we track current brightness, target peak, and attack progress.
// While attacking (attack < 1) brightness ramps linearly to peak; once
// peaked it decays multiplicatively by exp(-dt/TAU).

static float twkBright[LED_COUNT];   // current brightness 0..1
static float twkPeak[LED_COUNT];     // peak this sparkle is rising toward
static float twkAttack[LED_COUNT];   // attack progress 0..1 (>=1 = decaying)

static void resetTwinkleState() {
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        twkBright[i] = 0.0f;
        twkPeak[i]   = 0.0f;
        twkAttack[i] = 1.0f;  // idle LEDs sit "past attack" so they just decay
    }
}

static void igniteTwinkle(uint16_t i) {
    float peak = twinklePeakMin
        + randFloat() * (twinklePeakMax - twinklePeakMin);
    // Collision: keep the brighter sparkle, but always restart the attack
    // so a fresh hit re-snaps on.
    if (peak < twkPeak[i] && twkBright[i] > peak) peak = twkPeak[i];
    twkPeak[i]   = peak;
    twkAttack[i] = 0.0f;
}

static void renderSparkleTwinkle(float dt) {
    // ── Poisson spawn (frame-rate independent) ──
    float expected = twinkleSpawnRate * dt;
    int nSpawn = (int)expected;                 // floor
    float frac = expected - (float)nSpawn;
    if (randFloat() < frac) nSpawn++;           // one more with prob = frac
    for (int s = 0; s < nSpawn; s++) {
        uint16_t i = (uint16_t)(xorshift32() % LED_COUNT);
        igniteTwinkle(i);
    }

    // ── Advance envelopes + render ──
    float attackStep = (twinkleAttackS > 0.0f) ? (dt / twinkleAttackS) : 1.0f;
    float decay = expf(-dt / twinkleTauS);

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        if (twkAttack[i] < 1.0f) {
            twkAttack[i] += attackStep;
            if (twkAttack[i] >= 1.0f) {
                twkAttack[i] = 1.0f;
                twkBright[i] = twkPeak[i];
            } else {
                twkBright[i] = twkPeak[i] * twkAttack[i];
            }
        } else {
            twkBright[i] *= decay;
            if (twkBright[i] < 0.002f) twkBright[i] = 0.0f;
        }

        float bright = twkBright[i];
        float linBright = fastGamma24(bright) * sparkleBrightnessCap;

        // Pure-white asymmetric write: Strip A drives W only (clean RGBW
        // white), Strip B has no W so we fold white into r=g=b. This path
        // can't go through outputPixel (asymmetric A vs B). Always uses
        // delta-sigma dither — twinkle is dim by design and needs it.
        bool pureWhite = (TWINKLE_COL_R == 0.0f
                          && TWINKLE_COL_G == 0.0f
                          && TWINKLE_COL_B == 0.0f);
        if (pureWhite) {
            float fW   = TWINKLE_COL_W       * linBright * globalBrightness;
            float fRGB = TWINKLE_RGB_FALLBACK * linBright * globalBrightness;
            uint16_t tW16   = (uint16_t)clampf(fW   * 256.0f, 0, 65535);
            uint16_t tRGB16 = (uint16_t)clampf(fRGB * 256.0f, 0, 65535);
            if (tW16   < 256) tW16   = 0;
            if (tRGB16 < 256) tRGB16 = 0;
            uint8_t w   = deltaSigma(dsW[i], tW16);
            uint8_t rgb = deltaSigma(dsB[i], tRGB16);
            stripA.SetPixelColor(i, RgbwColor(0, 0, 0, w));
            stripB.SetPixelColor(i, RgbColor(rgb, rgb, rgb));
        } else {
            outputPixel(i,
                TWINKLE_COL_R * linBright,
                TWINKLE_COL_G * linBright,
                TWINKLE_COL_B * linBright,
                TWINKLE_COL_W * linBright);
        }
    }
}

// ── Fire render ──────────────────────────────────────────────────

static void renderFire(float dt, bool withDropout, float tiltBlend) {
    fireTime = fmodf(fireTime + dt, 6283.1853f);
    float t = fireTime;

    bool isSilent = energy < 0.001f;
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);

    float attackAlpha = fminf(1.0f, dt / 0.050f);
    float decayAlpha  = fminf(1.0f, dt / 2.0f);

    float targetBrightness;
    if (isSilent) {
        targetBrightness = 0.25f;
    } else {
        targetBrightness = fmaxf(0.25f, energy);
    }

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
    if (isPercussiveOnly)
        colorTarget = 0.0f;
    else if (isSilent)
        colorTarget = 0.3f;
    else
        colorTarget = fmaxf(0.3f, energy);

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

    float base = fireBaseBrightness;
    if (base < FIRE_DEADBAND) base = 0.0f;

    float s = FIRE_FLICKER_SCALE;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float fi = (float)i;

        float noise = fastSin(fi * 7.3f + t * 2.5f) *
                       fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

        float noiseAmp = fmaxf(0.15f * s, 0.10f * s / fmaxf(base, 0.1f));
        float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
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
        outputPixel(i, fR, fG, fB, fW);
    }
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    parseSerialCommands();

    SensorPacket snap;
    uint32_t lastMs;
    portENTER_CRITICAL(&pktMux);
    snap   = latestPacket;
    lastMs = lastPacketMs;
    portEXIT_CRITICAL(&pktMux);

    float ax = (float)snap.ax;
    float ay = (float)snap.ay;
    float az = (float)snap.az;
    vecNormalize(ax, ay, az);
    uint16_t rawRms = snap.rawRms;
    bool micOn = snap.micEnabled != 0;
    bool connected = (lastMs > 0) && (now - lastMs < TIMEOUT_MS);

    if (connected) {
        updateCalibration(ax, ay, az);
    }

    float angleDeg = 0;
    float dot = restAx * ax + restAy * ay + restAz * az;
    if (calibrated && connected) {
        angleDeg = acosf(clampf(dot, -1.0f, 1.0f)) * (180.0f / M_PI);
    }

    if (currentAlg != ALG_QUIET_BLOOM) {
        bool gyroActive = (snap.gx != 0 || snap.gy != 0 || snap.gz != 0);
        if (connected && micOn) {
            energy = computeEnergy(rawRms, dt);
            onset = computeOnset(rawRms, dt);
        } else if (connected && !micOn && gyroActive) {
            computeMotionEnergy(
                latestPacket.ax, latestPacket.ay, latestPacket.az,
                latestPacket.gx, latestPacket.gy, latestPacket.gz,
                dt, energy, onset);
        } else {
            energy = 0.0f;
            onset = 0.0f;
        }
    }

    if (currentAlg == ALG_QUIET_BLOOM) {
        if (espnowPktCount != bloomPrevPktCount) {
            static uint32_t lastBloomPktMs = 0;
            float pktDt = (lastBloomPktMs > 0)
                ? (now - lastBloomPktMs) / 1000.0f
                : (1.0f / SENSOR_HZ);
            if (pktDt > 0.2f) pktDt = 0.2f;
            lastBloomPktMs = now;
            bloomProcessMotion(pktDt, now);
            bloomPrevPktCount = espnowPktCount;
        }
    }

    if (connected && calibrated && (pktCount % 25 == 0)) {
        const char *algName = "sparkle";
        if (currentAlg == ALG_FIRE_MELD) algName = "fire_meld";
        else if (currentAlg == ALG_FIRE_FLICKER) algName = "fire_flicker";
        else if (currentAlg == ALG_QUIET_BLOOM) algName = "bloom";

        unsigned long pkts = (unsigned long)pktCount;
        if (currentAlg == ALG_QUIET_BLOOM) {
            Serial.printf("[%s] rate=%.1f energy=%.3f flash0=%.3f pkts=%lu\n",
                algName, bloomMotionRate, bloomColonyEnergy, bloomFlashGlow[0],
                pkts);
        } else {
            Serial.printf("[%s] angle=%.1f rms=%d energy=%.3f onset=%.3f mic=%s pkts=%lu\n",
                algName, angleDeg, rawRms, energy, onset, micOn ? "on" : "off",
                pkts);
        }
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
        case ALG_OFF:
            stripA.ClearTo(RgbwColor(0, 0, 0, 0));
            stripB.ClearTo(RgbColor(0, 0, 0));
            break;
        case ALG_SPARKLE_BURST:
            renderSparkleSyllable(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB);
            break;
        case ALG_FIRE_MELD:
            renderFire(dt, false, tiltBlend);
            break;
        case ALG_FIRE_FLICKER:
            renderFire(dt, true, tiltBlend);
            break;
        case ALG_QUIET_BLOOM:
            renderQuietBloom(dt, now);
            break;
        case ALG_GRAVITY_PARTICLE:
            renderGravitySparkle(dt);
            break;
        case ALG_SPARKLE_SYLLABLE:
            renderSparkleSyllable(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB);
            break;
        case ALG_SPARKLE_TWINKLE:
            renderSparkleTwinkle(dt);
            break;
        case ALG_IDLE:
            renderIdle(dt);
            break;
        case ALG_WHITE_STRESS:
            renderQuietBloom(dt, now);
            break;
        case ALG_RAW_COLOR:
            if (rawDither) {
                uint16_t tR = (uint16_t)rawR * rawBrightness;  // 0–65025
                uint16_t tG = (uint16_t)rawG * rawBrightness;
                uint16_t tB = (uint16_t)rawB * rawBrightness;
                uint16_t tW = (uint16_t)rawW * rawBrightness;
                for (uint16_t i = 0; i < LED_COUNT; i++) {
                    uint8_t oR = deltaSigma(dsR[i], tR);
                    uint8_t oG = deltaSigma(dsG[i], tG);
                    uint8_t oB = deltaSigma(dsB[i], tB);
                    uint8_t oW = deltaSigma(dsW[i], tW);
                    setBothPixel(i, oR, oG, oB, oW);
                }
            }
            break;
    }

    // === STICK PIPELINE (green breathe) ===
    // 4-second breathing cycle, frame-rate independent.
    static float stickPhase = 0.0f;
    float stickBr = (sinf(stickPhase) + 1.0f) * 0.5f;
    uint8_t stickG = (uint8_t)(stickBr * 127.0f);
    RgbColor stickColor(0, stickG, 0);
    stripStkA.ClearTo(stickColor);
    stripStkB.ClearTo(stickColor);
    stickPhase += (2.0f * (float)M_PI / 4.0f) * dt;
    if (stickPhase > 6.2832f) stickPhase -= 6.2832f;

    {
        static uint32_t lastDbg = 0;
        if (now - lastDbg > 500) {
            lastDbg = now;
            static char line[512];
            int pos = 0;
            pos += snprintf(line + pos, sizeof(line) - pos, "[PX] espnow=%lu", (unsigned long)espnowPktCount);
            for (uint16_t j = 0; j < LED_COUNT; j += 5) {
                RgbwColor c = stripA.GetPixelColor(j);
                pos += snprintf(line + pos, sizeof(line) - pos,
                    " %d:%d/%d/%d/%d", j, c.R, c.G, c.B, c.W);
            }
            Serial.println(line);
        }
    }

    stripA.Show();
    stripB.Show();
    stripStkA.Show();
    stripStkB.Show();

    if (currentAlg == ALG_RAW_COLOR && rawDither && rawTargetFps > 0) {
        uint32_t frameUs = 1000000UL / rawTargetFps;
        uint32_t elapsed = micros() - now * 1000;
        if (elapsed < frameUs) delayMicroseconds(frameUs - elapsed);
    } else {
        delay(1);
    }
}
