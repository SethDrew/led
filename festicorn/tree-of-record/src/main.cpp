/*
 * TREE-OF-RECORD — ambient bloom on all 6 strips, driven by a BS-26.
 *
 * ESP32-D0WD-V3, 6 × WS2812B RGB strips on GPIO 4, 15, 17, 5, 18, 19,
 * 100 LEDs each. No senders, no motion — pure ambient breathing bloom
 * (ported from biolum board B), modulated by a BS-26 console:
 *
 *   tens (10-pos knob) → brightness (0.01–1.0)  uniform output scale
 *   hundreds (0–4)     → speed      (0.3–3.0)   breathing rate, 2=normal
 *   decmid (0–9)       → hue        36° rotation steps (full wheel)
 *   decinner (0–9)     → saturation 0=full color, 9=white
 *   DC switch          → dormancy on/off (DORMANCY_FRAC = dc ? 0.3 : 0.0)
 *   ones, decouter, AC, uA, V → parsed but unused (reserved)
 *
 * BS-26 live → knobs control. No signal → serial PARAMS control.
 * BS-26 state arrives as JSON over ESP-NOW broadcast on channel 1.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <ArduinoJson.h>
#include <math.h>
#include <fast_math.h>
#include <delta_sigma.h>
#include <oklch_lut.h>
#include <v1_packet.h>
#include <gyro_packet_v1.h>
#include <audio_packet_v1.h>

// ── Strip layout ─────────────────────────────────────────────────
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 100
#endif

static const uint8_t NUM_STRIPS = 6;

// ── Bloom parameters (runtime-tunable via operator) ─────────────
static float bloomBrightnessCap  = 0.15f;
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

#define SENSOR_HZ       25.0f

// Tilt mapping (used by ported sparkle effect)
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f

// ── Effect drive parameters ──────────────────────────────────────
// BS-26 live → knobs control. No signal → serial PARAMS control.
static float globalBrightness = 1.0f;   // serial-controlled, 0.0–1.0
static float speedScale       = 1.0f;   // serial-controlled, 0.2–3.0
static float renderBrightness = 1.0f;   // effective scale the render reads each frame

// ── BS-26 ESP-NOW config ─────────────────────────────────────────
#define FIXED_CHANNEL        1
#define HEARTBEAT_TIMEOUT_MS 3000

// ── BS-26 state ──────────────────────────────────────────────────
struct Bs26State {
    bool     dc;
    bool     ac;
    uint8_t  hundreds;
    uint8_t  tens;
    uint8_t  ones;
    uint16_t decade;     // 0–1199
    uint8_t  decouter;   // 0–11, outer ring
    uint8_t  decmid;     // 0–9, middle ring
    uint8_t  decinner;   // 0–9, inner ring
    uint8_t  ua_dac;     // 0–255
    uint8_t  ua_max;
    uint8_t  v_dac;
    uint8_t  v_max;
    uint32_t seq;
};

static volatile Bs26State bs26 = {};
static volatile uint32_t  bs26LastMs = 0;
static volatile uint32_t  bs26PktCount = 0;
static volatile bool      bs26Updated = false;

// ── Sensor layer (v1 accel / gyro / audio packets over ESP-NOW) ──
// Coexists with BS-26 JSON; disambiguated by packet length in onReceive.
// Raw packet fields are latched in the callback; loop() decodes + normalizes.
#define SENSOR_TIMEOUT_MS 3000

// Companding full-scale (matches v1 packet encoders).
#define V1_AMAG_FS  57000.0f
#define V1_GMAG_FS  57000.0f
#define V1_RMS_FS   8000.0f   // must match sender_v2 RMS_FS

// Production sender locks ±4g accel / ±1000 dps gyro.
#define V1_ACCEL_RANGE_G    4.0f
#define V1_GYRO_RANGE_DPS   1000.0f
#define V1_COUNTS_PER_G     (32768.0f / V1_ACCEL_RANGE_G)
#define V1_DPS_PER_LSB      (V1_GYRO_RANGE_DPS / 128.0f)   // int8 mean → dps

static volatile TelemetryPacketV1 accelPkt = {};
static volatile uint32_t accelLastMs = 0;
static volatile uint32_t accelPktCount = 0;

static volatile GyroPacketV1 gyroPkt = {};
static volatile uint32_t gyroLastMs = 0;
static volatile uint32_t gyroPktCount = 0;

static volatile AudioPacketV1 audioPkt = {};
static volatile uint32_t audioLastMs = 0;
static volatile uint32_t audioPktCount = 0;

// sqrt-companded uint8 → linear value: val = (byte/255)² * FS
static inline float decodeCompand(uint8_t b, float fs) {
    float t = (float)b / 255.0f;
    return t * t * fs;
}

// ── PRNG ─────────────────────────────────────────────────────────
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

// ── HSV ↔ RGB (h 0..360, s/v 0..1, rgb 0..255) ─────────────────
static void hsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
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
    r = (r1 + m) * 255.0f;
    g = (g1 + m) * 255.0f;
    b = (b1 + m) * 255.0f;
}

static void rgbToHsv(float r, float g, float b, float &h, float &s, float &v) {
    r /= 255.0f; g /= 255.0f; b /= 255.0f;
    float mx = fmaxf(fmaxf(r, g), b);
    float mn = fminf(fminf(r, g), b);
    float d = mx - mn;
    v = mx;
    s = (mx <= 0.0f) ? 0.0f : d / mx;
    if (d <= 0.0f)        h = 0.0f;
    else if (mx == r)     h = 60.0f * fmodf((g - b) / d, 6.0f);
    else if (mx == g)     h = 60.0f * ((b - r) / d + 2.0f);
    else                  h = 60.0f * ((r - g) / d + 4.0f);
    if (h < 0.0f) h += 360.0f;
}

// ── Bloom hue endpoints (runtime, rotated by palette index) ─────
#define BLOOM_HUE_A_R_BASE    0.0f
#define BLOOM_HUE_A_G_BASE  180.0f
#define BLOOM_HUE_A_B_BASE  120.0f
#define BLOOM_HUE_B_R_BASE  140.0f
#define BLOOM_HUE_B_G_BASE   20.0f
#define BLOOM_HUE_B_B_BASE  255.0f

static float bloomHueA_R = BLOOM_HUE_A_R_BASE;
static float bloomHueA_G = BLOOM_HUE_A_G_BASE;
static float bloomHueA_B = BLOOM_HUE_A_B_BASE;
static float bloomHueB_R = BLOOM_HUE_B_R_BASE;
static float bloomHueB_G = BLOOM_HUE_B_G_BASE;
static float bloomHueB_B = BLOOM_HUE_B_B_BASE;

static uint8_t paletteHueIdx = 0;   // 0–9, from decmid
static uint8_t paletteSatIdx = 0;   // 0–9, from decinner

// Rotate base endpoint hues by hueIdx * 36° and desaturate by satIdx.
static void applyPalette(uint8_t hueIdx, uint8_t satIdx) {
    float rot = (float)hueIdx * 36.0f;
    float satScale = 1.0f - (float)satIdx / 9.0f;  // 0=full sat, 9=white
    float h, s, v, r, g, b;
    rgbToHsv(BLOOM_HUE_A_R_BASE, BLOOM_HUE_A_G_BASE, BLOOM_HUE_A_B_BASE, h, s, v);
    hsvToRgb(fmodf(h + rot, 360.0f), s * satScale, v, r, g, b);
    bloomHueA_R = r; bloomHueA_G = g; bloomHueA_B = b;
    rgbToHsv(BLOOM_HUE_B_R_BASE, BLOOM_HUE_B_G_BASE, BLOOM_HUE_B_B_BASE, h, s, v);
    hsvToRgb(fmodf(h + rot, 360.0f), s * satScale, v, r, g, b);
    bloomHueB_R = r; bloomHueB_G = g; bloomHueB_B = b;
}

// ── ESP-NOW receive (BS-26 JSON state packets) ──────────────────
static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len < 2 || len > 250) return;

    // Fixed-size binary sensor packets take priority over JSON (disambiguated
    // by exact length). Latch raw bytes; loop() decodes and normalizes.
    if (len == (int)sizeof(TelemetryPacketV1)) {
        memcpy((void*)&accelPkt, data, sizeof(TelemetryPacketV1));
        accelLastMs = millis();
        accelPktCount++;
        return;
    }
    if (len == (int)sizeof(GyroPacketV1)) {
        memcpy((void*)&gyroPkt, data, sizeof(GyroPacketV1));
        gyroLastMs = millis();
        gyroPktCount++;
        return;
    }
    if (len == (int)sizeof(AudioPacketV1)) {
        memcpy((void*)&audioPkt, data, sizeof(AudioPacketV1));
        audioLastMs = millis();
        audioPktCount++;
        return;
    }

    StaticJsonDocument<768> doc;
    DeserializationError err = deserializeJson(doc, (const char*)data, len);
    if (err) return;

    if (!doc.containsKey("seq")) return;

    bs26.dc       = doc["dc"] | false;
    bs26.ac       = doc["ac"] | false;
    bs26.hundreds = doc["hundreds"] | 0;
    bs26.tens     = doc["tens"] | 0;
    bs26.ones     = doc["ones"] | 0;
    bs26.decade   = doc["decade"] | 0;
    bs26.decouter = doc["decouter"] | 0;
    bs26.decmid   = doc["decmid"] | 0;
    bs26.decinner = doc["decinner"] | 0;

    JsonObject ua = doc["ua"];
    if (ua) {
        bs26.ua_dac = ua["dac"] | 0;
        bs26.ua_max = ua["max"] | 0;
    }
    JsonObject v = doc["v"];
    if (v) {
        bs26.v_dac = v["dac"] | 0;
        bs26.v_max = v["max"] | 0;
    }

    bs26.seq = doc["seq"] | 0;
    bs26LastMs = millis();
    bs26PktCount++;
    bs26Updated = true;
}

// ── LED driver: 6 strips via RMT ─────────────────────────────────
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 15);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, 19);

static inline void setPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
        case 4: strip4.SetPixelColor(i, c); break;
        case 5: strip5.SetPixelColor(i, c); break;
    }
}

static void showAll() {
    strip0.Show(); strip1.Show(); strip2.Show();
    strip3.Show(); strip4.Show(); strip5.Show();
}

// ── Generic delta-sigma output stage (shared by ported effects) ──
// Ported bulb-fleet/biolum effects write 0–255-range float color here;
// this scales by renderBrightness, converts to 16-bit, and dithers.
// (The bloom effect keeps its own per-pixel accumulators in BloomStrip.)
static uint16_t fxDitherR[NUM_STRIPS][LEDS_PER_STRIP];
static uint16_t fxDitherG[NUM_STRIPS][LEDS_PER_STRIP];
static uint16_t fxDitherB[NUM_STRIPS][LEDS_PER_STRIP];

static void resetFxDither() {
    memset(fxDitherR, 0, sizeof(fxDitherR));
    memset(fxDitherG, 0, sizeof(fxDitherG));
    memset(fxDitherB, 0, sizeof(fxDitherB));
}

// Global palette transform: rotate hue by paletteHueIdx×36° and scale
// saturation by (1 - paletteSatIdx/9), matching applyPalette()'s rule. This
// is the shared path for every effect that routes through setPixelDither, so
// the decmid/decinner knobs affect all of them. Bloom does NOT come through
// here (it uses setPixel directly and rotates its own base colors), so there
// is no double-rotation. No-op fast path when both knobs are at 0.
static inline void applyGlobalPalette(float &r, float &g, float &b) {
    if (paletteHueIdx == 0 && paletteSatIdx == 0) return;
    float h, sv, v;
    rgbToHsv(r, g, b, h, sv, v);
    h = fmodf(h + (float)paletteHueIdx * 36.0f, 360.0f);
    sv *= 1.0f - (float)paletteSatIdx / 9.0f;
    hsvToRgb(h, sv, v, r, g, b);
}

// fr/fg/fb in 0–255 linear range (already gamma-shaped by the effect),
// pre-brightness. renderBrightness applies here so the tens knob is the
// sole global output scale (no per-effect caps).
static inline void setPixelDither(uint8_t s, uint16_t i, float fr, float fg, float fb) {
    applyGlobalPalette(fr, fg, fb);
    fr *= renderBrightness;
    fg *= renderBrightness;
    fb *= renderBrightness;
    uint16_t t16R = (uint16_t)fminf(fr * 256.0f, 65535.0f);
    uint16_t t16G = (uint16_t)fminf(fg * 256.0f, 65535.0f);
    uint16_t t16B = (uint16_t)fminf(fb * 256.0f, 65535.0f);
    if ((t16R | t16G | t16B) == 0) {
        fxDitherR[s][i] = fxDitherG[s][i] = fxDitherB[s][i] = 0;
    }
    uint8_t r8 = deltaSigma(fxDitherR[s][i], t16R);
    uint8_t g8 = deltaSigma(fxDitherG[s][i], t16G);
    uint8_t b8 = deltaSigma(fxDitherB[s][i], t16B);
    setPixel(s, i, r8, g8, b8);
}

// Per-strip OKLCH hue offsets (~60° apart on 256-step wheel) so the 6
// strips span a rainbow chord. From bulb-fleet.
static const uint8_t STRIP_HUE_OFFSET[NUM_STRIPS] = { 0, 42, 85, 128, 170, 213 };

// ── Effect selection (BS-26 ones knob, 0–11) ────────────────────
enum EffectId : uint8_t {
    FX_AMBIENT_BLOOM = 0,
    FX_QUIET_BLOOM,
    FX_GRAVITY_PARTICLE,
    FX_SPARKLE_SYLLABLE,
    FX_FIRE_MELD,
    FX_FIRE_FLICKER,
    FX_RAINBOW,
    FX_NEBULA,
    FX_LEAF_WIND,
    FX_CREATURES,
    FX_LIGHT_THROUGH,
    FX_OFF_11,
    FX_COUNT
};

static uint8_t currentEffect = FX_AMBIENT_BLOOM;
static uint8_t prevEffect    = 0xFF;   // forces reset on first frame

// ── Audio/motion feature globals ────────────────────────────────
// The bulb-fleet effects (gravity, sparkle, fire, quiet bloom) consume
// audio/motion features. processSensors() drives these every frame from the
// v1 accel/gyro/audio packets. On sensor timeout they fall back to rest
// (energy/onset = 0, no tilt, accel flat) so each effect's idle/ambient
// branch renders unchanged.
struct SensorStub {
    int16_t ax, ay, az;   // accel counts (16384 = 1g); flat = (0,0,16384)
    int16_t gx, gy, gz;
};
static SensorStub sensorStub = { 0, 0, 16384, 0, 0, 0 };
static float fxEnergy   = 0.0f;   // 0–1 audio/motion energy
static float fxOnset    = 0.0f;   // 0–1 transient onset
static float fxTiltBlend = 0.0f;  // 0–1 tilt engagement
static float fxAngleDeg = 0.0f;   // tilt angle for hue mapping

// Creature interaction values, refreshed each frame from the sensor packets.
static float crShakeG_live  = 1.0f;   // accel magnitude in g (rest = 1g)
static float crScrollDps_live = 0.0f; // gyro yaw rate in deg/s

// ── Sensor → feature processing ──────────────────────────────────
// Decodes the v1 packets and drives the effect globals. Audio uses an
// adaptive floor + log scaling (ported from bulb-fleet computeEnergy/Onset,
// retuned for the decoded RMS scale). On timeout each layer falls back to its
// at-rest stub so the effects render their idle/ambient branch unchanged.
#define SNS_RMS_FLOOR_MIN  50.0f   // quiet-room floor in the 8000-FS domain
#define SNS_FLOOR_HEADROOM 1.4f
#define SNS_FLOOR_LEAK     0.005f
#define SNS_FLOOR_SNAP_EPS 0.05f
#define SNS_FLOOR_SOFT_SIG 0.6f

static float snsAdaptiveFloor = 0.0f;
static float snsRmsCeiling    = SNS_RMS_FLOOR_MIN;
static int   snsBelowFloorCnt = 0;
static float snsPrevRms       = 0.0f;
static float snsOnsetPeak     = 1e-6f;

static float snsUpdateFloor(float rms, float dt) {
    if (snsAdaptiveFloor < 1.0f) {
        snsAdaptiveFloor = fmaxf(rms, SNS_RMS_FLOOR_MIN);
        return snsAdaptiveFloor;
    }
    if (rms < snsAdaptiveFloor * (1.0f + SNS_FLOOR_SNAP_EPS)) {
        if (++snsBelowFloorCnt >= 3) {
            float target = fmaxf(rms, SNS_RMS_FLOOR_MIN);
            float snapAlpha = fminf(1.0f, dt / 0.11f);
            snsAdaptiveFloor += snapAlpha * (target - snsAdaptiveFloor);
        }
    } else {
        snsBelowFloorCnt = 0;
        float ratio = rms / fmaxf(snsAdaptiveFloor, 1.0f);
        float d = (ratio - 1.0f) / SNS_FLOOR_SOFT_SIG;
        snsAdaptiveFloor *= (1.0f + SNS_FLOOR_LEAK * dt * expf(-(d * d)));
    }
    snsAdaptiveFloor = fmaxf(snsAdaptiveFloor, SNS_RMS_FLOOR_MIN);
    return snsAdaptiveFloor;
}

static void processSensors(float dt) {
    uint32_t now = millis();

    // ── Accel ────────────────────────────────────────────────────
    if (accelLastMs > 0 && (now - accelLastMs) < SENSOR_TIMEOUT_MS) {
        TelemetryPacketV1 a;
        memcpy(&a, (const void*)&accelPkt, sizeof(a));

        // Per-axis means are (rawMean >> 8) → gravity/tilt vector in counts.
        // Reconstruct counts at the full-scale used by the stub (16384 = 1g).
        sensorStub.ax = (int16_t)((int)a.ax_mean << 8);
        sensorStub.ay = (int16_t)((int)a.ay_mean << 8);
        sensorStub.az = (int16_t)((int)a.az_mean << 8);

        float axg = (float)sensorStub.ax / 16384.0f;
        float ayg = (float)sensorStub.ay / 16384.0f;
        float azg = (float)sensorStub.az / 16384.0f;

        // Tilt: angle of the gravity vector away from the z (flat) axis.
        float horiz = sqrtf(axg * axg + ayg * ayg);
        fxAngleDeg = atan2f(horiz, fabsf(azg)) * 180.0f / (float)M_PI;
        fxTiltBlend = clampf((fxAngleDeg - DEADZONE_DEG)
                             / (MAX_ANGLE_DEG - DEADZONE_DEG), 0.0f, 1.0f);

        // Shake magnitude for creatures: decode amag_max (gravity included).
        float amagCounts = decodeCompand(a.amag_max, V1_AMAG_FS);
        crShakeG_live = amagCounts / V1_COUNTS_PER_G;
    } else {
        sensorStub.ax = 0; sensorStub.ay = 0; sensorStub.az = 16384;
        fxAngleDeg = 0.0f;
        fxTiltBlend = 0.0f;
        crShakeG_live = 1.0f;
    }

    // ── Gyro ─────────────────────────────────────────────────────
    if (gyroLastMs > 0 && (now - gyroLastMs) < SENSOR_TIMEOUT_MS) {
        GyroPacketV1 g;
        memcpy(&g, (const void*)&gyroPkt, sizeof(g));
        // gz_mean is the yaw-rate mean (int8 LSB ≈ V1_DPS_PER_LSB dps).
        crScrollDps_live = (float)g.gz_mean * V1_DPS_PER_LSB;
        sensorStub.gz = (int16_t)crScrollDps_live;
    } else {
        crScrollDps_live = 0.0f;
        sensorStub.gz = 0;
    }

    // ── Audio ────────────────────────────────────────────────────
    if (audioLastMs > 0 && (now - audioLastMs) < SENSOR_TIMEOUT_MS) {
        AudioPacketV1 au;
        memcpy(&au, (const void*)&audioPkt, sizeof(au));
        float rmsMean = decodeCompand(au.rms_mean, V1_RMS_FS);
        float rmsMax  = decodeCompand(au.rms_max,  V1_RMS_FS);

        // Energy: adaptive floor + log scaling against a leaky ceiling.
        snsRmsCeiling = fmaxf(SNS_RMS_FLOOR_MIN, snsRmsCeiling * expf(-0.0025f * dt));
        if (rmsMean > snsRmsCeiling) snsRmsCeiling = rmsMean;
        snsUpdateFloor(rmsMean, dt);
        float effFloor = snsAdaptiveFloor * SNS_FLOOR_HEADROOM;
        if (rmsMean < effFloor) {
            fxEnergy = 0.0f;
        } else {
            float db = 20.0f * log10f(rmsMean / effFloor);
            float dbRange = 20.0f * log10f(snsRmsCeiling / effFloor);
            if (dbRange < 1.0f) dbRange = 1.0f;
            fxEnergy = clampf(db / dbRange, 0.0f, 1.0f);
        }

        // Onset: in-window transient is the gap between window max and mean,
        // normalized against a decaying peak (mirrors computeOnset).
        float delta = fmaxf(0.0f, rmsMax - rmsMean);
        snsPrevRms = rmsMean;
        snsOnsetPeak = fmaxf(delta, snsOnsetPeak * expf(-1.3f * dt));
        fxOnset = (snsOnsetPeak > 1e-6f) ? clampf(delta / snsOnsetPeak, 0.0f, 1.0f) : 0.0f;
    } else {
        fxEnergy = 0.0f;
        fxOnset = 0.0f;
    }
}

// ── Per-strip bloom state ────────────────────────────────────────
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

static BloomStrip bloom[NUM_STRIPS];

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

// ── Bloom render (per strip, ambient only — no motion) ──────────
static void renderBloomStrip(uint8_t s, float dt) {
    BloomStrip &bs = bloom[s];

    // No motion processing — ambient breathing only.
    // Flash/energyBuffer stay at zero (no input).

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
        float baseR = lerpf(bloomHueA_R, bloomHueB_R, h);
        float baseG = lerpf(bloomHueA_G, bloomHueB_G, h);
        float baseB = lerpf(bloomHueA_B, bloomHueB_B, h);

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

        // Uniform global brightness scale before dither.
        oR *= renderBrightness;
        oG *= renderBrightness;
        oB *= renderBrightness;

        uint16_t t16R = (uint16_t)fminf(oR * 256.0f, 65535.0f);
        uint16_t t16G = (uint16_t)fminf(oG * 256.0f, 65535.0f);
        uint16_t t16B = (uint16_t)fminf(oB * 256.0f, 65535.0f);
        if ((t16R | t16G | t16B) == 0) {
            bs.ditherR[i] = bs.ditherG[i] = bs.ditherB[i] = 0;
        }
        uint8_t r8 = deltaSigma(bs.ditherR[i], t16R);
        uint8_t g8 = deltaSigma(bs.ditherG[i], t16G);
        uint8_t b8 = deltaSigma(bs.ditherB[i], t16B);

        setPixel(s, i, r8, g8, b8);
    }
}

// ── Gravity particle (ported from bulb-fleet renderGravitySparkle) ─
// Particles fall under accelerometer tilt and splat as OKLCH glows.
// Stubbed at rest (ax=0) → no gravity → particles settle, gentle static.
// No per-effect cap (renderBrightness via setPixelDither).
#define GS_PARTICLE_COUNT 7
#define GS_GRAVITY_SCALE  40.0f
#define GS_VELOCITY_DAMP  0.92f
#define GS_BOUNCE_REBOUND 0.5f
#define GS_SPLAT_RADIUS   2.5f

struct GsParticle {
    float pos;
    float vel;
    float bright;
    float hue;
};
static GsParticle gsParticles[NUM_STRIPS][GS_PARTICLE_COUNT];

static void resetGravity() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
            gsParticles[s][i].pos = (float)(LEDS_PER_STRIP - 1)
                * (float)i / (float)(GS_PARTICLE_COUNT - 1);
            gsParticles[s][i].vel = 0.0f;
            gsParticles[s][i].bright = 1.0f;
            float baseHue = 256.0f * (float)i / (float)GS_PARTICLE_COUNT;
            gsParticles[s][i].hue = fmodf(baseHue + (float)STRIP_HUE_OFFSET[s], 256.0f);
        }
    }
}

static void renderGravity(float dt) {
    float gravG = clampf((float)sensorStub.ax / 16384.0f, -1.5f, 1.5f);
    float accel = gravG * GS_GRAVITY_SCALE;
    float damp = fastDecay(GS_VELOCITY_DAMP, dt * 30.0f);

    static float accR[LEDS_PER_STRIP], accG[LEDS_PER_STRIP], accB[LEDS_PER_STRIP];
    const float maxPos = (float)(LEDS_PER_STRIP - 1);
    const float invTwoSigSq = 1.0f / (2.0f * GS_SPLAT_RADIUS * GS_SPLAT_RADIUS);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            accR[i] = 0; accG[i] = 0; accB[i] = 0;
        }

        for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
            GsParticle &p = gsParticles[s][i];

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
            int hi = center + 3; if (hi > (int)(LEDS_PER_STRIP - 1)) hi = LEDS_PER_STRIP - 1;
            for (int j = lo; j <= hi; j++) {
                float d = (float)j - p.pos;
                float w = expf(-(d * d) * invTwoSigSq);
                accR[j] += colR * w;
                accG[j] += colG * w;
                accB[j] += colB * w;
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r = accR[i];
            float g = accG[i];
            float b = accB[i];
            float maxCh = fmaxf(r, fmaxf(g, b));
            float bright = clampf(maxCh / 255.0f, 0.0f, 1.0f);
            float linBright = fastGamma24(bright);
            float norm = (bright > 0.001f) ? (linBright / bright) : 0.0f;
            setPixelDither(s, i,
                clampf(r * norm, 0.0f, 255.0f),
                clampf(g * norm, 0.0f, 255.0f),
                clampf(b * norm, 0.0f, 255.0f));
        }
    }
}

// ── Sparkle syllable (ported from bulb-fleet) ───────────────────
// Onset-triggered LED ignition over a dim warm base. Stubbed: no onset,
// no tilt → renders the quiet warm-amber floor. No per-effect cap.
#define SPARKLE_DEADBAND 0.08f

static float syllSparkle[NUM_STRIPS][LEDS_PER_STRIP];
static float syllDecayArr[NUM_STRIPS][LEDS_PER_STRIP];
static float syllEnvelope = 0.0f;
static float syllCooldown = 0.0f;

static void resetSparkle() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            syllSparkle[s][i] = 0.0f;
            syllDecayArr[s][i] = 0.94f;
        }
    }
    syllEnvelope = 0.0f;
    syllCooldown = 0.0f;
}

static void renderSparkle(float dt) {
    float energy = fxEnergy;
    float onsetNorm = fxOnset;
    float angleDeg = fxAngleDeg;
    float tiltBlend = fxTiltBlend;

    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > syllEnvelope)
        syllEnvelope += attackAlpha * (energy - syllEnvelope);
    else
        syllEnvelope += decayAlpha * (energy - syllEnvelope);

    syllCooldown = fmaxf(0.0f, syllCooldown - dt);

    if (onsetNorm > 0.1f && syllCooldown <= 0.0f) {
        syllCooldown = 0.060f;
        int nIgnite = (int)(LEDS_PER_STRIP * (0.25f + 0.25f * onsetNorm));
        float sparkVal = 0.6f + 0.4f * onsetNorm;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            static uint16_t indices[LEDS_PER_STRIP];
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) indices[i] = i;
            for (int i = 0; i < nIgnite; i++) {
                int j = i + (int)(xorshift32() % (LEDS_PER_STRIP - i));
                uint16_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
            for (int i = 0; i < nIgnite; i++) {
                syllSparkle[s][indices[i]] = sparkVal;
                syllDecayArr[s][indices[i]] = 0.92f + randFloat() * 0.05f;
            }
        }
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            syllSparkle[s][i] *= fastDecay(syllDecayArr[s][i], dt * 30.0f);

    float base = fminf(syllEnvelope, 0.15f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float tiltR = 0, tiltG = 0, tiltB = 0;
        if (tiltBlend > 0.0f) {
            float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
            if (hueFrac < 0.0f) hueFrac = 0.0f;
            if (hueFrac > 1.0f) hueFrac = 1.0f;
            uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255)
                                         + STRIP_HUE_OFFSET[s]) & 0xFF);
            tiltR = (float)oklchVarL[hueIdx][0];
            tiltG = (float)oklchVarL[hueIdx][1];
            tiltB = (float)oklchVarL[hueIdx][2];
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float sp = syllSparkle[s][i];
            float bright = base + sp * (1.0f - base);
            if (bright < SPARKLE_DEADBAND) bright = 0.0f;

            float colR = 255.0f;
            float colG = 180.0f + (240.0f - 180.0f) * sp;
            float colB =  80.0f + (200.0f -  80.0f) * sp;

            if (tiltBlend > 0.0f) {
                colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
                colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
                colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
            }

            float wFold = 127.0f * (1.0f - tiltBlend);
            colR = fminf(255.0f, colR + wFold);
            colG = fminf(255.0f, colG + wFold);
            colB = fminf(255.0f, colB + wFold);

            float linBright = fastGamma24(bright);
            setPixelDither(s, i,
                clampf(colR * linBright, 0.0f, 255.0f),
                clampf(colG * linBright, 0.0f, 255.0f),
                clampf(colB * linBright, 0.0f, 255.0f));
        }
    }
}

// ── Fire (ported from bulb-fleet renderFire) ────────────────────
// Audio-reactive flame; stubbed silent → calm amber idle flicker.
// withDropout distinguishes fire_meld (true) from fire_flicker (false).
// No per-effect cap.
#define FIRE_FLICKER_SCALE 3.0f
#define FIRE_DEADBAND      0.08f
#define FIRE_DROPOUT_DEPTH 0.85f

static float fireTime[NUM_STRIPS];
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

static void resetFire() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) fireTime[s] = (float)s * 1.37f;
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

static void renderFire(float dt, bool withDropout) {
    float energy = fxEnergy;
    float onset = fxOnset;
    bool isSilent = energy < 0.001f;
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);

    float attackAlpha = fminf(1.0f, dt / 0.050f);
    float decayAlpha  = fminf(1.0f, dt / 2.0f);

    float targetBrightness = isSilent ? 0.25f : fmaxf(0.25f, energy);
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
    if (isPercussiveOnly)   colorTarget = 0.0f;
    else if (isSilent)      colorTarget = 0.3f;
    else                    colorTarget = fmaxf(0.3f, energy);
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
    float sScl = FIRE_FLICKER_SCALE;

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        fireTime[s] = fmodf(fireTime[s] + dt, 6283.1853f);
        float t = fireTime[s];
        float sOff = (float)s * 17.0f;

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float fi = (float)i + sOff;
            float noise = fastSin(fi * 7.3f + t * 2.5f) *
                           fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;
            float noiseAmp = fmaxf(0.15f * sScl, 0.10f * sScl / fmaxf(base, 0.1f));
            float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                            + fireFlickerIntensity * (noise - 0.5f) * 0.25f * sScl;

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

            float linBright = fastGamma24(bright);
            setPixelDither(s, i,
                clampf(colR * linBright, 0.0f, 255.0f),
                clampf(colG * linBright, 0.0f, 255.0f),
                clampf(colB * linBright, 0.0f, 255.0f));
        }
    }
}

// ── Quiet bloom (ported from bulb-fleet renderQuietBloom) ───────
// Motion-reactive colony breathing. Stubbed: no motion drain → strips
// just breathe ambiently. No per-effect cap.
#define QB_SURPRISE_RATIO     3.0f
#define QB_ENERGY_MULTIPLIER  1.4f
#define QB_RECOVERY_RAMP      0.033f
#define QB_RECOVERY_SPREAD    0.70f
#define QB_BREATH_MIN_PERIOD  3.0f
#define QB_BREATH_MAX_PERIOD  8.0f
#define QB_BREATH_MIN_PEAK    0.65f
#define QB_BREATH_MAX_PEAK    1.00f
#define QB_BREATH_FLOOR       0.15f
#define QB_FLASH_DECAY_LO     0.96f
#define QB_FLASH_DECAY_HI     0.985f
#define QB_HUE_A_G   20.0f
#define QB_HUE_A_B  100.0f
#define QB_HUE_B_G   70.0f
#define QB_HUE_B_B  110.0f
#define QB_FLASH_G  150.0f
#define QB_FLASH_B  170.0f
#define QB_W_ONSET    0.5f

static float qbBreathPhase [NUM_STRIPS][LEDS_PER_STRIP];
static float qbBreathPeriod[NUM_STRIPS][LEDS_PER_STRIP];
static float qbBreathPeak  [NUM_STRIPS][LEDS_PER_STRIP];
static float qbHueT        [NUM_STRIPS][LEDS_PER_STRIP];
static float qbFlashGlow   [NUM_STRIPS][LEDS_PER_STRIP];
static float qbFlashDecay  [NUM_STRIPS][LEDS_PER_STRIP];
static float qbColonyEnergy = 1.0f;
static float qbDrainEnvelope = 0.0f;
static float qbHitIntensity = 0.0f;

static void resetQuietBloom() {
    qbColonyEnergy = 1.0f;
    qbDrainEnvelope = 0.0f;
    qbHitIntensity = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            qbBreathPhase[s][i] = randFloat();
            qbBreathPeriod[s][i] = QB_BREATH_MIN_PERIOD
                + randFloat() * (QB_BREATH_MAX_PERIOD - QB_BREATH_MIN_PERIOD);
            qbBreathPeak[s][i] = QB_BREATH_MIN_PEAK
                + randFloat() * (QB_BREATH_MAX_PEAK - QB_BREATH_MIN_PEAK);
            qbHueT[s][i] = randFloat();
            qbFlashGlow[s][i] = 0.0f;
            qbFlashDecay[s][i] = QB_FLASH_DECAY_LO
                + randFloat() * (QB_FLASH_DECAY_HI - QB_FLASH_DECAY_LO);
        }
    }
}

static void renderQuietBloom(float dt) {
    bool draining = qbDrainEnvelope > 0.001f;
    if (!draining) qbHitIntensity = 0.0f;

    // No motion source when stubbed → colony always recovering to full.
    qbColonyEnergy = fminf(1.0f, qbColonyEnergy + QB_RECOVERY_RAMP * dt);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            if (!draining) {
                qbFlashGlow[s][i] *= fastDecay(qbFlashDecay[s][i], dt * 30.0f);
                if (qbFlashGlow[s][i] < 0.005f) qbFlashGlow[s][i] = 0.0f;
            }

            qbBreathPhase[s][i] += dt / qbBreathPeriod[s][i];
            if (qbBreathPhase[s][i] >= 1.0f) qbBreathPhase[s][i] -= 1.0f;

            float wakeThresh = qbHueT[s][i] * QB_RECOVERY_SPREAD;
            float ledRecovery = clampf(
                (qbColonyEnergy - wakeThresh) / 0.30f, 0.0f, 1.0f);

            float breath = (fastSinPhase(qbBreathPhase[s][i]) * 0.5f + 0.5f);
            float breathGlow = QB_BREATH_FLOOR
                + breath * (qbBreathPeak[s][i] - QB_BREATH_FLOOR);
            breathGlow *= ledRecovery;

            if (draining) {
                float target = breathGlow * qbHitIntensity * QB_ENERGY_MULTIPLIER;
                if (target > qbFlashGlow[s][i]) {
                    qbFlashGlow[s][i] = target;
                    qbFlashDecay[s][i] = QB_FLASH_DECAY_LO
                        + randFloat() * (QB_FLASH_DECAY_HI - QB_FLASH_DECAY_LO);
                }
            }

            float g = fmaxf(breathGlow, qbFlashGlow[s][i]);
            float flashFrac = (qbFlashGlow[s][i] > breathGlow) ? 1.0f : 0.0f;
            float h = qbHueT[s][i];
            float colG = lerpf(lerpf(QB_HUE_A_G, QB_HUE_B_G, h), QB_FLASH_G, flashFrac);
            float colB = lerpf(lerpf(QB_HUE_A_B, QB_HUE_B_B, h), QB_FLASH_B, flashFrac);

            float linBright = fastGamma24(g);
            float oG = colG * linBright;
            float oB = colB * linBright;

            float wFrac = clampf((g - QB_W_ONSET) / (1.0f - QB_W_ONSET), 0.0f, 1.0f);
            float energyGate = clampf((qbColonyEnergy - 0.7f) / 0.3f, 0.0f, 1.0f);
            float wGate = fmaxf(energyGate, flashFrac);
            float wFold = wFrac * wGate * linBright * 200.0f;

            setPixelDither(s, i,
                clampf(wFold, 0.0f, 255.0f),
                clampf(oG + wFold, 0.0f, 255.0f),
                clampf(oB + wFold, 0.0f, 255.0f));
        }
    }

    if (draining) {
        float drain = qbDrainEnvelope * dt;
        drain = fminf(drain, qbColonyEnergy);
        qbColonyEnergy -= drain;
        qbDrainEnvelope *= expf(-4.07f * dt);
        if (qbDrainEnvelope <= 0.001f) qbDrainEnvelope = 0.0f;
    }
}

// ── Nebula (ported from bulb-fleet) ─────────────────────────────
// Breathing blue→magenta background + warm-white drifting orbs.
// No audio dependency. dt is pre-scaled by effSpeed at the call site;
// the intrinsic 0.3 tuning factor is kept (it is not the speed knob).
#define NEBULA_MAX_ORBS       5
#define NEBULA_ORB_TAIL       30.0f
#define NEBULA_ORB_BASE_SPEED 0.45f
#define NEBULA_SPAWN_CHANCE   0.03f   // per-frame @30fps reference; dt-scaled at spawn
#define NEBULA_MIN_LIFETIME   200     // frames @60fps reference; /60 to seconds at spawn
#define NEBULA_MAX_LIFETIME   300
#define NEBULA_INTRINSIC_SPD  0.3f

struct NebOrb {
    float pos;
    float vel;
    float age;
    float lifetime;
    bool active;
};

static float nebulaTime = 0.0f;
static NebOrb nebOrbs[NUM_STRIPS][NEBULA_MAX_ORBS];
static float nebDecay[NUM_STRIPS][LEDS_PER_STRIP];

static void resetNebula() {
    nebulaTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++)
            nebOrbs[s][o].active = false;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            nebDecay[s][i] = 0.0f;
    }
}

static void renderNebula(float dt) {
    float spd = NEBULA_INTRINSIC_SPD;
    nebulaTime += dt * spd;
    float t = nebulaTime * 60.0f;

    float decayPerFrame = 1.0f - (1.0f / NEBULA_ORB_TAIL);
    float decayPerSec = powf(decayPerFrame, 60.0f);
    float decay = powf(decayPerSec, dt);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            nebDecay[s][i] *= decay;
            if (nebDecay[s][i] < 0.01f) nebDecay[s][i] = 0.0f;
        }

        float spawnRoll = (float)(xorshift32() & 0xFFFF) / 65536.0f;
        if (spawnRoll < NEBULA_SPAWN_CHANCE * dt * 30.0f) {
            for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
                if (!nebOrbs[s][o].active) {
                    NebOrb &orb = nebOrbs[s][o];
                    orb.pos = randFloat() * (float)LEDS_PER_STRIP;
                    float dir = (xorshift32() & 1) ? 1.0f : -1.0f;
                    orb.vel = dir * NEBULA_ORB_BASE_SPEED * spd * (0.7f + randFloat() * 0.6f);
                    orb.age = 0.0f;
                    orb.lifetime = (NEBULA_MIN_LIFETIME
                        + (float)(xorshift32() % (NEBULA_MAX_LIFETIME - NEBULA_MIN_LIFETIME)))
                        / 60.0f;
                    orb.active = true;
                    break;
                }
            }
        }

        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
            NebOrb &orb = nebOrbs[s][o];
            if (!orb.active) continue;

            orb.age += dt;
            orb.pos += orb.vel * dt * 60.0f;
            orb.pos = fmodf(orb.pos + (float)LEDS_PER_STRIP, (float)LEDS_PER_STRIP);

            if (orb.age >= orb.lifetime) {
                orb.active = false;
                continue;
            }

            float lc = (float)orb.age / (float)orb.lifetime;
            float bright;
            if (lc < 0.4f) {
                float tf = lc / 0.4f;
                bright = tf * tf * (3.0f - 2.0f * tf);
            } else if (lc > 0.6f) {
                float tf = (1.0f - lc) / 0.4f;
                bright = tf * tf * (3.0f - 2.0f * tf);
            } else {
                bright = 1.0f;
            }

            int base = (int)orb.pos;
            int next = (base + 1) % LEDS_PER_STRIP;
            float frac = orb.pos - (float)base;
            float val0 = bright * 0.6f * (1.0f - frac);
            float val1 = bright * 0.6f * frac;
            if (base >= 0 && base < LEDS_PER_STRIP)
                nebDecay[s][base] = fminf(1.0f, nebDecay[s][base] + val0);
            if (next >= 0 && next < LEDS_PER_STRIP)
                nebDecay[s][next] = fminf(1.0f, nebDecay[s][next] + val1);
        }

        float sOff = (float)s * 0.167f;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float pos = (float)i / (float)LEDS_PER_STRIP;

            float breathing = 51.0f + 38.0f * fastSin((t * 0.0105f));
            float phase = pos + sOff + t * 0.006f;
            float spatial = 51.0f * (0.5f + 0.5f * fastSin(phase * 6.2832f));
            float bgBright = clampf(breathing + spatial, 0.0f, 153.0f) / 255.0f;

            float colorPhase = pos * 6.2832f + sOff * 6.2832f + t * 0.009f;
            float colorShift = 0.5f + 0.5f * fastSin(colorPhase);

            float bgR = (20.0f + colorShift * 235.0f) / 255.0f;
            float bgG = (30.0f - colorShift * 20.0f) / 255.0f;
            float bgB = (255.0f - colorShift * 125.0f) / 255.0f;

            float r = bgR * bgBright;
            float g = bgG * bgBright;
            float b = bgB * bgBright;

            float orbB = nebDecay[s][i];
            if (orbB > 0.01f) {
                r += orbB * 1.0f;
                g += orbB * 0.94f;
                b += orbB * 0.78f;
            }

            setPixelDither(s, i,
                clampf(r, 0.0f, 1.0f) * 255.0f,
                clampf(g, 0.0f, 1.0f) * 255.0f,
                clampf(b, 0.0f, 1.0f) * 255.0f);
        }
    }
}

// ── Rainbow (OKLCH scroll, ported from bulb-fleet renderIdle) ───
// No audio dependency. Pure ambient hue scroll. dt is pre-scaled by
// effSpeed at the call site. No per-effect brightness cap — output
// scale is renderBrightness via setPixelDither.
#define RAINBOW_SCROLL_SPEED  0.10f

static float rainbowPhase = 0.0f;

static void resetRainbow() {
    rainbowPhase = 0.0f;
}

static void renderRainbow(float dt) {
    rainbowPhase = fmodf(rainbowPhase + RAINBOW_SCROLL_SPEED * dt, 1.0f);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float stripOff = STRIP_HUE_OFFSET[s] / 256.0f;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float pos = (float)i / (float)LEDS_PER_STRIP;
            float hue = fmodf(pos + rainbowPhase + stripOff, 1.0f);
            uint8_t idx = (uint8_t)(hue * 255.0f);
            setPixelDither(s, i,
                (float)oklchVarL[idx][0],
                (float)oklchVarL[idx][1],
                (float)oklchVarL[idx][2]);
        }
    }
}

// ── Leaf wind (ported from bulb-fleet renderLeafWind) ───────────
// 1D drift along each strip: leaves blow from one end to the other, each LED
// lights by 1D distance to each leaf. Bulb-fleet's 2D topology.h is a hex map
// for a different installation, so tree-of-record collapses to a simple linear
// topology — LED index → position 0..1 (i / (LEDS_PER_STRIP-1)). Each leaf
// drifts independently per strip. No per-effect cap — output via setPixelDither.
#define LW_MAX_LEAVES     8     // per strip
#define LW_GLOW_RADIUS    0.10f
#define LW_GLOW_SQ2       (2.0f * LW_GLOW_RADIUS * LW_GLOW_RADIUS)
#define LW_WIND_SPEED     0.35f
#define LW_SPAWN_INTERVAL 0.5f
#define LW_FADE_IN        0.4f
#define LW_VEL_TAU        0.22f   // velocity EMA time constant (sec); = orig 0.85/frame @30fps
#define LW_TURBULENCE     0.3f
#define LW_BOOST_SPEED    0.25f
#define LW_BOOST_TC       2.5f

static const uint8_t LW_PALETTE[][3] = {
    {255, 140, 20}, {240, 100, 10}, {220, 60, 5}, {200, 40, 10},
    {180, 30, 5},   {255, 180, 40}, {160, 25, 5},
};
#define LW_PALETTE_SIZE (sizeof(LW_PALETTE) / sizeof(LW_PALETTE[0]))

struct LwLeaf {
    float pos, vel, boost, age, brightness;
    uint8_t r, g, b;
    bool active;
};
static LwLeaf lwLeaves[NUM_STRIPS][LW_MAX_LEAVES];
static float lwTime = 0.0f;
static float lwSpawnTimer[NUM_STRIPS];

static float lwNoise1d(float pos, float t, int seed) {
    return (fastSin(pos * 0.4f + t * 0.3f + seed * 7.3f)
          * fastSin(pos * 0.17f - t * 0.19f + seed * 3.1f + (float)M_PI * 0.5f)
          + fastSin(pos * 0.09f + t * 0.13f + seed * 1.7f) * 0.5f) / 1.5f;
}

static void resetLeafWind() {
    lwTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        lwSpawnTimer[s] = randFloat() * LW_SPAWN_INTERVAL;
        for (int i = 0; i < LW_MAX_LEAVES; i++) lwLeaves[s][i].active = false;
    }
}

static void lwSpawnLeaf(uint8_t s) {
    for (int i = 0; i < LW_MAX_LEAVES; i++) {
        if (lwLeaves[s][i].active) continue;
        LwLeaf &lf = lwLeaves[s][i];
        lf.active = true;
        lf.pos = -0.05f;   // enter from LED 0 end, blow toward the tip
        lf.boost = LW_BOOST_SPEED * (0.5f + randFloat() * 0.5f);
        lf.vel = 0.0f;
        lf.age = 0.0f;
        lf.brightness = 0.0f;
        int ci = (int)(xorshift32() % LW_PALETTE_SIZE);
        lf.r = LW_PALETTE[ci][0];
        lf.g = LW_PALETTE[ci][1];
        lf.b = LW_PALETTE[ci][2];
        return;
    }
}

static void renderLeafWind(float dt) {
    lwTime += dt;
    float boostDecay = expf(-dt / LW_BOOST_TC);
    float lwAlpha = fminf(1.0f, dt / LW_VEL_TAU);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        lwSpawnTimer[s] += dt;
        while (lwSpawnTimer[s] >= LW_SPAWN_INTERVAL) {
            lwSpawnTimer[s] -= LW_SPAWN_INTERVAL;
            lwSpawnLeaf(s);
        }

        for (int i = 0; i < LW_MAX_LEAVES; i++) {
            if (!lwLeaves[s][i].active) continue;
            LwLeaf &lf = lwLeaves[s][i];

            float noise = lwNoise1d(lf.pos * 5.0f + (float)s * 3.0f, lwTime, i);
            float speedMult = fmaxf(0.1f, 1.0f + noise * LW_TURBULENCE);
            float force = LW_WIND_SPEED * speedMult;

            lf.vel = fmaxf(LW_WIND_SPEED * 0.4f, lf.vel + lwAlpha * (force - lf.vel));
            lf.boost *= boostDecay;
            lf.pos += (lf.vel + lf.boost) * dt;
            lf.age += dt;

            lf.brightness = (lf.age < LW_FADE_IN) ? (lf.age / LW_FADE_IN) : 1.0f;

            if (lf.pos > 1.05f) lf.active = false;
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float lpos = (float)i / (float)(LEDS_PER_STRIP - 1);
            float totalGlow = 0.0f, cr = 0.0f, cg = 0.0f, cb = 0.0f;

            for (int li = 0; li < LW_MAX_LEAVES; li++) {
                if (!lwLeaves[s][li].active) continue;
                LwLeaf &lf = lwLeaves[s][li];
                float d = lpos - lf.pos;
                float intensity = expf(-(d * d) / LW_GLOW_SQ2) * lf.brightness;
                if (intensity < 0.005f) continue;
                totalGlow += intensity;
                cr += intensity * lf.r;
                cg += intensity * lf.g;
                cb += intensity * lf.b;
            }

            if (totalGlow > 0.01f) {
                cr /= totalGlow; cg /= totalGlow; cb /= totalGlow;
                float bright = fminf(totalGlow, 1.0f);
                float linBright = fastGamma24(bright);
                setPixelDither(s, i, cr * linBright, cg * linBright, cb * linBright);
            } else {
                setPixelDither(s, i, 0.0f, 0.0f, 0.0f);
            }
        }
    }
}

// ── Creatures (ported from biolum_mixed) ────────────────────────
// Bloom + crawl creatures drift along a virtual buffer per strip-pair.
// Shake (accel) and gyro-scroll interaction are driven by crShakeFeedG()/
// crScrollDps(), which processSensors() feeds from the v1 accel/gyro packets.
// On sensor timeout they return rest (1g shake, 0 rotation) and the visuals
// fall back to pure ambient drift. Virtual buffer is 100
// physical LEDs + a SCROLL_MARGIN each side (was 150/200 on the bullet build).
// No per-effect cap — output via setPixelDither.
#define CR_NUM_PAIRS     3
#define CR_MAX_CREATURES 7
#define CR_MAX_ANIMS     6
#define CR_SCROLL_MARGIN 12
#define CR_VIRTUAL_LEDS  (LEDS_PER_STRIP + 2 * CR_SCROLL_MARGIN)
#define CR_SPAWN_MARGIN  5
#define CR_DESPAWN_MARGIN (-5)

#define CR_COLOR_R   0.0f
#define CR_COLOR_G 180.0f
#define CR_COLOR_B 220.0f

// --- Shake interaction params (from biolum_mixed) ---
#define CR_SHAKE_THRESH_G       2.0f
#define CR_SHAKE_BUFFER_GAIN    0.8f
#define CR_SHAKE_BUFFER_MAX     0.5f
#define CR_SHAKE_DRAIN_RATE     0.8f
#define CR_SHAKE_BASE_FALL      0.5f
#define CR_SHAKE_FALL_SLOWDOWN  0.5f
#define CR_SHAKE_SCATTER_RADIUS 1.5f
#define CR_SHAKE_BRIGHT_BOOST   0.8f
#define CR_SHAKE_HUE_DRIFT      90.0f

// --- Gyro scroll params (from biolum_mixed) ---
#define CR_SCROLL_GAIN  ((float)CR_SCROLL_MARGIN / 360.0f)  // px per degree
#define CR_SCROLL_DECAY 0.3f

#define CR_PULSE_EXPANSION_SPEED 3.3f
#define CR_PULSE_TO_DRIFT_RATIO  1.9f
#define CR_AVG_DRIFT  (CR_PULSE_EXPANSION_SPEED / CR_PULSE_TO_DRIFT_RATIO)
#define CR_DRIFT_SPREAD 0.33f
#define CR_DRIFT_MIN  (CR_AVG_DRIFT * (1.0f - CR_DRIFT_SPREAD))
#define CR_DRIFT_MAX  (CR_AVG_DRIFT * (1.0f + CR_DRIFT_SPREAD))

#define CR_BLOOM_RADIUS        3.0f
#define CR_BLOOM_EDGE_SOFTNESS 0.8f
#define CR_BLOOM_RISE          2.0f
#define CR_BLOOM_HOLD          1.5f
#define CR_BLOOM_FALL          5.0f
#define CR_BLOOM_TOTAL  (CR_BLOOM_RISE + CR_BLOOM_HOLD + CR_BLOOM_FALL)
#define CR_BLOOM_EMIT_LO 1.2f
#define CR_BLOOM_EMIT_HI 2.8f

#define CR_CRAWL_RADIUS         8.0f
#define CR_CRAWL_PULSE_LIFETIME (CR_CRAWL_RADIUS / CR_PULSE_EXPANSION_SPEED)
#define CR_CRAWL_PULSE_FADE     1.4f
#define CR_CRAWL_TAIL_DECAY     4.0f
#define CR_CRAWL_EMIT_LO        1.2f
#define CR_CRAWL_EMIT_HI        2.8f

enum CrKind : uint8_t { CR_KIND_BLOOM, CR_KIND_CRAWL };

struct CrAnim { float age; bool active; };
struct CrCreature {
    float    pos, vel;
    CrKind   kind;
    bool     alive;
    float    emitTimer;
    CrAnim   anims[CR_MAX_ANIMS];
    float    hueOffset, hueSweep;
};
struct CrPair {
    CrCreature creatures[CR_MAX_CREATURES];
    float bufR[CR_VIRTUAL_LEDS], bufG[CR_VIRTUAL_LEDS], bufB[CR_VIRTUAL_LEDS];
};
static CrPair crPairs[CR_NUM_PAIRS];

// --- Global shake / scroll state (driven by processInteraction) ---
static float crShakeLevel   = 0.0f;   // smooth 0→1 controlling visual effects
static float crShakeBuffer  = 0.0f;   // energy buffer (spiky input → smooth drain)
static float crShakeTime    = 0.0f;   // accumulated seconds shaking
static float crScrollOffset = 0.0f;
static bool  crShakeActive  = false;
static float crShakeCooldown = 2.0f;
static bool  crLifecycleFrozen = false;
static float crShakeHueDrift = 0.0f;

static inline float crRandf(float lo, float hi) {
    return lo + randFloat() * (hi - lo);
}

// Sensor accessors — driven by processSensors() from the v1 accel/gyro
// packets. On timeout these fall back to rest (1g shake, 0 scroll).
static inline float crShakeFeedG() {
    return crShakeG_live;   // accel magnitude in g (rest = 1g → no shake)
}
static inline float crScrollDps() {
    return crScrollDps_live;  // gyro yaw rate in deg/s
}

static void crProcessInteraction(float dt) {
    float aG = crShakeFeedG();
    // At rest aG ≈ 1g (gravity), well under the 2g shake threshold.
    if (aG > CR_SHAKE_THRESH_G) {
        crShakeActive = true;
        float excess = aG - CR_SHAKE_THRESH_G;
        crShakeBuffer += excess * CR_SHAKE_BUFFER_GAIN * dt;
        if (crShakeBuffer > CR_SHAKE_BUFFER_MAX) crShakeBuffer = CR_SHAKE_BUFFER_MAX;
        crShakeTime += dt;
    } else {
        crShakeActive = false;
    }

    float drained = fminf(crShakeBuffer, CR_SHAKE_DRAIN_RATE * sqrtf(crShakeBuffer) * dt);
    crShakeBuffer -= drained;

    if (crShakeActive) {
        crShakeLevel = fminf(1.0f, crShakeLevel + drained);
    } else {
        float fallRate = CR_SHAKE_BASE_FALL / (1.0f + crShakeTime * CR_SHAKE_FALL_SLOWDOWN);
        crShakeLevel -= fallRate * dt;
        if (crShakeLevel <= 0.0f) { crShakeLevel = 0.0f; crShakeTime = 0.0f; }
    }

    if (crShakeLevel > 0.2f) crShakeCooldown = 0.0f;
    else                     crShakeCooldown += dt;
    crLifecycleFrozen = (crShakeLevel > 0.2f || crShakeCooldown < 0.3f);

    crShakeHueDrift += crShakeLevel * CR_SHAKE_HUE_DRIFT * dt;
    if (crShakeHueDrift > 360.0f) crShakeHueDrift -= 360.0f;

    // Gyro scroll: integrate angular velocity, decay back to center.
    crScrollOffset += crScrollDps() * dt * CR_SCROLL_GAIN;
    crScrollOffset *= expf(-dt / CR_SCROLL_DECAY);
}

static void crInitCreature(CrCreature &c) {
    c.kind = (xorshift32() & 1) ? CR_KIND_BLOOM : CR_KIND_CRAWL;
    float speed = crRandf(CR_DRIFT_MIN, CR_DRIFT_MAX);
    c.pos = crRandf(CR_SPAWN_MARGIN, CR_VIRTUAL_LEDS - CR_SPAWN_MARGIN);
    c.vel = (xorshift32() & 1) ? speed : -speed;
    c.alive = true;
    c.emitTimer = (c.kind == CR_KIND_BLOOM)
        ? crRandf(CR_BLOOM_EMIT_LO, CR_BLOOM_EMIT_HI)
        : crRandf(CR_CRAWL_EMIT_LO, CR_CRAWL_EMIT_HI);
    for (int i = 0; i < CR_MAX_ANIMS; i++) c.anims[i].active = false;
    c.hueOffset = crRandf(0.0f, 360.0f);
    c.hueSweep = crRandf(60.0f, 90.0f);
}

static void resetCreatures() {
    for (int p = 0; p < CR_NUM_PAIRS; p++)
        for (int c = 0; c < CR_MAX_CREATURES; c++)
            crInitCreature(crPairs[p].creatures[c]);
}

static void crEmitAnim(CrCreature &c) {
    for (int i = 0; i < CR_MAX_ANIMS; i++) {
        if (!c.anims[i].active) {
            c.anims[i].active = true;
            c.anims[i].age = 0.0f;
            return;
        }
    }
}

static void crUpdateCreature(CrCreature &c, float dt) {
    if (!c.alive) return;
    c.pos += c.vel * dt;
    if (c.pos < CR_DESPAWN_MARGIN || c.pos > CR_VIRTUAL_LEDS - CR_DESPAWN_MARGIN) {
        c.alive = false;
        return;
    }
    c.emitTimer -= dt;
    if (c.emitTimer <= 0.0f) {
        crEmitAnim(c);
        if (c.kind == CR_KIND_BLOOM) {
            float base = CR_BLOOM_TOTAL + crRandf(CR_BLOOM_EMIT_LO, CR_BLOOM_EMIT_HI);
            c.emitTimer = base * crRandf(0.33f, 2.0f);
        } else {
            c.emitTimer = CR_CRAWL_PULSE_LIFETIME + CR_CRAWL_PULSE_FADE
                          + crRandf(CR_CRAWL_EMIT_LO, CR_CRAWL_EMIT_HI);
        }
    }
    float maxAge = (c.kind == CR_KIND_BLOOM) ? CR_BLOOM_TOTAL
                   : (CR_CRAWL_PULSE_LIFETIME + CR_CRAWL_PULSE_FADE);
    for (int i = 0; i < CR_MAX_ANIMS; i++) {
        if (!c.anims[i].active) continue;
        c.anims[i].age += dt;
        if (c.anims[i].age >= maxAge) c.anims[i].active = false;
    }
}

// Shake-aware color: blend base teal toward a per-position rainbow.
static inline void crShakeColor(const CrCreature &c, float relOut,
                                float &pixR, float &pixG, float &pixB) {
    pixR = CR_COLOR_R; pixG = CR_COLOR_G; pixB = CR_COLOR_B;
    if (crShakeLevel > 0.01f) {
        float colorT = crShakeLevel;
        float hueFrac = (relOut + 1.0f) * 0.5f;
        float hue = fmodf(c.hueOffset + crShakeHueDrift + hueFrac * c.hueSweep, 360.0f);
        float rR, rG, rB;
        hsvToRgb(hue, 1.0f, 1.0f, rR, rG, rB);
        pixR = CR_COLOR_R * (1.0f - colorT) + rR * colorT;
        pixG = CR_COLOR_G * (1.0f - colorT) + rG * colorT;
        pixB = CR_COLOR_B * (1.0f - colorT) + rB * colorT;
    }
}

static void crRenderBloom(const CrCreature &c, float *bufR, float *bufG, float *bufB) {
    float center = c.pos;
    float halfWidth = CR_BLOOM_RADIUS + CR_BLOOM_EDGE_SOFTNESS;
    float totalSpan = halfWidth * (1.0f + crShakeLevel * CR_SHAKE_SCATTER_RADIUS);
    int lo = (int)(center - totalSpan - 1); if (lo < 0) lo = 0;
    int hi = (int)(center + totalSpan + 1); if (hi > CR_VIRTUAL_LEDS - 1) hi = CR_VIRTUAL_LEDS - 1;

    float brightMult = 1.0f + crShakeLevel * CR_SHAKE_BRIGHT_BOOST;

    for (int a = 0; a < CR_MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float age = c.anims[a].age;
        float envelope;
        if (age < CR_BLOOM_RISE) {
            envelope = age / CR_BLOOM_RISE;
        } else if (age < CR_BLOOM_RISE + CR_BLOOM_HOLD) {
            envelope = 1.0f;
        } else {
            float fallT = (age - CR_BLOOM_RISE - CR_BLOOM_HOLD) / CR_BLOOM_FALL;
            envelope = fmaxf(0.0f, 1.0f - fallT);
            envelope *= envelope;
        }
        if (envelope < 0.003f) continue;

        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;
            float srcDist = fabsf(relOut) * halfWidth;
            float spatial = (srcDist <= CR_BLOOM_RADIUS) ? 1.0f
                : expf(-(srcDist - CR_BLOOM_RADIUS) / CR_BLOOM_EDGE_SOFTNESS);

            float gapFade = 1.0f;
            if (crShakeLevel > 0.01f) {
                float gapSize = 0.3f * crShakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            float br = envelope * spatial * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR, pixG, pixB;
            crShakeColor(c, relOut, pixR, pixG, pixB);
            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void crRenderCrawl(const CrCreature &c, float *bufR, float *bufG, float *bufB) {
    float center = c.pos;
    float halfWidth = CR_CRAWL_RADIUS + CR_CRAWL_TAIL_DECAY;
    float totalSpan = halfWidth * (1.0f + crShakeLevel * CR_SHAKE_SCATTER_RADIUS);
    int lo = (int)(center - totalSpan - 1); if (lo < 0) lo = 0;
    int hi = (int)(center + totalSpan + 1); if (hi > CR_VIRTUAL_LEDS - 1) hi = CR_VIRTUAL_LEDS - 1;

    float brightMult = 1.0f + crShakeLevel * CR_SHAKE_BRIGHT_BOOST;

    for (int a = 0; a < CR_MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float t = fminf(c.anims[a].age / CR_CRAWL_PULSE_LIFETIME, 1.0f);
        float radius = t * CR_CRAWL_RADIUS;
        float fade = 1.0f;
        if (c.anims[a].age > CR_CRAWL_PULSE_LIFETIME) {
            float fadeT = (c.anims[a].age - CR_CRAWL_PULSE_LIFETIME) / CR_CRAWL_PULSE_FADE;
            fade = fmaxf(0.0f, 1.0f - fadeT);
            fade *= fade;
        }
        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;
            float srcDist = fabsf(relOut) * halfWidth;
            if (srcDist > radius) continue;
            float behind = radius - srcDist;
            float br = expf(-behind / CR_CRAWL_TAIL_DECAY);
            if (behind < 1.0f) br *= behind;

            float gapFade = 1.0f;
            if (crShakeLevel > 0.01f) {
                float gapSize = 0.3f * crShakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            br *= fade * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR, pixG, pixB;
            crShakeColor(c, relOut, pixR, pixG, pixB);
            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void renderCreatures(float dt) {
    crProcessInteraction(dt);
    int scrollPixels = (int)roundf(crScrollOffset);

    for (int p = 0; p < CR_NUM_PAIRS; p++) {
        CrPair &ps = crPairs[p];
        memset(ps.bufR, 0, sizeof(ps.bufR));
        memset(ps.bufG, 0, sizeof(ps.bufG));
        memset(ps.bufB, 0, sizeof(ps.bufB));

        for (int c = 0; c < CR_MAX_CREATURES; c++) {
            if (!crLifecycleFrozen) crUpdateCreature(ps.creatures[c], dt);
            if (!ps.creatures[c].alive) continue;
            if (ps.creatures[c].kind == CR_KIND_BLOOM)
                crRenderBloom(ps.creatures[c], ps.bufR, ps.bufG, ps.bufB);
            else
                crRenderCrawl(ps.creatures[c], ps.bufR, ps.bufG, ps.bufB);
        }
        for (int c = 0; c < CR_MAX_CREATURES; c++)
            if (!ps.creatures[c].alive) crInitCreature(ps.creatures[c]);

        uint8_t sA = p * 2, sB = p * 2 + 1;
        for (int i = 0; i < LEDS_PER_STRIP; i++) {
            int src = CR_SCROLL_MARGIN + i + scrollPixels;
            if (src < 0) src = 0;
            if (src >= CR_VIRTUAL_LEDS) src = CR_VIRTUAL_LEDS - 1;
            float vR = clampf(ps.bufR[src], 0.0f, 1.0f);
            float vG = clampf(ps.bufG[src], 0.0f, 1.0f);
            float vB = clampf(ps.bufB[src], 0.0f, 1.0f);
            // Original used vv (gamma 2.0); fastGamma24 matches the rest of
            // this build's pipeline. Output 0–255 range for setPixelDither.
            float oR = fastGamma24(vR) * 255.0f;
            float oG = fastGamma24(vG) * 255.0f;
            float oB = fastGamma24(vB) * 255.0f;
            setPixelDither(sA, i, oR, oG, oB);
            setPixelDither(sB, i, oR, oG, oB);
        }
    }
}

// ── Light Through (stormy sunset, slot 10) ──────────────────────
// Storm-ceiling base field (slate↔indigo) with warm bloom-patches that swell
// open then seal, like sun breaking through cloud. 6 strips are radial spokes
// from center: r = index/(LEDS-1) (0=trunk, 1=edge), spoke s at angle s×60°.
// Spawn/churn/growth all scale with fxDt (speed knob folds in upstream).
// Routes through setPixelDither like every other effect.
#define LT_MAX_PATCHES   6
#define LT_SPAWN_PERIOD  2.2f   // mean seconds between patches (Poisson)
#define LT_CHURN_RATE    0.12f

// Palette (0–255 linear).
#define LT_SLATE_R   50.0f
#define LT_SLATE_G   55.0f
#define LT_SLATE_B   80.0f
#define LT_INDIGO_R  35.0f
#define LT_INDIGO_G  35.0f
#define LT_INDIGO_B  60.0f
#define LT_AMBER_R  240.0f
#define LT_AMBER_G  180.0f
#define LT_AMBER_B   90.0f
#define LT_PEACH_R  220.0f
#define LT_PEACH_G  160.0f
#define LT_PEACH_B  110.0f
#define LT_SALMON_R 200.0f
#define LT_SALMON_G 140.0f
#define LT_SALMON_B 100.0f

struct LtPatch {
    float rc;        // radial center 0.15–0.85
    float cosTc;     // precomputed cos/sin of angular center
    float sinTc;
    float maxR;      // 0.25–0.7
    float life;      // 1.5–3.5 s
    float age;
    bool  active;
};
static LtPatch ltPatches[LT_MAX_PATCHES];
static float ltTime = 0.0f;
static float ltSpawnTimer = 0.0f;
static float ltSpokeCos[NUM_STRIPS];
static float ltSpokeSin[NUM_STRIPS];

// Layered-sine pseudo-noise in 0..1 (cheap, no Perlin).
static inline float ltNoise(float x) {
    return 0.5f + 0.5f * fastSin(x * 2.3f + fastSin(x * 1.7f + ltTime * 0.08f));
}

static void resetLightThrough() {
    ltTime = 0.0f;
    ltSpawnTimer = 0.0f;
    for (int i = 0; i < LT_MAX_PATCHES; i++) ltPatches[i].active = false;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float ang = (float)s * (60.0f * (float)M_PI / 180.0f);
        ltSpokeCos[s] = cosf(ang);
        ltSpokeSin[s] = sinf(ang);
    }
}

static void ltSpawnPatch() {
    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (ltPatches[i].active) continue;
        LtPatch &p = ltPatches[i];
        p.rc = 0.15f + randFloat() * 0.70f;        // [0.15, 0.85]
        float tc = randFloat() * 2.0f * (float)M_PI;
        p.cosTc = cosf(tc);
        p.sinTc = sinf(tc);
        p.maxR = 0.25f + randFloat() * 0.45f;       // [0.25, 0.7]
        p.life = 1.5f + randFloat() * 2.0f;         // [1.5, 3.5]
        p.age = 0.0f;
        p.active = true;
        return;
    }
}

static void renderLightThrough(float dt) {
    ltTime += dt;

    // Poisson spawn: probability dt/period per frame.
    ltSpawnTimer += dt;
    float spawnProb = dt / LT_SPAWN_PERIOD;
    if (randFloat() < spawnProb) ltSpawnPatch();

    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (!ltPatches[i].active) continue;
        ltPatches[i].age += dt;
        if (ltPatches[i].age >= ltPatches[i].life) ltPatches[i].active = false;
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float sc = ltSpokeCos[s];
        float ss = ltSpokeSin[s];
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r = (float)i / (float)(LEDS_PER_STRIP - 1);

            // Layer A — storm ceiling.
            float density = ltNoise(r * 3.0f + 0.7f * (float)s + ltTime * LT_CHURN_RATE);
            float centerLift = lerpf(1.0f, 0.85f, r);
            float outR = lerpf(LT_SLATE_R, LT_INDIGO_R, density) * centerLift;
            float outG = lerpf(LT_SLATE_G, LT_INDIGO_G, density) * centerLift;
            float outB = lerpf(LT_SLATE_B, LT_INDIGO_B, density) * centerLift;

            // LED planar position on the spoke.
            float px = r * sc;
            float py = r * ss;

            // Layer B — warm bloom-patches (sum where they overlap).
            for (int pi = 0; pi < LT_MAX_PATCHES; pi++) {
                if (!ltPatches[pi].active) continue;
                LtPatch &p = ltPatches[pi];
                float tau = p.age / p.life;
                float radius = p.maxR * fastSin((float)M_PI * tau);
                if (radius < 0.001f) continue;

                float dx = px - p.rc * p.cosTc;
                float dy = py - p.rc * p.sinTc;
                float d = sqrtf(dx * dx + dy * dy);

                float g = d / (0.6f * radius);
                float a = expf(-(g * g));
                if (a < 0.004f) continue;

                // Color: amber→peach→salmon by d/radius.
                float dr = clampf(d / radius, 0.0f, 1.0f);
                float pcR, pcG, pcB;
                if (dr < 0.5f) {
                    float t = dr / 0.5f;
                    pcR = lerpf(LT_AMBER_R, LT_PEACH_R, t);
                    pcG = lerpf(LT_AMBER_G, LT_PEACH_G, t);
                    pcB = lerpf(LT_AMBER_B, LT_PEACH_B, t);
                } else {
                    float t = (dr - 0.5f) / 0.5f;
                    pcR = lerpf(LT_PEACH_R, LT_SALMON_R, t);
                    pcG = lerpf(LT_PEACH_G, LT_SALMON_G, t);
                    pcB = lerpf(LT_PEACH_B, LT_SALMON_B, t);
                }

                outR = outR * (1.0f - a) + pcR * a + 0.15f * a * 255.0f;
                outG = outG * (1.0f - a) + pcG * a + 0.15f * a * 255.0f;
                outB = outB * (1.0f - a) + pcB * a;
            }

            setPixelDither(s, i,
                clampf(outR, 0.0f, 255.0f),
                clampf(outG, 0.0f, 255.0f),
                clampf(outB, 0.0f, 255.0f));
        }
    }
}

// ── Black out all strips (reserved/off slots) ───────────────────
static void renderOff() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            setPixel(s, i, 0, 0, 0);
}

// ── Effect reset dispatch (called on effect change) ─────────────
static void resetEffect(uint8_t fx) {
    resetFxDither();
    switch (fx) {
        case FX_AMBIENT_BLOOM:
            for (uint8_t s = 0; s < NUM_STRIPS; s++) resetBloomStrip(bloom[s]);
            break;
        case FX_RAINBOW:
            resetRainbow();
            break;
        case FX_NEBULA:
            resetNebula();
            break;
        case FX_GRAVITY_PARTICLE:
            resetGravity();
            break;
        case FX_SPARKLE_SYLLABLE:
            resetSparkle();
            break;
        case FX_FIRE_MELD:
        case FX_FIRE_FLICKER:
            resetFire();
            break;
        case FX_QUIET_BLOOM:
            resetQuietBloom();
            break;
        case FX_LEAF_WIND:
            resetLeafWind();
            break;
        case FX_CREATURES:
            resetCreatures();
            break;
        case FX_LIGHT_THROUGH:
            resetLightThrough();
            break;
        default:
            break;
    }
}

// ── Runtime parameter table ──────────────────────────────────────
struct Param {
    const char *name;
    enum Type { F32 } type;
    void *ptr;
    float lo, hi;
};

static const Param PARAMS[] = {
    { "BRIGHTNESS_CAP",   Param::F32, &bloomBrightnessCap,  0.05f, 1.0f  },
    { "GLOBAL_BRIGHTNESS",Param::F32, &globalBrightness,    0.0f,  1.0f  },
    { "SPEED_SCALE",      Param::F32, &speedScale,          0.2f,  3.0f  },
    { "BUFFER_DRAIN",     Param::F32, &bloomBufferDrain,    0.5f,  20.0f },
    { "FLASH_DECAY_RATE", Param::F32, &bloomFlashDecayRate, 0.2f,  10.0f },
    { "BREATH_FLOOR",     Param::F32, &bloomBreathFloor,    0.0f,  0.5f  },
    { "DORMANCY_MIN",     Param::F32, &bloomDormancyMin,   0.0f,  60.0f },
    { "DORMANCY_MAX",     Param::F32, &bloomDormancyMax,   0.0f,  60.0f },
    { "DORMANCY_FRAC",    Param::F32, &bloomDormancyFrac,  0.0f,  1.0f  },
};
static const size_t PARAM_COUNT = sizeof(PARAMS) / sizeof(PARAMS[0]);

static void dumpParam(const Param &p) {
    Serial.printf("[PARAM] %s=%.4f\n", p.name, *(float*)p.ptr);
}

static void dumpAllParams() {
    for (size_t i = 0; i < PARAM_COUNT; i++) dumpParam(PARAMS[i]);
}

static void setParamFromLine(const char *kv) {
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
            *(float*)p.ptr = v;
            dumpParam(p);
            return;
        }
    }
    Serial.printf("[PARAM] unknown key\n");
}

static void processLine(const char *line, uint8_t len) {
    if (len >= 2 && line[0] == '!' && line[1] == 'P') {
        if (len >= 3 && line[2] == '?') { dumpAllParams(); return; }
        if (len >= 4) { setParamFromLine(line + 2); return; }
    }
}

static char serialBuf[80];
static uint8_t serialBufLen = 0;

static void parseSerialCommands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBufLen > 0) {
                serialBuf[serialBufLen] = '\0';
                processLine(serialBuf, serialBufLen);
                serialBufLen = 0;
            }
        } else if (c >= 32 && c < 127 && serialBufLen < sizeof(serialBuf) - 1) {
            serialBuf[serialBufLen++] = c;
        }
    }
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0)); strip2.ClearTo(RgbColor(0));
    strip3.ClearTo(RgbColor(0)); strip4.ClearTo(RgbColor(0)); strip5.ClearTo(RgbColor(0));
    showAll();

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        resetBloomStrip(bloom[s]);
    }

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed");
        return;
    }
    esp_now_register_recv_cb(onReceive);

    Serial.printf("Tree-of-record ready — ch=%u\n", FIXED_CHANNEL);
    Serial.printf("  Bloom: 6 strips × %u LEDs (ambient, BS-26 driven)\n", LEDS_PER_STRIP);
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    // Decode sensor packets → effect feature globals (real-time dt).
    processSensors(dt);

    // ── Map BS-26 knobs to effect parameters ────────────────────
    // BS-26 live → knobs drive the effect. No signal → serial PARAMS.
    bool bs26Live = (bs26LastMs > 0 && (now - bs26LastMs) < HEARTBEAT_TIMEOUT_MS);
    float effBrightness = globalBrightness;
    float effSpeed      = speedScale;
    if (bs26Live) {
        Bs26State s;
        memcpy(&s, (const void*)&bs26, sizeof(s));

        // tens (10-position knob) → brightness 0.01–1.0 (linear)
        uint8_t tens = s.tens > 9 ? 9 : s.tens;
        effBrightness = lerpf(0.01f, 1.0f, (float)tens / 9.0f);

        // hundreds 0–4 → speed (index 2 = 1.0 normal); log-ish 5-step ramp
        static const float HUNDREDS_SPEED[5] = { 0.3f, 0.6f, 1.0f, 1.8f, 3.0f };
        uint8_t hi = s.hundreds > 4 ? 4 : s.hundreds;
        effSpeed = HUNDREDS_SPEED[hi];

        // decmid (0–9) → hue rotation, decinner (0–9) → saturation
        uint8_t hue = s.decmid > 9 ? 9 : s.decmid;
        uint8_t sat = s.decinner > 9 ? 9 : s.decinner;
        if (hue != paletteHueIdx || sat != paletteSatIdx) {
            paletteHueIdx = hue;
            paletteSatIdx = sat;
            applyPalette(paletteHueIdx, paletteSatIdx);
        }

        // ones (0–11) → effect select
        currentEffect = s.ones >= FX_COUNT ? FX_COUNT - 1 : s.ones;

        // DC switch → dormancy on/off
        bloomDormancyFrac = s.dc ? 0.3f : 0.0f;
    }

    renderBrightness = effBrightness;

    // Reset effect state on change.
    if (currentEffect != prevEffect) {
        resetEffect(currentEffect);
        prevEffect = currentEffect;
    }

    uint32_t t0 = micros();

    // ── Render the selected effect on all 6 strips ──────────────
    float fxDt = dt * effSpeed;
    switch (currentEffect) {
        case FX_AMBIENT_BLOOM:
            for (uint8_t s = 0; s < NUM_STRIPS; s++) renderBloomStrip(s, fxDt);
            break;
        case FX_RAINBOW:
            renderRainbow(fxDt);
            break;
        case FX_NEBULA:
            renderNebula(fxDt);
            break;
        case FX_GRAVITY_PARTICLE:
            renderGravity(fxDt);
            break;
        case FX_SPARKLE_SYLLABLE:
            renderSparkle(fxDt);
            break;
        case FX_FIRE_MELD:
            renderFire(fxDt, true);
            break;
        case FX_FIRE_FLICKER:
            renderFire(fxDt, false);
            break;
        case FX_QUIET_BLOOM:
            renderQuietBloom(fxDt);
            break;
        case FX_LEAF_WIND:
            renderLeafWind(fxDt);
            break;
        case FX_CREATURES:
            renderCreatures(fxDt);
            break;
        case FX_LIGHT_THROUGH:
            renderLightThrough(fxDt);
            break;
        default:
            // Reserved off slot (11): black.
            renderOff();
            break;
    }

    uint32_t t1 = micros();
    showAll();
    uint32_t t2 = micros();

    static uint32_t renderUsAccum = 0, showUsAccum = 0;
    renderUsAccum += (t1 - t0);
    showUsAccum   += (t2 - t1);

    parseSerialCommands();

    // Status logging
    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    if (now - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (now - lastLogMs);
        float avgRenderUs = (float)renderUsAccum / frameCount;
        float avgShowUs   = (float)showUsAccum / frameCount;
        Serial.printf("  FPS=%.1f  render=%.0fus  show=%.0fus  total=%.0fus\n",
                      fps, avgRenderUs, avgShowUs, avgRenderUs + avgShowUs);
        Serial.printf("  [bs26] %s pkts=%lu fx=%u bright=%.3f speed=%.2f hue=%u sat=%u dorm=%.2f\n",
                      bs26Live ? "LIVE" : "----",
                      (unsigned long)bs26PktCount,
                      currentEffect, renderBrightness, effSpeed,
                      paletteHueIdx, paletteSatIdx, bloomDormancyFrac);
        bool aLive = accelLastMs && (now - accelLastMs) < SENSOR_TIMEOUT_MS;
        bool gLive = gyroLastMs  && (now - gyroLastMs)  < SENSOR_TIMEOUT_MS;
        bool auLive = audioLastMs && (now - audioLastMs) < SENSOR_TIMEOUT_MS;
        Serial.printf("  [sns] accel=%s(%lu) gyro=%s(%lu) audio=%s(%lu) "
                      "energy=%.3f onset=%.3f tilt=%.2f@%.0f shakeG=%.2f scrollDps=%.0f\n",
                      aLive ? "L" : "-", (unsigned long)accelPktCount,
                      gLive ? "L" : "-", (unsigned long)gyroPktCount,
                      auLive ? "L" : "-", (unsigned long)audioPktCount,
                      fxEnergy, fxOnset, fxTiltBlend, fxAngleDeg,
                      crShakeG_live, crScrollDps_live);
        frameCount = 0;
        renderUsAccum = 0;
        showUsAccum = 0;
        lastLogMs = now;
    }
}
