/*
 * BULB-FLEET — sparkle_burst on 6 WS2812B RGB strips (classic ESP32).
 *
 * Listens to the same ESP-NOW SensorPacket broadcast as the 50-LED RGBW
 * receiver. Both nodes co-exist on the same channel; broadcast lands on
 * each receiver's onReceive independently. No coordination needed.
 *
 * Effect: sparkle_burst, ported from original-duck/src/bulb_receiver.cpp.
 *   - FixedRangeRMS energy (10k–50k floor/ceiling) + EnergyDelta onsets
 *   - Onset → ignite a random subset of LEDs per strip, each strip uses
 *     its own PRNG so the same onset lights different LEDs on each.
 *   - Per-LED exponential decay (~0.93 rate, 30 Hz reference)
 *   - RGBW→RGB: drop W; fold W luminance back into R+G+B so the warm
 *     amber base doesn't go pale. SPARKLE_BRIGHTNESS_CAP raised to
 *     compensate for missing W headroom.
 *   - Per-strip hue offset (60° / 42 LUT idx steps) when tilted — the 6
 *     strips paint a hue chord across the OKLCH wheel during engaged
 *     tilt, while the warm-amber base palette stays shared.
 *   - Engage / disengage / idle rainbow / disengage stagger fadeout —
 *     all preserved from bulb_receiver verbatim, scaled to per-strip.
 *
 * Wiring (matches biolum):
 *   GPIO 4  → strip 0   GPIO 5  → strip 3
 *   GPIO 16 → strip 1   GPIO 18 → strip 4
 *   GPIO 17 → strip 2   GPIO 19 → strip 5
 *
 * Serial: 's' sparkle (only effect), 'c' recalibrate, '?' identify.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <oklch_lut.h>
#include <fast_math.h>

// ── Topology ────────────────────────────────────────────────────────
#define NUM_STRIPS       6
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP   50
#endif
#define TOTAL_LEDS       (NUM_STRIPS * LEDS_PER_STRIP)

// Per-strip OKLCH hue offsets, ~60° apart (256-step wheel).
// Used for the tilt-overlay color so the 6 strips span a chord during
// engaged tilt. Idle rainbow shares this offset for cross-strip variety.
static const uint8_t STRIP_HUE_OFFSET[NUM_STRIPS] = {
    0, 42, 85, 128, 170, 213
};

// ── 6 NeoPixelBus instances on RMT channels 0..5 ────────────────────
// Sacrificial first pixel (HEAD_OFFSET=1): physical pixel 0 on each chain
// is never written (stays black) and acts as a signal regenerator,
// absorbing shifter→first-pixel glitches. Effect code keeps using logical
// indices 0..LEDS_PER_STRIP-1; stripSetPixel maps logical i → physical i+1
// and drops writes that would exceed the strip's physical length. Net
// result: visible LEDs = LEDS_PER_STRIP - 1 (= 99 of 100).
#define HEAD_OFFSET 1
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 16);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, 19);

static inline void stripSetPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    uint16_t pi = i + HEAD_OFFSET;  // skip sacrificial head pixel
    if (pi >= LEDS_PER_STRIP) return;  // tail logical LED has nowhere to go
    switch (s) {
        case 0: strip0.SetPixelColor(pi, c); break;
        case 1: strip1.SetPixelColor(pi, c); break;
        case 2: strip2.SetPixelColor(pi, c); break;
        case 3: strip3.SetPixelColor(pi, c); break;
        case 4: strip4.SetPixelColor(pi, c); break;
        case 5: strip5.SetPixelColor(pi, c); break;
    }
}

static inline void stripsShow() {
    strip0.Show(); strip1.Show(); strip2.Show();
    strip3.Show(); strip4.Show(); strip5.Show();
}

static inline void stripsBegin() {
    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
}

// ── Tunables ────────────────────────────────────────────────────────
#define SPARKLE_BRIGHTNESS_CAP 0.20f   // RGB-only — bumped 4× from RGBW (0.05)
#define SPARKLE_DEADBAND       0.08f
#define DEADZONE_DEG           10.0f
#define MAX_ANGLE_DEG          180.0f
#define BLEND_RANGE_DEG        40.0f
#define SENSOR_HZ              25.0f

// Fire (RGB-only — BRIGHTNESS_CAP bumped from 0.10 RGBW since W is gone)
#define BRIGHTNESS_CAP         0.60f
#define FIRE_FLICKER_SCALE     3.0f
#define FIRE_DEADBAND          0.08f
#define FIRE_DROPOUT_DEPTH     0.85f

// Quiet bloom (RGB-only — bumped from 0.25 RGBW)
#define BLOOM_BRIGHTNESS_CAP   0.40f
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
#define BLOOM_FLASH_DECAY_LO    0.96f
#define BLOOM_FLASH_DECAY_HI    0.985f
#define BLOOM_RECOVERY_RAMP     0.033f
#define BLOOM_RECOVERY_SPREAD   0.70f

// Bloom palette: blue/cyan; W luminance folded into all channels for RGB.
#define BLOOM_HUE_A_G          20.0f
#define BLOOM_HUE_A_B         100.0f
#define BLOOM_HUE_B_G          70.0f
#define BLOOM_HUE_B_B         110.0f
#define BLOOM_FLASH_G         150.0f
#define BLOOM_FLASH_B         170.0f
#define BLOOM_W_ONSET           0.5f

// Sparkle simple gate
#define SIMPLE_SPARKLE_FLOOR_RAW 47234.0f

// Audio
#define RMS_FLOOR              2000.0f
#define RMS_CEILING            15000.0f
#define RMS_PEAK_DECAY         0.9999f
#define DELTA_PEAK_DECAY       0.998f

// Timing
#define TIMEOUT_MS             500
#define IDLE_GRACE_MS          10000
#define IDLE_PERIOD_S          30.0f
#define IDLE_PEAK              0.18f   // bumped slightly — no W headroom

// Engage/disengage
#define ENGAGE_DOT_THRESH      -0.6f
#define DISENGAGE_DOT_THRESH   +0.6f
#define ENGAGE_HOLD_MS         300
#define DISENGAGE_HOLD_MS      10000
#define FADEOUT_WINDOW_MS      4000
#define FADEOUT_FLOOR          0.05f

// Channel discovery
#define CHANNEL_FALLBACK       1
#define CHANNEL_RESCAN_MS      (5UL * 60UL * 1000UL)
static const char* WIFI_SSID_TARGET = "cuteplant";
static uint8_t  currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs     = 0;

// ── Packet ──────────────────────────────────────────────────────────
struct __attribute__((packed)) SensorPacket {
    int16_t  ax, ay, az;
    int16_t  gx, gy, gz;
    uint16_t rawRms;
    uint8_t  micEnabled;
};

static volatile uint32_t lastPacketMs = 0;
static volatile uint32_t pktCount = 0;
static SensorPacket latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

static volatile uint32_t maxGapMs = 0;
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    pktCount++;
    uint32_t nowMs = millis();
    if (lastPacketMs != 0) {
        uint32_t gap = nowMs - lastPacketMs;
        if (gap > maxGapMs) maxGapMs = gap;
    }
    if (len == sizeof(SensorPacket)) {
        memcpy((void*)&latestPacket, data, sizeof(SensorPacket));
        lastPacketMs = nowMs;
    }
}

// ── Channel scan ────────────────────────────────────────────────────
static uint8_t scanForSsidChannel(const char* ssid) {
    int n = WiFi.scanNetworks(false, false, true, 120);
    uint8_t found = 0;
    int8_t bestRssi = -127;
    for (int i = 0; i < n; i++) {
        if (WiFi.SSID(i) == ssid) {
            int8_t rssi = WiFi.RSSI(i);
            if (rssi > bestRssi) { bestRssi = rssi; found = WiFi.channel(i); }
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

// ── Helpers ─────────────────────────────────────────────────────────
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

// Per-strip xorshift32 PRNGs — same audio onset triggers different LED
// ignition patterns on each strip.
static uint32_t prngState[NUM_STRIPS];
static inline uint32_t xorshift32(uint8_t s) {
    uint32_t x = prngState[s];
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    prngState[s] = x;
    return x;
}
static inline float randFloat(uint8_t s) {
    return (float)(xorshift32(s) & 0xFFFFFF) / 16777216.0f;
}

// ── Calibration ─────────────────────────────────────────────────────
static float restAx = 0, restAy = 0, restAz = 1.0f;
static bool  calibrated = false;
static float calSumAx = 0, calSumAy = 0, calSumAz = 0;
static uint32_t calSamples = 0;
static uint32_t calStartMs = 0;
#define CAL_DURATION_MS 2000

static void startCalibration() {
    calibrated = false;
    calSumAx = calSumAy = calSumAz = 0;
    calSamples = 0;
    calStartMs = millis();
    Serial.println("Calibrating — keep duck still for 2 seconds...");
}

static void updateCalibration(float ax, float ay, float az) {
    if (calStartMs == 0) startCalibration();
    if (!calibrated) {
        calSumAx += ax; calSumAy += ay; calSumAz += az;
        calSamples++;
        if (millis() - calStartMs >= CAL_DURATION_MS && calSamples > 0) {
            restAx = calSumAx / calSamples;
            restAy = calSumAy / calSamples;
            restAz = calSumAz / calSamples;
            vecNormalize(restAx, restAy, restAz);
            calibrated = true;
            Serial.printf("Calibrated! Rest: (%.4f, %.4f, %.4f)\n",
                          restAx, restAy, restAz);
        }
    }
}

// ── Audio: FixedRangeRMS + EnergyDelta ──────────────────────────────
static float frrCeiling = RMS_CEILING;
static float energy = 0.0f;
static float prevRms = 0.0f;
static float deltaPeak = 1e-6f;
static float onset = 0.0f;

static float computeEnergy(uint16_t rawRms) {
    float rms = (float)rawRms;
    frrCeiling = fmaxf(RMS_CEILING, frrCeiling * RMS_PEAK_DECAY);
    if (rms > frrCeiling) frrCeiling = rms;
    if (rms < RMS_FLOOR) return 0.0f;
    float db = 20.0f * log10f(rms / RMS_FLOOR);
    float dbRange = 20.0f * log10f(frrCeiling / RMS_FLOOR);
    return clampf(db / dbRange, 0.0f, 1.0f);
}

static float computeOnset(uint16_t rawRms) {
    float rms = (float)rawRms;
    float delta = fabsf(rms - prevRms);
    prevRms = rms;
    deltaPeak = fmaxf(delta, deltaPeak * DELTA_PEAK_DECAY);
    return (deltaPeak > 1e-6f) ? (delta / deltaPeak) : 0.0f;
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

static float computeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx*fx + fy*fy + fz*fz) / 131.0f;
}

static float computeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx*fx + fy*fy + fz*fz);
    return fabsf(mag - 16384.0f) / 16384.0f;
}

// ── Sparkle state — per-strip, per-LED ──────────────────────────────
static float    sparkle    [NUM_STRIPS][LEDS_PER_STRIP];
static float    decayRates [NUM_STRIPS][LEDS_PER_STRIP];
static float    envelope = 0.0f;
static float    cooldownRemaining = 0.0f;

// ── Fire render state — per-strip ───────────────────────────────────
// Per-strip floats so each strip has its own time-evolution but shares
// the audio-driven envelope/color machinery.
static float fireTime[NUM_STRIPS];
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

static void resetFireState() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Stagger phase per strip so they don't flicker in lockstep.
        fireTime[s] = (float)s * 1.37f;
    }
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

// ── Quiet bloom state — per-strip, per-LED ──────────────────────────
static float bloomBreathPhase [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomBreathPeriod[NUM_STRIPS][LEDS_PER_STRIP];
static float bloomBreathPeak  [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomHueT        [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomFlashGlow   [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomFlashDecay  [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomColonyEnergy = 1.0f;
static uint32_t bloomLastMotionMs = 0;
static float bloomMotionRate = 0.0f;
static float motionEMA = 0.0f;
static float drainEnvelope = 0.0f;
static float bloomHitIntensity = 0.0f;
static uint32_t bloomPrevPktCount = 0;

static void resetBloomState() {
    bloomColonyEnergy = 1.0f;
    bloomLastMotionMs = 0;
    bloomMotionRate = 0.0f;
    motionEMA = 0.0f;
    drainEnvelope = 0.0f;
    bloomHitIntensity = 0.0f;
    bloomPrevPktCount = 0;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            bloomBreathPhase[s][i]  = randFloat(s);
            bloomBreathPeriod[s][i] = BLOOM_BREATH_MIN_PERIOD
                + randFloat(s) * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
            bloomBreathPeak[s][i]   = BLOOM_BREATH_MIN_PEAK
                + randFloat(s) * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
            bloomHueT[s][i]         = randFloat(s);
            bloomFlashGlow[s][i]    = 0.0f;
            bloomFlashDecay[s][i]   = BLOOM_FLASH_DECAY_LO
                + randFloat(s) * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
        }
    }
}

// ── Effect switcher ─────────────────────────────────────────────────
enum Effect {
    EFF_FIRE_MELD = 0,
    EFF_FIRE_FLICKER,
    EFF_QUIET_BLOOM,
    EFF_SPARKLE_BURST,
    EFF_SPARKLE_SIMPLE,
    EFF_SPARKLE_PROGRESSION,
    EFF_COUNT
};
static Effect currentEffect = EFF_FIRE_MELD;

static const char* effectName(Effect e) {
    switch (e) {
        case EFF_FIRE_MELD:      return "fire_meld";
        case EFF_FIRE_FLICKER:   return "fire_flicker";
        case EFF_QUIET_BLOOM:    return "quiet_bloom";
        case EFF_SPARKLE_BURST:  return "sparkle_burst";
        case EFF_SPARKLE_SIMPLE: return "sparkle_simple";
        case EFF_SPARKLE_PROGRESSION: return "sparkle_progression";
        default: return "?";
    }
}

static void resetSparkleState() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            sparkle[s][i] = 0.0f;
            decayRates[s][i] = 0.92f + randFloat(s) * 0.05f;
        }
    }
    envelope = 0.0f;
    cooldownRemaining = 0.0f;
}

// ── Engage/disengage state machine ──────────────────────────────────
static bool engaged = true;  // bulb-fleet: always engaged, no flip-to-trigger
static uint32_t engageCandidateMs = 0;
static uint8_t engageHueOffset = 0;
static bool prevEngaged = false;
static uint8_t currentIdleHueIdx = 0;

// Brightness envelope (per-LED stagger) — shared one envelope across all
// strips since they all share the same engage/disengage fate. fadeStartMs
// per-LED-per-strip is independent so each strip's fadeout looks distinct.
static uint16_t fadeStartMs[NUM_STRIPS][LEDS_PER_STRIP];
static float    fadeoutMult[NUM_STRIPS][LEDS_PER_STRIP];
static float    envDirStartVal[NUM_STRIPS][LEDS_PER_STRIP];
static uint32_t envDirStartMs = 0;
static bool     envDirIsDown = false;

// ── Setup ───────────────────────────────────────────────────────────
void setup() {
    Serial.begin(460800);
    delay(300);

    stripsBegin();

    // Seed per-strip PRNGs
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        prngState[s] = esp_random();
        if (prngState[s] == 0) prngState[s] = 1 + s;
    }

    resetSparkleState();
    resetFireState();
    resetBloomState();

    // Per-LED, per-strip fade stagger — Fisher-Yates shuffle of fade-start
    // times in [0, span] so on disengage one LED dims every ~120ms.
    {
        const uint32_t span = DISENGAGE_HOLD_MS - FADEOUT_WINDOW_MS;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            uint16_t order[LEDS_PER_STRIP];
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) order[i] = i;
            for (uint16_t i = LEDS_PER_STRIP - 1; i > 0; i--) {
                uint16_t j = (uint16_t)(randFloat(s) * (float)(i + 1));
                if (j > i) j = i;
                uint16_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
            for (uint16_t k = 0; k < LEDS_PER_STRIP; k++) {
                fadeStartMs[s][order[k]] =
                    (uint16_t)((uint32_t)k * span / (LEDS_PER_STRIP - 1));
            }
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                fadeoutMult[s][i] = FADEOUT_FLOOR;
                envDirStartVal[s][i] = FADEOUT_FLOOR;
            }
        }
        envDirStartMs = millis();
        envDirIsDown = false;
    }

    // Boot flash — brief green sweep across strips so we know each output
    // is wired and addressable.
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            stripSetPixel(s, i, 0, 30, 0);
    stripsShow();
    delay(200);
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            stripSetPixel(s, i, 0, 0, 0);
    stripsShow();

    // ── Wi-Fi: STA, no AP join, channel from cuteplant ────────────
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    uint8_t ch = scanForSsidChannel(WIFI_SSID_TARGET);
    if (ch) Serial.printf("Found '%s' on ch=%u\n", WIFI_SSID_TARGET, ch);
    else    Serial.printf("'%s' not visible — fallback ch=%u\n",
                          WIFI_SSID_TARGET, CHANNEL_FALLBACK);
    applyChannel(ch ? ch : CHANNEL_FALLBACK);
    lastScanMs = millis();

    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.printf("[BOOT] role=bulb-fleet MAC=%s fw=bulb_fleet ch=%u\n",
                  WiFi.macAddress().c_str(), currentChannel);
    Serial.println("Commands: 's' cycle effect, 'c' recal, '?' identify");
    Serial.printf("[effect] %d %s\n", (int)currentEffect, effectName(currentEffect));
}

// ── Sparkle render: writes all 6 strips ─────────────────────────────
static void renderSparkleAll(float dt, float angleDeg, float tiltBlend) {
    bool isSilent = energy < 0.001f;

    // Asymmetric envelope: ~30ms attack, ~400ms decay
    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > envelope) envelope += attackAlpha * (energy - envelope);
    else                   envelope += decayAlpha  * (energy - envelope);

    cooldownRemaining = fmaxf(0.0f, cooldownRemaining - dt);

    float onsetThreshold = fmaxf(0.15f, 0.4f - envelope * 0.3f);
    bool ignite = (onset > onsetThreshold && cooldownRemaining <= 0.0f && !isSilent);

    if (ignite) {
        cooldownRemaining = fmaxf(0.050f, 0.150f - envelope * 0.10f);
        float onsetStrength = clampf(onset, 0.0f, 1.0f);
        int nIgnite = (int)(LEDS_PER_STRIP * (0.3f + 0.2f * onsetStrength));
        float sparkVal = 0.7f + 0.3f * onsetStrength;

        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            uint8_t indices[LEDS_PER_STRIP];
            for (uint8_t i = 0; i < LEDS_PER_STRIP; i++) indices[i] = i;
            for (int i = 0; i < nIgnite; i++) {
                int j = i + (int)(xorshift32(s) % (LEDS_PER_STRIP - i));
                uint8_t tmp = indices[i];
                indices[i] = indices[j]; indices[j] = tmp;
            }
            for (int i = 0; i < nIgnite; i++) {
                sparkle[s][indices[i]] = sparkVal;
                decayRates[s][indices[i]] = 0.92f + randFloat(s) * 0.05f;
            }
        }
    }

    // Per-LED decay (frame-rate independent)
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            sparkle[s][i] *= fastDecay(decayRates[s][i], dt * 30.0f);

    // Subtle shimmer between onsets
    if (!isSilent) {
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                float jitter = (randFloat(s) - 0.5f) * 0.02f;
                float nv = sparkle[s][i] + jitter;
                if (nv < 0.0f) nv = 0.0f;
                if (nv > sparkle[s][i] && jitter > 0) nv = sparkle[s][i];
                sparkle[s][i] = nv;
            }
    }

    float base = fminf(envelope, 0.2f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Per-strip tilt overlay color: rotate hue by strip offset
        float tiltR = 0, tiltG = 0, tiltB = 0;
        if (tiltBlend > 0.0f) {
            float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
            if (hueFrac < 0.0f) hueFrac = 0.0f;
            if (hueFrac > 1.0f) hueFrac = 1.0f;
            uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255)
                                         + engageHueOffset
                                         + STRIP_HUE_OFFSET[s]) & 0xFF);
            tiltR = (float)oklchVarL[hueIdx][0];
            tiltG = (float)oklchVarL[hueIdx][1];
            tiltB = (float)oklchVarL[hueIdx][2];
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float sp = sparkle[s][i];
            float bright = base + sp * (1.0f - base);
            if (bright < SPARKLE_DEADBAND) bright = 0.0f;

            // Warm-amber base, sparkle pushes toward warm white. Same per
            // strip — strips differ via tilt overlay + ignition pattern.
            float colR = 255.0f;
            float colG = 180.0f + (240.0f - 180.0f) * sp;
            float colB =  80.0f + (200.0f -  80.0f) * sp;

            if (tiltBlend > 0.0f) {
                colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
                colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
                colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
            }

            // RGBW→RGB W-fold: original W carried (1-tiltBlend)*255 worth
            // of pure-white luminance. With no W channel, fold half of it
            // into each RGB to keep amber from going pale + dim. The other
            // half is absorbed by the bumped SPARKLE_BRIGHTNESS_CAP.
            float wFold = 127.0f * (1.0f - tiltBlend);
            colR = fminf(255.0f, colR + wFold);
            colG = fminf(255.0f, colG + wFold);
            colB = fminf(255.0f, colB + wFold);

            float linBright = fastGamma24(bright) * SPARKLE_BRIGHTNESS_CAP;
            float fR = colR * linBright;
            float fG = colG * linBright;
            float fB = colB * linBright;

            // Envelope multiplier
            float m = fadeoutMult[s][i];
            fR *= m; fG *= m; fB *= m;

            uint8_t r = (uint8_t)clampf(fR, 0.0f, 255.0f);
            uint8_t g = (uint8_t)clampf(fG, 0.0f, 255.0f);
            uint8_t b = (uint8_t)clampf(fB, 0.0f, 255.0f);

            stripSetPixel(s, i, r, g, b);
        }
    }
}

// ── Sparkle simple (dBFS-gated, no onset detector) ──────────────────
static void renderSparkleSimple(float dt, float angleDeg, float tiltBlend) {
    float rawRms = (float)latestPacket.rawRms;
    bool gate = rawRms > SIMPLE_SPARKLE_FLOOR_RAW;
    bool isSilent = !gate;

    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    float gateVal = gate ? 1.0f : 0.0f;
    if (gateVal > envelope) envelope += attackAlpha * (gateVal - envelope);
    else                    envelope += decayAlpha  * (gateVal - envelope);

    cooldownRemaining = fmaxf(0.0f, cooldownRemaining - dt);

    if (gate && cooldownRemaining <= 0.0f) {
        cooldownRemaining = 0.100f;
        float db = 20.0f * log10f(rawRms / SIMPLE_SPARKLE_FLOOR_RAW);
        float intensity = clampf(db / 12.0f, 0.0f, 1.0f);
        int nIgnite = (int)(LEDS_PER_STRIP * (0.3f + 0.2f * intensity));
        float sparkVal = 0.7f + 0.3f * intensity;

        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            uint8_t indices[LEDS_PER_STRIP];
            for (uint8_t i = 0; i < LEDS_PER_STRIP; i++) indices[i] = i;
            for (int i = 0; i < nIgnite; i++) {
                int j = i + (int)(xorshift32(s) % (LEDS_PER_STRIP - i));
                uint8_t tmp = indices[i];
                indices[i] = indices[j]; indices[j] = tmp;
            }
            for (int i = 0; i < nIgnite; i++) {
                sparkle[s][indices[i]] = sparkVal;
                decayRates[s][indices[i]] = 0.92f + randFloat(s) * 0.05f;
            }
        }
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            sparkle[s][i] *= fastDecay(decayRates[s][i], dt * 30.0f);

    if (!isSilent) {
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                float jitter = (randFloat(s) - 0.5f) * 0.02f;
                float nv = sparkle[s][i] + jitter;
                if (nv < 0.0f) nv = 0.0f;
                if (nv > sparkle[s][i] && jitter > 0) nv = sparkle[s][i];
                sparkle[s][i] = nv;
            }
    }

    float base = fminf(envelope, 0.2f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float tiltR = 0, tiltG = 0, tiltB = 0;
        if (tiltBlend > 0.0f) {
            float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
            if (hueFrac < 0.0f) hueFrac = 0.0f;
            if (hueFrac > 1.0f) hueFrac = 1.0f;
            uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255)
                                         + engageHueOffset
                                         + STRIP_HUE_OFFSET[s]) & 0xFF);
            tiltR = (float)oklchVarL[hueIdx][0];
            tiltG = (float)oklchVarL[hueIdx][1];
            tiltB = (float)oklchVarL[hueIdx][2];
        }
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float sp = sparkle[s][i];
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

            float linBright = fastGamma24(bright) * SPARKLE_BRIGHTNESS_CAP;
            float fR = colR * linBright;
            float fG = colG * linBright;
            float fB = colB * linBright;
            float m = fadeoutMult[s][i];
            fR *= m; fG *= m; fB *= m;
            stripSetPixel(s, i,
                (uint8_t)clampf(fR, 0, 255),
                (uint8_t)clampf(fG, 0, 255),
                (uint8_t)clampf(fB, 0, 255));
        }
    }
}

// ── Sparkle progression: each onset advances index by 1, sparkles that
//    LED on every strip. Wraps at LEDS_PER_STRIP.
static uint16_t progIndex = 0;
static float progCooldown = 0.0f;
static void renderSparkleProgression(float dt, uint32_t now) {
    bool isSilent = energy < 0.001f;
    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > envelope) envelope += attackAlpha * (energy - envelope);
    else                   envelope += decayAlpha  * (energy - envelope);

    progCooldown = fmaxf(0.0f, progCooldown - dt);

    float onsetThreshold = fmaxf(0.15f, 0.4f - envelope * 0.3f);
    if (onset > onsetThreshold && progCooldown <= 0.0f && !isSilent) {
        progCooldown = 0.150f;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            sparkle[s][progIndex] = 1.0f;
            decayRates[s][progIndex] = 0.92f + randFloat(s) * 0.05f;
        }
        progIndex = (progIndex + 1) % LEDS_PER_STRIP;
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            sparkle[s][i] *= fastDecay(decayRates[s][i], dt * 30.0f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        uint8_t hueIdx = (uint8_t)((engageHueOffset + STRIP_HUE_OFFSET[s]) & 0xFF);
        float colR = (float)oklchVarL[hueIdx][0];
        float colG = (float)oklchVarL[hueIdx][1];
        float colB = (float)oklchVarL[hueIdx][2];
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float sp = sparkle[s][i];
            if (sp < SPARKLE_DEADBAND) {
                stripSetPixel(s, i, 0, 0, 0);
                continue;
            }
            float linBright = fastGamma24(sp) * SPARKLE_BRIGHTNESS_CAP;
            float m = fadeoutMult[s][i];
            stripSetPixel(s, i,
                (uint8_t)clampf(colR * linBright * m, 0, 255),
                (uint8_t)clampf(colG * linBright * m, 0, 255),
                (uint8_t)clampf(colB * linBright * m, 0, 255));
        }
    }
}

// ── Fire render (RGB, per-strip phase) ──────────────────────────────
// `withDropout=false` → fire_meld; `withDropout=true` → fire_flicker.
// W-fold: the original used pure-W floor at low brightness. RGB version
// folds avgRGB*(1-rgbBlend) into all three channels to keep the warm
// hue at low brightness (instead of going dim+colored).
static void renderFire(float dt, bool withDropout, float tiltBlend) {
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
    float colorTarget = (isPercussiveOnly || isSilent) ? 0.0f : energy;
    if (colorTarget > fireColorEnergy)
        fireColorEnergy += colorAttack * (colorTarget - fireColorEnergy);
    else
        fireColorEnergy += colorDecay * (colorTarget - fireColorEnergy);

    float ce = fireColorEnergy;
    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;
    const float amberR = 255.0f, amberG = 140.0f, amberB = 30.0f;
    const float redR   = 200.0f, redG   =  20.0f, redB   =  0.0f;
    const float whiteR = 180.0f, whiteG = 170.0f, whiteB = 160.0f;
    float baseColR, baseColG, baseColB;
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
    float scl = FIRE_FLICKER_SCALE;

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Per-strip time keeps each strip's flicker independent.
        fireTime[s] = fmodf(fireTime[s] + dt, 6283.1853f);
        float t = fireTime[s];
        // Per-strip spatial offset so noise patterns don't align.
        float sOff = (float)s * 17.0f;

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float fi = (float)i + sOff;
            float noise = fastSin(fi * 7.3f + t * 2.5f) *
                          fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;
            float noiseAmp = fmaxf(0.15f * scl, 0.10f * scl / fmaxf(base, 0.1f));
            float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                         + fireFlickerIntensity * (noise - 0.5f) * 0.25f * scl;

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

            // Direct RGB output — no tilt-gated W-fold (tiltBlend pinned to 1).
            float fR = oR;
            float fG = oG;
            float fB = oB;

            float m = fadeoutMult[s][i];
            fR *= m; fG *= m; fB *= m;
            // Swap R↔G for fire: hypothesis test — fire was rendering green
            // when sparkle (same feature) renders amber. Try reversed order.
            stripSetPixel(s, i,
                (uint8_t)clampf(fG, 0, 255),
                (uint8_t)clampf(fR, 0, 255),
                (uint8_t)clampf(fB, 0, 255));
        }
    }
}

// ── Bloom motion processing (call once per new packet) ──────────────
// Audio-disturbance bloom: noises with rawRms > BLOOM_AUDIO_THRESH disturb
// the colony. Above-threshold magnitude maps to hit intensity + drain.
#define BLOOM_AUDIO_THRESH   30000.0f
#define BLOOM_AUDIO_RANGE    35535.0f   // 65535 - 30000
static void bloomProcessMotion(uint32_t now) {
    float r = (float)latestPacket.rawRms;
    if (r <= BLOOM_AUDIO_THRESH) return;
    float over = (r - BLOOM_AUDIO_THRESH) / BLOOM_AUDIO_RANGE;  // 0..1
    if (over > 1.0f) over = 1.0f;
    bloomLastMotionMs = now;
    if (over > bloomHitIntensity) bloomHitIntensity = over;
    float newDrain = (0.6f + 0.4f * over) * (1.0f - DRAIN_ENVELOPE_DECAY) * 40.0f;
    if (newDrain > drainEnvelope) drainEnvelope = newDrain;
}

// ── Quiet bloom render ──────────────────────────────────────────────
static void renderQuietBloom(float dt, uint32_t now) {
    bool draining = drainEnvelope > 0.001f;
    if (!draining) bloomHitIntensity = 0.0f;

    if (now - bloomLastMotionMs > MOTION_SETTLE_MS)
        bloomColonyEnergy = fminf(1.0f, bloomColonyEnergy + BLOOM_RECOVERY_RAMP * dt);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            if (!draining) {
                bloomFlashGlow[s][i] *= fastDecay(bloomFlashDecay[s][i], dt * 30.0f);
                if (bloomFlashGlow[s][i] < 0.005f) bloomFlashGlow[s][i] = 0.0f;
            }
            bloomBreathPhase[s][i] += dt / bloomBreathPeriod[s][i];
            if (bloomBreathPhase[s][i] >= 1.0f) bloomBreathPhase[s][i] -= 1.0f;

            float wakeThresh = bloomHueT[s][i] * BLOOM_RECOVERY_SPREAD;
            float ledRecovery = clampf(
                (bloomColonyEnergy - wakeThresh) / 0.30f, 0.0f, 1.0f);

            float breath = (fastSinPhase(bloomBreathPhase[s][i]) * 0.5f + 0.5f);
            float breathGlow = BLOOM_BREATH_FLOOR
                + breath * (bloomBreathPeak[s][i] - BLOOM_BREATH_FLOOR);
            breathGlow *= ledRecovery;

            if (draining) {
                float target = breathGlow * bloomHitIntensity * ENERGY_MULTIPLIER;
                if (target > bloomFlashGlow[s][i]) {
                    bloomFlashGlow[s][i] = target;
                    bloomFlashDecay[s][i] = BLOOM_FLASH_DECAY_LO
                        + randFloat(s) * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
                }
            }

            float g = fmaxf(breathGlow, bloomFlashGlow[s][i]);
            float flashFrac = (bloomFlashGlow[s][i] > breathGlow) ? 1.0f : 0.0f;
            float h = bloomHueT[s][i];
            float colG = lerpf(lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h),
                               BLOOM_FLASH_G, flashFrac);
            float colB = lerpf(lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h),
                               BLOOM_FLASH_B, flashFrac);

            float linBright = fastGamma24(g) * BLOOM_BRIGHTNESS_CAP;
            float oG = colG * linBright;
            float oB = colB * linBright;

            // RGBW→RGB W-fold: original drove W as cyan-white onset above
            // BLOOM_W_ONSET. Fold W luminance equally into R/G/B so above
            // the threshold the bulb pales toward white instead of dim cyan.
            float wFrac = clampf((g - BLOOM_W_ONSET) / (1.0f - BLOOM_W_ONSET),
                                 0.0f, 1.0f);
            float energyGate = clampf((bloomColonyEnergy - 0.7f) / 0.3f, 0.0f, 1.0f);
            float wGate = fmaxf(energyGate, flashFrac);
            float wFold = wFrac * wGate * linBright * 200.0f;

            float fR = wFold;
            float fG = oG + wFold;
            float fB = oB + wFold;

            float m = fadeoutMult[s][i];
            fR *= m; fG *= m; fB *= m;
            stripSetPixel(s, i,
                (uint8_t)clampf(fR, 0, 255),
                (uint8_t)clampf(fG, 0, 255),
                (uint8_t)clampf(fB, 0, 255));
        }
    }

    if (draining) {
        float drain = drainEnvelope * dt;
        drain = fminf(drain, bloomColonyEnergy);
        bloomColonyEnergy -= drain;
        drainEnvelope *= DRAIN_ENVELOPE_DECAY;
        if (drainEnvelope <= 0.001f) drainEnvelope = 0.0f;
    }
}

// ── Idle rainbow ────────────────────────────────────────────────────
static void renderIdleRainbow(uint32_t now) {
    float idlePhase = fmodf(now / 1000.0f / IDLE_PERIOD_S, 1.0f);
    uint8_t baseHueIdx = (uint8_t)(idlePhase * 256.0f) & 0xFF;
    currentIdleHueIdx = baseHueIdx;
    float idleLin = fastGamma24(IDLE_PEAK);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Per-strip hue offset gives 6 hues simultaneously across the
        // sculpture — a slow rainbow chord rotating in unison.
        uint8_t hueIdx = (uint8_t)((baseHueIdx + STRIP_HUE_OFFSET[s]) & 0xFF);
        float baseR = (float)oklchVarL[hueIdx][0] * idleLin;
        float baseG = (float)oklchVarL[hueIdx][1] * idleLin;
        float baseB = (float)oklchVarL[hueIdx][2] * idleLin;

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float m = fadeoutMult[s][i];
            stripSetPixel(s, i,
                (uint8_t)clampf(baseR * m, 0, 255),
                (uint8_t)clampf(baseG * m, 0, 255),
                (uint8_t)clampf(baseB * m, 0, 255));
        }
    }
}

// ── Main loop ───────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f
                                  : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    // Channel rescan
    if (now - lastScanMs > CHANNEL_RESCAN_MS) {
        uint8_t newCh = scanForSsidChannel(WIFI_SSID_TARGET);
        if (newCh && newCh != currentChannel) {
            Serial.printf("[heal] channel %u -> %u\n", currentChannel, newCh);
            applyChannel(newCh);
        }
        lastScanMs = millis();
    }

    // Serial commands
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'c' || c == 'C') startCalibration();
        else if (c == 's' || c == 'S') {
            currentEffect = (Effect)(((int)currentEffect + 1) % (int)EFF_COUNT);
            resetSparkleState();
            resetFireState();
            resetBloomState();
            Serial.printf("[effect] %d %s\n", (int)currentEffect, effectName(currentEffect));
        }
        else if (c == '?') {
            Serial.printf("[BOOT] role=bulb-fleet MAC=%s fw=bulb_fleet ch=%u pkt=%lu\n",
                          WiFi.macAddress().c_str(), currentChannel,
                          (unsigned long)pktCount);
        }
    }

    // Latest packet → orientation + audio
    float ax = (float)latestPacket.ax;
    float ay = (float)latestPacket.ay;
    float az = (float)latestPacket.az;
    vecNormalize(ax, ay, az);
    uint16_t rawRms = latestPacket.rawRms;
    bool micOn = latestPacket.micEnabled != 0;
    bool connected = (lastPacketMs > 0) && (now - lastPacketMs < TIMEOUT_MS);

    if (connected) updateCalibration(ax, ay, az);

    float angleDeg = 0;
    float dot = restAx * ax + restAy * ay + restAz * az;
    if (calibrated && connected) {
        angleDeg = acosf(clampf(dot, -1.0f, 1.0f)) * (180.0f / M_PI);
    }

    // bulb-fleet: always engaged. Disengage state machine disabled.
    engaged = true;
    engageCandidateMs = 0;

    // Sample current idle hue every frame so engage edge can capture it
    {
        float liveIdlePhase = fmodf(now / 1000.0f / IDLE_PERIOD_S, 1.0f);
        currentIdleHueIdx = (uint8_t)(liveIdlePhase * 256.0f) & 0xFF;
    }
    bool engagedRisingEdge = engaged && !prevEngaged;
    if (engagedRisingEdge) engageHueOffset = currentIdleHueIdx;
    prevEngaged = engaged;

    // Brightness envelope direction
    bool envWantDown = engaged && (engageCandidateMs > 0);
    if (envWantDown != envDirIsDown) {
        envDirIsDown = envWantDown;
        envDirStartMs = now;
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
                envDirStartVal[s][i] = fadeoutMult[s][i];
    }
    {
        const uint32_t span = DISENGAGE_HOLD_MS - FADEOUT_WINDOW_MS;
        uint32_t elapsed = now - envDirStartMs;
        float target = envDirIsDown ? FADEOUT_FLOOR : 1.0f;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                uint32_t offset = envDirIsDown
                    ? fadeStartMs[s][i]
                    : (span - fadeStartMs[s][i]);
                int32_t windowMs = (int32_t)elapsed - (int32_t)offset;
                float p = clampf((float)windowMs / (float)FADEOUT_WINDOW_MS, 0.0f, 1.0f);
                fadeoutMult[s][i] = envDirStartVal[s][i] + p * (target - envDirStartVal[s][i]);
            }
        }
    }

    // Audio processing (sparkle path only — there's only sparkle)
    if (connected && micOn) {
        energy = computeEnergy(rawRms);
        onset = computeOnset(rawRms);
    } else if (connected && !micOn) {
        energy = 0.15f;
        onset = 0.0f;
    }

    // Telemetry @ ~1Hz
    if (connected && calibrated && (pktCount % 25 == 0)) {
        uint32_t sinceLast = (lastPacketMs == 0) ? 0 : (now - lastPacketMs);
        Serial.printf("[sparkle] rms=%u pkt=%lu since_last=%lums max_gap=%lums\n",
            rawRms, (unsigned long)pktCount,
            (unsigned long)sinceLast, (unsigned long)maxGapMs);
        maxGapMs = 0;
    }

    // Path select
    bool packetTimeout = (lastPacketMs == 0) || (now - lastPacketMs > IDLE_GRACE_MS);
    bool useRainbowIdle = !engaged || packetTimeout;

    if (useRainbowIdle) {
        renderIdleRainbow(now);
    } else {
        // No-duck mode: time-driven hue cycle, tiltBlend pinned to 1.0.
        float tiltBlend = 1.0f;
        float synthAngle = fmodf(now / 30000.0f, 1.0f) * (MAX_ANGLE_DEG - DEADZONE_DEG) + DEADZONE_DEG;

        // Bloom: process motion once per new packet
        if (currentEffect == EFF_QUIET_BLOOM && pktCount != bloomPrevPktCount) {
            bloomProcessMotion(now);
            bloomPrevPktCount = pktCount;
        }

        switch (currentEffect) {
            case EFF_FIRE_MELD:      renderFire(dt, false, tiltBlend); break;
            case EFF_FIRE_FLICKER:   renderFire(dt, true,  tiltBlend); break;
            case EFF_QUIET_BLOOM:    renderQuietBloom(dt, now); break;
            case EFF_SPARKLE_BURST:  renderSparkleAll(dt, synthAngle, tiltBlend); break;
            case EFF_SPARKLE_SIMPLE: renderSparkleSimple(dt, synthAngle, tiltBlend); break;
            case EFF_SPARKLE_PROGRESSION: renderSparkleProgression(dt, now); break;
            default: renderFire(dt, false, tiltBlend); break;
        }
    }

    stripsShow();
    delay(1);  // yield to WiFi
}
