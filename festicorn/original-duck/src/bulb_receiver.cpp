/*
 * BULB RECEIVER — ESP32-C3 LED controller (stationary, near the LED strip)
 *
 * Receives raw sensor data via ESP-NOW from the handheld sender,
 * computes FixedRangeRMS energy + EnergyDelta onset detection locally,
 * renders effects on SK6812 RGBW LEDs.
 *
 * Algorithms:
 *   sparkle_burst — onset-triggered sparkle with decay
 *   fire_meld     — per-LED flame flicker, energy-driven color
 *   fire_flicker  — fire_meld + sustain-triggered blown-fire dropout
 *   quiet_bloom   — motion-reactive bioluminescence (gyro/accel, no audio)
 *
 * Serial commands:
 *   's' — sparkle_burst
 *   'm' — fire_meld
 *   'f' — fire_flicker
 *   'b' — quiet_bloom
 *   'c' — recalibrate rest vector
 *
 * Wiring:
 *   GPIO 10 → LED data (SK6812 RGBW) — "bulbs" (50 LEDs)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include <oklch_lut.h>
#include <delta_sigma.h>
#include <fast_math.h>

// ── LED config ───────────────────────────────────────────────────
#ifndef LED_PIN_OVERRIDE
#define LED_PIN    10
#else
#define LED_PIN    LED_PIN_OVERRIDE
#endif
#define LED_COUNT  50

// ── Rendering ────────────────────────────────────────────────────
#define GAMMA 2.4f
#define BRIGHTNESS_CAP 0.10f
#define PURE_W_CEIL    0.10f  // below this: pure W, no RGB dies
#define PURE_W_BLEND   0.15f  // blend range above CEIL to full RGB

// ── Color / tilt mapping ─────────────────────────────────────────
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f
#define BLEND_RANGE_DEG 40.0f
#define SENSOR_HZ       25.0f

// ── FixedRangeRMS parameters (scaled to 24-bit I2S integer range) ─
// Floor=10000 chosen empirically from INMP441 ambient distribution at the
// bench (max ~12k, p99 ~10k). Cleanly clears the noise floor while still
// admitting close-talked / shouted voice and any musical environment
// (music RMS sits at 20k+, often saturating uint16). Quiet conversational
// voice at arm's length will not trigger — expected; user moves closer or
// speaks up to engage the duck. See library/test-vectors/inmp441-validation
// for per-condition distributions used to pick this number.
#define RMS_FLOOR       10000.0f
#define RMS_CEILING     50000.0f  // loud speech/clapping into mic
#define RMS_PEAK_DECAY  0.9999f

// ── EnergyDelta parameters ───────────────────────────────────────
#define DELTA_PEAK_DECAY 0.998f

// ── Sparkle parameters ───────────────────────────────────────────
#define SPARKLE_DEADBAND 0.08f

// ── Fire parameters ──────────────────────────────────────────────
#define FIRE_FLICKER_SCALE  3.0f
#define FIRE_DEADBAND       0.08f
#define FIRE_DROPOUT_DEPTH  0.85f  // fire_flicker only

// ── Quiet bloom parameters ───────────────────────────────────────
#define BLOOM_BRIGHTNESS_CAP   0.70f
#define BLOOM_NOISE_GATE       256    // per-channel: snap to 0 below 1 LSB in 8.8

// Movement processing
#define SURPRISE_EMA_UP        0.05f  // slow rise — spikes don't inflate baseline
#define SURPRISE_EMA_DOWN      0.2f   // fast fall — tracks decay
#define SURPRISE_RATIO         3.0f   // motion must exceed EMA * ratio to trigger
#define DRAIN_SCALE            100.0f // cubic drain normalizer: ~300 deg/s dumps colony
#define DRAIN_ENVELOPE_DECAY   0.85f  // per-frame decay: most drain in ~0.5s (12 frames)
#define FLASH_MOTION_SCALE     300.0f // deg/s for full log-compressed flash brightness
#define ENERGY_MULTIPLIER      1.4f   // flash can exceed stored energy by this factor
#define MOTION_SETTLE_MS       300    // ms of stillness before recovery begins

// Breathing (every LED, always on)
#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.65f
#define BLOOM_BREATH_MAX_PEAK   1.00f
#define BLOOM_BREATH_FLOOR      0.15f

// Flash decay
#define BLOOM_FLASH_DECAY_LO   0.96f
#define BLOOM_FLASH_DECAY_HI   0.985f

// Colony recovery
#define BLOOM_RECOVERY_RAMP    0.033f // 0→1 in ~30s (3x slower)
#define BLOOM_RECOVERY_SPREAD  0.70f  // per-LED wake thresholds span 0..0.7

// Color (blue/cyan palette — no R channel)
#define BLOOM_HUE_A_G   20.0f   // deep blue green
#define BLOOM_HUE_A_B  100.0f   // deep blue blue
#define BLOOM_HUE_B_G   70.0f   // teal green
#define BLOOM_HUE_B_B  110.0f   // teal blue
#define BLOOM_FLASH_G  150.0f   // flash cyan green
#define BLOOM_FLASH_B  170.0f   // flash cyan blue
#define BLOOM_W_ONSET    0.5f   // glow above this adds white channel

// ── ESP-NOW ──────────────────────────────────────────────────────
#define TIMEOUT_MS     500

// ── Wi-Fi channel discovery (matches bench-bulbs + v1 sender) ────
// Lock radio to whatever channel SSID `cuteplant` advertises on, with a
// 5-min rescan to heal AP channel drift. No AP join — ESP-NOW only.
#define CHANNEL_FALLBACK   1
#define CHANNEL_RESCAN_MS  (5UL * 60UL * 1000UL)
static const char* WIFI_SSID_TARGET = "cuteplant";
static uint8_t  currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs     = 0;

static uint8_t scanForSsidChannel(const char* ssid) {
    int n = WiFi.scanNetworks(/*async=*/false, /*show_hidden=*/false,
                              /*passive=*/true, /*max_ms_per_chan=*/120);
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

// LEGACY: 15-byte v0.1 SensorPacket. Will go silent the moment a v1 sender
// (festicorn/sender_rnd) is flashed nearby, since v1 packets are 16 B and
// onReceive's len-check will reject them. Kept here for the held-duck
// firmware that still uses audio RMS + shake-toggle.
struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;     // 6 bytes: raw accelerometer (±2g, 16384 = 1g)
    int16_t gx, gy, gz;     // 6 bytes: raw gyroscope (±250°/s, /131 = deg/s)
    uint16_t rawRms;        // 2 bytes: raw audio RMS
    uint8_t micEnabled;     // 1 byte: shake toggle state
};                          // 15 bytes total

// ── Algorithm enum ───────────────────────────────────────────────
enum Algorithm {
    ALG_SPARKLE_BURST,
    ALG_FIRE_MELD,
    ALG_FIRE_FLICKER,
    ALG_QUIET_BLOOM,
};

static Algorithm currentAlg = ALG_SPARKLE_BURST;

// ── Global state ─────────────────────────────────────────────────

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800);
// Delta-sigma accumulators per pixel, per channel
static uint16_t dsR[LED_COUNT], dsG[LED_COUNT], dsB[LED_COUNT], dsW[LED_COUNT];

// Latest received packet
static volatile bool packetReady = false;
static volatile uint32_t lastPacketMs = 0;
static volatile uint32_t pktCount = 0;
static SensorPacket latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

// Calibration — rest vector
static float restAx = 0, restAy = 0, restAz = 1.0f;
static bool calibrated = false;
static float calSumAx = 0, calSumAy = 0, calSumAz = 0;
static uint32_t calSamples = 0;
static uint32_t calStartMs = 0;
#define CAL_DURATION_MS 2000

// ── FixedRangeRMS state ──────────────────────────────────────────
static float frrCeiling = RMS_CEILING;
static float energy = 0.0f;

// ── EnergyDelta state ────────────────────────────────────────────
static float prevRms = 0.0f;
static float deltaPeak = 1e-6f;
static float onset = 0.0f;

// ── Sparkle burst render state ───────────────────────────────────
static float sparkle[LED_COUNT];
static float decayRates[LED_COUNT];
static float envelope = 0.0f;
static float cooldownRemaining = 0.0f;

// ── Fire render state (shared by fire_meld and fire_flicker) ─────
static float fireTime = 0.0f;
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
// Dropout state (fire_flicker only)
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

// ── Quiet bloom render state ─────────────────────────────────────
static float bloomBreathPhase[LED_COUNT];
static float bloomBreathPeriod[LED_COUNT];
static float bloomBreathPeak[LED_COUNT];
static float bloomHueT[LED_COUNT];        // color variation + recovery wake stagger
static float bloomFlashGlow[LED_COUNT];
static float bloomFlashDecay[LED_COUNT];
static float bloomColonyEnergy = 1.0f;
static uint32_t bloomLastMotionMs = 0;
static float bloomMotionRate = 0.0f;
static float motionEMA = 0.0f;
static float drainEnvelope = 0.0f;
static float bloomHitIntensity = 0.0f;
static uint32_t bloomPrevPktCount = 0;

// Simple PRNG state (xorshift32, seeded from esp_random at boot)
static uint32_t prngState;

// ── ESP-NOW receive callback ─────────────────────────────────────

void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    pktCount++;
    if (len == sizeof(SensorPacket)) {
        memcpy((void*)&latestPacket, data, sizeof(SensorPacket));
        lastPacketMs = millis();
        packetReady = true;
    }
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

// Returns float in [0, 1)
static inline float randFloat() {
    return (float)(xorshift32() & 0xFFFFFF) / 16777216.0f;
}

// ── Motion helpers (quiet bloom) ─────────────────────────────────

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

static float computeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx * fx + fy * fy + fz * fz) / 131.0f;  // ±250°/s range
}

static float computeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx * fx + fy * fy + fz * fz);
    return fabsf(mag - 16384.0f) / 16384.0f;  // 0 = stationary, 1 = ±1g jolt
}

// ── FixedRangeRMS: raw uint16 RMS → 0-1 energy ──────────────────

static float computeEnergy(uint16_t rawRms) {
    float rms = (float)rawRms;

    // Peak decay on ceiling — expands for loud input, decays back to base
    frrCeiling = fmaxf(RMS_CEILING, frrCeiling * RMS_PEAK_DECAY);
    if (rms > frrCeiling) frrCeiling = rms;

    // Hard gate
    if (rms < RMS_FLOOR) return 0.0f;

    // Log mapping over the current range
    float db = 20.0f * log10f(rms / RMS_FLOOR);
    float dbRange = 20.0f * log10f(frrCeiling / RMS_FLOOR);
    return clampf(db / dbRange, 0.0f, 1.0f);
}

// ── EnergyDelta: frame-to-frame RMS change → 0-1 onset ──────────

static float computeOnset(uint16_t rawRms) {
    float rms = (float)rawRms;
    float delta = fabsf(rms - prevRms);
    prevRms = rms;

    // Slow-decay peak normalization
    deltaPeak = fmaxf(delta, deltaPeak * DELTA_PEAK_DECAY);
    return (deltaPeak > 1e-6f) ? (delta / deltaPeak) : 0.0f;
}

// Forward declarations (defined after setup)
static void resetBloomState();

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    // ── LED strip init ────────────────────────────────────────────
    strip.begin();
    strip.setBrightness(255);

    // Seed delta-sigma accumulators
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
        dsR[i] = seed;
        dsG[i] = (seed + 64) & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;
    }
    // Init sparkle state
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] = 0.0f;
        decayRates[i] = 0.94f;
    }

    // Seed PRNG from hardware RNG
    prngState = esp_random();
    if (prngState == 0) prngState = 1;  // xorshift can't have 0 state

    // Pre-seed decay rates with some variation
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        decayRates[i] = 0.92f + randFloat() * 0.05f;
    }

    // Init bloom state (uses randFloat, so must be after PRNG seed)
    resetBloomState();

    // Boot flash — brief green
    for (uint16_t i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, 0, 40, 0, 0);
    strip.show();
    delay(200);
    strip.clear();
    strip.show();

    // ── Wi-Fi: STA mode + SSID-based channel discovery (no AP join) ──
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);  // C3 Super Mini regulator brownout fix (see engineering ledger)

    uint8_t ch = scanForSsidChannel(WIFI_SSID_TARGET);
    if (ch) {
        Serial.printf("Found '%s' on ch=%u\n", WIFI_SSID_TARGET, ch);
    } else {
        Serial.printf("'%s' not visible — falling back to ch=%u\n",
                      WIFI_SSID_TARGET, CHANNEL_FALLBACK);
    }
    applyChannel(ch ? ch : CHANNEL_FALLBACK);
    lastScanMs = millis();

    esp_err_t initResult = esp_now_init();
    esp_err_t cbResult = esp_now_register_recv_cb(onReceive);
    Serial.printf("ESP-NOW init=%d, cb=%d, ch=%u, MAC=%s\n",
        initResult, cbResult, currentChannel, WiFi.macAddress().c_str());

    Serial.println("Waiting for sender... (will auto-calibrate from first 2s of data)");
    Serial.println("Commands: 'c' recal, 's' sparkle, 'm' fire_meld, 'f' fire_flicker, 'b' bloom");
}

// ── Calibration ──────────────────────────────────────────────────

void startCalibration() {
    calibrated = false;
    calSumAx = 0; calSumAy = 0; calSumAz = 0;
    calSamples = 0;
    calStartMs = millis();
    Serial.println("Calibrating — keep duck still for 2 seconds...");
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

// ── Reset fire state (called on algorithm switch) ────────────────

static void resetFireState() {
    fireTime = 0.0f;
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

// Sparkle output instrumentation (peaks across strip, last frame)
static uint8_t sparkPeakR = 0, sparkPeakG = 0, sparkPeakB = 0, sparkPeakW = 0;
static float sparkPeakBright = 0.0f;
static float sparkPeakLin = 0.0f;

// ── Reset sparkle state (called on algorithm switch) ─────────────

static void resetSparkleState() {
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] = 0.0f;
        decayRates[i] = 0.92f + randFloat() * 0.05f;
    }
    envelope = 0.0f;
    cooldownRemaining = 0.0f;
}

// ── Reset bloom state (called on algorithm switch) ───────────────

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

// ── Quiet bloom: motion processing (call once per new packet) ────

static void bloomProcessMotion(float pktDt, uint32_t now) {
    float gyroRate = computeGyroRate(
        latestPacket.gx, latestPacket.gy, latestPacket.gz);
    float accelJolt = computeAccelJolt(
        latestPacket.ax, latestPacket.ay, latestPacket.az);
    bloomMotionRate = fmaxf(gyroRate, accelJolt * 300.0f);

    // Surprise detector: compare BEFORE updating EMA so spike doesn't dilute itself
    float surprise = fmaxf(0.0f, bloomMotionRate - motionEMA * SURPRISE_RATIO);

    // Asymmetric EMA: rise slowly (spikes don't inflate baseline), fall normally
    float alpha = (bloomMotionRate > motionEMA) ? SURPRISE_EMA_UP : SURPRISE_EMA_DOWN;
    motionEMA += alpha * (bloomMotionRate - motionEMA);

    if (surprise > 1.0f) {
        bloomLastMotionMs = now;

        // Store hit intensity for sustained flash in render path
        float hitIntensity = clampf(
            log2f(1.0f + surprise) / log2f(1.0f + FLASH_MOTION_SCALE),
            0.0f, 1.0f);
        if (hitIntensity > bloomHitIntensity) bloomHitIntensity = hitIntensity;

        // Set drain envelope — scaled so total geometric sum matches intended drain
        float normMotion = surprise / DRAIN_SCALE;
        float newDrain = normMotion * normMotion * normMotion
                       * (1.0f - DRAIN_ENVELOPE_DECAY);
        if (newDrain > drainEnvelope) drainEnvelope = newDrain;
    }
}

// ── Quiet bloom render ──────────────────────────────────────────

static void renderQuietBloom(float dt, uint32_t now) {
    // Check drain state (drain applied AFTER rendering so flash sees pre-drain energy)
    bool draining = drainEnvelope > 0.001f;
    if (!draining) {
        bloomHitIntensity = 0.0f;
    }

    // Colony recovery after motion settles
    if (now - bloomLastMotionMs > MOTION_SETTLE_MS) {
        bloomColonyEnergy = fminf(1.0f, bloomColonyEnergy + BLOOM_RECOVERY_RAMP * dt);
    }

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        // Flash decay (only when NOT being sustained by drain envelope)
        if (!draining) {
            bloomFlashGlow[i] *= fastDecay(bloomFlashDecay[i], dt * 30.0f);
            if (bloomFlashGlow[i] < 0.005f) bloomFlashGlow[i] = 0.0f;
        }

        // Breathing
        bloomBreathPhase[i] += dt / bloomBreathPeriod[i];
        if (bloomBreathPhase[i] >= 1.0f) bloomBreathPhase[i] -= 1.0f;

        // Per-LED staggered recovery
        float wakeThresh = bloomHueT[i] * BLOOM_RECOVERY_SPREAD;
        float ledRecovery = clampf(
            (bloomColonyEnergy - wakeThresh) / 0.30f, 0.0f, 1.0f);

        // Breathing glow × colony recovery
        float breath = (fastSinPhase(bloomBreathPhase[i]) * 0.5f + 0.5f);
        float breathGlow = BLOOM_BREATH_FLOOR
            + breath * (bloomBreathPeak[i] - BLOOM_BREATH_FLOOR);
        breathGlow *= ledRecovery;

        // Sustained flash: while drain is active, refresh flash from breathing
        if (draining) {
            float target = breathGlow * bloomHitIntensity * ENERGY_MULTIPLIER;
            if (target > bloomFlashGlow[i]) {
                bloomFlashGlow[i] = target;
                bloomFlashDecay[i] = BLOOM_FLASH_DECAY_LO
                    + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
            }
        }

        // Composite
        float g = fmaxf(breathGlow, bloomFlashGlow[i]);

        // Color: breathing = blue, flash = cyan
        float flashFrac = (bloomFlashGlow[i] > breathGlow) ? 1.0f : 0.0f;
        float h = bloomHueT[i];
        float colG = lerpf(lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h),
                           BLOOM_FLASH_G, flashFrac);
        float colB = lerpf(lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h),
                           BLOOM_FLASH_B, flashFrac);

        // Gamma + cap (bloom uses its own brightness cap)
        float linBright = fastGamma24(g) * BLOOM_BRIGHTNESS_CAP;

        float oG = colG * linBright;
        float oB = colB * linBright;

        // W channel: gated by colony energy so it stays blue during recovery
        float wFrac = clampf((g - BLOOM_W_ONSET) / (1.0f - BLOOM_W_ONSET),
                             0.0f, 1.0f);
        float energyGate = clampf((bloomColonyEnergy - 0.7f) / 0.3f, 0.0f, 1.0f);
        float wGate = fmaxf(energyGate, flashFrac);  // bypass gate during flash
        float oW = wFrac * wGate * linBright * 200.0f;

        // 8.8 fixed-point → delta-sigma
        uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

        // Per-channel noise gate
        if (tG16 < BLOOM_NOISE_GATE) tG16 = 0;
        if (tB16 < BLOOM_NOISE_GATE) tB16 = 0;
        if (tW16 < BLOOM_NOISE_GATE) tW16 = 0;

        uint8_t gc = deltaSigma(dsG[i], tG16);
        uint8_t b  = deltaSigma(dsB[i], tB16);
        uint8_t w  = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, 0, gc, b, w);
    }

    // Apply drain AFTER rendering so flash uses pre-drain energy
    if (draining) {
        float drain = drainEnvelope * dt;
        drain = fminf(drain, bloomColonyEnergy);
        bloomColonyEnergy -= drain;
        drainEnvelope *= DRAIN_ENVELOPE_DECAY;
        if (drainEnvelope <= 0.001f) drainEnvelope = 0.0f;
    }
}

// ── Sparkle burst render ─────────────────────────────────────────

static void renderSparkleBurst(float dt, float angleDeg, float tiltBlend,
                                float tiltR, float tiltG, float tiltB) {
    bool isSilent = energy < 0.001f;

    // Asymmetric envelope: ~30ms attack, ~400ms decay
    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > envelope)
        envelope += attackAlpha * (energy - envelope);
    else
        envelope += decayAlpha * (energy - envelope);

    // Cooldown tick
    cooldownRemaining = fmaxf(0.0f, cooldownRemaining - dt);

    // Onset threshold: lower when energy is high (more frequent sparkles)
    float onsetThreshold = fmaxf(0.15f, 0.4f - envelope * 0.3f);

    // Onset detection: ignite random subset of LEDs
    if (onset > onsetThreshold && cooldownRemaining <= 0.0f && !isSilent) {
        // Cooldown shorter at higher energy
        cooldownRemaining = fmaxf(0.050f, 0.150f - envelope * 0.10f);

        float onsetStrength = clampf(onset, 0.0f, 1.0f);
        int nIgnite = (int)(LED_COUNT * (0.3f + 0.2f * onsetStrength));

        // Fisher-Yates partial shuffle to pick nIgnite unique indices
        static uint8_t indices[LED_COUNT];
        for (uint8_t i = 0; i < LED_COUNT; i++) indices[i] = i;
        for (int i = 0; i < nIgnite; i++) {
            int j = i + (int)(xorshift32() % (LED_COUNT - i));
            uint8_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        float sparkVal = 0.7f + 0.3f * onsetStrength;
        for (int i = 0; i < nIgnite; i++) {
            sparkle[indices[i]] = sparkVal;
            decayRates[indices[i]] = 0.92f + randFloat() * 0.05f;
        }
    }

    // Per-LED decay (frame-rate independent: rate^(dt*30))
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] *= fastDecay(decayRates[i], dt * 30.0f);
    }

    // Subtle shimmer jitter between onsets (only when not silent)
    if (!isSilent) {
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float jitter = (randFloat() - 0.5f) * 0.02f;
            float newVal = sparkle[i] + jitter;
            if (newVal < 0.0f) newVal = 0.0f;
            if (newVal > sparkle[i] && jitter > 0) newVal = sparkle[i];
            sparkle[i] = newVal;
        }
    }

    // Base brightness: envelope capped at 50%
    float base = fminf(envelope, 0.2f);

    // Reset per-frame instrumentation peaks
    sparkPeakR = sparkPeakG = sparkPeakB = sparkPeakW = 0;
    sparkPeakBright = 0.0f;
    sparkPeakLin = 0.0f;

    // ── Per-LED color + brightness → RGBW output ─────────────────
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float s = sparkle[i];

        // Final per-LED brightness: base + sparkle fills gap up to 1.0
        float bright = base + s * (1.0f - base);

        // Deadband snap-to-zero
        if (bright < SPARKLE_DEADBAND) bright = 0.0f;

        // Color: warm amber base, sparkle interpolates toward white
        float colR = 255.0f;
        float colG = 180.0f + (240.0f - 180.0f) * s;
        float colB =  80.0f + (200.0f -  80.0f) * s;

        // Tilt hue overlay: blend in OKLCH color when tilted
        if (tiltBlend > 0.0f) {
            colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
            colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
            colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
        }

        // Apply brightness + gamma. Sparkle uses W as primary luminance:
        // full at upright, fades with tilt so OKLCH color takes over. RGB
        // carries warm-amber tint on top. The pure-W-at-low-brightness
        // curve used by bloom/fire isn't needed here — W always loaded.
        float linBright = fastGamma24(bright) * BRIGHTNESS_CAP;
        float colW = 255.0f * (1.0f - tiltBlend);
        float fR = colR * linBright;
        float fG = colG * linBright;
        float fB = colB * linBright;
        float fW = colW * linBright;

        // Scale to 16-bit for delta-sigma dithering
        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);

        // Per-channel noise gate: snap each channel independently to 0
        // if it would dither in the 0↔1 zone (target16 < 256)
        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, r, g, b, w);

        if (r > sparkPeakR) sparkPeakR = r;
        if (g > sparkPeakG) sparkPeakG = g;
        if (b > sparkPeakB) sparkPeakB = b;
        if (w > sparkPeakW) sparkPeakW = w;
        if (bright > sparkPeakBright) sparkPeakBright = bright;
        if (linBright > sparkPeakLin) sparkPeakLin = linBright;
    }
}

// ── Fire render (shared by fire_meld and fire_flicker) ───────────

static void renderFire(float dt, bool withDropout, float tiltBlend) {
    // Wrap at 1000·2π so float32 precision never collapses dt into a no-op
    // (would freeze flicker after ~1 day if user sat on fire algorithm).
    // t feeds only fastSin(); phase jump at wrap is invisible in chaotic noise.
    fireTime = fmodf(fireTime + dt, 6283.1853f);
    float t = fireTime;

    bool isSilent = energy < 0.001f;

    // Percussive-only detection: high delta relative to low energy
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);

    // --- Base brightness envelope ---
    // Asymmetric EMA: ~50ms attack, ~2s release
    float attackAlpha = fminf(1.0f, dt / 0.050f);
    float decayAlpha  = fminf(1.0f, dt / 2.0f);

    float targetBrightness;
    if (isSilent) {
        targetBrightness = 0.25f;  // hold at 25% during silence
    } else {
        targetBrightness = fmaxf(0.25f, energy);
    }

    if (targetBrightness > fireBaseBrightness)
        fireBaseBrightness += attackAlpha * (targetBrightness - fireBaseBrightness);
    else
        fireBaseBrightness += decayAlpha * (targetBrightness - fireBaseBrightness);

    // --- Flicker intensity: EMA-smoothed energy delta (~200ms TC) ---
    float flickerAlpha = fminf(1.0f, dt / 0.200f);
    float deltaTarget = isSilent ? 0.0f : onset;
    fireFlickerIntensity += flickerAlpha * (deltaTarget - fireFlickerIntensity);

    // --- Sustain-triggered dropout (fire_flicker only) ---
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

    // --- Color energy: sustained energy tracking for color mapping ---
    float colorAttack = fminf(1.0f, dt / 0.080f);
    float colorDecay  = fminf(1.0f, dt / 2.0f);

    float colorTarget;
    if (isPercussiveOnly || isSilent)
        colorTarget = 0.0f;
    else
        colorTarget = energy;

    if (colorTarget > fireColorEnergy)
        fireColorEnergy += colorAttack * (colorTarget - fireColorEnergy);
    else
        fireColorEnergy += colorDecay * (colorTarget - fireColorEnergy);

    // --- Base color from color_energy ---
    float ce = fireColorEnergy;
    float baseColR, baseColG, baseColB;

    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;

    // Color anchors
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

    // --- Deadband on base brightness ---
    float base = fireBaseBrightness;
    if (base < FIRE_DEADBAND) base = 0.0f;

    float s = FIRE_FLICKER_SCALE;

    // --- Per-LED rendering ---
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float fi = (float)i;

        // Per-LED noise: two incommensurate sines
        float noise = fastSin(fi * 7.3f + t * 2.5f) *
                       fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

        // Noise amplitude scaled by flicker_scale
        float noiseAmp = fmaxf(0.15f * s, 0.10f * s / fmaxf(base, 0.1f));
        float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                        + fireFlickerIntensity * (noise - 0.5f) * 0.25f * s;

        // --- Dropout per LED (fire_flicker only) ---
        float perLedDim = 0.0f;
        float colorRedShift = 0.0f;
        if (withDropout && dropoutAmount > 0.0f) {
            // Slow-drifting resilience per LED
            float resilience = fastSin(fi * 13.7f + t * 0.3f) *
                                fastSin(fi * 9.1f + t * 0.2f) * 0.5f + 0.5f;
            perLedDim = clampf(
                (dropoutAmount - resilience * 0.7f) / 0.3f, 0.0f, 1.0f
            ) * FIRE_DROPOUT_DEPTH;

            // Color red shift leads brightness
            colorRedShift = clampf(perLedDim / 0.3f, 0.0f, 1.0f);
        }

        // Apply brightness dropout
        bright *= (1.0f - perLedDim);
        bright = clampf(bright, 0.0f, 1.0f);

        // Per-LED color: blend base color toward ember red with dropout
        float colR = baseColR * (1.0f - colorRedShift) + redR * colorRedShift;
        float colG = baseColG * (1.0f - colorRedShift) + redG * colorRedShift;
        float colB = baseColB * (1.0f - colorRedShift) + redB * colorRedShift;

        // Apply brightness + gamma
        float linBright = fastGamma24(bright) * BRIGHTNESS_CAP;
        float oR = colR * linBright;
        float oG = colG * linBright;
        float oB = colB * linBright;

        // RGBW: pure W at low brightness, blend to RGB above threshold
        // (see engineering ledger: rgbw-pure-w-at-low-brightness)
        float maxCh_f = fmaxf(oR, fmaxf(oG, oB));
        float bFrac = maxCh_f / 255.0f;
        float rgbBlend = clampf((bFrac - PURE_W_CEIL) / PURE_W_BLEND, 0.0f, 1.0f);
        float avgRGB = (oR + oG + oB) / 3.0f;
        float fR = oR * rgbBlend;
        float fG = oG * rgbBlend;
        float fB = oB * rgbBlend;
        float fW = avgRGB * (1.0f - rgbBlend) * (1.0f - tiltBlend);

        // Scale to 16-bit for delta-sigma dithering
        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);

        // Per-channel noise gate: snap each channel independently to 0
        // if it would dither in the 0↔1 zone (target16 < 256)
        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, r, g, b, w);
    }
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;  // cap at 100ms to avoid jumps
    lastRenderMs = now;

    // Periodic SSID rescan — heals if the AP moves channel.
    if (now - lastScanMs > CHANNEL_RESCAN_MS) {
        uint8_t newCh = scanForSsidChannel(WIFI_SSID_TARGET);
        if (newCh && newCh != currentChannel) {
            Serial.printf("[heal] channel drift %u -> %u\n", currentChannel, newCh);
            applyChannel(newCh);
        }
        lastScanMs = millis();
    }

    // ── Serial commands ─────────────────────────────────────────
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'c' || c == 'C') {
            startCalibration();
        } else if (c == 's' || c == 'S') {
            currentAlg = ALG_SPARKLE_BURST;
            resetSparkleState();
            Serial.println("Algorithm: sparkle_burst");
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
        }
    }

    // ── Read latest packet & normalize accel to unit vector ─────
    float ax = (float)latestPacket.ax;
    float ay = (float)latestPacket.ay;
    float az = (float)latestPacket.az;
    vecNormalize(ax, ay, az);
    uint16_t rawRms = latestPacket.rawRms;
    bool micOn = latestPacket.micEnabled != 0;
    bool connected = (lastPacketMs > 0) && (now - lastPacketMs < TIMEOUT_MS);

    // ── Calibration accumulation ─────────────────────────────────
    if (connected) {
        updateCalibration(ax, ay, az);
    }

    // ── Compute tilt angle from rest ─────────────────────────────
    float angleDeg = 0;
    if (calibrated && connected) {
        float cosAngle = clampf(
            restAx * ax + restAy * ay + restAz * az,
            -1.0f, 1.0f
        );
        angleDeg = acosf(cosAngle) * (180.0f / M_PI);
    }

    // ── Audio processing (sparkle/fire only) ───────────────────
    if (currentAlg != ALG_QUIET_BLOOM) {
        if (connected && micOn) {
            energy = computeEnergy(rawRms);
            onset = computeOnset(rawRms);
        } else if (connected && !micOn) {
            energy = 0.15f;
            onset = 0.0f;
        }
    }

    // ── Motion processing (bloom only) ──────────────────────────
    if (currentAlg == ALG_QUIET_BLOOM && connected) {
        if (pktCount != bloomPrevPktCount) {
            static uint32_t lastBloomPktMs = 0;
            float pktDt = (lastBloomPktMs > 0)
                ? (now - lastBloomPktMs) / 1000.0f
                : (1.0f / SENSOR_HZ);
            if (pktDt > 0.2f) pktDt = 0.2f;
            lastBloomPktMs = now;
            bloomProcessMotion(pktDt, now);
            bloomPrevPktCount = pktCount;
        }
    }

    // ── Debug telemetry (every ~1s) ──────────────────────────────
    if (connected && calibrated && (pktCount % 25 == 0)) {
        const char *algName = "sparkle";
        if (currentAlg == ALG_FIRE_MELD) algName = "fire_meld";
        else if (currentAlg == ALG_FIRE_FLICKER) algName = "fire_flicker";
        else if (currentAlg == ALG_QUIET_BLOOM) algName = "bloom";

        if (currentAlg == ALG_QUIET_BLOOM) {
            Serial.printf("[%s] rate=%.1f energy=%.3f flash0=%.3f\n",
                algName, bloomMotionRate, bloomColonyEnergy, bloomFlashGlow[0]);
        } else if (currentAlg == ALG_SPARKLE_BURST) {
            Serial.printf("[%s] angle=%.1f rms=%d en=%.3f ons=%.3f peak b=%.2f lin=%.3f rgbw=%3u/%3u/%3u/%3u\n",
                algName, angleDeg, rawRms, energy, onset,
                sparkPeakBright, sparkPeakLin,
                sparkPeakR, sparkPeakG, sparkPeakB, sparkPeakW);
        } else {
            Serial.printf("[%s] angle=%.1f rms=%d energy=%.3f onset=%.3f mic=%s\n",
                algName, angleDeg, rawRms, energy, onset, micOn ? "on" : "off");
        }
    }

    // ── Timeout: breathing idle ──────────────────────────────────
    if (!connected) {
        float breath = (fastSin(now / 1000.0f * M_PI) + 1.0f) / 2.0f;
        // Render breathing on pure warm white
        uint16_t bW = (uint16_t)(fastGamma24(breath * 0.12f) * 65535.0f);
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            uint8_t w = deltaSigma(dsW[i], bW >> 8);
            strip.setPixelColor(i, 0, 0, 0, w);
        }
        strip.show();

        delay(1);
        return;
    }

    // ── Tilt → OKLCH hue (shared by all algorithms) ─────────────
    float tiltR = 0, tiltG = 0, tiltB = 0;
    float tiltBlend = 0.0f;
    if (angleDeg > DEADZONE_DEG) {
        float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
        if (hueFrac > 1.0f) hueFrac = 1.0f;
        uint8_t hueIdx = (uint8_t)(hueFrac * 255) % 256;
        tiltR = (float)oklchVarL[hueIdx][0];
        tiltG = (float)oklchVarL[hueIdx][1];
        tiltB = (float)oklchVarL[hueIdx][2];
        tiltBlend = (angleDeg - DEADZONE_DEG) / BLEND_RANGE_DEG;
        if (tiltBlend > 1.0f) tiltBlend = 1.0f;
    }

    // ── Render current algorithm ─────────────────────────────────
    switch (currentAlg) {
        case ALG_SPARKLE_BURST:
            renderSparkleBurst(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB);
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
    }

    strip.show();

    // Yield to WiFi task — critical on single-core ESP32-C3
    delay(1);
}
