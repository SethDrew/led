/*
 * RECEIVER — ESP32-C3 LED controller (stationary, near the LED strip)
 *
 * Receives raw sensor data via ESP-NOW from the handheld sender,
 * computes FixedRangeRMS energy + EnergyDelta onset detection locally,
 * renders effects on SK6812 RGBW LEDs.
 *
 * Algorithms:
 *   sparkle_burst — onset-triggered sparkle with decay
 *   fire_meld     — per-LED flame flicker, energy-driven color
 *   fire_flicker  — fire_meld + sustain-triggered blown-fire dropout
 *
 * Serial commands:
 *   's' — sparkle_burst
 *   'm' — fire_meld
 *   'f' — fire_flicker
 *   'c' — recalibrate rest vector
 *
 * Wiring:
 *   GPIO 10 → LED data (SK6812 RGBW)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_random.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include <oklch_lut.h>
#include <delta_sigma.h>

// ── LED config ───────────────────────────────────────────────────
#define LED_PIN    10
#define LED_COUNT  50

// ── Rendering ────────────────────────────────────────────────────
#define GAMMA 2.4f

// ── Color / tilt mapping ─────────────────────────────────────────
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f
#define BLEND_RANGE_DEG 40.0f
#define SENSOR_HZ       25.0f

// ── FixedRangeRMS parameters (scaled to 24-bit I2S integer range) ─
#define RMS_FLOOR       4000.0f   // above INMP441 ambient (~2400-3400)
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

// ── ESP-NOW ──────────────────────────────────────────────────────
#define TIMEOUT_MS     500

struct __attribute__((packed)) SensorPacket {
    float ax, ay, az;       // 12 bytes: smoothed accel unit vector
    uint16_t rawRms;        // 2 bytes: raw audio RMS
    uint8_t micEnabled;     // 1 byte: shake toggle state
};                          // 15 bytes total

// ── Algorithm enum ───────────────────────────────────────────────
enum Algorithm {
    ALG_SPARKLE_BURST,
    ALG_FIRE_MELD,
    ALG_FIRE_FLICKER,
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
static SensorPacket latestPacket = {0, 0, 1.0f, 0, 1};

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

    // Boot flash — brief green
    for (uint16_t i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, 0, 40, 0, 0);
    strip.show();
    delay(200);
    strip.clear();
    strip.show();

    // ── ESP-NOW init ──────────────────────────────────────────────
    WiFi.mode(WIFI_STA);
    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.println("Waiting for sender... (will auto-calibrate from first 2s of data)");
    Serial.println("Commands: 'c' recalibrate, 's' sparkle, 'm' fire_meld, 'f' fire_flicker");
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

// ── Reset sparkle state (called on algorithm switch) ─────────────

static void resetSparkleState() {
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] = 0.0f;
        decayRates[i] = 0.92f + randFloat() * 0.05f;
    }
    envelope = 0.0f;
    cooldownRemaining = 0.0f;
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
        sparkle[i] *= powf(decayRates[i], dt * 30.0f);
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

        // W channel: full when upright, fades with tilt
        float colW = 255.0f * (1.0f - tiltBlend);

        // Apply brightness + gamma
        float linBright = powf(bright, GAMMA);
        float oR = colR * linBright;
        float oG = colG * linBright;
        float oB = colB * linBright;
        float oW = colW * linBright;

        // Scale to 16-bit for delta-sigma dithering
        uint16_t tR16 = (uint16_t)clampf(oR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

        // Sub-LSB noise gate
        if (tR16 < 64) tR16 = 0;
        if (tG16 < 64) tG16 = 0;
        if (tB16 < 64) tB16 = 0;
        if (tW16 < 64) tW16 = 0;

        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, r, g, b, w);
    }
}

// ── Fire render (shared by fire_meld and fire_flicker) ───────────

static void renderFire(float dt, bool withDropout, float tiltBlend) {
    fireTime += dt;
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
        float noise = sinf(fi * 7.3f + t * 2.5f) *
                       sinf(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

        // Noise amplitude scaled by flicker_scale
        float noiseAmp = fmaxf(0.15f * s, 0.10f * s / fmaxf(base, 0.1f));
        float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                        + fireFlickerIntensity * (noise - 0.5f) * 0.25f * s;

        // --- Dropout per LED (fire_flicker only) ---
        float perLedDim = 0.0f;
        float colorRedShift = 0.0f;
        if (withDropout && dropoutAmount > 0.0f) {
            // Slow-drifting resilience per LED
            float resilience = sinf(fi * 13.7f + t * 0.3f) *
                                sinf(fi * 9.1f + t * 0.2f) * 0.5f + 0.5f;
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

        // W channel: warm white proportional to brightness, fades with tilt
        float colW = 255.0f * (1.0f - tiltBlend);

        // Apply brightness + gamma
        float linBright = powf(bright, GAMMA);
        float oR = colR * linBright;
        float oG = colG * linBright;
        float oB = colB * linBright;
        float oW = colW * linBright;

        // Scale to 16-bit for delta-sigma dithering
        uint16_t tR16 = (uint16_t)clampf(oR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

        // Sub-LSB noise gate
        if (tR16 < 64) tR16 = 0;
        if (tG16 < 64) tG16 = 0;
        if (tB16 < 64) tB16 = 0;
        if (tW16 < 64) tW16 = 0;

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
    float dt = 1.0f / SENSOR_HZ;  // ~40ms

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
        }
    }

    // ── Read latest packet ───────────────────────────────────────
    float ax = latestPacket.ax;
    float ay = latestPacket.ay;
    float az = latestPacket.az;
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

    // ── Audio processing ─────────────────────────────────────────
    if (connected && micOn) {
        energy = computeEnergy(rawRms);
        onset = computeOnset(rawRms);
    } else if (connected && !micOn) {
        // Mic disabled: hold a gentle base glow
        energy = 0.15f;
        onset = 0.0f;
    }

    // ── Debug telemetry (every ~1s) ──────────────────────────────
    if (connected && calibrated && (pktCount % 25 == 0)) {
        const char *algName = "sparkle";
        if (currentAlg == ALG_FIRE_MELD) algName = "fire_meld";
        else if (currentAlg == ALG_FIRE_FLICKER) algName = "fire_flicker";
        Serial.printf("[%s] angle=%.1f rms=%d energy=%.3f onset=%.3f mic=%s\n",
            algName, angleDeg, rawRms, energy, onset, micOn ? "on" : "off");
    }

    // ── Timeout: breathing idle ──────────────────────────────────
    if (!connected) {
        float breath = (sinf(now / 1000.0f * M_PI) + 1.0f) / 2.0f;
        // Render breathing on pure warm white
        uint16_t bW = (uint16_t)(powf(breath * 0.12f, GAMMA) * 65535.0f);
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
    }

    strip.show();

    // Yield to WiFi task — critical on single-core ESP32-C3
    delay(1);
}
