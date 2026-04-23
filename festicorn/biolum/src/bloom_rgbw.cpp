/*
 * BIOLUM BLOOM RGBW RECEIVER — classic ESP32, 1 SK6812 RGBW strip, 2 colonies
 *
 * RGBW port of bloom.cpp. Single strip, split into 2 colonies with independent
 * drain/recovery. Brings the latest RGB ideas (purple drift, per-colony state,
 * per-channel noise gate) together with the RGBW-specific W-channel handling
 * from bulbs/bulb_receiver.cpp quiet_bloom.
 *
 * Receives SensorPacket via ESP-NOW from the festicorn handheld sender.
 *
 * Wiring (SK6812 NEO_GRBW):
 *   GPIO 14 (LED_PIN) — 150 LEDs (first 75 = colony 0, next 75 = colony 1)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <delta_sigma.h>
#include <fast_math.h>

// ── Strip layout ─────────────────────────────────────────────────
#ifndef LED_PIN
#define LED_PIN 14
#endif
#ifndef LEDS_TOTAL
#define LEDS_TOTAL 150
#endif

static const uint8_t  COLONIES_PER_STRIP = 2;
static const uint8_t  NUM_COLONIES       = COLONIES_PER_STRIP;
static const uint16_t TOTAL_LEDS         = LEDS_TOTAL;
static const uint16_t LEDS_PER_COLONY    = TOTAL_LEDS / COLONIES_PER_STRIP;

// ── Bloom parameters ────────────────────────────────────────────
#define BLOOM_BRIGHTNESS_CAP_C0  0.10f   // first 75 LEDs
#define BLOOM_BRIGHTNESS_CAP_C1  0.06f   // last 75 LEDs
#define BLOOM_NOISE_GATE       256    // per-channel: snap to 0 below 1 LSB in 8.8

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
#define BLOOM_BREATH_FLOOR_MIN  0.15f       // floor at normal brightness
#define BLOOM_BREATH_FLOOR_MARGIN 0.75f     // <1.0 lets trough clip (some LEDs go dark)
#define BLOOM_B_CHANNEL_MAX     110.0f      // dominant channel used to derive floor

// Floor rises as CAP drops so the dominant B channel stays near the clip
// threshold during the breath cycle. Margin <1.0 means some LEDs dip below
// clip at trough and go fully dark — intentional at low brightness.
static inline float computeBreathFloor(float cap) {
    float gClip = powf(1.0f / (BLOOM_B_CHANNEL_MAX * cap), 1.0f / 2.4f);
    float f = BLOOM_BREATH_FLOOR_MARGIN * gClip;
    return (f > BLOOM_BREATH_FLOOR_MIN) ? f : BLOOM_BREATH_FLOOR_MIN;
}

#define BLOOM_FLASH_DECAY_LO   0.96f
#define BLOOM_FLASH_DECAY_HI   0.985f

#define BLOOM_RECOVERY_RAMP    0.033f
#define BLOOM_RECOVERY_SPREAD  0.70f

// Color (blue/cyan palette + subtle purple drift)
#define BLOOM_HUE_A_G   20.0f
#define BLOOM_HUE_A_B  100.0f
#define BLOOM_HUE_B_G   70.0f
#define BLOOM_HUE_B_B  110.0f
#define BLOOM_FLASH_G  150.0f
#define BLOOM_FLASH_B  170.0f
#define BLOOM_PURPLE_MAX  60.0f
#define BLOOM_PURPLE_RATE 0.15f

// W channel
#define BLOOM_W_ONSET        0.5f   // glow above this adds white channel
#define BLOOM_W_ENERGY_GATE  0.7f   // colony energy threshold for W
#define BLOOM_W_GAIN         200.0f // scales the W channel into 0..255*BRIGHTNESS_CAP

// ── ESP-NOW packet (matches handheld sender) ────────────────────
#define TIMEOUT_MS     500
#define SENSOR_HZ      25.0f

struct __attribute__((packed)) SensorPacket {
    int16_t  ax, ay, az;
    int16_t  gx, gy, gz;
    uint16_t rawRms;
    uint8_t  micEnabled;
};

static volatile uint32_t pktCount = 0;
static volatile uint32_t lastPacketMs = 0;
static SensorPacket latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    pktCount++;
    if (len == sizeof(SensorPacket)) {
        memcpy((void*)&latestPacket, data, sizeof(SensorPacket));
        lastPacketMs = millis();
    }
}

// ── LED driver: single SK6812 RGBW strip via RMT ────────────────
// Use Sk6812 timing (not Ws2812x) — SK6812 has its own NRZ timing spec that
// is close to but not identical to WS2812. The Ws2812xMethod's tighter
// timing produces glitching at the head of SK6812 strips. NeoGrbwFeature
// stores pixels as G,R,B,W; SetPixelColor(RgbwColor) handles the reorder.
static NeoPixelBus<NeoGrbwFeature, NeoEsp32Rmt0Sk6812Method> strip(TOTAL_LEDS, LED_PIN);

// ── Per-LED state (RGBW) ────────────────────────────────────────
static uint16_t dsR[TOTAL_LEDS], dsG[TOTAL_LEDS], dsB[TOTAL_LEDS], dsW[TOTAL_LEDS];
static float bloomBreathPhase[TOTAL_LEDS];
static float bloomBreathRate[TOTAL_LEDS];   // 1/period — precomputed
static float bloomBreathPeak[TOTAL_LEDS];
static float bloomHueT[TOTAL_LEDS];
static float bloomFlashGlow[TOTAL_LEDS];
static float bloomFlashDecay[TOTAL_LEDS];

// ── Per-colony state (independent drain/recovery; shared motion) ─
static float    bloomColonyEnergy[NUM_COLONIES];
static uint32_t bloomLastMotionMs[NUM_COLONIES];
static float    drainEnvelope[NUM_COLONIES];
static float    bloomHitIntensity[NUM_COLONIES];

// ── Shared motion-detector state ────────────────────────────────
static float    bloomMotionRate = 0.0f;
static float    motionEMA = 0.0f;
static uint32_t bloomPrevPktCount = 0;

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

// ── Motion helpers ──────────────────────────────────────────────
static float computeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx * fx + fy * fy + fz * fz) / 131.0f;
}
static float computeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx * fx + fy * fy + fz * fz);
    return fabsf(mag - 16384.0f) / 16384.0f;
}

// ── Bloom motion processing (per packet) ────────────────────────
static void bloomProcessMotion(uint32_t now) {
    float gyroRate  = computeGyroRate(latestPacket.gx, latestPacket.gy, latestPacket.gz);
    float accelJolt = computeAccelJolt(latestPacket.ax, latestPacket.ay, latestPacket.az);
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

// ── Render one frame ────────────────────────────────────────────
static void renderQuietBloom(float dt, uint32_t now) {
    // Per-colony pre-pass
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

    for (uint8_t seg = 0; seg < COLONIES_PER_STRIP; seg++) {
        uint8_t c = seg;
        bool draining = colonyDraining[c];
        float colonyEnergy = bloomColonyEnergy[c];
        float hitIntensity = bloomHitIntensity[c];
        float energyGate = clampf(
            (colonyEnergy - BLOOM_W_ENERGY_GATE) / (1.0f - BLOOM_W_ENERGY_GATE),
            0.0f, 1.0f);
        uint16_t segStart = seg * LEDS_PER_COLONY;
        uint16_t segEnd   = segStart + LEDS_PER_COLONY;
        float colonyCap = (c == 0) ? BLOOM_BRIGHTNESS_CAP_C0 : BLOOM_BRIGHTNESS_CAP_C1;
        float colonyFloor = computeBreathFloor(colonyCap);

        for (uint16_t i = segStart; i < segEnd; i++) {
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
            // Guard against floor > peak inverting the breath direction.
            float effFloor = fminf(colonyFloor, bloomBreathPeak[i] - 0.05f);
            float breathGlow = effFloor + breath * (bloomBreathPeak[i] - effFloor);
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

            // Subtle purple drift on R (slow sin, per-LED phase offset)
            float purplePhase = purpleTimeBase + h * 4.0f;
            float purpleMix = (fastSin(purplePhase) + 1.0f) * 0.5f;
            float colR = purpleMix * BLOOM_PURPLE_MAX;

            float linBright = fastGamma24(g) * colonyCap;
            float oR = colR * linBright;
            float oG = colG * linBright;
            float oB = colB * linBright;

            // W channel: on above glow threshold, gated by colony energy so it
            // stays color-only during recovery; flash bypasses the gate.
            float wFrac = clampf((g - BLOOM_W_ONSET) / (1.0f - BLOOM_W_ONSET),
                                 0.0f, 1.0f);
            float wGate = fmaxf(energyGate, flashFrac);
            float oW = wFrac * wGate * linBright * BLOOM_W_GAIN;

            uint16_t tR16 = (uint16_t)clampf(oR * 256.0f, 0, 65535);
            uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
            uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
            uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

            if (tR16 < BLOOM_NOISE_GATE) tR16 = 0;
            if (tG16 < BLOOM_NOISE_GATE) tG16 = 0;
            if (tB16 < BLOOM_NOISE_GATE) tB16 = 0;
            if (tW16 < BLOOM_NOISE_GATE) tW16 = 0;

            uint8_t r8 = deltaSigma(dsR[i], tR16);
            uint8_t g8 = deltaSigma(dsG[i], tG16);
            uint8_t b8 = deltaSigma(dsB[i], tB16);
            uint8_t w8 = deltaSigma(dsW[i], tW16);
            strip.SetPixelColor(i, RgbwColor(r8, g8, b8, w8));
        }
    }

    // Per-colony drain post-pass (flash uses pre-drain energy)
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

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.printf("biolum bloom RGBW — 1 strip × %u LEDs on GPIO %u (2 colonies)\n",
                  TOTAL_LEDS, LED_PIN);

    strip.Begin();
    strip.ClearTo(RgbwColor(0, 0, 0, 0));
    strip.Show();

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
        dsG[i] = (seed + 64)  & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;

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

    // Boot flash — brief warm white on the W channel
    strip.ClearTo(RgbwColor(0, 0, 0, 40));
    strip.Show();
    delay(200);
    strip.ClearTo(RgbwColor(0, 0, 0, 0));
    strip.Show();

    WiFi.mode(WIFI_STA);
    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.printf("WiFi ch=%d MAC=%s — waiting for sender...\n",
                  WiFi.channel(), WiFi.macAddress().c_str());
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    if (pktCount != bloomPrevPktCount) {
        bloomProcessMotion(now);
        bloomPrevPktCount = pktCount;
    }

    uint32_t t0 = micros();
    renderQuietBloom(dt, now);
    uint32_t renderUs = micros() - t0;

    uint32_t s0 = micros();
    strip.Show();
    uint32_t showUs = micros() - s0;

    // Telemetry (~1Hz)
    static uint32_t frameCount = 0;
    frameCount++;
    static uint64_t renderUsAccum = 0, showUsAccum = 0;
    renderUsAccum += renderUs;
    showUsAccum   += showUs;

    static uint32_t lastTelemetryMs = 0;
    static uint32_t lastTelemetryFrames = 0;
    bool connected = (lastPacketMs > 0) && (now - lastPacketMs < TIMEOUT_MS);
    if (now - lastTelemetryMs >= 1000) {
        float meanE = 0.0f;
        uint8_t draining = 0;
        for (uint8_t c = 0; c < NUM_COLONIES; c++) {
            meanE += bloomColonyEnergy[c];
            if (drainEnvelope[c] > 0.001f) draining++;
        }
        meanE /= NUM_COLONIES;
        uint32_t fps = frameCount - lastTelemetryFrames;
        uint32_t avgRender = fps ? (uint32_t)(renderUsAccum / fps) : 0;
        uint32_t avgShow   = fps ? (uint32_t)(showUsAccum / fps) : 0;
        lastTelemetryFrames = frameCount;
        lastTelemetryMs = now;
        renderUsAccum = 0;
        showUsAccum = 0;
        Serial.printf("[bloom-rgbw] %ufps r=%uus s=%uus rate=%.1f meanE=%.3f drain=%u %s\n",
                      fps, avgRender, avgShow,
                      bloomMotionRate, meanE, draining,
                      connected ? "connected" : "no-sender");
    }
}
