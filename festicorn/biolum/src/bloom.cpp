/*
 * BIOLUM BLOOM RECEIVER — classic ESP32, 4 WS2812B strips, 8 colonies
 *
 * Bioluminescence/quiet_bloom algorithm ported from rgb_bulbs/bulb_receiver.cpp.
 * Each strip is split in half → 2 colonies per pin × 4 pins = 8 colonies total.
 * All colonies receive the same motion input; each has independent drain/recovery.
 *
 * Receives SensorPacket via ESP-NOW from the festicorn handheld sender (broadcast).
 *
 * Wiring (NEO_RGB):
 *   GPIO 23, 22, 25, 26 — 200 LEDs each (first 100 = colony 2s, next 100 = colony 2s+1)
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
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 30
#endif

static const uint8_t  STRIP_PINS[]       = { 23, 22, 25, 26 };
static const uint8_t  NUM_STRIPS         = sizeof(STRIP_PINS) / sizeof(STRIP_PINS[0]);
static const uint8_t  COLONIES_PER_STRIP = 2;
static const uint8_t  NUM_COLONIES       = NUM_STRIPS * COLONIES_PER_STRIP;  // 8
static const uint16_t LEDS_PER_COLONY    = LEDS_PER_STRIP / COLONIES_PER_STRIP;
static const uint16_t TOTAL_LEDS         = (uint16_t)NUM_STRIPS * LEDS_PER_STRIP;

// LED → colony mapping: strip s, local li → colony (s * 2) + (li >= LEDS_PER_COLONY ? 1 : 0)
static inline uint8_t ledColony(uint8_t s, uint16_t li) {
    return (uint8_t)(s * COLONIES_PER_STRIP + (li / LEDS_PER_COLONY));
}

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

// ── LED driver: 4 strips via RMT, one channel per strip ──
// Each strip has its own RMT channel; Show() fires the kick non-blocking.
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP, 23);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 22);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 25);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP, 26);

// Refresh editing-buffer pointers each frame (NeoPixelBus swaps buffers on Show()).
static RgbColor* stripPixels[NUM_STRIPS];
static inline void refreshStripPointers() {
    // NeoRgbFeature stores pixels as packed R,G,B bytes — same layout as RgbColor.
    stripPixels[0] = (RgbColor*)strip0.Pixels(); stripPixels[1] = (RgbColor*)strip1.Pixels();
    stripPixels[2] = (RgbColor*)strip2.Pixels(); stripPixels[3] = (RgbColor*)strip3.Pixels();
}

static inline void clearAllStrips() {
    RgbColor off(0, 0, 0);
    strip0.ClearTo(off); strip1.ClearTo(off); strip2.ClearTo(off); strip3.ClearTo(off);
}

static inline void fillAllStrips(uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    strip0.ClearTo(c); strip1.ClearTo(c); strip2.ClearTo(c); strip3.ClearTo(c);
}

// Show() on the RMT method kicks off DMA and returns once previous
// transmission has finished. Calling all 4 back-to-back means the 4 RMT
// channels transmit in parallel (~6ms wire time at 200 LEDs).
static volatile uint32_t g_stripShowUs[NUM_STRIPS] = {0};

static inline void showAllStrips() {
    // Direct buffer writes via stripPixels[] don't set the dirty flag —
    // mark all strips dirty before Show() so RMT actually transmits.
    strip0.Dirty(); strip1.Dirty(); strip2.Dirty(); strip3.Dirty();
    // Show(false) skips the front/back buffer memcpy. We rewrite every pixel
    // every frame, so we don't need GetPixelColor to return last-shown values.
    uint32_t t;
    t = micros(); strip0.Show(false); g_stripShowUs[0] = micros() - t;
    t = micros(); strip1.Show(false); g_stripShowUs[1] = micros() - t;
    t = micros(); strip2.Show(false); g_stripShowUs[2] = micros() - t;
    t = micros(); strip3.Show(false); g_stripShowUs[3] = micros() - t;
}

// Per-LED state (TOTAL_LEDS sized — strip s, local i → s*LEDS_PER_STRIP + i)
static uint16_t dsR[TOTAL_LEDS], dsG[TOTAL_LEDS], dsB[TOTAL_LEDS];
static float bloomBreathPhase[TOTAL_LEDS];
static float bloomBreathRate[TOTAL_LEDS];   // 1/period — precomputed to avoid per-frame divide
static float bloomBreathPeak[TOTAL_LEDS];
static float bloomHueT[TOTAL_LEDS];
static float bloomFlashGlow[TOTAL_LEDS];
static float bloomFlashDecay[TOTAL_LEDS];

// Per-colony state (independent drain/recovery; shared motion input)
static float    bloomColonyEnergy[NUM_COLONIES];
static uint32_t bloomLastMotionMs[NUM_COLONIES];
static float    drainEnvelope[NUM_COLONIES];
static float    bloomHitIntensity[NUM_COLONIES];

// Shared motion-detector state (input identical for all colonies)
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

// ── Render one frame (writes to all strip pixel buffers) ────────
static void renderQuietBloom(float dt, uint32_t now) {
    // Per-colony pre-pass: drain bookkeeping and recovery
    bool colonyDraining[NUM_COLONIES];
    for (uint8_t c = 0; c < NUM_COLONIES; c++) {
        colonyDraining[c] = drainEnvelope[c] > 0.001f;
        if (!colonyDraining[c]) bloomHitIntensity[c] = 0.0f;
        if (now - bloomLastMotionMs[c] > MOTION_SETTLE_MS) {
            bloomColonyEnergy[c] = fminf(
                1.0f, bloomColonyEnergy[c] + BLOOM_RECOVERY_RAMP * dt);
        }
    }

    // Hoist time-dependent purple drift base out of per-LED loop
    const float purpleTimeBase = (float)now * (0.001f * BLOOM_PURPLE_RATE);
    const float decayExp = dt * 30.0f;

    refreshStripPointers();

    // Outer loop: each strip is split into 2 colony segments.
    // We iterate per-segment so colony state is hoisted out of the inner loop.
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

    // Per-colony drain post-pass (so flash uses pre-drain energy)
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

// ── Dual-core handshake ──────────────────────────────────────────
// Render runs on core 0, show on core 1 (where the RMT ISRs were registered
// in setup). Notifications: render → "renderDone" → show; show → "bufFree" → render.
// Frame time becomes max(render, show) instead of render + show.
static TaskHandle_t showTaskHandle   = nullptr;  // loop task (core 1)
static TaskHandle_t renderTaskHandle = nullptr;  // render task (core 0)
static volatile uint32_t g_renderUsLast = 0;
static volatile uint32_t g_showUsLast   = 0;

static void renderTask(void* /*param*/) {
    uint32_t lastRenderMs = 0;
    for (;;) {
        // Wait for show to release the buffer (initial give made in setup)
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        uint32_t now = millis();
        float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
        if (dt > 0.1f) dt = 0.1f;
        lastRenderMs = now;

        if (pktCount != bloomPrevPktCount) {
            bloomProcessMotion(now);
            bloomPrevPktCount = pktCount;
        }

        uint32_t t0 = micros();
        renderQuietBloom(dt, now);
        g_renderUsLast = micros() - t0;

        xTaskNotifyGive(showTaskHandle);
    }
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.printf("biolum bloom — %u strips × %u LEDs = %u total\n",
                  NUM_STRIPS, LEDS_PER_STRIP, TOTAL_LEDS);

    strip0.Begin(); strip1.Begin(); strip2.Begin(); strip3.Begin();
    clearAllStrips();
    showAllStrips();

    // Per-LED dither + bloom state seed
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

    // Boot flash — brief blue across all strips
    fillAllStrips(0, 0, 40);
    showAllStrips();
    delay(200);
    clearAllStrips();
    showAllStrips();

    WiFi.mode(WIFI_STA);
    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.printf("WiFi ch=%d MAC=%s — waiting for sender...\n",
                  WiFi.channel(), WiFi.macAddress().c_str());

    // ─── Benchmark: Show() scaling by N channels ───────────────────
    // Kick N channels sequentially, all after full settling. Does total = N × kick_time,
    // or does it plateau at wire_time regardless of N?
    Serial.println("[bench] Test — Show() scaling by channel count:");

    // N=1
    for (int rep = 0; rep < 3; rep++) {
        showAllStrips(); delay(15);
        strip0.Dirty();
        uint32_t t0 = micros();
        strip0.Show(false);
        uint32_t dt = micros() - t0;
        Serial.printf("  N=1 rep %d: %uus\n", rep, dt);
    }
    // N=2
    for (int rep = 0; rep < 3; rep++) {
        showAllStrips(); delay(15);
        strip0.Dirty(); strip1.Dirty();
        uint32_t t0 = micros();
        strip0.Show(false); strip1.Show(false);
        uint32_t dt = micros() - t0;
        Serial.printf("  N=2 rep %d: %uus\n", rep, dt);
    }
    // N=4
    for (int rep = 0; rep < 3; rep++) {
        showAllStrips(); delay(15);
        strip0.Dirty(); strip1.Dirty(); strip2.Dirty(); strip3.Dirty();
        uint32_t t0 = micros();
        strip0.Show(false); strip1.Show(false); strip2.Show(false); strip3.Show(false);
        uint32_t dt = micros() - t0;
        Serial.printf("  N=4 rep %d: %uus\n", rep, dt);
    }
    Serial.println("[bench] Done — entering normal frame loop");

    // Spawn render task on core 0 (loop runs on core 1, with the RMT ISRs).
    showTaskHandle = xTaskGetCurrentTaskHandle();
    xTaskCreatePinnedToCore(renderTask, "render", 8192, nullptr, 1,
                            &renderTaskHandle, 0);
    // Kick the first render so the pipeline starts.
    xTaskNotifyGive(renderTaskHandle);
}

// ── Main loop (show task on core 1) ──────────────────────────────
void loop() {
    // Wait for renderTask (core 0) to signal that a fresh frame is ready.
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    uint32_t t0 = micros();
    showAllStrips();
    g_showUsLast = micros() - t0;

    // Release the buffer back to renderTask so it can begin the next frame.
    xTaskNotifyGive(renderTaskHandle);

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
        float meanE = 0.0f;
        uint8_t draining = 0;
        for (uint8_t c = 0; c < NUM_COLONIES; c++) {
            meanE += bloomColonyEnergy[c];
            if (drainEnvelope[c] > 0.001f) draining++;
        }
        meanE /= NUM_COLONIES;
        uint32_t fps = frameCount - lastTelemetryFrames;
        uint32_t avgRender = (uint32_t)(renderUsAccum / fps);
        uint32_t avgShow   = (uint32_t)(showUsAccum / fps);
        lastTelemetryFrames = frameCount;
        lastTelemetryMs = now;
        renderUsAccum = 0;
        showUsAccum = 0;
        Serial.printf("[bloom] %ufps r=%uus s=%uus rate=%.1f meanE=%.3f %s\n",
                      fps, avgRender, avgShow,
                      bloomMotionRate, meanE,
                      connected ? "connected" : "no-sender");
        Serial.printf("  strip us: %u %u %u %u\n",
                      g_stripShowUs[0], g_stripShowUs[1], g_stripShowUs[2], g_stripShowUs[3]);
    }

}
