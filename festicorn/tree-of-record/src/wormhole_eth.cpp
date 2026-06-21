/*
 * Wormhole (ethernet-cable LED rig) — 4-strip variant of wormhole.cpp for
 * board led-eth-0 (classic ESP32, MAC 04:B2:47:82:74:20). The "ethernet"
 * here is just the physical cable carrying WS2812 data — NOT networked.
 *
 * 4 × WS2812 strips, 150 LEDs each, on GPIO 13 / 27 / 32 / 33 (RMT 0..3).
 * No sensors, no radio — a quiet standalone ambient piece. Build/flash with
 * env:wormhole-eth.
 *
 * Same slow per-spark sparkle envelopes (ease in over seconds, hold, slower
 * ease out) in a rust family with occasional soft warm whites; delta-sigma
 * dithering smooths the slow sub-8-bit tails.
 *
 * Tuned for a denser look than biolum-b: ~60% of the strip lit at steady
 * state (vs ~41% there). See SPARK_RATE below for the math.
 */

#include <Arduino.h>
#include <NeoPixelBus.h>
#include <esp_random.h>
#include <math.h>
#include <fast_math.h>
#include <delta_sigma.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 150
#endif

static const uint8_t NUM_STRIPS = 4;

// ── Tuning ───────────────────────────────────────────────────────
// Density: SPARK_RATE spawns/sec across all strips. Spawns landing on
// occupied pixels are skipped, so steady-state occupancy fraction solves
//   f = A / (1 + A),  A = SPARK_RATE * meanLifetime / totalPixels.
// Mean lifetime ≈ 2.5 (in) + 0.9 (hold) + 4.5 (out) ≈ 7.9 s.
// totalPixels = 4 * 150 = 600. Target f ≈ 0.60 → A = 1.5 →
//   SPARK_RATE = 1.5 * 600 / 7.9 ≈ 114/s  (≈360 LEDs lit, ~18/m at 30/m).
// (The 6×150 biolum-b build runs 80/s → ~41%.)
#define SPARK_RATE       114.0f
#define SPARK_IN_MIN       1.5f   // fade-in seconds
#define SPARK_IN_MAX       3.5f
#define SPARK_HOLD_MIN     0.4f
#define SPARK_HOLD_MAX     1.4f
#define SPARK_OUT_MIN      3.0f   // fade-out seconds (slower than in)
#define SPARK_OUT_MAX      6.0f
#define SPARK_PEAK_MIN     0.30f  // perceptual peak per spark
#define SPARK_PEAK_MAX     0.605f // 0.605 perceptual ^2.4 ≈ 30% linear
#define WHITE_FRACTION     0.28f  // chance a spark is soft white, not rust
#define WHITE_PEAK_CAP     0.605f // whites match the rust cap
#define MASTER_BRIGHTNESS  1.0f   // global cap, applied in linear domain

// Palette (gamma-domain 0–255). Rust = warm orange-brown; soft white =
// ~2700 K warm white. Eyeball-tune on hardware.
#define RUST_R  190.0f
#define RUST_G   58.0f
#define RUST_B   16.0f
#define SOFT_WHITE_R  255.0f
#define SOFT_WHITE_G  205.0f
#define SOFT_WHITE_B  145.0f

// ── Strips (RMT 0–3) on GPIO 13 / 27 / 32 / 33 ───────────────────
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP, 13);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 27);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 32);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP, 33);

static void setPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
    }
}

// ── Spark state ──────────────────────────────────────────────────
struct Spark {
    float t;        // age in seconds; < 0 → inactive
    float tIn, tHold, tOut;
    float peak;
    float r, g, b;  // gamma-domain color, 0–255
};

static Spark   sparks[NUM_STRIPS][LEDS_PER_STRIP];
static uint16_t ditherR[NUM_STRIPS][LEDS_PER_STRIP];
static uint16_t ditherG[NUM_STRIPS][LEDS_PER_STRIP];
static uint16_t ditherB[NUM_STRIPS][LEDS_PER_STRIP];
static float    spawnDebt = 0.0f;

static uint32_t rngState;
static inline uint32_t xorshift32() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return rngState;
}
static inline float randFloat() {
    return (float)(xorshift32() >> 8) / 16777216.0f;
}
static inline float randRange(float lo, float hi) {
    return lo + (hi - lo) * randFloat();
}

static void igniteSpark(Spark &sp) {
    sp.t     = 0.0f;
    sp.tIn   = randRange(SPARK_IN_MIN, SPARK_IN_MAX);
    sp.tHold = randRange(SPARK_HOLD_MIN, SPARK_HOLD_MAX);
    sp.tOut  = randRange(SPARK_OUT_MIN, SPARK_OUT_MAX);
    if (randFloat() < WHITE_FRACTION) {
        sp.peak = randRange(SPARK_PEAK_MIN, WHITE_PEAK_CAP);
        sp.r = SOFT_WHITE_R;
        sp.g = SOFT_WHITE_G;
        sp.b = SOFT_WHITE_B;
    } else {
        sp.peak = randRange(SPARK_PEAK_MIN, SPARK_PEAK_MAX);
        // Jitter rust hue per spark: drift G toward amber or brown.
        float j = randRange(-14.0f, 22.0f);
        sp.r = RUST_R;
        sp.g = RUST_G + j;
        sp.b = RUST_B + j * 0.2f;
    }
}

static inline float smoothstep01(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x * x * (3.0f - 2.0f * x);
}

// Envelope level 0–1 for a spark at age t; returns false when expired.
static inline bool sparkLevel(const Spark &sp, float &level) {
    float t = sp.t;
    if (t < sp.tIn) { level = smoothstep01(t / sp.tIn); return true; }
    t -= sp.tIn;
    if (t < sp.tHold) { level = 1.0f; return true; }
    t -= sp.tHold;
    if (t < sp.tOut) { level = smoothstep01(1.0f - t / sp.tOut); return true; }
    return false;
}

// ── Frame ────────────────────────────────────────────────────────
static void renderFrame(float dt) {
    // Poisson-ish spawning: accumulate fractional debt, ignite that many.
    spawnDebt += SPARK_RATE * dt;
    while (spawnDebt >= 1.0f) {
        spawnDebt -= 1.0f;
        uint8_t  s = xorshift32() % NUM_STRIPS;
        uint16_t i = xorshift32() % LEDS_PER_STRIP;
        // Occupied pixel → skip; density self-limits instead of restarting
        // mid-fade (a restart reads as a pop at slow timescales).
        if (sparks[s][i].t >= 0.0f) continue;
        igniteSpark(sparks[s][i]);
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            Spark &sp = sparks[s][i];
            float level = 0.0f;
            float colR = RUST_R, colG = RUST_G, colB = RUST_B;
            if (sp.t >= 0.0f) {
                sp.t += dt;
                if (sparkLevel(sp, level)) {
                    colR = sp.r; colG = sp.g; colB = sp.b;
                } else {
                    sp.t = -1.0f;
                    level = 0.0f;
                }
            }

            float bright = level * sp.peak;
            float linSpark = fastGamma24(bright);
            float oR = colR * linSpark * MASTER_BRIGHTNESS;
            float oG = colG * linSpark * MASTER_BRIGHTNESS;
            float oB = colB * linSpark * MASTER_BRIGHTNESS;

            uint16_t t16R = (uint16_t)fminf(oR * 256.0f, 65535.0f);
            uint16_t t16G = (uint16_t)fminf(oG * 256.0f, 65535.0f);
            uint16_t t16B = (uint16_t)fminf(oB * 256.0f, 65535.0f);
            uint8_t r8 = deltaSigma(ditherR[s][i], t16R);
            uint8_t g8 = deltaSigma(ditherG[s][i], t16G);
            uint8_t b8 = deltaSigma(ditherB[s][i], t16B);
            setPixel(s, i, r8, g8, b8);
        }
    }
}

void setup() {
    Serial.begin(115200);
    rngState = esp_random() | 1;

    strip0.Begin(); strip1.Begin(); strip2.Begin(); strip3.Begin();

    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            sparks[s][i].t = -1.0f;

    // Pre-seed mid-fade sparks so power-on doesn't start from black:
    // scatter ignitions at the target occupancy and age them randomly
    // into their envelopes.
    int preSeed = (int)(NUM_STRIPS * LEDS_PER_STRIP * 0.60f);
    for (int n = 0; n < preSeed; n++) {
        uint8_t  s = xorshift32() % NUM_STRIPS;
        uint16_t i = xorshift32() % LEDS_PER_STRIP;
        if (sparks[s][i].t >= 0.0f) continue;
        igniteSpark(sparks[s][i]);
        Spark &sp = sparks[s][i];
        sp.t = randFloat() * (sp.tIn + sp.tHold + sp.tOut) * 0.8f;
    }

    Serial.printf("[wormhole-eth] %u strips x %u LEDs, pins 13/27/32/33\n",
                  NUM_STRIPS, (unsigned)LEDS_PER_STRIP);
}

void loop() {
    static uint32_t lastUs = 0;
    uint32_t nowUs = micros();
    float dt = (lastUs == 0) ? 0.001f : (float)(nowUs - lastUs) * 1e-6f;
    lastUs = nowUs;
    if (dt > 0.1f) dt = 0.1f;

    renderFrame(dt);

    strip0.Show(); strip1.Show(); strip2.Show(); strip3.Show();
}
