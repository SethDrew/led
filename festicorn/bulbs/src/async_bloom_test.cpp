/*
 * BLOOM TEST — Adafruit NeoPixel (blocking, proven signal) + fast_math LUTs
 *
 * Same breathing + flash math as bulb_receiver.cpp (quiet_bloom).
 * Uses Adafruit NeoPixel (known-good SK6812 timing) to isolate
 * whether flicker is signal/library or effect/math.
 *
 * 50 SK6812 RGBW LEDs on GPIO 10.
 * No WiFi/ESP-NOW — pure render loop.
 *
 * Serial commands:
 *   'f' — trigger a flash (simulates motion event)
 *   't' — toggle CSV telemetry (LED 0 pipeline data every frame)
 *   '1' — per-channel noise gate (default, the fix)
 *   '2' — coordinated noise gate (snap all if max < threshold)
 *   '3' — no noise gate (raw delta-sigma)
 */

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include <fast_math.h>
#include <delta_sigma.h>

#define LED_PIN    10
#define LED_COUNT  50

#define GAMMA          2.4f
#define BRIGHTNESS_CAP 0.70f
#define NOISE_GATE     24

// Breathing
#define BREATH_MIN_PERIOD 3.0f
#define BREATH_MAX_PERIOD 8.0f
#define BREATH_MIN_PEAK   0.50f
#define BREATH_MAX_PEAK   1.00f
#define BREATH_FLOOR      0.03f

// Flash
#define FLASH_DECAY_LO    0.96f
#define FLASH_DECAY_HI    0.985f

// Color
#define HUE_A_G  20.0f
#define HUE_A_B  100.0f
#define HUE_B_G  70.0f
#define HUE_B_B  110.0f
#define FLASH_G  150.0f
#define FLASH_B  170.0f
#define W_ONSET  0.5f

// Adafruit NeoPixel: known-good SK6812 RGBW timing (blocking show)
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800);

// Delta-sigma accumulators
static uint16_t dsG[LED_COUNT], dsB[LED_COUNT], dsW[LED_COUNT];

// Per-LED state
static float breathPhase[LED_COUNT];
static float breathPeriod[LED_COUNT];
static float breathPeak[LED_COUNT];
static float hueT[LED_COUNT];
static float flashGlow[LED_COUNT];
static float flashDecay[LED_COUNT];

// PRNG
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

// Per-LED output buffer (for Adafruit: r, g, b, w per pixel)
struct RGBW { uint8_t r, g, b, w; };
static RGBW frameBuf[LED_COUNT];

// FPS tracking
static uint32_t frameCount = 0;
static uint32_t lastFpsMs = 0;

// Telemetry + gate mode
enum GateMode { GATE_PER_CHANNEL = 1, GATE_COORDINATED = 2, GATE_NONE = 3 };
static GateMode gateMode = GATE_PER_CHANNEL;
static bool telemetryOn = false;

static const char* gateModeName(GateMode m) {
    switch (m) {
        case GATE_PER_CHANNEL:  return "per-channel";
        case GATE_COORDINATED:  return "coordinated";
        case GATE_NONE:         return "none";
    }
    return "?";
}

static void printTelemetryHeader() {
    Serial.printf("# gate=%s\n", gateModeName(gateMode));
    Serial.println("ms,dt_us,g,rawG,rawB,rawW,gateG,gateB,gateW,outG,outB,outW");
}

void setup() {
    Serial.begin(460800);
    delay(300);

    strip.begin();
    strip.setBrightness(255);

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    // Seed delta-sigma
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
        dsG[i] = (seed + 64) & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;
    }

    // Randomize per-LED state
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        breathPhase[i] = randFloat();
        breathPeriod[i] = BREATH_MIN_PERIOD + randFloat() * (BREATH_MAX_PERIOD - BREATH_MIN_PERIOD);
        breathPeak[i] = BREATH_MIN_PEAK + randFloat() * (BREATH_MAX_PEAK - BREATH_MIN_PEAK);
        hueT[i] = randFloat();
        flashGlow[i] = 0.0f;
        flashDecay[i] = FLASH_DECAY_LO + randFloat() * (FLASH_DECAY_HI - FLASH_DECAY_LO);
    }

    // Boot flash
    for (uint16_t i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, 0, 30, 40, 0);
    strip.show();
    delay(200);
    strip.clear();
    strip.show();
    delay(100);

    lastFpsMs = millis();
    Serial.println("Bloom pipeline telemetry");
    Serial.println("  50 SK6812 RGBW on GPIO 10");
    Serial.println("  'f' flash  't' telemetry  '1' per-ch gate  '2' coord gate  '3' no gate");
    Serial.printf("  Gate: %s\n", gateModeName(gateMode));
}

void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : 0.033f;
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    // Serial commands
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'f' || c == 'F') {
            for (uint16_t i = 0; i < LED_COUNT; i++) {
                flashGlow[i] = 0.6f + 0.4f * randFloat();
                flashDecay[i] = FLASH_DECAY_LO + randFloat() * (FLASH_DECAY_HI - FLASH_DECAY_LO);
            }
            Serial.println(">> Flash triggered");
        } else if (c == 't' || c == 'T') {
            telemetryOn = !telemetryOn;
            if (telemetryOn) printTelemetryHeader();
            else Serial.println(">> Telemetry OFF");
        } else if (c == '1') {
            gateMode = GATE_PER_CHANNEL;
            Serial.printf(">> Gate: %s\n", gateModeName(gateMode));
            if (telemetryOn) printTelemetryHeader();
        } else if (c == '2') {
            gateMode = GATE_COORDINATED;
            Serial.printf(">> Gate: %s\n", gateModeName(gateMode));
            if (telemetryOn) printTelemetryHeader();
        } else if (c == '3') {
            gateMode = GATE_NONE;
            Serial.printf(">> Gate: %s\n", gateModeName(gateMode));
            if (telemetryOn) printTelemetryHeader();
        }
    }

    // --- Render ---
    float frameFactor = dt * 30.0f;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        // Flash decay
        flashGlow[i] *= fastDecay(flashDecay[i], frameFactor);
        if (flashGlow[i] < 0.005f) flashGlow[i] = 0.0f;

        // Breathing
        breathPhase[i] += dt / breathPeriod[i];
        if (breathPhase[i] >= 1.0f) breathPhase[i] -= 1.0f;

        float breath = fastSinPhase(breathPhase[i]) * 0.5f + 0.5f;
        float breathGlow = BREATH_FLOOR + breath * (breathPeak[i] - BREATH_FLOOR);

        // Composite
        float g = fmaxf(breathGlow, flashGlow[i]);

        // Color
        float flashFrac = (flashGlow[i] > breathGlow) ? 1.0f : 0.0f;
        float h = hueT[i];
        float colG = lerpf(lerpf(HUE_A_G, HUE_B_G, h), FLASH_G, flashFrac);
        float colB = lerpf(lerpf(HUE_A_B, HUE_B_B, h), FLASH_B, flashFrac);

        // Gamma + cap
        float linBright = fastGamma24(g) * BRIGHTNESS_CAP;

        float oG = colG * linBright;
        float oB = colB * linBright;

        // W channel for flash peaks
        float wFrac = clampf((g - W_ONSET) / (1.0f - W_ONSET), 0.0f, 1.0f);
        float oW = wFrac * linBright * 200.0f;

        // 8.8 fixed-point for delta-sigma
        uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

        // Save pre-gate values for telemetry
        uint16_t rawG = tG16, rawB = tB16, rawW = tW16;

        // Apply selected noise gate
        if (gateMode == GATE_PER_CHANNEL) {
            // Per-channel: snap individual channels to 0
            // if they'd dither in the 0↔1 zone (target16 < 256).
            if (tG16 < 256) tG16 = 0;
            if (tB16 < 256) tB16 = 0;
            if (tW16 < 256) tW16 = 0;
        } else if (gateMode == GATE_COORDINATED) {
            // Coordinated: snap ALL to 0 if max < threshold.
            // Misses weak channels when dominant is above threshold.
            uint16_t maxCh = tG16;
            if (tB16 > maxCh) maxCh = tB16;
            if (tW16 > maxCh) maxCh = tW16;
            if (maxCh < 256) { tG16 = 0; tB16 = 0; tW16 = 0; }
        }
        // GATE_NONE: no gate, raw delta-sigma

        uint8_t gc = deltaSigma(dsG[i], tG16);
        uint8_t b  = deltaSigma(dsB[i], tB16);
        uint8_t w  = deltaSigma(dsW[i], tW16);

        // Telemetry: log LED 0 pipeline state
        if (i == 0 && telemetryOn) {
            uint32_t dt_us = (uint32_t)(dt * 1000000.0f);
            Serial.printf("%lu,%lu,%.3f,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                now, dt_us, g, rawG, rawB, rawW, tG16, tB16, tW16, gc, b, w);
        }

        strip.setPixelColor(i, 0, gc, b, w);
    }

    strip.show();
    delay(1);

    // FPS
    frameCount++;
    if (now - lastFpsMs >= 2000) {
        float fps = frameCount * 1000.0f / (now - lastFpsMs);
        Serial.printf("[async] %.0f fps\n", fps);
        frameCount = 0;
        lastFpsMs = now;
    }
}
