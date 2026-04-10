/*
 * RECEIVER_3BULBS — ESP32-C3 bioluminescence controller
 *
 * Receives raw sensor data via ESP-NOW from the handheld sender,
 * renders "quiet bloom" bioluminescence on 3 x SK6812 RGBW sculptures.
 *
 * Input: accelerometer + gyroscope (no mic).
 *
 * Effect: quiet bloom — continuous energy transfer model
 *   - Stillness: all LEDs breathe in blue, full coverage over ~5s
 *   - Movement: immediate flash proportional to motion × remaining colony energy
 *   - Energy pool: colony brightness × 1.4; motion drains this continuously
 *   - Sustained movement: diminishing flashes as energy depletes
 *   - Recovery: LEDs dark then slowly return staggered over 3-10s per LED
 *
 * Wiring (2 physical strips, 3 logical sculptures):
 *   GPIO 21 -> Strip 0: 50 LEDs  -> Sculpture A
 *   GPIO 10 -> Strip 1: 100 LEDs -> Sculpture B (0-49), Sculpture C (50-99)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_random.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include <delta_sigma.h>
#include <fast_math.h>

// -- Physical strips ----------------------------------------------------------
#define PIN_STRIP0      21
#define PIN_STRIP1      10
#define STRIP0_COUNT    50
#define STRIP1_COUNT    100

// -- Logical sculptures -------------------------------------------------------
#define NUM_SCULPTURES     3
#define LEDS_PER_SCULPTURE 50
#define TOTAL_LEDS         (NUM_SCULPTURES * LEDS_PER_SCULPTURE)

// -- Rendering ----------------------------------------------------------------
#define GAMMA          2.4f
#define BRIGHTNESS_CAP 0.70f
#define NOISE_GATE     256      // per-channel: snap to 0 below 1 LSB in 8.8 (kills 0↔1 dither flicker)
#define SENSOR_HZ      25.0f

// -- Movement processing ------------------------------------------------------
#define MOTION_NOISE_FLOOR 12.0f   // deg/s: below this = gyro noise
#define DRAIN_SCALE        800.0f  // motion (deg/s above noise) to drain colony in ~1s at max
#define FLASH_MOTION_SCALE 300.0f  // deg/s for full log-compressed flash brightness
#define ENERGY_MULTIPLIER  1.4f    // flash can exceed stored energy by this factor
#define MOTION_SETTLE_MS   300     // ms of stillness before recovery begins

// -- Breathing (every LED, always on) -----------------------------------------
#define BREATH_MIN_PERIOD 3.0f
#define BREATH_MAX_PERIOD 8.0f
#define BREATH_MIN_PEAK   0.50f
#define BREATH_MAX_PEAK   1.00f
#define BREATH_FLOOR      0.03f

// -- Flash decay --------------------------------------------------------------
#define FLASH_DECAY_LO    0.96f
#define FLASH_DECAY_HI    0.985f

// -- Colony recovery ----------------------------------------------------------
#define RECOVERY_RAMP_RATE    0.10f   // 0→1 in ~10s (first LED wakes ~3s, last ~10s)
#define RECOVERY_WAKE_SPREAD  0.70f   // per-LED wake thresholds span 0..0.7

// -- Color --------------------------------------------------------------------
#define HUE_A_G  20.0f    // deep blue green
#define HUE_A_B  100.0f   // deep blue blue
#define HUE_B_G  70.0f    // teal green
#define HUE_B_B  110.0f   // teal blue

#define FLASH_G  150.0f   // flash cyan green
#define FLASH_B  170.0f   // flash cyan blue

#define W_ONSET  0.5f     // glow above this adds white channel

// -- ESP-NOW ------------------------------------------------------------------
#define TIMEOUT_MS  500

struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;     // raw accelerometer (±2g, 16384 = 1g)
    int16_t gx, gy, gz;     // raw gyroscope (±250°/s, /131 = deg/s)
    uint16_t rawRms;        // raw audio RMS
    uint8_t micEnabled;     // shake toggle
};

// -- Global state -------------------------------------------------------------

static Adafruit_NeoPixel strip0(STRIP0_COUNT, PIN_STRIP0, NEO_GRBW + NEO_KHZ800);
static Adafruit_NeoPixel strip1(STRIP1_COUNT, PIN_STRIP1, NEO_GRBW + NEO_KHZ800);

// Delta-sigma accumulators
static uint16_t dsR[TOTAL_LEDS], dsG[TOTAL_LEDS], dsB[TOTAL_LEDS], dsW[TOTAL_LEDS];

// Per-LED state
static float breathPhase[TOTAL_LEDS];
static float breathPeriod[TOTAL_LEDS];
static float breathPeak[TOTAL_LEDS];
static float hueT[TOTAL_LEDS];          // 0-1 color variation + recovery stagger seed
static float flashGlow[TOTAL_LEDS];     // current flash brightness (decays each frame)
static float flashDecay[TOTAL_LEDS];    // per-LED flash decay rate

// Colony energy: 1.0 = full (all breathing), 0.0 = depleted (dark).
// Motion drains this; flash brightness = drain × multiplier.
// Breathing dims as energy drops (per-LED staggered).
static float colonyEnergy = 1.0f;
static uint32_t lastMotionMs = 0;

// ESP-NOW packet state
static volatile uint32_t lastPacketMs = 0;
static volatile uint32_t pktCount = 0;
static SensorPacket latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

// PRNG
static uint32_t prngState;

// -- ESP-NOW callback ---------------------------------------------------------

void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    pktCount++;
    if (len == sizeof(SensorPacket)) {
        memcpy((void *)&latestPacket, data, sizeof(SensorPacket));
        lastPacketMs = millis();
    }
}

// -- Helpers ------------------------------------------------------------------

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
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

// Map logical (sculpture, led) to physical strip
static inline void setPixel(uint8_t sculpture, uint16_t led,
                            uint8_t r, uint8_t g, uint8_t b, uint8_t w) {
    if (sculpture == 0) {
        strip0.setPixelColor(led, r, g, b, w);
    } else {
        uint16_t pixel = (sculpture == 1) ? led : (LEDS_PER_SCULPTURE + led);
        strip1.setPixelColor(pixel, r, g, b, w);
    }
}

// -- Movement processing ------------------------------------------------------

static float computeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx * fx + fy * fy + fz * fz) / 131.0f;  // ±250°/s range
}

static float computeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx * fx + fy * fy + fz * fz);
    return fabsf(mag - 16384.0f) / 16384.0f;  // 0 = stationary, 1 = ±1g jolt
}

// -- Quiet bloom render -------------------------------------------------------

static void renderQuietBloom(float dt) {

    for (uint8_t s = 0; s < NUM_SCULPTURES; s++) {
        uint16_t base = s * LEDS_PER_SCULPTURE;

        for (uint16_t i = 0; i < LEDS_PER_SCULPTURE; i++) {
            uint16_t idx = base + i;

            // Flash decay (linear approximation of powf for base near 1.0)
            flashGlow[idx] *= fastDecay(flashDecay[idx], dt * 30.0f);
            if (flashGlow[idx] < 0.005f) flashGlow[idx] = 0.0f;

            // Advance breathing phase (wraps continuously)
            breathPhase[idx] += dt / breathPeriod[idx];
            if (breathPhase[idx] >= 1.0f) breathPhase[idx] -= 1.0f;

            // Per-LED colony recovery: staggered wake based on hueT.
            // Each LED has a wake threshold; LEDs with low thresholds
            // return first as colonyEnergy ramps back up.
            float wakeThresh = hueT[idx] * RECOVERY_WAKE_SPREAD;
            float ledRecovery = clampf(
                (colonyEnergy - wakeThresh) / 0.30f,
                0.0f, 1.0f);

            // Breathing glow modulated by colony energy (per-LED staggered)
            float breath = (fastSinPhase(breathPhase[idx]) * 0.5f + 0.5f);
            float breathGlow = BREATH_FLOOR + breath * (breathPeak[idx] - BREATH_FLOOR);
            breathGlow *= ledRecovery;

            // Composite: max of breathing and flash
            float g = fmaxf(breathGlow, flashGlow[idx]);

            // Color: breathing = per-LED warm blue, flash = cooler cyan
            float flashFrac = (flashGlow[idx] > breathGlow) ? 1.0f : 0.0f;
            float h = hueT[idx];
            float colG = lerpf(lerpf(HUE_A_G, HUE_B_G, h), FLASH_G, flashFrac);
            float colB = lerpf(lerpf(HUE_A_B, HUE_B_B, h), FLASH_B, flashFrac);

            // Brightness: gamma + cap
            float linBright = fastGamma24(g) * BRIGHTNESS_CAP;

            float oG = colG * linBright;
            float oB = colB * linBright;

            // W channel for flash peaks
            float wFrac = clampf((g - W_ONSET) / (1.0f - W_ONSET), 0.0f, 1.0f);
            float oW = wFrac * linBright * 200.0f;

            // Scale to 16-bit for delta-sigma
            uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
            uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
            uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

            // Per-channel noise gate: snap individual channels to 0
            // if they'd dither in the 0↔1 zone (target16 < 256)
            if (tG16 < NOISE_GATE) tG16 = 0;
            if (tB16 < NOISE_GATE) tB16 = 0;
            if (tW16 < NOISE_GATE) tW16 = 0;

            uint8_t gc = deltaSigma(dsG[idx], tG16);
            uint8_t b = deltaSigma(dsB[idx], tB16);
            uint8_t w = deltaSigma(dsW[idx], tW16);
            setPixel(s, i, 0, gc, b, w);
        }
    }
}

// -- Setup --------------------------------------------------------------------

void setup() {
    Serial.begin(460800);
    delay(300);

    strip0.begin();
    strip0.setBrightness(255);
    strip1.begin();
    strip1.setBrightness(255);

    // Seed PRNG FIRST — everything below needs randomness
    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    // Seed delta-sigma accumulators
    for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / TOTAL_LEDS);
        dsR[i] = seed;
        dsG[i] = (seed + 64) & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;
    }

    // Init per-LED state: randomize breathing params for organic variation
    for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
        breathPhase[i] = randFloat();
        breathPeriod[i] = BREATH_MIN_PERIOD + randFloat() * (BREATH_MAX_PERIOD - BREATH_MIN_PERIOD);
        breathPeak[i] = BREATH_MIN_PEAK + randFloat() * (BREATH_MAX_PEAK - BREATH_MIN_PEAK);
        hueT[i] = randFloat();
        flashGlow[i] = 0.0f;
        flashDecay[i] = FLASH_DECAY_LO + randFloat() * (FLASH_DECAY_HI - FLASH_DECAY_LO);
    }

    // Boot flash: brief teal
    for (uint16_t i = 0; i < STRIP0_COUNT; i++)
        strip0.setPixelColor(i, 0, 30, 40, 0);
    for (uint16_t i = 0; i < STRIP1_COUNT; i++)
        strip1.setPixelColor(i, 0, 30, 40, 0);
    strip0.show();
    strip1.show();
    delay(200);
    strip0.clear(); strip0.show();
    strip1.clear(); strip1.show();

    // ESP-NOW init
    WiFi.mode(WIFI_STA);
    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.println("receiver_3bulbs: quiet bloom (continuous energy)");
    Serial.printf("Sculptures: %d x %d LEDs | strip0 GPIO %d (%d), strip1 GPIO %d (%d)\n",
        NUM_SCULPTURES, LEDS_PER_SCULPTURE,
        PIN_STRIP0, STRIP0_COUNT, PIN_STRIP1, STRIP1_COUNT);
    Serial.println("Waiting for sender...");
}

// -- Main loop ----------------------------------------------------------------

void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    bool connected = (lastPacketMs > 0) && (now - lastPacketMs < TIMEOUT_MS);

    // Timeout: slow breathing idle
    if (!connected) {
        float breath = (fastSin(now / 2000.0f * M_PI) + 1.0f) / 2.0f;
        uint16_t bB = (uint16_t)(fastGamma24(breath * 0.12f) * 65535.0f);
        for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
            uint8_t b = deltaSigma(dsB[i], bB >> 8);
            uint8_t s = i / LEDS_PER_SCULPTURE;
            uint16_t led = i % LEDS_PER_SCULPTURE;
            setPixel(s, led, 0, 0, b, 0);
        }
        strip0.show();
        strip1.show();
        delay(1);
        return;
    }

    // ── Process new packets (25 Hz) ─────────────────────────────────
    static uint32_t prevPktCount = 0;
    static float motionRate = 0.0f;
    static uint32_t lastPktProcessMs = 0;

    if (pktCount != prevPktCount) {
        float pktDt = (lastPktProcessMs > 0)
            ? (now - lastPktProcessMs) / 1000.0f
            : (1.0f / SENSOR_HZ);
        if (pktDt > 0.2f) pktDt = 0.2f;
        lastPktProcessMs = now;

        // Combine gyro (rotation) + accel jolt (linear movement)
        float gyroRate = computeGyroRate(
            latestPacket.gx, latestPacket.gy, latestPacket.gz);
        float accelJolt = computeAccelJolt(
            latestPacket.ax, latestPacket.ay, latestPacket.az);
        motionRate = gyroRate + accelJolt * 300.0f;

        float motionAboveNoise = fmaxf(0.0f, motionRate - MOTION_NOISE_FLOOR);

        if (motionAboveNoise > 1.0f) {
            lastMotionMs = now;

            // Drain colony energy (linear — vigorous motion drains fast)
            float drain = (motionAboveNoise / DRAIN_SCALE) * pktDt;
            drain = fminf(drain, colonyEnergy);
            colonyEnergy -= drain;

            // Flash: log-compressed motion × remaining energy × multiplier
            // Log compression: gentle (40 deg/s) ≈ 0.65, vigorous (900) ≈ 1.0
            float logMotion = log2f(1.0f + motionAboveNoise)
                            / log2f(1.0f + FLASH_MOTION_SCALE);
            float flashLevel = clampf(logMotion, 0.0f, 1.0f)
                             * colonyEnergy
                             * ENERGY_MULTIPLIER;

            // Apply flash to all LEDs (slight per-LED variation)
            for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
                float target = flashLevel * (0.85f + 0.15f * randFloat());
                if (target > flashGlow[i]) {
                    flashGlow[i] = target;
                    flashDecay[i] = FLASH_DECAY_LO
                        + randFloat() * (FLASH_DECAY_HI - FLASH_DECAY_LO);
                }
            }
        }

        prevPktCount = pktCount;
    }

    // ── Colony recovery: begins after motion settles ─────────────────
    if (now - lastMotionMs > MOTION_SETTLE_MS) {
        colonyEnergy = fminf(1.0f, colonyEnergy + RECOVERY_RAMP_RATE * dt);
    }

    // ── Telemetry (~1s) ─────────────────────────────────────────────
    static uint32_t lastTelemetryMs = 0;
    if (now - lastTelemetryMs >= 1000) {
        lastTelemetryMs = now;
        Serial.printf("[bloom] rate=%.1f  energy=%.3f  flash0=%.3f\n",
            motionRate, colonyEnergy, flashGlow[0]);
    }

    // ── Render ───────────────────────────────────────────────────────
    renderQuietBloom(dt);

    strip0.show();
    strip1.show();

    // FPS reporting (every 2s)
    static uint32_t fpsFrameCount = 0;
    static uint32_t fpsLastMs = 0;
    fpsFrameCount++;
    if (now - fpsLastMs >= 2000) {
        float fps = fpsFrameCount * 1000.0f / (now - fpsLastMs);
        Serial.printf("[bloom] %.0f fps\n", fps);
        fpsFrameCount = 0;
        fpsLastMs = now;
    }

    delay(1);
}
