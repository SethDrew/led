/*
 * GYRO MIC FADE — ESP32-C3 standalone
 *
 * Single-board: MPU-6050 (I2C) + INMP441 (I2S) + SK6812 RGBW (NeoPixel).
 * No laptop bridge — all signal processing runs on-chip.
 *
 * Tilt angle from rest → OKLCH rainbow hue (white at rest via W channel)
 * Audio energy (log 15dB + sticky floor + asymmetric exp EMA) → brightness (0%–70%)
 * Delta-sigma dithering on all 4 RGBW channels, gamma 2.4
 *
 * Wiring:
 *   GPIO 21 → SCL (MPU-6050)
 *   GPIO 20 → SDA (MPU-6050)
 *   GPIO 6  → SCK (INMP441)
 *   GPIO 5  → WS  (INMP441)
 *   GPIO 0  → SD  (INMP441)
 *   GPIO 1  → LED data (SK6812 RGBW)
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include <oklch_lut.h>
#include <delta_sigma.h>

// ── Pin assignments ──────────────────────────────────────────────
#define LED_PIN    1
#define LED_COUNT  50
#define SDA_PIN    8
#define SCL_PIN    7
#define AD0_PIN    20    // driven LOW at boot to force MPU address 0x68
#define I2S_SCK    6
#define I2S_WS     5
#define I2S_SD     0
#define MPU_ADDR   0x68

// ── I2S config ───────────────────────────────────────────────────
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320    // 20ms at 16kHz — matches IMU cadence
#define DMA_BUF_COUNT 4

// ── Rendering ────────────────────────────────────────────────────
#define GAMMA 2.4f

// ── Signal processing parameters ─────────────────────────────────
// All EMA alphas computed for 50 Hz sensor rate
#define SENSOR_HZ     50.0f
#define SENSOR_MS     20      // millis between sensor reads

#define NORM_TC       5.0f    // seconds, audio mean EMA
#define BRIGHT_TC     5.0f    // seconds, brightness integral
#define ACCEL_TC      0.3f    // seconds, accelerometer smoothing
#define COLOR_TC      0.2f    // seconds, output color smoothing
#define SIGMOID_GAIN  2.0f    // steepness of brightness response

// Brightness range (byte values, go through gamma 2.4 on output)
#define MIN_BRIGHT    0       // off at silence
#define MAX_BRIGHT    178     // ~70%

// Tilt mapping
#define DEADZONE_DEG  8.0f
#define MAX_ANGLE_DEG 90.0f
#define BLEND_RANGE_DEG 15.0f

// ── Global state ─────────────────────────────────────────────────

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800);

// Delta-sigma accumulators per pixel, per channel
static uint16_t dsR[LED_COUNT], dsG[LED_COUNT], dsB[LED_COUNT], dsW[LED_COUNT];

// Current target RGBW + brightness (updated at sensor rate)
static float targetR = 0, targetG = 0, targetB = 0, targetW = 255;
static uint8_t targetBright = MIN_BRIGHT;

// EMA state (all floats, updated at 50 Hz)
static float floorEma = 0;       // noise floor tracking (sticky min EMA)
static float brightEma = 0;      // log-scale + asymmetric exponential EMA
static bool normInitialized = false;

// Smoothed accelerometer (unit vector)
static float smoothAx, smoothAy, smoothAz;
// Rest (calibration) accelerometer vector
static float restAx, restAy, restAz;
static bool calibrated = false;

// Smoothed output RGBW (float for sub-LSB EMA)
static float smoothR = 0, smoothG = 0, smoothB = 0, smoothW = 255;

// EMA alphas (precomputed)
static float alphaNorm, alphaBright, alphaAccel, alphaColor;

// Shake detection (toggles mic on/off)
// Counts direction reversals above 1.5g within a time window
static bool micEnabled = true;
static float gravityEma = 16384.0f;  // tracks gravity baseline (~1g in raw units)
static int8_t lastShakeSign = 0;     // -1, 0, +1
static uint8_t reversalCount = 0;
static uint32_t firstReversalMs = 0;
static uint32_t shakeCooldownUntil = 0;
#define SHAKE_THRESH      6000   // raw units above/below gravity (~0.37g)
#define SHAKE_REVERSALS   3      // direction changes needed
#define SHAKE_WINDOW_MS   800    // must complete within this
#define SHAKE_COOLDOWN_MS 1500   // ignore after trigger

// Timing
static uint32_t lastSensorMs = 0;
static uint32_t frameCount = 0;

// ── Helper functions ─────────────────────────────────────────────

static inline float vecLen(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}

static void vecNormalize(float &x, float &y, float &z) {
    float len = vecLen(x, y, z);
    if (len > 0) { x /= len; y /= len; z /= len; }
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ── I2S setup ────────────────────────────────────────────────────

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

// ── IMU read ─────────────────────────────────────────────────────

static int16_t imuAx, imuAy, imuAz;

void readIMU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6);  // only need accel (6 bytes)
    imuAx = (Wire.read() << 8) | Wire.read();
    imuAy = (Wire.read() << 8) | Wire.read();
    imuAz = (Wire.read() << 8) | Wire.read();
}

// ── Audio peak read ──────────────────────────────────────────────

int32_t readAudioPeak() {
    int32_t audio_buf[DMA_BUF_LEN];
    size_t bytes_read = 0;
    int32_t peak = 0;
    while (i2s_read(I2S_PORT, audio_buf, sizeof(audio_buf), &bytes_read, 0) == ESP_OK && bytes_read > 0) {
        int num = bytes_read / sizeof(int32_t);
        for (int i = 0; i < num; i++) {
            int32_t s = abs(audio_buf[i] >> 8);  // 24-bit from 32-bit left-aligned
            if (s > peak) peak = s;
        }
    }
    return peak;
}

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    // I2S must be initialized BEFORE I2C on ESP32-C3 (known bus conflict)
    setupI2S();

    pinMode(AD0_PIN, OUTPUT);
    digitalWrite(AD0_PIN, LOW);   // force MPU addr to 0x68
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    // Wake MPU-6050
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B);
    Wire.write(0x00);
    Wire.endTransmission(true);
    delay(100);

    // Enable DLPF (accel 44 Hz, gyro 42 Hz)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A);
    Wire.write(0x03);
    Wire.endTransmission(true);

    // Init LED strip
    strip.begin();
    strip.setBrightness(255);

    // Seed delta-sigma accumulators (decorrelate per-pixel dither)
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
        dsR[i] = seed;
        dsG[i] = (seed + 64) & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;
    }

    // Boot flash — brief red
    for (uint16_t i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, 40, 0, 0, 0);
    strip.show();
    delay(200);
    strip.clear();
    strip.show();

    // Precompute EMA alphas
    alphaNorm  = 2.0f / (NORM_TC  * SENSOR_HZ + 1.0f);
    alphaBright = 2.0f / (BRIGHT_TC * SENSOR_HZ + 1.0f);
    alphaAccel = 2.0f / (ACCEL_TC * SENSOR_HZ + 1.0f);
    alphaColor = 2.0f / (COLOR_TC * SENSOR_HZ + 1.0f);

    // ── Accelerometer calibration (2 seconds) ────────────────────
    Serial.println("Calibrating — keep still for 2 seconds...");
    float sumAx = 0, sumAy = 0, sumAz = 0;
    int calSamples = 0;
    uint32_t calStart = millis();

    while (millis() - calStart < 2000) {
        readIMU();
        sumAx += imuAx;
        sumAy += imuAy;
        sumAz += imuAz;
        calSamples++;
        delay(SENSOR_MS);
    }

    restAx = sumAx / calSamples;
    restAy = sumAy / calSamples;
    restAz = sumAz / calSamples;
    vecNormalize(restAx, restAy, restAz);

    smoothAx = restAx;
    smoothAy = restAy;
    smoothAz = restAz;
    calibrated = true;

    Serial.printf("Rest vector: (%.4f, %.4f, %.4f)\n", restAx, restAy, restAz);
    Serial.println("Running!");

    lastSensorMs = millis();
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t now = millis();

    // ── Sensor update at 50 Hz ───────────────────────────────────
    if (now - lastSensorMs >= SENSOR_MS) {
        lastSensorMs = now;

        readIMU();
        int32_t audioPeak = readAudioPeak();

        // ── Shake detection (direction-reversal counting) ─────────
        float rawMag = vecLen((float)imuAx, (float)imuAy, (float)imuAz);
        gravityEma += 0.01f * (rawMag - gravityEma);  // slow-track gravity
        float dynamic = rawMag - gravityEma;

        if (now > shakeCooldownUntil) {
            if (fabsf(dynamic) > SHAKE_THRESH) {
                int8_t sign = (dynamic > 0) ? 1 : -1;
                if (sign != lastShakeSign && lastShakeSign != 0) {
                    if (reversalCount == 0) firstReversalMs = now;
                    reversalCount++;
                    if (reversalCount >= SHAKE_REVERSALS
                        && (now - firstReversalMs) < SHAKE_WINDOW_MS) {
                        micEnabled = !micEnabled;
                        shakeCooldownUntil = now + SHAKE_COOLDOWN_MS;
                        reversalCount = 0;
                        lastShakeSign = 0;
                        Serial.printf("Mic %s\n", micEnabled ? "ON" : "OFF");
                    }
                }
                lastShakeSign = sign;
            }
        }
        if (reversalCount > 0 && (now - firstReversalMs) > SHAKE_WINDOW_MS) {
            reversalCount = 0;
            lastShakeSign = 0;
        }

        // ── Smooth accelerometer ─────────────────────────────────
        float rawAx = (float)imuAx;
        float rawAy = (float)imuAy;
        float rawAz = (float)imuAz;
        vecNormalize(rawAx, rawAy, rawAz);

        smoothAx += alphaAccel * (rawAx - smoothAx);
        smoothAy += alphaAccel * (rawAy - smoothAy);
        smoothAz += alphaAccel * (rawAz - smoothAz);
        vecNormalize(smoothAx, smoothAy, smoothAz);

        // ── Tilt angle ───────────────────────────────────────────
        float cosAngle = clampf(
            restAx * smoothAx + restAy * smoothAy + restAz * smoothAz,
            -1.0f, 1.0f
        );
        float angleDeg = acosf(cosAngle) * (180.0f / M_PI);

        // ── Tilt → target RGBW ───────────────────────────────────
        float tR, tG, tB, tW;

        if (angleDeg < DEADZONE_DEG) {
            // Pure W channel — warm white LED die
            tR = 0; tG = 0; tB = 0; tW = 255;
        } else {
            // Map angle to hue index
            float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
            if (hueFrac > 1.0f) hueFrac = 1.0f;
            uint8_t hueIdx = (uint8_t)(hueFrac * 255) % 256;

            uint8_t lr = oklchVarL[hueIdx][0];
            uint8_t lg = oklchVarL[hueIdx][1];
            uint8_t lb = oklchVarL[hueIdx][2];

            // Crossfade W → RGB
            float blend = (angleDeg - DEADZONE_DEG) / BLEND_RANGE_DEG;
            if (blend > 1.0f) blend = 1.0f;

            tR = lr * blend;
            tG = lg * blend;
            tB = lb * blend;
            tW = 255.0f * (1.0f - blend);
        }

        // ── Smooth output color (EMA on RGBW) ────────────────────
        smoothR += alphaColor * (tR - smoothR);
        smoothG += alphaColor * (tG - smoothG);
        smoothB += alphaColor * (tB - smoothB);
        smoothW += alphaColor * (tW - smoothW);

        targetR = smoothR;
        targetG = smoothG;
        targetB = smoothB;
        targetW = smoothW;

        // ── Audio → brightness ────────────────────────────────────
        if (micEnabled) {
            float peak = (float)audioPeak;

            if (!normInitialized) {
                floorEma = peak > 0 ? peak : 1.0f;
                normInitialized = true;
            }

            // Floor EMA: fast down (4x), very sticky up (0.1x)
            float alphaFloor = 2.0f / (30.0f * SENSOR_HZ + 1.0f);
            if (peak < floorEma)
                floorEma += alphaFloor * 4.0f * (peak - floorEma);
            else
                floorEma += alphaFloor * 0.1f * (peak - floorEma);

            // Log-scale brightness
            float aboveFloor = peak - floorEma;
            if (aboveFloor < 1.0f) aboveFloor = 1.0f;
            float dB = 20.0f * log10f(aboveFloor / (floorEma > 1.0f ? floorEma : 1.0f));
            float target = clampf(dB / 15.0f, 0.0f, 1.0f);

            // Asymmetric exponential EMA: fast attack (~60ms), slow decay (~5s)
            if (target > brightEma)
                brightEma += 0.3f * (target - brightEma);
            else
                brightEma += 0.004f * (target - brightEma);

            int bright = (int)(MIN_BRIGHT + brightEma * (MAX_BRIGHT - MIN_BRIGHT));
            bright = bright < MIN_BRIGHT ? MIN_BRIGHT : (bright > MAX_BRIGHT ? MAX_BRIGHT : bright);
            targetBright = (uint8_t)bright;
        } else {
            // Mic off: decay brightEma toward 0, output fixed 15%
            brightEma += 0.004f * (0.0f - brightEma);
            targetBright = 38;  // ~15% of 255
        }

        // ── Periodic debug output ────────────────────────────────
        frameCount++;
        if (frameCount % 50 == 0) {
            Serial.printf("angle=%5.1f  bright=%3d  mic=%s  accelMag=%.0f  audio=%ld\n",
                angleDeg, targetBright, micEnabled ? "on" : "off", rawMag, audioPeak);
        }
    }

    // ── Dither + show (runs every loop iteration, ~300 fps) ──────
    float brightLinear = powf((float)targetBright / 255.0f, GAMMA);
    uint8_t cR = (uint8_t)clampf(targetR, 0, 255);
    uint8_t cG = (uint8_t)clampf(targetG, 0, 255);
    uint8_t cB = (uint8_t)clampf(targetB, 0, 255);
    uint8_t cW = (uint8_t)clampf(targetW, 0, 255);

    uint16_t tR16 = (uint16_t)(cR * brightLinear * 256.0f);
    uint16_t tG16 = (uint16_t)(cG * brightLinear * 256.0f);
    uint16_t tB16 = (uint16_t)(cB * brightLinear * 256.0f);
    uint16_t tW16 = (uint16_t)(cW * brightLinear * 256.0f);
    if (tR16 < 64) tR16 = 0;
    if (tG16 < 64) tG16 = 0;
    if (tB16 < 64) tB16 = 0;
    if (tW16 < 64) tW16 = 0;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, r, g, b, w);
    }

    strip.show();
}
