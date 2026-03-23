/*
 * SENDER — ESP32-C3 handheld sensor pipe (goes inside the duck)
 *
 * Reads MPU-6050 (I2C) + INMP441 (I2S), streams raw sensor data
 * to receiver via ESP-NOW at 25 Hz. No calibration, no color logic —
 * all interpretation lives on the receiver.
 *
 * Wiring:
 *   GPIO 21 → SCL (MPU-6050)
 *   GPIO 20 → SDA (MPU-6050)
 *   GPIO 6  → SCK (INMP441)
 *   GPIO 5  → WS  (INMP441)
 *   GPIO 0  → SD  (INMP441)
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <esp_now.h>
#include <math.h>

// ── Pin assignments ──────────────────────────────────────────────
#define SDA_PIN    20
#define SCL_PIN    21
#define I2S_SCK    6
#define I2S_WS     5
#define I2S_SD     0
#define MPU_ADDR   0x68

// ── I2S config ───────────────────────────────────────────────────
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320
#define DMA_BUF_COUNT 4

// ── Timing ───────────────────────────────────────────────────────
#define SENSOR_HZ     25.0f
#define SENSOR_MS     40
#define ACCEL_TC      0.3f

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct __attribute__((packed)) SensorPacket {
    float ax, ay, az;       // 12 bytes: smoothed accel unit vector
    uint16_t rawRms;        // 2 bytes: raw audio RMS, no normalization
    uint8_t micEnabled;     // 1 byte: shake toggle state
};                          // 15 bytes total

// ── Global state ─────────────────────────────────────────────────
static float alphaAccel;

// Smoothed accelerometer (unit vector)
static float smoothAx = 0, smoothAy = 0, smoothAz = 1.0f;

// Shake detection
static bool micEnabled = true;
static float gravityEma = 16384.0f;
static int8_t lastShakeSign = 0;
static uint8_t reversalCount = 0;
static uint32_t firstReversalMs = 0;
static uint32_t shakeCooldownUntil = 0;
#define SHAKE_THRESH      6000
#define SHAKE_REVERSALS   3
#define SHAKE_WINDOW_MS   800
#define SHAKE_COOLDOWN_MS 1500

// IMU raw
static int16_t imuAx, imuAy, imuAz;

// Timing
static uint32_t lastSensorMs = 0;
static uint32_t frameCount = 0;

// ── Helpers ──────────────────────────────────────────────────────

static inline float vecLen(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}

static void vecNormalize(float &x, float &y, float &z) {
    float len = vecLen(x, y, z);
    if (len > 0) { x /= len; y /= len; z /= len; }
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

void readIMU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6);
    imuAx = (Wire.read() << 8) | Wire.read();
    imuAy = (Wire.read() << 8) | Wire.read();
    imuAz = (Wire.read() << 8) | Wire.read();
}

// ── Audio RMS read ───────────────────────────────────────────────

uint16_t readAudioRMS() {
    int32_t audio_buf[DMA_BUF_LEN];
    size_t bytes_read = 0;
    int64_t sum_sq = 0;
    int samplesRead = 0;
    while (i2s_read(I2S_PORT, audio_buf, sizeof(audio_buf), &bytes_read, 0) == ESP_OK && bytes_read > 0) {
        int num = bytes_read / sizeof(int32_t);
        for (int i = 0; i < num; i++) {
            int32_t s = audio_buf[i] >> 8;  // 24-bit
            sum_sq += (int64_t)s * s;
        }
        samplesRead += num;
    }
    if (samplesRead == 0) return 0;
    return (uint16_t)sqrtf((float)sum_sq / samplesRead);
}

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    // ── ESP-NOW init ──────────────────────────────────────────────
    WiFi.mode(WIFI_STA);
    esp_now_init();

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    // ── Sensors (I2S before I2C — known ESP32-C3 bus conflict) ────
    setupI2S();

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

    // Precompute EMA alpha
    alphaAccel = 2.0f / (ACCEL_TC * SENSOR_HZ + 1.0f);

    // Seed smoothed accel from first reading
    readIMU();
    smoothAx = (float)imuAx;
    smoothAy = (float)imuAy;
    smoothAz = (float)imuAz;
    vecNormalize(smoothAx, smoothAy, smoothAz);

    Serial.println("Sending!");
    lastSensorMs = millis();
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t now = millis();

    if (now - lastSensorMs < SENSOR_MS) return;
    lastSensorMs = now;

    readIMU();
    uint16_t rms = readAudioRMS();

    // ── Shake detection ──────────────────────────────────────────
    float rawMag = vecLen((float)imuAx, (float)imuAy, (float)imuAz);
    gravityEma += 0.01f * (rawMag - gravityEma);
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
                }
            }
            lastShakeSign = sign;
        }
    }
    if (reversalCount > 0 && (now - firstReversalMs) > SHAKE_WINDOW_MS) {
        reversalCount = 0;
        lastShakeSign = 0;
    }

    // ── Smooth accelerometer ─────────────────────────────────────
    float rawAx = (float)imuAx;
    float rawAy = (float)imuAy;
    float rawAz = (float)imuAz;
    vecNormalize(rawAx, rawAy, rawAz);

    smoothAx += alphaAccel * (rawAx - smoothAx);
    smoothAy += alphaAccel * (rawAy - smoothAy);
    smoothAz += alphaAccel * (rawAz - smoothAz);
    vecNormalize(smoothAx, smoothAy, smoothAz);

    // ── Send packet ──────────────────────────────────────────────
    SensorPacket pkt;
    pkt.ax = smoothAx;
    pkt.ay = smoothAy;
    pkt.az = smoothAz;
    pkt.rawRms = micEnabled ? rms : 0;
    pkt.micEnabled = micEnabled ? 1 : 0;

    esp_now_send(broadcastAddr, (uint8_t*)&pkt, sizeof(pkt));

    // ── Debug output ─────────────────────────────────────────────
    frameCount++;
    if (frameCount % 25 == 0) {
        Serial.printf("ax=%.3f ay=%.3f az=%.3f  rms=%5d  mic=%s\n",
            smoothAx, smoothAy, smoothAz, rms, micEnabled ? "on" : "off");
    }
}
