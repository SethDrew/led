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

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;     // 6 bytes: raw accelerometer (±2g, 16384 = 1g)
    int16_t gx, gy, gz;     // 6 bytes: raw gyroscope (±250°/s, /131 = deg/s)
    uint16_t rawRms;        // 2 bytes: raw audio RMS
    uint8_t micEnabled;     // 1 byte: shake toggle state
};                          // 15 bytes total

// ── Global state ─────────────────────────────────────────────────

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

// IMU raw (accel + gyro)
static int16_t imuAx, imuAy, imuAz;
static int16_t imuGx, imuGy, imuGz;

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
    Wire.requestFrom(MPU_ADDR, 14);  // accel(6) + temp(2) + gyro(6)
    imuAx = (Wire.read() << 8) | Wire.read();
    imuAy = (Wire.read() << 8) | Wire.read();
    imuAz = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read();  // skip temperature
    imuGx = (Wire.read() << 8) | Wire.read();
    imuGy = (Wire.read() << 8) | Wire.read();
    imuGz = (Wire.read() << 8) | Wire.read();
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

    // Seed first IMU reading
    readIMU();

    Serial.println("Sending (raw accel+gyro)!");
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

    // ── Send packet (raw sensor data, no processing) ───────────
    SensorPacket pkt;
    pkt.ax = imuAx;
    pkt.ay = imuAy;
    pkt.az = imuAz;
    pkt.gx = imuGx;
    pkt.gy = imuGy;
    pkt.gz = imuGz;
    pkt.rawRms = micEnabled ? rms : 0;
    pkt.micEnabled = micEnabled ? 1 : 0;

    esp_now_send(broadcastAddr, (uint8_t*)&pkt, sizeof(pkt));

    // ── Debug output ─────────────────────────────────────────────
    frameCount++;
    if (frameCount % 25 == 0) {
        float gyroMag = vecLen((float)imuGx, (float)imuGy, (float)imuGz) / 131.0f;
        Serial.printf("a=%6d,%6d,%6d  g=%.1f°/s  rms=%5d  mic=%s\n",
            imuAx, imuAy, imuAz, gyroMag, rms, micEnabled ? "on" : "off");
    }
}
