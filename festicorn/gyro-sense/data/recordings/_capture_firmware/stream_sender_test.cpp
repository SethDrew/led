/*
 * STREAM SENDER — ESP32-C3 streams raw IMU + audio RMS over ESP-NOW.
 *
 * Parallel approach to sender_recorder.cpp's LittleFS pipeline. Pure RAM +
 * radio: poll IMU at 200 Hz, latch audio RMS at the I2S DMA rate (~100 Hz),
 * batch into ESP-NOW broadcast packets. Bridge ESP32 receives + forwards
 * over USB serial to host.
 *
 * Packet layout (max 250 B ESP-NOW payload):
 *   StreamHeader (8 B): magic 'STR1', uint16 seq, uint8 sample_count, uint8 cfg
 *     cfg low nibble  = AFS_SEL (0=±2g,  1=±4g,  2=±8g,  3=±16g)
 *     cfg high nibble = FS_SEL  (0=±250, 1=±500, 2=±1000, 3=±2000 °/s)
 *     legacy senders write 0 → consumers treat as ±2g / ±250°/s
 *   Sample[N]   (14 B each): int16 ax,ay,az,gx,gy,gz; uint16 rms
 *   At N=17 → 8 + 238 = 246 B per packet, ~85 ms cadence at 200 Hz.
 *
 * Hardware: ESP32-C3 super-mini, MPU-6050/clone (I2C), INMP441 (I2S SCK=6 WS=5 SD=0).
 *           Canonical wiring: SDA=8, SCL=7, AD0=20 (driven LOW for 0x68).
 *           IMU full-scale ranges configurable via build flags
 *           ACCEL_RANGE_G / GYRO_RANGE_DPS (default 2 / 250).
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <math.h>

// ── Pins ─────────────────────────────────────────────────────────
#ifndef SDA_PIN
#define SDA_PIN  8
#endif
#ifndef SCL_PIN
#define SCL_PIN  7
#endif
#ifndef AD0_PIN
#define AD0_PIN  20    // driven LOW at boot to force MPU address 0x68
#endif
#define I2S_SCK  6
#define I2S_WS   5
#define I2S_SD   0
#define MPU_ADDR 0x68

// ── IMU full-scale range (build-time) ────────────────────────────
#ifndef ACCEL_RANGE_G
#define ACCEL_RANGE_G 2
#endif
#ifndef GYRO_RANGE_DPS
#define GYRO_RANGE_DPS 250
#endif

#if   ACCEL_RANGE_G == 2
  #define ACCEL_AFS_SEL 0
#elif ACCEL_RANGE_G == 4
  #define ACCEL_AFS_SEL 1
#elif ACCEL_RANGE_G == 8
  #define ACCEL_AFS_SEL 2
#elif ACCEL_RANGE_G == 16
  #define ACCEL_AFS_SEL 3
#else
  #error "ACCEL_RANGE_G must be 2, 4, 8, or 16"
#endif

#if   GYRO_RANGE_DPS == 250
  #define GYRO_FS_SEL 0
#elif GYRO_RANGE_DPS == 500
  #define GYRO_FS_SEL 1
#elif GYRO_RANGE_DPS == 1000
  #define GYRO_FS_SEL 2
#elif GYRO_RANGE_DPS == 2000
  #define GYRO_FS_SEL 3
#else
  #error "GYRO_RANGE_DPS must be 250, 500, 1000, or 2000"
#endif

#define HEADER_CFG ((uint8_t)(ACCEL_AFS_SEL | (GYRO_FS_SEL << 4)))

// ── I2S ──────────────────────────────────────────────────────────
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   160      // ~10 ms @ 16 kHz → RMS refresh ~100 Hz
#define DMA_BUF_COUNT 4

// ── Streaming ────────────────────────────────────────────────────
#define IMU_HZ          200
#define IMU_PERIOD_US   (1000000 / IMU_HZ)
#define BATCH_SIZE      17                  // 17 * 14 = 238 B body
#define STREAM_MAGIC    0x31525453          // 'STR1' little-endian

struct __attribute__((packed)) StreamHeader {
    uint32_t magic;
    uint16_t seq;
    uint8_t  sample_count;
    uint8_t  cfg;             // low nibble AFS_SEL, high nibble FS_SEL
};

struct __attribute__((packed)) Sample {
    int16_t  ax, ay, az;
    int16_t  gx, gy, gz;
    uint16_t rms;
};

static_assert(sizeof(StreamHeader) == 8, "header must be 8 B");
static_assert(sizeof(Sample) == 14, "sample must be 14 B");

// Packet buffer: header + BATCH_SIZE samples
#define PACKET_BYTES (sizeof(StreamHeader) + BATCH_SIZE * sizeof(Sample))
static_assert(PACKET_BYTES <= 250, "ESP-NOW max payload is 250 B");

static uint8_t packetBuf[PACKET_BYTES];
static uint8_t batchFill = 0;
static uint16_t seqCounter = 0;

// ── Sensor state ─────────────────────────────────────────────────
static int16_t  imuAx, imuAy, imuAz;
static int16_t  imuGx, imuGy, imuGz;
static uint16_t latchedRms = 0;
static bool     imuPresent = false;
static uint32_t i2cResets  = 0;
static uint32_t imuFails   = 0;
static uint32_t sentPackets = 0;
static uint32_t sendErrs    = 0;

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
#define ESPNOW_CHANNEL 1   // bridge fixes itself to channel 1

// ── I2S setup ────────────────────────────────────────────────────
static void setupI2S() {
    i2s_config_t cfg = {
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
    i2s_pin_config_t pins = {
        .bck_io_num = I2S_SCK,
        .ws_io_num  = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
    i2s_set_pin(I2S_PORT, &pins);
}

// ── Sensor reads ─────────────────────────────────────────────────
static void initMPU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); Wire.write(0x00);                  // wake
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A); Wire.write(0x01);                  // DLPF 188 Hz
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1B); Wire.write(GYRO_FS_SEL << 3);      // gyro full-scale
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C); Wire.write(ACCEL_AFS_SEL << 3);    // accel full-scale
    Wire.endTransmission(true);
}

static void resetI2C() {
    Wire.end();
    delay(1);
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(100000);
    initMPU();
}

static bool readIMU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom(MPU_ADDR, 14) != 14) return false;
    imuAx = (Wire.read() << 8) | Wire.read();
    imuAy = (Wire.read() << 8) | Wire.read();
    imuAz = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read();                // skip temperature
    imuGx = (Wire.read() << 8) | Wire.read();
    imuGy = (Wire.read() << 8) | Wire.read();
    imuGz = (Wire.read() << 8) | Wire.read();
    return true;
}

static uint16_t pollAudioRMS() {
    int32_t buf[DMA_BUF_LEN];
    size_t bytes_read = 0;
    int64_t sum_sq = 0;
    int n = 0;
    while (i2s_read(I2S_PORT, buf, sizeof(buf), &bytes_read, 0) == ESP_OK && bytes_read > 0) {
        int got = bytes_read / sizeof(int32_t);
        for (int i = 0; i < got; i++) {
            int32_t s = buf[i] >> 8;          // 24-bit
            sum_sq += (int64_t)s * s;
        }
        n += got;
    }
    if (!n) return 0;
    return (uint16_t)sqrtf((float)sum_sq / n);
}

// ── ESP-NOW send callback (just for error counting) ──────────────
static void onSent(const uint8_t *mac, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) sendErrs++;
}

// ── Batch / send ─────────────────────────────────────────────────
static void flushBatch() {
    if (batchFill == 0) return;
    StreamHeader hdr = { STREAM_MAGIC, seqCounter++, batchFill, HEADER_CFG };
    memcpy(packetBuf, &hdr, sizeof(hdr));
    size_t bytes = sizeof(StreamHeader) + batchFill * sizeof(Sample);
    esp_now_send(broadcastAddr, packetBuf, bytes);
    sentPackets++;
    batchFill = 0;
}

static void appendSample(const Sample &s) {
    uint8_t *body = packetBuf + sizeof(StreamHeader);
    memcpy(&body[batchFill * sizeof(Sample)], &s, sizeof(Sample));
    batchFill++;
    if (batchFill >= BATCH_SIZE) flushBatch();
}

// ── Status LED ───────────────────────────────────────────────────
#define STATUS_LED_PIN  8
#define STATUS_LED_ON   LOW
#define STATUS_LED_OFF  HIGH

static void updateStatusLed() {
    static uint32_t lastToggle = 0;
    static bool ledOn = false;
    static bool initialized = false;
    if (!initialized) {
        pinMode(STATUS_LED_PIN, OUTPUT);
        digitalWrite(STATUS_LED_PIN, STATUS_LED_OFF);
        initialized = true;
    }
    uint32_t now = millis();
    if (now - lastToggle >= 250) {           // 2 Hz blink while streaming
        lastToggle = now;
        ledOn = !ledOn;
        digitalWrite(STATUS_LED_PIN, ledOn ? STATUS_LED_ON : STATUS_LED_OFF);
    }
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(460800);

    // ── ESP-NOW init ──────────────────────────────────────────────
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);    // C3 Super Mini regulator brownout fix
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        while (1) delay(1000);
    }
    esp_now_register_send_cb(onSent);

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    Serial.printf("Stream sender ch=%d MAC=%s\n",
                  WiFi.channel(), WiFi.macAddress().c_str());

    // ── Sensors (I2S before I2C — known ESP32-C3 bus conflict) ────
#ifndef DISABLE_AUDIO
    setupI2S();
#endif

    pinMode(AD0_PIN, OUTPUT);
    digitalWrite(AD0_PIN, LOW);                  // force MPU addr 0x68
    delay(5);
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(100000);
    Serial.printf("I2C: SDA=%d SCL=%d\n", SDA_PIN, SCL_PIN);

    Wire.beginTransmission(MPU_ADDR);
    if (Wire.endTransmission() == 0) {
        imuPresent = true;
        initMPU();
        delay(50);
        Serial.printf("IMU: ok (accel=±%dg gyro=±%ddps)\n",
                      ACCEL_RANGE_G, GYRO_RANGE_DPS);
    } else {
        Serial.println("IMU: NOT FOUND");
    }
}

// ── Streaming tick (call as fast as possible) ────────────────────
static void streamTick() {
    static uint32_t nextUs = 0;
    uint32_t now = micros();
    if ((int32_t)(now - nextUs) < 0) return;
    nextUs = now + IMU_PERIOD_US;

    if (imuPresent) {
        static uint8_t failStreak = 0;
        if (readIMU()) {
            failStreak = 0;
        } else {
            // Mark sample as bad so analysis can interpolate.
            imuAx = imuAy = imuAz = imuGx = imuGy = imuGz = -1;
            imuFails++;
            if (++failStreak >= 3) {
                resetI2C();
                i2cResets++;
                failStreak = 0;
            }
        }
    }
#ifndef DISABLE_AUDIO
    uint16_t rms = pollAudioRMS();
    if (rms != 0) latchedRms = rms;
#endif

    Sample s = { imuAx, imuAy, imuAz, imuGx, imuGy, imuGz, latchedRms };
    appendSample(s);
}

// ── Loop ─────────────────────────────────────────────────────────
void loop() {
    streamTick();

    // Periodic stats over USB serial (host-side only; no impact on stream)
    static uint32_t lastStatMs = 0;
    uint32_t now = millis();
    if (now - lastStatMs >= 2000) {
        lastStatMs = now;
        Serial.printf("seq=%u pkts=%u sendErr=%u imuFail=%u i2cReset=%u rms=%u\n",
                      (unsigned)seqCounter, (unsigned)sentPackets,
                      (unsigned)sendErrs, (unsigned)imuFails,
                      (unsigned)i2cResets, (unsigned)latchedRms);
    }
    updateStatusLed();
}
