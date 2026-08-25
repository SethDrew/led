/*
 * SENDER_BNO — duck sensor pipe on a BNO08x board (drop-in for sender.cpp)
 *
 * Same 15-byte SensorPacket, same ESP-NOW channel discovery, same
 * shake-to-toggle-mic gesture as the frozen sender.cpp — so the existing
 * bulb_receiver lights up unchanged. Only the IMU source differs: this
 * reads a BNO08x (on-chip fusion, SHTP over I2C) and emulates MPU-6050 raw
 * units in the packet so the receiver's tuning (16384 = 1g, gyro /131)
 * holds. Accel is the fused SH2_ACCELEROMETER (m/s², gravity included,
 * matching MPU raw semantics).
 *
 * Wiring (this bench board, MAC 14:63:93:6F:FF:78):
 *   GPIO 2  → SDA (BNO08x)      GPIO 21 → WS  (INMP441)
 *   GPIO 3  → SCL (BNO08x)      GPIO 20 → SCK (INMP441)
 *   GPIO 0  → CS  (idle HIGH)   GPIO 9  → SD  (INMP441)
 *   GPIO 1  → ADD (float)       GPIO 5  → RST (wedge recovery)
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <math.h>

// ── Pin assignments ──────────────────────────────────────────────
#ifndef SDA_PIN
#define SDA_PIN 2
#endif
#ifndef SCL_PIN
#define SCL_PIN 3
#endif
#ifndef BNO_CS
#define BNO_CS 0
#endif
#ifndef BNO_ADD
#define BNO_ADD 1
#endif
#ifndef BNO_RST
#define BNO_RST 5
#endif
#define I2S_SCK 20
#define I2S_WS  21
#define I2S_SD  9

// ── I2S config ───────────────────────────────────────────────────
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320
#define DMA_BUF_COUNT 4

// ── Timing ───────────────────────────────────────────────────────
#define SENSOR_MS     40   // 25 Hz

// ── MPU-6050 unit emulation ──────────────────────────────────────
// Receiver assumes 16384 counts = 1g (m/s²/9.80665) and gyro counts /131 = °/s.
static const float ACCEL_MS2_TO_COUNTS = 16384.0f / 9.80665f;   // ≈1670.7
static const float GYRO_RADS_TO_COUNTS = 57.29578f * 131.0f;    // ≈7505.7

static inline int16_t clip16(float v) {
    if (v >  32767.0f) return  32767;
    if (v < -32768.0f) return -32768;
    return (int16_t)lrintf(v);
}

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ── Wi-Fi channel discovery (verbatim from sender.cpp) ───────────
#define CHANNEL_FALLBACK   1
#define CHANNEL_RESCAN_MS  (5UL * 60UL * 1000UL)
static const char* WIFI_SSID_TARGET = "cuteplant";
static uint8_t  currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs     = 0;

static uint8_t scanForSsidChannel(const char* ssid) {
    int n = WiFi.scanNetworks(/*async=*/false, /*show_hidden=*/false,
                              /*passive=*/true, /*max_ms_per_chan=*/120);
    uint8_t found = 0;
    int8_t bestRssi = -127;
    for (int i = 0; i < n; i++) {
        if (WiFi.SSID(i) == ssid) {
            int8_t rssi = WiFi.RSSI(i);
            if (rssi > bestRssi) { bestRssi = rssi; found = WiFi.channel(i); }
        }
    }
    WiFi.scanDelete();
    return found;
}

static void applyChannel(uint8_t ch) {
    if (ch == 0) ch = CHANNEL_FALLBACK;
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    currentChannel = ch;
}

struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;     // 6 bytes: raw accelerometer (±2g, 16384 = 1g)
    int16_t gx, gy, gz;     // 6 bytes: raw gyroscope (±250°/s, /131 = deg/s)
    uint16_t rawRms;        // 2 bytes: raw audio RMS (0 when micEnabled=0)
    uint8_t micEnabled;     // 1 byte: shake toggle state
};                          // 15 bytes total

// ── BNO08x ───────────────────────────────────────────────────────
// The driver's own hard-reset-in-begin leaves this module handshaking but
// never streaming; RST is pulsed manually as a last resort instead.
static Adafruit_BNO08x bno(-1);
static sh2_SensorValue_t ev;
static uint8_t bnoAddr = 0;
static float accMs2X = 0, accMs2Y = 0, accMs2Z = 9.80665f;  // gravity at rest
static float gyrRadX = 0, gyrRadY = 0, gyrRadZ = 0;

// ── Shake detection (sender-local UI gesture) ────────────────────
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

static int16_t imuAx, imuAy, imuAz;
static int16_t imuGx, imuGy, imuGz;

static uint32_t lastSensorMs = 0;
static uint32_t frameCount = 0;

static inline float vecLen(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}

// ── I2S setup (verbatim from sender.cpp) ─────────────────────────
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

// ── BNO bring-up ─────────────────────────────────────────────────
static bool bnoBegin(uint8_t a) {
    if (bno.begin_I2C(a, &Wire, 0)) return true;
    // Boot-time clock stretching can pass a scan yet kill the handshake;
    // 10 kHz always lands.
    Wire.setClock(10000);
    if (bno.begin_I2C(a, &Wire, 0)) return true;
    // A wedged BNO08x ACKs but won't handshake; only a hard reset clears it.
    pinMode(BNO_RST, OUTPUT);
    digitalWrite(BNO_RST, LOW);
    delay(20);
    pinMode(BNO_RST, INPUT);
    delay(600);
    return bno.begin_I2C(a, &Wire, 0);
}

static void setupBNO() {
    pinMode(BNO_CS, OUTPUT);
    digitalWrite(BNO_CS, HIGH);   // nCS must idle HIGH to select I2C mode
    pinMode(BNO_ADD, INPUT);      // module pullup straps the address

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    for (uint8_t a : (const uint8_t[]){0x4B, 0x4A}) {
        if (bnoBegin(a)) { bnoAddr = a; break; }
    }
    if (!bnoAddr) { Serial.println("[bno] not found"); return; }
    Wire.setClock(400000);
    Serial.printf("[bno] up at 0x%02X\n", bnoAddr);

    bno.enableReport(SH2_ACCELEROMETER, 20000);        // 50 Hz, m/s²
    bno.enableReport(SH2_GYROSCOPE_CALIBRATED, 20000); // 50 Hz, rad/s
}

// Drain the SHTP queue; cache the freshest accel + gyro.
static void pollBNO() {
    while (bnoAddr && bno.getSensorEvent(&ev)) {
        switch (ev.sensorId) {
            case SH2_ACCELEROMETER:
                accMs2X = ev.un.accelerometer.x;
                accMs2Y = ev.un.accelerometer.y;
                accMs2Z = ev.un.accelerometer.z;
                break;
            case SH2_GYROSCOPE_CALIBRATED:
                gyrRadX = ev.un.gyroscope.x;
                gyrRadY = ev.un.gyroscope.y;
                gyrRadZ = ev.un.gyroscope.z;
                break;
        }
    }
}

// ── Audio RMS read (verbatim from sender.cpp) ────────────────────
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

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    uint8_t ch = scanForSsidChannel(WIFI_SSID_TARGET);
    if (ch) {
        Serial.printf("Found '%s' on ch=%u\n", WIFI_SSID_TARGET, ch);
    } else {
        Serial.printf("'%s' not visible — falling back to ch=%u\n",
                      WIFI_SSID_TARGET, CHANNEL_FALLBACK);
    }
    applyChannel(ch ? ch : CHANNEL_FALLBACK);
    lastScanMs = millis();

    esp_now_init();
    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    setupI2S();   // I2S before I2C — known ESP32-C3 bus conflict
    setupBNO();

    Serial.printf("ESP-NOW ready ch=%u\n", currentChannel);
    Serial.printf("[BOOT] role=duck MAC=%s fw=sender_bno\n",
                  WiFi.macAddress().c_str());
    lastSensorMs = millis();
}

static void handleSerialQuery() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '?') {
            Serial.printf("[BOOT] role=duck MAC=%s fw=sender_bno mic=%d\n",
                WiFi.macAddress().c_str(), micEnabled ? 1 : 0);
        }
    }
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();

    handleSerialQuery();
    pollBNO();

    if (now - lastScanMs > CHANNEL_RESCAN_MS) {
        uint8_t newCh = scanForSsidChannel(WIFI_SSID_TARGET);
        if (newCh && newCh != currentChannel) {
            Serial.printf("[heal] channel drift %u -> %u\n", currentChannel, newCh);
            applyChannel(newCh);
        }
        lastScanMs = millis();
    }

    if (now - lastSensorMs < SENSOR_MS) return;
    lastSensorMs = now;

    imuAx = clip16(accMs2X * ACCEL_MS2_TO_COUNTS);
    imuAy = clip16(accMs2Y * ACCEL_MS2_TO_COUNTS);
    imuAz = clip16(accMs2Z * ACCEL_MS2_TO_COUNTS);
    imuGx = clip16(gyrRadX * GYRO_RADS_TO_COUNTS);
    imuGy = clip16(gyrRadY * GYRO_RADS_TO_COUNTS);
    imuGz = clip16(gyrRadZ * GYRO_RADS_TO_COUNTS);

    uint16_t rms = readAudioRMS();

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
                    Serial.printf("[shake] mic=%d\n", micEnabled ? 1 : 0);
                }
            }
            lastShakeSign = sign;
        }
    }
    if (reversalCount > 0 && (now - firstReversalMs) > SHAKE_WINDOW_MS) {
        reversalCount = 0;
        lastShakeSign = 0;
    }

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

    frameCount++;
    if (frameCount % 25 == 0) {
        float gyroMag = vecLen((float)imuGx, (float)imuGy, (float)imuGz) / 131.0f;
        Serial.printf("a=%6d,%6d,%6d  g=%.1f°/s  rms=%5d  mic=%d\n",
            imuAx, imuAy, imuAz, gyroMag, rms, micEnabled ? 1 : 0);
    }
}
