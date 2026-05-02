/*
 * SENDER — ESP32-C3 handheld sensor pipe (goes inside the duck)
 *
 * Dumb pipe: reads MPU-6050 (I2C) + INMP441 (I2S), broadcasts a 15-byte
 * SensorPacket via ESP-NOW at 25 Hz. Always transmits — engage/disengage
 * lives on the receiver. Only sender-local UI gesture is shake-to-toggle
 * the mic-enabled flag (rawRms zeroed at TX when disabled).
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
#include <esp_wifi.h>
#include <math.h>

// ── Pin assignments ──────────────────────────────────────────────
#define SDA_PIN     20
#define SCL_PIN     21
#define I2S_SCK     6
#define I2S_WS      5
#define I2S_SD      0
#define MPU_ADDR    0x68

// ── I2S config ───────────────────────────────────────────────────
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320
#define DMA_BUF_COUNT 4

// ── Timing ───────────────────────────────────────────────────────
#define SENSOR_MS     40   // 25 Hz

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ── Wi-Fi channel discovery ──────────────────────────────────────
// Receiver locks its radio to whatever channel SSID `cuteplant` advertises
// on. Sender does the same scan so both ends agree without configuration.
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

// ── Global state ─────────────────────────────────────────────────

// Shake detection (sender-local UI gesture: 3 reversals in 800 ms toggles mic)
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

    // ── Wi-Fi: STA mode + SSID-based channel discovery (no AP join) ──
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

    // ── ESP-NOW init ──────────────────────────────────────────────
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

    // Wake MPU-6050 — retry, since a prior session may have left it in
    // CYCLE mode (low-power wake @5Hz), so most I2C transactions miss the
    // wake window. Pulse the wake write every 50ms for up to 2s until a
    // WHO_AM_I read returns a valid ID.
    for (int attempt = 0; attempt < 40; attempt++) {
        Wire.beginTransmission(MPU_ADDR);
        Wire.write(0x6B);
        Wire.write(0x00);
        Wire.endTransmission(true);
        delay(50);
        Wire.beginTransmission(MPU_ADDR);
        Wire.write(0x75);  // WHO_AM_I
        if (Wire.endTransmission(false) == 0) {
            Wire.requestFrom(MPU_ADDR, (uint8_t)1);
            if (Wire.available()) {
                uint8_t v = Wire.read();
                if (v == 0x68 || v == 0x71) {
                    Serial.printf("MPU awake after %d attempts (WHO=0x%02x)\n",
                                  attempt + 1, v);
                    break;
                }
            }
        }
    }

    // Full register restore. A previous session may have left the MPU in
    // WOM-cycle mode with PWR_MGMT_2=0xC7 (gyro standby + LP_WAKE), and
    // the wake-write above only clears PWR_MGMT_1 — gyro stays standby
    // and reads zero unless we explicitly clear PWR_MGMT_2.
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6C);
    Wire.write(0x00);                  // PWR_MGMT_2: all axes on, no cycle
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A);
    Wire.write(0x03);                  // CONFIG: DLPF (accel 44Hz, gyro 42Hz)
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1B);
    Wire.write(0x00);                  // GYRO_CONFIG: ±250°/s
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C);
    Wire.write(0x00);                  // ACCEL_CONFIG: ±2g, no HPF
    Wire.endTransmission(true);

    Serial.printf("ESP-NOW ready ch=%u\n", currentChannel);
    Serial.printf("[BOOT] role=duck MAC=%s fw=sender\n",
                  WiFi.macAddress().c_str());
    lastSensorMs = millis();
}

// ── Serial query: '?' re-prints the boot banner so a tailer attached
// mid-run can identify which board it's looking at without resetting it.
static void handleSerialQuery() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '?') {
            Serial.printf("[BOOT] role=duck MAC=%s fw=sender mic=%d\n",
                WiFi.macAddress().c_str(), micEnabled ? 1 : 0);
        }
    }
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t now = millis();

    handleSerialQuery();

    // Periodic SSID rescan — heals if the AP moves channel.
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

    readIMU();
    uint16_t rms = readAudioRMS();

    // ── Shake detection (toggles mic) ────────────────────────────
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

    // ── Send packet — always ─────────────────────────────────────
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

    // ── Debug output (1 Hz) ──────────────────────────────────────
    frameCount++;
    if (frameCount % 25 == 0) {
        float gyroMag = vecLen((float)imuGx, (float)imuGy, (float)imuGz) / 131.0f;
        Serial.printf("a=%6d,%6d,%6d  g=%.1f°/s  rms=%5d  mic=%d\n",
            imuAx, imuAy, imuAz, gyroMag, rms, micEnabled ? 1 : 0);
    }
}
