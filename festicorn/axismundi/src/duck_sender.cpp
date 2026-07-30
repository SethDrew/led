/*
 * DUCK SENDER (axismundi) — mic RMS + gravity vector, one packet, 25 Hz.
 *
 * ESP32-C3 duck totem (MAC 38:44:BE:45:D9:CC): INMP441 mic + MPU-6050.
 * Broadcasts DuckPacketV1 (14 B) on ESP-NOW channel 1 for axismundi's energy
 * waterfall — mic RMS pours the water, the gravity vector colors it.
 *
 * Distinct from original-duck/src/sender_v[123].cpp (frozen, deployed): those
 * emit the 3-layer v1 telemetry trio gated on the tree-of-record record
 * switch. This one is a single always-on console packet, which is what the
 * axismundi input contract asks for (full self-contained state, gate on
 * size + magic + ver).
 *
 * Duck pinout (ESP32-C3, outlier wiring):
 *   GPIO 20 → SDA (MPU-6050)
 *   GPIO 21 → SCL (MPU-6050)
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
#include <duck_packet_v1.h>

#define SDA_PIN    20
#define SCL_PIN    21
#define I2S_SCK    6
#define I2S_WS     5
#define I2S_SD     0
#define MPU_ADDR   0x68

#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320
#define DMA_BUF_COUNT 4

#define ACCEL_RANGE_G  4
#define ACCEL_AFS_SEL  1

#define IMU_HZ         200
#define IMU_PERIOD_US  (1000000 / IMU_HZ)
#define WINDOW_SAMPLES 8               // 8 × 5 ms = 40 ms → 25 Hz emit

#define FIXED_CHANNEL 1

static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct WindowAccum {
    int32_t axSum, aySum, azSum;
    uint8_t filled;
    float   rmsMax, rmsSum;
    uint8_t rmsCount;
};
static WindowAccum win;

static int16_t  imuAx, imuAy, imuAz;
static bool     imuPresent = false;
static uint8_t  seqCounter = 0;
static uint32_t windowsSent = 0;
static uint32_t imuFails = 0;
static uint32_t i2cResets = 0;
static uint32_t sendErrs = 0;

static inline int8_t clampI8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline uint8_t companding_encode(float val, float fs) {
    if (val <= 0.0f) return 0;
    float r = val / fs;
    if (r > 1.0f) r = 1.0f;
    int b = (int)(sqrtf(r) * 255.0f + 0.5f);
    return (uint8_t)(b > 255 ? 255 : b);
}

static inline float companding_decode(uint8_t b, float fs) {
    float r = (float)b / 255.0f;
    return r * r * fs;
}

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
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
    i2s_set_pin(I2S_PORT, &pins);
}

static float readAudioRMS() {
    int32_t buf[DMA_BUF_LEN];
    size_t bytesRead = 0;
    int64_t sumSq = 0;
    int samples = 0;
    while (i2s_read(I2S_PORT, buf, sizeof(buf), &bytesRead, 0) == ESP_OK && bytesRead > 0) {
        int n = bytesRead / sizeof(int32_t);
        for (int i = 0; i < n; i++) {
            int32_t s = buf[i] >> 8;   // 24-bit
            sumSq += (int64_t)s * s;
        }
        samples += n;
    }
    if (samples == 0) return 0.0f;
    return sqrtf((float)sumSq / samples);
}

static void initMPU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); Wire.write(0x00);
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6C); Wire.write(0x00);
    Wire.endTransmission(true);
    // DLPF 0x01 = accel 184 Hz BW, required for 200 Hz sampling.
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A); Wire.write(0x01);
    Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C); Wire.write(ACCEL_AFS_SEL << 3);
    Wire.endTransmission(true);
}

// A reset that lands mid-read leaves the MPU driving SDA low forever and the
// probe then fails until the duck is power-cycled. Clock the slave out of that
// byte by hand before touching Wire.
static void i2cBusRecover() {
    Wire.end();
    pinMode(SDA_PIN, INPUT_PULLUP);
    pinMode(SCL_PIN, OUTPUT);
    for (int i = 0; i < 9 && digitalRead(SDA_PIN) == LOW; i++) {
        digitalWrite(SCL_PIN, LOW);  delayMicroseconds(5);
        digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
    }
    pinMode(SDA_PIN, OUTPUT);
    digitalWrite(SDA_PIN, LOW);  delayMicroseconds(5);
    digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
    digitalWrite(SDA_PIN, HIGH); delayMicroseconds(5);
    pinMode(SDA_PIN, INPUT_PULLUP);
    pinMode(SCL_PIN, INPUT_PULLUP);
}

static bool probeMPU() {
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    Wire.beginTransmission(MPU_ADDR);
    return Wire.endTransmission() == 0;
}

static void resetI2C() {
    i2cBusRecover();
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    initMPU();
    i2cResets++;
}

static bool readIMU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom(MPU_ADDR, 6) != 6) return false;
    imuAx = (Wire.read() << 8) | Wire.read();
    imuAy = (Wire.read() << 8) | Wire.read();
    imuAz = (Wire.read() << 8) | Wire.read();
    // Gravity is always on, so an all-zero triple means the MPU is ACKing but
    // wedged (asleep / mid-reset) rather than reporting an orientation.
    return !(imuAx == 0 && imuAy == 0 && imuAz == 0);
}

static void resetWindow() {
    memset(&win, 0, sizeof(win));
}

static void finalizeAndEmit() {
    DuckPacketV1 pkt;
    pkt.magic = DUCK_MAGIC;
    pkt.ver   = DUCK_VER;
    pkt.seq   = seqCounter++;
    pkt.flags = imuPresent ? 0 : DUCK_FLAG_NO_IMU;
    pkt.upMs  = millis();

    if (imuPresent) {
        pkt.ax = clampI8((int)(win.axSum / WINDOW_SAMPLES) >> 8);
        pkt.ay = clampI8((int)(win.aySum / WINDOW_SAMPLES) >> 8);
        pkt.az = clampI8((int)(win.azSum / WINDOW_SAMPLES) >> 8);
    } else {
        pkt.ax = 0; pkt.ay = 0; pkt.az = 0;
    }

    float rmsMean = (win.rmsCount > 0) ? (win.rmsSum / win.rmsCount) : 0.0f;
    pkt.rms_mean = companding_encode(rmsMean,    DUCK_RMS_FS);
    pkt.rms_max  = companding_encode(win.rmsMax, DUCK_RMS_FS);

    if (esp_now_send(broadcastAddr, (uint8_t*)&pkt, sizeof(pkt)) != ESP_OK) sendErrs++;
    windowsSent++;

    if ((windowsSent % 25) == 0) {
        Serial.printf("seq=%3u  rms(max/mean)=%6.0f/%6.0f  a=%4d,%4d,%4d  "
                      "errs=send/imu/i2c=%u/%u/%u  ch=%d\n",
                      pkt.seq,
                      companding_decode(pkt.rms_max,  DUCK_RMS_FS),
                      companding_decode(pkt.rms_mean, DUCK_RMS_FS),
                      pkt.ax, pkt.ay, pkt.az,
                      (unsigned)sendErrs, (unsigned)imuFails, (unsigned)i2cResets,
                      WiFi.channel());
    }

    resetWindow();
}

static void onSent(const uint8_t * /*mac*/, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) sendErrs++;
}

void setup() {
    Serial.begin(460800);
    delay(300);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

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

    Serial.printf("[BOOT] role=duck fw=axismundi-duck MAC=%s ch=%d\n",
                  WiFi.macAddress().c_str(), WiFi.channel());

    // I2S before I2C — known ESP32-C3 bus conflict.
    setupI2S();

    for (int attempt = 0; attempt < 3 && !imuPresent; attempt++) {
        if (attempt) { i2cBusRecover(); delay(20); }
        imuPresent = probeMPU();
    }
    if (imuPresent) {
        initMPU();
        delay(50);
        readIMU();
        Serial.printf("IMU: ok (accel=±%dg)\n", ACCEL_RANGE_G);
    } else {
        Serial.println("IMU: NOT FOUND — audio still sent, re-probing in background");
    }

    resetWindow();
    Serial.println("DuckPacketV1 14B, 200 Hz inner / 25 Hz emit");
}

static void sampleTick() {
    static uint32_t nextUs = 0;
    static uint8_t  failStreak = 0;

    uint32_t now = micros();
    if ((int32_t)(now - nextUs) < 0) return;
    nextUs = now + IMU_PERIOD_US;

    if (imuPresent) {
        if (!readIMU()) {
            imuFails++;
            if (++failStreak >= 3) { resetI2C(); failStreak = 0; }
            return;
        }
        failStreak = 0;
        win.axSum += imuAx;
        win.aySum += imuAy;
        win.azSum += imuAz;
    }
    win.filled++;

    // Read the mic every tick (8× per 40 ms window) so a transient in one
    // ~5 ms slice pushes rms_max above rms_mean. Skip empty reads (no fresh
    // DMA samples) so they don't drag the mean down.
    float rms = readAudioRMS();
    if (rms > 0.0f) {
        if (rms > win.rmsMax) win.rmsMax = rms;
        win.rmsSum += rms;
        win.rmsCount++;
    }

    if (win.filled >= WINDOW_SAMPLES) finalizeAndEmit();
}

void loop() {
    sampleTick();

    // A duck that came up on a stuck bus would otherwise stay colorless until
    // someone power-cycles it, so keep looking.
    if (!imuPresent) {
        static uint32_t nextProbeMs = 0;
        if ((int32_t)(millis() - nextProbeMs) >= 0) {
            nextProbeMs = millis() + 2000;
            i2cBusRecover();
            if (probeMPU()) {
                imuPresent = true;
                initMPU();
                Serial.println("IMU: found on re-probe");
            }
        }
    }
}
