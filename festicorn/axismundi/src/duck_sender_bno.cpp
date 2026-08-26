/*
 * DUCK SENDER — BNO08x board (primetime replacement for duck_sender.cpp)
 *
 * Same DuckPacketV1 wire contract as duck_sender.cpp: mic RMS + gravity
 * vector, 14 B on ESP-NOW channel 1, 200 Hz inner tick / 25 Hz emit.
 * Receivers cannot tell the two senders apart. Only the IMU layer differs:
 * a BNO08x (SHTP over I2C, fused accel in m/s²) replaces the MPU-6050
 * register reads. Accel is rescaled so 1 g ≈ 32 packet counts, matching the
 * MPU at ±4g — receivers spec ratios-only, but the magnitude matches anyway.
 *
 * Bench BNO08x board (ESP32-C3, MAC 14:63:93:6F:FF:78):
 *   GPIO 2  → SDA (BNO08x)      GPIO 20 → SCK (INMP441)
 *   GPIO 3  → SCL (BNO08x)      GPIO 21 → WS  (INMP441)
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
#include <duck_packet_v1.h>

#define SDA_PIN    2
#define SCL_PIN    3
#define BNO_CS     0
#define BNO_ADD    1
#define BNO_RST    5
#define I2S_SCK    20
#define I2S_WS     21
#define I2S_SD     9

#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320
#define DMA_BUF_COUNT 4

#define TICK_HZ        200
#define TICK_PERIOD_US (1000000 / TICK_HZ)
#define WINDOW_SAMPLES 8               // 8 × 5 ms = 40 ms → 25 Hz emit

// m/s² → MPU-±4g-equivalent counts (8192/g), so finalize's >>8 lands at
// the same ~32 counts per g the MPU duck emits.
#define MS2_TO_COUNTS  (8192.0f / 9.80665f)

#define ACCEL_STALL_MS 1000

#define FIXED_CHANNEL 1

static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct WindowAccum {
    int32_t axSum, aySum, azSum;
    uint8_t filled;
    float   rmsMax, rmsSum;
    uint8_t rmsCount;
};
static WindowAccum win;

// The driver's own hard-reset-in-begin leaves this module handshaking but
// never streaming; RST is pulsed manually in bnoInit as a last resort.
static Adafruit_BNO08x bno(-1);
static sh2_SensorValue_t ev;

static int16_t  imuAx, imuAy, imuAz;
static bool     imuPresent = false;
static uint32_t lastAccelMs = 0;
static uint8_t  seqCounter = 0;
static uint32_t windowsSent = 0;
static uint32_t imuFails = 0;
static uint32_t bnoResets = 0;
static uint32_t sendErrs = 0;

static inline int8_t clampI8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline int16_t clampI16(float v) {
    if (v >  32767.0f) return  32767;
    if (v < -32768.0f) return -32768;
    return (int16_t)lrintf(v);
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

static bool enableAccelReport() {
    return bno.enableReport(SH2_ACCELEROMETER, 10000);   // 100 Hz, m/s²
}

static bool bnoBeginEither() {
    return bno.begin_I2C(0x4B, &Wire, 0) || bno.begin_I2C(0x4A, &Wire, 0);
}

static bool bnoInit() {
    Wire.setClock(400000);
    bool ok = bnoBeginEither();
    // Boot-time clock stretching can pass a scan yet kill the SHTP
    // handshake; a 10 kHz handshake always lands.
    if (!ok) {
        Wire.setClock(10000);
        ok = bnoBeginEither();
    }
    // A wedged BNO08x ACKs but won't handshake; only a hard reset clears
    // it. Pulse RST then release to input — the module pullup owns the line.
    if (!ok) {
        bnoResets++;
        pinMode(BNO_RST, OUTPUT);
        digitalWrite(BNO_RST, LOW);
        delay(20);
        pinMode(BNO_RST, INPUT);
        delay(600);
        ok = bnoBeginEither();
    }
    Wire.setClock(400000);
    if (!ok) return false;
    lastAccelMs = millis();
    return enableAccelReport();
}

static void pollBNO() {
    if (!imuPresent) return;
    if (bno.wasReset()) {
        bnoResets++;
        enableAccelReport();
    }
    while (bno.getSensorEvent(&ev)) {
        if (ev.sensorId == SH2_ACCELEROMETER) {
            imuAx = clampI16(ev.un.accelerometer.x * MS2_TO_COUNTS);
            imuAy = clampI16(ev.un.accelerometer.y * MS2_TO_COUNTS);
            imuAz = clampI16(ev.un.accelerometer.z * MS2_TO_COUNTS);
            lastAccelMs = millis();
        }
    }
    // Gravity is always on, so a silent accel stream means the BNO wedged
    // rather than went still — drop to absent and let the re-probe revive it.
    if (millis() - lastAccelMs > ACCEL_STALL_MS) {
        imuFails++;
        imuPresent = false;
    }
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
                      "errs=send/imu/rst=%u/%u/%u  ch=%d\n",
                      pkt.seq,
                      companding_decode(pkt.rms_max,  DUCK_RMS_FS),
                      companding_decode(pkt.rms_mean, DUCK_RMS_FS),
                      pkt.ax, pkt.ay, pkt.az,
                      (unsigned)sendErrs, (unsigned)imuFails, (unsigned)bnoResets,
                      WiFi.channel());
    }

    resetWindow();
}

static void onSent(const uint8_t * /*mac*/, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) sendErrs++;
}

void setup() {
    Serial.begin(460800);
    // Drop output when no host drains CDC — a full TX buffer otherwise
    // blocks printf forever, freezing the sender when nothing is attached.
    Serial.setTxTimeoutMs(0);
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

    Serial.printf("[BOOT] role=duck fw=axismundi-duck-bno MAC=%s ch=%d\n",
                  WiFi.macAddress().c_str(), WiFi.channel());

    // I2S before I2C — known ESP32-C3 bus conflict.
    setupI2S();

    pinMode(BNO_CS, OUTPUT);
    digitalWrite(BNO_CS, HIGH);    // nCS must idle HIGH to stay in I2C mode
    pinMode(BNO_ADD, INPUT);       // module pullup straps the address
    Wire.begin(SDA_PIN, SCL_PIN);

    imuPresent = bnoInit();
    if (imuPresent) {
        Serial.println("IMU: BNO08x ok (fused accel, 100 Hz)");
    } else {
        Serial.println("IMU: NOT FOUND — audio still sent, re-probing in background");
    }

    resetWindow();
    Serial.println("DuckPacketV1 14B, 200 Hz inner / 25 Hz emit");
}

static void sampleTick() {
    static uint32_t nextUs = 0;

    uint32_t now = micros();
    if ((int32_t)(now - nextUs) < 0) return;
    nextUs = now + TICK_PERIOD_US;

    if (imuPresent) {
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
    pollBNO();
    sampleTick();

    // A duck that came up wedged would otherwise stay colorless until
    // someone power-cycles it, so keep looking.
    if (!imuPresent) {
        static uint32_t nextProbeMs = 0;
        if ((int32_t)(millis() - nextProbeMs) >= 0) {
            nextProbeMs = millis() + 2000;
            if (bnoInit()) {
                imuPresent = true;
                Serial.println("IMU: found on re-probe");
            }
        }
    }
}
