/*
 * SENDER_NS — sender.cpp + WebRTC noise suppression on the ESP32-C3.
 *
 * Identical to sender.cpp except readAudioRMS() pipes each 10ms (160-sample)
 * frame through the legacy WebRTC float NS module before computing RMS.
 * NS aggressiveness fixed at mode 2 ("Aggressive", ~15 dB suppression),
 * which is the closest legacy equivalent to the modern APM kVeryHigh that
 * the Python prototype validated. Mode 3 (kVeryHigh) is NOT exposed by the
 * legacy float API in cpuimage/WebRTC_NS — see lib/webrtc_ns/.
 *
 * I2S DMA buffer is 320 samples = exactly two NS frames per read.
 *
 * ESP32-C3 has no hardware FPU — every float op is libgcc soft-float.
 * Run-time check: a 25Hz packet cadence allows ~40ms wall time per cycle;
 * NS budget is roughly the time to process ~6 frames per second of audio
 * (DMA reads as much as is queued, currently 320 samples = 20ms per pass,
 * so ~50 NS frames/sec). If CPU is the bottleneck, drop to mode 1 (Medium)
 * or skip every other frame — RMS averages anyway.
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <math.h>

extern "C" {
#include "noise_suppression.h"
}

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
#define DMA_BUF_LEN   320          // 20ms = exactly 2 NS frames
#define DMA_BUF_COUNT 4
#define NS_FRAME_LEN  160          // 10ms @ 16kHz — fixed by webrtc NS

// ── Timing ───────────────────────────────────────────────────────
#define SENSOR_MS     40

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

#define CHANNEL_FALLBACK   1
#define CHANNEL_RESCAN_MS  (5UL * 60UL * 1000UL)
static const char* WIFI_SSID_TARGET = "cuteplant";
static uint8_t  currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs     = 0;

static uint8_t scanForSsidChannel(const char* ssid) {
    int n = WiFi.scanNetworks(false, false, true, 120);
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
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    uint16_t rawRms;            // RMS of NS-cleaned audio (still named rawRms
                                // for wire compatibility with the receiver)
    uint8_t micEnabled;
};

// ── Global state ─────────────────────────────────────────────────

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

// ── NS state ─────────────────────────────────────────────────────
static NsHandle* nsHandle = nullptr;

static inline float vecLen(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}

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

void setupNS() {
    nsHandle = WebRtcNs_Create();
    if (!nsHandle) {
        Serial.println("[ns] WebRtcNs_Create failed");
        return;
    }
    if (WebRtcNs_Init(nsHandle, SAMPLE_RATE) != 0) {
        Serial.println("[ns] init failed");
        WebRtcNs_Free(nsHandle);
        nsHandle = nullptr;
        return;
    }
    // Mode 2 = Aggressive (~15 dB). Closest legacy equivalent to kVeryHigh.
    WebRtcNs_set_policy(nsHandle, 2);
    Serial.println("[ns] ready: mode=2 (Aggressive), 16kHz, 10ms frames");
}

void readIMU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14);
    imuAx = (Wire.read() << 8) | Wire.read();
    imuAy = (Wire.read() << 8) | Wire.read();
    imuAz = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read();
    imuGx = (Wire.read() << 8) | Wire.read();
    imuGy = (Wire.read() << 8) | Wire.read();
    imuGz = (Wire.read() << 8) | Wire.read();
}

// Convert one DMA buffer (32-bit I2S, MSB-aligned 24-bit data) to int16
// frames suitable for the NS module, run NS, accumulate clean RMS.
static uint16_t processBufferThroughNS(const int32_t* audio_buf, int num_samples) {
    if (!nsHandle || num_samples < NS_FRAME_LEN) return 0;

    int64_t sum_sq = 0;
    int totalCleaned = 0;

    int frames = num_samples / NS_FRAME_LEN;
    for (int f = 0; f < frames; f++) {
        yield();
        int16_t inFrame[NS_FRAME_LEN];
        int16_t outFrame[NS_FRAME_LEN];

        // Pull 16-bit MSBs out of the 32-bit I2S word. INMP441 is 24-bit
        // left-justified in the upper bits of a 32-bit slot; >>16 gives the
        // top 16 bits which is the appropriate range for WebRTC NS (expects
        // ~int16 PCM). Drops 8 LSBs of dynamic range — fine for voice.
        for (int i = 0; i < NS_FRAME_LEN; i++) {
            int32_t s = audio_buf[f * NS_FRAME_LEN + i] >> 16;
            if (s > 32767) s = 32767;
            else if (s < -32768) s = -32768;
            inFrame[i] = (int16_t)s;
        }

        const int16_t* inBands[1]  = { inFrame };
        int16_t*       outBands[1] = { outFrame };
        WebRtcNs_Analyze(nsHandle, inFrame);
        WebRtcNs_Process(nsHandle, inBands, 1, outBands);

        for (int i = 0; i < NS_FRAME_LEN; i++) {
            int32_t s = outFrame[i];
            sum_sq += (int64_t)s * s;
        }
        totalCleaned += NS_FRAME_LEN;
    }

    if (totalCleaned == 0) return 0;
    return (uint16_t)sqrtf((float)sum_sq / totalCleaned);
}

uint16_t readAudioRMS() {
    static int32_t audio_buf[DMA_BUF_LEN];
    size_t bytes_read = 0;
    int64_t accum_sq = 0;
    int accum_n = 0;

    int passes = 0;
    while (passes < 4
           && i2s_read(I2S_PORT, audio_buf, sizeof(audio_buf), &bytes_read, 0) == ESP_OK
           && bytes_read > 0) {
        int num = bytes_read / sizeof(int32_t);
        uint16_t rms = processBufferThroughNS(audio_buf, num);
        accum_sq += (int64_t)rms * rms * num;
        accum_n  += num;
        passes++;
    }
    if (accum_n == 0) return 0;
    return (uint16_t)sqrtf((float)accum_sq / accum_n);
}

void setup() {
    Serial.begin(460800);
    delay(300);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    uint8_t ch = scanForSsidChannel(WIFI_SSID_TARGET);
    if (ch) Serial.printf("Found '%s' on ch=%u\n", WIFI_SSID_TARGET, ch);
    else    Serial.printf("'%s' not visible — falling back to ch=%u\n",
                          WIFI_SSID_TARGET, CHANNEL_FALLBACK);
    applyChannel(ch ? ch : CHANNEL_FALLBACK);
    lastScanMs = millis();

    esp_now_init();
    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    setupI2S();
    setupNS();

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    for (int attempt = 0; attempt < 40; attempt++) {
        Wire.beginTransmission(MPU_ADDR);
        Wire.write(0x6B);
        Wire.write(0x00);
        Wire.endTransmission(true);
        delay(50);
        Wire.beginTransmission(MPU_ADDR);
        Wire.write(0x75);
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

    Wire.beginTransmission(MPU_ADDR); Wire.write(0x6C); Wire.write(0x00); Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x1A); Wire.write(0x03); Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x1B); Wire.write(0x00); Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x1C); Wire.write(0x00); Wire.endTransmission(true);

    Serial.printf("ESP-NOW ready ch=%u\n", currentChannel);
    Serial.printf("[BOOT] role=duck MAC=%s fw=sender_ns\n",
                  WiFi.macAddress().c_str());
    lastSensorMs = millis();
}

static void handleSerialQuery() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '?') {
            Serial.printf("[BOOT] role=duck MAC=%s fw=sender_ns mic=%d\n",
                WiFi.macAddress().c_str(), micEnabled ? 1 : 0);
        }
    }
}

void loop() {
    uint32_t now = millis();

    handleSerialQuery();

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

    uint32_t t0 = micros();
    readIMU();
    uint16_t rms = readAudioRMS();
    uint32_t dtUs = micros() - t0;

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
    pkt.ax = imuAx; pkt.ay = imuAy; pkt.az = imuAz;
    pkt.gx = imuGx; pkt.gy = imuGy; pkt.gz = imuGz;
    pkt.rawRms = micEnabled ? rms : 0;
    pkt.micEnabled = micEnabled ? 1 : 0;
    esp_now_send(broadcastAddr, (uint8_t*)&pkt, sizeof(pkt));

    frameCount++;
    if (frameCount % 25 == 0) {
        float gyroMag = vecLen((float)imuGx, (float)imuGy, (float)imuGz) / 131.0f;
        Serial.printf("a=%6d,%6d,%6d  g=%.1f°/s  rms=%5d  mic=%d  loop=%luus\n",
            imuAx, imuAy, imuAz, gyroMag, rms, micEnabled ? 1 : 0,
            (unsigned long)dtUs);
    }
}
