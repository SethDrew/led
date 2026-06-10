/*
 * SENDER V3 — sender_v2 + lo-fi audio WAVEFORM streaming.
 *
 * Identical to sender_v2 (frozen sibling) but additionally streams the decimated
 * 8 kHz/8-bit mic waveform as AudioStreamPacketV1 (204 B) so tree-of-record can
 * record it to a slot and replay it out a 3.5 mm jack.
 *
 * Streaming is gated on the KAYLAB (BS-26) RECORD SWITCH, not the shake-mute.
 * The totem listens to the same BS-26 JSON broadcast the tree hears
 * ({"ac":0|1,...}) and mirrors the tree's rising-edge record toggle locally,
 * bounded to 30 s — so the waveform flows only while a slot is being recorded.
 * The BS-26 is the single source of switch truth; both consumers derive from it.
 *
 * Reads MPU-6050 (I2C accel + gyro) at 200 Hz and INMP441 (I2S mic), then
 * emits three independent layer packets over ESP-NOW broadcast at 25 Hz:
 *
 *   AccelPacketV1  → TelemetryPacketV1 (16 B)  per-axis accel max/min/mean
 *                                              + companded |a| max/mean
 *   GyroPacketV1   → GyroPacketV1     (12 B)   per-axis gyro max/min/mean
 *                                              + companded |gyro| max
 *   AudioPacketV1  → AudioPacketV1     (5 B)   companded RMS max/mean
 *
 * All three share one uint16 seq counter, incremented each 25 Hz tick.
 * The sender does NOT classify — schema is general-purpose telemetry;
 * classification lives on the receiver.
 *
 * IMU windowing + companding ported from festicorn/gyro-sense/src/sender.cpp.
 * I2S mic setup ported from this board's frozen sender.cpp.
 *
 * Duck pinout (ESP32-C3, outlier wiring):
 *   GPIO 20 → SDA (MPU-6050)
 *   GPIO 21 → SCL (MPU-6050)
 *   GPIO 6  → SCK (INMP441)
 *   GPIO 5  → WS  (INMP441)
 *   GPIO 0  → SD  (INMP441)
 *
 * ESP-NOW broadcast on channel 1 (matches tree-of-record FIXED_CHANNEL).
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <Preferences.h>
#include <math.h>
#include "v1_packet.h"
#include "gyro_packet_v1.h"
#include "audio_packet_v1.h"
#include "audio_stream_packet_v1.h"

// ── Pin assignments (duck outlier wiring) ────────────────────────
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

// ── IMU full-scale range (build-time) ────────────────────────────
#ifndef ACCEL_RANGE_G
#define ACCEL_RANGE_G 4
#endif
#ifndef GYRO_RANGE_DPS
#define GYRO_RANGE_DPS 1000
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

// ── Sampling / window cadence ────────────────────────────────────
#define IMU_HZ           200
#define IMU_PERIOD_US    (1000000 / IMU_HZ)   // 5000 µs
#define WINDOW_SAMPLES   8                    // 8 × 5 ms = 40 ms → 25 Hz emit

// Companding full-scales (counts) — matched to ±4g / ±1000 dps rails.
// Encoder: byte = clamp_u8( sqrt(val/FS) * 255 )
// Decoder: val  = (byte/255)² * FS
#define AMAG_FS          57000.0f
#define GMAG_FS          57000.0f

// Audio RMS companding full-scale. RMS is computed from 24-bit samples
// (audio >> 8), so values live in a ~tens-of-thousands to low-hundreds-of-
// thousands domain — NOT int16. Measured INMP441 RMS (bench, this mic):
// silent-room floor ~10k, loud music 30k–146k (mean ~85k), close transient
// spikes toward ~250k. FS=200000 clears loud content with headroom; only
// the very loudest claps approach saturation. Decoder must match (V1_RMS_FS).
#define RMS_FS           200000.0f

// Per-sample axis-clip threshold (≈99.2% of int16 rail).
#define CLIP_THRESH      32500

// ── Audio waveform stream (see audio_stream_packet_v1.h) ─────────
// Same 24-bit (audio>>8) domain as the RMS above. QUANT_SHIFT is the by-ear
// loudness knob (lower = louder/more clipping); tuned separately later.
#define AUDIO_QUANT_SHIFT 11
#define AUDIO_DC_SHIFT    10    // one-pole DC-removal HPF, ~1.2 Hz @ 8 kHz

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Matches tree-of-record's FIXED_CHANNEL.
#define FIXED_CHANNEL 1

// ── Window aggregation state ─────────────────────────────────────
struct WindowAccum {
    int16_t  axMax,  ayMax,  azMax;
    int16_t  axMin,  ayMin,  azMin;
    int32_t  axSum,  aySum,  azSum;
    int16_t  gxMax,  gyMax,  gzMax;
    int16_t  gxMin,  gyMin,  gzMin;
    int32_t  gxSum,  gySum,  gzSum;
    float    amagMax,  amagSum;
    float    gmagMax,  gmagSum;
    uint8_t  accelClips;
    uint8_t  gyroSats;
    uint8_t  filled;
    // Audio RMS over the window (one read per IMU window finalize)
    float    rmsMax,  rmsSum;
    uint8_t  rmsCount;
};
static WindowAccum win;
static uint16_t    seqCounter = 0;

// ── Shake-to-toggle mic mute (sender-local UI gesture) ───────────
// Ported from this board's frozen sender.cpp: 3 reversals of the dynamic
// accel magnitude (raw |a| minus a slow gravity EMA) within 800 ms toggles
// the mic. Sampled off the same 200 Hz accel stream. When muted, the audio
// packet's rms fields are zeroed and AUDIO_FLAG_MIC_MUTED is set so the
// receiver knows audio is alive but intentionally silent. Default: enabled.
#define AUDIO_FLAG_MIC_MUTED 0x01
// Original sender.cpp ran the IMU at ±2g (16384 counts/g) and used a 6000-count
// dynamic threshold. sender_v2 runs at ±ACCEL_RANGE_G, so counts/g scale by
// (2 / ACCEL_RANGE_G). Scale both the gravity seed and the threshold to keep
// the gesture feeling identical regardless of build-time range.
#define ACCEL_COUNTS_PER_G   (32768.0f / (float)ACCEL_RANGE_G)
#define SHAKE_THRESH         (6000.0f * ACCEL_COUNTS_PER_G / 16384.0f)
#define SHAKE_REVERSALS      3
#define SHAKE_WINDOW_MS      800
#define SHAKE_COOLDOWN_MS    1500

static bool     micEnabled        = true;
static float    shakeGravityEma   = ACCEL_COUNTS_PER_G;   // seed ≈ 1g
static int8_t   shakeLastSign     = 0;
static uint8_t  shakeReversals    = 0;
static uint32_t shakeFirstRevMs   = 0;
static uint32_t shakeCooldownUntil = 0;

// IMU read buffer
static int16_t imuAx, imuAy, imuAz;
static int16_t imuGx, imuGy, imuGz;
static bool    imuPresent = false;

// Gyro bias estimated at boot / background. Subtracted before any
// magnitude / saturation use. Accel bias is NOT subtracted (gravity is
// the desired DC component).
static int16_t gyroBias[3] = {0, 0, 0};
static bool    gyroBiasReady = false;

// Diagnostics
static uint32_t windowsSent = 0;
static uint32_t imuFails    = 0;
static uint32_t i2cResets   = 0;
static uint32_t sendErrs    = 0;

// ── Helpers ──────────────────────────────────────────────────────

static inline int8_t clampI8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline uint8_t clampU8(int v) {
    if (v > 255) return 255;
    if (v <   0) return   0;
    return (uint8_t)v;
}

static inline uint8_t companding_encode(float val, float fs) {
    if (val <= 0.0f) return 0;
    float r = val / fs;
    if (r > 1.0f) r = 1.0f;
    return clampU8((int)(sqrtf(r) * 255.0f + 0.5f));
}

static inline float companding_decode(uint8_t byte_val, float fs) {
    float r = (float)byte_val / 255.0f;
    return r * r * fs;
}

// ── I2S setup ────────────────────────────────────────────────────

static void setupI2S() {
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

// ── Audio waveform stream state ──────────────────────────────────
// Streaming mirrors the tree's record state, learned from the BS-26 broadcast.
#define WF_MAX_MS  30000                 // 30 s recording cap (matches the tree)
static AudioStreamPacketV1 audioStreamPkt;
static uint8_t  audioFill   = 0;
static uint16_t audioSeq    = 0;
static int32_t  audioDcEma  = 0;
static int32_t  audioHeld   = 0;
static bool     audioHavePrev = false;
static volatile bool wfStreaming = false;   // true while mirroring a record
static volatile bool wfAcLatch   = false;   // record switch consumed-while-high
static uint32_t wfStartMs   = 0;

static void wfResetPacketizer() {
    audioFill = 0;
    audioHavePrev = false;
}

static void emitAudioStream() {
    audioStreamPkt.seq       = audioSeq++;
    audioStreamPkt.rate_code = AUDIO_STREAM_RATE_8K;
    audioStreamPkt.n         = audioFill;
    if (esp_now_send(broadcastAddr, (uint8_t*)&audioStreamPkt, sizeof(audioStreamPkt)) != ESP_OK)
        sendErrs++;
    audioFill = 0;
}

// BS-26 (kaylab) JSON broadcast listener. We only need the record switch ("ac"),
// a maintained on/off switch read as a LEVEL — the same semantics the tree runs.
// While the switch is high the totem streams effects+waveform; while it's low it
// goes silent. A latch keeps the 30 s cap from restarting until the switch is
// released, so both ends start/stop together.
static void onBsRecv(const uint8_t * /*mac*/, const uint8_t *data, int len) {
    if (len < 6 || data[0] != '{') return;            // JSON only
    char buf[160];
    int n = (len < (int)sizeof(buf) - 1) ? len : (int)sizeof(buf) - 1;
    memcpy(buf, data, n); buf[n] = '\0';
    const char *p = strstr(buf, "\"ac\":");
    if (!p) return;
    p += 5;
    while (*p == ' ') p++;
    bool ac = (*p == '1');
    if (!ac) {                                        // released → re-armed, stop
        wfAcLatch = false;
        wfStreaming = false;
    } else if (!wfStreaming && !wfAcLatch) {          // switch on (not capped) → start
        wfStreaming = true;
        wfAcLatch   = true;
        wfStartMs   = millis();
        wfResetPacketizer();
    }
}

// Drain all pending I2S samples, return the window RMS (for the 5 B envelope),
// and — while mirroring a record — decimate/DC-remove/quantize the waveform
// into AudioStreamPacketV1 packets.
static float readAudioRMS() {
    // 30 s safety cap so a missed "stop" can't stream forever.
    if (wfStreaming && (millis() - wfStartMs) >= WF_MAX_MS) wfStreaming = false;
    if (!wfStreaming) wfResetPacketizer();

    int32_t audio_buf[DMA_BUF_LEN];
    size_t bytes_read = 0;
    int64_t sum_sq = 0;
    int samplesRead = 0;
    while (i2s_read(I2S_PORT, audio_buf, sizeof(audio_buf), &bytes_read, 0) == ESP_OK && bytes_read > 0) {
        int num = bytes_read / sizeof(int32_t);
        for (int i = 0; i < num; i++) {
            int32_t s = audio_buf[i] >> 8;  // 24-bit
            sum_sq += (int64_t)s * s;
            if (wfStreaming) {
                // ÷2 average-decimate 16 kHz → 8 kHz
                if (!audioHavePrev) { audioHeld = s; audioHavePrev = true; }
                else {
                    int32_t avg = (audioHeld + s) >> 1;
                    audioHavePrev = false;
                    audioDcEma += (avg - audioDcEma) >> AUDIO_DC_SHIFT;   // DC removal
                    int32_t ac = avg - audioDcEma;
                    int32_t q = (ac >> AUDIO_QUANT_SHIFT) + 128;          // 8-bit quantize
                    if (q < 0) q = 0; else if (q > 255) q = 255;
                    audioStreamPkt.samples[audioFill++] = (uint8_t)q;
                    if (audioFill >= AUDIO_STREAM_SAMPLES) emitAudioStream();
                }
            }
        }
        samplesRead += num;
    }
    if (samplesRead == 0) return 0.0f;
    return sqrtf((float)sum_sq / samplesRead);
}

// ── MPU init / read ──────────────────────────────────────────────

static void initMPU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); Wire.write(0x00);                  // wake
    Wire.endTransmission(true);
    // PWR_MGMT_2: all axes on, no cycle (clears any prior WOM-cycle state
    // that would leave the gyro in standby reading zero).
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6C); Wire.write(0x00);
    Wire.endTransmission(true);
    // DLPF 0x01 = accel 184 Hz / gyro 188 Hz BW, required for 200 Hz sampling.
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A); Wire.write(0x01);
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
    Wire.setClock(400000);
    initMPU();
    i2cResets++;
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
    if (gyroBiasReady) {
        int32_t cx = (int32_t)imuGx - gyroBias[0];
        int32_t cy = (int32_t)imuGy - gyroBias[1];
        int32_t cz = (int32_t)imuGz - gyroBias[2];
        if (cx >  32767) cx =  32767; else if (cx < -32768) cx = -32768;
        if (cy >  32767) cy =  32767; else if (cy < -32768) cy = -32768;
        if (cz >  32767) cz =  32767; else if (cz < -32768) cz = -32768;
        imuGx = (int16_t)cx;
        imuGy = (int16_t)cy;
        imuGz = (int16_t)cz;
    }
    return true;
}

// ── NVS bias storage ─────────────────────────────────────────────
static Preferences prefs;

static bool loadBiasFromNVS() {
    prefs.begin("gyro", true);
    bool valid = prefs.getBool("valid", false);
    if (valid) {
        gyroBias[0] = prefs.getShort("bx", 0);
        gyroBias[1] = prefs.getShort("by", 0);
        gyroBias[2] = prefs.getShort("bz", 0);
    }
    prefs.end();
    return valid;
}

static void saveBiasToNVS() {
    prefs.begin("gyro", false);
    prefs.putBool("valid", true);
    prefs.putShort("bx", gyroBias[0]);
    prefs.putShort("by", gyroBias[1]);
    prefs.putShort("bz", gyroBias[2]);
    prefs.end();
}

// ── Background gyro-bias calibration ─────────────────────────────
static int16_t med3(int16_t a, int16_t b, int16_t c) {
    if (a > b) { int16_t t = a; a = b; b = t; }
    if (b > c) { b = c; }
    if (a > b) { b = a; }
    return b;
}

static float madVariance(const int16_t* buf, int n) {
    int64_t sum = 0;
    for (int i = 0; i < n; i++) sum += buf[i];
    float med = (float)sum / n;
    float absDevs[100];  // BG_ROLL_N max
    for (int i = 0; i < n; i++) {
        float d = buf[i] - med;
        absDevs[i] = (d >= 0) ? d : -d;
    }
    for (int i = 1; i < n; i++) {
        float key = absDevs[i];
        int j = i - 1;
        while (j >= 0 && absDevs[j] > key) {
            absDevs[j + 1] = absDevs[j];
            j--;
        }
        absDevs[j + 1] = key;
    }
    float madVal = (n % 2 == 0)
        ? 0.5f * (absDevs[n/2 - 1] + absDevs[n/2])
        : absDevs[n/2];
    constexpr float K = 1.4826f;
    return K * K * madVal * madVal;
}

#define BG_ROLL_N           100       // 0.5s @ 200 Hz
#define BG_STILL_THRESH     200.0f    // counts variance sum
#define BG_BIAS_TARGET      600       // 3s of still samples
#define BG_BIAS_DIFF_THRESH 3         // counts change to trigger NVS save

static int16_t  bgRollX[BG_ROLL_N], bgRollY[BG_ROLL_N], bgRollZ[BG_ROLL_N];
static int      bgRollIdx  = 0;
static int      bgRollFill = 0;
static int16_t  bgHistX[3], bgHistY[3], bgHistZ[3];
static int      bgHistIdx  = 0;
static int      bgHistFill = 0;
static int64_t  bgBsx = 0, bgBsy = 0, bgBsz = 0;
static int      bgBn  = 0;
static uint32_t bgLastReportMs = 0;

static void bgCalibrateTick() {
    int16_t rawGx = imuGx, rawGy = imuGy, rawGz = imuGz;
    if (gyroBiasReady) {
        rawGx += gyroBias[0]; rawGy += gyroBias[1]; rawGz += gyroBias[2];
    }

    bgHistX[bgHistIdx] = rawGx;
    bgHistY[bgHistIdx] = rawGy;
    bgHistZ[bgHistIdx] = rawGz;
    bgHistIdx = (bgHistIdx + 1) % 3;
    if (bgHistFill < 3) bgHistFill++;

    int16_t fx, fy, fz;
    if (bgHistFill >= 3) {
        fx = med3(bgHistX[0], bgHistX[1], bgHistX[2]);
        fy = med3(bgHistY[0], bgHistY[1], bgHistY[2]);
        fz = med3(bgHistZ[0], bgHistZ[1], bgHistZ[2]);
    } else {
        fx = rawGx; fy = rawGy; fz = rawGz;
    }

    bgRollX[bgRollIdx] = fx;
    bgRollY[bgRollIdx] = fy;
    bgRollZ[bgRollIdx] = fz;
    bgRollIdx = (bgRollIdx + 1) % BG_ROLL_N;
    if (bgRollFill < BG_ROLL_N) bgRollFill++;
    if (bgRollFill < BG_ROLL_N) return;

    float vsum = madVariance(bgRollX, BG_ROLL_N)
               + madVariance(bgRollY, BG_ROLL_N)
               + madVariance(bgRollZ, BG_ROLL_N);
    bool still = (vsum < BG_STILL_THRESH);

    if (still) {
        bgBsx += fx; bgBsy += fy; bgBsz += fz;
        bgBn++;
    } else if (bgBn > 0) {
        bgBsx = bgBsy = bgBsz = 0;
        bgBn = 0;
    }

    uint32_t nowMs = millis();
    if (nowMs - bgLastReportMs >= 2000) {
        bgLastReportMs = nowMs;
        Serial.printf("[bg-cal] var=%.0f %s bn=%d/%d bias=%d/%d/%d\n",
                      vsum, still ? "still" : "moving", bgBn, BG_BIAS_TARGET,
                      gyroBias[0], gyroBias[1], gyroBias[2]);
    }

    if (bgBn >= BG_BIAS_TARGET) {
        int16_t newBx = (int16_t)(bgBsx / bgBn);
        int16_t newBy = (int16_t)(bgBsy / bgBn);
        int16_t newBz = (int16_t)(bgBsz / bgBn);
        int diff = abs(newBx - gyroBias[0]) + abs(newBy - gyroBias[1]) + abs(newBz - gyroBias[2]);

        gyroBias[0] = newBx;
        gyroBias[1] = newBy;
        gyroBias[2] = newBz;
        gyroBiasReady = true;

        if (diff >= BG_BIAS_DIFF_THRESH) {
            saveBiasToNVS();
            Serial.printf("[bg-cal] UPDATED + SAVED bx=%d by=%d bz=%d (diff=%d, n=%d)\n",
                          newBx, newBy, newBz, diff, bgBn);
        } else {
            Serial.printf("[bg-cal] confirmed bx=%d by=%d bz=%d (diff=%d, no save)\n",
                          newBx, newBy, newBz, diff);
        }

        bgBsx = bgBsy = bgBsz = 0;
        bgBn = 0;
    }
}

// ── Window accumulation ──────────────────────────────────────────

static void resetWindow() {
    win.axMax = win.ayMax = win.azMax = INT16_MIN;
    win.axMin = win.ayMin = win.azMin = INT16_MAX;
    win.axSum = win.aySum = win.azSum = 0;
    win.gxMax = win.gyMax = win.gzMax = INT16_MIN;
    win.gxMin = win.gyMin = win.gzMin = INT16_MAX;
    win.gxSum = win.gySum = win.gzSum = 0;
    win.amagMax = win.gmagMax = 0.0f;
    win.amagSum = win.gmagSum = 0.0f;
    win.accelClips = 0;
    win.gyroSats   = 0;
    win.filled     = 0;
    win.rmsMax = win.rmsSum = 0.0f;
    win.rmsCount = 0;
}

static void accumulateSample() {
    if (imuAx > win.axMax) win.axMax = imuAx;
    if (imuAy > win.ayMax) win.ayMax = imuAy;
    if (imuAz > win.azMax) win.azMax = imuAz;
    if (imuAx < win.axMin) win.axMin = imuAx;
    if (imuAy < win.ayMin) win.ayMin = imuAy;
    if (imuAz < win.azMin) win.azMin = imuAz;
    win.axSum += imuAx;
    win.aySum += imuAy;
    win.azSum += imuAz;

    if (imuGx > win.gxMax) win.gxMax = imuGx;
    if (imuGy > win.gyMax) win.gyMax = imuGy;
    if (imuGz > win.gzMax) win.gzMax = imuGz;
    if (imuGx < win.gxMin) win.gxMin = imuGx;
    if (imuGy < win.gyMin) win.gyMin = imuGy;
    if (imuGz < win.gzMin) win.gzMin = imuGz;
    win.gxSum += imuGx;
    win.gySum += imuGy;
    win.gzSum += imuGz;

    float fax = (float)imuAx, fay = (float)imuAy, faz = (float)imuAz;
    float fgx = (float)imuGx, fgy = (float)imuGy, fgz = (float)imuGz;
    float amag = sqrtf(fax*fax + fay*fay + faz*faz);
    float gmag = sqrtf(fgx*fgx + fgy*fgy + fgz*fgz);
    if (amag > win.amagMax) win.amagMax = amag;
    if (gmag > win.gmagMax) win.gmagMax = gmag;
    win.amagSum += amag;
    win.gmagSum += gmag;

    if (abs(imuAx) >= CLIP_THRESH || abs(imuAy) >= CLIP_THRESH || abs(imuAz) >= CLIP_THRESH) {
        if (win.accelClips < 15) win.accelClips++;
    }
    if (abs(imuGx) >= CLIP_THRESH || abs(imuGy) >= CLIP_THRESH || abs(imuGz) >= CLIP_THRESH) {
        if (win.gyroSats < 15) win.gyroSats++;
    }

    win.filled++;
}

static void finalizeAndEmit() {
    // ── Accel packet ─────────────────────────────────────────────
    int16_t axMean = (int16_t)(win.axSum / WINDOW_SAMPLES);
    int16_t ayMean = (int16_t)(win.aySum / WINDOW_SAMPLES);
    int16_t azMean = (int16_t)(win.azSum / WINDOW_SAMPLES);

    int axMaxAc = (int)win.axMax - (int)axMean;
    int ayMaxAc = (int)win.ayMax - (int)ayMean;
    int azMaxAc = (int)win.azMax - (int)azMean;
    int axMinAc = (int)win.axMin - (int)axMean;
    int ayMinAc = (int)win.ayMin - (int)ayMean;
    int azMinAc = (int)win.azMin - (int)azMean;

    uint16_t seq = seqCounter++;

    TelemetryPacketV1 pkt;
    pkt.seq      = seq;
    pkt.ax_max   = clampI8(axMaxAc >> 8);
    pkt.ay_max   = clampI8(ayMaxAc >> 8);
    pkt.az_max   = clampI8(azMaxAc >> 8);
    pkt.ax_min   = clampI8(axMinAc >> 8);
    pkt.ay_min   = clampI8(ayMinAc >> 8);
    pkt.az_min   = clampI8(azMinAc >> 8);
    pkt.ax_mean  = clampI8((int)axMean >> 8);
    pkt.ay_mean  = clampI8((int)ayMean >> 8);
    pkt.az_mean  = clampI8((int)azMean >> 8);
    pkt.amag_max  = companding_encode(win.amagMax,                AMAG_FS);
    pkt.amag_mean = companding_encode(win.amagSum / WINDOW_SAMPLES, AMAG_FS);
    pkt.gmag_max  = companding_encode(win.gmagMax,                GMAG_FS);
    pkt.gmag_mean = companding_encode(win.gmagSum / WINDOW_SAMPLES, GMAG_FS);
    pkt.flags    = (uint8_t)((win.accelClips << 4) | (win.gyroSats & 0x0F));

    // Gate ALL forwarding on the record switch — confirmed design intent
    // (2026-06-10): the tree only listens while recording; idle = silent,
    // effects run ambient. Don't "fix" this into always-on telemetry.
    // Packets are still built so the serial debug below stays informative.
    if (wfStreaming && esp_now_send(broadcastAddr, (uint8_t*)&pkt, sizeof(pkt)) != ESP_OK) sendErrs++;
    windowsSent++;

    // ── Gyro packet ──────────────────────────────────────────────
    int16_t gxMean = (int16_t)(win.gxSum / WINDOW_SAMPLES);
    int16_t gyMean = (int16_t)(win.gySum / WINDOW_SAMPLES);
    int16_t gzMean = (int16_t)(win.gzSum / WINDOW_SAMPLES);

    int gxMaxAc = (int)win.gxMax - (int)gxMean;
    int gyMaxAc = (int)win.gyMax - (int)gyMean;
    int gzMaxAc = (int)win.gzMax - (int)gzMean;
    int gxMinAc = (int)win.gxMin - (int)gxMean;
    int gyMinAc = (int)win.gyMin - (int)gyMean;
    int gzMinAc = (int)win.gzMin - (int)gzMean;

    GyroPacketV1 gpkt;
    gpkt.seq      = seq;
    gpkt.gx_max   = clampI8(gxMaxAc >> 8);
    gpkt.gy_max   = clampI8(gyMaxAc >> 8);
    gpkt.gz_max   = clampI8(gzMaxAc >> 8);
    gpkt.gx_min   = clampI8(gxMinAc >> 8);
    gpkt.gy_min   = clampI8(gyMinAc >> 8);
    gpkt.gz_min   = clampI8(gzMinAc >> 8);
    gpkt.gx_mean  = clampI8((int)gxMean >> 8);
    gpkt.gy_mean  = clampI8((int)gyMean >> 8);
    gpkt.gz_mean  = clampI8((int)gzMean >> 8);
    gpkt.gmag_max = companding_encode(win.gmagMax, GMAG_FS);

    if (wfStreaming && esp_now_send(broadcastAddr, (uint8_t*)&gpkt, sizeof(gpkt)) != ESP_OK) sendErrs++;

    // ── Audio packet ─────────────────────────────────────────────
    float rmsMean = (win.rmsCount > 0) ? (win.rmsSum / win.rmsCount) : 0.0f;

    AudioPacketV1 apkt;
    apkt.seq      = seq;
    if (micEnabled) {
        apkt.rms_mean = companding_encode(rmsMean,    RMS_FS);
        apkt.rms_max  = companding_encode(win.rmsMax, RMS_FS);
        apkt.flags    = 0;
    } else {
        // Muted: still send so the receiver knows audio is alive, but zero
        // the levels and flag the mute so it can show silence intentionally.
        apkt.rms_mean = 0;
        apkt.rms_max  = 0;
        apkt.flags    = AUDIO_FLAG_MIC_MUTED;
    }

    if (wfStreaming && esp_now_send(broadcastAddr, (uint8_t*)&apkt, sizeof(apkt)) != ESP_OK) sendErrs++;

    // ── Periodic serial verification (~1 Hz) ─────────────────────
    if ((windowsSent % 25) == 0) {
        const float CPG = 32768.0f / (float)ACCEL_RANGE_G;
        const float CPDPS = 32768.0f / (float)GYRO_RANGE_DPS;
        float amag_max_g  = companding_decode(pkt.amag_max,  AMAG_FS) / CPG;
        float gmag_max_d  = companding_decode(pkt.gmag_max,  GMAG_FS) / CPDPS;
        float rms_max_v   = companding_decode(apkt.rms_max,  RMS_FS);
        float rms_mean_v  = companding_decode(apkt.rms_mean, RMS_FS);
        Serial.printf(
            "seq=%5u  amag_max=%4.2f g  gmag_max=%6.0f dps  "
            "rms(max/mean)=%5.0f/%5.0f  mic=%s  clip=%u sat=%u  "
            "errs=send/imu/i2c=%u/%u/%u  ch=%d\n",
            seq, amag_max_g, gmag_max_d, rms_max_v, rms_mean_v,
            micEnabled ? "on" : "MUTE",
            pkt.flags >> 4, pkt.flags & 0x0F,
            (unsigned)sendErrs, (unsigned)imuFails, (unsigned)i2cResets,
            WiFi.channel());
    }

    resetWindow();
}

// ── ESP-NOW send callback (error counting) ───────────────────────
static void onSent(const uint8_t * /*mac*/, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) sendErrs++;
}

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    // ── Wi-Fi / ESP-NOW init ──────────────────────────────────────
    // 8.5 dBm TX is the validated production value for the C3 Super Mini.
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);

    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    Serial.printf("Channel fixed to %d\n", FIXED_CHANNEL);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        while (1) delay(1000);
    }
    esp_now_register_send_cb(onSent);
    esp_now_register_recv_cb(onBsRecv);   // listen for BS-26 record switch

    Serial.printf("[BOOT] role=duck MAC=%s fw=sender_v3 ch=%d\n",
                  WiFi.macAddress().c_str(), WiFi.channel());

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

    Wire.beginTransmission(MPU_ADDR);
    if (Wire.endTransmission() == 0) {
        imuPresent = true;
        initMPU();
        delay(50);
        Serial.printf("IMU: ok (accel=±%dg gyro=±%ddps)\n",
                      ACCEL_RANGE_G, GYRO_RANGE_DPS);
        readIMU();   // prime

        if (loadBiasFromNVS()) {
            gyroBiasReady = true;
            Serial.printf("Gyro bias loaded from NVS: bx=%d by=%d bz=%d\n",
                          gyroBias[0], gyroBias[1], gyroBias[2]);
        } else {
            Serial.println("No saved bias — starting with zero, will calibrate in background");
        }
    } else {
        Serial.println("IMU: NOT FOUND — sender will idle (audio still sent)");
    }

    resetWindow();
    Serial.printf("3-layer telemetry: accel 16B / gyro 12B / audio 5B, "
                  "200 Hz inner / 25 Hz emit\n");
    Serial.printf("audio waveform: 8 kHz/8-bit, 204 B/pkt, gated on BS-26 "
                  "record switch (QUANT_SHIFT=%d)\n", AUDIO_QUANT_SHIFT);
}

// ── Shake-to-toggle detection (runs per 200 Hz accel sample) ─────
static void shakeDetectTick() {
    float rawMag = sqrtf((float)imuAx * imuAx
                       + (float)imuAy * imuAy
                       + (float)imuAz * imuAz);
    shakeGravityEma += 0.01f * (rawMag - shakeGravityEma);
    float dynamic = rawMag - shakeGravityEma;

    uint32_t now = millis();
    if (now > shakeCooldownUntil && fabsf(dynamic) > SHAKE_THRESH) {
        int8_t sign = (dynamic > 0) ? 1 : -1;
        if (sign != shakeLastSign && shakeLastSign != 0) {
            if (shakeReversals == 0) shakeFirstRevMs = now;
            shakeReversals++;
            if (shakeReversals >= SHAKE_REVERSALS
                && (now - shakeFirstRevMs) < SHAKE_WINDOW_MS) {
                // Mic is tied to the record switch, NOT shake: all sends are
                // gated on wfStreaming, so the mic is effectively live only
                // while recording. Shake no longer mutes (kept detection as a
                // no-op cooldown to avoid disturbing IMU windowing).
                shakeCooldownUntil = now + SHAKE_COOLDOWN_MS;
                shakeReversals = 0;
                shakeLastSign = 0;
            }
        }
        shakeLastSign = sign;
    }
    if (shakeReversals > 0 && (now - shakeFirstRevMs) > SHAKE_WINDOW_MS) {
        shakeReversals = 0;
        shakeLastSign = 0;
    }
}

// ── Sample tick (gated by IMU_PERIOD_US) ─────────────────────────

static void sampleTick() {
    static uint32_t nextUs = 0;
    static uint8_t  failStreak = 0;

    uint32_t now = micros();
    if ((int32_t)(now - nextUs) < 0) return;
    nextUs = now + IMU_PERIOD_US;

    if (!imuPresent) {
        // No IMU — still emit a window every 8 ticks so the audio layer
        // keeps flowing at 25 Hz. IMU fields stay at their reset extremes
        // and encode to neutral/zero values.
        if (++win.filled >= WINDOW_SAMPLES) {
            float rms = readAudioRMS();
            win.rmsMax = rms; win.rmsSum = rms; win.rmsCount = 1;
            finalizeAndEmit();
        }
        return;
    }

    if (!readIMU()) {
        imuFails++;
        if (++failStreak >= 3) {
            resetI2C();
            failStreak = 0;
        }
        return;
    }
    failStreak = 0;

    shakeDetectTick();
    bgCalibrateTick();
    accumulateSample();
    // Read the mic every tick (8× per 40 ms window) so a transient in one
    // ~5 ms slice pushes rms_max above rms_mean → real onset signal. Skip
    // empty reads (no fresh DMA samples) so they don't drag the mean down.
    float rms = readAudioRMS();
    if (rms > 0.0f) {
        if (rms > win.rmsMax) win.rmsMax = rms;
        win.rmsSum += rms;
        win.rmsCount++;
    }
    if (win.filled >= WINDOW_SAMPLES) {
        finalizeAndEmit();
    }
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    sampleTick();
}
