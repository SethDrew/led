/*
 * SENDER — ESP32-C3 handheld sensor pipe (goes inside the bulb)
 *
 * Reads MPU-6050 (I2C) at 200 Hz internally, summarizes 8 samples per
 * 40 ms window into a 16-byte v1 telemetry packet, broadcasts via
 * ESP-NOW at 25 Hz. The sender does NOT classify gestures — schema is
 * general-purpose telemetry; classification lives on the receiver and
 * can evolve without re-flashing.
 *
 * Wire schema (engineering ledger entry bulb-imu-telemetry-wire-schema-v1):
 *   0-1   uint16 seq                           (wraps)
 *   2-4   int8   ax_max,  ay_max,  az_max      AC-coupled max,  raw>>8
 *   5-7   int8   ax_min,  ay_min,  az_min      AC-coupled min
 *   8-10  int8   ax_mean, ay_mean, az_mean     gravity mean
 *   11    uint8  amag_max     sqrt-companded, FS≈57000 counts (±4g rail)
 *   12    uint8  amag_mean    same companding
 *   13    uint8  gmag_max     sqrt-companded, FS≈57000 counts (±1000 dps)
 *   14    uint8  gmag_mean    same companding
 *   15    uint8  flags        hi-nibble accel-clip count, lo gyro-sat count
 *
 * Range locked at ±4g / ±1000 dps (engineering ledger
 * bulb-imu-range-decision-4g-1000dps).
 *
 * Canonical bulb-sender wiring (ESP32-C3):
 *   GPIO 8  → SDA (MPU-6050 / clone)
 *   GPIO 7  → SCL
 *   GPIO 20 → AD0 (driven LOW at boot to force I2C addr 0x68)
 *   (audio path deferred — INMP441 wires unused in v1)
 *
 * The single mic+gyro outlier board uses a different wiring (SDA=20/SCL=21);
 * it is NOT supported by this firmware. Standardize the hardware to match.
 */

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <math.h>
#include "wifi_credentials_local.h"
#include "v1_packet.h"

// ── Pin assignments ──────────────────────────────────────────────
#define SDA_PIN    8
#define SCL_PIN    7
#define AD0_PIN    20      // driven LOW so MPU answers at 0x68
#define MPU_ADDR   0x68

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

// Companding full-scales picked from observed peaks at ±4g / ±1000 dps:
//   peak |a_ac| 6.5–7.6 g ≈ 53k–62k counts on ±4g rail
//   peak |gyro| 1456–1732 dps ≈ 47k–57k counts on ±1000 dps rail
// Encoder: byte = clamp_u8( sqrt(val/FS) * 255 )
// Decoder: val  = (byte/255)² * FS
#define AMAG_FS          57000.0f
#define GMAG_FS          57000.0f

// Per-sample axis-clip threshold. 32500 ≈ 99.2% of the int16 rail; cheap
// way to count saturated samples per window into the flags byte without
// having to know the raw rail exactly. Same threshold for gyro.
#define CLIP_THRESH      32500

// ── ESP-NOW ──────────────────────────────────────────────────────
static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Channel rescan interval — receiver follows the home AP; if router moves
// channel we'd otherwise fall silent. Scan is sync (~2-3s freeze) so keep
// the cadence loose.
#define CHANNEL_RESCAN_MS (5UL * 60UL * 1000UL)
#define CHANNEL_FALLBACK   1

static uint8_t  currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs     = 0;

// v1 wire packet — defined in lib/v1_telemetry/v1_packet.h (shared with receivers).

// ── Window aggregation state ─────────────────────────────────────
struct WindowAccum {
    // Per-axis raw running stats (counts, int16 domain)
    int16_t  axMax,  ayMax,  azMax;
    int16_t  axMin,  ayMin,  azMin;
    int32_t  axSum,  aySum,  azSum;
    // Magnitude stats (counts; sqrt computed at finalize)
    float    amagMax,  amagSum;
    float    gmagMax,  gmagSum;
    // Saturation counts for flags byte (0..15 each)
    uint8_t  accelClips;
    uint8_t  gyroSats;
    uint8_t  filled;
};
static WindowAccum win;
static uint16_t    seqCounter = 0;

// IMU read buffer
static int16_t imuAx, imuAy, imuAz;
static int16_t imuGx, imuGy, imuGz;
static bool    imuPresent = false;

// Gyro bias estimated at boot (still-rest average). Subtracted from raw gyro
// before any magnitude / saturation use. Accel bias is NOT subtracted —
// gravity is the desired DC component (mean-semantics Option A).
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

// Encode a magnitude (counts) into the sqrt-companded byte format.
// FS is the full-scale value the encoding saturates at.
static inline uint8_t companding_encode(float val, float fs) {
    if (val <= 0.0f) return 0;
    float r = val / fs;
    if (r > 1.0f) r = 1.0f;
    return clampU8((int)(sqrtf(r) * 255.0f + 0.5f));
}

// Decoded magnitude in counts from a companded byte (for serial debug).
static inline float companding_decode(uint8_t byte_val, float fs) {
    float r = (float)byte_val / 255.0f;
    return r * r * fs;
}

// ── Channel discovery ────────────────────────────────────────────
static uint8_t scanForSsidChannel(const char* ssid) {
    int n = WiFi.scanNetworks(/*async=*/false, /*show_hidden=*/false,
                              /*passive=*/true, /*max_ms_per_chan=*/120);
    uint8_t found = 0;
    int8_t bestRssi = -127;
    for (int i = 0; i < n; i++) {
        if (WiFi.SSID(i) == ssid) {
            int8_t rssi = WiFi.RSSI(i);
            if (rssi > bestRssi) {
                bestRssi = rssi;
                found = WiFi.channel(i);
            }
        }
    }
    WiFi.scanDelete();
    return found;
}

static void applyChannel(uint8_t ch) {
    if (ch == 0) ch = CHANNEL_FALLBACK;
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    currentChannel = ch;
}

// ── MPU init / read ──────────────────────────────────────────────

static void initMPU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); Wire.write(0x00);                  // wake
    Wire.endTransmission(true);
    // DLPF: 0x01 = accel 184 Hz / gyro 188 Hz BW. Required for 200 Hz
    // sampling. The legacy sender wrote 0x03 (44 Hz) which severely
    // smeared sub-10 ms tap impulses — see engineering-ledger
    // bulb-imu-telemetry-wire-schema-v1 + signal-engineer DLPF analysis.
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
        // Saturating subtract — bias is small (~50–200 counts) so overflow
        // is impossible in practice, but be explicit anyway.
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

// Variance-gated rest detection. Maintains a rolling 0.5 s window (100
// samples @ 200 Hz) of raw gyro per axis. Each tick computes total
// variance across 3 axes; when below STILL_VAR_THRESH we accumulate the
// sample into the bias estimator. If motion is detected mid-calibration
// the qualified-still counter resets. Once we have ~3 s of qualified-still
// samples (BIAS_TARGET) we latch and proceed.
//
// This lets the bulb self-calibrate the moment someone sets it down,
// rather than assuming bench-only static start.
static void calibrateGyroBias(float settleSecs) {
    constexpr int   ROLL_N          = 100;        // 0.5 s @ 200 Hz
    constexpr float STILL_VAR_THRESH = 50.0f;     // counts² per axis (sum across 3 axes)
    const int       BIAS_TARGET     = (int)(settleSecs * IMU_HZ);

    static int16_t  rollX[ROLL_N], rollY[ROLL_N], rollZ[ROLL_N];
    int   rollIdx = 0;
    int   rollFill = 0;

    int64_t bsx = 0, bsy = 0, bsz = 0;   // bias accumulators
    int     bn = 0;                       // qualified-still samples

    uint32_t nextUs = micros();
    const uint32_t periodUs = 1000000UL / IMU_HZ;
    uint32_t lastReportMs = millis();

    while (bn < BIAS_TARGET) {
        uint32_t now = micros();
        if ((int32_t)(now - nextUs) < 0) continue;
        nextUs += periodUs;
        if (!readIMU()) continue;        // bias not ready => raw values

        rollX[rollIdx] = imuGx;
        rollY[rollIdx] = imuGy;
        rollZ[rollIdx] = imuGz;
        rollIdx = (rollIdx + 1) % ROLL_N;
        if (rollFill < ROLL_N) rollFill++;

        if (rollFill < ROLL_N) continue; // need full window before judging

        // Compute per-axis variance (population). Window is small enough
        // that O(N) per tick is fine on the C3.
        int64_t mx = 0, my = 0, mz = 0;
        for (int i = 0; i < ROLL_N; i++) {
            mx += rollX[i]; my += rollY[i]; mz += rollZ[i];
        }
        float mxf = (float)mx / ROLL_N;
        float myf = (float)my / ROLL_N;
        float mzf = (float)mz / ROLL_N;
        float vx = 0, vy = 0, vz = 0;
        for (int i = 0; i < ROLL_N; i++) {
            float dx = rollX[i] - mxf;
            float dy = rollY[i] - myf;
            float dz = rollZ[i] - mzf;
            vx += dx*dx; vy += dy*dy; vz += dz*dz;
        }
        vx /= ROLL_N; vy /= ROLL_N; vz /= ROLL_N;
        float vsum = vx + vy + vz;

        bool still = (vsum < STILL_VAR_THRESH);

        if (still) {
            bsx += imuGx; bsy += imuGy; bsz += imuGz;
            bn++;
        } else if (bn > 0) {
            // Motion detected mid-calibration — discard partial bias.
            bsx = bsy = bsz = 0;
            bn = 0;
        }

        uint32_t nowMs = millis();
        if (nowMs - lastReportMs >= 250) {
            lastReportMs = nowMs;
            if (still) {
                Serial.printf("[bias] settling... var=%.1f bn=%d/%d\n",
                              vsum, bn, BIAS_TARGET);
            } else {
                Serial.printf("[bias] waiting for stillness... var=%.1f\n", vsum);
            }
        }
    }

    gyroBias[0] = (int16_t)(bsx / bn);
    gyroBias[1] = (int16_t)(bsy / bn);
    gyroBias[2] = (int16_t)(bsz / bn);
    gyroBiasReady = true;
    Serial.printf("[bias] settled bx=%d by=%d bz=%d counts (n=%d)\n",
                  gyroBias[0], gyroBias[1], gyroBias[2], bn);
}

// ── Window accumulation ──────────────────────────────────────────

static void resetWindow() {
    win.axMax = win.ayMax = win.azMax = INT16_MIN;
    win.axMin = win.ayMin = win.azMin = INT16_MAX;
    win.axSum = win.aySum = win.azSum = 0;
    win.amagMax = win.gmagMax = 0.0f;
    win.amagSum = win.gmagSum = 0.0f;
    win.accelClips = 0;
    win.gyroSats   = 0;
    win.filled     = 0;
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

    // Magnitudes in counts (float to keep sqrt cheap on C3)
    float fax = (float)imuAx, fay = (float)imuAy, faz = (float)imuAz;
    float fgx = (float)imuGx, fgy = (float)imuGy, fgz = (float)imuGz;
    float amag = sqrtf(fax*fax + fay*fay + faz*faz);
    float gmag = sqrtf(fgx*fgx + fgy*fgy + fgz*fgz);
    if (amag > win.amagMax) win.amagMax = amag;
    if (gmag > win.gmagMax) win.gmagMax = gmag;
    win.amagSum += amag;
    win.gmagSum += gmag;

    // Per-axis saturation count (any axis ≥ CLIP_THRESH counts as one
    // sample). Counter saturates at 15 (4-bit nibble).
    if (abs(imuAx) >= CLIP_THRESH || abs(imuAy) >= CLIP_THRESH || abs(imuAz) >= CLIP_THRESH) {
        if (win.accelClips < 15) win.accelClips++;
    }
    if (abs(imuGx) >= CLIP_THRESH || abs(imuGy) >= CLIP_THRESH || abs(imuGz) >= CLIP_THRESH) {
        if (win.gyroSats < 15) win.gyroSats++;
    }

    win.filled++;
}

static void finalizeAndEmit() {
    int16_t axMean = (int16_t)(win.axSum / WINDOW_SAMPLES);
    int16_t ayMean = (int16_t)(win.aySum / WINDOW_SAMPLES);
    int16_t azMean = (int16_t)(win.azSum / WINDOW_SAMPLES);

    // AC max/min = signed deviation of raw extremes from window mean.
    // Then >>8 maps int16 → int8; saturating clamp guards the corner case
    // where a single-window swing exceeds ±32k counts (only possible
    // mid-impact when one axis crosses the rail).
    int axMaxAc = (int)win.axMax - (int)axMean;
    int ayMaxAc = (int)win.ayMax - (int)ayMean;
    int azMaxAc = (int)win.azMax - (int)azMean;
    int axMinAc = (int)win.axMin - (int)axMean;
    int ayMinAc = (int)win.ayMin - (int)ayMean;
    int azMinAc = (int)win.azMin - (int)azMean;

    TelemetryPacketV1 pkt;
    pkt.seq      = seqCounter++;
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

    esp_err_t r = esp_now_send(broadcastAddr, (uint8_t*)&pkt, sizeof(pkt));
    if (r != ESP_OK) sendErrs++;
    windowsSent++;

    // ── Periodic serial verification (every ~1 s) ────────────────
    // Decode the bytes back to physical units so we can eyeball that
    // the packet is sane before declaring this firmware flight-ready.
    if ((windowsSent % 25) == 0) {
        // 1 g ≈ 16384/(ACCEL_RANGE_G/2) counts on ±2g, or 16384/2 on ±4g, etc.
        // Generic: counts_per_g = 32768 / ACCEL_RANGE_G
        const float CPG = 32768.0f / (float)ACCEL_RANGE_G;
        const float CPDPS = 32768.0f / (float)GYRO_RANGE_DPS;
        float amag_max_g  = companding_decode(pkt.amag_max,  AMAG_FS) / CPG;
        float amag_mean_g = companding_decode(pkt.amag_mean, AMAG_FS) / CPG;
        float gmag_max_d  = companding_decode(pkt.gmag_max,  GMAG_FS) / CPDPS;
        float gmag_mean_d = companding_decode(pkt.gmag_mean, GMAG_FS) / CPDPS;
        float ax_mean_g   = ((float)pkt.ax_mean * 256.0f) / CPG;
        float ay_mean_g   = ((float)pkt.ay_mean * 256.0f) / CPG;
        float az_mean_g   = ((float)pkt.az_mean * 256.0f) / CPG;
        Serial.printf(
            "seq=%5u  amag(max/mean)=%4.2f/%4.2f g  "
            "gmag=%6.0f/%6.0f dps  grav=(%+.2f,%+.2f,%+.2f) g  "
            "clip=%u sat=%u  errs=send/imu/i2c=%u/%u/%u  ch=%d\n",
            pkt.seq,
            amag_max_g, amag_mean_g,
            gmag_max_d, gmag_mean_d,
            ax_mean_g, ay_mean_g, az_mean_g,
            pkt.flags >> 4, pkt.flags & 0x0F,
            (unsigned)sendErrs, (unsigned)imuFails, (unsigned)i2cResets,
            WiFi.channel());
    }

    resetWindow();
}

// ── ESP-NOW send callback (just for error counting) ──────────────
static void onSent(const uint8_t * /*mac*/, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) sendErrs++;
}

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    // ── ESP-NOW init ──────────────────────────────────────────────
    // 8.5 dBm TX is the validated production value; see engineering ledger
    // entry esp32-c3-super-mini-tx-defect for why full power breaks TX on
    // the C3 Super Mini batches we use.
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);

    Serial.printf("Scanning for '%s'...\n", WIFI_SSID);
    uint8_t ch = scanForSsidChannel(WIFI_SSID);
    if (ch) {
        Serial.printf("Found '%s' on ch=%d\n", WIFI_SSID, ch);
    } else {
        Serial.printf("'%s' not visible — falling back to ch=%d\n", WIFI_SSID, CHANNEL_FALLBACK);
    }
    applyChannel(ch);
    lastScanMs = millis();

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        while (1) delay(1000);
    }
    esp_now_register_send_cb(onSent);

    Serial.printf("Sender ch=%d MAC=%s\n", WiFi.channel(), WiFi.macAddress().c_str());

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    // ── I2C / MPU-6050 ────────────────────────────────────────────
    pinMode(AD0_PIN, OUTPUT);
    digitalWrite(AD0_PIN, LOW);          // force MPU addr to 0x68
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
        Serial.println("Gyro bias: variance-gated rest detection — set bulb down to calibrate.");
        calibrateGyroBias(3.0f);
    } else {
        Serial.println("IMU: NOT FOUND — sender will idle");
    }

    resetWindow();
    Serial.printf("v1 telemetry: 16 B/pkt, 200 Hz inner / 25 Hz emit, "
                  "amag_FS=%.0f gmag_FS=%.0f\n", AMAG_FS, GMAG_FS);
}

// ── Sample tick (called each pass, gated by IMU_PERIOD_US) ───────

static void sampleTick() {
    static uint32_t nextUs = 0;
    static uint8_t  failStreak = 0;

    uint32_t now = micros();
    if ((int32_t)(now - nextUs) < 0) return;
    nextUs = now + IMU_PERIOD_US;

    if (!imuPresent) return;

    if (!readIMU()) {
        // Mark the sample's contribution as "no change" rather than
        // injecting -1 garbage. Don't accumulate. After 3 fails in a row,
        // bus-reset.
        imuFails++;
        if (++failStreak >= 3) {
            resetI2C();
            failStreak = 0;
        }
        return;
    }
    failStreak = 0;

    accumulateSample();
    if (win.filled >= WINDOW_SAMPLES) {
        finalizeAndEmit();
    }
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t nowMs = millis();

    // Periodic channel rescan — heals if router moved channel since boot.
    // Sync scan freezes the loop ~2-3s; receiver tolerates the gap. Window
    // accumulator is reset after the rescan to avoid emitting a packet
    // built from a partial window plus a 3 s stale tail.
    if (nowMs - lastScanMs > CHANNEL_RESCAN_MS) {
        uint8_t ch = scanForSsidChannel(WIFI_SSID);
        if (ch && ch != currentChannel) {
            Serial.printf("[heal] channel drift %d -> %d\n", currentChannel, ch);
            applyChannel(ch);
        }
        lastScanMs = millis();
        resetWindow();
    }

    sampleTick();
}
