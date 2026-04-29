/*
 * SENDER RECORDER — capture raw IMU + audio RMS to LittleFS for later analysis.
 *
 * Behavior: always starts a new recording on boot, into the next free
 *   /recN.bin (N = max existing index + 1). Recording continues regardless
 *   of host CDC presence; commands work in parallel when laptop attached.
 *   Cleanup is manual: dump + erase when connected to laptop.
 *
 * Commands: r=restart record, s=stop, d=dump-all (base64), e=erase-all, i=info
 *
 * File format /recN.bin:
 *   header (16 B): 'DUCK', uint16 ver, uint16 sample_hz=200,
 *                  uint32 sample_count, uint32 cfg
 *     ver=1: ±2g, ±250°/s assumed (cfg unused, written as 0)
 *     ver=2: cfg byte 0 = AFS_SEL (0=2g, 1=4g, 2=8g, 3=16g)
 *            cfg byte 1 = FS_SEL  (0=250, 1=500, 2=1000, 3=2000 °/s)
 *            cfg bytes 2-3 reserved
 *   body  (14 B/sample): int16 ax,ay,az,gx,gy,gz; uint16 rms
 *
 * Hardware: ESP32-C3 super-mini, MPU-6050/clone (I2C), INMP441 (I2S SCK=6 WS=5 SD=0).
 *           DLPF 188 Hz to keep tap edges. IMU full-scale ranges configurable
 *           via build flags ACCEL_RANGE_G / GYRO_RANGE_DPS (default 2 / 250).
 *           Canonical wiring: SDA=8, SCL=7, AD0=20 (driven LOW for 0x68).
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <LittleFS.h>
#include <esp_log.h>

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

#define HEADER_VER 2
#define HEADER_CFG ((uint32_t)ACCEL_AFS_SEL | ((uint32_t)GYRO_FS_SEL << 8))

// ── I2S ──────────────────────────────────────────────────────────
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   160      // ~10 ms @ 16 kHz → RMS refresh ~100 Hz
#define DMA_BUF_COUNT 4

// ── Recording ────────────────────────────────────────────────────
#define IMU_HZ          200
#define IMU_PERIOD_US   (1000000 / IMU_HZ)
#define MAX_RECORD_SEC  180
#define MAX_FILE_BYTES  (1024 * 1024)        // 1 MB cap (~6 min @ 14 B/200 Hz)
#define WRITE_BUF_BYTES 1024                 // ~73 samples, ~365 ms at 200 Hz

// ── File ─────────────────────────────────────────────────────────
#define REC_PREFIX "/rec"
#define REC_SUFFIX ".bin"
#define MAX_FILES   64
#define MAGIC    0x4B435544                  // 'DUCK' little-endian

struct __attribute__((packed)) RecordHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t sample_hz;
    uint32_t sample_count;
    uint32_t reserved;
};

struct __attribute__((packed)) Sample {
    int16_t  ax, ay, az;
    int16_t  gx, gy, gz;
    uint16_t rms;
};

static_assert(sizeof(RecordHeader) == 16, "header must be 16 B");
static_assert(sizeof(Sample) == 14, "sample must be 14 B");

// ── State ────────────────────────────────────────────────────────
static int16_t  imuAx, imuAy, imuAz;
static int16_t  imuGx, imuGy, imuGz;
static uint16_t latchedRms = 0;
static bool     imuPresent = false;

static File     recFile;
static uint8_t  writeBuf[WRITE_BUF_BYTES];
static size_t   writeBufFill = 0;
static uint32_t sampleCount  = 0;
static uint32_t recStartMs   = 0;
static bool     recording    = false;
static bool     filePresent  = false;       // cached, refreshed on lifecycle events only
static char     currentPath[16] = "";
static int      currentIndex    = -1;

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
    Wire.setClock(400000);
    initMPU();
}

static uint32_t i2cResets = 0;
static uint32_t i2cFails  = 0;

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

// ── Recording control ────────────────────────────────────────────
// flushBuffer: cheap path on buffer-full — append-only, no metadata update.
// finalizeFile: expensive path on stop — writes header with sample_count.
// Tradeoff: power loss before stop yields a file with sample_count=0 in the
// header; the body is intact and recoverable by counting bytes.
static void flushBuffer() {
#ifdef DISABLE_FS_WRITES
    writeBufFill = 0;          // discard buffer; isolation test for I2C/FS contention
    return;
#else
    if (!recFile || !writeBufFill) return;
    recFile.write(writeBuf, writeBufFill);
    writeBufFill = 0;
#endif
}

static void finalizeFile() {
    if (!recFile) return;
    flushBuffer();
    size_t pos = recFile.position();
    recFile.seek(0);
    RecordHeader hdr = { MAGIC, HEADER_VER, IMU_HZ, sampleCount, HEADER_CFG };
    recFile.write((uint8_t*)&hdr, sizeof(hdr));
    recFile.seek(pos);
    recFile.flush();
}

static void writeSample(const Sample &s) {
    if (writeBufFill + sizeof(Sample) > sizeof(writeBuf)) {
        flushBuffer();
    }
    memcpy(&writeBuf[writeBufFill], &s, sizeof(Sample));
    writeBufFill += sizeof(Sample);
    sampleCount++;
}

// Parse a filename like "rec5.bin" or "/rec5.bin" → 5; returns -1 if not ours.
static int parseIndex(const char *name) {
    if (*name == '/') name++;
    if (strncmp(name, "rec", 3) != 0) return -1;
    const char *p = name + 3;
    if (*p < '0' || *p > '9') return -1;
    int n = 0;
    while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; }
    if (strcmp(p, ".bin") != 0) return -1;
    return n;
}

static int nextFreeIndex() {
    File root = LittleFS.open("/");
    int max_idx = -1;
    File f = root.openNextFile();
    while (f) {
        int idx = parseIndex(f.name());
        if (idx > max_idx) max_idx = idx;
        f = root.openNextFile();
    }
    return max_idx + 1;
}

static void buildPath(int idx, char *out, size_t cap) {
    snprintf(out, cap, "%s%d%s", REC_PREFIX, idx, REC_SUFFIX);
}

static void startRecord() {
    if (recording) return;
    currentIndex = nextFreeIndex();
    buildPath(currentIndex, currentPath, sizeof(currentPath));

    recFile = LittleFS.open(currentPath, "w");
    if (!recFile) {
        Serial.printf("ERR: open %s\n", currentPath);
        return;
    }
    RecordHeader hdr = { MAGIC, HEADER_VER, IMU_HZ, 0, HEADER_CFG };
    recFile.write((uint8_t*)&hdr, sizeof(hdr));

    sampleCount  = 0;
    writeBufFill = 0;
    recStartMs   = millis();
    recording    = true;
    filePresent  = true;
    Serial.printf("REC start: %s\n", currentPath);
}

static void stopRecord() {
    if (!recording) return;
    finalizeFile();
    recFile.close();
    recording = false;
    Serial.printf("REC stop: %u samples (%.2f s)\n",
                  (unsigned)sampleCount, sampleCount / (float)IMU_HZ);
}

static void recordTick() {
    static uint32_t nextUs = 0;
    uint32_t now = micros();
    if ((int32_t)(now - nextUs) < 0) return;
    nextUs = now + IMU_PERIOD_US;

    if (imuPresent) {
        static uint8_t failStreak = 0;
        if (readIMU()) {
            failStreak = 0;
        } else {
            i2cFails++;
            imuAx = imuAy = imuAz = imuGx = imuGy = imuGz = -1;
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
    writeSample(s);

    if (sampleCount >= MAX_FILE_BYTES / sizeof(Sample)
        || (millis() - recStartMs) >= (MAX_RECORD_SEC * 1000UL)) {
        stopRecord();
        Serial.println("REC limit reached, idle.");
    }
}

// ── Commands ─────────────────────────────────────────────────────
// Walk root, list each rec*.bin with size and approximate sample count
// (derived from byte size; the persisted header has the authoritative count
// once the file is finalized).
static void cmdInfo() {
    File root = LittleFS.open("/");
    File f = root.openNextFile();
    int n = 0;
    while (f) {
        int idx = parseIndex(f.name());
        if (idx >= 0) {
            uint32_t bytes = f.size();
            uint32_t samples = bytes >= sizeof(RecordHeader)
                ? (bytes - sizeof(RecordHeader)) / sizeof(Sample) : 0;
            Serial.printf("FILE rec%d: ~%u samples (%.2f s), %u bytes%s\n",
                          idx, (unsigned)samples,
                          samples / (float)IMU_HZ, (unsigned)bytes,
                          (idx == currentIndex && recording) ? " RECORDING" : "");
            n++;
        }
        f = root.openNextFile();
    }
    if (n == 0) Serial.println("FILES: none");
    Serial.printf("FS:   %u / %u bytes used\n",
                  (unsigned)LittleFS.usedBytes(), (unsigned)LittleFS.totalBytes());
    Serial.printf("I2C resets: %u  fails: %u\n",
                  (unsigned)i2cResets, (unsigned)i2cFails);
}

static void dumpOne(int idx, const char *path) {
    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("ERR: open %s\n", path);
        return;
    }
    size_t n = f.size();
    Serial.printf("FILE idx=%d bytes=%u\n", idx, (unsigned)n);
    static const char b64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    uint8_t buf[252];                      // multiple of 3 → no padding mid-stream
    while (true) {
        size_t got = f.read(buf, sizeof(buf));
        if (got == 0) break;
        char line[400];
        size_t li = 0;
        for (size_t i = 0; i < got; i += 3) {
            uint32_t v = (uint32_t)buf[i] << 16;
            bool b = (i + 1 < got), c = (i + 2 < got);
            if (b) v |= (uint32_t)buf[i+1] << 8;
            if (c) v |= buf[i+2];
            line[li++] = b64[(v >> 18) & 63];
            line[li++] = b64[(v >> 12) & 63];
            line[li++] = b ? b64[(v >> 6) & 63] : '=';
            line[li++] = c ? b64[v & 63]        : '=';
        }
        Serial.write((uint8_t*)line, li);
        Serial.write('\n');
    }
    f.close();
}

static void cmdDump() {
    if (recording) stopRecord();             // close write handle before reading
    int indices[MAX_FILES];
    int nFiles = 0;
    File root = LittleFS.open("/");
    File f = root.openNextFile();
    while (f && nFiles < MAX_FILES) {
        int idx = parseIndex(f.name());
        if (idx >= 0) indices[nFiles++] = idx;
        f = root.openNextFile();
    }
    // sort ascending so dumps are chronological
    for (int i = 1; i < nFiles; i++) {
        int v = indices[i], j = i - 1;
        while (j >= 0 && indices[j] > v) { indices[j+1] = indices[j]; j--; }
        indices[j+1] = v;
    }
    Serial.printf("DUMPBEGIN files=%d\n", nFiles);
    char path[16];
    for (int i = 0; i < nFiles; i++) {
        buildPath(indices[i], path, sizeof(path));
        dumpOne(indices[i], path);
    }
    Serial.println("DUMPEND");
}

static void cmdErase() {
    if (recording) stopRecord();
    char paths[MAX_FILES][16];
    int n = 0;
    File root = LittleFS.open("/");
    File f = root.openNextFile();
    while (f && n < MAX_FILES) {
        int idx = parseIndex(f.name());
        if (idx >= 0) {
            const char *name = f.name();
            if (*name != '/') snprintf(paths[n], sizeof(paths[n]), "/%s", name);
            else { strncpy(paths[n], name, sizeof(paths[n]) - 1);
                   paths[n][sizeof(paths[n]) - 1] = '\0'; }
            n++;
        }
        f = root.openNextFile();
    }
    for (int i = 0; i < n; i++) LittleFS.remove(paths[i]);
    Serial.printf("ERASED %d files\n", n);
    filePresent = false;
    currentIndex = -1;
    currentPath[0] = '\0';
}

static void printMenu() {
    Serial.println("\n=== DUCK RECORDER ===");
    cmdInfo();
    Serial.println("Commands: r=record, s=stop, d=dump, e=erase, i=info");
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(460800);
    esp_log_level_set("vfs_api", ESP_LOG_NONE);   // silence harmless missing-file log

    if (!LittleFS.begin(true)) {
        Serial.println("LittleFS mount failed");
        while (1) delay(1000);
    }

    setupI2S();
    pinMode(AD0_PIN, OUTPUT);
    digitalWrite(AD0_PIN, LOW);                  // force MPU addr 0x68
    delay(5);
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    Serial.printf("I2C: SDA=%d SCL=%d AD0=%d\n", SDA_PIN, SCL_PIN, AD0_PIN);

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

    // Always record on boot — preserves prior captures by writing to next free
    // /recN.bin. Cleanup is manual via 'e' from the laptop.
    printMenu();
    startRecord();
}

// ── Status LED on GPIO 8 (built-in on most C3 super-mini clones, active LOW) ─
// Modes:  recording → 1 Hz blink   idle-with-file → 4 Hz blink   empty → off
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
    uint32_t period;
    if (recording) {
        period = 500;                          // 1 Hz blink
    } else if (filePresent) {
        period = 125;                          // 4 Hz blink — file present, idle
    } else {
        digitalWrite(STATUS_LED_PIN, STATUS_LED_OFF);
        return;
    }
    if (now - lastToggle >= period) {
        lastToggle = now;
        ledOn = !ledOn;
        digitalWrite(STATUS_LED_PIN, ledOn ? STATUS_LED_ON : STATUS_LED_OFF);
    }
}

// ── Loop ─────────────────────────────────────────────────────────
void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r' || c == '\n' || c == ' ') continue;
        switch (c) {
            case 'r': stopRecord(); startRecord(); break;
            case 's': stopRecord(); break;
            case 'd': cmdDump();    break;
            case 'e': stopRecord(); cmdErase(); break;
            case 'i': cmdInfo();    break;
            default:  Serial.printf("?: %c\n", c); break;
        }
    }
    if (recording) recordTick();
    updateStatusLed();
}
