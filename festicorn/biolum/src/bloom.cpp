/*
 * BIOLUM — 6-strip quiet bloom, one gyro sender per strip.
 *
 * Classic ESP32, 6 × WS2812B RGB strips on GPIO 4, 16, 17, 5, 18, 19.
 * Each strip is driven by its own gyro sender via ESP-NOW. Senders are
 * identified by MAC and mapped to strip index at compile time.
 *
 * Bloom algorithm ported verbatim from bench-bulbs ALG_QUIET_BLOOM.
 * Each strip has fully independent bloom state (breath, hue drift,
 * flash, gate). Motion from each gyro only affects its own strip.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <fast_math.h>
#include "v1_packet.h"
#include <delta_sigma.h>

// ── Strip layout ─────────────────────────────────────────────────
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 100
#endif

static const uint8_t NUM_STRIPS = 6;

// ── Bloom parameters (runtime-tunable via operator) ─────────────
static float bloomBrightnessCapSrc = 0.03f;   // LEDs 0–9 (source cluster, very sparse)
static float bloomBrightnessCapA  = 0.10f;   // LEDs 10–64
static float bloomBrightnessCapB  = 0.50f;   // LEDs 65–99
static float bloomBufferDrain     = 15.0f;
static float bloomFlashDecayRate  = 3.0f;
static float bloomBreathFloor     = 0.15f;
static float surpriseRiseTau     = 0.80f;
static float surpriseFallTau     = 0.15f;
static float surpriseRatio       = 1.8f;
static float gyroWeight          = 0.012f;
static float motionGain          = 2.0f;

#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.65f
#define BLOOM_BREATH_MAX_PEAK   1.00f

#define BLOOM_FLASH_DECAY_LO   0.96f
#define BLOOM_FLASH_DECAY_HI   0.985f

#define BLOOM_HUE_A_R    0.0f
#define BLOOM_HUE_A_G  180.0f
#define BLOOM_HUE_A_B  120.0f

#define BLOOM_HUE_B_R  140.0f
#define BLOOM_HUE_B_G   20.0f
#define BLOOM_HUE_B_B  255.0f

#define BLOOM_FLASH_R  200.0f
#define BLOOM_FLASH_G  120.0f
#define BLOOM_FLASH_B  255.0f

#define BLOOM_HUE_DRIFT_MIN  (1.0f / 45.0f)
#define BLOOM_HUE_DRIFT_MAX  (1.0f / 15.0f)

#define BLOOM_BLACKOUT_HOLD_S 1.0f

#define GATE_ON_THRESH  0.7f
#define GATE_OFF_THRESH 1.2f

#define SENSOR_HZ       25.0f
#define TIMEOUT_MS     500

// ── v1 telemetry decode ──────────────────────────────────────────
#define MAG_FS           57000.0f
#define COUNTS_PER_G      8192.0f

static inline float amagGFromByte(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * MAG_FS / COUNTS_PER_G;
}

#define COUNTS_PER_DPS 32.8f

static inline float gmagDpsFromByte(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * MAG_FS / COUNTS_PER_DPS;
}

// ── PRNG ─────────────────────────────────────────────────────────
static uint32_t prngState;

static inline uint32_t xorshift32() {
    prngState ^= prngState << 13;
    prngState ^= prngState >> 17;
    prngState ^= prngState << 5;
    return prngState;
}

static inline float randFloat() {
    return (float)(xorshift32() & 0xFFFFFF) / 16777216.0f;
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// ── Gyro sender MAC → strip mapping ─────────────────────────────
// Last two bytes of each sender MAC, matched in onReceive.
struct SenderDef {
    uint8_t mac4;
    uint8_t mac5;
};

static const SenderDef SENDER_MAP[NUM_STRIPS] = {
    { 0x0B, 0x88 },  // strip 0 (pin 4)  — 14:63:93:70:0B:88
    { 0x93, 0xB0 },  // strip 1 (pin 16) — 14:63:93:6E:93:B0
    { 0xAE, 0x4C },  // strip 2 (pin 17) — 14:63:93:6E:AE:4C
    { 0xAA, 0x2C },  // strip 3 (pin 5)  — 14:63:93:6E:AA:2C
    { 0xF5, 0xF8 },  // strip 4 (pin 18) — 10:00:3B:B0:F5:F8
    { 0x9E, 0xF0 },  // strip 5 (pin 19) — 10:00:3B:B1:9E:F0
};

// ── ESP-NOW command dispatch ─────────────────────────────────────
#define CMD_PKT_MAGIC 0xC0
static volatile bool espnowCmdPending = false;
static char espnowCmdBuf[64];
static volatile uint8_t espnowCmdLen = 0;

// ── Per-sender ESP-NOW state ─────────────────────────────────────
static volatile uint32_t senderLastMs[NUM_STRIPS] = {0};
static volatile uint8_t  senderPeakAmag[NUM_STRIPS] = {0};
static volatile uint8_t  senderPeakGmag[NUM_STRIPS] = {0};
static volatile uint32_t senderPktCount[NUM_STRIPS] = {0};
static volatile uint16_t senderSeq[NUM_STRIPS] = {0};
static volatile uint16_t senderPrevSeq[NUM_STRIPS] = {0};

void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    // Telemetry is exactly 16 bytes; commands are any other length with 0xC0 prefix.
    // This prevents seq=0xC0 in a telemetry packet from being misread as a command.
    if (len != sizeof(TelemetryPacketV1) && len >= 2 && data[0] == CMD_PKT_MAGIC && !espnowCmdPending) {
        uint8_t cmdLen = (len - 1 < (int)sizeof(espnowCmdBuf) - 1) ? len - 1 : sizeof(espnowCmdBuf) - 1;
        memcpy(espnowCmdBuf, data + 1, cmdLen);
        espnowCmdBuf[cmdLen] = '\0';
        espnowCmdLen = cmdLen;
        espnowCmdPending = true;
        return;
    }
    if (len != sizeof(TelemetryPacketV1)) return;

    int idx = -1;
    for (int i = 0; i < NUM_STRIPS; i++) {
        if (mac[4] == SENDER_MAP[i].mac4 && mac[5] == SENDER_MAP[i].mac5) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return;

    TelemetryPacketV1 pkt;
    memcpy(&pkt, data, sizeof(TelemetryPacketV1));
    if (pkt.amag_mean > senderPeakAmag[idx]) senderPeakAmag[idx] = pkt.amag_mean;
    if (pkt.gmag_mean > senderPeakGmag[idx]) senderPeakGmag[idx] = pkt.gmag_mean;
    senderPrevSeq[idx] = senderSeq[idx];
    senderSeq[idx] = pkt.seq;
    senderLastMs[idx] = millis();
    senderPktCount[idx]++;
}

// ── LED driver: 6 strips via RMT ─────────────────────────────────
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 16);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, 19);

static inline void setPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
        case 4: strip4.SetPixelColor(i, c); break;
        case 5: strip5.SetPixelColor(i, c); break;
    }
}

static void showAll() {
    strip0.Show(); strip1.Show(); strip2.Show();
    strip3.Show(); strip4.Show(); strip5.Show();
}

// ── Per-strip bloom state ────────────────────────────────────────
struct BloomStrip {
    float breathPhase[LEDS_PER_STRIP];
    float breathPeriod[LEDS_PER_STRIP];
    float breathPeak[LEDS_PER_STRIP];
    float hueT[LEDS_PER_STRIP];
    float hueDrift[LEDS_PER_STRIP];
    float blackoutTimer[LEDS_PER_STRIP];
    bool  gateOff[LEDS_PER_STRIP];
    uint16_t ditherR[LEDS_PER_STRIP];
    uint16_t ditherG[LEDS_PER_STRIP];
    uint16_t ditherB[LEDS_PER_STRIP];
    float flash;
    float energyBuffer;
    uint32_t prevPktCount;
    float motionEma;
    bool  emaSeeded;
};

static BloomStrip bloom[NUM_STRIPS];

static void resetBloomStrip(BloomStrip &bs) {
    bs.flash = 0.0f;
    bs.energyBuffer = 0.0f;
    bs.prevPktCount = 0;
    bs.motionEma = 0.0f;
    bs.emaSeeded = false;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        bs.breathPhase[i]  = randFloat();
        bs.breathPeriod[i] = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bs.breathPeak[i]   = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bs.hueT[i]         = randFloat();
        float rate = BLOOM_HUE_DRIFT_MIN
            + randFloat() * (BLOOM_HUE_DRIFT_MAX - BLOOM_HUE_DRIFT_MIN);
        bs.hueDrift[i]     = (randFloat() > 0.5f) ? rate : -rate;
        bs.blackoutTimer[i] = 0.0f;
        bs.gateOff[i]      = false;
        bs.ditherR[i] = 0;
        bs.ditherG[i] = 0;
        bs.ditherB[i] = 0;
    }
}

// ── Bloom motion processing (per strip) ──────────────────────────
static void bloomProcessMotion(uint8_t s) {
    uint8_t peakAmag = senderPeakAmag[s];
    uint8_t peakGmag = senderPeakGmag[s];
    senderPeakAmag[s] = 0;
    senderPeakGmag[s] = 0;

    uint16_t dSeq = senderSeq[s] - senderPrevSeq[s];
    float pktDt = (dSeq > 0 && dSeq < 250) ? (float)dSeq / SENSOR_HZ : (1.0f / SENSOR_HZ);

    float aG   = amagGFromByte(peakAmag);
    float gDps = gmagDpsFromByte(peakGmag);
    float motion = fmaxf(aG, gDps * gyroWeight);

    BloomStrip &bs = bloom[s];

    if (!bs.emaSeeded) {
        bs.motionEma = motion;
        bs.emaSeeded = true;
        return;
    }

    float alpha = fminf(1.0f, pktDt / ((motion > bs.motionEma) ? surpriseRiseTau : surpriseFallTau));
    bs.motionEma += alpha * (motion - bs.motionEma);

    float excess = fmaxf(0.0f, motion - bs.motionEma * surpriseRatio);
    bs.energyBuffer += excess * pktDt * motionGain;
    if (bs.energyBuffer > 0.5f) bs.energyBuffer = 0.5f;
}

// ── Bloom render (per strip) ─────────────────────────────────────
static void renderBloomStrip(uint8_t s, float dt) {
    BloomStrip &bs = bloom[s];

    float drain = bs.energyBuffer * bloomBufferDrain * dt;
    if (bs.energyBuffer < 0.05f) drain += 0.1f * dt;
    drain = fminf(drain, bs.energyBuffer);
    bs.energyBuffer -= drain;
    bs.flash = fminf(1.0f, bs.flash + drain);

    bs.flash *= expf(-bloomFlashDecayRate * dt);
    if (bs.flash < 0.005f) bs.flash = 0.0f;

    // Hoist per-strip constants out of pixel loop
    float flashSrc = bs.flash * bloomBrightnessCapSrc;
    float flashA = bs.flash * bloomBrightnessCapA;
    float flashB = bs.flash * bloomBrightnessCapB;
    float flashFrac = (bs.flash > 0.1f) ? clampf(bs.flash / 0.3f, 0.0f, 1.0f) : 0.0f;
    float oneMinusFF = 1.0f - flashFrac;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        float breath = fastSinPhase(bs.breathPhase[i]) * 0.5f + 0.5f;
        float breathGlow = bloomBreathFloor + breath * (bs.breathPeak[i] - bloomBreathFloor);
        float cap = (i < 10) ? bloomBrightnessCapSrc : (i >= 65) ? bloomBrightnessCapB : bloomBrightnessCapA;
        float breathLin = fastGamma24(breathGlow) * cap;
        float flashLin  = (i < 10) ? flashSrc : (i >= 65) ? flashB : flashA;

        bs.breathPhase[i] += dt / bs.breathPeriod[i];
        if (bs.breathPhase[i] >= 1.0f) bs.breathPhase[i] -= 1.0f;
        bs.hueT[i] += bs.hueDrift[i] * dt;
        if (bs.hueT[i] > 1.0f) bs.hueT[i] -= 1.0f;
        else if (bs.hueT[i] < 0.0f) bs.hueT[i] += 1.0f;

        float h = bs.hueT[i];
        float baseR = lerpf(BLOOM_HUE_A_R, BLOOM_HUE_B_R, h);
        float baseG = lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h);
        float baseB = lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h);

        float oR = baseR * breathLin + (baseR * oneMinusFF + BLOOM_FLASH_R * flashFrac) * flashLin;
        float oG = baseG * breathLin + (baseG * oneMinusFF + BLOOM_FLASH_G * flashFrac) * flashLin;
        float oB = baseB * breathLin + (baseB * oneMinusFF + BLOOM_FLASH_B * flashFrac) * flashLin;

        // Hysteresis noise gate with blackout hold
        float maxCh = fmaxf(fmaxf(oR, oG), oB);
        float thresh = bs.gateOff[i] ? GATE_OFF_THRESH : GATE_ON_THRESH;
        if (maxCh < thresh) {
            oR = oG = oB = 0.0f;
            bs.gateOff[i] = true;
            bs.blackoutTimer[i] = 0.0f;
        } else if (bs.gateOff[i]) {
            bs.blackoutTimer[i] += dt;
            if (bs.blackoutTimer[i] < BLOOM_BLACKOUT_HOLD_S) {
                oR = oG = oB = 0.0f;
            } else {
                bs.gateOff[i] = false;
                bs.blackoutTimer[i] = 0.0f;
            }
        }

        uint16_t t16R = (uint16_t)fminf(oR * 256.0f, 65535.0f);
        uint16_t t16G = (uint16_t)fminf(oG * 256.0f, 65535.0f);
        uint16_t t16B = (uint16_t)fminf(oB * 256.0f, 65535.0f);
        if ((t16R | t16G | t16B) == 0) {
            bs.ditherR[i] = bs.ditherG[i] = bs.ditherB[i] = 0;
        }
        uint8_t r8 = deltaSigma(bs.ditherR[i], t16R);
        uint8_t g8 = deltaSigma(bs.ditherG[i], t16G);
        uint8_t b8 = deltaSigma(bs.ditherB[i], t16B);

        setPixel(s, i, r8, g8, b8);
    }
}

#define FIXED_CHANNEL 6

// ── Runtime parameter table ──────────────────────────────────────
struct Param {
    const char *name;
    enum Type { F32 } type;
    void *ptr;
    float lo, hi;
};

static const Param PARAMS[] = {
    { "BRIGHTNESS_CAP_A", Param::F32, &bloomBrightnessCapA,  0.05f, 1.0f  },
    { "BRIGHTNESS_CAP_B", Param::F32, &bloomBrightnessCapB,  0.05f, 1.0f  },
    { "SURPRISE_RISE_TAU", Param::F32, &surpriseRiseTau,      0.1f,  5.0f  },
    { "SURPRISE_FALL_TAU", Param::F32, &surpriseFallTau,     0.05f, 2.0f  },
    { "SURPRISE_RATIO",    Param::F32, &surpriseRatio,       1.5f,  10.0f },
    { "GYRO_WEIGHT",       Param::F32, &gyroWeight,          0.0f,  0.1f  },
    { "MOTION_GAIN",       Param::F32, &motionGain,          0.5f,  20.0f },
    { "BUFFER_DRAIN",      Param::F32, &bloomBufferDrain,    0.5f,  20.0f },
    { "FLASH_DECAY_RATE",  Param::F32, &bloomFlashDecayRate, 0.2f,  10.0f },
    { "BREATH_FLOOR",      Param::F32, &bloomBreathFloor,    0.0f,  0.5f  },
};
static const size_t PARAM_COUNT = sizeof(PARAMS) / sizeof(PARAMS[0]);

static void dumpParam(const Param &p) {
    Serial.printf("[PARAM] %s=%.4f\n", p.name, *(float*)p.ptr);
}

static void dumpAllParams() {
    for (size_t i = 0; i < PARAM_COUNT; i++) dumpParam(PARAMS[i]);
}

static void setParamFromLine(const char *kv) {
    while (*kv == ' ') kv++;
    const char *eq = strchr(kv, '=');
    if (!eq) { Serial.println("[PARAM] bad syntax"); return; }
    size_t nameLen = (size_t)(eq - kv);
    const char *valStr = eq + 1;
    for (size_t i = 0; i < PARAM_COUNT; i++) {
        const Param &p = PARAMS[i];
        if (strlen(p.name) == nameLen && strncmp(p.name, kv, nameLen) == 0) {
            float v = atof(valStr);
            if (v < p.lo) v = p.lo;
            if (v > p.hi) v = p.hi;
            *(float*)p.ptr = v;
            dumpParam(p);
            return;
        }
    }
    Serial.printf("[PARAM] unknown key\n");
}

static void processLine(const char *line, uint8_t len) {
    if (len >= 2 && line[0] == '!' && line[1] == 'P') {
        if (len >= 3 && line[2] == '?') { dumpAllParams(); return; }
        if (len >= 4) { setParamFromLine(line + 2); return; }
    }
}

static char serialBuf[80];
static uint8_t serialBufLen = 0;

static void parseSerialCommands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBufLen > 0) {
                serialBuf[serialBufLen] = '\0';
                processLine(serialBuf, serialBufLen);
                serialBufLen = 0;
            }
        } else if (c >= 32 && c < 127 && serialBufLen < sizeof(serialBuf) - 1) {
            serialBuf[serialBufLen++] = c;
        }
    }
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    // Init strips
    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0)); strip2.ClearTo(RgbColor(0));
    strip3.ClearTo(RgbColor(0)); strip4.ClearTo(RgbColor(0)); strip5.ClearTo(RgbColor(0));
    showAll();

    // Init bloom state
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        resetBloomStrip(bloom[s]);
    }

    // WiFi + ESP-NOW
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed");
        return;
    }
    esp_now_register_recv_cb(onReceive);

    Serial.printf("Biolum bloom ready — ch=%u, %u strips × %u LEDs\n",
                  FIXED_CHANNEL, NUM_STRIPS, LEDS_PER_STRIP);
    Serial.println("Sender map:");
    for (int i = 0; i < NUM_STRIPS; i++) {
        Serial.printf("  strip %d (pin %d) ← XX:XX:XX:XX:%02X:%02X\n",
                      i,
                      (i == 0) ? 4 : (i == 1) ? 16 : (i == 2) ? 17 :
                      (i == 3) ? 5 : (i == 4) ? 18 : 19,
                      SENDER_MAP[i].mac4, SENDER_MAP[i].mac5);
    }
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    uint32_t t0 = micros();

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        if (senderPktCount[s] != bloom[s].prevPktCount) {
            bloomProcessMotion(s);
            bloom[s].prevPktCount = senderPktCount[s];
        }
        renderBloomStrip(s, dt);
    }

    uint32_t t1 = micros();
    showAll();
    uint32_t t2 = micros();

    static uint32_t renderUsAccum = 0, showUsAccum = 0;
    renderUsAccum += (t1 - t0);
    showUsAccum   += (t2 - t1);

    // Process ESP-NOW commands
    if (espnowCmdPending) {
        processLine(espnowCmdBuf, espnowCmdLen);
        espnowCmdPending = false;
    }

    parseSerialCommands();

    // Status logging
    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    if (now - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (now - lastLogMs);
        float avgRenderUs = (float)renderUsAccum / frameCount;
        float avgShowUs   = (float)showUsAccum / frameCount;
        Serial.printf("  FPS=%.1f  render=%.0fus  show=%.0fus  total=%.0fus\n",
                      fps, avgRenderUs, avgShowUs, avgRenderUs + avgShowUs);
        frameCount = 0;
        renderUsAccum = 0;
        showUsAccum = 0;
        lastLogMs = now;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            bool active = senderLastMs[s] > 0 && (now - senderLastMs[s] < TIMEOUT_MS);
            uint16_t mid = LEDS_PER_STRIP / 2;
            float ph = bloom[s].breathPhase[mid];
            float breathVal = (fastSinPhase(ph) * 0.5f + 0.5f);
            float breathGlow = bloomBreathFloor + breathVal * (bloom[s].breathPeak[mid] - bloomBreathFloor);
            float breathLin = fastGamma24(breathGlow) * ((mid < 65) ? bloomBrightnessCapA : bloomBrightnessCapB);
            float h = bloom[s].hueT[mid];
            float sampleG = lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h) * breathLin;
            Serial.printf("  [%d] %s pkts=%lu flash=%.4f buf=%.4f ema=%.3f gate=%d\n",
                          s, active ? "LIVE" : "----",
                          (unsigned long)senderPktCount[s], bloom[s].flash,
                          bloom[s].energyBuffer, bloom[s].motionEma,
                          bloom[s].gateOff[mid] ? 1 : 0);
        }
    }
}
