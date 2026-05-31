/*
 * BIOLUM Board B — 4 ambient bloom strips + 2 creature strips on one ESP32.
 *
 * Classic ESP32, 6 × WS2812B RGB strips on GPIO 4, 16, 17, 5, 18, 19.
 *
 * Strips 0-3 (pins 4, 16, 17, 5): ambient bloom breathing — same algorithm
 *   as board A but NO gyro/motion response. Breathing only, 100 LEDs each.
 *
 * Strips 4-5 (pins 18, 19): bioluminescent creature animation with shake
 *   interaction from the F8:08 sender. 150 LEDs each, mirrored pair (one
 *   buffer). Virtual 200px buffer with 25px scroll margin each side.
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

#define CREATURE_LEDS    150
#define SCROLL_MARGIN     25
#define VIRTUAL_LEDS    (CREATURE_LEDS + 2 * SCROLL_MARGIN)

static const uint8_t NUM_STRIPS      = 6;
static const uint8_t NUM_BLOOM_STRIPS = 4;
#define MAX_CREATURES 7
#define MAX_ANIMS     6

// ── Bloom parameters (runtime-tunable via operator) ─────────────
static float bloomBrightnessCapA  = 0.15f;   // LEDs 0–64
static float bloomBrightnessCapB  = 0.15f;   // LEDs 65–99
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

static float bloomDormancyMin  = 3.0f;
static float bloomDormancyMax  = 7.0f;
static float bloomDormancyFrac = 0.0f;

#define GATE_ON_THRESH  0.7f
#define GATE_OFF_THRESH 1.2f

#define SENSOR_HZ       25.0f
#define TIMEOUT_MS     500

// ── Creature bloom constants (CBOOM_ prefix avoids BLOOM_ collision) ─
static constexpr float CREATURE_COLOR_R = 0.0f;
static constexpr float CREATURE_COLOR_G = 180.0f;
static constexpr float CREATURE_COLOR_B = 220.0f;

static constexpr float PULSE_EXPANSION_SPEED = 3.3f;
static constexpr float PULSE_TO_DRIFT_RATIO  = 1.9f;
static constexpr float AVG_DRIFT = PULSE_EXPANSION_SPEED / PULSE_TO_DRIFT_RATIO;
static constexpr float DRIFT_SPREAD = 0.33f;
static constexpr float DRIFT_MIN = AVG_DRIFT * (1.0f - DRIFT_SPREAD);
static constexpr float DRIFT_MAX = AVG_DRIFT * (1.0f + DRIFT_SPREAD);

static constexpr int SPAWN_MARGIN   =  5;
static constexpr int DESPAWN_MARGIN = -5;

static constexpr float CBOOM_RADIUS        = 3.0f;
static constexpr float CBOOM_EDGE_SOFTNESS = 0.8f;
static constexpr float CBOOM_RISE          = 2.0f;
static constexpr float CBOOM_HOLD          = 1.5f;
static constexpr float CBOOM_FALL          = 5.0f;
static constexpr float CBOOM_TOTAL         = CBOOM_RISE + CBOOM_HOLD + CBOOM_FALL;
static constexpr float CBOOM_EMIT_LO       = 1.2f;
static constexpr float CBOOM_EMIT_HI       = 2.8f;

static constexpr float CRAWL_RADIUS         = 8.0f;
static constexpr float CRAWL_PULSE_LIFETIME = CRAWL_RADIUS / PULSE_EXPANSION_SPEED;
static constexpr float CRAWL_PULSE_FADE     = 1.4f;
static constexpr float CRAWL_TAIL_DECAY     = 4.0f;
static constexpr float CRAWL_EMIT_LO        = 1.2f;
static constexpr float CRAWL_EMIT_HI        = 2.8f;

// ── Shake params ─────────────────────────────────────────────────
static constexpr float SHAKE_THRESH_G       = 2.0f;
static constexpr float SHAKE_BUFFER_GAIN    = 0.8f;
static constexpr float SHAKE_BUFFER_MAX     = 0.5f;
static constexpr float SHAKE_DRAIN_RATE     = 0.8f;
static constexpr float SHAKE_BASE_FALL      = 0.5f;
static constexpr float SHAKE_FALL_SLOWDOWN  = 0.5f;
static constexpr float SHAKE_SCATTER_RADIUS = 1.5f;
static constexpr float SHAKE_BRIGHT_BOOST   = 0.8f;
static constexpr float SHAKE_HUE_DRIFT      = 90.0f;

// ── v1 telemetry decode (bloom strips) ───────────────────────────
#define MAG_FS           57000.0f
#define COUNTS_PER_G      8192.0f

static inline float amagGFromByte_bloom(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * MAG_FS / COUNTS_PER_G;
}

#define COUNTS_PER_DPS 32.8f

static inline float gmagDpsFromByte(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * MAG_FS / COUNTS_PER_DPS;
}

// ── Creature accel decode (±4g range, different formula) ─────────
static inline float amagGFromByte_creature(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * MAG_FS / (32768.0f / 4.0f);
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

static inline float randFloatRange(float lo, float hi) {
    return lo + randFloat() * (hi - lo);
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// ── HSV to RGB (h 0..360, s/v 0..1, output 0..255) ─────────────
static void hsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r1, g1, b1;
    if      (h < 60)  { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else              { r1 = c; g1 = 0; b1 = x; }
    r = (r1 + m) * 255.0f;
    g = (g1 + m) * 255.0f;
    b = (b1 + m) * 255.0f;
}

// ── Gyro sender MAC → bloom strip mapping (strips 0-3 only) ────
struct SenderDef {
    uint8_t mac4;
    uint8_t mac5;
};

static const SenderDef SENDER_MAP[NUM_BLOOM_STRIPS] = {
    { 0x0B, 0x88 },  // strip 0 (pin 4)  — 14:63:93:70:0B:88
    { 0x93, 0xB0 },  // strip 1 (pin 16) — 14:63:93:6E:93:B0
    { 0xAE, 0x4C },  // strip 2 (pin 17) — 14:63:93:6E:AE:4C
    { 0xAA, 0x2C },  // strip 3 (pin 5)  — 14:63:93:6E:AA:2C
};

// ── Creature sender (F8:08) ─────────────────────────────────────
static const uint8_t CREATURE_SENDER_MAC4 = 0xF8;
static const uint8_t CREATURE_SENDER_MAC5 = 0x08;

// ── ESP-NOW command dispatch ─────────────────────────────────────
#define CMD_PKT_MAGIC 0xC0
static volatile bool espnowCmdPending = false;
static char espnowCmdBuf[64];
static volatile uint8_t espnowCmdLen = 0;

// ── Per-sender ESP-NOW state (bloom strips 0-3) ─────────────────
static volatile uint32_t senderLastMs[NUM_BLOOM_STRIPS] = {0};
static volatile uint8_t  senderPeakAmag[NUM_BLOOM_STRIPS] = {0};
static volatile uint8_t  senderPeakGmag[NUM_BLOOM_STRIPS] = {0};
static volatile uint32_t senderPktCount[NUM_BLOOM_STRIPS] = {0};
static volatile uint16_t senderSeq[NUM_BLOOM_STRIPS] = {0};
static volatile uint16_t senderPrevSeq[NUM_BLOOM_STRIPS] = {0};

// ── Creature sender ESP-NOW state ───────────────────────────────
static volatile uint32_t creatureSenderLastMs  = 0;
static volatile uint8_t  creatureSenderAmag    = 0;
static volatile uint32_t creatureSenderPktCount = 0;

void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    // Command packets
    if (len != sizeof(TelemetryPacketV1) && len >= 2 && data[0] == CMD_PKT_MAGIC && !espnowCmdPending) {
        uint8_t cmdLen = (len - 1 < (int)sizeof(espnowCmdBuf) - 1) ? len - 1 : sizeof(espnowCmdBuf) - 1;
        memcpy(espnowCmdBuf, data + 1, cmdLen);
        espnowCmdBuf[cmdLen] = '\0';
        espnowCmdLen = cmdLen;
        espnowCmdPending = true;
        return;
    }
    if (len != sizeof(TelemetryPacketV1)) return;

    // Check creature sender first
    if (mac[4] == CREATURE_SENDER_MAC4 && mac[5] == CREATURE_SENDER_MAC5) {
        TelemetryPacketV1 pkt;
        memcpy(&pkt, data, sizeof(pkt));
        creatureSenderAmag = pkt.amag_max;
        creatureSenderLastMs = millis();
        creatureSenderPktCount++;
        return;
    }

    // Check bloom senders (strips 0-3)
    for (int i = 0; i < NUM_BLOOM_STRIPS; i++) {
        if (mac[4] == SENDER_MAP[i].mac4 && mac[5] == SENDER_MAP[i].mac5) {
            TelemetryPacketV1 pkt;
            memcpy(&pkt, data, sizeof(TelemetryPacketV1));
            if (pkt.amag_mean > senderPeakAmag[i]) senderPeakAmag[i] = pkt.amag_mean;
            if (pkt.gmag_mean > senderPeakGmag[i]) senderPeakGmag[i] = pkt.gmag_mean;
            senderPrevSeq[i] = senderSeq[i];
            senderSeq[i] = pkt.seq;
            senderLastMs[i] = millis();
            senderPktCount[i]++;
            return;
        }
    }
}

// ── LED driver: 6 strips via RMT ─────────────────────────────────
// Strips 0-3: LEDS_PER_STRIP (100) LEDs
// Strips 4-5: CREATURE_LEDS (150) LEDs
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 16);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(CREATURE_LEDS,  18);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(CREATURE_LEDS,  19);

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

// ── Per-strip bloom state (strips 0-3 only) ─────────────────────
struct BloomStrip {
    float breathPhase[LEDS_PER_STRIP];
    float breathPeriod[LEDS_PER_STRIP];
    float breathPeak[LEDS_PER_STRIP];
    float hueT[LEDS_PER_STRIP];
    float hueDrift[LEDS_PER_STRIP];
    float blackoutTimer[LEDS_PER_STRIP];
    float dormancyDur[LEDS_PER_STRIP];
    float dormancyRoll[LEDS_PER_STRIP];
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

static BloomStrip bloom[NUM_BLOOM_STRIPS];

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
        bs.dormancyDur[i]   = bloomDormancyMin + randFloat() * (bloomDormancyMax - bloomDormancyMin);
        bs.dormancyRoll[i]  = randFloat();
        bs.gateOff[i]      = false;
        bs.ditherR[i] = 0;
        bs.ditherG[i] = 0;
        bs.ditherB[i] = 0;
    }
}

// ── Bloom render (per strip, ambient only — no motion) ──────────
static void renderBloomStrip(uint8_t s, float dt) {
    BloomStrip &bs = bloom[s];

    // No motion processing — ambient breathing only.
    // Flash/energyBuffer stay at zero (no gyro input).

    float drain = bs.energyBuffer * bloomBufferDrain * dt;
    if (bs.energyBuffer < 0.05f) drain += 0.1f * dt;
    drain = fminf(drain, bs.energyBuffer);
    bs.energyBuffer -= drain;
    bs.flash = fminf(1.0f, bs.flash + drain);

    bs.flash *= expf(-bloomFlashDecayRate * dt);
    if (bs.flash < 0.005f) bs.flash = 0.0f;

    float flashA = bs.flash * bloomBrightnessCapA;
    float flashB = bs.flash * bloomBrightnessCapB;
    float flashFrac = (bs.flash > 0.1f) ? clampf(bs.flash / 0.3f, 0.0f, 1.0f) : 0.0f;
    float oneMinusFF = 1.0f - flashFrac;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        float breath = fastSinPhase(bs.breathPhase[i]) * 0.5f + 0.5f;
        float breathGlow = bloomBreathFloor + breath * (bs.breathPeak[i] - bloomBreathFloor);
        bool zoneB = (i >= 65);
        float cap = zoneB ? bloomBrightnessCapB : bloomBrightnessCapA;
        float breathLin = fastGamma24(breathGlow) * cap;
        float flashLin  = zoneB ? flashB : flashA;

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

        // Hysteresis noise gate with dormancy hold
        float maxCh = fmaxf(fmaxf(oR, oG), oB);
        float thresh = bs.gateOff[i] ? GATE_OFF_THRESH : GATE_ON_THRESH;
        if (maxCh < thresh) {
            oR = oG = oB = 0.0f;
            if (!bs.gateOff[i] && bs.dormancyRoll[i] < bloomDormancyFrac) {
                bs.gateOff[i] = true;
                bs.blackoutTimer[i] = 0.0f;
                bs.dormancyDur[i] = bloomDormancyMin
                    + randFloat() * (bloomDormancyMax - bloomDormancyMin);
            }
        } else if (bs.gateOff[i]) {
            bs.blackoutTimer[i] += dt;
            if (bs.blackoutTimer[i] < bs.dormancyDur[i]) {
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

// ── Creature types ──────────────────────────────────────────────
enum CreatureKind : uint8_t { KIND_BLOOM, KIND_CRAWL };

struct Anim {
    float age;
    bool  active;
};

struct Creature {
    float        pos;
    float        vel;
    CreatureKind kind;
    bool         alive;
    float        emitTimer;
    Anim         anims[MAX_ANIMS];
    float        hueOffset;
    float        hueSweep;
};

// ── Per-pair state (1 pair: strips 4-5 mirror same buffer) ──────
struct PairState {
    Creature creatures[MAX_CREATURES];
    float    buf[VIRTUAL_LEDS];
    float    bufR[VIRTUAL_LEDS];
    float    bufG[VIRTUAL_LEDS];
    float    bufB[VIRTUAL_LEDS];
};

static PairState creaturePair;

// ── Global shake state ──────────────────────────────────────────
static float shakeLevel    = 0.0f;
static float shakeBuffer   = 0.0f;
static float shakeTime     = 0.0f;
static bool  shakeActive   = false;
static float shakeCooldown = 2.0f;
static bool  lifecycleFrozen = false;
static float shakeHueDrift = 0.0f;

// ── Creature lifecycle ──────────────────────────────────────────
static void initCreature(Creature& c) {
    c.kind = (xorshift32() % 2) ? KIND_BLOOM : KIND_CRAWL;
    float speed = randFloatRange(DRIFT_MIN, DRIFT_MAX);
    c.pos = randFloatRange(SPAWN_MARGIN, VIRTUAL_LEDS - SPAWN_MARGIN);
    c.vel = (xorshift32() % 2) ? speed : -speed;
    c.alive = true;

    if (c.kind == KIND_BLOOM) {
        c.emitTimer = randFloatRange(CBOOM_EMIT_LO, CBOOM_EMIT_HI);
    } else {
        c.emitTimer = randFloatRange(CRAWL_EMIT_LO, CRAWL_EMIT_HI);
    }

    for (int i = 0; i < MAX_ANIMS; i++) c.anims[i].active = false;
    c.hueOffset = randFloatRange(0.0f, 360.0f);
    c.hueSweep = randFloatRange(60.0f, 90.0f);
}

static void emitAnim(Creature& c) {
    for (int i = 0; i < MAX_ANIMS; i++) {
        if (!c.anims[i].active) {
            c.anims[i].active = true;
            c.anims[i].age = 0.0f;
            return;
        }
    }
}

static void updateCreature(Creature& c, float dt) {
    if (!c.alive) return;

    c.pos += c.vel * dt;

    if (c.pos < DESPAWN_MARGIN || c.pos > VIRTUAL_LEDS - DESPAWN_MARGIN) {
        c.alive = false;
        return;
    }

    c.emitTimer -= dt;
    if (c.emitTimer <= 0.0f) {
        emitAnim(c);
        if (c.kind == KIND_BLOOM) {
            float base = CBOOM_TOTAL + randFloatRange(CBOOM_EMIT_LO, CBOOM_EMIT_HI);
            c.emitTimer = base * randFloatRange(0.33f, 2.0f);
        } else {
            c.emitTimer = CRAWL_PULSE_LIFETIME + CRAWL_PULSE_FADE
                          + randFloatRange(CRAWL_EMIT_LO, CRAWL_EMIT_HI);
        }
    }

    float maxAge = (c.kind == KIND_BLOOM) ? CBOOM_TOTAL
                                          : (CRAWL_PULSE_LIFETIME + CRAWL_PULSE_FADE);
    for (int i = 0; i < MAX_ANIMS; i++) {
        if (!c.anims[i].active) continue;
        c.anims[i].age += dt;
        if (c.anims[i].age >= maxAge) c.anims[i].active = false;
    }
}

// ── Creature render: bloom ──────────────────────────────────────
static void creatureRenderBloom(const Creature& c, float* bufR, float* bufG, float* bufB) {
    float center = c.pos;
    float halfWidth = CBOOM_RADIUS + CBOOM_EDGE_SOFTNESS;
    float totalSpan = halfWidth * (1.0f + shakeLevel * SHAKE_SCATTER_RADIUS);
    int lo = max(0, (int)(center - totalSpan - 1));
    int hi = min(VIRTUAL_LEDS - 1, (int)(center + totalSpan + 1));

    float brightMult = 1.0f + shakeLevel * SHAKE_BRIGHT_BOOST;
    float colorT = shakeLevel;

    for (int a = 0; a < MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float age = c.anims[a].age;

        float envelope;
        if (age < CBOOM_RISE) {
            envelope = age / CBOOM_RISE;
        } else if (age < CBOOM_RISE + CBOOM_HOLD) {
            envelope = 1.0f;
        } else {
            float fallT = (age - CBOOM_RISE - CBOOM_HOLD) / CBOOM_FALL;
            envelope = fmaxf(0.0f, 1.0f - fallT);
            envelope *= envelope;
        }
        if (envelope < 0.003f) continue;

        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;

            float srcDist = fabsf(relOut) * halfWidth;

            float spatial;
            if (srcDist <= CBOOM_RADIUS) {
                spatial = 1.0f;
            } else {
                spatial = expf(-(srcDist - CBOOM_RADIUS) / CBOOM_EDGE_SOFTNESS);
            }

            float gapFade = 1.0f;
            if (shakeLevel > 0.01f) {
                float gapSize = 0.3f * shakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            float br = envelope * spatial * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR = CREATURE_COLOR_R, pixG = CREATURE_COLOR_G, pixB = CREATURE_COLOR_B;
            if (shakeLevel > 0.01f) {
                float hueFrac = (relOut + 1.0f) * 0.5f;
                float hue = fmodf(c.hueOffset + shakeHueDrift + hueFrac * c.hueSweep, 360.0f);
                float rainR, rainG, rainB;
                hsvToRgb(hue, 1.0f, 1.0f, rainR, rainG, rainB);
                pixR = CREATURE_COLOR_R * (1.0f - colorT) + rainR * colorT;
                pixG = CREATURE_COLOR_G * (1.0f - colorT) + rainG * colorT;
                pixB = CREATURE_COLOR_B * (1.0f - colorT) + rainB * colorT;
            }

            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

// ── Creature render: crawl ──────────────────────────────────────
static void creatureRenderCrawl(const Creature& c, float* bufR, float* bufG, float* bufB) {
    float center = c.pos;
    float halfWidth = CRAWL_RADIUS + CRAWL_TAIL_DECAY;
    float totalSpan = halfWidth * (1.0f + shakeLevel * SHAKE_SCATTER_RADIUS);
    int lo = max(0, (int)(center - totalSpan - 1));
    int hi = min(VIRTUAL_LEDS - 1, (int)(center + totalSpan + 1));

    float brightMult = 1.0f + shakeLevel * SHAKE_BRIGHT_BOOST;
    float colorT = shakeLevel;

    for (int a = 0; a < MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;

        float t = fminf(c.anims[a].age / CRAWL_PULSE_LIFETIME, 1.0f);
        float radius = t * CRAWL_RADIUS;

        float fade = 1.0f;
        if (c.anims[a].age > CRAWL_PULSE_LIFETIME) {
            float fadeT = (c.anims[a].age - CRAWL_PULSE_LIFETIME) / CRAWL_PULSE_FADE;
            fade = fmaxf(0.0f, 1.0f - fadeT);
            fade *= fade;
        }

        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;

            float srcDist = fabsf(relOut) * halfWidth;
            if (srcDist > radius) continue;

            float behind = radius - srcDist;
            float br = expf(-behind / CRAWL_TAIL_DECAY);
            if (behind < 1.0f) br *= behind;

            float gapFade = 1.0f;
            if (shakeLevel > 0.01f) {
                float gapSize = 0.3f * shakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            br *= fade * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR = CREATURE_COLOR_R, pixG = CREATURE_COLOR_G, pixB = CREATURE_COLOR_B;
            if (shakeLevel > 0.01f) {
                float hueFrac = (relOut + 1.0f) * 0.5f;
                float hue = fmodf(c.hueOffset + shakeHueDrift + hueFrac * c.hueSweep, 360.0f);
                float rainR, rainG, rainB;
                hsvToRgb(hue, 1.0f, 1.0f, rainR, rainG, rainB);
                pixR = CREATURE_COLOR_R * (1.0f - colorT) + rainR * colorT;
                pixG = CREATURE_COLOR_G * (1.0f - colorT) + rainG * colorT;
                pixB = CREATURE_COLOR_B * (1.0f - colorT) + rainB * colorT;
            }

            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void renderCreature(const Creature& c, float* bufR, float* bufG, float* bufB) {
    if (!c.alive) return;
    if (c.kind == KIND_BLOOM) creatureRenderBloom(c, bufR, bufG, bufB);
    else                      creatureRenderCrawl(c, bufR, bufG, bufB);
}

// ── Shake processing ────────────────────────────────────────────
static void processShake(float dt) {
    uint32_t now = millis();
    bool sensorLive = (creatureSenderLastMs > 0 && (now - creatureSenderLastMs) < TIMEOUT_MS);

    if (sensorLive) {
        float aG = amagGFromByte_creature(creatureSenderAmag);
        if (aG > SHAKE_THRESH_G) {
            shakeActive = true;
            float excess = aG - SHAKE_THRESH_G;
            shakeBuffer += excess * SHAKE_BUFFER_GAIN * dt;
            if (shakeBuffer > SHAKE_BUFFER_MAX) shakeBuffer = SHAKE_BUFFER_MAX;
            shakeTime += dt;
        } else {
            shakeActive = false;
        }
    }

    float drained = fminf(shakeBuffer, SHAKE_DRAIN_RATE * sqrtf(shakeBuffer) * dt);
    shakeBuffer -= drained;

    if (shakeActive) {
        shakeLevel += drained;
        if (shakeLevel > 1.0f) shakeLevel = 1.0f;
    } else {
        float fallRate = SHAKE_BASE_FALL / (1.0f + shakeTime * SHAKE_FALL_SLOWDOWN);
        shakeLevel -= fallRate * dt;
        if (shakeLevel <= 0.0f) {
            shakeLevel = 0.0f;
            shakeTime = 0.0f;
        }
    }

    if (shakeLevel > 0.2f) {
        shakeCooldown = 0.0f;
    } else {
        shakeCooldown += dt;
    }
    lifecycleFrozen = (shakeLevel > 0.2f || shakeCooldown < 0.3f);

    shakeHueDrift += shakeLevel * SHAKE_HUE_DRIFT * dt;
    if (shakeHueDrift > 360.0f) shakeHueDrift -= 360.0f;
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
    { "DORMANCY_MIN",      Param::F32, &bloomDormancyMin,   0.0f,  60.0f },
    { "DORMANCY_MAX",      Param::F32, &bloomDormancyMax,   0.0f,  60.0f },
    { "DORMANCY_FRAC",     Param::F32, &bloomDormancyFrac,  0.0f,  1.0f  },
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

    // Init bloom state (strips 0-3)
    for (uint8_t s = 0; s < NUM_BLOOM_STRIPS; s++) {
        resetBloomStrip(bloom[s]);
    }

    // Init creatures
    for (int c = 0; c < MAX_CREATURES; c++) {
        initCreature(creaturePair.creatures[c]);
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

    Serial.printf("Biolum board B ready — ch=%u\n", FIXED_CHANNEL);
    Serial.printf("  Bloom: strips 0-3 × %u LEDs (ambient, no motion)\n", LEDS_PER_STRIP);
    Serial.printf("  Creatures: strips 4-5 × %u LEDs (virtual=%d)\n", CREATURE_LEDS, VIRTUAL_LEDS);
    Serial.printf("  Creature sender: XX:XX:XX:XX:%02X:%02X\n",
                  CREATURE_SENDER_MAC4, CREATURE_SENDER_MAC5);
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    uint32_t t0 = micros();

    // ── Bloom strips 0-3 (ambient breathing, no motion) ─────────
    for (uint8_t s = 0; s < NUM_BLOOM_STRIPS; s++) {
        renderBloomStrip(s, dt);
    }

    // ── Creature strips 4-5 (shake interaction) ─────────────────
    processShake(dt);

    PairState& ps = creaturePair;
    memset(ps.bufR, 0, sizeof(ps.bufR));
    memset(ps.bufG, 0, sizeof(ps.bufG));
    memset(ps.bufB, 0, sizeof(ps.bufB));

    for (int c = 0; c < MAX_CREATURES; c++) {
        if (!lifecycleFrozen) updateCreature(ps.creatures[c], dt);
        renderCreature(ps.creatures[c], ps.bufR, ps.bufG, ps.bufB);
    }

    for (int c = 0; c < MAX_CREATURES; c++) {
        if (!ps.creatures[c].alive) initCreature(ps.creatures[c]);
    }

    // Output creature buffer to strips 4 and 5 (mirrored — same pixels)
    for (int i = 0; i < CREATURE_LEDS; i++) {
        int srcIdx = SCROLL_MARGIN + i;
        if (srcIdx >= VIRTUAL_LEDS) srcIdx = VIRTUAL_LEDS - 1;

        float vR = ps.bufR[srcIdx];
        float vG = ps.bufG[srcIdx];
        float vB = ps.bufB[srcIdx];
        if (vR > 1.0f) vR = 1.0f;
        if (vG > 1.0f) vG = 1.0f;
        if (vB > 1.0f) vB = 1.0f;
        // v*v gamma, direct 0-255, no dithering
        vR = vR * vR;
        vG = vG * vG;
        vB = vB * vB;

        uint8_t r = (uint8_t)(vR * 255.0f);
        uint8_t g = (uint8_t)(vG * 255.0f);
        uint8_t b = (uint8_t)(vB * 255.0f);
        setPixel(4, i, r, g, b);
        setPixel(5, i, r, g, b);
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

        // Bloom strip status
        for (uint8_t s = 0; s < NUM_BLOOM_STRIPS; s++) {
            uint16_t mid = LEDS_PER_STRIP / 2;
            Serial.printf("  [bloom %d] gate=%d\n",
                          s, bloom[s].gateOff[mid] ? 1 : 0);
        }

        // Creature status
        bool sensorLive = (creatureSenderLastMs > 0 && (now - creatureSenderLastMs) < TIMEOUT_MS);
        int alive = 0;
        for (int c = 0; c < MAX_CREATURES; c++) {
            if (creaturePair.creatures[c].alive) alive++;
        }
        Serial.printf("  [creatures] %s pkts=%lu shake=%.3f buf=%.3f alive=%d\n",
                      sensorLive ? "LIVE" : "----",
                      (unsigned long)creatureSenderPktCount,
                      shakeLevel, shakeBuffer, alive);
    }
}
