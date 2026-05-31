// Biolum mixed — ambient bloom + crawl creatures drifting across 6 strips.
// Ported from audio-reactive/effects/biolum_mixed.py.
//
// Interactive features (from single gyro sender via ESP-NOW):
//   Shake (accel) → creatures disintegrate: pixels scatter outward, color
//     shifts to rainbow. 3s sustained → fade out, 5s → all die. Reverses
//     smoothly on stop.
//   Rotation (gyro Z) → scrolls the entire animation left/right. Angular
//     velocity integration with decay. Full rotation ≈ 20% strip shift.
//     Renders into 200px buffer with 25px margin each side.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <NeoPixelBus.h>
#include <math.h>
#include "v1_packet.h"
#include "gyro_packet_v1.h"

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 150
#endif

#define VIRTUAL_LEDS   200
#define SCROLL_MARGIN  25

#define NUM_PAIRS     3
#define NUM_STRIPS    6
#define MAX_CREATURES 7

static const uint8_t STRIP_PINS[NUM_STRIPS] = { 4, 15, 17, 5, 18, 19 };

// --- Color (teal/cyan) ---
static const float COLOR_R = 0.0f;
static const float COLOR_G = 180.0f;
static const float COLOR_B = 220.0f;

// --- Drift ---
static constexpr float PULSE_EXPANSION_SPEED = 3.3f;
static constexpr float PULSE_TO_DRIFT_RATIO  = 1.9f;
static constexpr float AVG_DRIFT = PULSE_EXPANSION_SPEED / PULSE_TO_DRIFT_RATIO;
static constexpr float DRIFT_SPREAD = 0.33f;
static constexpr float DRIFT_MIN = AVG_DRIFT * (1.0f - DRIFT_SPREAD);
static constexpr float DRIFT_MAX = AVG_DRIFT * (1.0f + DRIFT_SPREAD);

static constexpr int SPAWN_MARGIN   =  5;
static constexpr int DESPAWN_MARGIN = -5;

// --- Bloom params ---
static constexpr float BLOOM_RADIUS        = 3.0f;
static constexpr float BLOOM_EDGE_SOFTNESS = 0.8f;
static constexpr float BLOOM_RISE          = 2.0f;
static constexpr float BLOOM_HOLD          = 1.5f;
static constexpr float BLOOM_FALL          = 5.0f;
static constexpr float BLOOM_TOTAL         = BLOOM_RISE + BLOOM_HOLD + BLOOM_FALL;
static constexpr float BLOOM_EMIT_LO       = 1.2f;
static constexpr float BLOOM_EMIT_HI       = 2.8f;

// --- Crawl params ---
static constexpr float CRAWL_RADIUS         = 8.0f;
static constexpr float CRAWL_PULSE_LIFETIME = CRAWL_RADIUS / PULSE_EXPANSION_SPEED;
static constexpr float CRAWL_PULSE_FADE     = 1.4f;
static constexpr float CRAWL_TAIL_DECAY     = 4.0f;
static constexpr float CRAWL_EMIT_LO        = 1.2f;
static constexpr float CRAWL_EMIT_HI        = 2.8f;

// --- Max anims per creature ---
#define MAX_ANIMS 6


// --- Shake params ---
static constexpr float SHAKE_THRESH_G       = 2.0f;
static constexpr float SHAKE_BUFFER_GAIN    = 0.8f;           // energy per g above threshold per second
static constexpr float SHAKE_BUFFER_MAX     = 0.5f;
static constexpr float SHAKE_DRAIN_RATE     = 0.8f;           // drain scale factor (sqrt mode)
static constexpr float SHAKE_BASE_FALL      = 0.5f;           // base fall rate (per second)
static constexpr float SHAKE_FALL_SLOWDOWN  = 0.5f;           // extra seconds of recovery per second shaken
static constexpr float SHAKE_SCATTER_RADIUS = 1.5f;           // max scatter = 1.5× creature half-width each side
static constexpr float SHAKE_BRIGHT_BOOST   = 0.8f;           // extra brightness at max shake
static constexpr float SHAKE_HUE_DRIFT      = 90.0f;          // degrees/sec hue rotation at max shake

// --- Gyro scroll params ---
// At ±1000 dps, int8 gx_mean LSB = 7.8 dps. We need counts_per_dps to decode.
static constexpr float COUNTS_PER_DPS = 32.768f;  // 32768 / 1000
// Full rotation (360°) → SCROLL_MARGIN pixels of shift
static constexpr float SCROLL_GAIN    = (float)SCROLL_MARGIN / 360.0f;  // px per degree
static constexpr float SCROLL_DECAY   = 0.3f;  // decay time constant in seconds

// --- ESP-NOW ---
#define FIXED_CHANNEL 6
#define SENSOR_HZ     25.0f
#define TIMEOUT_MS    500

// F8:08 sender MAC last two bytes
static const uint8_t SENDER_MAC4 = 0xF8;
static const uint8_t SENDER_MAC5 = 0x08;

// v1 telemetry decode
#define MAG_FS 57000.0f
static inline float amagGFromByte(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * MAG_FS / (32768.0f / 4.0f);  // ±4g
}

// --- Types ---
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
    float        hueOffset;       // random starting hue for this creature's rainbow
    float        hueSweep;        // degrees of hue range (60-90° for smooth 2-3 color wash)
};

// --- Per-pair state (3 logical strips) ---
struct PairState {
    Creature creatures[MAX_CREATURES];
    float    buf[VIRTUAL_LEDS];
    float    bufR[VIRTUAL_LEDS];
    float    bufG[VIRTUAL_LEDS];
    float    bufB[VIRTUAL_LEDS];
};

static PairState pairs[NUM_PAIRS];

// --- Global shake / scroll state ---
static float shakeLevel = 0.0f;       // smooth 0→1 controlling all visual effects
static float shakeBuffer = 0.0f;      // energy buffer (spiky input → smooth drain)
static float shakeTime = 0.0f;        // accumulated seconds of shaking (for proportional recovery)
static float scrollOffset = 0.0f;
static bool  shakeActive = false;
static float shakeCooldown = 2.0f;    // seconds since shakeLevel dropped below 0.2 (lifecycle resumes after 1s)
static bool  lifecycleFrozen = false;
static float shakeHueDrift = 0.0f;    // accumulated hue rotation from sustained shake

// --- ESP-NOW received data ---
static volatile uint32_t senderLastMs   = 0;
static volatile uint8_t  senderAmag     = 0;
static volatile int8_t   senderGzMean   = 0;
static volatile uint32_t senderPktCount = 0;
static volatile bool     gotGyroPacket  = false;

void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (mac[4] != SENDER_MAC4 || mac[5] != SENDER_MAC5) return;

    if (len == sizeof(TelemetryPacketV1)) {
        TelemetryPacketV1 pkt;
        memcpy(&pkt, data, sizeof(pkt));
        senderAmag = pkt.amag_max;
        senderLastMs = millis();
        senderPktCount++;
    } else if (len == sizeof(GyroPacketV1)) {
        GyroPacketV1 gpkt;
        memcpy(&gpkt, data, sizeof(gpkt));
        senderGzMean = gpkt.gz_mean;
        gotGyroPacket = true;
    }
}

// --- NeoPixelBus strip objects ---
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod>* strip0;
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod>* strip1;
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod>* strip2;
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod>* strip3;
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod>* strip4;
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod>* strip5;

static inline float randf(float lo, float hi) {
    return lo + (float)random(10000) / 10000.0f * (hi - lo);
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// HSV to RGB (h in 0..360, s and v in 0..1, output 0..255)
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

static void initCreature(Creature& c) {
    c.kind = random(2) ? KIND_BLOOM : KIND_CRAWL;
    float speed = randf(DRIFT_MIN, DRIFT_MAX);
    c.pos = randf(SPAWN_MARGIN, VIRTUAL_LEDS - SPAWN_MARGIN);
    c.vel = random(2) ? speed : -speed;
    c.alive = true;

    if (c.kind == KIND_BLOOM) {
        c.emitTimer = randf(BLOOM_EMIT_LO, BLOOM_EMIT_HI);
    } else {
        c.emitTimer = randf(CRAWL_EMIT_LO, CRAWL_EMIT_HI);
    }

    for (int i = 0; i < MAX_ANIMS; i++) c.anims[i].active = false;
    c.hueOffset = randf(0.0f, 360.0f);
    c.hueSweep = randf(60.0f, 90.0f);
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
            float base = BLOOM_TOTAL + randf(BLOOM_EMIT_LO, BLOOM_EMIT_HI);
            c.emitTimer = base * randf(0.33f, 2.0f);
        } else {
            c.emitTimer = CRAWL_PULSE_LIFETIME + CRAWL_PULSE_FADE
                          + randf(CRAWL_EMIT_LO, CRAWL_EMIT_HI);
        }
    }

    float maxAge = (c.kind == KIND_BLOOM) ? BLOOM_TOTAL
                                          : (CRAWL_PULSE_LIFETIME + CRAWL_PULSE_FADE);
    for (int i = 0; i < MAX_ANIMS; i++) {
        if (!c.anims[i].active) continue;
        c.anims[i].age += dt;
        if (c.anims[i].age >= maxAge) c.anims[i].active = false;
    }

}

// Sub-pixel screen blend: deposits brightness split across two adjacent pixels
static inline void blendSubpixel(float* bufR, float* bufG, float* bufB,
                                 float pos, float normR, float normG, float normB) {
    if (pos < 0.0f || pos >= (float)(VIRTUAL_LEDS - 1)) return;
    int lo = (int)pos;
    float frac = pos - (float)lo;
    float wLo = 1.0f - frac;
    float wHi = frac;

    float rLo = normR * wLo, gLo = normG * wLo, bLo = normB * wLo;
    float rHi = normR * wHi, gHi = normG * wHi, bHi = normB * wHi;

    bufR[lo] = bufR[lo] + rLo - bufR[lo] * rLo;
    bufG[lo] = bufG[lo] + gLo - bufG[lo] * gLo;
    bufB[lo] = bufB[lo] + bLo - bufB[lo] * bLo;
    if (lo + 1 < VIRTUAL_LEDS) {
        bufR[lo+1] = bufR[lo+1] + rHi - bufR[lo+1] * rHi;
        bufG[lo+1] = bufG[lo+1] + gHi - bufG[lo+1] * gHi;
        bufB[lo+1] = bufB[lo+1] + bHi - bufB[lo+1] * bHi;
    }
}

static void renderBloom(const Creature& c, float* buf, float* bufR, float* bufG, float* bufB) {
    float center = c.pos;
    float halfWidth = BLOOM_RADIUS + BLOOM_EDGE_SOFTNESS;
    // Total span expands with shake: original halfWidth stretched by scatter
    float totalSpan = halfWidth * (1.0f + shakeLevel * SHAKE_SCATTER_RADIUS);
    int lo = max(0, (int)(center - totalSpan - 1));
    int hi = min(VIRTUAL_LEDS - 1, (int)(center + totalSpan + 1));

    float brightMult = 1.0f + shakeLevel * SHAKE_BRIGHT_BOOST;
    float colorT = shakeLevel;

    for (int a = 0; a < MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float age = c.anims[a].age;

        float envelope;
        if (age < BLOOM_RISE) {
            envelope = age / BLOOM_RISE;
        } else if (age < BLOOM_RISE + BLOOM_HOLD) {
            envelope = 1.0f;
        } else {
            float fallT = (age - BLOOM_RISE - BLOOM_HOLD) / BLOOM_FALL;
            envelope = fmaxf(0.0f, 1.0f - fallT);
            envelope *= envelope;
        }
        if (envelope < 0.003f) continue;

        for (int i = lo; i <= hi; i++) {
            // Reverse map: output pixel -> creature-local coordinate
            float relOut = ((float)i - center) / totalSpan;  // -1..+1 in output space
            if (fabsf(relOut) > 1.0f) continue;

            // Map back to source position for brightness calc
            // At shake=0, totalSpan=halfWidth so srcDist = |relOut| * halfWidth (original)
            // At shake=1, output is stretched but source brightness comes from compressed position
            float srcDist = fabsf(relOut) * halfWidth;

            float spatial;
            if (srcDist <= BLOOM_RADIUS) {
                spatial = 1.0f;
            } else {
                spatial = expf(-(srcDist - BLOOM_RADIUS) / BLOOM_EDGE_SOFTNESS);
            }

            // Dark gap in center at high shake
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

            // Color: blend from base to rainbow based on shakeLevel
            float pixR = COLOR_R, pixG = COLOR_G, pixB = COLOR_B;
            if (shakeLevel > 0.01f) {
                float hueFrac = (relOut + 1.0f) * 0.5f;
                float hue = fmodf(c.hueOffset + shakeHueDrift + hueFrac * c.hueSweep, 360.0f);
                float rainR, rainG, rainB;
                hsvToRgb(hue, 1.0f, 1.0f, rainR, rainG, rainB);
                pixR = COLOR_R * (1.0f - colorT) + rainR * colorT;
                pixG = COLOR_G * (1.0f - colorT) + rainG * colorT;
                pixB = COLOR_B * (1.0f - colorT) + rainB * colorT;
            }

            // Screen blend directly into buffer (no subpixel needed — we're at integer positions)
            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void renderCrawl(const Creature& c, float* buf, float* bufR, float* bufG, float* bufB) {
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

            // Reverse map to source distance
            float srcDist = fabsf(relOut) * halfWidth;
            if (srcDist > radius) continue;

            float behind = radius - srcDist;
            float br = expf(-behind / CRAWL_TAIL_DECAY);
            if (behind < 1.0f) br *= behind;

            // Dark gap in center
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

            float pixR = COLOR_R, pixG = COLOR_G, pixB = COLOR_B;
            if (shakeLevel > 0.01f) {
                float hueFrac = (relOut + 1.0f) * 0.5f;
                float hue = fmodf(c.hueOffset + shakeHueDrift + hueFrac * c.hueSweep, 360.0f);
                float rainR, rainG, rainB;
                hsvToRgb(hue, 1.0f, 1.0f, rainR, rainG, rainB);
                pixR = COLOR_R * (1.0f - colorT) + rainR * colorT;
                pixG = COLOR_G * (1.0f - colorT) + rainG * colorT;
                pixB = COLOR_B * (1.0f - colorT) + rainB * colorT;
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

static void renderCreature(const Creature& c, float* buf, float* bufR, float* bufG, float* bufB) {
    if (!c.alive) return;
    if (c.kind == KIND_BLOOM) renderBloom(c, buf, bufR, bufG, bufB);
    else                      renderCrawl(c, buf, bufR, bufG, bufB);
}

static inline void setPixel(int strip, int pixel, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor color(r, g, b);
    switch (strip) {
        case 0: strip0->SetPixelColor(pixel, color); break;
        case 1: strip1->SetPixelColor(pixel, color); break;
        case 2: strip2->SetPixelColor(pixel, color); break;
        case 3: strip3->SetPixelColor(pixel, color); break;
        case 4: strip4->SetPixelColor(pixel, color); break;
        case 5: strip5->SetPixelColor(pixel, color); break;
    }
}

static inline void showStrip(int s) {
    switch (s) {
        case 0: strip0->Show(); break;
        case 1: strip1->Show(); break;
        case 2: strip2->Show(); break;
        case 3: strip3->Show(); break;
        case 4: strip4->Show(); break;
        case 5: strip5->Show(); break;
    }
}

// --- Shake + scroll processing ---
static void processInteraction(float dt) {
    uint32_t now = millis();
    bool sensorLive = (senderLastMs > 0 && (now - senderLastMs) < TIMEOUT_MS);

    // Feed energy buffer from accel
    if (sensorLive) {
        float aG = amagGFromByte(senderAmag);
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

    // Drain buffer proportional to sqrt(buffer) — fast when full, gentle when low
    float drained = fminf(shakeBuffer, SHAKE_DRAIN_RATE * sqrtf(shakeBuffer) * dt);
    shakeBuffer -= drained;

    if (shakeActive) {
        shakeLevel += drained;
        if (shakeLevel > 1.0f) shakeLevel = 1.0f;
    } else {
        // Not shaking: level decays (proportional to how long we shook)
        float fallRate = SHAKE_BASE_FALL / (1.0f + shakeTime * SHAKE_FALL_SLOWDOWN);
        shakeLevel -= fallRate * dt;
        if (shakeLevel <= 0.0f) {
            shakeLevel = 0.0f;
            shakeTime = 0.0f;
        }
    }

    // Lifecycle pause: freeze creatures during significant shake + 1s cooldown
    if (shakeLevel > 0.2f) {
        shakeCooldown = 0.0f;
    } else {
        shakeCooldown += dt;
    }
    lifecycleFrozen = (shakeLevel > 0.2f || shakeCooldown < 0.3f);

    // Hue drift during sustained shake — keeps visuals alive at max
    shakeHueDrift += shakeLevel * SHAKE_HUE_DRIFT * dt;
    if (shakeHueDrift > 360.0f) shakeHueDrift -= 360.0f;

    // Gyro scroll disabled — shake makes gyro too noisy.
    if (gotGyroPacket) {
        gotGyroPacket = false;
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("\nbiolum_mixed — interactive creatures on 6 strips");

    randomSeed(analogRead(0) ^ (micros() << 16));

    strip0 = new NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod>(LEDS_PER_STRIP, STRIP_PINS[0]);
    strip1 = new NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod>(LEDS_PER_STRIP, STRIP_PINS[1]);
    strip2 = new NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod>(LEDS_PER_STRIP, STRIP_PINS[2]);
    strip3 = new NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod>(LEDS_PER_STRIP, STRIP_PINS[3]);
    strip4 = new NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod>(LEDS_PER_STRIP, STRIP_PINS[4]);
    strip5 = new NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod>(LEDS_PER_STRIP, STRIP_PINS[5]);

    strip0->Begin(); strip1->Begin(); strip2->Begin();
    strip3->Begin(); strip4->Begin(); strip5->Begin();

    for (int p = 0; p < NUM_PAIRS; p++) {
        for (int c = 0; c < MAX_CREATURES; c++) {
            initCreature(pairs[p].creatures[c]);
        }
    }

    // WiFi + ESP-NOW
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
    } else {
        esp_now_register_recv_cb(onReceive);
        Serial.printf("ESP-NOW ready ch=%d, sender=XX:XX:XX:XX:%02X:%02X\n",
                      FIXED_CHANNEL, SENDER_MAC4, SENDER_MAC5);
    }

    Serial.printf("  %d pairs × %d LEDs (virtual=%d, margin=%d), %d creatures each\n",
                  NUM_PAIRS, LEDS_PER_STRIP, VIRTUAL_LEDS, SCROLL_MARGIN, MAX_CREATURES);
}

static uint32_t lastMicros = 0;

void loop() {
    uint32_t now = micros();
    float dt = (lastMicros == 0) ? 0.016f : (now - lastMicros) * 1e-6f;
    if (dt > 0.1f) dt = 0.016f;
    lastMicros = now;

    processInteraction(dt);

    int scrollPixels = (int)roundf(scrollOffset);

    for (int p = 0; p < NUM_PAIRS; p++) {
        PairState& ps = pairs[p];

        memset(ps.bufR, 0, sizeof(ps.bufR));
        memset(ps.bufG, 0, sizeof(ps.bufG));
        memset(ps.bufB, 0, sizeof(ps.bufB));

        for (int c = 0; c < MAX_CREATURES; c++) {
            if (!lifecycleFrozen) updateCreature(ps.creatures[c], dt);
            renderCreature(ps.creatures[c], ps.buf, ps.bufR, ps.bufG, ps.bufB);
        }

        for (int c = 0; c < MAX_CREATURES; c++) {
            if (!ps.creatures[c].alive) initCreature(ps.creatures[c]);
        }


        int pinA = p * 2;
        int pinB = p * 2 + 1;

        // Output: read from virtual buffer with scroll offset, write to physical strip
        for (int i = 0; i < LEDS_PER_STRIP; i++) {
            int srcIdx = SCROLL_MARGIN + i + scrollPixels;
            if (srcIdx < 0) srcIdx = 0;
            if (srcIdx >= VIRTUAL_LEDS) srcIdx = VIRTUAL_LEDS - 1;

            float vR = ps.bufR[srcIdx];
            float vG = ps.bufG[srcIdx];
            float vB = ps.bufB[srcIdx];
            // Clamp
            if (vR > 1.0f) vR = 1.0f;
            if (vG > 1.0f) vG = 1.0f;
            if (vB > 1.0f) vB = 1.0f;
            // Gamma
            vR = vR * vR;
            vG = vG * vG;
            vB = vB * vB;

            uint8_t r = (uint8_t)(vR * 255.0f);
            uint8_t g = (uint8_t)(vG * 255.0f);
            uint8_t b = (uint8_t)(vB * 255.0f);
            setPixel(pinA, i, r, g, b);
            setPixel(pinB, i, r, g, b);
        }

        showStrip(pinA);
        showStrip(pinB);
    }

    // --- Diagnostic prints (every 2s) ---
    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    uint32_t nowMs = millis();
    if (nowMs - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (nowMs - lastLogMs);
        bool sensorLive = (senderLastMs > 0 && (nowMs - senderLastMs) < TIMEOUT_MS);
        float aG = amagGFromByte(senderAmag);
        float gzDps = (float)senderGzMean * 256.0f / COUNTS_PER_DPS;

        Serial.printf("FPS=%.1f  sensor=%s pkts=%lu  amag=%.2fg  gz=%.1fdps\n",
                      fps, sensorLive ? "LIVE" : "----",
                      (unsigned long)senderPktCount, aG, gzDps);
        Serial.printf("  shake=%.3f buf=%.3f time=%.1fs %s  alive=",
                      shakeLevel, shakeBuffer, shakeTime,
                      shakeActive ? "SHAKING" : "idle");

        // Count alive creatures across all pairs
        for (int p = 0; p < NUM_PAIRS; p++) {
            int alive = 0;
            for (int c = 0; c < MAX_CREATURES; c++) {
                if (pairs[p].creatures[c].alive) alive++;
            }
            Serial.printf("%d ", alive);
        }
        Serial.println();

        frameCount = 0;
        lastLogMs = nowMs;
    }
}
