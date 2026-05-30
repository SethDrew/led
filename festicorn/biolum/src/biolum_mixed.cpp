// Biolum mixed — ambient bloom + crawl creatures drifting across 6 strips.
// Ported from audio-reactive/effects/biolum_mixed.py (standalone, no audio/gyro).

#include <Arduino.h>
#include <NeoPixelBus.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 150
#endif

#define NUM_PAIRS     3
#define NUM_STRIPS    6
#define MAX_CREATURES 7

// 3 logical strips, each mirrored across a pair of physical pins.
// Pair 0: pins 4, 15   Pair 1: pins 17, 5   Pair 2: pins 18, 19
static const uint8_t STRIP_PINS[NUM_STRIPS] = { 4, 15, 17, 5, 18, 19 };

// --- Color (teal/cyan) ---
static const uint8_t COLOR_R = 0;
static const uint8_t COLOR_G = 180;
static const uint8_t COLOR_B = 220;

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
};

// --- Per-pair state (3 logical strips) ---
struct PairState {
    Creature creatures[MAX_CREATURES];
    float    buf[LEDS_PER_STRIP];
};

static PairState pairs[NUM_PAIRS];

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

static void initCreature(Creature& c) {
    c.kind = random(2) ? KIND_BLOOM : KIND_CRAWL;
    float speed = randf(DRIFT_MIN, DRIFT_MAX);
    c.pos = randf(SPAWN_MARGIN, LEDS_PER_STRIP - SPAWN_MARGIN);
    c.vel = random(2) ? speed : -speed;
    c.alive = true;

    if (c.kind == KIND_BLOOM) {
        c.emitTimer = randf(BLOOM_EMIT_LO, BLOOM_EMIT_HI);
    } else {
        c.emitTimer = randf(CRAWL_EMIT_LO, CRAWL_EMIT_HI);
    }

    for (int i = 0; i < MAX_ANIMS; i++) c.anims[i].active = false;
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

    if (c.pos < DESPAWN_MARGIN || c.pos > LEDS_PER_STRIP - DESPAWN_MARGIN) {
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

static void renderBloom(const Creature& c, float* buf) {
    float center = c.pos;
    int lo = max(0, (int)(center - BLOOM_RADIUS - BLOOM_EDGE_SOFTNESS - 1));
    int hi = min(LEDS_PER_STRIP - 1, (int)(center + BLOOM_RADIUS + BLOOM_EDGE_SOFTNESS + 1));

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
            float dist = fabsf(i - center);
            float spatial;
            if (dist <= BLOOM_RADIUS) {
                spatial = 1.0f;
            } else {
                spatial = expf(-(dist - BLOOM_RADIUS) / BLOOM_EDGE_SOFTNESS);
            }
            float br = envelope * spatial;
            if (br > 0.003f) {
                buf[i] = buf[i] + br - buf[i] * br;
            }
        }
    }
}

static void renderCrawl(const Creature& c, float* buf) {
    float center = c.pos;
    int lo = max(0, (int)(center - CRAWL_RADIUS - CRAWL_TAIL_DECAY - 2));
    int hi = min(LEDS_PER_STRIP - 1, (int)(center + CRAWL_RADIUS + CRAWL_TAIL_DECAY + 2));

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
            float distFromCenter = fabsf(i - center);
            if (distFromCenter > radius) continue;

            float behind = radius - distFromCenter;
            float br = expf(-behind / CRAWL_TAIL_DECAY);
            if (behind < 1.0f) br *= behind;
            br *= fade;

            if (br > 0.003f) {
                buf[i] = buf[i] + br - buf[i] * br;
            }
        }
    }
}

static void renderCreature(const Creature& c, float* buf) {
    if (!c.alive) return;
    if (c.kind == KIND_BLOOM) renderBloom(c, buf);
    else                      renderCrawl(c, buf);
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

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("\nbiolum_mixed — ambient creatures on 6 strips");

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

    Serial.printf("  %d pairs × %d LEDs, %d creatures each\n",
                  NUM_PAIRS, LEDS_PER_STRIP, MAX_CREATURES);
}

static uint32_t lastMicros = 0;

void loop() {
    uint32_t now = micros();
    float dt = (lastMicros == 0) ? 0.016f : (now - lastMicros) * 1e-6f;
    if (dt > 0.1f) dt = 0.016f;
    lastMicros = now;

    for (int p = 0; p < NUM_PAIRS; p++) {
        PairState& ps = pairs[p];

        memset(ps.buf, 0, sizeof(ps.buf));

        for (int c = 0; c < MAX_CREATURES; c++) {
            updateCreature(ps.creatures[c], dt);
            renderCreature(ps.creatures[c], ps.buf);
        }

        for (int c = 0; c < MAX_CREATURES; c++) {
            if (!ps.creatures[c].alive) initCreature(ps.creatures[c]);
        }

        int pinA = p * 2;
        int pinB = p * 2 + 1;

        for (int i = 0; i < LEDS_PER_STRIP; i++) {
            float v = ps.buf[i];
            if (v > 1.0f) v = 1.0f;
            v = v * v;
            uint8_t r = (uint8_t)(v * COLOR_R);
            uint8_t g = (uint8_t)(v * COLOR_G);
            uint8_t b = (uint8_t)(v * COLOR_B);
            setPixel(pinA, i, r, g, b);
            setPixel(pinB, i, r, g, b);
        }

        showStrip(pinA);
        showStrip(pinB);
    }
}
