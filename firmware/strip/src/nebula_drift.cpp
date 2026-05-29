/*
 * Bioluminescent jellyfish: 7 creatures drift slowly along the strip,
 * each periodically emitting expanding cyan rings. Delta-time based.
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12

#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

#ifndef COLOR_ORDER
#define COLOR_ORDER NEO_GRB
#endif

#define BRIGHTNESS 60

// --- Creatures ---
#define NUM_CREATURES  7
#define DRIFT_SPEED    1.0f   // pixels per second

// --- Pulses ---
#define MAX_PULSES     10
#define LIFETIME_S     1.0f   // seconds
#define MAX_RADIUS     13.3f  // pixels over lifetime (was 40, now 1/3)
#define TAIL_LEN       12     // decay distance in pixels

// Color
#define R_BASE  0
#define G_BASE 180
#define B_BASE 220

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, COLOR_ORDER + NEO_KHZ800);

static uint8_t decayLUT[64];

void buildLUT() {
  for (uint8_t i = 0; i < 64; i++) {
    decayLUT[i] = (uint8_t)(255.0f * exp(-(float)i / TAIL_LEN));
  }
}

static uint16_t rngState;
uint16_t rng() {
  rngState ^= rngState << 7;
  rngState ^= rngState >> 9;
  rngState ^= rngState << 8;
  return rngState;
}

struct Creature {
  float pos;
  float vel;         // pixels per second
  float emitTimer;   // seconds until next pulse
};

struct Pulse {
  float age;         // seconds, <0 = inactive
  int16_t centerX8;
};

Creature creatures[NUM_CREATURES];
Pulse pulses[MAX_PULSES];
uint8_t nextPulse = 0;

static uint16_t accum[NUM_PIXELS];

unsigned long prevMicros;

void spawnPulse(float center) {
  Pulse &p = pulses[nextPulse];
  p.centerX8 = (int16_t)(center * 8);
  p.age = 0.0f;
  nextPulse = (nextPulse + 1) % MAX_PULSES;
}

float randomEmitInterval() {
  return 0.3f + (rng() % 700) * 0.001f; // 0.3–1.0s
}

void setup() {
  strip.begin();
  strip.setBrightness(BRIGHTNESS);
  strip.clear();
  strip.show();
  buildLUT();

  rngState = analogRead(A0) | 1;

  for (uint8_t i = 0; i < NUM_CREATURES; i++) {
    float base = (float)(i + 1) * NUM_PIXELS / (NUM_CREATURES + 1);
    creatures[i].pos = base + (int8_t)(rng() % 20) - 10;
    creatures[i].vel = ((rng() & 1) ? 1.0f : -1.0f) * DRIFT_SPEED;
    creatures[i].emitTimer = randomEmitInterval();
  }

  for (uint8_t i = 0; i < MAX_PULSES; i++) pulses[i].age = -1.0f;
  prevMicros = micros();
}

void loop() {
  unsigned long now = micros();
  float dt = (now - prevMicros) * 0.000001f;
  prevMicros = now;
  if (dt > 0.1f) dt = 0.1f; // clamp on first frame or glitch

  // Update creatures
  for (uint8_t c = 0; c < NUM_CREATURES; c++) {
    Creature &cr = creatures[c];

    cr.pos += cr.vel * dt;

    // Random walk on velocity
    if ((rng() & 31) == 0) {
      cr.vel += ((rng() & 1) ? 0.25f : -0.25f);
      if (cr.vel >  3.0f) cr.vel =  3.0f;
      if (cr.vel < -3.0f) cr.vel = -3.0f;
    }

    if (cr.pos < 8)              { cr.vel = abs(cr.vel); }
    if (cr.pos > NUM_PIXELS - 8) { cr.vel = -abs(cr.vel); }

    cr.emitTimer -= dt;
    if (cr.emitTimer <= 0) {
      spawnPulse(cr.pos);
      cr.emitTimer = randomEmitInterval();
    }
  }

  memset(accum, 0, sizeof(accum));

  for (uint8_t p = 0; p < MAX_PULSES; p++) {
    if (pulses[p].age < 0) continue;

    float t = pulses[p].age / LIFETIME_S; // 0..1
    if (t > 1.0f) { pulses[p].age = -1.0f; continue; }

    uint16_t radiusX8 = (uint16_t)(t * MAX_RADIUS * 8);

    float fade = 1.0f - t;
    fade *= fade;
    uint8_t fadeU8 = (uint8_t)(fade * 255);

    int16_t centerX8 = pulses[p].centerX8;
    int16_t ctr = centerX8 >> 3;
    int16_t rad = (radiusX8 >> 3) + TAIL_LEN + 1;
    int16_t lo = ctr - rad; if (lo < 0) lo = 0;
    int16_t hi = ctr + rad; if (hi >= NUM_PIXELS) hi = NUM_PIXELS - 1;

    for (int16_t i = lo; i <= hi; i++) {
      int16_t distX8 = abs((int16_t)(i << 3) - centerX8);
      if (distX8 > radiusX8) continue;

      uint16_t behindX8 = radiusX8 - distX8;
      uint8_t behindPx = behindX8 >> 3;
      uint8_t frac = behindX8 & 7; // sub-pixel 0-7

      uint8_t d0 = (behindPx < 64) ? decayLUT[behindPx] : 0;
      uint8_t d1 = (behindPx + 1 < 64) ? decayLUT[behindPx + 1] : 0;
      uint8_t decay = d0 + (((int16_t)(d1 - d0) * frac) >> 3);

      // Leading edge fade-in: pixels within 1px of front get soft ramp
      uint16_t aheadX8 = radiusX8 - distX8; // same as behindX8 but semantically "how close to front"
      if (aheadX8 < 8) {
        decay = ((uint16_t)decay * aheadX8) >> 3;
      }

      uint16_t bright = ((uint16_t)decay * fadeU8) >> 8;
      accum[i] += bright;
    }

    pulses[p].age += dt;
  }

  for (int i = 0; i < NUM_PIXELS; i++) {
    uint16_t v = accum[i];
    if (v > 255) v = 255;
    uint8_t v2 = ((uint16_t)v * v) >> 8;
    uint8_t r = ((uint16_t)R_BASE * v2) >> 8;
    uint8_t g = ((uint16_t)G_BASE * v2) >> 8;
    uint8_t b = ((uint16_t)B_BASE * v2) >> 8;
    strip.setPixelColor(i, r, g, b);
  }

  strip.show();
}
