// take-root — 5-entity LED-synced crawler with dithered fade.
//
// A soft white "ring" sweeps UP through the tree at a constant LED rate.
// All trunk entities share a single `trunk_head` counter (LEDs from base).
// When the head crosses LED 8 (the rejoin trigger), two LED-synced spawn
// entities fire from the interconnection point on the opposite side of the
// branch: one going OUT through the back leg (branch-in) and one going UP
// through the upper trunk (trunk-split).
//
// Entities (5 total):
//   1. backside         — strip 1, idx 0..33      (head = trunk_head)
//   2. front-right-base — strip 2, idx 0..36      (head = trunk_head)
//   3. front-left-base  — strip 3, idx 0..12      (head = trunk_head)
//                                                  includes branch-out (8..12)
//   4. branch-in        — strip 3, idx 19..13     (REVERSED on wire,
//                                                  head = trunk_head − 8)
//   5. trunk-split      — strip 3, idx 20..37     (head = trunk_head − 8)
//
// Wire-index ↔ entity-position mapping:
//   front-left-base: p = idx              (idx 0..12)
//   branch-in:       p = 19 − idx         (idx 19 = p 0, idx 13 = p 6)
//   trunk-split:     p = idx − 20         (idx 20 = p 0, idx 37 = p 17)
//
// Topology heights (for reference, NOT used by this effect):
//   backside (strip 1):       H = -5 + idx
//   front-right-base (s2):    H = idx (i≤8); 8 (i=9..11); 8+0.8(i-11) (i≥12)
//   front-left-base trunk-1:  H = -1 + idx (idx 0..7), attach at idx 8 (H=7)
//   rejoin (idx 19↔20):       H = 8 (one pitch above attach)
//
// Both ring and crawlers use the SK6812 W channel with first-order
// delta-sigma temporal dithering for smooth sub-byte fades at low brightness.
//
// GPIO 12 strapping caveat: never INPUT_PULLUP. NeoPixel begin() configures
// OUTPUT after strapping latches; 74HCT245 input is high-Z so ESP32 internal
// pull-down wins at boot.

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

// First-order delta-sigma ditherer. Inlined from festicorn/lib/delta_sigma.
static inline uint8_t deltaSigma(uint16_t &accum, uint16_t target16) {
    accum += target16;
    uint8_t out = accum >> 8;
    accum &= 0xFF;
    return out;
}

// --- Tunables ---------------------------------------------------------------
#define MAX_W                24       // peak W-channel level
#define SWEEP_MS             10000.0f
#define PAUSE_MS             2000.0f
#define FRAME_MS             20

#define HALF_WIDTH           2.0f     // fade footprint = 2*HALF_WIDTH LEDs
#define MAX_BASE_LEN         37.0f    // longest base entity (front-right-base)
#define REJOIN_TRIGGER_LED   8.0f     // trunk_head value that fires entities 4+5

// --- Strip objects ----------------------------------------------------------
Adafruit_NeoPixel strip1(NUM_LEDS, STRIP1_PIN, NEO_RGBW + NEO_KHZ800);
Adafruit_NeoPixel strip2(NUM_LEDS, STRIP2_PIN, NEO_RGBW + NEO_KHZ800);
Adafruit_NeoPixel strip3(NUM_LEDS, STRIP3_PIN, NEO_RGBW + NEO_KHZ800);

// Delta-sigma accumulators (one per LED, persistent across frames).
static uint16_t accum1[34] = {0};
static uint16_t accum2[37] = {0};
static uint16_t accum3[38] = {0};

// --- Symmetric triangular fade ----------------------------------------------
static inline float fade(float head, float pos, float half_width) {
    float d = head - pos;
    if (d < 0.0f) d = -d;
    if (d >= half_width) return 0.0f;
    return 1.0f - (d / half_width);
}

// Push a float intensity (0..1) through delta-sigma and write the W channel.
static inline void write_w(Adafruit_NeoPixel& strip, int wire, uint16_t& accum, float intensity) {
    if (intensity < 0.0f) intensity = 0.0f;
    if (intensity > 1.0f) intensity = 1.0f;
    uint16_t target16 = (uint16_t)(intensity * MAX_W * 256.0f);
    uint8_t w = deltaSigma(accum, target16);
    strip.setPixelColor(wire, Adafruit_NeoPixel::Color(0, 0, 0, w));
}

void setup() {
    strip1.begin();
    strip2.begin();
    strip3.begin();
    // Brightness left at 255 — manual attenuation via MAX_W keeps the
    // delta-sigma dithering from being re-quantized by the strip's scaler.
    strip1.setBrightness(255);
    strip2.setBrightness(255);
    strip3.setBrightness(255);
    strip1.clear(); strip1.show();
    strip2.clear(); strip2.show();
    strip3.clear(); strip3.show();
}

void loop() {
    static uint32_t cycle_start = millis();
    uint32_t now = millis();
    float t = (float)(now - cycle_start);

    if (t > SWEEP_MS + PAUSE_MS) {
        cycle_start = now;
        t = 0.0f;
    }

    // trunk_head: shared LED counter for the 3 base entities. Starts -HALF_WIDTH
    // (head fully below LED 0) and ends MAX_BASE_LEN + HALF_WIDTH (head fully
    // above LED 36) so the fade enters and exits cleanly.
    const float TRAVEL = MAX_BASE_LEN + 2.0f * HALF_WIDTH;
    float trunk_head = (t < SWEEP_MS)
        ? (-HALF_WIDTH + TRAVEL * (t / SWEEP_MS))
        : (MAX_BASE_LEN + HALF_WIDTH);

    // Spawned-crawler heads: same LED-rate, offset by the rejoin trigger.
    float spawn_head = trunk_head - REJOIN_TRIGGER_LED;

    // === Entity 1: backside (strip 1) =======================================
    for (int i = 0; i < 34; i++) {
        write_w(strip1, i, accum1[i], fade(trunk_head, (float)i, HALF_WIDTH));
    }

    // === Entity 2: front-right-base (strip 2) ===============================
    for (int i = 0; i < 37; i++) {
        write_w(strip2, i, accum2[i], fade(trunk_head, (float)i, HALF_WIDTH));
    }

    // === Entity 3: front-left-base (strip 3 idx 0..12) ======================
    for (int i = 0; i <= 12; i++) {
        write_w(strip3, i, accum3[i], fade(trunk_head, (float)i, HALF_WIDTH));
    }

    // === Entity 4: branch-in (strip 3 idx 19..13, reversed on wire) =========
    for (int i = 13; i <= 19; i++) {
        float p = (float)(19 - i);
        write_w(strip3, i, accum3[i], fade(spawn_head, p, HALF_WIDTH));
    }

    // === Entity 5: trunk-split (strip 3 idx 20..37) =========================
    for (int i = 20; i <= 37; i++) {
        float p = (float)(i - 20);
        write_w(strip3, i, accum3[i], fade(spawn_head, p, HALF_WIDTH));
    }

    strip1.show();
    strip2.show();
    strip3.show();
    delay(FRAME_MS);
}
