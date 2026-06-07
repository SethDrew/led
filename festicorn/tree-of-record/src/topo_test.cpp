/*
 * TOPO_TEST — LED-to-stick position mapping utility for tree-of-record.
 *
 * Lights all 6 strips with a STATIC repeating 10-LED R/G/B block pattern so
 * you can count stick boundaries by color blocks:
 *   LEDs 0-9 red, 10-19 green, 20-29 blue, 30-39 red, ... across all 100.
 *
 * Not compiled by default. To build/flash this instead of main.cpp, point
 * build_src_filter at it, e.g. add a temporary env to platformio.ini:
 *
 *   build_src_filter = -<*> +<topo_test.cpp>
 *
 * Hardware (matches main.cpp): ESP32-D0WD-V3, 6 × WS2812B RGB strips,
 * 100 LEDs each, GPIO 4/15/17/5/18/19, RMT channels 0..5, NeoRgbFeature.
 *
 * Startup ID flash: each strip flashes its distinct color in turn so you know
 * which pin is which, then all strips settle into the static block pattern:
 *   strip 0 = red    (GPIO 4)
 *   strip 1 = green  (GPIO 15)
 *   strip 2 = blue   (GPIO 17)
 *   strip 3 = yellow (GPIO 5)
 *   strip 4 = cyan   (GPIO 18)
 *   strip 5 = magenta(GPIO 19)
 */

#include <Arduino.h>
#include <NeoPixelBus.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 100
#endif

static const uint8_t NUM_STRIPS = 6;

// ── Tuning ───────────────────────────────────────────────────────
#define ID_FLASH_MS   800    // per-strip ID color flash at startup
#define BLOCK_SIZE    10     // LEDs per color block in the static pattern

// Static block colors, cycled every BLOCK_SIZE LEDs: red, green, blue.
static const uint8_t BLOCK_COLOR[3][3] = {
    { 120,   0,   0 },   // red
    {   0, 120,   0 },   // green
    {   0,   0, 120 },   // blue
};

// ── LED drivers: 6 strips via RMT (same as main.cpp) ─────────────
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 15);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, 19);

static const uint8_t STRIP_GPIO[NUM_STRIPS] = { 4, 15, 17, 5, 18, 19 };

// Distinct per-strip ID color (R, G, B), kept dim-ish so markers read clearly.
static const uint8_t STRIP_COLOR[NUM_STRIPS][3] = {
    { 120,   0,   0 },   // 0 red
    {   0, 120,   0 },   // 1 green
    {   0,   0, 120 },   // 2 blue
    { 100, 100,   0 },   // 3 yellow
    {   0, 100, 100 },   // 4 cyan
    { 100,   0, 100 },   // 5 magenta
};

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

static inline void clearStrip(uint8_t s) {
    switch (s) {
        case 0: strip0.ClearTo(RgbColor(0)); break;
        case 1: strip1.ClearTo(RgbColor(0)); break;
        case 2: strip2.ClearTo(RgbColor(0)); break;
        case 3: strip3.ClearTo(RgbColor(0)); break;
        case 4: strip4.ClearTo(RgbColor(0)); break;
        case 5: strip5.ClearTo(RgbColor(0)); break;
    }
}

static inline void showStrip(uint8_t s) {
    switch (s) {
        case 0: strip0.Show(); break;
        case 1: strip1.Show(); break;
        case 2: strip2.Show(); break;
        case 3: strip3.Show(); break;
        case 4: strip4.Show(); break;
        case 5: strip5.Show(); break;
    }
}

static void clearAll() {
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0)); strip2.ClearTo(RgbColor(0));
    strip3.ClearTo(RgbColor(0)); strip4.ClearTo(RgbColor(0)); strip5.ClearTo(RgbColor(0));
    strip0.Show(); strip1.Show(); strip2.Show();
    strip3.Show(); strip4.Show(); strip5.Show();
}

// Fill a whole strip with its ID color and show it.
static void flashStripId(uint8_t s) {
    clearAll();
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
        setPixel(s, i, STRIP_COLOR[s][0], STRIP_COLOR[s][1], STRIP_COLOR[s][2]);
    showStrip(s);
}

// Paint the static R/G/B block pattern on one strip (does not Show).
static void drawBlocks(uint8_t s) {
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        const uint8_t *c = BLOCK_COLOR[(i / BLOCK_SIZE) % 3];
        setPixel(s, i, c[0], c[1], c[2]);
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);

    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
    clearAll();

    Serial.println("\nTOPO_TEST — static R/G/B block pattern");
    Serial.printf("  %u strips x %u LEDs, %u-LED blocks (red/green/blue repeating)\n",
                  NUM_STRIPS, LEDS_PER_STRIP, BLOCK_SIZE);

    // ID flash: each strip flashes its color in turn so the user can map pins.
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        Serial.printf("  ID flash strip %u (GPIO %u)\n", s, STRIP_GPIO[s]);
        flashStripId(s);
        delay(ID_FLASH_MS);
    }
    clearAll();

    // Settle into the static block pattern on all strips simultaneously.
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        drawBlocks(s);
        showStrip(s);
    }
    Serial.println("  static block pattern shown on all 6 strips");
}

void loop() {
    // Static pattern — nothing to animate.
    delay(1000);
}
