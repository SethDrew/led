/*
 * TOPO_TEST — LED-to-stick position mapping utility for tree-of-record.
 *
 * Walks a lit group of LEDs down each of the 6 strips so you can see, by
 * eye, which physical stick a given strip/index range lands on, then note
 * e.g. "stick 1 on strip 0 = LEDs 0-23".
 *
 * Not compiled by default. To build/flash this instead of main.cpp, point
 * build_src_filter at it, e.g. add a temporary env to platformio.ini:
 *
 *   build_src_filter = -<*> +<topo_test.cpp>
 *
 * Hardware (matches main.cpp): ESP32-D0WD-V3, 6 × WS2812B RGB strips,
 * 100 LEDs each, GPIO 4/16/17/5/18/19, RMT channels 0..5, NeoGrbFeature.
 *
 * Per-strip ID flash: each strip gets a distinct color so you know which
 * pin is active before the walk:
 *   strip 0 = red    (GPIO 4)
 *   strip 1 = green  (GPIO 16)
 *   strip 2 = blue   (GPIO 17)
 *   strip 3 = yellow (GPIO 5)
 *   strip 4 = cyan   (GPIO 18)
 *   strip 5 = magenta(GPIO 19)
 *
 * Walk: a group of WALK_GROUP LEDs lights in the strip's ID color and steps
 * down the strip every WALK_STEP_MS. LEDs at multiples of 10 light white as
 * counting markers so you can read positions at a glance.
 *
 * Serial control (115200):
 *   p  — pause / resume the walk
 *   n  — skip to the next strip (or space)
 *   r  — restart the current strip from index 0
 * Serial output prints the active strip and the leading LED index as it walks.
 */

#include <Arduino.h>
#include <NeoPixelBus.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 100
#endif

static const uint8_t NUM_STRIPS = 6;

// ── Walk tuning ──────────────────────────────────────────────────
#define WALK_GROUP    4      // LEDs lit at once in the moving group
#define WALK_STEP     3      // index advance per step
#define WALK_STEP_MS  250    // dwell per step (ms)
#define ID_FLASH_MS   800    // whole-strip ID color flash before walking
#define MARKER_EVERY  10     // every Nth LED is a white counting marker

// ── LED drivers: 6 strips via RMT (same as main.cpp) ─────────────
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 16);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, 19);

static const uint8_t STRIP_GPIO[NUM_STRIPS] = { 4, 16, 17, 5, 18, 19 };

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

// ── Walk state ───────────────────────────────────────────────────
static uint8_t  curStrip = 0;
static int      walkPos  = 0;       // leading index of the moving group
static bool     paused   = false;
static bool     idShown  = false;   // whether the ID flash for curStrip ran
static uint32_t lastStepMs = 0;

static void beginStrip(uint8_t s) {
    curStrip = s;
    walkPos  = 0;
    idShown  = false;
    Serial.printf("\n=== STRIP %u  (GPIO %u, %s) ===\n",
                  s, STRIP_GPIO[s],
                  s == 0 ? "red" : s == 1 ? "green" : s == 2 ? "blue" :
                  s == 3 ? "yellow" : s == 4 ? "cyan" : "magenta");
}

static void nextStrip() {
    beginStrip((curStrip + 1) % NUM_STRIPS);
}

// Draw the current strip: moving group in ID color, markers in white.
static void drawWalk() {
    clearStrip(curStrip);

    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        if (i % MARKER_EVERY == 0)
            setPixel(curStrip, i, 60, 60, 60);   // dim white counting marker
    }

    for (int k = 0; k < WALK_GROUP; k++) {
        int idx = walkPos + k;
        if (idx >= 0 && idx < LEDS_PER_STRIP)
            setPixel(curStrip, idx,
                     STRIP_COLOR[curStrip][0],
                     STRIP_COLOR[curStrip][1],
                     STRIP_COLOR[curStrip][2]);
    }

    showStrip(curStrip);
}

static void handleSerial() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        switch (c) {
            case 'p': case 'P':
                paused = !paused;
                Serial.printf("[%s]\n", paused ? "PAUSED" : "RESUMED");
                break;
            case 'n': case 'N': case ' ':
                Serial.println("[next strip]");
                nextStrip();
                break;
            case 'r': case 'R':
                Serial.println("[restart strip]");
                walkPos = 0;
                idShown = false;
                break;
            default:
                break;
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);

    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
    clearAll();

    Serial.println("\nTOPO_TEST — LED stick mapping");
    Serial.printf("  %u strips x %u LEDs, group=%u step=%u dwell=%ums, marker every %u\n",
                  NUM_STRIPS, LEDS_PER_STRIP, WALK_GROUP, WALK_STEP, WALK_STEP_MS, MARKER_EVERY);
    Serial.println("  serial: p=pause/resume  n/space=next strip  r=restart strip");

    beginStrip(0);
    lastStepMs = millis();
}

void loop() {
    handleSerial();

    if (!idShown) {
        flashStripId(curStrip);
        delay(ID_FLASH_MS);
        idShown = true;
        lastStepMs = millis();
        drawWalk();
        Serial.printf("  led=%d (group %d-%d)\n",
                      walkPos, walkPos, walkPos + WALK_GROUP - 1);
        return;
    }

    if (paused) return;

    uint32_t now = millis();
    if (now - lastStepMs < WALK_STEP_MS) return;
    lastStepMs = now;

    walkPos += WALK_STEP;
    if (walkPos >= LEDS_PER_STRIP) {
        Serial.printf("  strip %u done\n", curStrip);
        nextStrip();
        return;
    }

    drawWalk();
    Serial.printf("  led=%d (group %d-%d)\n",
                  walkPos, walkPos, walkPos + WALK_GROUP - 1);
}
