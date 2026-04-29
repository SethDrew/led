// Rainbow + gamma A/B test
// Two OKLCH rainbows side by side, cycling through brightness.
// Left side: linear dimming. Right side: gamma-corrected dimming.
// Folded strip layout: left runs reversed so they align when folded.

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#ifndef TEST_STRIP_PIN
#define TEST_STRIP_PIN 32
#endif
#ifndef TEST_STRIP_LEDS
#define TEST_STRIP_LEDS 150
#endif

Adafruit_NeoPixel strip(TEST_STRIP_LEDS, TEST_STRIP_PIN, NEO_GRB + NEO_KHZ800);

// Adafruit gamma8 table (gamma=2.8)
static const uint8_t gamma8[] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,
    5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,
   10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
   17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
   25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,
   37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,
   51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,
   69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,
   90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,
  115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,
  144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,
  177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,
  215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255
};

// OKLCH variable-L rainbow (from effects.cpp)
static const uint8_t oklch[256][3] = {
    {251,  3, 64}, {247,  1, 57}, {238,  1, 51}, {229,  1, 46},
    {220,  1, 41}, {212,  1, 36}, {204,  1, 32}, {196,  1, 28},
    {189,  1, 24}, {182,  1, 21}, {176,  1, 19}, {170,  1, 16},
    {165,  1, 14}, {160,  1, 12}, {156,  1, 10}, {152,  1,  8},
    {149,  1,  6}, {147,  1,  5}, {145,  1,  4}, {144,  1,  2},
    {143,  1,  1}, {143,  1,  0}, {137,  3,  0}, {133,  4,  0},
    {130,  6,  0}, {128,  8,  0}, {126,  9,  0}, {125, 11,  0},
    {125, 12,  0}, {125, 14,  0}, {126, 16,  0}, {127, 17,  0},
    {128, 19,  0}, {130, 21,  0}, {132, 23,  0}, {135, 25,  0},
    {138, 28,  0}, {141, 30,  1}, {144, 32,  1}, {148, 35,  1},
    {151, 38,  1}, {155, 41,  1}, {159, 44,  1}, {163, 47,  1},
    {166, 50,  1}, {170, 53,  1}, {173, 56,  1}, {176, 60,  1},
    {179, 63,  1}, {182, 66,  1}, {184, 70,  1}, {186, 73,  1},
    {187, 76,  1}, {188, 79,  1}, {188, 82,  1}, {188, 85,  1},
    {187, 87,  1}, {186, 90,  1}, {184, 92,  1}, {182, 93,  1},
    {179, 95,  1}, {176, 96,  1}, {173, 98,  1}, {169, 99,  1},
    {166,100,  1}, {163,101,  1}, {159,103,  1}, {156,104,  1},
    {153,105,  1}, {149,107,  1}, {146,108,  1}, {143,109,  1},
    {139,110,  1}, {136,112,  1}, {133,113,  1}, {129,114,  1},
    {126,116,  1}, {122,117,  1}, {119,118,  1}, {115,120,  1},
    {111,121,  1}, {108,123,  1}, {104,124,  1}, {100,126,  1},
    { 96,127,  1}, { 92,129,  1}, { 88,131,  1}, { 83,132,  1},
    { 79,134,  1}, { 74,136,  1}, { 70,138,  1}, { 65,139,  1},
    { 60,141,  1}, { 55,143,  1}, { 49,146,  1}, { 44,148,  1},
    { 38,150,  1}, { 32,153,  1}, { 25,155,  1}, { 19,158,  1},
    { 12,161,  1}, {  4,164,  1}, {  2,164,  6}, {  2,163, 12},
    {  2,162, 17}, {  2,161, 23}, {  2,161, 28}, {  2,160, 33},
    {  2,159, 38}, {  2,159, 43}, {  2,158, 47}, {  2,157, 52},
    {  2,157, 56}, {  2,156, 60}, {  2,156, 64}, {  2,155, 68},
    {  2,155, 72}, {  2,154, 76}, {  2,154, 79}, {  2,153, 83},
    {  2,153, 86}, {  2,152, 90}, {  2,152, 93}, {  2,151, 96},
    {  2,151,100}, {  2,150,103}, {  2,150,106}, {  2,149,109},
    {  2,149,112}, {  2,148,115}, {  2,148,118}, {  2,148,121},
    {  2,147,125}, {  2,147,128}, {  2,146,131}, {  2,146,134},
    {  2,145,137}, {  2,145,140}, {  2,145,143}, {  2,144,146},
    {  2,144,149}, {  2,143,152}, {  2,143,155}, {  2,142,158},
    {  2,142,161}, {  2,141,165}, {  2,141,168}, {  2,141,171},
    {  2,140,175}, {  2,140,178}, {  2,139,182}, {  2,139,185},
    {  2,138,189}, {  2,137,193}, {  2,137,196}, {  2,136,200},
    {  2,136,204}, {  2,135,209}, {  2,135,213}, {  2,134,217},
    {  2,133,222}, {  2,133,227}, {  2,132,232}, {  2,131,237},
    {  2,130,243}, {  2,130,248}, {  3,129,252}, {  6,127,252},
    {  9,126,252}, { 12,125,252}, { 15,124,252}, { 18,122,252},
    { 21,121,252}, { 24,120,252}, { 26,119,252}, { 29,118,252},
    { 31,117,252}, { 34,116,252}, { 37,115,252}, { 39,114,252},
    { 40,111,252}, { 40,109,252}, { 40,105,252}, { 40,101,251},
    { 39, 97,251}, { 37, 92,251}, { 36, 86,251}, { 34, 81,251},
    { 32, 74,251}, { 31, 68,251}, { 29, 62,250}, { 27, 55,250},
    { 26, 48,250}, { 25, 42,250}, { 24, 35,250}, { 24, 29,249},
    { 24, 23,249}, { 24, 17,249}, { 24, 12,249}, { 25,  6,248},
    { 25,  2,248}, { 25,  1,225}, { 25,  1,201}, { 24,  1,180},
    { 24,  1,162}, { 23,  0,146}, { 23,  0,132}, { 23,  0,121},
    { 22,  0,111}, { 22,  0,103}, { 22,  0, 96}, { 23,  0, 91},
    { 23,  0, 86}, { 24,  0, 83}, { 25,  0, 81}, { 26,  0, 80},
    { 28,  0, 79}, { 29,  0, 79}, { 32,  0, 80}, { 35,  0, 82},
    { 38,  0, 85}, { 42,  0, 88}, { 46,  0, 91}, { 51,  0, 96},
    { 57,  0,100}, { 64,  0,106}, { 72,  0,112}, { 80,  1,118},
    { 90,  1,124}, {101,  1,131}, {112,  1,138}, {125,  1,144},
    {138,  1,151}, {153,  1,157}, {168,  1,163}, {184,  1,169},
    {201,  1,174}, {218,  1,178}, {235,  1,181}, {251,  2,182},
    {252,  9,174}, {252, 16,167}, {252, 21,160}, {252, 25,154},
    {252, 28,148}, {252, 30,142}, {252, 31,136}, {252, 30,130},
    {252, 29,123}, {252, 26,116}, {252, 23,108}, {252, 20,101},
    {251, 17, 93}, {251, 13, 86}, {251, 10, 78}, {251,  6, 71},
};

static const uint16_t RAMP_LEN = 28;
static const uint16_t OFFSET = 61;
static const uint16_t GAP = 2;          // pixel gap between sides to align fold
static const uint32_t CYCLE_MS = 12000;  // fade portion (6s up + 6s down)
static const uint32_t HOLD_MS = 2000;    // hold at max
static const uint32_t TOTAL_MS = CYCLE_MS + HOLD_MS;

// Hybrid gamma: gamma curve 0-204, then linear ramp from gamma8[204] to 255
// Smooth crossover — no discontinuity at the join point.
static const uint8_t CROSSOVER = 204;
static uint8_t gammaHybrid(uint8_t v) {
    if (v <= CROSSOVER) {
        return gamma8[v];
    }
    // Linear ramp from gamma8[CROSSOVER] to 255
    uint8_t base = gamma8[CROSSOVER];  // = 137
    return base + (uint16_t)(v - CROSSOVER) * (255 - base) / (255 - CROSSOVER);
}

static uint32_t lastLog = 0;
static bool tablePrinted = false;

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n[Rainbow Gamma Test — hybrid comparison]");
    Serial.printf("  %d LEDs on pin %d\n", TEST_STRIP_LEDS, TEST_STRIP_PIN);
    Serial.printf("  Left (reversed): pixels %d-%d LINEAR\n", OFFSET, OFFSET + RAMP_LEN - 1);
    Serial.printf("  Right:           pixels %d-%d HYBRID gamma\n", OFFSET + RAMP_LEN + GAP, OFFSET + RAMP_LEN + GAP + RAMP_LEN - 1);
    Serial.printf("  Gap: %d pixels between sides\n", GAP);

    strip.begin();
    strip.setBrightness(255);
}

void loop() {
    uint32_t ms = millis();

    // Triangle wave with hold at peak:
    // 0 → CYCLE_MS/2: ramp up 0→255
    // CYCLE_MS/2 → CYCLE_MS/2+HOLD_MS: hold at 255
    // CYCLE_MS/2+HOLD_MS → CYCLE_MS+HOLD_MS: ramp down 255→0
    uint32_t phase = ms % TOTAL_MS;
    uint16_t brightness;
    uint32_t halfCycle = CYCLE_MS / 2;

    if (phase < halfCycle) {
        // Ramp up
        brightness = (uint32_t)phase * 255 / halfCycle;
    } else if (phase < halfCycle + HOLD_MS) {
        // Hold at max
        brightness = 255;
    } else {
        // Ramp down
        brightness = (uint32_t)(TOTAL_MS - phase) * 255 / halfCycle;
    }
    if (brightness > 255) brightness = 255;

    // Gamma-correct the brightness, not the individual channels
    uint8_t gammaBr = gammaHybrid(brightness);

    // Print full RGB table once at max brightness
    if (brightness == 255 && !tablePrinted) {
        tablePrinted = true;
        Serial.println("\n=== RGB values at max brightness (br=255) ===");
        Serial.println("Pixel | HYBRID (left)       | LINEAR (right)      | Match?");
        Serial.println("------+---------------------+---------------------+-------");
        uint8_t gbr = gammaHybrid(255);  // gamma(255) should be 255
        for (uint16_t i = 0; i < RAMP_LEN; i++) {
            uint8_t hueIdx = (uint8_t)((uint32_t)i * 255 / (RAMP_LEN - 1));
            uint8_t hr = (uint16_t)oklch[hueIdx][0] * gbr / 255;
            uint8_t hg = (uint16_t)oklch[hueIdx][1] * gbr / 255;
            uint8_t hb = (uint16_t)oklch[hueIdx][2] * gbr / 255;
            uint8_t lr = oklch[hueIdx][0];
            uint8_t lg = oklch[hueIdx][1];
            uint8_t lb = oklch[hueIdx][2];
            Serial.printf(" %2d   | (%3d, %3d, %3d)     | (%3d, %3d, %3d)     | %s\n",
                i, hr, hg, hb, lr, lg, lb,
                (hr==lr && hg==lg && hb==lb) ? "YES" : "NO");
        }
        Serial.println();
    }
    if (brightness < 200) tablePrinted = false;  // reset for next cycle

    // Debug log every 200ms
    if (ms - lastLog >= 200) {
        lastLog = ms;
        Serial.printf("br=%3d  gammaBr=%3d\n", brightness, gammaBr);
    }

    // Clear
    for (uint16_t i = 0; i < strip.numPixels(); i++) {
        strip.setPixelColor(i, 0, 0, 0);
    }

    // Left side (reversed): HYBRID gamma dimming (gamma applied to brightness)
    for (uint16_t i = 0; i < RAMP_LEN; i++) {
        uint8_t hueIdx = (uint8_t)((uint32_t)i * 255 / (RAMP_LEN - 1));
        uint8_t r = (uint16_t)oklch[hueIdx][0] * gammaBr / 255;
        uint8_t g = (uint16_t)oklch[hueIdx][1] * gammaBr / 255;
        uint8_t b = (uint16_t)oklch[hueIdx][2] * gammaBr / 255;
        strip.setPixelColor(OFFSET + (RAMP_LEN - 1 - i), r, g, b);
    }

    // Right side (after gap): LINEAR dimming (brightness applied directly)
    for (uint16_t i = 0; i < RAMP_LEN; i++) {
        uint8_t hueIdx = (uint8_t)((uint32_t)i * 255 / (RAMP_LEN - 1));
        uint8_t r = (uint16_t)oklch[hueIdx][0] * brightness / 255;
        uint8_t g = (uint16_t)oklch[hueIdx][1] * brightness / 255;
        uint8_t b = (uint16_t)oklch[hueIdx][2] * brightness / 255;
        strip.setPixelColor(OFFSET + RAMP_LEN + GAP + i, r, g, b);
    }

    strip.show();
    delay(16);
}
