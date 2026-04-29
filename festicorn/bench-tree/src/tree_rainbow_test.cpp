/*
 * TREE + STRIP with Web UI
 *
 * Tree: Sap flow animation on 3 strips (pins 27/13/14, 197 LEDs) — standalone
 * Strip: Web-controllable effects on pin 32 (150 LEDs) — rainbow/gradient + palettes
 *
 * Both run independently on the same ESP32.
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoOTA.h>
#include <Adafruit_NeoPixel.h>
#include "TreeTopology.h"
#include "TreeEffect.h"
#include "foregrounds/SapFlowForeground.h"

#define WIFI_SSID "cuteplant"
#define WIFI_PASSWORD "bigboiredwood"

#ifndef TEST_STRIP_PIN
#define TEST_STRIP_PIN 32
#endif
#ifndef TEST_STRIP_LEDS
#define TEST_STRIP_LEDS 150
#endif

// ═══════════════════════════════════════════════════════════════════
// Hardware
// ═══════════════════════════════════════════════════════════════════

Tree tree;
Adafruit_NeoPixel testStrip(TEST_STRIP_LEDS, TEST_STRIP_PIN, NEO_GRB + NEO_KHZ800);
WebServer server(80);

// ═══════════════════════════════════════════════════════════════════
// Effect state (strip only)
// ═══════════════════════════════════════════════════════════════════

struct RGB { uint8_t r, g, b; };

enum Effect { RAINBOW, GRADIENT };
enum Palette {
    SAP_FLOW, OKLCH_RAINBOW,
    RED_BLUE, CYAN_GOLD, GREEN_PURPLE, ORANGE_TEAL, MAGENTA_CYAN, SUNSET_SKY,
    BLUE_WASH, RED_WASH, GREEN_WASH, PURPLE_WASH, GOLD_WASH,
    CUSTOM
};

struct EffectState {
    Effect effect = RAINBOW;
    uint8_t brightness = 128;
    uint32_t cycleTimeMs = 8000;
    Palette palette = OKLCH_RAINBOW;
    uint8_t chroma = 255;  // 0=grey, 255=full saturation
    RGB customStops[16];
    uint8_t customCount = 0;
} state;

// ═══════════════════════════════════════════════════════════════════
// Gamma
// ═══════════════════════════════════════════════════════════════════

static const uint8_t gamma8[256] = {
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

static uint8_t gammaHybrid(uint8_t v) {
    if (v == 0) return 0;
    const uint8_t CROSSOVER = 153;
    uint8_t g;
    if (v <= CROSSOVER) {
        g = gamma8[v];
    } else {
        uint8_t base = gamma8[CROSSOVER];
        g = (uint8_t)(base + (uint16_t)(v - CROSSOVER) * (255 - base) / (255 - CROSSOVER));
    }
    // Floor: gamma8 maps 0-40 to 0-1, but br=1 gives all-black after
    // integer truncation (252*1/255=0). Minimum br=2 ensures at least
    // 1-bit visible output for any non-zero brightness.
    return g < 2 ? 2 : g;
}

// ═══════════════════════════════════════════════════════════════════
// OKLCH variable-L rainbow LUT (256 entries, PROGMEM)
// ═══════════════════════════════════════════════════════════════════

static const uint8_t oklchVarL[256][3] PROGMEM = {
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

static inline void lutRead(const uint8_t lut[][3], uint8_t idx, uint8_t &r, uint8_t &g, uint8_t &b) {
    r = pgm_read_byte(&lut[idx][0]);
    g = pgm_read_byte(&lut[idx][1]);
    b = pgm_read_byte(&lut[idx][2]);
}

// Uniform chroma scaling — blend toward BT.601 luminance grey
static void applyChroma(uint8_t &r, uint8_t &g, uint8_t &b, uint8_t chroma) {
    if (chroma >= 255) return;
    uint8_t grey = ((uint16_t)r * 77 + (uint16_t)g * 150 + (uint16_t)b * 29) >> 8;
    r = grey + ((int16_t)(r - grey) * chroma / 255);
    g = grey + ((int16_t)(g - grey) * chroma / 255);
    b = grey + ((int16_t)(b - grey) * chroma / 255);
}

// ═══════════════════════════════════════════════════════════════════
// Palette data
// ═══════════════════════════════════════════════════════════════════

static const RGB sapFlow[] = {
    {0, 40, 0}, {0, 70, 0}, {10, 110, 10}, {34, 180, 34}, {80, 255, 80}
};
static const uint8_t sapFlowCount = sizeof(sapFlow) / sizeof(sapFlow[0]);

static const RGB oklchRainbow[] = {
    {251,3,64}, {188,82,1}, {2,155,72}, {3,129,252}, {184,1,169}
};
static const uint8_t oklchRainbowCount = sizeof(oklchRainbow) / sizeof(oklchRainbow[0]);

static const RGB redBlue[] = {
    {251,3,64},{251,15,90},{252,26,117},{252,30,140},{252,20,162},
    {239,1,181},{179,1,167},{126,1,145},{86,1,121},{57,0,100},
    {40,0,86},{29,0,79},{24,0,82},{23,0,130},{25,6,248},{26,48,250}
};
static const uint8_t redBlueCount = sizeof(redBlue) / sizeof(redBlue[0]);

static const RGB cyanGold[] = {
    {2,144,145},{2,139,184},{2,131,236},{29,118,252},{34,80,251},
    {25,6,248},{23,0,90},{53,0,97},{191,1,171},{252,30,127},
    {224,1,43},{150,1,7},{125,12,0},{151,37,1},{187,76,1},{166,100,1}
};
static const uint8_t cyanGoldCount = sizeof(cyanGold) / sizeof(cyanGold[0]);

static const RGB greenPurple[] = {
    {2,163,12},{2,157,51},{2,153,82},{2,149,108},{2,146,132},
    {2,143,156},{2,139,182},{2,135,212},{2,129,252},{25,120,252},
    {40,108,252},{30,67,251},{24,17,249},{23,0,149},{23,0,84},{40,0,86}
};
static const uint8_t greenPurpleCount = sizeof(greenPurple) / sizeof(greenPurple[0]);

static const RGB orangeTeal[] = {
    {137,27,0},{158,43,1},{179,63,1},{188,82,1},{179,95,1},
    {158,103,1},{138,111,1},{116,119,1},{92,129,1},{65,139,1},
    {31,153,1},{2,163,14},{2,158,45},{2,155,71},{2,152,92},{2,149,112}
};
static const uint8_t orangeTealCount = sizeof(orangeTeal) / sizeof(orangeTeal[0]);

static const RGB magentaCyan[] = {
    {252,20,162},{251,17,93},{202,1,31},{148,1,6},{126,10,0},
    {140,29,0},{176,59,1},{186,90,1},{155,104,1},{120,118,1},
    {80,133,1},{26,155,1},{2,159,39},{2,153,80},{2,149,114},{2,144,145}
};
static const uint8_t magentaCyanCount = sizeof(magentaCyan) / sizeof(magentaCyan[0]);

static const RGB sunsetSky[] = {
    {50,40,130},{70,40,125},{95,45,120},{125,50,115},{150,55,105},
    {175,50,85},{195,50,70},{210,60,55},{220,90,60},{215,115,55},
    {200,120,40},{185,95,20},{175,60,5},{170,40,2},{185,70,1},{210,110,15}
};
static const uint8_t sunsetSkyCount = sizeof(sunsetSky) / sizeof(sunsetSky[0]);

static const RGB blueWash[] = {
    {32,72,251},{37,74,220},{43,76,191},{49,77,165},
    {56,78,140},{63,78,118},{70,78,97},{78,78,78}
};
static const uint8_t blueWashCount = sizeof(blueWash) / sizeof(blueWash[0]);

static const RGB redWash[] = {
    {145,1,4},{128,7,7},{111,12,11},{95,18,16},
    {79,23,20},{65,28,25},{50,32,31},{37,37,37}
};
static const uint8_t redWashCount = sizeof(redWash) / sizeof(redWash[0]);

static const RGB greenWash[] = {
    {2,163,12},{16,156,23},{30,148,35},{45,141,48},
    {60,133,62},{76,125,76},{92,116,91},{108,108,108}
};
static const uint8_t greenWashCount = sizeof(greenWash) / sizeof(greenWash[0]);

static const RGB purpleWash[] = {
    {29,0,79},{27,3,67},{24,6,55},{22,8,45},
    {20,10,36},{18,12,28},{17,14,21},{15,15,15}
};
static const uint8_t purpleWashCount = sizeof(purpleWash) / sizeof(purpleWash[0]);

static const RGB goldWash[] = {
    {166,100,1},{157,102,13},{148,104,25},{139,105,39},
    {131,106,54},{123,107,70},{115,107,88},{108,108,108}
};
static const uint8_t goldWashCount = sizeof(goldWash) / sizeof(goldWash[0]);

// ═══════════════════════════════════════════════════════════════════
// Strip render functions
// ═══════════════════════════════════════════════════════════════════

void renderStripRainbow() {
    uint32_t ms = millis();
    uint16_t n = testStrip.numPixels();
    uint8_t br = gammaHybrid(state.brightness);
    uint16_t baseHue = (uint16_t)((uint64_t)(ms % state.cycleTimeMs) * 65536ULL / state.cycleTimeMs);

    for (uint16_t i = 0; i < n; i++) {
        uint16_t hue = baseHue + (i * 65536UL / n);
        uint8_t idx = hue >> 8;
        uint8_t r, g, b;
        lutRead(oklchVarL, idx, r, g, b);
        applyChroma(r, g, b, state.chroma);
        r = (uint16_t)r * br / 255;
        g = (uint16_t)g * br / 255;
        b = (uint16_t)b * br / 255;
        testStrip.setPixelColor(i, r, g, b);
    }
    testStrip.show();
}

static uint32_t lerpPalette(const RGB *pal, uint8_t count, float t) {
    float idx = t * (count - 1);
    uint8_t lo = (uint8_t)idx;
    uint8_t hi = lo + 1;
    if (hi >= count) hi = count - 1;
    float frac = idx - lo;
    uint8_t r = pal[lo].r + (pal[hi].r - pal[lo].r) * frac;
    uint8_t g = pal[lo].g + (pal[hi].g - pal[lo].g) * frac;
    uint8_t b = pal[lo].b + (pal[hi].b - pal[lo].b) * frac;
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

void renderStripGradient() {
    uint16_t n = testStrip.numPixels();
    const RGB *pal;
    uint8_t count;
    switch (state.palette) {
        case OKLCH_RAINBOW:     pal = oklchRainbow;     count = oklchRainbowCount;     break;
        case RED_BLUE:          pal = redBlue;          count = redBlueCount;          break;
        case CYAN_GOLD:         pal = cyanGold;         count = cyanGoldCount;         break;
        case GREEN_PURPLE:      pal = greenPurple;      count = greenPurpleCount;      break;
        case ORANGE_TEAL:       pal = orangeTeal;       count = orangeTealCount;       break;
        case MAGENTA_CYAN:      pal = magentaCyan;      count = magentaCyanCount;      break;
        case SUNSET_SKY:        pal = sunsetSky;        count = sunsetSkyCount;        break;
        case BLUE_WASH:         pal = blueWash;         count = blueWashCount;         break;
        case RED_WASH:          pal = redWash;          count = redWashCount;          break;
        case GREEN_WASH:        pal = greenWash;        count = greenWashCount;        break;
        case PURPLE_WASH:       pal = purpleWash;       count = purpleWashCount;       break;
        case GOLD_WASH:         pal = goldWash;         count = goldWashCount;         break;
        case CUSTOM:
            if (state.customCount > 0) {
                pal = state.customStops;
                count = state.customCount;
            } else {
                pal = sapFlow; count = sapFlowCount;
            }
            break;
        default:                pal = sapFlow;          count = sapFlowCount;          break;
    }

    uint8_t br = gammaHybrid(state.brightness);
    for (uint16_t i = 0; i < n; i++) {
        float t = (count > 1) ? (float)i / (n - 1) : 0.0f;
        uint32_t c = lerpPalette(pal, count, t);
        uint8_t r = (c >> 16) & 0xFF;
        uint8_t g = (c >> 8) & 0xFF;
        uint8_t b = c & 0xFF;
        applyChroma(r, g, b, state.chroma);
        r = (uint16_t)r * br / 255;
        g = (uint16_t)g * br / 255;
        b = (uint16_t)b * br / 255;
        testStrip.setPixelColor(i, r, g, b);
    }
    testStrip.show();
}

// ═══════════════════════════════════════════════════════════════════
// Web UI (PROGMEM)
// ═══════════════════════════════════════════════════════════════════

static const char HTML_PAGE[] PROGMEM = R"rawliteral(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tree Strip</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    min-height: 100vh; padding: 20px;
    display: flex; flex-direction: column; align-items: center;
  }
  h1 { font-size: 1.8em; margin-bottom: 4px; color: #fff; }
  h2 { font-size: 1.1em; margin-bottom: 24px; color: #888; font-weight: 400; }
  .section { width: 100%; max-width: 400px; margin-bottom: 20px; }
  .section-label { font-size: 0.85em; color: #888; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px; }
  .btn-group { display: flex; gap: 8px; }
  .btn {
    flex: 1; padding: 12px; border: 2px solid #333; border-radius: 8px;
    background: #16213e; color: #e0e0e0; font-size: 1em; cursor: pointer;
    transition: all 0.2s; text-align: center;
  }
  .btn:hover { border-color: #555; }
  .btn.active { border-color: #e94560; }
  .btn-effect.active { background: #e94560; color: #fff; }
  .slider-container { display: flex; align-items: center; gap: 12px; }
  .slider-container input[type=range] { flex: 1; accent-color: #e94560; }
  .slider-val { min-width: 40px; text-align: right; font-variant-numeric: tabular-nums; }
  .palette-group-label { font-size: 0.75em; color: #666; text-transform: uppercase;
    letter-spacing: 1px; margin: 10px 0 4px 0; }
  .palette-group-label:first-child { margin-top: 0; }
  .palette-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .solid-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .palette-btn {
    height: 48px; border-radius: 8px; border: 2px solid #333;
    cursor: pointer; transition: all 0.2s;
  }
  .palette-btn.active { border-color: #e94560; box-shadow: 0 0 8px rgba(233,69,96,0.5); }
  .palette-btn:hover { border-color: #555; }
  .status {
    margin-top: 20px; padding: 12px; background: #0f3460; border-radius: 8px;
    font-size: 0.85em; text-align: center; width: 100%; max-width: 400px;
  }
  .hidden { display: none; }
</style>
</head>
<body>
<h1>Tree Strip</h1>
<h2>150 LEDs on D32</h2>

<div class="section">
  <div class="section-label">Effect</div>
  <div class="btn-group">
    <button class="btn btn-effect" id="btn-rainbow" onclick="setEffect('rainbow')">Rainbow</button>
    <button class="btn btn-effect" id="btn-gradient" onclick="setEffect('gradient')">Gradient</button>
  </div>
</div>

<div class="section">
  <div class="section-label">Brightness</div>
  <div class="slider-container">
    <input type="range" id="brightness" min="0" max="255" value="128"
      oninput="document.getElementById('br-val').textContent=this.value"
      onchange="postVal('/brightness','brightness',this.value)">
    <span class="slider-val" id="br-val">128</span>
  </div>
</div>

<div class="section">
  <div class="section-label">Chroma</div>
  <div class="slider-container">
    <input type="range" id="chroma" min="0" max="255" value="255"
      oninput="document.getElementById('ch-val').textContent=Math.round(this.value/2.55)+'%'"
      onchange="postVal('/chroma','chroma',this.value)">
    <span class="slider-val" id="ch-val">100%</span>
  </div>
</div>

<div class="section" id="sec-cycletime">
  <div class="section-label">Cycle Time</div>
  <div class="slider-container">
    <input type="range" id="cycletime" min="0" max="1000" step="1" value="71"
      oninput="document.getElementById('ct-val').textContent=fmtCycle(sliderToMs(this.value))"
      onchange="postVal('/cycletime','cycletime',sliderToMs(this.value))">
    <span class="slider-val" id="ct-val">8.0s</span>
  </div>
</div>

<div class="section hidden" id="sec-palette">
  <div class="section-label">Palette</div>
  <div id="palette-container"></div>
</div>

<div class="status" id="status">Connecting...</div>

<script>
var GRADIENTS = [
  {id:'sap_flow', css:'linear-gradient(90deg, rgb(0,40,0), rgb(0,70,0), rgb(10,110,10), rgb(34,180,34), rgb(80,255,80))'},
  {id:'oklch_rainbow', css:'linear-gradient(90deg, rgb(251,3,64), rgb(188,82,1), rgb(2,155,72), rgb(3,129,252), rgb(184,1,169))'},
  {id:'red_blue', css:'linear-gradient(90deg, rgb(251,3,64), rgb(252,30,140), rgb(126,1,145), rgb(29,0,79), rgb(26,48,250))'},
  {id:'cyan_gold', css:'linear-gradient(90deg, rgb(2,144,145), rgb(29,118,252), rgb(53,0,97), rgb(150,1,7), rgb(166,100,1))'},
  {id:'green_purple', css:'linear-gradient(90deg, rgb(2,163,12), rgb(2,149,108), rgb(2,135,212), rgb(30,67,251), rgb(40,0,86))'},
  {id:'orange_teal', css:'linear-gradient(90deg, rgb(137,27,0), rgb(188,82,1), rgb(116,119,1), rgb(2,163,14), rgb(2,149,112))'},
  {id:'magenta_cyan', css:'linear-gradient(90deg, rgb(252,20,162), rgb(148,1,6), rgb(186,90,1), rgb(26,155,1), rgb(2,144,145))'},
  {id:'sunset_sky', css:'linear-gradient(90deg, rgb(50,40,130), rgb(125,50,115), rgb(210,60,55), rgb(175,60,5), rgb(210,110,15))'}
];
var SOLIDS = [
  {id:'s_red',     hex:'fb0340', css:'rgb(251,3,64)'},
  {id:'s_orange',  hex:'bc5201', css:'rgb(188,82,1)'},
  {id:'s_yellow',  hex:'a66401', css:'rgb(166,100,1)'},
  {id:'s_green',   hex:'02a30c', css:'rgb(2,163,12)'},
  {id:'s_cyan',    hex:'029091', css:'rgb(2,144,145)'},
  {id:'s_blue',    hex:'2048fb', css:'rgb(32,72,251)'},
  {id:'s_purple',  hex:'1d004f', css:'rgb(29,0,79)'},
  {id:'s_magenta', hex:'fc14a2', css:'rgb(252,20,162)'}
];

var container = document.getElementById('palette-container');
var allPaletteIds = [];
function buildGrid(items, gridClass) {
  var grid = document.createElement('div');
  grid.className = gridClass || 'palette-grid';
  items.forEach(function(p) {
    allPaletteIds.push(p.id);
    var btn = document.createElement('button');
    btn.className = 'palette-btn';
    btn.id = 'btn-' + p.id;
    btn.style.background = p.css;
    if (p.hex) {
      btn.onclick = function() { setCustomColor(p.hex); };
    } else {
      btn.onclick = function() { setPalette(p.id); };
    }
    grid.appendChild(btn);
  });
  return grid;
}
function addLabel(text) {
  var lbl = document.createElement('div');
  lbl.className = 'palette-group-label';
  lbl.textContent = text;
  container.appendChild(lbl);
}
addLabel('Gradients');
container.appendChild(buildGrid(GRADIENTS, 'palette-grid'));
addLabel('Solid Colors');
container.appendChild(buildGrid(SOLIDS, 'solid-grid'));

function sliderToMs(pos) {
  pos = Number(pos);
  if (pos <= 600) return Math.round(1000 + pos * (59000 / 600));
  return Math.round(60000 * Math.pow(10, (pos - 600) / 400));
}
function msToSlider(ms) {
  if (ms <= 60000) return Math.round((ms - 1000) * 600 / 59000);
  return Math.round(600 + 400 * Math.log10(ms / 60000));
}
function fmtCycle(ms) {
  var s = ms / 1000;
  if (s < 60) return s.toFixed(1) + 's';
  var m = Math.floor(s / 60), sec = Math.round(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}
function postVal(path, key, val) {
  fetch(path, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: key+'='+val});
}
function setEffect(e) {
  fetch('/effect', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: 'effect='+e}).then(function(){pollStatus()});
}
function setPalette(p) {
  fetch('/palette', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: 'palette='+p}).then(function(){pollStatus()});
}
function setCustomColor(hex) {
  fetch('/custom', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: 'c='+hex}).then(function(){pollStatus()});
}
function updateUI(s) {
  document.getElementById('btn-rainbow').classList.toggle('active', s.effect==='rainbow');
  document.getElementById('btn-gradient').classList.toggle('active', s.effect==='gradient');
  document.getElementById('brightness').value = s.brightness;
  document.getElementById('br-val').textContent = s.brightness;
  document.getElementById('chroma').value = s.chroma;
  document.getElementById('ch-val').textContent = Math.round(s.chroma/2.55)+'%';
  document.getElementById('cycletime').value = msToSlider(s.cycleTime);
  document.getElementById('ct-val').textContent = fmtCycle(s.cycleTime);
  document.getElementById('sec-cycletime').classList.toggle('hidden', s.effect!=='rainbow');
  document.getElementById('sec-palette').classList.toggle('hidden', s.effect!=='gradient');
  allPaletteIds.forEach(function(pid) {
    var el = document.getElementById('btn-' + pid);
    if (el) el.classList.toggle('active', s.palette === pid);
  });
  document.getElementById('status').textContent = 'IP: '+s.ip+' | Signal: '+s.rssi+' dBm';
}
function pollStatus() {
  fetch('/status').then(function(r){return r.json()}).then(updateUI).catch(function(){
    document.getElementById('status').textContent = 'Disconnected';
  });
}
pollStatus();
setInterval(pollStatus, 5000);
</script>
</body>
</html>)rawliteral";

// ═══════════════════════════════════════════════════════════════════
// Web handlers
// ═══════════════════════════════════════════════════════════════════

static const char* paletteName(Palette p) {
    switch (p) {
        case SAP_FLOW:      return "sap_flow";
        case OKLCH_RAINBOW: return "oklch_rainbow";
        case RED_BLUE:      return "red_blue";
        case CYAN_GOLD:     return "cyan_gold";
        case GREEN_PURPLE:  return "green_purple";
        case ORANGE_TEAL:   return "orange_teal";
        case MAGENTA_CYAN:  return "magenta_cyan";
        case SUNSET_SKY:    return "sunset_sky";
        case BLUE_WASH:     return "blue_wash";
        case RED_WASH:      return "red_wash";
        case GREEN_WASH:    return "green_wash";
        case PURPLE_WASH:   return "purple_wash";
        case GOLD_WASH:     return "gold_wash";
        case CUSTOM:        return "custom";
        default:            return "oklch_rainbow";
    }
}

void handleRoot() {
    server.send_P(200, "text/html", HTML_PAGE);
}

void handleStatus() {
    String json = "{";
    json += "\"effect\":\"" + String(state.effect == RAINBOW ? "rainbow" : "gradient") + "\",";
    json += "\"brightness\":" + String(state.brightness) + ",";
    json += "\"cycleTime\":" + String(state.cycleTimeMs) + ",";
    json += "\"palette\":\"" + String(paletteName(state.palette)) + "\",";
    json += "\"chroma\":" + String(state.chroma) + ",";
    json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
    json += "\"ip\":\"" + WiFi.localIP().toString() + "\"";
    json += "}";
    server.send(200, "application/json", json);
}

void handleSetEffect() {
    if (server.hasArg("effect")) {
        String e = server.arg("effect");
        if (e == "rainbow") state.effect = RAINBOW;
        else if (e == "gradient") state.effect = GRADIENT;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetBrightness() {
    if (server.hasArg("brightness")) {
        int b = server.arg("brightness").toInt();
        if (b >= 0 && b <= 255) state.brightness = b;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetCycleTime() {
    if (server.hasArg("cycletime")) {
        long ct = server.arg("cycletime").toInt();
        if (ct >= 1000 && ct <= 600000) state.cycleTimeMs = ct;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetPalette() {
    if (server.hasArg("palette")) {
        String p = server.arg("palette");
        if (p == "sap_flow")           state.palette = SAP_FLOW;
        else if (p == "oklch_rainbow") state.palette = OKLCH_RAINBOW;
        else if (p == "red_blue")      state.palette = RED_BLUE;
        else if (p == "cyan_gold")     state.palette = CYAN_GOLD;
        else if (p == "green_purple")  state.palette = GREEN_PURPLE;
        else if (p == "orange_teal")   state.palette = ORANGE_TEAL;
        else if (p == "magenta_cyan")  state.palette = MAGENTA_CYAN;
        else if (p == "sunset_sky")    state.palette = SUNSET_SKY;
        else if (p == "blue_wash")     state.palette = BLUE_WASH;
        else if (p == "red_wash")      state.palette = RED_WASH;
        else if (p == "green_wash")    state.palette = GREEN_WASH;
        else if (p == "purple_wash")   state.palette = PURPLE_WASH;
        else if (p == "gold_wash")     state.palette = GOLD_WASH;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetChroma() {
    if (server.hasArg("chroma")) {
        int c = server.arg("chroma").toInt();
        if (c >= 0 && c <= 255) state.chroma = c;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetCustomPalette() {
    if (server.hasArg("c")) {
        String s = server.arg("c");
        uint8_t count = 0;
        int start = 0;
        while (start < (int)s.length() && count < 16) {
            int comma = s.indexOf(',', start);
            if (comma < 0) comma = s.length();
            String hex = s.substring(start, comma);
            uint32_t val = strtoul(hex.c_str(), NULL, 16);
            state.customStops[count].r = (val >> 16) & 0xFF;
            state.customStops[count].g = (val >> 8) & 0xFF;
            state.customStops[count].b = val & 0xFF;
            count++;
            start = comma + 1;
        }
        state.customCount = count;
        state.palette = CUSTOM;
        state.effect = GRADIENT;
    }
    server.send(200, "text/plain", "OK");
}

// ═══════════════════════════════════════════════════════════════════
// Setup & Loop
// ═══════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    Serial.println("=== TREE + STRIP ===");

    // Tree
    tree.begin();
    Serial.print("Tree LEDs: ");
    Serial.println(tree.getNumLEDs());

    // Strip
    testStrip.begin();
    testStrip.setBrightness(255);
    testStrip.clear();
    testStrip.show();
    Serial.printf("Strip: %d LEDs on pin %d\n", TEST_STRIP_LEDS, TEST_STRIP_PIN);

    // WiFi
    WiFi.mode(WIFI_STA);
    WiFi.setHostname("tree-esp32");
    WiFi.setMinSecurity(WIFI_AUTH_WPA_PSK);
    Serial.printf("[WiFi] Connecting to '%s'...", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 15000) {
        delay(500);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());

        // OTA
        ArduinoOTA.setHostname("tree-esp32");
        ArduinoOTA.onStart([]() { Serial.println("[OTA] Start"); });
        ArduinoOTA.onEnd([]() { Serial.println("\n[OTA] Done"); });
        ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
            Serial.printf("[OTA] %u%%\r", progress * 100 / total);
        });
        ArduinoOTA.begin();
        Serial.println("[OTA] Ready");

        // Web server
        server.on("/", HTTP_GET, handleRoot);
        server.on("/status", HTTP_GET, handleStatus);
        server.on("/effect", HTTP_POST, handleSetEffect);
        server.on("/brightness", HTTP_POST, handleSetBrightness);
        server.on("/cycletime", HTTP_POST, handleSetCycleTime);
        server.on("/palette", HTTP_POST, handleSetPalette);
        server.on("/chroma", HTTP_POST, handleSetChroma);
        server.on("/custom", HTTP_POST, handleSetCustomPalette);
        server.begin();
        Serial.println("[Web] Server started on port 80");
    } else {
        Serial.printf("\n[WiFi] FAILED (status=%d) — running without network\n", WiFi.status());
    }

    Serial.println("Ready!");
}

void loop() {
    ArduinoOTA.handle();
    server.handleClient();

    // Tree: standalone sap flow (always runs)
    static SapFlowForeground sap(&tree, 34, 139, 34, 8);
    sap.update();
    sap.render();

    // Strip: web-controlled effect
    if (state.effect == GRADIENT) {
        renderStripGradient();
    } else {
        renderStripRainbow();
    }

    delay(16); // ~60fps
}
