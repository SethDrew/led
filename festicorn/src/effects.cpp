#include "effects.h"

#ifndef LED_OFFSET
#define LED_OFFSET 0
#endif

uint8_t devColorBuf[NUM_PIXELS * 3] = {0};
bool devColorFresh = false;
float effectPhase = 0.0f;

// ── Adafruit gamma8 table (gamma=2.8) ──────────────────────────────
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

// ── Hybrid gamma: perceptual curve below 60%, linear ramp to max ───
static const uint8_t CROSSOVER = 153;  // 60% of 255

uint8_t gammaHybrid(uint8_t v) {
    if (v == 0) return 0;
    uint8_t g;
    if (v <= CROSSOVER) {
        g = gamma8[v];
    } else {
        uint8_t base = gamma8[CROSSOVER];  // = 137
        g = (uint8_t)(base + (uint16_t)(v - CROSSOVER) * (255 - base) / (255 - CROSSOVER));
    }
    // Floor: gamma8 maps 0-40 to 0-1, but br=1 gives all-black after
    // integer truncation (252*1/255=0). Minimum br=2 ensures at least
    // 1-bit visible output for any non-zero brightness.
    return g < 2 ? 2 : g;
}

// ── BT.601 chroma desaturation ────────────────────────────────────
// Blend toward per-pixel luminance grey.
// Source: ITU-R BT.601-7 (2011), Y = 0.299R + 0.587G + 0.114B.
// Cited in COLOR_ENGINEERING.md — not yet validated on hardware (ledger
// entry oklch-color-solid-coverage, status: spark, confidence: high).
static void applyChroma(uint8_t &r, uint8_t &g, uint8_t &b, uint8_t chroma) {
    if (chroma >= 255) return;
    uint16_t grey = ((uint16_t)r * 77 + (uint16_t)g * 150 + (uint16_t)b * 29) >> 8;
    // 77/256≈0.299, 150/256≈0.587, 29/256≈0.114
    r = (uint8_t)(((uint16_t)grey * (255 - chroma) + (uint16_t)r * chroma) / 255);
    g = (uint8_t)(((uint16_t)grey * (255 - chroma) + (uint16_t)g * chroma) / 255);
    b = (uint8_t)(((uint16_t)grey * (255 - chroma) + (uint16_t)b * chroma) / 255);
}

// ── OKLCH constant-L rainbow LUT ───────────────────────────────────
// L=0.75 everywhere, per-hue max chroma (98% of gamut boundary). 256 entries.
// OKLab -> linear sRGB -> 8-bit. No additional gamma needed.
static const uint8_t oklchConstL[256][3] = {
    {252,  53, 102}, {252,  53,  99}, {252,  54,  95}, {252,  54,  92},
    {252,  55,  89}, {252,  55,  86}, {252,  56,  83}, {252,  56,  80},
    {252,  57,  77}, {252,  57,  75}, {252,  57,  72}, {252,  58,  69},
    {252,  58,  67}, {252,  58,  64}, {252,  59,  62}, {252,  59,  59},
    {252,  59,  57}, {252,  60,  55}, {252,  60,  52}, {252,  60,  50},
    {252,  61,  48}, {252,  61,  45}, {252,  61,  43}, {252,  62,  41},
    {252,  62,  39}, {252,  62,  37}, {252,  63,  35}, {252,  63,  32},
    {252,  63,  30}, {252,  64,  28}, {252,  64,  26}, {252,  64,  24},
    {252,  65,  22}, {252,  65,  20}, {252,  65,  17}, {252,  65,  15},
    {252,  66,  13}, {252,  66,  11}, {252,  66,   9}, {252,  67,   7},
    {252,  67,   4}, {252,  67,   2}, {249,  69,   1}, {244,  70,   1},
    {240,  72,   1}, {235,  74,   1}, {231,  75,   1}, {227,  77,   1},
    {223,  78,   1}, {219,  80,   1}, {215,  81,   1}, {211,  83,   1},
    {207,  84,   1}, {204,  86,   1}, {200,  87,   1}, {196,  88,   1},
    {193,  90,   1}, {189,  91,   1}, {186,  92,   1}, {183,  94,   1},
    {179,  95,   1}, {176,  96,   1}, {173,  98,   1}, {169,  99,   1},
    {166, 100,   1}, {163, 101,   1}, {159, 103,   1}, {156, 104,   1},
    {153, 105,   1}, {149, 107,   1}, {146, 108,   1}, {143, 109,   1},
    {139, 110,   1}, {136, 112,   1}, {133, 113,   1}, {129, 114,   1},
    {126, 116,   1}, {122, 117,   1}, {119, 118,   1}, {115, 120,   1},
    {111, 121,   1}, {108, 123,   1}, {104, 124,   1}, {100, 126,   1},
    { 96, 127,   1}, { 92, 129,   1}, { 88, 131,   1}, { 83, 132,   1},
    { 79, 134,   1}, { 74, 136,   1}, { 70, 138,   1}, { 65, 139,   1},
    { 60, 141,   1}, { 55, 143,   1}, { 49, 146,   1}, { 44, 148,   1},
    { 38, 150,   1}, { 32, 153,   1}, { 25, 155,   1}, { 19, 158,   1},
    { 12, 161,   1}, {  4, 164,   1}, {  2, 164,   6}, {  2, 163,  12},
    {  2, 162,  17}, {  2, 161,  23}, {  2, 161,  28}, {  2, 160,  33},
    {  2, 159,  38}, {  2, 159,  43}, {  2, 158,  47}, {  2, 157,  52},
    {  2, 157,  56}, {  2, 156,  60}, {  2, 156,  64}, {  2, 155,  68},
    {  2, 155,  72}, {  2, 154,  76}, {  2, 154,  79}, {  2, 153,  83},
    {  2, 153,  86}, {  2, 152,  90}, {  2, 152,  93}, {  2, 151,  96},
    {  2, 151, 100}, {  2, 150, 103}, {  2, 150, 106}, {  2, 149, 109},
    {  2, 149, 112}, {  2, 148, 115}, {  2, 148, 118}, {  2, 148, 121},
    {  2, 147, 125}, {  2, 147, 128}, {  2, 146, 131}, {  2, 146, 134},
    {  2, 145, 137}, {  2, 145, 140}, {  2, 145, 143}, {  2, 144, 146},
    {  2, 144, 149}, {  2, 143, 152}, {  2, 143, 155}, {  2, 142, 158},
    {  2, 142, 161}, {  2, 141, 165}, {  2, 141, 168}, {  2, 141, 171},
    {  2, 140, 175}, {  2, 140, 178}, {  2, 139, 182}, {  2, 139, 185},
    {  2, 138, 189}, {  2, 137, 193}, {  2, 137, 196}, {  2, 136, 200},
    {  2, 136, 204}, {  2, 135, 209}, {  2, 135, 213}, {  2, 134, 217},
    {  2, 133, 222}, {  2, 133, 227}, {  2, 132, 232}, {  2, 131, 237},
    {  2, 130, 243}, {  2, 130, 248}, {  3, 129, 252}, {  6, 127, 252},
    {  9, 126, 252}, { 12, 125, 252}, { 15, 124, 252}, { 18, 122, 252},
    { 21, 121, 252}, { 24, 120, 252}, { 26, 119, 252}, { 29, 118, 252},
    { 31, 117, 252}, { 34, 116, 252}, { 37, 115, 252}, { 39, 114, 252},
    { 42, 113, 252}, { 44, 112, 252}, { 46, 111, 252}, { 49, 110, 252},
    { 51, 109, 252}, { 54, 108, 252}, { 56, 107, 252}, { 58, 106, 252},
    { 61, 105, 252}, { 63, 104, 252}, { 65, 104, 252}, { 68, 103, 252},
    { 70, 102, 252}, { 72, 101, 252}, { 75, 100, 252}, { 77,  99, 252},
    { 80,  98, 252}, { 82,  97, 252}, { 85,  96, 252}, { 87,  95, 252},
    { 90,  94, 252}, { 92,  93, 252}, { 95,  92, 252}, { 98,  91, 252},
    {100,  90, 252}, {103,  89, 252}, {106,  87, 252}, {109,  86, 252},
    {112,  85, 252}, {115,  84, 252}, {118,  83, 252}, {122,  81, 252},
    {125,  80, 252}, {128,  79, 252}, {132,  77, 252}, {136,  76, 252},
    {140,  74, 252}, {144,  73, 252}, {148,  71, 252}, {152,  69, 252},
    {157,  68, 252}, {162,  66, 252}, {167,  64, 252}, {172,  62, 252},
    {178,  60, 252}, {184,  57, 252}, {190,  55, 252}, {197,  52, 252},
    {204,  49, 252}, {212,  46, 252}, {221,  43, 252}, {230,  40, 252},
    {240,  36, 252}, {251,  32, 252}, {252,  33, 241}, {252,  34, 229},
    {252,  36, 219}, {252,  37, 209}, {252,  39, 200}, {252,  40, 191},
    {252,  41, 183}, {252,  42, 176}, {252,  43, 169}, {252,  44, 163},
    {252,  45, 156}, {252,  46, 151}, {252,  47, 145}, {252,  47, 140},
    {252,  48, 135}, {252,  49, 130}, {252,  49, 126}, {252,  50, 121},
    {252,  51, 117}, {252,  51, 113}, {252,  52, 109}, {252,  52, 106},
};

// ── OKLCH variable-L rainbow LUT ───────────────────────────────────
// Hue-dependent lightness: red (~30 deg) L~0.52, purple (~300 deg) L~0.38,
// green/cyan/yellow L=0.75. Smooth cosine interpolation between dips.
// Per-hue max chroma at 98% gamut boundary. Generated by gen_oklch_varL_lut.py.
static const uint8_t oklchVarL[256][3] = {
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

// ── Rainbow Cycle (OKLCH variable-L, full strip) ───────────────────
void renderRainbow(Adafruit_NeoPixel &strip, const EffectState &state) {
    uint16_t n = strip.numPixels();

    // Hybrid gamma on brightness scalar (not per-channel)
    uint8_t br = gammaHybrid(state.brightness);

    // Clear unused leading pixels
    for (uint16_t i = 0; i < LED_OFFSET && i < n; i++) {
        strip.setPixelColor(i, 0);
    }

    uint16_t visible = n - LED_OFFSET;
    uint16_t baseHue = (uint16_t)(effectPhase * 65536.0f);

    for (uint16_t i = 0; i < visible; i++) {
        uint16_t hue = baseHue + (i * 65536UL / visible);
        uint8_t idx = hue >> 8;
        uint8_t r = (uint16_t)oklchVarL[idx][0] * br / 255;
        uint8_t g = (uint16_t)oklchVarL[idx][1] * br / 255;
        uint8_t b = (uint16_t)oklchVarL[idx][2] * br / 255;
        applyChroma(r, g, b, state.chroma);
        strip.setPixelColor(i + LED_OFFSET, r, g, b);
    }
}

// ── Color Gradient ─────────────────────────────────────────────────
// Linear interpolation across palette color stops mapped 0 -> NUM_PIXELS-1.

struct RGB {
    uint8_t r, g, b;
};

// --- Legacy palette: Sap Flow (non-OKLCH, hand-tuned) ---
static const RGB sapFlow[] = {
    {0, 40, 0},       // very dark green
    {0, 70, 0},       // dark green
    {10, 110, 10},    // mid-dark green
    {34, 180, 34},    // mid green
    {80, 255, 80}     // bright green
};
static const uint8_t sapFlowCount = sizeof(sapFlow) / sizeof(sapFlow[0]);

// --- OKLCH variable-L rainbow — 5 stops sampled from oklchVarL LUT ---
static const RGB oklchRainbow[] = {
    {251,   3,  64},  // red    (idx 0)
    {188,  82,   1},  // yellow (idx 54)
    {  2, 155,  72},  // green  (idx 116)
    {  3, 129, 252},  // blue   (idx 160)
    {184,   1, 169}   // purple (idx 232)
};
static const uint8_t oklchRainbowCount = sizeof(oklchRainbow) / sizeof(oklchRainbow[0]);

// ── Category 2: OKLCH hue-arc gradients (computed by gen_palettes.py) ──

// Red -> Blue: hue 0 -> 270 deg OKLCH (CW via magenta/purple, non-uniform)
static const RGB redBlue[] = {
    {251,   3,  64},
    {251,  15,  90},
    {252,  26, 117},
    {252,  30, 140},
    {252,  20, 162},
    {239,   1, 181},
    {179,   1, 167},
    {126,   1, 145},
    { 86,   1, 121},
    { 57,   0, 100},
    { 40,   0,  86},
    { 29,   0,  79},
    { 24,   0,  82},
    { 23,   0, 130},
    { 25,   6, 248},
    { 26,  48, 250}
};
static const uint8_t redBlueCount = sizeof(redBlue) / sizeof(redBlue[0]);

// Cyan -> Gold: hue 195 -> 90 deg OKLCH
static const RGB cyanGold[] = {
    {  2, 144, 145},
    {  2, 139, 184},
    {  2, 131, 236},
    { 29, 118, 252},
    { 34,  80, 251},
    { 25,   6, 248},
    { 23,   0,  90},
    { 53,   0,  97},
    {191,   1, 171},
    {252,  30, 127},
    {224,   1,  43},
    {150,   1,   7},
    {125,  12,   0},
    {151,  37,   1},
    {187,  76,   1},
    {166, 100,   1}
};
static const uint8_t cyanGoldCount = sizeof(cyanGold) / sizeof(cyanGold[0]);

// Green -> Purple: hue 145 -> 310 deg OKLCH
static const RGB greenPurple[] = {
    {  2, 163,  12},
    {  2, 157,  51},
    {  2, 153,  82},
    {  2, 149, 108},
    {  2, 146, 132},
    {  2, 143, 156},
    {  2, 139, 182},
    {  2, 135, 212},
    {  2, 129, 252},
    { 25, 120, 252},
    { 40, 108, 252},
    { 30,  67, 251},
    { 24,  17, 249},
    { 23,   0, 149},
    { 23,   0,  84},
    { 40,   0,  86}
};
static const uint8_t greenPurpleCount = sizeof(greenPurple) / sizeof(greenPurple[0]);

// Orange -> Teal: hue 50 -> 180 deg OKLCH
static const RGB orangeTeal[] = {
    {137,  27,   0},
    {158,  43,   1},
    {179,  63,   1},
    {188,  82,   1},
    {179,  95,   1},
    {158, 103,   1},
    {138, 111,   1},
    {116, 119,   1},
    { 92, 129,   1},
    { 65, 139,   1},
    { 31, 153,   1},
    {  2, 163,  14},
    {  2, 158,  45},
    {  2, 155,  71},
    {  2, 152,  92},
    {  2, 149, 112}
};
static const uint8_t orangeTealCount = sizeof(orangeTeal) / sizeof(orangeTeal[0]);

// Magenta -> Cyan: hue 340 -> 195 deg OKLCH
static const RGB magentaCyan[] = {
    {252,  20, 162},
    {251,  17,  93},
    {202,   1,  31},
    {148,   1,   6},
    {126,  10,   0},
    {140,  29,   0},
    {176,  59,   1},
    {186,  90,   1},
    {155, 104,   1},
    {120, 118,   1},
    { 80, 133,   1},
    { 26, 155,   1},
    {  2, 159,  39},
    {  2, 153,  80},
    {  2, 149, 114},
    {  2, 144, 145}
};
static const uint8_t magentaCyanCount = sizeof(magentaCyan) / sizeof(magentaCyan[0]);

// Sunset Sky: deep indigo -> purple -> dusty rose -> coral -> amber
static const RGB sunsetSky[] = {
    { 50,  40, 130},
    { 70,  40, 125},
    { 95,  45, 120},
    {125,  50, 115},
    {150,  55, 105},
    {175,  50,  85},
    {195,  50,  70},
    {210,  60,  55},
    {220,  90,  60},
    {215, 115,  55},
    {200, 120,  40},
    {185,  95,  20},
    {175,  60,   5},
    {170,  40,   2},
    {185,  70,   1},
    {210, 110,  15}
};
static const uint8_t sunsetSkyCount = sizeof(sunsetSky) / sizeof(sunsetSky[0]);

// ── OKLCH chroma sweeps (computed by gen_palettes.py) ──

// Blue Wash: hue=265 deg, L=0.674, C=0.168 -> 0
static const RGB blueWash[] = {
    { 32,  72, 251},
    { 37,  74, 220},
    { 43,  76, 191},
    { 49,  77, 165},
    { 56,  78, 140},
    { 63,  78, 118},
    { 70,  78,  97},
    { 78,  78,  78}
};
static const uint8_t blueWashCount = sizeof(blueWash) / sizeof(blueWash[0]);

// Red Wash: hue=25 deg, L=0.525, C=0.210 -> 0
static const RGB redWash[] = {
    {145,   1,   4},
    {128,   7,   7},
    {111,  12,  11},
    { 95,  18,  16},
    { 79,  23,  20},
    { 65,  28,  25},
    { 50,  32,  31},
    { 37,  37,  37}
};
static const uint8_t redWashCount = sizeof(redWash) / sizeof(redWash[0]);

// Green Wash: hue=145 deg, L=0.750, C=0.232 -> 0
static const RGB greenWash[] = {
    {  2, 163,  12},
    { 16, 156,  23},
    { 30, 148,  35},
    { 45, 141,  48},
    { 60, 133,  62},
    { 76, 125,  76},
    { 92, 116,  91},
    {108, 108, 108}
};
static const uint8_t greenWashCount = sizeof(greenWash) / sizeof(greenWash[0]);

// Purple Wash: hue=305 deg, L=0.389, C=0.200 -> 0
static const RGB purpleWash[] = {
    { 29,   0,  79},
    { 27,   3,  67},
    { 24,   6,  55},
    { 22,   8,  45},
    { 20,  10,  36},
    { 18,  12,  28},
    { 17,  14,  21},
    { 15,  15,  15}
};
static const uint8_t purpleWashCount = sizeof(purpleWash) / sizeof(purpleWash[0]);

// Gold Wash: hue=90 deg, L=0.750, C=0.151 -> 0
static const RGB goldWash[] = {
    {166, 100,   1},
    {157, 102,  13},
    {148, 104,  25},
    {139, 105,  39},
    {131, 106,  54},
    {123, 107,  70},
    {115, 107,  88},
    {108, 108, 108}
};
static const uint8_t goldWashCount = sizeof(goldWash) / sizeof(goldWash[0]);

// ── Palette interpolation (wrap-around) ───────────────────────────

static uint32_t lerpPalette(const RGB *pal, uint8_t count, float t) {
    // Wrap-around: last stop blends back to first for smooth cycling
    float idx = t * count;
    uint8_t lo = (uint8_t)idx % count;
    uint8_t hi = (lo + 1) % count;
    float frac = idx - (int)idx;
    uint8_t r = pal[lo].r + (pal[hi].r - pal[lo].r) * frac;
    uint8_t g = pal[lo].g + (pal[hi].g - pal[lo].g) * frac;
    uint8_t b = pal[lo].b + (pal[hi].b - pal[lo].b) * frac;
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

void renderGradient(Adafruit_NeoPixel &strip, const EffectState &state) {
    uint16_t n = strip.numPixels();
    const RGB *pal;
    uint8_t count;
    switch (state.palette) {
        case OKLCH_RAINBOW:     pal = oklchRainbow;     count = oklchRainbowCount;     break;
        // Category 2: Hue-arc gradients
        case RED_BLUE:          pal = redBlue;          count = redBlueCount;          break;
        case CYAN_GOLD:         pal = cyanGold;         count = cyanGoldCount;         break;
        case GREEN_PURPLE:      pal = greenPurple;      count = greenPurpleCount;      break;
        case ORANGE_TEAL:       pal = orangeTeal;       count = orangeTealCount;       break;
        case MAGENTA_CYAN:      pal = magentaCyan;      count = magentaCyanCount;      break;
        case SUNSET_SKY:        pal = sunsetSky;        count = sunsetSkyCount;        break;
        // Chroma sweeps
        case BLUE_WASH:         pal = blueWash;         count = blueWashCount;         break;
        case RED_WASH:          pal = redWash;          count = redWashCount;          break;
        case GREEN_WASH:        pal = greenWash;        count = greenWashCount;        break;
        case PURPLE_WASH:       pal = purpleWash;       count = purpleWashCount;       break;
        case GOLD_WASH:         pal = goldWash;         count = goldWashCount;         break;
        default:                pal = sapFlow;          count = sapFlowCount;          break;
    }

    // Hybrid gamma on brightness scalar (not per-channel)
    uint8_t br = gammaHybrid(state.brightness);

    // Clear unused leading pixels
    for (uint16_t i = 0; i < LED_OFFSET && i < n; i++) {
        strip.setPixelColor(i, 0);
    }

    float offset = effectPhase;

    uint16_t visible = n - LED_OFFSET;
    for (uint16_t i = 0; i < visible; i++) {
        float t = (float)i / visible + offset;
        t -= (int)t;
        uint32_t c = lerpPalette(pal, count, t);
        uint8_t r = (uint16_t)((c >> 16) & 0xFF) * br / 255;
        uint8_t g = (uint16_t)((c >> 8) & 0xFF) * br / 255;
        uint8_t b = (uint16_t)(c & 0xFF) * br / 255;
        applyChroma(r, g, b, state.chroma);
        strip.setPixelColor(i + LED_OFFSET, r, g, b);
    }
}

// ── Dev Color (external preview buffer) ──────────────────────────
void renderDevColor(Adafruit_NeoPixel &strip, const EffectState &state) {
    uint16_t n = strip.numPixels();

    // Hybrid gamma on brightness scalar (not per-channel)
    uint8_t br = gammaHybrid(state.brightness);

    // Clear unused leading pixels
    for (uint16_t i = 0; i < LED_OFFSET && i < n; i++) {
        strip.setPixelColor(i, 0);
    }

    uint16_t visible = n - LED_OFFSET;
    for (uint16_t i = 0; i < visible; i++) {
        uint8_t r = (uint16_t)devColorBuf[i * 3 + 0] * br / 255;
        uint8_t g = (uint16_t)devColorBuf[i * 3 + 1] * br / 255;
        uint8_t b = (uint16_t)devColorBuf[i * 3 + 2] * br / 255;
        strip.setPixelColor(i + LED_OFFSET, r, g, b);
    }
}
