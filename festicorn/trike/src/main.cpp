#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <fast_math.h>
#include <oklch_lut.h>

#define LED_PIN     4
#define BTN_PIN     3
#define NUM_PIXELS  150

static Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

enum Effect { RAINBOW, WEAVE, SOLID_PINK, EFFECT_COUNT };

static Effect   effect      = RAINBOW;
static float    brightF     = 128.0f;
static float    effectPhase = 0.0f;

static const uint32_t CYCLE_MS = 8000;
static const float    SPEEDS[] = { 1.0f, 2.0f, 0.5f };
static uint8_t        speedIdx = 0;

static const uint8_t PINK[3] = { 255, 25, 90 };
static const uint8_t TEAL[3] = { 0, 190, 160 };
// Must divide NUM_PIXELS exactly, or the weave jumps when effectPhase wraps.
static const float WEAVE_PERIOD = 15.0f;

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

static const uint8_t CROSSOVER = 153;

static uint8_t gammaHybrid(uint8_t v) {
    if (v == 0) return 0;
    uint8_t g;
    if (v <= CROSSOVER) {
        g = gamma8[v];
    } else {
        uint8_t base = gamma8[CROSSOVER];
        g = (uint8_t)(base + (uint16_t)(v - CROSSOVER) * (255 - base) / (255 - CROSSOVER));
    }
    return g < 2 ? 2 : g;
}

static void renderRainbow() {
    uint8_t br = gammaHybrid((uint8_t)brightF);
    uint16_t baseHue = (uint16_t)(effectPhase * 65536.0f);
    for (uint16_t i = 0; i < NUM_PIXELS; i++) {
        uint16_t hue = baseHue + (uint32_t)i * 65536UL / NUM_PIXELS;
        uint8_t idx = hue >> 8;
        strip.setPixelColor(i,
            (uint16_t)oklchVarL[idx][0] * br / 255,
            (uint16_t)oklchVarL[idx][1] * br / 255,
            (uint16_t)oklchVarL[idx][2] * br / 255);
    }
}

static void renderWeave() {
    uint8_t br = gammaHybrid((uint8_t)brightF);
    float drift = effectPhase * ((float)NUM_PIXELS / WEAVE_PERIOD);
    for (uint16_t i = 0; i < NUM_PIXELS; i++) {
        float t = 0.5f + 0.5f * fastSin(6.2831853f * ((float)i / WEAVE_PERIOD - drift));
        float seam = 0.65f + 0.35f * fabsf(2.0f * t - 1.0f);
        float s = seam * (float)br / 255.0f;
        strip.setPixelColor(i,
            (uint8_t)((PINK[0] * t + TEAL[0] * (1.0f - t)) * s),
            (uint8_t)((PINK[1] * t + TEAL[1] * (1.0f - t)) * s),
            (uint8_t)((PINK[2] * t + TEAL[2] * (1.0f - t)) * s));
    }
}

static void renderSolidPink() {
    uint8_t br = gammaHybrid((uint8_t)brightF);
    strip.fill(strip.Color(
        (uint16_t)PINK[0] * br / 255,
        (uint16_t)PINK[1] * br / 255,
        (uint16_t)PINK[2] * br / 255));
}

#define DEBOUNCE_MS   20
#define HOLD_MS      400
#define DOUBLE_MS    300
#define RAMP_PER_S   150.0f
#define BRIGHT_MIN     8.0f

static bool     rawPrev = false, debounced = false;
static uint32_t btnChangeMs = 0, pressStartMs = 0, clickPendingMs = 0;
static bool     holdActive = false, clickPending = false, doubleHandled = false;
static int8_t   rampDir = 1;

static void nextEffect() {
    effect = (Effect)((effect + 1) % EFFECT_COUNT);
    static const char *names[] = { "rainbow", "weave", "solid_pink" };
    Serial.printf("effect: %s\n", names[effect]);
}

static void updateButton(uint32_t now, float dt) {
    bool raw = digitalRead(BTN_PIN) == LOW;
    if (raw != rawPrev) { btnChangeMs = now; rawPrev = raw; }

    if ((now - btnChangeMs) >= DEBOUNCE_MS && raw != debounced) {
        debounced = raw;
        if (debounced) {
            pressStartMs = now;
            if (clickPending && (now - clickPendingMs) <= DOUBLE_MS) {
                clickPending = false;
                doubleHandled = true;
                speedIdx = (speedIdx + 1) % 3;
                Serial.printf("speed: %.1fx\n", SPEEDS[speedIdx]);
            } else {
                doubleHandled = false;
            }
        } else {
            if (holdActive) {
                holdActive = false;
                rampDir = -rampDir;
            } else if (!doubleHandled && (now - pressStartMs) < HOLD_MS) {
                clickPending = true;
                clickPendingMs = now;
            }
        }
    }

    if (debounced && !holdActive && !doubleHandled && (now - pressStartMs) >= HOLD_MS) {
        holdActive = true;
        clickPending = false;
        Serial.printf("brightness ramp %s\n", rampDir > 0 ? "up" : "down");
    }

    if (holdActive) {
        brightF += (float)rampDir * RAMP_PER_S * dt;
        if (brightF > 255.0f) brightF = 255.0f;
        if (brightF < BRIGHT_MIN) brightF = BRIGHT_MIN;
    }

    if (clickPending && (now - clickPendingMs) > DOUBLE_MS) {
        clickPending = false;
        nextEffect();
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(BTN_PIN, INPUT_PULLUP);
    strip.begin();
    strip.setBrightness(255);
    strip.show();
    Serial.println("trike ready — press: effect, hold: brightness, double: speed");
}

void loop() {
    uint32_t now = millis();
    static uint32_t lastFrameMs = 0;
    float dt = (lastFrameMs > 0) ? (now - lastFrameMs) / 1000.0f : (1.0f / 60.0f);
    if (dt > 0.1f) dt = 0.1f;
    lastFrameMs = now;

    updateButton(now, dt);

    effectPhase += (dt * 1000.0f / (float)CYCLE_MS) * SPEEDS[speedIdx];
    if (effectPhase >= 1.0f) effectPhase -= (int)effectPhase;

    switch (effect) {
        case WEAVE:      renderWeave();     break;
        case SOLID_PINK: renderSolidPink(); break;
        default:         renderRainbow();   break;
    }
    strip.show();
    delay(8);
}
