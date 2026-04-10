/*
 * BLUE vs WHITE — side-by-side strobe comparison on SK6812 RGBW
 *
 * Serial commands toggle between modes:
 *   'l' — lite mode (minimal math, maximum fps)
 *   'h' — heavy mode (fire-equivalent per-pixel math)
 *
 * Starts in lite mode. Watch for strobe appearing when switching
 * to heavy mode — if it does, frame rate is the cause.
 *
 * LEDs 0–24:  Blue channel only (R=0, G=0, B=fade, W=0)
 * LEDs 25–49: White channel only (R=0, G=0, B=0, W=fade)
 *
 * GPIO 10 → SK6812 RGBW (50 LEDs)
 */

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include <delta_sigma.h>

#define LED_PIN    10
#define LED_COUNT  50
#define SPLIT      25

#define GAMMA      2.4f
#define PEAK_BRIGHT 0.20f
#define PERIOD_S    8.0f

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800);

static uint16_t dsR[LED_COUNT], dsG[LED_COUNT], dsB[LED_COUNT], dsW[LED_COUNT];

// FPS tracking
static uint32_t frameCount = 0;
static uint32_t lastFpsMs = 0;

// Mode: false = lite (minimal math), true = heavy (fire-equivalent math)
static bool heavyMode = false;

// Dummy accumulator to prevent optimizer from removing heavy math
static volatile float sinkVal = 0.0f;

void setup() {
    Serial.begin(460800);
    strip.begin();
    strip.setBrightness(255);

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
        dsR[i] = seed;
        dsG[i] = (seed + 64) & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;
    }

    lastFpsMs = millis();
    Serial.println("Blue vs White strobe test");
    Serial.println("  LEDs 0-24:  BLUE channel");
    Serial.println("  LEDs 25-49: WHITE channel");
    Serial.println("  Send 'l' for lite mode, 'h' for heavy mode");
    Serial.println("  Starting in LITE mode");
}

void loop() {
    // Serial commands
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'l' || c == 'L') {
            heavyMode = false;
            Serial.println(">> LITE mode (minimal math)");
        } else if (c == 'h' || c == 'H') {
            heavyMode = true;
            Serial.println(">> HEAVY mode (fire-equivalent math)");
        }
    }

    float t = millis() / 1000.0f;

    // Slow sine: 0 → 1 → 0
    float phase = sinf(t * 2.0f * M_PI / PERIOD_S) * 0.5f + 0.5f;
    float bright = phase * PEAK_BRIGHT;

    // Gamma correction
    float lin = powf(bright, GAMMA);

    // Convert to 8.8 fixed-point
    uint16_t target16 = (uint16_t)(lin * 255.0f * 256.0f);

    // Noise gate
    if (target16 < 64) target16 = 0;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        // In heavy mode: simulate fire-equivalent per-pixel math
        // (two sinf, one powf, color blending, RGBW routing)
        if (heavyMode) {
            float fi = (float)i;
            float noise = sinf(fi * 7.3f + t * 2.5f) *
                           sinf(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;
            float fakeBright = bright * (1.0f + 0.15f * (noise - 0.5f));
            float fakeLin = powf(fakeBright, GAMMA);
            // Color blending (simulate amber → red interpolation)
            float colR = 255.0f * (1.0f - noise * 0.3f) + 200.0f * noise * 0.3f;
            float colG = 140.0f * (1.0f - noise * 0.3f) +  20.0f * noise * 0.3f;
            float colB =  30.0f * (1.0f - noise * 0.3f);
            // RGBW routing (simulate the full path)
            float oR = colR * fakeLin;
            float oG = colG * fakeLin;
            float oB = colB * fakeLin;
            float maxCh_f = fmaxf(oR, fmaxf(oG, oB));
            float bFrac = maxCh_f / 255.0f;
            float rgbBlend = fminf(1.0f, fmaxf(0.0f, (bFrac - 0.10f) / 0.15f));
            float avgRGB = (oR + oG + oB) / 3.0f;
            // Consume result so optimizer can't remove it
            sinkVal = avgRGB * rgbBlend;
        }

        // Actual LED output is always the same simple test pattern
        uint8_t r, g, b, w;
        if (i < SPLIT) {
            r = 0; g = 0;
            b = deltaSigma(dsB[i], target16);
            w = 0;
            deltaSigma(dsR[i], 0);
            deltaSigma(dsG[i], 0);
            deltaSigma(dsW[i], 0);
        } else {
            r = 0; g = 0; b = 0;
            w = deltaSigma(dsW[i], target16);
            deltaSigma(dsR[i], 0);
            deltaSigma(dsG[i], 0);
            deltaSigma(dsB[i], 0);
        }

        strip.setPixelColor(i, r, g, b, w);
    }

    strip.show();
    delay(1);

    // FPS reporting every 2 seconds
    frameCount++;
    uint32_t now = millis();
    if (now - lastFpsMs >= 2000) {
        float fps = frameCount * 1000.0f / (now - lastFpsMs);
        Serial.printf("[%s] %.0f fps\n", heavyMode ? "HEAVY" : "LITE", fps);
        frameCount = 0;
        lastFpsMs = now;
    }
}
