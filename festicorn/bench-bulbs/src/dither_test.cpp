/*
 * DITHER TEST — dithered vs stepped for values 1–10/255
 *
 * Drives both GPIO 4 and GPIO 18 (50× WS2812B RGB each).
 *   First 25: delta-sigma dithered
 *   Last 25:  stepped (rounded to nearest integer)
 *
 * Fade 1/255 → 10/255 over 10s, then back, repeating. White.
 * Frame rate capped at 200fps.
 */

#include <Arduino.h>
#include <NeoPixelBus.h>
#include <delta_sigma.h>

#define LED_COUNT  50
#define SPLIT      25
#define TARGET_FPS 200
#define FRAME_US   (1000000 / TARGET_FPS)

static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> stripA(LED_COUNT, 4);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> stripB(LED_COUNT, 18);

static uint16_t ditherAccum[SPLIT][3] = {};
static float phase = 0.0f;

void setup() {
    Serial.begin(115200);
    delay(500);
    stripA.Begin();
    stripB.Begin();
    stripA.Show();
    stripB.Show();
    memset(ditherAccum, 0, sizeof(ditherAccum));

    Serial.println("\n=== DITHER vs STEP TEST (200fps) ===");
    Serial.println("First 25: dithered");
    Serial.println("Last 25:  stepped (rounded)");
    Serial.println("Fade 1→10/255 over 10s, then back\n");
}

void loop() {
    uint32_t frameStart = micros();
    float dt = 1.0f / TARGET_FPS;

    // Triangle wave: 0→1 over 10s, 1→0 over 10s
    phase += dt / 5.0f;
    if (phase >= 2.0f) phase -= 2.0f;
    float t = (phase < 1.0f) ? phase : (2.0f - phase);

    // Map t 0→1 to brightness 1.0→10.0 (/255)
    float brightness = 1.0f + t * 9.0f;
    uint16_t target16 = (uint16_t)(brightness * 256.0f);
    uint8_t stepped = (uint8_t)(brightness + 0.5f);

    // First 25: dithered
    for (int i = 0; i < SPLIT; i++) {
        uint8_t r = deltaSigma(ditherAccum[i][0], target16);
        uint8_t g = deltaSigma(ditherAccum[i][1], target16);
        uint8_t b = deltaSigma(ditherAccum[i][2], target16);
        stripA.SetPixelColor(i, RgbColor(r, g, b));
        stripB.SetPixelColor(i, RgbColor(r, g, b));
    }

    // Last 25: stepped (no dither)
    for (int i = 0; i < SPLIT; i++) {
        stripA.SetPixelColor(SPLIT + i, RgbColor(stepped, stepped, stepped));
        stripB.SetPixelColor(SPLIT + i, RgbColor(stepped, stepped, stepped));
    }

    stripA.Show();
    stripB.Show();

    static uint32_t frames = 0;
    static uint32_t lastStatMs = 0;
    frames++;
    uint32_t now = millis();
    if (now - lastStatMs >= 1000) {
        float fps = frames * 1000.0f / (now - lastStatMs);
        Serial.printf("[FPS %.0f | %.1f/255 | t16=%u | stepped=%u]\n",
                      fps, brightness, target16, stepped);
        frames = 0;
        lastStatMs = now;
    }

    uint32_t elapsed = micros() - frameStart;
    if (elapsed < FRAME_US) {
        delayMicroseconds(FRAME_US - elapsed);
    }
}
