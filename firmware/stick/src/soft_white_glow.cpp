/*
 * SOFT WHITE GLOW — Dithering A/B test
 *
 * First 25 LEDs: manual temporal dithering
 * Last 25 LEDs: no dithering (truncated uint8)
 * Both use W channel via FastLED RGBW, same target brightness.
 * Pin 10, 50 LEDs, ESP32-C3, SK6812 RGBW.
 */

#include <FastLED.h>

#define LED_PIN    10
#define LED_COUNT  50

CRGB leds[LED_COUNT];

static const uint8_t MAX_BRIGHT = 51;  // 20% of 255

uint32_t frameCount = 0;

void setup() {
    Serial.begin(115200);

    FastLED.addLeds<SK6812, LED_PIN, GRB>(leds, LED_COUNT)
        .setRgbw(RgbwDefault());
    // Disable FastLED's built-in dithering — we do it manually
    FastLED.setDither(0);
    FastLED.setBrightness(255);

    // Brief red flash
    fill_solid(leds, LED_COUNT, CRGB(40, 0, 0));
    FastLED.show();
    delay(200);

    Serial.println("Dither A/B: first 25 dithered, last 25 raw");
}

void loop() {
    float t = millis() / 1000.0f;
    // 8 sec cycle, 0..1 range
    float linear = (sin(t * 0.785f) + 1.0f) * 0.5f;
    // Floor at 5 to avoid 0↔1 zone, ceiling at 51
    float target = 5.0f + linear * (MAX_BRIGHT - 5.0f);

    // Dithered value: use frame count to decide round up or down
    uint8_t lo = (uint8_t)target;
    float frac = target - lo;
    // 16-phase dither for finer sub-steps
    uint8_t dithered = (frac > (float)(frameCount & 15) / 16.0f) ? lo + 1 : lo;

    // Raw truncated value
    uint8_t raw = lo;

    // First 25: dithered white
    for (uint16_t i = 0; i < 25; i++)
        leds[i] = CRGB(dithered, dithered, dithered);

    // Last 25: raw white (no dithering)
    for (uint16_t i = 25; i < LED_COUNT; i++)
        leds[i] = CRGB(raw, raw, raw);

    FastLED.show();
    frameCount++;
}
