/*
 * SOFT WHITE GLOW — NeoPixel baseline (no dithering)
 * Linear sine on RGB warm white, no gamma, same speed as FastLED version.
 * For A/B comparison.
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN    10
#define LED_COUNT  50

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800);

static const float BREATHE_SPEED = 0.004f;
static const uint8_t MAX_BRIGHT = 100;

float phase = 0.0f;

void setup() {
    Serial.begin(115200);
    strip.begin();
    strip.setBrightness(255);
    strip.show();

    // Brief red flash
    for (uint16_t i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, strip.Color(40, 0, 0, 0));
    strip.show();
    delay(200);
    strip.clear();
    strip.show();

    Serial.println("Soft White Glow — NeoPixel baseline");
}

void loop() {
    float linear = (sin(phase) + 1.0f) * 0.5f;
    uint8_t b = (uint8_t)(linear * MAX_BRIGHT);

    // Same warm white via RGB, scaled by brightness
    uint8_t r = (uint8_t)((uint16_t)b * 255 / MAX_BRIGHT);
    uint8_t g = (uint8_t)((uint16_t)b * 200 / MAX_BRIGHT);
    uint8_t bl = (uint8_t)((uint16_t)b * 120 / MAX_BRIGHT);

    for (uint16_t i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, strip.Color(r, g, bl, 0));

    strip.show();

    phase += BREATHE_SPEED;
    if (phase > 6.2832f) phase -= 6.2832f;
}
