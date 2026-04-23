// Minimal SK6812 RGBW diagnostic: first 3 LEDs at ~20% red, rest off.
// Same library as bulb_receiver.cpp (Adafruit_NeoPixel, GRBW, 800kHz) so we
// isolate signal-integrity / wiring from the bloom_rgbw render code.
//
// Wiring: GPIO 23 → strip DIN. Strip +5V/GND on a capable PSU.
// If the first LEDs still glitch with this, it's a signal/level issue
// (direct-drive at 3.3V into SK6812) — not code.

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#ifndef LED_PIN
#define LED_PIN 23
#endif
#ifndef LED_COUNT
#define LED_COUNT 150
#endif

// 20% of 255 ≈ 51 — well above the 0↔1 dither dead zone, firmly in spec.
static const uint8_t RED_LEVEL = 51;
static const uint8_t LIT_COUNT = 3;

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800);

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.printf("rgbw min-test: chase of %u red LEDs across %u total on GPIO %u\n",
                  LIT_COUNT, LED_COUNT, LED_PIN);

    strip.begin();
    strip.setBrightness(255);
    strip.clear();
    strip.show();
}

void loop() {
    // Chase: LIT_COUNT red LEDs step along the strip, one position every 100ms.
    // Full strip every LED_COUNT * 100ms ≈ 15s for 150. Refreshes at 50Hz.
    static uint16_t head = 0;
    static uint32_t lastStepMs = 0;
    uint32_t now = millis();
    if (now - lastStepMs >= 100) {
        lastStepMs = now;
        head = (head + 1) % LED_COUNT;
        Serial.printf("  head=%u\n", head);
    }

    strip.clear();
    for (uint8_t k = 0; k < LIT_COUNT; ++k) {
        uint16_t i = (head + k) % LED_COUNT;
        strip.setPixelColor(i, RED_LEVEL, 0, 0, 0);
    }
    strip.show();
    delay(20);
}
