#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#define STRIP_A_LEDS 12
#define STRIP_B_LEDS 11
#define TOTAL_LEDS (STRIP_A_LEDS + STRIP_B_LEDS)
#define BRIGHTNESS 60

Adafruit_NeoPixel stripA(STRIP_A_LEDS, PIN_A, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripB(STRIP_B_LEDS, PIN_B, NEO_GRB + NEO_KHZ800);

uint16_t hueOffset = 0;

void setup() {
    stripA.begin();
    stripB.begin();
    stripA.setBrightness(BRIGHTNESS);
    stripB.setBrightness(BRIGHTNESS);
}

void loop() {
    // Rainbow spread across both strips as one continuous run
    for (int i = 0; i < STRIP_A_LEDS; i++) {
        uint16_t hue = hueOffset + (i * 65536L / TOTAL_LEDS);
        stripA.setPixelColor(i, Adafruit_NeoPixel::ColorHSV(hue));
    }
    for (int i = 0; i < STRIP_B_LEDS; i++) {
        uint16_t hue = hueOffset + ((i + STRIP_A_LEDS) * 65536L / TOTAL_LEDS);
        stripB.setPixelColor(i, Adafruit_NeoPixel::ColorHSV(hue));
    }

    stripA.show();
    stripB.show();

    hueOffset += 256;
    delay(20);
}
