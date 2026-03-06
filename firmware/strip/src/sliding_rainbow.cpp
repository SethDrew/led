/*
 * Sliding rainbow: 30-LED rainbow window, position controlled by pot on A5.
 * Standalone animation — no serial streaming needed.
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12
#define POT_PIN A5

#ifndef NUM_PIXELS
#define NUM_PIXELS 300
#endif

#ifndef COLOR_ORDER
#define COLOR_ORDER NEO_RGB
#endif

#define RAINBOW_LEN 30
#define BRIGHTNESS 40  // ~15%

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, COLOR_ORDER + NEO_KHZ800);

float potSmoothed = 0;

void setup() {
  strip.begin();
  strip.setBrightness(BRIGHTNESS);
  strip.clear();
  strip.show();
  potSmoothed = analogRead(POT_PIN);
}

void loop() {
  potSmoothed += (analogRead(POT_PIN) - potSmoothed) * 0.3;

  int maxTravel = NUM_PIXELS - RAINBOW_LEN;
  int pos = (int)((uint32_t)(potSmoothed + 0.5) * maxTravel / 1023);

  strip.clear();

  for (uint8_t j = 0; j < RAINBOW_LEN; j++) {
    int16_t idx = pos + j;
    if (idx >= 0 && idx < NUM_PIXELS) {
      uint16_t hue = (uint32_t)j * 65536 / RAINBOW_LEN;
      strip.setPixelColor(idx, strip.gamma32(strip.ColorHSV(hue)));
    }
  }

  strip.show();
  delay(25);
}
