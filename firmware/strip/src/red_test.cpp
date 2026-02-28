/*
 * Test — last 13 LEDs at 30% white, rest off.
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12

#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  strip.begin();
  strip.clear();

  strip.show();
}

uint16_t hueOffset = 0;

void loop() {
  for (int i = 0; i < 17; i++) {
    uint16_t hue = (i * 65536L / 17 + hueOffset) % 65536;
    strip.setPixelColor(NUM_PIXELS - 17 - 7 + i, strip.ColorHSV(hue, 255, 255));
  }
  strip.show();
  // 1 LED-width = 65536/17 ≈ 3855 hue units over 500ms
  // At 30fps (33ms): 3855 / (500/33) ≈ 254 per frame
  hueOffset += 254;
  delay(33);
}
