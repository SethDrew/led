/*
 * Rainbow Cycle — full rainbow gradient across strip, scrolling.
 * One complete cycle every CYCLE_MS milliseconds.
 */

#include <Adafruit_NeoPixel.h>

#ifndef STRIP_PIN
#define STRIP_PIN 27
#endif

#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

#define CYCLE_MS 8000

Adafruit_NeoPixel strip(NUM_PIXELS, STRIP_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  strip.begin();
  strip.setBrightness(128);
  strip.show();
}

void loop() {
  uint32_t now = millis();
  // Hue offset scrolls 0-65535 over CYCLE_MS
  uint16_t hue_offset = (uint16_t)((uint32_t)(now % CYCLE_MS) * 65536UL / CYCLE_MS);

  for (int i = 0; i < NUM_PIXELS; i++) {
    uint16_t hue = hue_offset + (uint16_t)((uint32_t)i * 65536UL / NUM_PIXELS);
    strip.setPixelColor(i, strip.gamma32(strip.ColorHSV(hue, 255, 255)));
  }
  strip.show();
  delay(16);
}
