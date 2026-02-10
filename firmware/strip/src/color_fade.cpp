/*
 * Slow Color Fade — standalone effect, no streaming required.
 * Smoothly cycles through hues across the full strip.
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12

#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

// HSV-to-RGB (hue 0-65535, sat/val 0-255)
// We use the NeoPixel built-in ColorHSV for this.

uint16_t baseHue = 0;

void setup() {
  strip.begin();
  strip.setBrightness(255);
  strip.show();
}

void loop() {
  // Each LED offset slightly in hue for a rainbow gradient that drifts
  for (int i = 0; i < NUM_PIXELS; i++) {
    uint16_t hue = baseHue + (uint16_t)((uint32_t)i * 65536 / NUM_PIXELS);
    // 40% brightness (102/255), full saturation
    uint32_t color = strip.ColorHSV(hue, 255, 102);
    color = strip.gamma32(color);
    strip.setPixelColor(i, color);
  }
  strip.show();

  // Slow drift: full cycle in ~40 seconds
  baseHue += 100;
  delay(25);
}
