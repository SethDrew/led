#ifndef PULSING_COLOR_BACKGROUND_H
#define PULSING_COLOR_BACKGROUND_H

#include "Effect.h"

// Pulsing solid color background
class PulsingColorBackground : public BackgroundEffect {
private:
  int numPixels;
  uint8_t r, g, b;
  float minBrightness;
  float maxBrightness;
  float pulseSpeed;
  float currentBrightness;
  unsigned long frameCount;

public:
  PulsingColorBackground(int pixels, uint8_t red, uint8_t green, uint8_t blue,
                         float min_bright = 0.2, float max_bright = 0.8, float speed = 0.02)
    : numPixels(pixels), r(red), g(green), b(blue),
      minBrightness(min_bright), maxBrightness(max_bright),
      pulseSpeed(speed), currentBrightness(0), frameCount(0) {}

  // Optionally set from HSV
  void setColorHSV(uint8_t hue, uint8_t sat = 255, uint8_t val = 255) {
    hsvToRgb(hue, sat, val, r, g, b);
  }

  void update() override {
    frameCount++;

    // Sine wave pulsing
    float t = (float)frameCount * pulseSpeed;
    float wave = 0.5 + 0.5 * sin(t);  // 0.0 to 1.0
    currentBrightness = minBrightness + (maxBrightness - minBrightness) * wave;
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    int final_r = r * currentBrightness;
    int final_g = g * currentBrightness;
    int final_b = b * currentBrightness;

    for (int i = 0; i < numPixels; i++) {
      if (blend == REPLACE) {
        buffer[i][0] = final_r;
        buffer[i][1] = final_g;
        buffer[i][2] = final_b;
      } else if (blend == ADD) {
        buffer[i][0] = min(255, (int)buffer[i][0] + final_r);
        buffer[i][1] = min(255, (int)buffer[i][1] + final_g);
        buffer[i][2] = min(255, (int)buffer[i][2] + final_b);
      }
    }
  }
};

#endif
