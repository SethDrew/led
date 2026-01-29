#ifndef SOLID_COLOR_BACKGROUND_H
#define SOLID_COLOR_BACKGROUND_H

#include "Effect.h"

// Simple solid color background
class SolidColorBackground : public BackgroundEffect {
private:
  int numPixels;
  uint8_t r, g, b;
  float brightness;  // 0.0 to 1.0

public:
  SolidColorBackground(int pixels, uint8_t red, uint8_t green, uint8_t blue, float bright = 1.0)
    : numPixels(pixels), r(red), g(green), b(blue), brightness(bright) {}

  // Optionally set from HSV
  void setColorHSV(uint8_t hue, uint8_t sat = 255, uint8_t val = 255) {
    hsvToRgb(hue, sat, val, r, g, b);
  }

  void setBrightness(float bright) {
    brightness = constrain(bright, 0.0, 1.0);
  }

  void update() override {
    // Static background - nothing to update
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    int final_r = r * brightness;
    int final_g = g * brightness;
    int final_b = b * brightness;

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
