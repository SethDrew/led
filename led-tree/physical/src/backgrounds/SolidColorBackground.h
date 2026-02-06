#ifndef SOLID_COLOR_BACKGROUND_H
#define SOLID_COLOR_BACKGROUND_H

#include "../TreeEffect.h"

// Solid color background - fills entire tree with one color
class SolidColorBackground : public TreeBackgroundEffect {
private:
  uint8_t r, g, b;
  float intensity;

public:
  SolidColorBackground(Tree* tree, uint8_t red, uint8_t green, uint8_t blue, float inten = 1.0)
    : TreeBackgroundEffect(tree), r(red), g(green), b(blue), intensity(inten) {}

  void update() override {
    // Static background, nothing to update
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    uint8_t scaled_r = (uint16_t)r * intensity;
    uint8_t scaled_g = (uint16_t)g * intensity;
    uint8_t scaled_b = (uint16_t)b * intensity;

    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      buffer[i][0] = scaled_r;
      buffer[i][1] = scaled_g;
      buffer[i][2] = scaled_b;
    }
  }

  // Allow changing color dynamically
  void setColor(uint8_t red, uint8_t green, uint8_t blue) {
    r = red;
    g = green;
    b = blue;
  }
};

#endif
