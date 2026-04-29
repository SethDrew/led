#ifndef TWINKLE_FOREGROUND_H
#define TWINKLE_FOREGROUND_H

#include "../TreeEffect.h"

// Twinkle effect - random LEDs twinkle like stars
// Good for adding sparkle to any background
class TwinkleForeground : public TreeForegroundEffect {
private:
  uint8_t* brightness;  // Current brightness of each LED
  uint8_t r, g, b;
  uint8_t twinkleChance;  // Probability out of 100
  uint8_t decayRate;      // How fast twinkles fade

public:
  TwinkleForeground(Tree* tree, uint8_t red, uint8_t green, uint8_t blue,
                    uint8_t chance = 3, uint8_t decay = 20)
    : TreeForegroundEffect(tree),
      r(red), g(green), b(blue),
      twinkleChance(chance),
      decayRate(decay) {

    brightness = new uint8_t[tree->getNumLEDs()];
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      brightness[i] = 0;
    }
  }

  ~TwinkleForeground() {
    delete[] brightness;
  }

  void update() override {
    // Decay all LEDs
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      if (brightness[i] > decayRate) {
        brightness[i] -= decayRate;
      } else {
        brightness[i] = 0;
      }
    }

    // Randomly trigger new twinkles
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      if (random(100) < twinkleChance) {
        brightness[i] = 200 + random(55);  // 200-255
      }
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = ADD) override {
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      if (brightness[i] > 5) {
        uint8_t scaled_r = ((uint16_t)r * brightness[i]) / 255;
        uint8_t scaled_g = ((uint16_t)g * brightness[i]) / 255;
        uint8_t scaled_b = ((uint16_t)b * brightness[i]) / 255;

        if (blend == ADD) {
          addColor(buffer, i, scaled_r, scaled_g, scaled_b);
        } else {
          buffer[i][0] = scaled_r;
          buffer[i][1] = scaled_g;
          buffer[i][2] = scaled_b;
        }
      }
    }
  }
};

#endif
