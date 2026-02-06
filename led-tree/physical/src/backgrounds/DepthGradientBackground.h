#ifndef DEPTH_GRADIENT_BACKGROUND_H
#define DEPTH_GRADIENT_BACKGROUND_H

#include "../TreeEffect.h"

// Gradient background based on tree depth
// Goes from baseColor at depth 0 to tipColor at max depth
class DepthGradientBackground : public TreeBackgroundEffect {
private:
  uint8_t baseR, baseG, baseB;  // Color at base (depth 0)
  uint8_t tipR, tipG, tipB;     // Color at tips (max depth)
  float intensity;

public:
  DepthGradientBackground(Tree* tree,
                          uint8_t br, uint8_t bg, uint8_t bb,  // Base color
                          uint8_t tr, uint8_t tg, uint8_t tb,  // Tip color
                          float inten = 1.0)
    : TreeBackgroundEffect(tree),
      baseR(br), baseG(bg), baseB(bb),
      tipR(tr), tipG(tg), tipB(tb),
      intensity(inten) {}

  void update() override {
    // Static gradient, nothing to update
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      uint8_t depth = tree->getDepth(i);

      // Calculate interpolation factor (0.0 at base, 1.0 at max depth)
      float t = (float)depth / MAX_DEPTH;

      // Linear interpolation between base and tip colors
      uint8_t r = baseR + (tipR - baseR) * t;
      uint8_t g = baseG + (tipG - baseG) * t;
      uint8_t b = baseB + (tipB - baseB) * t;

      // Apply intensity
      buffer[i][0] = (uint16_t)r * intensity;
      buffer[i][1] = (uint16_t)g * intensity;
      buffer[i][2] = (uint16_t)b * intensity;
    }
  }
};

#endif
