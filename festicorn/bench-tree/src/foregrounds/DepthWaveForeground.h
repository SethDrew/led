#ifndef DEPTH_WAVE_FOREGROUND_H
#define DEPTH_WAVE_FOREGROUND_H

#include "../TreeEffect.h"

// Depth wave - wave flows up and down the tree based on depth
// Optimized: Renders directly to tree (no buffer)
class DepthWaveForeground : public TreeForegroundEffect {
private:
  int offset;
  int direction;
  float waveWidth;
  uint8_t r, g, b;
  uint16_t frameCount;
  uint16_t pauseAtReversal;

public:
  DepthWaveForeground(Tree* tree, uint8_t red, uint8_t green, uint8_t blue, float width = 5.0)
    : TreeForegroundEffect(tree),
      offset(0),
      direction(1),
      waveWidth(width),
      r(red), g(green), b(blue),
      frameCount(0),
      pauseAtReversal(0) {}

  void update() override {
    frameCount++;

    // Handle pause at reversal
    if (pauseAtReversal > 0) {
      pauseAtReversal--;
      return;
    }

    // Advance wave
    offset += direction;

    // Reverse direction at ends
    if (offset > MAX_DEPTH + 10 || offset < -10) {
      direction *= -1;
      pauseAtReversal = 25;  // Pause for ~1 second at 25 FPS
    }
  }

  void render() override {
    // Clear tree first
    tree->clear();

    // Render wave
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      int effectiveDepth = tree->getEffectiveDepth(i);
      float distance = abs(effectiveDepth - offset);

      if (distance <= waveWidth) {
        tree->setNodeColor(i, r, g, b);
      }
    }

    tree->show();
  }

  // Allow changing wave color
  void setColor(uint8_t red, uint8_t green, uint8_t blue) {
    r = red;
    g = green;
    b = blue;
  }
};

#endif
