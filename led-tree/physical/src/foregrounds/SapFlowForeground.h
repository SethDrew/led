#ifndef SAP_FLOW_FOREGROUND_H
#define SAP_FLOW_FOREGROUND_H

#include "../TreeEffect.h"

// Sap flow effect - particles flow upward through the tree
// Optimized: Renders directly to tree (no buffer)
class SapFlowForeground : public TreeForegroundEffect {
private:
  struct SapParticle {
    float depth;
    float velocity;
    uint8_t brightness;
    bool active;
  };

  static const uint8_t MAX_PARTICLES = 6;  // Reduced from 8 to save RAM
  SapParticle particles[MAX_PARTICLES];

  uint8_t r, g, b;
  uint16_t frameCount;
  uint8_t spawnChance;

public:
  SapFlowForeground(Tree* tree, uint8_t red, uint8_t green, uint8_t blue, uint8_t spawn = 5)
    : TreeForegroundEffect(tree),
      r(red), g(green), b(blue),
      frameCount(0),
      spawnChance(spawn) {

    for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
      particles[i].active = false;
    }
  }

  void update() override {
    frameCount++;

    // Try to spawn new particle
    if (random(100) < spawnChance) {
      for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
        if (!particles[i].active) {
          particles[i].active = true;
          particles[i].depth = 0;
          particles[i].velocity = 0.3 + random(20) / 100.0;
          particles[i].brightness = 150 + random(105);
          break;
        }
      }
    }

    // Update particles
    for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
      if (!particles[i].active) continue;

      particles[i].depth += particles[i].velocity;

      if (particles[i].depth > MAX_DEPTH + 5) {
        particles[i].active = false;
      }
    }
  }

  void render() override {
    // Clear tree
    tree->clear();

    // Render each particle
    for (uint8_t p = 0; p < MAX_PARTICLES; p++) {
      if (!particles[p].active) continue;

      float particleDepth = particles[p].depth;
      uint8_t brightness = particles[p].brightness;

      // Render particle with soft edges
      for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
        uint8_t ledDepth = tree->getDepth(i);
        float distance = abs(ledDepth - particleDepth);

        if (distance < 3.0) {
          float falloff = 1.0 - (distance / 3.0);
          falloff = falloff * falloff;

          uint8_t scaled_r = ((uint16_t)r * brightness * falloff) / 255;
          uint8_t scaled_g = ((uint16_t)g * brightness * falloff) / 255;
          uint8_t scaled_b = ((uint16_t)b * brightness * falloff) / 255;

          tree->setNodeColor(i, scaled_r, scaled_g, scaled_b);
        }
      }
    }

    tree->show();
  }
};

#endif
