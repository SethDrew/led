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

  static const uint8_t MAX_PARTICLES = 12;  // More particles for denser flow
  SapParticle particles[MAX_PARTICLES];

  uint8_t r, g, b;
  uint16_t frameCount;
  uint8_t spawnChance;
  uint8_t framesSinceLastSpawn;
  uint8_t minFramesBetweenSpawn;
  uint8_t maxFramesBetweenSpawn;

  // Background color state for smooth 1-value-per-step transitions
  uint8_t currentR, currentG, currentB;
  int8_t directionR, directionG, directionB;

public:
  SapFlowForeground(Tree* tree, uint8_t red, uint8_t green, uint8_t blue, uint8_t spawn = 5)
    : TreeForegroundEffect(tree),
      r(red), g(green), b(blue),
      frameCount(0),
      spawnChance(spawn),
      framesSinceLastSpawn(0),
      minFramesBetweenSpawn(20),   // Minimum 20 frames (800ms at 25 FPS)
      maxFramesBetweenSpawn(120),  // Maximum 120 frames (4.8s at 25 FPS)
      currentR(200), currentG(255), currentB(100),  // Start at bright lime green
      directionR(-1), directionG(-1), directionB(-1) { // Start moving toward dark green

    for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
      particles[i].active = false;
    }
  }

  void update() override {
    frameCount++;
    framesSinceLastSpawn++;

    // Try to spawn new particle with timing constraints
    bool shouldSpawn = false;

    if (framesSinceLastSpawn >= maxFramesBetweenSpawn) {
      // Force spawn if max time exceeded
      shouldSpawn = true;
    } else if (framesSinceLastSpawn >= minFramesBetweenSpawn && random(100) < spawnChance) {
      // Random spawn if minimum time met
      shouldSpawn = true;
    }

    if (shouldSpawn) {
      for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
        if (!particles[i].active) {
          particles[i].active = true;
          particles[i].depth = 0;
          particles[i].velocity = 0.4;  // Fixed velocity prevents particles from crossing
          particles[i].brightness = 150 + random(105);
          framesSinceLastSpawn = 0;
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
    // Solid deep green background at 1% brightness
    uint8_t baseBrightness = 3;  // ~1% of 255

    // Deep forest green (darkest)
    uint8_t bgR = (baseBrightness * 10) / 255;
    uint8_t bgG = (baseBrightness * 100) / 255;
    uint8_t bgB = (baseBrightness * 10) / 255;

    // Apply background to all LEDs
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      tree->setNodeColor(i, bgR, bgG, bgB);
    }

    // Render each particle additively on top of background
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

          uint8_t particle_r = ((uint16_t)r * brightness * falloff) / 255;
          uint8_t particle_g = ((uint16_t)g * brightness * falloff) / 255;
          uint8_t particle_b = ((uint16_t)b * brightness * falloff) / 255;

          // Add particle color to background (additive blending)
          uint8_t final_r = min(255, bgR + particle_r);
          uint8_t final_g = min(255, bgG + particle_g);
          uint8_t final_b = min(255, bgB + particle_b);

          tree->setNodeColor(i, final_r, final_g, final_b);
        }
      }
    }

    tree->show();
  }
};

#endif
