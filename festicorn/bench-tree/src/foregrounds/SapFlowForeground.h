#ifndef SAP_FLOW_FOREGROUND_H
#define SAP_FLOW_FOREGROUND_H

#include "../TreeEffect.h"

// Sap flow effect - particles flow upward through the tree
// Optimized: Renders directly to tree (no buffer)
class SapFlowForeground : public TreeForegroundEffect {
private:
  struct SapParticle {
    float depth;
    float velocityPerSec;
    uint8_t brightness;
    bool active;
  };

  static const uint8_t MAX_PARTICLES = 12;
  SapParticle particles[MAX_PARTICLES];

  uint8_t r, g, b;
  uint32_t frameCount;

  // Time-based spawn control — everything in seconds, independent of FPS.
  // Defaults captured from the frame-based implementation at delay(40)/~21fps.
  float defaultVelocityPerSec;
  float minSpawnIntervalSec;
  float maxSpawnIntervalSec;
  float spawnRatePerSec;
  float timeSinceLastSpawnSec;
  unsigned long lastUpdateMs;

  // Legacy background color state (currently unused; render() uses a fixed bg).
  uint8_t currentR, currentG, currentB;
  int8_t directionR, directionG, directionB;

public:
  SapFlowForeground(Tree* tree, uint8_t red, uint8_t green, uint8_t blue,
                    float velocityPerSec = 7.25f,
                    float minIntervalSec = 1.91f,
                    float maxIntervalSec = 12.73f,
                    float spawnRate = 1.68f)
    : TreeForegroundEffect(tree),
      r(red), g(green), b(blue),
      frameCount(0),
      defaultVelocityPerSec(velocityPerSec),
      minSpawnIntervalSec(minIntervalSec),
      maxSpawnIntervalSec(maxIntervalSec),
      spawnRatePerSec(spawnRate),
      timeSinceLastSpawnSec(0),
      lastUpdateMs(0),
      currentR(200), currentG(255), currentB(100),
      directionR(-1), directionG(-1), directionB(-1) {

    for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
      particles[i].active = false;
    }
  }

  void update() override {
    unsigned long now = millis();
    if (lastUpdateMs == 0) lastUpdateMs = now;
    float dt = (now - lastUpdateMs) / 1000.0f;
    if (dt > 0.1f) dt = 0.1f;  // clamp on long pauses (e.g. WiFi blips)
    lastUpdateMs = now;

    frameCount++;
    timeSinceLastSpawnSec += dt;

    bool shouldSpawn = false;
    if (timeSinceLastSpawnSec >= maxSpawnIntervalSec) {
      shouldSpawn = true;
    } else if (timeSinceLastSpawnSec >= minSpawnIntervalSec) {
      // Poisson-ish: P(spawn this frame) = rate * dt
      long threshold = (long)(spawnRatePerSec * dt * 1000000.0f);
      if (random(1000000) < threshold) {
        shouldSpawn = true;
      }
    }

    if (shouldSpawn) {
      for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
        if (!particles[i].active) {
          particles[i].active = true;
          particles[i].depth = 0;
          particles[i].velocityPerSec = defaultVelocityPerSec;
          particles[i].brightness = 80 + random(70);
          timeSinceLastSpawnSec = 0;
#ifdef DIAG_LOG
          Serial.print("S "); Serial.print(now);
          Serial.print(" "); Serial.println(i);
#endif
          break;
        }
      }
    }

    for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
      if (!particles[i].active) continue;
      particles[i].depth += particles[i].velocityPerSec * dt;
      if (particles[i].depth > MAX_DEPTH + 5) {
        particles[i].active = false;
      }
    }

#ifdef DIAG_LOG
    uint8_t ac = 0;
    float minDepth = 999;
    for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
      if (particles[i].active) {
        ac++;
        if (particles[i].depth < minDepth) minDepth = particles[i].depth;
      }
    }
    Serial.print("F "); Serial.print(now);
    Serial.print(" "); Serial.print(ac);
    Serial.print(" "); Serial.print(timeSinceLastSpawnSec, 3);
    Serial.print(" "); Serial.println(minDepth, 2);
#endif
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
