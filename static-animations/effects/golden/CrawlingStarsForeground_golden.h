#ifndef CRAWLING_STARS_FOREGROUND_H
#define CRAWLING_STARS_FOREGROUND_H

#include "Effect.h"

// Crawling stars - glowing orbs that move along the strip
// OPTIMIZED: Uses uint8_t instead of float (75% memory savings)
class CrawlingStarsForeground : public ForegroundEffect {
private:
  Orb orbs[10];
  uint8_t* ledValues;  // Changed from float* to uint8_t*
  int numPixels;
  int maxOrbs;
  float orbSize;
  float speed;

  // Star color (warm white)
  const uint8_t STAR_R = 255;
  const uint8_t STAR_G = 240;
  const uint8_t STAR_B = 200;

public:
  CrawlingStarsForeground(int pixels, int max_orbs, float orb_size, float orb_speed)
    : numPixels(pixels), maxOrbs(max_orbs), orbSize(orb_size), speed(orb_speed) {

    ledValues = new uint8_t[pixels];

    for (int i = 0; i < 10; i++) {
      orbs[i].active = false;
    }
    for (int i = 0; i < pixels; i++) {
      ledValues[i] = 0;
    }
  }

  ~CrawlingStarsForeground() {
    delete[] ledValues;
  }

  void update() override {
    // Decay all LEDs (multiply by decay factor, similar to float version)
    // decayFactor = 1.0 - (1.0 / orbSize), converted to 8-bit integer math
    // We use 256 as "1.0" for fixed-point math
    uint16_t decayFactor = 256 - (256 / (uint16_t)(orbSize + 0.5));

    for (int i = 0; i < numPixels; i++) {
      ledValues[i] = ((uint16_t)ledValues[i] * decayFactor) >> 8;  // >> 8 divides by 256
      if (ledValues[i] < 3) ledValues[i] = 0;  // Threshold (was 0.01 * 255)
    }

    // Count active orbs
    int activeCount = 0;
    for (int i = 0; i < 10; i++) {
      if (orbs[i].active) activeCount++;
    }

    // Spawn new orb
    if (activeCount < maxOrbs && random(100) < 3) {
      for (int i = 0; i < 10; i++) {
        if (!orbs[i].active) {
          orbs[i].active = true;
          orbs[i].position = random(0, numPixels);
          orbs[i].velocity = (random(0, 2) == 0 ? 1 : -1) * speed * (0.5 + random(100) / 200.0);
          orbs[i].age = 0;
          orbs[i].lifetime = 40 + random(100);
          break;
        }
      }
    }

    // Update orbs
    for (int i = 0; i < 10; i++) {
      if (!orbs[i].active) continue;

      orbs[i].age += 1.0;
      if (orbs[i].age >= orbs[i].lifetime) {
        orbs[i].active = false;
        continue;
      }

      orbs[i].position += orbs[i].velocity;
      if (orbs[i].position < 0) orbs[i].position += numPixels;
      if (orbs[i].position >= numPixels) orbs[i].position -= numPixels;

      // Calculate lifecycle brightness (smoothstep fade in/out)
      float lifecycle = orbs[i].age / orbs[i].lifetime;
      float lifeBrightness = 1.0;
      if (lifecycle < 0.4) {
        float t = lifecycle / 0.4;
        lifeBrightness = t * t * (3.0 - 2.0 * t);
      } else if (lifecycle > 0.6) {
        float t = (1.0 - lifecycle) / 0.4;
        lifeBrightness = t * t * (3.0 - 2.0 * t);
      }

      // Add orb brightness to LED (0.6 scale factor = 153 in 8-bit)
      int orbPixel = (int)orbs[i].position;
      if (orbPixel >= 0 && orbPixel < numPixels) {
        uint16_t addition = (uint16_t)(lifeBrightness * 153);  // 0.6 * 255
        uint16_t newValue = (uint16_t)ledValues[orbPixel] + addition;
        ledValues[orbPixel] = (newValue > 255) ? 255 : (uint8_t)newValue;
      }
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = ADD) override {
    for (int i = 0; i < numPixels; i++) {
      if (ledValues[i] > 2) {  // Threshold (was 0.01 * 255)
        // Scale color by brightness using 16-bit math to avoid overflow
        uint16_t r = ((uint16_t)STAR_R * ledValues[i]) / 255;
        uint16_t g = ((uint16_t)STAR_G * ledValues[i]) / 255;
        uint16_t b = ((uint16_t)STAR_B * ledValues[i]) / 255;

        if (blend == ADD) {
          // Additive blending with saturation
          uint16_t new_r = (uint16_t)buffer[i][0] + r;
          uint16_t new_g = (uint16_t)buffer[i][1] + g;
          uint16_t new_b = (uint16_t)buffer[i][2] + b;

          buffer[i][0] = (new_r > 255) ? 255 : (uint8_t)new_r;
          buffer[i][1] = (new_g > 255) ? 255 : (uint8_t)new_g;
          buffer[i][2] = (new_b > 255) ? 255 : (uint8_t)new_b;
        } else if (blend == REPLACE) {
          buffer[i][0] = r;
          buffer[i][1] = g;
          buffer[i][2] = b;
        }
      }
    }
  }
};

#endif
