#ifndef CRAWLING_STARS_FOREGROUND_H
#define CRAWLING_STARS_FOREGROUND_H

#include "Effect.h"

// Crawling stars - glowing orbs that move along the strip
class CrawlingStarsForeground : public ForegroundEffect {
private:
  Orb orbs[10];
  float* ledValues;
  int numPixels;
  int maxOrbs;
  float orbSize;
  float speed;

  // Star color (warm white)
  const int STAR_R = 255;
  const int STAR_G = 240;
  const int STAR_B = 200;

public:
  CrawlingStarsForeground(int pixels, int max_orbs, float orb_size, float orb_speed)
    : numPixels(pixels), maxOrbs(max_orbs), orbSize(orb_size), speed(orb_speed) {

    ledValues = new float[pixels];

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
    // Decay all LEDs
    float decayFactor = 1.0 - (1.0 / orbSize);
    for (int i = 0; i < numPixels; i++) {
      ledValues[i] *= decayFactor;
      if (ledValues[i] < 0.01) ledValues[i] = 0.0;
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

      // Calculate lifecycle brightness
      float lifecycle = orbs[i].age / orbs[i].lifetime;
      float lifeBrightness = 1.0;
      if (lifecycle < 0.4) {
        float t = lifecycle / 0.4;
        lifeBrightness = t * t * (3.0 - 2.0 * t);
      } else if (lifecycle > 0.6) {
        float t = (1.0 - lifecycle) / 0.4;
        lifeBrightness = t * t * (3.0 - 2.0 * t);
      }

      int orbPixel = (int)orbs[i].position;
      if (orbPixel >= 0 && orbPixel < numPixels) {
        ledValues[orbPixel] = min(1.0f, ledValues[orbPixel] + lifeBrightness * 0.6f);
      }
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = ADD) override {
    for (int i = 0; i < numPixels; i++) {
      if (ledValues[i] > 0.01) {
        int r = STAR_R * ledValues[i];
        int g = STAR_G * ledValues[i];
        int b = STAR_B * ledValues[i];

        if (blend == ADD) {
          // Additive blending
          buffer[i][0] = min(255, (int)buffer[i][0] + r);
          buffer[i][1] = min(255, (int)buffer[i][1] + g);
          buffer[i][2] = min(255, (int)buffer[i][2] + b);
        } else if (blend == REPLACE) {
          // Replace mode
          buffer[i][0] = r;
          buffer[i][1] = g;
          buffer[i][2] = b;
        }
      }
    }
  }
};

#endif
