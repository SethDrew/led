#ifndef SPARKS_FOREGROUND_H
#define SPARKS_FOREGROUND_H

#include "Effect.h"

// Random spark explosions - bursts of yellow-white particles
class SparksForeground : public ForegroundEffect {
private:
  Orb sparks[35];
  int numPixels;
  int explosionTimer;
  int explosionInterval;

public:
  SparksForeground(int pixels) : numPixels(pixels), explosionTimer(0), explosionInterval(100) {
    for (int i = 0; i < 35; i++) {
      sparks[i].active = false;
    }
  }

  void spawnExplosion(float position) {
    // Spawn 35 sparks at the explosion point
    for (int i = 0; i < 35; i++) {
      sparks[i].active = true;
      sparks[i].position = position;
      sparks[i].velocity = (random(20, 60) / 100.0) * (random(0, 2) == 0 ? 1 : -1);
      sparks[i].age = 0;
      sparks[i].lifetime = 25 + random(30);  // 25-55 frames
    }
  }

  void update() override {
    // Trigger random explosions
    explosionTimer++;
    if (explosionTimer >= explosionInterval) {
      float randomPos = random(5, numPixels - 5);
      spawnExplosion(randomPos);
      explosionTimer = 0;
      explosionInterval = 80 + random(60);  // 80-140 frames between explosions
    }

    // Update all sparks
    for (int i = 0; i < 35; i++) {
      if (!sparks[i].active) continue;

      sparks[i].age += 1.0;
      if (sparks[i].age >= sparks[i].lifetime) {
        sparks[i].active = false;
        continue;
      }

      sparks[i].position += sparks[i].velocity;

      // Deactivate if off strip
      if (sparks[i].position < 0 || sparks[i].position >= numPixels) {
        sparks[i].active = false;
        continue;
      }
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = ADD) override {
    for (int i = 0; i < 35; i++) {
      if (!sparks[i].active) continue;

      // Calculate brightness with flickering
      float lifecycle = sparks[i].age / sparks[i].lifetime;
      float brightness = 1.0 - lifecycle;
      brightness = brightness * brightness;  // Quadratic falloff
      float flicker = 0.7 + (random(0, 60) / 100.0);
      brightness = brightness * flicker;
      brightness = min(1.0f, max(0.0f, brightness));

      // Color variants - yellow-white emphasis
      int colorChoice = i % 7;
      int r, g, b;
      switch(colorChoice) {
        case 0: r = 255 * brightness; g = 255 * brightness; b = 150 * brightness; break;  // Bright yellow-white
        case 1: r = 255 * brightness; g = 255 * brightness; b = 100 * brightness; break;  // Pale yellow
        case 2: r = 255 * brightness; g = 240 * brightness; b = 200 * brightness; break;  // Warm white
        case 3: r = 255 * brightness; g = 255 * brightness; b = 180 * brightness; break;  // Cream
        case 4: r = 255 * brightness; g = 255 * brightness; b = 0; break;                 // Pure yellow
        case 5: r = 255 * brightness; g = 245 * brightness; b = 150 * brightness; break;  // Golden white
        default: r = 255 * brightness; g = 255 * brightness; b = 120 * brightness; break;  // Soft yellow
      }

      int pixelPos = (int)sparks[i].position;
      if (pixelPos >= 0 && pixelPos < numPixels) {
        if (blend == ADD) {
          buffer[pixelPos][0] = min(255, (int)buffer[pixelPos][0] + r);
          buffer[pixelPos][1] = min(255, (int)buffer[pixelPos][1] + g);
          buffer[pixelPos][2] = min(255, (int)buffer[pixelPos][2] + b);
        } else if (blend == REPLACE) {
          buffer[pixelPos][0] = r;
          buffer[pixelPos][1] = g;
          buffer[pixelPos][2] = b;
        }
      }
    }
  }
};

#endif
