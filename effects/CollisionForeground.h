#ifndef COLLISION_FOREGROUND_H
#define COLLISION_FOREGROUND_H

#include "Effect.h"

// Collision sequence - two crawlers approach, collide, and explode into sparks
class CollisionForeground : public ForegroundEffect {
private:
  enum State { APPROACHING, COLLIDING, SPARKLING, RESET };
  State state;

  int numPixels;
  const float waveWidth = 6.0;
  const int CRAWLER_R = 255;
  const int CRAWLER_G = 240;
  const int CRAWLER_B = 200;

  float crawler1Pos;
  float crawler2Pos;
  float crawler1Speed;
  float crawler2Speed;
  int crawler1SpawnDelay;
  int crawler2SpawnDelay;
  int framesSinceReset;
  int collisionPoint;
  int resetDelay;

  Orb sparks[35];

public:
  CollisionForeground(int pixels) : numPixels(pixels) {
    state = APPROACHING;
    crawler1Pos = -8;
    crawler2Pos = pixels + 8;

    // Same speed for both to ensure they meet at center
    crawler1Speed = 0.8;
    crawler2Speed = 0.8;

    crawler1SpawnDelay = 0;
    crawler2SpawnDelay = 0;
    framesSinceReset = 0;
    collisionPoint = 0;
    resetDelay = 0;

    for (int i = 0; i < 35; i++) {
      sparks[i].active = false;
    }
  }

  void update() override {
    switch(state) {
      case APPROACHING:
        {
          framesSinceReset++;

          // Move crawlers
          if (framesSinceReset >= crawler1SpawnDelay) {
            crawler1Pos += crawler1Speed;
          }
          if (framesSinceReset >= crawler2SpawnDelay) {
            crawler2Pos -= crawler2Speed;
          }

          // Check for collision
          if (crawler1Pos + waveWidth >= crawler2Pos - waveWidth) {
            state = COLLIDING;
            collisionPoint = (int)((crawler1Pos + crawler2Pos) / 2);
          }
        }
        break;

      case COLLIDING:
        {
          // Continue moving
          crawler1Pos += crawler1Speed;
          crawler2Pos -= crawler2Speed;

          // Check if both passed through
          if (crawler1Pos - waveWidth >= collisionPoint &&
              crawler2Pos + waveWidth <= collisionPoint) {
            // Spawn sparks
            for (int i = 0; i < 35; i++) {
              sparks[i].active = true;
              sparks[i].position = collisionPoint;
              sparks[i].velocity = (random(20, 60) / 100.0) * (random(0, 2) == 0 ? 1 : -1);
              sparks[i].age = 0;
              sparks[i].lifetime = 25 + random(30);
            }
            state = SPARKLING;
          }
        }
        break;

      case SPARKLING:
        {
          bool anyActive = false;

          for (int i = 0; i < 35; i++) {
            if (!sparks[i].active) continue;
            anyActive = true;

            sparks[i].age += 1.0;
            if (sparks[i].age >= sparks[i].lifetime) {
              sparks[i].active = false;
              continue;
            }

            sparks[i].position += sparks[i].velocity;
            if (sparks[i].position < 0 || sparks[i].position >= numPixels) {
              sparks[i].active = false;
              continue;
            }
          }

          if (!anyActive) {
            state = RESET;
            resetDelay = 60;
          }
        }
        break;

      case RESET:
        {
          resetDelay--;
          if (resetDelay <= 0) {
            crawler1Pos = -8;
            crawler2Pos = numPixels + 8;

            // Use same speed for both so they meet at center
            float speed = 0.5 + (random(0, 100) / 100.0);
            crawler1Speed = speed;
            crawler2Speed = speed;

            crawler1SpawnDelay = random(0, 40);
            crawler2SpawnDelay = random(0, 40);
            framesSinceReset = 0;
            state = APPROACHING;
          }
        }
        break;
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = ADD) override {
    // Render crawlers during APPROACHING and COLLIDING
    if (state == APPROACHING || state == COLLIDING) {
      // Crawler 1
      if (framesSinceReset >= crawler1SpawnDelay) {
        int start1 = max(0, (int)(crawler1Pos - waveWidth));
        int end1 = min(numPixels - 1, (int)(crawler1Pos + waveWidth + 1));

        if (state == COLLIDING) {
          end1 = min(collisionPoint, end1);
        }

        for (int i = start1; i <= end1; i++) {
          float distance = abs((float)i - crawler1Pos);
          if (distance <= waveWidth) {
            float brightness = cos(distance / waveWidth * 3.14159 / 2);
            int r = CRAWLER_R * brightness;
            int g = CRAWLER_G * brightness;
            int b = CRAWLER_B * brightness;

            if (blend == ADD) {
              buffer[i][0] = min(255, (int)buffer[i][0] + r);
              buffer[i][1] = min(255, (int)buffer[i][1] + g);
              buffer[i][2] = min(255, (int)buffer[i][2] + b);
            }
          }
        }
      }

      // Crawler 2
      if (framesSinceReset >= crawler2SpawnDelay) {
        int start2 = max(0, (int)(crawler2Pos - waveWidth));
        int end2 = min(numPixels - 1, (int)(crawler2Pos + waveWidth + 1));

        if (state == COLLIDING) {
          start2 = max(collisionPoint, start2);
        }

        for (int i = start2; i <= end2; i++) {
          float distance = abs((float)i - crawler2Pos);
          if (distance <= waveWidth) {
            float brightness = cos(distance / waveWidth * 3.14159 / 2);
            int r = CRAWLER_R * brightness;
            int g = CRAWLER_G * brightness;
            int b = CRAWLER_B * brightness;

            if (blend == ADD) {
              buffer[i][0] = min(255, (int)buffer[i][0] + r);
              buffer[i][1] = min(255, (int)buffer[i][1] + g);
              buffer[i][2] = min(255, (int)buffer[i][2] + b);
            }
          }
        }
      }
    }

    // Render sparks during SPARKLING
    if (state == SPARKLING) {
      for (int i = 0; i < 35; i++) {
        if (!sparks[i].active) continue;

        // Calculate brightness with flickering
        float lifecycle = sparks[i].age / sparks[i].lifetime;
        float brightness = 1.0 - lifecycle;
        brightness = brightness * brightness;
        float flicker = 0.7 + (random(0, 60) / 100.0);
        brightness = brightness * flicker;
        brightness = min(1.0f, max(0.0f, brightness));

        // Yellow-white colors
        int colorChoice = i % 7;
        int r, g, b;
        switch(colorChoice) {
          case 0: r = 255 * brightness; g = 255 * brightness; b = 150 * brightness; break;
          case 1: r = 255 * brightness; g = 255 * brightness; b = 100 * brightness; break;
          case 2: r = 255 * brightness; g = 240 * brightness; b = 200 * brightness; break;
          case 3: r = 255 * brightness; g = 255 * brightness; b = 180 * brightness; break;
          case 4: r = 255 * brightness; g = 255 * brightness; b = 0; break;
          case 5: r = 255 * brightness; g = 245 * brightness; b = 150 * brightness; break;
          default: r = 255 * brightness; g = 255 * brightness; b = 120 * brightness; break;
        }

        int pixelPos = (int)sparks[i].position;
        if (pixelPos >= 0 && pixelPos < numPixels) {
          if (blend == ADD) {
            buffer[pixelPos][0] = min(255, (int)buffer[pixelPos][0] + r);
            buffer[pixelPos][1] = min(255, (int)buffer[pixelPos][1] + g);
            buffer[pixelPos][2] = min(255, (int)buffer[pixelPos][2] + b);
          }
        }
      }
    }
  }
};

#endif
