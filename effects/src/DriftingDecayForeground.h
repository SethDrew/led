#ifndef DRIFTING_DECAY_FOREGROUND_H
#define DRIFTING_DECAY_FOREGROUND_H

#include "Effect.h"

// Drifting decay - white crawlers that drift apart and decay to colors
class DriftingDecayForeground : public ForegroundEffect {
private:
  class DriftCrawler {
  private:
    struct Particle {
      float position;
      float velocity;
      uint8_t r, g, b;
      float brightness;  // 0.0 to 1.0
    };

    Particle particles[5];
    uint8_t state;  // 0=fade_in, 1=drift, 2=done
    int age;
    uint8_t rDecayRate, gDecayRate, bDecayRate;
    int numPixels;

  public:
    DriftCrawler(int pixels) : numPixels(pixels), state(2) {}

    void spawn(float centerPos) {
      rDecayRate = random(6, 12);
      gDecayRate = random(6, 12);
      bDecayRate = random(6, 12);

      for (int i = 0; i < 5; i++) {
        particles[i].position = centerPos + (i - 2);
        particles[i].velocity = (random(-40, 40) / 100.0);
        particles[i].r = 255;
        particles[i].g = 255;
        particles[i].b = 255;
        particles[i].brightness = 0.0;
      }
      state = 0;
      age = 0;
    }

    void update() {
      if (state == 2) return;

      age++;

      if (state == 0) {  // Fade in
        for (int i = 0; i < 5; i++) {
          particles[i].brightness = min(1.0f, particles[i].brightness + 0.05f);
        }
        if (particles[0].brightness >= 1.0f) {
          state = 1;
        }
      } else if (state == 1) {  // Drift and decay
        for (int i = 0; i < 5; i++) {
          particles[i].position += particles[i].velocity;

          int totalRGB = particles[i].r + particles[i].g + particles[i].b;
          if (totalRGB > 200) {
            int newR = (particles[i].r > rDecayRate) ? particles[i].r - rDecayRate : 0;
            int newG = (particles[i].g > gDecayRate) ? particles[i].g - gDecayRate : 0;
            int newB = (particles[i].b > bDecayRate) ? particles[i].b - bDecayRate : 0;
            int newTotal = newR + newG + newB;

            if (newTotal >= 200) {
              particles[i].r = newR;
              particles[i].g = newG;
              particles[i].b = newB;
            }
          }

          float brightDecay = random(4, 7) / 1000.0f;
          particles[i].brightness = max(0.0f, particles[i].brightness - brightDecay);
        }

        bool allDead = true;
        for (int i = 0; i < 5; i++) {
          if (particles[i].brightness > 0.01f) {
            allDead = false;
            break;
          }
        }
        if (allDead) {
          state = 2;
        }
      }
    }

    void render(uint8_t buffer[][3]) {
      if (state == 2) return;

      for (int i = 0; i < 5; i++) {
        int pixelPos = (int)particles[i].position;
        if (pixelPos >= 0 && pixelPos < numPixels && particles[i].brightness > 0.01f) {
          int r = particles[i].r * particles[i].brightness;
          int g = particles[i].g * particles[i].brightness;
          int b = particles[i].b * particles[i].brightness;

          buffer[pixelPos][0] = min(255, (int)buffer[pixelPos][0] + r);
          buffer[pixelPos][1] = min(255, (int)buffer[pixelPos][1] + g);
          buffer[pixelPos][2] = min(255, (int)buffer[pixelPos][2] + b);
        }
      }
    }

    bool isActive() {
      return state != 2;
    }

    float getCenterPosition() {
      if (state == 2) return -999;
      float sum = 0;
      for (int i = 0; i < 5; i++) {
        sum += particles[i].position;
      }
      return sum / 5.0;
    }
  };

  const int MAX_CRAWLERS = 4;
  DriftCrawler* crawlers[4];
  int numPixels;
  int frameCount;

public:
  DriftingDecayForeground(int pixels) : numPixels(pixels), frameCount(0) {
    for (int i = 0; i < MAX_CRAWLERS; i++) {
      crawlers[i] = new DriftCrawler(pixels);
    }
  }

  ~DriftingDecayForeground() {
    for (int i = 0; i < MAX_CRAWLERS; i++) {
      delete crawlers[i];
    }
  }

  void update() override {
    frameCount++;

    // Spawn new crawler every 50 frames
    if (frameCount % 50 == 0) {
      for (int i = 0; i < MAX_CRAWLERS; i++) {
        if (!crawlers[i]->isActive()) {
          bool validPosition = false;
          float randomPos = 0;
          int attempts = 0;

          while (!validPosition && attempts < 20) {
            randomPos = random(20, numPixels - 20);
            validPosition = true;

            // Check distance from all active crawlers
            for (int j = 0; j < MAX_CRAWLERS; j++) {
              if (j != i && crawlers[j]->isActive()) {
                float otherPos = crawlers[j]->getCenterPosition();
                if (otherPos > -100 && fabs(randomPos - otherPos) < 15) {
                  validPosition = false;
                  break;
                }
              }
            }
            attempts++;
          }

          if (validPosition) {
            crawlers[i]->spawn(randomPos);
          }
          break;
        }
      }
    }

    // Update all crawlers
    for (int i = 0; i < MAX_CRAWLERS; i++) {
      crawlers[i]->update();
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = ADD) override {
    // Render all crawlers (they handle additive blending internally)
    for (int i = 0; i < MAX_CRAWLERS; i++) {
      crawlers[i]->render(buffer);
    }
  }
};

#endif
