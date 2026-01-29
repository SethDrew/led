#ifndef FRAGMENTATION_FOREGROUND_H
#define FRAGMENTATION_FOREGROUND_H

#include "Effect.h"

// Fragmentation - white crawler decomposes into RGB particles and reforms
class FragmentationForeground : public ForegroundEffect {
private:
  enum State { STABLE, FRAGMENTING, REFORMING };
  State state;

  int numPixels;
  const float centerPos;
  const float waveWidth = 6.0;
  const int numParticles = 40;

  ColorParticle particles[40];
  int stableFrames;
  int fragmentFrames;

public:
  FragmentationForeground(int pixels)
    : numPixels(pixels), centerPos(pixels / 2.0),
      state(STABLE), stableFrames(0), fragmentFrames(0) {
    for (int i = 0; i < numParticles; i++) {
      particles[i].active = false;
    }
  }

  void update() override {
    switch(state) {
      case STABLE:
        {
          stableFrames++;
          if (stableFrames >= 60) {  // Stay stable for 60 frames
            state = FRAGMENTING;
            stableFrames = 0;
            fragmentFrames = 0;

            // Initialize particles from crawler pixels
            int particleIndex = 0;

            for (int pixelPos = (int)(centerPos - waveWidth); pixelPos <= (int)(centerPos + waveWidth) && particleIndex < numParticles - 2; pixelPos++) {
              if (pixelPos < 0 || pixelPos >= numPixels) continue;

              float distance = fabs((float)pixelPos - centerPos);
              if (distance <= waveWidth) {
                float brightness = cos(distance / waveWidth * 3.14159 / 2);

                if (brightness > 0.3) {
                  int spawnFrame = (int)((waveWidth - distance) / waveWidth * 50);

                  // Create 3 particles (R, G, B) from this pixel
                  for (int component = 0; component < 3; component++) {
                    if (particleIndex >= numParticles) break;

                    particles[particleIndex].active = false;
                    particles[particleIndex].position = (float)pixelPos;
                    particles[particleIndex].spawnFrame = spawnFrame;
                    particles[particleIndex].velocity = ((random(0, 80) - 40) / 100.0);
                    particles[particleIndex].originalVelocity = particles[particleIndex].velocity;

                    int scaledBrightness = (int)(255 * brightness);
                    if (component == 0) {
                      particles[particleIndex].r = scaledBrightness;
                      particles[particleIndex].g = 0;
                      particles[particleIndex].b = 0;
                    } else if (component == 1) {
                      particles[particleIndex].r = 0;
                      particles[particleIndex].g = scaledBrightness;
                      particles[particleIndex].b = 0;
                    } else {
                      particles[particleIndex].r = 0;
                      particles[particleIndex].g = 0;
                      particles[particleIndex].b = scaledBrightness;
                    }

                    particleIndex++;
                  }
                }
              }
            }

            // Deactivate unused slots
            for (int i = particleIndex; i < numParticles; i++) {
              particles[i].active = false;
            }
          }
        }
        break;

      case FRAGMENTING:
        {
          fragmentFrames++;

          // Activate particles
          for (int i = 0; i < numParticles; i++) {
            if (!particles[i].active && fragmentFrames >= particles[i].spawnFrame) {
              particles[i].active = true;
            }

            if (particles[i].active) {
              particles[i].position += particles[i].velocity;
            }
          }

          if (fragmentFrames >= 100) {
            state = REFORMING;
            fragmentFrames = 0;

            // Reverse velocities
            for (int i = 0; i < numParticles; i++) {
              if (particles[i].active) {
                particles[i].velocity = -particles[i].originalVelocity * 1.0;
              }
            }
          }
        }
        break;

      case REFORMING:
        {
          fragmentFrames++;

          bool allReturned = true;

          // Move particles and deactivate in reverse order
          for (int i = 0; i < numParticles; i++) {
            if (particles[i].active) {
              int despawnFrame = 100 - particles[i].spawnFrame * 2;

              if (fragmentFrames >= despawnFrame) {
                particles[i].active = false;
                continue;
              }

              allReturned = false;
              particles[i].position += particles[i].velocity;
            }
          }

          if (allReturned || fragmentFrames >= 100) {
            for (int i = 0; i < numParticles; i++) {
              particles[i].active = false;
            }
            state = STABLE;
            stableFrames = 0;
          }
        }
        break;
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    switch(state) {
      case STABLE:
        {
          // Show white crawler at center
          for (int i = 0; i < numPixels; i++) {
            float distance = fabs((float)i - centerPos);
            if (distance <= waveWidth) {
              float b = cos(distance / waveWidth * 3.14159 / 2);
              int brightness = 255 * b;

              if (blend == REPLACE) {
                buffer[i][0] = brightness;
                buffer[i][1] = brightness;
                buffer[i][2] = brightness;
              } else if (blend == ADD) {
                buffer[i][0] = min(255, (int)buffer[i][0] + brightness);
                buffer[i][1] = min(255, (int)buffer[i][1] + brightness);
                buffer[i][2] = min(255, (int)buffer[i][2] + brightness);
              }
            }
          }
        }
        break;

      case FRAGMENTING:
        {
          // Render crawler being consumed from outside-in
          for (int i = 0; i < numPixels; i++) {
            float distance = fabs((float)i - centerPos);
            if (distance <= waveWidth) {
              float brightness = cos(distance / waveWidth * 3.14159 / 2);
              int pixelSpawnFrame = (int)((waveWidth - distance) / waveWidth * 50);

              if (fragmentFrames < pixelSpawnFrame && brightness > 0.3) {
                int b = 255 * brightness;
                if (blend == REPLACE) {
                  buffer[i][0] = b;
                  buffer[i][1] = b;
                  buffer[i][2] = b;
                } else if (blend == ADD) {
                  buffer[i][0] = min(255, (int)buffer[i][0] + b);
                  buffer[i][1] = min(255, (int)buffer[i][1] + b);
                  buffer[i][2] = min(255, (int)buffer[i][2] + b);
                }
              }
            }
          }

          // Render particles
          for (int i = 0; i < numParticles; i++) {
            if (particles[i].active) {
              int pixelPos = (int)particles[i].position;
              if (pixelPos >= 0 && pixelPos < numPixels) {
                int framesSinceSpawn = fragmentFrames - particles[i].spawnFrame;
                float brightness = 1.0 - (framesSinceSpawn / 100.0);
                brightness = max(0.0f, min(1.0f, brightness));

                int r = particles[i].r * brightness;
                int g = particles[i].g * brightness;
                int b = particles[i].b * brightness;

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
        }
        break;

      case REFORMING:
        {
          // Render reforming crawler
          float crawlerBrightness = fragmentFrames / 100.0;
          crawlerBrightness = min(1.0f, crawlerBrightness);

          for (int i = 0; i < numPixels; i++) {
            float distance = fabs((float)i - centerPos);
            if (distance <= waveWidth) {
              float b = cos(distance / waveWidth * 3.14159 / 2) * crawlerBrightness;
              int brightness = 255 * b;

              if (blend == REPLACE) {
                buffer[i][0] = brightness;
                buffer[i][1] = brightness;
                buffer[i][2] = brightness;
              } else if (blend == ADD) {
                buffer[i][0] = min(255, (int)buffer[i][0] + brightness);
                buffer[i][1] = min(255, (int)buffer[i][1] + brightness);
                buffer[i][2] = min(255, (int)buffer[i][2] + brightness);
              }
            }
          }

          // Render particles
          for (int i = 0; i < numParticles; i++) {
            if (particles[i].active) {
              int pixelPos = (int)particles[i].position;
              if (pixelPos >= 0 && pixelPos < numPixels) {
                float brightness = min(1.0f, fragmentFrames / 50.0);

                int r = particles[i].r * brightness;
                int g = particles[i].g * brightness;
                int b = particles[i].b * brightness;

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
        }
        break;
    }
  }
};

#endif
