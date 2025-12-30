#ifndef NEBULA_BACKGROUND_H
#define NEBULA_BACKGROUND_H

#include "Effect.h"

// Nebula background - breathing waves with blue-to-magenta color shifts
class NebulaBackground : public BackgroundEffect {
private:
  float* ledValues;
  int numPixels;
  unsigned long frameCount;

  // Background wave parameters
  const float BREATH_FREQUENCY = 0.035;
  const float BREATH_CENTER = 0.20;
  const float BREATH_AMPLITUDE = 0.15;
  const float SPATIAL_AMPLITUDE = 0.20;
  const float SPATIAL_SPEED = 0.02;
  const float BACKGROUND_MAX = 0.60;

public:
  NebulaBackground(int pixels) : numPixels(pixels), frameCount(0) {
    ledValues = new float[pixels];
    for (int i = 0; i < pixels; i++) {
      ledValues[i] = 0;
    }
  }

  ~NebulaBackground() {
    delete[] ledValues;
  }

  void update() override {
    frameCount++;

    // Global breathing effect
    float t = (float)frameCount;
    float breathing = BREATH_CENTER + BREATH_AMPLITUDE * sin(t * BREATH_FREQUENCY);

    // Calculate spatial wave for each LED
    for (int i = 0; i < numPixels; i++) {
      float normalized_pos = (float)i / numPixels;
      float phase = normalized_pos + t * SPATIAL_SPEED;
      float spatial = SPATIAL_AMPLITUDE * (0.5 + 0.5 * cos(2.0 * 3.14159 * phase));
      float combined = breathing + spatial;
      ledValues[i] = min(BACKGROUND_MAX, combined);
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    for (int i = 0; i < numPixels; i++) {
      // Create color variation based on position and time
      float colorPhase = (float)i / numPixels * 3.14159 * 2.0 + (float)frameCount * 0.03;
      float colorShift = 0.5 + 0.5 * sin(colorPhase);

      // Shift between pure blue and intense magenta
      int bg_r = 20 + colorShift * 235;
      int bg_g = 30 - colorShift * 20;
      int bg_b = 255 - colorShift * 125;

      // Apply brightness and write to buffer
      buffer[i][0] = bg_r * ledValues[i];
      buffer[i][1] = bg_g * ledValues[i];
      buffer[i][2] = bg_b * ledValues[i];
    }
  }
};

#endif
