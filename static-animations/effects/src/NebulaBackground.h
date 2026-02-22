#ifndef NEBULA_BACKGROUND_H
#define NEBULA_BACKGROUND_H

#include "Effect.h"

// Nebula background - breathing waves with blue-to-magenta color shifts
// OPTIMIZED: Uses uint8_t instead of float (75% memory savings)
class NebulaBackground : public BackgroundEffect {
private:
  uint8_t* ledValues;  // Changed from float* to uint8_t*
  int numPixels;
  unsigned long frameCount;

  // Background wave parameters (scaled to 0-255 range)
  const float BREATH_FREQUENCY = 0.035;
  const uint8_t BREATH_CENTER = 51;      // ~0.20 * 255
  const uint8_t BREATH_AMPLITUDE = 38;   // ~0.15 * 255
  const uint8_t SPATIAL_AMPLITUDE = 51;  // ~0.20 * 255
  const float SPATIAL_SPEED = 0.02;
  const uint8_t BACKGROUND_MAX = 153;    // ~0.60 * 255

public:
  NebulaBackground(int pixels) : numPixels(pixels), frameCount(0) {
    ledValues = new uint8_t[pixels];
    for (int i = 0; i < pixels; i++) {
      ledValues[i] = 0;
    }
  }

  ~NebulaBackground() {
    delete[] ledValues;
  }

  void update() override {
    frameCount++;

    // Global breathing effect (returns -1.0 to 1.0, we scale to 0-255)
    float t = (float)frameCount;
    int16_t breathing = BREATH_CENTER + (int16_t)(BREATH_AMPLITUDE * sin(t * BREATH_FREQUENCY));

    // Calculate spatial wave for each LED
    for (int i = 0; i < numPixels; i++) {
      float normalized_pos = (float)i / numPixels;
      float phase = normalized_pos + t * SPATIAL_SPEED;

      // Cosine returns -1.0 to 1.0, we want 0 to SPATIAL_AMPLITUDE
      int16_t spatial = (int16_t)(SPATIAL_AMPLITUDE * (0.5 + 0.5 * cos(2.0 * 3.14159 * phase)));

      // Combine and clamp to valid range
      int16_t combined = breathing + spatial;
      if (combined < 0) combined = 0;
      if (combined > BACKGROUND_MAX) combined = BACKGROUND_MAX;

      ledValues[i] = (uint8_t)combined;
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    for (int i = 0; i < numPixels; i++) {
      // Create color variation based on position and time
      float colorPhase = (float)i / numPixels * 3.14159 * 2.0 + (float)frameCount * BREATH_FREQUENCY;
      float colorShift = 0.5 + 0.5 * sin(colorPhase);

      // Shift between pure blue and intense magenta
      uint8_t bg_r = 20 + colorShift * 235;
      uint8_t bg_g = 30 - colorShift * 20;
      uint8_t bg_b = 255 - colorShift * 125;

      // Apply brightness using 16-bit math to avoid overflow, then scale back
      // brightness is 0-153, color is 0-255
      buffer[i][0] = ((uint16_t)bg_r * ledValues[i]) / 255;
      buffer[i][1] = ((uint16_t)bg_g * ledValues[i]) / 255;
      buffer[i][2] = ((uint16_t)bg_b * ledValues[i]) / 255;
    }
  }
};

#endif
