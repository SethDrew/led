#ifndef ENHANCED_CRAWL_FOREGROUND_H
#define ENHANCED_CRAWL_FOREGROUND_H

#include "Effect.h"

// Enhanced crawl - smooth wave with multiple color modes
class EnhancedCrawlForeground : public ForegroundEffect {
private:
  int numPixels;
  float waveWidth;
  int colorMode;       // 0: solid, 1: rainbow gradient, 2: color-shifting, 3: warm white
  int baseHue;         // Base hue for solid color mode
  float colorShiftSpeed;
  int offset;          // Wave position
  float shiftingHue;   // For color-shifting mode

public:
  EnhancedCrawlForeground(int pixels, float wave_width, int color_mode, int base_hue, float color_shift_speed)
    : numPixels(pixels), waveWidth(wave_width), colorMode(color_mode),
      baseHue(base_hue), colorShiftSpeed(color_shift_speed),
      offset(-(int)wave_width), shiftingHue(0) {}

  void update() override {
    // Move the wave forward
    offset++;

    // Update shifting hue for color-shift mode
    if (colorMode == 2) {
      shiftingHue += colorShiftSpeed;
      if (shiftingHue >= 256) shiftingHue = 0;
    }

    // Check if wave has fully exited the strip (trailing edge cleared)
    if (offset > numPixels + (int)waveWidth) {
      offset = -(int)waveWidth;  // Reset to start just before the strip
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    for (int i = 0; i < numPixels; i++) {
      // Calculate distance from current LED to wave center
      float distance = abs(i - offset);

      // Create soft gradient using cosine for smooth falloff
      float brightness = 0.0;
      if (distance <= waveWidth) {
        brightness = max(0.0, cos(distance / waveWidth * 3.14159 / 2));
      }

      uint8_t r, g, b;

      switch(colorMode) {
        case 0: // Solid color mode - entire wave is one hue
          hsvToRgb((uint8_t)baseHue, 255, (uint8_t)(brightness * 255), r, g, b);
          break;

        case 1: // Rainbow gradient mode - wave itself is a rainbow
          {
            // Map distance within wave to a hue (creates rainbow across the wave)
            uint8_t hue = (uint8_t)((distance / waveWidth) * 255);
            hsvToRgb(hue, 255, (uint8_t)(brightness * 255), r, g, b);
          }
          break;

        case 2: // Color-shifting mode - entire wave cycles through colors over time
          hsvToRgb((uint8_t)shiftingHue, 255, (uint8_t)(brightness * 255), r, g, b);
          break;

        case 3: // Custom RGB mode - warm white
        default:
          r = 255 * brightness;
          g = 240 * brightness;
          b = 200 * brightness;
          break;
      }

      if (blend == REPLACE) {
        buffer[i][0] = r;
        buffer[i][1] = g;
        buffer[i][2] = b;
      } else if (blend == ADD) {
        buffer[i][0] = min(255, (int)buffer[i][0] + r);
        buffer[i][1] = min(255, (int)buffer[i][1] + g);
        buffer[i][2] = min(255, (int)buffer[i][2] + b);
      }
    }
  }
};

#endif
