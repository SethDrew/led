#ifndef RAINBOW_CIRCLE_FOREGROUND_H
#define RAINBOW_CIRCLE_FOREGROUND_H

#include "Effect.h"

// Rainbow circle passing vertically through the LED strip
class RainbowCircleForeground : public ForegroundEffect {
private:
  int numPixels;
  float radius;
  float speed;
  float verticalPos;
  int pauseFrames;
  const int PAUSE_DURATION = 2000;

public:
  RainbowCircleForeground(int pixels, float circle_radius, float circle_speed)
    : numPixels(pixels), radius(circle_radius), speed(circle_speed),
      verticalPos(-circle_radius), pauseFrames(0) {}

  void update() override {
    // Handle pause between cycles
    if (pauseFrames > 0) {
      pauseFrames--;
      return;
    }

    // Move the circle vertically through the strip
    verticalPos += speed;

    // Reset when circle fully exits below the strip
    if (verticalPos > radius) {
      verticalPos = -radius;
      pauseFrames = PAUSE_DURATION;
    }
  }

  void render(uint8_t buffer[][3], BlendMode blend = REPLACE) override {
    // During pause, don't render anything
    if (pauseFrames > 0) {
      return;
    }

    // Center of the LED strip (where the circle passes through)
    const float stripCenter = numPixels / 2.0;

    // Calculate if the horizontal strip intersects the circle at this vertical position
    float distanceFromCenter = abs(verticalPos);

    if (distanceFromCenter <= radius) {
      // Calculate horizontal distance from center to intersection points
      // Using circle equation: x² + y² = r²  →  x = ±sqrt(r² - y²)
      float halfWidth = sqrt(radius * radius - distanceFromCenter * distanceFromCenter);

      // Calculate the two intersection points
      int leftPoint = (int)(stripCenter - halfWidth);
      int rightPoint = (int)(stripCenter + halfWidth);

      // Determine hue based on vertical position through the circle
      uint8_t hue = (uint8_t)((verticalPos + radius) / (2.0 * radius) * 255);

      // Fill interior with 10% brightness
      uint8_t fill_r, fill_g, fill_b;
      hsvToRgb(hue, 255, 25, fill_r, fill_g, fill_b);

      for (int i = max(0, leftPoint); i <= min(numPixels - 1, rightPoint); i++) {
        if (blend == REPLACE) {
          buffer[i][0] = fill_r;
          buffer[i][1] = fill_g;
          buffer[i][2] = fill_b;
        } else if (blend == ADD) {
          buffer[i][0] = min(255, (int)buffer[i][0] + fill_r);
          buffer[i][1] = min(255, (int)buffer[i][1] + fill_g);
          buffer[i][2] = min(255, (int)buffer[i][2] + fill_b);
        }
      }

      // Light up edge points at full brightness (overwrite the fill)
      uint8_t edge_r, edge_g, edge_b;
      hsvToRgb(hue, 255, 255, edge_r, edge_g, edge_b);

      if (leftPoint >= 0 && leftPoint < numPixels) {
        if (blend == REPLACE) {
          buffer[leftPoint][0] = edge_r;
          buffer[leftPoint][1] = edge_g;
          buffer[leftPoint][2] = edge_b;
        } else if (blend == ADD) {
          buffer[leftPoint][0] = min(255, (int)buffer[leftPoint][0] + edge_r);
          buffer[leftPoint][1] = min(255, (int)buffer[leftPoint][1] + edge_g);
          buffer[leftPoint][2] = min(255, (int)buffer[leftPoint][2] + edge_b);
        }
      }

      if (rightPoint >= 0 && rightPoint < numPixels && rightPoint != leftPoint) {
        if (blend == REPLACE) {
          buffer[rightPoint][0] = edge_r;
          buffer[rightPoint][1] = edge_g;
          buffer[rightPoint][2] = edge_b;
        } else if (blend == ADD) {
          buffer[rightPoint][0] = min(255, (int)buffer[rightPoint][0] + edge_r);
          buffer[rightPoint][1] = min(255, (int)buffer[rightPoint][1] + edge_g);
          buffer[rightPoint][2] = min(255, (int)buffer[rightPoint][2] + edge_b);
        }
      }
    }
  }
};

#endif
