#ifndef EFFECT_H
#define EFFECT_H

#include <Arduino.h>

// Blend modes for compositing layers
enum BlendMode {
  REPLACE,  // Foreground replaces background
  ADD,      // Additive blending (sum with cap at 255)
  ALPHA,    // Alpha blending (not yet implemented)
  MULTIPLY  // Multiply blending (not yet implemented)
};

// Common data structure for particle-based effects
struct Orb {
  float position;    // Current position along strip
  float velocity;    // Speed and direction of movement
  float age;         // How long the orb has existed
  float lifetime;    // Total lifetime of this orb
  bool active;       // Whether this orb slot is in use
};

// Color particle for fragmentation effect
struct ColorParticle {
  float position;
  float velocity;
  float originalVelocity;  // Store for reversal
  uint8_t spawnFrame;      // When this particle should activate (0-255 frames)
  uint8_t r, g, b;         // RGB color components
  bool active;
  float homePosition;      // Original position to return to when reforming
};

// HSV to RGB conversion utility (H: 0-255, S: 0-255, V: 0-255)
inline void hsvToRgb(uint8_t h, uint8_t s, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b) {
  if (s == 0) {
    r = g = b = v;
    return;
  }

  uint8_t region = h / 43;
  uint8_t remainder = (h - (region * 43)) * 6;

  uint8_t p = (v * (255 - s)) >> 8;
  uint8_t q = (v * (255 - ((s * remainder) >> 8))) >> 8;
  uint8_t t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

  switch (region) {
    case 0:  r = v; g = t; b = p; break;
    case 1:  r = q; g = v; b = p; break;
    case 2:  r = p; g = v; b = t; break;
    case 3:  r = p; g = q; b = v; break;
    case 4:  r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
}

// Base class for all effects
class Effect {
public:
  virtual ~Effect() {}

  // Update effect state (called each frame)
  virtual void update() = 0;

  // Render effect to buffer
  // buffer: RGB buffer [NUM_PIXELS][3]
  // blend: How to blend with existing buffer content
  virtual void render(uint8_t buffer[][3], BlendMode blend = ADD) = 0;
};

// Base class for background effects
class BackgroundEffect : public Effect {
public:
  // Backgrounds typically use REPLACE mode
  virtual void render(uint8_t buffer[][3], BlendMode blend = REPLACE) = 0;
};

// Base class for foreground effects
class ForegroundEffect : public Effect {
public:
  // Foregrounds typically use ADD mode
  virtual void render(uint8_t buffer[][3], BlendMode blend = ADD) = 0;
};

#endif
