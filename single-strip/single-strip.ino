#include <Adafruit_NeoPixel.h>

// Define pin for LED strip
#define LED_PIN 6

// Define the number of NeoPixels in strip
#define NUM_PIXELS 100

// Create NeoPixel object
Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

// Orb structure for particle animations
struct Orb {
  float position;    // Current position along strip
  float velocity;    // Speed and direction of movement
  float age;         // How long the orb has existed
  float lifetime;    // Total lifetime of this orb
  bool active;       // Whether this orb slot is in use
};

// Forward declarations
void softWhiteCrawlAnimation();
void softWhiteCrawl(float waveWidth);
void crawlingStarsAnimation();
void crawlingStars(int maxOrbs, float orbSize, float speed);
void nebulaAnimation();
void nebulaWithBackground(int maxOrbs, float orbSize, float speed);
void updateBackgroundWaves(float* ledValues_background, unsigned long frameCount);
void updateOrbs(Orb* orbs, float* ledValues_stars, int maxOrbs, float orbSize, float speed);
void renderComposited(const float* ledValues_stars, const float* ledValues_background, unsigned long frameCount);
void rainbowCircleAnimation();
void rainbowCircle(float circleWidth, float speed);
void enhancedCrawlAnimation();
void enhancedCrawl(float waveWidth, int colorMode, int baseHue, float colorShiftSpeed);
void collisionAnimation();
void collision();

void setup() {
  Serial.begin(115200);
  Serial.println("Single Strip LED Animations");

  // Initialize strip
  strip.begin();
  strip.setBrightness(200);
  strip.show();
}

void loop() {
  // softWhiteCrawlAnimation(); delay(20);
  // crawlingStarsAnimation(); delay(20);
  // nebulaAnimation(); delay(20);
  // rainbowCircleAnimation();
  // enhancedCrawlAnimation(); delay(20);
  collisionAnimation(); delay(20);
}

// ===== Animation Wrappers =====

void softWhiteCrawlAnimation() {
  float waveWidth = 4.0;  // How many LEDs the gradient spans
  softWhiteCrawl(waveWidth);
}

void crawlingStarsAnimation() {
  int maxOrbs = 5;        // Maximum number of orbs at once
  float orbSize = 7.0;    // Size of each orb glow (in LEDs)
  float speed = 0.3;     // How fast orbs drift
  crawlingStars(maxOrbs, orbSize, speed);
}

void nebulaAnimation() {
  int maxOrbs = 5;        // Maximum number of orbs at once
  float orbSize = 7.0;    // Size of each orb glow (in LEDs)
  float speed = 0.6;     // How fast orbs drift
  nebulaWithBackground(maxOrbs, orbSize, speed);
}

void rainbowCircleAnimation() {
  float circleWidth = 15.0;  // Width of the circular wave
  float speed = 0.1;         // How fast the circle moves (slower for better resolution)
  rainbowCircle(circleWidth, speed);
}

void enhancedCrawlAnimation() {
  float waveWidth = 15.0;     // How many LEDs the gradient spans
  int colorMode = 1;          // 0=solid, 1=rainbow gradient, 2=color-shifting, 3=custom RGB
  int baseHue = 0;            // Starting hue (0-255) - used in solid/color-shift modes
  float colorShiftSpeed = 0.5; // Speed of color changes (used in color-shift mode)
  enhancedCrawl(waveWidth, colorMode, baseHue, colorShiftSpeed);
}

void collisionAnimation() {
  collision();
}

// ===== Helper Functions for Nebula Animation =====

// Update background pulsating waves combining breathing and spatial effects
void updateBackgroundWaves(float* ledValues_background, unsigned long frameCount) {
  const float BREATH_FREQUENCY = 0.035;
  const float BREATH_CENTER = 0.09;
  const float BREATH_AMPLITUDE = 0.06;
  const float SPATIAL_AMPLITUDE = 0.10;
  const float SPATIAL_SPEED = 0.02;
  const float BACKGROUND_MAX = 0.25;

  // Global breathing effect - entire strip pulses together
  float t = (float)frameCount;
  float breathing = BREATH_CENTER + BREATH_AMPLITUDE * sin(t * BREATH_FREQUENCY);

  // Calculate spatial wave for each LED
  for (int i = 0; i < NUM_PIXELS; i++) {
    float normalized_pos = (float)i / NUM_PIXELS;
    float phase = normalized_pos + t * SPATIAL_SPEED;

    // Cosine wave creates smooth gradients traveling along the strip
    float spatial = SPATIAL_AMPLITUDE * (0.5 + 0.5 * cos(2.0 * 3.14159 * phase));

    // Combine breathing and spatial waves
    float combined = breathing + spatial;

    // Apply cap to prevent over-brightness
    ledValues_background[i] = min(BACKGROUND_MAX, combined);
  }
}

// Update and manage particle orbs
void updateOrbs(Orb* orbs, float* ledValues_stars, int maxOrbs, float orbSize, float speed) {
  // Count active orbs
  int activeCount = 0;
  for (int i = 0; i < 10; i++) {
    if (orbs[i].active) activeCount++;
  }

  // Spawn new orb if below max and random chance
  if (activeCount < maxOrbs && random(100) < 3) {  // 3% chance per frame
    // Find empty slot
    for (int i = 0; i < 10; i++) {
      if (!orbs[i].active) {
        orbs[i].active = true;
        orbs[i].position = random(0, NUM_PIXELS);
        orbs[i].velocity = (random(0, 2) == 0 ? 1 : -1) * speed * (0.5 + random(100) / 200.0);
        orbs[i].age = 0;
        orbs[i].lifetime = 40 + random(100);  // 40-140 frames lifetime
        break;
      }
    }
  }

  // Update each orb
  for (int i = 0; i < 10; i++) {
    if (!orbs[i].active) continue;

    // Age the orb
    orbs[i].age += 1.0;

    // Deactivate if lifetime exceeded
    if (orbs[i].age >= orbs[i].lifetime) {
      orbs[i].active = false;
      continue;
    }

    // Move the orb
    orbs[i].position += orbs[i].velocity;

    // Wrap position
    if (orbs[i].position < 0) orbs[i].position += NUM_PIXELS;
    if (orbs[i].position >= NUM_PIXELS) orbs[i].position -= NUM_PIXELS;

    // Calculate lifecycle brightness (fade in, stay, fade out)
    float lifecycle = orbs[i].age / orbs[i].lifetime;
    float lifeBrightness = 1.0;

    if (lifecycle < 0.4) {
      // Fade in during first 40% of life
      float t = lifecycle / 0.4;
      lifeBrightness = t * t * (3.0 - 2.0 * t);  // Smoothstep for gentle fade
    } else if (lifecycle > 0.6) {
      // Fade out during last 40% of life
      float t = (1.0 - lifecycle) / 0.4;
      lifeBrightness = t * t * (3.0 - 2.0 * t);  // Smoothstep for gentle fade
    }

    // Add orb's brightness to current position (additive blending, capped at 1.0)
    int orbPixel = (int)orbs[i].position;
    if (orbPixel >= 0 && orbPixel < NUM_PIXELS) {
      ledValues_stars[orbPixel] = min(1.0f, ledValues_stars[orbPixel] + lifeBrightness * 0.6f);
    }
  }
}

// Render composite of stars and background with color blending
void renderComposited(const float* ledValues_stars, const float* ledValues_background, unsigned long frameCount) {
  // Warm white for stars
  const int STAR_COLOR_R = 255;
  const int STAR_COLOR_G = 240;
  const int STAR_COLOR_B = 200;

  for (int i = 0; i < NUM_PIXELS; i++) {
    // Calculate star contribution
    int r_stars = STAR_COLOR_R * ledValues_stars[i];
    int g_stars = STAR_COLOR_G * ledValues_stars[i];
    int b_stars = STAR_COLOR_B * ledValues_stars[i];

    // Create color variation in nebula background based on position and time
    // Simplified to 1 sin call per pixel instead of 3 for better performance
    float colorPhase = (float)i / NUM_PIXELS * 3.14159 * 2.0 + (float)frameCount * 0.03;
    float colorShift = 0.5 + 0.5 * sin(colorPhase);

    // Shift between pure blue (low) and intense magenta (high) for maximum contrast
    // Pure blue: (20,30,255), Intense magenta: (255,10,130)
    int bg_r = 20 + colorShift * 235;   // 20-255
    int bg_g = 30 - colorShift * 20;    // 30-10
    int bg_b = 255 - colorShift * 125;  // 255-130

    // Apply background brightness
    int r_bg = bg_r * ledValues_background[i];
    int g_bg = bg_g * ledValues_background[i];
    int b_bg = bg_b * ledValues_background[i];

    // Additive blend with saturation cap
    int r_final = min(255, r_stars + r_bg);
    int g_final = min(255, g_stars + g_bg);
    int b_final = min(255, b_stars + b_bg);

    strip.setPixelColor(i, strip.Color(r_final, g_final, b_final));
  }

  strip.show();
}

// ===== Core Animation Functions =====

// Soft white crawl effect - a gentle wave of warm white light
// waveWidth: how many LEDs the gradient spans
void softWhiteCrawl(float waveWidth) {
  static int offset = 0;  // Track position of the wave

  // Soft warm white color (slightly yellowish for warmth)
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

  for(int i = 0; i < NUM_PIXELS; i++) {
    // Calculate distance from current LED to wave center (no wrapping)
    float distance = abs(i - offset);

    // Create soft gradient using cosine for smooth falloff
    float brightness = 0.0;
    if (distance <= waveWidth) {
      brightness = max(0.0, cos(distance / waveWidth * 3.14159 / 2));
    }

    // Apply brightness to warm white color
    int r = warmWhiteR * brightness;
    int g = warmWhiteG * brightness;
    int b = warmWhiteB * brightness;

    // Set color on strip
    strip.setPixelColor(i, strip.Color(r, g, b));
  }

  // Update strip
  strip.show();

  // Move the wave forward
  offset++;

  // Check if wave has fully exited the strip (trailing edge cleared)
  if (offset > NUM_PIXELS + (int)waveWidth) {
    offset = -(int)waveWidth;  // Reset to start just before the strip
  }
}

// Enhanced crawl effect with dynamic color control
// waveWidth: how many LEDs the gradient spans
// colorMode: 0=solid color, 1=rainbow gradient, 2=color-shifting wave, 3=custom RGB
// baseHue: hue value (0-255) for solid/color-shift modes
// colorShiftSpeed: speed of hue changes in color-shift mode
void enhancedCrawl(float waveWidth, int colorMode, int baseHue, float colorShiftSpeed) {
  static int offset = 0;           // Track position of the wave
  static float shiftingHue = 0;    // For color-shifting mode

  for(int i = 0; i < NUM_PIXELS; i++) {
    // Calculate distance from current LED to wave center
    float distance = abs(i - offset);

    // Create soft gradient using cosine for smooth falloff
    float brightness = 0.0;
    if (distance <= waveWidth) {
      brightness = max(0.0, cos(distance / waveWidth * 3.14159 / 2));
    }

    uint32_t color;

    switch(colorMode) {
      case 0: // Solid color mode - entire wave is one hue
        color = strip.ColorHSV((baseHue * 256) % 65536, 255, (int)(brightness * 255));
        break;

      case 1: // Rainbow gradient mode - wave itself is a rainbow
        {
          // Map distance within wave to a hue (creates rainbow across the wave)
          int hue = (int)((distance / waveWidth) * 255);
          color = strip.ColorHSV((hue * 256) % 65536, 255, (int)(brightness * 255));
        }
        break;

      case 2: // Color-shifting mode - entire wave cycles through colors over time
        color = strip.ColorHSV(((int)shiftingHue * 256) % 65536, 255, (int)(brightness * 255));
        break;

      case 3: // Custom RGB mode - warm white (like original)
      default:
        {
          int r = 255 * brightness;
          int g = 240 * brightness;
          int b = 200 * brightness;
          color = strip.Color(r, g, b);
        }
        break;
    }

    strip.setPixelColor(i, color);
  }

  strip.show();

  // Move the wave forward
  offset++;

  // Update shifting hue for color-shift mode
  if (colorMode == 2) {
    shiftingHue += colorShiftSpeed;
    if (shiftingHue >= 256) shiftingHue = 0;
  }

  // Check if wave has fully exited the strip (trailing edge cleared)
  if (offset > NUM_PIXELS + (int)waveWidth) {
    offset = -(int)waveWidth;  // Reset to start just before the strip
  }
}

// Crawling stars effect - star-like orbs that fade in, drift, and fade out independently
// Uses history-based decay: only orb positions are set to brightness, everything else decays
// maxOrbs: maximum number of orbs at once
// orbSize: size of each orb's glow (determines decay rate)
// speed: how fast orbs drift along the strip
void crawlingStars(int maxOrbs, float orbSize, float speed) {
  static Orb orbs[10];  // Pool of orb slots (max 10)
  static float ledValues[NUM_PIXELS] = {0};  // Store brightness for each LED
  static bool initialized = false;

  // Initialize orbs on first run
  if (!initialized) {
    for (int i = 0; i < 10; i++) {
      orbs[i].active = false;
    }
    initialized = true;
  }

  // Decay all LEDs based on orb size (creates natural glow/trail)
  float decayFactor = 1.0 - (1.0 / orbSize);
  for (int i = 0; i < NUM_PIXELS; i++) {
    ledValues[i] *= decayFactor;
    if (ledValues[i] < 0.01) {
      ledValues[i] = 0.0;  // Cut off very dim values
    }
  }

  // Count active orbs
  int activeCount = 0;
  for (int i = 0; i < 10; i++) {
    if (orbs[i].active) activeCount++;
  }

  // Spawn new orb if below max and random chance
  if (activeCount < maxOrbs && random(100) < 3) {  // 3% chance per frame
    // Find empty slot
    for (int i = 0; i < 10; i++) {
      if (!orbs[i].active) {
        orbs[i].active = true;
        orbs[i].position = random(0, NUM_PIXELS);
        orbs[i].velocity = (random(0, 2) == 0 ? 1 : -1) * speed * (0.5 + random(100) / 200.0);
        orbs[i].age = 0;
        orbs[i].lifetime = 40 + random(100);  // 40-140 frames lifetime
        break;
      }
    }
  }

  // Update each orb
  for (int i = 0; i < 10; i++) {
    if (!orbs[i].active) continue;

    // Age the orb
    orbs[i].age += 1.0;

    // Deactivate if lifetime exceeded
    if (orbs[i].age >= orbs[i].lifetime) {
      orbs[i].active = false;
      continue;
    }

    // Move the orb
    orbs[i].position += orbs[i].velocity;

    // Wrap position
    if (orbs[i].position < 0) orbs[i].position += NUM_PIXELS;
    if (orbs[i].position >= NUM_PIXELS) orbs[i].position -= NUM_PIXELS;

    // Calculate lifecycle brightness (fade in, stay, fade out)
    float lifecycle = orbs[i].age / orbs[i].lifetime;
    float lifeBrightness = 1.0;

    if (lifecycle < 0.4) {
      // Fade in during first 40% of life
      float t = lifecycle / 0.4;
      lifeBrightness = t * t * (3.0 - 2.0 * t);  // Smoothstep for gentle fade
    } else if (lifecycle > 0.6) {
      // Fade out during last 40% of life
      float t = (1.0 - lifecycle) / 0.4;
      lifeBrightness = t * t * (3.0 - 2.0 * t);  // Smoothstep for gentle fade
    }

    // Add orb's brightness to current position (additive blending, capped at 1.0)
    int orbPixel = (int)orbs[i].position;
    if (orbPixel >= 0 && orbPixel < NUM_PIXELS) {
      ledValues[orbPixel] = min(1.0f, ledValues[orbPixel] + lifeBrightness * 0.6f);
    }
  }

  // Render all LEDs from stored values
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

  for (int i = 0; i < NUM_PIXELS; i++) {
    float brightness = ledValues[i];
    int r = warmWhiteR * brightness;
    int g = warmWhiteG * brightness;
    int b = warmWhiteB * brightness;
    strip.setPixelColor(i, strip.Color(r, g, b));
  }

  // Update strip
  strip.show();
}

// Nebula effect - ethereal pulsating background with drifting star orbs
// Combines breathing waves, spatial gradients, and particle effects
// maxOrbs: maximum number of orbs at once
// orbSize: size of each orb's glow (determines decay rate)
// speed: how fast orbs drift along the strip
void nebulaWithBackground(int maxOrbs, float orbSize, float speed) {
  static Orb orbs[10];  // Pool of orb slots (max 10)
  static float ledValues_stars[NUM_PIXELS] = {0};      // Star particle layer
  static float ledValues_background[NUM_PIXELS] = {0}; // Pulsating background layer
  static unsigned long frameCount = 0;                  // Time tracking
  static bool initialized = false;

  // Initialize orbs on first run
  if (!initialized) {
    for (int i = 0; i < 10; i++) {
      orbs[i].active = false;
    }
    initialized = true;
  }

  // Decay star layer based on orb size (creates natural glow/trail)
  float decayFactor = 1.0 - (1.0 / orbSize);
  for (int i = 0; i < NUM_PIXELS; i++) {
    ledValues_stars[i] *= decayFactor;
    if (ledValues_stars[i] < 0.01) {
      ledValues_stars[i] = 0.0;  // Cut off very dim values
    }
  }

  // Update background pulsating waves
  updateBackgroundWaves(ledValues_background, frameCount);

  // Update and render particle orbs
  updateOrbs(orbs, ledValues_stars, maxOrbs, orbSize, speed);

  // Render composite of stars and background
  renderComposited(ledValues_stars, ledValues_background, frameCount);

  // Increment frame counter for animation timing
  frameCount++;
}

// Rainbow circle effect - visualizes a 2D circle passing through the 1D strip
// Shows the cross-section: starts as 1 point, expands to 2 points, then contracts back
// radius: radius of the circle
// speed: how fast the circle passes through
void rainbowCircle(float radius, float speed) {
  static float verticalPos = -radius;  // Circle's vertical position (starts above strip)
  static int pauseFrames = 0;           // Counter for pause between cycles
  const int PAUSE_DURATION = 2000;      // Pause frames (no delay, so higher count needed)

  // Center of the LED strip (where the circle passes through)
  const float stripCenter = NUM_PIXELS / 2.0;

  // Handle pause between cycles
  if (pauseFrames > 0) {
    // Clear all LEDs during pause
    for (int i = 0; i < NUM_PIXELS; i++) {
      strip.setPixelColor(i, 0);
    }
    strip.show();
    pauseFrames--;
    return;
  }

  // Clear all LEDs first
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, 0);
  }

  // Calculate if the horizontal strip intersects the circle at this vertical position
  float distanceFromCenter = abs(verticalPos);

  if (distanceFromCenter <= radius) {
    // Calculate horizontal distance from center to intersection points
    // Using circle equation: x² + y² = r²  →  x = ±sqrt(r² - y²)
    float halfWidth = sqrt(radius * radius - distanceFromCenter * distanceFromCenter);

    // Calculate the two intersection points (or one if at the very edge)
    int leftPoint = (int)(stripCenter - halfWidth);
    int rightPoint = (int)(stripCenter + halfWidth);

    // Determine hue based on vertical position through the circle
    int hue = (int)((verticalPos + radius) / (2.0 * radius) * 255);

    // Fill interior with 10% brightness
    for (int i = max(0, leftPoint); i <= min(NUM_PIXELS - 1, rightPoint); i++) {
      uint32_t fillColor = strip.ColorHSV((hue * 256) % 65536, 255, 25);  // 10% brightness
      strip.setPixelColor(i, fillColor);
    }

    // Light up edge points at full brightness (overwrite the fill)
    if (leftPoint >= 0 && leftPoint < NUM_PIXELS) {
      uint32_t edgeColor = strip.ColorHSV((hue * 256) % 65536, 255, 255);
      strip.setPixelColor(leftPoint, edgeColor);
    }

    if (rightPoint >= 0 && rightPoint < NUM_PIXELS && rightPoint != leftPoint) {
      uint32_t edgeColor = strip.ColorHSV((hue * 256) % 65536, 255, 255);
      strip.setPixelColor(rightPoint, edgeColor);
    }
  }

  strip.show();

  // Move the circle vertically through the strip
  verticalPos += speed;

  // Reset when circle fully exits below the strip
  if (verticalPos > radius) {
    verticalPos = -radius;
    pauseFrames = PAUSE_DURATION;  // Start pause before next cycle
  }
}

// Collision effect - two soft white crawlers collide and create firework sparks
// Crawlers start at opposite ends, move toward each other, and explode on collision
void collision() {
  // State machine
  enum State { APPROACHING, COLLIDING, PAUSED, SPARKLING, RESET };
  static State state = APPROACHING;

  static float crawler1Pos = 0;
  static float crawler2Pos = NUM_PIXELS - 1;
  const float crawlerSpeed = 0.8;
  const float waveWidth = 6.0;

  static int collisionPoint = 0;
  static int pauseFrames = 0;
  static int resetDelay = 0;

  // Spark particles
  static Orb sparks[35];
  static bool sparksInitialized = false;

  if (!sparksInitialized) {
    for (int i = 0; i < 35; i++) {
      sparks[i].active = false;
    }
    sparksInitialized = true;
  }

  // Clear all LEDs
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, 0);
  }

  // State machine
  switch(state) {
    case APPROACHING:
      {
        // Move crawlers
        crawler1Pos += crawlerSpeed;
        crawler2Pos -= crawlerSpeed;

        // Calculate brightness for both crawlers
        float brightness[NUM_PIXELS];
        for (int i = 0; i < NUM_PIXELS; i++) {
          brightness[i] = 0;
        }

        // Add crawler 1 brightness
        for (int i = 0; i < NUM_PIXELS; i++) {
          float distance = abs((float)i - crawler1Pos);
          if (distance <= waveWidth) {
            float b = cos(distance / waveWidth * 3.14159 / 2);
            if (b > 0) brightness[i] += b;
          }
        }

        // Add crawler 2 brightness
        for (int i = 0; i < NUM_PIXELS; i++) {
          float distance = abs((float)i - crawler2Pos);
          if (distance <= waveWidth) {
            float b = cos(distance / waveWidth * 3.14159 / 2);
            if (b > 0) brightness[i] += b;
          }
        }

        // Render all pixels
        for (int i = 0; i < NUM_PIXELS; i++) {
          if (brightness[i] > 0) {
            float b = min(1.0f, brightness[i]);
            int r = 255 * b;
            int g = 240 * b;
            int blue = 200 * b;
            strip.setPixelColor(i, strip.Color(r, g, blue));
          }
        }

        // Check for collision when fronts touch
        if (crawler1Pos + waveWidth >= crawler2Pos - waveWidth) {
          state = COLLIDING;
          collisionPoint = (int)((crawler1Pos + crawler2Pos) / 2);
        }
      }
      break;

    case COLLIDING:
      {
        // Continue moving forward
        crawler1Pos += crawlerSpeed;
        crawler2Pos -= crawlerSpeed;

        // Calculate brightness with masking
        float brightness[NUM_PIXELS];
        for (int i = 0; i < NUM_PIXELS; i++) {
          brightness[i] = 0;
        }

        // Crawler 1: only draw LEDs up to and including collision point
        for (int i = 0; i < NUM_PIXELS; i++) {
          if (i <= collisionPoint) {
            float distance = abs((float)i - crawler1Pos);
            if (distance <= waveWidth) {
              float b = cos(distance / waveWidth * 3.14159 / 2);
              if (b > 0) brightness[i] += b;
            }
          }
        }

        // Crawler 2: only draw LEDs from collision point onwards
        for (int i = 0; i < NUM_PIXELS; i++) {
          if (i >= collisionPoint) {
            float distance = abs((float)i - crawler2Pos);
            if (distance <= waveWidth) {
              float b = cos(distance / waveWidth * 3.14159 / 2);
              if (b > 0) brightness[i] += b;
            }
          }
        }

        // Render all pixels
        for (int i = 0; i < NUM_PIXELS; i++) {
          if (brightness[i] > 0) {
            float b = min(1.0f, brightness[i]);
            int r = 255 * b;
            int g = 240 * b;
            int blue = 200 * b;
            strip.setPixelColor(i, strip.Color(r, g, blue));
          }
        }

        // Check if both tails have passed through collision point
        if (crawler1Pos - waveWidth >= collisionPoint &&
            crawler2Pos + waveWidth <= collisionPoint) {
          state = PAUSED;
          pauseFrames = 2;
        }
      }
      break;

    case PAUSED:
      {
        pauseFrames--;
        if (pauseFrames <= 0) {
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
          if (sparks[i].position < 0 || sparks[i].position >= NUM_PIXELS) {
            sparks[i].active = false;
            continue;
          }

          // Calculate brightness with flickering
          float lifecycle = sparks[i].age / sparks[i].lifetime;
          float brightness = 1.0 - lifecycle;
          brightness = brightness * brightness;
          float flicker = 0.7 + (random(0, 60) / 100.0);
          brightness = brightness * flicker;
          brightness = min(1.0f, max(0.0f, brightness));

          // Color variants
          int colorChoice = i % 7;
          int r, g, b;
          switch(colorChoice) {
            case 0: r = 255 * brightness; g = 255 * brightness; b = 30 * brightness; break;
            case 1: r = 255 * brightness; g = 240 * brightness; b = 200 * brightness; break;
            case 2: r = 255 * brightness; g = 255 * brightness; b = 120 * brightness; break;
            case 3: r = 255 * brightness; g = 255 * brightness; b = 255 * brightness; break;
            case 4: r = 255 * brightness; g = 220 * brightness; b = 80 * brightness; break;
            case 5: r = 255 * brightness; g = 200 * brightness; b = 100 * brightness; break;
            default: r = 255 * brightness; g = 250 * brightness; b = 180 * brightness; break;
          }

          int pixelPos = (int)sparks[i].position;
          if (pixelPos >= 0 && pixelPos < NUM_PIXELS) {
            uint32_t existing = strip.getPixelColor(pixelPos);
            uint8_t er = (existing >> 16) & 0xFF;
            uint8_t eg = (existing >> 8) & 0xFF;
            uint8_t eb = existing & 0xFF;
            strip.setPixelColor(pixelPos, strip.Color(
              min(255, (int)er + r),
              min(255, (int)eg + g),
              min(255, (int)eb + b)
            ));
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
          crawler1Pos = 0;
          crawler2Pos = NUM_PIXELS - 1;
          state = APPROACHING;
        }
      }
      break;
  }

  strip.show();
}
