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
  nebulaAnimation(); delay(20);
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

// ===== Helper Functions for Nebula Animation =====

// Update background pulsating waves combining breathing and spatial effects
void updateBackgroundWaves(float* ledValues_background, unsigned long frameCount) {
  const float BREATH_FREQUENCY = 0.035;
  const float BREATH_CENTER = 0.08;
  const float BREATH_AMPLITUDE = 0.07;
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
        orbs[i].lifetime = 40 + random(160);  // 40-200 frames lifetime
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
        orbs[i].lifetime = 40 + random(160);  // 40-200 frames lifetime
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
