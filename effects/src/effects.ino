/*
 * LED EFFECTS - MODULAR HEADER-BASED SYSTEM
 *
 * This demonstrates a clean, modular LED effects library where:
 * - Each effect is defined in its own header file
 * - You only include the effects you need
 * - Effects are composited using a shared buffer system
 * - Backgrounds and foregrounds can be mixed freely
 *
 * ARCHITECTURE:
 *   Background Effects (REPLACE blend):
 *     - NebulaBackground          : Breathing waves with color shifts
 *     - SolidColorBackground      : Static solid color
 *     - PulsingColorBackground    : Pulsing solid color
 *
 *   Foreground Effects (ADD blend):
 *     - CrawlingStarsForeground   : Glowing orbs that drift
 *     - SparksForeground          : Random spark explosions
 *     - CollisionForeground       : Crawlers that collide and explode
 *     - RainbowCircleForeground   : Rainbow circle passing through
 *     - EnhancedCrawlForeground   : Smooth wave with color modes
 *     - FragmentationForeground   : White crawler decomposes to RGB
 *     - DriftingDecayForeground   : White crawlers drift and decay
 *
 * MEMORY OPTIMIZATION:
 *   - Only include headers you need
 *   - Effects are instantiated only when used
 *   - Static allocation for active effects
 *
 * TO ADD A NEW EFFECT:
 *   1. Create YourEffect.h inheriting from BackgroundEffect or ForegroundEffect
 *   2. Implement update() and render() methods
 *   3. Include the header below
 *   4. Instantiate and use in your animation function
 */

#include <Adafruit_NeoPixel.h>

// ===== INCLUDE ONLY THE EFFECTS YOU NEED =====

#include "Effect.h"
#include "NebulaBackground.h"
#include "SolidColorBackground.h"
#include "PulsingColorBackground.h"
#include "CrawlingStarsForeground.h"
#include "SparksForeground.h"
#include "CollisionForeground.h"
#include "RainbowCircleForeground.h"
#include "EnhancedCrawlForeground.h"
#include "FragmentationForeground.h"
#include "DriftingDecayForeground.h"

// ===== HARDWARE CONFIGURATION =====

#define LED_PIN 12

// NUM_PIXELS can be set via build flags (see platformio.ini)
// Default to 25 for Wokwi simulation if not defined
#ifndef NUM_PIXELS
#define NUM_PIXELS 25
#endif

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

// ===== RENDERING PARAMETERS =====

// GLOBAL_BRIGHTNESS_PERCENT can be set via build flags (see platformio.ini)
// Default to 100% for Wokwi simulation if not defined
#ifndef GLOBAL_BRIGHTNESS_PERCENT
#define GLOBAL_BRIGHTNESS_PERCENT 100
#endif

const float GLOBAL_BRIGHTNESS = GLOBAL_BRIGHTNESS_PERCENT / 100.0;  // 0.0 to 1.0
const int MIN_LED_VALUE = 5;           // Minimum per channel for WS2812B

// ===== SHARED BUFFER SYSTEM =====

static uint8_t pixelBuffer[NUM_PIXELS][3];

// ===== BUFFER UTILITIES =====

void clearBuffer(uint8_t buffer[][3]) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    buffer[i][0] = 0;
    buffer[i][1] = 0;
    buffer[i][2] = 0;
  }
}

void applyBrightness(uint8_t buffer[][3], float brightness) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    buffer[i][0] = buffer[i][0] * brightness;
    buffer[i][1] = buffer[i][1] * brightness;
    buffer[i][2] = buffer[i][2] * brightness;
  }
}

void applyMinThreshold(uint8_t buffer[][3], int minValue) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    for (int c = 0; c < 3; c++) {
      if (buffer[i][c] > 0 && buffer[i][c] < minValue) {
        buffer[i][c] = minValue;
      }
    }
  }
}

void pushToStrip(uint8_t buffer[][3]) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, strip.Color(buffer[i][0], buffer[i][1], buffer[i][2]));
  }
  strip.show();
}

// ===== COMPOSITOR =====

void renderLayeredFrame(BackgroundEffect* bg, ForegroundEffect* fg) {
  clearBuffer(pixelBuffer);
  if (bg) bg->render(pixelBuffer, REPLACE);
  if (fg) fg->render(pixelBuffer, ADD);
  applyBrightness(pixelBuffer, GLOBAL_BRIGHTNESS);
  applyMinThreshold(pixelBuffer, MIN_LED_VALUE);
  pushToStrip(pixelBuffer);
}

// ===== EXAMPLE ANIMATIONS =====

// Example 1: Nebula background with crawling stars
void nebulaStarsAnimation() {
  static NebulaBackground bg(NUM_PIXELS);
  static CrawlingStarsForeground fg(NUM_PIXELS, 5, 7.0, 0.3);

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// Example 2: Nebula background with random sparks
void nebulaSparksAnimation() {
  static NebulaBackground bg(NUM_PIXELS);
  static SparksForeground fg(NUM_PIXELS);

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// Example 3: Nebula background with collision sequence
void nebulaCollisionAnimation() {
  static NebulaBackground bg(NUM_PIXELS);
  static CollisionForeground fg(NUM_PIXELS);

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// Example 4: Solid purple background with rainbow circle
void purpleRainbowAnimation() {
  static SolidColorBackground bg(NUM_PIXELS, 80, 0, 120, 0.3);  // Purple at 30%
  static RainbowCircleForeground fg(NUM_PIXELS, 10.0, 0.1);

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// Example 5: Pulsing blue background with enhanced crawl
void pulsingCrawlAnimation() {
  static PulsingColorBackground bg(NUM_PIXELS, 0, 50, 150, 0.1, 0.4, 0.03);  // Blue pulse
  static EnhancedCrawlForeground fg(NUM_PIXELS, 8.0, 3, 0, 0.0);  // Warm white wave

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// Example 6: Black background with fragmentation effect
void fragmentationAnimation() {
  static SolidColorBackground bg(NUM_PIXELS, 0, 0, 0, 0.0);  // Black
  static FragmentationForeground fg(NUM_PIXELS);

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// Example 7: Black background with drifting decay
void driftingDecayAnimation() {
  static SolidColorBackground bg(NUM_PIXELS, 0, 0, 0, 0.0);  // Black
  static DriftingDecayForeground fg(NUM_PIXELS);

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// Example 8: No background, just rainbow circle
void rainbowCircleOnlyAnimation() {
  static RainbowCircleForeground fg(NUM_PIXELS, 12.0, 0.08);

  fg.update();
  renderLayeredFrame(NULL, &fg);
}

// Example 9: Custom - pulsing red with sparks
void redSparksAnimation() {
  static PulsingColorBackground bg(NUM_PIXELS, 50, 0, 0, 0.05, 0.15, 0.02);  // Dark red pulse
  static SparksForeground fg(NUM_PIXELS);

  bg.update();
  fg.update();
  renderLayeredFrame(&bg, &fg);
}

// ===== MAIN =====

void setup() {
  Serial.begin(115200);
  Serial.println("LED Effects - Modular Header System");
  Serial.print("NUM_PIXELS: ");
  Serial.println(NUM_PIXELS);

  strip.begin();
  strip.setBrightness(200);
  strip.show();

  Serial.println("Ready!");
}

void loop() {
  // Uncomment one animation to run:

  nebulaStarsAnimation();           // Nebula + stars ← ACTIVE
  // nebulaSparksAnimation();       // Nebula + sparks
  // nebulaCollisionAnimation();    // Nebula + collision
  // purpleRainbowAnimation();      // Purple + rainbow
  // pulsingCrawlAnimation();       // Pulsing blue + crawl
  // fragmentationAnimation();      // Fragmentation
  // driftingDecayAnimation();      // Drifting decay
  // rainbowCircleOnlyAnimation();  // Rainbow circle only
  // fireworksAnimation();          // Fireworks style

  // No delay - run as fast as possible for smooth animations!
}
