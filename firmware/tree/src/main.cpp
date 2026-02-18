/*
 * TREE EFFECTS - MODULAR SYSTEM (Memory Optimized)
 *
 * Renders directly to tree strips (no buffer overhead)
 * Each animation runs one effect at a time
 *
 * Available Effects:
 *   - DepthWaveForeground: Wave flowing up/down tree
 *   - SapFlowForeground: Particles rising through tree
 *
 * Modes:
 *   - STREAMING_MODE: Receive RGB data from serial (see streaming_receiver.cpp)
 *   - Normal mode: Run preset animations
 */

#include "TreeTopology.h"

#ifndef STREAMING_MODE

#include "TreeEffect.h"
#include "foregrounds/DepthWaveForeground.h"
#include "foregrounds/SapFlowForeground.h"

// Global tree instance
Tree tree;

// Standalone sculpture on pin 12 (not part of tree topology)
#ifndef SCULPTURE_PIN
#define SCULPTURE_PIN 13
#endif
#ifndef SCULPTURE_LEDS
#define SCULPTURE_LEDS 120  // 5m FCOB WS2811: 24 ICs/m × 5m = 120 addressable pixels (36 LEDs per IC)
#endif
// WS2811 FCOB 12V strip
Adafruit_NeoPixel sculpture(SCULPTURE_LEDS, SCULPTURE_PIN, NEO_GRB + NEO_KHZ800);

// ===== ANIMATIONS =====

// Animation 1: Classic green wave
void classicWaveAnimation() {
  static DepthWaveForeground wave(&tree, 34, 139, 34, 5.0);  // Forest green
  wave.update();
  wave.render();
}

// Animation 2: Forest green sap flow
void sapFlowAnimation() {
  static SapFlowForeground sap(&tree, 34, 139, 34, 8);  // Forest green, moderate spawn rate
  sap.update();
  sap.render();
}

// Animation 3: Blue wave
void blueWaveAnimation() {
  static DepthWaveForeground wave(&tree, 0, 100, 255, 6.0);  // Blue
  wave.update();
  wave.render();
}

// Animation 4: Orange wave
void orangeWaveAnimation() {
  static DepthWaveForeground wave(&tree, 255, 140, 0, 5.0);  // Orange
  wave.update();
  wave.render();
}

// Animation 5: White sap
void whiteSapAnimation() {
  static SapFlowForeground sap(&tree, 255, 255, 255, 6);  // White
  sap.update();
  sap.render();
}

// Animation 6: Solid white - all LEDs on
void solidWhiteAnimation() {
  static bool initialized = false;
  if (!initialized) {
    for (uint8_t i = 0; i < tree.getNumLEDs(); i++) {
      tree.setNodeColor(i, 255, 255, 255);
    }
    tree.show();
    initialized = true;
  }
}

// ===== MAIN =====

void setup() {
  Serial.begin(115200);
  Serial.println("=================================");
  Serial.println("LED TREE");
  Serial.println("=================================");
  Serial.println("Tree Effects - Memory Optimized");
  Serial.print("Total LEDs: ");
  Serial.println(tree.getNumLEDs());

  tree.begin();

  // Sculpture: solid red on pin 12
  sculpture.begin();
  sculpture.setBrightness(8);  // Low brightness for testing
  for (int i = 0; i < SCULPTURE_LEDS; i++) {
    sculpture.setPixelColor(i, 255, 0, 0);
  }
  sculpture.show();
  Serial.print("Sculpture on pin ");
  Serial.print(SCULPTURE_PIN);
  Serial.println(": all red");

  Serial.println("Ready!");
}

void loop() {
  // Uncomment one animation:

  // classicWaveAnimation();   // Green wave
  sapFlowAnimation();          // Green sap flow ← ACTIVE
  // blueWaveAnimation();      // Blue wave
  // orangeWaveAnimation();    // Orange wave
  // whiteSapAnimation();      // White sap
  // solidWhiteAnimation();    // Solid white - all LEDs

  delay(40);  // ~25 FPS
}

#endif  // !STREAMING_MODE
