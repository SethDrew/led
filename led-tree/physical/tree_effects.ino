/*
 * TREE EFFECTS - MODULAR SYSTEM (Memory Optimized)
 *
 * Renders directly to tree strips (no buffer overhead)
 * Each animation runs one effect at a time
 *
 * Available Effects:
 *   - DepthWaveForeground: Wave flowing up/down tree
 *   - SapFlowForeground: Particles rising through tree
 */

#include "TreeTopology.h"
#include "src/TreeEffect.h"
#include "src/foregrounds/DepthWaveForeground.h"
#include "src/foregrounds/SapFlowForeground.h"

// Global tree instance
Tree tree;

// ===== ANIMATIONS =====

// Animation 1: Classic green wave
void classicWaveAnimation() {
  static DepthWaveForeground wave(&tree, 34, 139, 34, 5.0);  // Forest green
  wave.update();
  wave.render();
}

// Animation 2: Bright green sap flow
void sapFlowAnimation() {
  static SapFlowForeground sap(&tree, 100, 255, 100, 8);  // Bright green
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

// ===== MAIN =====

void setup() {
  Serial.begin(115200);
  Serial.println("Tree Effects - Memory Optimized");
  Serial.print("Total LEDs: ");
  Serial.println(tree.getNumLEDs());

  tree.begin();
  Serial.println("Ready!");
}

void loop() {
  // Uncomment one animation:

  classicWaveAnimation();      // Green wave ← ACTIVE
  // sapFlowAnimation();       // Green sap flow
  // blueWaveAnimation();      // Blue wave
  // orangeWaveAnimation();    // Orange wave
  // whiteSapAnimation();      // White sap

  delay(40);  // ~25 FPS
}
