/*
 * LED TREE — Standalone Animation
 *
 * Tree: Sap flow animation (3 strips, 197 LEDs)
 *
 * Available Effects:
 *   - DepthWaveForeground: Wave flowing up/down tree
 *   - SapFlowForeground: Particles rising through tree
 */

#include "TreeTopology.h"

#ifndef STREAMING_MODE

#include "TreeEffect.h"
#include "foregrounds/DepthWaveForeground.h"
#include "foregrounds/SapFlowForeground.h"

// Global tree instance
Tree tree;

// ===== TREE ANIMATIONS =====

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
  Serial.print("Tree LEDs: ");
  Serial.println(tree.getNumLEDs());

  tree.begin();

  Serial.println("Tree: sap flow");
  Serial.println("Ready!");
}

void loop() {
  sapFlowAnimation();
  delay(40);  // ~25 FPS
}

#endif  // !STREAMING_MODE
