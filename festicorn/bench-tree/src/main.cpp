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
  // Time-based defaults captured from prior frame-based version (delay(40), ~21fps).
  static SapFlowForeground sap(&tree, 34, 139, 34);
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
  // ~75% of forest sap rate (old spawnChance=6 vs 8).
  static SapFlowForeground sap(&tree, 255, 255, 255, 7.25f, 1.91f, 12.73f, 1.26f);
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
  // No artificial frame cap. Adafruit_NeoPixel::show() blocks during WS2812
  // transmission (~6ms across 3 strips), giving a natural ~120fps ceiling.
}

#endif  // !STREAMING_MODE
