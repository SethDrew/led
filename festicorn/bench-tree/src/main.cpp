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
#include "foregrounds/SapFlowForeground.h"

// Global tree instance
Tree tree;

// ===== TREE ANIMATION =====

// Forest green sap flow
void sapFlowAnimation() {
  // Time-based defaults captured from prior frame-based version (delay(40), ~21fps).
  static SapFlowForeground sap(&tree, 34, 139, 34);
  sap.update();
  sap.render();
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
