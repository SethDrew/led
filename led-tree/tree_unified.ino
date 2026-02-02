/*
 * Tree Visualization - Unified Topology
 *
 * Single physical tree with 197 LEDs across 3 strips
 * See TreeTopology.h for complete structure documentation
 *
 * Current animation: Depth wave flowing up and down the tree
 */

#include "TreeTopology.h"

Tree tree;

// Animation state
int offset = 0;
int direction = 1;

// Animation parameters
const uint8_t waveColor[] = {34, 139, 34};  // Forest green (R, G, B)
const float waveWidth = 5.0;

void setup() {
  Serial.begin(115200);
  Serial.println("Tree Visualization - Unified Topology");
  Serial.print("Total LEDs: ");
  Serial.println(TOTAL_TREE_LEDS);
  Serial.print("Max depth: ");
  Serial.println(MAX_DEPTH);

  tree.begin();

  Serial.println("Starting animation...");
}

void loop() {
  // Clear all LEDs
  tree.clear();

  // Render wave at current offset depth
  for (int i = 0; i < tree.getNumLEDs(); i++) {
    TreeNode& node = tree.getNode(i);

    // Calculate effective depth for this node (includes animation offset)
    int effectiveDepth = node.depth + node.depthOffset;

    // Calculate distance from wave center
    float distance = abs(effectiveDepth - offset);

    // If within wave width, light up this LED
    if (distance <= waveWidth) {
      tree.setNodeColor(i, waveColor[0], waveColor[1], waveColor[2]);
    }
  }

  // Show the frame
  tree.show();

  // Advance animation
  offset += direction;

  // Reverse direction at ends (with overshoot for smooth transition)
  if (offset > MAX_DEPTH + 10 || offset < -10) {
    delay(1000);  // Pause at reversal
    direction *= -1;
  }

  delay(40);  // ~25 FPS
}
