/*
 * Brown Color Comparison
 * Bottom half: Saddle Brown (139, 69, 19)
 * Top half: Burnt Orange (180, 90, 20)
 */

#include "TreeTopology.h"

Tree tree;

// Two brown options
const uint8_t SADDLE_BROWN_R = 139;
const uint8_t SADDLE_BROWN_G = 69;
const uint8_t SADDLE_BROWN_B = 19;

const uint8_t BURNT_ORANGE_R = 180;
const uint8_t BURNT_ORANGE_G = 90;
const uint8_t BURNT_ORANGE_B = 20;

const uint8_t MID_DEPTH = MAX_DEPTH / 2;  // 35

void setup() {
  Serial.begin(115200);
  Serial.println("Brown Comparison Test");
  Serial.print("Bottom half (depth 0-");
  Serial.print(MID_DEPTH);
  Serial.println("): Saddle Brown RGB(139, 69, 19)");
  Serial.print("Top half (depth ");
  Serial.print(MID_DEPTH + 1);
  Serial.print("-");
  Serial.print(MAX_DEPTH);
  Serial.println("): Burnt Orange RGB(180, 90, 20)");

  tree.begin();

  // Render two-color gradient
  for (uint8_t i = 0; i < tree.getNumLEDs(); i++) {
    uint8_t depth = tree.getDepth(i);

    if (depth <= MID_DEPTH) {
      // Bottom half: Saddle Brown
      tree.setNodeColor(i, SADDLE_BROWN_R, SADDLE_BROWN_G, SADDLE_BROWN_B);
    } else {
      // Top half: Burnt Orange
      tree.setNodeColor(i, BURNT_ORANGE_R, BURNT_ORANGE_G, BURNT_ORANGE_B);
    }
  }
  tree.show();

  Serial.println("Comparison displayed!");
}

void loop() {
  // Static display
  delay(1000);
}
