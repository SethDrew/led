/*
 * Y-Tree Visualization with 3rd Branch
 * Main strip (pin 13): Trunk + 2 branches at Y-split
 * Branch C (pin 12): 5 LEDs branching from trunk at position 25
 *
 * Structure:
 * - Trunk: LEDs 0-38 on pin 13
 * - Branch A: LEDs 38-61 on pin 13 (Y-split at 38)
 * - Branch B: LEDs 62-91 on pin 13 (Y-split at 38)
 * - Branch C: 5 LEDs on pin 12 (branches at trunk position 25)
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN_MAIN 13
#define LED_PIN_BRANCH_C 12

// Tree structure
#define TRUNK_START 0
#define TRUNK_END 38
#define BRANCH_A_START 38
#define BRANCH_A_END 61
#define BRANCH_B_START 62
#define BRANCH_B_END 91  // Branch 2 is 30 LEDs long (62-91)
#define BRANCH_C_LEDS 6  // Branch C on separate pin
#define BRANCH_C_SPLIT 25  // Branches from trunk at position 25

#define SPLIT_POINT 38  // Where the main Y occurs

#define TOTAL_LEDS_MAIN 92  // LEDs on main strip (pin 13)
#define TOTAL_TREE_NODES 98  // Total nodes in tree structure (92 + 6)

Adafruit_NeoPixel stripMain(TOTAL_LEDS_MAIN, LED_PIN_MAIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripBranchC(BRANCH_C_LEDS, LED_PIN_BRANCH_C, NEO_GRB + NEO_KHZ800);

// Depth structure for tree visualization
#define MAX_DEPTH 67  // SPLIT_POINT (38) + longest branch (Branch B: 30 LEDs = 29 additional depth)
struct LEDNode {
  Adafruit_NeoPixel* strip;  // Which strip this LED is on
  int ledIndex;               // Index on that strip
  int depth;                  // Depth in tree for animation
};

LEDNode treeStructure[TOTAL_TREE_NODES];

// Colors
uint32_t barkBrown;
uint32_t leafGreen;
uint32_t springGreen;

int32_t direction = 1;

void initializeTreeStructure() {
  int nodeIndex = 0;

  // TRUNK (depth 0 to 38)
  for (int i = TRUNK_START; i <= TRUNK_END; i++) {
    treeStructure[nodeIndex].strip = &stripMain;
    treeStructure[nodeIndex].ledIndex = i;
    treeStructure[nodeIndex].depth = i;
    nodeIndex++;
  }

  // BRANCH A - main Y-split left branch (depth 38 onwards)
  int branchALength = BRANCH_A_END - BRANCH_A_START + 1;
  for (int i = 0; i < branchALength; i++) {
    treeStructure[nodeIndex].strip = &stripMain;
    treeStructure[nodeIndex].ledIndex = BRANCH_A_START + i;
    treeStructure[nodeIndex].depth = SPLIT_POINT + i;
    nodeIndex++;
  }

  // BRANCH B - main Y-split right branch (also depth 38 onwards, parallel to Branch A)
  int branchBLength = BRANCH_B_END - BRANCH_B_START + 1;
  for (int i = 0; i < branchBLength; i++) {
    treeStructure[nodeIndex].strip = &stripMain;
    treeStructure[nodeIndex].ledIndex = BRANCH_B_START + i;
    treeStructure[nodeIndex].depth = SPLIT_POINT + i;
    nodeIndex++;
  }

  // BRANCH C - side branch on pin 12 (branches at trunk position 25)
  for (int i = 0; i < BRANCH_C_LEDS; i++) {
    treeStructure[nodeIndex].strip = &stripBranchC;
    treeStructure[nodeIndex].ledIndex = i;
    treeStructure[nodeIndex].depth = BRANCH_C_SPLIT + i;
    nodeIndex++;
  }

  Serial.print("Initialized tree with ");
  Serial.print(nodeIndex);
  Serial.println(" total LEDs");
  Serial.print("  Trunk: ");
  Serial.print(TRUNK_END - TRUNK_START + 1);
  Serial.println(" LEDs");
  Serial.print("  Branch A: ");
  Serial.print(branchALength);
  Serial.println(" LEDs");
  Serial.print("  Branch B: ");
  Serial.print(branchBLength);
  Serial.println(" LEDs");
  Serial.print("  Branch C: ");
  Serial.print(BRANCH_C_LEDS);
  Serial.print(" LEDs (splits at trunk pos ");
  Serial.print(BRANCH_C_SPLIT);
  Serial.println(")");
}

void setup() {
  Serial.begin(115200);
  Serial.println("Y-Tree LED Visualization");
  Serial.print("Split point at LED: ");
  Serial.println(SPLIT_POINT);

  stripMain.begin();
  stripMain.setBrightness(128);  // 50% brightness
  stripMain.clear();
  stripMain.show();

  stripBranchC.begin();
  stripBranchC.setBrightness(128);
  stripBranchC.clear();
  stripBranchC.show();

  // Initialize tree structure
  initializeTreeStructure();

  // Startup test - flash each branch
  Serial.println("Startup test...");

  // Flash trunk brown
  for (int i = TRUNK_START; i <= TRUNK_END; i++) {
    stripMain.setPixelColor(i, barkBrown);
  }
  stripMain.show();
  delay(1000);

  // Flash Branch A green
  stripMain.clear();
  for (int i = BRANCH_A_START; i <= BRANCH_A_END; i++) {
    stripMain.setPixelColor(i, leafGreen);
  }
  stripMain.show();
  delay(1000);

  // Flash Branch B green
  stripMain.clear();
  for (int i = BRANCH_B_START; i <= BRANCH_B_END; i++) {
    stripMain.setPixelColor(i, leafGreen);
  }
  stripMain.show();
  delay(1000);

  // Flash Branch C green
  stripMain.clear();
  for (int i = 0; i < BRANCH_C_LEDS; i++) {
    stripBranchC.setPixelColor(i, leafGreen);
  }
  stripBranchC.show();
  delay(1000);

  stripMain.clear();
  stripMain.show();
  stripBranchC.clear();
  stripBranchC.show();

  Serial.println("Starting animation...");

  // Define colors
  barkBrown = stripMain.Color(139, 69, 19);       // Brown for trunk
  leafGreen = stripMain.Color(34, 139, 34);        // Forest green
  springGreen = stripMain.Color(144, 238, 144);    // Light green
}

void loop() {
  // Green wave crawl at 2x speed (40ms delay)
  colorCrawl(40);
}

// Slow crawl of green wave (no brown background)
void colorCrawl(int wait) {
  static int offset = 0;

  // Green color for wave
  int greenR = 34, greenG = 139, greenB = 34;

  // Wave width
  float waveWidth = 5.0;

  // Apply wave to all LEDs based on their depth
  // LEDs at the same depth will light simultaneously (branch splits!)
  for (int i = 0; i < TOTAL_TREE_NODES; i++) {
    LEDNode& node = treeStructure[i];

    float distance;

    // Special handling for Branch C - always animate 0→5 (outward)
    if (node.strip == &stripBranchC) {
      if (direction > 0) {
        // Upward wave: normal depth calculation
        distance = abs(node.depth - offset);
      } else {
        // Downward wave: flip the depth so it still goes 0→5, delayed by 1 LED
        // Branch C depths are 25-30, so flip them: 30→25, 29→26, etc.
        int flippedDepth = (BRANCH_C_SPLIT * 2 + BRANCH_C_LEDS - 1) - node.depth;
        distance = abs(flippedDepth - (offset - 1));  // Delay by 1 LED on down motion
      }
    } else {
      // Normal branches - use regular depth
      distance = abs(node.depth - offset);
    }

    // Calculate brightness for green wave (0 = off, 1 = full green)
    float brightness = 0.0;
    if (distance <= waveWidth) {
      brightness = (cos(distance / waveWidth * 3.14159) + 1.0) / 2.0;
    }

    // Green wave on black background
    int r = greenR * brightness;
    int g = greenG * brightness;
    int b = greenB * brightness;

    // Set color on physical LED using the correct strip
    node.strip->setPixelColor(node.ledIndex, node.strip->Color(r, g, b));
  }

  // Show both strips
  stripMain.show();
  stripBranchC.show();

  // Move wave forward
  offset += 1 * direction;

  // Reverse direction at ends
  if (offset > MAX_DEPTH + (int)waveWidth || offset < (int)waveWidth * -1) {
    delay(1000);
    direction = direction * -1;
  }

  delay(wait);
}
