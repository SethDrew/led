/*
 * New Tree Visualization
 * Pin D11: Single strip with trunk and branches
 *
 * Structure:
 * - Trunk: LEDs 0-69 (depth 0-69)
 * - LED 70: Main path continuation (depth 70)
 * - Branch A: LEDs 71-72 (splits at depth 43)
 * - Branch B: LEDs 73-98 (splits at depth 43)
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 11

// Tree structure
#define TRUNK_START 0
#define TRUNK_END 69
#define MAIN_CONTINUATION 70
#define BRANCH_A_START 71
#define BRANCH_A_END 72
#define BRANCH_B_START 73
#define BRANCH_B_END 98

#define SPLIT_POINT 43  // Where branches A and B split off

#define TOTAL_LEDS 99  // 0-98
#define MAX_DEPTH 70   // Deepest point (LED 70)

Adafruit_NeoPixel strip(TOTAL_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

// Depth structure for tree visualization
struct LEDNode {
  int ledIndex;   // Index on strip
  int depth;      // Depth in tree for animation
};

LEDNode treeStructure[TOTAL_LEDS];

// Colors
uint32_t leafGreen;

int32_t direction = 1;

void initializeTreeStructure() {
  int nodeIndex = 0;

  // TRUNK (depth 0 to 69)
  for (int i = TRUNK_START; i <= TRUNK_END; i++) {
    treeStructure[nodeIndex].ledIndex = i;
    treeStructure[nodeIndex].depth = i;
    nodeIndex++;
  }

  // MAIN CONTINUATION (depth 70)
  treeStructure[nodeIndex].ledIndex = MAIN_CONTINUATION;
  treeStructure[nodeIndex].depth = 70;
  nodeIndex++;

  // BRANCH A - stumpy branch (splits at depth 43)
  int branchALength = BRANCH_A_END - BRANCH_A_START + 1;
  for (int i = 0; i < branchALength; i++) {
    treeStructure[nodeIndex].ledIndex = BRANCH_A_START + i;
    treeStructure[nodeIndex].depth = SPLIT_POINT + i;
    nodeIndex++;
  }

  // BRANCH B - longer branch (splits at depth 43)
  int branchBLength = BRANCH_B_END - BRANCH_B_START + 1;
  for (int i = 0; i < branchBLength; i++) {
    treeStructure[nodeIndex].ledIndex = BRANCH_B_START + i;
    treeStructure[nodeIndex].depth = SPLIT_POINT + i;
    nodeIndex++;
  }

  Serial.print("Initialized tree with ");
  Serial.print(nodeIndex);
  Serial.println(" total LEDs");
  Serial.print("  Trunk: ");
  Serial.print(TRUNK_END - TRUNK_START + 1);
  Serial.println(" LEDs (depths 0-69)");
  Serial.print("  Main continuation: 1 LED (depth 70)");
  Serial.println();
  Serial.print("  Branch A: ");
  Serial.print(branchALength);
  Serial.print(" LEDs (splits at depth ");
  Serial.print(SPLIT_POINT);
  Serial.println(")");
  Serial.print("  Branch B: ");
  Serial.print(branchBLength);
  Serial.print(" LEDs (splits at depth ");
  Serial.print(SPLIT_POINT);
  Serial.println(")");
}

void setup() {
  Serial.begin(115200);
  Serial.println("New Tree LED Visualization");
  Serial.print("Split point at depth: ");
  Serial.println(SPLIT_POINT);

  strip.begin();
  strip.setBrightness(128);  // 50% brightness
  strip.clear();
  strip.show();

  // Initialize tree structure
  initializeTreeStructure();

  // Startup test - flash each section
  Serial.println("Startup test...");

  // Flash trunk green
  for (int i = TRUNK_START; i <= TRUNK_END; i++) {
    strip.setPixelColor(i, strip.Color(0, 255, 0));
  }
  strip.show();
  delay(1000);

  // Flash main continuation
  strip.clear();
  strip.setPixelColor(MAIN_CONTINUATION, strip.Color(0, 255, 0));
  strip.show();
  delay(1000);

  // Flash Branch A
  strip.clear();
  for (int i = BRANCH_A_START; i <= BRANCH_A_END; i++) {
    strip.setPixelColor(i, strip.Color(0, 255, 0));
  }
  strip.show();
  delay(1000);

  // Flash Branch B
  strip.clear();
  for (int i = BRANCH_B_START; i <= BRANCH_B_END; i++) {
    strip.setPixelColor(i, strip.Color(0, 255, 0));
  }
  strip.show();
  delay(1000);

  strip.clear();
  strip.show();

  Serial.println("Starting animation...");

  // Define colors
  leafGreen = strip.Color(34, 139, 34);  // Forest green
}

void loop() {
  // Green wave crawl at 2x speed (40ms delay)
  colorCrawl(40);
}

// Green wave animation
void colorCrawl(int wait) {
  static int offset = 0;

  // Green color for wave
  int greenR = 34, greenG = 139, greenB = 34;

  // Wave width
  float waveWidth = 5.0;

  // Apply wave to all LEDs based on their depth
  for (int i = 0; i < TOTAL_LEDS; i++) {
    LEDNode& node = treeStructure[i];

    float distance = abs(node.depth - offset);

    // Calculate brightness for green wave
    float brightness = 0.0;
    if (distance <= waveWidth) {
      brightness = (cos(distance / waveWidth * 3.14159) + 1.0) / 2.0;
    }

    // Green wave on black background
    int r = greenR * brightness;
    int g = greenG * brightness;
    int b = greenB * brightness;

    strip.setPixelColor(node.ledIndex, strip.Color(r, g, b));
  }

  strip.show();

  // Move wave forward
  offset += 1 * direction;

  // Reverse direction at ends
  if (offset > MAX_DEPTH + (int)waveWidth || offset < (int)waveWidth * -1) {
    delay(1000);
    direction = direction * -1;
  }

  delay(wait);
}
