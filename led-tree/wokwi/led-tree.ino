#include <Adafruit_NeoPixel.h>

// Pin definitions for each branch
#define TRUNK_PIN 6
#define LOWER_LEFT_PIN 7
#define RIGHT_PIN 8
#define UPPER_LEFT_PIN 9
#define RIGHT_TWIG_PIN 10

// LED counts per branch
#define TRUNK_LEDS 10
#define LOWER_LEFT_LEDS 7
#define RIGHT_LEDS 5
#define UPPER_LEFT_LEDS 4
#define RIGHT_TWIG_LEDS 4

// Create NeoPixel objects for each branch
Adafruit_NeoPixel trunk(TRUNK_LEDS, TRUNK_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel lowerLeft(LOWER_LEFT_LEDS, LOWER_LEFT_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel rightBranch(RIGHT_LEDS, RIGHT_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel upperLeft(UPPER_LEFT_LEDS, UPPER_LEFT_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel rightTwig(RIGHT_TWIG_LEDS, RIGHT_TWIG_PIN, NEO_GRB + NEO_KHZ800);

// Depth-based tree structure for branching animations
// Each LED is assigned a depth (distance from trunk base)
// LEDs at the same depth light up simultaneously, creating branch splits

#define MAX_TREE_DEPTH 20  // Maximum depth in the tree

// Structure to hold physical LED with its depth
struct LEDNode {
  Adafruit_NeoPixel* strip;
  int index;
  int depth;
};

// Total number of LEDs in the tree
#define TOTAL_LEDS 30

// Array of all LEDs with their depths
LEDNode treeStructure[TOTAL_LEDS];

// Tree colors
uint32_t barkBrown;
uint32_t leafGreen;
uint32_t springGreen;

int32_t direction = 1; // Direction of color crawl

// Initialize tree structure with depth mapping
void initializeTreeStructure() {
  int nodeIndex = 0;

  // Trunk base to right branch junction (depth 0-6)
  for (int i = 0; i < 7; i++) {
    treeStructure[nodeIndex++] = {&trunk, i, i};
  }

  // RIGHT BRANCH PATH from depth 7
  // Right branch main (depth 7-10)
  for (int i = 0; i < 4; i++) {
    treeStructure[nodeIndex++] = {&rightBranch, i, 7 + i};
  }

  // Right branch splits at depth 10:
  // - Right twig (depth 11-14)
  for (int i = 0; i < 4; i++) {
    treeStructure[nodeIndex++] = {&rightTwig, i, 11 + i};
  }
  // - Right branch tip (depth 11)
  treeStructure[nodeIndex++] = {&rightBranch, 4, 11};

  // TRUNK CONTINUATION PATH from depth 7
  // Trunk continues (depth 7-9)
  for (int i = 0; i < 3; i++) {
    treeStructure[nodeIndex++] = {&trunk, 7 + i, 7 + i};
  }

  // Left branch path from depth 10
  // Left branch main (depth 10-14)
  for (int i = 0; i < 5; i++) {
    treeStructure[nodeIndex++] = {&lowerLeft, i, 10 + i};
  }

  // Left branch splits at depth 14:
  // - Upper left twig (depth 15-18)
  for (int i = 0; i < 4; i++) {
    treeStructure[nodeIndex++] = {&upperLeft, i, 15 + i};
  }
  // - Left branch tip (depth 15-16)
  for (int i = 0; i < 2; i++) {
    treeStructure[nodeIndex++] = {&lowerLeft, 5 + i, 15 + i};
  }

  Serial.print("Initialized tree with ");
  Serial.print(nodeIndex);
  Serial.println(" LEDs");
}

void setup() {
  Serial.begin(115200);
  Serial.println("Asymmetric LED Tree");

  // Initialize all branches
  trunk.begin();
  trunk.setBrightness(200);
  trunk.show();

  lowerLeft.begin();
  lowerLeft.setBrightness(200);
  lowerLeft.show();

  rightBranch.begin();
  rightBranch.setBrightness(200);
  rightBranch.show();

  upperLeft.begin();
  upperLeft.setBrightness(200);
  upperLeft.show();

  rightTwig.begin();
  rightTwig.setBrightness(200);
  rightTwig.show();

  // Initialize tree structure with depth mapping
  initializeTreeStructure();

  // Define colors
  barkBrown = trunk.Color(139, 69, 19);      // Brown for trunk
  leafGreen = trunk.Color(34, 139, 34);       // Forest green
  springGreen = trunk.Color(144, 238, 144);   // Light green
}

void loop() {
  // Slow crawl of brown and green
  colorCrawl(100);
}


// Set brightness for all branches
void setAllBrightness(int brightness) {
  // Trunk stays brown
  for(int i = 0; i < TRUNK_LEDS; i++) {
    uint8_t r = 139 * brightness / 100;
    uint8_t g = 69 * brightness / 100;
    uint8_t b = 19 * brightness / 100;
    trunk.setPixelColor(i, trunk.Color(r, g, b));
  }
  trunk.show();

  // Branches get green with varying brightness
  for(int i = 0; i < LOWER_LEFT_LEDS; i++) {
    uint8_t r = 34 * brightness / 100;
    uint8_t g = 139 * brightness / 100;
    uint8_t b = 34 * brightness / 100;
    lowerLeft.setPixelColor(i, lowerLeft.Color(r, g, b));
  }
  lowerLeft.show();

  for(int i = 0; i < RIGHT_LEDS; i++) {
    rightBranch.setPixelColor(i, rightBranch.Color(34 * brightness / 100, 139 * brightness / 100, 34 * brightness / 100));
  }
  rightBranch.show();

  for(int i = 0; i < UPPER_LEFT_LEDS; i++) {
    upperLeft.setPixelColor(i, upperLeft.Color(144 * brightness / 100, 238 * brightness / 100, 144 * brightness / 100));
  }
  upperLeft.show();

  for(int i = 0; i < RIGHT_TWIG_LEDS; i++) {
    rightTwig.setPixelColor(i, rightTwig.Color(144 * brightness / 100, 238 * brightness / 100, 144 * brightness / 100));
  }
  rightTwig.show();
}

// Clear all LEDs
void clearAll() {
  trunk.clear();
  trunk.show();
  lowerLeft.clear();
  lowerLeft.show();
  rightBranch.clear();
  rightBranch.show();
  upperLeft.clear();
  upperLeft.show();
  rightTwig.clear();
  rightTwig.show();
}

// Slow crawl of brown and green with branching effect
void colorCrawl(int wait) {
  static int offset = 0;  // Start at trunk base

  // Brown and green colors
  int brownR = 139, brownG = 69, brownB = 19;
  int greenR = 34, greenG = 139, greenB = 34;

  // Wave width
  float waveWidth = 3.0;

  // Apply wave to all LEDs based on their depth
  // LEDs at the same depth will light simultaneously (creating branch splits)
  for (int i = 0; i < TOTAL_LEDS; i++) {
    LEDNode& node = treeStructure[i];

    // Calculate distance from wave center based on LED depth
    float distance = abs(node.depth - offset);

    // Calculate blend between brown and green based on position in wave
    float blend = 0.0;
    if (distance <= waveWidth) {
      // Use cosine for smooth transition
      blend = (cos(distance / waveWidth * 3.14159) + 1.0) / 2.0;  // 0 to 1
    }

    // Interpolate between brown and green
    int r = brownR * (1 - blend) + greenR * blend;
    int g = brownG * (1 - blend) + greenG * blend;
    int b = brownB * (1 - blend) + greenB * blend;

    // Set color on physical LED
    node.strip->setPixelColor(node.index, node.strip->Color(r, g, b));
  }

  // Show all strips
  trunk.show();
  lowerLeft.show();
  rightBranch.show();
  upperLeft.show();
  rightTwig.show();

  // Move wave forward
  offset += 1 * direction;

  // Reset when wave completes full cycle through tree depth
  if (offset > MAX_TREE_DEPTH + (int)waveWidth || offset < (int)waveWidth * -1) {
    delay(1000);
    direction = direction * -1;
  }

  delay(wait);
}

// Apply color wave to a single branch
void applyWaveToBranch(Adafruit_NeoPixel &branch, int numLEDs, int offset, float waveWidth,
                       int color1R, int color1G, int color1B, int color2R, int color2G, int color2B) {
  for(int i = 0; i < numLEDs; i++) {
    float distance = abs(i - offset);

    // Calculate blend between brown and green based on position in wave
    float blend = 0.0;
    if (distance <= waveWidth) {
      // Use cosine for smooth transition
      blend = (cos(distance / waveWidth * 3.14159) + 1.0) / 2.0;  // 0 to 1
    }

    // Interpolate between brown and green
    int r = color1R * (1 - blend) + color2R * blend;
    int g = color1G * (1 - blend) + color2G * blend;
    int b = color1B * (1 - blend) + color2B * blend;

    branch.setPixelColor(i, branch.Color(r, g, b));
  }
  branch.show();
}
