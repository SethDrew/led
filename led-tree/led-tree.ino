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

// Virtual tree structure - maps continuous index (0-29) to physical strips
// Total virtual LEDs in tree
#define VIRTUAL_TREE_SIZE 30

// Trunk segments (split at right branch junction at position 6 and left branch at position 9)
const int trunkPart1[] = {0, 1, 2, 3, 4, 5, 6};  // Base to right branch (7 LEDs)
const int trunkPart1Size = 7;
const int trunkPart2[] = {16, 17, 18};  // Right branch to left branch junction (3 LEDs)
const int trunkPart2Size = 3;

// Right branch segments (splits at position 3 for right twig)
const int rightBranchPart1[] = {7, 8, 9, 10};  // Junction to right twig split (4 LEDs)
const int rightBranchPart1Size = 4;
const int rightBranchPart2[] = {15};  // After right twig to tip (1 LED)
const int rightBranchPart2Size = 1;

// Right twig (connects at rightBranch[3])
const int rightTwig_virtual[] = {11, 12, 13, 14};  // 4 LEDs
const int rightTwigSize_virtual = 4;

// Left branch segments (splits at position 4 for upper left twig)
const int leftBranchPart1[] = {19, 20, 21, 22, 23};  // Junction to upper left split (5 LEDs)
const int leftBranchPart1Size = 5;
const int leftBranchPart2[] = {28, 29};  // After upper left twig to tip (2 LEDs)
const int leftBranchPart2Size = 2;

// Upper left twig (connects at leftBranch[4])
const int upperLeftTwig_virtual[] = {24, 25, 26, 27};  // 4 LEDs
const int upperLeftTwigSize_virtual = 4;

// Structure to hold physical strip mapping
struct PhysicalLED {
  Adafruit_NeoPixel* strip;
  int index;
};

// Tree colors
uint32_t barkBrown;
uint32_t leafGreen;
uint32_t springGreen;

int32_t direction = 1; // Direction of color crawl

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

// Map virtual tree index to physical strip and LED index
PhysicalLED getPhysicalLED(int virtualIndex) {
  PhysicalLED led;

  // Trunk part 1: virtual indices 0-6 -> trunk[0-6]
  if (virtualIndex >= 0 && virtualIndex <= 6) {
    led.strip = &trunk;
    led.index = virtualIndex;
    return led;
  }

  // Right branch part 1: virtual indices 7-10 -> rightBranch[0-3]
  if (virtualIndex >= 7 && virtualIndex <= 10) {
    led.strip = &rightBranch;
    led.index = virtualIndex - 7;
    return led;
  }

  // Right twig: virtual indices 11-14 -> rightTwig[0-3]
  if (virtualIndex >= 11 && virtualIndex <= 14) {
    led.strip = &rightTwig;
    led.index = virtualIndex - 11;
    return led;
  }

  // Right branch part 2: virtual index 15 -> rightBranch[4]
  if (virtualIndex == 15) {
    led.strip = &rightBranch;
    led.index = 4;
    return led;
  }

  // Trunk part 2: virtual indices 16-18 -> trunk[7-9]
  if (virtualIndex >= 16 && virtualIndex <= 18) {
    led.strip = &trunk;
    led.index = virtualIndex - 16 + 7;
    return led;
  }

  // Left branch part 1: virtual indices 19-23 -> lowerLeft[0-4]
  if (virtualIndex >= 19 && virtualIndex <= 23) {
    led.strip = &lowerLeft;
    led.index = virtualIndex - 19;
    return led;
  }

  // Upper left twig: virtual indices 24-27 -> upperLeft[0-3]
  if (virtualIndex >= 24 && virtualIndex <= 27) {
    led.strip = &upperLeft;
    led.index = virtualIndex - 24;
    return led;
  }

  // Left branch part 2: virtual indices 28-29 -> lowerLeft[5-6]
  if (virtualIndex >= 28 && virtualIndex <= 29) {
    led.strip = &lowerLeft;
    led.index = virtualIndex - 28 + 5;
    return led;
  }

  // Default case (should not happen)
  led.strip = &trunk;
  led.index = 0;
  return led;
}

// Set color on virtual tree using virtual index
void setVirtualPixelColor(int virtualIndex, uint32_t color) {
  if (virtualIndex < 0 || virtualIndex >= VIRTUAL_TREE_SIZE) return;

  PhysicalLED led = getPhysicalLED(virtualIndex);
  led.strip->setPixelColor(led.index, color);
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

// Slow crawl of brown and green across virtual tree structure
void colorCrawl(int wait) {
  static int offset = 0;  // Start at trunk base

  // Brown and green colors
  int brownR = 139, brownG = 69, brownB = 19;
  int greenR = 34, greenG = 139, greenB = 34;

  // Wave width
  float waveWidth = 3.0;

  // Apply wave to entire virtual tree
  for (int i = 0; i < VIRTUAL_TREE_SIZE; i++) {
    float distance = abs(i - offset);

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

    // Set color on virtual tree (maps to physical strips)
    PhysicalLED led = getPhysicalLED(i);
    led.strip->setPixelColor(led.index, led.strip->Color(r, g, b));
  }

  // Show all strips
  trunk.show();
  lowerLeft.show();
  rightBranch.show();
  upperLeft.show();
  rightTwig.show();

  // Move wave forward
  offset += 1 * direction;

  // Reset when wave completes full cycle through virtual tree
  if (offset > VIRTUAL_TREE_SIZE + (int)waveWidth || offset < (int)waveWidth * -1) {
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
