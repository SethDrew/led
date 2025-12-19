#include <Arduino.h>
#line 1 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
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

// Tree colors
uint32_t barkBrown;
uint32_t leafGreen;
uint32_t springGreen;

#line 29 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void setup();
#line 60 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void loop();
#line 66 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void growthAnimation(int wait);
#line 100 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void windEffect(int wait, int duration);
#line 121 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void shimmerBranch(Adafruit_NeoPixel &branch, int numLEDs, uint32_t baseColor);
#line 134 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void breatheEffect(int wait, int cycles);
#line 150 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void setAllBrightness(int brightness);
#line 186 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void clearAll();
#line 200 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void colorCrawl(int wait);
#line 233 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
void applyWaveToBranch(Adafruit_NeoPixel &branch, int numLEDs, int offset, float waveWidth, int color1R, int color1G, int color1B, int color2R, int color2G, int color2B);
#line 29 "/Users/KO16K39/Documents/led/led-complex/led-complex.ino"
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

// Growth animation - starts from trunk, spreads to branches
void growthAnimation(int wait) {
  clearAll();

  // Grow trunk from bottom to top
  for(int i = 0; i < TRUNK_LEDS; i++) {
    trunk.setPixelColor(i, barkBrown);
    trunk.show();
    delay(wait);
  }

  // Branches grow simultaneously from their bases
  int maxLen = max(max(LOWER_LEFT_LEDS, RIGHT_LEDS), max(UPPER_LEFT_LEDS, RIGHT_TWIG_LEDS));
  for(int i = 0; i < maxLen; i++) {
    if(i < LOWER_LEFT_LEDS) {
      lowerLeft.setPixelColor(i, leafGreen);
      lowerLeft.show();
    }
    if(i < RIGHT_LEDS) {
      rightBranch.setPixelColor(i, leafGreen);
      rightBranch.show();
    }
    if(i < UPPER_LEFT_LEDS) {
      upperLeft.setPixelColor(i, springGreen);
      upperLeft.show();
    }
    if(i < RIGHT_TWIG_LEDS) {
      rightTwig.setPixelColor(i, springGreen);
      rightTwig.show();
    }
    delay(wait);
  }
}

// Wind effect - random shimmering in branches
void windEffect(int wait, int duration) {
  unsigned long startTime = millis();

  while(millis() - startTime < duration) {
    // Keep trunk steady (brown)
    for(int i = 0; i < TRUNK_LEDS; i++) {
      trunk.setPixelColor(i, barkBrown);
    }
    trunk.show();

    // Random shimmer in branches
    shimmerBranch(lowerLeft, LOWER_LEFT_LEDS, leafGreen);
    shimmerBranch(rightBranch, RIGHT_LEDS, leafGreen);
    shimmerBranch(upperLeft, UPPER_LEFT_LEDS, springGreen);
    shimmerBranch(rightTwig, RIGHT_TWIG_LEDS, springGreen);

    delay(wait);
  }
}

// Helper function to create shimmer effect on a branch
void shimmerBranch(Adafruit_NeoPixel &branch, int numLEDs, uint32_t baseColor) {
  for(int i = 0; i < numLEDs; i++) {
    // Random brightness variation
    int brightness = random(50, 100);
    uint8_t r = ((baseColor >> 16) & 0xFF) * brightness / 100;
    uint8_t g = ((baseColor >> 8) & 0xFF) * brightness / 100;
    uint8_t b = (baseColor & 0xFF) * brightness / 100;
    branch.setPixelColor(i, branch.Color(r, g, b));
  }
  branch.show();
}

// Breathing effect - entire tree pulses
void breatheEffect(int wait, int cycles) {
  for(int cycle = 0; cycle < cycles; cycle++) {
    // Breathe in (brighten)
    for(int brightness = 50; brightness <= 100; brightness += 5) {
      setAllBrightness(brightness);
      delay(wait);
    }
    // Breathe out (dim)
    for(int brightness = 100; brightness >= 50; brightness -= 5) {
      setAllBrightness(brightness);
      delay(wait);
    }
  }
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

// Slow crawl of brown and green across all branches
void colorCrawl(int wait) {
  static int offset = 0;

  // Brown and green colors
  int brownR = 139, brownG = 69, brownB = 19;
  int greenR = 34, greenG = 139, greenB = 34;

  // Wave width
  float waveWidth = 3.0;

  // Apply wave to each branch
  applyWaveToBranch(trunk, TRUNK_LEDS, offset, waveWidth, brownR, brownG, brownB, greenR, greenG, greenB);
  applyWaveToBranch(lowerLeft, LOWER_LEFT_LEDS, offset, waveWidth, brownR, brownG, brownB, greenR, greenG, greenB);
  applyWaveToBranch(rightBranch, RIGHT_LEDS, offset, waveWidth, brownR, brownG, brownB, greenR, greenG, greenB);
  applyWaveToBranch(upperLeft, UPPER_LEFT_LEDS, offset, waveWidth, brownR, brownG, brownB, greenR, greenG, greenB);
  applyWaveToBranch(rightTwig, RIGHT_TWIG_LEDS, offset, waveWidth, brownR, brownG, brownB, greenR, greenG, greenB);

  // Move wave forward
  offset++;

  // Get max length for proper cycling
  int maxLen = max(max(TRUNK_LEDS, LOWER_LEFT_LEDS), max(RIGHT_LEDS, max(UPPER_LEFT_LEDS, RIGHT_TWIG_LEDS)));

  // Reset when wave completes
  if (offset > maxLen + (int)waveWidth) {
    delay(200);
    offset = -(int)waveWidth;
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

