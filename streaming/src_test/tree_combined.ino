/*
 * Combined Tree Visualization - Memory Optimized
 * Controls both old and new trees on one Arduino
 *
 * Old tree (pins 13 & 12):
 * - Pin 13: Trunk 0-38, Branch A 38-61, Branch B 62-91
 * - Pin 12: Branch C 0-5 (splits at trunk depth 25)
 *
 * New tree (pin 11):
 * - Trunk 0-69, LED 70 continuation
 * - Branch A 71-72, Branch B 73-98 (split at depth 43)
 */

#include <Adafruit_NeoPixel.h>

// Old tree
#define OLD_LED_PIN_MAIN 13
#define OLD_LED_PIN_BRANCH_C 12
#define OLD_TOTAL_LEDS_MAIN 92
#define OLD_BRANCH_C_LEDS 6

// New tree
#define NEW_LED_PIN 11
#define NEW_TOTAL_LEDS 99

#define GLOBAL_MAX_DEPTH 70

Adafruit_NeoPixel oldStripMain(OLD_TOTAL_LEDS_MAIN, OLD_LED_PIN_MAIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel oldStripBranchC(OLD_BRANCH_C_LEDS, OLD_LED_PIN_BRANCH_C, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel newStrip(NEW_TOTAL_LEDS, NEW_LED_PIN, NEO_GRB + NEO_KHZ800);

int8_t direction = 1;
int8_t offset = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("Combined Tree - Memory Optimized");

  oldStripMain.begin();
  oldStripMain.setBrightness(153);  // 60% brightness
  oldStripMain.clear();
  oldStripMain.show();

  oldStripBranchC.begin();
  oldStripBranchC.setBrightness(153);  // 60% brightness
  oldStripBranchC.clear();
  oldStripBranchC.show();

  newStrip.begin();
  newStrip.setBrightness(153);  // 60% brightness
  newStrip.clear();
  newStrip.show();

  Serial.println("Starting animation...");
}

void loop() {
  const int8_t greenR = 34, greenG = 139, greenB = 34;  // Forest green
  const float waveWidth = 5.0;

  // OLD TREE - Main strip pin 13
  // Trunk 0-38
  for (int8_t i = 0; i <= 38; i++) {
    float distance = abs(i - offset);
    if (distance <= waveWidth) {
      oldStripMain.setPixelColor(i, oldStripMain.Color(greenR, greenG, greenB));
    } else {
      oldStripMain.setPixelColor(i, 0);
    }
  }

  // Branch A 38-61 (depths 38-61)
  for (int8_t i = 38; i <= 61; i++) {
    int8_t depth = i;
    float distance = abs(depth - offset);
    if (distance <= waveWidth) {
      oldStripMain.setPixelColor(i, oldStripMain.Color(greenR, greenG, greenB));
    } else {
      oldStripMain.setPixelColor(i, 0);
    }
  }

  // Branch B 62-91 (depths 38-67)
  for (int8_t i = 62; i <= 91; i++) {
    int8_t depth = 38 + (i - 62);
    float distance = abs(depth - offset);
    if (distance <= waveWidth) {
      oldStripMain.setPixelColor(i, oldStripMain.Color(greenR, greenG, greenB));
    } else {
      oldStripMain.setPixelColor(i, 0);
    }
  }

  // OLD TREE - Branch C pin 12 (depths 25-30, always outward)
  for (int8_t i = 0; i < OLD_BRANCH_C_LEDS; i++) {
    int8_t depth = 25 + i;
    float distance;
    if (direction > 0) {
      distance = abs(depth - offset);
    } else {
      int8_t flippedDepth = (25 * 2 + OLD_BRANCH_C_LEDS - 1) - depth;
      distance = abs(flippedDepth - (offset - 1));
    }
    if (distance <= waveWidth) {
      oldStripBranchC.setPixelColor(i, oldStripBranchC.Color(greenR, greenG, greenB));
    } else {
      oldStripBranchC.setPixelColor(i, 0);
    }
  }

  // NEW TREE - pin 11 (advanced by 1 LED step)
  // Trunk 0-69
  for (int8_t i = 0; i <= 69; i++) {
    float distance = abs(i - (offset + 1));
    if (distance <= waveWidth) {
      newStrip.setPixelColor(i, newStrip.Color(greenR, greenG, greenB));
    } else {
      newStrip.setPixelColor(i, 0);
    }
  }

  // LED 70 (depth 70)
  {
    float distance = abs(70 - (offset + 1));
    if (distance <= waveWidth) {
      newStrip.setPixelColor(70, newStrip.Color(greenR, greenG, greenB));
    } else {
      newStrip.setPixelColor(70, 0);
    }
  }

  // Branch A 71-72 (depths 43-44)
  for (int8_t i = 71; i <= 72; i++) {
    int8_t depth = 43 + (i - 71);
    float distance = abs(depth - (offset + 1));
    if (distance <= waveWidth) {
      newStrip.setPixelColor(i, newStrip.Color(greenR, greenG, greenB));
    } else {
      newStrip.setPixelColor(i, 0);
    }
  }

  // Branch B 73-98 (depths 43-68)
  for (int8_t i = 73; i <= 98; i++) {
    int8_t depth = 43 + (i - 73);
    float distance = abs(depth - (offset + 1));
    if (distance <= waveWidth) {
      newStrip.setPixelColor(i, newStrip.Color(greenR, greenG, greenB));
    } else {
      newStrip.setPixelColor(i, 0);
    }
  }

  oldStripMain.show();
  oldStripBranchC.show();
  newStrip.show();

  offset += direction;

  if (offset > GLOBAL_MAX_DEPTH + 10 || offset < -10) {
    delay(1000);
    direction *= -1;
  }

  delay(40);
}
