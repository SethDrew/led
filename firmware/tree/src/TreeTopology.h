#ifndef TREE_TOPOLOGY_H
#define TREE_TOPOLOGY_H

#include <Adafruit_NeoPixel.h>

/*
 * TREE TOPOLOGY - Single Physical Tree (Memory Optimized)
 *
 * This tree has 197 total LEDs across 3 physical strips
 *
 * STRIP 1 (Pin 13) - Lower Tree Section - 92 LEDs
 *   └─ Lower Trunk: LEDs 0-38 (depth 0-38)
 *       ├─ Branch A: LEDs 38-61 (depth 38-61) - splits at depth 38
 *       └─ Branch B: LEDs 62-91 (depth 38-67) - splits at depth 38
 *
 * STRIP 2 (Pin 12) - Side Branch - 6 LEDs
 *   └─ Branch C: LEDs 0-5 (depth 25-30) - splits from trunk at depth 25
 *
 * STRIP 3 (Pin 11) - Upper Tree Section - 99 LEDs
 *   └─ Upper Trunk: LEDs 0-70 (depth 0-70) - continues from strip 1
 *       ├─ Branch D: LEDs 71-72 (depth 43-44) - splits at depth 43
 *       └─ Branch E: LEDs 73-98 (depth 43-68) - splits at depth 43
 */

// Pin assignments (overridable via build flags for ESP32 etc.)
#ifndef STRIP1_PIN
#define STRIP1_PIN 13
#endif
#ifndef STRIP2_PIN
#define STRIP2_PIN 12
#endif
#ifndef STRIP3_PIN
#define STRIP3_PIN 11
#endif

// Strip LED counts
#define STRIP1_LEDS 92
#define STRIP2_LEDS 6
#define STRIP3_LEDS 99

// Tree topology constants
#define MAX_DEPTH 70

// Compact LED node - only 3 bytes per LED instead of 10+
struct CompactNode {
  uint8_t stripId;      // 0=strip1, 1=strip2, 2=strip3
  uint8_t stripIndex;   // Index on that strip
  uint8_t depth;        // Depth in tree (0-70)
};

// Tree class manages the physical tree structure
class Tree {
private:
  Adafruit_NeoPixel strip1;
  Adafruit_NeoPixel strip2;
  Adafruit_NeoPixel strip3;

  // Compact representation - 3 bytes × 197 = 591 bytes (vs ~2000 bytes before!)
  CompactNode nodes[197];
  uint8_t numNodes;

  void initializeTopology() {
    uint8_t idx = 0;

    // STRIP 1 - Lower tree section (pin 13)

    // Lower trunk: LEDs 0-38, depths 0-38
    for (uint8_t i = 0; i <= 38; i++) {
      nodes[idx].stripId = 0;
      nodes[idx].stripIndex = i;
      nodes[idx].depth = i;
      idx++;
    }

    // Branch A: LEDs 39-61, depths 39-61 (LED 38 is the fork point, already in trunk)
    for (uint8_t i = 39; i <= 61; i++) {
      nodes[idx].stripId = 0;
      nodes[idx].stripIndex = i;
      nodes[idx].depth = i;
      idx++;
    }

    // Branch B: LEDs 62-91, depths 38-67
    for (uint8_t i = 62; i <= 91; i++) {
      nodes[idx].stripId = 0;
      nodes[idx].stripIndex = i;
      nodes[idx].depth = 38 + (i - 62);
      idx++;
    }

    // STRIP 2 - Side branch (pin 12)

    // Branch C: LEDs 0-5, depths 25-30
    for (uint8_t i = 0; i < 6; i++) {
      nodes[idx].stripId = 1;
      nodes[idx].stripIndex = i;
      nodes[idx].depth = 25 + i;
      idx++;
    }

    // STRIP 3 - Upper tree section (pin 11)

    // Upper trunk: LEDs 0-70, depths 0-70
    for (uint8_t i = 0; i <= 70; i++) {
      nodes[idx].stripId = 2;
      nodes[idx].stripIndex = i;
      nodes[idx].depth = i;
      idx++;
    }

    // Branch D: LEDs 71-72, depths 43-44
    for (uint8_t i = 71; i <= 72; i++) {
      nodes[idx].stripId = 2;
      nodes[idx].stripIndex = i;
      nodes[idx].depth = 43 + (i - 71);
      idx++;
    }

    // Branch E: LEDs 73-98, depths 43-68
    for (uint8_t i = 73; i <= 98; i++) {
      nodes[idx].stripId = 2;
      nodes[idx].stripIndex = i;
      nodes[idx].depth = 43 + (i - 73);
      idx++;
    }

    numNodes = idx;
  }

  // Get strip pointer by ID
  inline Adafruit_NeoPixel* getStrip(uint8_t stripId) {
    switch(stripId) {
      case 0: return &strip1;
      case 1: return &strip2;
      case 2: return &strip3;
      default: return &strip1;
    }
  }

  // Get depth offset for animation (strip 3 has +1 offset)
  inline uint8_t getDepthOffset(uint8_t stripId) {
    return (stripId == 2) ? 1 : 0;
  }

public:
  Tree() :
    strip1(STRIP1_LEDS, STRIP1_PIN, NEO_GRB + NEO_KHZ800),
    strip2(STRIP2_LEDS, STRIP2_PIN, NEO_GRB + NEO_KHZ800),
    strip3(STRIP3_LEDS, STRIP3_PIN, NEO_GRB + NEO_KHZ800) {
    initializeTopology();
  }

  void begin() {
    strip1.begin();
    strip1.setBrightness(255);  // 100% brightness
    strip1.clear();
    strip1.show();

    strip2.begin();
    strip2.setBrightness(255);
    strip2.clear();
    strip2.show();

    strip3.begin();
    strip3.setBrightness(255);
    strip3.clear();
    strip3.show();
  }

  void show() {
    strip1.show();
    strip2.show();
    strip3.show();
  }

  void clear() {
    strip1.clear();
    strip2.clear();
    strip3.clear();
  }

  // Get number of LEDs
  uint8_t getNumLEDs() {
    return numNodes;
  }

  // Get depth of a node
  uint8_t getDepth(uint8_t nodeIndex) {
    return nodes[nodeIndex].depth;
  }

  // Get effective depth (with animation offset)
  uint8_t getEffectiveDepth(uint8_t nodeIndex) {
    return nodes[nodeIndex].depth + getDepthOffset(nodes[nodeIndex].stripId);
  }

  // Set color of a specific node
  void setNodeColor(uint8_t nodeIndex, uint8_t r, uint8_t g, uint8_t b) {
    CompactNode& node = nodes[nodeIndex];
    Adafruit_NeoPixel* strip = getStrip(node.stripId);
    strip->setPixelColor(node.stripIndex, strip->Color(r, g, b));
  }

  // Direct access to strips for advanced users
  Adafruit_NeoPixel& getStrip1() { return strip1; }
  Adafruit_NeoPixel& getStrip2() { return strip2; }
  Adafruit_NeoPixel& getStrip3() { return strip3; }
};

#endif
