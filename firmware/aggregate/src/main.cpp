/*
 * AGGREGATE FIRMWARE
 *
 * Runs TWO concurrent systems on ESP32:
 * 1. Tree animation (sap flow) on 3 strips - 197 LEDs total
 *    - Strip 1: Pin 27 (92 LEDs)
 *    - Strip 2: Pin 13 (6 LEDs)
 *    - Strip 3: Pin 14 (99 LEDs)
 *
 * 2. Streaming receiver on Pin 33 - 300 LEDs
 *    - Protocol: [0xFF][0xAA][RGB x 300][XOR_CHECKSUM]
 *    - 1Mbps serial, non-blocking state machine
 *    - NEO_GRB color order, WS2812B
 */

#include <Adafruit_NeoPixel.h>
#include "TreeTopology.h"
#include "foregrounds/SapFlowForeground.h"

// ============================================================================
// TREE ANIMATION SETUP
// ============================================================================

Tree tree;
SapFlowForeground* sapFlow;

unsigned long lastTreeFrame = 0;
const unsigned long TREE_FRAME_INTERVAL = 40; // ~25 FPS

// ============================================================================
// STREAMING RECEIVER SETUP
// ============================================================================

#ifndef STREAM_PIN
#define STREAM_PIN 33
#endif
#ifndef STREAM_LEDS
#define STREAM_LEDS 300
#endif

Adafruit_NeoPixel streamStrip(STREAM_LEDS, STREAM_PIN, NEO_GRB + NEO_KHZ800);

const uint8_t START1 = 0xFF;
const uint8_t START2 = 0xAA;
const uint16_t FRAME_BYTES = STREAM_LEDS * 3;

// Frame buffer (900 bytes - fine on ESP32)
uint8_t buf[STREAM_LEDS * 3];

enum RxState { WAIT_SYNC1, WAIT_SYNC2, READ_FRAME };
RxState rxState = WAIT_SYNC1;
uint16_t bytesRead = 0;
uint8_t runningXor = 0;
unsigned long frameStartTime = 0;

// Stats
unsigned long streamFrames = 0;
unsigned long streamDropped = 0;
unsigned long lastStats = 0;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(1000000);

  // Initialize tree
  tree.begin();
  sapFlow = new SapFlowForeground(&tree, 50, 255, 50, 5); // Green sap flow

  // Initialize streaming receiver
  streamStrip.begin();
  streamStrip.setBrightness(255);
  streamStrip.clear();
  streamStrip.show();

  Serial.println("Aggregate firmware initialized");
  Serial.println("Tree: 197 LEDs on 3 strips (pins 27/13/14)");
  Serial.println("Stream: 300 LEDs on pin 33");
}

// ============================================================================
// STREAMING RECEIVER (Non-blocking)
// ============================================================================

void processStreamSerial() {
  while (Serial.available()) {
    uint8_t b = Serial.read();

    switch (rxState) {
      case WAIT_SYNC1:
        if (b == START1) {
          rxState = WAIT_SYNC2;
        }
        break;

      case WAIT_SYNC2:
        if (b == START2) {
          rxState = READ_FRAME;
          bytesRead = 0;
          runningXor = 0;
          frameStartTime = millis();
        } else if (b == START1) {
          // Stay in WAIT_SYNC2 for consecutive 0xFF
        } else {
          rxState = WAIT_SYNC1;
        }
        break;

      case READ_FRAME:
        if (bytesRead < FRAME_BYTES) {
          buf[bytesRead++] = b;
          runningXor ^= b;
        } else {
          // Checksum byte received
          if (runningXor == b) {
            // Valid frame - update strip
            for (uint16_t i = 0; i < STREAM_LEDS; i++) {
              uint16_t idx = i * 3;
              streamStrip.setPixelColor(i, buf[idx], buf[idx+1], buf[idx+2]);
            }
            streamStrip.show();
            streamFrames++;
          } else {
            // Checksum failed
            streamDropped++;
          }
          rxState = WAIT_SYNC1;
        }
        break;
    }
  }

  // Frame timeout (50ms)
  if (rxState == READ_FRAME && (millis() - frameStartTime) > 50) {
    rxState = WAIT_SYNC1;
    streamDropped++;
  }
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Process streaming receiver (non-blocking, every iteration)
  processStreamSerial();

  // Update tree animation at ~25 FPS
  unsigned long now = millis();
  if (now - lastTreeFrame >= TREE_FRAME_INTERVAL) {
    lastTreeFrame = now;
    sapFlow->update();
    sapFlow->render();
  }

  // Print stats every 2 seconds
  if (now - lastStats > 2000) {
    unsigned long elapsed = now - lastStats;

    Serial.print("Stream FPS: ");
    Serial.print(streamFrames * 1000 / elapsed);
    if (streamDropped > 0) {
      Serial.print(" drop: ");
      Serial.print(streamDropped);
    }
    Serial.println();

    streamFrames = 0;
    streamDropped = 0;
    lastStats = now;
  }
}
