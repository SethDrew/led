/*
 * LED TREE STREAMING RECEIVER
 *
 * Receives RGB frame data from computer via serial and displays on tree LEDs.
 * Protocol: [0xFF] [0xAA] [R0] [G0] [B0] ... [R196] [G196] [B196]
 *
 * Key design:
 * - 197 nodes (3 strips), 591 bytes of RGB data per frame
 * - Reads all bytes into flat buffer first, THEN copies to tree pixels
 *   (matches proven strip receiver approach — Serial.read() byte-by-byte)
 * - Uses TreeTopology node mapping (strip1: 92 LEDs, strip2: 6 LEDs, strip3: 99 LEDs)
 * - 1Mbps baud rate for speed
 *
 * Memory: ~591 frameBuffer + ~591 NeoPixel + ~591 topology = ~1773 bytes (of 2048)
 */

#include <Arduino.h>
#include "TreeTopology.h"

// Protocol bytes
const uint8_t START_BYTE_1 = 0xFF;
const uint8_t START_BYTE_2 = 0xAA;

#define NUM_NODES 197

// Flat frame buffer — read all bytes here first, then copy to tree
uint8_t frameBuffer[NUM_NODES * 3];

// Global tree instance
Tree tree;

// Stats
unsigned long framesReceived = 0;
unsigned long lastStatsTime = 0;

void setup() {
  Serial.begin(1000000);  // 1 Mbps

  tree.begin();

  Serial.println("=================================");
  Serial.println("LED TREE");
  Serial.println("=================================");
  Serial.println("LED Tree Streaming Receiver Ready");
  Serial.print("Tree nodes: ");
  Serial.println(tree.getNumLEDs());
  Serial.println("Waiting for frames...");
}

void loop() {
  // Wait for start sequence
  if (Serial.available() >= 2) {
    if (Serial.read() == START_BYTE_1 && Serial.read() == START_BYTE_2) {
      int bytesNeeded = NUM_NODES * 3;  // 591 bytes
      int bytesRead = 0;

      // Read with timeout — byte-by-byte into flat buffer (proven approach)
      unsigned long startTime = millis();
      while (bytesRead < bytesNeeded && (millis() - startTime) < 100) {
        if (Serial.available()) {
          frameBuffer[bytesRead++] = Serial.read();
        }
      }

      // If we got a complete frame, copy to tree and display
      if (bytesRead == bytesNeeded) {
        for (int i = 0; i < NUM_NODES; i++) {
          int idx = i * 3;
          tree.setNodeColor(i, frameBuffer[idx], frameBuffer[idx+1], frameBuffer[idx+2]);
        }
        tree.show();

        framesReceived++;

        // Print stats every second
        if (millis() - lastStatsTime > 1000) {
          Serial.print("FPS: ");
          Serial.println(framesReceived);
          framesReceived = 0;
          lastStatsTime = millis();
        }
      } else {
        Serial.print("Warning: Incomplete frame (got ");
        Serial.print(bytesRead);
        Serial.print("/");
        Serial.print(bytesNeeded);
        Serial.println(" bytes)");
      }
    }
  }
}
