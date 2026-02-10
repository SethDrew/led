/*
 * LED TREE STREAMING RECEIVER
 *
 * Receives RGB frame data from computer via serial and displays on tree LEDs.
 * Protocol: [0xFF] [0xAA] [R0] [G0] [B0] ... [R196] [G196] [B196]
 *
 * Key design:
 * - 197 nodes (3 strips), 591 bytes of RGB data per frame
 * - Reads bytes directly into strip pixels (NO frame buffer to save RAM)
 * - Uses TreeTopology node mapping (strip1: 92 LEDs, strip2: 6 LEDs, strip3: 99 LEDs)
 * - 1Mbps baud rate for speed
 *
 * Memory usage: ~591 bytes for NeoPixel strip buffers + ~600 bytes for topology = ~1200 bytes
 * (ATmega328 has 2KB RAM, so this fits comfortably)
 */

#include <Arduino.h>
#include "TreeTopology.h"

// Protocol bytes
const uint8_t START_BYTE_1 = 0xFF;
const uint8_t START_BYTE_2 = 0xAA;

// Global tree instance
Tree tree;

// Stats
unsigned long framesReceived = 0;
unsigned long lastStatsTime = 0;

void setup() {
  // High baud rate for better FPS
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
      // Read RGB data for all nodes
      int bytesNeeded = tree.getNumLEDs() * 3;  // 197 * 3 = 591 bytes
      int bytesRead = 0;

      // Temporary buffer for bulk reading (faster than single reads)
      // We use a small rolling buffer to avoid allocating 591 bytes
      uint8_t buf[30];  // Read in chunks of 30 bytes (10 pixels at a time)

      // Read with timeout
      unsigned long startTime = millis();
      uint8_t nodeIndex = 0;

      while (bytesRead < bytesNeeded && (millis() - startTime) < 100) {
        // How many bytes can we read?
        int available = Serial.available();
        if (available > 0) {
          // Read up to buffer size
          int toRead = min(available, (int)sizeof(buf));
          toRead = min(toRead, bytesNeeded - bytesRead);

          int actuallyRead = Serial.readBytes(buf, toRead);

          // Process the bytes we just read
          for (int i = 0; i < actuallyRead; i++) {
            int offset = bytesRead % 3;

            // Accumulate RGB values
            static uint8_t r, g, b;
            if (offset == 0) {
              r = buf[i];
            } else if (offset == 1) {
              g = buf[i];
            } else {  // offset == 2
              b = buf[i];
              // We have a complete pixel - write it to the correct strip
              tree.setNodeColor(nodeIndex, r, g, b);
              nodeIndex++;
            }

            bytesRead++;
          }
        }
      }

      // If we got a complete frame, display it
      if (bytesRead == bytesNeeded) {
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
        // Incomplete frame - just skip it
        Serial.print("Warning: Incomplete frame (got ");
        Serial.print(bytesRead);
        Serial.print("/");
        Serial.print(bytesNeeded);
        Serial.println(" bytes)");
      }
    }
  }
}
