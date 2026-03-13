/*
 * LED TREE STREAMING RECEIVER
 *
 * Receives RGB frame data from computer via serial and displays on tree LEDs.
 * Protocol: [0xFF][0xAA][R0 G0 B0 ... R196 G196 B196][XOR_CHECKSUM]
 * XOR_CHECKSUM = XOR of all RGB bytes. Invalid frames hold last good display.
 *
 * State machine sync recovery: scans for 0xFF+0xAA pair byte-by-byte,
 * so a single corrupted byte can't cascade into multi-frame glitching.
 *
 * Key design:
 * - 197 nodes (3 strips), 591 bytes of RGB data per frame + 1 checksum byte
 * - Double-buffered: reads into frameBuffer, copies to tree on valid checksum
 *   (tree has complex topology mapping via setNodeColor — no direct-buffer)
 * - Uses TreeTopology node mapping (strip1: 92 LEDs, strip2: 6 LEDs, strip3: 99 LEDs)
 * - Device ID support via EEPROM (CMD_IDENTIFY / CMD_SET_ID)
 * - Quiet when idle (no stats printing) so avrdude can sync for uploads
 * - 1Mbps baud rate
 *
 * Memory: ~591 frameBuffer + ~591 NeoPixel x3 + ~591 topology = ~1773 bytes (of 2048)
 */

#include <Arduino.h>
#include <EEPROM.h>
#include "TreeTopology.h"

// Protocol bytes
const uint8_t START1 = 0xFF;
const uint8_t START2 = 0xAA;

// Device identity — stored in EEPROM address 0
// Query: send 0xFE → responds [0xFE, id]
// Set:   send 0xFD <id> → writes EEPROM, responds [0xFD, id]
#define CMD_IDENTIFY   0xFE
#define CMD_SET_ID     0xFD
#define EEPROM_ID_ADDR 0
uint8_t deviceId;

#define NUM_NODES 197
const uint16_t FRAME_BYTES = (uint16_t)NUM_NODES * 3;

// Double buffer — read all bytes here first, then copy to tree on valid checksum
uint8_t frameBuffer[NUM_NODES * 3];

// Global tree instance
Tree tree;

// Receiver state machine — resilient sync recovery
enum RxState { WAIT_SYNC1, WAIT_SYNC2, READ_FRAME, WAIT_SET_ID };
RxState rxState = WAIT_SYNC1;
uint16_t bytesRead = 0;
uint8_t runningXor = 0;
unsigned long frameStartTime = 0;

// Stats
unsigned long frames = 0;
unsigned long dropped = 0;
unsigned long lastStats = 0;

void setup() {
  Serial.begin(1000000);  // 1 Mbps
  deviceId = EEPROM.read(EEPROM_ID_ADDR);

  tree.begin();

  Serial.println("=================================");
  Serial.println("LED TREE");
  Serial.println("=================================");
  Serial.println("LED Tree Streaming Receiver Ready");
  Serial.print("Tree nodes: ");
  Serial.println(tree.getNumLEDs());
  Serial.print("Device ID: ");
  Serial.println(deviceId);
  Serial.println("Waiting for frames...");
}

void loop() {
  // State machine processes one byte at a time for robust sync recovery.
  // Protocol: [0xFF][0xAA][RGB x 197][XOR_CHECKSUM]
  // On checksum fail or timeout: hold last good frame, scan for next sync.
  while (Serial.available()) {
    uint8_t b = Serial.read();

    switch (rxState) {
      case WAIT_SYNC1:
        if (b == START1) {
          rxState = WAIT_SYNC2;
        } else if (b == CMD_IDENTIFY) {
          Serial.write(CMD_IDENTIFY);
          Serial.write(deviceId);
        } else if (b == CMD_SET_ID) {
          rxState = WAIT_SET_ID;
        }
        break;

      case WAIT_SET_ID: {
        deviceId = b;
        EEPROM.update(EEPROM_ID_ADDR, deviceId);
        Serial.write(CMD_SET_ID);
        Serial.write(deviceId);
        rxState = WAIT_SYNC1;
        break;
      }

      case WAIT_SYNC2:
        if (b == START2) {
          // Found sync pair — start reading frame
          rxState = READ_FRAME;
          bytesRead = 0;
          runningXor = 0;
          frameStartTime = millis();
        } else if (b == START1) {
          // Consecutive 0xFF — stay, next byte might be 0xAA
        } else {
          rxState = WAIT_SYNC1;
        }
        break;

      case READ_FRAME:
        if (bytesRead < FRAME_BYTES) {
          frameBuffer[bytesRead++] = b;
          runningXor ^= b;
        } else {
          // This byte is the XOR checksum
          if (runningXor == b) {
            // Valid frame — copy buffer to tree topology and display
            for (uint16_t i = 0; i < NUM_NODES; i++) {
              uint16_t idx = i * 3;
              tree.setNodeColor(i, frameBuffer[idx], frameBuffer[idx+1], frameBuffer[idx+2]);
            }
            tree.show();
            frames++;
          } else {
            // Checksum mismatch — hold last good frame
            dropped++;
          }
          rxState = WAIT_SYNC1;
        }
        break;
    }
  }

  // Frame timeout — abandon incomplete frame if data stops arriving
  if (rxState == READ_FRAME && (millis() - frameStartTime) > 50) {
    rxState = WAIT_SYNC1;
    dropped++;
  }

  // Stats every 2s — only print if actively receiving
  // (keeps serial line quiet when idle, so avrdude can sync for uploads)
  if (millis() - lastStats > 2000) {
    if (frames > 0 || dropped > 0) {
      unsigned long elapsed = millis() - lastStats;
      Serial.print("FPS: ");
      Serial.print(frames * 1000 / elapsed);
      if (dropped > 0) {
        Serial.print(" drop:");
        Serial.print(dropped);
      }
      Serial.println();
    }
    frames = 0;
    dropped = 0;
    lastStats = millis();
  }
}
