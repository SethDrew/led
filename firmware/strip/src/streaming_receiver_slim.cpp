/*
 * SLIM STREAMING RECEIVER — for large LED counts on ATmega328 (2KB RAM)
 *
 * Receives directly into NeoPixel's internal buffer to avoid double-buffering.
 * Trade-off: on checksum fail, partial frame is already in the pixel buffer,
 * so we skip show() but can't roll back. Good enough — corrupted frames are
 * rare and the next valid frame overwrites within 33ms.
 *
 * Protocol: [0xFF][0xAA][R0][G0][B0]...[Rn][Gn][Bn][XOR_CHECKSUM]
 */

#include <Adafruit_NeoPixel.h>
#include <EEPROM.h>

#define CMD_IDENTIFY 0xFE
#define CMD_SET_ID   0xFD
#define EEPROM_ID_ADDR 0
uint8_t deviceId;

#define LED_PIN 12

#ifndef NUM_PIXELS
#define NUM_PIXELS 300
#endif

#ifndef COLOR_ORDER
#define COLOR_ORDER NEO_GRB
#endif

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, COLOR_ORDER + NEO_KHZ800);

const uint8_t START1 = 0xFF;
const uint8_t START2 = 0xAA;
const uint16_t FRAME_BYTES = (uint16_t)NUM_PIXELS * 3;

// Pointer to NeoPixel's internal buffer (set in setup)
uint8_t* pixBuf;

enum RxState { WAIT_SYNC1, WAIT_SYNC2, READ_FRAME, WAIT_SET_ID };
RxState rxState = WAIT_SYNC1;
uint16_t bytesRead = 0;
uint8_t runningXor = 0;
unsigned long frameStartTime = 0;

unsigned long frames = 0;
unsigned long dropped = 0;
unsigned long lastStats = 0;

void setup() {
  Serial.begin(1000000);
  deviceId = EEPROM.read(EEPROM_ID_ADDR);
  strip.begin();
  strip.setBrightness(255);
  strip.clear();
  strip.show();

  pixBuf = strip.getPixels();
}

void loop() {
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
          rxState = READ_FRAME;
          bytesRead = 0;
          runningXor = 0;
          frameStartTime = millis();
        } else if (b == START1) {
          // consecutive 0xFF — stay
        } else {
          rxState = WAIT_SYNC1;
        }
        break;

      case READ_FRAME:
        if (bytesRead < FRAME_BYTES) {
          // Write directly into NeoPixel buffer.
          // NeoPixel stores in native color order, but setPixelColor
          // handles reordering. For direct buffer writes, we need to
          // account for color order. The protocol sends RGB; NeoPixel
          // may store GRB. We reorder on the fly.
          uint16_t pixIdx = bytesRead / 3;
          uint8_t  channel = bytesRead % 3;

          // Map incoming RGB channel to NeoPixel buffer position
          // NeoPixel buffer layout depends on color order:
          //   NEO_GRB: byte 0=G, byte 1=R, byte 2=B
          //   NEO_RGB: byte 0=R, byte 1=G, byte 2=B
          uint16_t base = pixIdx * 3;
#if COLOR_ORDER == NEO_GRB
          // Incoming: R=0, G=1, B=2 → Buffer: G=0, R=1, B=2
          static const uint8_t remap[3] = {1, 0, 2};
          pixBuf[base + remap[channel]] = b;
#else
          // NEO_RGB: no reorder needed
          pixBuf[base + channel] = b;
#endif

          runningXor ^= b;
          bytesRead++;
        } else {
          // Checksum byte
          if (runningXor == b) {
            strip.show();
            frames++;
          } else {
            // DEBUG: print mismatch
            Serial.print("XOR:");
            Serial.print(runningXor);
            Serial.print(" got:");
            Serial.print(b);
            Serial.print(" n:");
            Serial.println(bytesRead);
            dropped++;
          }
          rxState = WAIT_SYNC1;
        }
        break;
    }
  }

  // Frame timeout
  if (rxState == READ_FRAME && (millis() - frameStartTime) > 50) {
    rxState = WAIT_SYNC1;
    dropped++;
  }

  // Stats every 2s — only print if we've received at least one frame
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
