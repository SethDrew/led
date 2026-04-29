/*
 * DIAMOND STREAMING RECEIVER
 *
 * Serial in → 73 LEDs out on pin 33.
 * Protocol: [0xFF][0xAA][R0][G0][B0]...[R72][G72][B72][XOR_CHECKSUM]
 * XOR checksum + state machine sync recovery — holds last good frame on error.
 * 1Mbps baud, NEO_RGB color order (WS2811 COB).
 */

#include <Adafruit_NeoPixel.h>

#ifndef DIAMOND_PIN
#define DIAMOND_PIN 33
#endif
#ifndef DIAMOND_LEDS
#define DIAMOND_LEDS 73
#endif

Adafruit_NeoPixel strip(DIAMOND_LEDS, DIAMOND_PIN, NEO_RGB + NEO_KHZ800);

const uint8_t START1 = 0xFF;
const uint8_t START2 = 0xAA;
const uint16_t FRAME_BYTES = DIAMOND_LEDS * 3;

uint8_t buf[73 * 3];

// Receiver state machine — resilient sync recovery
enum RxState { WAIT_SYNC1, WAIT_SYNC2, READ_FRAME };
RxState rxState = WAIT_SYNC1;
uint16_t bytesRead = 0;
uint8_t runningXor = 0;
unsigned long frameStartTime = 0;

unsigned long frames = 0;
unsigned long dropped = 0;
unsigned long lastStats = 0;

void setup() {
  Serial.begin(1000000);
  strip.begin();
  strip.setBrightness(255);
  strip.clear();
  strip.show();

  Serial.println("Diamond receiver ready");
  Serial.print("LEDs: ");
  Serial.print(DIAMOND_LEDS);
  Serial.print(" pin: ");
  Serial.println(DIAMOND_PIN);
}

void loop() {
  // State machine processes one byte at a time for robust sync recovery.
  // Protocol: [0xFF][0xAA][RGB x N][XOR_CHECKSUM]
  // On checksum fail or timeout: hold last good frame, scan for next sync.
  while (Serial.available()) {
    uint8_t b = Serial.read();

    switch (rxState) {
      case WAIT_SYNC1:
        if (b == START1) rxState = WAIT_SYNC2;
        break;

      case WAIT_SYNC2:
        if (b == START2) {
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
          buf[bytesRead++] = b;
          runningXor ^= b;
        } else {
          // XOR checksum byte
          if (runningXor == b) {
            for (uint8_t i = 0; i < DIAMOND_LEDS; i++) {
              uint16_t idx = i * 3;
              strip.setPixelColor(i, buf[idx], buf[idx+1], buf[idx+2]);
            }
            strip.show();
            frames++;
          } else {
            dropped++;
          }
          rxState = WAIT_SYNC1;
        }
        break;
    }
  }

  // Frame timeout — abandon incomplete frame
  if (rxState == READ_FRAME && (millis() - frameStartTime) > 50) {
    rxState = WAIT_SYNC1;
    dropped++;
  }

  // Stats every 2s
  if (millis() - lastStats > 2000) {
    unsigned long elapsed = millis() - lastStats;
    Serial.print("FPS: ");
    Serial.print(frames * 1000 / elapsed);
    if (dropped > 0) {
      Serial.print(" drop:");
      Serial.print(dropped);
    }
    Serial.println();
    frames = 0;
    dropped = 0;
    lastStats = millis();
  }
}
