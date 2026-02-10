/*
 * AUDIO-REACTIVE LED STREAMING RECEIVER
 *
 * Receives RGB frame data from computer via serial and displays on tree LEDs.
 * Protocol: [0xFF] [0xAA] [R0] [G0] [B0] [R1] [G1] [B1] ... [R196] [G196] [B196]
 *
 * Copied from led/streaming/src/streaming_receiver.ino
 * Last synced: 2026-02-03
 * Modified for tree topology (197 LEDs, 3 strips)
 */

#include <Adafruit_NeoPixel.h>

// Tree topology - 3 strips
#define STRIP1_PIN 13
#define STRIP2_PIN 12
#define STRIP3_PIN 11

#define STRIP1_LEDS 92
#define STRIP2_LEDS 6
#define STRIP3_LEDS 99

#ifndef NUM_LEDS
#define NUM_LEDS 197
#endif

Adafruit_NeoPixel strip1(STRIP1_LEDS, STRIP1_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip2(STRIP2_LEDS, STRIP2_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip3(STRIP3_LEDS, STRIP3_PIN, NEO_GRB + NEO_KHZ800);

// Frame buffer
uint8_t frameBuffer[NUM_LEDS * 3];

// Protocol bytes
const uint8_t START_BYTE_1 = 0xFF;
const uint8_t START_BYTE_2 = 0xAA;

// Stats
unsigned long framesReceived = 0;
unsigned long lastStatsTime = 0;

void setup() {
  // High baud rate for better FPS
  Serial.begin(1000000);  // 1 Mbps

  strip1.begin();
  strip1.setBrightness(51);  // 20% brightness
  strip1.show();

  strip2.begin();
  strip2.setBrightness(51);
  strip2.show();

  strip3.begin();
  strip3.setBrightness(51);
  strip3.show();

  Serial.println("Audio-Reactive LED Streaming Receiver Ready");
  Serial.print("Waiting for frames (");
  Serial.print(NUM_LEDS);
  Serial.println(" LEDs across 3 strips)...");
}

void loop() {
  // Wait for start sequence
  if (Serial.available() >= 2) {
    if (Serial.read() == START_BYTE_1 && Serial.read() == START_BYTE_2) {
      // Read RGB data for all LEDs
      int bytesNeeded = NUM_LEDS * 3;
      int bytesRead = 0;

      // Read with timeout
      unsigned long startTime = millis();
      while (bytesRead < bytesNeeded && (millis() - startTime) < 100) {
        if (Serial.available()) {
          frameBuffer[bytesRead++] = Serial.read();
        }
      }

      // If we got a complete frame, display it
      if (bytesRead == bytesNeeded) {
        // Update LEDs from buffer, mapping to correct strips
        int ledIndex = 0;

        // Strip 1: LEDs 0-91
        for (int i = 0; i < STRIP1_LEDS; i++) {
          int idx = ledIndex * 3;
          strip1.setPixelColor(i, frameBuffer[idx], frameBuffer[idx+1], frameBuffer[idx+2]);
          ledIndex++;
        }

        // Strip 2: LEDs 92-97
        for (int i = 0; i < STRIP2_LEDS; i++) {
          int idx = ledIndex * 3;
          strip2.setPixelColor(i, frameBuffer[idx], frameBuffer[idx+1], frameBuffer[idx+2]);
          ledIndex++;
        }

        // Strip 3: LEDs 98-196
        for (int i = 0; i < STRIP3_LEDS; i++) {
          int idx = ledIndex * 3;
          strip3.setPixelColor(i, frameBuffer[idx], frameBuffer[idx+1], frameBuffer[idx+2]);
          ledIndex++;
        }

        strip1.show();
        strip2.show();
        strip3.show();

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
        Serial.println("Warning: Incomplete frame");
      }
    }
  }
}
