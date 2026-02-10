/*
 * LED STREAMING RECEIVER / WHITE TEST
 *
 * Receives RGB frame data from computer via serial and displays on LEDs.
 * Protocol: [0xFF] [0xAA] [R0] [G0] [B0] [R1] [G1] [B1] ... [R149] [G149] [B149]
 *
 * If WHITE_TEST is defined, displays solid white at 60% brightness instead.
 *
 * FUTURE: Can add frame buffering here (store 1-2 frames for smooth playback)
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12

// NUM_PIXELS set by build flag
#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

#ifdef WHITE_TEST
// WHITE TEST MODE - 10% brightness = 26 out of 255
#define WHITE_BRIGHTNESS 26
#else

// Frame buffer
uint8_t frameBuffer[NUM_PIXELS * 3];

// Protocol bytes
const uint8_t START_BYTE_1 = 0xFF;
const uint8_t START_BYTE_2 = 0xAA;

// Stats
unsigned long framesReceived = 0;
unsigned long lastStatsTime = 0;
#endif

void setup() {
#ifdef WHITE_TEST
  // WHITE TEST MODE
  Serial.begin(115200);

  strip.begin();
  strip.setBrightness(255);

  // Set all LEDs to white at 10% brightness
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, WHITE_BRIGHTNESS, WHITE_BRIGHTNESS, WHITE_BRIGHTNESS);
  }
  strip.show();

  Serial.println("LED White Test - 10% Brightness");
  Serial.print("LEDs: ");
  Serial.println(NUM_PIXELS);
  Serial.print("RGB Value: ");
  Serial.println(WHITE_BRIGHTNESS);
  Serial.print("Brightness: 10% (");
  Serial.print(WHITE_BRIGHTNESS);
  Serial.println("/255)");
  Serial.println("Running...");
#else
  // STREAMING RECEIVER MODE
  Serial.begin(1000000);  // 1 Mbps

  strip.begin();
  strip.setBrightness(255);

  // Init: show a few frames to sync strip
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, 50, 0, 0);
  }
  strip.show();
  delay(100);
  strip.clear();
  strip.show();
  delay(100);

  Serial.println("LED Streaming Receiver Ready");
  Serial.print("Waiting for frames (");
  Serial.print(NUM_PIXELS);
  Serial.println(" LEDs)...");
#endif
}

void loop() {
#ifdef WHITE_TEST
  // WHITE TEST MODE - Nothing to do, LEDs stay on
  delay(1000);
#else
  // STREAMING RECEIVER MODE
  // Wait for start sequence
  if (Serial.available() >= 2) {
    if (Serial.read() == START_BYTE_1 && Serial.read() == START_BYTE_2) {
      // Read RGB data for all LEDs
      int bytesNeeded = NUM_PIXELS * 3;
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
        // Update LEDs from frame buffer
        for (int i = 0; i < NUM_PIXELS; i++) {
          int idx = i * 3;
          strip.setPixelColor(i, frameBuffer[idx], frameBuffer[idx+1], frameBuffer[idx+2]);
        }
        strip.show();

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

  // NOTE: Future buffering would go here
  // - Store incoming frames in a ring buffer
  // - Display from buffer while receiving next frame
  // - Smooth out any computer pauses
#endif
}
