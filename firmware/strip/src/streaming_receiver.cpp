/*
 * LED STREAMING RECEIVER / WHITE TEST
 *
 * Receives RGB frame data from computer via serial and displays on LEDs.
 * Protocol: [0xFF] [0xAA] [R0] [G0] [B0] ... [Rn] [Gn] [Bn] [XOR_CHECKSUM]
 * XOR_CHECKSUM = XOR of all RGB bytes. Invalid frames hold last good display.
 *
 * State machine sync recovery: scans for 0xFF+0xAA pair byte-by-byte,
 * so a single corrupted byte can't cascade into multi-frame glitching.
 *
 * If WHITE_TEST is defined, displays solid white at 60% brightness instead.
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12

// NUM_PIXELS set by build flag
#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

// Color order: set via build flag (-DCOLOR_ORDER=NEO_GRB or NEO_RGB)
#ifndef COLOR_ORDER
#define COLOR_ORDER NEO_GRB
#endif

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, COLOR_ORDER + NEO_KHZ800);

#ifdef WHITE_TEST
// WHITE TEST MODE - 10% brightness = 26 out of 255
#define WHITE_BRIGHTNESS 26
#else

// Frame buffer
uint8_t frameBuffer[NUM_PIXELS * 3];

// Protocol bytes
const uint8_t START_BYTE_1 = 0xFF;
const uint8_t START_BYTE_2 = 0xAA;
const uint16_t FRAME_BYTES = NUM_PIXELS * 3;

// Receiver state machine — resilient sync recovery
enum RxState { WAIT_SYNC1, WAIT_SYNC2, READ_FRAME };
RxState rxState = WAIT_SYNC1;
uint16_t bytesRead = 0;
uint8_t runningXor = 0;
unsigned long frameStartTime = 0;

// Stats
unsigned long framesReceived = 0;
unsigned long framesDropped = 0;
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
  //
  // State machine processes one byte at a time for robust sync recovery.
  // Protocol: [0xFF][0xAA][RGB x N][XOR_CHECKSUM]
  // On checksum fail or timeout: hold last good frame, scan for next sync.
  while (Serial.available()) {
    uint8_t b = Serial.read();

    switch (rxState) {
      case WAIT_SYNC1:
        if (b == START_BYTE_1) rxState = WAIT_SYNC2;
        break;

      case WAIT_SYNC2:
        if (b == START_BYTE_2) {
          // Found sync pair — start reading frame
          rxState = READ_FRAME;
          bytesRead = 0;
          runningXor = 0;
          frameStartTime = millis();
        } else if (b == START_BYTE_1) {
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
            // Valid frame — update LEDs
            for (uint16_t i = 0; i < NUM_PIXELS; i++) {
              uint16_t idx = i * 3;
              strip.setPixelColor(i, frameBuffer[idx], frameBuffer[idx+1], frameBuffer[idx+2]);
            }
            strip.show();
            framesReceived++;
          } else {
            // Checksum mismatch — hold last good frame
            framesDropped++;
          }
          rxState = WAIT_SYNC1;
        }
        break;
    }
  }

  // Frame timeout — abandon incomplete frame if data stops arriving
  if (rxState == READ_FRAME && (millis() - frameStartTime) > 50) {
    rxState = WAIT_SYNC1;
    framesDropped++;
  }

  // Stats every 2s
  if (millis() - lastStatsTime > 2000) {
    unsigned long elapsed = millis() - lastStatsTime;
    Serial.print("FPS: ");
    Serial.print(framesReceived * 1000 / elapsed);
    if (framesDropped > 0) {
      Serial.print(" drop:");
      Serial.print(framesDropped);
    }
    Serial.println();
    framesReceived = 0;
    framesDropped = 0;
    lastStatsTime = millis();
  }
#endif
}
