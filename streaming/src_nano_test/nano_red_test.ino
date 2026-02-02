/*
 * Arduino Nano Simple RED Test
 * Sets first 10 LEDs to red at half brightness
 * Nano uses 5V logic - should work directly with WS2812B!
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12
#define NUM_PIXELS 10

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== Arduino Nano RED Test ===");
  Serial.println("Pin: 12");
  Serial.println("Setting first 10 LEDs to RED...");

  strip.begin();
  strip.setBrightness(128);  // Half brightness

  // Set all 10 pixels to RED
  for(int i=0; i<NUM_PIXELS; i++) {
    strip.setPixelColor(i, 255, 0, 0);  // Bright red
  }
  strip.show();

  Serial.println("Done! LEDs should be RED.");
}

void loop() {
  delay(1000);
  Serial.println(".");  // Heartbeat
}
