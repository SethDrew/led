/*
 * UNIVERSAL LED STRIP TESTER
 *
 * Works with ANY strip length! Starts with a few LEDs and shows you how many work.
 * Shows a chaser pattern - only lights up a few LEDs at a time so it won't fail
 * even if you specify more LEDs than you have connected.
 *
 * Watch the serial monitor - it will tell you how far the chaser gets!
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 13
#define MAX_PIXELS 300  // Set this high - won't hurt if you have fewer!

Adafruit_NeoPixel strip(MAX_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

int chaserPos = 0;
int chaserLength = 5;  // Number of LEDs in the chaser
unsigned long lastUpdate = 0;
int updateDelay = 10;  // Speed of chaser (ms) - 10ms = 5x faster than 50ms

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("==================================");
  Serial.println("  UNIVERSAL LED STRIP TESTER");
  Serial.println("==================================");
  Serial.println();
  Serial.print("Testing up to ");
  Serial.print(MAX_PIXELS);
  Serial.println(" LEDs");
  Serial.println("Watch the chaser pattern!");
  Serial.println("It will cycle through your strip.");
  Serial.println();

  strip.begin();
  strip.setBrightness(128);  // 50% brightness
  strip.clear();
  strip.show();
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastUpdate > updateDelay) {
    lastUpdate = currentTime;

    // Clear all LEDs
    strip.clear();

    // Light up the chaser (cool blue to hot pink gradient)
    for (int i = 0; i < chaserLength; i++) {
      int pos = (chaserPos + i) % MAX_PIXELS;

      // Gradient from cool blue (trailing) to hot pink (leading)
      float t = (float)i / (chaserLength - 1);  // 0.0 at trailing, 1.0 at leading

      // Cool blue (70% brightness): RGB(0, 105, 178) at t=0
      // Hot pink (high contrast): RGB(255, 0, 127) at t=1
      uint8_t r = 0 + (t * 255);
      uint8_t g = 105 - (t * 105);
      uint8_t b = 178 - (t * 51);

      strip.setPixelColor(pos, strip.Color(r, g, b));
    }

    strip.show();

    // Print status every 10 LEDs
    if (chaserPos % 10 == 0) {
      Serial.print("Chaser at LED: ");
      Serial.println(chaserPos);
    }

    // Move chaser forward
    chaserPos++;
    if (chaserPos >= MAX_PIXELS) {
      chaserPos = 0;
      Serial.println();
      Serial.println("--- Completed one full cycle ---");
      Serial.println();
    }
  }
}
