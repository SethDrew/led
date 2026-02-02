/*
 * 3-LED Test
 * Lights up the first 3 LEDs in different colors to test wiring
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12
#define NUM_PIXELS 3

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  Serial.println("3-LED Test Starting...");

  strip.begin();
  strip.setBrightness(128);  // 50% brightness

  // LED 0: Red
  strip.setPixelColor(0, strip.Color(255, 0, 0));

  // LED 1: Green
  strip.setPixelColor(1, strip.Color(0, 255, 0));

  // LED 2: Blue
  strip.setPixelColor(2, strip.Color(0, 0, 255));

  strip.show();

  Serial.println("LEDs should be lit:");
  Serial.println("  LED 0: RED");
  Serial.println("  LED 1: GREEN");
  Serial.println("  LED 2: BLUE");
}

void loop() {
  // Nothing to do - LEDs stay on
  delay(1000);
}
