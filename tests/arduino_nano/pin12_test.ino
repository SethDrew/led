/*
 * Pin 12 Test - Blink Branch C
 * Tests ONLY pin 12 with bright colors
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN_12 12
#define NUM_LEDS 5

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN_12, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Pin 12 Test - Branch C Only");

  strip.begin();
  strip.setBrightness(255);  // FULL brightness
  strip.show();
}

void loop() {
  // Cycle through bright colors
  Serial.println("RED");
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(255, 0, 0));
  }
  strip.show();
  delay(1000);

  Serial.println("GREEN");
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(0, 255, 0));
  }
  strip.show();
  delay(1000);

  Serial.println("BLUE");
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(0, 0, 255));
  }
  strip.show();
  delay(1000);

  Serial.println("WHITE");
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(255, 255, 255));
  }
  strip.show();
  delay(1000);

  Serial.println("OFF");
  strip.clear();
  strip.show();
  delay(1000);
}
