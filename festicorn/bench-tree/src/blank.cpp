/*
 * Blank - turns off all LEDs
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12
#define NUM_PIXELS 150

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  strip.begin();
  strip.clear();
  strip.show();
  Serial.println("LEDs off");
}

void loop() {
  delay(1000);
}
