/*
 * All White Test
 * Lights all LEDs solid white
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN_MAIN 13
#define LED_PIN_BRANCH_C 12

#define TOTAL_LEDS_MAIN 92
#define BRANCH_C_LEDS 5

Adafruit_NeoPixel stripMain(TOTAL_LEDS_MAIN, LED_PIN_MAIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripBranchC(BRANCH_C_LEDS, LED_PIN_BRANCH_C, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("All White Test");
  Serial.print("Pin 13 LEDs: ");
  Serial.println(TOTAL_LEDS_MAIN);
  Serial.print("Pin 12 LEDs: ");
  Serial.println(BRANCH_C_LEDS);

  stripMain.begin();
  stripMain.setBrightness(255);  // Full brightness
  stripMain.clear();

  stripBranchC.begin();
  stripBranchC.setBrightness(255);  // Full brightness
  stripBranchC.clear();

  // Flash Branch C first to test
  Serial.println("Flashing Branch C (pin 12)...");
  for (int i = 0; i < BRANCH_C_LEDS; i++) {
    stripBranchC.setPixelColor(i, stripBranchC.Color(255, 0, 0));  // Red first
  }
  stripBranchC.show();
  delay(2000);

  // Then white
  Serial.println("Setting Branch C to white...");
  for (int i = 0; i < BRANCH_C_LEDS; i++) {
    stripBranchC.setPixelColor(i, stripBranchC.Color(255, 255, 255));
  }
  stripBranchC.show();

  // Set main strip to white
  Serial.println("Setting main strip to white...");
  for (int i = 0; i < TOTAL_LEDS_MAIN; i++) {
    stripMain.setPixelColor(i, stripMain.Color(255, 255, 255));
  }
  stripMain.show();

  Serial.println("All LEDs should be white");
}

void loop() {
  // Nothing - just stay white
  delay(1000);
}
