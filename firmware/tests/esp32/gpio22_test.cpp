/*
 * Simple GPIO 22 Test
 * Toggles GPIO 22 HIGH (3.3V) and LOW (0V) every second
 * Measure with multimeter - should see voltage switching
 */

#include <Arduino.h>

#define TEST_PIN 22

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== GPIO 22 Test ===");
  Serial.println("Pin will toggle HIGH/LOW every second");
  Serial.println("Measure voltage with multimeter between GPIO 22 and GND");
  Serial.println("Should see: 3.3V → 0V → 3.3V → 0V...\n");

  pinMode(TEST_PIN, OUTPUT);
}

void loop() {
  digitalWrite(TEST_PIN, HIGH);
  Serial.println("GPIO 22 = HIGH (3.3V)");
  delay(1000);

  digitalWrite(TEST_PIN, LOW);
  Serial.println("GPIO 22 = LOW (0V)");
  delay(1000);
}
