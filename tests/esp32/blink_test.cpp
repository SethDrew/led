/*
 * ESP32 Built-in LED Blink Test
 * Proves ESP32 is working - you'll see the onboard LED blinking
 */

#include <Arduino.h>

#define LED_BUILTIN 2  // Most ESP32 boards have LED on GPIO 2

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== ESP32 Blink Test ===");
  Serial.println("Built-in LED should be blinking!");
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("LED ON");
  delay(500);

  digitalWrite(LED_BUILTIN, LOW);
  Serial.println("LED OFF");
  delay(500);
}
