/*
 * ESP32 Multi-Pin Test
 * Lights up first 10 LEDs on ALL valid GPIO pins simultaneously
 * Just look at your LED strip - whichever pin it's wired to will light up!
 */

#include <Adafruit_NeoPixel.h>

#define NUM_PIXELS 10
#define BRIGHTNESS 128  // Half brightness (0-255)

// Valid GPIO pins for NeoPixel on ESP32
const int test_pins[] = {
  0, 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33
};
const int num_pins = sizeof(test_pins) / sizeof(test_pins[0]);

Adafruit_NeoPixel* strips[20];  // Array to hold all strip objects

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== ESP32 Multi-Pin Test ===");
  Serial.println("Activating ALL pins simultaneously...");
  Serial.println("Your LEDs should light up RED now!\n");
  Serial.println("Active GPIO pins:");

  // Initialize all pins
  for (int i = 0; i < num_pins; i++) {
    int pin = test_pins[i];
    Serial.print("  GPIO ");
    Serial.println(pin);

    strips[i] = new Adafruit_NeoPixel(NUM_PIXELS, pin, NEO_GRB + NEO_KHZ800);
    strips[i]->begin();
    strips[i]->setBrightness(BRIGHTNESS);

    // Set first 10 pixels to RED
    for(int j = 0; j < NUM_PIXELS; j++) {
      strips[i]->setPixelColor(j, 255, 0, 0);  // Red
    }
    strips[i]->show();
  }

  Serial.println("\nAll pins active! Check which GPIO pin lights up your LEDs.");
  Serial.println("LEDs will stay RED. Press reset to restart test.");
}

void loop() {
  // Continuously refresh all strips to keep data pins active
  for (int i = 0; i < num_pins; i++) {
    strips[i]->show();
  }
  delay(100);  // Refresh 10 times per second

  // Also print status
  static int count = 0;
  if (count++ % 10 == 0) {
    Serial.println("Refreshing all pins...");
  }
}
