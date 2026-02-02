/*
 * ESP32 Pin Scanner
 * Systematically tests all valid GPIO pins to find which one controls your LEDs
 * Watch your LED strip - when it lights up RED, note the pin number from serial output
 */

#include <Adafruit_NeoPixel.h>

#ifndef NUM_PIXELS
#define NUM_PIXELS 10
#endif

// Valid GPIO pins for NeoPixel on ESP32 (avoiding flash pins and input-only pins)
const int test_pins[] = {
  0, 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33
};
const int num_pins = sizeof(test_pins) / sizeof(test_pins[0]);

Adafruit_NeoPixel* strip = nullptr;
int current_pin_index = 0;

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n\n=== ESP32 Pin Scanner ===");
  Serial.println("Testing all valid GPIO pins...");
  Serial.println("WATCH YOUR LED STRIP - note which pin makes it light up RED!\n");
  delay(2000);
}

void loop() {
  if (current_pin_index >= num_pins) {
    Serial.println("\n=== Scan Complete ===");
    Serial.println("Did any pin light up your LEDs?");
    Serial.println("If not, check:");
    Serial.println("  - LED strip is getting 5V external power");
    Serial.println("  - Common ground between ESP32 and power supply");
    Serial.println("  - Data wire is connected");
    Serial.println("  - LED strip is functional");
    Serial.println("\nRestarting scan in 10 seconds...");
    delay(10000);
    current_pin_index = 0;
    return;
  }

  int pin = test_pins[current_pin_index];

  Serial.print("Testing GPIO ");
  Serial.print(pin);
  Serial.print(" ... ");
  Serial.flush();

  // Clean up previous strip if exists
  if (strip != nullptr) {
    strip->clear();
    strip->show();
    delete strip;
    delay(100);
  }

  // Create new strip on current pin
  strip = new Adafruit_NeoPixel(NUM_PIXELS, pin, NEO_GRB + NEO_KHZ800);
  strip->begin();
  strip->setBrightness(50);

  // Set all pixels to RED
  for(int i = 0; i < NUM_PIXELS; i++) {
    strip->setPixelColor(i, 255, 0, 0);  // Bright red
  }
  strip->show();

  Serial.println("(LEDs should be RED now if this is the right pin)");

  // Wait 3 seconds for observation
  delay(3000);

  // Turn off
  strip->clear();
  strip->show();

  // Short delay before next pin
  delay(500);

  current_pin_index++;
}
