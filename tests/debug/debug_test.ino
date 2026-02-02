/*
 * LED DEBUG TEST
 * Tests basic LED strip connectivity with serial output
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12
#define NUM_PIXELS 10  // Test with just 10

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  Serial.println("=== LED DEBUG TEST ===");
  Serial.print("LED Pin: ");
  Serial.println(LED_PIN);
  Serial.print("Num LEDs: ");
  Serial.println(NUM_PIXELS);

  strip.begin();
  strip.setBrightness(255);
  strip.show(); // Initialize all pixels to 'off'

  Serial.println("Strip initialized!");
}

void loop() {
  // Test 1: All red
  Serial.println("Test 1: All RED (brightness 50)");
  for(int i=0; i<NUM_PIXELS; i++) {
    strip.setPixelColor(i, 50, 0, 0);
  }
  strip.show();
  delay(2000);

  // Test 2: All green
  Serial.println("Test 2: All GREEN (brightness 50)");
  for(int i=0; i<NUM_PIXELS; i++) {
    strip.setPixelColor(i, 0, 50, 0);
  }
  strip.show();
  delay(2000);

  // Test 3: All blue
  Serial.println("Test 3: All BLUE (brightness 50)");
  for(int i=0; i<NUM_PIXELS; i++) {
    strip.setPixelColor(i, 0, 0, 50);
  }
  strip.show();
  delay(2000);

  // Test 4: All white
  Serial.println("Test 4: All WHITE (brightness 30)");
  for(int i=0; i<NUM_PIXELS; i++) {
    strip.setPixelColor(i, 30, 30, 30);
  }
  strip.show();
  delay(2000);

  // Test 5: All off
  Serial.println("Test 5: All OFF");
  for(int i=0; i<NUM_PIXELS; i++) {
    strip.setPixelColor(i, 0, 0, 0);
  }
  strip.show();
  delay(2000);

  Serial.println("---");
}
