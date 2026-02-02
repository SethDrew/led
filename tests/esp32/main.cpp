/*
 * ESP32 LED Test
 * Cycles through colors to test LED strip functionality
 */

#include <Adafruit_NeoPixel.h>

#ifndef LED_PIN
#define LED_PIN 16  // GPIO 16
#endif

#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

// Forward declarations
void colorWipe(uint32_t color, int wait);
uint32_t Wheel(byte WheelPos);

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== ESP32 LED Simple Test ===");
  Serial.print("LED Pin: GPIO ");
  Serial.println(LED_PIN);
  Serial.print("Num LEDs: ");
  Serial.println(NUM_PIXELS);

  strip.begin();
  strip.setBrightness(50);  // Lower brightness for initial test

  // Skip first pixel (sacrificial), set pixels 1-10 to RED
  Serial.println("Skipping first pixel, setting pixels 1-10 to RED...");
  strip.setPixelColor(0, 0, 0, 0);  // First pixel OFF (sacrificial)
  for(int i=1; i<=10; i++) {
    strip.setPixelColor(i, 255, 0, 0);  // Bright red
  }
  strip.show();

  Serial.println("Test complete - LEDs should be RED");
}

void loop() {
  // Do nothing - just keep LEDs red
  delay(1000);
  Serial.println(".");  // Heartbeat
}

// Fill strip with a color (one pixel at a time)
void colorWipe(uint32_t color, int wait) {
  for(int i=0; i<NUM_PIXELS; i++) {
    strip.setPixelColor(i, color);
    strip.show();
    delay(wait);
  }
}

// Rainbow wheel (input 0-255)
uint32_t Wheel(byte WheelPos) {
  WheelPos = 255 - WheelPos;
  if(WheelPos < 85) {
    return strip.Color(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if(WheelPos < 170) {
    WheelPos -= 85;
    return strip.Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return strip.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
}
