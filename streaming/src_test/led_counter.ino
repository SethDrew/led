/*
 * LED Counter - Color changes every 10 LEDs
 * Pin D11 - helps count total LEDs on strip
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 11
#define MAX_LEDS 110  // Full range for counting

Adafruit_NeoPixel strip(MAX_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

// Color palette - distinct colors for counting
uint32_t colors[] = {
  strip.Color(255, 0, 0),     // 0-9: Red
  strip.Color(0, 255, 0),     // 10-19: Green
  strip.Color(0, 0, 255),     // 20-29: Blue
  strip.Color(255, 255, 0),   // 30-39: Yellow
  strip.Color(0, 255, 255),   // 40-49: Cyan
  strip.Color(255, 0, 255),   // 50-59: Magenta
  strip.Color(255, 255, 255), // 60-69: White
  strip.Color(255, 128, 0),   // 70-79: Orange
  strip.Color(128, 0, 255),   // 80-89: Purple
  strip.Color(128, 255, 0),   // 90-99: Lime
  strip.Color(255, 0, 128),   // 100-109: Pink
  strip.Color(0, 255, 128)    // 110-119: Aqua
};

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("LED Counter - Pin D11");
  Serial.println("Colors change every 10 LEDs");
  Serial.println("Count the color blocks to find total LEDs");
  Serial.println();

  strip.begin();
  strip.setBrightness(128);  // 50% brightness
  strip.clear();

  // Set colors - every 10 LEDs changes color
  for (int i = 0; i < MAX_LEDS; i++) {
    int colorIndex = i / 10;  // 0-9 = 0, 10-19 = 1, etc.
    strip.setPixelColor(i, colors[colorIndex]);
  }

  strip.show();

  Serial.println("LED pattern set:");
  Serial.println("  0-9: Red");
  Serial.println("  10-19: Green");
  Serial.println("  20-29: Blue");
  Serial.println("  30-39: Yellow");
  Serial.println("  40-49: Cyan");
  Serial.println("  50-59: Magenta");
  Serial.println("  60-69: White");
  Serial.println("  70-79: Orange");
  Serial.println("  80-89: Purple");
  Serial.println("  90-99: Lime");
  Serial.println("  100-109: Pink");
  Serial.println("  110-119: Aqua");
}

void loop() {
  // Just hold the pattern
  delay(1000);
}
