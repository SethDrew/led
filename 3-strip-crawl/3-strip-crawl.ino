#include <Adafruit_NeoPixel.h>

// Define pins for each LED strip
#define STRIP1_PIN 6
#define STRIP2_PIN 7
#define STRIP3_PIN 8

// Define the number of NeoPixels per strip
#define NUM_PIXELS 30

// Create NeoPixel objects for each strip
Adafruit_NeoPixel strip1(NUM_PIXELS, STRIP1_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip2(NUM_PIXELS, STRIP2_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip3(NUM_PIXELS, STRIP3_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  Serial.println("Multi-Strip Soft White Crawl");

  // Initialize all three strips
  strip1.begin();
  strip1.setBrightness(200);
  strip1.show();

  strip2.begin();
  strip2.setBrightness(200);
  strip2.show();

  strip3.begin();
  strip3.setBrightness(200);
  strip3.show();
}

void loop() {
  // Continuous soft white crawl across all strips
  softWhiteCrawl(50); 
}

// Soft white crawl effect - a gentle wave of warm white light
void softWhiteCrawl(int wait) {
  static int offset = 0;  // Track position of the wave

  // Soft warm white color (slightly yellowish for warmth)
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

  // Wave width - how many LEDs the gradient spans
  float waveWidth = 4.0;

  for(int i = 0; i < NUM_PIXELS; i++) {
    // Calculate distance from current LED to wave center (no wrapping)
    float distance = abs(i - offset);

    // Create soft gradient using cosine for smooth falloff
    float brightness = 0.0;
    if (distance <= waveWidth) {
      brightness = max(0.0, cos(distance / waveWidth * 3.14159 / 2));
    }

    // Apply brightness to warm white color
    int r = warmWhiteR * brightness;
    int g = warmWhiteG * brightness;
    int b = warmWhiteB * brightness;

    // Set same color on all three strips (synchronized)
    strip1.setPixelColor(i, strip1.Color(r, g, b));
    strip2.setPixelColor(i, strip2.Color(r, g, b));
    strip3.setPixelColor(i, strip3.Color(r, g, b));
  }

  // Update all strips
  strip1.show();
  strip2.show();
  strip3.show();

  // Move the wave forward
  offset++;

  // Check if wave has fully exited the strip (trailing edge cleared)
  if (offset > NUM_PIXELS + (int)waveWidth) {
    delay(3000);  // Pause after wave fully exits
    offset = -(int)waveWidth;  // Reset to start just before the strip
  }

  delay(wait);
}
