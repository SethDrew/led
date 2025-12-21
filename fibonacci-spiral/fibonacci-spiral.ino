#include <Adafruit_NeoPixel.h>

// Fibonacci spiral configuration
// We'll use 10 LED strip segments arranged in a golden spiral pattern
#define NUM_SEGMENTS 10

// Pin definitions for each segment
const int segmentPins[NUM_SEGMENTS] = {6, 7, 8, 9, 10, 11, 12, 13, A0, A1};

// Number of LEDs per segment (increases as spiral grows)
const int segmentLengths[NUM_SEGMENTS] = {5, 8, 10, 13, 16, 20, 24, 28, 32, 36};

// Create NeoPixel objects for each segment
Adafruit_NeoPixel segments[NUM_SEGMENTS];

// Total number of LEDs
int totalLEDs = 0;

// Depth mapping - each segment gets sequential depths
struct SegmentDepth {
  int segmentIndex;
  int ledIndex;
  int depth;
};

#define MAX_LEDS 200
SegmentDepth depthMap[MAX_LEDS];
int maxDepth = 0;

// Animation state
int animationOffset = 0;
int animationDirection = 1;

void setup() {
  Serial.begin(115200);
  Serial.println("Fibonacci Spiral - Linear LED Strips");

  // Initialize all segments
  int currentDepth = 0;
  for (int i = 0; i < NUM_SEGMENTS; i++) {
    segments[i] = Adafruit_NeoPixel(segmentLengths[i], segmentPins[i], NEO_GRB + NEO_KHZ800);
    segments[i].begin();
    segments[i].setBrightness(200);
    segments[i].clear();
    segments[i].show();

    // Map depths for this segment
    for (int j = 0; j < segmentLengths[i]; j++) {
      depthMap[totalLEDs].segmentIndex = i;
      depthMap[totalLEDs].ledIndex = j;
      depthMap[totalLEDs].depth = currentDepth;
      totalLEDs++;
      currentDepth++;
    }
  }

  maxDepth = currentDepth - 1;

  Serial.print("Initialized ");
  Serial.print(NUM_SEGMENTS);
  Serial.print(" segments with ");
  Serial.print(totalLEDs);
  Serial.print(" total LEDs (max depth: ");
  Serial.print(maxDepth);
  Serial.println(")");
}

void loop() {
  // Spiral wave animation
  spiralWave(30);
}

// Animate a rainbow wave traveling along the spiral
void spiralWave(int wait) {
  // Clear all segments
  for (int i = 0; i < NUM_SEGMENTS; i++) {
    segments[i].clear();
  }

  // Wave parameters
  int waveLength = 15;

  // Apply wave based on depth
  for (int i = 0; i < totalLEDs; i++) {
    int depth = depthMap[i].depth;
    int distance = abs(depth - animationOffset);

    if (distance < waveLength) {
      // Calculate brightness based on distance (wave shape)
      float brightness = (cos(distance * PI / waveLength) + 1.0) / 2.0;

      // Calculate hue based on position in spiral (rainbow effect)
      int hue = (depth * 65536L / maxDepth) % 65536;

      // Convert HSV to RGB with brightness
      uint32_t color = segments[0].ColorHSV(hue, 255, (int)(brightness * 255));

      // Set the LED
      int seg = depthMap[i].segmentIndex;
      int led = depthMap[i].ledIndex;
      segments[seg].setPixelColor(led, color);
    }
  }

  // Show all segments
  for (int i = 0; i < NUM_SEGMENTS; i++) {
    segments[i].show();
  }

  // Move wave forward
  animationOffset += animationDirection;

  // Reverse direction at ends
  if (animationOffset >= maxDepth + waveLength) {
    animationDirection = -1;
    delay(500);
  } else if (animationOffset < -waveLength) {
    animationDirection = 1;
    delay(500);
  }

  delay(wait);
}
