#include <Adafruit_NeoPixel.h>

// Define pins for each LED strip
#define STRIP1_PIN 6
#define STRIP2_PIN 7
#define STRIP3_PIN 8
#define STRIP4_PIN 9
#define STRIP5_PIN 10

// Define the number of NeoPixels per strip
#define NUM_PIXELS 30
#define NUM_STRIPS 5

// Create NeoPixel objects for each strip
Adafruit_NeoPixel strip1(NUM_PIXELS, STRIP1_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip2(NUM_PIXELS, STRIP2_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip3(NUM_PIXELS, STRIP3_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip4(NUM_PIXELS, STRIP4_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip5(NUM_PIXELS, STRIP5_PIN, NEO_GRB + NEO_KHZ800);

// Array of strip pointers for easier iteration
Adafruit_NeoPixel* strips[NUM_STRIPS] = {&strip1, &strip2, &strip3, &strip4, &strip5};

void setup() {
  Serial.begin(115200);
  Serial.println("5-Strip LED Animations");

  // Initialize all five strips
  for (int i = 0; i < NUM_STRIPS; i++) {
    strips[i]->begin();
    strips[i]->setBrightness(200);
    strips[i]->show();
  }
}

void loop() {
  // Switch between animations
  // Comment/uncomment to choose:

  // softWhiteCrawl(50);  // Original synchronized crawl
  // helix(30);  // Helix animation
  // sineWave(40);  // Sine wave traveling across strips
  diagonalCrawl(40, false, 3.5);  // Diagonal line traveling across strips
  // Parameters: (wait_ms, wrap_mode, frequency)
  // wrap_mode: true = wrap around, false = bounce back
  // frequency: number of times it crosses all strips over the LED length (1.0 = once)
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

    // Set same color on all strips (synchronized)
    for (int strip = 0; strip < NUM_STRIPS; strip++) {
      strips[strip]->setPixelColor(i, strips[strip]->Color(r, g, b));
    }
  }

  // Update all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->show();
  }

  // Move the wave forward
  offset++;

  // Check if wave has fully exited the strip (trailing edge cleared)
  if (offset > NUM_PIXELS + (int)waveWidth) {
    delay(3000);  // Pause after wave fully exits
    offset = -(int)waveWidth;  // Reset to start just before the strip
  }

  delay(wait);
}

// Helix effect - soft white 
void helix(int wait) {
  static float stripOffset = 0.0;  // Horizontal position (travels across strips)

  // Clear all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->clear();
  }

  // Helix parameters
  float helixTurns = .5;  // Number of complete rotations over strip length
  float waveWidth = 1.5;   // Width of the glowing section (in strips)

  // Soft warm white color (same as crawl)
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

  // Draw helix - for each LED height, calculate which strip(s) should be lit
  for (int i = 0; i < NUM_PIXELS; i++) {
    // Calculate which strip should be lit at this height
    // The spiral rotates as you go up the LEDs
    float angle = (i * helixTurns * 2.0 * PI / NUM_PIXELS);

    // Convert angle to strip index (0-4.99...)
    float baseStrip = (sin(angle) + 1.0) / 2.0 * (NUM_STRIPS - 0.01);

    // Add the offset to make it travel across strips
    float targetStrip = baseStrip + stripOffset;

    // Wrap the strip index
    while (targetStrip >= NUM_STRIPS) targetStrip -= NUM_STRIPS;
    while (targetStrip < 0) targetStrip += NUM_STRIPS;

    // Light up the primary strip and blend to adjacent strips
    for (int strip = 0; strip < NUM_STRIPS; strip++) {
      // Calculate distance from this strip to the target strip (with wrapping)
      float distance = strip - targetStrip;
      if (distance > NUM_STRIPS / 2.0) distance -= NUM_STRIPS;
      if (distance < -NUM_STRIPS / 2.0) distance += NUM_STRIPS;
      distance = abs(distance);

      // Calculate brightness based on distance
      float brightness = 0.0;
      if (distance <= waveWidth) {
        brightness = (cos(distance / waveWidth * PI) + 1.0) / 2.0;
      }

      if (brightness > 0.05) {
        // Apply brightness to warm white color
        int r = warmWhiteR * brightness;
        int g = warmWhiteG * brightness;
        int b = warmWhiteB * brightness;

        strips[strip]->setPixelColor(i, strips[strip]->Color(r, g, b));
      }
    }
  }

  // Show all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->show();
  }

  // Move helix across strips
  stripOffset += 0.08;

  // Wrap strip offset
  if (stripOffset >= NUM_STRIPS) {
    stripOffset -= NUM_STRIPS;
  }

  delay(wait);
}

// Sine wave - travels across strips with fading trail
void sineWave(int wait) {
  static float phase = 0.0;  // Wave position/phase

  // Clear all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->clear();
  }

  // Sine wave parameters
  float frequency = 2.0;      // How many complete waves fit in the strip length
  float trailLength = 10.0;   // How many LEDs of fading trail (configurable)

  // Soft warm white color
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

  // Draw sine wave
  for (int i = 0; i < NUM_PIXELS; i++) {
    // Calculate distance from the lead (current position)
    // Only draw LEDs that are behind or at the lead
    float distanceFromLead = phase - i;

    // Only draw if within trail length (fading behind the lead)
    if (distanceFromLead >= 0 && distanceFromLead <= trailLength) {
      // Calculate brightness - fades as we get further from lead
      float brightness = 1.0 - (distanceFromLead / trailLength);
      brightness = max(0.0, brightness);

      // Calculate sine wave value at this position
      float angle = (i + phase) * frequency * 2.0 * PI / NUM_PIXELS;
      float sineValue = sin(angle);

      // Map sine value (-1 to 1) to strip index (0 to 4)
      float stripFloat = (sineValue + 1.0) / 2.0 * (NUM_STRIPS - 0.01);
      int primaryStrip = (int)stripFloat;
      int secondaryStrip = (primaryStrip + 1) % NUM_STRIPS;
      float blend = stripFloat - primaryStrip;

      // Apply brightness to warm white
      int r = warmWhiteR * brightness;
      int g = warmWhiteG * brightness;
      int b = warmWhiteB * brightness;

      // Set color on primary strip
      if (primaryStrip >= 0 && primaryStrip < NUM_STRIPS) {
        strips[primaryStrip]->setPixelColor(i, strips[primaryStrip]->Color(r, g, b));
      }

      // Blend to secondary strip for smooth transitions
      if (blend > 0.1 && secondaryStrip < NUM_STRIPS) {
        int r2 = warmWhiteR * brightness * blend;
        int g2 = warmWhiteG * brightness * blend;
        int b2 = warmWhiteB * brightness * blend;
        strips[secondaryStrip]->setPixelColor(i, strips[secondaryStrip]->Color(r2, g2, b2));
      }
    }
  }

  // Show all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->show();
  }

  // Move wave forward
  phase += 0.5;

  // Wrap phase
  if (phase >= NUM_PIXELS) {
    phase -= NUM_PIXELS;
  }

  delay(wait);
}

// Diagonal crawl - linear diagonal line traveling across strips with fading trail
// wrapMode: true = wrap around, false = bounce back
// frequency: how many times the pattern crosses all strips over the LED length (1.0 = once)
void diagonalCrawl(int wait, bool wrapMode, float frequency) {
  static float phase = 0.0;  // Current position
  static int direction = 1;  // 1 = forward, -1 = backward (for bounce mode)

  // Clear all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->clear();
  }

  // Diagonal parameters
  float trailLength = 8.0;   // How many LEDs of fading trail (configurable)

  // Soft warm white color
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

  // Draw diagonal line
  for (int i = 0; i < NUM_PIXELS; i++) {
    // Calculate distance from the lead (current position)
    float distanceFromLead;

    if (direction == 1) {
      // Forward: trail behind the lead
      distanceFromLead = phase - i;
    } else {
      // Backward: trail behind the lead (in reverse)
      distanceFromLead = i - phase;
    }

    // Only draw if within trail length (fading behind the lead)
    if (distanceFromLead >= 0 && distanceFromLead <= trailLength) {
      // Calculate brightness - fades as we get further from lead
      float brightness = 1.0 - (distanceFromLead / trailLength);
      brightness = max(0.0, brightness);

      // Calculate which strip this LED position maps to (diagonal line)
      // Frequency controls how many times the pattern repeats over the strip length
      float stripFloat;
      if (wrapMode || direction == 1) {
        // Forward direction or wrap mode
        stripFloat = fmod(i * NUM_STRIPS * frequency / (float)NUM_PIXELS, NUM_STRIPS);
      } else {
        // Backward direction in bounce mode - reverse the strip order
        stripFloat = NUM_STRIPS - 1 - fmod(i * NUM_STRIPS * frequency / (float)NUM_PIXELS, NUM_STRIPS);
      }

      int primaryStrip = (int)stripFloat;
      int secondaryStrip = (primaryStrip + 1) % NUM_STRIPS;
      float blend = stripFloat - primaryStrip;

      // Apply brightness to warm white
      int r = warmWhiteR * brightness;
      int g = warmWhiteG * brightness;
      int b = warmWhiteB * brightness;

      // Set color on primary strip
      if (primaryStrip >= 0 && primaryStrip < NUM_STRIPS) {
        strips[primaryStrip]->setPixelColor(i, strips[primaryStrip]->Color(r, g, b));
      }

      // Blend to secondary strip for smooth transitions
      if (blend > 0.1 && secondaryStrip < NUM_STRIPS) {
        int r2 = warmWhiteR * brightness * blend;
        int g2 = warmWhiteG * brightness * blend;
        int b2 = warmWhiteB * brightness * blend;
        strips[secondaryStrip]->setPixelColor(i, strips[secondaryStrip]->Color(r2, g2, b2));
      }
    }
  }

  // Show all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->show();
  }

  // Move forward or backward
  phase += 0.5 * direction;

  // Wrap or bounce based on mode
  if (wrapMode) {
    // Wrap mode - continuous loop
    if (phase >= NUM_PIXELS) {
      phase -= NUM_PIXELS;
    }
    if (phase < 0) {
      phase += NUM_PIXELS;
    }
  } else {
    // Bounce mode - reverse direction at ends
    if (phase >= NUM_PIXELS) {
      phase = NUM_PIXELS - 0.5;
      direction = -1;
    }
    if (phase < 0) {
      phase = 0.5;
      direction = 1;
    }
  }

  delay(wait);
}


