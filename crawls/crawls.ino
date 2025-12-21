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

  // softWhiteCrawlAnimation(); delay(50);
  // helixAnimation(); delay(30);
  // sineWaveAnimation(); delay(40);
  // diagonalCrawlSimpleAnimation(); delay(40);
  diagonalCrawlAnimation(); delay(40);
}

// ===== Animation Wrappers =====
// These wrappers document the parameters for each animation

void softWhiteCrawlAnimation() {
  float waveWidth = 4.0;  // How many LEDs the gradient spans
  softWhiteCrawl(waveWidth);
}

void helixAnimation() {
  float helixTurns = 0.5;  // Number of complete rotations over strip length
  float speed = 0.08;      // How fast it moves across strips per frame
  helix(helixTurns, speed);
}

void sineWaveAnimation() {
  float frequency = 2.0;    // How many complete waves fit in the strip length
  float trailLength = 10.0; // How many LEDs of fading trail
  float speed = 0.5;        // How fast the wave moves per frame
  sineWave(frequency, trailLength, speed);
}

void diagonalCrawlSimpleAnimation() {
  float trailLength = 8.0;  // How many LEDs of fading trail
  float pauseLength = 10.0; // Pause duration at edges (in phase units)
  float speed = 0.5;        // How fast the pulse moves per frame
  bool wrapMode = false;    // true = wrap around, false = bounce back
  float frequency = 3.3;    // How many times it crosses all strips (1.0 = once)
  diagonalCrawlSimple(trailLength, pauseLength, speed, wrapMode, frequency);
}

void diagonalCrawlAnimation() {
  float trailLength = 8.0;  // How many LEDs of fading trail
  float pauseLength = 10.0; // Pause duration at edges (in phase units)
  float speed = 0.5;        // How fast the pulse moves per frame
  bool wrapMode = false;    // true = wrap around, false = bounce back
  float frequency = 3.3;    // How many times it crosses all strips (1.0 = once)
  diagonalCrawl(trailLength, pauseLength, speed, wrapMode, frequency);
}

// ===== Core Animation Functions =====


// Soft white crawl effect - a gentle wave of warm white light
// waveWidth: how many LEDs the gradient spans
void softWhiteCrawl(float waveWidth) {
  static int offset = 0;  // Track position of the wave

  // Soft warm white color (slightly yellowish for warmth)
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

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
    offset = -(int)waveWidth;  // Reset to start just before the strip
  }
}

// Helix effect - soft white
// helixTurns: number of complete rotations over strip length
// speed: how fast it moves across strips per frame
void helix(float helixTurns, float speed) {
  static float stripOffset = 0.0;  // Horizontal position (travels across strips)

  // Clear all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->clear();
  }

  // Helix parameters
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
  stripOffset += speed;

  // Wrap strip offset
  if (stripOffset >= NUM_STRIPS) {
    stripOffset -= NUM_STRIPS;
  }
}

// Sine wave - travels across strips with fading trail
// frequency: how many complete waves fit in the strip length
// trailLength: how many LEDs of fading trail
// speed: how fast the wave moves per frame
void sineWave(float frequency, float trailLength, float speed) {
  static float phase = 0.0;  // Wave position/phase

  // Clear all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->clear();
  }

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
  phase += speed;

  // Wrap phase
  if (phase >= NUM_PIXELS) {
    phase -= NUM_PIXELS;
  }
}

// Diagonal crawl (simple single-pulse) - linear diagonal line traveling across strips with fading trail
// trailLength: how many LEDs of fading trail
// pauseLength: pause duration at edges (in phase units)
// speed: how fast the pulse moves per frame
// wrapMode: true = wrap around, false = bounce back
// frequency: how many times the pattern crosses all strips over the LED length (1.0 = once)
void diagonalCrawlSimple(float trailLength, float pauseLength, float speed, bool wrapMode, float frequency) {
  static float phase = 0.0;  // Current position
  static int direction = 1;  // 1 = forward, -1 = backward (for bounce mode)

  // Clear all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->clear();
  }

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

      // Calculate where this point in the trail was (for cross-strip fading)
      float trailPhase = phase - distanceFromLead * direction;

      // Calculate which strip this position maps to (diagonal line)
      float stripFloat;
      if (wrapMode || direction == 1) {
        // Forward direction or wrap mode
        stripFloat = fmod(trailPhase * NUM_STRIPS * frequency / (float)NUM_PIXELS, NUM_STRIPS);
      } else {
        // Backward direction in bounce mode - reverse the strip order
        float temp = fmod(trailPhase * NUM_STRIPS * frequency / (float)NUM_PIXELS, NUM_STRIPS);
        stripFloat = fmod(NUM_STRIPS - 1 - temp + NUM_STRIPS, NUM_STRIPS);
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
  phase += speed * direction;

  // Wrap or bounce based on mode (with pause at edges)
  if (wrapMode) {
    // Wrap mode - continuous loop with pause
    if (phase >= NUM_PIXELS + pauseLength) {
      phase = -pauseLength;
    }
    if (phase < -pauseLength) {
      phase = NUM_PIXELS + pauseLength - speed;
    }
  } else {
    // Bounce mode - reverse direction at ends with pause
    if (phase >= NUM_PIXELS + pauseLength) {
      phase = NUM_PIXELS + pauseLength - speed;
      direction = -1;
    }
    if (phase < -pauseLength) {
      phase = -pauseLength + speed;
      direction = 1;
    }
  }
}

// Diagonal crawl (multi-pulse) - linear diagonal line traveling across strips with fading trail
// trailLength: how many LEDs of fading trail
// pauseLength: pause duration at edges (in phase units)
// speed: how fast the pulse moves per frame
// wrapMode: true = wrap around, false = bounce back
// frequency: how many times the pattern crosses all strips over the LED length (1.0 = once)
void diagonalCrawl(float trailLength, float pauseLength, float speed, bool wrapMode, float frequency) {
  #define NUM_PULSES 3
  #define TRAIL_HISTORY_SIZE 20

  // Multiple pulses - synchronized direction to avoid intersections
  static float phases[NUM_PULSES] = {0.0, 16.67, 33.33};  // Evenly spaced
  static int direction = 1;  // Single direction for all pulses (synchronized)
  static float phaseHistories[NUM_PULSES][TRAIL_HISTORY_SIZE];
  static int historyIndices[NUM_PULSES] = {0, 0, 0};
  static bool historyFilleds[NUM_PULSES] = {false, false, false};

  // Store current phases in history for all pulses
  for (int p = 0; p < NUM_PULSES; p++) {
    phaseHistories[p][historyIndices[p]] = phases[p];
    historyIndices[p] = (historyIndices[p] + 1) % TRAIL_HISTORY_SIZE;
    if (historyIndices[p] == 0) historyFilleds[p] = true;
  }

  // Clear all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->clear();
  }

  // Soft warm white color
  int warmWhiteR = 255;
  int warmWhiteG = 240;
  int warmWhiteB = 200;

  // Draw diagonal line using position history for smooth trailing
  // Process all pulses and accumulate maximum brightness
  for (int i = 0; i < NUM_PIXELS; i++) {
    float maxBrightness = 0.0;
    float stripFloat = 0.0;

    // Check all pulses
    for (int p = 0; p < NUM_PULSES; p++) {
      // Check all history positions for this pulse
      int historyLength = historyFilleds[p] ? TRAIL_HISTORY_SIZE : historyIndices[p];
      for (int h = 0; h < historyLength && h < trailLength; h++) {
        int histIdx = (historyIndices[p] - 1 - h + TRAIL_HISTORY_SIZE) % TRAIL_HISTORY_SIZE;
        float histPhase = phaseHistories[p][histIdx];

        // Calculate distance from this historical position
        float distance = abs(i - histPhase);

        // Calculate brightness based on distance (temporal fade based on history position)
        if (distance < 1.0) {
          float brightness = (1.0 - (h / trailLength)) * (1.0 - distance);
          if (brightness > maxBrightness) {
            maxBrightness = brightness;

            // Calculate which strip this position maps to (diagonal line)
            // Use histPhase (where the pulse actually is) not i (which LED we're drawing)
            // Frequency controls how many times the pattern repeats over the strip length
            if (wrapMode || direction == 1) {
              // Forward direction or wrap mode
              stripFloat = fmod(histPhase * NUM_STRIPS * frequency / (float)NUM_PIXELS, NUM_STRIPS);
            } else {
              // Backward direction in bounce mode - reverse the strip order
              // Add NUM_STRIPS before final fmod to prevent negative values
              float temp = fmod(histPhase * NUM_STRIPS * frequency / (float)NUM_PIXELS, NUM_STRIPS);
              stripFloat = fmod(NUM_STRIPS - 1 - temp + NUM_STRIPS, NUM_STRIPS);
            }
          }
        }
      }
    }

    // Only draw if brightness is significant
    if (maxBrightness > 0.01) {
      int primaryStrip = (int)stripFloat;
      int secondaryStrip = (primaryStrip + 1) % NUM_STRIPS;
      float blend = stripFloat - primaryStrip;

      // Apply brightness to warm white
      int r = warmWhiteR * maxBrightness;
      int g = warmWhiteG * maxBrightness;
      int b = warmWhiteB * maxBrightness;

      // Set color on primary strip
      if (primaryStrip >= 0 && primaryStrip < NUM_STRIPS) {
        strips[primaryStrip]->setPixelColor(i, strips[primaryStrip]->Color(r, g, b));
      }

      // Blend to secondary strip for smooth transitions
      if (blend > 0.1 && secondaryStrip < NUM_STRIPS) {
        int r2 = warmWhiteR * maxBrightness * blend;
        int g2 = warmWhiteG * maxBrightness * blend;
        int b2 = warmWhiteB * maxBrightness * blend;
        strips[secondaryStrip]->setPixelColor(i, strips[secondaryStrip]->Color(r2, g2, b2));
      }
    }
  }

  // Show all strips
  for (int strip = 0; strip < NUM_STRIPS; strip++) {
    strips[strip]->show();
  }

  // Move all pulses forward or backward (synchronized)
  for (int p = 0; p < NUM_PULSES; p++) {
    phases[p] += speed * direction;
  }

  // Check if all pulses have passed the boundary (synchronized direction change)
  if (wrapMode) {
    // Wrap mode - wrap all pulses when they pass boundaries
    for (int p = 0; p < NUM_PULSES; p++) {
      if (phases[p] >= NUM_PIXELS + pauseLength) {
        phases[p] -= (NUM_PIXELS + 2 * pauseLength);
      }
      if (phases[p] < -pauseLength) {
        phases[p] += (NUM_PIXELS + 2 * pauseLength);
      }
    }
  } else {
    // Bounce mode - reverse all pulses together when the trailing one finishes its pause
    // Find the trailing pulse (the one furthest behind)
    float trailingPhase = phases[0];
    for (int p = 1; p < NUM_PULSES; p++) {
      if (direction == 1) {
        // Going forward: trailing is the smallest (furthest behind)
        trailingPhase = min(trailingPhase, phases[p]);
      } else {
        // Going backward: trailing is the largest (furthest behind)
        trailingPhase = max(trailingPhase, phases[p]);
      }
    }

    // Reverse when the trailing pulse completes its pause off-screen
    if (direction == 1 && trailingPhase >= NUM_PIXELS + pauseLength) {
      direction = -1;
    } else if (direction == -1 && trailingPhase <= -pauseLength) {
      direction = 1;
    }
  }
}


