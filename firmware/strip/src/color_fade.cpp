/*
 * Traveling Rainbow — per-channel gamma correction experiment
 * Pot on A3 controls position. 60-LED rainbow travels from start to end.
 * Custom per-channel gamma for richer warm tones (orange, teal, etc.)
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12
#define POT_PIN A3

#ifndef NUM_PIXELS
#define NUM_PIXELS 150
#endif

#define RAINBOW_LENGTH 60

// Per-channel gamma values - tweak these to fix color issues
// Standard is ~2.2, but we can adjust per channel for better warm tones
#define GAMMA_RED   2.2
#define GAMMA_GREEN 2.4  // Higher gamma = darker green = better orange/warm tones
#define GAMMA_BLUE  2.2

// White balance scale factors (0.0-1.0) - adjust if white looks tinted
#define BALANCE_RED   1.0
#define BALANCE_GREEN 1.0
#define BALANCE_BLUE  0.95  // Slightly reduce blue if strip looks too cool

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

float potSmoothed = 0;
int potLast = -1;

// Apply per-channel gamma correction with white balance
uint32_t gammaCorrect(uint8_t r, uint8_t g, uint8_t b) {
  // Normalize to 0.0-1.0
  float rf = r / 255.0;
  float gf = g / 255.0;
  float bf = b / 255.0;

  // Apply white balance FIRST (on linear values)
  rf *= BALANCE_RED;
  gf *= BALANCE_GREEN;
  bf *= BALANCE_BLUE;

  // Apply per-channel gamma correction
  rf = pow(rf, GAMMA_RED);
  gf = pow(gf, GAMMA_GREEN);
  bf = pow(bf, GAMMA_BLUE);

  // Convert back to 0-255
  uint8_t r_out = (uint8_t)(rf * 255.0 + 0.5);
  uint8_t g_out = (uint8_t)(gf * 255.0 + 0.5);
  uint8_t b_out = (uint8_t)(bf * 255.0 + 0.5);

  return strip.Color(r_out, g_out, b_out);
}

void setup() {
  strip.begin();
  strip.setBrightness(255);
  strip.clear();
  strip.show();
  potSmoothed = analogRead(POT_PIN);
  potLast = (int)(potSmoothed + 0.5);
}

void loop() {
  potSmoothed += (analogRead(POT_PIN) - potSmoothed) * 0.4;
  int potMapped = (int)(potSmoothed + 0.5);
  if (abs(potMapped - potLast) > 3) {
    potLast = potMapped;
  }

  // Pot controls rainbow position
  int maxTravel = NUM_PIXELS - RAINBOW_LENGTH;
  int rainbowStart = (int)((uint32_t)potLast * maxTravel / 1023);

  // Clear entire strip first
  strip.clear();

  // Draw 60-LED rainbow starting at rainbowStart
  for (int i = 0; i < RAINBOW_LENGTH; i++) {
    int ledPos = rainbowStart + i;
    if (ledPos >= 0 && ledPos < NUM_PIXELS) {
      // Full rainbow across 60 LEDs
      uint16_t hue = (uint16_t)((uint32_t)i * 65536 / RAINBOW_LENGTH);

      // Convert HSV to RGB (Adafruit does this internally, we need raw RGB)
      uint32_t colorHSV = strip.ColorHSV(hue, 255, 128);  // 50% brightness for camera
      uint8_t r = (colorHSV >> 16) & 0xFF;
      uint8_t g = (colorHSV >> 8) & 0xFF;
      uint8_t b = colorHSV & 0xFF;

      // Apply our custom per-channel gamma correction
      uint32_t color = gammaCorrect(r, g, b);
      strip.setPixelColor(ledPos, color);
    }
  }

  strip.show();
  delay(25);
}
