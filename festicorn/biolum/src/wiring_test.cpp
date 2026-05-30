// Biolum wiring test — 6 strips, each an alternating color pair.
// Walk the install pin-by-pin to verify data lines and label GPIO.
//
// Pin → pattern (alternating every 10 LEDs):
//   4  RED/YELLOW      16 GREEN/CYAN      17 BLUE/MAGENTA
//   5  YELLOW/WHITE    18 CYAN/RED        19 MAGENTA/GREEN

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 30
#endif

#define CHUNK 10

struct StripDef {
  uint8_t  pin;
  uint32_t colorA;
  uint32_t colorB;
  const char* name;
};

static const StripDef STRIPS[] = {
  {  4, 0x400000, 0x402000, "RED/YELLOW"     },
  { 15, 0x004000, 0x004040, "GREEN/CYAN"     },
  { 17, 0x000040, 0x400040, "BLUE/MAGENTA"   },
  {  5, 0x402000, 0x404040, "YELLOW/WHITE"   },
  { 18, 0x004040, 0x400000, "CYAN/RED"       },
  { 19, 0x400040, 0x004000, "MAGENTA/GREEN"  },
};
static const size_t NUM_STRIPS = sizeof(STRIPS) / sizeof(STRIPS[0]);

static Adafruit_NeoPixel* strips[NUM_STRIPS];

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("biolum wiring test — color pairs");
  Serial.printf("  %u strips, %u LEDs each, %u LEDs per chunk\n",
                (unsigned)NUM_STRIPS, LEDS_PER_STRIP, CHUNK);

  for (size_t i = 0; i < NUM_STRIPS; ++i) {
    strips[i] = new Adafruit_NeoPixel(LEDS_PER_STRIP, STRIPS[i].pin, NEO_GRB + NEO_KHZ800);
    strips[i]->begin();
    Serial.printf("  pin %2u → %s\n", STRIPS[i].pin, STRIPS[i].name);
  }
  Serial.println("ready — walk pins one by one");
}

void loop() {
  for (size_t i = 0; i < NUM_STRIPS; ++i) {
    for (uint16_t p = 0; p < LEDS_PER_STRIP; ++p) {
      uint32_t c = ((p / CHUNK) % 2 == 0) ? STRIPS[i].colorA : STRIPS[i].colorB;
      strips[i]->setPixelColor(p, c);
    }
    strips[i]->show();
  }
  delay(33);
}
