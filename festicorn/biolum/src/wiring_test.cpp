// Biolum wiring test — 4 strips, each a distinct color.
// Walk the install pin-by-pin to verify data lines.
//
// Pin → color:
//   23 red       22 green     25 blue      26 white
//
// Brightness held low (~25%) so all 4 strips at once stays well under PSU budget.

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 30
#endif

struct StripDef {
  uint8_t  pin;
  uint32_t color;
  const char* name;
};

// 8-bit colors at moderate brightness (≈64/255). Each value chosen so the
// dominant channel(s) read clearly when sighted on the strip.
static const StripDef STRIPS[] = {
  { 23, 0x400000, "red"     },
  { 22, 0x004000, "green"   },
  { 25, 0x000040, "blue"    },
  { 26, 0x404040, "white"   },
};
static const size_t NUM_STRIPS = sizeof(STRIPS) / sizeof(STRIPS[0]);

static Adafruit_NeoPixel* strips[NUM_STRIPS];

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("biolum wiring test");
  Serial.printf("  %u strips, %u LEDs each\n", (unsigned)NUM_STRIPS, LEDS_PER_STRIP);

  for (size_t i = 0; i < NUM_STRIPS; ++i) {
    strips[i] = new Adafruit_NeoPixel(LEDS_PER_STRIP, STRIPS[i].pin, NEO_RGB + NEO_KHZ800);
    strips[i]->begin();
    strips[i]->clear();
    for (uint16_t p = 0; p < LEDS_PER_STRIP; ++p) {
      strips[i]->setPixelColor(p, STRIPS[i].color);
    }
    strips[i]->show();
    Serial.printf("  pin %2u → %s (0x%06X)\n",
                  STRIPS[i].pin, STRIPS[i].name, STRIPS[i].color);
  }
  Serial.println("ready — walk pins one by one");
}

void loop() {
  // Re-push every frame so strips plugged in after boot start displaying.
  for (size_t i = 0; i < NUM_STRIPS; ++i) {
    for (uint16_t p = 0; p < LEDS_PER_STRIP; ++p) {
      strips[i]->setPixelColor(p, STRIPS[i].color);
    }
    strips[i]->show();
  }
  delay(33);
}
