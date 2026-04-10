/*
 * RGB_TEST_3BULBS — solid color on each sculpture for wiring verification
 *
 * Sculpture A (strip0, GPIO 21, 50 LEDs): GREEN
 * Sculpture B (strip1 0-49, GPIO 10):     RED
 * Sculpture C (strip1 50-99, GPIO 10):    BLUE
 */

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#define PIN_STRIP0   21
#define PIN_STRIP1   10
#define STRIP0_COUNT 50
#define STRIP1_COUNT 100

static Adafruit_NeoPixel strip0(STRIP0_COUNT, PIN_STRIP0, NEO_GRBW + NEO_KHZ800);
static Adafruit_NeoPixel strip1(STRIP1_COUNT, PIN_STRIP1, NEO_GRBW + NEO_KHZ800);

void setup() {
    Serial.begin(460800);
    delay(300);

    strip0.begin();
    strip0.setBrightness(255);
    strip1.begin();
    strip1.setBrightness(255);

    // Sculpture A: green
    for (uint16_t i = 0; i < STRIP0_COUNT; i++)
        strip0.setPixelColor(i, 0, 40, 0, 0);

    // Sculpture B: red
    for (uint16_t i = 0; i < 50; i++)
        strip1.setPixelColor(i, 40, 0, 0, 0);

    // Sculpture C: blue
    for (uint16_t i = 50; i < 100; i++)
        strip1.setPixelColor(i, 0, 0, 40, 0);

    strip0.show();
    strip1.show();

    Serial.println("rgb_test_3bulbs: A=GREEN B=RED C=BLUE");
}

void loop() {
    // Sculpture A: green
    for (uint16_t i = 0; i < STRIP0_COUNT; i++)
        strip0.setPixelColor(i, 0, 40, 0, 0);

    // Sculpture B: red
    for (uint16_t i = 0; i < 50; i++)
        strip1.setPixelColor(i, 40, 0, 0, 0);

    // Sculpture C: blue
    for (uint16_t i = 50; i < 100; i++)
        strip1.setPixelColor(i, 0, 0, 40, 0);

    strip0.show();
    strip1.show();
    delay(20);
}
