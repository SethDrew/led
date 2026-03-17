/*
 * DIFFUSER STICK — GREEN BREATHE
 *
 * All LEDs fade in and out together, pure green, max 50% brightness.
 * Runs entirely on the ESP32.
 */

#include <Adafruit_NeoPixel.h>

#define TOTAL_LEDS (STRIP_A_COUNT + STRIP_B_COUNT)

Adafruit_NeoPixel stripA(STRIP_A_COUNT, STRIP_A_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripB(STRIP_B_COUNT, STRIP_B_PIN, NEO_GRB + NEO_KHZ800);

// Max green value at 50% brightness
static const uint8_t MAX_GREEN = 127;

// Breathe cycle ~4 seconds at 60 fps
static const float BREATHE_SPEED = 0.025f;
float phase = 0.0f;

static const unsigned long FRAME_MS = 16;

void setup() {
    Serial.begin(115200);

    stripA.begin();
    stripA.setBrightness(255);
    stripB.begin();
    stripB.setBrightness(255);

    // Startup flash
    for (uint16_t i = 0; i < STRIP_A_COUNT; i++) stripA.setPixelColor(i, 50, 0, 0);
    for (uint16_t i = 0; i < STRIP_B_COUNT; i++) stripB.setPixelColor(i, 50, 0, 0);
    stripA.show(); stripB.show();
    delay(200);
    stripA.clear(); stripA.show();
    stripB.clear(); stripB.show();

    Serial.println("Diffuser Stick — Green Breathe");
}

void loop() {
    unsigned long frameStart = millis();

    // Sine wave: 0..1 range
    float brightness = (sin(phase) + 1.0f) * 0.5f;
    uint8_t g = (uint8_t)(brightness * MAX_GREEN);

    for (uint16_t i = 0; i < STRIP_A_COUNT; i++)
        stripA.setPixelColor(i, 0, g, 0);
    for (uint16_t i = 0; i < STRIP_B_COUNT; i++)
        stripB.setPixelColor(i, 0, g, 0);

    stripA.show();
    stripB.show();

    phase += BREATHE_SPEED;
    if (phase > 6.2832f) phase -= 6.2832f;

    unsigned long elapsed = millis() - frameStart;
    if (elapsed < FRAME_MS)
        delay(FRAME_MS - elapsed);
}
