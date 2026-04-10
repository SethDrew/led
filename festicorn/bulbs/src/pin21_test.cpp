// GPIO 21 LED test — slow hue fade through all chromas
#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

Adafruit_NeoPixel strip(10, 21, NEO_GRB + NEO_KHZ800);

#define HUE_CYCLE_MS 12000  // 12 seconds per full hue rotation

void setup() {
    Serial.begin(460800);
    strip.begin();
    strip.setBrightness(60);
    Serial.println("pin21_test: slow hue fade on GPIO 21");
}

void loop() {
    uint16_t hue = (uint16_t)((millis() % HUE_CYCLE_MS) * 65536UL / HUE_CYCLE_MS);
    uint32_t color = strip.ColorHSV(hue, 255, 255);
    uint32_t gamma = strip.gamma32(color);
    for (int i = 0; i < 10; i++)
        strip.setPixelColor(i, gamma);
    strip.show();
    delay(10);
}
