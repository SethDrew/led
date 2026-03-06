/*
 * Slow fading rainbow on butterfly section (LEDs 185-299) of 300-LED strip.
 * Runs autonomously on Nano. Supports EEPROM device identification.
 */

#include <Adafruit_NeoPixel.h>
#include <EEPROM.h>

#define LED_PIN 12
#define NUM_PIXELS 300
#define BUTTERFLY_START 185
#define BUTTERFLY_COUNT 115
#define BRIGHTNESS 40  // ~15%

#define CMD_IDENTIFY 0xFE
#define CMD_SET_ID   0xFD
#define EEPROM_ID_ADDR 0
uint8_t deviceId;

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_RGB + NEO_KHZ800);

void setup() {
  Serial.begin(1000000);
  deviceId = EEPROM.read(EEPROM_ID_ADDR);
  strip.begin();
  strip.setBrightness(BRIGHTNESS);
  strip.clear();
  strip.show();
}

uint16_t offset = 0;

void loop() {
  // Handle identify commands
  while (Serial.available()) {
    uint8_t b = Serial.read();
    if (b == CMD_IDENTIFY) {
      Serial.write(CMD_IDENTIFY);
      Serial.write(deviceId);
    } else if (b == CMD_SET_ID) {
      while (!Serial.available()) {}  // wait for id byte
      deviceId = Serial.read();
      EEPROM.update(EEPROM_ID_ADDR, deviceId);
      Serial.write(CMD_SET_ID);
      Serial.write(deviceId);
    }
  }

  // Dark on strip section
  for (uint16_t i = 0; i < BUTTERFLY_START; i++) {
    strip.setPixelColor(i, 0, 0, 0);
  }

  // Slow-moving rainbow on butterfly — full cycle every ~8s
  for (uint16_t i = 0; i < BUTTERFLY_COUNT; i++) {
    uint16_t hue = ((uint32_t)i * 65536 / BUTTERFLY_COUNT + (uint32_t)offset * 273) % 65536;
    strip.setPixelColor(BUTTERFLY_START + i, strip.gamma32(strip.ColorHSV(hue)));
  }

  strip.show();
  offset++;
  delay(33);  // ~30 FPS
}
