/*
 * STREAMING RECEIVER — 12-Effect Matrix (Wokwi Virtual)
 *
 * 12 rows × 20 LEDs = 240 total, rendered as a NeoPixel grid.
 * Each row shows one WLED audio-reactive effect.
 *
 * Protocol: [0xFF] [0xAA] [240 × RGB] = 722 bytes per frame
 *
 * Row layout (top to bottom):
 *   0  Juggles       4  Blurz        8  Freqpixels
 *   1  Midnoise      5  DJLight      9  Freqwave
 *   2  Noisemeter    6  Freqmap     10  Noisemove
 *   3  Plasmoid      7  Freqmatrix  11  Rocktaves
 */

#include <Adafruit_NeoPixel.h>

#define TOTAL_LEDS 240

Adafruit_NeoPixel strip(TOTAL_LEDS, 6, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(1000000);
  strip.begin();
  strip.show();
}

void loop() {
  if (Serial.available() >= 2) {
    if (Serial.read() == 0xFF && Serial.read() == 0xAA) {
      int bytesNeeded = TOTAL_LEDS * 3;
      uint8_t buf[60];
      uint8_t rgb[3];
      int got = 0;
      unsigned long t0 = millis();

      while (got < bytesNeeded && (millis() - t0) < 200) {
        int avail = Serial.available();
        if (avail > 0) {
          int toRead = min(avail, min((int)sizeof(buf), bytesNeeded - got));
          int n = Serial.readBytes(buf, toRead);
          for (int i = 0; i < n; i++) {
            int byteIdx = got + i;
            int ch = byteIdx % 3;
            rgb[ch] = buf[i];
            if (ch == 2) {
              strip.setPixelColor(byteIdx / 3, rgb[0], rgb[1], rgb[2]);
            }
          }
          got += n;
        }
      }

      if (got == bytesNeeded) {
        strip.show();
      }
    }
  }
}
