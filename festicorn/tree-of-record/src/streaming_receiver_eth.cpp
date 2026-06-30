// streaming_receiver_eth — host-streamed pixel sink for the led-eth board.
//
// The laptop (audio-reactive viewer / runner.py) does all the audio DSP and
// effect rendering, then streams finished RGB frames over USB serial. This
// firmware is a dumb sink: parse a frame, mirror it across all 4 strips, show.
//
// Protocol (matches audio-reactive/effects/runner.py SerialLEDOutput):
//   [0xFF][0xAA][R0][G0][B0]...[Rn][Gn][Bn][XOR_CHECKSUM]
//   XOR_CHECKSUM = XOR of all RGB bytes. Bad checksum / timeout → hold last
//   good frame, resync on the next 0xFF 0xAA pair (single bad byte can't
//   cascade into multi-frame glitching).
//
// Host receives NUM_LEDS pixels; we write the same NUM_LEDS to each of the 4
// strips ("duplicate the effect across all four"). For independent 600-LED
// output, raise NUM_LEDS to 600 and split the buffer per strip instead.
//
// Identify: send 0xFE → responds [0xFE, mac0..mac5] (factory STA MAC), so the
// viewer's controllers.json auto-matches this board to its port by MAC.
//
// 4× WS2812 (GRB), 150 LEDs each, GPIO 13/27/32/33, RMT0..3. CH340 @ 460800.

#include <Arduino.h>
#include <NeoPixelBus.h>
#include <esp_mac.h>

#ifndef NUM_LEDS
#define NUM_LEDS 150
#endif
#ifndef STRIP0_PIN
#define STRIP0_PIN 13
#endif
#ifndef STRIP1_PIN
#define STRIP1_PIN 27
#endif
#ifndef STRIP2_PIN
#define STRIP2_PIN 32
#endif
#ifndef STRIP3_PIN
#define STRIP3_PIN 33
#endif

// One dedicated RMT channel per strip — same driver config as wormhole_eth /
// tree_of_record_eth, the validated 4-strip setup for this board.
NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(NUM_LEDS, STRIP0_PIN);
NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(NUM_LEDS, STRIP1_PIN);
NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(NUM_LEDS, STRIP2_PIN);
NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(NUM_LEDS, STRIP3_PIN);

const uint8_t START1 = 0xFF;
const uint8_t START2 = 0xAA;
const uint8_t CMD_IDENTIFY = 0xFE;
const uint16_t FRAME_BYTES = (uint16_t)NUM_LEDS * 3;

uint8_t frameBuf[NUM_LEDS * 3];

enum RxState { WAIT_SYNC1, WAIT_SYNC2, READ_FRAME };
RxState rxState = WAIT_SYNC1;
uint16_t bytesRead = 0;
uint8_t runningXor = 0;
unsigned long frameStartTime = 0;

static void showAll() {
  strip0.Show();
  strip1.Show();
  strip2.Show();
  strip3.Show();
}

static void fillAll(uint8_t r, uint8_t g, uint8_t b) {
  RgbColor c(r, g, b);
  for (int i = 0; i < NUM_LEDS; i++) {
    strip0.SetPixelColor(i, c);
    strip1.SetPixelColor(i, c);
    strip2.SetPixelColor(i, c);
    strip3.SetPixelColor(i, c);
  }
}

void setup() {
  Serial.setRxBufferSize(2048);  // 4×150 frames are ~1.8 KB on the wire
  Serial.begin(460800);

  strip0.Begin();
  strip1.Begin();
  strip2.Begin();
  strip3.Begin();

  // Brief boot blink so a dark board reads as "alive, waiting for frames".
  fillAll(20, 0, 0);
  showAll();
  delay(120);
  fillAll(0, 0, 0);
  showAll();
}

void loop() {
  while (Serial.available()) {
    uint8_t b = Serial.read();

    switch (rxState) {
      case WAIT_SYNC1:
        if (b == START1) {
          rxState = WAIT_SYNC2;
        } else if (b == CMD_IDENTIFY) {
          uint8_t mac[6];
          esp_read_mac(mac, ESP_MAC_WIFI_STA);
          Serial.write(CMD_IDENTIFY);
          Serial.write(mac, 6);
        }
        break;

      case WAIT_SYNC2:
        if (b == START2) {
          rxState = READ_FRAME;
          bytesRead = 0;
          runningXor = 0;
          frameStartTime = millis();
        } else if (b == START1) {
          // consecutive 0xFF — stay, next byte might be 0xAA
        } else {
          rxState = WAIT_SYNC1;
        }
        break;

      case READ_FRAME:
        if (bytesRead < FRAME_BYTES) {
          frameBuf[bytesRead] = b;
          runningXor ^= b;
          bytesRead++;
        } else {
          // this byte is the XOR checksum
          if (runningXor == b) {
            for (int i = 0; i < NUM_LEDS; i++) {
              uint8_t r = frameBuf[i * 3];
              uint8_t g = frameBuf[i * 3 + 1];
              uint8_t bl = frameBuf[i * 3 + 2];
              RgbColor c(r, g, bl);
              strip0.SetPixelColor(i, c);
              strip1.SetPixelColor(i, c);
              strip2.SetPixelColor(i, c);
              strip3.SetPixelColor(i, c);
            }
            showAll();
          }
          // bad checksum: drop, hold last good frame
          rxState = WAIT_SYNC1;
        }
        break;
    }
  }

  // Abandon a stalled partial frame so a mid-frame stop can't wedge the parser.
  if (rxState == READ_FRAME && (millis() - frameStartTime) > 50) {
    rxState = WAIT_SYNC1;
  }
}
