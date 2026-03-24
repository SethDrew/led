/*
 * STREAM RECEIVER — Serial RGBW → dual SK6812 RGBW strips
 *
 * Receives RGBW frames via USB serial and splits across two LED strips
 * for A/B effect comparison on a single ESP32-C3.
 *
 * Protocol: [0xFF][0xAA][RGBW x 100][XOR_CHECKSUM]
 *   LEDs 0-49  → Strip A (GPIO 10)
 *   LEDs 50-99 → Strip B (GPIO 21)
 *
 * RGBW bytes are passed through directly to the SK6812 — no conversion.
 * Python-side effects have full control over all four channels.
 *
 * Flashed onto the RECEIVER board (Board B, MAC 10:00:3B:B0:6E:04).
 *
 * Usage:
 *   pio run -e stream_recv -t upload
 *   python compare.py sparkle_burst flicker_flame_warmth --mic --port /dev/cu.usbmodem112201
 */

#include <Adafruit_NeoPixel.h>
#include <streaming_protocol.h>

#define PIN_A      10
#define PIN_B      21
#define LEDS_PER   50
#define TOTAL_LEDS (LEDS_PER * 2)
#define BPP        4  // bytes per pixel: R, G, B, W

Adafruit_NeoPixel stripA(LEDS_PER, PIN_A, NEO_GRBW + NEO_KHZ800);
Adafruit_NeoPixel stripB(LEDS_PER, PIN_B, NEO_GRBW + NEO_KHZ800);

uint8_t buf[TOTAL_LEDS * BPP];
StreamReceiver rx;
unsigned long frameStartTime = 0;
bool receiving = false;

// Stats
unsigned long framesOk = 0;
unsigned long framesBad = 0;
unsigned long lastStats = 0;

void showFrame() {
    for (uint16_t i = 0; i < LEDS_PER; i++) {
        uint16_t idx = i * BPP;
        stripA.setPixelColor(i, buf[idx], buf[idx+1], buf[idx+2], buf[idx+3]);
    }
    for (uint16_t i = 0; i < LEDS_PER; i++) {
        uint16_t idx = (LEDS_PER + i) * BPP;
        stripB.setPixelColor(i, buf[idx], buf[idx+1], buf[idx+2], buf[idx+3]);
    }
    stripA.show();
    stripB.show();
}

void setup() {
    Serial.begin(1000000);
    streamReceiverInit(rx, TOTAL_LEDS, buf, BPP);

    stripA.begin();
    stripA.setBrightness(255);
    stripB.begin();
    stripB.setBrightness(255);

    // Startup flash: strip A amber, strip B blue-white, then off
    for (uint16_t i = 0; i < LEDS_PER; i++) {
        stripA.setPixelColor(i, 40, 20, 5, 0);
        stripB.setPixelColor(i, 5, 15, 40, 0);
    }
    stripA.show();
    stripB.show();
    delay(300);
    stripA.clear(); stripA.show();
    stripB.clear(); stripB.show();
}

void loop() {
    // Read all available bytes — loop back quickly if mid-frame
    // to handle USB-CDC buffer splits (256-byte USB RX buffer)
    int avail = Serial.available();
    while (avail > 0) {
        uint8_t b = Serial.read();
        avail--;

        if (streamReceiverFeed(rx, b)) {
            showFrame();
            framesOk++;
            receiving = false;
        } else if (rx.state >= 2) {
            if (!receiving) {
                frameStartTime = millis();
                receiving = true;
            }
        } else if (receiving) {
            framesBad++;
            receiving = false;
        }
    }

    // If mid-frame, briefly yield then re-check for more bytes
    // This handles USB-CDC packets arriving in chunks
    if (receiving) {
        unsigned long elapsed = millis() - frameStartTime;
        if (elapsed > 100) {
            // True timeout — no data for 100ms
            streamReceiverInit(rx, TOTAL_LEDS, buf, BPP);
            framesBad++;
            receiving = false;
        }
        // Otherwise just loop back — more bytes should arrive shortly
    }

    // Stats every 5s
    unsigned long now = millis();
    if (now - lastStats > 5000) {
        if (framesOk > 0 || framesBad > 0) {
            unsigned long elapsed = now - lastStats;
            Serial.printf("FPS: %lu", framesOk * 1000 / elapsed);
            if (framesBad > 0) Serial.printf(" drop: %lu", framesBad);
            Serial.println();
        }
        framesOk = 0;
        framesBad = 0;
        lastStats = now;
    }
}
