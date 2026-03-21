/*
 * DIFFUSER STICK — STREAMING RECEIVER
 *
 * ESP32 with two LED strips on separate data pins.
 * Receives RGB frames via serial and splits across both strips.
 *
 * Protocol: [0xFF][0xAA][RGB x TOTAL_LEDS][XOR_CHECKSUM]
 * Strip A (pin 32): first STRIP_A_COUNT LEDs
 * Strip B (pin 12): next STRIP_B_COUNT LEDs
 */

#include <Adafruit_NeoPixel.h>
#include <streaming_protocol.h>

#define TOTAL_LEDS (STRIP_A_COUNT + STRIP_B_COUNT)

Adafruit_NeoPixel stripA(STRIP_A_COUNT, STRIP_A_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripB(STRIP_B_COUNT, STRIP_B_PIN, NEO_GRB + NEO_KHZ800);

// Frame buffer and receiver
uint8_t buf[TOTAL_LEDS * 3];
StreamReceiver rx;
unsigned long frameStartTime = 0;
bool receiving = false;

// Stats
unsigned long framesOk = 0;
unsigned long framesBad = 0;
unsigned long lastStats = 0;

void showFrame() {
    // Strip A: LEDs 0 .. STRIP_A_COUNT-1
    for (uint16_t i = 0; i < STRIP_A_COUNT; i++) {
        uint16_t idx = i * 3;
        stripA.setPixelColor(i, buf[idx], buf[idx + 1], buf[idx + 2]);
    }
    // Strip B: LEDs STRIP_A_COUNT .. TOTAL_LEDS-1
    for (uint16_t i = 0; i < STRIP_B_COUNT; i++) {
        uint16_t idx = (STRIP_A_COUNT + i) * 3;
        stripB.setPixelColor(i, buf[idx], buf[idx + 1], buf[idx + 2]);
    }
    stripA.show();
    stripB.show();
}

void setup() {
    Serial.begin(1000000);
    streamReceiverInit(rx, TOTAL_LEDS, buf);

    stripA.begin();
    stripA.setBrightness(255);
    stripB.begin();
    stripB.setBrightness(255);

    // Startup flash: brief red then off
    for (uint16_t i = 0; i < STRIP_A_COUNT; i++) stripA.setPixelColor(i, 50, 0, 0);
    for (uint16_t i = 0; i < STRIP_B_COUNT; i++) stripB.setPixelColor(i, 50, 0, 0);
    stripA.show();
    stripB.show();
    delay(200);
    stripA.clear(); stripA.show();
    stripB.clear(); stripB.show();

    Serial.println("Diffuser Stick Streaming Receiver");
    Serial.printf("Strip A: pin %d, %d LEDs\n", STRIP_A_PIN, STRIP_A_COUNT);
    Serial.printf("Strip B: pin %d, %d LEDs\n", STRIP_B_PIN, STRIP_B_COUNT);
    Serial.printf("Total: %d LEDs, waiting for frames...\n", TOTAL_LEDS);
}

void loop() {
    while (Serial.available()) {
        uint8_t b = Serial.read();

        if (streamReceiverFeed(rx, b)) {
            showFrame();
            framesOk++;
            receiving = false;
        } else if (rx.state >= 2) {
            // Entered DATA or CHECK state — start timeout clock
            if (!receiving) {
                frameStartTime = millis();
                receiving = true;
            }
        } else if (receiving) {
            // Was receiving but dropped back to SYNC — checksum failed
            framesBad++;
            receiving = false;
        }
    }

    // Frame timeout
    if (receiving && (millis() - frameStartTime) > 50) {
        streamReceiverInit(rx, TOTAL_LEDS, buf);
        framesBad++;
        receiving = false;
    }

    // Stats every 2s
    unsigned long now = millis();
    if (now - lastStats > 2000) {
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
