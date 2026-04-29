/*
 * STREAM BRIDGE — classic ESP32 receives ESP-NOW from the C3 stream sender
 * and forwards each packet over USB serial to the host.
 *
 * Wire framing (host stream_capture.py decodes this):
 *   0xAA 0x55              — sync magic
 *   uint8  payload_len     — bytes that follow (excluding CRC)
 *   payload[payload_len]   — raw ESP-NOW packet (StreamHeader + samples)
 *   uint8  crc8            — CRC-8/CCITT (poly 0x07, init 0x00) over payload
 *
 * Serial baud: 1_000_000. Theoretical max load ~3 kB/s.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#define ESPNOW_CHANNEL 1
#define FRAME_SYNC0    0xAA
#define FRAME_SYNC1    0x55

// CRC-8 / CCITT (poly 0x07, init 0x00). Tiny, plenty for 250 B frames.
static uint8_t crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : (crc << 1);
        }
    }
    return crc;
}

static volatile uint32_t pktCount = 0;

// ESP-NOW receive callback runs on WiFi task — keep it short.
// Serial.write is buffered + non-blocking enough at 1 Mbaud for 250 B at <40 Hz.
static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len <= 0 || len > 250) return;
    pktCount++;

    uint8_t hdr[3] = { FRAME_SYNC0, FRAME_SYNC1, (uint8_t)len };
    Serial.write(hdr, 3);
    Serial.write(data, len);
    uint8_t crc = crc8(data, len);
    Serial.write(&crc, 1);
}

void setup() {
    Serial.begin(1000000);
    delay(100);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Lock to the same channel the sender uses; promiscuous toggle is the
    // recipe that has worked in this project for broadcast rx on ESP32.
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println("# ESP-NOW init FAILED");
        while (1) delay(1000);
    }
    esp_now_register_recv_cb(onReceive);

    Serial.printf("# Stream bridge ready ch=%d MAC=%s\n",
                  WiFi.channel(), WiFi.macAddress().c_str());
}

void loop() {
    // Optional liveness ping over serial as a comment line.
    // Host parser must skip non-frame bytes — frames are unambiguous via 0xAA 0x55.
    static uint32_t lastMs = 0;
    uint32_t now = millis();
    if (now - lastMs >= 5000) {
        lastMs = now;
        Serial.printf("# bridge pkts=%u\n", (unsigned)pktCount);
    }
    delay(10);
}
