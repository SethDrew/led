// Minimal ESP-NOW receive test — no LEDs, no libraries
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>

static volatile uint32_t pktCount = 0;
static volatile int lastLen = 0;

void onRecv(const uint8_t *mac, const uint8_t *data, int len) {
    pktCount++;
    lastLen = len;
}

void setup() {
    Serial.begin(460800);
    delay(500);
    WiFi.mode(WIFI_STA);
    Serial.printf("init: %d\n", esp_now_init());
    Serial.printf("cb: %d\n", esp_now_register_recv_cb(onRecv));
    Serial.printf("MAC: %s\n", WiFi.macAddress().c_str());
}

void loop() {
    Serial.printf("pkts=%lu  lastLen=%d\n", pktCount, lastLen);
    delay(1000);
}
