#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// Fixed-ch1 per-packet probe: one JSON line per clicky packet (ms + seq)
// for over-the-air loss/jitter measurement. Flash to any spare classic ESP32.

struct Rec {
    uint32_t ms;
    uint8_t seq;
    uint16_t pos[6];
};

#define RING_SIZE 64
static Rec ring[RING_SIZE];
static volatile uint8_t ringHead = 0;
static uint8_t ringTail = 0;

static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len != 18 || data[0] != 0x1C || data[1] != 0xC1) return;
    uint8_t h = ringHead;
    if ((uint8_t)(h - ringTail) >= RING_SIZE) return;
    Rec &r = ring[h & (RING_SIZE - 1)];
    r.ms = millis();
    r.seq = data[3];
    memcpy(r.pos, data + 6, 12);
    ringHead = h + 1;
}

void setup() {
    Serial.begin(115200);
    delay(500);
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    esp_now_init();
    esp_now_register_recv_cb(onReceive);
    Serial.printf("\n[BOOT] clicky seq probe ch1 MAC=%s\n", WiFi.macAddress().c_str());
}

void loop() {
    while (ringTail != (uint8_t)ringHead) {
        Rec &r = ring[ringTail & (RING_SIZE - 1)];
        ringTail++;
        Serial.printf("{\"ms\":%lu,\"seq\":%u,\"p\":[%u,%u,%u,%u,%u,%u]}\n",
                      (unsigned long)r.ms, r.seq, r.pos[0], r.pos[1], r.pos[2],
                      r.pos[3], r.pos[4], r.pos[5]);
    }
    delay(2);
}
