// ESP-NOW → USB bridge for the console dashboard: every packet heard on
// channel 1 is forwarded as A5 5A | len | payload | chk, chk = len XOR
// payload bytes (the framing console-dashboard/dashboard.py BridgeSource
// parses). No other serial output — the dashboard probe identifies the
// bridge by the A5 5A prefix itself.
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#define BRIDGE_CHANNEL 1
#define RING_SIZE 64
#define MAX_PAYLOAD 64

struct Pkt {
    uint8_t data[MAX_PAYLOAD];
    uint8_t len;
};

static Pkt ring[RING_SIZE];
static volatile uint8_t ringHead = 0;
static uint8_t ringTail = 0;

static void onReceive(const uint8_t* mac, const uint8_t* data, int len) {
    (void)mac;
    if (len <= 0 || len > MAX_PAYLOAD) return;
    uint8_t h = ringHead;
    if ((uint8_t)(h - ringTail) >= RING_SIZE) return;
    Pkt &p = ring[h & (RING_SIZE - 1)];
    p.len = (uint8_t)len;
    memcpy(p.data, data, p.len);
    ringHead = h + 1;
}

void setup() {
    Serial.begin(460800);
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_ps(WIFI_PS_NONE);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(BRIDGE_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    esp_now_init();
    esp_now_register_recv_cb(onReceive);
    esp_wifi_set_channel(BRIDGE_CHANNEL, WIFI_SECOND_CHAN_NONE);
}

void loop() {
    while (ringTail != (uint8_t)ringHead) {
        Pkt &p = ring[ringTail & (RING_SIZE - 1)];
        ringTail++;
        uint8_t hdr[3] = { 0xA5, 0x5A, p.len };
        uint8_t chk = p.len;
        for (uint8_t i = 0; i < p.len; i++) chk ^= p.data[i];
        Serial.write(hdr, 3);
        Serial.write(p.data, p.len);
        Serial.write(&chk, 1);
    }
    delay(1);
}
