/*
 * BRIDGE — ESP-NOW <-> USB serial pipe.
 *
 * Mirrors sender WiFi/ESP-NOW setup exactly for reliable TX.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#define FIXED_CHANNEL 6

static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static const uint8_t FRAME_M0 = 0xA5;
static const uint8_t FRAME_M1 = 0x5A;
#define MAX_PAYLOAD 250

// ── ESP-NOW RX → laptop ─────────────────────────────────────────────
static void onRecv(const uint8_t* mac, const uint8_t* data, int len) {
    if (len <= 0 || len > MAX_PAYLOAD) return;
    uint8_t hdr[3] = { FRAME_M0, FRAME_M1, (uint8_t)len };
    Serial.write(hdr, 3);
    Serial.write(data, len);
    uint8_t x = (uint8_t)len;
    for (int i = 0; i < len; i++) x ^= data[i];
    Serial.write(&x, 1);
}

static void onSent(const uint8_t* mac, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) {
        Serial.printf("[SEND] FAIL\n");
    }
}

// ── Laptop → ESP-NOW TX (frame parser state machine) ────────────────
enum RxState { WAIT_M0, WAIT_M1, WAIT_LEN, READ_PAYLOAD, READ_XOR };
static RxState rxState = WAIT_M0;
static uint8_t rxLen = 0;
static uint8_t rxBuf[MAX_PAYLOAD];
static uint16_t rxIdx = 0;

static void serialTick() {
    while (Serial.available()) {
        uint8_t b = (uint8_t)Serial.read();
        if (b == '?' && rxState == WAIT_M0) {
            Serial.printf("[BOOT] role=bridge MAC=%s fw=bridge ch=%u\n",
                          WiFi.macAddress().c_str(), (unsigned)WiFi.channel());
            continue;
        }
        switch (rxState) {
            case WAIT_M0:
                if (b == FRAME_M0) rxState = WAIT_M1;
                break;
            case WAIT_M1:
                rxState = (b == FRAME_M1) ? WAIT_LEN : WAIT_M0;
                break;
            case WAIT_LEN:
                if (b == 0 || b > MAX_PAYLOAD) { rxState = WAIT_M0; break; }
                rxLen = b;
                rxIdx = 0;
                rxState = READ_PAYLOAD;
                break;
            case READ_PAYLOAD:
                rxBuf[rxIdx++] = b;
                if (rxIdx >= rxLen) rxState = READ_XOR;
                break;
            case READ_XOR: {
                uint8_t x = rxLen;
                for (int i = 0; i < rxLen; i++) x ^= rxBuf[i];
                if (x == b) {
                    esp_now_send(broadcastAddr, rxBuf, rxLen);
                }
                rxState = WAIT_M0;
                break;
            }
        }
    }
}

void setup() {
    Serial.begin(460800);
    delay(300);

    // Mirror sender WiFi setup exactly
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);

    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        while (1) delay(1000);
    }
    esp_now_register_recv_cb(onRecv);
    esp_now_register_send_cb(onSent);

    Serial.printf("[BOOT] role=bridge MAC=%s fw=bridge ch=%d\n",
                  WiFi.macAddress().c_str(), WiFi.channel());

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
}

void loop() {
    serialTick();
}
