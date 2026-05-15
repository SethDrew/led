/*
 * BRIDGE — ESP-NOW <-> USB serial pipe.
 *
 * Listens for broadcast ESP-NOW packets on the same channel as the duck
 * (discovered via "cuteplant" SSID scan) and forwards each packet to the
 * laptop framed over USB CDC. Conversely, when the laptop writes a frame
 * back, the bridge re-broadcasts it via ESP-NOW so the bulb receiver sees
 * the same 15-byte SensorPacket bytes during playback.
 *
 * Frame format (both directions, identical):
 *   [0xA5][0x5A][LEN:1][PAYLOAD: LEN bytes][XOR8: LEN ^ payload bytes]
 *
 * The bridge does not parse the payload — it is opaque. Length is bounded
 * to 250 (ESP-NOW MTU) so a corrupted byte cannot stall the parser.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

static const char* WIFI_SSID_TARGET = "cuteplant";
#define CHANNEL_FALLBACK 1
#define CHANNEL_RESCAN_MS (5UL * 60UL * 1000UL)

static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static uint8_t currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs = 0;

static const uint8_t FRAME_M0 = 0xA5;
static const uint8_t FRAME_M1 = 0x5A;
#define MAX_PAYLOAD 250

static uint8_t scanForSsidChannel(const char* ssid) {
    int n = WiFi.scanNetworks(false, false, true, 120);
    uint8_t found = 0;
    int8_t bestRssi = -127;
    for (int i = 0; i < n; i++) {
        if (WiFi.SSID(i) == ssid) {
            int8_t rssi = WiFi.RSSI(i);
            if (rssi > bestRssi) { bestRssi = rssi; found = WiFi.channel(i); }
        }
    }
    WiFi.scanDelete();
    return found;
}

static void applyChannel(uint8_t ch) {
    if (ch == 0) ch = CHANNEL_FALLBACK;
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    currentChannel = ch;
}

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

// ── Laptop → ESP-NOW TX (frame parser state machine) ────────────────
enum RxState { WAIT_M0, WAIT_M1, WAIT_LEN, READ_PAYLOAD, READ_XOR };
static RxState rxState = WAIT_M0;
static uint8_t rxLen = 0;
static uint8_t rxBuf[MAX_PAYLOAD];
static uint16_t rxIdx = 0;

static void serialTick() {
    while (Serial.available()) {
        uint8_t b = (uint8_t)Serial.read();
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

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    uint8_t ch = scanForSsidChannel(WIFI_SSID_TARGET);
    Serial.printf("[BOOT] role=bridge MAC=%s fw=bridge ch=%u ssid_found=%d\n",
                  WiFi.macAddress().c_str(), ch ? ch : CHANNEL_FALLBACK, ch ? 1 : 0);
    applyChannel(ch ? ch : CHANNEL_FALLBACK);
    lastScanMs = millis();

    esp_now_init();
    esp_now_register_recv_cb(onRecv);

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
}

void loop() {
    serialTick();

    uint32_t now = millis();
    if (now - lastScanMs > CHANNEL_RESCAN_MS) {
        uint8_t newCh = scanForSsidChannel(WIFI_SSID_TARGET);
        if (newCh && newCh != currentChannel) applyChannel(newCh);
        lastScanMs = millis();
    }
}
