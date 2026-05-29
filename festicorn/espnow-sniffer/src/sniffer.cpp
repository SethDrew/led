#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "v1_packet.h"

static const char* WIFI_SSID_TARGET = "cuteplant";
#define CHANNEL_FALLBACK 1
#define CHANNEL_RESCAN_MS (5UL * 60UL * 1000UL)

static uint8_t currentChannel = CHANNEL_FALLBACK;
static uint32_t lastScanMs = 0;

struct SenderStats {
    uint8_t mac[6];
    volatile uint32_t pktCount;
    volatile uint32_t lastMs;
    volatile uint16_t lastSeq;
    volatile bool isV1;
    volatile bool isNew;
    volatile TelemetryPacketV1 lastPkt;
};

#define MAX_SENDERS 16
static volatile SenderStats senders[MAX_SENDERS];
static volatile uint8_t numSenders = 0;

void onReceive(const uint8_t* mac, const uint8_t* data, int len) {
    int idx = -1;
    for (uint8_t i = 0; i < numSenders; i++) {
        if (memcmp((const void*)senders[i].mac, mac, 6) == 0) { idx = i; break; }
    }
    if (idx < 0) {
        if (numSenders >= MAX_SENDERS) return;
        idx = numSenders++;
        memcpy((void*)senders[idx].mac, mac, 6);
        senders[idx].pktCount = 0;
        senders[idx].lastSeq = 0;
        senders[idx].isV1 = false;
        senders[idx].isNew = true;
    }

    senders[idx].pktCount++;
    senders[idx].lastMs = millis();

    if (len == sizeof(TelemetryPacketV1)) {
        TelemetryPacketV1 pkt;
        memcpy(&pkt, data, sizeof(pkt));
        senders[idx].lastSeq = pkt.seq;
        senders[idx].isV1 = true;
        memcpy((void*)&senders[idx].lastPkt, &pkt, sizeof(pkt));

        // Emit binary frame: [0xA5][0x5A][LEN+6][MAC:6][PAYLOAD:LEN][XOR8]
        uint8_t frameLen = 6 + len;
        uint8_t frame[2 + 1 + 6 + sizeof(TelemetryPacketV1) + 1];
        frame[0] = 0xA5;
        frame[1] = 0x5A;
        frame[2] = frameLen;
        memcpy(&frame[3], mac, 6);
        memcpy(&frame[9], data, len);
        uint8_t xor8 = 0;
        for (uint8_t i = 2; i < 3 + frameLen; i++) xor8 ^= frame[i];
        frame[3 + frameLen] = xor8;
        Serial.write(frame, 4 + frameLen);
    }
}

#define AMAG_FS 57000.0f
#define GMAG_FS 57000.0f
#define COUNTS_PER_G (32768.0f / 4.0f)
#define COUNTS_PER_DPS (32768.0f / 1000.0f)
static inline float magFromByte(uint8_t b) {
    float n = (float)b / 255.0f;
    return n * n * AMAG_FS;
}
static inline float amagG(uint8_t b) { return magFromByte(b) / COUNTS_PER_G; }
static inline float gmagDps(uint8_t b) { return magFromByte(b) / COUNTS_PER_DPS; }
static inline float axisMeanG(int8_t m) { return ((float)m * 256.0f) / COUNTS_PER_G; }

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

void setup() {
    Serial.begin(460800);
    delay(500);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    uint8_t ch = scanForSsidChannel(WIFI_SSID_TARGET);
    Serial.printf("\n[BOOT] espnow-sniffer MAC=%s ch=%u ssid_found=%d\n",
                  WiFi.macAddress().c_str(), ch ? ch : CHANNEL_FALLBACK, ch ? 1 : 0);
    applyChannel(ch ? ch : CHANNEL_FALLBACK);
    lastScanMs = millis();

    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.println("[READY] listening for ESP-NOW packets...\n");
}

void loop() {
    uint32_t now = millis();

    // Print new sender alerts
    uint8_t n = numSenders;
    for (uint8_t i = 0; i < n; i++) {
        if (senders[i].isNew) {
            senders[i].isNew = false;
            Serial.printf("[NEW] sender %u: %02X:%02X:%02X:%02X:%02X:%02X  %s\n",
                          i,
                          senders[i].mac[0], senders[i].mac[1], senders[i].mac[2],
                          senders[i].mac[3], senders[i].mac[4], senders[i].mac[5],
                          senders[i].isV1 ? "v1" : "unknown");
        }
    }

    // Summary every 3 seconds
    static uint32_t lastSummaryMs = 0;
    if (now - lastSummaryMs > 3000 && n > 0) {
        lastSummaryMs = now;
        Serial.printf("\n=== %u senders | %lus uptime ===\n", n, now / 1000);
        for (uint8_t i = 0; i < n; i++) {
            uint32_t age = now - senders[i].lastMs;
            const char* status = (age > 1000) ? " STALE" : "";
            if (senders[i].isV1 && age <= 1000) {
                TelemetryPacketV1 p;
                memcpy(&p, (const void*)&senders[i].lastPkt, sizeof(p));
                Serial.printf("  [%u] %02X:%02X  seq=%-5u amag=%.2f/%.2fg gmag=%d/%ddps grav=(%.2f,%.2f,%.2f) clip=%u sat=%u\n",
                              i, senders[i].mac[4], senders[i].mac[5],
                              p.seq,
                              amagG(p.amag_max), amagG(p.amag_mean),
                              (int)gmagDps(p.gmag_max), (int)gmagDps(p.gmag_mean),
                              axisMeanG(p.ax_mean), axisMeanG(p.ay_mean), axisMeanG(p.az_mean),
                              (p.flags >> 4) & 0x0F, p.flags & 0x0F);
            } else {
                Serial.printf("  [%u] %02X:%02X  pkts=%-6lu seq=%-5u %s%s\n",
                              i, senders[i].mac[4], senders[i].mac[5],
                              senders[i].pktCount, senders[i].lastSeq,
                              senders[i].isV1 ? "v1" : "??",
                              status);
            }
        }
        Serial.println();
    }

    // Channel rescan
    if (now - lastScanMs > CHANNEL_RESCAN_MS) {
        uint8_t newCh = scanForSsidChannel(WIFI_SSID_TARGET);
        if (newCh && newCh != currentChannel) {
            Serial.printf("[RESCAN] ch %u -> %u\n", currentChannel, newCh);
            applyChannel(newCh);
        }
        lastScanMs = millis();
    }

    delay(50);
}
