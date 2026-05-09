// noodles — serial streaming receiver, two strips mirrored.
// Protocol: [0xFF][0xAA][R0][G0][B0]...[Rn][Gn][Bn][XOR_CHECKSUM]
// Receives NUM_LEDS pixels, writes to the LAST NUM_LEDS of each 150-LED strip.
//
// Identification: send 0xFE → responds [0xFE, mac0..mac5] (factory MAC).
//
// ESP-NOW relay: receives sensor packets from duck/v1 senders on the
// `cuteplant` channel, forwards to host as:
//   [0xFB][15 bytes]  — v0.1 SensorPacket (duck)
//   [0xFA][16 bytes]  — v1 TelemetryPacketV1 (gyro-sense)

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <esp_mac.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "v1_packet.h"

#define PHYSICAL_LEDS 150
#define OFFSET (PHYSICAL_LEDS - NUM_LEDS)

Adafruit_NeoPixel stripA(PHYSICAL_LEDS, STRIP_A_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripB(PHYSICAL_LEDS, STRIP_B_PIN, NEO_GRB + NEO_KHZ800);

const uint8_t START1 = 0xFF;
const uint8_t START2 = 0xAA;
const uint8_t CMD_IDENTIFY = 0xFE;
const uint8_t CMD_POT = 0xFC;
const uint8_t CMD_DUCK = 0xFB;
const uint8_t CMD_V1   = 0xFA;
const uint16_t FRAME_BYTES = (uint16_t)NUM_LEDS * 3;

// ── ESP-NOW duck relay ───────────────────────────────────────────
struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    uint16_t rawRms;
    uint8_t micEnabled;
};  // 15 bytes

static volatile bool duckPacketReady = false;
static SensorPacket duckLatest;
static volatile bool v1PacketReady = false;
static TelemetryPacketV1 v1Latest;

#define DUCK_CHANNEL_FALLBACK   1
#define DUCK_CHANNEL_RESCAN_MS  (5UL * 60UL * 1000UL)
static const char* DUCK_SSID_TARGET = "cuteplant";
static uint8_t  duckChannel  = DUCK_CHANNEL_FALLBACK;
static uint32_t duckLastScanMs = 0;

static uint8_t duckScanForSsidChannel(const char* ssid) {
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

static void duckApplyChannel(uint8_t ch) {
    if (ch == 0) ch = DUCK_CHANNEL_FALLBACK;
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    duckChannel = ch;
}

void onDuckRecv(const uint8_t *mac, const uint8_t *data, int len) {
    if (len == sizeof(SensorPacket)) {
        memcpy((void*)&duckLatest, data, sizeof(SensorPacket));
        duckPacketReady = true;
    } else if (len == sizeof(TelemetryPacketV1)) {
        memcpy((void*)&v1Latest, data, sizeof(TelemetryPacketV1));
        v1PacketReady = true;
    }
}

#define POT_PIN 34
#define POT_INTERVAL_MS 50

uint8_t frameBuf[450]; // max 150 * 3

enum RxState { WAIT_SYNC1, WAIT_SYNC2, READ_FRAME };
RxState rxState = WAIT_SYNC1;
uint16_t bytesRead = 0;
uint8_t runningXor = 0;
unsigned long frameStartTime = 0;
unsigned long lastPotTime = 0;

void setup() {
    Serial.setRxBufferSize(1024);
    Serial.begin(460800);
    analogReadResolution(10);
    stripA.begin();
    stripB.begin();
    stripA.setBrightness(255);
    stripB.setBrightness(255);
    stripA.clear(); stripA.show();
    stripB.clear(); stripB.show();

    // ── Wi-Fi STA + ESP-NOW for duck sensor relay ────────────────
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    uint8_t ch = duckScanForSsidChannel(DUCK_SSID_TARGET);
    duckApplyChannel(ch ? ch : DUCK_CHANNEL_FALLBACK);
    duckLastScanMs = millis();
    esp_now_init();
    esp_now_register_recv_cb(onDuckRecv);
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
                    if (runningXor == b) {
                        for (int i = 0; i < NUM_LEDS; i++) {
                            uint8_t r = frameBuf[i * 3];
                            uint8_t g = frameBuf[i * 3 + 1];
                            uint8_t bl = frameBuf[i * 3 + 2];
                            uint32_t c = stripA.Color(r, g, bl);
                            stripA.setPixelColor(OFFSET + i, c);
                            stripB.setPixelColor(OFFSET + i, c);
                        }
                        stripA.show();
                        stripB.show();
                    }
                    rxState = WAIT_SYNC1;
                }
                break;
        }
    }

    if (rxState == READ_FRAME && (millis() - frameStartTime) > 50) {
        rxState = WAIT_SYNC1;
    }

    unsigned long now = millis();
    if (now - lastPotTime >= POT_INTERVAL_MS) {
        lastPotTime = now;
        uint16_t val = analogRead(POT_PIN);
        Serial.write(CMD_POT);
        Serial.write((uint8_t)(val >> 8));
        Serial.write((uint8_t)(val & 0xFF));
    }

    // Forward latest duck packet, if a new one arrived since last loop.
    if (duckPacketReady) {
        noInterrupts();
        SensorPacket pkt = duckLatest;
        duckPacketReady = false;
        interrupts();
        Serial.write(CMD_DUCK);
        Serial.write((const uint8_t*)&pkt, sizeof(pkt));
    }

    // Forward latest v1 telemetry packet.
    if (v1PacketReady) {
        noInterrupts();
        TelemetryPacketV1 pkt = v1Latest;
        v1PacketReady = false;
        interrupts();
        Serial.write(CMD_V1);
        Serial.write((const uint8_t*)&pkt, sizeof(pkt));
    }

    // Periodic SSID rescan for channel drift healing.
    if (now - duckLastScanMs > DUCK_CHANNEL_RESCAN_MS) {
        uint8_t newCh = duckScanForSsidChannel(DUCK_SSID_TARGET);
        if (newCh && newCh != duckChannel) {
            duckApplyChannel(newCh);
        }
        duckLastScanMs = now;
    }
}
