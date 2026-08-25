#include <Arduino.h>
#include <NeoPixelBus.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <v1_packet.h>
#include <gyro_packet_v1.h>

#define FIXED_CHANNEL 6

struct RxSlot {
    uint8_t mac[6];
    TelemetryPacketV1 pkt;
    volatile uint32_t count;
    volatile bool fresh;
};
static RxSlot rxSlots[4];
static volatile uint32_t rxOther = 0;
static volatile uint32_t rxAny = 0;
static volatile uint32_t rxV1 = 0;
static uint8_t curChannel = FIXED_CHANNEL;
static uint32_t lastHopMs = 0;
static bool hopping = true;

static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    rxAny++;
    if (len != sizeof(TelemetryPacketV1)) { rxOther++; return; }
    rxV1++;
    for (int i = 0; i < 4; i++) {
        RxSlot &s = rxSlots[i];
        if (s.count == 0 || memcmp(s.mac, mac, 6) == 0) {
            memcpy(s.mac, mac, 6);
            memcpy(&s.pkt, data, sizeof(s.pkt));
            s.count++;
            s.fresh = true;
            return;
        }
    }
}

static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> stripA(LED_COUNT_A, LED_PIN_A);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> stripB(LED_COUNT_B, LED_PIN_B);

static bool swapped = false;
static bool prevBtn = true;
static uint32_t lastEdgeMs = 0;

static uint32_t lastScanMs = 0;
static uint32_t scansOk = 0, scansMiss = 0;
static bool pinsFlipped = false;

static uint8_t whoAmI(uint8_t addr, uint8_t reg) {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return 0;
    if (Wire.requestFrom(addr, (uint8_t)1) != 1) return 0;
    return Wire.read();
}

static void gyroScan() {
    uint8_t found[8];
    uint8_t n = 0;
    for (uint8_t a = 0x08; a <= 0x77 && n < 8; a++) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) found[n++] = a;
    }
    if (n == 0) {
        scansMiss++;
        int sda = digitalRead(GYRO_SDA), scl = digitalRead(GYRO_SCL);
        Serial.printf("{\"gyro\":\"none\",\"pins\":\"%s\",\"sda\":%d,"
                      "\"scl\":%d,\"ok\":%lu,\"miss\":%lu}\n",
                      pinsFlipped ? "6/5" : "5/6", sda, scl,
                      (unsigned long)scansOk, (unsigned long)scansMiss);
        if (scansMiss % 3 == 0) {
            pinsFlipped = !pinsFlipped;
            Wire.end();
            if (pinsFlipped) Wire.begin(GYRO_SCL, GYRO_SDA, 100000);
            else Wire.begin(GYRO_SDA, GYRO_SCL, 100000);
        }
        return;
    }
    scansOk++;
    for (uint8_t i = 0; i < n; i++) {
        uint8_t a = found[i];
        uint8_t who = 0;
        const char* id = "unknown";
        if (a == 0x68 || a == 0x69) {
            who = whoAmI(a, 0x75);
            if (who == 0x68) id = "MPU-6050";
            else if (who == 0x70) id = "MPU-6500";
            else if (who == 0x71) id = "MPU-9250";
            else if (who == 0x12) id = "ICM-20602";
        } else if (a == 0x6A || a == 0x6B) {
            who = whoAmI(a, 0x0F);
            if (who == 0x6C) id = "LSM6DSO";
            else if (who == 0x69) id = "LSM6DS3";
        } else if (a == 0x4A || a == 0x4B) {
            id = "BNO08x";
        }
        Serial.printf("{\"gyro\":\"0x%02X\",\"who\":\"0x%02X\",\"id\":\"%s\","
                      "\"pins\":\"%d/%d\",\"ok\":%lu,\"miss\":%lu,"
                      "\"ch\":%d,\"other\":%lu}\n",
                      a, who, id, pinsFlipped ? GYRO_SCL : GYRO_SDA,
                      pinsFlipped ? GYRO_SDA : GYRO_SCL,
                      (unsigned long)scansOk, (unsigned long)scansMiss,
                      WiFi.channel(), (unsigned long)rxOther);
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    Wire.begin(GYRO_SDA, GYRO_SCL, 100000);
    WiFi.mode(WIFI_STA);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    if (esp_now_init() == ESP_OK) esp_now_register_recv_cb(onReceive);
    stripA.Begin();
    stripB.Begin();
    stripA.Show();
    stripB.Show();
}

template <typename T>
static void renderRainbow(T& strip, uint16_t count, float t) {
    for (uint16_t i = 0; i < count; i++) {
        float hue = fmodf(t + (float)i / count, 1.0f);
        strip.SetPixelColor(i, HslColor(hue, 1.0f, 0.25f));
    }
}

template <typename T>
static void renderBlue(T& strip, uint16_t count) {
    RgbColor blue(0, 0, 100);
    for (uint16_t i = 0; i < count; i++) strip.SetPixelColor(i, blue);
}

void loop() {
    bool btn = digitalRead(BUTTON_PIN);
    uint32_t now = millis();
    if (prevBtn && !btn && now - lastEdgeMs > 40) {
        swapped = !swapped;
        lastEdgeMs = now;
        Serial.printf("{\"btn\":\"press\",\"swapped\":%d}\n", swapped);
    }
    prevBtn = btn;

    float t = now * 0.00005f;
    if (!swapped) {
        renderRainbow(stripA, LED_COUNT_A, t);
        renderBlue(stripB, LED_COUNT_B);
    } else {
        renderBlue(stripA, LED_COUNT_A);
        renderRainbow(stripB, LED_COUNT_B, t);
    }
    stripA.Show();
    while (!stripB.CanShow()) {}
    stripB.Show();

    if (now - lastScanMs >= 2000) {
        lastScanMs = now;
        gyroScan();
    }
    if (hopping && now - lastHopMs >= 3000) {
        Serial.printf("{\"hop\":%u,\"frames\":%lu,\"other\":%lu}\n",
                      curChannel, (unsigned long)rxAny,
                      (unsigned long)rxOther);
        if (rxV1 > 0) hopping = false;
        else {
            rxAny = 0;
            rxOther = 0;
            curChannel = curChannel >= 11 ? 1 : curChannel + 1;
            esp_wifi_set_channel(curChannel, WIFI_SECOND_CHAN_NONE);
        }
        lastHopMs = now;
    }
    for (int i = 0; i < 4; i++) {
        RxSlot &s = rxSlots[i];
        if (!s.fresh) continue;
        s.fresh = false;
        Serial.printf("{\"rx\":\"%02X:%02X:%02X:%02X:%02X:%02X\",\"seq\":%u,"
                      "\"amag_max\":%u,\"gmag_max\":%u,\"n\":%lu,"
                      "\"other\":%lu}\n",
                      s.mac[0], s.mac[1], s.mac[2], s.mac[3], s.mac[4],
                      s.mac[5], s.pkt.seq, s.pkt.amag_max, s.pkt.gmag_max,
                      (unsigned long)s.count, (unsigned long)rxOther);
    }
    delay(16);
}
