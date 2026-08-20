#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#define MAX_SENDERS 16
#define RING_SIZE 32

struct Pkt {
    uint8_t mac[6];
    uint8_t data[64];
    uint8_t len;
    uint32_t ms;
};

struct Sender {
    uint8_t mac[6];
    uint32_t count;
    uint32_t lastMs;
    uint8_t lastLen;
    uint8_t chan;
    bool used;
};

static const uint8_t HOP_CHANNELS[] = {1, 6, 11};
static uint8_t hopIdx = 0;
static uint8_t curChannel = 1;

static Pkt ring[RING_SIZE];
static volatile uint8_t ringHead = 0;
static uint8_t ringTail = 0;
static Sender senders[MAX_SENDERS];
static volatile uint32_t totalPkts = 0;

static void onReceive(const uint8_t* mac, const uint8_t* data, int len) {
    totalPkts++;
    uint8_t h = ringHead;
    if ((uint8_t)(h - ringTail) >= RING_SIZE) return;
    Pkt &p = ring[h & (RING_SIZE - 1)];
    memcpy(p.mac, mac, 6);
    p.len = (len > 64) ? 64 : len;
    memcpy(p.data, data, p.len);
    p.ms = millis();
    ringHead = h + 1;
}

static Sender* findSender(const uint8_t* mac) {
    for (int i = 0; i < MAX_SENDERS; i++)
        if (senders[i].used && memcmp(senders[i].mac, mac, 6) == 0)
            return &senders[i];
    for (int i = 0; i < MAX_SENDERS; i++) {
        if (!senders[i].used) {
            senders[i].used = true;
            memcpy(senders[i].mac, mac, 6);
            senders[i].count = 0;
            return &senders[i];
        }
    }
    return nullptr;
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    esp_now_init();
    esp_now_register_recv_cb(onReceive);

    Serial.printf("\n[BOOT] generic espnow sniffer MAC=%s hop=1,6,11\n", WiFi.macAddress().c_str());
}

static void setChannel(uint8_t ch) {
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    curChannel = ch;
}

void loop() {
    while (ringTail != (uint8_t)ringHead) {
        Pkt &p = ring[ringTail & (RING_SIZE - 1)];
        ringTail++;
        Sender* s = findSender(p.mac);
        bool first = s && s->count == 0;
        if (s) {
            s->count++;
            s->lastMs = p.ms;
            s->lastLen = p.len;
            s->chan = curChannel;
        }
        if (first) {
            Serial.printf("[NEW] ch=%u %02X:%02X:%02X:%02X:%02X:%02X len=%u hex=",
                          curChannel,
                          p.mac[0], p.mac[1], p.mac[2], p.mac[3], p.mac[4], p.mac[5], p.len);
            for (int i = 0; i < p.len && i < 32; i++) Serial.printf("%02X", p.data[i]);
            Serial.println();
        }
    }

    static uint32_t lastLogMs = 0;
    static uint32_t lastHopMs = 0;
    uint32_t now = millis();
    if (now - lastHopMs > 2000) {
        lastHopMs = now;
        hopIdx = (hopIdx + 1) % (sizeof(HOP_CHANNELS) / sizeof(HOP_CHANNELS[0]));
        setChannel(HOP_CHANNELS[hopIdx]);
    }
    if (now - lastLogMs > 2000) {
        lastLogMs = now;
        Serial.printf("[%lus] ch=%u total=%lu\n", now / 1000, curChannel, (unsigned long)totalPkts);
        for (int i = 0; i < MAX_SENDERS; i++) {
            if (!senders[i].used) continue;
            Sender &s = senders[i];
            Serial.printf("  ch=%u %02X:%02X:%02X:%02X:%02X:%02X pkts=%lu len=%u age=%lums\n",
                          s.chan,
                          s.mac[0], s.mac[1], s.mac[2], s.mac[3], s.mac[4], s.mac[5],
                          (unsigned long)s.count, s.lastLen, (unsigned long)(now - s.lastMs));
        }
    }
    delay(5);
}
