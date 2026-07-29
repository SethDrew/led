#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#include "rfproto.h"

// 27 quarter-dBm (6.75) is the only level both ESP32 and C3 accept with exact
// readback; classic quantizes onto a coarse grid that has no 34.
static const int8_t NODE_TXQ = 27;
static const uint32_t BEACON_MS = 100;
static const uint32_t WINDOW_MS = 2000;

static const uint8_t KNOWN[][6] = {
    {0x10, 0x00, 0x3B, 0xB1, 0x3E, 0x80},
    {0xB0, 0xCB, 0xD8, 0x8A, 0x85, 0xF4},
    {0x14, 0x63, 0x93, 0x6F, 0xFF, 0x78},
};
static const int N_KNOWN = sizeof(KNOWN) / 6;

#if CONFIG_IDF_TARGET_ESP32C3
static const int LED_PIN = 8;
static const int LED_ON = LOW;
#else
static const int LED_PIN = 2;
static const int LED_ON = HIGH;
#endif

struct Acc {
    uint32_t count;
    int32_t rssiSum;
    int8_t rssiMin;
    int8_t rssiMax;
    int32_t nfSum;
};

static uint8_t peers[4][6];
static int nPeers = 0;
static volatile Acc acc[4];
static NodeReport snapshot;
static uint8_t selfMac[6];
static uint8_t txBuf[250];
static uint16_t beaconSeq = 0;
static uint32_t windowIdx = 0;
static bool heardLastWindow = false;

static const uint8_t BCAST[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static int peerIndex(const uint8_t *mac) {
    for (int i = 0; i < nPeers; i++)
        if (memcmp(peers[i], mac, 6) == 0) return i;
    return -1;
}

static void onPromisc(void *buf, wifi_promiscuous_pkt_type_t type) {
    const wifi_promiscuous_pkt_t *p = (wifi_promiscuous_pkt_t *)buf;
    int len = p->rx_ctrl.sig_len;
    if (len < 28) return;

    int idx = peerIndex(p->payload + 10);
    if (idx < 0) return;

    const uint8_t *body = p->payload + 24;
    int bodyLen = len - 24 - 4;
    if (bodyLen < (int)sizeof(RfHdr)) return;
    bool ours = false;
    for (int i = 0; i + 4 <= bodyLen; i++) {
        uint32_t m;
        memcpy(&m, body + i, 4);
        if (m == RFB_MAGIC) { ours = true; break; }
    }
    if (!ours) return;

    Acc &a = (Acc &)acc[idx];
    int8_t r = (int8_t)p->rx_ctrl.rssi;
    if (a.count == 0) { a.rssiMin = r; a.rssiMax = r; }
    if (r < a.rssiMin) a.rssiMin = r;
    if (r > a.rssiMax) a.rssiMax = r;
    a.rssiSum += r;
    a.nfSum += p->rx_ctrl.noise_floor;
    a.count++;
}

static int8_t forceTxq() {
    int8_t rb = 0;
    for (int i = 0; i < 20; i++) {
        esp_wifi_set_max_tx_power(NODE_TXQ);
        esp_wifi_get_max_tx_power(&rb);
        if (rb == NODE_TXQ) break;
        delay(20);
    }
    return rb;
}

static void rollWindow() {
    NodeReport r = {};
#if CONFIG_IDF_TARGET_ESP32C3
    r.chip = RFCHIP_C3;
#else
    r.chip = RFCHIP_ESP32;
#endif
    int8_t rb = 0;
    esp_wifi_get_max_tx_power(&rb);
    if (rb != NODE_TXQ) rb = forceTxq();
    r.txq_readback = rb;
    r.uptime_s = millis() / 1000;
    r.window = ++windowIdx;
    r.window_ms = WINDOW_MS;
    r.n_peers = nPeers;

    static uint32_t prevTotal = 0;
    uint32_t total = 0;
    for (int i = 0; i < nPeers; i++) {
        Acc a;
        noInterrupts();
        a = (Acc &)acc[i];
        interrupts();

        memcpy(r.peer[i].mac, peers[i], 6);
        r.peer[i].count = a.count;
        r.peer[i].rssi_sum = a.rssiSum;
        r.peer[i].nf_sum = a.nfSum;
        r.peer[i].rssi_min = a.count ? a.rssiMin : 0;
        r.peer[i].rssi_max = a.count ? a.rssiMax : 0;
        total += a.count;
    }
    heardLastWindow = (total != prevTotal);
    prevTotal = total;
    snapshot = r;
}

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, !LED_ON);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_ps(WIFI_PS_NONE);
    esp_wifi_get_mac(WIFI_IF_STA, selfMac);

    esp_now_init();

    for (int i = 0; i < N_KNOWN && nPeers < 4; i++) {
        if (memcmp(KNOWN[i], selfMac, 6) == 0) continue;
        memcpy(peers[nPeers], KNOWN[i], 6);
        memset((void *)&acc[nPeers], 0, sizeof(Acc));
        nPeers++;
    }

    esp_now_peer_info_t pi = {};
    memcpy(pi.peer_addr, BCAST, 6);
    pi.channel = CHANNEL;
    pi.ifidx = WIFI_IF_STA;
    pi.encrypt = false;
    esp_now_add_peer(&pi);

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE);
    wifi_promiscuous_filter_t filt = {};
    filt.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT;
    esp_wifi_set_promiscuous_filter(&filt);
    esp_wifi_set_promiscuous_rx_cb(onPromisc);

    // must come after promiscuous setup: enabling it shifts which quarter-dBm
    // levels the driver will accept, so setting power earlier silently misses.
    forceTxq();

    rollWindow();
}

void loop() {
    static uint32_t tBeacon = 0, tWindow = 0, tLed = 0;
    static bool ledState = false;
    uint32_t now = millis();

    if (now - tWindow >= WINDOW_MS) {
        tWindow = now;
        rollWindow();
    }

    if (now - tBeacon >= BEACON_MS) {
        tBeacon = now;
        RfHdr h;
        h.magic = RFB_MAGIC;
        h.type = PKT_REPORT;
        h.txq = snapshot.txq_readback;
        h.seq = beaconSeq++;
        h.t_send = micros();
        memcpy(txBuf, &h, sizeof(h));
        memcpy(txBuf + sizeof(h), &snapshot, sizeof(snapshot));
        esp_now_send(BCAST, txBuf, sizeof(h) + sizeof(snapshot));
    }

    uint32_t period = heardLastWindow ? 1000 : 3000;
    if (!ledState && now - tLed >= period) {
        tLed = now;
        ledState = true;
        digitalWrite(LED_PIN, LED_ON);
    } else if (ledState && now - tLed >= 50) {
        ledState = false;
        digitalWrite(LED_PIN, !LED_ON);
    }
}
