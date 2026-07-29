#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#include "rfproto.h"

static const int MAX_SAMPLES = 512;

struct RxStat {
    uint32_t count;
    int32_t rssiSum;
    int32_t rssiMin;
    int32_t rssiMax;
    int32_t nfSum;
    uint32_t rateBits;
    uint16_t lastSeq;
    uint16_t firstSeq;
    bool haveSeq;
};

static uint8_t peers[MAX_PEERS][6];
static RxStat rxStats[MAX_PEERS];
static NodeReport nodeRpt[MAX_PEERS];
static bool nodeRptValid[MAX_PEERS];
static int nPeers = 0;

static volatile bool cbFired = false;
static volatile esp_now_send_status_t cbStatus = ESP_NOW_SEND_FAIL;
static volatile uint32_t cbMicros = 0;

static volatile bool echoPending = false;
static uint8_t echoDst[6];
static RfHdr echoHdr;

static uint8_t txBuf[250];
static int16_t sampAck[MAX_SAMPLES];
static uint32_t sampLat[MAX_SAMPLES];

static int8_t curTxq = 0;

static volatile uint32_t ambCount = 0;
static volatile int32_t ambNfSum = 0;
static volatile int32_t ambRssiSum = 0;
static volatile int8_t ambNfMin = 127;
static volatile int8_t ambNfMax = -128;

static int peerIndex(const uint8_t *mac) {
    for (int i = 0; i < nPeers; i++)
        if (memcmp(peers[i], mac, 6) == 0) return i;
    return -1;
}

static void macToStr(const uint8_t *m, char *out) {
    sprintf(out, "%02X:%02X:%02X:%02X:%02X:%02X", m[0], m[1], m[2], m[3], m[4], m[5]);
}

static bool parseMac(const char *s, uint8_t *out) {
    unsigned v[6];
    if (sscanf(s, "%x:%x:%x:%x:%x:%x", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) return false;
    for (int i = 0; i < 6; i++) out[i] = (uint8_t)v[i];
    return true;
}

static void onSent(const uint8_t *mac, esp_now_send_status_t status) {
    cbMicros = micros();
    cbStatus = status;
    cbFired = true;
}

static void onRecv(const uint8_t *mac, const uint8_t *data, int len) {
    if (len < (int)sizeof(RfHdr)) return;
    RfHdr h;
    memcpy(&h, data, sizeof(h));
    if (h.magic != RFB_MAGIC) return;
    if (h.type == PKT_PING && !echoPending) {
        memcpy(echoDst, mac, 6);
        echoHdr = h;
        echoHdr.type = PKT_PONG;
        echoPending = true;
    }
}

static void onPromisc(void *buf, wifi_promiscuous_pkt_type_t type) {
    const wifi_promiscuous_pkt_t *p = (wifi_promiscuous_pkt_t *)buf;
    int len = p->rx_ctrl.sig_len;

    int8_t nf = p->rx_ctrl.noise_floor;
    ambNfSum += nf;
    ambRssiSum += p->rx_ctrl.rssi;
    if (nf < ambNfMin) ambNfMin = nf;
    if (nf > ambNfMax) ambNfMax = nf;
    ambCount++;

    if (len < 28) return;
    const uint8_t *src = p->payload + 10;
    int idx = peerIndex(src);
    if (idx < 0) return;

    const uint8_t *body = p->payload + 24;
    int bodyLen = len - 24 - 4;
    if (bodyLen < (int)sizeof(RfHdr)) return;
    int found = -1;
    for (int i = 0; i + (int)sizeof(RfHdr) <= bodyLen; i++) {
        uint32_t m;
        memcpy(&m, body + i, 4);
        if (m == RFB_MAGIC) { found = i; break; }
    }
    if (found < 0) return;
    RfHdr h;
    memcpy(&h, body + found, sizeof(h));

    RxStat &s = rxStats[idx];
    int r = p->rx_ctrl.rssi;
    if (s.count == 0) { s.rssiMin = r; s.rssiMax = r; }
    if (r < s.rssiMin) s.rssiMin = r;
    if (r > s.rssiMax) s.rssiMax = r;
    s.rssiSum += r;
    s.nfSum += p->rx_ctrl.noise_floor;
    s.rateBits |= (1u << (p->rx_ctrl.rate & 31));
    if (!s.haveSeq) { s.firstSeq = h.seq; s.haveSeq = true; }
    s.lastSeq = h.seq;
    s.count++;

    if (h.type == PKT_REPORT && bodyLen - found >= (int)(sizeof(RfHdr) + sizeof(NodeReport))) {
        memcpy(&nodeRpt[idx], body + found + sizeof(RfHdr), sizeof(NodeReport));
        nodeRptValid[idx] = true;
    }
}

static void lockChannel() {
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE);
}

static void cmdId() {
    char mac[18];
    uint8_t m[6];
    esp_wifi_get_mac(WIFI_IF_STA, m);
    macToStr(m, mac);
    int8_t p = 0;
    esp_wifi_get_max_tx_power(&p);
#if CONFIG_IDF_TARGET_ESP32C3
    const char *chip = "esp32c3";
#else
    const char *chip = "esp32";
#endif
    Serial.printf("ID mac=%s chip=%s idf=%s cpu=%u txq=%d\n",
                  mac, chip, esp_get_idf_version(), (unsigned)getCpuFrequencyMhz(), (int)p);
}

static void cmdTxp(int q) {
    esp_err_t e = esp_wifi_set_max_tx_power((int8_t)q);
    int8_t rb = 0;
    esp_wifi_get_max_tx_power(&rb);
    curTxq = rb;
    Serial.printf("TXP req=%d err=%d readback=%d dbm=%.2f\n", q, (int)e, (int)rb, rb / 4.0);
}

static void cmdPeer(const char *arg) {
    uint8_t m[6];
    if (!parseMac(arg, m)) { Serial.println("ERR badmac"); return; }
    if (peerIndex(m) >= 0) { Serial.println("PEER dup"); return; }
    if (nPeers >= MAX_PEERS) { Serial.println("ERR full"); return; }
    esp_now_peer_info_t pi = {};
    memcpy(pi.peer_addr, m, 6);
    pi.channel = CHANNEL;
    pi.ifidx = WIFI_IF_STA;
    pi.encrypt = false;
    esp_err_t e = esp_now_add_peer(&pi);
    memcpy(peers[nPeers], m, 6);
    memset(&rxStats[nPeers], 0, sizeof(RxStat));
    nPeers++;
    Serial.printf("PEER ok idx=%d err=%d\n", nPeers - 1, (int)e);
}

static void cmdRxClear() {
    memset(rxStats, 0, sizeof(rxStats));
    memset(nodeRptValid, 0, sizeof(nodeRptValid));
    Serial.println("RXCLEAR ok");
}

static void cmdRxStat() {
    for (int i = 0; i < nPeers; i++) {
        char mac[18];
        macToStr(peers[i], mac);
        RxStat &s = rxStats[i];
        if (s.count == 0) {
            Serial.printf("RX src=%s n=0\n", mac);
        } else {
            uint16_t span = (uint16_t)(s.lastSeq - s.firstSeq) + 1;
            Serial.printf("RX src=%s n=%lu rssi_mean=%.2f rssi_min=%ld rssi_max=%ld nf_mean=%.2f rates=0x%lx "
                          "firstseq=%u lastseq=%u span=%u loss_pct=%.2f\n",
                          mac, (unsigned long)s.count, (double)s.rssiSum / s.count,
                          (long)s.rssiMin, (long)s.rssiMax, (double)s.nfSum / s.count,
                          (unsigned long)s.rateBits, s.firstSeq, s.lastSeq, span,
                          span ? 100.0 * (span - (double)s.count) / span : 0.0);
        }
    }
    Serial.println("RXSTAT end");
}

static void cmdNodeStat() {
    for (int i = 0; i < nPeers; i++) {
        if (!nodeRptValid[i]) continue;
        char mac[18], pm[18];
        macToStr(peers[i], mac);
        const NodeReport &r = nodeRpt[i];
        Serial.printf("NODE src=%s chip=%s txq=%d dbm=%.2f uptime_s=%lu window=%lu window_ms=%u\n",
                      mac, r.chip == RFCHIP_C3 ? "esp32c3" : "esp32",
                      (int)r.txq_readback, r.txq_readback / 4.0,
                      (unsigned long)r.uptime_s, (unsigned long)r.window, r.window_ms);
        for (int j = 0; j < r.n_peers && j < 4; j++) {
            macToStr(r.peer[j].mac, pm);
            uint32_t c = r.peer[j].count;
            Serial.printf("NODEHEARD src=%s heard=%s n=%lu rssi_sum=%ld nf_sum=%ld "
                          "rssi_mean=%.2f rssi_min=%d rssi_max=%d\n",
                          mac, pm, (unsigned long)c,
                          (long)r.peer[j].rssi_sum, (long)r.peer[j].nf_sum,
                          c ? (double)r.peer[j].rssi_sum / c : 0.0,
                          (int)r.peer[j].rssi_min, (int)r.peer[j].rssi_max);
        }
    }
    Serial.println("NODESTAT end");
}

static void cmdBurst(const char *macs, int count, int interval, int len, int wantEcho) {
    uint8_t dst[6];
    if (!parseMac(macs, dst)) { Serial.println("ERR badmac"); return; }
    if (count > MAX_SAMPLES) count = MAX_SAMPLES;
    if (len < (int)sizeof(RfHdr)) len = sizeof(RfHdr);
    if (len > 250) len = 250;

    int nAck = 0, nTimeout = 0;
    uint32_t t0all = micros();

    for (int i = 0; i < count; i++) {
        RfHdr h;
        h.magic = RFB_MAGIC;
        h.type = wantEcho ? PKT_PING : PKT_FLOOD;
        h.txq = curTxq;
        h.seq = i;
        h.t_send = micros();
        memset(txBuf, 0x5A, len);
        memcpy(txBuf, &h, sizeof(h));

        cbFired = false;
        uint32_t t0 = micros();
        esp_err_t e = esp_now_send(dst, txBuf, len);
        if (e != ESP_OK) {
            sampAck[i] = -1;
            sampLat[i] = 0;
        } else {
            uint32_t deadline = t0 + 100000;
            while (!cbFired && (int32_t)(micros() - deadline) < 0) {
            }
            if (cbFired) {
                sampAck[i] = (cbStatus == ESP_NOW_SEND_SUCCESS) ? 1 : 0;
                sampLat[i] = cbMicros - t0;
                if (sampAck[i]) nAck++;
            } else {
                sampAck[i] = -1;
                sampLat[i] = 0;
                nTimeout++;
            }
        }
        if (interval > 0) delay(interval);
    }
    uint32_t elapsed = micros() - t0all;

    double sum = 0, sumsq = 0;
    uint32_t lmin = 0xFFFFFFFF, lmax = 0;
    int n = 0;
    for (int i = 0; i < count; i++) {
        if (sampAck[i] != 1) continue;
        double v = sampLat[i];
        sum += v;
        sumsq += v * v;
        if (sampLat[i] < lmin) lmin = sampLat[i];
        if (sampLat[i] > lmax) lmax = sampLat[i];
        n++;
    }
    double mean = n ? sum / n : 0;
    double sd = n > 1 ? sqrt(sumsq / n - mean * mean) : 0;

    Serial.printf("BURST dst=%s txq=%d n=%d ack=%d fail=%d timeout=%d elapsed_us=%lu "
                  "ack_lat_mean_us=%.1f ack_lat_sd_us=%.1f ack_lat_min_us=%lu ack_lat_max_us=%lu\n",
                  macs, (int)curTxq, count, nAck, count - nAck - nTimeout, nTimeout,
                  (unsigned long)elapsed, mean, sd,
                  (unsigned long)(n ? lmin : 0), (unsigned long)lmax);
}

static void cmdScan() {
    esp_wifi_set_promiscuous(false);
    int n = WiFi.scanNetworks(false, true);
    for (int i = 0; i < n; i++) {
        char mac[18];
        macToStr(WiFi.BSSID(i), mac);
        Serial.printf("AP bssid=%s rssi=%d ch=%d ssid=%s\n",
                      mac, WiFi.RSSI(i), WiFi.channel(i), WiFi.SSID(i).c_str());
    }
    WiFi.scanDelete();
    lockChannel();
    Serial.printf("SCAN end n=%d\n", n);
}

static void cmdNoise(int ms) {
    if (ms <= 0) ms = 3000;
    ambCount = 0;
    ambNfSum = 0;
    ambRssiSum = 0;
    ambNfMin = 127;
    ambNfMax = -128;
    uint32_t t0 = millis();
    while (millis() - t0 < (uint32_t)ms) delay(10);
    uint32_t n = ambCount;
    Serial.printf("NOISE ms=%d frames=%lu nf_mean=%.2f nf_min=%d nf_max=%d rssi_mean=%.2f frames_per_s=%.1f\n",
                  ms, (unsigned long)n,
                  n ? (double)ambNfSum / n : 0.0, (int)ambNfMin, (int)ambNfMax,
                  n ? (double)ambRssiSum / n : 0.0,
                  n * 1000.0 / ms);
}

static void handleLine(char *line) {
    char *sp = strchr(line, ' ');
    char *arg = sp ? sp + 1 : (char *)"";
    if (sp) *sp = 0;

    if (!strcmp(line, "ID")) cmdId();
    else if (!strcmp(line, "TXP")) cmdTxp(atoi(arg));
    else if (!strcmp(line, "PEER")) cmdPeer(arg);
    else if (!strcmp(line, "RXCLEAR")) cmdRxClear();
    else if (!strcmp(line, "RXSTAT")) cmdRxStat();
    else if (!strcmp(line, "NODESTAT")) cmdNodeStat();
    else if (!strcmp(line, "SCAN")) cmdScan();
    else if (!strcmp(line, "NOISE")) cmdNoise(atoi(arg));
    else if (!strcmp(line, "BURST")) {
        char macs[32];
        int count = 50, interval = 5, len = 32, echo = 0;
        if (sscanf(arg, "%31s %d %d %d %d", macs, &count, &interval, &len, &echo) >= 1)
            cmdBurst(macs, count, interval, len, echo);
        else Serial.println("ERR args");
    }
    else if (!strcmp(line, "PING")) Serial.println("PONG");
    else Serial.printf("ERR unknown '%s'\n", line);
}

void setup() {
    Serial.begin(115200);
    delay(300);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_ps(WIFI_PS_NONE);

    esp_now_init();
    esp_now_register_send_cb(onSent);
    esp_now_register_recv_cb(onRecv);

    lockChannel();
    wifi_promiscuous_filter_t filt = {};
    filt.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT;
    esp_wifi_set_promiscuous_filter(&filt);
    esp_wifi_set_promiscuous_rx_cb(onPromisc);

    Serial.println();
    cmdId();
    Serial.println("READY");
}

void loop() {
    static char buf[128];
    static int len = 0;

    if (echoPending) {
        RfHdr h = echoHdr;
        memset(txBuf, 0x5A, 32);
        memcpy(txBuf, &h, sizeof(h));
        if (peerIndex(echoDst) >= 0) esp_now_send(echoDst, txBuf, 32);
        echoPending = false;
    }

    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            buf[len] = 0;
            if (len) handleLine(buf);
            len = 0;
        } else if (len < (int)sizeof(buf) - 1) {
            buf[len++] = c;
        }
    }
}
