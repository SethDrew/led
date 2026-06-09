#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <ArduinoJson.h>

// BS-26 knob stability monitor: receives ESP-NOW JSON packets on ch1,
// prints every packet's knob fields, flags any change from previous.

struct KnobState {
    int ones, tens, hundreds, decouter, decmid, decinner, seq;
    unsigned long up;
    int boot;
    int rawH, rawT, rawO, adsT, adsO, adsRef;
    bool valid;
};

static volatile bool pktReady = false;
static uint8_t pktBuf[300];
static int pktLen = 0;
static uint8_t pktMac[6];

static void onReceive(const uint8_t* mac, const uint8_t* data, int len) {
    if (pktReady) return;  // drop if main loop hasn't consumed yet
    if (len < 2 || len > 299) return;
    memcpy(pktBuf, data, len);
    pktBuf[len] = 0;
    pktLen = len;
    memcpy(pktMac, mac, 6);
    pktReady = true;
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

    // Add broadcast peer (same fix as tree-of-record)
    esp_now_peer_info_t peer = {};
    memset(peer.peer_addr, 0xFF, 6);
    peer.channel = 1;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    // Re-assert channel after init
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    Serial.printf("\n[BOOT] bs26-stability-monitor MAC=%s ch=1\n",
                  WiFi.macAddress().c_str());
    Serial.println("[READY] waiting for BS-26 packets...\n");
    Serial.println("seq     ones tens hund dout dmid din  | up(s)  boot | rawH  rawT  rawO  adsT  adsO  aRef | changes");
    Serial.println("------- ---- ---- ---- ---- ---- ---- | ------ ---- | ----- ----- ----- ----- ----- ---- | -------");
}

void loop() {
    static KnobState prev = {};
    static uint32_t pktCount = 0;
    static uint32_t changeCount = 0;

    if (!pktReady) {
        // Heartbeat every 5s if no packets
        static uint32_t lastHb = 0;
        if (millis() - lastHb > 5000) {
            lastHb = millis();
            Serial.printf("[%lus] pkts=%lu changes=%lu (waiting...)\n",
                          millis() / 1000, pktCount, changeCount);
        }
        delay(10);
        return;
    }

    // Parse JSON
    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, (const char*)pktBuf, pktLen);
    pktReady = false;

    if (err || (!doc.containsKey("s") && !doc.containsKey("seq"))) {
        Serial.printf("[?] non-JSON or no seq: len=%d from %02X:%02X\n",
                      pktLen, pktMac[4], pktMac[5]);
        return;
    }

    KnobState cur;
    // Support both old (long key) and new (short key) formats
    cur.seq      = doc["s"] | (doc["seq"] | 0);
    cur.ones     = doc["o"] | (doc["ones"] | -1);
    cur.tens     = doc["t"] | (doc["tens"] | -1);
    cur.hundreds = doc["h"] | (doc["hundreds"] | -1);
    cur.decouter = doc["do"] | (doc["decouter"] | -1);
    cur.decmid   = doc["dm"] | (doc["decmid"] | -1);
    cur.decinner = doc["di"] | (doc["decinner"] | -1);
    cur.up       = doc["up"] | 0UL;
    cur.boot     = doc["b"] | (doc["boot"] | -1);
    cur.rawH     = doc["rh"] | (doc["raw"]["hundreds"] | -1);
    cur.rawT     = doc["rt"] | (doc["raw"]["tens"] | -1);
    cur.rawO     = doc["ro"] | (doc["raw"]["ones"] | -1);
    cur.adsT     = doc["at"] | (doc["raw"]["adsTens"] | -1);
    cur.adsO     = doc["ao"] | (doc["raw"]["adsOnes"] | -1);
    cur.adsRef   = doc["ar"] | -1;
    cur.valid    = true;
    pktCount++;

    // Check for changes
    char changes[128] = "";
    if (prev.valid) {
        char* p = changes;
        if (cur.ones != prev.ones)         p += sprintf(p, "ones:%d→%d ", prev.ones, cur.ones);
        if (cur.tens != prev.tens)         p += sprintf(p, "tens:%d→%d ", prev.tens, cur.tens);
        if (cur.hundreds != prev.hundreds) p += sprintf(p, "hund:%d→%d ", prev.hundreds, cur.hundreds);
        if (cur.decouter != prev.decouter) p += sprintf(p, "dout:%d→%d ", prev.decouter, cur.decouter);
        if (cur.decmid != prev.decmid)     p += sprintf(p, "dmid:%d→%d ", prev.decmid, cur.decmid);
        if (cur.decinner != prev.decinner) p += sprintf(p, "din:%d→%d ", prev.decinner, cur.decinner);
        if (cur.boot != prev.boot && prev.boot > 0) p += sprintf(p, "**REBOOT** ");
        if (changes[0]) changeCount++;
    }

    Serial.printf("%-7d %4d %4d %4d %4d %4d %4d | %5lu %4d | %5d %5d %5d %5d %5d %4d | %s\n",
                  cur.seq, cur.ones, cur.tens, cur.hundreds,
                  cur.decouter, cur.decmid, cur.decinner,
                  cur.up / 1000, cur.boot,
                  cur.rawH, cur.rawT, cur.rawO, cur.adsT, cur.adsO, cur.adsRef,
                  changes[0] ? changes : "—");

    prev = cur;
}
