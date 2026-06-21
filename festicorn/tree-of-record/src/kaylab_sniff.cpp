/*
 * kaylab_sniff — ESP-NOW presence sniffer. Answers one question: is the BS-26
 * (Kay Lab, MAC E0:8C:FE:57:A1:18) actually transmitting, and on what channel?
 *
 * No LEDs. Brings up the radio in STA mode, registers an ESP-NOW receive
 * callback, and HOPS the Wi-Fi channel 1..13 (~600 ms dwell each). esp_now RX
 * only fires when we're parked on the sender's channel, so a hit tells us both
 * "Kaylab is alive" and "it's on channel N". If channel N != 1, that explains
 * why led-eth-0 (locked to ch 1) hears nothing.
 *
 * Per-second line prints the current dwell channel and total packets seen, with
 * a breakdown of any Kaylab hit (its channel + payload length). Flash to any
 * spare board near Kaylab (e.g. biolum-B). Build: env:kaylab-sniff.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// BS-26 / Kaylab broadcast sender.
static const uint8_t KAYLAB_MAC[6] = {0xE0, 0x8C, 0xFE, 0x57, 0xA1, 0x18};

static volatile uint32_t totalPkts   = 0;
static volatile uint32_t kaylabPkts  = 0;
static volatile uint8_t  kaylabChan   = 0;   // channel where Kaylab was last heard
static volatile int      kaylabLen    = 0;
static volatile uint8_t  lastSrc[6]   = {0};
static volatile uint8_t  curChannel   = 1;

static bool isKaylab(const uint8_t *m) {
    for (uint8_t i = 0; i < 6; i++) if (m[i] != KAYLAB_MAC[i]) return false;
    return true;
}

static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    totalPkts++;
    for (uint8_t i = 0; i < 6; i++) lastSrc[i] = mac[i];
    if (isKaylab(mac)) {
        kaylabPkts++;
        kaylabChan = curChannel;
        kaylabLen  = len;
    }
}

void setup() {
    Serial.begin(115200);
    delay(300);
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    if (esp_now_init() != ESP_OK) {
        Serial.println("[sniff] esp_now_init FAILED");
        return;
    }
    esp_now_register_recv_cb(onReceive);

    Serial.println("[sniff] hunting for Kaylab E0:8C:FE:57:A1:18 across channels 1..13");
    Serial.println("[sniff] (esp_now RX only fires on the sender's channel; dwell ~600ms each)");
}

void loop() {
    static uint8_t  ch         = 1;
    static uint32_t hopMs      = 0;
    static uint32_t lastLogMs  = 0;
    const  uint32_t DWELL_MS   = 600;

    uint32_t now = millis();

#ifdef SNIFF_LOCK_CHANNEL
    // Park on one channel so event-driven (on-change) bursts aren't missed.
    static bool locked = false;
    if (!locked) { curChannel = SNIFF_LOCK_CHANNEL;
                   esp_wifi_set_channel(SNIFF_LOCK_CHANNEL, WIFI_SECOND_CHAN_NONE);
                   locked = true; }
#else
    if (now - hopMs >= DWELL_MS) {
        ch = (uint8_t)(ch % 13) + 1;
        curChannel = ch;
        esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
        hopMs = now;
    }
#endif

    if (now - lastLogMs >= 1000) {
        if (kaylabPkts > 0) {
            Serial.printf("[sniff] *** KAYLAB HEARD *** ch=%u  len=%d  kaylabPkts=%lu  (total RX=%lu)\n",
                          kaylabChan, kaylabLen, (unsigned long)kaylabPkts, (unsigned long)totalPkts);
        } else {
            Serial.printf("[sniff] dwell ch=%u  total RX=%lu  lastSrc=%02X:%02X:%02X:%02X:%02X:%02X  (no Kaylab yet)\n",
                          curChannel, (unsigned long)totalPkts,
                          lastSrc[0], lastSrc[1], lastSrc[2], lastSrc[3], lastSrc[4], lastSrc[5]);
        }
        lastLogMs = now;
    }
}
