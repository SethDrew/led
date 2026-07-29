#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <kettle_packet_v1.h>

#define ENC_A 20
#define ENC_B 10

static const uint8_t BTN_PINS[] = {21, 9, 7, 6, 5};
static const int NUM_BTNS = sizeof(BTN_PINS);

#define FIXED_CHANNEL 1
#define SEND_MIN_INTERVAL_MS 20
#define HEARTBEAT_MS 1000

static volatile int32_t encCount = 0;
static volatile uint8_t prevState = 0;

static const int8_t QDEC[16] = {
    0, -1, +1, 0,
    +1, 0, 0, -1,
    -1, 0, 0, +1,
    0, +1, -1, 0
};

static hw_timer_t* pollTimer = nullptr;

static void IRAM_ATTR onPoll() {
    uint32_t g = REG_READ(GPIO_IN_REG);
    uint8_t s = (((g >> ENC_A) & 1) << 1) | ((g >> ENC_B) & 1);
    if (s != prevState) {
        encCount += QDEC[(prevState << 2) | s];
        prevState = s;
    }
}

static const uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static KettlePacketV1 pkt = {KETTLE_MAGIC, KETTLE_VER, 0, 0, 0, 0};

void setup() {
    Serial.begin(115200);
    delay(1500);
    pinMode(ENC_A, INPUT_PULLUP);
    pinMode(ENC_B, INPUT_PULLUP);
    for (int i = 0; i < NUM_BTNS; i++) pinMode(BTN_PINS[i], INPUT_PULLUP);
    prevState = (digitalRead(ENC_A) << 1) | digitalRead(ENC_B);
    pollTimer = timerBegin(0, 80, true);
    timerAttachInterrupt(pollTimer, &onPoll, true);
    timerAlarmWrite(pollTimer, 50, true);
    timerAlarmEnable(pollTimer);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    if (esp_now_init() != ESP_OK) {
        Serial.println("{\"err\":\"espnow-init\"}");
        return;
    }
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    // This board (10:00:3B:B1:3E:80) has the SuperMini TX defect: PA dies
    // above 15.75 dBm (catalog + engineering ledger esp32-c3-super-mini-tx-defect).
    // 51 = 2 dB under this unit's load-measured brownout cliff; raising it collapses TX
    // (ledger: esp-now-c3-brownout-cliff-duty-cycle-dependent)
    esp_wifi_set_max_tx_power(51);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BROADCAST_ADDR, 6);
    peer.channel = FIXED_CHANNEL;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
}

void loop() {
    static int stable[NUM_BTNS], lastRaw[NUM_BTNS];
    static uint32_t lastEdge[NUM_BTNS], presses[NUM_BTNS];
    static uint32_t lastSend = 0, lastBeat = 0;
    static int32_t lastSentCount = INT32_MIN;
    static int lastSentBtnMask = -1;
    static uint8_t repeats = 0;
    static bool init = false;
    uint32_t now = millis();

    if (!init) {
        init = true;
        for (int i = 0; i < NUM_BTNS; i++) {
            stable[i] = HIGH; lastRaw[i] = HIGH; lastEdge[i] = 0; presses[i] = 0;
        }
    }

    int btnMask = 0;
    for (int i = 0; i < NUM_BTNS; i++) {
        int raw = digitalRead(BTN_PINS[i]);
        if (raw != lastRaw[i]) { lastRaw[i] = raw; lastEdge[i] = now; }
        if (now - lastEdge[i] > 15 && raw != stable[i]) {
            stable[i] = raw;
            if (stable[i] == LOW) presses[i]++;
        }
        if (stable[i] == LOW) btnMask |= (1 << i);
    }

    int32_t c = encCount;
    // Broadcast has no ACK/retry and this link measured up to 57-80% raw frame
    // loss (see engineering/library/WIRELESS.md). Each change is transmitted 3x
    // over ~40 ms; state is absolute so duplicates are idempotent.
    if (c != lastSentCount || btnMask != lastSentBtnMask) repeats = 3;
    bool due = (now - lastBeat >= HEARTBEAT_MS);
    if ((repeats && now - lastSend >= SEND_MIN_INTERVAL_MS) || due) {
        if (repeats) repeats--;
        lastSentCount = c;
        lastSentBtnMask = btnMask;
        lastBeat = now;
        lastSend = now;

        pkt.seq++;
        pkt.btnBits = (uint8_t)btnMask;
        pkt.enc = c;
        pkt.upMs = now;
        esp_now_send(BROADCAST_ADDR, (const uint8_t*)&pkt, sizeof(pkt));

        uint8_t s = prevState;
        Serial.printf("{\"c\":%ld,\"a\":%d,\"b\":%d,\"btn\":[%d,%d,%d,%d,%d],"
                      "\"n\":[%lu,%lu,%lu,%lu,%lu],\"up\":%lu,\"seq\":%u}\n",
                      (long)c, (s >> 1) & 1, s & 1,
                      (btnMask >> 0) & 1, (btnMask >> 1) & 1, (btnMask >> 2) & 1,
                      (btnMask >> 3) & 1, (btnMask >> 4) & 1,
                      presses[0], presses[1], presses[2], presses[3], presses[4],
                      now, pkt.seq);
    }
    delay(2);
}
