/*
 * BS-26 Retrofit Receiver — skeleton for biolum board B.
 *
 * Listens for BS-26 JSON state packets over ESP-NOW broadcast.
 * Parses into a typed struct and logs changes. Proof-of-life:
 * strip 0 pixel 0 = green when receiving, red when timed out.
 *
 * Hardware: ESP32-D0WD-V3 (biolum board B)
 *   2 × WS2812B RGB strips on GPIO 18, 19
 *   100 LEDs each
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <NeoPixelBus.h>
#include <ArduinoJson.h>

// ── Config ──────────────────────────────────────────────────────
#define FIXED_CHANNEL    1
#define HEARTBEAT_TIMEOUT_MS 3000
#define LEDS_PER_STRIP  100

// ── BS-26 state ─────────────────────────────────────────────────
struct Bs26State {
    bool     dc;
    bool     ac;
    uint8_t  hundreds;   // 1–5
    uint8_t  tens;       // 0–11
    uint8_t  ones;       // 0–9
    uint16_t decade;     // 0–1199
    uint8_t  ua_dac;
    uint8_t  ua_max;
    uint8_t  v_dac;
    uint8_t  v_max;
    uint32_t seq;
};

static volatile Bs26State bs26 = {};
static volatile uint32_t  bs26LastMs = 0;
static volatile uint32_t  bs26PktCount = 0;
static volatile bool      bs26Updated = false;

// ── JSON parse buffer (ESP-NOW callback context) ────────────────
static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len < 2 || len > 250) return;

    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, (const char*)data, len);
    if (err) return;

    // Must have seq field to be a valid state packet
    if (!doc.containsKey("seq")) return;

    bs26.dc       = doc["dc"] | false;
    bs26.ac       = doc["ac"] | false;
    bs26.hundreds = doc["hundreds"] | 0;
    bs26.tens     = doc["tens"] | 0;
    bs26.ones     = doc["ones"] | 0;
    bs26.decade   = doc["decade"] | 0;

    JsonObject ua = doc["ua"];
    if (ua) {
        bs26.ua_dac = ua["dac"] | 0;
        bs26.ua_max = ua["max"] | 0;
    }
    JsonObject v = doc["v"];
    if (v) {
        bs26.v_dac = v["dac"] | 0;
        bs26.v_max = v["max"] | 0;
    }

    bs26.seq = doc["seq"] | 0;
    bs26LastMs = millis();
    bs26PktCount++;
    bs26Updated = true;
}

// ── LED strips ──────────────────────────────────────────────────
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 19);

static void clearAll() {
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0));
}

static void showAll() {
    strip0.Show(); strip1.Show();
}

// ── Setup ───────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);

    strip0.Begin(); strip1.Begin();
    clearAll();
    showAll();

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed");
        return;
    }
    esp_now_register_recv_cb(onReceive);

    Serial.printf("BS-26 receiver ready — ch=%u\n", FIXED_CHANNEL);
}

// ── Main loop ───────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    bool alive = (bs26LastMs > 0 && (now - bs26LastMs) < HEARTBEAT_TIMEOUT_MS);

    clearAll();
    if (alive) {
        Bs26State s;
        memcpy(&s, (const void*)&bs26, sizeof(s));

        // Strip 0: decade bar graph (0–1199 mapped to 0–99 LEDs)
        int barLen = (int)((float)s.decade / 1199.0f * LEDS_PER_STRIP);
        if (barLen > LEDS_PER_STRIP) barLen = LEDS_PER_STRIP;
        for (int i = 0; i < barLen; i++) {
            uint8_t hue = (uint8_t)(i * 255 / LEDS_PER_STRIP);
            // Simple rainbow: rotate through R→G→B
            uint8_t r, g, b;
            if (hue < 85)      { r = 255 - hue * 3; g = hue * 3;       b = 0; }
            else if (hue < 170){ hue -= 85; r = 0;   g = 255 - hue * 3; b = hue * 3; }
            else               { hue -= 170; r = hue * 3; g = 0;        b = 255 - hue * 3; }
            strip0.SetPixelColor(i, RgbColor(r / 4, g / 4, b / 4));
        }

        // Strip 1: switch/knob indicators
        // DC switch: pixels 0-9
        RgbColor dcColor = s.dc ? RgbColor(0, 40, 0) : RgbColor(10, 0, 0);
        for (int i = 0; i < 10; i++) strip1.SetPixelColor(i, dcColor);
        // AC switch: pixels 10-19
        RgbColor acColor = s.ac ? RgbColor(0, 0, 40) : RgbColor(10, 0, 0);
        for (int i = 10; i < 20; i++) strip1.SetPixelColor(i, acColor);
        // uA meter: pixels 30-69
        if (s.ua_max > 0) {
            int uaLen = (int)((float)s.ua_dac / (float)s.ua_max * 40);
            for (int i = 0; i < uaLen && i < 40; i++)
                strip1.SetPixelColor(30 + i, RgbColor(40, 20, 0));
        }
        // V meter: pixels 70-99
        if (s.v_max > 0) {
            int vLen = (int)((float)s.v_dac / (float)s.v_max * 30);
            for (int i = 0; i < vLen && i < 30; i++)
                strip1.SetPixelColor(70 + i, RgbColor(0, 20, 40));
        }
    } else {
        // No signal: red pulse on first 10 pixels both strips
        for (int i = 0; i < 10; i++) {
            strip0.SetPixelColor(i, RgbColor(30, 0, 0));
            strip1.SetPixelColor(i, RgbColor(30, 0, 0));
        }
    }
    showAll();

    // Log on state change
    if (bs26Updated) {
        bs26Updated = false;
        Bs26State s;
        memcpy(&s, (const void*)&bs26, sizeof(s));
        Serial.printf("[BS26] seq=%lu dc=%d ac=%d decade=%u (%u/%u/%u) ua=%u/%u v=%u/%u\n",
                      (unsigned long)s.seq, s.dc, s.ac, s.decade,
                      s.hundreds, s.tens, s.ones,
                      s.ua_dac, s.ua_max, s.v_dac, s.v_max);
    }

    // Periodic status
    static uint32_t lastLogMs = 0;
    if (now - lastLogMs > 5000) {
        Serial.printf("[STATUS] %s pkts=%lu\n",
                      alive ? "CONNECTED" : "NO SIGNAL",
                      (unsigned long)bs26PktCount);
        lastLogMs = now;
    }

    delay(16);
}
