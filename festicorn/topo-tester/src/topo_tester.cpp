/*
 * topo-tester — visual validation tool for the 2D LED topology.
 *
 * Same hardware as bulb-fleet (6× WS2812B, 100 LEDs each). Cycle modes
 * via serial ('1'..'6') or UDP port 4211 (single-byte payload).
 *
 *   1 strip-id      — solid color per strip (R/G/B/Y/C/M) so user can
 *                     match strip index to physical path
 *   2 sequential    — one strip white at a time, 3s per strip
 *   3 direction     — gradient green(LED0) → red(LED99) per strip
 *   4 radial        — color by distance from vertex 0 (red near, blue far)
 *   5 quadrant      — color by 2D quadrant (TL=R, TR=G, BL=B, BR=Y)
 *   6 wave          — animated radial wave from vertex 0
 *   7 pot-fill      — fill each strip proportional to potentiometer (GPIO 34)
 *
 * Potentiometer: GPIO 34 (ADC1_CH6), wiped against 3.3V. Raw 12-bit value
 * (0–4095) and mapped byte (0–255) printed every 500ms. `g_potByte` is
 * available globally so other modes can use it (e.g. as a brightness knob).
 */

#include <Arduino.h>
#include <WiFi.h>
#include <AsyncUDP.h>
#include <NeoPixelBus.h>
#include <math.h>
#include "topology.h"

#ifndef WIFI_SSID
#define WIFI_SSID "cuteplant"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS "bigboiredwood"
#endif

#define CMD_PORT 4211

#define POT_PIN     34
#define POT_PRINT_MS 500

// Global pot state — readable from any mode.
volatile uint16_t g_potRaw  = 0;     // 0..4095
volatile uint8_t  g_potByte = 0;     // 0..255

static AsyncUDP cmdUdp;

static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(TOPO_LEDS,  4);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(TOPO_LEDS, 16);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(TOPO_LEDS, 17);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(TOPO_LEDS,  5);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(TOPO_LEDS, 18);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(TOPO_LEDS, 19);

static volatile uint8_t currentMode = 8;
static uint8_t lastMode = 0;

// Global brightness cap — installations are bright; keep test modes mild.
static const float BRIGHTNESS = 0.4f;

static inline uint8_t scaleB(float v) {
    v *= BRIGHTNESS;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (uint8_t)v;
}

static inline void setPx(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    if (i >= TOPO_LEDS) return;
    RgbColor c(scaleB(r), scaleB(g), scaleB(b));
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
        case 4: strip4.SetPixelColor(i, c); break;
        case 5: strip5.SetPixelColor(i, c); break;
    }
}

static inline void stripsBegin() {
    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
}

static inline void stripsShow() {
    strip0.Show(); strip1.Show(); strip2.Show();
    while (!strip0.CanShow() || !strip1.CanShow() || !strip2.CanShow()) {}
    strip3.Show(); strip4.Show(); strip5.Show();
}

static inline void clearAll() {
    RgbColor k(0,0,0);
    strip0.ClearTo(k); strip1.ClearTo(k); strip2.ClearTo(k);
    strip3.ClearTo(k); strip4.ClearTo(k); strip5.ClearTo(k);
}

// ── Modes ────────────────────────────────────────────────────────

// 1: each strip a solid identifying color.
static const uint8_t STRIP_COLORS[6][3] = {
    {255,   0,   0}, // strip 0 — red
    {  0, 255,   0}, // strip 1 — green
    {  0,   0, 255}, // strip 2 — blue
    {255, 255,   0}, // strip 3 — yellow
    {  0, 255, 255}, // strip 4 — cyan
    {255,   0, 255}, // strip 5 — magenta
};

static void renderStripId() {
    for (uint8_t s = 0; s < TOPO_STRIPS; s++)
        for (uint16_t i = 0; i < TOPO_LEDS; i++)
            setPx(s, i, STRIP_COLORS[s][0], STRIP_COLORS[s][1], STRIP_COLORS[s][2]);
}

// 2: one strip lit (white) at a time, advancing every 3s.
static void renderSequential() {
    uint32_t now = millis();
    uint8_t active = (now / 3000) % TOPO_STRIPS;
    for (uint8_t s = 0; s < TOPO_STRIPS; s++) {
        uint8_t v = (s == active) ? 255 : 0;
        for (uint16_t i = 0; i < TOPO_LEDS; i++)
            setPx(s, i, v, v, v);
    }
}

// 3: gradient green (LED 0) → red (LED 99).
static void renderDirection() {
    for (uint8_t s = 0; s < TOPO_STRIPS; s++) {
        for (uint16_t i = 0; i < TOPO_LEDS; i++) {
            float t = (float)i / (float)(TOPO_LEDS - 1);
            uint8_t r = (uint8_t)(t * 255.0f);
            uint8_t g = (uint8_t)((1.0f - t) * 255.0f);
            setPx(s, i, r, g, 0);
        }
    }
}

// Helpers using topology positions.
static inline float dist2D(Vec2 a, Vec2 b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return sqrtf(dx*dx + dy*dy);
}

// 4: radial — red close to vertex 0, blue far.
static void renderRadial() {
    Vec2 origin = VERTICES[0];
    // Find max distance for normalization.
    float maxD = 0.0f;
    for (uint8_t s = 0; s < TOPO_STRIPS; s++)
        for (uint16_t i = 0; i < TOPO_LEDS; i++) {
            float d = dist2D(ledPos[s][i], origin);
            if (d > maxD) maxD = d;
        }
    if (maxD < 1e-6f) maxD = 1.0f;

    for (uint8_t s = 0; s < TOPO_STRIPS; s++) {
        for (uint16_t i = 0; i < TOPO_LEDS; i++) {
            float t = dist2D(ledPos[s][i], origin) / maxD;
            if (t > 1.0f) t = 1.0f;
            uint8_t r = (uint8_t)((1.0f - t) * 255.0f);
            uint8_t b = (uint8_t)(t * 255.0f);
            setPx(s, i, r, 0, b);
        }
    }
}

// 5: quadrant — split around the centroid of the shape (~0.5, 0.5).
static void renderQuadrant() {
    const float cx = 0.5f, cy = 0.5f;
    for (uint8_t s = 0; s < TOPO_STRIPS; s++) {
        for (uint16_t i = 0; i < TOPO_LEDS; i++) {
            Vec2 p = ledPos[s][i];
            bool right = p.x >= cx;
            bool top   = p.y >= cy;
            uint8_t r=0, g=0, b=0;
            if      ( top && !right) { r=255; }              // TL red
            else if ( top &&  right) { g=255; }              // TR green
            else if (!top && !right) { b=255; }              // BL blue
            else                     { r=255; g=255; }       // BR yellow
            setPx(s, i, r, g, b);
        }
    }
}

// 6: radial wave from vertex 0.
static void renderWave(float dt) {
    static float phase = 0.0f;
    phase += dt * 0.6f;
    Vec2 origin = VERTICES[0];
    const float k = 6.0f; // spatial frequency (cycles per unit distance)
    for (uint8_t s = 0; s < TOPO_STRIPS; s++) {
        for (uint16_t i = 0; i < TOPO_LEDS; i++) {
            float d = dist2D(ledPos[s][i], origin);
            float w = 0.5f + 0.5f * sinf(k * d - phase * 6.2832f);
            uint8_t v = (uint8_t)(w * 255.0f);
            // Cool teal palette so the wave reads as one coherent surface.
            setPx(s, i, 0, v, (uint8_t)(v / 2));
        }
    }
}

// 7: pot-fill — light first N LEDs of each strip in white, where
// N = round(potByte/255 * TOPO_LEDS). Strip-id color tints the lit portion
// so you can still tell strips apart.
static void renderPotFill() {
    uint16_t lit = (uint16_t)((uint32_t)g_potByte * TOPO_LEDS / 255);
    for (uint8_t s = 0; s < TOPO_STRIPS; s++) {
        for (uint16_t i = 0; i < TOPO_LEDS; i++) {
            if (i < lit) {
                setPx(s, i,
                      STRIP_COLORS[s][0],
                      STRIP_COLORS[s][1],
                      STRIP_COLORS[s][2]);
            } else {
                setPx(s, i, 0, 0, 0);
            }
        }
    }
}

// 8: connected-only mapping. Installation has 4 strips wired up; light
// each with a distinct color, leave the unused strips dark.
//   strip 0 (diagram blue)   → red
//   strip 2 (diagram orange) → green
//   strip 4 (diagram black)  → blue
//   strip 5 (diagram pink)   → white
//   strips 1 and 3 → off
static void renderConnectedFour() {
    for (uint16_t i = 0; i < TOPO_LEDS; i++) {
        setPx(0, i, 255,   0,   0);
        setPx(1, i,   0,   0,   0);
        setPx(2, i,   0, 255,   0);
        setPx(3, i,   0,   0,   0);
        setPx(4, i,   0,   0, 255);
        setPx(5, i, 255, 255, 255);
    }
}

// ── Potentiometer ────────────────────────────────────────────────

static void potUpdate() {
    static uint32_t lastPrint = 0;
    int raw = analogRead(POT_PIN);            // 0..4095 (12-bit)
    if (raw < 0) raw = 0;
    if (raw > 4095) raw = 4095;
    g_potRaw  = (uint16_t)raw;
    g_potByte = (uint8_t)(raw >> 4);          // 12-bit → 8-bit

    uint32_t now = millis();
    if (now - lastPrint >= POT_PRINT_MS) {
        lastPrint = now;
        Serial.printf("[POT] raw=%4u  byte=%3u\n",
                      (unsigned)g_potRaw, (unsigned)g_potByte);
    }
}

// ── Command handling ─────────────────────────────────────────────

static void setMode(uint8_t m) {
    if (m < 1 || m > 8) return;
    currentMode = m;
    Serial.printf("[MODE] %u\n", m);
}

static void parseSerial() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c >= '1' && c <= '8') setMode((uint8_t)(c - '0'));
    }
}

// ── Setup / loop ─────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    // ADC setup for potentiometer on GPIO 34 (input-only, ADC1_CH6).
    analogReadResolution(12);
    analogSetPinAttenuation(POT_PIN, ADC_11db);  // full 0–3.3V range
    pinMode(POT_PIN, INPUT);

    initTopology();
    stripsBegin();
    clearAll();
    stripsShow();

    Serial.printf("[BOOT] topo-tester — connecting to %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    WiFi.setAutoReconnect(true);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
        delay(500);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WIFI] IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WIFI] not connected — serial-only");
    }

    if (cmdUdp.listen(CMD_PORT)) {
        cmdUdp.onPacket([](AsyncUDPPacket packet) {
            if (packet.length() < 1) return;
            char c = (char)packet.data()[0];
            if (c >= '1' && c <= '7') {
                setMode((uint8_t)(c - '0'));
                packet.printf("MODE:%c", c);
            }
        });
        Serial.printf("[UDP] cmd on port %d\n", CMD_PORT);
    }

    Serial.println("Modes: 1 strip-id  2 sequential  3 direction  4 radial  5 quadrant  6 wave  7 pot-fill");
}

void loop() {
    static uint32_t lastMs = 0;
    uint32_t now = millis();
    float dt = (lastMs > 0) ? (now - lastMs) / 1000.0f : 0.016f;
    if (dt > 0.1f) dt = 0.1f;
    lastMs = now;

    parseSerial();
    potUpdate();

    if (currentMode != lastMode) {
        clearAll();
        lastMode = currentMode;
    }

    switch (currentMode) {
        case 1: renderStripId();   break;
        case 2: renderSequential(); break;
        case 3: renderDirection(); break;
        case 4: renderRadial();    break;
        case 5: renderQuadrant();  break;
        case 6: renderWave(dt);    break;
        case 7: renderPotFill();   break;
        case 8: renderConnectedFour(); break;
    }

    stripsShow();
    delay(16);
}
