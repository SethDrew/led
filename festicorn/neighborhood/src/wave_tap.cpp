/*
 * NEIGHBORHOOD — wave_tap: taps spawn leaves at the strip END, blow toward LED 0.
 *
 * Resurrected from biolum wave_pulse (git 84a2608), direction reversed.
 * Tap = hard gate peak_axis_ac_g >= 0.50 g + 120 ms refractory — every
 * strike fires, drumming works (contrast: biolum quiet_bloom surprise-EMA).
 *
 * Hardware: classic ESP32, WS2811 RGB, 50 LEDs (all 6 GPIOs driven: 4, 15, 17, 5, 18, 19).
 * Sender: v1 16-byte TelemetryPacketV1 @ 25 Hz, ESP-NOW channel 6.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <fast_math.h>
#include <oklch_lut.h>
#include "v1_packet.h"

#define FIXED_CHANNEL 6

static const uint8_t  STRIP_PINS[] = { 4, 15, 17, 5, 18, 19 };
static const uint8_t  NUM_STRIPS   = sizeof(STRIP_PINS) / sizeof(STRIP_PINS[0]);
static const uint16_t STRIP_LEN[NUM_STRIPS] = { 50, 50, 50, 50, 50, 50 };

#define BRIGHTNESS_CAP   0.30f
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG  180.0f

#define COUNTS_PER_G     8192.0f
#define COUNTS_PER_INT8   256.0f

#define WP_MAX_LEAVES       16
#define WP_SPEED            20.0f
#define WP_LEAF_SIGMA        1.5f
#define WP_FADE_IN_TIME      0.20f
#define WP_FADE_OUT_LEDS     8.0f
#define WP_MAX_BRIGHTNESS    0.85f
#define WP_DEFAULT_R        50.0f
#define WP_DEFAULT_G       200.0f
#define WP_DEFAULT_B        80.0f
#define WP_TAP_THRESH_G      0.50f
#define WP_COOLDOWN_MS     120

struct WPLeaf {
    float pos;
    float colR, colG, colB;
    float brightness;
    float age;
    bool  active;
};

static WPLeaf   wpLeaves[NUM_STRIPS][WP_MAX_LEAVES];
static uint32_t wpLastHitMs = 0;
static uint32_t tapCount    = 0;
static float    lastPeakG   = 0.0f;

static volatile uint32_t pktCount = 0;
static volatile uint32_t lastPacketMs = 0;
static TelemetryPacketV1 latestPacket = {0};
static uint32_t prevPktCount = 0;

static volatile uint8_t lastMac45[2] = {0, 0};

static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len != sizeof(TelemetryPacketV1)) return;
    memcpy((void*)&latestPacket, data, sizeof(TelemetryPacketV1));
    lastMac45[0] = mac[4];
    lastMac45[1] = mac[5];
    lastPacketMs = millis();
    pktCount++;
}

static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(STRIP_LEN[0], STRIP_PINS[0]);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(STRIP_LEN[1], STRIP_PINS[1]);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(STRIP_LEN[2], STRIP_PINS[2]);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(STRIP_LEN[3], STRIP_PINS[3]);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(STRIP_LEN[4], STRIP_PINS[4]);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(STRIP_LEN[5], STRIP_PINS[5]);

static inline void setPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
        case 4: strip4.SetPixelColor(i, c); break;
        case 5: strip5.SetPixelColor(i, c); break;
    }
}

static void showAll() {
    // Stagger Show()s — see engineering ledger esp32-rmt-simultaneous-show-glitch-frames
    strip0.Show(); strip1.Show(); strip2.Show();
    while (!strip0.CanShow() || !strip1.CanShow() || !strip2.CanShow()) {}
    strip3.Show(); strip4.Show(); strip5.Show();
}

static uint32_t prngState;
static inline uint32_t xorshift32() {
    prngState ^= prngState << 13;
    prngState ^= prngState >> 17;
    prngState ^= prngState << 5;
    return prngState;
}
static inline float randFloat() {
    return (float)(xorshift32() & 0xFFFFFF) / 16777216.0f;
}
static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline float axisMeanG(int8_t m) {
    return ((float)m * COUNTS_PER_INT8) / COUNTS_PER_G;
}
static void vecNormalize(float &x, float &y, float &z) {
    float len = sqrtf(x*x + y*y + z*z);
    if (len > 0) { x /= len; y /= len; z /= len; }
}

static float restAx = 0, restAy = 0, restAz = 1;
static bool  calibrated = false;
static float calSumAx = 0, calSumAy = 0, calSumAz = 0;
static uint32_t calSamples = 0;
static uint32_t calStartMs = 0;
#define CAL_DURATION_MS 2000

static void startCalibration() {
    calibrated = false;
    calSumAx = 0; calSumAy = 0; calSumAz = 0;
    calSamples = 0;
    calStartMs = millis();
    Serial.println("Calibrating — keep sender still for 2 seconds...");
}

static void updateCalibration(float ax, float ay, float az) {
    if (calibrated) return;
    calSumAx += ax; calSumAy += ay; calSumAz += az;
    calSamples++;
    if (millis() - calStartMs >= CAL_DURATION_MS && calSamples > 0) {
        restAx = calSumAx / calSamples;
        restAy = calSumAy / calSamples;
        restAz = calSumAz / calSamples;
        vecNormalize(restAx, restAy, restAz);
        calibrated = true;
        Serial.printf("Calibrated! Rest vector: (%.4f, %.4f, %.4f)\n",
                      restAx, restAy, restAz);
    }
}

static void wpPickColor(float angleDeg, float &cR, float &cG, float &cB) {
    if (angleDeg <= DEADZONE_DEG) {
        cR = WP_DEFAULT_R; cG = WP_DEFAULT_G; cB = WP_DEFAULT_B;
        return;
    }
    float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
    if (hueFrac > 1.0f) hueFrac = 1.0f;
    uint8_t hueIdx = (uint8_t)(hueFrac * 255.0f);
    cR = (float)oklchVarL[hueIdx][0];
    cG = (float)oklchVarL[hueIdx][1];
    cB = (float)oklchVarL[hueIdx][2];
}

static void wavePulseProcessMotion(uint32_t now, float angleDeg) {
    float ax_pk = fmaxf(fabsf((float)latestPacket.ax_max), fabsf((float)latestPacket.ax_min));
    float ay_pk = fmaxf(fabsf((float)latestPacket.ay_max), fabsf((float)latestPacket.ay_min));
    float az_pk = fmaxf(fabsf((float)latestPacket.az_max), fabsf((float)latestPacket.az_min));
    float peak_axis_ac_g = fmaxf(ax_pk, fmaxf(ay_pk, az_pk))
                         * (COUNTS_PER_INT8 / COUNTS_PER_G);
    lastPeakG = peak_axis_ac_g;

    if (peak_axis_ac_g < WP_TAP_THRESH_G) return;
    if ((now - wpLastHitMs) < WP_COOLDOWN_MS) return;
    wpLastHitMs = now;
    tapCount++;

    Serial.printf("[tap] peak=%.2fg angle=%.0f\n", peak_axis_ac_g, angleDeg);

    float overshoot = (peak_axis_ac_g - WP_TAP_THRESH_G) / 2.5f;
    float hitIntensity = clampf(overshoot, 0.0f, 1.0f);
    int nLeaves = 1 + (int)(hitIntensity * 3.0f + 0.5f);
    if (nLeaves > 4) nLeaves = 4;

    float baseR, baseG, baseB;
    wpPickColor(angleDeg, baseR, baseG, baseB);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float endPos = (float)(STRIP_LEN[s] - 1);
        for (int n = 0; n < nLeaves; n++) {
            int8_t slot = -1;
            for (uint8_t j = 0; j < WP_MAX_LEAVES; j++) {
                if (!wpLeaves[s][j].active) { slot = j; break; }
            }
            if (slot < 0) break;

            WPLeaf &lf = wpLeaves[s][slot];
            lf.pos       = endPos + (float)n * 0.7f + randFloat() * 0.5f;
            lf.colR       = clampf(baseR * (0.9f + 0.2f * randFloat()), 0.0f, 255.0f);
            lf.colG       = clampf(baseG * (0.9f + 0.2f * randFloat()), 0.0f, 255.0f);
            lf.colB       = clampf(baseB * (0.9f + 0.2f * randFloat()), 0.0f, 255.0f);
            lf.brightness = 0.0f;
            lf.age        = 0.0f;
            lf.active     = true;
        }
    }
}

static void renderWavePulse(float dt) {
    float sigma_sq2  = 2.0f * WP_LEAF_SIGMA * WP_LEAF_SIGMA;

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t i = 0; i < WP_MAX_LEAVES; i++) {
            WPLeaf &lf = wpLeaves[s][i];
            if (!lf.active) continue;

            lf.pos -= WP_SPEED * dt;
            lf.age += dt;

            lf.brightness = (lf.age < WP_FADE_IN_TIME)
                            ? (lf.age / WP_FADE_IN_TIME) : 1.0f;

            if (lf.pos < WP_FADE_OUT_LEDS && lf.pos >= 0.0f)
                lf.brightness *= lf.pos / WP_FADE_OUT_LEDS;

            // Deactivate at LED 0 — rendering past it lets the gaussian
            // tail bleed back into the first LEDs at full weight.
            if (lf.pos <= 0.0f)
                lf.active = false;
        }

        for (uint16_t li = 0; li < STRIP_LEN[s]; li++) {
            float glow = 0.0f, oR = 0.0f, oG = 0.0f, oB = 0.0f;
            for (uint8_t j = 0; j < WP_MAX_LEAVES; j++) {
                const WPLeaf &lf = wpLeaves[s][j];
                if (!lf.active) continue;
                float d = (float)li - lf.pos;
                float w = expf(-(d * d) / sigma_sq2) * lf.brightness;
                glow += w;
                oR   += w * lf.colR;
                oG   += w * lf.colG;
                oB   += w * lf.colB;
            }

            uint8_t r8 = 0, g8 = 0, b8 = 0;
            if (glow > 1e-6f) {
                float bright = clampf(glow, 0.0f, 1.0f) * WP_MAX_BRIGHTNESS;
                float linBright = fastGamma24(bright) * BRIGHTNESS_CAP;
                r8 = (uint8_t)clampf((oR / glow) * linBright, 0.0f, 255.0f);
                g8 = (uint8_t)clampf((oG / glow) * linBright, 0.0f, 255.0f);
                b8 = (uint8_t)clampf((oB / glow) * linBright, 0.0f, 255.0f);
            }
            setPixel(s, li, r8, g8, b8);
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(300);
    prngState = esp_random() | 1;

    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
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

    Serial.printf("Neighborhood wave_tap ready — ch=%u MAC=%s\n",
                  FIXED_CHANNEL, WiFi.macAddress().c_str());
    Serial.println("Taps spawn at strip END, blow toward LED 0. 'c' = recalibrate");
}

void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / 60.0f);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    bool connected = lastPacketMs > 0 && (now - lastPacketMs) < 3000;

    float angleDeg = 0.0f;
    if (connected && pktCount != prevPktCount) {
        prevPktCount = pktCount;

        static uint32_t prevArriveMs = 0;
        static uint16_t prevSeq = 0;
        uint32_t arriveMs = lastPacketMs;
        float ax_pk = fmaxf(fabsf((float)latestPacket.ax_max), fabsf((float)latestPacket.ax_min));
        float ay_pk = fmaxf(fabsf((float)latestPacket.ay_max), fabsf((float)latestPacket.ay_min));
        float az_pk = fmaxf(fabsf((float)latestPacket.az_max), fabsf((float)latestPacket.az_min));
        float pk = fmaxf(ax_pk, fmaxf(ay_pk, az_pk)) * (COUNTS_PER_INT8 / COUNTS_PER_G);
        uint16_t dseq = latestPacket.seq - prevSeq;
        if (pk >= 0.15f || dseq != 1 || (arriveMs - prevArriveMs) > 80) {
            Serial.printf("PKT t=%lu mac=%02X:%02X seq=%u dseq=%u dms=%lu peak=%.2f\n",
                          (unsigned long)arriveMs, lastMac45[0], lastMac45[1],
                          latestPacket.seq, dseq,
                          (unsigned long)(arriveMs - prevArriveMs), pk);
        }
        prevArriveMs = arriveMs;
        prevSeq = latestPacket.seq;
        float ax = axisMeanG(latestPacket.ax_mean);
        float ay = axisMeanG(latestPacket.ay_mean);
        float az = axisMeanG(latestPacket.az_mean);
        vecNormalize(ax, ay, az);

        if (calStartMs == 0) startCalibration();
        updateCalibration(ax, ay, az);

        if (calibrated) {
            float cosAngle = clampf(restAx * ax + restAy * ay + restAz * az, -1.0f, 1.0f);
            angleDeg = acosf(cosAngle) * (180.0f / M_PI);
        }
        wavePulseProcessMotion(now, angleDeg);
    }

    renderWavePulse(dt);
    showAll();

    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'c') startCalibration();
    }

    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    if (now - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (now - lastLogMs);
        int activeLeaves = 0;
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint8_t j = 0; j < WP_MAX_LEAVES; j++)
                if (wpLeaves[s][j].active) activeLeaves++;
        Serial.printf("FPS=%.1f %s pkts=%lu taps=%lu peak=%.2fg leaves=%d cal=%d\n",
                      fps, connected ? "LIVE" : "----",
                      (unsigned long)pktCount, (unsigned long)tapCount,
                      lastPeakG, activeLeaves, calibrated ? 1 : 0);
        frameCount = 0;
        lastLogMs = now;
    }
}
