/*
 * RGBW BULBS STANDALONE — classic ESP32, 1 SK6812 RGBW strip, 2 colonies
 *
 * Autonomous port of festicorn/biolum/src/bloom_rgbw.cpp. No ESP-NOW receiver;
 * runs the breath cycle purely from internal state. WiFi station mode + ArduinoOTA
 * for wireless flashing (no USB access to the installation).
 *
 * Wiring (SK6812 NEO_GRBW):
 *   GPIO 14 (LED_PIN) — 100 LEDs (first 50 = colony 0, next 50 = colony 1)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiGeneric.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <ESPAsyncWebServer.h>
#include <esp_random.h>
#include <esp_system.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <delta_sigma.h>
#include <fast_math.h>

// ── Strip layout ─────────────────────────────────────────────────
#ifndef LED_PIN
#define LED_PIN 14
#endif
#ifndef LEDS_TOTAL
#define LEDS_TOTAL 150
#endif

static const uint8_t  COLONIES_PER_STRIP = 2;
static const uint8_t  NUM_COLONIES       = COLONIES_PER_STRIP;
static const uint16_t TOTAL_LEDS         = LEDS_TOTAL;
static const uint16_t LEDS_PER_COLONY    = TOTAL_LEDS / COLONIES_PER_STRIP;

// ── WiFi / OTA ──────────────────────────────────────────────────
static const char *WIFI_SSID = "cuteplant";
static const char *WIFI_PASS = "bigboiredwood";
static const char *OTA_HOSTNAME = "rgbw-bulbs";

// ── Bloom parameters ────────────────────────────────────────────
#define BLOOM_BRIGHTNESS_CAP_C0  0.10f   // first 75 LEDs
#define BLOOM_BRIGHTNESS_CAP_C1  0.06f   // last 75 LEDs
#define BLOOM_NOISE_GATE       256    // per-channel: snap to 0 below 1 LSB in 8.8

#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.65f
#define BLOOM_BREATH_MAX_PEAK   1.00f
#define BLOOM_BREATH_FLOOR_MIN  0.15f       // floor at normal brightness
#define BLOOM_BREATH_FLOOR_MARGIN 0.75f     // <1.0 lets trough clip (some LEDs go dark)
#define BLOOM_B_CHANNEL_MAX     110.0f      // dominant channel used to derive floor

// Floor rises as CAP drops so the dominant B channel stays near the clip
// threshold during the breath cycle. Margin <1.0 means some LEDs dip below
// clip at trough and go fully dark — intentional at low brightness.
static inline float computeBreathFloor(float cap) {
    float gClip = powf(1.0f / (BLOOM_B_CHANNEL_MAX * cap), 1.0f / 2.4f);
    float f = BLOOM_BREATH_FLOOR_MARGIN * gClip;
    return (f > BLOOM_BREATH_FLOOR_MIN) ? f : BLOOM_BREATH_FLOOR_MIN;
}

// Color (blue/cyan palette + subtle purple drift)
#define BLOOM_HUE_A_G   20.0f
#define BLOOM_HUE_A_B  100.0f
#define BLOOM_HUE_B_G   70.0f
#define BLOOM_HUE_B_B  110.0f
#define BLOOM_PURPLE_MAX  60.0f
#define BLOOM_PURPLE_RATE 0.15f

// W channel
#define BLOOM_W_ONSET        0.5f   // glow above this adds white channel
#define BLOOM_W_GAIN         200.0f // scales the W channel into 0..255*BRIGHTNESS_CAP

// ── Fire parameters (ported from festicorn/bulbs/src/bulb_receiver.cpp) ─────
#define FIRE_FLICKER_SCALE  3.0f
#define FIRE_DEADBAND       0.08f
// W-channel blending thresholds for RGBW output
#define FIRE_PURE_W_CEIL    0.04f   // below this fraction, pure W channel only
#define FIRE_PURE_W_BLEND   0.08f   // blend zone width above PURE_W_CEIL

// Color anchors (RGB 0–255 scale, pre-gamma)
static const float FIRE_AMBER_R = 255.0f, FIRE_AMBER_G = 140.0f, FIRE_AMBER_B = 30.0f;
static const float FIRE_RED_R   = 200.0f, FIRE_RED_G   =  20.0f, FIRE_RED_B   =  0.0f;

// Fire render state
static float fireTime              = 0.0f;
static float fireBaseBrightness    = 0.5f;  // autonomous: held at 0.5 (mid-flame)
static float fireFlickerIntensity  = 0.0f;
static float fireColorEnergy       = 1.0f;  // fixed at 1.0 → full red palette

static void resetFireState() {
    fireTime             = 0.0f;
    fireBaseBrightness   = 0.5f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy      = 1.0f;
}

// ── Leaf Wind parameters (ported from audio-reactive/effects/leaf_wind.py) ───
// Topology collapsed to a single linear branch (LEDs 0..TOTAL_LEDS-1).
// Wind always blows left→right (positive direction). Onset detection replaced
// by a timer-based Poisson-style spawn rate modulated by a slow sine so the
// arrival of leaves feels uneven and natural.
#define LW_MAX_LEAVES       12
#define LW_WIND_SPEED       9.0f
#define LW_TURBULENCE       0.5f
#define LW_DAMPING          0.92f
#define LW_BOOST_SPEED      25.0f
#define LW_BOOST_TC         2.0f
#define LW_LEAF_SIGMA       1.5f
#define LW_FADE_IN_TIME     0.3f
#define LW_FADE_OUT_LEDS    10.0f
#define LW_MAX_BRIGHTNESS   0.85f
#define LW_STALL_RADIUS     3.0f
#define LW_STALL_TIMEOUT    3.0f
// Autonomous spawn: base rate in leaves/second, scaled by speed slider
#define LW_SPAWN_RATE_BASE  0.8f   // average leaves/sec at speed=1.0
// RGBW routing thresholds — lower ceiling than fire keeps green identity further down
#define LW_PURE_W_CEIL  0.02f
#define LW_PURE_W_BLEND 0.06f

// Foliage palette: {R, G, B} in 0–255 float
static const float LW_PALETTE[][3] = {
    { 40, 220,  30},   // bright spring green
    { 20, 180,  40},   // mid leaf green
    { 10, 120,  20},   // deep forest green
    {120, 210,  20},   // yellow-green highlight
    { 90, 140,  30},   // olive
    {160, 240,  80},   // pale lime
    { 10,  80,  15},   // dark moss
};
static const uint8_t LW_PALETTE_SIZE = 7;

struct LWLeaf {
    float pos;
    float vel;
    float boost;
    float colR, colG, colB;
    float brightness;
    float age;
    float refPos;
    float refTime;
    bool  active;
};

static LWLeaf lwLeaves[LW_MAX_LEAVES];
static float  lwTime         = 0.0f;
static float  lwSpawnTimer   = 0.0f;  // counts down to next spawn attempt

static void resetLeafWindState() {
    lwTime       = 0.0f;
    lwSpawnTimer = 0.0f;
    for (uint8_t i = 0; i < LW_MAX_LEAVES; i++) lwLeaves[i].active = false;
}

// ── Web server + runtime state ───────────────────────────────────
enum Effect : uint8_t { EFFECT_BLOOM = 0, EFFECT_FIRE = 1, EFFECT_LEAF_WIND = 2 };
static const char *EFFECT_NAMES[] = { "bloom", "fire", "leaf_wind" };
static const uint8_t NUM_EFFECTS = 3;

struct WebState {
    int      brightness = 10;      // 0–100; slider=10 → C0=0.10,C1=0.06 (default); slider=100 → C0=1.0,C1=0.60
    float    speed      = 1.0f;    // 0.25–4.0 multiplier on breath/flicker rate
    Effect   effect     = EFFECT_BLOOM;
    uint32_t fps        = 0;
};
static WebState state;
static AsyncWebServer server(80);
// Cache index.html in RAM at boot to avoid LittleFS blocking the render loop
static String cachedIndexHtml;

// ── Frame capture (UDP, port 4555) ───────────────────────────────
// POST /capture?effect=NAME&seconds=N: stream raw post-delta-sigma RGBW
// frames to the requester's IP on UDP port 4555 for N seconds.
// Packet layout: 4-byte LE frame index + (TOTAL_LEDS * 4) bytes RGBW.
static WiFiUDP        captureUdp;
static bool           captureActive  = false;
static uint32_t       captureEndMs   = 0;
static uint32_t       captureFrameIdx = 0;
static IPAddress      captureClientIP;
static const uint16_t CAPTURE_PORT   = 4555;
static Effect         capturePrevEffect = EFFECT_BLOOM;
// Capture packet buffer: 4 (index) + LEDS_TOTAL*4 (RGBW)
static uint8_t        captureBuf[4 + LEDS_TOTAL * 4];

static const char FALLBACK_PAGE[] PROGMEM =
    "<!DOCTYPE html><html><body><h2>No UI uploaded.</h2>"
    "<p>Flash LittleFS image with: pio run -e bloom_standalone -t uploadfs</p>"
    "</body></html>";

static void setupWebServer() {
    if (!LittleFS.begin(true)) {
        Serial.println("[FS] LittleFS mount failed");
    } else {
        File f = LittleFS.open("/index.html", "r");
        if (f) {
            cachedIndexHtml = f.readString();
            f.close();
            Serial.printf("[FS] Cached index.html: %u bytes\n", cachedIndexHtml.length());
        }
    }

    DefaultHeaders::Instance().addHeader("Access-Control-Allow-Origin", "*");

    server.on("/", HTTP_GET, [](AsyncWebServerRequest *req) {
        if (cachedIndexHtml.length() > 0)
            req->send(200, "text/html", cachedIndexHtml);
        else
            req->send_P(200, "text/html", FALLBACK_PAGE);
    });

    server.on("/status", HTTP_GET, [](AsyncWebServerRequest *req) {
        String json = "{";
        json += "\"brightness\":" + String(state.brightness) + ",";
        json += "\"speed\":"      + String(state.speed, 2)   + ",";
        json += "\"effect\":\""   + String(EFFECT_NAMES[state.effect]) + "\",";
        json += "\"fps\":"        + String(state.fps)         + ",";
        json += "\"rssi\":"       + String(WiFi.isConnected() ? WiFi.RSSI() : 0) + ",";
        json += "\"ip\":\""       + WiFi.localIP().toString() + "\",";
        json += "\"uptime_s\":"   + String(millis() / 1000)   + "}";
        req->send(200, "application/json", json);
    });

    server.on("/effects", HTTP_GET, [](AsyncWebServerRequest *req) {
        String json = "[";
        for (uint8_t i = 0; i < NUM_EFFECTS; i++) {
            if (i) json += ",";
            json += "\"" + String(EFFECT_NAMES[i]) + "\"";
        }
        json += "]";
        req->send(200, "application/json", json);
    });

    server.on("/effect", HTTP_POST, [](AsyncWebServerRequest *req) {
        if (req->hasParam("effect", true)) {
            String e = req->getParam("effect", true)->value();
            for (uint8_t i = 0; i < NUM_EFFECTS; i++) {
                if (e == EFFECT_NAMES[i]) {
                    if (state.effect != (Effect)i) {
                        state.effect = (Effect)i;
                        if (state.effect == EFFECT_FIRE)      resetFireState();
                        if (state.effect == EFFECT_LEAF_WIND) resetLeafWindState();
                    }
                    break;
                }
            }
        }
        req->send(200, "text/plain", "OK");
    });

    server.on("/brightness", HTTP_POST, [](AsyncWebServerRequest *req) {
        if (req->hasParam("brightness", true)) {
            int b = req->getParam("brightness", true)->value().toInt();
            if (b >= 0 && b <= 100) state.brightness = b;
        }
        req->send(200, "text/plain", "OK");
    });

    server.on("/speed", HTTP_POST, [](AsyncWebServerRequest *req) {
        if (req->hasParam("speed", true)) {
            float s = req->getParam("speed", true)->value().toFloat();
            if (s >= 0.25f && s <= 4.0f) state.speed = s;
        }
        req->send(200, "text/plain", "OK");
    });

    // POST /capture?effect=NAME&seconds=N
    // Switches to the requested effect and streams post-delta-sigma RGBW
    // frames as UDP packets to the requester for N seconds, then restores.
    server.on("/capture", HTTP_POST, [](AsyncWebServerRequest *req) {
        int seconds = 5;
        if (req->hasParam("seconds", true)) {
            seconds = req->getParam("seconds", true)->value().toInt();
            if (seconds < 1) seconds = 1;
            if (seconds > 60) seconds = 60;
        }
        if (req->hasParam("effect", true)) {
            String e = req->getParam("effect", true)->value();
            for (uint8_t i = 0; i < NUM_EFFECTS; i++) {
                if (e == EFFECT_NAMES[i]) {
                    capturePrevEffect = state.effect;
                    state.effect = (Effect)i;
                    if (state.effect == EFFECT_FIRE)      resetFireState();
                    if (state.effect == EFFECT_LEAF_WIND) resetLeafWindState();
                    break;
                }
            }
        }
        captureClientIP  = req->client()->remoteIP();
        captureEndMs     = millis() + (uint32_t)seconds * 1000;
        captureFrameIdx  = 0;
        captureActive    = true;
        captureUdp.begin(CAPTURE_PORT);
        Serial.printf("[capture] streaming %ds to %s:%u\n",
                      seconds, captureClientIP.toString().c_str(), CAPTURE_PORT);
        req->send(200, "text/plain", "OK");
    });

    // Serve snapshot PNGs from LittleFS — explicit handler avoids serveStatic
    // panicking on a missing /snapshots/ directory (LittleFS drops empty dirs).
    server.on("/snapshots/*", HTTP_GET, [](AsyncWebServerRequest *req) {
        // req->url() is e.g. "/snapshots/bloom.png" — use directly as LittleFS path
        String path = req->url();
        if (!LittleFS.exists(path)) { req->send(404); return; }
        req->send(LittleFS, path, "image/png");
    });

    server.begin();
    Serial.println("[Web] Async server started on port 80");
}

// ── LED driver: single SK6812 RGBW strip via RMT ────────────────
// Use Sk6812 timing (not Ws2812x) — SK6812 has its own NRZ timing spec.
// NeoGrbwFeature stores pixels as G,R,B,W.
static NeoPixelBus<NeoGrbwFeature, NeoEsp32Rmt0Sk6812Method> strip(TOTAL_LEDS, LED_PIN);

// ── Per-LED state (RGBW) ────────────────────────────────────────
static uint16_t dsR[TOTAL_LEDS], dsG[TOTAL_LEDS], dsB[TOTAL_LEDS], dsW[TOTAL_LEDS];
static float bloomBreathPhase[TOTAL_LEDS];
static float bloomBreathRate[TOTAL_LEDS];
static float bloomBreathPeak[TOTAL_LEDS];
static float bloomHueT[TOTAL_LEDS];

// ── PRNG ────────────────────────────────────────────────────────
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
static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// SK6812 RGBW routing (SK6812_OUTPUT_PIPELINE.md §2) — for fade-through-white effects only.
// NOT suitable for chroma-driven effects (fire, leaf_wind): threshold is on post-cap
// output so the brightness slider drags everything into W. Use achromatic extraction instead.
static inline void sk6812Route(float oR, float oG, float oB,
                               float pure_w_ceil, float pure_w_blend,
                               float &fR, float &fG, float &fB, float &fW) {
    float maxCh = fmaxf(oR, fmaxf(oG, oB));
    float bFrac = maxCh / 255.0f;
    float rgbBlend = clampf((bFrac - pure_w_ceil) / pure_w_blend, 0.0f, 1.0f);
    float avgRGB = (oR + oG + oB) / 3.0f;
    fR = oR * rgbBlend;
    fG = oG * rgbBlend;
    fB = oB * rgbBlend;
    fW = avgRGB * (1.0f - rgbBlend);
}

// ── Render one frame ────────────────────────────────────────────
static void renderQuietBloom(float dt, uint32_t now) {
    const float purpleTimeBase = (float)now * (0.001f * BLOOM_PURPLE_RATE);

    for (uint8_t seg = 0; seg < COLONIES_PER_STRIP; seg++) {
        uint16_t segStart = seg * LEDS_PER_COLONY;
        uint16_t segEnd   = segStart + LEDS_PER_COLONY;
        // slider=10 → C0=0.10, C1=0.06; slider=100 → C0=1.0, C1=0.60
        float colonyCap = (seg == 0) ? (state.brightness / 100.0f)
                                     : (state.brightness * 0.006f);
        if (colonyCap < 0.001f) colonyCap = 0.001f;
        float colonyFloor = computeBreathFloor(colonyCap);

        for (uint16_t i = segStart; i < segEnd; i++) {
            bloomBreathPhase[i] += dt * bloomBreathRate[i] * state.speed;
            if (bloomBreathPhase[i] >= 1.0f) bloomBreathPhase[i] -= 1.0f;

            float h = bloomHueT[i];
            float breath = (fastSinPhase(bloomBreathPhase[i]) * 0.5f + 0.5f);
            // Guard against floor > peak inverting the breath direction.
            float effFloor = fminf(colonyFloor, bloomBreathPeak[i] - 0.05f);
            float g = effFloor + breath * (bloomBreathPeak[i] - effFloor);

            float colG = lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h);
            float colB = lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h);

            // Subtle purple drift on R (slow sin, per-LED phase offset)
            float purplePhase = purpleTimeBase + h * 4.0f;
            float purpleMix = (fastSin(purplePhase) + 1.0f) * 0.5f;
            float colR = purpleMix * BLOOM_PURPLE_MAX;

            float linBright = fastGamma24(g) * colonyCap;
            float oR = colR * linBright;
            float oG = colG * linBright;
            float oB = colB * linBright;

            // W channel: on above glow threshold.
            float wFrac = clampf((g - BLOOM_W_ONSET) / (1.0f - BLOOM_W_ONSET),
                                 0.0f, 1.0f);
            float oW = wFrac * linBright * BLOOM_W_GAIN;

            uint16_t tR16 = (uint16_t)clampf(oR * 256.0f, 0, 65535);
            uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
            uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
            uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

            if (tR16 < BLOOM_NOISE_GATE) tR16 = 0;
            if (tG16 < BLOOM_NOISE_GATE) tG16 = 0;
            if (tB16 < BLOOM_NOISE_GATE) tB16 = 0;
            if (tW16 < BLOOM_NOISE_GATE) tW16 = 0;

            uint8_t r8 = deltaSigma(dsR[i], tR16);
            uint8_t g8 = deltaSigma(dsG[i], tG16);
            uint8_t b8 = deltaSigma(dsB[i], tB16);
            uint8_t w8 = deltaSigma(dsW[i], tW16);
            strip.SetPixelColor(i, RgbwColor(r8, g8, b8, w8));
        }
    }
}

// ── Autonomous fire render (ported from festicorn/bulbs/src/bulb_receiver.cpp) ──
// Audio inputs (energy/onset) replaced with fixed autonomous values:
//   fireBaseBrightness held at 0.5 (mid-flame); speed slider drives flicker tempo.
static void renderFire(float dt) {
    fireTime += dt * state.speed;
    float t = fireTime;

    float base = fireBaseBrightness;
    if (base < FIRE_DEADBAND) base = 0.0f;

    // Color: ce=1.0 → deep red (no white blend, full red shift)
    float ce = fireColorEnergy;  // fixed at 1.0 on reset
    float baseColR, baseColG, baseColB;
    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;
    if (ce < WHITE_BLEND_THRESHOLD) {
        float tw = 1.0f - (ce / WHITE_BLEND_THRESHOLD);
        baseColR = FIRE_AMBER_R * (1.0f - tw) + 180.0f * tw;
        baseColG = FIRE_AMBER_G * (1.0f - tw) + 170.0f * tw;
        baseColB = FIRE_AMBER_B * (1.0f - tw) + 160.0f * tw;
    } else {
        float tr = (ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD);
        if (tr > 1.0f) tr = 1.0f;
        baseColR = FIRE_AMBER_R * (1.0f - tr) + FIRE_RED_R * tr;
        baseColG = FIRE_AMBER_G * (1.0f - tr) + FIRE_RED_G * tr;
        baseColB = FIRE_AMBER_B * (1.0f - tr) + FIRE_RED_B * tr;
    }

    // brightness cap from slider (fire uses C0 cap for whole strip)
    float cap = state.brightness / 100.0f;
    if (cap < 0.001f) cap = 0.001f;

    float s = FIRE_FLICKER_SCALE;

    for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
        float fi = (float)i;

        // Per-LED noise: two incommensurate sines (same formula as source)
        float noise = fastSin(fi * 7.3f + t * 2.5f) *
                      fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

        float noiseAmp = fmaxf(0.15f * s, 0.10f * s / fmaxf(base, 0.1f));
        float bright = base * (1.0f + noiseAmp * (noise - 0.5f));
        bright = clampf(bright, 0.0f, 1.0f);

        float linBright = fastGamma24(bright) * cap;
        float oR = baseColR * linBright;
        float oG = baseColG * linBright;
        float oB = baseColB * linBright;

        float wFire = fminf(oR, fminf(oG, oB));
        float fR = oR - wFire;
        float fG = oG - wFire;
        float fB = oB - wFire;
        float fW = wFire;
        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);

        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        uint8_t r8 = deltaSigma(dsR[i], tR16);
        uint8_t g8 = deltaSigma(dsG[i], tG16);
        uint8_t b8 = deltaSigma(dsB[i], tB16);
        uint8_t w8 = deltaSigma(dsW[i], tW16);
        strip.SetPixelColor(i, RgbwColor(r8, g8, b8, w8));
    }
}

// ── Leaf Wind render (ported from audio-reactive/effects/leaf_wind.py) ───────
// Linear strip: one branch (0..TOTAL_LEDS-1), wind blows in +pos direction.
// Onset detection replaced by a timer-based spawn modulated by a slow sine so
// arrivals feel uneven. Speed slider scales spawn rate and wind speed.
static void renderLeafWind(float dt) {
    lwTime += dt;
    float t = lwTime;

    // --- Autonomous spawn ---
    // Spawn rate (leaves/sec) scaled by speed slider; slow sine adds unevenness
    float spawnRate = LW_SPAWN_RATE_BASE * state.speed
                      * (0.7f + 0.3f * (fastSin(t * 0.4f) * 0.5f + 0.5f));
    lwSpawnTimer -= dt;
    if (lwSpawnTimer <= 0.0f) {
        // Find a free slot
        int8_t slot = -1;
        for (uint8_t i = 0; i < LW_MAX_LEAVES; i++) {
            if (!lwLeaves[i].active) { slot = i; break; }
        }
        if (slot >= 0) {
            uint8_t pal = (uint8_t)(xorshift32() % LW_PALETTE_SIZE);
            LWLeaf &lf = lwLeaves[slot];
            lf.pos       = -(float)(xorshift32() % 3);  // spawn just off left edge
            lf.vel       = 0.0f;
            lf.boost     = LW_BOOST_SPEED * (0.5f + 0.5f * randFloat());
            lf.colR      = LW_PALETTE[pal][0];
            lf.colG      = LW_PALETTE[pal][1];
            lf.colB      = LW_PALETTE[pal][2];
            lf.brightness = 0.0f;
            lf.age       = 0.0f;
            lf.refPos    = lf.pos;
            lf.refTime   = 0.0f;
            lf.active    = true;
        }
        // Next interval: Poisson-ish (mean = 1/rate, jitter ±30%)
        float mean = (spawnRate > 0.01f) ? 1.0f / spawnRate : 1.0f;
        lwSpawnTimer = mean * (0.7f + 0.6f * randFloat());
    }

    // --- Update leaves ---
    float boostDecay = expf(-dt / LW_BOOST_TC);
    float sigma_sq2  = 2.0f * LW_LEAF_SIGMA * LW_LEAF_SIGMA;

    for (uint8_t i = 0; i < LW_MAX_LEAVES; i++) {
        LWLeaf &lf = lwLeaves[i];
        if (!lf.active) continue;

        // 1D noise via two incommensurate sines (matches Python _noise1d)
        float noiseVal = fastSin(lf.pos * 0.4f + t * 0.3f + (float)i * 7.3f)
                       * fastSin(lf.pos * 0.17f - t * 0.19f + (float)i * 3.1f) * 0.5f
                       + fastSin(lf.pos * 0.09f + t * 0.13f + (float)i * 1.7f) * 0.33f;
        float speedMult = fmaxf(1.0f + noiseVal * LW_TURBULENCE, 0.05f);

        float force = LW_WIND_SPEED * state.speed * speedMult;
        lf.vel = lf.vel * LW_DAMPING + force * (1.0f - LW_DAMPING);
        lf.boost *= boostDecay;
        lf.pos  += (lf.vel + lf.boost) * dt;
        lf.age  += dt;

        // Stall detection
        float elapsed = lf.age - lf.refTime;
        if (elapsed > LW_STALL_TIMEOUT) {
            if (fabsf(lf.pos - lf.refPos) < LW_STALL_RADIUS) {
                lf.active = false; continue;
            }
            lf.refPos  = lf.pos;
            lf.refTime = lf.age;
        }

        // Fade in
        lf.brightness = (lf.age < LW_FADE_IN_TIME)
                        ? (lf.age / LW_FADE_IN_TIME) : 1.0f;

        // Fade out near exit (right edge)
        float distToExit = (float)(TOTAL_LEDS - 1) - lf.pos;
        if (distToExit < LW_FADE_OUT_LEDS && distToExit >= 0.0f)
            lf.brightness *= distToExit / LW_FADE_OUT_LEDS;

        // Remove when off right edge + margin
        if (lf.pos > (float)TOTAL_LEDS - 1 + LW_LEAF_SIGMA * 3.0f)
            lf.active = false;
    }

    // --- Render: Gaussian splat per leaf, additive, then normalize color ---
    float cap = state.brightness / 100.0f;
    if (cap < 0.001f) cap = 0.001f;

    for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
        float glow = 0.0f, oR = 0.0f, oG = 0.0f, oB = 0.0f;

        for (uint8_t j = 0; j < LW_MAX_LEAVES; j++) {
            const LWLeaf &lf = lwLeaves[j];
            if (!lf.active) continue;
            float d = (float)i - lf.pos;
            float w = expf(-(d * d) / sigma_sq2) * lf.brightness;
            glow += w;
            oR   += w * lf.colR;
            oG   += w * lf.colG;
            oB   += w * lf.colB;
        }

        float fR = 0.0f, fG = 0.0f, fB = 0.0f;
        if (glow > 1e-6f) {
            float bright = clampf(glow, 0.0f, 1.0f) * LW_MAX_BRIGHTNESS;
            float linBright = fastGamma24(bright) * cap;
            fR = (oR / glow) * linBright;
            fG = (oG / glow) * linBright;
            fB = (oB / glow) * linBright;
        }

        float wLeaf = fminf(fR, fminf(fG, fB));
        float lfR = fR - wLeaf;
        float lfG = fG - wLeaf;
        float lfB = fB - wLeaf;
        float lfW = wLeaf;
        uint16_t tR16 = (uint16_t)clampf(lfR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(lfG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(lfB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(lfW * 256.0f, 0, 65535);

        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        strip.SetPixelColor(i, RgbwColor(
            deltaSigma(dsR[i], tR16),
            deltaSigma(dsG[i], tG16),
            deltaSigma(dsB[i], tB16),
            deltaSigma(dsW[i], tW16)));
    }
}

// ── WiFi + OTA setup ────────────────────────────────────────────
static void onWiFiDisconnect(WiFiEvent_t, WiFiEventInfo_t) {
    Serial.println("[wifi] disconnected — reconnecting");
    WiFi.reconnect();
}

static void setupWiFiAndOTA() {
    WiFi.persistent(false);
    WiFi.setAutoReconnect(true);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.setHostname(OTA_HOSTNAME);
    WiFi.onEvent(onWiFiDisconnect, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.printf("Connecting to %s", WIFI_SSID);
    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
        delay(250);
        Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("WiFi ok — IP=%s RSSI=%d\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
        ArduinoOTA.setHostname(OTA_HOSTNAME);
        ArduinoOTA.onStart([]() { Serial.println("OTA start"); });
        ArduinoOTA.onEnd([]()   { Serial.println("\nOTA end"); });
        ArduinoOTA.onProgress([](unsigned int p, unsigned int t) {
            Serial.printf("OTA %u%%\r", (p * 100) / t);
        });
        ArduinoOTA.onError([](ota_error_t e) {
            Serial.printf("OTA err %u\n", e);
        });
        ArduinoOTA.begin();
        Serial.printf("OTA ready @ %s.local\n", OTA_HOSTNAME);
    } else {
        Serial.println("WiFi failed — running without OTA");
    }
}

// mDNS re-announce intentionally omitted — MDNS.end()/.begin() churn broke
// ArduinoOTA's UDP:3232 listener. Upload by raw IP instead of .local.

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.printf("rgbw-bulbs-standalone — 1 strip × %u LEDs on GPIO %u (2 colonies)\n",
                  TOTAL_LEDS, LED_PIN);
    // Reset reason: 1=POWERON 3=SW 4=OWDT 5=DEEPSLEEP 6=SDIO 7=TG0WDT 8=TG1WDT
    // 9=RTCWDT_SYS 14=BROWNOUT 15=SDIO — 14 = PSU dipped under BOD threshold.
    Serial.printf("boot rst=%d cpuMHz=%u\n",
                  (int)esp_reset_reason(), ESP.getCpuFreqMHz());

    strip.Begin();
    strip.ClearTo(RgbwColor(0, 0, 0, 0));
    strip.Show();

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / TOTAL_LEDS);
        dsR[i] = seed;
        dsG[i] = (seed + 64)  & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;

        bloomBreathPhase[i]  = randFloat();
        float period = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bloomBreathRate[i]   = 1.0f / period;
        bloomBreathPeak[i]   = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bloomHueT[i]         = randFloat();
    }

    // Boot flash — brief warm white on the W channel
    strip.ClearTo(RgbwColor(0, 0, 0, 40));
    strip.Show();
    delay(200);
    strip.ClearTo(RgbwColor(0, 0, 0, 0));
    strip.Show();

    setupWiFiAndOTA();
    setupWebServer();
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    ArduinoOTA.handle();

    uint32_t now = millis();

    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : 0.04f;
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    uint32_t t0 = micros();
    switch (state.effect) {
        case EFFECT_FIRE:      renderFire(dt);            break;
        case EFFECT_LEAF_WIND: renderLeafWind(dt);        break;
        default:               renderQuietBloom(dt, now); break;
    }
    uint32_t renderUs = micros() - t0;

    uint32_t s0 = micros();
    strip.Show();
    uint32_t showUs = micros() - s0;

    // Frame capture: send post-delta-sigma RGBW via UDP
    if (captureActive) {
        if (now >= captureEndMs) {
            captureActive = false;
            captureUdp.stop();
            state.effect = capturePrevEffect;
            Serial.printf("[capture] done — %u frames sent\n", captureFrameIdx);
        } else {
            // Pack frame index (LE uint32) + raw RGBW bytes
            captureBuf[0] = captureFrameIdx & 0xFF;
            captureBuf[1] = (captureFrameIdx >> 8)  & 0xFF;
            captureBuf[2] = (captureFrameIdx >> 16) & 0xFF;
            captureBuf[3] = (captureFrameIdx >> 24) & 0xFF;
            for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
                RgbwColor c = strip.GetPixelColor<RgbwColor>(i);
                captureBuf[4 + i * 4 + 0] = c.R;
                captureBuf[4 + i * 4 + 1] = c.G;
                captureBuf[4 + i * 4 + 2] = c.B;
                captureBuf[4 + i * 4 + 3] = c.W;
            }
            captureUdp.beginPacket(captureClientIP, CAPTURE_PORT);
            captureUdp.write(captureBuf, sizeof(captureBuf));
            captureUdp.endPacket();
            captureFrameIdx++;
        }
    }

    // Telemetry (~1Hz)
    static uint32_t frameCount = 0;
    frameCount++;
    static uint64_t renderUsAccum = 0, showUsAccum = 0;
    renderUsAccum += renderUs;
    showUsAccum   += showUs;

    static uint32_t lastTelemetryMs = 0;
    static uint32_t lastTelemetryFrames = 0;
    if (now - lastTelemetryMs >= 1000) {
        uint32_t fps = frameCount - lastTelemetryFrames;
        state.fps = fps;
        uint32_t avgRender = fps ? (uint32_t)(renderUsAccum / fps) : 0;
        uint32_t avgShow   = fps ? (uint32_t)(showUsAccum / fps) : 0;
        lastTelemetryFrames = frameCount;
        lastTelemetryMs = now;
        renderUsAccum = 0;
        showUsAccum = 0;
        Serial.printf("[bloom] %ufps r=%uus s=%uus up=%lus heap=%u rssi=%d rst=%d ip=%s\n",
                      fps, avgRender, avgShow,
                      (unsigned long)(now / 1000),
                      ESP.getFreeHeap(),
                      WiFi.isConnected() ? WiFi.RSSI() : 0,
                      (int)esp_reset_reason(),
                      WiFi.isConnected() ? WiFi.localIP().toString().c_str() : "n/a");
    }
}
