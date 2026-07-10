/*
 * TREE-OF-RECORD — ambient bloom on all 6 strips, driven by a BS-26.
 *
 * ESP32-D0WD-V3, 6 × WS2812B RGB strips on GPIO 4, 15, 17, 5, 18, 19,
 * 100 LEDs each. No senders, no motion — pure ambient breathing bloom
 * (ported from biolum board B), modulated by a BS-26 console:
 *
 *   tens (10-pos knob) → brightness; 0 = fully off
 *   ones (0–11)        → speed (12 steps, log-spaced 0.2–8.0)
 *   decouter (0–11)    → effect select, grouped: 0–2 audio field effects
 *                        (bloom, fire, heat), 3–4 audio particle effects
 *                        (sparkle, creatures), 5 ambient particles (leaf-wind),
 *                        6–9 passive (light-through, nebula, galaxy, rainbow),
 *                        10 bouquet, 11 collision
 *   hundreds (0–4)     → unused (was record-slot select; record/playback
 *                        ripped out 2026-07 — restore from git history)
 *   decmid (0–9)       → hue        36° rotation steps (full wheel)
 *   decinner (0–9)     → saturation 5=original, 0=boosted to full, 9=white
 *   DC switch          → unused (was record/playback trigger)
 *   AC switch          → ignored (lever broke at install, June 2026 — contact
 *                        stuck closed/chattering.
 *                        See AC_SWITCH_FAILURE_AND_DC_FALLBACK.md)
 *
 * BS-26 live → knobs control. No signal → serial PARAMS control.
 * BS-26 state arrives as JSON over ESP-NOW broadcast on channel 1.
 *
 * ── INVARIANTS (hold these when editing any effect) ──────────────
 * 1. Two clocks. Effects receive fxDt (= dt × speed knob × SPEED_SCALE) for
 *    motion, fades, spawning. Anything that REACTS to audio or sensors —
 *    envelopes, onset cooldowns, energy EMAs — uses real gDt. The ones knob
 *    shapes motion, never reaction latency.
 * 2. Gamma goes on a scalar brightness (fastGamma24), which then scales all
 *    three channels uniformly. Never gamma R/G/B individually — it shifts
 *    hue on mixed colors as they dim.
 * 3. New effects take color from oklchVarL (perceptually tuned LUT). The
 *    hand-RGB palettes in fire/nebula/light-through/leaf-wind/creatures are
 *    deliberate per-effect artistic choices, not the default.
 * 4. All output goes through setPixelScaled — that's where the hue/sat
 *    knobs, renderBrightness, and the floor policy live. Bloom is the one
 *    sanctioned exception (rotates its own endpoints via applyPalette).
 * 5. PARAMS[] exposes values tuned live in the field; constants tuned once
 *    stay #defines. Don't expose by reflex.
 * 6. All decay/smoothing is dt-based (festicorn/CLAUDE.md). Legacy
 *    fastDecay(rate, dt×30) sites are dt-correct — the ×30 is a reference
 *    frame rate baked into the rate constant, not an FPS assumption to fix.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <ArduinoJson.h>
#include <math.h>
#include <fast_math.h>
#include <oklch_lut.h>
#include <v1_packet.h>
#include <gyro_packet_v1.h>
#include <audio_packet_v1.h>
#include <audio_stream_packet_v1.h>

// ── Strip layout ─────────────────────────────────────────────────
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 100
#endif

static const uint8_t NUM_STRIPS = 4;   // led-eth-0 variant: 4 strips (tree is 6)

// ── Bloom parameters (runtime-tunable via operator) ─────────────
static float bloomBrightnessCap  = 1.64f;   // master ambient level (post-gamma,
                                            // ×baseColor). Raised 1.0→1.64 to land
                                            // bloom's per-lit-pixel mean in the
                                            // 18–35%-of-255 ambient band.
static float bloomBufferDrain     = 15.0f;
static float bloomFlashDecayRate  = 3.0f;
// Glow trough (pulses, doesn't blink). 0.28 is the contrast-vs-gate limit:
// trough output at knob=0.1 is ~0.85, just above GATE_ON_THRESH 0.7 — lower
// floors blink at low knob.
static float bloomBreathFloor     = 0.28f;
// Audio-energy overdrive: audioEnergy5s drives a per-pixel boost ADDED after the
// brightness knob/cap, so loud audio punches bright pixels THROUGH the cap toward
// saturation (up to bloomAudioCeil ≈ 2–3×) even when the brightness knob is low,
// while below-mean pixels are pushed dark. Knob sets the quiet ambient level.
static float bloomStretchGain     = 2.5f;   // overdrive level at full energy (× cap)
static float bloomAudioCeil       = 3.0f;   // max level; ×baseColor saturates at 255
// Ambient contrast stretch: expand each pixel's glow away from the field mean
// BEFORE gamma, always on (audio stretch above is separate, energy-driven).
// Makes breathing variance visible at low knob, where pure ratio compresses
// into the bottom of the code range. >1 lets troughs cross the noise gate and
// wink fully off — intentional (reads as dormancy).
static float bloomAmbientContrast = 1.6f;

#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.75f
#define BLOOM_BREATH_MAX_PEAK   1.00f

#define BLOOM_FLASH_R  200.0f
#define BLOOM_FLASH_G  120.0f
#define BLOOM_FLASH_B  255.0f

#define BLOOM_HUE_DRIFT_MIN  (1.0f / 45.0f)
#define BLOOM_HUE_DRIFT_MAX  (1.0f / 15.0f)

static float bloomDormancyMin  = 3.0f;
static float bloomDormancyMax  = 7.0f;
static float bloomDormancyFrac = 0.0f;


#define GATE_ON_THRESH  0.7f
#define GATE_OFF_THRESH 1.2f

#define SENSOR_HZ       25.0f

// Tilt mapping (used by ported sparkle effect)
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f

// ── Effect drive parameters ──────────────────────────────────────
// BS-26 live → knobs control. No signal → serial PARAMS control.
static float globalBrightness = 1.0f;   // serial-controlled, 0.0–1.0
static float speedScale       = 1.0f;   // serial-controlled, 0.2–3.0
static float renderBrightness = 1.0f;   // ceiling: max output scale
static float renderFloor      = 0.0f;   // reserved — was proportional floor (removed: killed animation dynamics)
static float gDt = 1.0f / 60.0f;        // current frame dt, set each loop iteration

// ── BS-26 ESP-NOW config ─────────────────────────────────────────
#define FIXED_CHANNEL        1
#define HEARTBEAT_TIMEOUT_MS 6000

// ── BS-26 state ──────────────────────────────────────────────────
struct Bs26State {
    bool     dc;
    bool     ac;
    uint8_t  hundreds;
    uint8_t  tens;
    uint8_t  ones;
    uint16_t decade;     // 0–1199
    uint8_t  decouter;   // 0–11, outer ring
    uint8_t  decmid;     // 0–9, middle ring
    uint8_t  decinner;   // 0–9, inner ring
    uint8_t  ua_dac;     // 0–255
    uint8_t  ua_max;
    uint8_t  v_dac;
    uint8_t  v_max;
    uint32_t seq;
};

static volatile Bs26State bs26 = {};
static volatile uint32_t  bs26LastMs = 0;
static volatile uint32_t  bs26PktCount = 0;
static volatile bool      bs26Updated = false;
// Guards the multi-field bs26 write (onReceive task) against the bulk read
// (loop). Without it a torn read can momentarily mix fields and spike brightness.
static portMUX_TYPE bs26Mux = portMUX_INITIALIZER_UNLOCKED;

// ── Sensor layer (v1 accel / gyro / audio packets over ESP-NOW) ──
// Coexists with BS-26 JSON; disambiguated by packet length in onReceive.
// Raw packet fields are latched in the callback; loop() decodes + normalizes.
#define SENSOR_TIMEOUT_MS 3000

// Companding full-scale (matches v1 packet encoders).
#define V1_AMAG_FS  57000.0f
#define V1_GMAG_FS  57000.0f
#define V1_RMS_FS   200000.0f   // must match sender_v2 RMS_FS

// Production sender locks ±4g accel / ±1000 dps gyro.
#define V1_ACCEL_RANGE_G    4.0f
#define V1_GYRO_RANGE_DPS   1000.0f
#define V1_COUNTS_PER_G     (32768.0f / V1_ACCEL_RANGE_G)
#define V1_DPS_PER_LSB      (V1_GYRO_RANGE_DPS / 128.0f)   // int8 mean → dps

static volatile TelemetryPacketV1 accelPkt = {};
static volatile uint32_t accelLastMs = 0;
static volatile uint32_t accelPktCount = 0;

static volatile GyroPacketV1 gyroPkt = {};
static volatile uint32_t gyroLastMs = 0;
static volatile uint32_t gyroPktCount = 0;

static volatile AudioPacketV1 audioPkt = {};
static volatile uint32_t audioLastMs = 0;
static volatile uint32_t audioPktCount = 0;

// sqrt-companded uint8 → linear value: val = (byte/255)² * FS
static inline float decodeCompand(uint8_t b, float fs) {
    float t = (float)b / 255.0f;
    return t * t * fs;
}

// ── PRNG ─────────────────────────────────────────────────────────
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

// ── HSV ↔ RGB (h 0..360, s/v 0..1, rgb 0..255) ─────────────────
static void hsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r1, g1, b1;
    if      (h < 60)  { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else              { r1 = c; g1 = 0; b1 = x; }
    r = (r1 + m) * 255.0f;
    g = (g1 + m) * 255.0f;
    b = (b1 + m) * 255.0f;
}

static void rgbToHsv(float r, float g, float b, float &h, float &s, float &v) {
    r /= 255.0f; g /= 255.0f; b /= 255.0f;
    float mx = fmaxf(fmaxf(r, g), b);
    float mn = fminf(fminf(r, g), b);
    float d = mx - mn;
    v = mx;
    s = (mx <= 0.0f) ? 0.0f : d / mx;
    if (d <= 0.0f)        h = 0.0f;
    else if (mx == r)     h = 60.0f * fmodf((g - b) / d, 6.0f);
    else if (mx == g)     h = 60.0f * ((b - r) / d + 2.0f);
    else                  h = 60.0f * ((r - g) / d + 4.0f);
    if (h < 0.0f) h += 360.0f;
}

// ── Bloom hue endpoints (runtime, rotated by palette index) ─────
#define BLOOM_HUE_A_R_BASE    0.0f
#define BLOOM_HUE_A_G_BASE  180.0f
#define BLOOM_HUE_A_B_BASE  120.0f
#define BLOOM_HUE_B_R_BASE  140.0f
#define BLOOM_HUE_B_G_BASE   20.0f
#define BLOOM_HUE_B_B_BASE  255.0f

static float bloomHueA_R = BLOOM_HUE_A_R_BASE;
static float bloomHueA_G = BLOOM_HUE_A_G_BASE;
static float bloomHueA_B = BLOOM_HUE_A_B_BASE;
static float bloomHueB_R = BLOOM_HUE_B_R_BASE;
static float bloomHueB_G = BLOOM_HUE_B_G_BASE;
static float bloomHueB_B = BLOOM_HUE_B_B_BASE;

static uint8_t paletteHueIdx = 0;   // 0–9, from decmid
static uint8_t paletteSatIdx = 5;   // 0–9, from decinner; 5 = neutral (originals)

// Saturation knob curve: 5 leaves the effect's original saturation untouched.
// 0–4 boosts toward full saturation, linear in knob position, weighted by
// existing saturation with an ease-out (s·(2−s)) so mid-vivid colors push
// hard while neutral pixels (s≈0, hue meaningless) stay neutral. 6–9
// desaturates on a quadratic falloff so 6–7 stay colorful and only 8–9
// read as washed → white.
static inline float satKnob(float s, uint8_t satIdx) {
    if (satIdx < 5) {
        float t = (5.0f - (float)satIdx) / 5.0f;
        float w = s * (2.0f - s);
        return s + (1.0f - s) * t * w;
    }
    float u = (float)(satIdx - 5) / 4.0f;
    return s * (1.0f - u * u);
}

// Rotate base endpoint hues by hueIdx * 36° and re-saturate by satIdx.
static void applyPalette(uint8_t hueIdx, uint8_t satIdx) {
    float rot = (float)hueIdx * 36.0f;
    float h, s, v, r, g, b;
    rgbToHsv(BLOOM_HUE_A_R_BASE, BLOOM_HUE_A_G_BASE, BLOOM_HUE_A_B_BASE, h, s, v);
    hsvToRgb(fmodf(h + rot, 360.0f), satKnob(s, satIdx), v, r, g, b);
    bloomHueA_R = r; bloomHueA_G = g; bloomHueA_B = b;
    rgbToHsv(BLOOM_HUE_B_R_BASE, BLOOM_HUE_B_G_BASE, BLOOM_HUE_B_B_BASE, h, s, v);
    hsvToRgb(fmodf(h + rot, 360.0f), satKnob(s, satIdx), v, r, g, b);
    bloomHueB_R = r; bloomHueB_G = g; bloomHueB_B = b;
}

// ── ESP-NOW receive (BS-26 JSON state packets) ──────────────────
static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len < 2 || len > 250) return;

    // Fixed-size binary sensor packets take priority over JSON (disambiguated
    // by exact length). Latch raw bytes; loop() decodes and normalizes.
    if (len == (int)sizeof(TelemetryPacketV1)) {
        memcpy((void*)&accelPkt, data, sizeof(TelemetryPacketV1));
        accelLastMs = millis();
        accelPktCount++;
        return;
    }
    if (len == (int)sizeof(GyroPacketV1)) {
        memcpy((void*)&gyroPkt, data, sizeof(GyroPacketV1));
        gyroLastMs = millis();
        gyroPktCount++;
        return;
    }
    if (len == (int)sizeof(AudioPacketV1)) {
        memcpy((void*)&audioPkt, data, sizeof(AudioPacketV1));
        audioLastMs = millis();
        audioPktCount++;
        return;
    }
    if (len == (int)sizeof(AudioStreamPacketV1)) {
        return;   // sender_v3 waveform stream — unused since playback rip-out
    }

    StaticJsonDocument<768> doc;
    DeserializationError err = deserializeJson(doc, (const char*)data, len);
    if (err) return;

    if (!doc.containsKey("s") && !doc.containsKey("seq")) return;

    // Build the full state in a scratch copy, then publish it to bs26 in one
    // critical section so loop never reads a half-updated struct.
    Bs26State next = {};
    // ac/dc arrive as integers ("ac":1) over ESP-NOW, not JSON booleans. The
    // `| false` default-operator gates on is<bool>(), which is FALSE for an
    // integer, so it always returned the default — switches never registered.
    // as<bool>() converts the integer (1→true) instead of type-gating it.
    next.dc       = doc["dc"].as<bool>();
    next.ac       = doc["ac"].as<bool>();
    next.hundreds = doc["h"]  | (doc["hundreds"] | 0);
    next.tens     = doc["t"]  | (doc["tens"] | 0);
    next.ones     = doc["o"]  | (doc["ones"] | 0);
    next.decade   = doc["dec"]| (doc["decade"] | 0);
    next.decouter = doc["do"] | (doc["decouter"] | 0);
    next.decmid   = doc["dm"] | (doc["decmid"] | 0);
    next.decinner = doc["di"] | (doc["decinner"] | 0);
    next.ua_dac   = doc["ua"] | 0;
    next.ua_max   = 0;
    next.v_dac    = doc["v"]  | 0;
    next.v_max    = 0;
    next.seq      = doc["s"]  | (doc["seq"] | 0);

    portENTER_CRITICAL(&bs26Mux);
    memcpy((void*)&bs26, &next, sizeof(bs26));
    portEXIT_CRITICAL(&bs26Mux);

    bs26LastMs = millis();
    bs26PktCount++;
    bs26Updated = true;
}

// ── LED driver: 4 strips via RMT (led-eth-0: GPIO 13/27/32/33) ───
// GRB wire order (NeoGrbFeature) to match led-eth-0's LEDs — same as the
// working wormhole_eth build. (The 6-strip tree uses RGB on its product.)
#ifndef STRIP0_PIN
#define STRIP0_PIN 13
#endif
#ifndef STRIP1_PIN
#define STRIP1_PIN 27
#endif
#ifndef STRIP2_PIN
#define STRIP2_PIN 32
#endif
#ifndef STRIP3_PIN
#define STRIP3_PIN 33
#endif
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP, STRIP0_PIN);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, STRIP1_PIN);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, STRIP2_PIN);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP, STRIP3_PIN);

static inline void setPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
    }
}

static void showAll() {
    // Stagger the Show()s to avoid RMT ISR contention: many channels refilling
    // simultaneously can miss ISR deadlines when a WiFi/ESP-NOW interrupt lands
    // mid-frame, underrunning a channel into corrupt (wrong-color, back-of-
    // strip) glitch frames. The CanShow() wait spreads the refills (and the
    // switching current). See engineering ledger
    // esp32-rmt-simultaneous-show-glitch-frames; fix proven on bulb-fleet.
    strip0.Show(); strip1.Show(); strip2.Show();
    while (!strip0.CanShow() || !strip1.CanShow() || !strip2.CanShow()) {}
    strip3.Show();
}

// ── Stick topology ──────────────────────────────────────────────
// Physical mapping (from topo_test): each "stick" is a contiguous LED range
// on a strip. start/end are inclusive LED indices. Strips 1 and 3 have no
// sticks. Hand-measured against the static block test pattern.
struct Stick { uint8_t strip; uint8_t start; uint8_t end; };
// led-eth-0 variant: no hand-measured physical map — each of the 4 strips is
// one full contiguous run. (The tree's sticks referenced strips 4/5, which
// don't exist here.) Gravity seeds particles across these ranges.
static const Stick STICKS[] = {
    { 0, 0, LEDS_PER_STRIP - 1 },
    { 1, 0, LEDS_PER_STRIP - 1 },
    { 2, 0, LEDS_PER_STRIP - 1 },
    { 3, 0, LEDS_PER_STRIP - 1 },
};
static const uint8_t STICK_COUNT = sizeof(STICKS) / sizeof(STICKS[0]);

// ── Generic output stage (shared by ported effects) ─────────────
// Ported bulb-fleet/biolum effects write 0–255-range float color here;
// this scales by renderBrightness and truncates to 8-bit. No dithering.

// ── Hysteretic floor: anti-flicker policy for sub-LSB channels ───
// Each channel per pixel has a timer (seconds, float). Positive = time spent
// below threshold (trending toward dormant). Negative = time spent above
// (trending toward active). State transitions:
//   ACTIVE  → DORMANT: timer reaches +FLOOR_DEACTIVATE_S (channel goes dark)
//   DORMANT → ACTIVE:  timer reaches -FLOOR_REACTIVATE_S (channel re-enables)
// While active and sub-threshold: output floored at 1/255 (steady, no flicker).
// While dormant: output true zero regardless of target.
#define FLOOR_DEACTIVATE_S  1.0f
#define FLOOR_REACTIVATE_S  0.5f
#define FLOOR_THRESHOLD     256    // target16 below this = sub-LSB danger zone

static float fxFloorTimer[NUM_STRIPS][LEDS_PER_STRIP][1];

static void resetFxDither() {
    memset(fxFloorTimer, 0, sizeof(fxFloorTimer));
}

static inline uint16_t applyFloorPolicy(uint16_t target16, float &timer, float dt) {
    if (target16 < FLOOR_THRESHOLD) {
        timer += dt;
        if (timer >= FLOOR_DEACTIVATE_S) {
            timer = FLOOR_DEACTIVATE_S;
            return 0;  // dormant: true zero
        }
        return FLOOR_THRESHOLD;  // active but sub-LSB: floor at 1/255
    } else {
        timer -= dt;
        if (timer <= -FLOOR_REACTIVATE_S) {
            timer = 0.0f;  // fully reactivated
            return target16;
        }
        if (timer > 0.0f) {
            return 0;  // was dormant, waiting to reactivate
        }
        return target16;  // active, above threshold
    }
}

// Global palette transform: rotate hue by paletteHueIdx×36° and re-saturate
// via satKnob() (5 = original, <5 boost, >5 desaturate), matching
// applyPalette()'s rule. This is the shared path for every effect that routes
// through setPixelScaled, so the decmid/decinner knobs affect all of them.
// Bloom does NOT come through here (it uses setPixel directly and rotates its
// own base colors), so there is no double-rotation. No-op fast path at the
// neutral knob position.
static inline void applyGlobalPalette(float &r, float &g, float &b) {
    if (paletteHueIdx == 0 && paletteSatIdx == 5) return;
    float h, sv, v;
    rgbToHsv(r, g, b, h, sv, v);
    h = fmodf(h + (float)paletteHueIdx * 36.0f, 360.0f);
    sv = satKnob(sv, paletteSatIdx);
    hsvToRgb(h, sv, v, r, g, b);
}

// fr/fg/fb in 0–255 linear range (already gamma-shaped by the effect).
static inline void setPixelScaled(uint8_t s, uint16_t i, float fr, float fg, float fb) {
    applyGlobalPalette(fr, fg, fb);
    if (fr > 0.0f || fg > 0.0f || fb > 0.0f) {
        fr *= renderBrightness;
        fg *= renderBrightness;
        fb *= renderBrightness;
    }
    uint16_t t16R = (uint16_t)fminf(fr * 256.0f, 65535.0f);
    uint16_t t16G = (uint16_t)fminf(fg * 256.0f, 65535.0f);
    uint16_t t16B = (uint16_t)fminf(fb * 256.0f, 65535.0f);
    uint16_t maxT16 = t16R > t16G ? (t16R > t16B ? t16R : t16B)
                                   : (t16G > t16B ? t16G : t16B);
    uint16_t floorResult = applyFloorPolicy(maxT16, fxFloorTimer[s][i][0], gDt);
    if (floorResult == 0 && maxT16 < FLOOR_THRESHOLD) {
        t16R = t16G = t16B = 0;
    }
    uint8_t r8 = t16R >> 8;
    uint8_t g8 = t16G >> 8;
    uint8_t b8 = t16B >> 8;
    setPixel(s, i, r8, g8, b8);
}

// Per-strip OKLCH hue offsets (~60° apart on 256-step wheel) so the 6
// strips span a rainbow chord. From bulb-fleet.
static const uint8_t STRIP_HUE_OFFSET[NUM_STRIPS] = { 0, 64, 128, 192 };

// ── Effect selection (BS-26 decouter ring, 0–11) ────────────────
// Knob order groups by what drives the effect: audio-reactive first (0–4,
// quietest response to busiest), then ambient effects that ignore the
// sensors entirely (5–8, organic → cosmic → pure color).
// Positions ≥ FX_COUNT clamp to the last effect.
enum EffectId : uint8_t {
    // audio-reactive: field effects
    FX_AMBIENT_BLOOM = 0,   // slow energy swell (audioEnergy5s contrast)
    FX_FIRE,                // energy = flame height, onsets = flicker
    FX_HEAT_DIFFUSION,      // energy injects heat at rotating point; PDE spreads it
    // audio-reactive: particle effects
    FX_SPARKLE_SYLLABLE,    // discrete onsets spawn sparks (+ tilt hue)
    FX_CREATURES,           // speech-energy excitement (was accel shake)
    // ambient: particle effects
    FX_LEAF_WIND,
    // ambient: passive
    FX_LIGHT_THROUGH,
    FX_NEBULA,
    FX_GALAXY,              // dark-purple slow breath + drifting desaturated galaxies
    FX_RAINBOW,
    FX_BOUQUET,
    FX_COLLISION,           // crawlers annihilate into ignition burst
    FX_COUNT
};

static uint8_t currentEffect = FX_AMBIENT_BLOOM;
static uint8_t prevEffect    = 0xFF;   // forces reset on first frame

// ── Shared per-effect buffers (DRAM is tight) ────────────────────
// Effects are mutually exclusive and fully re-initialized on switch
// (resetEffect), so large per-pixel state can share one allocation.
// Add a union member per effect that needs a field; never read across
// effects, and make sure the effect's reset writes every cell it reads.
static union {
    float neb[NUM_STRIPS][LEDS_PER_STRIP];   // nebula: orb trail decay
    float heat[NUM_STRIPS][LEDS_PER_STRIP];  // heat diffusion: temperature
} fxField;

// ── Audio/motion feature globals ────────────────────────────────
// The bulb-fleet effects (gravity, sparkle, fire, quiet bloom) consume
// audio/motion features. processSensors() drives these every frame from the
// v1 accel/gyro/audio packets. On sensor timeout they fall back to rest
// (energy/onset = 0, no tilt, accel flat) so each effect's idle/ambient
// branch renders unchanged.
struct SensorStub {
    int16_t ax, ay, az;   // accel counts (16384 = 1g); flat = (0,0,16384)
    int16_t gx, gy, gz;
};
static SensorStub sensorStub = { 0, 0, 16384, 0, 0, 0 };
static float fxEnergy   = 0.0f;   // 0–1 audio/motion energy
static float fxOnset    = 0.0f;   // 0–1 transient onset
static float fxTiltBlend = 0.0f;  // 0–1 tilt engagement
static float fxAngleDeg = 0.0f;   // tilt angle for hue mapping

// "Energy in the system": leaky integral of fxEnergy over ~tau seconds. Builds
// while audio is present, bleeds off over ~tau when it stops. Drives the bloom
// contrast stretch. energyForce >= 0 overrides it for live tuning without audio.
static float audioEnergy5s = 0.0f;
static float energy5sTau   = 3.0f;   // integration/decay window (s)
static float energyForce   = -1.0f;  // -1 = use real audio; 0–1 = forced override

// ── Sensor → feature processing ──────────────────────────────────
// Decodes the v1 packets and drives the effect globals. Audio uses an
// adaptive floor + log scaling (ported from bulb-fleet computeEnergy/Onset,
// retuned for the decoded RMS scale). On timeout each layer falls back to its
// at-rest stub so the effects render their idle/ambient branch unchanged.
// Hard energy floor: below this, mean RMS reads as ambient/no-intent and
// fxEnergy is zeroed. Measured @ FS=200000 (this mic): silent room ~2k,
// room music ~16-37k, distant speech ~13-37k, direct speech peaks 92-133k.
// 20k sits above music/distant talk so only deliberate close speech drives
// energy. The adaptive floor floats above this when ambient is louder.
// Serial-tunable (FLOOR_MIN): the 20k default was calibrated for close
// speech in a quiet room; field analysis of rec_0 (install day) showed
// venue-distance speech peaks only 33-51k at the mic — and ~0.4x that
// once wind rumble is excluded — so the venue floor likely wants to sit
// lower once a windscreen is on the mic.
static float snsRmsFloorMin = 20000.0f;
#define SNS_RMS_FLOOR_MIN  snsRmsFloorMin
// Ceiling seed + leak floor. The ceiling used to boot at the 20k floor min —
// below the 28k effective floor — which clamped dbRange to its 1 dB minimum
// and mapped the FIRST sound after boot/playback-start to fxEnergy = 1.0
// regardless of loudness (gravity slammed outward at first speech). Seed at
// the measured direct-close-speech scale (peaks 92-133k, max 141k @
// FS=200000) so the first utterance lands ~0.7-1.0 honestly; louder peaks
// still raise the ceiling above this.
#define SNS_RMS_CEIL_MIN   130000.0f
// Onset rise floor — minimum absolute flux (fast − slow envelope) to count
// as an attack. Set just above silent-room mic jitter (~2.6k max) so noise
// can't manufacture phantom onsets. The detector ALSO requires the attack
// envelope to clear the same adaptive presence floor that gates fxEnergy —
// field finding (install day 2026-06-11): wind gusts are sharp broadband
// rises over a quiet bed, and flux alone fired sparkle at max strength
// (rise/baseline ≈ 1 over quiet) while staying entirely below the 20k
// presence floor. Direct claps/speech spike well above the floor, so
// clap-to-interact survives the shared gate.
#define SNS_ONSET_FLOOR    6000.0f
#define SNS_FLOOR_HEADROOM 1.4f
#define SNS_FLOOR_LEAK     0.005f
#define SNS_FLOOR_SNAP_EPS 0.05f
#define SNS_FLOOR_SOFT_SIG 0.6f

static float snsAdaptiveFloor = 0.0f;
static float snsRmsCeiling    = SNS_RMS_CEIL_MIN;
static int   snsBelowFloorCnt = 0;

// Energy-flux onset detector state + feel knobs (tunable live via !P).
// Onset = rmsMean rising sharply above its recent baseline — a discrete event
// at each attack, unlike the old self-normalizing max/mean ratio that sat near
// 1.0 through all continuous sound and fired on the noise floor.
static float onsetFast    = 0.0f;   // ~30 ms energy envelope
static float onsetSlow    = 0.0f;   // ~300 ms baseline
static float onsetLock    = 0.0f;   // refractory countdown (s)
static float onsetRefrac  = 0.120f; // refractory lockout: max ~1/this fires/s
static float onsetRiseFrac= 0.5f;   // fire when (fast-slow) > riseFrac*slow (floored at SNS_ONSET_FLOOR)

static float snsUpdateFloor(float rms, float dt) {
    if (snsAdaptiveFloor < 1.0f) {
        snsAdaptiveFloor = fmaxf(rms, SNS_RMS_FLOOR_MIN);
        return snsAdaptiveFloor;
    }
    if (rms < snsAdaptiveFloor * (1.0f + SNS_FLOOR_SNAP_EPS)) {
        if (++snsBelowFloorCnt >= 3) {
            float target = fmaxf(rms, SNS_RMS_FLOOR_MIN);
            float snapAlpha = fminf(1.0f, dt / 0.11f);
            snsAdaptiveFloor += snapAlpha * (target - snsAdaptiveFloor);
        }
    } else {
        snsBelowFloorCnt = 0;
        float ratio = rms / fmaxf(snsAdaptiveFloor, 1.0f);
        float d = (ratio - 1.0f) / SNS_FLOOR_SOFT_SIG;
        snsAdaptiveFloor *= (1.0f + SNS_FLOOR_LEAK * dt * expf(-(d * d)));
    }
    snsAdaptiveFloor = fmaxf(snsAdaptiveFloor, SNS_RMS_FLOOR_MIN);
    return snsAdaptiveFloor;
}

// Reseed the adaptive floor/ceiling/onset state to declaration defaults. Called
// at playback start so re-derived energy/onset converges cleanly instead of
// inheriting the prior live-audio state.
static void audioResetAdaptiveState() {
    snsAdaptiveFloor = 0.0f;
    snsRmsCeiling    = SNS_RMS_CEIL_MIN;
    snsBelowFloorCnt = 0;
    onsetFast = 0.0f;
    onsetSlow = 0.0f;
    onsetLock = 0.0f;
}

// Stage-2 → stage-3 audio derivation: companded RMS bytes → fxEnergy/fxOnset
// via the adaptive floor + log scaling against a leaky ceiling, plus onset from
// the window max/mean gap. Single source of truth for both live packets and
// recorded-frame playback, so silent replay reproduces the captured drive.
// Byte-envelope entry point: decode the companded RMS bytes (live AudioPacketV1
// and legacy recorded frames) and run the shared float core below.
static void audioDeriveFeaturesRms(float rmsMean, float rmsMax, float dt);
static void audioDeriveFeatures(uint8_t rmsMeanByte, uint8_t rmsMaxByte, float dt) {
    audioDeriveFeaturesRms(decodeCompand(rmsMeanByte, V1_RMS_FS),
                           decodeCompand(rmsMaxByte,  V1_RMS_FS), dt);
}

// Core: RMS floats (24-bit domain) → fxEnergy/fxOnset. Single source of truth
// for live packets, legacy envelope frames, and waveform-derived playback.
static void audioDeriveFeaturesRms(float rmsMean, float rmsMax, float dt) {
    // Energy: adaptive floor + log scaling against a leaky ceiling.
    snsRmsCeiling = fmaxf(SNS_RMS_CEIL_MIN, snsRmsCeiling * expf(-0.0025f * dt));
    if (rmsMean > snsRmsCeiling) snsRmsCeiling = rmsMean;
    snsUpdateFloor(rmsMean, dt);
    float effFloor = snsAdaptiveFloor * SNS_FLOOR_HEADROOM;
    if (rmsMean < effFloor) {
        fxEnergy = 0.0f;
    } else {
        float db = 20.0f * log10f(rmsMean / effFloor);
        float dbRange = 20.0f * log10f(snsRmsCeiling / effFloor);
        if (dbRange < 1.0f) dbRange = 1.0f;
        fxEnergy = clampf(db / dbRange, 0.0f, 1.0f);
    }

    // Onset: positive flux of the energy envelope. A syllable/word attack is
    // rmsMean RISING above its recent baseline — a discrete event. The old
    // detector used the within-window max/mean gap normalized by a decaying
    // peak, which sat near 1.0 through all continuous sound and fired on the
    // noise floor (over-active sparkle). Two EMAs (fast envelope, slow
    // baseline); fire on the rise crossing an adaptive threshold, then a
    // refractory lockout swallows the attack's sustain so it fires once.
    onsetFast += fminf(1.0f, dt / 0.030f) * (rmsMean - onsetFast);
    onsetSlow += fminf(1.0f, dt / 0.300f) * (rmsMean - onsetSlow);
    float rise   = onsetFast - onsetSlow;
    float thresh = fmaxf(SNS_ONSET_FLOOR, onsetRiseFrac * onsetSlow);  // self-scales to the room
    onsetLock = fmaxf(0.0f, onsetLock - dt);
    // Presence gate: the attack envelope must clear the same effective floor
    // that gates fxEnergy (see SNS_ONSET_FLOOR comment — wind-over-quiet fix).
    bool present = onsetFast > effFloor;
    if (present && rise > thresh && onsetLock <= 0.0f) {
        onsetLock = onsetRefrac;
        fxOnset = clampf(rise / fmaxf(onsetSlow, 1.0f), 0.0f, 1.0f);   // strength for sparkle density
    } else {
        fxOnset = 0.0f;                                                // discrete: zero between events
    }
}

static void processSensors(float dt) {
    uint32_t now = millis();

    // ── Accel ────────────────────────────────────────────────────
    // Liveness deltas are signed: the RX callback updates xLastMs
    // asynchronously, so a packet landing after `now` was captured makes
    // (now - xLastMs) wrap to ~2^32 unsigned — a fresh packet would read as
    // timed out, zeroing energy and reseeding the adaptive state for a frame.
    if (accelLastMs > 0 && (int32_t)(now - accelLastMs) < (int32_t)SENSOR_TIMEOUT_MS) {
        TelemetryPacketV1 a;
        memcpy(&a, (const void*)&accelPkt, sizeof(a));

        // Per-axis means are (rawMean >> 8) → gravity/tilt vector in counts.
        // Reconstruct counts at the full-scale used by the stub (16384 = 1g).
        sensorStub.ax = (int16_t)((int)a.ax_mean << 8);
        sensorStub.ay = (int16_t)((int)a.ay_mean << 8);
        sensorStub.az = (int16_t)((int)a.az_mean << 8);

        float axg = (float)sensorStub.ax / 16384.0f;
        float ayg = (float)sensorStub.ay / 16384.0f;
        float azg = (float)sensorStub.az / 16384.0f;

        // Tilt: angle of the gravity vector away from the z (flat) axis.
        float horiz = sqrtf(axg * axg + ayg * ayg);
        fxAngleDeg = atan2f(horiz, fabsf(azg)) * 180.0f / (float)M_PI;
        fxTiltBlend = clampf((fxAngleDeg - DEADZONE_DEG)
                             / (MAX_ANGLE_DEG - DEADZONE_DEG), 0.0f, 1.0f);
    } else {
        sensorStub.ax = 0; sensorStub.ay = 0; sensorStub.az = 16384;
        fxAngleDeg = 0.0f;
        fxTiltBlend = 0.0f;
    }

    // ── Gyro ─────────────────────────────────────────────────────
    if (gyroLastMs > 0 && (int32_t)(now - gyroLastMs) < (int32_t)SENSOR_TIMEOUT_MS) {
        GyroPacketV1 g;
        memcpy(&g, (const void*)&gyroPkt, sizeof(g));
        // gz_mean is the yaw-rate mean (int8 LSB ≈ V1_DPS_PER_LSB dps).
        sensorStub.gz = (int16_t)((float)g.gz_mean * V1_DPS_PER_LSB);
    } else {
        sensorStub.gz = 0;
    }

    // ── Audio ────────────────────────────────────────────────────
    if (audioLastMs > 0 && (int32_t)(now - audioLastMs) < (int32_t)SENSOR_TIMEOUT_MS) {
        AudioPacketV1 au;
        memcpy(&au, (const void*)&audioPkt, sizeof(au));
        audioDeriveFeatures(au.rms_mean, au.rms_max, dt);
    } else {
        // Audio packets stopped (sender off / out of range). Zero the outputs and
        // reset the adaptive floor/ceiling/onset-peak so they don't freeze stale
        // through the dropout and resume mid-decay — packets returning re-seed
        // cleanly, same as at playback start.
        fxEnergy = 0.0f;
        fxOnset = 0.0f;
        audioResetAdaptiveState();
    }

    // Slow "energy in the system": leaky integral of fxEnergy. Builds while
    // audio is present, bleeds off over ~energy5sTau when it stops.
    float a5 = fminf(1.0f, dt / energy5sTau);
    audioEnergy5s += a5 * (fxEnergy - audioEnergy5s);
}

// ── Per-strip bloom state ────────────────────────────────────────
struct BloomStrip {
    float breathPhase[LEDS_PER_STRIP];
    float breathPeriod[LEDS_PER_STRIP];
    float breathPeak[LEDS_PER_STRIP];
    float hueT[LEDS_PER_STRIP];
    float hueDrift[LEDS_PER_STRIP];
    float blackoutTimer[LEDS_PER_STRIP];
    float dormancyDur[LEDS_PER_STRIP];
    float dormancyRoll[LEDS_PER_STRIP];
    bool  gateOff[LEDS_PER_STRIP];
    uint16_t ditherR[LEDS_PER_STRIP];
    uint16_t ditherG[LEDS_PER_STRIP];
    uint16_t ditherB[LEDS_PER_STRIP];
    float flash;
    float energyBuffer;
};

static BloomStrip bloom[NUM_STRIPS];

static void resetBloomStrip(BloomStrip &bs) {
    bs.flash = 0.0f;
    bs.energyBuffer = 0.0f;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        bs.breathPhase[i]  = randFloat();
        bs.breathPeriod[i] = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bs.breathPeak[i]   = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bs.hueT[i]         = randFloat();
        float rate = BLOOM_HUE_DRIFT_MIN
            + randFloat() * (BLOOM_HUE_DRIFT_MAX - BLOOM_HUE_DRIFT_MIN);
        bs.hueDrift[i]     = (randFloat() > 0.5f) ? rate : -rate;
        bs.blackoutTimer[i] = 0.0f;
        bs.dormancyDur[i]   = bloomDormancyMin + randFloat() * (bloomDormancyMax - bloomDormancyMin);
        bs.dormancyRoll[i]  = randFloat();
        bs.gateOff[i]      = false;
        bs.ditherR[i] = 0;
        bs.ditherG[i] = 0;
        bs.ditherB[i] = 0;
    }
}

// ── Bloom render (per strip, ambient only — no motion) ──────────
static void renderBloomStrip(uint8_t s, float dt) {
    BloomStrip &bs = bloom[s];

    // No motion processing — ambient breathing only.
    // Flash/energyBuffer stay at zero (no input).

    float drain = bs.energyBuffer * bloomBufferDrain * dt;
    if (bs.energyBuffer < 0.05f) drain += 0.1f * dt;
    drain = fminf(drain, bs.energyBuffer);
    bs.energyBuffer -= drain;
    bs.flash = fminf(1.0f, bs.flash + drain);

    bs.flash *= expf(-bloomFlashDecayRate * dt);
    if (bs.flash < 0.005f) bs.flash = 0.0f;

    float flashLin = bs.flash * bloomBrightnessCap;
    float flashFrac = (bs.flash > 0.1f) ? clampf(bs.flash / 0.3f, 0.0f, 1.0f) : 0.0f;
    float oneMinusFF = 1.0f - flashFrac;

    // Pass 1: breath glow per pixel + field mean (the pivot for the stretch).
    float glow[LEDS_PER_STRIP];
    float glowSum = 0.0f;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        float breath = fastSinPhase(bs.breathPhase[i]) * 0.5f + 0.5f;
        glow[i] = bloomBreathFloor + breath * (bs.breathPeak[i] - bloomBreathFloor);
        glowSum += glow[i];
    }
    float glowMean = glowSum * (1.0f / (float)LEDS_PER_STRIP);
    // Audio energy → per-pixel overdrive added AFTER the knob/cap. Pixels above
    // the field mean blow through toward saturation; below-mean pixels go dark.
    float eStretch = (energyForce >= 0.0f) ? energyForce : audioEnergy5s;
    float audioDrive = bloomStretchGain * eStretch;   // 0 .. ~gain

    // Pass 2: ambient (knob-scaled) + audio overdrive (knob-independent).
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        // Signed deviation from the field mean, normalized to ~[-1, +1.2].
        float w = clampf((glow[i] - glowMean) * 4.0f, -1.0f, 1.5f);
        // Ambient breathing: dim when the brightness knob is low. Contrast
        // stretch around the field mean first — see bloomAmbientContrast.
        float stretched = clampf(glowMean + (glow[i] - glowMean) * bloomAmbientContrast,
                                 0.0f, 1.0f);
        float ambient = fastGamma24(stretched) * bloomBrightnessCap * renderBrightness;
        // Audio punch is full-scale linear (NOT knob-scaled) so it blows through.
        float breathLin = clampf(ambient + audioDrive * w, 0.0f, bloomAudioCeil);

        bs.breathPhase[i] += dt / bs.breathPeriod[i];
        if (bs.breathPhase[i] >= 1.0f) bs.breathPhase[i] -= 1.0f;
        bs.hueT[i] += bs.hueDrift[i] * dt;
        if (bs.hueT[i] > 1.0f) bs.hueT[i] -= 1.0f;
        else if (bs.hueT[i] < 0.0f) bs.hueT[i] += 1.0f;

        float h = bs.hueT[i];
        float baseR = lerpf(bloomHueA_R, bloomHueB_R, h);
        float baseG = lerpf(bloomHueA_G, bloomHueB_G, h);
        float baseB = lerpf(bloomHueA_B, bloomHueB_B, h);

        float oR = baseR * breathLin + (baseR * oneMinusFF + BLOOM_FLASH_R * flashFrac) * flashLin;
        float oG = baseG * breathLin + (baseG * oneMinusFF + BLOOM_FLASH_G * flashFrac) * flashLin;
        float oB = baseB * breathLin + (baseB * oneMinusFF + BLOOM_FLASH_B * flashFrac) * flashLin;

        // Hysteresis noise gate with dormancy hold
        float maxCh = fmaxf(fmaxf(oR, oG), oB);
        float thresh = bs.gateOff[i] ? GATE_OFF_THRESH : GATE_ON_THRESH;
        if (maxCh < thresh) {
            oR = oG = oB = 0.0f;
            if (!bs.gateOff[i] && bs.dormancyRoll[i] < bloomDormancyFrac) {
                bs.gateOff[i] = true;
                bs.blackoutTimer[i] = 0.0f;
                bs.dormancyDur[i] = bloomDormancyMin
                    + randFloat() * (bloomDormancyMax - bloomDormancyMin);
            }
        } else if (bs.gateOff[i]) {
            bs.blackoutTimer[i] += dt;
            if (bs.blackoutTimer[i] < bs.dormancyDur[i]) {
                oR = oG = oB = 0.0f;
            } else {
                bs.gateOff[i] = false;
                bs.blackoutTimer[i] = 0.0f;
            }
        }

        // renderBrightness already folded into `ambient`; the audio overdrive is
        // intentionally NOT scaled by it so loud audio punches through a low knob.

        uint16_t t16R = (uint16_t)fminf(oR * 256.0f, 65535.0f);
        uint16_t t16G = (uint16_t)fminf(oG * 256.0f, 65535.0f);
        uint16_t t16B = (uint16_t)fminf(oB * 256.0f, 65535.0f);
        if ((t16R | t16G | t16B) == 0) {
            bs.ditherR[i] = bs.ditherG[i] = bs.ditherB[i] = 0;
        }
        uint8_t r8 = t16R >> 8;
        uint8_t g8 = t16G >> 8;
        uint8_t b8 = t16B >> 8;

        setPixel(s, i, r8, g8, b8);
    }
}

// ── Sparkle syllable (ported from bulb-fleet) ───────────────────
// Onset-triggered LED ignition over a dim warm base. Without audio the
// idle twinkle below keeps it alive: sparse spontaneous single-LED
// sparks, dimmer and slower-fading than speech sparks — a quiet preview
// of what talking does. No per-effect cap.
#define SPARKLE_DEADBAND 0.08f
// Idle twinkle: starfield. Mean spontaneous sparks/sec across ALL strips,
// sized so ~25% of LEDs are lit at any moment. Measured on hardware:
// 95/s → 17% lit; 140/s → 19%; 220/s → ~25% (returns diminish as spawns
// increasingly land on occupied pixels and are dropped). (History: 5/s
// read as broken at low knob; 18/s chill but sparse.)
#define SPARKLE_IDLE_RATE 220.0f

static float syllSparkle[NUM_STRIPS][LEDS_PER_STRIP];
static float syllDecayArr[NUM_STRIPS][LEDS_PER_STRIP];
static float syllEnvelope = 0.0f;
static float syllCooldown = 0.0f;
static float syllIdleDebt = 0.0f;

static void resetSparkle() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            syllSparkle[s][i] = 0.0f;
            syllDecayArr[s][i] = 0.94f;
        }
    }
    syllEnvelope = 0.0f;
    syllCooldown = 0.0f;
    syllIdleDebt = 0.0f;
}

static void renderSparkle(float dt) {
    float energy = fxEnergy;
    float onsetNorm = fxOnset;
    float angleDeg = fxAngleDeg;
    float tiltBlend = fxTiltBlend;

    // gDt, not the speed-scaled dt: the ones knob shapes motion, never
    // reaction latency — envelope and onset cooldown track speech in real
    // time at any speed (same rule as gravity's gsEnergyEma).
    float attackAlpha = fminf(1.0f, gDt / 0.030f);
    float decayAlpha  = fminf(1.0f, gDt / 0.400f);
    if (energy > syllEnvelope)
        syllEnvelope += attackAlpha * (energy - syllEnvelope);
    else
        syllEnvelope += decayAlpha * (energy - syllEnvelope);

    syllCooldown = fmaxf(0.0f, syllCooldown - gDt);

    if (onsetNorm > 0.1f && syllCooldown <= 0.0f) {
        syllCooldown = 0.060f;
        int nIgnite = (int)(LEDS_PER_STRIP * (0.25f + 0.25f * onsetNorm));
        float sparkVal = 0.6f + 0.4f * onsetNorm;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            static uint16_t indices[LEDS_PER_STRIP];
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) indices[i] = i;
            for (int i = 0; i < nIgnite; i++) {
                int j = i + (int)(xorshift32() % (LEDS_PER_STRIP - i));
                uint16_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
            for (int i = 0; i < nIgnite; i++) {
                syllSparkle[s][indices[i]] = sparkVal;
                syllDecayArr[s][indices[i]] = 0.92f + randFloat() * 0.05f;
            }
        }
    }

    // Idle twinkle: warm-white starfield in silence. Val 0.45–0.75 lands
    // mid-high on the color ramp below → soft warm white stars (speech
    // sparks run 0.6–1.0 and ignite a third of the strip at once, so the
    // reaction still reads as a clear step up). Spawns landing on a live
    // spark are skipped — density self-limits without restart pops.
    // Ambient motion → speed-scaled dt.
    syllIdleDebt += SPARKLE_IDLE_RATE * dt;
    while (syllIdleDebt >= 1.0f) {
        syllIdleDebt -= 1.0f;
        uint8_t  s = (uint8_t)(xorshift32() % NUM_STRIPS);
        uint16_t i = (uint16_t)(xorshift32() % LEDS_PER_STRIP);
        if (syllSparkle[s][i] > 0.05f) continue;   // don't restart a live spark
        // Idle spark birth value — each star's intrinsic brightness. Raised
        // 0.45+0.30·r → 0.90+0.20·r so the per-lit-pixel mean (averaged over
        // fading stars) clears the bottom of the ambient band. Same spawn
        // rate/count (coverage unchanged); brighter stars ride a touch
        // warmer-white, as the speech sparks already do.
        syllSparkle[s][i]  = 0.90f + randFloat() * 0.20f;
        syllDecayArr[s][i] = 0.975f + randFloat() * 0.015f;
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            syllSparkle[s][i] *= fastDecay(syllDecayArr[s][i], dt * 30.0f);

    float base = fminf(syllEnvelope, 0.15f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float tiltR = 0, tiltG = 0, tiltB = 0;
        if (tiltBlend > 0.0f) {
            float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
            if (hueFrac < 0.0f) hueFrac = 0.0f;
            if (hueFrac > 1.0f) hueFrac = 1.0f;
            uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255)
                                         + STRIP_HUE_OFFSET[s]) & 0xFF);
            tiltR = (float)oklchVarL[hueIdx][0];
            tiltG = (float)oklchVarL[hueIdx][1];
            tiltB = (float)oklchVarL[hueIdx][2];
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float sp = syllSparkle[s][i];
            float bright = base + sp * (1.0f - base);
            if (bright < SPARKLE_DEADBAND) bright = 0.0f;

            float colR = 255.0f;
            float colG = 180.0f + (240.0f - 180.0f) * sp;
            float colB =  80.0f + (200.0f -  80.0f) * sp;

            if (tiltBlend > 0.0f) {
                colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
                colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
                colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
            }

            float wFold = 127.0f * (1.0f - tiltBlend);
            colR = fminf(255.0f, colR + wFold);
            colG = fminf(255.0f, colG + wFold);
            colB = fminf(255.0f, colB + wFold);

            float linBright = fastGamma24(bright);
            setPixelScaled(s, i,
                clampf(colR * linBright, 0.0f, 255.0f),
                clampf(colG * linBright, 0.0f, 255.0f),
                clampf(colB * linBright, 0.0f, 255.0f));
        }
    }
}

// ── Fire — audio-driven flame (ported from bulb-fleet renderFire) ───
// Base brightness rides speech energy (50 ms attack, 2 s decay; faint ember
// glow in silence). Onsets pump a flicker term on top. Sustained steady
// sound slowly "burns down" patches of LEDs toward ember-red (dropout) so
// held notes read as settling coals; transients re-ignite them. Color rides
// energy too: cool white when quiet → amber → red when loud; percussive-only
// input (onset with little energy) suppresses color toward white.
// Palette anchors are WS2811-corrected for these green-dominant LEDs (~2×
// red): G pulled well below the bulb-fleet values so amber doesn't read
// yellow and white doesn't read green. Eyeball-tune on hardware.
#define FIRE_FLICKER_SCALE  3.0f
// Silent ember floor — fire's intrinsic base flame intensity with no audio
// (the level base brightness rides in silence, and the floor it never drops
// below when sound is present). Raised 0.25→0.78 so the idle ember genuinely
// glows in the ambient band instead of reading near-black; speech still climbs
// from here toward full. This deliberately brightens fire's idle character.
#define FIRE_EMBER_FLOOR    0.78f
#define FIRE_DEADBAND       0.08f
#define FIRE_DROPOUT_DEPTH  0.85f
// bulb-fleet shipped two slots (FIRE_MELD without dropout, FIRE_FLICKER
// with); we have one slot, so dropout is a compile-time toggle to compare.
static const bool FIRE_WITH_DROPOUT = true;

// Audio-free simplification (no audio input wired right now): when set, fire
// drops all fxEnergy/fxOnset/dropout/color-energy machinery and runs as a
// deterministic flame — fixed height + color with the autonomous product-of-
// sines flicker only. Matches the steady-state look the audio form settles to
// in silence. Set to 0 to restore the full audio-reactive fire.
#define FIRE_NO_AUDIO     1
#define FIRE_FLAME_HEIGHT FIRE_EMBER_FLOOR   // fixed base brightness, audio-free

static float fireTime[NUM_STRIPS];
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

static void resetFire() {
    // Stagger phase per strip so flicker isn't lockstep.
    for (uint8_t s = 0; s < NUM_STRIPS; s++) fireTime[s] = (float)s * 1.37f;
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

static void renderFire(float dt) {
#if FIRE_NO_AUDIO
    // Audio removed — deterministic flame: fixed height + idle amber-red color
    // (the steady state the audio form settles to in silence), no onset pump,
    // no dropout. The product-of-sines flicker further down is autonomous.
    float base = FIRE_FLAME_HEIGHT;
    float dropoutAmount = 0.0f;
    fireFlickerIntensity = 0.0f;
    float baseColR = 231.4f, baseColG = 49.3f, baseColB = 2.9f;
    const float redR = 200.0f, redG = 15.0f, redB = 0.0f;  // red-shift target (unused: dropout off)
#else
    // Audio-feature EMAs use gDt (real time), not the speed-scaled dt: the
    // ones knob shapes flame motion, never reaction latency — same convention
    // as gravity's gsEnergyEma.
    float energy = fxEnergy;
    float onset  = fxOnset;
    bool isSilent = energy < 0.001f;
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);

    float attackAlpha = fminf(1.0f, gDt / 0.050f);
    float decayAlpha  = fminf(1.0f, gDt / 2.0f);
    float targetBrightness = isSilent ? FIRE_EMBER_FLOOR : fmaxf(FIRE_EMBER_FLOOR, energy);
    if (targetBrightness > fireBaseBrightness)
        fireBaseBrightness += attackAlpha * (targetBrightness - fireBaseBrightness);
    else
        fireBaseBrightness += decayAlpha * (targetBrightness - fireBaseBrightness);

    // fxOnset is discrete (zero between events, refractory-gated) where
    // bulb-fleet's onset was continuous; the 200 ms EMA turns each event
    // into a flicker pulse that decays back — pumps on attacks either way.
    float flickerAlpha = fminf(1.0f, gDt / 0.200f);
    float deltaTarget = isSilent ? 0.0f : onset;
    fireFlickerIntensity += flickerAlpha * (deltaTarget - fireFlickerIntensity);

    float dropoutAmount = 0.0f;
    if (FIRE_WITH_DROPOUT) {
        // Sustained sound = energy present but barely changing → coals settle.
        float energyDeriv = (energy - firePrevEnergyForDeriv) / fmaxf(gDt, 0.001f);
        firePrevEnergyForDeriv = energy;
        float derivAlpha = fminf(1.0f, gDt / 0.200f);
        fireEnergyDerivSmooth += derivAlpha * (energyDeriv - fireEnergyDerivSmooth);
        bool isSustaining = (!isSilent && energy > 0.05f
                             && fabsf(fireEnergyDerivSmooth) <= 0.5f);
        if (isSustaining)
            fireDropoutAmount = fminf(1.0f, fireDropoutAmount + gDt * 0.35f);
        else
            fireDropoutAmount = fmaxf(0.0f, fireDropoutAmount - gDt * 1.0f);
        dropoutAmount = fireDropoutAmount;
    }

    float colorAttack = fminf(1.0f, gDt / 0.080f);
    float colorDecay  = fminf(1.0f, gDt / 2.0f);
    float colorTarget;
    if (isPercussiveOnly)  colorTarget = 0.0f;
    else if (isSilent)     colorTarget = 0.3f;
    else                   colorTarget = fmaxf(0.3f, energy);
    if (colorTarget > fireColorEnergy)
        fireColorEnergy += colorAttack * (colorTarget - fireColorEnergy);
    else
        fireColorEnergy += colorDecay * (colorTarget - fireColorEnergy);

    // Base color from smoothed color-energy: white below the blend threshold,
    // amber through the mid range, red at RED_FULL and above.
    float ce = fireColorEnergy;
    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;
    const float amberR = 255.0f, amberG = 75.0f, amberB = 5.0f;
    const float redR   = 200.0f, redG   = 15.0f, redB   = 0.0f;
    const float whiteR = 180.0f, whiteG = 85.0f, whiteB = 160.0f;
    float baseColR, baseColG, baseColB;
    if (ce < WHITE_BLEND_THRESHOLD) {
        float tw = 1.0f - (ce / WHITE_BLEND_THRESHOLD);
        baseColR = amberR * (1.0f - tw) + whiteR * tw;
        baseColG = amberG * (1.0f - tw) + whiteG * tw;
        baseColB = amberB * (1.0f - tw) + whiteB * tw;
    } else {
        float tr = (ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD);
        if (tr > 1.0f) tr = 1.0f;
        baseColR = amberR * (1.0f - tr) + redR * tr;
        baseColG = amberG * (1.0f - tr) + redG * tr;
        baseColB = amberB * (1.0f - tr) + redB * tr;
    }

    float base = fireBaseBrightness;
    if (base < FIRE_DEADBAND) base = 0.0f;
#endif  // FIRE_NO_AUDIO
    float sScl = FIRE_FLICKER_SCALE;

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        fireTime[s] = fmodf(fireTime[s] + dt, 6283.1853f);
        float t = fireTime[s];
        // Per-strip spatial offset so noise patterns don't align.
        float sOff = (float)s * 17.0f;

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float fi = (float)i + sOff;

            float noise = fastSin(fi * 7.3f + t * 2.5f) *
                          fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

            // Relative flicker grows as the base dims so quiet fire still
            // visibly dances; flicker term adds onset-driven sparkle on top.
            float noiseAmp = fmaxf(0.15f * sScl, 0.10f * sScl / fmaxf(base, 0.1f));
            float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                         + fireFlickerIntensity * (noise - 0.5f) * 0.25f * sScl;

            float perLedDim = 0.0f;
            float colorRedShift = 0.0f;
            if (FIRE_WITH_DROPOUT && dropoutAmount > 0.0f) {
                // Slow-moving resilience field: low-resilience LEDs burn down
                // first, dimming and red-shifting toward embers.
                float resilience = fastSin(fi * 13.7f + t * 0.3f) *
                                   fastSin(fi * 9.1f + t * 0.2f) * 0.5f + 0.5f;
                perLedDim = clampf(
                    (dropoutAmount - resilience * 0.7f) / 0.3f, 0.0f, 1.0f
                ) * FIRE_DROPOUT_DEPTH;
                colorRedShift = clampf(perLedDim / 0.3f, 0.0f, 1.0f);
            }

            bright *= (1.0f - perLedDim);
            bright = clampf(bright, 0.0f, 1.0f);

            float colR = baseColR * (1.0f - colorRedShift) + redR * colorRedShift;
            float colG = baseColG * (1.0f - colorRedShift) + redG * colorRedShift;
            float colB = baseColB * (1.0f - colorRedShift) + redB * colorRedShift;

            float linBright = fastGamma24(bright);
            setPixelScaled(s, i,
                clampf(colR * linBright, 0.0f, 255.0f),
                clampf(colG * linBright, 0.0f, 255.0f),
                clampf(colB * linBright, 0.0f, 255.0f));
        }
    }
}

// ── Nebula (ported from bulb-fleet) ─────────────────────────────
// Breathing blue→magenta background + warm-white drifting orbs.
// No audio dependency. dt is pre-scaled by effSpeed at the call site;
// the intrinsic 0.3 tuning factor is kept (it is not the speed knob).
#define NEBULA_MAX_ORBS       5
#define NEBULA_ORB_TAIL       30.0f
#define NEBULA_ORB_BASE_SPEED 0.45f
#define NEBULA_SPAWN_CHANCE   0.03f   // per-frame @30fps reference; dt-scaled at spawn
#define NEBULA_MIN_LIFETIME   200     // frames @60fps reference; /60 to seconds at spawn
#define NEBULA_MAX_LIFETIME   300
#define NEBULA_INTRINSIC_SPD  0.3f

struct NebOrb {
    float pos;
    float vel;
    float age;
    float lifetime;
    bool active;
};

static float nebulaTime = 0.0f;
static NebOrb nebOrbs[NUM_STRIPS][NEBULA_MAX_ORBS];
#define nebDecay fxField.neb   // shared per-effect field buffer

static void resetNebula() {
    nebulaTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++)
            nebOrbs[s][o].active = false;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            nebDecay[s][i] = 0.0f;
    }
}

static void renderNebula(float dt) {
    float spd = NEBULA_INTRINSIC_SPD;
    nebulaTime += dt * spd;
    float t = nebulaTime * 60.0f;

    float decayPerFrame = 1.0f - (1.0f / NEBULA_ORB_TAIL);
    float decayPerSec = powf(decayPerFrame, 60.0f);
    float decay = powf(decayPerSec, dt);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            nebDecay[s][i] *= decay;
            if (nebDecay[s][i] < 0.01f) nebDecay[s][i] = 0.0f;
        }

        float spawnRoll = (float)(xorshift32() & 0xFFFF) / 65536.0f;
        if (spawnRoll < NEBULA_SPAWN_CHANCE * dt * 30.0f) {
            for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
                if (!nebOrbs[s][o].active) {
                    NebOrb &orb = nebOrbs[s][o];
                    orb.pos = randFloat() * (float)LEDS_PER_STRIP;
                    float dir = (xorshift32() & 1) ? 1.0f : -1.0f;
                    orb.vel = dir * NEBULA_ORB_BASE_SPEED * spd * (0.7f + randFloat() * 0.6f);
                    orb.age = 0.0f;
                    orb.lifetime = (NEBULA_MIN_LIFETIME
                        + (float)(xorshift32() % (NEBULA_MAX_LIFETIME - NEBULA_MIN_LIFETIME)))
                        / 60.0f;
                    orb.active = true;
                    break;
                }
            }
        }

        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
            NebOrb &orb = nebOrbs[s][o];
            if (!orb.active) continue;

            orb.age += dt;
            orb.pos += orb.vel * dt * 60.0f;
            orb.pos = fmodf(orb.pos + (float)LEDS_PER_STRIP, (float)LEDS_PER_STRIP);

            if (orb.age >= orb.lifetime) {
                orb.active = false;
                continue;
            }

            float lc = (float)orb.age / (float)orb.lifetime;
            float bright;
            if (lc < 0.4f) {
                float tf = lc / 0.4f;
                bright = tf * tf * (3.0f - 2.0f * tf);
            } else if (lc > 0.6f) {
                float tf = (1.0f - lc) / 0.4f;
                bright = tf * tf * (3.0f - 2.0f * tf);
            } else {
                bright = 1.0f;
            }

            int base = (int)orb.pos;
            int next = (base + 1) % LEDS_PER_STRIP;
            float frac = orb.pos - (float)base;
            float val0 = bright * 0.6f * (1.0f - frac);
            float val1 = bright * 0.6f * frac;
            if (base >= 0 && base < LEDS_PER_STRIP)
                nebDecay[s][base] = fminf(1.0f, nebDecay[s][base] + val0);
            if (next >= 0 && next < LEDS_PER_STRIP)
                nebDecay[s][next] = fminf(1.0f, nebDecay[s][next] + val1);
        }

        float sOff = (float)s * 0.167f;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float pos = (float)i / (float)LEDS_PER_STRIP;

            // Background field level scaled ×1.5 (51→76, 38→57, 51→76, cap
            // 153→230) to lift nebula's per-lit-pixel mean into the ambient
            // band. Same breathing/spatial shape, just a brighter cloud.
            float breathing = 76.0f + 57.0f * fastSin((t * 0.0105f));
            float phase = pos + sOff + t * 0.006f;
            float spatial = 76.0f * (0.5f + 0.5f * fastSin(phase * 6.2832f));
            float bgBright = clampf(breathing + spatial, 0.0f, 230.0f) / 255.0f;

            float colorPhase = pos * 6.2832f + sOff * 6.2832f + t * 0.009f;
            float colorShift = 0.5f + 0.5f * fastSin(colorPhase);

            float bgR = (20.0f + colorShift * 235.0f) / 255.0f;
            float bgG = (30.0f - colorShift * 20.0f) / 255.0f;
            float bgB = (255.0f - colorShift * 125.0f) / 255.0f;

            float r = bgR * bgBright;
            float g = bgG * bgBright;
            float b = bgB * bgBright;

            float orbB = nebDecay[s][i];
            if (orbB > 0.01f) {
                // Orb highlight scaled ×1.5 to match the brightened background
                // (1.0/0.94/0.78 → 1.5/1.41/1.17), holding the orb:cloud ratio.
                r += orbB * 1.5f;
                g += orbB * 1.41f;
                b += orbB * 1.17f;
            }

            setPixelScaled(s, i,
                clampf(r, 0.0f, 1.0f) * 255.0f,
                clampf(g, 0.0f, 1.0f) * 255.0f,
                clampf(b, 0.0f, 1.0f) * 255.0f);
        }
    }
}

// ── Rainbow (OKLCH scroll, ported from bulb-fleet renderIdle) ───
// No audio dependency. Pure ambient hue scroll. dt is pre-scaled by
// effSpeed at the call site. No per-effect brightness cap — output
// scale is renderBrightness via setPixelScaled.
#define RAINBOW_SCROLL_SPEED  0.10f
// Value (V) level applied to the OKLCH scroll — rainbow's intrinsic brightness.
// 1.11 lifts the per-lit-pixel mean to the top of the ambient band (it's the
// brightest effect, mapped to ~89). Brightest hues clip slightly at 255.
#define RAINBOW_VALUE         1.11f

static float rainbowPhase = 0.0f;

static void resetRainbow() {
    rainbowPhase = 0.0f;
}

static void renderRainbow(float dt) {
    rainbowPhase = fmodf(rainbowPhase + RAINBOW_SCROLL_SPEED * dt, 1.0f);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float stripOff = STRIP_HUE_OFFSET[s] / 256.0f;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float pos = (float)i / (float)LEDS_PER_STRIP;
            float hue = fmodf(pos + rainbowPhase + stripOff, 1.0f);
            uint8_t idx = (uint8_t)(hue * 255.0f);
            setPixelScaled(s, i,
                (float)oklchVarL[idx][0] * RAINBOW_VALUE,
                (float)oklchVarL[idx][1] * RAINBOW_VALUE,
                (float)oklchVarL[idx][2] * RAINBOW_VALUE);
        }
    }
}

// ── Leaf wind (ported from bulb-fleet renderLeafWind) ───────────
// 1D drift along each strip: leaves blow from one end to the other, each LED
// lights by 1D distance to each leaf. Bulb-fleet's 2D topology.h is a hex map
// for a different installation, so tree-of-record collapses to a simple linear
// topology — LED index → position 0..1 (i / (LEDS_PER_STRIP-1)). Each leaf
// drifts independently per strip. No per-effect cap — output via setPixelScaled.
#define LW_MAX_LEAVES     8     // per strip
// Per-leaf intrinsic brightness (value scalar on the leaf's glow, pre-gamma).
// 1.25 lifts leaf-wind's per-lit-pixel mean into the ambient band. Applied after
// the totalGlow>0.01 lit gate so coverage (which pixels light) is unchanged.
#define LW_LEAF_VALUE     1.25f
#define LW_GLOW_RADIUS    0.05f
#define LW_GLOW_SQ2       (2.0f * LW_GLOW_RADIUS * LW_GLOW_RADIUS)
#define LW_WIND_SPEED     0.35f
#define LW_SPAWN_INTERVAL 0.9f
#define LW_FADE_IN        0.4f
#define LW_VEL_TAU        0.22f   // velocity EMA time constant (sec); = orig 0.85/frame @30fps
#define LW_TURBULENCE     0.3f
// Launch gust. Was 0.25 / TC 2.5s — but flight takes ~2.5s, so the gust used to
// decay away mid-journey and the leaf braked ~11% toward the tip (measured).
// TC now 20s (>> flight) makes the gust a near-constant per-leaf speed offset:
// no tip-braking, leaves still ride different gusts. Magnitude trimmed 0.25→0.157
// so overall speed is unchanged (sim: 0.455→0.448).
#define LW_BOOST_SPEED    0.157f
#define LW_BOOST_TC       20.0f

static const uint8_t LW_PALETTE[][3] = {
    {255, 140, 20}, {240, 100, 10}, {220, 60, 5}, {200, 40, 10},
    {180, 30, 5},   {255, 180, 40}, {160, 25, 5},
};
#define LW_PALETTE_SIZE (sizeof(LW_PALETTE) / sizeof(LW_PALETTE[0]))

struct LwLeaf {
    float pos, vel, boost, age, brightness;
    uint8_t r, g, b;
    bool active;
};
static LwLeaf lwLeaves[NUM_STRIPS][LW_MAX_LEAVES];
static float lwTime = 0.0f;
static float lwSpawnTimer[NUM_STRIPS];

static float lwNoise1d(float pos, float t, int seed) {
    return (fastSin(pos * 0.4f + t * 0.3f + seed * 7.3f)
          * fastSin(pos * 0.17f - t * 0.19f + seed * 3.1f + (float)M_PI * 0.5f)
          + fastSin(pos * 0.09f + t * 0.13f + seed * 1.7f) * 0.5f) / 1.5f;
}

static void resetLeafWind() {
    lwTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        lwSpawnTimer[s] = randFloat() * LW_SPAWN_INTERVAL;
        for (int i = 0; i < LW_MAX_LEAVES; i++) lwLeaves[s][i].active = false;
    }
}

static void lwSpawnLeaf(uint8_t s) {
    for (int i = 0; i < LW_MAX_LEAVES; i++) {
        if (lwLeaves[s][i].active) continue;
        LwLeaf &lf = lwLeaves[s][i];
        lf.active = true;
        lf.pos = -0.05f;   // enter from LED 0 end, blow toward the tip
        lf.boost = LW_BOOST_SPEED * (0.5f + randFloat() * 0.5f);
        lf.vel = 0.0f;
        lf.age = 0.0f;
        lf.brightness = 0.0f;
        int ci = (int)(xorshift32() % LW_PALETTE_SIZE);
        lf.r = LW_PALETTE[ci][0];
        lf.g = LW_PALETTE[ci][1];
        lf.b = LW_PALETTE[ci][2];
        return;
    }
}

static void renderLeafWind(float dt) {
    lwTime += dt;
    float boostDecay = expf(-dt / LW_BOOST_TC);
    float lwAlpha = fminf(1.0f, dt / LW_VEL_TAU);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        lwSpawnTimer[s] += dt;
        while (lwSpawnTimer[s] >= LW_SPAWN_INTERVAL) {
            lwSpawnTimer[s] -= LW_SPAWN_INTERVAL;
            lwSpawnLeaf(s);
        }

        for (int i = 0; i < LW_MAX_LEAVES; i++) {
            if (!lwLeaves[s][i].active) continue;
            LwLeaf &lf = lwLeaves[s][i];

            float noise = lwNoise1d(lf.pos * 5.0f + (float)s * 3.0f, lwTime, i);
            float speedMult = fmaxf(0.1f, 1.0f + noise * LW_TURBULENCE);
            float force = LW_WIND_SPEED * speedMult;

            lf.vel = fmaxf(LW_WIND_SPEED * 0.4f, lf.vel + lwAlpha * (force - lf.vel));
            lf.boost *= boostDecay;
            lf.pos += (lf.vel + lf.boost) * dt;
            lf.age += dt;

            lf.brightness = (lf.age < LW_FADE_IN) ? (lf.age / LW_FADE_IN) : 1.0f;

            if (lf.pos > 1.05f) lf.active = false;
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float lpos = (float)i / (float)(LEDS_PER_STRIP - 1);
            float totalGlow = 0.0f, cr = 0.0f, cg = 0.0f, cb = 0.0f;

            for (int li = 0; li < LW_MAX_LEAVES; li++) {
                if (!lwLeaves[s][li].active) continue;
                LwLeaf &lf = lwLeaves[s][li];
                float d = lpos - lf.pos;
                float intensity = expf(-(d * d) / LW_GLOW_SQ2) * lf.brightness;
                if (intensity < 0.005f) continue;
                totalGlow += intensity;
                cr += intensity * lf.r;
                cg += intensity * lf.g;
                cb += intensity * lf.b;
            }

            if (totalGlow > 0.01f) {
                cr /= totalGlow; cg /= totalGlow; cb /= totalGlow;
                float bright = fminf(totalGlow * LW_LEAF_VALUE, 1.0f);
                float linBright = fastGamma24(bright);
                setPixelScaled(s, i, cr * linBright, cg * linBright, cb * linBright);
            } else {
                setPixelScaled(s, i, 0.0f, 0.0f, 0.0f);
            }
        }
    }
}

// ── Creatures (ported from biolum_mixed) ────────────────────────
// Bloom + crawl creatures drift along a virtual buffer, one independent
// simulation per physical strip (no mirroring — every strip is its own world).
// Excitement is audio-driven: the speech-energy envelope (same signal as
// gravity's spread) stands in for the original accel shake — this install
// has no motion input, so the old shake/gyro-scroll feeds were retired.
// In silence the visuals fall back to pure ambient drift. Virtual buffer is
// 100 physical LEDs + a SCROLL_MARGIN each side (margin kept for the buffer
// layout even though gyro scroll is gone).
// No per-effect cap — output via setPixelScaled.
#define CR_NUM_UNITS     NUM_STRIPS  // one independent creature world per strip
                                      // (was hardcoded 6 from the 6-strip fork →
                                      // OOB fxFloorTimer[4/5] write → reboot)
#define CR_MAX_CREATURES 7
#define CR_MAX_ANIMS     6
#define CR_SCROLL_MARGIN 12
#define CR_VIRTUAL_LEDS  (LEDS_PER_STRIP + 2 * CR_SCROLL_MARGIN)
#define CR_SPAWN_MARGIN  5
#define CR_DESPAWN_MARGIN (-5)

#define CR_COLOR_R   0.0f
#define CR_COLOR_G 180.0f
#define CR_COLOR_B 220.0f
// Creature field brightness (value scalar on the rendered glow, pre-gamma).
// 1.20 lifts creatures' per-lit-pixel mean into the ambient band. Applied to the
// already-composited buffer so the bloom/crawl envelope shapes and the teal
// color are preserved; the >0 pixels are the same (coverage unchanged).
#define CR_VALUE     1.20f

// --- Excitement params (buffer/drain machinery from biolum_mixed shake) ---
// Audio feed: smoothed speech energy above CR_AUDIO_THRESH stands in for the
// old accel excess-g. After the ceiling calibration, deliberate close speech
// rides ~0.7–1.0 and room music ~0.2, so 0.25 keeps music from exciting the
// creatures. CR_AUDIO_GAIN maps the max excess (~0.75) onto the ~2 scale the
// buffer gain/drain constants were tuned for.
#define CR_AUDIO_THRESH         0.25f
#define CR_AUDIO_GAIN           2.7f
#define CR_SHAKE_BUFFER_GAIN    0.8f
#define CR_SHAKE_BUFFER_MAX     0.5f
#define CR_SHAKE_DRAIN_RATE     0.8f
#define CR_SHAKE_BASE_FALL      0.5f
#define CR_SHAKE_FALL_SLOWDOWN  0.5f
#define CR_SHAKE_SCATTER_RADIUS 1.5f
#define CR_SHAKE_BRIGHT_BOOST   0.8f
#define CR_SHAKE_HUE_DRIFT      90.0f

#define CR_PULSE_EXPANSION_SPEED 3.3f
// Ratio doubled (was 1.9) → drift halved → each creature lives ~2x longer on
// screen. Pulse expansion speed is untouched, so animations look identical.
#define CR_PULSE_TO_DRIFT_RATIO  3.8f
#define CR_AVG_DRIFT  (CR_PULSE_EXPANSION_SPEED / CR_PULSE_TO_DRIFT_RATIO)
#define CR_DRIFT_SPREAD 0.33f
#define CR_DRIFT_MIN  (CR_AVG_DRIFT * (1.0f - CR_DRIFT_SPREAD))
#define CR_DRIFT_MAX  (CR_AVG_DRIFT * (1.0f + CR_DRIFT_SPREAD))

#define CR_BLOOM_RADIUS        3.0f
#define CR_BLOOM_EDGE_SOFTNESS 0.8f
#define CR_BLOOM_RISE          2.0f
#define CR_BLOOM_HOLD          1.5f
#define CR_BLOOM_FALL          5.0f
#define CR_BLOOM_TOTAL  (CR_BLOOM_RISE + CR_BLOOM_HOLD + CR_BLOOM_FALL)
#define CR_BLOOM_EMIT_LO 1.2f
#define CR_BLOOM_EMIT_HI 2.8f

#define CR_CRAWL_RADIUS         8.0f
#define CR_CRAWL_PULSE_LIFETIME (CR_CRAWL_RADIUS / CR_PULSE_EXPANSION_SPEED)
#define CR_CRAWL_PULSE_FADE     1.4f
#define CR_CRAWL_TAIL_DECAY     4.0f
#define CR_CRAWL_EMIT_LO        1.2f
#define CR_CRAWL_EMIT_HI        2.8f

enum CrKind : uint8_t { CR_KIND_BLOOM, CR_KIND_CRAWL };

struct CrAnim { float age; bool active; };
struct CrCreature {
    float    pos, vel;
    CrKind   kind;
    bool     alive;
    float    emitTimer;
    CrAnim   anims[CR_MAX_ANIMS];
    float    hueOffset, hueSweep;
};
struct CrUnit {
    CrCreature creatures[CR_MAX_CREATURES];
    float bufR[CR_VIRTUAL_LEDS], bufG[CR_VIRTUAL_LEDS], bufB[CR_VIRTUAL_LEDS];
};
static CrUnit crUnits[CR_NUM_UNITS];

// --- Global excitement state (driven by processInteraction) ---
static float crShakeLevel   = 0.0f;   // smooth 0→1 controlling visual effects
static float crShakeBuffer  = 0.0f;   // energy buffer (spiky input → smooth drain)
static float crShakeTime    = 0.0f;   // accumulated seconds excited
static bool  crShakeActive  = false;
static float crShakeCooldown = 2.0f;
static bool  crLifecycleFrozen = false;
static float crShakeHueDrift = 0.0f;
static float crEnergyEma    = 0.0f;   // smoothed speech energy (audio feed)

static inline float crRandf(float lo, float hi) {
    return lo + randFloat() * (hi - lo);
}

static void crProcessInteraction(float dt) {
    // gDt for the envelope, not the speed-scaled dt: the ones knob shapes the
    // creatures' motion, never reaction latency (gravity convention).
    crEnergyEma += fminf(1.0f, gDt / 0.15f) * (fxEnergy - crEnergyEma);
    float excess = (crEnergyEma - CR_AUDIO_THRESH) * CR_AUDIO_GAIN;
    if (excess > 0.0f) {
        crShakeActive = true;
        crShakeBuffer += excess * CR_SHAKE_BUFFER_GAIN * dt;
        if (crShakeBuffer > CR_SHAKE_BUFFER_MAX) crShakeBuffer = CR_SHAKE_BUFFER_MAX;
        crShakeTime += dt;
    } else {
        crShakeActive = false;
    }

    float drained = fminf(crShakeBuffer, CR_SHAKE_DRAIN_RATE * sqrtf(crShakeBuffer) * dt);
    crShakeBuffer -= drained;

    if (crShakeActive) {
        crShakeLevel = fminf(1.0f, crShakeLevel + drained);
    } else {
        float fallRate = CR_SHAKE_BASE_FALL / (1.0f + crShakeTime * CR_SHAKE_FALL_SLOWDOWN);
        crShakeLevel -= fallRate * dt;
        if (crShakeLevel <= 0.0f) { crShakeLevel = 0.0f; crShakeTime = 0.0f; }
    }

    if (crShakeLevel > 0.2f) crShakeCooldown = 0.0f;
    else                     crShakeCooldown += dt;
    crLifecycleFrozen = (crShakeLevel > 0.2f || crShakeCooldown < 0.3f);

    crShakeHueDrift += crShakeLevel * CR_SHAKE_HUE_DRIFT * dt;
    if (crShakeHueDrift > 360.0f) crShakeHueDrift -= 360.0f;
}

static void crInitCreature(CrCreature &c) {
    c.kind = (xorshift32() & 1) ? CR_KIND_BLOOM : CR_KIND_CRAWL;
    float speed = crRandf(CR_DRIFT_MIN, CR_DRIFT_MAX);
    c.pos = crRandf(CR_SPAWN_MARGIN, CR_VIRTUAL_LEDS - CR_SPAWN_MARGIN);
    c.vel = (xorshift32() & 1) ? speed : -speed;
    c.alive = true;
    c.emitTimer = (c.kind == CR_KIND_BLOOM)
        ? crRandf(CR_BLOOM_EMIT_LO, CR_BLOOM_EMIT_HI)
        : crRandf(CR_CRAWL_EMIT_LO, CR_CRAWL_EMIT_HI);
    for (int i = 0; i < CR_MAX_ANIMS; i++) c.anims[i].active = false;
    c.hueOffset = crRandf(0.0f, 360.0f);
    c.hueSweep = crRandf(60.0f, 90.0f);
}

static void resetCreatures() {
    for (int p = 0; p < CR_NUM_UNITS; p++) {
        for (int c = 0; c < CR_MAX_CREATURES; c++) {
            CrCreature &cr = crUnits[p].creatures[c];
            crInitCreature(cr);
            // Spawn visible: pre-age one anim partway into its envelope so
            // the field is alive the instant the mode is entered, instead
            // of dark through the first emit delay + bloom rise (~4 s).
            float maxAge = (cr.kind == CR_KIND_BLOOM) ? CR_BLOOM_TOTAL
                           : (CR_CRAWL_PULSE_LIFETIME + CR_CRAWL_PULSE_FADE);
            cr.anims[0].active = true;
            cr.anims[0].age = crRandf(0.0f, maxAge * 0.6f);
        }
    }
}

static void crEmitAnim(CrCreature &c) {
    for (int i = 0; i < CR_MAX_ANIMS; i++) {
        if (!c.anims[i].active) {
            c.anims[i].active = true;
            c.anims[i].age = 0.0f;
            return;
        }
    }
}

static void crUpdateCreature(CrCreature &c, float dt) {
    if (!c.alive) return;
    c.pos += c.vel * dt;
    if (c.pos < CR_DESPAWN_MARGIN || c.pos > CR_VIRTUAL_LEDS - CR_DESPAWN_MARGIN) {
        c.alive = false;
        return;
    }
    c.emitTimer -= dt;
    if (c.emitTimer <= 0.0f) {
        crEmitAnim(c);
        if (c.kind == CR_KIND_BLOOM) {
            float base = CR_BLOOM_TOTAL + crRandf(CR_BLOOM_EMIT_LO, CR_BLOOM_EMIT_HI);
            c.emitTimer = base * crRandf(0.33f, 2.0f);
        } else {
            c.emitTimer = CR_CRAWL_PULSE_LIFETIME + CR_CRAWL_PULSE_FADE
                          + crRandf(CR_CRAWL_EMIT_LO, CR_CRAWL_EMIT_HI);
        }
    }
    float maxAge = (c.kind == CR_KIND_BLOOM) ? CR_BLOOM_TOTAL
                   : (CR_CRAWL_PULSE_LIFETIME + CR_CRAWL_PULSE_FADE);
    for (int i = 0; i < CR_MAX_ANIMS; i++) {
        if (!c.anims[i].active) continue;
        c.anims[i].age += dt;
        if (c.anims[i].age >= maxAge) c.anims[i].active = false;
    }
}

// Shake-aware color: blend base teal toward a per-position rainbow.
static inline void crShakeColor(const CrCreature &c, float relOut,
                                float &pixR, float &pixG, float &pixB) {
    pixR = CR_COLOR_R; pixG = CR_COLOR_G; pixB = CR_COLOR_B;
    if (crShakeLevel > 0.01f) {
        float colorT = crShakeLevel;
        float hueFrac = (relOut + 1.0f) * 0.5f;
        float hue = fmodf(c.hueOffset + crShakeHueDrift + hueFrac * c.hueSweep, 360.0f);
        float rR, rG, rB;
        hsvToRgb(hue, 1.0f, 1.0f, rR, rG, rB);
        pixR = CR_COLOR_R * (1.0f - colorT) + rR * colorT;
        pixG = CR_COLOR_G * (1.0f - colorT) + rG * colorT;
        pixB = CR_COLOR_B * (1.0f - colorT) + rB * colorT;
    }
}

static void crRenderBloom(const CrCreature &c, float *bufR, float *bufG, float *bufB) {
    float center = c.pos;
    float halfWidth = CR_BLOOM_RADIUS + CR_BLOOM_EDGE_SOFTNESS;
    float totalSpan = halfWidth * (1.0f + crShakeLevel * CR_SHAKE_SCATTER_RADIUS);
    int lo = (int)(center - totalSpan - 1); if (lo < 0) lo = 0;
    int hi = (int)(center + totalSpan + 1); if (hi > CR_VIRTUAL_LEDS - 1) hi = CR_VIRTUAL_LEDS - 1;

    float brightMult = 1.0f + crShakeLevel * CR_SHAKE_BRIGHT_BOOST;

    for (int a = 0; a < CR_MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float age = c.anims[a].age;
        float envelope;
        if (age < CR_BLOOM_RISE) {
            envelope = age / CR_BLOOM_RISE;
        } else if (age < CR_BLOOM_RISE + CR_BLOOM_HOLD) {
            envelope = 1.0f;
        } else {
            float fallT = (age - CR_BLOOM_RISE - CR_BLOOM_HOLD) / CR_BLOOM_FALL;
            envelope = fmaxf(0.0f, 1.0f - fallT);
            envelope *= envelope;
        }
        if (envelope < 0.003f) continue;

        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;
            float srcDist = fabsf(relOut) * halfWidth;
            float spatial = (srcDist <= CR_BLOOM_RADIUS) ? 1.0f
                : expf(-(srcDist - CR_BLOOM_RADIUS) / CR_BLOOM_EDGE_SOFTNESS);

            float gapFade = 1.0f;
            if (crShakeLevel > 0.01f) {
                float gapSize = 0.3f * crShakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            float br = envelope * spatial * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR, pixG, pixB;
            crShakeColor(c, relOut, pixR, pixG, pixB);
            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void crRenderCrawl(const CrCreature &c, float *bufR, float *bufG, float *bufB) {
    float center = c.pos;
    float halfWidth = CR_CRAWL_RADIUS + CR_CRAWL_TAIL_DECAY;
    float totalSpan = halfWidth * (1.0f + crShakeLevel * CR_SHAKE_SCATTER_RADIUS);
    int lo = (int)(center - totalSpan - 1); if (lo < 0) lo = 0;
    int hi = (int)(center + totalSpan + 1); if (hi > CR_VIRTUAL_LEDS - 1) hi = CR_VIRTUAL_LEDS - 1;

    float brightMult = 1.0f + crShakeLevel * CR_SHAKE_BRIGHT_BOOST;

    for (int a = 0; a < CR_MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float t = fminf(c.anims[a].age / CR_CRAWL_PULSE_LIFETIME, 1.0f);
        float radius = t * CR_CRAWL_RADIUS;
        float fade = 1.0f;
        if (c.anims[a].age > CR_CRAWL_PULSE_LIFETIME) {
            float fadeT = (c.anims[a].age - CR_CRAWL_PULSE_LIFETIME) / CR_CRAWL_PULSE_FADE;
            fade = fmaxf(0.0f, 1.0f - fadeT);
            fade *= fade;
        }
        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;
            float srcDist = fabsf(relOut) * halfWidth;
            if (srcDist > radius) continue;
            float behind = radius - srcDist;
            float br = expf(-behind / CR_CRAWL_TAIL_DECAY);
            if (behind < 1.0f) br *= behind;

            float gapFade = 1.0f;
            if (crShakeLevel > 0.01f) {
                float gapSize = 0.3f * crShakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            br *= fade * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR, pixG, pixB;
            crShakeColor(c, relOut, pixR, pixG, pixB);
            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void renderCreatures(float dt) {
    crProcessInteraction(dt);

    for (int p = 0; p < CR_NUM_UNITS; p++) {
        CrUnit &ps = crUnits[p];
        memset(ps.bufR, 0, sizeof(ps.bufR));
        memset(ps.bufG, 0, sizeof(ps.bufG));
        memset(ps.bufB, 0, sizeof(ps.bufB));

        for (int c = 0; c < CR_MAX_CREATURES; c++) {
            if (!crLifecycleFrozen) crUpdateCreature(ps.creatures[c], dt);
            if (!ps.creatures[c].alive) continue;
            if (ps.creatures[c].kind == CR_KIND_BLOOM)
                crRenderBloom(ps.creatures[c], ps.bufR, ps.bufG, ps.bufB);
            else
                crRenderCrawl(ps.creatures[c], ps.bufR, ps.bufG, ps.bufB);
        }
        for (int c = 0; c < CR_MAX_CREATURES; c++)
            if (!ps.creatures[c].alive) crInitCreature(ps.creatures[c]);

        // One unit drives one physical strip — no mirroring.
        uint8_t s = p;
        for (int i = 0; i < LEDS_PER_STRIP; i++) {
            int src = CR_SCROLL_MARGIN + i;
            float vR = clampf(ps.bufR[src] * CR_VALUE, 0.0f, 1.0f);
            float vG = clampf(ps.bufG[src] * CR_VALUE, 0.0f, 1.0f);
            float vB = clampf(ps.bufB[src] * CR_VALUE, 0.0f, 1.0f);
            // Gamma the scalar brightness and rescale channels proportionally
            // (gravity's pattern). Per-channel gamma — what the original port
            // did — shifts hue on mixed colors as they dim (doctrine rule:
            // gamma the brightness multiplier, never individual channels).
            float bright = fmaxf(vR, fmaxf(vG, vB));
            float norm = (bright > 0.001f) ? fastGamma24(bright) / bright : 0.0f;
            setPixelScaled(s, i, vR * norm * 255.0f,
                                 vG * norm * 255.0f,
                                 vB * norm * 255.0f);
        }
    }
}

// ── Light Through (stormy sunset, slot 10) ──────────────────────
// Storm-ceiling base field (slate↔indigo) with warm bloom-patches that swell
// open then seal, like sun breaking through cloud. 6 strips are radial spokes
// from center: r = index/(LEDS-1) (0=trunk, 1=edge), spoke s at angle s×60°.
// Spawn/churn/growth all scale with fxDt (speed knob folds in upstream).
// Routes through setPixelScaled like every other effect.
#define LT_MAX_PATCHES   6
#define LT_SPAWN_PERIOD  2.2f   // mean seconds between patches (Poisson)
#define LT_CHURN_RATE    0.12f

// Palette (0–255 linear). All entries scaled ×1.44 from the original storm
// palette (slate 50/55/80, indigo 35/35/60, amber 240/180/90, peach 220/160/110,
// salmon 200/140/100) so the field's intrinsic brightness lifts the per-lit-pixel
// mean into the ambient band; hue ratios are preserved (uniform per-color gain),
// and the already-bright warm patch cores clip slightly toward white ("sun").
#define LT_SLATE_R   72.0f
#define LT_SLATE_G   79.0f
#define LT_SLATE_B  115.0f
#define LT_INDIGO_R  50.0f
#define LT_INDIGO_G  50.0f
#define LT_INDIGO_B  86.0f
#define LT_AMBER_R  346.0f
#define LT_AMBER_G  259.0f
#define LT_AMBER_B  130.0f
#define LT_PEACH_R  317.0f
#define LT_PEACH_G  230.0f
#define LT_PEACH_B  158.0f
#define LT_SALMON_R 288.0f
#define LT_SALMON_G 202.0f
#define LT_SALMON_B 144.0f

struct LtPatch {
    float rc;        // radial center 0.15–0.85
    float cosTc;     // precomputed cos/sin of angular center
    float sinTc;
    float maxR;      // 0.25–0.7
    float life;      // 1.5–3.5 s
    float age;
    bool  active;
};
static LtPatch ltPatches[LT_MAX_PATCHES];
static float ltTime = 0.0f;
static float ltSpawnTimer = 0.0f;
static float ltSpokeCos[NUM_STRIPS];
static float ltSpokeSin[NUM_STRIPS];

// Layered-sine pseudo-noise in 0..1 (cheap, no Perlin).
static inline float ltNoise(float x) {
    return 0.5f + 0.5f * fastSin(x * 2.3f + fastSin(x * 1.7f + ltTime * 0.08f));
}

static void resetLightThrough() {
    ltTime = 0.0f;
    ltSpawnTimer = 0.0f;
    for (int i = 0; i < LT_MAX_PATCHES; i++) ltPatches[i].active = false;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float ang = (float)s * (60.0f * (float)M_PI / 180.0f);
        ltSpokeCos[s] = cosf(ang);
        ltSpokeSin[s] = sinf(ang);
    }
}

static void ltSpawnPatch() {
    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (ltPatches[i].active) continue;
        LtPatch &p = ltPatches[i];
        p.rc = 0.15f + randFloat() * 0.70f;        // [0.15, 0.85]
        float tc = randFloat() * 2.0f * (float)M_PI;
        p.cosTc = cosf(tc);
        p.sinTc = sinf(tc);
        p.maxR = 0.25f + randFloat() * 0.45f;       // [0.25, 0.7]
        p.life = 1.5f + randFloat() * 2.0f;         // [1.5, 3.5]
        p.age = 0.0f;
        p.active = true;
        return;
    }
}

static void renderLightThrough(float dt) {
    ltTime += dt;

    // Poisson spawn: probability dt/period per frame.
    ltSpawnTimer += dt;
    float spawnProb = dt / LT_SPAWN_PERIOD;
    if (randFloat() < spawnProb) ltSpawnPatch();

    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (!ltPatches[i].active) continue;
        ltPatches[i].age += dt;
        if (ltPatches[i].age >= ltPatches[i].life) ltPatches[i].active = false;
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float sc = ltSpokeCos[s];
        float ss = ltSpokeSin[s];
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r = (float)i / (float)(LEDS_PER_STRIP - 1);

            // Layer A — storm ceiling.
            float density = ltNoise(r * 3.0f + 0.7f * (float)s + ltTime * LT_CHURN_RATE);
            float centerLift = lerpf(1.0f, 0.85f, r);
            float outR = lerpf(LT_SLATE_R, LT_INDIGO_R, density) * centerLift;
            float outG = lerpf(LT_SLATE_G, LT_INDIGO_G, density) * centerLift;
            float outB = lerpf(LT_SLATE_B, LT_INDIGO_B, density) * centerLift;

            // LED planar position on the spoke.
            float px = r * sc;
            float py = r * ss;

            // Layer B — warm bloom-patches (sum where they overlap).
            for (int pi = 0; pi < LT_MAX_PATCHES; pi++) {
                if (!ltPatches[pi].active) continue;
                LtPatch &p = ltPatches[pi];
                float tau = p.age / p.life;
                float radius = p.maxR * fastSin((float)M_PI * tau);
                if (radius < 0.001f) continue;

                float dx = px - p.rc * p.cosTc;
                float dy = py - p.rc * p.sinTc;
                float d = sqrtf(dx * dx + dy * dy);

                float g = d / (0.6f * radius);
                float a = expf(-(g * g));
                if (a < 0.004f) continue;

                // Color: amber→peach→salmon by d/radius.
                float dr = clampf(d / radius, 0.0f, 1.0f);
                float pcR, pcG, pcB;
                if (dr < 0.5f) {
                    float t = dr / 0.5f;
                    pcR = lerpf(LT_AMBER_R, LT_PEACH_R, t);
                    pcG = lerpf(LT_AMBER_G, LT_PEACH_G, t);
                    pcB = lerpf(LT_AMBER_B, LT_PEACH_B, t);
                } else {
                    float t = (dr - 0.5f) / 0.5f;
                    pcR = lerpf(LT_PEACH_R, LT_SALMON_R, t);
                    pcG = lerpf(LT_PEACH_G, LT_SALMON_G, t);
                    pcB = lerpf(LT_PEACH_B, LT_SALMON_B, t);
                }

                outR = outR * (1.0f - a) + pcR * a + 0.15f * a * 255.0f;
                outG = outG * (1.0f - a) + pcG * a + 0.15f * a * 255.0f;
                outB = outB * (1.0f - a) + pcB * a;
            }

            setPixelScaled(s, i,
                clampf(outR, 0.0f, 255.0f),
                clampf(outG, 0.0f, 255.0f),
                clampf(outB, 0.0f, 255.0f));
        }
    }
}

// ── Heat diffusion (ported from audio-reactive heat_diffusion.py) ──
// 1D heat equation per strip: speech energy injects heat at a slowly
// patrolling point, diffusion spreads it, cooling sinks it. Blackbody
// ramp black→deep red→orange→white; sustained loud speech reaches
// white-hot at the source (steady-state peak ≈ rate / (2·√(α·c))).
// A passive sparkle overlay drifts on top: 1–3 px wide, fading in and
// out over a few seconds at nominal speed, soft warm white.
#define HEAT_ALPHA       40.0f   // diffusivity (px²/s)
#define HEAT_COOLING     0.9f    // heat sink rate (1/s)
#define HEAT_ROT_PERIOD  18.0f   // s per strip traversal at nominal speed
#define HEAT_MAX_STEP    (0.4f / HEAT_ALPHA)  // explicit-Euler stability bound
#define HEAT_SPARKS      4       // passive sparkle slots per strip
// Idle ember: minimum injection so silence still shows a red blob roaming
// with the patrol point. Steady-state peak T ≈ 1.17 × floor: 0.25 → ≈0.29,
// a fully saturated red right at the edge of orange — clearly alive in
// silence, but speech still has the whole orange→white range to climb.
static float heatIdleFloor = 0.78f;  // live-tunable: HEAT_IDLE_FLOOR.
// Intrinsic silent-injection level → steady-state ember temperature (T≈1.17×floor
// through heatRamp). Raised 0.25→0.78 to lift the idle ember's per-lit-pixel mean
// into the 18–35%-of-255 band; the hotter blob reads brighter (toward orange/white)
// while speech still has the top of the ramp to climb. Deliberate idle brightening.

static float heatInjRate = 14.0f;  // heat/s at fxEnergy = 1
#define heatT fxField.heat         // shared per-effect field buffer
static float heatInjPos[NUM_STRIPS];
static float heatInjDir[NUM_STRIPS];

struct HeatSpark {
    uint16_t pos;     // left edge
    uint8_t  width;   // 1–3 px
    float    phase;   // 0–1 through the fade-in/out cycle
    float    period;  // seconds for the full in+out
    float    idle;    // seconds until (re)spawn
    bool     active;
};
static HeatSpark heatSparks[NUM_STRIPS][HEAT_SPARKS];

static void resetHeatDiffusion() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) heatT[s][i] = 0.0f;
        heatInjPos[s] = randFloat() * (float)LEDS_PER_STRIP;
        heatInjDir[s] = (xorshift32() & 1) ? 1.0f : -1.0f;
        for (uint8_t k = 0; k < HEAT_SPARKS; k++) {
            heatSparks[s][k].active = false;
            heatSparks[s][k].idle = randFloat() * 3.0f;
        }
    }
}

// Blackbody-ish ramp; channels saturate in sequence so T>1 reads white.
static inline void heatRamp(float T, float &r, float &g, float &b) {
    r = clampf(T * 3.0f, 0.0f, 1.0f);
    g = clampf((T - 0.30f) * 1.6f, 0.0f, 1.0f);
    b = clampf((T - 0.80f) * 3.0f, 0.0f, 1.0f);
}

static void renderHeatDiffusion(float dt) {
    static float lap[LEDS_PER_STRIP];

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float *T = heatT[s];

        // Patrol the injection point (motion → speed-scaled dt).
        heatInjPos[s] += heatInjDir[s]
                       * ((float)LEDS_PER_STRIP / HEAT_ROT_PERIOD) * dt;
        if (heatInjPos[s] < 0.0f) {
            heatInjPos[s] = 0.0f;
            heatInjDir[s] = 1.0f;
        } else if (heatInjPos[s] > (float)(LEDS_PER_STRIP - 1)) {
            heatInjPos[s] = (float)(LEDS_PER_STRIP - 1);
            heatInjDir[s] = -1.0f;
        }

        // Inject heat from speech energy. gDt, not the speed-scaled dt:
        // how strongly audio is expressed must not change with the ones
        // knob (same rule as gravity/sparkle). The idle floor keeps an
        // ember roaming in silence — see heatIdleFloor.
        float inject = heatInjRate * fmaxf(fxEnergy, heatIdleFloor) * gDt;
        if (inject > 0.0f) {
            int c = (int)heatInjPos[s];
            T[c] += inject * 0.5f;
            if (c > 0)                  T[c - 1] += inject * 0.25f;
            if (c < LEDS_PER_STRIP - 1) T[c + 1] += inject * 0.25f;
        }

        // Diffuse + cool, substepped to stay inside the stability bound
        // when the speed knob stretches dt.
        float rem = dt;
        while (rem > 0.0f) {
            float step = fminf(rem, HEAT_MAX_STEP);
            rem -= step;
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                float left  = (i > 0) ? T[i - 1] : T[i];
                float right = (i < LEDS_PER_STRIP - 1) ? T[i + 1] : T[i];
                lap[i] = left + right - 2.0f * T[i];
            }
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                T[i] += (HEAT_ALPHA * lap[i] - HEAT_COOLING * T[i]) * step;
                if (T[i] < 0.0f) T[i] = 0.0f;
            }
        }

        // Passive sparkle lifecycle (ambient motion → speed-scaled dt).
        for (uint8_t k = 0; k < HEAT_SPARKS; k++) {
            HeatSpark &sp = heatSparks[s][k];
            if (!sp.active) {
                sp.idle -= dt;
                if (sp.idle <= 0.0f) {
                    sp.active = true;
                    sp.pos = (uint16_t)(xorshift32() % (LEDS_PER_STRIP - 2));
                    sp.width = 1 + (uint8_t)(xorshift32() % 3);
                    sp.period = 2.5f + randFloat() * 2.5f;
                    sp.phase = 0.0f;
                }
            } else {
                sp.phase += dt / sp.period;
                if (sp.phase >= 1.0f) {
                    sp.active = false;
                    sp.idle = 1.0f + randFloat() * 3.0f;
                }
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r, g, b;
            heatRamp(T[i], r, g, b);

            for (uint8_t k = 0; k < HEAT_SPARKS; k++) {
                HeatSpark &sp = heatSparks[s][k];
                if (!sp.active) continue;
                if (i < sp.pos || i >= sp.pos + sp.width) continue;
                float env = fastSin((float)M_PI * sp.phase);
                env = env * env * 0.42f;   // sin² in/out, peak 0.42
                r += env;
                g += env * 0.92f;          // soft warm white
                b += env * 0.72f;
            }

            float bright = fmaxf(r, fmaxf(g, b));
            if (bright > 1.0f) {
                float inv = 1.0f / bright;
                r *= inv; g *= inv; b *= inv;
                bright = 1.0f;
            }
            float norm = (bright > 0.001f) ? fastGamma24(bright) / bright : 0.0f;
            setPixelScaled(s, i, r * norm * 255.0f, g * norm * 255.0f,
                           b * norm * 255.0f);
        }
    }
}

// ── Black out all strips (reserved/off slots) ───────────────────
// ── Galaxy (slot 8) ─────────────────────────────────────────────
// A dark-purple background that breathes very slowly — a full fade in and out
// over ~GAL_BG_PERIOD seconds — with soft "galaxies" drifting into view and
// fading out in front of it. Ambient: no audio/sensor dependency. Each strip
// runs its own population. Color is a deliberate desaturated-random palette
// (an INVARIANT 3 exception, like fire/nebula/light-through): galaxies want
// pale tints, not the saturated oklchVarL hues. Routed through setPixelScaled
// like every other effect, so the hue/sat knobs and floor policy apply.
//
// Population math: spawn and death both scale with fxDt, so steady-state count
// ≈ GAL_BIRTH_RATE × lifetime ≈ 1.5 per strip and is INVARIANT to the speed
// knob — the ones knob changes the pace, not how many galaxies are present.
#define GAL_MAX_PER_STRIP  7       // headroom above the ~2.5 mean (Poisson tail)
#define GAL_BG_PERIOD      20.0f   // s — full fade in+out at nominal speed (ones=8)
#define GAL_BG_PEAK        0.20f   // background scalar ceiling (kept dim → "dark")
#define GAL_BG_R           0.50f   // dark-purple background color (0..1 linear)
#define GAL_BG_G           0.00f
#define GAL_BG_B           0.72f
#define GAL_LIFE_MIN       1.0f    // galaxy lifetime range (s); nominal ~3 (±2)
#define GAL_LIFE_MAX       5.0f
#define GAL_DRIFT_MIN      5.0f    // LEDs traversed over a whole lifetime
#define GAL_DRIFT_MAX      10.0f
#define GAL_SIGMA_MIN      2.5f    // glow half-width (LEDs)
#define GAL_SIGMA_MAX      4.5f
#define GAL_PEAK           0.60f   // per-galaxy brightness ceiling (0..1)
#define GAL_SAT_MIN        0.18f   // desaturated colors
#define GAL_SAT_MAX        0.40f
#define GAL_BIRTH_RATE     0.83f   // spawns/s/strip → ~2.5 present (rate×life, ~3s)
#define GAL_FADE_FRAC      0.30f   // fraction of life spent fading in / out
// Per-pixel sparkle ("sparkler") inside each galaxy: individual LEDs pop bright
// and multiplicatively decay, scattered around the galaxy center — replaces the
// old smooth traveling-wave shimmer. Same decay model as the basic sparkle fx.
#define GAL_BED_LEVEL      0.22f   // faint smooth glow bed; sparks ride on top
#define GAL_SPARK_RATE     22.0f   // spark births/s per galaxy (× its envelope)
#define GAL_SPARK_DECAY    0.86f   // per-(1/30s) multiplicative decay (snappy pop)
#define GAL_SPARK_SPREAD   1.5f    // spark scatter, in units of σ, around center

struct Galaxy {
    float pos;        // LED position
    float vel;        // LEDs/s drift (signed)
    float age;        // s
    float lifetime;   // s
    float sigma;      // glow half-width (LEDs)
    float cutoff;     // 3.2σ render bound (LEDs)
    float r, g, b;    // base color 0..1 (desaturated)
    float sparkDebt;  // fractional spark-birth accumulator
    bool  active;
};
static Galaxy galaxies[NUM_STRIPS][GAL_MAX_PER_STRIP];
static float  galBgPhase = 0.0f;   // seconds into the background breath
// Per-pixel sparkle field (decaying brightness, 0..~1) for the "sparkler" twinkle.
static float  galSpark[NUM_STRIPS][LEDS_PER_STRIP];

static void galSpawn(Galaxy &gx, float startAge) {
    gx.pos      = randFloat() * (float)LEDS_PER_STRIP;
    gx.lifetime = GAL_LIFE_MIN + randFloat() * (GAL_LIFE_MAX - GAL_LIFE_MIN);
    float drift = GAL_DRIFT_MIN + randFloat() * (GAL_DRIFT_MAX - GAL_DRIFT_MIN);
    float dir   = (xorshift32() & 1) ? 1.0f : -1.0f;
    gx.vel      = dir * drift / gx.lifetime;
    gx.sigma    = GAL_SIGMA_MIN + randFloat() * (GAL_SIGMA_MAX - GAL_SIGMA_MIN);
    gx.cutoff   = gx.sigma * 3.2f;
    float hue   = randFloat() * 360.0f;
    float sat   = GAL_SAT_MIN + randFloat() * (GAL_SAT_MAX - GAL_SAT_MIN);
    float cr, cg, cb;
    hsvToRgb(hue, sat, 1.0f, cr, cg, cb);
    gx.r = cr / 255.0f; gx.g = cg / 255.0f; gx.b = cb / 255.0f;
    gx.sparkDebt = 0.0f;
    gx.age    = startAge;
    gx.active = true;
}

// Smoothstep fade in (first GAL_FADE_FRAC of life), hold, fade out (last).
static inline float galEnvelope(const Galaxy &gx) {
    float lc = gx.age / gx.lifetime;
    if (lc < GAL_FADE_FRAC) {
        float tf = lc / GAL_FADE_FRAC;
        return tf * tf * (3.0f - 2.0f * tf);
    } else if (lc > 1.0f - GAL_FADE_FRAC) {
        float tf = (1.0f - lc) / GAL_FADE_FRAC;
        return tf * tf * (3.0f - 2.0f * tf);
    }
    return 1.0f;
}

static void resetGalaxy() {
    galBgPhase = GAL_BG_PERIOD * 0.25f;   // start mid-rise so entry isn't black
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) galSpark[s][i] = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t g = 0; g < GAL_MAX_PER_STRIP; g++)
            galaxies[s][g].active = false;
        // Pre-spawn 1–2 at random points in their lives so the field is alive
        // the instant the mode is entered (mirrors creatures' spawn-visible).
        uint8_t pre = 1 + (uint8_t)(xorshift32() & 1);
        for (uint8_t k = 0; k < pre; k++) {
            galSpawn(galaxies[s][k], 0.0f);
            galaxies[s][k].age = randFloat() * galaxies[s][k].lifetime;
        }
    }
}

static void renderGalaxy(float dt) {
    // Background breath: full fade in+out over GAL_BG_PERIOD. 0.5−0.5cos →
    // dark at phase 0, dim-purple peak at the half-period.
    galBgPhase += dt;
    if (galBgPhase >= GAL_BG_PERIOD) galBgPhase -= GAL_BG_PERIOD;
    float bgLevel = GAL_BG_PEAK *
        (0.5f - 0.5f * cosf(6.2831853f * (galBgPhase / GAL_BG_PERIOD)));
    float bgR = GAL_BG_R * bgLevel;
    float bgG = GAL_BG_G * bgLevel;
    float bgB = GAL_BG_B * bgLevel;

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Advance / retire galaxies on this strip.
        for (uint8_t g = 0; g < GAL_MAX_PER_STRIP; g++) {
            Galaxy &gx = galaxies[s][g];
            if (!gx.active) continue;
            gx.age += dt;
            gx.pos += gx.vel * dt;
            if (gx.age >= gx.lifetime) gx.active = false;
        }
        // Poisson birth into a free slot.
        if (randFloat() < GAL_BIRTH_RATE * dt) {
            for (uint8_t g = 0; g < GAL_MAX_PER_STRIP; g++)
                if (!galaxies[s][g].active) { galSpawn(galaxies[s][g], 0.0f); break; }
        }
        // Per-frame envelope (0..1) and brightness amplitude for each galaxy.
        float env[GAL_MAX_PER_STRIP];
        float amp[GAL_MAX_PER_STRIP];
        for (uint8_t g = 0; g < GAL_MAX_PER_STRIP; g++) {
            env[g] = galaxies[s][g].active ? galEnvelope(galaxies[s][g]) : 0.0f;
            amp[g] = env[g] * GAL_PEAK;
        }

        // Decay the per-pixel sparkle field (snappy "sparkler" pops). Same
        // multiplicative-decay model as the basic sparkle effect.
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            galSpark[s][i] *= fastDecay(GAL_SPARK_DECAY, dt * 30.0f);

        // Birth new sparks scattered around each galaxy's center, at a rate that
        // scales with the galaxy's envelope (brighter galaxies sparkle harder).
        // Spread is < cutoff, so every spark lands inside the rendered footprint.
        for (uint8_t k = 0; k < GAL_MAX_PER_STRIP; k++) {
            if (env[k] <= 0.0f) continue;
            Galaxy &gx = galaxies[s][k];
            gx.sparkDebt += GAL_SPARK_RATE * dt * env[k];
            while (gx.sparkDebt >= 1.0f) {
                gx.sparkDebt -= 1.0f;
                float off = (randFloat() * 2.0f - 1.0f) * gx.sigma * GAL_SPARK_SPREAD;
                int li = (int)(gx.pos + off + 0.5f);
                if (li >= 0 && li < LEDS_PER_STRIP && galSpark[s][li] < 0.05f)
                    galSpark[s][li] = 0.85f + randFloat() * 0.30f;
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r = bgR, g = bgG, b = bgB;
            float spark = galSpark[s][i];
            for (uint8_t k = 0; k < GAL_MAX_PER_STRIP; k++) {
                if (amp[k] <= 0.0f) continue;
                Galaxy &gx = galaxies[s][k];
                float d = (float)i - gx.pos;
                if (d < -gx.cutoff || d > gx.cutoff) continue;
                float spatial = expf(-(d * d) / (2.0f * gx.sigma * gx.sigma));
                // Faint smooth bed + bright per-pixel sparkle riding on top.
                float bri = amp[k] * spatial *
                            (GAL_BED_LEVEL + (1.0f - GAL_BED_LEVEL) * spark);
                float cr = gx.r * bri, cg = gx.g * bri, cb = gx.b * bri;
                // Screen-blend galaxies over the background and each other.
                r = r + cr - r * cr;
                g = g + cg - g * cg;
                b = b + cb - b * cb;
            }
            setPixelScaled(s, i, r * 255.0f, g * 255.0f, b * 255.0f);
        }
    }
}

// ── Bouquet (anchored wildflowers, gusts roll root→tip) ─────────
#define BQ_FLOWERS        20
#define BQ_CLUSTERS       8
#define BQ_SIGMA          0.010f
#define BQ_SIGMA_SQ2      (2.0f * BQ_SIGMA * BQ_SIGMA)
#define BQ_LEAN_MAX       0.018f
#define BQ_SPRING_W       6.0f
#define BQ_SPRING_ZETA    0.35f
#define BQ_IDLE_FLUTTER   0.06f
#define BQ_WIND_FLUTTER   0.55f
#define BQ_FLOWER_VALUE   0.9f
#define BQ_CALM_MEAN      0.5f
#define BQ_CALM_MAX       2.0f
#define BQ_GUST_POOL      2
#define BQ_GUST_SPEED_MIN 0.15f
#define BQ_GUST_SPEED_MAX 0.35f
#define BQ_GUST_SIGMA     0.11f
#define BQ_GUST_DIE_FRAC  0.33f
#define BQ_GUST_DIE_TC    0.8f
// wind-phase gain: ruffle flicker freq ≈ GAIN·|dwx/dt|, peaks at gust edges
#define BQ_RUFFLE_GAIN    180.0f
#define BQ_FLOWER_RUFFLE  0.43f

static const uint8_t BQ_PALETTE[][3] = {
    {255, 45, 85},   {220, 25, 130}, {255, 105, 15}, {255, 185, 40},
    {235, 205, 150}, {150, 55, 230}, {255, 60, 35},
};
#define BQ_PALETTE_SIZE (sizeof(BQ_PALETTE) / sizeof(BQ_PALETTE[0]))

struct BqFlower {
    float pos, freq, phase, lean, leanVel;
    uint8_t r, g, b;
};
static BqFlower bqFlowers[NUM_STRIPS][BQ_FLOWERS];

struct BqWind {
    bool gusting;
    float front, speed, strength, life, calmLeft;
};
static BqWind bqWind[NUM_STRIPS][BQ_GUST_POOL];
static float bqTime = 0.0f;

static inline float bqCalmDraw() {
    return fminf(BQ_CALM_MAX,
                 -BQ_CALM_MEAN * logf(fmaxf(randFloat(), 1e-6f)));
}

static inline float bqWindAt(const BqWind *pool, float x) {
    float wx = 0.0f;
    for (int g = 0; g < BQ_GUST_POOL; g++) {
        const BqWind &w = pool[g];
        if (!w.gusting) continue;
        float d = x - w.front;
        float v = w.strength
                * expf(-(d * d) / (2.0f * BQ_GUST_SIGMA * BQ_GUST_SIGMA));
        if (v > wx) wx = v;
    }
    return wx;
}

static void resetBouquet() {
    bqTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        int f = 0;
        for (int c = 0; c < BQ_CLUSTERS; c++) {
            float center = (c + 0.5f) / (float)BQ_CLUSTERS
                         + (randFloat() - 0.5f) * 0.10f;
            int count = (c & 1) ? 2 : 3;
            for (int k = 0; k < count && f < BQ_FLOWERS; k++, f++) {
                BqFlower &fl = bqFlowers[s][f];
                fl.pos = fminf(0.97f, fmaxf(0.03f,
                    center + (randFloat() - 0.5f) * 0.06f));
                fl.freq = 1.5f + randFloat() * 2.0f;
                fl.phase = randFloat() * 2.0f * (float)M_PI;
                fl.lean = 0.0f;
                fl.leanVel = 0.0f;
                const uint8_t *col = BQ_PALETTE[xorshift32() % BQ_PALETTE_SIZE];
                fl.r = col[0]; fl.g = col[1]; fl.b = col[2];
            }
        }
        for (int g = 0; g < BQ_GUST_POOL; g++) {
            bqWind[s][g].gusting = false;
            bqWind[s][g].calmLeft = randFloat() * BQ_CALM_MAX
                                  + g * BQ_CALM_MAX;
        }
    }
}

static void renderBouquet(float dt) {
    bqTime += dt;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        BqWind *pool = bqWind[s];
        for (int g = 0; g < BQ_GUST_POOL; g++) {
            BqWind &w = pool[g];
            if (!w.gusting) {
                w.calmLeft -= dt;
                if (w.calmLeft <= 0.0f) {
                    bool entryClear = true;
                    float aheadSpeed = BQ_GUST_SPEED_MAX;
                    for (int o = 0; o < BQ_GUST_POOL; o++) {
                        if (o == g || !pool[o].gusting) continue;
                        if (pool[o].front < 3.0f * BQ_GUST_SIGMA)
                            entryClear = false;
                        aheadSpeed = fminf(aheadSpeed, pool[o].speed);
                    }
                    if (entryClear) {
                        w.gusting = true;
                        w.front = -3.0f * BQ_GUST_SIGMA;
                        w.speed = fminf(aheadSpeed, BQ_GUST_SPEED_MIN
                                + randFloat()
                                * (BQ_GUST_SPEED_MAX - BQ_GUST_SPEED_MIN));
                        w.strength = 0.4f + randFloat() * 0.6f;
                        w.life = (randFloat() < BQ_GUST_DIE_FRAC)
                               ? (0.3f + randFloat() * 0.4f) / w.speed
                               : 1e9f;
                    }
                }
            } else {
                w.front += w.speed * dt;
                w.life -= dt;
                if (w.life < 0.0f) w.strength *= expf(-dt / BQ_GUST_DIE_TC);
                if (w.front > 1.0f + 3.0f * BQ_GUST_SIGMA
                    || w.strength < 0.03f) {
                    w.gusting = false;
                    w.calmLeft = bqCalmDraw();
                }
            }
        }

        float fcen[BQ_FLOWERS], fbri[BQ_FLOWERS];
        for (int f = 0; f < BQ_FLOWERS; f++) {
            BqFlower &fl = bqFlowers[s][f];
            float wx = bqWindAt(pool, fl.pos);
            float acc = BQ_SPRING_W * BQ_SPRING_W * (wx - fl.lean)
                      - 2.0f * BQ_SPRING_ZETA * BQ_SPRING_W * fl.leanVel;
            fl.leanVel += acc * dt;
            fl.lean += fl.leanVel * dt;
            fcen[f] = fl.pos + fl.lean * BQ_LEAN_MAX;
            float amp = BQ_IDLE_FLUTTER
                      + BQ_WIND_FLUTTER * fminf(1.0f, fabsf(fl.lean));
            fbri[f] = BQ_FLOWER_VALUE * fmaxf(0.0f,
                1.0f + amp * fastSin(2.0f * (float)M_PI * fl.freq * bqTime
                                     + fl.phase));
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float x = (float)i / (float)(LEDS_PER_STRIP - 1);
            float wx = bqWindAt(pool, x);
            float ruffleSin = fastSin(x * 23.0f + s * 5.1f
                                + bqTime * 0.4f + BQ_RUFFLE_GAIN * wx);
            float shimmer = 0.7f + 0.3f * ruffleSin;
            float base = shimmer * (1.0f + 0.8f * wx);
            float cr = base * 3.0f, cg = base * 14.0f, cb = base * 2.0f;
            float flowerRuffle = 1.0f + BQ_FLOWER_RUFFLE * wx * ruffleSin;

            float fr = 0.0f, fg = 0.0f, fb = 0.0f, wsum = 0.0f, imax = 0.0f;
            for (int f = 0; f < BQ_FLOWERS; f++) {
                float d = x - fcen[f];
                if (fabsf(d) > 4.0f * BQ_SIGMA) continue;
                float intensity = expf(-(d * d) / BQ_SIGMA_SQ2) * fbri[f]
                                * flowerRuffle;
                BqFlower &fl = bqFlowers[s][f];
                float wgt = intensity * intensity * intensity;
                fr += wgt * fl.r;
                fg += wgt * fl.g;
                fb += wgt * fl.b;
                wsum += wgt;
                if (intensity > imax) imax = intensity;
            }
            if (wsum > 0.0f) {
                cr += imax * fr / wsum;
                cg += imax * fg / wsum;
                cb += imax * fb / wsum;
            }
            setPixelScaled(s, i, cr, cg, cb);
        }
    }
}


// ── Collision (crawlers annihilate into an ignition burst) ──────
#define CO_CALM_MEAN     5.0f
#define CO_CALM_MIN      1.5f
#define CO_CRAWL_W       6.0f
#define CO_SPEED_MIN     30.0f
#define CO_SPEED_MAX     60.0f
#define CO_MASS_MIN      0.7f
#define CO_MASS_MAX      1.6f
#define CO_FRAGS         16
#define CO_FRAG_DRAG_TC  0.6f
#define CO_FRAG_LIFE_MIN 0.4f
#define CO_FRAG_LIFE_MAX 1.1f
#define CO_TRAIL         0.30f
#define CO_SPARKLES      24
#define CO_SPKL_KICK     120.0f
#define CO_SPKL_DRAG_TC  0.12f
#define CO_SPKL_LIFE_MIN 0.5f
#define CO_SPKL_LIFE_MAX 1.3f
#define CO_SPKL_DUTY     0.45f
#define CO_IDLE_RATE     8.0f
#define CO_DEADBAND      0.05f
static float coShatterK = 1.0f;

enum CoPhase : uint8_t { CO_CALM, CO_APPROACH };
struct CoStrip {
    uint8_t phase;
    bool collided;
    float p1, p2, v1, v2, m1, m2, w1, w2, impact, timer;
};
struct CoFrag { float pos, vel, life, maxLife, bright; };
struct CoSparkle { float pos, vel, life, maxLife; };
static CoStrip co[NUM_STRIPS];
static CoFrag coFrag[NUM_STRIPS][CO_FRAGS];
static CoSparkle coSpkl[NUM_STRIPS][CO_SPARKLES];
static float coSpark[NUM_STRIPS][LEDS_PER_STRIP];
static float coDecayArr[NUM_STRIPS][LEDS_PER_STRIP];
static float coIdleDebt = 0.0f;

static void resetCollision() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        co[s].phase = CO_CALM;
        co[s].collided = false;
        co[s].timer = randFloat() * CO_CALM_MEAN;
        for (int k = 0; k < CO_FRAGS; k++) coFrag[s][k].life = 0.0f;
        for (int k = 0; k < CO_SPARKLES; k++) coSpkl[s][k].life = 0.0f;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            coSpark[s][i] = 0.0f;
            coDecayArr[s][i] = 0.94f;
        }
    }
    coIdleDebt = 0.0f;
}

static inline float coBlobBright(float d, float w) {
    float a = fabsf(d) / w;
    if (a >= 1.0f) return 0.0f;
    return fastSinPhase(0.25f * (1.0f - a));
}

static void renderCollision(float dt) {
    coIdleDebt += CO_IDLE_RATE * dt;
    while (coIdleDebt >= 1.0f) {
        coIdleDebt -= 1.0f;
        uint8_t  s = (uint8_t)(xorshift32() % NUM_STRIPS);
        uint16_t i = (uint16_t)(xorshift32() % LEDS_PER_STRIP);
        if (coSpark[s][i] > 0.05f) continue;
        coSpark[s][i]    = 0.35f + randFloat() * 0.25f;
        coDecayArr[s][i] = 0.975f + randFloat() * 0.015f;
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        CoStrip &c = co[s];
        switch (c.phase) {
        case CO_CALM:
            c.timer -= dt;
            if (c.timer <= 0.0f) {
                c.phase = CO_APPROACH;
                c.collided = false;
                c.v1 = CO_SPEED_MIN
                     + randFloat() * (CO_SPEED_MAX - CO_SPEED_MIN);
                c.v2 = CO_SPEED_MIN
                     + randFloat() * (CO_SPEED_MAX - CO_SPEED_MIN);
                c.m1 = CO_MASS_MIN
                     + randFloat() * (CO_MASS_MAX - CO_MASS_MIN);
                c.m2 = CO_MASS_MIN
                     + randFloat() * (CO_MASS_MAX - CO_MASS_MIN);
                c.w1 = CO_CRAWL_W * c.m1;
                c.w2 = CO_CRAWL_W * c.m2;
                c.p1 = -c.w1;
                c.p2 = (float)(LEDS_PER_STRIP - 1) + c.w2;
            }
            break;
        case CO_APPROACH:
            c.p1 += c.v1 * dt;
            c.p2 -= c.v2 * dt;
            if (!c.collided && c.p1 + c.w1 >= c.p2 - c.w2) {
                c.collided = true;
                c.impact = 0.5f * (c.p1 + c.p2);
                // Shatter: fragment velocities drawn in the COM frame and
                // de-meaned so total momentum m1·v1 - m2·v2 is conserved.
                float vCom = (c.m1 * c.v1 - c.m2 * c.v2) / (c.m1 + c.m2);
                float vRel = c.v1 + c.v2;
                float sv[CO_FRAGS];
                float mean = 0.0f;
                for (int k = 0; k < CO_FRAGS; k++) {
                    float r = randFloat() * 2.0f - 1.0f;
                    sv[k] = coShatterK * vRel * r * fabsf(r);
                    mean += sv[k];
                }
                mean /= (float)CO_FRAGS;
                for (int k = 0; k < CO_FRAGS; k++) {
                    CoFrag &fr = coFrag[s][k];
                    fr.pos = c.impact;
                    fr.vel = vCom + (sv[k] - mean);
                    fr.maxLife = CO_FRAG_LIFE_MIN + randFloat()
                               * (CO_FRAG_LIFE_MAX - CO_FRAG_LIFE_MIN);
                    fr.life = fr.maxLife;
                    fr.bright = 0.80f + randFloat() * 0.20f;
                }
                for (int k = 0; k < CO_SPARKLES; k++) {
                    CoSparkle &sk = coSpkl[s][k];
                    float r = randFloat() * 2.0f - 1.0f;
                    sk.pos = c.impact;
                    sk.vel = vCom + CO_SPKL_KICK * r * fabsf(r);
                    sk.maxLife = CO_SPKL_LIFE_MIN + randFloat()
                               * (CO_SPKL_LIFE_MAX - CO_SPKL_LIFE_MIN);
                    sk.life = sk.maxLife;
                }
            }
            if (c.collided && c.p1 >= c.impact + c.w1
                && c.p2 <= c.impact - c.w2) {
                c.phase = CO_CALM;
                c.timer = CO_CALM_MIN
                        - CO_CALM_MEAN * logf(fmaxf(randFloat(), 1e-6f));
            }
            break;
        }

        float fragGlow[LEDS_PER_STRIP];
        memset(fragGlow, 0, sizeof(fragGlow));
        for (int k = 0; k < CO_FRAGS; k++) {
            CoFrag &fr = coFrag[s][k];
            if (fr.life <= 0.0f) continue;
            fr.life -= dt;
            fr.vel *= expf(-dt / CO_FRAG_DRAG_TC);
            fr.pos += fr.vel * dt;
            if (fr.pos < 0.0f || fr.pos > (float)(LEDS_PER_STRIP - 1)) {
                fr.life = 0.0f;
                continue;
            }
            float u = fr.life / fr.maxLife;
            float b = fr.bright * u * u;
            int i0 = (int)fr.pos;
            float frac = fr.pos - (float)i0;
            fragGlow[i0] += b * (1.0f - frac);
            if (i0 + 1 < LEDS_PER_STRIP) fragGlow[i0 + 1] += b * frac;
            if (coSpark[s][i0] < CO_TRAIL * b) {
                coSpark[s][i0]    = CO_TRAIL * b;
                coDecayArr[s][i0] = 0.88f + randFloat() * 0.04f;
            }
        }
        for (int k = 0; k < CO_SPARKLES; k++) {
            CoSparkle &sk = coSpkl[s][k];
            if (sk.life <= 0.0f) continue;
            sk.life -= dt;
            sk.vel *= expf(-dt / CO_SPKL_DRAG_TC);
            sk.pos += sk.vel * dt;
            if (sk.pos < 0.0f || sk.pos > (float)(LEDS_PER_STRIP - 1)) {
                sk.life = 0.0f;
                continue;
            }
            if (randFloat() >= CO_SPKL_DUTY) continue;
            float u = sk.life / sk.maxLife;
            fragGlow[(int)(sk.pos + 0.5f)] += u * u
                                            * (0.7f + 0.3f * randFloat());
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            coSpark[s][i] *= fastDecay(coDecayArr[s][i], dt * 30.0f);

        bool crawling = (c.phase == CO_APPROACH);
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float sp = coSpark[s][i];
            if (sp < CO_DEADBAND) sp = 0.0f;
            sp = fminf(1.0f, sp + fragGlow[i]);
            float linSp = fastGamma24(sp);
            float r = 255.0f * linSp;
            float g = (180.0f + 60.0f * sp) * linSp;
            float b = (80.0f + 120.0f * sp) * linSp;
            if (crawling) {
                float x = (float)i;
                float b1 = (!c.collided || x <= c.impact)
                         ? coBlobBright(x - c.p1, c.w1) : 0.0f;
                float b2 = (!c.collided || x >= c.impact)
                         ? coBlobBright(x - c.p2, c.w2) : 0.0f;
                float linB = fastGamma24(fminf(1.0f, b1 + b2));
                r += 255.0f * linB;
                g += 240.0f * linB;
                b += 200.0f * linB;
            }
            setPixelScaled(s, i, fminf(255.0f, r), fminf(255.0f, g),
                           fminf(255.0f, b));
        }
    }
}

static void renderOff() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            setPixel(s, i, 0, 0, 0);
}

// ── Effect reset dispatch (called on effect change) ─────────────
static void resetEffect(uint8_t fx) {
    resetFxDither();
    switch (fx) {
        case FX_AMBIENT_BLOOM:
            for (uint8_t s = 0; s < NUM_STRIPS; s++) resetBloomStrip(bloom[s]);
            break;
        case FX_FIRE:
            resetFire();
            break;
        case FX_HEAT_DIFFUSION:
            resetHeatDiffusion();
            break;
        case FX_SPARKLE_SYLLABLE:
            resetSparkle();
            break;
        case FX_CREATURES:
            resetCreatures();
            break;
        case FX_LEAF_WIND:
            resetLeafWind();
            break;
        case FX_LIGHT_THROUGH:
            resetLightThrough();
            break;
        case FX_NEBULA:
            resetNebula();
            break;
        case FX_GALAXY:
            resetGalaxy();
            break;
        case FX_RAINBOW:
            resetRainbow();
            break;
        case FX_BOUQUET:
            resetBouquet();
            break;
        case FX_COLLISION:
            resetCollision();
            break;
        default:
            break;
    }
}

// ── Runtime parameter table ──────────────────────────────────────
struct Param {
    const char *name;
    enum Type { F32 } type;
    void *ptr;
    float lo, hi;
};

static const Param PARAMS[] = {
    { "BRIGHTNESS_CAP",   Param::F32, &bloomBrightnessCap,  0.05f, 2.0f  },
    { "GLOBAL_BRIGHTNESS",Param::F32, &globalBrightness,    0.0f,  1.0f  },
    { "SPEED_SCALE",      Param::F32, &speedScale,          0.2f,  3.0f  },
    { "BUFFER_DRAIN",     Param::F32, &bloomBufferDrain,    0.5f,  20.0f },
    { "FLASH_DECAY_RATE", Param::F32, &bloomFlashDecayRate, 0.2f,  10.0f },
    { "BREATH_FLOOR",     Param::F32, &bloomBreathFloor,    0.0f,  0.5f  },
    { "DORMANCY_MIN",     Param::F32, &bloomDormancyMin,   0.0f,  60.0f },
    { "DORMANCY_MAX",     Param::F32, &bloomDormancyMax,   0.0f,  60.0f },
    { "DORMANCY_FRAC",    Param::F32, &bloomDormancyFrac,  0.0f,  1.0f  },
    { "ONSET_REFRAC",     Param::F32, &onsetRefrac,        0.02f, 1.0f  },
    { "ONSET_RISEFRAC",   Param::F32, &onsetRiseFrac,      0.05f, 4.0f  },
    { "STRETCH_GAIN",     Param::F32, &bloomStretchGain,   0.0f,  6.0f  },
    { "AUDIO_CEIL",       Param::F32, &bloomAudioCeil,     1.0f,  4.0f  },
    { "ENERGY5S_TAU",     Param::F32, &energy5sTau,        0.5f,  20.0f },
    { "ENERGY_FORCE",     Param::F32, &energyForce,       -1.0f,  1.0f  },
    { "HEAT_INJ_RATE",    Param::F32, &heatInjRate,        1.0f,  60.0f },
    { "FLOOR_MIN",        Param::F32, &snsRmsFloorMin,  2000.0f, 80000.0f },
    { "HEAT_IDLE_FLOOR",  Param::F32, &heatIdleFloor,      0.0f,  1.0f   },
    { "CO_SHATTER_K",     Param::F32, &coShatterK,         0.2f,  4.0f  },
};
static const size_t PARAM_COUNT = sizeof(PARAMS) / sizeof(PARAMS[0]);

static void dumpParam(const Param &p) {
    Serial.printf("[PARAM] %s=%.4f\n", p.name, *(float*)p.ptr);
}

static void dumpAllParams() {
    for (size_t i = 0; i < PARAM_COUNT; i++) dumpParam(PARAMS[i]);
}

static void setParamFromLine(const char *kv) {
    while (*kv == ' ') kv++;
    const char *eq = strchr(kv, '=');
    if (!eq) { Serial.println("[PARAM] bad syntax"); return; }
    size_t nameLen = (size_t)(eq - kv);
    const char *valStr = eq + 1;
    for (size_t i = 0; i < PARAM_COUNT; i++) {
        const Param &p = PARAMS[i];
        if (strlen(p.name) == nameLen && strncmp(p.name, kv, nameLen) == 0) {
            float v = atof(valStr);
            if (v < p.lo) v = p.lo;
            if (v > p.hi) v = p.hi;
            *(float*)p.ptr = v;
            dumpParam(p);
            return;
        }
    }
    Serial.printf("[PARAM] unknown key\n");
}

static void processLine(const char *line, uint8_t len) {
    if (len >= 2 && line[0] == '!' && line[1] == 'P') {
        if (len >= 3 && line[2] == '?') { dumpAllParams(); return; }
        if (len >= 4) { setParamFromLine(line + 2); return; }
    }
}

static char serialBuf[80];
static uint8_t serialBufLen = 0;

static void parseSerialCommands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBufLen > 0) {
                serialBuf[serialBufLen] = '\0';
                processLine(serialBuf, serialBufLen);
                serialBufLen = 0;
            }
        } else if (c >= 32 && c < 127 && serialBufLen < sizeof(serialBuf) - 1) {
            serialBuf[serialBufLen++] = c;
        }
    }
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    strip0.Begin(); strip1.Begin(); strip2.Begin(); strip3.Begin();
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0));
    strip2.ClearTo(RgbColor(0)); strip3.ClearTo(RgbColor(0));
    showAll();

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        resetBloomStrip(bloom[s]);
    }

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

    // Some core/IDF builds reset the WiFi channel during esp_now_init — re-assert
    // it so RX stays on the sender's channel.
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

    // Register the broadcast peer so broadcast frames from the BS-26 sender are
    // accepted (mirrors the sender's broadcast peer setup).
    static const uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BROADCAST_ADDR, 6);
    peer.channel = FIXED_CHANNEL;
    peer.encrypt = false;
    esp_now_add_peer(&peer);


    Serial.printf("Tree-of-record (led-eth-0, 4-strip) ready — ch=%u\n", FIXED_CHANNEL);
    Serial.printf("  4 strips × %u LEDs on GPIO 13/27/32/33 (BS-26 driven)\n", LEDS_PER_STRIP);
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    gDt = dt;
    lastRenderMs = now;

    // Decode sensor packets → effect feature globals (real-time dt).
    processSensors(dt);

    // ── Map BS-26 knobs to effect parameters ────────────────────
    // BS-26 live → knobs drive the effect. No signal → serial PARAMS.
    bool bs26Live = (bs26LastMs > 0 && (int32_t)(now - bs26LastMs) < (int32_t)HEARTBEAT_TIMEOUT_MS);
    bool bs26EverSeen = (bs26LastMs > 0);
    // Last knob-derived brightness/speed, held across brief packet gaps so a
    // dropout doesn't snap brightness to the serial-param default. The other
    // knob params (hue/sat/decouter/effect) are persistent globals and
    // simply retain their last value when a frame is skipped.
    static float bs26LastBrightness = 1.0f;
    static float bs26LastSpeed      = 1.0f;

    float effBrightness = globalBrightness;
    float effSpeed      = speedScale;
    if (bs26Live) {
        Bs26State s;
        portENTER_CRITICAL(&bs26Mux);
        memcpy(&s, (const void*)&bs26, sizeof(s));
        portEXIT_CRITICAL(&bs26Mux);

        // tens (10-position knob) → brightness; 0 = fully off
        uint8_t tens = s.tens > 9 ? 9 : s.tens;
        if (tens == 0) {
            effBrightness = 0.0f;
        } else {
            effBrightness = powf((float)tens / 9.0f, 2.5f);
        }

        // ones (0–11) → speed. Fine log resolution 0.2→1.0 across positions 0–8
        // (ratio ≈1.22/step), then compressed octave doublings 2/4/8 at the top.
        static const float ONES_SPEED[12] = {
            0.20f, 0.24f, 0.30f, 0.37f, 0.45f, 0.55f, 0.67f, 0.82f, 1.00f,
            2.00f, 4.00f, 8.00f
        };
        uint8_t spd = s.ones > 11 ? 11 : s.ones;
        effSpeed = ONES_SPEED[spd];

        // decmid (0–9) → hue rotation, decinner (0–9) → saturation
        uint8_t hue = s.decmid > 9 ? 9 : s.decmid;
        uint8_t sat = s.decinner > 9 ? 9 : s.decinner;
        if (hue != paletteHueIdx || sat != paletteSatIdx) {
            paletteHueIdx = hue;
            paletteSatIdx = sat;
            applyPalette(paletteHueIdx, paletteSatIdx);
        }

        // decouter (0–11) → effect select
        currentEffect = s.decouter >= FX_COUNT ? FX_COUNT - 1 : s.decouter;

        bs26LastBrightness = effBrightness;
        bs26LastSpeed      = effSpeed;
    } else if (bs26EverSeen) {
        // Brief packet gap — hold the last knob-derived values rather than
        // reverting to the serial-param defaults (which would spike brightness).
        effBrightness = bs26LastBrightness;
        effSpeed      = bs26LastSpeed;
    }

    bloomDormancyFrac = 0.0f;

    renderBrightness = effBrightness;
    renderFloor = effBrightness * 0.2f;

    // tens=0 → all off (blank and skip rendering)
    if (effBrightness == 0.0f) {
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
                setPixel(s, i, 0, 0, 0);
        showAll();
        return;
    }

    // Reset effect state on change, but debounce so spinning the ones knob
    // through intermediate slots (e.g. gravity, whose reset does a white seed
    // flash) doesn't fire their resets in passing. Only the settled selection
    // resets.
    static uint8_t pendingEffect = 0xFF;
    static uint32_t effectSettleMs = 0;
    if (currentEffect != pendingEffect) {
        pendingEffect = currentEffect;
        effectSettleMs = now;
    }
    if (currentEffect != prevEffect &&
        (prevEffect == 0xFF || now - effectSettleMs >= 250)) {
        resetEffect(currentEffect);
        prevEffect = currentEffect;
    }

    uint32_t t0 = micros();

    // ── Render the selected effect on all 6 strips ──────────────
    // Per-effect speed normalization so the ones knob feels consistent.
    static const float SPEED_SCALE[] = {
        1.4f,   // FX_AMBIENT_BLOOM
        1.0f,   // FX_FIRE
        1.0f,   // FX_HEAT_DIFFUSION
        1.0f,   // FX_SPARKLE_SYLLABLE
        1.5f,   // FX_CREATURES
        0.25f,  // FX_LEAF_WIND
        1.0f,   // FX_LIGHT_THROUGH
        1.5f,   // FX_NEBULA
        1.0f,   // FX_GALAXY
        2.0f,   // FX_RAINBOW
        1.0f,   // FX_BOUQUET
        1.0f,   // FX_COLLISION
    };
    float fxDt = dt * effSpeed * SPEED_SCALE[currentEffect];
    switch (currentEffect) {
        case FX_AMBIENT_BLOOM:
            for (uint8_t s = 0; s < NUM_STRIPS; s++) renderBloomStrip(s, fxDt);
            break;
        case FX_FIRE:
            renderFire(fxDt);
            break;
        case FX_HEAT_DIFFUSION:
            renderHeatDiffusion(fxDt);
            break;
        case FX_SPARKLE_SYLLABLE:
            renderSparkle(fxDt);
            break;
        case FX_CREATURES:
            renderCreatures(fxDt);
            break;
        case FX_LEAF_WIND:
            renderLeafWind(fxDt);
            break;
        case FX_LIGHT_THROUGH:
            renderLightThrough(fxDt);
            break;
        case FX_NEBULA:
            renderNebula(fxDt);
            break;
        case FX_GALAXY:
            renderGalaxy(fxDt);
            break;
        case FX_RAINBOW:
            renderRainbow(fxDt);
            break;
        case FX_BOUQUET:
            renderBouquet(fxDt);
            break;
        case FX_COLLISION:
            renderCollision(fxDt);
            break;
        default:
            renderOff();
            break;
    }

    uint32_t t1 = micros();
    showAll();
    uint32_t t2 = micros();

    static uint32_t renderUsAccum = 0, showUsAccum = 0;
    renderUsAccum += (t1 - t0);
    showUsAccum   += (t2 - t1);

    parseSerialCommands();

    // Status logging
    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    if (now - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (now - lastLogMs);
        float avgRenderUs = (float)renderUsAccum / frameCount;
        float avgShowUs   = (float)showUsAccum / frameCount;
        Serial.printf("  FPS=%.1f  render=%.0fus  show=%.0fus  total=%.0fus\n",
                      fps, avgRenderUs, avgShowUs, avgRenderUs + avgShowUs);
        RgbColor px = strip0.GetPixelColor(50);
        Serial.printf("  [px50] R=%u G=%u B=%u\n", px.R, px.G, px.B);
        {
            // Field span + lit count on strip 0 (per-pixel max channel), for
            // any effect — objective "is anything visible" readout.
            uint8_t bMin = 255, bMax = 0;
            uint16_t lit = 0;
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                RgbColor c = strip0.GetPixelColor(i);
                uint8_t m = c.R; if (c.G > m) m = c.G; if (c.B > m) m = c.B;
                if (m < bMin) bMin = m;
                if (m > bMax) bMax = m;
                if (m > 0) lit++;
            }
            float eDbg = (energyForce >= 0.0f) ? energyForce : audioEnergy5s;
            Serial.printf("  [span] %u..%u lit=%u/%u e5s=%.3f force=%.2f\n",
                          bMin, bMax, lit, (unsigned)LEDS_PER_STRIP,
                          audioEnergy5s, eDbg);
        }
        Serial.printf("  [bs26] %s pkts=%lu fx=%u bright=%.3f speed=%.2f hue=%u sat=%u dorm=%.2f\n",
                      bs26Live ? "LIVE" : "----",
                      (unsigned long)bs26PktCount,
                      currentEffect, renderBrightness, effSpeed,
                      paletteHueIdx, paletteSatIdx, bloomDormancyFrac);
        bool aLive = accelLastMs && (int32_t)(now - accelLastMs) < (int32_t)SENSOR_TIMEOUT_MS;
        bool gLive = gyroLastMs  && (int32_t)(now - gyroLastMs)  < (int32_t)SENSOR_TIMEOUT_MS;
        bool auLive = audioLastMs && (int32_t)(now - audioLastMs) < (int32_t)SENSOR_TIMEOUT_MS;
        Serial.printf("  [sns] accel=%s(%lu) gyro=%s(%lu) audio=%s(%lu) "
                      "energy=%.3f onset=%.3f tilt=%.2f@%.0f crExcite=%.2f\n",
                      aLive ? "L" : "-", (unsigned long)accelPktCount,
                      gLive ? "L" : "-", (unsigned long)gyroPktCount,
                      auLive ? "L" : "-", (unsigned long)audioPktCount,
                      fxEnergy, fxOnset, fxTiltBlend, fxAngleDeg,
                      crShakeLevel);
        frameCount = 0;
        renderUsAccum = 0;
        showUsAccum = 0;
        lastLogMs = now;
    }
}
