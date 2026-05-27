/*
 * BULB-FLEET — phone WiFi UDP architecture (ported from road-bulbs).
 *
 * Phone (Sensor Logger app) → WiFi UDP → ESP32 listener. No ESP-NOW.
 * Connects to phone hotspot SSID "cuteplant".
 *
 * 6× WS2812B RGB strips on a classic ESP32. Same physical board as
 * biolum/. Each strip gets a per-strip OKLCH hue offset so the 6 strips
 * fan out a rainbow chord during tilt.
 *
 *   strip0: GPIO 4   RMT0     strip3: GPIO 5   RMT3
 *   strip1: GPIO 16  RMT1     strip4: GPIO 18  RMT4
 *   strip2: GPIO 17  RMT2     strip5: GPIO 19  RMT5
 *
 * No sacrificial LED — level-shifter glitch fix verified.
 *
 * UDP ports: 4210 sensor, 4211 cmd, 4212 discovery, 4213 onset.
 * Effects ported from road-bulbs/src/receiver.cpp.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <AsyncUDP.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <esp_random.h>
#include <esp_system.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <oklch_lut.h>
#include <fast_math.h>

// ── WiFi credentials (phone hotspot) ────────────────────────────
#ifndef WIFI_SSID
#define WIFI_SSID "cuteplant"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS "bigboiredwood"
#endif

static AsyncUDP udp;
static AsyncUDP cmdUdp;
static AsyncUDP discoverUdp;
static AsyncUDP onsetUdp;
#define UDP_PORT 4210
#define CMD_PORT 4211
#define DISCOVER_PORT 4212
#define ONSET_PORT 4213
static portMUX_TYPE pktMux = portMUX_INITIALIZER_UNLOCKED;

// ── Topology ────────────────────────────────────────────────────
#define NUM_STRIPS       6
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP   100
#endif
#define LED_COUNT        LEDS_PER_STRIP  // per-strip effect math uses this
#define HEAD_OFFSET      0

// Per-strip OKLCH hue offsets (~60° apart on 256-step wheel) — gives the
// 6 strips simultaneous hues that span a rainbow chord during tilt.
static const uint8_t STRIP_HUE_OFFSET[NUM_STRIPS] = {
    0, 42, 85, 128, 170, 213
};

// ── Global sliders (set via UDP cmd) ────────────────────────────
static float globalBrightness  = 0.50f;  // 0.0–1.0, slider is the single brightness control
static float globalSensitivity = 1.0f;   // 0.05–2.0, scales RMS_CEILING
static float globalSpeed       = 1.0f;   // 0–2x multiplier, midpoint slider = 1.0

static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP,  4);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, 16);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, 17);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP,  5);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, 18);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, 19);

// Write pixel i on strip s.
static inline void stripSetPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    float br = globalBrightness;
    if (br != 1.0f) {
        r = (uint8_t)fminf(r * br, 255.0f);
        g = (uint8_t)fminf(g * br, 255.0f);
        b = (uint8_t)fminf(b * br, 255.0f);
    }
    RgbColor c(r, g, b);
    uint16_t pi = i + HEAD_OFFSET;
    if (pi >= LEDS_PER_STRIP) return;
    switch (s) {
        case 0: strip0.SetPixelColor(pi, c); break;
        case 1: strip1.SetPixelColor(pi, c); break;
        case 2: strip2.SetPixelColor(pi, c); break;
        case 3: strip3.SetPixelColor(pi, c); break;
        case 4: strip4.SetPixelColor(pi, c); break;
        case 5: strip5.SetPixelColor(pi, c); break;
    }
}

static inline void stripsBegin() {
    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
}

static inline void stripsShow() {
    // Stagger in two batches of 3 to avoid RMT ISR contention.
    // 6 channels refilling simultaneously can miss ISR deadlines
    // when WiFi interrupts land, causing all-white glitch frames.
    strip0.Show(); strip1.Show(); strip2.Show();
    while (!strip0.CanShow() || !strip1.CanShow() || !strip2.CanShow()) {}
    strip3.Show(); strip4.Show(); strip5.Show();
}

static inline void stripsClear() {
    RgbColor black(0, 0, 0);
    strip0.ClearTo(black); strip1.ClearTo(black); strip2.ClearTo(black);
    strip3.ClearTo(black); strip4.ClearTo(black); strip5.ClearTo(black);
}

// Fill logical pixels of all strips with the same color (used for boot flashes).
static inline void fillAll(uint8_t r, uint8_t g, uint8_t b) {
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            stripSetPixel(s, i, r, g, b);
}

// ── Rendering ────────────────────────────────────────────────────
#define GAMMA 2.4f
// RGB-only — bumped from road-bulbs RGBW values since W headroom is gone.
#define BRIGHTNESS_CAP          1.0f
#define SPARKLE_BRIGHTNESS_CAP  1.0f

// ── Color / tilt mapping ─────────────────────────────────────────
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f
#define BLEND_RANGE_DEG 40.0f
#define SENSOR_HZ       25.0f

// ── FixedRangeRMS parameters ─────────────────────────────────────
#define RMS_CEILING     5000.0f

// ── Adaptive floor ───────────────────────────────────────────────
#define MIC_NOISE_FLOOR_MIN    50.0f
#define FLOOR_LEAK_RATE        0.005f
#define FLOOR_SNAP_EPSILON     0.05f
#define FLOOR_SNAP_CONSECUTIVE 3
#define FLOOR_SNAP_MIN_RATIO   0.4f
#define FLOOR_SOFT_SIGMA       0.6f
#define FLOOR_LONG_DRIFT       0.001f
#define FLOOR_HEADROOM         1.4f

// ── Sparkle ──────────────────────────────────────────────────────
#define SPARKLE_DEADBAND 0.08f

// ── Fire ─────────────────────────────────────────────────────────
#define FIRE_FLICKER_SCALE  3.0f
#define FIRE_DEADBAND       0.08f
#define FIRE_DROPOUT_DEPTH  0.85f

// ── Quiet bloom (RGB — no W channel) ─────────────────────────────
#define BLOOM_BRIGHTNESS_CAP   1.0f

#define SURPRISE_RATIO         3.0f
#define DRAIN_SCALE            100.0f
#define DRAIN_ENVELOPE_DECAY   0.85f
#define FLASH_MOTION_SCALE     300.0f
#define ENERGY_MULTIPLIER      1.4f
#define MOTION_SETTLE_MS       300

#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.65f
#define BLOOM_BREATH_MAX_PEAK   1.00f
#define BLOOM_BREATH_FLOOR      0.15f

#define BLOOM_FLASH_DECAY_LO   0.96f
#define BLOOM_FLASH_DECAY_HI   0.985f

#define BLOOM_RECOVERY_RAMP    0.033f
#define BLOOM_RECOVERY_SPREAD  0.70f

#define BLOOM_HUE_A_G   20.0f
#define BLOOM_HUE_A_B  100.0f
#define BLOOM_HUE_B_G   70.0f
#define BLOOM_HUE_B_B  110.0f
#define BLOOM_FLASH_G  150.0f
#define BLOOM_FLASH_B  170.0f
#define BLOOM_W_ONSET    0.5f

// ── Timeouts ─────────────────────────────────────────────────────
#define TIMEOUT_MS     500


struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    uint16_t rawRms;
    uint8_t micEnabled;
};  // 15 bytes

enum Algorithm {
    ALG_OFF,
    ALG_SPARKLE_BURST,        // kept in enum for cmd compat; renders syllable
    ALG_FIRE_MELD,
    ALG_FIRE_FLICKER,
    ALG_QUIET_BLOOM,
    ALG_GRAVITY_PARTICLE,
    ALG_SPARKLE_SYLLABLE,
    ALG_RAINBOW,
    ALG_NEBULA,
};

static Algorithm currentAlg = ALG_GRAVITY_PARTICLE;

// ── Gravity sparkle ──────────────────────────────────────────────
#define GS_PARTICLE_COUNT     7
#define GS_GRAVITY_SCALE      40.0f
#define GS_VELOCITY_DAMP      0.92f
#define GS_BOUNCE_REBOUND     0.5f
#define GS_SPLAT_RADIUS       2.5f
#define GS_BRIGHTNESS_CAP     1.0f

struct GsParticle {
    float pos;
    float vel;
    float bright;
    float hue;
};
// Per-strip particles so each strip animates independently with its own
// hue palette (offset by STRIP_HUE_OFFSET in hue space).
static GsParticle gsParticles[NUM_STRIPS][GS_PARTICLE_COUNT];

// ── Global state ─────────────────────────────────────────────────

static volatile uint32_t lastPacketMs = 0;
static volatile uint32_t pktCount = 0;
static SensorPacket latestPacket = {0, 0, 16384, 0, 0, 0, 0, 1};

// Calibration
static float restAx = 0, restAy = 0, restAz = 1.0f;
static bool calibrated = false;
static float calSumAx = 0, calSumAy = 0, calSumAz = 0;
static uint32_t calSamples = 0;
static uint32_t calStartMs = 0;
#define CAL_DURATION_MS 2000

// FixedRangeRMS
static float frrCeiling = RMS_CEILING;
static float energy = 0.0f;

// Adaptive floor state
static float adaptiveFloor = 0.0f;
static float longMin = 0.0f;
static int belowFloorCount = 0;

static float prevRms = 0.0f;
static float deltaPeak = 1e-6f;
static float onset = 0.0f;

// Syllable onset from phone (port 4213)
static volatile uint8_t syllOnsetStrength = 0;
static volatile uint32_t syllOnsetMs = 0;

// Fire — per-strip time-evolution so strips don't lockstep flicker.
static float fireTime[NUM_STRIPS];
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

// Bloom — per-strip, per-LED.
static float bloomBreathPhase [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomBreathPeriod[NUM_STRIPS][LEDS_PER_STRIP];
static float bloomBreathPeak  [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomHueT        [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomFlashGlow   [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomFlashDecay  [NUM_STRIPS][LEDS_PER_STRIP];
static float bloomColonyEnergy = 1.0f;
static uint32_t bloomLastMotionMs = 0;
static float bloomMotionRate = 0.0f;
static float motionEMA = 0.0f;
static float drainEnvelope = 0.0f;
static float bloomHitIntensity = 0.0f;
static uint32_t bloomPrevPktCount = 0;

static uint32_t prngState;

static uint8_t engageHueOffset = 0;

// ── MAC for identification ───────────────────────────────────────
static char macStr[18] = {0};
static void initMacStr() {
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

// ── Helpers ──────────────────────────────────────────────────────

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float vecLen(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}

static void vecNormalize(float &x, float &y, float &z) {
    float len = vecLen(x, y, z);
    if (len > 0) { x /= len; y /= len; z /= len; }
}

static inline uint32_t xorshift32() {
    prngState ^= prngState << 13;
    prngState ^= prngState >> 17;
    prngState ^= prngState << 5;
    return prngState;
}

static inline float randFloat() {
    return (float)(xorshift32() & 0xFFFFFF) / 16777216.0f;
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

static float computeGyroRate(int16_t gx, int16_t gy, int16_t gz) {
    float fx = (float)gx, fy = (float)gy, fz = (float)gz;
    return sqrtf(fx * fx + fy * fy + fz * fz) / 131.0f;
}

static float computeAccelJolt(int16_t ax, int16_t ay, int16_t az) {
    float fx = (float)ax, fy = (float)ay, fz = (float)az;
    float mag = sqrtf(fx * fx + fy * fy + fz * fz);
    return fabsf(mag - 16384.0f) / 16384.0f;
}

static float updateAdaptiveFloor(float rms, float dt) {
    if (adaptiveFloor < 1.0f) {
        adaptiveFloor = fmaxf(rms, MIC_NOISE_FLOOR_MIN);
        longMin = adaptiveFloor;
        return adaptiveFloor;
    }

    if (rms < longMin) {
        longMin = fmaxf(rms, MIC_NOISE_FLOOR_MIN);
    } else {
        longMin *= (1.0f + FLOOR_LONG_DRIFT * dt);
    }

    if (rms < adaptiveFloor * (1.0f + FLOOR_SNAP_EPSILON)) {
        belowFloorCount++;
        if (belowFloorCount >= FLOOR_SNAP_CONSECUTIVE) {
            float target = fmaxf(rms, FLOOR_SNAP_MIN_RATIO * longMin);
            target = fmaxf(target, MIC_NOISE_FLOOR_MIN);
            float snapAlpha = fminf(1.0f, dt / 0.11f);
            adaptiveFloor += snapAlpha * (target - adaptiveFloor);
        }
    } else {
        belowFloorCount = 0;
        float ratio = rms / fmaxf(adaptiveFloor, 1.0f);
        float d = (ratio - 1.0f) / FLOOR_SOFT_SIGMA;
        float weight = expf(-(d * d));
        adaptiveFloor *= (1.0f + FLOOR_LEAK_RATE * dt * weight);
    }

    adaptiveFloor = fmaxf(adaptiveFloor, MIC_NOISE_FLOOR_MIN);
    return adaptiveFloor;
}

static float computeEnergy(uint16_t rawRms, float dt) {
    float rms = (float)rawRms;
    float scaledCeiling = RMS_CEILING / globalSensitivity;
    frrCeiling = fmaxf(scaledCeiling, frrCeiling * expf(-0.0025f * dt));
    if (rms > frrCeiling) frrCeiling = rms;
    updateAdaptiveFloor(rms, dt);
    float effectiveFloor = adaptiveFloor * FLOOR_HEADROOM;
    if (rms < effectiveFloor) return 0.0f;
    float db = 20.0f * log10f(rms / effectiveFloor);
    float dbRange = 20.0f * log10f(frrCeiling / effectiveFloor);
    if (dbRange < 1.0f) dbRange = 1.0f;
    return clampf(db / dbRange, 0.0f, 1.0f);
}

static float computeOnset(uint16_t rawRms, float dt) {
    float rms = (float)rawRms;
    float delta = fabsf(rms - prevRms);
    prevRms = rms;
    float decay = expf(-1.3f * dt);
    deltaPeak = fmaxf(delta, deltaPeak * decay);
    return (deltaPeak > 1e-6f) ? (delta / deltaPeak) : 0.0f;
}

// Motion-derived energy/onset (substitute when mic is off)
static float motionRms = 0.0f;
static float motionRmsEma = 0.0f;
static float prevMotionRms = 0.0f;
static float motionDeltaPeak = 0.0f;

static void computeMotionEnergy(int16_t ax, int16_t ay, int16_t az,
                                 int16_t gx, int16_t gy, int16_t gz,
                                 float dt,
                                 float &outEnergy, float &outOnset) {
    float gyroRate = computeGyroRate(gx, gy, gz);
    float accelJolt = computeAccelJolt(ax, ay, az);
    float raw = fmaxf(gyroRate * 200.0f, accelJolt * 40000.0f);
    motionRms = raw;
    float emaAlpha = (raw > motionRmsEma) ? fminf(1.0f, dt / 0.11f)
                                          : fminf(1.0f, dt / 0.77f);
    motionRmsEma += emaAlpha * (raw - motionRmsEma);
    float ceiling = fmaxf(10000.0f, motionRmsEma * 3.0f);
    outEnergy = clampf(raw / ceiling, 0.0f, 1.0f);
    float delta = fabsf(raw - prevMotionRms);
    prevMotionRms = raw;
    motionDeltaPeak = fmaxf(delta, motionDeltaPeak * expf(-0.51f * dt));
    outOnset = (motionDeltaPeak > 1e-6f) ? (delta / motionDeltaPeak) : 0.0f;
}

static void resetBloomState();
static void resetFireState();
static void resetGravitySparkle();
static void resetSyllableState();
void startCalibration();
void handleSerialCommand(char c);
static void handleSliderCommand(const uint8_t *data, size_t len);

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    initMacStr();

    stripsBegin();

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    resetSyllableState();
    resetFireState();
    resetBloomState();
    resetGravitySparkle();

    // Boot: green while connecting WiFi.
    fillAll(0, 40, 0);
    stripsShow();

    Serial.printf("[BOOT] Connecting to WiFi: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    WiFi.setAutoReconnect(true);
    uint32_t wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 30000) {
        delay(500);
        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WIFI] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
        fillAll(0, 0, 40);
        stripsShow();
        delay(300);
    } else {
        Serial.println("\n[WIFI] FAILED — running in idle-only mode");
        fillAll(40, 0, 0);
        stripsShow();
        delay(500);
    }

    if (udp.listen(UDP_PORT)) {
        udp.onPacket([](AsyncUDPPacket packet) {
            if (packet.length() != sizeof(SensorPacket)) return;
            uint32_t now = millis();
            portENTER_CRITICAL(&pktMux);
            memcpy((void*)&latestPacket, packet.data(), sizeof(SensorPacket));
            lastPacketMs = now;
            pktCount++;
            portEXIT_CRITICAL(&pktMux);
        });
        Serial.printf("[UDP] Listening on port %d\n", UDP_PORT);
    } else {
        Serial.println("[UDP] FAILED to bind port");
    }

    if (cmdUdp.listen(CMD_PORT)) {
        cmdUdp.onPacket([](AsyncUDPPacket packet) {
            if (packet.length() < 1) return;
            char c = (char)packet.data()[0];
            if ((c == 'B' || c == 'S' || c == 'V') && packet.length() >= 2) {
                handleSliderCommand(packet.data(), packet.length());
                packet.printf("OK:%c=%u", c, packet.data()[1]);
                return;
            }
            handleSerialCommand(c);
            const char *name = "unknown";
            switch (currentAlg) {
                case ALG_OFF:             name = "off"; break;
                case ALG_SPARKLE_BURST:   name = "sparkle"; break;
                case ALG_FIRE_MELD:       name = "fire"; break;
                case ALG_FIRE_FLICKER:    name = "flicker"; break;
                case ALG_QUIET_BLOOM:     name = "bloom"; break;
                case ALG_GRAVITY_PARTICLE: name = "gravity"; break;
                case ALG_SPARKLE_SYLLABLE: name = "syllable"; break;
                case ALG_RAINBOW:            name = "rainbow"; break;
                case ALG_NEBULA:          name = "nebula"; break;
            }
            packet.printf("FX:%s", name);
        });
        Serial.printf("[CMD] Listening on port %d\n", CMD_PORT);
    }

    if (discoverUdp.listen(DISCOVER_PORT)) {
        discoverUdp.onPacket([](AsyncUDPPacket packet) {
            if (packet.length() >= 5 && memcmp(packet.data(), "ROAD?", 5) == 0) {
                char reply[64];
                snprintf(reply, sizeof(reply), "ROAD!bulb-fleet|%s", macStr);
                packet.printf("%s", reply);
            }
        });
        Serial.printf("[DISC] Discovery on port %d\n", DISCOVER_PORT);
    }

    if (onsetUdp.listen(ONSET_PORT)) {
        onsetUdp.onPacket([](AsyncUDPPacket packet) {
            if (packet.length() == 4 &&
                packet.data()[0] == 0xAA &&
                packet.data()[3] == 0x55) {
                syllOnsetStrength = packet.data()[1];
                syllOnsetMs = millis();
            }
        });
        Serial.printf("[ONSET] Listening on port %d\n", ONSET_PORT);
    }

    if (MDNS.begin("bulb-fleet")) {
        MDNS.addService("bulb-fleet", "udp", UDP_PORT);
        Serial.println("[MDNS] bulb-fleet.local");
    }

    stripsClear();
    stripsShow();

    Serial.println("Commands: 'c' recal, 's' sparkle, 'm' fire_meld, 'f' fire_flicker, 'b' bloom, 'g' gravity_particle, 'y' syllable, '?' identify");
    Serial.printf("[BOOT] role=bulb-fleet MAC=%s fw=bulb_fleet_wifi\n", macStr);
}

// ── Calibration ──────────────────────────────────────────────────

void startCalibration() {
    calibrated = false;
    calSumAx = 0; calSumAy = 0; calSumAz = 0;
    calSamples = 0;
    calStartMs = millis();
    Serial.println("Calibrating — keep rig still for 2 seconds...");
}

void updateCalibration(float ax, float ay, float az) {
    if (calStartMs == 0) {
        startCalibration();
    }

    if (!calibrated) {
        calSumAx += ax;
        calSumAy += ay;
        calSumAz += az;
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
}

static void resetFireState() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Stagger phase per strip so flicker isn't lockstep.
        fireTime[s] = (float)s * 1.37f;
    }
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

static void resetBloomState() {
    bloomColonyEnergy = 1.0f;
    bloomLastMotionMs = 0;
    bloomMotionRate = 0.0f;
    motionEMA = 0.0f;
    drainEnvelope = 0.0f;
    bloomHitIntensity = 0.0f;
    bloomPrevPktCount = 0;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            bloomBreathPhase[s][i] = randFloat();
            bloomBreathPeriod[s][i] = BLOOM_BREATH_MIN_PERIOD
                + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
            bloomBreathPeak[s][i] = BLOOM_BREATH_MIN_PEAK
                + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
            bloomHueT[s][i] = randFloat();
            bloomFlashGlow[s][i] = 0.0f;
            bloomFlashDecay[s][i] = BLOOM_FLASH_DECAY_LO
                + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
        }
    }
}

static void resetGravitySparkle() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
            gsParticles[s][i].pos = (float)(LED_COUNT - 1) * (float)i / (float)(GS_PARTICLE_COUNT - 1);
            gsParticles[s][i].vel = 0.0f;
            gsParticles[s][i].bright = 1.0f;
            // Hue: spread within the strip + offset per strip so the 6
            // strips show different rainbow chords.
            float baseHue = 256.0f * (float)i / (float)GS_PARTICLE_COUNT;
            gsParticles[s][i].hue = fmodf(baseHue + (float)STRIP_HUE_OFFSET[s], 256.0f);
        }
    }
}

// ── Gravity sparkle render ───────────────────────────────────────

static void renderGravitySparkle(float dt) {
    float gravG = clampf((float)latestPacket.ax / 16384.0f, -1.5f, 1.5f);
    float accel = gravG * GS_GRAVITY_SCALE;
    float damp = fastDecay(GS_VELOCITY_DAMP, dt * 30.0f);

    static float accR[LED_COUNT], accG[LED_COUNT], accB[LED_COUNT];
    const float maxPos = (float)(LED_COUNT - 1);
    const float invTwoSigSq = 1.0f / (2.0f * GS_SPLAT_RADIUS * GS_SPLAT_RADIUS);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            accR[i] = 0; accG[i] = 0; accB[i] = 0;
        }

        for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
            GsParticle &p = gsParticles[s][i];

            p.vel = p.vel * damp + accel * dt;
            p.pos += p.vel * dt;

            if (p.pos < 0.0f) {
                p.pos = 0.0f;
                if (p.vel < 0.0f) p.vel = -p.vel * GS_BOUNCE_REBOUND;
            } else if (p.pos > maxPos) {
                p.pos = maxPos;
                if (p.vel > 0.0f) p.vel = -p.vel * GS_BOUNCE_REBOUND;
            }

            uint8_t hueIdx = (uint8_t)((uint32_t)p.hue & 0xFF);
            float colR = (float)oklchVarL[hueIdx][0];
            float colG = (float)oklchVarL[hueIdx][1];
            float colB = (float)oklchVarL[hueIdx][2];

            int center = (int)(p.pos + 0.5f);
            int lo = center - 3; if (lo < 0) lo = 0;
            int hi = center + 3; if (hi > (int)(LED_COUNT - 1)) hi = LED_COUNT - 1;
            for (int j = lo; j <= hi; j++) {
                float d = (float)j - p.pos;
                float w = expf(-(d * d) * invTwoSigSq);
                accR[j] += colR * w;
                accG[j] += colG * w;
                accB[j] += colB * w;
            }
        }

        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float r = accR[i];
            float g = accG[i];
            float b = accB[i];
            float maxCh = fmaxf(r, fmaxf(g, b));
            float bright = clampf(maxCh / 255.0f, 0.0f, 1.0f);
            float linBright = fastGamma24(bright) * GS_BRIGHTNESS_CAP;
            float norm = (bright > 0.001f) ? (linBright / bright) : 0.0f;
            uint8_t rr = (uint8_t)clampf(r * norm + 0.5f, 0, 255);
            uint8_t gg = (uint8_t)clampf(g * norm + 0.5f, 0, 255);
            uint8_t bb = (uint8_t)clampf(b * norm + 0.5f, 0, 255);
            stripSetPixel(s, i, rr, gg, bb);
        }
    }
}

// ── Sparkle syllable state ───────────────────────────────────────
// Per-strip, per-LED — each onset ignites independent random LEDs on each
// strip so the 6 strips have varied patterns from the same trigger.
static float syllSparkle[NUM_STRIPS][LED_COUNT];
static float syllDecay  [NUM_STRIPS][LED_COUNT];
static float syllEnvelope = 0.0f;
static float syllCooldown = 0.0f;

static void resetSyllableState() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            syllSparkle[s][i] = 0.0f;
            syllDecay[s][i]   = 0.94f;
        }
    }
    syllEnvelope = 0.0f;
    syllCooldown = 0.0f;
}

// ── Idle rainbow wash ───────────────────────────────────────────
static float idlePhase = 0.0f;
#define IDLE_BRIGHTNESS 1.0f
#define IDLE_SPEED      0.10f

static void renderIdle(float dt) {
    idlePhase = fmodf(idlePhase + IDLE_SPEED * globalSpeed * dt, 1.0f);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float stripOff = STRIP_HUE_OFFSET[s] / 256.0f;
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float pos = (float)i / (float)LED_COUNT;
            float hue = fmodf(pos + idlePhase + stripOff, 1.0f);
            uint8_t idx = (uint8_t)(hue * 255.0f);
            float bright = fastGamma24(IDLE_BRIGHTNESS);
            stripSetPixel(s, i,
                (uint8_t)(oklchVarL[idx][0] * bright),
                (uint8_t)(oklchVarL[idx][1] * bright),
                (uint8_t)(oklchVarL[idx][2] * bright));
        }
    }
}

// ── Nebula ──────────────────────────────────────────────────────
#define NEBULA_MAX_ORBS      5
#define NEBULA_ORB_TAIL      30.0f
#define NEBULA_ORB_BASE_SPEED 0.45f
#define NEBULA_SPAWN_CHANCE  0.03f
#define NEBULA_MIN_LIFETIME  200
#define NEBULA_MAX_LIFETIME  300
#define NEBULA_BRIGHTNESS    1.0f

struct NebOrb {
    float pos;
    float vel;
    uint16_t age;
    uint16_t lifetime;
    bool active;
};

static float nebulaTime = 0.0f;
static NebOrb nebOrbs[NUM_STRIPS][NEBULA_MAX_ORBS];
static float nebDecay[NUM_STRIPS][LEDS_PER_STRIP];

static void resetNebula() {
    nebulaTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++)
            nebOrbs[s][o].active = false;
        for (uint16_t i = 0; i < LED_COUNT; i++)
            nebDecay[s][i] = 0.0f;
    }
}

static void renderNebula(float dt) {
    float spd = globalSpeed * 0.3f;
    nebulaTime += dt * spd;
    float t = nebulaTime * 60.0f;

    // Decay rate: convert per-frame decay to time-based
    float decayPerFrame = 1.0f - (1.0f / NEBULA_ORB_TAIL);
    float decayPerSec = powf(decayPerFrame, 60.0f);
    float decay = powf(decayPerSec, dt);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Decay orb brightness buffer
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            nebDecay[s][i] *= decay;
            if (nebDecay[s][i] < 0.01f) nebDecay[s][i] = 0.0f;
        }

        // Spawn orbs
        float spawnRoll = (float)(xorshift32() & 0xFFFF) / 65536.0f;
        if (spawnRoll < NEBULA_SPAWN_CHANCE) {
            for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
                if (!nebOrbs[s][o].active) {
                    NebOrb &orb = nebOrbs[s][o];
                    orb.pos = randFloat() * (float)LED_COUNT;
                    float dir = (xorshift32() & 1) ? 1.0f : -1.0f;
                    orb.vel = dir * NEBULA_ORB_BASE_SPEED * spd * (0.7f + randFloat() * 0.6f);
                    orb.age = 0;
                    orb.lifetime = NEBULA_MIN_LIFETIME + (uint16_t)(xorshift32() % (NEBULA_MAX_LIFETIME - NEBULA_MIN_LIFETIME));
                    orb.active = true;
                    break;
                }
            }
        }

        // Update orbs
        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
            NebOrb &orb = nebOrbs[s][o];
            if (!orb.active) continue;

            orb.age++;
            orb.pos += orb.vel * dt * 60.0f;
            orb.pos = fmodf(orb.pos + (float)LED_COUNT, (float)LED_COUNT);

            if (orb.age >= orb.lifetime) {
                orb.active = false;
                continue;
            }

            // Smoothstep lifecycle brightness
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

            // Sub-pixel interpolation into decay buffer
            int base = (int)orb.pos;
            int next = (base + 1) % LED_COUNT;
            float frac = orb.pos - (float)base;
            float val0 = bright * 0.6f * (1.0f - frac);
            float val1 = bright * 0.6f * frac;
            if (base >= 0 && base < LED_COUNT)
                nebDecay[s][base] = fminf(1.0f, nebDecay[s][base] + val0);
            if (next >= 0 && next < LED_COUNT)
                nebDecay[s][next] = fminf(1.0f, nebDecay[s][next] + val1);
        }

        // Render
        float sOff = (float)s * 0.167f; // strip phase offset
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float pos = (float)i / (float)LED_COUNT;

            // Background: breathing wave
            float breathing = 51.0f + 38.0f * fastSin((t * 0.0105f));
            float phase = pos + sOff + t * 0.006f;
            float spatial = 51.0f * (0.5f + 0.5f * fastSin(phase * 6.2832f));
            float bgBright = clampf(breathing + spatial, 0.0f, 153.0f) / 255.0f;

            // Color variation: blue to magenta
            float colorPhase = pos * 6.2832f + sOff * 6.2832f + t * 0.009f;
            float colorShift = 0.5f + 0.5f * fastSin(colorPhase);

            float bgR = (20.0f + colorShift * 235.0f) / 255.0f;
            float bgG = (30.0f - colorShift * 20.0f) / 255.0f;
            float bgB = (255.0f - colorShift * 125.0f) / 255.0f;

            float r = bgR * bgBright;
            float g = bgG * bgBright;
            float b = bgB * bgBright;

            // Orb layer: warm white additive
            float orbB = nebDecay[s][i];
            if (orbB > 0.01f) {
                r += orbB * 1.0f;
                g += orbB * 0.94f;
                b += orbB * 0.78f;
            }

            float cap = NEBULA_BRIGHTNESS;
            stripSetPixel(s, i,
                (uint8_t)clampf(r * cap * 255.0f, 0, 255),
                (uint8_t)clampf(g * cap * 255.0f, 0, 255),
                (uint8_t)clampf(b * cap * 255.0f, 0, 255));
        }
    }
}

void handleSerialCommand(char c) {
    if (c == 'c' || c == 'C') {
        startCalibration();
    } else if (c == 's') {
        // Compat: 's' was sparkle_burst on road-bulbs. We render syllable.
        currentAlg = ALG_SPARKLE_SYLLABLE;
        resetSyllableState();
        Serial.println("Algorithm: sparkle_syllable (s alias)");
    } else if (c == 'm' || c == 'M') {
        currentAlg = ALG_FIRE_MELD;
        resetFireState();
        Serial.println("Algorithm: fire_meld");
    } else if (c == 'f' || c == 'F') {
        currentAlg = ALG_FIRE_FLICKER;
        resetFireState();
        Serial.println("Algorithm: fire_flicker");
    } else if (c == 'b' || c == 'B') {
        currentAlg = ALG_QUIET_BLOOM;
        resetBloomState();
        Serial.println("Algorithm: quiet_bloom");
    } else if (c == 'g' || c == 'G') {
        currentAlg = ALG_GRAVITY_PARTICLE;
        resetGravitySparkle();
        Serial.println("Algorithm: gravity_particle");
    } else if (c == 'y' || c == 'Y') {
        currentAlg = ALG_SPARKLE_SYLLABLE;
        resetSyllableState();
        Serial.println("Algorithm: sparkle_syllable");
    } else if (c == 'n' || c == 'N') {
        currentAlg = ALG_NEBULA;
        resetNebula();
        Serial.println("Algorithm: nebula");
    } else if (c == 'i' || c == 'I') {
        currentAlg = ALG_RAINBOW;
        idlePhase = 0.0f;
        Serial.println("Algorithm: rainbow");
    } else if (c == 'x' || c == 'X') {
        currentAlg = ALG_OFF;
        stripsClear();
        stripsShow();
        Serial.println("Algorithm: off");
    } else if (c == '?') {
        Serial.printf("[BOOT] role=bulb-fleet MAC=%s fw=bulb_fleet_wifi\n", macStr);
    }
}

static void handleSliderCommand(const uint8_t *data, size_t len) {
    if (len < 2) return;
    char cmd = (char)data[0];
    uint8_t val = data[1];
    if (cmd == 'B') {
        globalBrightness = val / 255.0f;
        Serial.printf("[SLIDER] brightness=%.0f%%\n", globalBrightness * 100.0f);
    } else if (cmd == 'S') {
        globalSensitivity = fmaxf(0.05f, val / 128.0f);
        Serial.printf("[SLIDER] sensitivity=%.2f\n", globalSensitivity);
    } else if (cmd == 'V') {
        globalSpeed = fmaxf(0.01f, val / 128.0f);
        Serial.printf("[SLIDER] speed=%.2f\n", globalSpeed);
    }
}

static void parseSerialCommands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c >= 32 && c < 127) {
            handleSerialCommand(c);
        }
    }
}

// ── Bloom motion ─────────────────────────────────────────────────

static void bloomProcessMotion(float pktDt, uint32_t now) {
    float gyroRate = computeGyroRate(
        latestPacket.gx, latestPacket.gy, latestPacket.gz);
    float accelJolt = computeAccelJolt(
        latestPacket.ax, latestPacket.ay, latestPacket.az);
    bloomMotionRate = fmaxf(gyroRate, accelJolt * 300.0f);

    float surprise = fmaxf(0.0f, bloomMotionRate - motionEMA * SURPRISE_RATIO);

    float alpha = (bloomMotionRate > motionEMA) ? fminf(1.0f, pktDt / 0.77f)
                                                : fminf(1.0f, pktDt / 0.16f);
    motionEMA += alpha * (bloomMotionRate - motionEMA);

    if (surprise > 1.0f) {
        bloomLastMotionMs = now;
        float hitIntensity = clampf(
            log2f(1.0f + surprise) / log2f(1.0f + FLASH_MOTION_SCALE),
            0.0f, 1.0f);
        if (hitIntensity > bloomHitIntensity) bloomHitIntensity = hitIntensity;

        float normMotion = surprise / DRAIN_SCALE;
        float newDrain = normMotion * normMotion * normMotion
                       * (1.0f - DRAIN_ENVELOPE_DECAY);
        if (newDrain > drainEnvelope) drainEnvelope = newDrain;
    }
}

// Quiet bloom — RGB-only. The original RGBW path drove a W channel above
// BLOOM_W_ONSET to pale-cyan the bloom. With no W, fold the wFold value
// into all three channels so brighter bloom moments shift toward white.
static void renderQuietBloom(float dt, uint32_t now) {
    bool draining = drainEnvelope > 0.001f;
    if (!draining) {
        bloomHitIntensity = 0.0f;
    }

    if (now - bloomLastMotionMs > MOTION_SETTLE_MS) {
        bloomColonyEnergy = fminf(1.0f, bloomColonyEnergy + BLOOM_RECOVERY_RAMP * dt);
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            if (!draining) {
                bloomFlashGlow[s][i] *= fastDecay(bloomFlashDecay[s][i], dt * 30.0f);
                if (bloomFlashGlow[s][i] < 0.005f) bloomFlashGlow[s][i] = 0.0f;
            }

            bloomBreathPhase[s][i] += dt / bloomBreathPeriod[s][i];
            if (bloomBreathPhase[s][i] >= 1.0f) bloomBreathPhase[s][i] -= 1.0f;

            float wakeThresh = bloomHueT[s][i] * BLOOM_RECOVERY_SPREAD;
            float ledRecovery = clampf(
                (bloomColonyEnergy - wakeThresh) / 0.30f, 0.0f, 1.0f);

            float breath = (fastSinPhase(bloomBreathPhase[s][i]) * 0.5f + 0.5f);
            float breathGlow = BLOOM_BREATH_FLOOR
                + breath * (bloomBreathPeak[s][i] - BLOOM_BREATH_FLOOR);
            breathGlow *= ledRecovery;

            if (draining) {
                float target = breathGlow * bloomHitIntensity * ENERGY_MULTIPLIER;
                if (target > bloomFlashGlow[s][i]) {
                    bloomFlashGlow[s][i] = target;
                    bloomFlashDecay[s][i] = BLOOM_FLASH_DECAY_LO
                        + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
                }
            }

            float g = fmaxf(breathGlow, bloomFlashGlow[s][i]);

            float flashFrac = (bloomFlashGlow[s][i] > breathGlow) ? 1.0f : 0.0f;
            float h = bloomHueT[s][i];
            float colG = lerpf(lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h),
                               BLOOM_FLASH_G, flashFrac);
            float colB = lerpf(lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h),
                               BLOOM_FLASH_B, flashFrac);

            float linBright = fastGamma24(g) * BLOOM_BRIGHTNESS_CAP;
            float oG = colG * linBright;
            float oB = colB * linBright;

            // W-fold for RGB: pale toward white when bloom is bright.
            float wFrac = clampf((g - BLOOM_W_ONSET) / (1.0f - BLOOM_W_ONSET),
                                 0.0f, 1.0f);
            float energyGate = clampf((bloomColonyEnergy - 0.7f) / 0.3f, 0.0f, 1.0f);
            float wGate = fmaxf(energyGate, flashFrac);
            float wFold = wFrac * wGate * linBright * 200.0f;

            float fR = wFold;
            float fG = oG + wFold;
            float fB = oB + wFold;

            stripSetPixel(s, i,
                (uint8_t)clampf(fR, 0, 255),
                (uint8_t)clampf(fG, 0, 255),
                (uint8_t)clampf(fB, 0, 255));
        }
    }

    if (draining) {
        float drain = drainEnvelope * dt;
        drain = fminf(drain, bloomColonyEnergy);
        bloomColonyEnergy -= drain;
        drainEnvelope *= expf(-4.07f * dt);
        if (drainEnvelope <= 0.001f) drainEnvelope = 0.0f;
    }
}

// ── Sparkle syllable render (RGB, per-strip) ─────────────────────
// Onset detection runs on phone (44kHz waveform), arrives as 4-byte
// async UDP packets on port 4213. ESP32 only renders. Each strip uses
// the same envelope/onset but independently shuffles which LEDs ignite,
// and applies a per-strip OKLCH hue offset to the tilt color.

static void renderSparkleSyllable(float dt, float angleDeg, float tiltBlend) {
    uint32_t now = millis();
    uint32_t onsetAge = now - syllOnsetMs;
    uint8_t strength = syllOnsetStrength;

    bool gotOnset = (syllOnsetMs > 0 && onsetAge < 100);
    float onsetNorm = gotOnset ? (strength / 255.0f) : 0.0f;

    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > syllEnvelope)
        syllEnvelope += attackAlpha * (energy - syllEnvelope);
    else
        syllEnvelope += decayAlpha * (energy - syllEnvelope);

    syllCooldown = fmaxf(0.0f, syllCooldown - dt);

    if (gotOnset && onsetNorm > 0.1f && syllCooldown <= 0.0f) {
        syllOnsetMs = 0;
        syllCooldown = 0.060f;

        int nIgnite = (int)(LED_COUNT * (0.25f + 0.25f * onsetNorm));
        float sparkVal = 0.6f + 0.4f * onsetNorm;

        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            static uint16_t indices[LED_COUNT];
            for (uint16_t i = 0; i < LED_COUNT; i++) indices[i] = i;
            for (int i = 0; i < nIgnite; i++) {
                int j = i + (int)(xorshift32() % (LED_COUNT - i));
                uint16_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
            for (int i = 0; i < nIgnite; i++) {
                syllSparkle[s][indices[i]] = sparkVal;
                syllDecay[s][indices[i]]   = 0.92f + randFloat() * 0.05f;
            }
        }
    }

    // Per-LED decay (frame-rate independent via fastDecay).
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LED_COUNT; i++)
            syllSparkle[s][i] *= fastDecay(syllDecay[s][i], dt * 30.0f);

    float base = fminf(syllEnvelope, 0.15f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Per-strip tilt color (hue offset rotates the rainbow chord).
        float tiltR = 0, tiltG = 0, tiltB = 0;
        if (tiltBlend > 0.0f) {
            float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
            if (hueFrac < 0.0f) hueFrac = 0.0f;
            if (hueFrac > 1.0f) hueFrac = 1.0f;
            uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255)
                                         + engageHueOffset
                                         + STRIP_HUE_OFFSET[s]) & 0xFF);
            tiltR = (float)oklchVarL[hueIdx][0];
            tiltG = (float)oklchVarL[hueIdx][1];
            tiltB = (float)oklchVarL[hueIdx][2];
        }

        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float sp = syllSparkle[s][i];
            float bright = base + sp * (1.0f - base);
            if (bright < SPARKLE_DEADBAND) bright = 0.0f;

            // Warm-amber base, sparkle pushes toward warm white.
            float colR = 255.0f;
            float colG = 180.0f + (240.0f - 180.0f) * sp;
            float colB =  80.0f + (200.0f -  80.0f) * sp;

            if (tiltBlend > 0.0f) {
                colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
                colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
                colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
            }

            // RGBW→RGB W-fold: the road-bulbs path drove W = 255 *
            // (1-tiltBlend). Fold half that luminance into each RGB so the
            // warm amber doesn't go dim/colored. Headroom comes from the
            // bumped SPARKLE_BRIGHTNESS_CAP.
            float wFold = 127.0f * (1.0f - tiltBlend);
            colR = fminf(255.0f, colR + wFold);
            colG = fminf(255.0f, colG + wFold);
            colB = fminf(255.0f, colB + wFold);

            float linBright = fastGamma24(bright) * SPARKLE_BRIGHTNESS_CAP;
            float fR = colR * linBright;
            float fG = colG * linBright;
            float fB = colB * linBright;

            stripSetPixel(s, i,
                (uint8_t)clampf(fR, 0, 255),
                (uint8_t)clampf(fG, 0, 255),
                (uint8_t)clampf(fB, 0, 255));
        }
    }
}

// ── Fire render (RGB, per-strip phase) ───────────────────────────
// W-fold: original road-bulbs drove pure-W floor at low brightness via
// rgbBlend gating. With no W, fold avgRGB*(1-rgbBlend) back into all
// three channels so low-brightness fire keeps its warm color.

static void renderFire(float dt, bool withDropout, float tiltBlend) {
    bool isSilent = energy < 0.001f;
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);

    float attackAlpha = fminf(1.0f, dt / 0.050f);
    float decayAlpha  = fminf(1.0f, dt / 2.0f);

    float targetBrightness;
    if (isSilent) {
        targetBrightness = 0.25f;
    } else {
        targetBrightness = fmaxf(0.25f, energy);
    }

    if (targetBrightness > fireBaseBrightness)
        fireBaseBrightness += attackAlpha * (targetBrightness - fireBaseBrightness);
    else
        fireBaseBrightness += decayAlpha * (targetBrightness - fireBaseBrightness);

    float flickerAlpha = fminf(1.0f, dt / 0.200f);
    float deltaTarget = isSilent ? 0.0f : onset;
    fireFlickerIntensity += flickerAlpha * (deltaTarget - fireFlickerIntensity);

    float dropoutAmount = 0.0f;
    if (withDropout) {
        float energyDeriv = (energy - firePrevEnergyForDeriv) / fmaxf(dt, 0.001f);
        firePrevEnergyForDeriv = energy;
        float derivAlpha = fminf(1.0f, dt / 0.200f);
        fireEnergyDerivSmooth += derivAlpha * (energyDeriv - fireEnergyDerivSmooth);

        bool isSustaining = (!isSilent && energy > 0.05f
                             && fabsf(fireEnergyDerivSmooth) <= 0.5f);
        if (isSustaining)
            fireDropoutAmount = fminf(1.0f, fireDropoutAmount + dt * 0.35f);
        else
            fireDropoutAmount = fmaxf(0.0f, fireDropoutAmount - dt * 1.0f);

        dropoutAmount = fireDropoutAmount;
    }

    float colorAttack = fminf(1.0f, dt / 0.080f);
    float colorDecay  = fminf(1.0f, dt / 2.0f);

    float colorTarget;
    if (isPercussiveOnly)
        colorTarget = 0.0f;
    else if (isSilent)
        colorTarget = 0.3f;
    else
        colorTarget = fmaxf(0.3f, energy);

    if (colorTarget > fireColorEnergy)
        fireColorEnergy += colorAttack * (colorTarget - fireColorEnergy);
    else
        fireColorEnergy += colorDecay * (colorTarget - fireColorEnergy);

    float ce = fireColorEnergy;
    float baseColR, baseColG, baseColB;

    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;

    const float amberR = 255.0f, amberG = 140.0f, amberB = 30.0f;
    const float redR   = 200.0f, redG   =  20.0f, redB   =  0.0f;
    const float whiteR = 180.0f, whiteG = 170.0f, whiteB = 160.0f;

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

    float sScl = FIRE_FLICKER_SCALE;

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        fireTime[s] = fmodf(fireTime[s] + dt, 6283.1853f);
        float t = fireTime[s];
        // Per-strip spatial offset so noise patterns don't align.
        float sOff = (float)s * 17.0f;

        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float fi = (float)i + sOff;

            float noise = fastSin(fi * 7.3f + t * 2.5f) *
                           fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

            float noiseAmp = fmaxf(0.15f * sScl, 0.10f * sScl / fmaxf(base, 0.1f));
            float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                            + fireFlickerIntensity * (noise - 0.5f) * 0.25f * sScl;

            float perLedDim = 0.0f;
            float colorRedShift = 0.0f;
            if (withDropout && dropoutAmount > 0.0f) {
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

            float linBright = fastGamma24(bright) * BRIGHTNESS_CAP;
            float oR = colR * linBright;
            float oG = colG * linBright;
            float oB = colB * linBright;

            // Direct RGB — no W channel to fold into.
            stripSetPixel(s, i,
                (uint8_t)clampf(oR, 0, 255),
                (uint8_t)clampf(oG, 0, 255),
                (uint8_t)clampf(oB, 0, 255));
        }
    }
    (void)tiltBlend;
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    parseSerialCommands();

    SensorPacket snap;
    uint32_t lastMs;
    portENTER_CRITICAL(&pktMux);
    snap   = latestPacket;
    lastMs = lastPacketMs;
    portEXIT_CRITICAL(&pktMux);

    float ax = (float)snap.ax;
    float ay = (float)snap.ay;
    float az = (float)snap.az;
    vecNormalize(ax, ay, az);
    uint16_t rawRms = snap.rawRms;
    bool micOn = snap.micEnabled != 0;
    bool connected = (lastMs > 0) && (now - lastMs < TIMEOUT_MS);

    if (connected) {
        updateCalibration(ax, ay, az);
    }

    float angleDeg = 0;
    float dot = restAx * ax + restAy * ay + restAz * az;
    if (calibrated && connected) {
        angleDeg = acosf(clampf(dot, -1.0f, 1.0f)) * (180.0f / M_PI);
    }

    if (currentAlg != ALG_QUIET_BLOOM) {
        bool gyroActive = (snap.gx != 0 || snap.gy != 0 || snap.gz != 0);
        if (connected && micOn) {
            energy = computeEnergy(rawRms, dt);
            onset = computeOnset(rawRms, dt);
        } else if (connected && !micOn && gyroActive) {
            computeMotionEnergy(
                latestPacket.ax, latestPacket.ay, latestPacket.az,
                latestPacket.gx, latestPacket.gy, latestPacket.gz,
                dt, energy, onset);
        } else {
            energy = 0.0f;
            onset = 0.0f;
        }
    }

    if (currentAlg == ALG_QUIET_BLOOM && !connected) {
        bloomHitIntensity = 0.0f;
        drainEnvelope *= expf(-4.07f * dt);
        if (drainEnvelope <= 0.001f) drainEnvelope = 0.0f;
    }

    if (currentAlg == ALG_QUIET_BLOOM && connected) {
        if (pktCount != bloomPrevPktCount) {
            static uint32_t lastBloomPktMs = 0;
            float pktDt = (lastBloomPktMs > 0)
                ? (now - lastBloomPktMs) / 1000.0f
                : (1.0f / SENSOR_HZ);
            if (pktDt > 0.2f) pktDt = 0.2f;
            lastBloomPktMs = now;
            bloomProcessMotion(pktDt, now);
            bloomPrevPktCount = pktCount;
        }
    }

    if (connected && calibrated && (pktCount % 25 == 0)) {
        const char *algName = "sparkle";
        if (currentAlg == ALG_FIRE_MELD) algName = "fire_meld";
        else if (currentAlg == ALG_FIRE_FLICKER) algName = "fire_flicker";
        else if (currentAlg == ALG_QUIET_BLOOM) algName = "bloom";
        else if (currentAlg == ALG_GRAVITY_PARTICLE) algName = "gravity";
        else if (currentAlg == ALG_SPARKLE_SYLLABLE) algName = "syllable";

        unsigned long pkts = (unsigned long)pktCount;
        if (currentAlg == ALG_QUIET_BLOOM) {
            Serial.printf("[%s] rate=%.1f energy=%.3f flash00=%.3f pkts=%lu\n",
                algName, bloomMotionRate, bloomColonyEnergy, bloomFlashGlow[0][0],
                pkts);
        } else {
            Serial.printf("[%s] angle=%.1f rms=%d energy=%.3f onset=%.3f mic=%s pkts=%lu\n",
                algName, angleDeg, rawRms, energy, onset, micOn ? "on" : "off",
                pkts);
        }
    }

    float tiltBlend = 0.0f;
    if (angleDeg > DEADZONE_DEG) {
        tiltBlend = (angleDeg - DEADZONE_DEG) / BLEND_RANGE_DEG;
        if (tiltBlend > 1.0f) tiltBlend = 1.0f;
    }

    switch (currentAlg) {
        case ALG_OFF:
            stripsClear();
            break;
        case ALG_SPARKLE_BURST:
            renderSparkleSyllable(dt, angleDeg, tiltBlend);
            break;
        case ALG_FIRE_MELD:
            renderFire(dt, false, tiltBlend);
            break;
        case ALG_FIRE_FLICKER:
            renderFire(dt, true, tiltBlend);
            break;
        case ALG_QUIET_BLOOM:
            renderQuietBloom(dt, now);
            break;
        case ALG_GRAVITY_PARTICLE:
            renderGravitySparkle(dt);
            break;
        case ALG_SPARKLE_SYLLABLE:
            renderSparkleSyllable(dt, angleDeg, tiltBlend);
            break;
        case ALG_RAINBOW:
            renderIdle(dt);
            break;
        case ALG_NEBULA:
            renderNebula(dt);
            break;
    }

    stripsShow();
    delay(1);
}
