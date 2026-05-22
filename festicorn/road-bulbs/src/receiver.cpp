/*
 * ROAD BULBS — classic ESP32 LED controller (portable rig)
 *
 * Phone (Sensor Logger app) → WiFi HTTP POST → ESP32 web server.
 * No laptop needed. ESP32 connects to phone hotspot and listens for
 * Sensor Logger JSON pushes on port 80.
 *
 * Algorithms: sparkle_burst, fire_meld, fire_flicker,
 *   quiet_bloom, gravity_particle, sparkle_syllable
 *
 * Serial commands (over USB for debug):
 *   's' sparkle_burst, 'm' fire_meld, 'f' fire_flicker,
 *   'b' quiet_bloom, 'g' gravity_particle, 'y' sparkle_syllable,
 *   'c' recalibrate, 'e' toggle force-engage, '?' identify
 *
 * Wiring: GPIO 13 → SK6812 RGBW bulbs (50 LEDs).
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <AsyncUDP.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <esp_random.h>
#include <esp_system.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include <oklch_lut.h>
#include <delta_sigma.h>
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

// ── LED config ───────────────────────────────────────────────────
#ifndef LED_PIN_OVERRIDE
#define LED_PIN    13
#else
#define LED_PIN    LED_PIN_OVERRIDE
#endif
#define LED_COUNT  50

// ── Rendering ────────────────────────────────────────────────────
#define GAMMA 2.4f
#define BRIGHTNESS_CAP 0.10f
#define SPARKLE_BRIGHTNESS_CAP 0.05f
#define PURE_W_CEIL    0.10f
#define PURE_W_BLEND   0.15f

// ── Color / tilt mapping ─────────────────────────────────────────
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f
#define BLEND_RANGE_DEG 40.0f
#define SENSOR_HZ       25.0f

// ── FixedRangeRMS parameters ─────────────────────────────────────
#define RMS_CEILING     5000.0f

// ── Adaptive floor (P4-guarded) ──────────────────────────────────
#define MIC_NOISE_FLOOR_MIN    50.0f    // physical mic floor (Galaxy S24 quiet room ~60)
#define FLOOR_LEAK_RATE        0.005f   // multiplicative upward leak per second
#define FLOOR_SNAP_EPSILON     0.05f    // ratio within which rms counts as "at floor"
#define FLOOR_SNAP_CONSECUTIVE 3        // require N consecutive below-floor before snapping
#define FLOOR_SNAP_MIN_RATIO   0.4f     // don't collapse below this × long_min (breath-pause guard)
#define FLOOR_SOFT_SIGMA       0.6f     // soft-weight gate width for upward leak
#define FLOOR_LONG_DRIFT       0.001f   // long-min upward drift per second
#define FLOOR_HEADROOM         1.4f     // signal must exceed floor × this to register as energy

// ── Sparkle ──────────────────────────────────────────────────────
#define SPARKLE_DEADBAND 0.08f

// ── Fire ─────────────────────────────────────────────────────────
#define FIRE_FLICKER_SCALE  3.0f
#define FIRE_DEADBAND       0.08f
#define FIRE_DROPOUT_DEPTH  0.85f

// ── Quiet bloom ──────────────────────────────────────────────────
#define BLOOM_BRIGHTNESS_CAP   0.25f
#define BLOOM_NOISE_GATE       256

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
    ALG_SPARKLE_BURST,
    ALG_FIRE_MELD,
    ALG_FIRE_FLICKER,
    ALG_QUIET_BLOOM,
    ALG_GRAVITY_PARTICLE,
    ALG_SPARKLE_SYLLABLE,
};

static Algorithm currentAlg = ALG_GRAVITY_PARTICLE;

// ── Gravity sparkle ──────────────────────────────────────────────
// Rainbow particles drift along the strip pulled by phone tilt. Flat
// phone → no motion; tilted → all particles slide toward the low end.
#define GS_PARTICLE_COUNT     7
#define GS_GRAVITY_SCALE      40.0f   // LEDs/s² per g of tilt
#define GS_VELOCITY_DAMP      0.92f   // applied as ^(dt*30) per frame
#define GS_BOUNCE_REBOUND     0.5f    // |v| × this on bounce
#define GS_SPLAT_RADIUS       2.5f    // LEDs; gaussian-ish σ
#define GS_BRIGHTNESS_CAP     0.45f

struct GsParticle {
    float pos;        // 0..LED_COUNT-1
    float vel;        // LEDs/s
    float bright;     // 0..1
    float hue;        // 0..256, fixed per particle
};
static GsParticle gsParticles[GS_PARTICLE_COUNT];

// ── Global state ─────────────────────────────────────────────────

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800);
static uint16_t dsR[LED_COUNT], dsG[LED_COUNT], dsB[LED_COUNT], dsW[LED_COUNT];

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
static float adaptiveFloor = 0.0f;     // 0 = uninitialized, first packet sets it
static float longMin = 0.0f;
static int belowFloorCount = 0;

static float prevRms = 0.0f;
static float deltaPeak = 1e-6f;
static float onset = 0.0f;

// Syllable onset from phone (port 4213)
static volatile uint8_t syllOnsetStrength = 0;
static volatile uint32_t syllOnsetMs = 0;

// Sparkle
static float sparkle[LED_COUNT];
static float decayRates[LED_COUNT];
static float envelope = 0.0f;
static float cooldownRemaining = 0.0f;

// Fire
static float fireTime = 0.0f;
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

// Bloom
static float bloomBreathPhase[LED_COUNT];
static float bloomBreathPeriod[LED_COUNT];
static float bloomBreathPeak[LED_COUNT];
static float bloomHueT[LED_COUNT];
static float bloomFlashGlow[LED_COUNT];
static float bloomFlashDecay[LED_COUNT];
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

// ── Helpers (forward decl for HTTP handler) ─────────────────────
static inline float clampf(float v, float lo, float hi);

// ── UDP sensor receiver ──────────────────────────────────────────
// 15-byte binary SensorPacket arrives on UDP port 4210 at ~25 Hz.
// Handler runs on the AsyncUDP task (separate from the LED render loop).

static void parseSerialCommands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c >= 32 && c < 127) {
            extern void handleSerialCommand(char c);
            handleSerialCommand(c);
        }
    }
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
    frrCeiling = fmaxf(RMS_CEILING, frrCeiling * expf(-0.0025f * dt));
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

// Motion-derived energy/onset (substitute for mic when mic is off)
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
static void resetSparkleState();
static void resetFireState();
static void resetGravitySparkle();
void startCalibration();
void handleSerialCommand(char c);

// ── Setup ────────────────────────────────────────────────────────

void setup() {
    Serial.begin(460800);
    delay(300);

    initMacStr();

    strip.begin();
    strip.setBrightness(255);

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        uint16_t seed = (uint16_t)((uint32_t)i * 256 / LED_COUNT);
        dsR[i] = seed;
        dsG[i] = (seed + 64) & 0xFF;
        dsB[i] = (seed + 128) & 0xFF;
        dsW[i] = (seed + 192) & 0xFF;
    }
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] = 0.0f;
        decayRates[i] = 0.94f;
    }
    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        decayRates[i] = 0.92f + randFloat() * 0.05f;
    }


    resetBloomState();
    resetGravitySparkle();

    // Show green while connecting WiFi
    for (uint16_t i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, 0, 40, 0, 0);
    strip.show();

    // Connect to phone hotspot
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
        // Flash blue to confirm connection
        for (uint16_t i = 0; i < LED_COUNT; i++)
            strip.setPixelColor(i, 0, 0, 40, 0);
        strip.show();
        delay(300);
    } else {
        Serial.println("\n[WIFI] FAILED — running in idle-only mode");
        // Flash red to indicate failure
        for (uint16_t i = 0; i < LED_COUNT; i++)
            strip.setPixelColor(i, 40, 0, 0, 0);
        strip.show();
        delay(500);
    }

    // Start AsyncUDP listener (runs on its own task — off the LED render loop).
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

    // Command UDP on port 4211 — single byte selects effect, replies with name
    if (cmdUdp.listen(CMD_PORT)) {
        cmdUdp.onPacket([](AsyncUDPPacket packet) {
            if (packet.length() < 1) return;
            char c = (char)packet.data()[0];
            handleSerialCommand(c);
            const char *name = "unknown";
            switch (currentAlg) {
                case ALG_SPARKLE_BURST:   name = "sparkle"; break;
                case ALG_FIRE_MELD:       name = "fire"; break;
                case ALG_FIRE_FLICKER:    name = "flicker"; break;
                case ALG_QUIET_BLOOM:     name = "bloom"; break;
                case ALG_GRAVITY_PARTICLE: name = "gravity"; break;
                case ALG_SPARKLE_SYLLABLE: name = "syllable"; break;
            }
            packet.printf("FX:%s", name);
        });
        Serial.printf("[CMD] Listening on port %d\n", CMD_PORT);
    }

    // Discovery: phone broadcasts "ROAD?" on 4212, we reply "ROAD!" + MAC
    if (discoverUdp.listen(DISCOVER_PORT)) {
        discoverUdp.onPacket([](AsyncUDPPacket packet) {
            if (packet.length() >= 5 && memcmp(packet.data(), "ROAD?", 5) == 0) {
                char reply[64];
                snprintf(reply, sizeof(reply), "ROAD!%s", macStr);
                packet.printf("%s", reply);
            }
        });
        Serial.printf("[DISC] Discovery on port %d\n", DISCOVER_PORT);
    }

    // Onset packets from phone: [0xAA, strength, band, 0x55]
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

    // mDNS
    if (MDNS.begin("road-bulbs")) {
        MDNS.addService("road-bulbs", "udp", UDP_PORT);
        Serial.println("[MDNS] road-bulbs.local");
    }

    strip.clear();
    strip.show();

    Serial.println("Commands: 'c' recal, 's' sparkle, 'm' fire_meld, 'f' fire_flicker, 'b' bloom, 'g' gravity_particle, 'e' force-engage, '?' identify");
    Serial.printf("[BOOT] role=bulbs MAC=%s fw=road_bulbs_wifi\n", macStr);
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
    fireTime = 0.0f;
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

static uint8_t sparkPeakR = 0, sparkPeakG = 0, sparkPeakB = 0, sparkPeakW = 0;
static float sparkPeakBright = 0.0f;
static float sparkPeakLin = 0.0f;

static void resetSparkleState() {
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] = 0.0f;
        decayRates[i] = 0.92f + randFloat() * 0.05f;
    }
    envelope = 0.0f;
    cooldownRemaining = 0.0f;
}

static void resetBloomState() {
    bloomColonyEnergy = 1.0f;
    bloomLastMotionMs = 0;
    bloomMotionRate = 0.0f;
    motionEMA = 0.0f;
    drainEnvelope = 0.0f;
    bloomHitIntensity = 0.0f;
    bloomPrevPktCount = 0;
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        bloomBreathPhase[i] = randFloat();
        bloomBreathPeriod[i] = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bloomBreathPeak[i] = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bloomHueT[i] = randFloat();
        bloomFlashGlow[i] = 0.0f;
        bloomFlashDecay[i] = BLOOM_FLASH_DECAY_LO
            + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
    }
}

// ── Gravity sparkle reset / spawn ────────────────────────────────

static void resetGravitySparkle() {
    for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
        gsParticles[i].pos = (float)(LED_COUNT - 1) * (float)i / (float)(GS_PARTICLE_COUNT - 1);
        gsParticles[i].vel = 0.0f;
        gsParticles[i].bright = 1.0f;
        gsParticles[i].hue = 256.0f * (float)i / (float)GS_PARTICLE_COUNT;
    }
}

// ── Gravity sparkle render ───────────────────────────────────────

static void renderGravitySparkle(float dt) {
    float gravG = clampf((float)latestPacket.ax / 16384.0f, -1.5f, 1.5f);
    float accel = gravG * GS_GRAVITY_SCALE;
    float damp = fastDecay(GS_VELOCITY_DAMP, dt * 30.0f);

    static float accR[LED_COUNT], accG[LED_COUNT], accB[LED_COUNT];
    for (uint16_t i = 0; i < LED_COUNT; i++) {
        accR[i] = 0; accG[i] = 0; accB[i] = 0;
    }

    const float maxPos = (float)(LED_COUNT - 1);
    const float invTwoSigSq = 1.0f / (2.0f * GS_SPLAT_RADIUS * GS_SPLAT_RADIUS);

    for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
        GsParticle &p = gsParticles[i];

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
        strip.setPixelColor(i, rr, gg, bb, 0);
    }
}

// ── Serial command dispatch (called from frame parser idle state) ─

void handleSerialCommand(char c) {
    if (c == 'c' || c == 'C') {
        startCalibration();
    } else if (c == 's') {
        currentAlg = ALG_SPARKLE_BURST;
        resetSparkleState();
        Serial.println("Algorithm: sparkle_burst");
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
    } else if (c == '?') {
        Serial.printf("[BOOT] role=bulbs MAC=%s fw=road_bulbs\n", macStr);
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

static void renderQuietBloom(float dt, uint32_t now) {
    bool draining = drainEnvelope > 0.001f;
    if (!draining) {
        bloomHitIntensity = 0.0f;
    }

    if (now - bloomLastMotionMs > MOTION_SETTLE_MS) {
        bloomColonyEnergy = fminf(1.0f, bloomColonyEnergy + BLOOM_RECOVERY_RAMP * dt);
    }

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        if (!draining) {
            bloomFlashGlow[i] *= fastDecay(bloomFlashDecay[i], dt * 30.0f);
            if (bloomFlashGlow[i] < 0.005f) bloomFlashGlow[i] = 0.0f;
        }

        bloomBreathPhase[i] += dt / bloomBreathPeriod[i];
        if (bloomBreathPhase[i] >= 1.0f) bloomBreathPhase[i] -= 1.0f;

        float wakeThresh = bloomHueT[i] * BLOOM_RECOVERY_SPREAD;
        float ledRecovery = clampf(
            (bloomColonyEnergy - wakeThresh) / 0.30f, 0.0f, 1.0f);

        float breath = (fastSinPhase(bloomBreathPhase[i]) * 0.5f + 0.5f);
        float breathGlow = BLOOM_BREATH_FLOOR
            + breath * (bloomBreathPeak[i] - BLOOM_BREATH_FLOOR);
        breathGlow *= ledRecovery;

        if (draining) {
            float target = breathGlow * bloomHitIntensity * ENERGY_MULTIPLIER;
            if (target > bloomFlashGlow[i]) {
                bloomFlashGlow[i] = target;
                bloomFlashDecay[i] = BLOOM_FLASH_DECAY_LO
                    + randFloat() * (BLOOM_FLASH_DECAY_HI - BLOOM_FLASH_DECAY_LO);
            }
        }

        float g = fmaxf(breathGlow, bloomFlashGlow[i]);

        float flashFrac = (bloomFlashGlow[i] > breathGlow) ? 1.0f : 0.0f;
        float h = bloomHueT[i];
        float colG = lerpf(lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h),
                           BLOOM_FLASH_G, flashFrac);
        float colB = lerpf(lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h),
                           BLOOM_FLASH_B, flashFrac);

        float linBright = fastGamma24(g) * BLOOM_BRIGHTNESS_CAP;

        float oG = colG * linBright;
        float oB = colB * linBright;

        float wFrac = clampf((g - BLOOM_W_ONSET) / (1.0f - BLOOM_W_ONSET),
                             0.0f, 1.0f);
        float energyGate = clampf((bloomColonyEnergy - 0.7f) / 0.3f, 0.0f, 1.0f);
        float wGate = fmaxf(energyGate, flashFrac);
        float oW = wFrac * wGate * linBright * 200.0f;

        uint16_t tG16 = (uint16_t)clampf(oG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(oB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(oW * 256.0f, 0, 65535);

        if (tG16 < BLOOM_NOISE_GATE) tG16 = 0;
        if (tB16 < BLOOM_NOISE_GATE) tB16 = 0;
        if (tW16 < BLOOM_NOISE_GATE) tW16 = 0;

        uint8_t gc = deltaSigma(dsG[i], tG16);
        uint8_t b  = deltaSigma(dsB[i], tB16);
        uint8_t w  = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, 0, gc, b, w);
    }

    if (draining) {
        float drain = drainEnvelope * dt;
        drain = fminf(drain, bloomColonyEnergy);
        bloomColonyEnergy -= drain;
        drainEnvelope *= expf(-4.07f * dt);
        if (drainEnvelope <= 0.001f) drainEnvelope = 0.0f;
    }
}

// ── Sparkle burst ────────────────────────────────────────────────

static void renderSparkleBurst(float dt, float angleDeg, float tiltBlend,
                                float tiltR, float tiltG, float tiltB) {
    bool isSilent = energy < 0.001f;

    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > envelope)
        envelope += attackAlpha * (energy - envelope);
    else
        envelope += decayAlpha * (energy - envelope);

    cooldownRemaining = fmaxf(0.0f, cooldownRemaining - dt);

    float onsetThreshold = fmaxf(0.15f, 0.4f - envelope * 0.3f);

    if (onset > onsetThreshold && cooldownRemaining <= 0.0f && !isSilent) {
        cooldownRemaining = fmaxf(0.050f, 0.150f - envelope * 0.10f);

        float onsetStrength = clampf(onset, 0.0f, 1.0f);
        int nIgnite = (int)(LED_COUNT * (0.3f + 0.2f * onsetStrength));

        static uint8_t indices[LED_COUNT];
        for (uint8_t i = 0; i < LED_COUNT; i++) indices[i] = i;
        for (int i = 0; i < nIgnite; i++) {
            int j = i + (int)(xorshift32() % (LED_COUNT - i));
            uint8_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        float sparkVal = 0.7f + 0.3f * onsetStrength;
        for (int i = 0; i < nIgnite; i++) {
            sparkle[indices[i]] = sparkVal;
            decayRates[indices[i]] = 0.92f + randFloat() * 0.05f;
        }
    }

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        sparkle[i] *= fastDecay(decayRates[i], dt * 30.0f);
    }

    if (!isSilent) {
        for (uint16_t i = 0; i < LED_COUNT; i++) {
            float jitter = (randFloat() - 0.5f) * 0.02f;
            float newVal = sparkle[i] + jitter;
            if (newVal < 0.0f) newVal = 0.0f;
            if (newVal > sparkle[i] && jitter > 0) newVal = sparkle[i];
            sparkle[i] = newVal;
        }
    }

    float base = fminf(envelope, 0.2f);

    sparkPeakR = sparkPeakG = sparkPeakB = sparkPeakW = 0;
    sparkPeakBright = 0.0f;
    sparkPeakLin = 0.0f;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float s = sparkle[i];
        float bright = base + s * (1.0f - base);
        if (bright < SPARKLE_DEADBAND) bright = 0.0f;

        float colR = 255.0f;
        float colG = 180.0f + (240.0f - 180.0f) * s;
        float colB =  80.0f + (200.0f -  80.0f) * s;

        if (tiltBlend > 0.0f) {
            colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
            colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
            colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
        }

        float linBright = fastGamma24(bright) * SPARKLE_BRIGHTNESS_CAP;
        float colW = 255.0f * (1.0f - tiltBlend);
        float fR = colR * linBright;
        float fG = colG * linBright;
        float fB = colB * linBright;
        float fW = colW * linBright;

        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);

        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, r, g, b, w);

        if (r > sparkPeakR) sparkPeakR = r;
        if (g > sparkPeakG) sparkPeakG = g;
        if (b > sparkPeakB) sparkPeakB = b;
        if (w > sparkPeakW) sparkPeakW = w;
        if (bright > sparkPeakBright) sparkPeakBright = bright;
        if (linBright > sparkPeakLin) sparkPeakLin = linBright;
    }
}

// ── Sparkle syllable ─────────────────────────────────────────────
// Onset detection runs on phone (44kHz waveform), arrives as 4-byte
// async UDP packets on port 4213. ESP32 only renders.

static float syllSparkle[LED_COUNT];
static float syllDecay[LED_COUNT];
static float syllEnvelope = 0.0f;
static float syllCooldown = 0.0f;

static void resetSyllableState() {
    memset(syllSparkle, 0, sizeof(syllSparkle));
    syllEnvelope = 0.0f;
    syllCooldown = 0.0f;
}

static void renderSparkleSyllable(float dt, float angleDeg, float tiltBlend,
                                   float tiltR, float tiltG, float tiltB) {
    uint32_t now = millis();
    uint32_t onsetAge = now - syllOnsetMs;
    uint8_t strength = syllOnsetStrength;

    // Consume onset if fresh (< 100ms old)
    bool gotOnset = (syllOnsetMs > 0 && onsetAge < 100);
    float onsetNorm = gotOnset ? (strength / 255.0f) : 0.0f;

    // Envelope tracks energy from status packets (same as sparkle_burst)
    float attackAlpha = fminf(1.0f, dt / 0.030f);
    float decayAlpha  = fminf(1.0f, dt / 0.400f);
    if (energy > syllEnvelope)
        syllEnvelope += attackAlpha * (energy - syllEnvelope);
    else
        syllEnvelope += decayAlpha * (energy - syllEnvelope);

    syllCooldown = fmaxf(0.0f, syllCooldown - dt);

    if (gotOnset && onsetNorm > 0.1f && syllCooldown <= 0.0f) {
        syllOnsetMs = 0;  // mark consumed
        syllCooldown = 0.060f;

        int nIgnite = (int)(LED_COUNT * (0.25f + 0.25f * onsetNorm));
        static uint8_t indices[LED_COUNT];
        for (uint8_t i = 0; i < LED_COUNT; i++) indices[i] = i;
        for (int i = 0; i < nIgnite; i++) {
            int j = i + (int)(xorshift32() % (LED_COUNT - i));
            uint8_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        float sparkVal = 0.6f + 0.4f * onsetNorm;
        for (int i = 0; i < nIgnite; i++) {
            syllSparkle[indices[i]] = sparkVal;
            syllDecay[indices[i]] = 0.92f + randFloat() * 0.05f;
        }
    }

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        syllSparkle[i] *= fastDecay(syllDecay[i], dt * 30.0f);
    }

    float base = fminf(syllEnvelope, 0.15f);

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float s = syllSparkle[i];
        float bright = base + s * (1.0f - base);
        if (bright < SPARKLE_DEADBAND) bright = 0.0f;

        float colR = 255.0f;
        float colG = 180.0f + (240.0f - 180.0f) * s;
        float colB =  80.0f + (200.0f -  80.0f) * s;

        if (tiltBlend > 0.0f) {
            colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
            colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
            colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
        }

        float linBright = fastGamma24(bright) * SPARKLE_BRIGHTNESS_CAP;
        float colW = 255.0f * (1.0f - tiltBlend);
        float fR = colR * linBright;
        float fG = colG * linBright;
        float fB = colB * linBright;
        float fW = colW * linBright;

        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);

        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, r, g, b, w);
    }
}

// ── Fire render ──────────────────────────────────────────────────

static void renderFire(float dt, bool withDropout, float tiltBlend) {
    fireTime = fmodf(fireTime + dt, 6283.1853f);
    float t = fireTime;

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
    if (isPercussiveOnly || isSilent)
        colorTarget = 0.0f;
    else
        colorTarget = energy;

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

    float s = FIRE_FLICKER_SCALE;

    for (uint16_t i = 0; i < LED_COUNT; i++) {
        float fi = (float)i;

        float noise = fastSin(fi * 7.3f + t * 2.5f) *
                       fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

        float noiseAmp = fmaxf(0.15f * s, 0.10f * s / fmaxf(base, 0.1f));
        float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                        + fireFlickerIntensity * (noise - 0.5f) * 0.25f * s;

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

        float maxCh_f = fmaxf(oR, fmaxf(oG, oB));
        float bFrac = maxCh_f / 255.0f;
        float rgbBlend = clampf((bFrac - PURE_W_CEIL) / PURE_W_BLEND, 0.0f, 1.0f);
        float avgRGB = (oR + oG + oB) / 3.0f;
        float fR = oR * rgbBlend;
        float fG = oG * rgbBlend;
        float fB = oB * rgbBlend;
        float fW = avgRGB * (1.0f - rgbBlend) * (1.0f - tiltBlend);

        uint16_t tR16 = (uint16_t)clampf(fR * 256.0f, 0, 65535);
        uint16_t tG16 = (uint16_t)clampf(fG * 256.0f, 0, 65535);
        uint16_t tB16 = (uint16_t)clampf(fB * 256.0f, 0, 65535);
        uint16_t tW16 = (uint16_t)clampf(fW * 256.0f, 0, 65535);

        if (tR16 < 256) tR16 = 0;
        if (tG16 < 256) tG16 = 0;
        if (tB16 < 256) tB16 = 0;
        if (tW16 < 256) tW16 = 0;

        uint8_t r = deltaSigma(dsR[i], tR16);
        uint8_t g = deltaSigma(dsG[i], tG16);
        uint8_t b = deltaSigma(dsB[i], tB16);
        uint8_t w = deltaSigma(dsW[i], tW16);
        strip.setPixelColor(i, r, g, b, w);
    }
}

// ── Main loop ────────────────────────────────────────────────────

void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;

    // HTTP runs on AsyncTCP task automatically. Just handle serial here.
    parseSerialCommands();

    // Snapshot shared sensor state under mutex (AsyncTCP task writes it).
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
    // Effects that read latestPacket directly (e.g. gravity sparkle) keep
    // working — single 16-bit reads are atomic on ESP32 anyway. Snapshot
    // is used here for self-consistent multi-field math below.
    (void)snap;

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

        unsigned long pkts = (unsigned long)pktCount;
        if (currentAlg == ALG_QUIET_BLOOM) {
            Serial.printf("[%s] rate=%.1f energy=%.3f flash0=%.3f pkts=%lu\n",
                algName, bloomMotionRate, bloomColonyEnergy, bloomFlashGlow[0],
                pkts);
        } else if (currentAlg == ALG_SPARKLE_BURST) {
            Serial.printf("[%s] angle=%.1f rms=%d en=%.3f ons=%.3f peak b=%.2f lin=%.3f rgbw=%3u/%3u/%3u/%3u pkts=%lu\n",
                algName, angleDeg, rawRms, energy, onset,
                sparkPeakBright, sparkPeakLin,
                sparkPeakR, sparkPeakG, sparkPeakB, sparkPeakW, pkts);
        } else {
            Serial.printf("[%s] angle=%.1f rms=%d energy=%.3f onset=%.3f mic=%s pkts=%lu\n",
                algName, angleDeg, rawRms, energy, onset, micOn ? "on" : "off",
                pkts);
        }
    }


    float tiltR = 0, tiltG = 0, tiltB = 0;
    float tiltBlend = 0.0f;
    if (angleDeg > DEADZONE_DEG) {
        float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
        if (hueFrac > 1.0f) hueFrac = 1.0f;
        uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255) + engageHueOffset) & 0xFF);
        tiltR = (float)oklchVarL[hueIdx][0];
        tiltG = (float)oklchVarL[hueIdx][1];
        tiltB = (float)oklchVarL[hueIdx][2];
        tiltBlend = (angleDeg - DEADZONE_DEG) / BLEND_RANGE_DEG;
        if (tiltBlend > 1.0f) tiltBlend = 1.0f;
    }

    switch (currentAlg) {
        case ALG_SPARKLE_BURST:
            renderSparkleBurst(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB);
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
            renderSparkleSyllable(dt, angleDeg, tiltBlend, tiltR, tiltG, tiltB);
            break;
    }



    strip.show();
    delay(1);
}
