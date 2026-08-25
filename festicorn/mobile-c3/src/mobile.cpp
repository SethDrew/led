// Wiring/hardware: catalog/boards.yaml firmware_by_role.mobile-c3.
// Effects ported from tree-of-record (audio stripped) + neighborhood wave_tap.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <fast_math.h>
#include <v1_packet.h>

#define FIXED_CHANNEL 6
#define N_WASH  LED_COUNT_A
#define N_FX    LED_COUNT_B

static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> stripWash(N_WASH, LED_PIN_A);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> stripFx(N_FX, LED_PIN_B);

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
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

// ── Global controls ──────────────────────────────────────────────
static float   brightF  = 128.0f;
static uint8_t speedIdx = 2;
static const float SPEEDS[5] = { 0.24f, 0.33f, 0.45f, 0.61f, 0.82f };

enum FxId : uint8_t { FX_BLOOM, FX_FIRE, FX_NEBULA, FX_LIGHT_THROUGH, FX_GYRO_PULSE, FX_COUNT };
static const char *FX_NAMES[FX_COUNT] =
    { "bloom", "fire", "nebula", "light_through", "gyro_pulse" };
static uint8_t currentFx = FX_BLOOM;

enum WashId : uint8_t { WASH_PINK_BLUE, WASH_TEAL_ORANGE, WASH_GREEN_INDIGO,
                        WASH_LAVENDER_CRIMSON, WASH_COUNT };
static const char *WASH_NAMES[WASH_COUNT] = { "pink_blue", "teal_orange", "green_indigo",
                                              "lavender_crimson" };
static uint8_t currentWash = WASH_PINK_BLUE;

static const float WASH_ENDS[WASH_COUNT][2][3] = {
    { { 255.0f,  25.0f,  90.0f }, {  20.0f,  60.0f, 255.0f } },
    { {   0.0f, 190.0f, 160.0f }, { 255.0f,  90.0f,  10.0f } },
    { {  10.0f,  90.0f,  30.0f }, {  45.0f,  20.0f, 120.0f } },
    { { 170.0f, 110.0f, 255.0f }, { 255.0f,  20.0f,  10.0f } },
};

// ── ESP-NOW v1 telemetry (biolum bulb-sender, gyro_pulse input) ──
#define COUNTS_PER_G    8192.0f
#define COUNTS_PER_INT8  256.0f

static volatile uint32_t pktCount = 0;
static volatile uint32_t lastPacketMs = 0;
static TelemetryPacketV1 latestPacket = {0};
static uint32_t prevPktCount = 0;

static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len != sizeof(TelemetryPacketV1)) return;
    memcpy((void*)&latestPacket, data, sizeof(TelemetryPacketV1));
    lastPacketMs = millis();
    pktCount++;
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
    Serial.println("calibrating — keep sender still 2s");
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
        Serial.printf("calibrated rest=(%.3f,%.3f,%.3f)\n", restAx, restAy, restAz);
    }
}

// ── OKLCH wash gradients (rebuilt on wash change; endpoints in sRGB) ─
static float washBuf[N_WASH][3];

static inline float srgbToLin(float c) {
    c /= 255.0f;
    return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}
static inline float linToSrgb(float c) {
    c = clampf(c, 0.0f, 1.0f);
    return (c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f)) * 255.0f;
}

static void rgbToOklab(float r, float g, float b, float &L, float &A, float &B) {
    r = srgbToLin(r); g = srgbToLin(g); b = srgbToLin(b);
    float l = cbrtf(0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b);
    float m = cbrtf(0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b);
    float s = cbrtf(0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b);
    L = 0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s;
    A = 1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s;
    B = 0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s;
}

static bool oklabToRgb(float L, float A, float B, float &r, float &g, float &b) {
    float l = L + 0.3963377774f * A + 0.2158037573f * B;
    float m = L - 0.1055613458f * A - 0.0638541728f * B;
    float s = L - 0.0894841775f * A - 1.2914855480f * B;
    l = l * l * l; m = m * m * m; s = s * s * s;
    float lr = 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    float lg = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    float lb = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
    bool inGamut = lr >= -0.001f && lr <= 1.001f &&
                   lg >= -0.001f && lg <= 1.001f &&
                   lb >= -0.001f && lb <= 1.001f;
    r = linToSrgb(lr); g = linToSrgb(lg); b = linToSrgb(lb);
    return inGamut;
}

static void rebuildWash() {
    const float (*we)[3] = WASH_ENDS[currentWash];
    float La, Aa, Ba, Lb, Ab, Bb;
    rgbToOklab(we[0][0], we[0][1], we[0][2], La, Aa, Ba);
    rgbToOklab(we[1][0], we[1][1], we[1][2], Lb, Ab, Bb);
    float C0 = sqrtf(Aa * Aa + Ba * Ba), h0 = atan2f(Ba, Aa);
    float C1 = sqrtf(Ab * Ab + Bb * Bb), h1 = atan2f(Bb, Ab);
    float dh = h1 - h0;
    if (dh > (float)M_PI)  dh -= 2.0f * (float)M_PI;
    if (dh < -(float)M_PI) dh += 2.0f * (float)M_PI;

    for (uint16_t i = 0; i < N_WASH; i++) {
        float t = 0.5f - 0.5f * cosf((float)M_PI * (float)i / (float)(N_WASH - 1));
        float L = lerpf(La, Lb, t);
        float C = lerpf(C0, C1, t);
        float h = h0 + dh * t;
        float r, g, b;
        while (!oklabToRgb(L, C * cosf(h), C * sinf(h), r, g, b) && C > 0.001f)
            C *= 0.9f;
        washBuf[i][0] = r; washBuf[i][1] = g; washBuf[i][2] = b;
    }
}

// ── Effect frame buffer (0–255 linear floats, gamma'd per effect) ─
static float fb[N_FX][3];

static inline void fbSet(uint16_t i, float r, float g, float b) {
    fb[i][0] = r; fb[i][1] = g; fb[i][2] = b;
}

// ── Bloom (tree-of-record ambient branch: breathing hue field) ───
#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.75f
#define BLOOM_BREATH_MAX_PEAK   1.00f
#define BLOOM_HUE_DRIFT_MIN  (1.0f / 45.0f)
#define BLOOM_HUE_DRIFT_MAX  (1.0f / 15.0f)
#define BLOOM_HUE_A_R    0.0f
#define BLOOM_HUE_A_G  180.0f
#define BLOOM_HUE_A_B  120.0f
#define BLOOM_HUE_B_R  140.0f
#define BLOOM_HUE_B_G   20.0f
#define BLOOM_HUE_B_B  255.0f
#define GATE_ON_THRESH  0.7f
#define GATE_OFF_THRESH 1.2f

static float bloomCap      = 1.64f;
static float bloomFloor    = 0.28f;
static float bloomContrast = 1.6f;

static float bloomBreathPhase[N_FX];
static float bloomBreathPeriod[N_FX];
static float bloomBreathPeak[N_FX];
static float bloomHueT[N_FX];
static float bloomHueDrift[N_FX];
static bool  bloomGateOff[N_FX];

static void resetBloom() {
    for (uint16_t i = 0; i < N_FX; i++) {
        bloomBreathPhase[i]  = randFloat();
        bloomBreathPeriod[i] = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bloomBreathPeak[i]   = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bloomHueT[i]         = randFloat();
        float rate = BLOOM_HUE_DRIFT_MIN
            + randFloat() * (BLOOM_HUE_DRIFT_MAX - BLOOM_HUE_DRIFT_MIN);
        bloomHueDrift[i]     = (randFloat() > 0.5f) ? rate : -rate;
        bloomGateOff[i]      = false;
    }
}

static void renderBloom(float dt) {
    float glow[N_FX];
    float glowSum = 0.0f;
    for (uint16_t i = 0; i < N_FX; i++) {
        float breath = fastSinPhase(bloomBreathPhase[i]) * 0.5f + 0.5f;
        glow[i] = bloomFloor + breath * (bloomBreathPeak[i] - bloomFloor);
        glowSum += glow[i];
    }
    float glowMean = glowSum / (float)N_FX;

    for (uint16_t i = 0; i < N_FX; i++) {
        float stretched = clampf(glowMean + (glow[i] - glowMean) * bloomContrast, 0.0f, 1.0f);
        float breathLin = fastGamma24(stretched) * bloomCap;

        bloomBreathPhase[i] += dt / bloomBreathPeriod[i];
        if (bloomBreathPhase[i] >= 1.0f) bloomBreathPhase[i] -= 1.0f;
        bloomHueT[i] += bloomHueDrift[i] * dt;
        if (bloomHueT[i] > 1.0f) bloomHueT[i] -= 1.0f;
        else if (bloomHueT[i] < 0.0f) bloomHueT[i] += 1.0f;

        float h = bloomHueT[i];
        float oR = lerpf(BLOOM_HUE_A_R, BLOOM_HUE_B_R, h) * breathLin;
        float oG = lerpf(BLOOM_HUE_A_G, BLOOM_HUE_B_G, h) * breathLin;
        float oB = lerpf(BLOOM_HUE_A_B, BLOOM_HUE_B_B, h) * breathLin;

        float maxCh = fmaxf(fmaxf(oR, oG), oB);
        float thresh = bloomGateOff[i] ? GATE_OFF_THRESH : GATE_ON_THRESH;
        if (maxCh < thresh) {
            oR = oG = oB = 0.0f;
            bloomGateOff[i] = true;
        } else {
            bloomGateOff[i] = false;
        }
        fbSet(i, oR, oG, oB);
    }
}

// ── Fire (tree-of-record, audio energy → slow autonomous wander) ─
#define FIRE_FLICKER_SCALE 3.0f
#define FIRE_BASE_LEVEL    0.78f

static float fireTime   = 0.0f;
static float fireWander = 0.0f;

static void resetFire() {
    fireTime = 0.0f;
    fireWander = randFloat() * 100.0f;
}

static void renderFire(float dt) {
    fireTime = fmodf(fireTime + dt, 6283.1853f);
    fireWander += dt;

    float base = FIRE_BASE_LEVEL
        + 0.10f * fastSin(fireWander * 0.31f) * fastSin(fireWander * 0.117f);
    float ce = 0.30f + 0.15f * (0.5f + 0.5f * fastSin(fireWander * 0.083f));

    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;
    const float amberR = 255.0f, amberG = 75.0f, amberB = 5.0f;
    const float redR   = 200.0f, redG   = 15.0f, redB   = 0.0f;
    float tr = clampf((ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD),
                      0.0f, 1.0f);
    float colR = lerpf(amberR, redR, tr);
    float colG = lerpf(amberG, redG, tr);
    float colB = lerpf(amberB, redB, tr);

    float t = fireTime;
    float noiseAmp = fmaxf(0.15f * FIRE_FLICKER_SCALE,
                           0.10f * FIRE_FLICKER_SCALE / fmaxf(base, 0.1f));
    for (uint16_t i = 0; i < N_FX; i++) {
        float fi = (float)i;
        float noise = fastSin(fi * 7.3f + t * 2.5f) *
                      fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;
        float bright = clampf(base * (1.0f + noiseAmp * (noise - 0.5f)), 0.0f, 1.0f);
        float linBright = fastGamma24(bright);
        fbSet(i, colR * linBright, colG * linBright, colB * linBright);
    }
}

// ── Nebula (tree-of-record port, single strip) ───────────────────
#define NEBULA_MAX_ORBS       5
#define NEBULA_ORB_TAIL       30.0f
#define NEBULA_ORB_BASE_SPEED 0.45f
#define NEBULA_SPAWN_CHANCE   0.03f
#define NEBULA_MIN_LIFETIME   200
#define NEBULA_MAX_LIFETIME   300
#define NEBULA_INTRINSIC_SPD  0.3f

struct NebOrb { float pos, vel, age, lifetime; bool active; };

static float  nebulaTime = 0.0f;
static NebOrb nebOrbs[NEBULA_MAX_ORBS];
static float  nebDecay[N_FX];

static void resetNebula() {
    nebulaTime = 0.0f;
    for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) nebOrbs[o].active = false;
    for (uint16_t i = 0; i < N_FX; i++) nebDecay[i] = 0.0f;
}

static void renderNebula(float dt) {
    float spd = NEBULA_INTRINSIC_SPD;
    nebulaTime += dt * spd;
    float t = nebulaTime * 60.0f;

    float decayPerFrame = 1.0f - (1.0f / NEBULA_ORB_TAIL);
    float decay = powf(powf(decayPerFrame, 60.0f), dt);

    for (uint16_t i = 0; i < N_FX; i++) {
        nebDecay[i] *= decay;
        if (nebDecay[i] < 0.01f) nebDecay[i] = 0.0f;
    }

    float spawnRoll = (float)(xorshift32() & 0xFFFF) / 65536.0f;
    if (spawnRoll < NEBULA_SPAWN_CHANCE * dt * 30.0f) {
        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
            if (!nebOrbs[o].active) {
                NebOrb &orb = nebOrbs[o];
                orb.pos = randFloat() * (float)N_FX;
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
        NebOrb &orb = nebOrbs[o];
        if (!orb.active) continue;
        orb.age += dt;
        orb.pos += orb.vel * dt * 60.0f;
        orb.pos = fmodf(orb.pos + (float)N_FX, (float)N_FX);
        if (orb.age >= orb.lifetime) { orb.active = false; continue; }

        float lc = orb.age / orb.lifetime;
        float bright;
        if (lc < 0.4f)      { float tf = lc / 0.4f;          bright = tf * tf * (3.0f - 2.0f * tf); }
        else if (lc > 0.6f) { float tf = (1.0f - lc) / 0.4f; bright = tf * tf * (3.0f - 2.0f * tf); }
        else                { bright = 1.0f; }

        int base = (int)orb.pos;
        int next = (base + 1) % N_FX;
        float frac = orb.pos - (float)base;
        nebDecay[base] = fminf(1.0f, nebDecay[base] + bright * 0.6f * (1.0f - frac));
        nebDecay[next] = fminf(1.0f, nebDecay[next] + bright * 0.6f * frac);
    }

    for (uint16_t i = 0; i < N_FX; i++) {
        float pos = (float)i / (float)N_FX;
        float breathing = 76.0f + 57.0f * fastSin(t * 0.0105f);
        float spatial = 76.0f * (0.5f + 0.5f * fastSin((pos + t * 0.006f) * 6.2832f));
        float bgBright = clampf(breathing + spatial, 0.0f, 230.0f) / 255.0f;

        float colorShift = 0.5f + 0.5f * fastSin(pos * 6.2832f + t * 0.009f);
        float r = ((20.0f + colorShift * 235.0f) / 255.0f) * bgBright;
        float g = ((30.0f - colorShift * 20.0f) / 255.0f) * bgBright;
        float b = ((255.0f - colorShift * 125.0f) / 255.0f) * bgBright;

        float orbB = nebDecay[i];
        if (orbB > 0.01f) {
            r += orbB * 1.5f;
            g += orbB * 1.41f;
            b += orbB * 1.17f;
        }
        fbSet(i, clampf(r, 0.0f, 1.0f) * 255.0f,
                 clampf(g, 0.0f, 1.0f) * 255.0f,
                 clampf(b, 0.0f, 1.0f) * 255.0f);
    }
}

// ── Light Through (tree-of-record, collapsed from radial to 1D) ──
#define LT_MAX_PATCHES  6
#define LT_SPAWN_PERIOD 2.2f
#define LT_CHURN_RATE   0.12f
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

struct LtPatch { float rc, maxR, life, age; bool active; };
static LtPatch ltPatches[LT_MAX_PATCHES];
static float ltTime = 0.0f;

static inline float ltNoise(float x) {
    return 0.5f + 0.5f * fastSin(x * 2.3f + fastSin(x * 1.7f + ltTime * 0.08f));
}

static void resetLightThrough() {
    ltTime = 0.0f;
    for (int i = 0; i < LT_MAX_PATCHES; i++) ltPatches[i].active = false;
}

static void ltSpawnPatch() {
    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (ltPatches[i].active) continue;
        LtPatch &p = ltPatches[i];
        p.rc   = 0.10f + randFloat() * 0.80f;
        p.maxR = 0.12f + randFloat() * 0.23f;
        p.life = 1.5f + randFloat() * 2.0f;
        p.age  = 0.0f;
        p.active = true;
        return;
    }
}

static void renderLightThrough(float dt) {
    ltTime += dt;
    if (randFloat() < dt / LT_SPAWN_PERIOD) ltSpawnPatch();

    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (!ltPatches[i].active) continue;
        ltPatches[i].age += dt;
        if (ltPatches[i].age >= ltPatches[i].life) ltPatches[i].active = false;
    }

    for (uint16_t i = 0; i < N_FX; i++) {
        float r = (float)i / (float)(N_FX - 1);
        float density = ltNoise(r * 3.0f + ltTime * LT_CHURN_RATE);
        float centerLift = lerpf(1.0f, 0.85f, r);
        float outR = lerpf(LT_SLATE_R, LT_INDIGO_R, density) * centerLift;
        float outG = lerpf(LT_SLATE_G, LT_INDIGO_G, density) * centerLift;
        float outB = lerpf(LT_SLATE_B, LT_INDIGO_B, density) * centerLift;

        for (int pi = 0; pi < LT_MAX_PATCHES; pi++) {
            if (!ltPatches[pi].active) continue;
            LtPatch &p = ltPatches[pi];
            float tau = p.age / p.life;
            float radius = p.maxR * fastSin((float)M_PI * tau);
            if (radius < 0.001f) continue;

            float d = fabsf(r - p.rc);
            float g = d / (0.6f * radius);
            float a = expf(-(g * g));
            if (a < 0.004f) continue;

            float dr = clampf(d / radius, 0.0f, 1.0f);
            float pcR, pcG, pcB;
            if (dr < 0.5f) {
                float tt = dr / 0.5f;
                pcR = lerpf(LT_AMBER_R, LT_PEACH_R, tt);
                pcG = lerpf(LT_AMBER_G, LT_PEACH_G, tt);
                pcB = lerpf(LT_AMBER_B, LT_PEACH_B, tt);
            } else {
                float tt = (dr - 0.5f) / 0.5f;
                pcR = lerpf(LT_PEACH_R, LT_SALMON_R, tt);
                pcG = lerpf(LT_PEACH_G, LT_SALMON_G, tt);
                pcB = lerpf(LT_PEACH_B, LT_SALMON_B, tt);
            }
            outR = outR * (1.0f - a) + pcR * a + 0.15f * a * 255.0f;
            outG = outG * (1.0f - a) + pcG * a + 0.15f * a * 255.0f;
            outB = outB * (1.0f - a) + pcB * a;
        }
        fbSet(i, clampf(outR, 0.0f, 255.0f),
                 clampf(outG, 0.0f, 255.0f),
                 clampf(outB, 0.0f, 255.0f));
    }
}

// ── Gyro pulse (neighborhood wave_tap port, single strip) ────────
#define WP_MAX_LEAVES    16
#define WP_SPEED         20.0f
#define WP_LEAF_SIGMA     1.5f
#define WP_FADE_IN_TIME   0.20f
#define WP_FADE_OUT_LEDS  8.0f
#define WP_MAX_BRIGHTNESS 0.85f
#define WP_DEFAULT_R     50.0f
#define WP_DEFAULT_G    200.0f
#define WP_DEFAULT_B     80.0f
#define WP_TAP_THRESH_G   0.50f
#define WP_COOLDOWN_MS  120
#define DEADZONE_DEG     10.0f
#define MAX_ANGLE_DEG   180.0f

struct WPLeaf { float pos, colR, colG, colB, brightness, age; bool active; };

static WPLeaf   wpLeaves[WP_MAX_LEAVES];
static uint32_t wpLastHitMs = 0;
static uint32_t tapCount    = 0;
static float    lastPeakG   = 0.0f;

static void resetGyroPulse() {
    for (uint8_t i = 0; i < WP_MAX_LEAVES; i++) wpLeaves[i].active = false;
}

static void wpPickColor(float angleDeg, float &cR, float &cG, float &cB) {
    if (angleDeg <= DEADZONE_DEG) {
        cR = WP_DEFAULT_R; cG = WP_DEFAULT_G; cB = WP_DEFAULT_B;
        return;
    }
    float hueFrac = clampf((angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG),
                           0.0f, 1.0f);
    float h6 = hueFrac * 6.0f;
    int   seg = (int)h6;
    float f = h6 - seg;
    float full = 255.0f;
    switch (seg) {
        case 0:  cR = full;            cG = full * f;        cB = 0;               break;
        case 1:  cR = full * (1 - f);  cG = full;            cB = 0;               break;
        case 2:  cR = 0;               cG = full;            cB = full * f;        break;
        case 3:  cR = 0;               cG = full * (1 - f);  cB = full;            break;
        case 4:  cR = full * f;        cG = 0;               cB = full;            break;
        default: cR = full;            cG = 0;               cB = full * (1 - f);  break;
    }
}

static void gyroPulseProcessMotion(uint32_t now, float angleDeg) {
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

    float hitIntensity = clampf((peak_axis_ac_g - WP_TAP_THRESH_G) / 2.5f, 0.0f, 1.0f);
    int nLeaves = 1 + (int)(hitIntensity * 3.0f + 0.5f);
    if (nLeaves > 4) nLeaves = 4;

    float baseR, baseG, baseB;
    wpPickColor(angleDeg, baseR, baseG, baseB);

    float endPos = (float)(N_FX - 1);
    for (int n = 0; n < nLeaves; n++) {
        int8_t slot = -1;
        for (uint8_t j = 0; j < WP_MAX_LEAVES; j++) {
            if (!wpLeaves[j].active) { slot = j; break; }
        }
        if (slot < 0) break;
        WPLeaf &lf = wpLeaves[slot];
        lf.pos        = endPos + (float)n * 0.7f + randFloat() * 0.5f;
        lf.colR       = clampf(baseR * (0.9f + 0.2f * randFloat()), 0.0f, 255.0f);
        lf.colG       = clampf(baseG * (0.9f + 0.2f * randFloat()), 0.0f, 255.0f);
        lf.colB       = clampf(baseB * (0.9f + 0.2f * randFloat()), 0.0f, 255.0f);
        lf.brightness = 0.0f;
        lf.age        = 0.0f;
        lf.active     = true;
    }
}

static void renderGyroPulse(float dt) {
    float sigma_sq2 = 2.0f * WP_LEAF_SIGMA * WP_LEAF_SIGMA;

    for (uint8_t i = 0; i < WP_MAX_LEAVES; i++) {
        WPLeaf &lf = wpLeaves[i];
        if (!lf.active) continue;
        lf.pos -= WP_SPEED * dt;
        lf.age += dt;
        lf.brightness = (lf.age < WP_FADE_IN_TIME) ? (lf.age / WP_FADE_IN_TIME) : 1.0f;
        if (lf.pos < WP_FADE_OUT_LEDS && lf.pos >= 0.0f)
            lf.brightness *= lf.pos / WP_FADE_OUT_LEDS;
        // Deactivate at LED 0 — rendering past it lets the gaussian
        // tail bleed back into the first LEDs at full weight.
        if (lf.pos <= 0.0f) lf.active = false;
    }

    for (uint16_t li = 0; li < N_FX; li++) {
        float glow = 0.0f, oR = 0.0f, oG = 0.0f, oB = 0.0f;
        for (uint8_t j = 0; j < WP_MAX_LEAVES; j++) {
            const WPLeaf &lf = wpLeaves[j];
            if (!lf.active) continue;
            float d = (float)li - lf.pos;
            float w = expf(-(d * d) / sigma_sq2) * lf.brightness;
            glow += w;
            oR   += w * lf.colR;
            oG   += w * lf.colG;
            oB   += w * lf.colB;
        }
        if (glow > 1e-6f) {
            float linBright = fastGamma24(clampf(glow, 0.0f, 1.0f) * WP_MAX_BRIGHTNESS);
            fbSet(li, clampf((oR / glow) * linBright, 0.0f, 255.0f),
                      clampf((oG / glow) * linBright, 0.0f, 255.0f),
                      clampf((oB / glow) * linBright, 0.0f, 255.0f));
        } else {
            fbSet(li, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ── Button (trike FSM: hold=brightness, 1=speed, 2=effect, 3=wash) ─
#define DEBOUNCE_MS   20
#define HOLD_MS      400
#define CLICK_GAP_MS 300
#define RAMP_PER_S    50.0f
#define BRIGHT_MIN     8.0f

static bool     rawPrev = false, debounced = false;
static uint32_t btnChangeMs = 0, pressStartMs = 0, lastReleaseMs = 0;
static bool     holdActive = false;
static uint8_t  clickCount = 0;
static int8_t   rampDir = 1;

static void resetFx(uint8_t fx) {
    switch (fx) {
        case FX_BLOOM:         resetBloom();        break;
        case FX_FIRE:          resetFire();         break;
        case FX_NEBULA:        resetNebula();       break;
        case FX_LIGHT_THROUGH: resetLightThrough(); break;
        case FX_GYRO_PULSE:    resetGyroPulse();    break;
    }
}

static void dispatchClicks(uint8_t n) {
    if (n == 1) {
        speedIdx = (speedIdx + 1) % 5;
        Serial.printf("speed: %.1fx\n", SPEEDS[speedIdx]);
    } else if (n == 2) {
        currentFx = (currentFx + 1) % FX_COUNT;
        resetFx(currentFx);
        Serial.printf("effect: %s\n", FX_NAMES[currentFx]);
    } else {
        currentWash = (currentWash + 1) % WASH_COUNT;
        rebuildWash();
        Serial.printf("wash: %s\n", WASH_NAMES[currentWash]);
    }
}

static void updateButton(uint32_t now, float dt) {
    bool raw = digitalRead(BUTTON_PIN) == LOW;
    if (raw != rawPrev) { btnChangeMs = now; rawPrev = raw; }

    if ((now - btnChangeMs) >= DEBOUNCE_MS && raw != debounced) {
        debounced = raw;
        if (debounced) {
            pressStartMs = now;
        } else {
            if (holdActive) {
                holdActive = false;
                rampDir = -rampDir;
                clickCount = 0;
            } else if ((now - pressStartMs) < HOLD_MS) {
                clickCount++;
                lastReleaseMs = now;
            }
        }
    }

    if (debounced && !holdActive && (now - pressStartMs) >= HOLD_MS) {
        holdActive = true;
        clickCount = 0;
        Serial.printf("brightness ramp %s\n", rampDir > 0 ? "up" : "down");
    }

    if (holdActive) {
        brightF = clampf(brightF + (float)rampDir * RAMP_PER_S * dt,
                         BRIGHT_MIN, 255.0f);
    }

    if (clickCount > 0 && !debounced && (now - lastReleaseMs) > CLICK_GAP_MS) {
        dispatchClicks(clickCount);
        clickCount = 0;
    }
}

// ── Output ───────────────────────────────────────────────────────
static void showFrame() {
    float gb = fastGamma24(brightF / 255.0f);

    for (uint16_t i = 0; i < N_WASH; i++) {
        stripWash.SetPixelColor(i, RgbColor(
            (uint8_t)(washBuf[i][0] * gb),
            (uint8_t)(washBuf[i][1] * gb),
            (uint8_t)(washBuf[i][2] * gb)));
    }
    for (uint16_t i = 0; i < N_FX; i++) {
        stripFx.SetPixelColor(i, RgbColor(
            (uint8_t)clampf(fb[i][0] * gb, 0.0f, 255.0f),
            (uint8_t)clampf(fb[i][1] * gb, 0.0f, 255.0f),
            (uint8_t)clampf(fb[i][2] * gb, 0.0f, 255.0f)));
    }
    // Stagger Show()s — see engineering ledger esp32-rmt-simultaneous-show-glitch-frames
    stripWash.Show();
    while (!stripWash.CanShow()) {}
    stripFx.Show();
}

void setup() {
    Serial.begin(115200);
    delay(300);
    prngState = esp_random() | 1;
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    stripWash.Begin();
    stripFx.Begin();
    stripWash.Show();
    stripFx.Show();

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    if (esp_now_init() == ESP_OK) esp_now_register_recv_cb(onReceive);

    resetFx(currentFx);
    rebuildWash();
    Serial.printf("mobile-c3 ready ch=%u — press: speed, double: effect, "
                  "triple: wash, hold: brightness. 'c' = recalibrate\n",
                  FIXED_CHANNEL);
}

void loop() {
    uint32_t now = millis();
    static uint32_t lastFrameMs = 0;
    float dt = (lastFrameMs > 0) ? (now - lastFrameMs) / 1000.0f : (1.0f / 60.0f);
    if (dt > 0.1f) dt = 0.1f;
    lastFrameMs = now;

    updateButton(now, dt);

    bool connected = lastPacketMs > 0 && (now - lastPacketMs) < 3000;
    float angleDeg = 0.0f;
    if (connected && pktCount != prevPktCount) {
        prevPktCount = pktCount;
        float ax = ((float)latestPacket.ax_mean * COUNTS_PER_INT8) / COUNTS_PER_G;
        float ay = ((float)latestPacket.ay_mean * COUNTS_PER_INT8) / COUNTS_PER_G;
        float az = ((float)latestPacket.az_mean * COUNTS_PER_INT8) / COUNTS_PER_G;
        vecNormalize(ax, ay, az);
        if (calStartMs == 0) startCalibration();
        updateCalibration(ax, ay, az);
        if (calibrated) {
            float cosAngle = clampf(restAx * ax + restAy * ay + restAz * az, -1.0f, 1.0f);
            angleDeg = acosf(cosAngle) * (180.0f / (float)M_PI);
        }
        if (currentFx == FX_GYRO_PULSE) gyroPulseProcessMotion(now, angleDeg);
    }

    float fxDt = dt * SPEEDS[speedIdx];
    switch (currentFx) {
        case FX_BLOOM:         renderBloom(fxDt);        break;
        case FX_FIRE:          renderFire(fxDt);         break;
        case FX_NEBULA:        renderNebula(fxDt);       break;
        case FX_LIGHT_THROUGH: renderLightThrough(fxDt); break;
        case FX_GYRO_PULSE:    renderGyroPulse(fxDt);    break;
    }

    showFrame();

    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'c') startCalibration();
    }

    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    if (now - lastLogMs > 2000) {
        Serial.printf("fps=%.1f %s fx=%s wash=%s spd=%.1f br=%d pkts=%lu taps=%lu peak=%.2fg\n",
                      frameCount * 1000.0f / (now - lastLogMs),
                      connected ? "LIVE" : "----",
                      FX_NAMES[currentFx], WASH_NAMES[currentWash],
                      SPEEDS[speedIdx], (int)brightF,
                      (unsigned long)pktCount, (unsigned long)tapCount, lastPeakG);
        frameCount = 0;
        lastLogMs = now;
    }
    delay(5);
}
