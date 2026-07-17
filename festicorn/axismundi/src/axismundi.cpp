/*
 * AXISMUNDI — pot-driven particle field on strips of arbitrary length.
 *
 * ESP32-D0WD-V3 (biolum-b, MAC B4:BF:E9:C9:01:38). 6 × WS2812B strips, GRB.
 *   slot 0  GPIO 4   150 LEDs
 *   slot 1  GPIO 15   50 LEDs   (GPIO15 is a strapping pin — post-boot data
 *   slot 2  GPIO 17   50 LEDs    out is fine, but suspect it first if a strip
 *   slot 3  GPIO 5    50 LEDs    attached at boot ever wedges startup)
 *   slot 4  GPIO 18   50 LEDs
 *   slot 5  GPIO 19  150 LEDs
 *
 * Forked from tree-of-record's clicky effect. The difference that names this
 * project: strips are NOT equal length. The particle field lives entirely in
 * normalized [0,1] space and is rasterized per-strip at each strip's own
 * length (constant-PROPORTION policy) — a blob covers the same fraction of a
 * 50-LED strip as a 150-LED one, so all LED-space tunings scale by len/REF_LEN.
 * Flip RASTER_PROPORTIONAL to 0 for constant-PIXEL-width instead.
 *
 * Driven by a clicky-pots console (6 push-pull pots + buttons) over ESP-NOW
 * broadcast on channel 1. Every packet is full panel state.
 *
 *   pot position  → particle position along the strip (fixed per-pot hue)
 *   pull out      → split the blob into a spaced comb (×4 spacing)
 *   btn 0 shards  → burst of free-flying shards from each particle
 *   btn 1 pump    → grow particle size
 *   btn 2 hue-rot → +30° per press
 *   btn 3 gather  → hold to pull all particles to the field mean
 *   btn 4 fade    → hold to desaturate+fade to black (fires a crackle)
 *   btn 5 trail   → hold to paint a fading trail
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <math.h>
#include <fast_math.h>
#include <clicky_packet_v1.h>

// ── Strip layout ─────────────────────────────────────────────────
static const uint8_t  NUM_STRIPS = 6;
static const uint16_t STRIP_LEN[NUM_STRIPS] = {150, 50, 50, 50, 50, 150};
#define MAX_LEDS   150
#define REF_LEN    100.0f   // LED-space tunings were authored at this length
#define RASTER_PROPORTIONAL 1

#define FIXED_CHANNEL 1

#define MASTER_BRIGHTNESS 1.0f   // brightness ceiling (pot 5 fully out), 0..1
#define CLICKY_BRIGHT_POT  5      // this pot is a brightness fader, not a particle
#define BRIGHT_FLOOR       0.0f   // brightness at pot 5 fully in (full black)

static float gBrightness = MASTER_BRIGHTNESS;

static float gDt = 1.0f / 60.0f;   // current frame dt, set each loop iteration

// ── clicky-pots panel state (over ESP-NOW) ──────────────────────
static volatile ClickyPacketV1 clickyPkt = {};
static volatile uint32_t clickyLastMs = 0;
static volatile uint32_t clickyPktCount = 0;

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

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// ── HSV → RGB (h 0..360, s/v 0..1, rgb 0..255) ─────────────────
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

// ── ESP-NOW receive (clicky packets only) ───────────────────────
static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len != (int)sizeof(ClickyPacketV1)) return;
    ClickyPacketV1 cp;
    memcpy(&cp, data, sizeof(cp));
    if (cp.magic != CLICKY_MAGIC || cp.ver != CLICKY_VER) return;
    memcpy((void*)&clickyPkt, &cp, sizeof(cp));
    clickyLastMs = millis();
    clickyPktCount++;
}

// ── LED driver: 6 strips via RMT (GPIO 4/15/17/5/18/19), GRB order ──
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(STRIP_LEN[0], 4);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(STRIP_LEN[1], 15);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(STRIP_LEN[2], 17);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(STRIP_LEN[3], 5);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(STRIP_LEN[4], 18);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(STRIP_LEN[5], 19);

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
    // Stagger the Show()s to avoid RMT ISR contention (see engineering ledger
    // esp32-rmt-simultaneous-show-glitch-frames).
    strip0.Show(); strip1.Show(); strip2.Show();
    while (!strip0.CanShow() || !strip1.CanShow() || !strip2.CanShow()) {}
    strip3.Show(); strip4.Show(); strip5.Show();
}

// ── Hysteretic floor: anti-flicker policy for sub-LSB channels ───
// Per-pixel timer (seconds). Positive = time below threshold (→ dormant),
// negative = time above (→ active). See original for the full state machine.
#define FLOOR_DEACTIVATE_S  1.0f
#define FLOOR_REACTIVATE_S  0.5f
#define FLOOR_THRESHOLD     256    // target16 below this = sub-LSB danger zone

static float fxFloorTimer[NUM_STRIPS][MAX_LEDS];

static void resetFxDither() {
    memset(fxFloorTimer, 0, sizeof(fxFloorTimer));
}

static inline uint16_t applyFloorPolicy(uint16_t target16, float &timer, float dt) {
    if (target16 < FLOOR_THRESHOLD) {
        timer += dt;
        if (timer >= FLOOR_DEACTIVATE_S) {
            timer = FLOOR_DEACTIVATE_S;
            return 0;
        }
        return FLOOR_THRESHOLD;
    } else {
        timer -= dt;
        if (timer <= -FLOOR_REACTIVATE_S) {
            timer = 0.0f;
            return target16;
        }
        if (timer > 0.0f) return 0;
        return target16;
    }
}

// fr/fg/fb in 0–255 linear range (already gamma-shaped by the effect).
static inline void setPixelScaled(uint8_t s, uint16_t i, float fr, float fg, float fb) {
    uint16_t t16R = (uint16_t)fminf(fr * 256.0f * gBrightness, 65535.0f);
    uint16_t t16G = (uint16_t)fminf(fg * 256.0f * gBrightness, 65535.0f);
    uint16_t t16B = (uint16_t)fminf(fb * 256.0f * gBrightness, 65535.0f);
    uint16_t maxT16 = t16R > t16G ? (t16R > t16B ? t16R : t16B)
                                   : (t16G > t16B ? t16G : t16B);
    uint16_t floorResult = applyFloorPolicy(maxT16, fxFloorTimer[s][i], gDt);
    if (floorResult == 0 && maxT16 < FLOOR_THRESHOLD) {
        t16R = t16G = t16B = 0;
    }
    setPixel(s, i, t16R >> 8, t16G >> 8, t16B >> 8);
}

// ── Clicky particles (pot-driven) ───────────────────────────────
// One particle per pot, in normalized [0,1] space, rasterized onto every
// strip at that strip's own length. Pot position = particle position along
// the strip, fixed per-pot hue. Pulling the knob out splits the blob apart
// into a spaced comb.
#define CLICKY_SIGMA_NORM    0.025f  // = 2.5 LED on a REF_LEN strip
#define CLICKY_PULL_NORM     0.04f   // comb spacing = 4 LED on a REF_LEN strip
#define CLICKY_POS_TAU       0.05f
// Ruffle: spatial sine rippling through each blob (cycles held proportional
// to blob width by evaluating on the REF_LEN-normalized index).
#define CLICKY_RUFFLE_AMP    0.45f
#define CLICKY_RUFFLE_FREQ   1.4f    // rad per REF-LED
#define CLICKY_RUFFLE_SPEED  7.0f    // rad per second

static const float CLICKY_HUES[CLICKY_NUM_POTS] = {0, 60, 120, 180, 240, 300};
static float clickyPos[CLICKY_NUM_POTS];
static bool  clickyPosInit = false;
// Ruffle time phase, wrapped to [0, 2π): an unbounded seconds accumulator
// loses float resolution after ~36 h uptime (ulp > frame dt) and the ruffle
// visibly freezes.
static float clickyRufflePhase = 0.0f;

// Buttons (bit = panel position, left to right). See header comment.
#define CLICKY_BTN_SHARDS  0
#define CLICKY_BTN_PUMP    1
#define CLICKY_BTN_HUEROT  2
#define CLICKY_BTN_GATHER  3
#define CLICKY_BTN_TRAIL   5
#define CLICKY_BTN_FADE    4
#define CLICKY_TRAIL_K     1.4f    // exp fade: ~0.1% left after 5 s
#define CLICKY_PAINT_S     0.6f    // dwell time to saturate paint to its cap
#define CLICKY_PAINT_CAP   160.0f  // max paint level (vs 255-scale particle)
#define CLICKY_FADE_S      1.5f
#define CLICKY_RECOVER_S   0.7f
#define CLICKY_CRACKLE_S   0.5f
#define CLICKY_PUMP_KICK   0.9f    // sigma multiplier added per press
#define CLICKY_PUMP_MAX    3.0f
#define CLICKY_PUMP_TAU    0.6f
#define CLICKY_MOVE_EPS    0.0015f // per-frame smoothed-pos delta = "moving"
#define CLICKY_GATHER_S    0.25f   // ramp time to fully gathered

// Shard field lives in normalized space: pos in [0,1], vel in units/sec.
#define CLICKY_NUM_SHARDS  48
struct ClickyShard {
    float pos, vel, hue, life, maxLife;
};
static ClickyShard clickyShards[CLICKY_NUM_SHARDS];

static float clickyTrail[NUM_STRIPS][MAX_LEDS][3];
static float clickyFade = 1.0f;
static float clickyCrackleT = 0.0f;
static float clickyPump[CLICKY_NUM_POTS];
static int   clickyShardNext = 0;
static float clickyGather = 0.0f;
static float clickyHueRot[CLICKY_NUM_POTS];
static float clickyMovedAgo[CLICKY_NUM_POTS];
static uint8_t clickyPrevBtn = 0;

static void resetClickyParticles() {
    resetFxDither();
    clickyPosInit = false;
    memset(clickyTrail, 0, sizeof(clickyTrail));
    memset(clickyShards, 0, sizeof(clickyShards));
    memset(clickyHueRot, 0, sizeof(clickyHueRot));
    for (int k = 0; k < CLICKY_NUM_POTS; k++) {
        clickyMovedAgo[k] = 1e9f;
        clickyPump[k] = 1.0f;
    }
    clickyFade = 1.0f;
    clickyCrackleT = 0.0f;
    clickyGather = 0.0f;
    clickyPrevBtn = 0;
    clickyShardNext = 0;
}

// Rasterize the shared normalized field onto one strip of length `len`.
static void rasterStrip(uint8_t s, uint16_t len, const ClickyPacketV1 &pkt,
                        float gatherT, float meanPos, float dt) {
    static float buf[MAX_LEDS][3];
    static float cov[MAX_LEDS];
    memset(buf, 0, sizeof(float) * len * 3);
    memset(cov, 0, sizeof(float) * len);

#if RASTER_PROPORTIONAL
    float sigScale = (float)len / REF_LEN;   // LED-space per REF-LED
#else
    float sigScale = 1.0f;
#endif

    for (int k = 0; k < CLICKY_NUM_POTS; k++) {
        if (k == CLICKY_BRIGHT_POT) continue;
        float ownN = clickyPos[k];
        float centerN = lerpf(ownN, meanPos, gatherT);
        float center = centerN * (float)(len - 1);
        float sigma = fmaxf(0.5f, CLICKY_SIGMA_NORM * REF_LEN * sigScale * clickyPump[k]);

        if (clickyCrackleT > 0.0f) {
            float env = clickyCrackleT / CLICKY_CRACKLE_S;
            for (int n = 0; n < 4; n++) {
                if (randFloat() > env) continue;
                int i = (int)lroundf(center + (randFloat() - 0.5f) * 4.0f * sigma);
                if (i < 0 || i >= (int)len) continue;
                float v = (8.0f + randFloat() * 30.0f) * env;
                buf[i][0] = buf[i][1] = buf[i][2] = v;
                cov[i] = 1.0f;
            }
            continue;
        }
        if (clickyFade <= 0.0f) continue;

        float r, g, b;
        hsvToRgb(fmodf(CLICKY_HUES[k] + clickyHueRot[k], 360.0f),
                 sqrtf(clickyFade), 1.0f, r, g, b);

        int spacing = 1;
        if (pkt.pullBits & (1 << k))
            spacing = (int)fmaxf(1.0f, lroundf(CLICKY_PULL_NORM * REF_LEN * sigScale));
        int reach = (int)(4.0f * sigma);
        for (int j = -reach; j <= reach; j++) {
            int i = (int)lroundf(center) + j * spacing;
            if (i < 0 || i >= (int)len) continue;
            float d = (float)j / sigma;
            // Evaluate ruffle on the REF_LEN-normalized index so the number of
            // ripples across a blob is the same on a 50- and a 150-LED strip.
            float ei = (float)i / (float)len * REF_LEN;
            float ruffle = 1.0f + CLICKY_RUFFLE_AMP
                * fastSin(ei * CLICKY_RUFFLE_FREQ
                          + clickyRufflePhase
                          + (float)k * 2.1f);
            float w = expf(-d * d) * ruffle * clickyFade;
            buf[i][0] += r * w;
            buf[i][1] += g * w;
            buf[i][2] += b * w;
            cov[i] += w;
        }
    }

    // Shards: free-flying, max-blended. Normalized pos → this strip's LEDs.
    for (int n = 0; n < CLICKY_NUM_SHARDS; n++) {
        ClickyShard &sh = clickyShards[n];
        if (sh.life <= 0.0f) continue;
        float fadeF = fmaxf(0.0f, sh.life / sh.maxLife);
        float r, g, b;
        hsvToRgb(sh.hue, 1.0f, 1.0f, r, g, b);
        int i = (int)lroundf(sh.pos * (float)(len - 1));
        for (int o = -1; o <= 1; o++) {
            int ii = i + o;
            if (ii < 0 || ii >= (int)len) continue;
            float w = fadeF * (o == 0 ? 0.4f : 0.125f);
            buf[ii][0] = fmaxf(buf[ii][0], r * w);
            buf[ii][1] = fmaxf(buf[ii][1], g * w);
            buf[ii][2] = fmaxf(buf[ii][2], b * w);
            cov[ii] = fmaxf(cov[ii], w);
        }
    }

    // Paintbrush trail (per strip, since strips differ in length).
    bool trailHeld = pkt.btnBits & (1 << CLICKY_BTN_TRAIL);
    if (trailHeld) {
        float rate = dt * (CLICKY_PAINT_CAP / 255.0f) / CLICKY_PAINT_S;
        for (uint16_t i = 0; i < len; i++)
            for (int c = 0; c < 3; c++)
                clickyTrail[s][i][c] = fminf(CLICKY_PAINT_CAP,
                                             clickyTrail[s][i][c] + buf[i][c] * rate);
    } else {
        float decay = expf(-CLICKY_TRAIL_K * dt);
        for (uint16_t i = 0; i < len; i++)
            for (int c = 0; c < 3; c++)
                clickyTrail[s][i][c] *= decay;
    }

    for (uint16_t i = 0; i < len; i++) {
        float bg = 1.0f - fminf(cov[i], 1.0f);
        float r = buf[i][0] + clickyTrail[s][i][0] * bg;
        float g = buf[i][1] + clickyTrail[s][i][1] * bg;
        float b = buf[i][2] + clickyTrail[s][i][2] * bg;
        setPixelScaled(s, i, fminf(r, 255.0f), fminf(g, 255.0f), fminf(b, 255.0f));
    }
}

static void renderClickyParticles(float dt) {
    ClickyPacketV1 pkt;
    memcpy(&pkt, (const void*)&clickyPkt, sizeof(pkt));

    // ── State update (strip-independent, once per frame) ──────────
    clickyRufflePhase += dt * CLICKY_RUFFLE_SPEED;
    if (clickyRufflePhase >= 2.0f * (float)M_PI)
        clickyRufflePhase -= 2.0f * (float)M_PI;
    float alpha = fminf(1.0f, dt / CLICKY_POS_TAU);

    bool fadeHeld   = pkt.btnBits & (1 << CLICKY_BTN_FADE);
    bool gatherHeld = pkt.btnBits & (1 << CLICKY_BTN_GATHER);
    uint8_t pressed = pkt.btnBits & ~clickyPrevBtn;
    clickyPrevBtn = pkt.btnBits;

    float prevFade = clickyFade;
    if (fadeHeld) clickyFade = fmaxf(0.0f, clickyFade - dt / CLICKY_FADE_S);
    else          clickyFade = fminf(1.0f, clickyFade + dt / CLICKY_RECOVER_S);
    if (prevFade > 0.0f && clickyFade <= 0.0f) clickyCrackleT = CLICKY_CRACKLE_S;

    if (gatherHeld) clickyGather = fminf(1.0f, clickyGather + dt / CLICKY_GATHER_S);
    else            clickyGather = fmaxf(0.0f, clickyGather - dt / CLICKY_GATHER_S);
    float gatherT = clickyGather * clickyGather * (3.0f - 2.0f * clickyGather);

    float meanPos = 0.0f;
    for (int k = 0; k < CLICKY_NUM_POTS; k++) {
        float target = (float)pkt.pos[k] / 10000.0f;
        if (!clickyPosInit) clickyPos[k] = target;
        float before = clickyPos[k];
        clickyPos[k] += alpha * (target - clickyPos[k]);
        if (fabsf(clickyPos[k] - before) > CLICKY_MOVE_EPS) clickyMovedAgo[k] = 0.0f;
        else clickyMovedAgo[k] += dt;
        if (k != CLICKY_BRIGHT_POT) meanPos += clickyPos[k];
    }
    meanPos /= (float)(CLICKY_NUM_POTS - 1);
    clickyPosInit = true;

    gBrightness = MASTER_BRIGHTNESS
        * (BRIGHT_FLOOR + (1.0f - BRIGHT_FLOOR) * clickyPos[CLICKY_BRIGHT_POT]);

    if (pressed & (1 << CLICKY_BTN_HUEROT))
        for (int k = 0; k < CLICKY_NUM_POTS; k++)
            clickyHueRot[k] = fmodf(clickyHueRot[k] + 30.0f, 360.0f);

    if (pressed & (1 << CLICKY_BTN_PUMP))
        for (int k = 0; k < CLICKY_NUM_POTS; k++)
            clickyPump[k] = fminf(CLICKY_PUMP_MAX, clickyPump[k] + CLICKY_PUMP_KICK);
    for (int k = 0; k < CLICKY_NUM_POTS; k++)
        clickyPump[k] = 1.0f + (clickyPump[k] - 1.0f) * expf(-dt / CLICKY_PUMP_TAU);

    if (pressed & (1 << CLICKY_BTN_SHARDS)) {
        for (int k = 0; k < CLICKY_NUM_POTS; k++) {
            if (k == CLICKY_BRIGHT_POT) continue;
            float centerN = clickyPos[k];
            float edgeN = 2.0f * CLICKY_SIGMA_NORM * clickyPump[k];
            int burst = 3 + (int)(randFloat() * 4.0f);
            for (int m = 0; m < burst; m++) {
                ClickyShard &sh = clickyShards[clickyShardNext];
                clickyShardNext = (clickyShardNext + 1) % CLICKY_NUM_SHARDS;
                float dir = (randFloat() < 0.5f) ? -1.0f : 1.0f;
                sh.pos = centerN + dir * edgeN;
                sh.vel = dir * (0.25f + randFloat() * 0.20f);   // units/sec
                sh.hue = fmodf(CLICKY_HUES[k] + clickyHueRot[k]
                               + (randFloat() - 0.5f) * 50.0f + 360.0f, 360.0f);
                sh.maxLife = 0.20f + randFloat() * 0.15f;
                sh.life = sh.maxLife;
            }
        }
    }

    // Integrate shards once (rasterStrip reads them read-only).
    for (int n = 0; n < CLICKY_NUM_SHARDS; n++) {
        ClickyShard &sh = clickyShards[n];
        if (sh.life <= 0.0f) continue;
        sh.life -= dt;
        sh.pos += sh.vel * dt;
    }

    // ── Rasterize onto every strip ────────────────────────────────
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        rasterStrip(s, STRIP_LEN[s], pkt, gatherT, meanPos, dt);

    if (clickyCrackleT > 0.0f) clickyCrackleT -= dt;
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0));
    strip2.ClearTo(RgbColor(0)); strip3.ClearTo(RgbColor(0));
    strip4.ClearTo(RgbColor(0)); strip5.ClearTo(RgbColor(0));
    showAll();

    resetClickyParticles();

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

    static const uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BROADCAST_ADDR, 6);
    peer.channel = FIXED_CHANNEL;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    Serial.printf("Axismundi (clicky, 6-strip) ready — ch=%u\n", FIXED_CHANNEL);
    Serial.printf("  strips: 4=%u 15=%u 17=%u 5=%u 18=%u 19=%u\n",
                  STRIP_LEN[0], STRIP_LEN[1], STRIP_LEN[2],
                  STRIP_LEN[3], STRIP_LEN[4], STRIP_LEN[5]);
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / 60.0f);
    if (dt > 0.1f) dt = 0.1f;
    gDt = dt;
    lastRenderMs = now;

    // No panel seen yet → hold dark.
    if (clickyLastMs == 0) {
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint16_t i = 0; i < STRIP_LEN[s]; i++)
                setPixel(s, i, 0, 0, 0);
        showAll();
        return;
    }

    renderClickyParticles(dt);
    showAll();

    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    if (now - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (now - lastLogMs);
        bool clickyLive = (int32_t)(now - clickyLastMs) < 3000;
        Serial.printf("  FPS=%.1f  [clicky] %s pkts=%lu pull=%02X btn=%02X pos0=%u\n",
                      fps, clickyLive ? "LIVE" : "----",
                      (unsigned long)clickyPktCount,
                      clickyPkt.pullBits, clickyPkt.btnBits, clickyPkt.pos[0]);
        frameCount = 0;
        lastLogMs = now;
    }
}
