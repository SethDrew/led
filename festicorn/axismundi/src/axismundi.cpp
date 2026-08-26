/*
 * AXISMUNDI — pot-driven particle field on strips of arbitrary length.
 *
 * ESP32 (led-eth-1, MAC A4:F0:0F:61:40:18). 4 × WS2812B strips, GRB,
 * 150 LEDs each, GPIO 13/27/32/33 — data leaves the board on RJ45
 * conductors (cable as plain wire, not ethernet).
 *
 * Forked from tree-of-record's clicky effect. The difference that names this
 * project: strips may be ANY length (this install happens to run four equal
 * 150s — set STRIP_LEN per strip). The particle field lives entirely in
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
 *   btn 0 desat   → hold to fade chroma to 20% (pastel), release restores
 *   btn 1 pump    → hold to grow particle size
 *   btn 2 hue-rot → hold to rotate hue continuously
 *   btn 3 gather  → hold to pull all particles to the field mean
 *   btn 4 fade    → hold to desaturate+fade to black (fires a crackle)
 *   btn 5 trail   → hold to paint a fading trail
 *
 * Also driven by the kettle-knob console (infinite-spin encoder + 5 buttons)
 * over ESP-NOW. Spin turns the whole picture: each strip is a closed ring
 * and the composed frame rotates around it, one knob revolution per ring
 * trip, signed with direction.
 *
 *   btn A flywheel → hold; spin adds velocity and the picture coasts
 *   btn B twist    → hold; rings fan apart into a helix as you spin
 *   btn C counter  → hold; rings 1 and 3 turn the other way
 *   btn D crackle  → hold; sparks pinned to physical positions
 *
 * And by the duck console (mic + IMU totem) over ESP-NOW: mic energy pours an
 * energy waterfall down every strip, and turning the duck over changes the
 * water's color — the tilt-hue interaction of tree-of-record's sparkle effect,
 * re-aimed at the falling water. Any console may be absent. See README.md.
 *
 * Layer order matters: the clicky field is composed in logical ring space and
 * rotated by the carousel, while the duck waterfall and the kettle crackle are
 * composited afterwards in physical space, so spin never drags them off their
 * spot on the sculpture. Clicky layers map their hue through the shared
 * rotation base; the duck water is its own palette (white at rest, duck
 * rotation colors it) and deliberately ignores hue-rot. Red-with-blue is kept
 * off the piece by the choice of the six pot seeds alone; see CLICKY_HUES.
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
#include <clicky_packet_v1.h>
#include <kettle_packet_v1.h>
#include <duck_packet_v1.h>
#include <kettle_energy.h>
#include <duck_energy.h>
#include <hue_arc.h>

// ── Strip layout ─────────────────────────────────────────────────
// Rendering is per SEGMENT: each segment rasterizes the shared normalized
// field over its own length (the viz twin's per-strip raster, mirrored here)
// and rotates by its parent wire's carousel ring. fold>0 marks a canopy chain
// of `fold` daisy-chained out-and-back radials — logical len is one radius,
// mirrored onto every leg. AM_MOCKUP = backyard mockup, one board; default
// (no flag) = led-eth-1 full rig, unchanged.
// wfRev flips the duck waterfall's flow within the segment. Mockup uses it
// for the rising-sap story: roots flow tip->trunk and helix base->top (both
// wired head-first at trunk/top), canopy radiates hub->tip.
// rgb: product sends R,G,B on the wire (WS2811 bullets) vs the strips' G,R,B —
// verify each physical batch at bring-up, flip the flag if the label lied.
// guy: safety layer — dim slow rainbow, bypasses every effect and the carousel.
struct AmSeg { uint8_t strip; uint16_t off, len; uint8_t fold, wfRev, rgb, guy; };
#define AM_NUM_RINGS 4
#if defined(AM_MOCKUP)
  static const AmSeg SEG[] = {
    {0, 0, 25, 2, 0}, {1, 0, 25, 2, 0},
    {2, 0, 85, 0, 1}, {2, 85, 65, 0, 1}, {2, 150, 75, 0, 1}, {2, 225, 75, 0, 1},
    {3, 0, 85, 0, 1}, {3, 85, 75, 0, 1}, {3, 160, 75, 0, 1}, {3, 235, 75, 0, 1},
  };
  #define AM_NS 10
  static const uint16_t PHYS_LEN[AM_NUM_RINGS] = {100, 100, 300, 310};
  #define MAX_LEDS 85
#elif defined(AM_PLAYA_CANOPY)
  static const AmSeg SEG[] = {
    {0, 0, 100, 0, 0, 1, 0}, {1, 0, 100, 0, 0, 1, 0},
    {2, 0, 100, 0, 0, 1, 0}, {3, 0,  75, 0, 0, 1, 0},
  };
  #define AM_NS 4
  static const uint16_t PHYS_LEN[AM_NUM_RINGS] = {100, 100, 100, 100};
  #define MAX_LEDS 100
#elif defined(AM_PLAYA_RAYS)
  static const AmSeg SEG[] = {
    {0, 0, 50, 0, 0, 1, 0}, {0, 50, 150, 0, 0, 0, 1},
    {1, 0, 50, 0, 0, 1, 0}, {1, 50, 150, 0, 0, 0, 1},
    {2, 0, 50, 0, 0, 1, 0}, {2, 50, 150, 0, 0, 0, 1},
    {3, 0, 50, 0, 0, 1, 0}, {3, 50, 150, 0, 0, 0, 1},
  };
  #define AM_NS 8
  static const uint16_t PHYS_LEN[AM_NUM_RINGS] = {200, 200, 200, 200};
  #define MAX_LEDS 150
#elif defined(AM_PLAYA_TRUNK)
  static const AmSeg SEG[] = {
    {0, 0, 150, 0, 1, 0, 0}, {0, 150, 150, 0, 0, 0, 0},
    {1, 0, 50, 0, 0, 1, 0}, {1, 50, 150, 0, 0, 0, 1},
    {2, 0, 50, 0, 0, 1, 0}, {2, 50, 150, 0, 0, 0, 1},
    {3, 0, 50, 0, 0, 1, 0}, {3, 50, 150, 0, 0, 0, 1},
  };
  #define AM_NS 8
  static const uint16_t PHYS_LEN[AM_NUM_RINGS] = {300, 200, 200, 200};
  #define MAX_LEDS 150
#elif defined(AM_PLAYA_ROOTS)
  static const AmSeg SEG[] = {
    {0, 0, 75, 0, 1, 0, 0}, {0, 75, 75, 0, 0, 0, 0},
    {1, 0, 75, 0, 1, 0, 0}, {1, 75, 75, 0, 0, 0, 0},
    {2, 0, 75, 0, 1, 0, 0}, {2, 75, 75, 0, 0, 0, 0},
    {3, 0, 75, 0, 1, 0, 0}, {3, 75, 75, 0, 0, 0, 0},
  };
  #define AM_NS 8
  static const uint16_t PHYS_LEN[AM_NUM_RINGS] = {150, 150, 150, 150};
  #define MAX_LEDS 75
#else
  static const AmSeg SEG[] = {
    {0, 0, 150, 0, 0}, {1, 0, 150, 0, 0}, {2, 0, 150, 0, 0}, {3, 0, 150, 0, 0},
  };
  #define AM_NS 4
  static const uint16_t PHYS_LEN[AM_NUM_RINGS] = {150, 150, 150, 150};
  #define MAX_LEDS 150
#endif
static const uint8_t NUM_STRIPS = AM_NS;
static uint16_t STRIP_LEN[AM_NS];
#define REF_LEN    100.0f   // LED-space tunings were authored at this length
#define RASTER_PROPORTIONAL 1

#define AXFX_NS     NUM_STRIPS
#define AXFX_MAXLEN MAX_LEDS
#define AXFX_LEN(s) STRIP_LEN[s]
#include "axfx_ambient.h"

// Each segment's [j0,j1] slice of the sap journey (0 = root tips, 1 = canopy
// tips), read through the segment's own u after the wfRev flip. Bench layout
// carries no roots/helix semantics: all four rise in parallel over the span.
// Canopy has no table: its runs bend at the rim (spoke out, rim up to the
// corner, rim back down), so sapRegisterSegs builds a per-LED j array instead.
#define AM_J_MIDEDGE (AXFX_SAP_J_HELIX + 0.25f * (124.6f / 138.3f))
#if defined(AM_MOCKUP)
static const float SEG_J[AM_NS][2] = {
    {AXFX_SAP_J_HELIX, 1.0f},             {AXFX_SAP_J_HELIX, 1.0f},
    {AXFX_SAP_J_ROOTS, AXFX_SAP_J_HELIX}, {0.0f, AXFX_SAP_J_ROOTS},
    {0.0f, AXFX_SAP_J_ROOTS},             {0.0f, AXFX_SAP_J_ROOTS},
    {AXFX_SAP_J_ROOTS, AXFX_SAP_J_HELIX}, {0.0f, AXFX_SAP_J_ROOTS},
    {0.0f, AXFX_SAP_J_ROOTS},             {0.0f, AXFX_SAP_J_ROOTS},
};
#elif defined(AM_PLAYA_CANOPY)
#elif defined(AM_PLAYA_RAYS)
static const float SEG_J[AM_NS][2] = {
    {AXFX_SAP_J_HELIX, 1.0f}, {0.0f, 0.0f},
    {AXFX_SAP_J_HELIX, 1.0f}, {0.0f, 0.0f},
    {AXFX_SAP_J_HELIX, 1.0f}, {0.0f, 0.0f},
    {AXFX_SAP_J_HELIX, 1.0f}, {0.0f, 0.0f},
};
#elif defined(AM_PLAYA_TRUNK)
static const float SEG_J[AM_NS][2] = {
    {AXFX_SAP_J_ROOTS, AXFX_SAP_J_HELIX}, {AXFX_SAP_J_ROOTS, AXFX_SAP_J_HELIX},
    {AXFX_SAP_J_HELIX, 1.0f}, {0.0f, 0.0f},
    {AXFX_SAP_J_HELIX, 1.0f}, {0.0f, 0.0f},
    {AXFX_SAP_J_HELIX, 1.0f}, {0.0f, 0.0f},
};
#elif defined(AM_PLAYA_ROOTS)
static const float SEG_J[AM_NS][2] = {
    {0.0f, AXFX_SAP_J_ROOTS}, {0.0f, AXFX_SAP_J_ROOTS},
    {0.0f, AXFX_SAP_J_ROOTS}, {0.0f, AXFX_SAP_J_ROOTS},
    {0.0f, AXFX_SAP_J_ROOTS}, {0.0f, AXFX_SAP_J_ROOTS},
    {0.0f, AXFX_SAP_J_ROOTS}, {0.0f, AXFX_SAP_J_ROOTS},
};
#else
static const float SEG_J[AM_NS][2] = {
    {0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f},
};
#endif

static void sapRegisterSegs() {
#if defined(AM_PLAYA_CANOPY)
    static float j[MAX_LEDS];
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        int len = SEG[s].len;
        for (int i = 0; i < len; i++) {
            if (i < 50)
                j[i] = AXFX_SAP_J_HELIX
                     + (AM_J_MIDEDGE - AXFX_SAP_J_HELIX) * ((float)i / 49.0f);
            else if (i < 75)
                j[i] = AM_J_MIDEDGE
                     + (1.0f - AM_J_MIDEDGE) * ((float)(i - 50) / 24.0f);
            else
                j[i] = 1.0f - (1.0f - AM_J_MIDEDGE) * ((float)(i - 75) / 24.0f);
        }
        axfxSapSetSegJ(s, len, j);
    }
#else
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        axfxSapSetSeg(s, SEG[s].guy ? 0 : SEG[s].len,
                      SEG_J[s][0], SEG_J[s][1], SEG[s].wfRev != 0);
#endif
}

#define FIXED_CHANNEL 1

#define MASTER_BRIGHTNESS 1.0f   // brightness ceiling (pot 5 fully out), 0..1
#define CLICKY_BRIGHT_POT  5      // this pot is a brightness fader, not a particle
#define BRIGHT_FLOOR       0.0f   // brightness at pot 5 fully in (full black)

static float gBrightness = MASTER_BRIGHTNESS;

// Resting show: with no live clicky console the particle field keeps playing,
// half bright at a fifth speed, easing between states so heartbeat dropouts
// don't pop.
#define CLICKY_LIVE_MS    3000
#define CLICKY_REST_DIM   0.5f
#define CLICKY_REST_SPEED 0.2f
#define CLICKY_REST_TAU_S 0.75f
static float clickyRest = 0.0f;

// ── clicky-pots panel state (over ESP-NOW) ──────────────────────
static volatile ClickyPacketV1 clickyPkt = {};
static volatile uint32_t clickyLastMs = 0;
static volatile uint32_t clickyPktCount = 0;

// ── kettle-knob console state (over ESP-NOW) ────────────────────
static volatile KettlePacketV1 kettlePkt = {};
static volatile uint32_t kettleLastMs = 0;
static volatile uint32_t kettlePktCount = 0;

// ── duck console state (over ESP-NOW) ───────────────────────────
static volatile DuckPacketV1 duckPkt = {};
static volatile uint32_t duckLastMs = 0;
static volatile uint32_t duckPktCount = 0;

// RF diagnostics: every kettle arrival ring-buffered in the RX callback and
// drained to serial in loop(), so seq gaps give exact broadcast loss.
struct KettleArrival { uint32_t ms; uint8_t seq; int32_t enc; };
static KettleArrival kettleArr[64];
static volatile uint8_t kettleArrHead = 0;
static uint8_t kettleArrTail = 0;

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

// Raw arrivals before any magic/ver filtering: a rejected packet is otherwise
// indistinguishable from an absent sender at the per-console counters.
static volatile uint32_t rxAnyCount = 0;
static volatile uint32_t rxDropCount = 0;
static volatile uint16_t rxLastLen = 0;
static volatile uint16_t rxLastMagic = 0;
static volatile uint16_t rxDropLen = 0;
static volatile uint16_t rxDropMagic = 0;
static volatile uint8_t  rxDropVer = 0;

static inline void rxNoteDrop(const uint8_t *data, int len) {
    rxDropCount++;
    rxDropLen = (uint16_t)len;
    if (len >= 2) rxDropMagic = (uint16_t)(data[0] | (data[1] << 8));
    if (len >= 3) rxDropVer = data[2];
}

// ── ESP-NOW receive (clicky packets only) ───────────────────────
static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    rxAnyCount++;
    rxLastLen = (uint16_t)len;
    if (len >= 2) rxLastMagic = (uint16_t)(data[0] | (data[1] << 8));
    if (len == (int)sizeof(ClickyPacketV1)) {
        ClickyPacketV1 cp;
        memcpy(&cp, data, sizeof(cp));
        if (cp.magic != CLICKY_MAGIC || cp.ver != CLICKY_VER) { rxNoteDrop(data, len); return; }
        memcpy((void*)&clickyPkt, &cp, sizeof(cp));
        clickyLastMs = millis();
        clickyPktCount++;
    } else if (len == (int)sizeof(KettlePacketV1)) {
        KettlePacketV1 kp;
        memcpy(&kp, data, sizeof(kp));
        if (kp.magic != KETTLE_MAGIC || kp.ver != KETTLE_VER) { rxNoteDrop(data, len); return; }
        memcpy((void*)&kettlePkt, &kp, sizeof(kp));
        kettleLastMs = millis();
        kettlePktCount++;
        KettleArrival &a = kettleArr[kettleArrHead & 63];
        a.ms = millis(); a.seq = kp.seq; a.enc = kp.enc;
        kettleArrHead++;
    } else if (len == (int)sizeof(DuckPacketV1)) {
        DuckPacketV1 dp;
        memcpy(&dp, data, sizeof(dp));
        if (dp.magic != DUCK_MAGIC || dp.ver != DUCK_VER) { rxNoteDrop(data, len); return; }
        memcpy((void*)&duckPkt, &dp, sizeof(dp));
        duckLastMs = millis();
        duckPktCount++;
    } else {
        rxNoteDrop(data, len);
    }
}

// ── LED driver: 4 strips (GPIO 13/27/32/33), GRB order ──
// I2S1 parallel DMA: whole frame streams from one buffer, no mid-frame refill
// ISR, so ESP-NOW/WiFi interrupts can't bit-slip it (RMT did; A/B 2026-07-29).
#define OUT_DRV "I2S1X8"
static NeoPixelBus<NeoGrbFeature, NeoEsp32I2s1X8Ws2812xMethod> strip0(PHYS_LEN[0], 13);
static NeoPixelBus<NeoGrbFeature, NeoEsp32I2s1X8Ws2812xMethod> strip1(PHYS_LEN[1], 27);
static NeoPixelBus<NeoGrbFeature, NeoEsp32I2s1X8Ws2812xMethod> strip2(PHYS_LEN[2], 32);
static NeoPixelBus<NeoGrbFeature, NeoEsp32I2s1X8Ws2812xMethod> strip3(PHYS_LEN[3], 33);

static inline void setPixelRaw(uint8_t s, uint16_t i, const RgbColor &c) {
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
    }
}

static inline void setPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    const AmSeg &sg = SEG[s];
    // GRB host feature: swapping the args makes the wire bytes land R,G,B for
    // bullet segments.
    RgbColor c = sg.rgb ? RgbColor(g, r, b) : RgbColor(r, g, b);
    if (sg.fold == 0) { setPixelRaw(sg.strip, sg.off + i, c); return; }
    uint16_t radial = 2 * sg.len;
    for (uint8_t k = 0; k < sg.fold; k++) {
        uint16_t base = sg.off + k * radial;
        setPixelRaw(sg.strip, base + i, c);
        setPixelRaw(sg.strip, base + radial - 1 - i, c);
    }
}

// Twin: ax_paint_straps. Bypasses setPixelScaled's hueArcRolloff on purpose —
// a safety layer must not be gradeable to black by the kettle arc.
#define STRAP_LUM     0.055f
#define STRAP_CYCLE_S 60.0f
static void paintGuys(uint32_t now) {
    float hue = (float)(now % (uint32_t)(STRAP_CYCLE_S * 1000.0f))
                * (360.0f / (STRAP_CYCLE_S * 1000.0f));
    float r, g, b;
    hsvToRgb(hue, 1.0f, 1.0f, r, g, b);
    uint8_t R = (uint8_t)(r * STRAP_LUM);
    uint8_t G = (uint8_t)(g * STRAP_LUM);
    uint8_t B = (uint8_t)(b * STRAP_LUM);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        if (!SEG[s].guy) continue;
        for (uint16_t i = 0; i < STRIP_LEN[s]; i++)
            setPixel(s, i, R, G, B);
    }
}

static uint32_t outFillUs[NUM_STRIPS];
static uint32_t outShowUs;

static void showAll() {
    uint32_t t0 = micros();
    // One shared DMA stream: every instance must fill its mux slot each frame
    // or the transfer never kicks. No partial shows, no CanShow() between.
    strip0.Show(); uint32_t t1 = micros();
    strip1.Show(); uint32_t t2 = micros();
    strip2.Show(); uint32_t t3 = micros();
    strip3.Show(); uint32_t t4 = micros();
    outFillUs[0] = t1 - t0; outFillUs[1] = t2 - t1;
    outFillUs[2] = t3 - t2; outFillUs[3] = t4 - t3;
    outShowUs = t4 - t0;
}

// fr/fg/fb in 0–255 linear range (already gamma-shaped by the effect).
// No anti-flicker floor here: the hysteretic sub-LSB policy inherited from
// tree-of-record assumes parked content and erased the rotating carousel
// frame (simulated 94-97% light loss while spinning).
static inline void setPixelScaled(uint8_t s, uint16_t i, float fr, float fg, float fb) {
    hueArcRolloff(fr, fg, fb, 255.0f);
    uint16_t t16R = (uint16_t)fminf(fr * 256.0f, 65535.0f);
    uint16_t t16G = (uint16_t)fminf(fg * 256.0f, 65535.0f);
    uint16_t t16B = (uint16_t)fminf(fb * 256.0f, 65535.0f);
    setPixel(s, i, t16R >> 8, t16G >> 8, t16B >> 8);
}

// ── Kettle carousel (spin-driven) ───────────────────────────────
// The spin drives kettle.rot — the carousel phase the output stage rotates
// the whole composed frame by. Math in lib/kettle_energy, shared with the
// viz twin.
static KettleState kettle;

#define LOOP_RECORD 1
#if LOOP_RECORD
#include "loop_record.h"
#endif

#define KETTLE_LEAK_TAU      6.0f
#define KETTLE_LEAK_DELAY_S  1.0f
#define KETTLE_LEAK_RAMP_S   2.0f
#define KETTLE_LEAK_EPS_REVS 0.02f

static void kettleUpdate(float dt) {
    KettlePacketV1 pkt;
    memcpy(&pkt, (const void*)&kettlePkt, sizeof(pkt));
    if (kettleLastMs != 0) {
        kettleInput(kettle, pkt.enc, pkt.btnBits, pkt.upMs);
#if LOOP_RECORD
        lrEncFeed(kettle, pkt.enc, pkt.upMs);
        lrButton(pkt.btnBits, millis());
#endif
    }
    float rotBefore = kettle.rot;
    kettleRotStep(kettle, dt);
    float adv = kettle.rot - rotBefore;
    adv -= roundf(adv);
    float speed = (dt > 0.0f) ? adv / dt : 0.0f;

    // Drift-home (mirrors the viz twin): waits out a deadband after the
    // spin stops, then ramps in and drains the carousel back to zero.
    static float idleT = 0.0f;
    if (fabsf(speed) >= KETTLE_LEAK_EPS_REVS) idleT = 0.0f;
    else idleT += dt;
    if (idleT > KETTLE_LEAK_DELAY_S) {
        float u = fminf(1.0f, (idleT - KETTLE_LEAK_DELAY_S) / KETTLE_LEAK_RAMP_S);
        float f = expf(-dt * u / KETTLE_LEAK_TAU);
        for (int r = 0; r < kettle.numRings; r++) {
            float d = kettle.ringRot[r];
            d -= roundf(d);
            kettle.ringRot[r] = kettleWrap01(d * f);
        }
        float d = kettle.rot;
        d -= roundf(d);
        kettle.rot = kettleWrap01(d * f);
    }
}

// ── Composed frame + carousel output ────────────────────────────
// The clicky field renders into frameBuf in LOGICAL ring space and the output
// stage reads each strip as a ring rotated by that ring's phase. The duck
// waterfall and the kettle crackle render into PHYSICAL space and are added
// after the rotation, so spin slides the particle field underneath while they
// stay pinned to the sculpture. Rotating them with the frame is what made the
// old spin-energy sparkle read as glitches.
#define KETTLE_CRK_LEVEL 220.0f   // frameBuf is 0..255 scale

static float frameBuf[NUM_STRIPS][MAX_LEDS][3];
static float duckBuf[NUM_STRIPS][MAX_LEDS][3];

static void outputRotated() {
    static float phys[MAX_LEDS][3];
    static float crk[MAX_LEDS];
    float restB = CLICKY_REST_DIM + (1.0f - CLICKY_REST_DIM) * clickyRest;
    paintGuys(millis());
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        if (SEG[s].guy) continue;
        uint16_t len = STRIP_LEN[s];
        const float (*sap)[3] = axfxSapRow(s);
        float off = kettleRot(kettle, SEG[s].strip) * (float)len;
        for (uint16_t i = 0; i < len; i++) {
            float srcF = fmodf((float)i - off, (float)len);
            if (srcF < 0.0f) srcF += (float)len;
            uint16_t i0 = (uint16_t)srcF;
            if (i0 >= len) i0 = len - 1;
            uint16_t i1 = (uint16_t)((i0 + 1) % len);
            float fr = srcF - (float)i0;
            phys[i][0] = lerpf(frameBuf[s][i0][0], frameBuf[s][i1][0], fr);
            phys[i][1] = lerpf(frameBuf[s][i0][1], frameBuf[s][i1][1], fr);
            phys[i][2] = lerpf(frameBuf[s][i0][2], frameBuf[s][i1][2], fr);
        }
        memset(crk, 0, sizeof(float) * len);
        kettleCrackleSplat(kettle, SEG[s].strip, len, crk);
        for (uint16_t i = 0; i < len; i++) {
            // Neutral, not warm white: the crackle is additive over every
            // other layer, and an off-arc tint near-cancels the arc's cyan
            // chroma and leaves the residual hue pointing anywhere -- the
            // one way past the red/blue arc guarantee.
            float c = crk[i] * KETTLE_CRK_LEVEL;
            // The sap wipe crossfades the clicky particle layer ALONE: duck and
            // crackle play through it untouched. gBrightness (pot 5) is that
            // same clicky-layer fader — never the duck or the crackle.
            float cf = gBrightness * restB * (1.0f - axfxWipeAt(s, i));
            float cr = phys[i][0] * cf + duckBuf[s][i][0] + sap[i][0] + c;
            float cg = phys[i][1] * cf + duckBuf[s][i][1] + sap[i][1] + c;
            float cb = phys[i][2] * cf + duckBuf[s][i][2] + sap[i][2] + c;
#if LOOP_RECORD
            lrTap(s, i, cr, cg, cb);
#endif
            setPixelScaled(s, i, cr, cg, cb);
        }
    }
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

// Red spans [340,20] and blue [225,265], so the short way round from blue's
// high edge to red's low edge is 75 deg: any two hues sitting one in red and
// one in blue are at least that far apart. These six span 70, under that, so no
// rotation can light both bands at once however far the palette turns. The span
// is centred in the 205 deg gap on the far side, as clear of both as it fits.
static const float CLICKY_HUES[CLICKY_NUM_POTS] = {0, 60, 120, 180, 272, 300};
// The rotation base. btn 2 spins it; the duck reads it too, so all layers rotate
// as one palette rather than each owning a slice of the wheel.
static float gHueRotDeg = 0.0f;
static float clickyPos[CLICKY_NUM_POTS];
static bool  clickyPosInit = false;
// Ruffle time phase, wrapped to [0, 2π): an unbounded seconds accumulator
// loses float resolution after ~36 h uptime (ulp > frame dt) and the ruffle
// visibly freezes.
static float clickyRufflePhase = 0.0f;

// Buttons (bit = panel position, left to right). See header comment.
#define CLICKY_BTN_DESAT   0
#define CLICKY_BTN_PUMP    1
#define CLICKY_BTN_HUEROT  2
#define CLICKY_BTN_GATHER  3
#define CLICKY_BTN_TRAIL   5
#define CLICKY_BTN_FADE    4
#define CLICKY_TRAIL_K     1.4f    // exp fade: ~0.1% left after 5 s
#define CLICKY_PAINT_S     0.6f    // dwell time to saturate paint to its cap
#define CLICKY_PAINT_STOP_S 0.3f   // paint gate fade-out after the knob stops
#define CLICKY_PAINT_CAP   160.0f  // max paint level (vs 255-scale particle)
#define CLICKY_FADE_S      1.5f
#define CLICKY_RECOVER_S   0.7f
#define CLICKY_CRACKLE_S   0.5f
#define CLICKY_PUMP_MAX    3.0f
#define CLICKY_PUMP_TAU    0.6f
#define CLICKY_PUMP_RISE_S 1.0f    // hold time to swell from 1.0 to PUMP_MAX
#define CLICKY_HUEROT_DEG_S  90.0f // hue rotation rate (deg/sec) while held
#define CLICKY_RAINBOW_BURST 10    // shards per pull/push gesture (= comb teeth ±1..±5)
#define CLICKY_EMIT_LIFE     0.45f // pull-out: shard reaches its comb tooth as it dies
#define CLICKY_ABSORB_LIFE   0.50f // push-in: shard converges from its tooth onto center
#define CLICKY_SPLIT_S       0.10f // comb spacing ramp time, 1 ↔ full, on pull/push
#define CLICKY_MOVE_EPS    0.0015f // per-frame smoothed-pos delta = "moving"
#define CLICKY_GATHER_S    0.25f   // ramp time to fully gathered
#define CLICKY_DESAT_MIN   0.5f    // chroma floor while desat button held
#define CLICKY_DESAT_S     0.3f    // ramp time full color <-> desaturated

// Shard field lives in normalized space: pos in [0,1], vel in units/sec.
#define CLICKY_NUM_SHARDS  48
struct ClickyShard {
    float pos, vel, hue, life, maxLife;
};
static ClickyShard clickyShards[CLICKY_NUM_SHARDS];

// The trail stores a LEVEL plus a LOGICAL hue, never a rendered RGB: a stored
// color is a stored arc position, and paint outlives the rotation that made it,
// so baked RGB would leave old strokes stranded outside the current arc.
static float clickyTrailLvl[NUM_STRIPS][MAX_LEDS];
static float clickyTrailHue[NUM_STRIPS][MAX_LEDS];
static float clickyFade = 1.0f;
static float clickyCrackleT = 0.0f;
static float clickyPump[CLICKY_NUM_POTS];
static int   clickyShardNext = 0;
static float clickyGather = 0.0f;
static float clickyMovedAgo[CLICKY_NUM_POTS];
static float clickyDesat = 1.0f;
static uint8_t clickyPrevPull = 0;
static float clickySplit[CLICKY_NUM_POTS];
static float clickySplitEase[CLICKY_NUM_POTS];

static void resetClickyParticles() {
    clickyPosInit = false;
    memset(clickyTrailLvl, 0, sizeof(clickyTrailLvl));
    memset(clickyTrailHue, 0, sizeof(clickyTrailHue));
    memset(clickyShards, 0, sizeof(clickyShards));
    gHueRotDeg = 0.0f;
    memset(clickySplit, 0, sizeof(clickySplit));
    memset(clickySplitEase, 0, sizeof(clickySplitEase));
    for (int k = 0; k < CLICKY_NUM_POTS; k++) {
        clickyMovedAgo[k] = 1e9f;
        clickyPump[k] = 1.0f;
    }
    clickyFade = 1.0f;
    clickyCrackleT = 0.0f;
    clickyGather = 0.0f;
    clickyDesat = 1.0f;
    clickyPrevPull = 0;
    clickyShardNext = 0;
}

// Rasterize the shared normalized field onto one strip of length `len`.
static void rasterStrip(uint8_t s, uint16_t len, const ClickyPacketV1 &pkt,
                        float gatherT, float meanPos, float dt) {
    static float buf[MAX_LEDS][3];
    static float cov[MAX_LEDS];
    static float bufHue[MAX_LEDS];
    static float bufW[MAX_LEDS];
    static int8_t bufPot[MAX_LEDS];
    memset(buf, 0, sizeof(float) * len * 3);
    memset(cov, 0, sizeof(float) * len);
    memset(bufW, 0, sizeof(float) * len);
    memset(bufPot, 0xFF, sizeof(int8_t) * len);

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
                i = ((i % (int)len) + (int)len) % (int)len;
                float v = (8.0f + randFloat() * 30.0f) * env;
                buf[i][0] = buf[i][1] = buf[i][2] = v;
                cov[i] = 1.0f;
            }
            continue;
        }
        if (clickyFade <= 0.0f) continue;

        float r, g, b;
        hsvToRgb(hueWrap360(CLICKY_HUES[k] + gHueRotDeg),
                 sqrtf(clickyFade), 1.0f, r, g, b);

        float fullSpacing = fmaxf(1.0f, CLICKY_PULL_NORM * REF_LEN * sigScale);
        float spacingF = lerpf(1.0f, fullSpacing, clickySplitEase[k]);
        int reach = (int)(4.0f * sigma);
        for (int j = -reach; j <= reach; j++) {
            float tc = center + (float)j * spacingF;
            float d = (float)j / sigma;
            float w = expf(-d * d) * clickyFade;
            // Sub-pixel splat only while the split animates; settled comb
            // snaps to single crisp pixels (and settled blob keeps the old
            // integer-center look).
            float tcs = (clickySplitEase[k] <= 0.001f || clickySplitEase[k] >= 0.999f)
                ? (float)lroundf(tc) : tc;
            tcs -= floorf(tcs / (float)len) * (float)len;
            int i0 = (int)tcs;
            float fr = tcs - (float)i0;
            i0 %= (int)len;
            int i1 = (i0 + 1) % (int)len;
            float w0 = w * (1.0f - fr);
            if (w0 > bufW[i0]) { bufW[i0] = w0; bufHue[i0] = CLICKY_HUES[k]; bufPot[i0] = (int8_t)k; }
            buf[i0][0] += r * w0;
            buf[i0][1] += g * w0;
            buf[i0][2] += b * w0;
            cov[i0] += w0;
            float w1 = w * fr;
            if (w1 > bufW[i1]) { bufW[i1] = w1; bufHue[i1] = CLICKY_HUES[k]; bufPot[i1] = (int8_t)k; }
            buf[i1][0] += r * w1;
            buf[i1][1] += g * w1;
            buf[i1][2] += b * w1;
            cov[i1] += w1;
        }
    }

    // Shards: free-flying, max-blended. Normalized pos → this strip's LEDs.
    for (int n = 0; n < CLICKY_NUM_SHARDS; n++) {
        ClickyShard &sh = clickyShards[n];
        if (sh.life <= 0.0f) continue;
        float fadeF = fmaxf(0.0f, sh.life / sh.maxLife);
        float r, g, b;
        hsvToRgb(hueWrap360(sh.hue + gHueRotDeg), 1.0f, 1.0f, r, g, b);
        int i = (int)lroundf(sh.pos * (float)(len - 1));
        for (int o = -1; o <= 1; o++) {
            int ii = ((i + o) % (int)len + (int)len) % (int)len;
            float w = fadeF * (o == 0 ? 0.4f : 0.125f);
            // Brightest-wins on the whole pixel, not per channel: a per-channel
            // max of two colors yields a hue that is neither of them, which
            // walks the shard off the arc.
            float sr = r * w, sg = g * w, sb = b * w;
            if (fmaxf(sr, fmaxf(sg, sb)) >
                fmaxf(buf[ii][0], fmaxf(buf[ii][1], buf[ii][2]))) {
                buf[ii][0] = sr; buf[ii][1] = sg; buf[ii][2] = sb;
                bufW[ii] = w; bufHue[ii] = sh.hue; bufPot[ii] = -1;
            }
            cov[ii] = fmaxf(cov[ii], w);
        }
    }

    // Paintbrush trail (per strip, since strips differ in length).
    bool trailHeld = pkt.btnBits & (1 << CLICKY_BTN_TRAIL);
    if (trailHeld) {
        float rate = dt * (CLICKY_PAINT_CAP / 255.0f) / CLICKY_PAINT_S;
        for (uint16_t i = 0; i < len; i++) {
            float add = fmaxf(fmaxf(buf[i][0], buf[i][1]), buf[i][2]) * rate;
            if (add <= 0.0f) continue;
            int8_t p = bufPot[i];
            if (p >= 0) {
                float gate = 1.0f - clickyMovedAgo[p] / CLICKY_PAINT_STOP_S;
                if (gate <= 0.0f) continue;
                add *= gate;
            }
            float lvl = clickyTrailLvl[s][i];
            float dh = fmodf(bufHue[i] - clickyTrailHue[s][i] + 540.0f, 360.0f)
                       - 180.0f;
            clickyTrailHue[s][i] = hueWrap360(clickyTrailHue[s][i]
                                              + dh * (add / (add + lvl + 1e-6f)));
            clickyTrailLvl[s][i] = fminf(CLICKY_PAINT_CAP, lvl + add);
        }
    } else {
        float decay = expf(-CLICKY_TRAIL_K * dt);
        for (uint16_t i = 0; i < len; i++)
            clickyTrailLvl[s][i] *= decay;
    }

    // Ruffle is applied ONCE here, over blobs + trail summed: a texture on the
    // combined field, so paint under a blob adds color without changing the
    // ripple depth. One shared carrier on the REF_LEN-normalized index — same
    // ripple count on a 50- and 150-LED strip, and it factors out of any
    // overlap instead of cancelling (measured 0.45 -> 0.22 depth with
    // per-blob phases).
    for (uint16_t i = 0; i < len; i++) {
        float ei = (float)i / (float)len * REF_LEN;
        float ruffle = 1.0f + CLICKY_RUFFLE_AMP
            * fastSin(ei * CLICKY_RUFFLE_FREQ + clickyRufflePhase);
        float bg = clickyTrailLvl[s][i] * (1.0f / 255.0f);
        float tr = 0.0f, tg = 0.0f, tb = 0.0f;
        if (bg > 0.0f)
            hsvToRgb(hueWrap360(clickyTrailHue[s][i] + gHueRotDeg), 1.0f, 1.0f, tr, tg, tb);
        float cr = buf[i][0] + tr * bg;
        float cg = buf[i][1] + tg * bg;
        float cb = buf[i][2] + tb * bg;
        if (clickyDesat < 0.999f) {
            float luma = 0.299f * cr + 0.587f * cg + 0.114f * cb;
            cr = luma + (cr - luma) * clickyDesat;
            cg = luma + (cg - luma) * clickyDesat;
            cb = luma + (cb - luma) * clickyDesat;
        }
        frameBuf[s][i][0] = cr * ruffle;
        frameBuf[s][i][1] = cg * ruffle;
        frameBuf[s][i][2] = cb * ruffle;
    }
}

// Rainbow burst from a knob gesture, aimed at the comb geometry. One shard
// per tooth (j = ±1..±5 at spacing CLICKY_PULL_NORM). outward = pull-out:
// shards launch from the particle and arrive at their tooth as they die —
// the motion blur of the split. !outward = push-in: shards spawn at the
// tooth positions and converge onto the particle. Hues span the wheel.
static void spawnRainbowBurst(float centerN, bool outward) {
    for (int m = 0; m < CLICKY_RAINBOW_BURST; m++) {
        ClickyShard &sh = clickyShards[clickyShardNext];
        clickyShardNext = (clickyShardNext + 1) % CLICKY_NUM_SHARDS;
        int j = (m / 2 + 1) * ((m & 1) ? 1 : -1);
        float toothN = (float)j * CLICKY_PULL_NORM;
        sh.hue = 360.0f * (float)m / (float)CLICKY_RAINBOW_BURST
                 + (randFloat() - 0.5f) * 20.0f;
        if (outward) {
            sh.maxLife = CLICKY_EMIT_LIFE * (0.8f + randFloat() * 0.4f);
            sh.pos = centerN;
            sh.vel = toothN / sh.maxLife;
        } else {
            sh.maxLife = CLICKY_ABSORB_LIFE * (0.8f + randFloat() * 0.4f);
            sh.pos = centerN + toothN;
            sh.vel = -toothN / sh.maxLife;
        }
        sh.life = sh.maxLife;
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
    bool desatHeld  = pkt.btnBits & (1 << CLICKY_BTN_DESAT);
    bool pumpHeld   = pkt.btnBits & (1 << CLICKY_BTN_PUMP);
    bool hueRotHeld = pkt.btnBits & (1 << CLICKY_BTN_HUEROT);

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

    if (hueRotHeld)
        gHueRotDeg = hueWrap360(gHueRotDeg + CLICKY_HUEROT_DEG_S * dt);

    if (pumpHeld) {
        float rise = dt * (CLICKY_PUMP_MAX - 1.0f) / CLICKY_PUMP_RISE_S;
        for (int k = 0; k < CLICKY_NUM_POTS; k++)
            clickyPump[k] = fminf(CLICKY_PUMP_MAX, clickyPump[k] + rise);
    } else {
        for (int k = 0; k < CLICKY_NUM_POTS; k++)
            clickyPump[k] = 1.0f + (clickyPump[k] - 1.0f) * expf(-dt / CLICKY_PUMP_TAU);
    }

    uint8_t pulledEdge = pkt.pullBits & ~clickyPrevPull;
    uint8_t pushedEdge = ~pkt.pullBits & clickyPrevPull;
    clickyPrevPull = pkt.pullBits;
    for (int k = 0; k < CLICKY_NUM_POTS; k++) {
        if (k == CLICKY_BRIGHT_POT) continue;
        if (pulledEdge & (1 << k)) spawnRainbowBurst(clickyPos[k], true);
        if (pushedEdge & (1 << k)) spawnRainbowBurst(clickyPos[k], false);
        if (pkt.pullBits & (1 << k))
            clickySplit[k] = fminf(1.0f, clickySplit[k] + dt / CLICKY_SPLIT_S);
        else
            clickySplit[k] = fmaxf(0.0f, clickySplit[k] - dt / CLICKY_SPLIT_S);
        float t = clickySplit[k];
        clickySplitEase[k] = t * t * (3.0f - 2.0f * t);
    }

    {
        float target = desatHeld ? CLICKY_DESAT_MIN : 1.0f;
        float da = fminf(1.0f, dt / CLICKY_DESAT_S);
        clickyDesat += da * (target - clickyDesat);
    }

    // Integrate shards once (rasterStrip reads them read-only). Positions
    // wrap: the field is a ring, so shards launched off one end come around.
    for (int n = 0; n < CLICKY_NUM_SHARDS; n++) {
        ClickyShard &sh = clickyShards[n];
        if (sh.life <= 0.0f) continue;
        sh.life -= dt;
        sh.pos += sh.vel * dt;
        sh.pos -= floorf(sh.pos);
    }

    // ── Rasterize onto every strip ────────────────────────────────
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        rasterStrip(s, STRIP_LEN[s], pkt, gatherT, meanPos, dt);

    if (clickyCrackleT > 0.0f) clickyCrackleT -= dt;
}

// ── Duck: audio features + rotate-to-change-color ───────────────
// Feature extraction and the renderers live in lib/duck_energy, shared
// with the viz twin. This file owns only the per-strip state
// and the wiring into the composed frame.
// AX_DUCK_SPARKLE picks the renderer on the SAME tuned features: 1 = onset
// sparkle bursts (original-duck look), 0 = the energy waterfall.
#define AX_DUCK_SPARKLE 0

static DuckFeatures duck;
static DuckSparkleCtl duckSp;

static float wfLevel[NUM_STRIPS][MAX_LEDS];
static float wfHue[NUM_STRIPS][MAX_LEDS];
static float wfTilt[NUM_STRIPS][MAX_LEDS];
static float wfDec[NUM_STRIPS][MAX_LEDS];

static void resetWaterfall() {
    memset(wfLevel, 0, sizeof(wfLevel));
    memset(wfHue, 0, sizeof(wfHue));
    memset(wfTilt, 0, sizeof(wfTilt));
    memset(wfDec, 0, sizeof(wfDec));
    duckSp.env = 0.0f;
}

static bool duckLive() {
    uint32_t now = millis();
    return duckLastMs != 0
        && (int32_t)(now - duckLastMs) < (int32_t)DUCK_TIMEOUT_MS;
}

static void duckUpdate(float dt) {
    DuckPacketV1 pkt;
    memcpy(&pkt, (const void*)&duckPkt, sizeof(pkt));
    bool wasCal = duck.calibrated;
    duckFeaturesUpdate(duck, pkt, dt, millis(), duckLive());
    if (duck.calibrated && !wasCal)
        Serial.printf("duck rest vector (%.3f, %.3f, %.3f)\n",
                      duck.restAx, duck.restAy, duck.restAz);
}

static void renderWaterfall(float dt, bool live) {
    memset(duckBuf, 0, sizeof(duckBuf));
#if AX_DUCK_SPARKLE
    float ignite = duckSparkleFrame(duckSp, duck, live, dt);
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        duckSparkleStep(wfLevel[s], wfHue[s], wfTilt[s], wfDec[s], STRIP_LEN[s],
                        duck, duckSp, ignite, dt, randFloat, duckBuf[s]);
#else
    float inject = duckWaterfallInject(duck, live);
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        duckWaterfallStep(wfLevel[s], wfHue[s], wfTilt[s], STRIP_LEN[s],
                          duck, inject, dt, randFloat, duckBuf[s], SEG[s].wfRev);
#endif
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);

    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        STRIP_LEN[s] = SEG[s].len;

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    strip0.Begin(); strip1.Begin();
    strip2.Begin(); strip3.Begin();
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0));
    strip2.ClearTo(RgbColor(0)); strip3.ClearTo(RgbColor(0));
    showAll();
    Serial.printf("boot drv=%s built=%s %s heap=%u\n",
                  OUT_DRV, __DATE__, __TIME__, ESP.getFreeHeap());

    for (int k = 0; k < CLICKY_NUM_POTS; k++)
        clickyPkt.pos[k] = (k == CLICKY_BRIGHT_POT) ? 10000 : 5000;
    resetClickyParticles();
    resetWaterfall();
    duckFeaturesReset(duck);
    kettleReset(kettle);
    kettleSetRings(kettle, AM_NUM_RINGS);
    axfxSapReset();
    sapRegisterSegs();

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_ps(WIFI_PS_NONE);

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

#if defined(AM_MOCKUP)
    Serial.printf("Axismundi MOCKUP (%u segs: 2 canopy chains + 2x helix+3roots) ready — ch=%u\n",
                  NUM_STRIPS, FIXED_CHANNEL);
#elif defined(AM_PLAYA_CANOPY)
    Serial.printf("Axismundi PLAYA CANOPY (4x rim Y-pair, bullets) ready — ch=%u\n", FIXED_CHANNEL);
#elif defined(AM_PLAYA_RAYS)
    Serial.printf("Axismundi PLAYA RAYS (spokes 0/2/4/6 + guys) ready — ch=%u\n", FIXED_CHANNEL);
#elif defined(AM_PLAYA_TRUNK)
    Serial.printf("Axismundi PLAYA TRUNK (helix + spokes 1/3/5 + guys) ready — ch=%u\n", FIXED_CHANNEL);
#elif defined(AM_PLAYA_ROOTS)
    Serial.printf("Axismundi PLAYA ROOTS (4x double-meander pole) ready — ch=%u\n", FIXED_CHANNEL);
#else
    Serial.printf("Axismundi (clicky, 4-strip) ready — ch=%u\n", FIXED_CHANNEL);
#endif
#if !defined(AM_MOCKUP) && (defined(AM_PLAYA_CANOPY) || defined(AM_PLAYA_RAYS) || defined(AM_PLAYA_TRUNK) || defined(AM_PLAYA_ROOTS))
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        Serial.printf("  seg %u: wire %u off %u len %u rgb %u guy %u\n",
                      s, SEG[s].strip, SEG[s].off, SEG[s].len, SEG[s].rgb, SEG[s].guy);
#elif defined(AM_MOCKUP)
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        Serial.printf("  seg %u: wire %u off %u len %u fold %u\n",
                      s, SEG[s].strip, SEG[s].off, SEG[s].len, SEG[s].fold);
#else
    Serial.printf("  strips: 13=%u 27=%u 32=%u 33=%u\n",
                  STRIP_LEN[0], STRIP_LEN[1], STRIP_LEN[2], STRIP_LEN[3]);
#endif
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / 60.0f);
    if (dt > 0.1f) dt = 0.1f;
    lastRenderMs = now;


#if DIAG_KETTLE_LOG
    while (kettleArrTail != (uint8_t)kettleArrHead) {
        KettleArrival &a = kettleArr[kettleArrTail & 63];
        kettleArrTail++;
        Serial.printf("KP t=%lu seq=%u enc=%ld\n",
                      (unsigned long)a.ms, a.seq, (long)a.enc);
    }
    static uint32_t lastRotLogMs = 0;
    if (now - lastRotLogMs >= 100) {
        lastRotLogMs = now;
        Serial.printf("RT t=%lu rot=%.4f pend=%.4f vel=%.3f fly=%d ctr=%d\n",
                      (unsigned long)now, kettle.rot, kettle.rotPending,
                      kettle.vel, (int)kettleHeld(kettle, KETTLE_BTN_FLYWHEEL),
                      (int)kettleHeld(kettle, KETTLE_BTN_COUNTER));
    }
#else
    kettleArrTail = (uint8_t)kettleArrHead;
#endif

    uint32_t renderT0 = micros();
    kettleUpdate(dt);
#if LOOP_RECORD
    lrTick(now, dt);
#endif
    duckUpdate(dt);
    bool clickyOn = clickyLastMs != 0
        && (int32_t)(now - clickyLastMs) < CLICKY_LIVE_MS;
    clickyRest += fminf(1.0f, dt / CLICKY_REST_TAU_S)
        * ((clickyOn ? 1.0f : 0.0f) - clickyRest);
    renderClickyParticles(dt * (CLICKY_REST_SPEED
        + (1.0f - CLICKY_REST_SPEED) * clickyRest));
    renderWaterfall(dt, duckLive());
    axfxSapFrame(kettleLastMs != 0
                 && ((kettlePkt.btnBits >> AXFX_SAP_BTN) & 1), dt);
    outputRotated();
#if LOOP_RECORD
    lrApply(now);
#endif
    uint32_t renderUs = micros() - renderT0;
    showAll();

    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    static uint32_t showSum = 0, showMax = 0, renderSum = 0, renderMax = 0;
    showSum += outShowUs;   if (outShowUs > showMax) showMax = outShowUs;
    renderSum += renderUs;  if (renderUs > renderMax) renderMax = renderUs;
    frameCount++;
    if (now - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (now - lastLogMs);
        Serial.printf("[rx] any=%lu drop=%lu lastLen=%u lastMagic=%04X"
                      " dropLen=%u dropMagic=%04X dropVer=%u\n",
                      (unsigned long)rxAnyCount, (unsigned long)rxDropCount,
                      rxLastLen, rxLastMagic,
                      rxDropLen, rxDropMagic, rxDropVer);
        Serial.printf("[out] drv=%s show=%lu/%lu render=%lu/%lu fill=%lu,%lu,%lu,%lu heap=%u\n",
                      OUT_DRV,
                      (unsigned long)(showSum / frameCount), (unsigned long)showMax,
                      (unsigned long)(renderSum / frameCount), (unsigned long)renderMax,
                      (unsigned long)outFillUs[0], (unsigned long)outFillUs[1],
                      (unsigned long)outFillUs[2], (unsigned long)outFillUs[3],
                      ESP.getFreeHeap());
        showSum = showMax = renderSum = renderMax = 0;
        bool clickyLive = clickyLastMs != 0 && (int32_t)(now - clickyLastMs) < 3000;
        bool kettleLive = kettleLastMs != 0 && (int32_t)(now - kettleLastMs) < 3000;
        bool duckIsLive = duckLive();
        Serial.printf("  FPS=%.1f  [clicky] %s pkts=%lu pull=%02X btn=%02X pos=%u,%u,%u,%u,%u,%u"
                      "  [kettle] %s pkts=%lu enc=%ld btn=%02X rot=%.3f"
                      "  [duck] %s pkts=%lu e=%.2f on=%.2f hue=%.0f tilt=%.2f cal=%d\n",
                      fps, clickyLive ? "LIVE" : "----",
                      (unsigned long)clickyPktCount,
                      clickyPkt.pullBits, clickyPkt.btnBits,
                      clickyPkt.pos[0], clickyPkt.pos[1], clickyPkt.pos[2],
                      clickyPkt.pos[3], clickyPkt.pos[4], clickyPkt.pos[5],
                      kettleLive ? "LIVE" : "----",
                      (unsigned long)kettlePktCount,
                      (long)kettlePkt.enc, kettlePkt.btnBits, kettle.rot,
                      duckIsLive ? "LIVE" : "----",
                      (unsigned long)duckPktCount,
                      duck.energy, duck.onset, duck.hueIdx, duck.tilt,
                      duck.calibrated ? 1 : 0);
        frameCount = 0;
        lastLogMs = now;
    }
}
