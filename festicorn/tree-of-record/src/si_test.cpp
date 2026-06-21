/*
 * si_test — WS2812 signal-integrity / reflection stress test for led-eth-0
 * (LED data over an ethernet CABLE used as a plain conductor, not networked).
 *
 * Purpose: provoke data-line reflections as hard as possible and make any
 * resulting bit error trivially visible to the eye. Method = hold a DIM,
 * UNIFORM field at maximum refresh rate. A clean line shows a dead-solid
 * uniform color. A reflection that flips a data bit corrupts one pixel for
 * one frame → against the dim field it reads as a bright/wrong-colored SPARK.
 *
 * 4 × WS2812, 150 LEDs each, GPIO 13/27/32/33 (RMT 0..3). Build: env:si-test.
 *
 * Deliberately kept DIM (≤96/255) so glitches are unambiguously DATA errors,
 * not brownout. (Whole-strip dimming that tracks brightness = power, not
 * reflection — test that separately at full white.)
 *
 * Phases cycle automatically; each is a uniform field held for a few seconds
 * at full refresh. Phase name + measured fps print on serial so you know what
 * you're looking at and can correlate a spark to a phase.
 */

#include <Arduino.h>
#include <NeoPixelBus.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 150
#endif

// Strip count + pins are build-flag parameterized so the SAME source runs
// byte-identical on different boards (e.g. the ethernet rig led-eth-0 and the
// direct-wired control biolum-b). Override SI_NUM_STRIPS / SI_PIN0..5 per env.
// Defaults = led-eth-0 (4 strips on GPIO 13/27/32/33).
#ifndef SI_NUM_STRIPS
#define SI_NUM_STRIPS 4
#endif
#ifndef SI_PIN0
#define SI_PIN0 13
#endif
#ifndef SI_PIN1
#define SI_PIN1 27
#endif
#ifndef SI_PIN2
#define SI_PIN2 32
#endif
#ifndef SI_PIN3
#define SI_PIN3 33
#endif
#ifndef SI_PIN4
#define SI_PIN4 18
#endif
#ifndef SI_PIN5
#define SI_PIN5 19
#endif

static const uint8_t NUM_STRIPS = SI_NUM_STRIPS;

// All six RMT objects always exist (RMT channel is a compile-time template
// arg), but only the first NUM_STRIPS are Begin()'d / driven — the rest are
// inert (no hardware touched until Begin()).
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP, SI_PIN0);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, SI_PIN1);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, SI_PIN2);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP, SI_PIN3);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, SI_PIN4);
static NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, SI_PIN5);

static void stripBegin(uint8_t s) {
    switch (s) {
        case 0: strip0.Begin(); break;  case 1: strip1.Begin(); break;
        case 2: strip2.Begin(); break;  case 3: strip3.Begin(); break;
        case 4: strip4.Begin(); break;  case 5: strip5.Begin(); break;
    }
}
static bool stripCanShow(uint8_t s) {
    switch (s) {
        case 0: return strip0.CanShow();  case 1: return strip1.CanShow();
        case 2: return strip2.CanShow();  case 3: return strip3.CanShow();
        case 4: return strip4.CanShow();  case 5: return strip5.CanShow();
    }
    return true;
}
static void stripDirty(uint8_t s) {
    switch (s) {
        case 0: strip0.Dirty(); break;  case 1: strip1.Dirty(); break;
        case 2: strip2.Dirty(); break;  case 3: strip3.Dirty(); break;
        case 4: strip4.Dirty(); break;  case 5: strip5.Dirty(); break;
    }
}
static void stripShow(uint8_t s) {
    switch (s) {
        case 0: strip0.Show(); break;  case 1: strip1.Show(); break;
        case 2: strip2.Show(); break;  case 3: strip3.Show(); break;
        case 4: strip4.Show(); break;  case 5: strip5.Show(); break;
    }
}

// ── Test phases ──────────────────────────────────────────────────
// Each phase fills every pixel with one color. Rationale per phase:
//   DIM-GRAY-8   primary detector: any high-bit flip → bright spark.
//   TRANS-AA     0xAA/0x55/0xAA — maximizes 1↔0 transitions on the wire,
//                the worst case for edge reflections corrupting the next bit.
//   DIM-R/G/B-8  isolate which color channel's bits are fragile (tells you
//                where in the 24-bit word the eye is closing).
//   MID-GRAY-96  a bit more current/drive; sparks here but not at 8 hint at
//                a level/timing-dependent margin.
//   OFF-BLACK    all-zero frame: a flipped bit lights a pixel from pure
//                darkness — the most catchable single-pixel detector.
//   HEAD-AA      TRANS-AA color on only the first `head` LEDs, rest BLACK.
//                Same head signal as TRANS-AA but ~near-zero total current.
//                Power-vs-signal discriminator: if a head pixel still
//                glitches here, total current (power) is NOT the cause.
// Slimmed to the two phases that actually exercise the fault, found during
// bring-up of led-eth-0: the glitch is transition-driven (0xAA pattern), not
// brightness/current — HEAD-AA (12 LEDs, ~no current) still glitches, so it's
// signal integrity / reflection, not power. The dim/solid/per-channel and
// off-black phases never triggered and were removed. Keep this pair as the
// fast fix-verification loop: watch the intermittent flip on TRANS-AA, and
// re-confirm power is irrelevant on HEAD-AA.
struct Phase { const char *name; uint8_t r, g, b; float hold_s; uint16_t head; };
static const Phase PHASES[] = {
    {"TRANS-AA", 170, 85, 170, 30.0f, 0},   // full-strip max-transition trigger
    {"HEAD-AA",  170, 85, 170, 15.0f, 12},  // same head signal, ~no current
};
static const uint8_t NUM_PHASES = sizeof(PHASES) / sizeof(PHASES[0]);

static void setPixel(uint8_t s, uint16_t i, RgbColor v) {
    switch (s) {
        case 0: strip0.SetPixelColor(i, v); break;
        case 1: strip1.SetPixelColor(i, v); break;
        case 2: strip2.SetPixelColor(i, v); break;
        case 3: strip3.SetPixelColor(i, v); break;
        case 4: strip4.SetPixelColor(i, v); break;
        case 5: strip5.SetPixelColor(i, v); break;
    }
}

// head == 0 → fill the whole strip; head > 0 → first `head` LEDs get the
// color, the rest go black (low-current power-vs-signal discriminator).
static void fillAll(uint8_t r, uint8_t g, uint8_t b, uint16_t head) {
    RgbColor c(r, g, b);
    RgbColor off(0, 0, 0);
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        RgbColor v = (head == 0 || i < head) ? c : off;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) setPixel(s, i, v);
    }
}

void setup() {
    Serial.begin(115200);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) stripBegin(s);
    static const int PINS[6] = {SI_PIN0, SI_PIN1, SI_PIN2, SI_PIN3, SI_PIN4, SI_PIN5};
    Serial.printf("[si_test] %u strips x %u LEDs, pins", NUM_STRIPS, (unsigned)LEDS_PER_STRIP);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) Serial.printf(" %d", PINS[s]);
    Serial.println();
    Serial.println("[si_test] TRANS-AA / HEAD-AA at max refresh. Watch for "
                   "bright/wrong sparks against the flat field = bit errors.");
}

void loop() {
    static uint8_t  phase     = 0xFF;   // force a load on first iteration
    static uint32_t phaseEndMs = 0;
    static uint32_t frames     = 0;
    static uint32_t lastFpsMs  = 0;

    uint32_t now = millis();

    // Advance phase when its hold expires.
    if (phase == 0xFF || (int32_t)(now - phaseEndMs) >= 0) {
        phase = (phase == 0xFF) ? 0 : (uint8_t)((phase + 1) % NUM_PHASES);
        const Phase &p = PHASES[phase];
        fillAll(p.r, p.g, p.b, p.head);
        phaseEndMs = now + (uint32_t)(p.hold_s * 1000.0f);
        Serial.printf("[si_test] phase %s  rgb(%u,%u,%u)  head %u  hold %.0fs\n",
                      p.name, p.r, p.g, p.b, (unsigned)p.head, p.hold_s);
    }

    // Hammer the wire: re-send the (unchanged) buffer as fast as the strips
    // allow. More frames = more dice rolls for an intermittent reflection.
    // The buffer is only Dirtied on a phase change, so Show() would otherwise
    // no-op after the first frame and the wire would go silent — force Dirty()
    // every cycle, but only when all driven RMT channels have finished the
    // prior send, so each Show() is a real transmission (`frames` = honest fps).
    bool allReady = true;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) allReady &= stripCanShow(s);
    if (allReady) {
        for (uint8_t s = 0; s < NUM_STRIPS; s++) { stripDirty(s); stripShow(s); }
        frames++;
    }

    if (now - lastFpsMs >= 1000) {
        Serial.printf("[si_test]   %s  %lu fps\n",
                      PHASES[phase].name, (unsigned long)frames);
        frames = 0;
        lastFpsMs = now;
    }
}
