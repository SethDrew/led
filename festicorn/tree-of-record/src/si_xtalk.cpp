/*
 * si_xtalk — crosstalk / shared-ground-return discriminator for WS2812 data
 * over ethernet cable (led-eth-0 and the biolum-b control). Companion to
 * si_test.cpp, which drives IDENTICAL data on every line — a blind spot for
 * inter-line coupling, because correlated switching is common-mode.
 *
 * This test DEcorrelates the lines to expose ground-return bounce / crosstalk:
 *   SOLO-sK   strip K = max-transition aggressor (0xAA), all others BLACK.
 *             Does K glitch with its neighbors quiet? → K's OWN path
 *             (reflection / cross-domain ground), not coupling.
 *   VICT-sK   strip K = quiet victim (BLACK), all others = 0xAA aggressors.
 *             Does the quiet K light up from its neighbors switching? → pure
 *             inter-line crosstalk / shared-ground-return bounce.
 *
 * Read the pair per strip: clean SOLO but dirty VICT = coupling; dirty SOLO =
 * own-line. A BLACK victim is the most sensitive detector — a coupled pulse
 * misreads a 0→1 and lights a pixel out of pure darkness.
 *
 * Strip count + pins are build-flag parameterized (same scheme as si_test) so
 * this runs byte-identical on the ethernet rig and the direct-wired control.
 * Build: env:si-xtalk (led-eth-0) / env:si-xtalk-biolum (biolum-b).
 */

#include <Arduino.h>
#include <NeoPixelBus.h>

#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 150
#endif

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

// Aggressor = the 0xAA/0x55/0xAA max-transition color from si_test's TRANS-AA.
static const uint8_t AGG_R = 170, AGG_G = 85, AGG_B = 170;

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

// aggMask bit s set → strip s is an aggressor (0xAA), else BLACK.
static void fillPattern(uint8_t aggMask) {
    RgbColor agg(AGG_R, AGG_G, AGG_B);
    RgbColor off(0, 0, 0);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        RgbColor v = (aggMask >> s) & 1 ? agg : off;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) setPixel(s, i, v);
    }
}

// Phases generated per strip: for each K, SOLO-K then VICT-K.
//   phase p → K = p/2; even p = SOLO, odd p = VICTIM.
static const uint8_t NUM_PHASES = 2 * SI_NUM_STRIPS;
static const float    HOLD_S    = 10.0f;

static uint8_t aggMaskFor(uint8_t phase, char *nameOut) {
    uint8_t k    = phase / 2;
    bool    solo = (phase % 2) == 0;
    uint8_t full = (uint8_t)((1u << NUM_STRIPS) - 1);
    uint8_t mask = solo ? (uint8_t)(1u << k) : (uint8_t)(full & ~(1u << k));
    sprintf(nameOut, "%s-s%u", solo ? "SOLO" : "VICT", k);
    return mask;
}

void setup() {
    Serial.begin(115200);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) stripBegin(s);
    static const int PINS[6] = {SI_PIN0, SI_PIN1, SI_PIN2, SI_PIN3, SI_PIN4, SI_PIN5};
    Serial.printf("[si_xtalk] %u strips x %u LEDs, pins", NUM_STRIPS, (unsigned)LEDS_PER_STRIP);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) Serial.printf(" %d", PINS[s]);
    Serial.println();
    Serial.println("[si_xtalk] SOLO-sK: only K switches (own-path test). "
                   "VICT-sK: K is the quiet victim, rest blast 0xAA "
                   "(coupling test). Watch strip K.");
}

void loop() {
    static uint8_t  phase     = 0xFF;
    static uint32_t phaseEndMs = 0;
    static uint32_t frames     = 0;
    static uint32_t lastFpsMs  = 0;
    static char     name[16];

    uint32_t now = millis();

    if (phase == 0xFF || (int32_t)(now - phaseEndMs) >= 0) {
        phase = (phase == 0xFF) ? 0 : (uint8_t)((phase + 1) % NUM_PHASES);
        uint8_t mask = aggMaskFor(phase, name);
        fillPattern(mask);
        phaseEndMs = now + (uint32_t)(HOLD_S * 1000.0f);
        Serial.printf("[si_xtalk] phase %s  aggMask 0x%02X  watch strip %u\n",
                      name, mask, phase / 2);
    }

    bool allReady = true;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) allReady &= stripCanShow(s);
    if (allReady) {
        for (uint8_t s = 0; s < NUM_STRIPS; s++) { stripDirty(s); stripShow(s); }
        frames++;
    }

    if (now - lastFpsMs >= 1000) {
        Serial.printf("[si_xtalk]   %s  %lu fps\n", name, (unsigned long)frames);
        frames = 0;
        lastFpsMs = now;
    }
}
