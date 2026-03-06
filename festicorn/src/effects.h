#pragma once
#include <Adafruit_NeoPixel.h>

enum Effect {
    RAINBOW,
    GRADIENT
};

enum Palette {
    // Legacy
    SAP_FLOW,
    OKLCH_RAINBOW,
    // Category 2: OKLCH hue-arc gradients
    RED_BLUE,
    CYAN_GOLD,
    GREEN_PURPLE,
    ORANGE_TEAL,
    MAGENTA_CYAN,
    // Category 3: OKLCH chroma sweeps
    BLUE_WASH,
    RED_WASH,
    GREEN_WASH,
    PURPLE_WASH,
    GOLD_WASH
};

struct EffectState {
    Effect effect = RAINBOW;
    uint8_t brightness = 128;  // 50%
    uint32_t cycleTimeMs = 8000;
    Palette palette = SAP_FLOW;
};

void renderRainbow(Adafruit_NeoPixel &strip, const EffectState &state);
void renderGradient(Adafruit_NeoPixel &strip, const EffectState &state);
uint8_t gammaHybrid(uint8_t v);
