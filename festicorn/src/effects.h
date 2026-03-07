#pragma once
#include <Adafruit_NeoPixel.h>

enum Effect {
    RAINBOW,
    GRADIENT
};

enum Palette {
    // Gradients
    SAP_FLOW,
    OKLCH_RAINBOW,
    RED_BLUE,
    CYAN_GOLD,
    GREEN_PURPLE,
    ORANGE_TEAL,
    MAGENTA_CYAN,
    SUNSET_SKY,
    // Chroma sweeps
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
