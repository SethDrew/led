#pragma once
#include <Adafruit_NeoPixel.h>

enum Effect {
    RAINBOW,
    GRADIENT,
    DEV_COLOR
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
    uint8_t chroma = 255;      // 255 = full color, 0 = greyscale
    uint32_t cycleTimeMs = 8000;
    Palette palette = SAP_FLOW;
};

extern uint8_t devColorBuf[NUM_PIXELS * 3];
extern bool devColorFresh;
extern float effectPhase;  // 0.0 to 1.0, accumulated in main loop

void renderRainbow(Adafruit_NeoPixel &strip, const EffectState &state);
void renderGradient(Adafruit_NeoPixel &strip, const EffectState &state);
void renderDevColor(Adafruit_NeoPixel &strip, const EffectState &state);
uint8_t gammaHybrid(uint8_t v);
