// engine.cpp — the live LED engine, compiled to WASM.
//
// This is the SAME effects.h the firmware and the host sim use — included, not
// copied (anti-drift). Compiled with Emscripten it runs the real effect math
// live in the browser (Three.js calls render(t) per frame) AND under Node (a
// headless script dumps frames), so one artifact replaces both the host sim and
// the matplotlib MP4.
//
// Plain C ABI, no Embind. Buffers are static and fixed-size; combined with a
// fixed WASM heap (no ALLOW_MEMORY_GROWTH) the Float32Array views JS holds over
// them never detach. We emit LINEAR RGB (0..1) and let the renderer apply the
// display transform — effects.h authors in linear; the device sink owns gamma.

#include "effects.h"   // pulls in geometry/topology.h (LED_POS, HC_NUM_LEDS)

extern "C" {

static RGBf g_buf[HC_NUM_LEDS];          // effect scratch (effects.h writes here)
static float g_rgb[HC_NUM_LEDS * 3];     // interleaved linear RGB, returned to JS
static float g_pos[HC_NUM_LEDS * 3];     // interleaved positions, returned to JS
static int   g_effect = 0;

int led_count()      { return HC_NUM_LEDS; }
int effect_count()   { return HC_NUM_EFFECTS; }
const char* effect_name(int i) {
    return (i >= 0 && i < HC_NUM_EFFECTS) ? HC_EFFECTS[i].name : "";
}
void set_effect(int i) { if (i >= 0 && i < HC_NUM_EFFECTS) g_effect = i; }
int  get_effect()      { return g_effect; }

// Geometry, exposed from the engine so positions have a single runtime source.
const float* positions() {
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        g_pos[i*3+0] = LED_POS[i].x;
        g_pos[i*3+1] = LED_POS[i].y;
        g_pos[i*3+2] = LED_POS[i].z;
    }
    return g_pos;
}

// Render the active effect at time t; returns a pointer to N*3 linear floats.
const float* render(float t) {
    HC_EFFECTS[g_effect].fn(g_buf, t);
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        g_rgb[i*3+0] = g_buf[i].r;
        g_rgb[i*3+1] = g_buf[i].g;
        g_rgb[i*3+2] = g_buf[i].b;
    }
    return g_rgb;
}

} // extern "C"
