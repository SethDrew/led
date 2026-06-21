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
static float g_brightness = 1.0f;        // global master, applied at output

int led_count()      { return HC_NUM_LEDS; }
int effect_count()   { return HC_NUM_EFFECTS; }
const char* effect_name(int i) {
    return (i >= 0 && i < HC_NUM_EFFECTS) ? HC_EFFECTS[i].name : "";
}
void set_effect(int i) { if (i >= 0 && i < HC_NUM_EFFECTS) g_effect = i; }
int  get_effect()      { return g_effect; }

void  set_brightness(float b) { g_brightness = b; }
float get_brightness()        { return g_brightness; }

// Live tunables — straight passthrough to the HC_PARAMS registry in effects.h.
int         param_count()        { return HC_NUM_PARAMS; }
const char* param_effect(int i)  { return (i >= 0 && i < HC_NUM_PARAMS) ? HC_PARAMS[i].effect : ""; }
const char* param_name(int i)    { return (i >= 0 && i < HC_NUM_PARAMS) ? HC_PARAMS[i].name : ""; }
float       param_lo(int i)      { return (i >= 0 && i < HC_NUM_PARAMS) ? HC_PARAMS[i].lo : 0.0f; }
float       param_hi(int i)      { return (i >= 0 && i < HC_NUM_PARAMS) ? HC_PARAMS[i].hi : 0.0f; }
float       param_get(int i)     { return (i >= 0 && i < HC_NUM_PARAMS) ? *HC_PARAMS[i].ptr : 0.0f; }
void        param_set(int i, float v) { if (i >= 0 && i < HC_NUM_PARAMS) *HC_PARAMS[i].ptr = v; }

// Geometry, exposed from the engine so positions have a single runtime source.
const float* positions() {
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        g_pos[i*3+0] = LED_POS[i].x;
        g_pos[i*3+1] = LED_POS[i].y;
        g_pos[i*3+2] = LED_POS[i].z;
    }
    return g_pos;
}

// Per-LED section kind (0=helix, 1=canopy, 2=root) from the strip table, so the
// viewer can size/dim each section independently without re-reading meta.json.
static int g_kind[HC_NUM_LEDS];
const int* kinds() {
    for (int s = 0; s < HC_NUM_STRIPS; s++)
        for (int j = 0; j < HC_STRIPS[s].count; j++)
            g_kind[HC_STRIPS[s].start + j] = HC_STRIPS[s].kind;
    return g_kind;
}

// Render the active effect at time t; returns a pointer to N*3 linear floats.
const float* render(float t) {
    HC_EFFECTS[g_effect].fn(g_buf, t);
    const float b = g_brightness;
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        g_rgb[i*3+0] = g_buf[i].r * b;
        g_rgb[i*3+1] = g_buf[i].g * b;
        g_rgb[i*3+2] = g_buf[i].b * b;
    }
    return g_rgb;
}

} // extern "C"
