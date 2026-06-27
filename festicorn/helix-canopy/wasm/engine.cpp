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
#include <string.h>    // strcmp (resample-spin effect gate)

extern "C" {

static RGBf g_buf[HC_NUM_LEDS];          // effect scratch (effects.h writes here)
static float g_rgb[HC_NUM_LEDS * 3];     // interleaved linear RGB, returned to JS
static float g_pos[HC_NUM_LEDS * 3];     // interleaved positions, returned to JS
static int   g_effect = 0;
static float g_brightness = 1.0f;        // global master, applied at output
static float g_trunk_bright = 1.0f;      // trunk-only dimmer (slider panel), output stage

int led_count()      { return HC_NUM_LEDS; }
int effect_count()   { return HC_NUM_EFFECTS; }
const char* effect_name(int i) {
    return (i >= 0 && i < HC_NUM_EFFECTS) ? HC_EFFECTS[i].name : "";
}
void set_effect(int i) { if (i >= 0 && i < HC_NUM_EFFECTS) g_effect = i; }
int  get_effect()      { return g_effect; }

void  set_brightness(float b) { g_brightness = b; }
float get_brightness()        { return g_brightness; }

// ── Bespoke control inputs (the per-installation widget bus) ───────────────
// Panel 1: momentary buttons launch a pulse (effect event into effects.h).
// Panel 2: a slider dims the trunk only (output-stage scale, all effects).
// (Panel 3, the spin dial, is a pure view transform and lives in the viewer.)
void  pulse(int btn, float t)      { hc_launch_pulse(btn, t); }
void  set_trunk_bright(float v)    { g_trunk_bright = v; }
float get_trunk_bright()           { return g_trunk_bright; }
// Panel 3, field-correct path: rotate the PATTERN (not the model) about the
// trunk axis. Effects that opt in sample at hc_pos(); spin is invisible on
// effects with no angular structure. Quantization is real (see effects.h).
void  set_spin(float rad)          { HC_SPIN = rad; }
float get_spin()                   { return HC_SPIN; }

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

// Per-LED kind lookup, built once from the strip table (0=helix/trunk). Used by
// the output stage so the trunk dimmer can scale only the trunk, in every effect.
static int  s_kind[HC_NUM_LEDS];
static bool s_kind_ready = false;
static void ensure_kind() {
    if (s_kind_ready) return;
    for (int s = 0; s < HC_NUM_STRIPS; s++)
        for (int j = 0; j < HC_STRIPS[s].count; j++)
            s_kind[HC_STRIPS[s].start + j] = HC_STRIPS[s].kind;
    s_kind_ready = true;
}

// ── Approach B: output-stage resample spin ─────────────────────────────────
// The wire-order originals (creatures, nebula) author in index space and
// have no field-correct hc_pos sampling, so HC_SPIN is invisible to them. To
// spin them faithfully we rigidly rotate the RENDERED image about the vertical
// axis and resample it back onto the fixed LED grid: output LED j shows the
// source pixel whose position, rotated by +HC_SPIN, lands on j — i.e. the
// source nearest to R(-HC_SPIN)*pos[j], which is exactly hc_pos(j). At spin 0
// this is the identity. The original look (run length, trunk energy, along-
// strand coherence) is preserved exactly because we replay g_buf untouched;
// the cost is a click as a lit blob crosses between discrete strands (the
// strands are sparse in azimuth — a topological snap, not a tuning artifact).
static bool effect_uses_resample(int e) {
    const char* n = HC_EFFECTS[e].name;
    return strcmp(n, "creatures") == 0 || strcmp(n, "nebula") == 0;
}

// Nearest source LED to a world point (brute force; rotation about z preserves
// (r,z) so the match stays within the same structural region by construction).
static inline int nearest_led(HcVec3 q) {
    int best = 0; float bd = 1e30f;
    for (int k = 0; k < HC_NUM_LEDS; k++) {
        const float dx = LED_POS[k].x - q.x, dy = LED_POS[k].y - q.y, dz = LED_POS[k].z - q.z;
        const float d = dx*dx + dy*dy + dz*dz;
        if (d < bd) { bd = d; best = k; }
    }
    return best;
}

// Render the active effect at time t; returns a pointer to N*3 linear floats.
const float* render(float t) {
    ensure_kind();
    HC_EFFECTS[g_effect].fn(g_buf, t);
    const float b = g_brightness;
    const bool resample = effect_uses_resample(g_effect) && HC_SPIN != 0.0f;
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        const int src = resample ? nearest_led(hc_pos(i)) : i;   // B: pull from rotated position
        const float bb = b * (s_kind[i] == 0 ? g_trunk_bright : 1.0f);   // trunk-only dimmer
        g_rgb[i*3+0] = g_buf[src].r * bb;
        g_rgb[i*3+1] = g_buf[src].g * bb;
        g_rgb[i*3+2] = g_buf[src].b * bb;
    }
    return g_rgb;
}

} // extern "C"
