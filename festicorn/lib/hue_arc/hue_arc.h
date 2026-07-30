// Hue helpers for the rotating clicky palette.
//
// Layers author a hue in degrees and add the shared rotation base to it, so the
// whole palette turns as one. Red-with-blue is handled where it belongs, in the
// choice of the six pot seeds: they sit inside a window narrower than the 75 deg
// from blue's high edge to red's low edge, so no rotation can put one seed in
// red while another is in blue. Nothing here enforces that; see CLICKY_HUES.

#ifndef HUE_ARC_H
#define HUE_ARC_H

#include <math.h>
#include <stdint.h>

static inline float hueWrap360(float h) {
    h = fmodf(h, 360.0f);
    return h < 0.0f ? h + 360.0f : h;
}

// Per-channel clipping is hue-breaking: when layers stack, one channel hits the
// ceiling first and drags the pixel off its hue. Roll off instead the way
// overexposure actually behaves -- hold the peak at the ceiling and desaturate
// toward white -- which scales all three channel differences by one factor and
// so leaves the hue exactly where the layer put it.
static inline void hueArcRolloff(float &r, float &g, float &b, float ceil) {
    float mx = fmaxf(r, fmaxf(g, b));
    if (mx <= ceil) {
        r = fmaxf(r, 0.0f); g = fmaxf(g, 0.0f); b = fmaxf(b, 0.0f);
        return;
    }
    float mn = fminf(r, fminf(g, b));
    if (mn < 0.0f) mn = 0.0f;
    float d = mx - mn;
    float k = (d > ceil) ? (ceil / d) : 1.0f;
    r = fmaxf(ceil - k * (mx - r), 0.0f);
    g = fmaxf(ceil - k * (mx - g), 0.0f);
    b = fmaxf(ceil - k * (mx - b), 0.0f);
}

#endif
