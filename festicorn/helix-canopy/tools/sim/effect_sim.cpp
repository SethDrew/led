// effect_sim.cpp — host-compiled 3D ledsim for helix-canopy.
//
// Runs a REAL effect render fn (src/effects.h, included directly — not copied)
// against the generated geometry (geometry/topology.h) and dumps pixels over
// time to CSV on stdout:
//
//     frame,t_s,pixel,x,y,z,r,g,b      (r,g,b are 0..255, gamma-encoded)
//
// The x,y,z columns are what make this a TOPOLOGY-aware sim: the pandas
// analyzers and the 3D viewer can reason about light in space, not just by
// strip index.
//
// Usage:
//   ./effect_sim <effect> <fps> <seconds>  > frames.csv
//   ./effect_sim pour 30 6 > frames.csv

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "effects.h"   // pulls in topology.h; defines HC_EFFECTS registry

// Simple gamma 2.2 encode of linear 0..1 -> 0..255. (A real device would use
// the shared gammaLUT/delta-sigma; for geometry+motion QA this is enough. When
// the device output stage is chosen, swap this for the real lib include.)
static inline unsigned char enc(float lin) {
    if (lin < 0) lin = 0; if (lin > 1) lin = 1;
    return (unsigned char)(powf(lin, 1.0f / 2.2f) * 255.0f + 0.5f);
}

int main(int argc, char** argv) {
    const char* name = (argc > 1) ? argv[1] : "pour";
    float fps        = (argc > 2) ? atof(argv[2]) : 30.0f;
    float seconds    = (argc > 3) ? atof(argv[3]) : 6.0f;

    HcEffectFn fn = nullptr;
    for (int e = 0; e < HC_NUM_EFFECTS; e++)
        if (strcmp(HC_EFFECTS[e].name, name) == 0) fn = HC_EFFECTS[e].fn;
    if (!fn) {
        fprintf(stderr, "unknown effect '%s'. available:", name);
        for (int e = 0; e < HC_NUM_EFFECTS; e++)
            fprintf(stderr, " %s", HC_EFFECTS[e].name);
        fprintf(stderr, "\n");
        return 1;
    }

    static RGBf buf[HC_NUM_LEDS];
    int n_frames = (int)(fps * seconds);

    printf("frame,t_s,pixel,x,y,z,r,g,b\n");
    for (int f = 0; f < n_frames; f++) {
        float t = f / fps;
        fn(buf, t);
        for (int i = 0; i < HC_NUM_LEDS; i++) {
            printf("%d,%.4f,%d,%.4f,%.4f,%.4f,%u,%u,%u\n",
                   f, t, i,
                   LED_POS[i].x, LED_POS[i].y, LED_POS[i].z,
                   enc(buf[i].r), enc(buf[i].g), enc(buf[i].b));
        }
    }
    fprintf(stderr, "rendered %d frames x %d LEDs (effect=%s, %.0ffps)\n",
            n_frames, HC_NUM_LEDS, name, fps);
    return 0;
}
