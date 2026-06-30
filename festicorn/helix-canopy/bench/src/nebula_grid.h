// Voxel-grid accelerated port of fx_nebula_native, for A/B benchmarking only.
// Lives in the bench, NOT in effects.h — this is an experiment.
//
// Two changes vs the original O(orbs x N) deposition:
//   1. Rotate the ~56 ORBS into the LEDs' fixed frame (rotation preserves
//      distance) instead of rotating all 1697 LEDs every frame. Kills the
//      per-frame rx/ry trig loop AND lets the grid be built once on static
//      LED_POS.
//   2. Bin LEDs into a voxel grid (cell = cutoff radius 3*sigma). Each orb only
//      visits the 3x3x3 cells around it -> O(orbs x k), k = local LED count.
#pragma once
#include "effects.h"

// ── static voxel grid over LED_POS (built once) ──────────────────────────────
namespace nbg {
  static const float CELL = 3.0f * 0.22f;        // = cutoff radius (sigma=0.22)
  static const int   MAXCELLS = 1280;            // actual grid ~850 cells
  static int   gnx, gny, gnz, gncells;
  static float gminx, gminy, gminz;
  static int   cell_start[MAXCELLS + 1];         // CSR offsets
  static int   led_order[HC_NUM_LEDS];           // LED indices sorted by cell
  static bool  built = false;

  static inline int cell_of(float x, float y, float z) {
    int ix = (int)((x - gminx) / CELL); if (ix < 0) ix = 0; if (ix >= gnx) ix = gnx - 1;
    int iy = (int)((y - gminy) / CELL); if (iy < 0) iy = 0; if (iy >= gny) iy = gny - 1;
    int iz = (int)((z - gminz) / CELL); if (iz < 0) iz = 0; if (iz >= gnz) iz = gnz - 1;
    return (iz * gny + iy) * gnx + ix;
  }

  static void build() {
    gminx = HC_BOUND_MIN.x - 0.01f; gminy = HC_BOUND_MIN.y - 0.01f; gminz = HC_BOUND_MIN.z - 0.01f;
    gnx = (int)((HC_BOUND_MAX.x - gminx) / CELL) + 1;
    gny = (int)((HC_BOUND_MAX.y - gminy) / CELL) + 1;
    gnz = (int)((HC_BOUND_MAX.z - gminz) / CELL) + 1;
    gncells = gnx * gny * gnz;
    static int count[MAXCELLS];
    for (int c = 0; c < gncells; c++) count[c] = 0;
    for (int i = 0; i < HC_NUM_LEDS; i++) count[cell_of(LED_POS[i].x, LED_POS[i].y, LED_POS[i].z)]++;
    cell_start[0] = 0;
    for (int c = 0; c < gncells; c++) cell_start[c + 1] = cell_start[c] + count[c];
    static int cursor[MAXCELLS];
    for (int c = 0; c < gncells; c++) cursor[c] = cell_start[c];
    for (int i = 0; i < HC_NUM_LEDS; i++) {
      int c = cell_of(LED_POS[i].x, LED_POS[i].y, LED_POS[i].z);
      led_order[cursor[c]++] = i;
    }
    built = true;
  }

  // Deposit one orb (already rotated into LED frame) using the grid.
  static inline void deposit_grid(float ox, float oy, float oz, float amp, float* led) {
    const float sigma = 0.22f, cut = 3.0f * sigma, cut2 = cut * cut;
    int cix = (int)((ox - gminx) / CELL), ciy = (int)((oy - gminy) / CELL), ciz = (int)((oz - gminz) / CELL);
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
      int ix = cix + dx, iy = ciy + dy, iz = ciz + dz;
      if (ix < 0 || iy < 0 || iz < 0 || ix >= gnx || iy >= gny || iz >= gnz) continue;
      int c = (iz * gny + iy) * gnx + ix;
      for (int k = cell_start[c]; k < cell_start[c + 1]; k++) {
        int i = led_order[k];
        float ddx = LED_POS[i].x - ox, ddy = LED_POS[i].y - oy, ddz = LED_POS[i].z - oz;
        float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
        if (d2 > cut2) continue;
        float g = expf(-d2 / (2.0f * sigma * sigma));
        float nv = led[i] + amp * g; led[i] = nv > 255 ? 255 : nv;
      }
    }
  }
  // Brute-force deposit (the original inner loop) — for the correctness self-test.
  static inline void deposit_brute(float ox, float oy, float oz, float amp, float* led) {
    const float sigma = 0.22f, cut = 3.0f * sigma, cut2 = cut * cut;
    for (int i = 0; i < HC_NUM_LEDS; i++) {
      float ddx = LED_POS[i].x - ox, ddy = LED_POS[i].y - oy, ddz = LED_POS[i].z - oz;
      float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
      if (d2 > cut2) continue;
      float g = expf(-d2 / (2.0f * sigma * sigma));
      float nv = led[i] + amp * g; led[i] = nv > 255 ? 255 : nv;
    }
  }
}

// Correctness: deposit a synthetic spread of orbs both ways, return max |diff|.
static inline float neb_grid_selftest() {
  if (!nbg::built) nbg::build();
  float* a = (float*)malloc(sizeof(float) * HC_NUM_LEDS);   // heap, not .bss
  float* b = (float*)malloc(sizeof(float) * HC_NUM_LEDS);
  if (!a || !b) return -1.0f;
  for (int i = 0; i < HC_NUM_LEDS; i++) { a[i] = 0; b[i] = 0; }
  uint32_t rng = 0x1234u;
  for (int o = 0; o < 60; o++) {
    float x = (xsf(rng) - 0.5f) * 6.0f, y = (xsf(rng) - 0.5f) * 6.0f, z = xsf(rng) * 4.5f;
    float amp = 100.0f + xsf(rng) * 53.0f;
    nbg::deposit_grid(x, y, z, amp, a);
    nbg::deposit_brute(x, y, z, amp, b);
  }
  float md = 0;
  for (int i = 0; i < HC_NUM_LEDS; i++) { float d = fabsf(a[i] - b[i]); if (d > md) md = d; }
  free(a); free(b);
  return md;
}

// Full effect, grid path + rotated orbs. Mirrors fx_nebula_native's sim exactly.
static inline void fx_nebula_native_grid(RGBf* out, float t) {
  if (!nbg::built) nbg::build();
  const float NEBULA_FPS = 30.0f;
  int maxOrbs = (int)((NEB_REF_ORBS * HC_NUM_LEDS / NEB_REF_PIXELS / 6) * NEB_density_nat);
  if (maxOrbs < 1) maxOrbs = 1;
  if (maxOrbs > NEB_MAXSLOTS) maxOrbs = NEB_MAXSLOTS;
  const float orbSize = NEB_orb_size;
  const float rise = NEB_rise_speed;
  float frame = t * NEBULA_FPS;
  float zspan = HC_BOUND_MAX.z - HC_BOUND_MIN.z;

  static bool init = false;
  static NebOrb3 orbs[NEB_MAXSLOTS];
  static float led[HC_NUM_LEDS];
  static uint32_t rng = 0xBADCAFEu;
  static float last_t = -1.0f, acc = 0.0f;
  if (!init) { for (int i = 0; i < NEB_MAXSLOTS; i++) orbs[i].active = false;
               for (int i = 0; i < HC_NUM_LEDS; i++) led[i] = 0; init = true; }

  const float BG_GAIN = 0.55f;
  for (int i = 0; i < HC_NUM_LEDS; i++) {
    float zn = (LED_POS[i].z - HC_BOUND_MIN.z) / (zspan > 1e-6f ? zspan : 1.0f);
    nebula_bg_pixel(zn, frame, out[i]);
    out[i].r *= BG_GAIN; out[i].g *= BG_GAIN; out[i].b *= BG_GAIN;
  }

  // rotate orbs into LED frame once per frame (was: rotate 1697 LEDs)
  float c = cosf(HC_SPIN), s = sinf(HC_SPIN);

  float decay = 1.0f - 1.0f / orbSize;
  int nsteps = neb_steps(t, NEBULA_FPS, last_t, acc);
  for (int st = 0; st < nsteps; st++) {
    for (int i = 0; i < HC_NUM_LEDS; i++) { led[i] *= decay; if (led[i] < 3) led[i] = 0; }
    int active = 0;
    for (int i = 0; i < NEB_MAXSLOTS; i++) if (orbs[i].active) active++;
    for (int i = 0; i < NEB_MAXSLOTS && active < maxOrbs; i++) {
      if (orbs[i].active) continue;
      if (xsf(rng) > 0.08f) continue;
      orbs[i].active = true;
      float ang = xsf(rng) * 6.2832f, rad = 0.30f;
      orbs[i].x = rad * cosf(ang); orbs[i].y = rad * sinf(ang);
      orbs[i].z = HC_BOUND_MIN.z + xsf(rng) * 0.3f;
      orbs[i].vx = (xsf(rng) - 0.5f) * 0.01f; orbs[i].vy = (xsf(rng) - 0.5f) * 0.01f;
      orbs[i].vz = rise * (0.7f + xsf(rng) * 0.6f);
      orbs[i].age = 0; orbs[i].life = 60.0f + xsf(rng) * 80.0f;
      active++;
    }
    for (int o = 0; o < NEB_MAXSLOTS; o++) {
      if (!orbs[o].active) continue;
      orbs[o].age += 1.0f;
      if (orbs[o].age >= orbs[o].life || orbs[o].z > HC_BOUND_MAX.z + 0.2f) { orbs[o].active = false; continue; }
      float orr = hypotf(orbs[o].x, orbs[o].y);
      if (orr > 1e-3f) { const float oacc = 0.0005f; orbs[o].vx += oacc * orbs[o].x / orr; orbs[o].vy += oacc * orbs[o].y / orr; }
      orbs[o].x += orbs[o].vx; orbs[o].y += orbs[o].vy; orbs[o].z += orbs[o].vz;
      float lc = orbs[o].age / orbs[o].life;
      float lb = (lc < 0.4f) ? smoothstep01(lc / 0.4f)
               : (lc > 0.6f) ? smoothstep01((1.0f - lc) / 0.4f) : 1.0f;
      // R(spin)*orb so we can compare against static LED_POS (= original's rx/ry compare)
      float ox = c * orbs[o].x - s * orbs[o].y;
      float oy = s * orbs[o].x + c * orbs[o].y;
      nbg::deposit_grid(ox, oy, orbs[o].z, lb * 153.0f, led);
    }
  }
  for (int i = 0; i < HC_NUM_LEDS; i++) if (led[i] > 2) star_add(out[i], led[i]);
}
