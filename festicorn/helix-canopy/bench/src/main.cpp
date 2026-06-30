// Render-only benchmark: how fast does each helix-canopy effect fill the full
// HC_NUM_LEDS buffer on this chip? No LED output, no CAN — pure render cost.
// Single-threaded (one core), which is what a normal render loop uses.
//
// Split build: all 8 effects' static scratch arrays can't coexist in DRAM, so
// we reference only a subset of fx_ functions per build. --gc-sections then
// drops the unreferenced effects AND their statics. Two flashes cover all 8:
//   (default)      -> heavy effects     (nebula_az, creatures_az)
//   -DLIGHT_BATCH  -> the lighter six
#include <Arduino.h>
#include "effects.h"   // pulls topology.h: LED_POS, HC_NUM_LEDS, fx_*, ...
#ifdef GRID_BENCH
#include "nebula_grid.h"
#endif

struct BenchE { const char* name; HcEffectFn fn; };
static const BenchE EFX[] = {
#ifdef LIGHT_BATCH
  { "pour",        fx_pour },
  { "pulse",       fx_pulse },
  { "axis_radial", fx_axis_radial },
  { "nebula",      fx_nebula },
  { "line",        fx_line },
  { "creatures",   fx_creatures_lin },
#else
  { "nebula_az",    fx_nebula_native },
  { "creatures_az", fx_creatures },
#endif
};
static const int N_EFX = sizeof(EFX) / sizeof(EFX[0]);

static RGBf* buf = nullptr;   // heap, not .bss

void setup() {
  Serial.begin(115200);
  delay(1500);
  buf = (RGBf*)malloc(sizeof(RGBf) * HC_NUM_LEDS);
  Serial.println();
  Serial.println("=== helix-canopy effect render benchmark ===");
  Serial.printf("HC_NUM_LEDS=%d  CPU=%d MHz  buf=%s\n",
                HC_NUM_LEDS, getCpuFrequencyMhz(), buf ? "ok" : "MALLOC FAILED");
  if (!buf) return;

  // Exercise interaction state so pulse/creatures/nebula do real work.
  for (int b = 0; b < 3; b++) hc_launch_pulse(b, 0.5f);
  HC_SPIN = 0.7f;

  const int   WARM = 30;          // let stateful sims settle
  const int   ITER = 200;         // timed frames
  const float dt   = 1.0f / 60.0f;

#ifdef GRID_BENCH
  Serial.printf("grid self-test (brute vs grid, identical orbs): max abs diff = %.4f\n",
                neb_grid_selftest());
  struct GB { const char* name; HcEffectFn fn; } gb[] = {
    { "nebula_az (brute)", fx_nebula_native },
    { "nebula_az (grid)",  fx_nebula_native_grid },
  };
  Serial.println("effect               us/frame      fps");
  Serial.println("-----------------------------------------");
  for (int e = 0; e < 2; e++) {
    float t = 0.0f;
    for (int i = 0; i < WARM; i++) { gb[e].fn(buf, t); t += dt; }
    uint32_t t0 = micros();
    for (int i = 0; i < ITER; i++) { gb[e].fn(buf, t); t += dt; }
    uint32_t us = micros() - t0;
    Serial.printf("%-18s %9.1f %8.1f\n", gb[e].name, (float)us / ITER, 1e6f / ((float)us / ITER));
  }
  Serial.println("=== done ===");
  return;
#endif

  Serial.println("effect            us/frame      fps");
  Serial.println("--------------------------------------");
  for (int e = 0; e < N_EFX; e++) {
    HcEffectFn fn = EFX[e].fn;
    float t = 0.0f;
    for (int i = 0; i < WARM; i++) { fn(buf, t); t += dt; }
    uint32_t t0 = micros();
    for (int i = 0; i < ITER; i++) { fn(buf, t); t += dt; }
    uint32_t us = micros() - t0;
    float per = (float)us / ITER;
    float fps = 1e6f / per;
    Serial.printf("%-14s %9.1f %8.1f\n", EFX[e].name, per, fps);
  }
  Serial.println("=== done ===");
}

void loop() {}
