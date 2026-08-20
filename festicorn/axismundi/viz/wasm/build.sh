#!/usr/bin/env bash
# build.sh — compile effects.h into the live WASM engine.
#
# Output engine.mjs + engine.wasm runs in BOTH the browser (Three.js viewer) and
# Node (headless frame dump) from a single artifact. Fixed heap (no memory
# growth) keeps JS's typed-array views over the pixel/position buffers from ever
# detaching — the one pitfall that silently zeros the LED colors.
set -euo pipefail
cd "$(dirname "$0")"

EMCC="${EMCC:-emcc}"

"$EMCC" engine.cpp ../../../lib/oklch_lut/oklch_lut.cpp -O2 -std=c++17 \
  -I../src -I../geometry -I../../src -I../../../lib/kettle_energy -I../../../lib/duck_energy \
  -I../../../lib/oklch_lut -I../../../lib/v1_telemetry -I../../../lib/hue_arc \
  -I../../../lib/fast_math \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,node \
  -sALLOW_MEMORY_GROWTH=0 -sINITIAL_MEMORY=16MB \
  -sEXPORTED_FUNCTIONS=_render,_positions,_kinds,_led_count,_effect_count,_effect_name,_set_effect,_get_effect,_set_brightness,_get_brightness,_pulse,_set_trunk_bright,_get_trunk_bright,_set_spin,_get_spin,_set_clicky,_set_kettle,_set_duck,_duck_probe_reset,_duck_probe_feed,_duck_const_count,_duck_const_name,_duck_const_value,_param_count,_param_effect,_param_name,_param_lo,_param_hi,_param_get,_param_set,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString,HEAPF32,HEAP32 \
  -o engine.mjs

echo "built $(pwd)/engine.mjs (+ engine.wasm)"
