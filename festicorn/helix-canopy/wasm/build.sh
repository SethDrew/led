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

"$EMCC" engine.cpp -O2 -std=c++17 \
  -I../src -I../geometry \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,node \
  -sALLOW_MEMORY_GROWTH=0 -sINITIAL_MEMORY=16MB \
  -sEXPORTED_FUNCTIONS=_render,_positions,_led_count,_effect_count,_effect_name,_set_effect,_get_effect,_set_brightness,_get_brightness,_param_count,_param_effect,_param_name,_param_lo,_param_hi,_param_get,_param_set,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString,HEAPF32 \
  -o engine.mjs

echo "built $(pwd)/engine.mjs (+ engine.wasm)"
