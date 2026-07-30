#!/usr/bin/env bash
# build.sh — refresh the dashboard's WASM engine from axismundi/viz.
#
# This directory holds no engine source. It used to hold a full copy of
# axismundi_fx.h / effects.h / engine.cpp / topology.h, which silently drifted
# behind the canonical sources (missed the duck layer, the per-ring carousel and the
# hue-arc rework) while still compiling against the LIVE shared headers in
# festicorn/lib — so a lib API change broke a build nobody was watching.
# Delegating keeps both the sources and the emcc flags single-source.
set -euo pipefail
cd "$(dirname "$0")"

SRC=../../../../axismundi/viz/wasm

"$SRC/build.sh"
cp "$SRC/engine.mjs" "$SRC/engine.wasm" .

echo "synced $(pwd)/engine.mjs (+ engine.wasm) from axismundi/viz"
