# LED viewer / IDE

Real-time, orbitable, scrubbable 3D playback of helix-canopy effects in the
browser. Two data sources behind one renderer:

- **`wasm` (live, default)** — `effects.h` compiled to WebAssembly runs the real
  effect math per frame, in the browser. Switch effects from the dropdown, no
  rebuild. This is the live IDE.
- **`frames` (precomputed)** — plays back a packed sim CSV. The Phase 0 fallback;
  also what you'd use to view a capture without building the engine.

Anti-drift holds in both: the colors come from the one `effects.h` (via WASM, or
via the host sim that produced the frames). The viewer only draws. The WASM
engine's output is byte-identical to the host C++ sim on deterministic effects.

## Layout

```
src/effects.h ─┬─► wasm/engine.cpp ──emcc──► wasm/engine.mjs + .wasm
               │                                   │        │
               │                          browser (live) ─┘        └─► node dump.mjs ─► CSV ─► analyze.py
               └─► tools/sim/effect_sim (legacy host sim, same CSV) ─────────────────────────────┘
geometry/layout.json ──► viewer/build_frames.py ──► viewer/data/  (frames-mode payload)
```

## Run it — hot-reload dev server (recommended)

```bash
cd ..            # helix-canopy root
node dev.mjs     # then open http://localhost:8777/viewer/
```

`dev.mjs` serves the tree, watches `src/effects.h` + `tools/geometry_gen.py`
(+ the wasm sources), rebuilds the WASM engine on save (~0.5s), and live-reloads
the browser — your camera is preserved across reloads, and build errors show in
the viewer overlay. Edit an effect, save, see it. A content hash gates rebuilds
so editor autosave/format touches that don't change bytes are ignored.

URLs: `?mode=wasm&fx=pour`, `?mode=frames` (precomputed fallback).

### Or static (no rebuild-on-save)

```bash
cd ../wasm && ./build.sh                 # build engine once
cd .. && python3 -m http.server 8777 -d . # then open /viewer/
```

Controls: drag to orbit, scroll to zoom, scrub the timeline, effect dropdown
(WASM only), toggle bloom/spin.

> Note: nebula keeps internal state across frames, so scrubbing backward in WASM
> mode advances the clock rather than rewinding the simulation. Deterministic
> effects (pour) are scrub-exact.

## Precomputed-frames mode

```bash
cd ../wasm && node dump.mjs nebula 30 20 > nebula_frames.csv   # WASM dump (or use tools/sim)
cd ../viewer && python3 build_frames.py --csv ../wasm/nebula_frames.csv --fps 30
# then open .../viewer/?mode=frames
```

## Headless capture (AI loop)

```bash
npm i playwright-core@1.49     # one-time; uses system Chrome, no browser download
node shoot.mjs --frame 300     # writes shots/*.png at several orbit angles
```

`shoot.mjs` drives the same viewer headless. The page exposes hooks for agents:
`window.__ledReady` (first frame composited), `window.__dumpColors()` (current
display-encoded buffer), `window.__setClock(t)` (pause + jump to time t). For a
pure data dump with no GPU, run `node ../wasm/dump.mjs` — same engine, CSV out.

## Generated (gitignored)

`data/`, `shots/`, `node_modules/` here; `engine.mjs`, `engine.wasm`, `*.csv` in
`../wasm/`. All regenerable from `build.sh` / `build_frames.py`.
