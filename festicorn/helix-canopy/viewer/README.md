# LED viewer — Phase 0 (interactive playback)

Real-time, orbitable, scrubbable 3D playback of helix-canopy effects in the
browser. This is **Phase 0** of the LED IDE: it does *not* run `effects.h` live
yet — it plays back frames the existing host sim already produces. It exists to
replace the slow, non-interactive matplotlib MP4 with something you can orbit
and scrub today, and to stand up the headless screenshot harness the AI loop
needs. Phase 1 swaps the precomputed frames for `effects.h` compiled to WASM.

## Pipeline

```
effect_sim (tools/sim)  ──CSV──►  build_frames.py  ──►  data/scene.json + data/frames.bin
                                                              │
                                              index.html (Three.js) ◄── browser / Playwright
```

Anti-drift is preserved: the colors come from the real `effects.h` via the host
sim. Nothing about the effect math is reimplemented here — the viewer only draws.

## Run it

```bash
# 1. generate frames from the sim (once per effect / geometry change)
cd ../tools/sim && make && ./effect_sim nebula 30 20 > nebula_frames.csv

# 2. pack them for the browser
cd ../../viewer && python3 build_frames.py        # writes data/

# 3. serve + open
python3 -m http.server 8777 -d .                  # then open http://localhost:8777/
```

Use a different effect or framerate:

```bash
cd ../tools/sim && ./effect_sim pour 30 12 > pour_frames.csv
cd ../../viewer && python3 build_frames.py --csv ../tools/sim/pour_frames.csv --fps 30
```

Controls: drag to orbit, scroll to zoom, scrub the timeline, toggle bloom/spin.

## Headless capture (AI loop)

```bash
npm i playwright-core@1.49     # one-time; uses system Chrome, no browser download
node shoot.mjs --frame 300     # writes shots/frame300_*.png at several orbit angles
```

`shoot.mjs` drives the same viewer a human uses, headless, so an agent can grab
frames at chosen camera angles + timeline positions and reason about them (or
diff against the nominal MP4 stills). Render runs well above realtime.

## Generated (gitignored)

`data/`, `shots/`, `node_modules/` are build artifacts — regenerate freely.
