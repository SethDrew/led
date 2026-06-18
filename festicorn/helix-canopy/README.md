# helix-canopy

A "T"-shaped sculpture: a vertical **double helix** post rising to a **filled
heptagon canopy** ceiling (~10 ft up). Sim-first — designed and iterated in 3D
before any hardware exists.

## The iteration loop

```
  tools/geometry_gen.py      params -> geometry/{layout.json, topology.h, meta.json}
  src/effects.h              the REAL render math (hardware-independent)
  tools/sim/effect_sim.cpp   host-compiles effects.h + topology.h -> CSV with xyz
  tools/analyze.py           CSV -> topology-aware metrics (the PRIMARY loop interface)
  tools/view_geometry.py     CSV -> multi-angle PNG (gut-check a single frame)
  tools/animate.py           CSV -> MP4/GIF (watch the effect play over time)
  (later) device firmware    #includes the SAME effects.h + topology.h -> no drift

Most iteration questions are answered by analyze.py (numbers), not by looking.
The image/video tools are for the occasional "does this read right" gut-check.
```

### Anti-drift, by construction
Geometry math lives in exactly one place (`geometry_gen.py`); it's emitted as
literals. Effect math lives in exactly one place (`src/effects.h`); the sim
`#include`s it, never copies it. There is no second copy of either to rot — the
ledsim doctrine enforced structurally.

## Usage

```bash
VENV=../../.venv/bin/python          # from this dir; adjust depth as needed

# 1. generate / regenerate geometry (edit the P dict in geometry_gen.py first)
$VENV tools/geometry_gen.py

# 2. preview the bare geometry (color by strip or height)
$VENV tools/view_geometry.py --color height

# 3. build + run an effect, dump pixels-over-time with xyz
cd tools/sim && make && ./effect_sim pour 30 6 > pour.csv

# 4a. ANALYZE (the usual loop step) — answer questions from the data
cd ../.. && $VENV tools/analyze.py --frames tools/sim/pour.csv

# 4b. look at one frame, or watch the whole thing
$VENV tools/view_geometry.py --frames tools/sim/pour.csv --frame 45
$VENV tools/animate.py --frames tools/sim/pour.csv --glow   # writes pour.mp4
```

## Layout
- `geometry/` — generated geometry. `layout.json` (OPC-style), `topology.h`
  (firmware reads this), `meta.json` (params + strip ranges). PNGs are scratch.
- `src/effects.h` — render functions + effect registry. Source of truth.
- `tools/sim/` — host-compiled 3D ledsim. `effect_sim` + `*.csv` are scratch.

## Strips (multi-controller — ~1300 LEDs total)
`helix_A`, `helix_B` (244 each), `canopy` (821, serpentine fill). Strip ranges
into the flat `LED_POS[]` array are in `meta.json` / `HC_STRIPS[]`. The canopy
density makes this a multi-ESP32 install — see strip counts when planning wiring.
