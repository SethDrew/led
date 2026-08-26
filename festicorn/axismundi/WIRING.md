# Axis Mundi — Playa LED Wiring (physical build spec)

What will physically be on playa, as decided 2026-08-20. Structural
dimensions come from the no-pull butterfly plan (side fixed 10 ft, R 138.3 in,
apothem 124.6 in, butterfly pushes each attach point 5.5 in outboard of its
corner, long spoke 140.8 in, coupling plate ⌀6 in). The machine-readable
source of truth for every derived number and LED position is
`viz/tools/geometry_gen.py` — change the plan there, never here.

## Structure

Regular heptagon canopy on webbing, one flat plane at 10 ft. No edge pull.
7 long spokes (hub coupling → butterfly attach points at the corners), 7 short
spokes (hub coupling → mid-edge). Center pole under the hub: 2 in diameter,
10 ft. 7 guys from the attach points down-and-out to ground anchors. 4
paper-mache root poles on the floor.

## LED products

| product | where | pitch | color order |
|---|---|---|---|
| WS2811 12 mm bullet strands, 50 nodes, 140 in lit | canopy (sectors + long spokes) | 72.6 mm | RGB (typical for bullets — **verify at bring-up**) |
| WS2812B IP67 strip, 5 m / 150 LED (30/m) | guys, helix, roots | 33.3 mm | GRB (**verify; wire order varies by product, see board catalog / ledger**) |

The two products have different color orders — the firmware output stage must
set color order **per segment**, and each new physical batch gets tested
before trusting the label.

## Runs (wire order)

- **8 canopy runs on 4 pins** — 2 × 50-node bullet strands each (280 in
  draped over a ~241.6 in path; the drape absorbs the slack). Two chains of
  four runs start at ONE short spoke (S0, the two runs ~5 cm apart) and wrap
  the rim in opposite directions; each run goes from the coupling edge out
  its own short spoke to the mid-edge, then one side-length of rim to the
  next short-spoke junction — so every short spoke carries one run and S0
  carries two. The two FINAL runs leave their last ~25 bulbs unlit so both
  chains terminate on the vertex diametrically opposite S0. JST Y-splitters
  pair each run with its mirror twin on one pin (4 pins × 2 strips): identical
  data on mirrored wire paths = whole-canopy reflection about the S0 axis.
  750 lit bulbs (6 × 100 + 2 × 75). How the renderer maps the field onto the
  spoke/rim segments lives in the fx twin (`ax_build_axfrac`), not here.
- **7 long-spoke + guy chains** — one 50-node bullet strand from the coupling
  edge out to the attach point (essentially taut: 140 in strand on a 140.8 in
  span), then ONE full 5 m / 150-LED WS2812B strip continuing down the guy.
  The ground anchor sits where the strip runs out (~12.9 ft past the attach
  point's floor projection). 200 px per chain, 1400 total.
- **Center helix** — WS2812B strips (same product as guys) wound around the
  center pole, phase-staggered, chained serpentine from the hub: down, then
  back up. Count is a generator parameter (`helix_strips`, currently 2 →
  ~24.6 turns each, 300 px).
- **4 roots** — one 5 m / 150-LED strip per pole, double-meander lay. 600 px.
  Driven by their own 4-output board (the 4×150 bench build).

Total ≈ 3000 px. Guys are a SAFETY layer: they hold a dim slow rainbow and
never participate in effects (see `ax_paint_straps` in the fx twin).

## Pin grouping — four 4-output boards, 16 pins, decided 2026-08-26

16 data lines land 1:1 on four 4-output eth boards (pins 13/27/32/33 each;
board identities assigned in the board catalog at deploy time). The only
splitters in the system are the canopy's mirror Y-pairs above. No data line
travels the trunk — every board drives locally.

| Board (site) | pin 13 | pin 27 | pin 32 | pin 33 |
|---|---|---|---|---|
| CANOPY (hub) | rim pair 1 | rim pair 2 | rim pair 3 | rim pair 4 |
| RAYS (hub) | spoke 0 | spoke 2 | spoke 4 | spoke 6 |
| TRUNK (hub) | helix serpentine | spoke 1 | spoke 3 | spoke 5 |
| ROOTS (base) | root A | root B | root C | root D |

Spokes are numbered from the vertex diametrically opposite S0 (the rim
seam) and alternate between RAYS and TRUNK, so a single board failure
darkens every other ray instead of a contiguous arc. All hub-board data
legs originate at the ⌀6 in coupling; the ROOTS board sits at the base PSU.
Per-pin wire lengths: canopy 100 px, spokes 200 px, helix 300 px,
roots 150 px.

## Open items

- Bullet-strand color order and strip color order unverified until the
  physical batches are tested.
