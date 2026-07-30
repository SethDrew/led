# Axis Mundi

A world-tree LED sculpture: **roots** at the floor, a **canopy** overhead,
and (probably) a **center** — vertical trunk/post connecting them. This
directory holds everything: the installation firmware (`src/`), the Blender
wiring mockups (`wiring/`, with POWER.md as the running power design doc),
and `viz/` — the 3D model geometry, the visualizer twin of this firmware
(`viz/src/axismundi_fx.h`), and the WASM engine the console dashboard's
sculpture panel builds from. (These grew up in the retired helix-canopy
project and were folded in when its viewer was superseded by the dashboard.)

## Hardware

- Controller: led-eth-1 (ESP32, MAC A4:F0:0F:61:40:18) — the bench/prototype
  board; the three eth boards are the candidate pool for the final install.
- 4 × WS2812B strips, GRB, 150 LEDs each, GPIO 13/27/32/33, data over RJ45
  conductors used as plain wire.
- Strip lengths are per-strip configurable; the particle/effect field lives in
  normalized [0,1] space and rasterizes at each strip's own length.

Board facts live in the catalog (`festicorn/catalog/boards.yaml`) — that file
is the source of truth for MACs, pins, and roles.

## Inputs

All inputs standardized on **ESP-NOW broadcast, channel 1** (CAN considered
for the kettle knob and dropped for convenience). Each input sends full,
self-contained state packets; receivers gate on packet size + magic + version.
Any input may be absent — the sculpture renders gracefully with whatever
consoles are alive.

1. **clicky-pots** — 6 push-pull pots + 6 buttons console.
   `ClickyPacketV1` (18 B): drives the particle field (see clicky-pots README).
2. **kettle-knob** — salvaged infinite-spin water-heater encoder (30 detents,
   EC11) + 5 buttons on an ESP32-C3 SuperMini. `KettlePacketV1` (13 B):
   drives the energy/sparkle system and per-region waves (below).
3. **duck sender** — I2S mic + MPU-6050 on an ESP32-C3, sending at 25 Hz.
   `DuckPacketV1` (14 B): mic RMS mean/max (sqrt-companded to a byte each)
   plus a gravity direction vector. Drives the energy waterfall (below).
   Source lives here, in `src/duck_sender.cpp` (env `duck-sender`) — it is
   a different packet and a different installation from `original-duck/`,
   which stays frozen.

## Kettle-knob effect: the carousel

Spin turns the **carousel**: every strip is treated as a closed ring, and
the entire composed frame — clicky particles, trails, all of it — rotates
around each ring, signed with spin direction. One knob revolution carries
the frame one full trip around the ring. Rotation is a raster-time
transform (render in logical ring space, read output rotated), shared math
in lib/kettle_energy.

How each host wraps its geometry into rings is per-host: the bench firmware
wraps each 150-LED strip end-to-start; the visualizer wraps each of its
strips the same way. The canopy's artistic wrapping (it is a radial entity,
not a strip) is an open design question.

The earlier spin-energy sparkle system was removed (2026-07-28): carousel
rotation scatters logical-space spawns to arbitrary physical positions,
which read as glitches rather than an effect. The four panel buttons are
all hold-to-actuate motion modes — flywheel (spin charges velocity, picture
coasts), helical twist (rings fan apart), counter-rotation (alternate rings
reverse), and physical-space crackle. The knob press remains unassigned.

## Duck effect: the energy waterfall

Audio energy is injected at the head of every strip and advects along it,
decaying as it travels — loud passages push a bright front down the strip,
and silence leaves it dark: the noise gate holds the injection at zero until
sound clears it, so ambient room tone produces nothing at all. Level is
normalized by an adaptive floor
and a leaky ceiling, so the effect self-scales to the room instead of
needing a gain knob; a steady tone correctly reads as background and fades
to nothing. Onsets (fast vs slow EMA, with a refractory lock) shove an
extra slug of energy in, which is what reads as beat response.

**Rotate the duck to change colour**, mirroring the original duck's sparkle
interaction: the first 2 s at rest calibrates a rest vector, then the angle
away from it (10° deadzone, saturating at 180°) picks a hue from the OKLCH
LUT and blends it in over the next 40°. Rotation is a *colour* control
only — the water level owns brightness, so axismundi uses the constant-L
LUT rather than the variable-L one the original duck uses. The original
banks on an RGBW white channel to carry luminance; on these RGB-only
WS2812B strips, variable-L would make rotation read partly as dimming.

Shared math lives in `lib/duck_energy` so the firmware and the
visualizer twin in `viz/` cannot drift apart.

## Coexistence with clicky-pots

The kettle layer emits no light of its own; it rotates whatever the clicky
layer renders. Kettle alone shows a dark sculpture; clicky alone renders
unrotated only until the kettle has ever spun. The clicky brightness pot
(pot 5) remains the global brightness fader when present.
