# mobile-c3

Portable battery-powered installation: two 50-LED WS2811 bulb strings on an
ESP32-C3, one momentary button for all control. Strip A shows a static color
wash; strip B runs an animated effect. All effects are standalone (no audio) —
the only external input is the biolum bulb-sender's IMU telemetry, consumed by
one effect. Firmware: `src/mobile.cpp` (env `mobile`, the default).

Wiring, pins, MAC, and strapping-pin hazards: `festicorn/catalog/boards.yaml`
(role `mobile-c3`, board MAC `14:63:93:70:42:AC`). Bulb color order is RGB —
owned by LED_LIBRARY.md in engineering/library.

## Controls (single button, GPIO 2)

| Gesture | Action |
|---|---|
| hold (≥400 ms) | brightness ramp, ~5 s full range; direction alternates each hold |
| single press | cycle 5 speeds (log-spaced 0.24–0.82, kaylab BS-26 ONES positions 1–7 band) |
| double click | next effect (strip B) |
| triple click | next wash (strip A) |

Boot defaults: bloom, pink_blue, speed 0.45×, brightness 128. Nothing
persists across power cycles.

GPIO 2 is a C3 strapping pin: do not hold the button while power is applied
or the board may not boot.

## Strip A — washes

Static gradients, cosine-eased spatially, interpolated in OKLab LCh
(shortest hue arc; out-of-gamut midpoints resolved by chroma reduction, never
per-channel clipping). Endpoints are sRGB in `WASH_ENDS`; the 50-px gradient
is rebuilt into `washBuf` on wash change only.

Order: pink_blue → teal_orange → green_indigo → lavender_crimson.

## Strip B — effects

All ported from sibling installations with audio reactivity stripped; each
name below states the source and what replaced the audio input.

| Effect | Source | De-audio substitution |
|---|---|---|
| bloom | tree-of-record `renderBloomStrip` | ambient breathing branch only; flash/dormancy machinery dropped |
| fire | tree-of-record `renderFire` | energy → slow layered-sine wander (base 0.68–0.88, color amber↔red ~1 min); onset/dropout dropped |
| nebula | tree-of-record `renderNebula` | none needed (was already audio-free) |
| light_through | tree-of-record `renderLightThrough` | none needed; radial 6-spoke geometry collapsed to 1D (patch center along strip, radii retuned 0.12–0.35) |
| gyro_pulse | neighborhood `wave_tap` (itself from biolum wave_pulse) | n/a — input is IMU telemetry, see below |

Two-clock invariant (from tree-of-record): motion scales with
`fxDt = dt × speed`; anything reacting to sensors (tap gate, refractory,
calibration) uses real dt. Output stage applies `fastGamma24` to the scalar
global brightness, never per-channel.

## gyro_pulse input

ESP-NOW receiver, fixed channel 6, accepts any 16-byte `TelemetryPacketV1`
(`lib/v1_telemetry/v1_packet.h`). Production source is a biolum bulb-sender
(MAC `10:00:3B:B0:F5:F8`, 25 Hz, ±4g). Tap = hard gate
peak_axis_ac_g ≥ 0.50 g + 120 ms refractory; taps spawn 1–4 pulses
(harder hit → more) at the strip end traveling toward LED 0. Tilt angle vs
the calibrated rest vector picks pulse hue (≤10° dead zone → default green).

Rest-vector calibration runs automatically on the first 2 s of packets after
boot; `c` over serial recalibrates. If the sender's resting orientation
changes mid-session, colors drift until recalibrated.

Taps are ignored in every other effect. The serial status line shows
`LIVE`/`----` for stream presence; a frozen `pkts=` counter means the sender
died (observed once after ~15 h: 1.3M packets then silence, cause not
isolated — sender battery suspected, receiver radio not ruled out).

## Build / flash

```bash
cd festicorn/mobile-c3
../../.venv/bin/pio run -e mobile -t upload
```

Identify the board by MAC before flashing — usbmodem suffixes reshuffle
(`esptool read_mac`). Serial 115200: 2 s status lines
(`fps … fx wash spd br pkts taps peak`), plus a line per control change and
per tap.

## Known state / open items

- Tap → pulse path has never been verified end-to-end on this board (sender
  was down during bring-up); everything upstream of the gate is verified.
- Settings persistence (NVS), ESP-NOW reinit watchdog, and auto-recalibration
  were considered and deliberately deferred until field use demands them.
- `src/rainbow_blue.cpp` (env `rainbow_blue`) is the retired bring-up
  firmware: strip smoke-test + gyro I2C scan + ESP-NOW channel sweep.
