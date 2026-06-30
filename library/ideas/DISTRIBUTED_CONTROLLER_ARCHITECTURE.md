# Distributed Controller Architecture — Design Exploration

**Status:** Exploration / pre-validated. Not built. Parked until there is a real
multi-controller system to design against. Captured from a working session
(June 2026) that started as two field bugs and turned into a scaling discussion.

**Relation to other ideas:** This is the *wired, multi-controller scale-up* of
the principle in REMOTE_JAM.md — *"don't stream LED data, stream the recipe and
cook locally."* The "central controller / edge computer" referenced here is the
kind of device described in toradex-reclaim.md (a reclaimed Apalis iMX6 as a
field LED orchestrator).

**Target topology under discussion:** a front panel (the kaylab/BS-26 knobs) →
a central controller (edge computer running the LED algorithms) → ~10 ESP32
"LED controllers," each driving 6 strips of ~150 LEDs (≈900 px/node, ≈9000 px
total) in real time.

---

## Part 1 — Two field bugs (the starting point)

### Bug A: freeze → dark → reboot under knob-spam

**Symptom:** when users changed many knobs at once, the LED animations would get
stuck, stop responding to knobs, then go dark and reboot. Looked like the LED
controller being overwhelmed.

**Diagnose by reset reason first — don't guess.** On every boot the ESP32 ROM
prints a reset reason (`rst:0x1 (POWERON_RESET)` on a clean boot). After a crash
it instead reports the cause — `Brownout detector was triggered`,
`TG0WDT_SYS_RESET` (watchdog), or `Guru Meditation Error` + backtrace (panic).
That one line collapses the guesswork into a category. The catch is reading it
live during a show, so the high-value addition is to **persist the last reset
reason** (RTC memory or a SPIFFS file) and print it at next boot, so a 2am crash
is still readable the next morning.

**Credible causes by inspection, distinguishable by that reset reason:**

1. **Brownout (power) — top suspect.** 6 strips × 150 = 900 WS2812 per node.
   Cranking brightness (`tens`) into a white-heavy effect/saturation spikes
   current; the PSU sags and the brownout detector resets the chip. Fits the
   symptom exactly: change knobs → LEDs lose power (go dark) → reboot.
2. **JSON parsing inside the ESP-NOW receive callback.** `onReceive` runs
   `deserializeJson` into a 768-byte `StaticJsonDocument` *in the radio
   callback's task context*, which has a limited stack. Knob-spam = a burst of
   JSON packets (the box sends on every change plus a heartbeat). Rapid parsing
   there risks a stack overflow or starving the WiFi task → "stops responding,
   then reboots." This smell is visible in the code today.
3. **A watchdog from something blocking** (SPIFFS / I2S) that only bites when it
   coincides with the flood.

**Reproducing it for TDD — several levels:**

1. **Reset-reason capture first.** One crash → the bucket.
2. **A packet flooder:** reflash a spare ESP32 (or temporarily the kaylab box)
   to broadcast bs26 JSON at 200–1000 pkt/s with randomized knob fields. Crashes
   with LEDs on a fixed *dim* effect → comms/handler path. Only crashes when
   brightness/white is also high → power.
3. **A fault-injection serial command** on the tree that synthesizes N random
   knob updates/sec straight into the handler, bypassing RF. Crashes under that
   → pure handler logic, bench-reproducible and unit-testable. Doesn't, but real
   RF flood does → RF/stack-context specific.
4. **Host-side unit test** of the pure logic (JSON decode → state struct →
   knob/effect state machine), compiled off-device and fuzzed with garbage
   payloads. Real TDD for the logic — but it can't catch power sag or RTOS
   timing faults; those need the on-device rig.

**Takeaway:** partly inspectable (the callback-JSON risk is real), but the reset
reason is the cheap oracle. Capture it before theorizing further.

### Bug B: ESP-NOW lossiness varied through the day

Expected behavior for the transport, not a bug. **ESP-NOW *broadcast* has no ACK
and no retransmit** — only *unicast* ESP-NOW gets link-layer acknowledgement +
automatic retries, and the duck broadcasts. So loss tracks 2.4 GHz conditions,
which swing through a festival day: crowds (human bodies are water → 2.4 GHz
absorbers), other APs/phones/Bluetooth, channel occupancy, and sender battery
sag. "Almost fully lossy sometimes, fine other times" is the signature of
best-effort broadcast in a crowd.

Cheapest win without new wiring: switch the audio stream to **unicast** (peer the
receiver's MAC) for MAC-layer ACK + auto-retransmit; pin the channel; disable
WiFi power-save (`esp_wifi_set_ps(WIFI_PS_NONE)`). Combined with seq-based
backfill on the receiver, that yields correct tempo *and* far fewer gaps.

---

## Part 2 — Wired coordination between controllers

### Why USB is the wrong tool

USB is host/device (master/slave), and the classic ESP32-D0WD has **no native
USB** — its port is a UART-to-USB bridge chip. A USB hub between two ESP32s gives
nothing peer-to-peer; you'd have to make one a USB *host*, which only the S2/S3
do natively and is a lot of complexity for no benefit. (USB *is* the right tool
in exactly one spot: front panel → edge computer, where the computer is a real
USB host and the panel is a device.)

### The decision that drives everything: where does rendering happen?

- **Central renders pixels, ESP32s are dumb frame-pushers** → Ethernet.
- **Central sends commands, ESP32s render locally** → RS-485 or CAN.

### The bandwidth math (decisive)

Per node: 900 px × 3 B (RGB) × 60 fps = **~1.3 Mbps**. Ten nodes = **~13 Mbps
aggregate**.

**RS-485 is one shared, half-duplex differential pair.** Every byte for every
node travels the same wire, one at a time. Line rate trades against distance:
~10 Mbps at <15 m (fragile — needs good transceivers, tight termination, short
stubs), ~1 Mbps at ~100 m (the safe, cheap-MAX485 default), ~100 kbps at >1 km.
Call it ~800 kbps usable at 1 Mbps after framing.

**Full-pixel streaming over one RS-485 bus — the hard limit:**
- 5400 px × 3 B = 16,200 B/frame → at 800 kbps ≈ **6 fps**.
- 9000 px × 3 B = 27,000 B/frame → ≈ **3.7 fps**.
- Addressing nodes individually doesn't help — the bus carries all nodes' bytes
  serially regardless. Brute-force options (10 Mbps short runs ≈ 37 fps but
  fragile; N parallel buses off N USB-485 dongles) converge toward "just use
  Ethernet."

So: **central computer streaming raw pixels to 9000 LEDs over one RS-485 bus →
no.** Fundamental to a shared serial bus.

### If you do want central pixel-render: Ethernet

```
 ┌──────────────┐    USB (computer = host)    ┌─────────────────────┐
 │ Front panel  │────────────────────────────▶│  Edge computer      │
 │ (kaylab ESP) │   knob state, low bandwidth  │  renders 9000 px    │
 └──────────────┘                              │  60 fps, audio FX   │
                                               └─────────┬───────────┘
                                                         │ Ethernet (DDP/sACN/Art-Net)
                                                  ┌──────▼──────┐
                                                  │  PoE switch │  (NOT a hub)
                                                  └──┬───┬───┬──┘
                                          ┌──────────┘   │   └──────────┐
                                     ESP32 #1        ESP32 #…        ESP32 #10
                                     6 strips        6 strips        6 strips
```

- **Switch, not hub.** A hub (shared collision domain) would collide all the
  traffic; a switch gives each port dedicated full-duplex bandwidth. 1.3 Mbps/node
  is ~1% of a 100M link.
- **PoE switch** powers each ESP32's *logic* over the same Cat5/6 run (one cable
  = data + board power). PoE does **not** power the LEDs.
- **ESP32 nodes need Ethernet:** integrated boards (Olimex ESP32-PoE /
  ESP32-PoE-ISO, or WT32-ETH01 with LAN8720), or a W5500 SPI-Ethernet module on
  a plain ESP32.
- **Protocol (all UDP):** **DDP** (cleanest for big custom pixel counts — no
  universe bookkeeping), **sACN/E1.31** (DMX-over-IP standard, 170 RGB px per
  universe so ~6 universes/node, has a sync packet), or **Art-Net** (similar,
  ArtSync). **WLED** on the nodes speaks all three out of the box, so you may not
  need custom driver firmware.
- **Sync:** use the protocol's sync packet so nodes flip buffers together; keep
  the switch on an isolated segment so other traffic doesn't jitter 60 fps.
- **Power is the real beast:** 9000 WS2812 at full white ≈ 540 A @ 5 V (2.7 kW);
  even ~20% duty ≈ 100 A. Local PSUs and power injection per node, fused, planned
  early. Independent of the data design.

---

## Part 3 — The compute-split (the recommended direction)

Send *parameters*, render *locally*. Collapses bandwidth ~1000×. Recognize that
the data has three classes with very different rates and audiences:

| Data | Rate needed | Audience | Size |
|---|---|---|---|
| Effect ID + params (brightness, speed, palette) | ~10 Hz (the "every 100 ms") | per-node | ~10 B |
| **Audio features** (energy, onset, bands, rms) | ~30–60 Hz (reactivity lags if slower) | **shared — broadcast once** | ~10 B |
| Sync tick (keep node clocks aligned) | ~1–10 Hz | broadcast | ~5 B |

Key correction to the naive "send everything every 100 ms": **audio features must
not ride the 100 ms channel** — reactivity needs ~50 Hz or sparkle/onset feels
mushy. And since the audio is identical for all nodes, **broadcast it once**, not
per-node.

**Total bus load, 10 nodes:**
- Audio broadcast: ~10 B × 50 Hz = **500 B/s** (shared)
- Per-node params: ~10 B × 10 Hz × 10 = **1,000 B/s**
- Sync: ~5 B × 10 Hz = **50 B/s**
- **≈ 1.6 KB/s ≈ 13 kbps → under 2% of a 1 Mbps bus.**

This is the existing kaylab→tree model (broadcast knob state, render locally)
scaled up. Real-time RMT timing/dithering stays on the ESP32; show logic + audio
+ per-node assignment move to the computer.

### Suggested packet format

Three message types, type byte first, little-endian:

```
AUDIO  (broadcast, ~50 Hz):  [0xA1][seq:1][energy:1][onset:1][band0..3:4][rms:1]              = 9 B
PARAM  (addressed, ~10 Hz):  [0xP1][nodeId:1][fxId:1][bright:1][speed:1][hue:1][sat:1][p0:1][p1:1] = 9 B
SYNC   (broadcast, ~5 Hz):   [0xS1][t_ms:4]                                                    = 5 B
```

- Companded uint8s for the audio fields — reuse the existing 5-byte
  `AudioPacketV1` envelope wholesale.
- `SYNC.t_ms` is the master clock; nodes phase-lock their effect timebase to it.
- `seq` on AUDIO lets nodes detect loss (apply the same backfill logic).

### Prefer CAN/TWAI over RS-485 for this

For a low-bandwidth, multi-node, noisy-environment *command* bus, the ESP32's
built-in **CAN (TWAI)** controller is the cleaner choice at the same ~$1
transceiver cost (SN65HVD230):

- Hardware framing, CRC, ACK, and **automatic retransmit** — error handling you'd
  otherwise hand-roll on RS-485.
- No DE direction-pin juggling (RS-485 half-duplex makes you flip TX/RX direction
  in software — a classic bug source).
- Built-in arbitration, so nodes can send telemetry back without a polling scheme.
- Multi-drop, 1 Mbps. Downside: classic CAN frames hold only 8 data bytes, so a
  9 B param = 2 frames — irrelevant at 13 kbps.

Pick **RS-485** instead only if you want the higher raw ceiling to occasionally
push *partial* pixel data (RS-485 at 1 Mbps leaves ~785 kbps of headroom; CAN
doesn't).

### The tradeoff

The param model can't express arbitrary per-pixel content — video, scrolling
text, one image mapped across all 9000 LEDs. If every effect is
generative/parametric (ours are), you lose nothing. True pixel-mapped content is
the one job that forces Ethernet; you could keep CAN/RS-485 for 99% of the show
and add Ethernet only for a pixel-mapped moment.

---

## Part 4 — Distributed per-pixel content without streaming pixels (shader / field model)

The insight that dissolves most of the "tradeoff" above: you *can* do globally
coordinated per-pixel content over the command bus **if each node knows where it
lives in the overall map and the effect is expressed as a function of space and
time rather than a buffer of pixels.** Stop thinking "buffer of pixels"; think
"the effect is `color(x, y, t, audio)`, and each node evaluates it only over the
pixels it owns." Per-pixel content with near-zero bandwidth, because pixels are
*derived*, not *transmitted*. Three ingredients:

### 1. The meta-index = a global coordinate map

Each node stores, once, a table mapping local pixel index → a position in a
**shared sculpture-wide coordinate system** (`x, y[, z]`). Static — measured at
install, loaded into flash/NVS, never sent with effect packets. The existing
hand-measured `STICKS[]` topology is the local seed of this; extend it so every
stick maps into one global frame with a single origin and axes (e.g. wind travels
along +x). The effect packet then carries **no map and no pixels** — just the
field's parameters; the node supplies the coordinates.

### 2. Effects as fields — the wind example, worked

Wind is the easy, perfect case because it's *stateless*:

```
front = speed * t_global                 // same number on every node for a given t
phase = pixel.x * k - front
value = noise(phase, pixel.y) * (0.3 + 0.7 * audio.energy)
color = palette(value)
```

Every node computes `front` from the same global clock, so the wind front is
continuous across node boundaries with zero coordination — node 7 needn't know
what node 6 drew, because they evaluate the identical function over adjacent
coordinates. Bandwidth stays ~13 kbps. Covers a huge swath: gradients, plasma,
scrolling rainbow, nebula, ripples-from-a-point, plane waves. The existing
`leaf-wind` is already this, phased per-strip; generalizing its phase to global
`x` *is* the upgrade.

### 3. Particle effects: deterministic shared simulation

Particles crossing the whole sculpture (gravity/polycule) seem to need handoff
between nodes — but don't, if the sim is **deterministic and replicated.** Every
node runs the *identical* particle simulation (same seed, same audio, same clock),
computes the full global particle state, and only *draws* the particles within its
own coordinate window. No handoff, no per-pixel data — just shared params + audio
+ clock. Cost is redundant compute, trivial for tens of particles.

General principle: **if global state is a pure function of (shared params, shared
clock, shared audio, fixed seed), every node can reproduce all of it and render
its slice.**

### The honest hard case: full-space PDEs

Heat diffusion and worley have state where a cell's next value depends on its
*neighbors' history*. At a node boundary the neighbor lives on another ESP32.
Three options, none as clean as the field model:

- **Halo exchange** — each node shares a thin strip of boundary values with its
  neighbors each frame (classic domain decomposition). Small data, but needs
  node-to-node routing and tight sync.
- **Replicate the whole PDE on every node** — works for fields/particles, but a
  full 9000-cell diffusion sim on every ESP32 is 10× redundant compute and the
  memory may not fit. Usually breaks down.
- **Keep PDEs node-local** — diffusion doesn't cross seams; heat becomes a
  per-region effect, not a sculpture-spanning one. Often fine artistically.

Stateless fields and replicated particle sims span the whole sculpture for free;
neighbor-coupled PDEs are the genuine exception. Don't let the wind case fool you
into assuming everything distributes cleanly.

### Sync becomes load-bearing

Once the wind front is `speed * t_global`, clock skew between nodes is a visible
seam. Budget: wind at 1 m/s with 30 px/m, 10 ms inter-node error = 0.3 px
mismatch (fine) — but free-running crystals *drift* seconds/hour, so resync
periodically. The master broadcasts `t_global`; each node runs a *disciplined*
local clock (track offset + rate, PLL-style) rather than free-running. On a
shared bus all nodes hear the broadcast within microseconds of each other, so
sub-millisecond alignment is very achievable.

### Prior art

**Pixelblaze** controllers run pattern *expressions* evaluated per-pixel using
each pixel's mapped 3D coordinate (`render(index)` with `pixelXYZ`), and its
**Firestorm** layer syncs multiple controllers' clocks for larger installs. Same
three ingredients — coordinate map, field function, shared clock. Worth studying
their map format and sync handshake before building.

---

## Part 5 — helix-canopy case study: render cost, the sim/render split, and master-on-a-node

**Status:** Reasoning from a June 2026 working session against the helix-canopy
sculpture (1697 px: 2× helix 594, 14× canopy 700, 7× root 403). **Everything in
this section is reasoning and design argument, NOT validated** — *except* the
render-fps measurements, which were taken on real hardware and are recorded in
the engineering ledger (`esp32-d0wd-helix-render-fps-2026-06-28`). Don't treat
the architecture conclusions below as confirmed until built.

### 5.1 The one measured input

A bench (ESP32-D0WD-V3, 240MHz, one core) rendering the full 1697-px buffer
clears 60 fps on 7 of 8 effects (124–950 fps); the two real-space particle-
deposition effects are the bottleneck (`creatures_az` 70, `nebula_az` 37 — the
latter fails 60). A voxel-grid rewrite of `nebula_az` measured 36→71 fps,
bit-identical output. Those numbers are the only validated facts here; the rest
is what we *infer* from them.

### 5.2 Why the heavy effects are slow (reasoning)

Not topology-awareness — `pour`/`pulse`/`axis_radial` use full 3D topology and
run 190–246 fps. The split is algorithmic:

- **Field (gather) effects:** `color[i] = f(position[i], t)`. Each LED
  independent, O(N), one pass. Cheap, and the ideal for distribution.
- **Scatter/deposition effects:** particles deposit onto nearby LEDs. In *index*
  space a particle touches a contiguous window (O(1) to find, cheap). In *real
  3D* space the nearby LEDs are an unstructured subset of the flat array, so with
  no spatial index every particle is tested against all N LEDs → O(particles×N).
  That nested loop is the cost (~100k distance evals/frame for `nebula_az`).

So it's fundamental (algorithmic), not sloppy code — but addressable: a voxel
grid (bin LEDs at cell = cutoff radius; each particle visits only its 3×3×3
neighbourhood) collapses O(P×N) → O(P×k). Rotating the ~56 orbs into the LEDs'
fixed frame (rotations preserve distance) also removes a per-frame 1697-LED trig
loop and lets the grid be built once on static `LED_POS`. Measured ~2× because
the residual O(N) work (background gradient, decay, compositing) becomes the
floor; slice-render (5.4) attacks that residual too.

### 5.3 Don't replicate the whole sim — broadcast the scene, render locally

The replicated-full-render model in Parts 3–4 (every node renders all 1697 px,
emits its slice) was a minimize-code-change expedient. It costs 3× redundant
render, the float-determinism fragility (needs bit-identical arithmetic on
identical chips), and a tight per-frame budget. The cleaner decomposition, for a
particle effect:

| stage | cost | shared/local |
|---|---|---|
| **simulate** (advance ~56 orbs: pos, size, brightness, life) | tiny, O(particles) | global — one scene |
| **render/deposit** (scene → per-LED colour) | expensive, O(LEDs) | per-LED, embarrassingly parallel |

The render splits because **each LED's colour depends only on (its position, t,
the particle list) — never on another LED.** So: master runs the sim once and
broadcasts the *scene* (t + params + particle list, ~1 KB/frame — state, not
pixels); each board renders only its own LED slice. This is the CPU→GPU split /
a render-farm tiling the image.

Payoffs (reasoned): kills the determinism gotcha (one authoritative sim, no
need for identical chips); divides render cost ~3× (worst `nebula_az` ~9 ms/board
even without the grid); and **reduces the sync burden** — orb positions are
*shipped*, not re-derived per board, so they stay consistent across boundaries
regardless of sub-ms clock skew. The clock still matters for the *field* parts
each board computes locally from t, and for interpolating between scene snapshots
if rendering above the broadcast rate.

Cost: a clean `effects.h` refactor — every effect becomes either a pure field
`f(position, t, params)` (loop over my slice) or `sim(t, params) → particles`
(master) + `deposit(particles) over my LED range` (node). Preserves the no-drift
doctrine: one source file, sim half on master, deposit half on nodes, host
sim/WASM runs both halves in one process.

### 5.4 What separates and what doesn't (reasoning)

The reframe works whenever the field is a **bounded superposition of localized
kernels with low-dimensional parameters** — linearity or point/local structure
gives a per-point evaluation that needs no neighbour state:

- particles = sum of blobs; **heat diffusion = sum of growing Gaussians** (linear
  + known heat-kernel Green's function: a source at `x₀,t₀` spreads as a Gaussian
  with `σ = √(2α·age)` and decaying peak — *architecturally identical to the orb
  effect*, just with age-varying σ/amplitude; a touch = a new source = a
  broadcast EVENT); **Worley = min distance to feature points** (point-native, no
  PDE, no neighbour history — broadcast the points, evaluate per LED); waves/
  plasma = sum of a few Fourier modes.

What genuinely resists is **nonlinear or globally-coupled dynamics**, where no
fixed-size superposition exists:

- **reaction-diffusion (Gray–Scott / Turing patterns)** — nonlinear local
  coupling; patterns emerge from interaction; needs the grid + neighbour stencil
  + halo exchange. (The aesthetically tempting one for a nature piece — this is
  the real exception, not heat/Worley.)
- **fluids (Navier–Stokes)** — pressure projection is a globally-coupled Poisson
  solve; particle methods (SPH, vortex particles) exist but reintroduce pairwise
  O(P²) coupling rather than removing it.
- **cellular automata** — discrete neighbour rules.

Dividing line: *does a point's value depend only on (shared scene + its own
coordinates)* → particle-able, broadcasts and slices for free; *or on the
evolving state of its neighbours* → stuck with grid + halo exchange. The current
helix palette is entirely on the easy side.

### 5.5 Master-on-a-node (revises the Part-4 Pi recommendation)

Given 5.1–5.3, an earlier "the ESP32 isn't strong enough to be master, use a Pi"
premise looks too conservative for this scale (reasoning, not built):

- The master's *extra* duties (broadcast SYNC/EVENT, hold params, run the cheap
  orb sim, queue scene broadcasts) are < 1 ms/frame. The expensive per-LED
  deposition is not a master duty — every node does its own slice.
- The D0WD is **dual-core**; the bench used one core. Layout: render on core 1,
  master duties on core 0. No need to render across two cores — single-core
  already clears 60 (and slice-render leaves large margin).

So: designate one render board as master, drop the Pi — **unless** audio lands
(≈50 Hz FFT/mel/onset could saturate the master's spare core → revisit a Pi) or
you want true pixel-mapped media (the one job that forces central pixel
streaming; bandwidth is a non-issue at 1697 px ≈ 2.4 Mbps, so streaming is also
*viable* here, just unnecessary for parametric/field content). SPOF is unchanged
either way; a silent-master failover on another node is possible if wanted.

---

## Open next artifacts (when there is a real system)

- Persistent reset-reason capture + the packet-flood / fault-injection repro rig
  (Bug A), as the first concrete deliverable regardless of architecture.
- Global coordinate map format (extending `STICKS[]` into one frame) + the
  node-side render loop that evaluates a field over its own coords against a
  disciplined clock.
- Exact CAN frame layout and bitrate budget for the real node count.
- Decision input still needed: can the duck/sources be physically tethered? If
  fixed near the tree → wire it and the RF-loss class disappears; if
  handheld/moving → harden ESP-NOW (unicast + retries + FEC) instead.
