# Input Availability UX — Design Options

Problem: Effects can require pot, IMU, v1 telemetry, or just mic audio. User picks an effect, the runtime accepts it silently, and broken effects render with stuck-default sensor values. There is no signal in the UI telling the user *what an effect needs* or *whether it's plugged in right now*. Two effects (`band_tendrils`, `jellyfish`) consume pot without declaring it.

The viewer already has a class filter bar (`audio-reactive` / `non-audio-reactive`) built on the brittle "starts with 'none'" heuristic in `isEffectAudioReactive` (app.js:1773). Any solution should retire that heuristic and key off `ref_interactivity` instead.

## Shared prerequisites (all options)

**Backend — new endpoint `/api/inputs/status`** polled at ~1 Hz from the UI. Returns:

```json
{
  "controller": {"noodles": "connected"|"disconnected", "strip": "..."},
  "audio":      {"state": "active"|"silent"|"missing", "device": "BlackHole"},
  "pot":        {"state": "available"|"stale"|"unavailable", "last_byte_ms": 230},
  "imu_duck":   {"state": "available"|"stale"|"unavailable", "last_packet_ms": 480},
  "v1":         {"state": "available"|"stale"|"unavailable", "last_packet_ms": 1200}
}
```

Implementation: `SerialLEDOutput` already parses the framed bytes (runner.py:380-411). Add a `last_seen_ms` per kind (`0xFC`, `0xFB`, `0xFA`); server polls runner state via the existing IPC/state file. Pot collapses to `unavailable` whenever noodles controller is disconnected (the single chokepoint named in the audit §3). `stale` = >1s without bytes while controller present. Pot can never be confidently "available" — best we have is "controller present + bytes flowing"; treat that as available, accept the "not moving" ambiguity.

**Effect metadata — backfill required regardless of option:**
- `band_tendrils.py`: add `ref_interactivity = 'hybrid'`, mention pot in `ref_input`.
- `jellyfish.py`: same; pot tweaks parameters.
- Add a structured `ref_inputs_required: list[Literal['audio','pot','imu_duck','v1','keyboard']]` to `base.py` so the UI doesn't have to parse `ref_input` prose. Populate per effect; `_discover_effects` ships it on the API payload.

Without these two pieces of work no UI option can be correct. Cost: ~1 hour of mechanical edits across ~10 effect files.

---

## Option A — Tab split + status cards

**Sketch.** Inside the Effects tab, two sub-tabs: `Audio` and `Interactive`. The Interactive sub-tab opens with a status strip across the top:

```
┌─ Inputs ──────────────────────────────────────────────┐
│ ● Noodles ctrl   ● Pot   ◌ Duck IMU   ✕ v1 telemetry  │
│   USB connected    flowing  silent 0.5s   not seen    │
└───────────────────────────────────────────────────────┘
[ effect cards filtered to ref_interactivity in {sensor,hybrid} ]
```

The Audio sub-tab shows only `ref_interactivity == 'audio'` effects and a smaller audio-only status line.

**Backend.** Endpoint above. No other server work.

**Frontend.** Replace the class-filter chip bar with a sub-tab control; render `InputStatusCard` component at the top of the Interactive sub-tab; filter `effectsList` by `ref_interactivity` rather than the string heuristic. `isEffectAudioReactive` deletes.

**Tradeoffs.** Cleanest mental model — "this whole tab is for hardware effects, here's what hardware you have." Cost: medium (new tab control + status card component). Teaches the user the input taxonomy. Downside: hybrids (`tilt_pendulum`) only live in one place; `visual`-only effects need a third home or get folded into Audio.

---

## Option B — Inline badges + gray-out

**Sketch.** Each effect card grows a row of small icon badges next to its name: 🎤 (audio), 🎛 (pot), 📐 (IMU), 📡 (v1), ⌨ (key). Badge is green when input is live, amber when stale, gray when unavailable. If any *required* badge is unavailable, the whole card grays out and becomes non-clickable; hovering shows a tooltip: *"Needs pot — noodles controller not connected."*

**Backend.** Same endpoint.

**Frontend.** Add badge row to each card in `renderEffectsCards` (app.js:2345). Compute `cardAvailable(eff, inputsStatus)`; bind a `disabled` class. Keep the existing class-filter bar but rewrite it as a multi-select: Audio / Pot / IMU / v1.

**Tradeoffs.** Most info-dense and most honest — user always sees *exactly* which inputs an effect wants and which are live. No effects hidden. Cost: small (one component per card). Downside: when most hardware is unplugged the grid is a sea of gray; teaches the input model but punishes the common case (mic-only sessions).

---

## Option C — Single list, collapse missing-hardware effects

**Sketch.** One flat list. Effects whose inputs are all satisfied render normally at the top. Effects that need missing hardware collapse into a folded section at the bottom:

```
[ playable effects … ]

▸ 6 effects need hardware that isn't connected
   ↳ pot_particle — needs pot, IMU  (noodles ctrl disconnected)
   ↳ rc_particle  — needs v1        (no v1 packets in 3s)
   …
```

Expanding the section lists them with reason text; clicking still launches (don't lock the user out of testing).

**Backend.** Same endpoint.

**Frontend.** Same filter computation as B, but instead of graying cards, partition the list and render the collapsed section. Smallest change to existing layout.

**Tradeoffs.** Lowest friction for the common mic-only case — confusing options disappear by default. Cheap to build (~1 evening). Downside: teaches the user the least — they may not realize the hidden effects exist. Doesn't surface *what* is connected, only *what's missing per effect*. Pair it with a small status pill in the header (`Inputs: mic ✓ · pot — · IMU —`) to recover some of that.

---

## Recommendation: **C, with the header status pill from A.**

This project is a creative tool used in two distinct modes — couch sessions with just the laptop mic, and on-site with the noodles controller and the duck. Mode C optimizes the common couch case (mic-only effects front and center, sensor effects out of the way but discoverable). The header status pill keeps Option A's pedagogical "here's what hardware you have right now" without spending a tab on it. Option B is the most "correct" answer but produces a worse default view for the most common session type and costs more to build. Sequence the work: backfill metadata + endpoint first (those are prerequisites for any option), ship C, and only escalate to B if users say the collapsed section is invisible.
