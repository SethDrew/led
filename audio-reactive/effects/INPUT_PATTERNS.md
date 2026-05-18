# Effect Input Patterns — Audit + Standardization Proposal

Read-only audit of how viewer effects consume physical inputs, plus a proposed canonical interface. Companion to `viewer/INPUT_AUDIT.md` (input map) and `viewer/INPUT_UX_OPTIONS.md` (UI). Source of truth for delivery wiring: `effects/runner.py:566-573`.

## 1. Per-input patterns observed

### 1.1 Pot (raw 0-1023)

Eight effects consume the pot. Six distinct patterns:

- **Position w/ EMA + absolute deadzone, no clamp** — `pot_particle.py:40-41` stores `pot_raw`; smoothing in `render` (`pot_particle.py:74-77`): `if abs(raw-smoothed) > 6.0: smoothed += (raw-smoothed)*alpha`. Then `position = (smoothed/1023.0)*(num_leds-1)`. Cleanest of the lot.
- **Delta-as-event (spawn/erase)** — `tilt_gravity.py:47-49` keeps `prev_pot` + `pot_raw`; render computes `delta = pot_raw - prev_pot` and triggers `_spawn_particle()` / `pop()` on `±15` threshold. Stateful, no smoothing.
- **Delta-as-agitation (kick + decay)** — `gas_crawlers.py:47-56` derives `agitation` from `|delta| > 8`, accumulates to 1.0, decays `*0.97` each frame. Two-rate EMA on color response (`render`, lines 76-78).
- **Direct rest-position** — `tilt_pendulum.py:36-37` uses `pot_raw/1023.0 * (num_leds-1)` raw, no smoothing. Spring physics absorbs jitter downstream.
- **Range-map for parameter** — `band_tendrils.py:201-204` `t = raw/1023.0; _pot_scale = 0.5 + t*7.5`. Used as a continuous knob. Same shape in deprecated `percussive_tendrils.py:143-147` which additionally derives `_pot_decay = 0.78 + t*0.18`.
- **Hue mapping with activation latch** — `jellyfish.py:206-209` `_pot_hue = raw/1023.0; _pot_active = True` (latches forever; never resets).

Divergences worth fixing:
- Deadzones use absolute counts (6, 8, 15) on raw 0-1023. Three different magnitudes encoding "same idea".
- Only `pot_particle` smooths; everyone else either eats jitter raw or relies on downstream physics to filter.
- `band_tendrils` and `jellyfish` declare neither `ref_input` nor `ref_interactivity` for pot — invisible to UI (confirmed in `INPUT_AUDIT.md` §2).
- Nothing handles "pot not present" — runner pushes default 512 unconditionally (`runner.py:349`, `:566-567`), so effects render mid-stuck silently.

### 1.2 Button / tap / explosion

Three different physical sources, three different code paths, zero shared abstraction:

- **Keyboard edge-trigger** — `pot_particle.py:50-52, 85-86`: tracks `keys`/`prev_keys`, fires on `'Digit1' in keys and 'Digit1' not in prev_keys`. Manual edge-detect on every consumer.
- **v1 `amag_max` threshold + cooldown** (single-mac) — `nebula_explosions.py:67-73, 130-139`: `set_v1_data` stores `amag_max`; render gates on `amag_max >= AMAG_TAP_THRESH and (elapsed - last_tap_time) > TAP_COOLDOWN_S`. Resets `amag_max = 0` after each render frame.
- **v1 multi-mac role assignment** — `rc_particle.py:57-86`: first MAC = controller (provides tilt), second MAC = trigger (provides tap via `amag_max`). Same threshold+cooldown pattern as nebula. Auto-role-assignment is unique to this effect.

Divergences:
- `AMAG_TAP_THRESH` is defined per-file. Not shared.
- Cooldown is per-effect.
- No abstraction unifies keyboard, pot-as-button, v1 tap — even though all three are "edge-triggered event".

### 1.3 IMU accel

Four effects, four nearly-identical implementations of "live high-pass via slow EMA baseline":

- `tilt_gravity.py:51-57`, `tilt_pendulum.py:39-46`, `gas_crawlers.py:58-64` — all identical: `raw_ax = ax/16384.0; if !baseline_ready: baseline=raw; baseline += (raw-baseline)*0.008; accel_x = raw - baseline`. **0.008 alpha is duplicated three times, with the same `~5s time constant at 25Hz` comment.** Strong copy-paste tell.
- `rc_particle.py:68-78` — different scheme: builds a tilt angle `atan2(ax, az)`, then high-passes the angle (not the component). Same `0.008` baseline alpha. Uses `v1` `ax_mean`/`az_mean` rather than raw `ax`.
- `pot_particle.py:43-48` — accel ignored; only consumes gyro magnitude.

Divergences:
- Three effects derive 1-axis gravity; one derives 2-axis tilt angle. Both valid roles, but no shared name.
- Nothing consumes a true gravity-vector (3-vec, low-pass).
- No effect handles missing IMU; default zeros silently freeze gravity at center.

### 1.4 IMU gyro

Two consumers, two formulations:

- `pot_particle.py:43-48`: `mag = sqrt(gx²+gy²+gz²) / 131.0` → degrees/sec. Mapped to hue: `min(gyro_dps/300, 1) * 0.65`.
- `rc_particle.py:80, 113-115`: uses v1 `gmag_mean` directly (already a magnitude on-device). Mapped to hue: `min(gmag/120, 1) * 0.65`.

Two different scaling constants (300 dps vs 120 raw) chosen empirically. No shared "rotation rate scalar 0..1".

### 1.5 v1 telemetry

`TelemetryPacketV1` parsed in `runner.py:392-410` into a 15-field dict:
`mac, seq, ax_max/min/mean, ay_max/min/mean, az_max/min/mean, amag_max, amag_mean, gmag_max, gmag_mean, flags`.

Field usage:
- `amag_max` — both v1 consumers (`nebula_explosions`, `rc_particle`) use it as tap detector.
- `ax_mean`, `az_mean` — `rc_particle` only, for tilt.
- `gmag_mean` — `rc_particle` only, for rotation hue.
- All other fields (min, ay_mean, amag_mean, gmag_max, flags, seq) — unused.

Staleness: `runner.py:570` guards `if led_output.v1_data is not None` but never clears it. Last frame persists forever after sender drops out.

### 1.6 Mic-derived signals

Per the audit, canonical pipeline lives in `signals.py`. Two outliers compute their own RMS in `process_audio` rather than consuming from a shared signal:
- `tilt_pendulum.py:48-49`: `rms = sqrt(mean(chunk**2))`
- `gas_crawlers.py:66-67`: same formula.

Both then smooth themselves. Not catastrophic — RMS is cheap — but it duplicates code and bypasses the EMA normalization landing elsewhere in the project (see open question in MEMORY.md about EMA inheritance).

## 2. Divergence summary

| Conceptual input | Most principled | Buggy / drift-prone | Pure duplication |
|---|---|---|---|
| Pot value | `pot_particle` (EMA + deadzone) | `jellyfish` (latch never resets) | three abs-deadzone constants |
| Pot delta | `gas_crawlers` (kick+decay) | `tilt_gravity` (raw delta, no debounce) | — |
| Pot range-map | `band_tendrils` (`0.5 + t*7.5`) | — | duplicated literal in deprecated file |
| Tap event | `nebula_explosions` (thresh+cooldown) | keyboard edge in `pot_particle` (manual) | `AMAG_TAP_THRESH` redefined in two files |
| Accel high-pass | tied; all three identical | `rc_particle` tilt-angle differs in semantics | 0.008 baseline alpha × 4 files |
| Gyro magnitude | `rc_particle` (uses v1 mag) | — | two scale constants chosen by hand |
| RMS | `signals.py` pipeline | `tilt_pendulum`, `gas_crawlers` (local recompute) | — |

## 3. Proposal — `effects/inputs.py` free functions (lighter path)

Steering note: user preference is for whatever is easiest to maintain and obvious to a future agent reading one effect cold. Drop the `InputContext` class proposal. Keep the runner exactly as it is (`set_pot_value` / `set_imu_data` / `set_v1_data` / `set_keys` setters survive). Move the duplicated EMA / threshold / scaling blocks into named free functions in a new `audio-reactive/effects/inputs.py`. Effects own their own state dicts; the helpers are stateless transforms.

The goal is **strictly mechanical de-duplication**, not an abstraction. A future agent should read `accel_high_pass(raw, state)` and know exactly what it does without opening the module.

```python
# audio-reactive/effects/inputs.py

# ── Constants ─────────────────────────────────────────────
POT_MAX = 1023
AMAG_TAP_THRESH = 130        # was duplicated in nebula_explosions, rc_particle
ACCEL_BASELINE_ALPHA = 0.008 # ~5s @ 25Hz — was duplicated 4×
GYRO_DPS_PER_COUNT = 1.0 / 131.0   # MPU-6050 ±250 dps full-scale

# ── Pot helpers (state dict owned by caller) ──────────────
def smooth_pot(raw, state, alpha=0.3, deadzone_raw=6.0):
    """EMA-smooth pot with absolute deadzone. state = {'smoothed': float}.
    Returns smoothed value in 0..POT_MAX domain. Caller divides if it wants 0..1."""

def pot_normalized(raw):
    """raw → 0..1, no smoothing. For effects whose downstream physics absorb jitter."""

def pot_scale(raw, lo, hi):
    """Linear range-map: lo + (raw/POT_MAX) * (hi-lo). Replaces `0.5 + t*7.5`."""

def pot_delta_kick(raw, state, threshold_raw=8):
    """Edge-detect on pot delta. state = {'prev': float}.
    Returns (delta, kick_strength_0_1) or (delta, 0.0) when below threshold.
    Caller picks: use as bool 'pressed', accumulate as agitation, etc."""

# ── IMU helpers ───────────────────────────────────────────
def accel_high_pass(raw_axis, state, alpha=ACCEL_BASELINE_ALPHA):
    """Track a slow EMA baseline, return raw - baseline.
    state = {'baseline': float, 'ready': bool}.
    Use: accel_x = accel_high_pass(data['ax']/16384.0, self._ax_state)"""

def gyro_dps_magnitude(data):
    """sqrt(gx²+gy²+gz²) / 131.0 → degrees/sec. From duck SensorPacket."""

# ── Tap / button helpers ──────────────────────────────────
def tap_edge(magnitude, state, threshold=AMAG_TAP_THRESH, cooldown_s=0.5, now=0.0):
    """Edge-triggered tap detector with cooldown.
    state = {'last_t': float}. Returns True once when fired."""

def key_edge(key, current_keys, prev_keys):
    """True iff key just transitioned from up to down."""
```

That's the whole module. Every function is one to six lines. No classes, no objects to construct, no lifecycle, no staleness flags (deferred — see §5).

**Effect file shape after migration** (example, `tilt_gravity.py`):

```python
from effects.inputs import accel_high_pass, pot_delta_kick

def __init__(self, ...):
    self._ax_state = {'baseline': 0.0, 'ready': False}
    self._pot_state = {'prev': 0.0}
    ...

def set_pot_value(self, raw):
    delta, _ = pot_delta_kick(raw, self._pot_state, threshold_raw=15)
    if delta > 15: self._spawn_particle()
    elif delta < -15 and len(self.particles) > 1: self.particles.pop()

def set_imu_data(self, data):
    self.accel_x = accel_high_pass(data.get('ax', 0) / 16384.0, self._ax_state)
```

Readable cold. No new types in scope. The 0.008 alpha lives in one place.

Keep the lightweight metadata field idea: add `ref_inputs_required` as a structured list (e.g. `['pot', 'accel']`) on the base class, distinct from the freeform `ref_input` human string. UI can introspect this later without parsing prose. Costs nothing now (just declare it on the ~10 sensor effects).

## 4. Migration — revised

All work is moving copy-pasted blocks into named functions. No interface changes.

**Step 1 — write `effects/inputs.py`** (~30 min). Eight short functions, plus the constants. No tests needed beyond a sanity import check; functions are trivial.

**Step 2 — convert effects** (~20 min each, ~3 hr total):
- `pot_particle.py` — `gyro_dps_magnitude`, keep keyboard edge as-is or use `key_edge`.
- `tilt_gravity.py` — `accel_high_pass`, `pot_delta_kick`.
- `tilt_pendulum.py` — `accel_high_pass`, `pot_normalized`.
- `gas_crawlers.py` — `accel_high_pass`, `pot_delta_kick`.
- `nebula_explosions.py` — replace local `AMAG_TAP_THRESH` import; use `tap_edge`.
- `rc_particle.py` — same, plus its own multi-mac role logic stays put.
- `band_tendrils.py:201-204` — `pot_scale(raw, 0.5, 8.0)`. Add `ref_input` declaration.
- `jellyfish.py:206-209` — `pot_normalized`. Add `ref_input` declaration.

**Step 3 — delete the duplicated constants** in `nebula_explosions:15` and `rc_particle:19`.

**Total: ~3-4 hours** including a careful read-through of each diff. Migration is per-file independent and trivially revertible; no shim needed.

What this does NOT do (deliberately):
- No staleness detection. The runner still pushes default 512/zero on disconnect. Adding staleness flags requires touching the runner and effect render paths together — a separate, larger change to scope later. Flagged in §5.
- No unified `button.pressed()`. Per §6 below, the three "tap" sources are genuinely different events and should keep distinct call sites; pretending they're one would mislead future agents more than help them.
- No InputContext, no metaclass tricks, no dependency injection.

---

## 5. Open questions — revised

1. **Staleness** — defer entirely? Or add a minimal `last_seen_ms` timestamp in the runner and let effects choose to react? Proposal: defer, it's its own change.
2. **`ref_inputs_required` metadata** — worth declaring now for the ~10 sensor effects, or defer to the UI design pass? Costs 1 line per file.
3. **RMS recompute in `tilt_pendulum`/`gas_crawlers`** — fold into this migration (add `compute_rms_from_chunk` to `inputs.py`), or leave until the EMA-band-signal refactor lands (per MEMORY.md open question)?
4. **Keyboard channel** — leave `key_edge` as an opt-in helper and let `pot_particle` keep its own pattern, or push every keyboard consumer to the helper? Only one effect uses keys today, so the duplication cost is zero.

---

## 6. Follow-up A — "Button" event sources

Three physical sources are loosely called "tap" or "button" across effects. The honest answer is they are **three different events** with different physics, latency profiles, and false-positive modes. Unifying them under `button.pressed()` would be the wrong call.

### 6.1 Keyboard key edge (`pot_particle.py:50-52, 85-86`)

- **Source:** stdin JSON from viewer (`runner.py:472-490`), GUI keyboard capture.
- **Semantics:** discrete, named (`Digit1`, etc.), carries a **hold state** (key remains in `_active_keys` while held). Edge detection is manual: `key in keys and key not in prev_keys`.
- **Latency:** ~1 render frame (~33 ms at 30 fps). Bounded by stdin read cadence.
- **Jitter:** none. Press is unambiguous.
- **False positives:** zero. The user actually pressed it.
- **What's unique:** named channels (multiple keys = multiple events), hold state, programmatic origin.

### 6.2 v1 accel-magnitude tap (`nebula_explosions.py:67-73, 130-139`, `rc_particle.py:82-86, 118-123`)

- **Source:** v1 sender computes per-window `amag_max` (sqrt-companded |accel|, gravity included, ±4g rail) on-device at 200 Hz inner sampling and emits 25 Hz; threshold of 130 corresponds roughly to a hard tap. See `festicorn/lib/v1_telemetry/v1_packet.h:23-25` and `gyro-sense/src/sender.cpp:91`.
- **Semantics:** **continuous magnitude exceeding a threshold**, not a discrete event. The sender does not classify — it streams stats. The "event" is a Schmitt-trigger receiver-side: `amag_max ≥ threshold AND elapsed - last_tap > cooldown`. There is no hold state — a sustained shake produces a tap every cooldown_s.
- **Latency:** sender window 40 ms + ESP-NOW + serial framing ≈ 50-80 ms typical, longer than keyboard.
- **Jitter:** windowed peak is stable, but the 40 ms window can split a sharp impulse across two packets and report a lower peak in each. False negatives on edge-aligned taps.
- **False positives:** rotating the controller hard while holding it can clear 130 counts on `amag_max` because gravity is included in `amag` (see the bug retro in `v1_packet.h:10-16`). Hard accelerations from walking can also fire it.
- **What's unique:** physical, embodied, no name (any motion above threshold fires). Cooldown is not debounce — it's rate-limiting because there's no down-edge.

### 6.3 Pot delta as event (`tilt_gravity.py:62-70`, `gas_crawlers.py:47-56`)

- **Source:** runner reads pot at frame rate; effect compares to previous frame.
- **Semantics:** **rate-of-change exceeding a threshold**. Closer to a velocity sensor than a button. In `tilt_gravity` the *sign* of delta matters (positive spawns, negative pops); in `gas_crawlers` only magnitude matters (kick agitation).
- **Latency:** one frame (~33 ms).
- **Jitter:** pot ADC noise without smoothing produces ~1-3 count jitter, well below the 8/15 thresholds used — fine. No false positives in practice.
- **False positives:** dropping the controller while turning the knob → continuous kicks. Not really wrong; matches the physical action.
- **What's unique:** continuous, signed, and the user can hold the knob mid-turn (state is the knob position, not the event).

### 6.4 Recommendation

**Keep them separate**, not unified. They are three different things:

- **Keyboard** = programmatic / debug trigger with hold state.
- **v1 tap** = physical impulse, rate-limited Schmitt trigger, false-positive-prone, no hold.
- **Pot delta** = signed velocity gate, no false positives, naturally rate-limited by user.

What a future agent who sees `button.pressed()` and assumes keyboard semantics would get burned by, in order of severity: (1) v1 tap firing repeatedly during sustained motion when they expected one-shot; (2) v1 tap firing during rotation (gravity-included amag); (3) pot delta carrying sign information that's been thrown away.

What `inputs.py` should expose:
- `key_edge(key, current, prev)` — for keyboard
- `tap_edge(magnitude, state, threshold, cooldown_s, now)` — for v1 / IMU magnitudes
- `pot_delta_kick(raw, state, threshold)` — returns `(delta, kick)`, caller picks how to use

Three named helpers, three call sites in the effect. A future agent reading `tap_edge(self.amag_max, ..., now=self.elapsed)` knows immediately this is a thresholded magnitude with cooldown, not a key press.

---

## 7. Follow-up B — Duck (v0.1) vs v1 sender

### 7.1 What they are

- **Duck (`festicorn/original-duck/src/sender.cpp`)** — raw-stream sender. 15-byte `SensorPacket`: `int16 ax,ay,az,gx,gy,gz; uint16 rawRms; uint8 micEnabled`. ESP-NOW @ 25 Hz. Includes an INMP441 mic (RMS computed on-device) and a shake-to-toggle-mic UI gesture. Runner CMD `0xFB`.
- **v1 / gyro-sense (`festicorn/gyro-sense/src/sender.cpp` + `lib/v1_telemetry/v1_packet.h`)** — aggregated-stats sender. 16-byte `TelemetryPacketV1`: per-axis max/min/mean (AC-coupled, gravity preserved in mean), magnitude max/mean (sqrt-companded), clip-flags. Sampled at 200 Hz internally, emitted at 25 Hz. Range locked at ±4g / ±1000 dps with on-boot variance-gated gyro bias calibration. No mic. Runner CMD `0xFA`.

### 7.2 Why both are still in flight

The v1 schema is genuinely newer and more capable for tap detection (200 Hz inner sampling captures impulses the duck's 25 Hz raw read smears), but the duck has the mic. Neither has been declared deprecated. No effect consumes both — `rc_particle` and `nebula_explosions` use v1; `pot_particle`, `tilt_gravity`, `tilt_pendulum`, `gas_crawlers` use duck IMU. They are addressing different hardware (the literal duck plush vs. the bulb-style gyro-sense boards).

Quick check: nothing consumes both `set_imu_data` and `set_v1_data` in the same effect (grep of `effects/`). They're not "two flavors of the same data" inside any single effect — they're two different physical devices.

### 7.3 Semantic overlap

- **Tilt:** `rc_particle` uses `atan2(v1.ax_mean, v1.az_mean)`. Could be done from duck `ax, az` raw — yes, but the duck's raw 25 Hz read has no inner-window averaging, so the tilt would be noisier. The v1 `ax_mean` is essentially a free 200 Hz → 25 Hz boxcar.
- **Gyro magnitude:** `pot_particle` uses `sqrt(gx²+gy²+gz²)/131` from duck. Could use v1 `gmag_mean`. v1's value is sqrt-companded (non-linear), bias-corrected, sampled at 200 Hz — better. But the duck has lower hardware cost (it's the same plush already in the user's hand).
- **Tap:** v1 wins decisively. The duck's 25 Hz raw IMU read cannot reliably catch <10 ms impulses; the v1 200 Hz inner sampling can, and emits the per-window peak. A duck-based tap detector would miss most real taps.
- **Mic RMS:** duck only. v1 has no audio path (audio deferred per v1 sender comment).

### 7.4 Is on-device aggregation buying us anything?

Yes, three things, in decreasing order of importance:

1. **Impulse capture at 25 Hz wire rate** — `amag_max` over an 8-sample inner window catches single-frame events that would otherwise be missed entirely. This is the core reason v1 exists; the duck cannot do this.
2. **Bias correction and DLPF tuning on-device** — variance-gated gyro bias calibration (3 s of qualified-still samples auto-detected) and the DLPF=184 Hz BW choice are baked into the sender. A raw-stream alternative would need to replicate this on the laptop.
3. **Bandwidth budget** — 16 B vs 15 B at 25 Hz is a wash. Not a real factor.

Counter-arguments for raw streaming:
- Laptop has more compute; could compute richer features (centroid, zero-cross, etc.) the sender can't.
- A single representation is simpler to reason about.

Verdict: keep both. They serve different physical devices and different sensing tasks. The duck does "always-on cuddly object that has a mic" and the v1 does "tap-capable bulb with no audio." Trying to retrofit one onto the other costs more than the duplication.

### 7.5 Recommendation

- **Status quo on firmware.** Both senders stay.
- **Document the roles in the runner.** Add a short comment at `runner.py:383-411` stating: duck = raw-stream IMU+mic, low cadence; v1 = aggregated stats with tap-grade impulse capture, no mic.
- **Don't try to share parser code.** They're different packets with different semantics — sharing would create more confusion than it saves.
- **Effects choose by sensor role**, not by source format. `tap_edge`-using effects should target v1; mic-aware effects should target duck. Document this in each effect's `ref_input` field.

---

## 8. UI decoration — fold into the migration

Goal: every effect card in the viewer's list gets an at-a-glance input-badge row; the detail panel shows per-input role + wiring blurb. Same lightweight spirit as §3 — data is declarative on the effect class, no new endpoints, no runtime plumbing.

### 8.1 Data model — two new class fields

The `ref_inputs_required` field from §3 is the badge data. Add one more sibling, `input_roles`, to carry the per-effect prose. Both live on the effect class, both are picked up by `_discover_effects` (`server.py:403`) the same way `ref_input` already is.

```python
class TiltGravity(AudioReactiveEffect):
    ref_input = 'accel (tilt) + pot (spawn/erase)'   # existing freeform, keep
    ref_interactivity = 'sensor'                      # existing, keep
    ref_inputs_required = ['accel', 'pot']            # NEW — badge row
    input_roles = {                                   # NEW — detail panel
        'accel': 'X-axis tilt provides gravity force on particles',
        'pot':   'Turn knob to spawn particles (+) or erase (-)',
    }
```

`ref_input` (freeform) stays — it's already in the API payload and the detail-row renderer at `app.js:2589`. We are not deleting fields; we are *adding* two structured siblings.

**Why a dict not a list of objects:** keying by canonical input id (`'pot'`, `'accel'`, `'gyro'`, `'v1_tap'`, `'keyboard'`, `'mic'`) lets the UI cross-reference the wiring blurb library without a join.

### 8.2 Canonical input ids + wiring blurbs

Single dict in `audio-reactive/effects/inputs.py` next to the helper functions. The same module that owns the constants owns the human-facing reference text; future agents only have one file to update when wiring changes.

```python
# audio-reactive/effects/inputs.py

INPUT_CATALOG = {
    'mic':      {'label': 'Mic',      'icon': '🎙', 'wiring': 'BlackHole virtual input or selected sounddevice. 44.1 kHz mono.'},
    'pot':      {'label': 'Pot',      'icon': '🎛', 'wiring': 'Noodles controller on USB; potentiometer ADC 0-1023 @ ~30 Hz.'},
    'accel':    {'label': 'Accel',    'icon': '📐', 'wiring': 'Duck sender (ESP-NOW → noodles). MPU-6050 ±2g, gravity included.'},
    'gyro':     {'label': 'Gyro',     'icon': '🌀', 'wiring': 'Duck sender. MPU-6050 ±250°/s; magnitude in deg/sec.'},
    'v1_tap':   {'label': 'v1 tap',   'icon': '👆', 'wiring': 'gyro-sense sender (ESP-NOW → noodles). amag_max @ 25 Hz, ±4g/±1000 dps.'},
    'keyboard': {'label': 'Key',      'icon': '⌨',  'wiring': 'Viewer stdin → runner. Digit1 etc. are the bound keys.'},
}
```

Single source of truth. The icon column can be a glyph, a CSS class name, or an SVG path — doesn't matter for the proposal, picks happen at UI time. (Note: caveman/no-emoji project preference — replace glyphs with CSS class names if the user prefers; structure is the same.)

### 8.3 Server-side change

In `server.py:_discover_effects` (line 403) and the per-entry dict it builds, add two passthrough lines after the existing `ref_input` line:

```python
'ref_inputs_required': getattr(eff_cls, 'ref_inputs_required', []),
'input_roles':         getattr(eff_cls, 'input_roles', {}),
```

Two lines × two sites (one for signal effects line ~420, one for full effects line ~430, possibly the unified render at line ~560 too). That's it server-side. No new endpoint; the existing effects-list payload carries the new fields. UI ships `INPUT_CATALOG` either as a static JSON the server serves once or inlined into `app.js` (it's ~6 entries, fine to inline).

### 8.4 UI change

Two surfaces, both already exist:

**Effect list cards** (`app.js` around `:2288` / `:2341` where `isEffectAudioReactive` is consulted today): after the card title, render a badge row from `eff.ref_inputs_required.map(id => INPUT_CATALOG[id])`. CSS: a small flex row of pill-shaped badges with icon + label. New CSS class in `style.css`, ~10 lines. Total new JS: ~15 lines per card-rendering site, can factor to one helper `renderInputBadges(eff)`.

**Effect detail panel** (`app.js:2589` where `ref_input` is rendered as a row today): replace/augment that single row with a structured "Inputs" section:

```
Inputs
  Pot       Turn knob to spawn particles (+) or erase (-)
            Noodles controller on USB; ADC 0-1023 @ ~30 Hz.
  Accel     X-axis tilt provides gravity force on particles
            Duck sender (ESP-NOW → noodles). MPU-6050 ±2g.
```

Two lines per input: per-effect role (from `input_roles[id]`), per-input wiring blurb (from `INPUT_CATALOG[id].wiring`). Keep the freeform `ref_input` field as a fallback / additional notes line for effects that haven't been migrated yet.

Files touched: `app.js` (~40 lines added, ~5 modified), `style.css` (~15 lines), `body.html` (probably 0 — both surfaces already have container elements). No new templates.

### 8.5 Migration order

Fold into the §4 work, per-effect. Each effect's conversion becomes a 3-step diff:

1. Replace duplicated EMA / threshold blocks with `inputs.py` helpers (§4).
2. Declare `ref_inputs_required` and `input_roles` on the class.
3. (Once any effect has them) ship the JS badge row + detail rewrite. Effects without the new fields fall back to the existing `ref_input` row — graceful degradation, partial migration is safe.

Effects without sensor inputs (mic-only) declare `ref_inputs_required = ['mic']` and a one-line `input_roles['mic']`. Visual-only effects declare `ref_inputs_required = []` and the badge row simply renders empty.

### 8.6 Effort estimate

Adds **≈2-3 hours** on top of the §4 migration:

- `INPUT_CATALOG` dict + server passthrough: 30 min.
- JS badge renderer + CSS: 60 min.
- JS detail panel rewrite: 45 min.
- Per-effect `ref_inputs_required` + `input_roles` declarations: ~3 min each × ~20 sensor-or-audio effects ≈ 60 min, mostly already implied by writing `ref_input` today.

**Combined §4 + §8 total: ≈6-7 hours.** Still well below the original 8-10 hour InputContext proposal, and now the UI deliverable is included.

### 8.7 Non-goals (deferred)

- **Live "input present / missing" indicators on the badges** (greying out a pot badge when the noodles controller isn't connected). Wants the staleness work flagged in §5 plus a websocket push from the runner. Worth doing later; intentionally not in this slice.
- **Click-to-filter** (click the pot badge → show only pot-using effects). Trivial to add once the data is in the payload, but a separate UX call.
- **Auto-derived `ref_inputs_required` from setter introspection** (`hasattr(effect, 'set_pot_value') → 'pot'`). Possible but lossy — `band_tendrils` and `jellyfish` would auto-acquire pot, which is the right answer, but `nebula_explosions`'s v1 tap usage couldn't be distinguished from `rc_particle`'s v1 tilt usage by setter alone. Explicit declaration is clearer.

---

## Revised executive summary (120 words)

The pot, accel high-pass, and `AMAG_TAP_THRESH` are copy-pasted 2-4× across effects with magic constants. Proposed fix is intentionally lightweight: a new `audio-reactive/effects/inputs.py` of ~8 stateless free functions (`smooth_pot`, `pot_delta_kick`, `accel_high_pass`, `tap_edge`, `key_edge`, `pot_scale`, etc.) plus shared constants. Runner stays unchanged; effects keep `set_pot_value` / `set_imu_data` / `set_v1_data` setters and own their state dicts. Total migration ≈3-4 hours. Honest finding on "button": keyboard, v1 amag tap, and pot delta are three different events (named-hold, thresholded-magnitude-with-cooldown, signed-velocity-gate) — keep three helpers, not one unified abstraction. Honest finding on senders: keep duck (raw + mic) and v1 (aggregated + tap-grade) separate; they serve different devices.

## 5. Open questions for the user

1. **InputContext vs. free functions?** The doc proposes the heavier option. The lighter option (utility module) is faster but doesn't solve staleness or UI introspection. Which?
2. **Shared `AMAG_TAP_THRESH`** — is the current value (whatever each file uses today) actually the right one across effects, or should each effect keep its own sensitivity?
3. **Staleness defaults** — when pot/IMU goes stale, should effects (a) freeze at last value, (b) drift to a "rest" default, or (c) be told and decide? Proposal above implies (c) via `.stale` flag.
4. **Multi-mac v1 role assignment** — should this be promoted to the InputContext (a fleet abstraction), or stay effect-local since only `rc_particle` uses it?
5. **RMS recompute in `tilt_pendulum`/`gas_crawlers`** — fold into this migration, or defer until the EMA-band-signal refactor lands (per the open question in MEMORY.md)?
6. **Keyboard channel** — currently only `pot_particle` uses it. Worth unifying with `button` or leave as a separate `ctx.keys`?

---

## Executive summary (120 words)

The pot, IMU accel, v1 tap, and gyro have each been implemented 2-4 different ways across ~10 effects, with the accel high-pass copy-pasted verbatim (alpha 0.008) in four files and `AMAG_TAP_THRESH` defined twice. Two effects (`band_tendrils`, `jellyfish`) hide a pot dependency from the UI by skipping `ref_input`. Nothing handles input staleness — runner pushes default 512/zero indefinitely.

Proposed canonical interface: a single `InputContext` injected via `set_inputs(ctx)` exposing `ctx.pot.value/delta/kick/scale/stale`, `ctx.imu.gravity()/tilt_angle()/shake()/rotation()/stale`, `ctx.button.pressed()` (unifies keyboard + v1 tap + pot-momentary), and `ctx.v1.controller/trigger/stale`. Migration is 8-10 hours via an additive shim — old setters keep working until each effect is converted. Six open questions in §5 worth a user call before starting.
