# v1 Wire-Schema Derivation

Single-page record of how the v1 16-byte ESP-NOW telemetry packet was
derived from raw IMU captures. Synthesized from three original working
documents (audit / threshold validation / tap-detector research) which
have been retired.

Cite: ledger entry `bulb-imu-telemetry-wire-schema-v1`.

## Why ±4g / ±1000°/s

The original sender ran the MPU-6050 at ±2g / ±250°/s. Severe gestures
pinned the rails:

- Hanging drumming bouts (`hang ep4`, `ep5`) saturated `accel_clip_frac`
  at 26% and `gyro_sat_frac` at 25.7% of samples.
- Held-bulb dance and severe drumming (duck `ep4`, `ep5`, `ep6`) all
  pegged `peak_axis_g` at 2.000 g — the rail value, not a real reading.
- Flag bits (`accel_clip`, `gyro_sat`) carried no information during
  severe events because they were permanently set.

Doubling accel range and quadrupling gyro range recovers headroom
without hurting resolution at the small-signal end (the LSB cost is
sub-noise-floor on this MPU). On the new ±4g / ±1000°/s captures the
worst severe-drumming episode shows `accel_clip_frac < 7%` and
`gyro_sat_frac < 2%`. Hardware-edge truth is restored. Range-bumped
firmware is the v1 default.

## Packet layout

16 bytes per packet, 25 Hz over ESP-NOW broadcast. **Option-A semantics**:
per-axis `_max` and `_min` bytes are AC-coupled against the per-window
mean (matching `(rawMax - rawMean) >> 8` in `sender.cpp`); per-axis
`_mean` bytes are **raw** (`rawMean >> 8`, gravity included) so receivers
can recover the gravity vector via EMA. `amag` and `gmag` use **raw**
magnitudes (gravity included) and are sqrt-companded with full-scale
`AMAG_FS = GMAG_FS = 57000`.

| Bytes | Field | Units / encoding |
|---|---|---|
| 0–1   | `seq` | uint16 LE packet counter |
| 2–4   | `ax_max`, `ay_max`, `az_max` | int8 AC-coupled per-axis max over 8 samples (LSB = 1/256 of full-scale-g; 256 cts/LSB at ±2g, 512 cts/LSB at ±4g) |
| 5–7   | `ax_min`, `ay_min`, `az_min` | int8 AC-coupled per-axis min |
| 8–10  | `ax_mean`, `ay_mean`, `az_mean` | int8 raw per-axis mean (`rawMean >> 8`) — gravity INCLUDED; receivers recover the gravity vector from these |
| 11    | `amag_max` | uint8 sqrt-companded peak \|a\| (raw, gravity included) |
| 12    | `amag_mean` | uint8 sqrt-companded mean \|a\| (raw, gravity included) |
| 13    | `gmag_max` | uint8 sqrt-companded peak \|gyro\| |
| 14    | `gmag_mean` | uint8 sqrt-companded mean \|gyro\| |
| 15    | `flags` | bit0=`accel_clip`, bit1=`gyro_sat`, bit2=`bgMotion`, bits3–7 reserved |
| (16)  | `rawRms` | optional 17th byte: log-companded mic RMS |

Gravity is intentionally *not* transmitted as an absolute vector; the
receiver recovers it via EMA over the per-axis `_mean` bytes (see Known
limitations). Saturation flags fire on raw int16 saturation
(`|i16| ≥ 32760`) regardless of range — saturation is a hardware fact,
not a unit fact.

The simulator at `tools/simulate_wire.py` carries `_grav_mean` as a debug
field for fidelity comparisons; the wire format does not.

## Threshold derivation

Two candidate trigger signals were evaluated against the locked taxonomy
(13 hand-segmented episodes across hanging and duck captures):

- **`peak_axis_ac_g`** — `max(|ax/ay/az_{max,min}|) * 256 / counts_per_g`.
  Gravity-removed by construction. Captures lateral taps cleanly.
- **`|amag_max - 1g|`** — old proxy. Underestimates lateral taps because
  raw `|a|` is dominated by the gravity vector.

Headline separation across captures:

| mode | signal | fire_min | no_fire_max | gap | clean? |
|---|---|---|---|---|---|
| hang | `peak_axis_ac_g` | 0.750 | 0.250 | 0.500 | YES |
| hang | `\|amag_max - 1g\|` | 0.891 | 0.285 | 0.606 | YES |
| duck | `peak_axis_ac_g` | 2.000 | 1.984 | 0.016 | YES (rail-limited) |
| duck | `\|amag_max - 1g\|` | 2.238 | 2.452 | -0.214 | NO |

`peak_axis_ac_g` wins on both modes. `|amag_max - 1g|` has structural
ambiguity: it triggers on both `|a|>1g` (taps) and `|a|<1g` (free-fall
dips), and inverts cleanly on the duck capture where free-dance amag
exceeds tap amag.

**Hanging — final threshold = 0.50 g.** Light pendulum decay sits at
0.25 g (no fire); medium tap fires at 0.75 g; severe drumming clips. The
between-episode baseline p99 is 0.016 g, ≈30× headroom. ROC sweep
confirms the entire `[0.32, 0.69] g` range gives TP=4/4, FP=0; pick the
midpoint for stress robustness.

**Duck — final threshold = 3.50 g + 200 ms refractory.** Amplitude alone
cannot cleanly separate held-bulb tap from free-hand dance — both peak
at 3.7–4.0 g in the ±4g supplementary captures (range_capture_protocol
labels). 3.50 g excludes shake/hard (1.56 g) and rest cleanly; still
fires on dance and severe drumming. **Documented v1 limitation**:
amplitude-only is inadequate for principled duck classification. The
receiver should EMA the per-axis `_mean` bytes (~1–2 s tau) to recover a
long-baseline gravity vector and reject motion oscillating with gravity.

**Crest gate dropped.** Proposed gate `(amag_max_byte / amag_mean_byte)² ≥ 4.5`
is structurally unreachable. Max observed across 13 episodes = 3.57 (duck
ep7 settle/decay); highest *tap* crest² = 2.83. Root cause: `amag` is
computed from raw accel (gravity included), so over a 40 ms window
`amag_mean ≈ 1g + ε`. For a 2g lateral tap, `max/mean ≈ 2/1 = 2 → 4`;
sqrt-companding compresses further. The gate value 4.5 assumed
gravity-removed `|a|`. **Drop the gate for v1**; rely on
`peak_axis_ac_g` + 200 ms cooldown. Revisit when a tap-ladder + shake-
ladder dataset lets us anchor the impulsive/rhythmic boundary.

## Validation result

Per-feature wire fidelity at the receiver (decoder applied to encoded
packets, compared to ground-truth windowed features from raw IMU):

| feature | ±2g baseline (mean rel err) | ±4g extended (median) | verdict |
|---|---|---|---|
| `peak_amag` | 0.6% | 0.4% | unchanged |
| `rms_amag` | 4.9% | 3.5% | unchanged |
| `crest` | 5.7% | 3.9% | unchanged |
| `ac_strength` | 18.0% | 17.9% | identical (Nyquist-limited, range-independent) |
| `n_peaks` | Δ ~1 | Δ 1 | unchanged |
| `stroke_rate_hz` | 28.4% | 14.3% | unchanged |
| `grav_angle_deg` | 2.7% (max 20%) | Δ 0.1° | unchanged |
| `accel_clip_frac` | Δ ~0.04 | Δ 0.0 | improved (less clipping at ±4g) |
| `gyro_sat_frac` | Δ ~0.02 | Δ 0.0 | improved |

The window-aggregation features (`ac_strength`, `n_peaks`,
`stroke_rate_hz`) are intrinsically lossy at 25 Hz packet rate and that
loss is range-independent. Everything else preserves to within
statistical noise of the ±2g baseline.

Classifier result, end-to-end at the wire, on the locked 13 episodes:

- **Kind correct: 13/13**
- **Severity correct: 12/13**

The single severity miss is `hang ep5` (drumming bout 2): tap/severe →
shake/severe at the kind level on one boundary review, and a tap/hard →
tap/severe re-call on `duck ep4`. Both are classifier-boundary choices,
not schema fidelity issues — the wire carries enough information to
re-decide either way.

sqrt-companding ceiling holds: `RANGE_AMAG=80000` sits comfortably above
the busiest packet observed (max byte = 224/255 = 88% headroom across
~5400 packets in the busiest captures). At ±4g the encoder's amag input
is half the count value of an equivalent ±2g event, so the byte ceiling
sits *lower* in g-space — no ceiling adjustment needed.

## Known limitations

- **Per-axis mean drifts during sustained motion.** The wire's
  per-axis `_mean` byte is the literal window mean, AC-coupled by
  subtracting itself; over a 40 ms window with sustained translation the
  per-window mean tracks the motion, not gravity. The receiver
  reconstructs an absolute gravity vector by EMA-ing the wire `_mean`
  bytes with a 1–2 s time constant. Static reorientation gestures
  (the `movement` kind) currently recover gravity to within ~20% of
  ground truth, which is the largest known v1 fidelity gap. v1.1 would
  close this with 3 explicit gravity bytes amortized at 5 Hz (~15 B/s).
- **Sub-window timing (<40 ms) is structurally absent.** Two taps inside
  one window collapse to one max value. Drumrolls above 25 Hz alias.
- **Content above 12.5 Hz aliases.** The MPU's DLPF is at 44 Hz BW;
  sampling at 25 Hz folds 12.5–44 Hz content into the band. If field
  deployment surfaces spurious "shake"-like behavior on quiet bulbs,
  aliased mechanical vibration is the prime suspect — mitigation is
  firmware (tighten DLPF to 0x06), not schema.
- **Inter-axis phase / rotation chirality is not preserved.** Per-axis
  envelope keeps amplitude per axis but not the phase relationship
  between axes. Clockwise vs counter-clockwise wrist twists look
  identical on the wire.
- **Severity classifier in `synth_severity.py`** must use physical
  units (g, °/s) — its previous count-based thresholds become
  half-magnitude under the ±4g range bump. Same root cause as
  `bulb_receiver.cpp`'s `computeGyroRate` / `computeAccelJolt`. Fix at
  the unit boundary, once.
