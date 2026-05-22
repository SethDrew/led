# Festicorn — Claude Instructions

## What This Is

Festicorn is an umbrella for standalone LED installation firmware. Each subdirectory is a finished, independently deployable project — a specific physical installation with its own hardware, sensor pipeline, LED topology, and artistic tuning.

Projects share low-level math primitives (via `festicorn/lib/`) but intentionally duplicate effect algorithms. Each installation is tuned for its physical context: a 6-LED bioluminescence piece and a 50-bulb portable rig want different sparkle decay rates, different brightness caps, different sensor processing. **Prefer specificity over abstraction.** A concrete, tuned effect in one project is better than a generic configurable effect that tries to serve all of them.

When code is genuinely hardware-independent and stable (LUTs, dithering, trig), share it in `lib/`. When code involves artistic or perceptual tuning, keep it local to the installation.

## Time-Based Math (critical)

Never use bare per-frame multipliers or fixed EMA alphas. All decay, smoothing, and integration must be dt-based so behavior is frame-rate independent.

**Decay (multiplicative):**
```cpp
// WRONG: assumes fixed frame rate
x *= 0.98f;

// RIGHT: pure time constant
x *= expf(-k * dt);  // k = -ln(desired_per_frame_rate) * assumed_fps
```

**EMA (additive):**
```cpp
// WRONG: fixed alpha per frame
ema += 0.05f * (target - ema);

// RIGHT: time-constant-based alpha
float alpha = fminf(1.0f, dt / tau);  // tau in seconds
ema += alpha * (target - ema);
```

To convert legacy constants: if old alpha was `A` at `F` Hz, then `tau = 1.0 / (A * F)`. If old decay was `D` at `F` Hz, then `k = -logf(D) * F`.

## Shared Libraries

`festicorn/lib/` contains math primitives shared across all installations:
- `fast_math` — sinLUT, gammaLUT, fastSin, fastGamma24, fastDecay
- `delta_sigma` — temporal dithering (16-bit accumulator → 8-bit output)
- `oklch_lut` — perceptual color LUT

Never inline these into project code. The sim (road-bulbs/tools/sim/) is the one exception — it copies them with a comment noting the source.

## Effect Code is Per-Installation

Each installation has its own tuned copy of effect algorithms (sparkle, fire, bloom, gravity). This is intentional — installations have different hardware, sensor packets, LED counts, and artistic tuning. Don't try to abstract effects into a shared library.

When fixing a bug in effect logic (like frame-rate coupling), grep sibling projects for the same pattern:
```bash
grep -r "the_buggy_pattern" festicorn/*/src/
```
The closest siblings are `road-bulbs` ↔ `original-duck` (same packet format, forked codebase).

## Sensor Packets Vary

Each installation defines its own `SensorPacket` struct. Don't assume 15 bytes or any particular field layout. Check the project's source.

## PlatformIO Build

Each subdirectory is a standalone PlatformIO project. `pio` lives in the project venv at `.venv/bin/pio`, not on PATH. Build with:
```bash
cd festicorn/<project>
../../.venv/bin/pio run
```

## Android Sensor App (sensor-app/)

The phone companion app lives at `festicorn/sensor-app/`. Standard Android Gradle project (not PlatformIO). Build with:
```bash
cd festicorn/sensor-app
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Frozen Installations

`original-duck` is frozen and deployed in the field. Changes require extra caution — test thoroughly and confirm with the user before modifying.
