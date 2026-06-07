# Composable Sensor Layer Protocol

How standalone (ESP-NOW / UDP) installations get sensor data from a handheld/wearable sender to the LED receiver. This documents the pattern as it exists in the codebase today, not a proposed design.

A **sender** reads physical sensors and broadcasts compact packets. A **receiver** consumes them and drives effects. Rather than one fixed packet, the wire format is a set of independent **layers** — a sender emits any subset, and the receiver picks out the layers it understands. This is the physical, on-the-wire counterpart to the abstract input roles in ARCHITECTURE.md §5: layers carry raw/summarized sensor data; the receiver's role binder turns that into EVENT / INTENSITY / MOOD / etc.

---

## The three layers

| Layer | Struct | Header | Size | Status |
|---|---|---|---|---|
| **Accel** | `TelemetryPacketV1` | `lib/v1_telemetry/v1_packet.h` | 16 B | Standardized |
| **Gyro** | `GyroPacketV1` | `lib/v1_telemetry/gyro_packet_v1.h` | 12 B | Standardized |
| **Audio** | *(none)* | — | — | **Not standardized** |

- **Accel** — MPU-6050 accelerometer, sampled 200 Hz, summarized into per-window per-axis max/min/mean plus companded magnitudes, emitted at 25 Hz. Full wire layout and the rationale behind every field live in the engineering ledger entry `bulb-imu-telemetry-wire-schema-v1` — do not duplicate it here.
- **Gyro** — companion gyro telemetry, same windowing and companding conventions as accel. Field layout is in `gyro_packet_v1.h`.
- **Audio** — see "Audio layer status" below. There is no standardized audio packet. What exists is a raw-RMS field carried inside an older, separate `SensorPacket` (the duck/phone format), not a layer in this scheme.

---

## How layers compose

Each layer is an independent fixed-size struct sharing a `uint16 seq` lead field. A sender broadcasts whichever layers it produces, each on its own cadence. The receiver's receive callback inspects each incoming packet and routes it by **payload size**:

```cpp
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len == sizeof(TelemetryPacketV1)) { /* accel */ }
    else if (len == sizeof(GyroPacketV1)) { /* gyro */ }
}
```

This is how `biolum` already consumes accel and gyro from a single sender — two layers, one link, disambiguated by size (see `biolum/src/biolum_mixed.cpp`). A receiver that only cares about shake reads only the accel layer and ignores the rest.

### Known fragility: no discriminator byte

Layers are told apart **only by `sizeof`**. There is no explicit `layer_id` / type tag on the wire. Today this works because the live layers (16 B, 12 B) and other coexisting packets (the 15 B duck `SensorPacket`, BS-26 JSON) all have distinct sizes. It is fragile: a future layer (e.g. an audio packet) that happens to match an existing size would be silently misparsed. Any receiver that handles multiple packet types **must** gate on exact size (`len == sizeof(...)`) — never `len >= ...`.

---

## Current senders

| Sender | Hardware | Layers emitted | Transport |
|---|---|---|---|
| **gyro-sense** | ESP32-C3 + MPU-6050 | Accel + Gyro | ESP-NOW |
| **Duck v1** *(frozen)* | ESP32-C3 + MPU-6050 + INMP441 mic | Raw IMU + raw RMS (legacy 15 B `SensorPacket`, **not** the layer structs) | ESP-NOW |
| **Phone app** | Android IMU + mic | Raw IMU + raw RMS (15 B `SensorPacket`) + onset side-packet | UDP |
| **BS-26** | Knob/switch/meter panel | Knobs/switches only (JSON, no sensor layers) | ESP-NOW |

Only **gyro-sense** speaks the layer protocol (accel + gyro). The duck and phone predate it and ship the older flat `SensorPacket` (raw accel/gyro/RMS in one 15 B struct) plus, for the phone, a separate 4 B onset packet. BS-26 carries no sensor layers at all.

---

## Mapping layers to input roles

Layers are raw/summarized signals; the receiver binds them to the abstract input roles of ARCHITECTURE.md §5. The same role can be filled from different layers depending on what a sender provides.

| Layer field | Typical input role(s) | Notes |
|---|---|---|
| Accel `amag_max` / `amag_mean` | INTENSITY, EVENT | Shake/strike magnitude. `biolum` drives bloom energy from `amag_max`. |
| Accel per-axis `_mean` | MOOD, POSITION | At-rest gravity direction → tilt → hue/position. |
| Gyro `gz_mean` (and other axes) | RATE, POSITION | Rotational rate; `biolum` uses `gz_mean` for scroll direction. |
| Audio envelope (raw RMS, today) | INTENSITY | Loudness → brightness/size. |
| Audio onset (phone side-packet, today) | EVENT | Discrete trigger (spawn/flash). |

A receiver may also *derive* roles a layer doesn't ship directly — e.g. deriving an EVENT onset from frame-to-frame change in the audio envelope, or recovering a clean gravity vector by EMA-ing the accel `_mean` bytes over ~1–2 s.

---

## Audio layer status

The audio layer is **not standardized**. Duck v1's raw RMS (a single `uint16` inside the flat `SensorPacket`) is the working stop-gap — an unnormalized loudness number that each receiver re-normalizes itself (adaptive floor + log-dB scaling, duplicated across `road-bulbs` and `bulb-fleet`). The phone additionally sends a separate 4 B onset packet, the only sender-side onset detection in the system.

This is deliberate. Audio is harder to standardize than movement: feature extraction is often effect-specific, transient timing is latency-sensitive in a way gestures are not, and audio streams are richer than IMU summaries. The current expectation is that new audio features get built **bespoke, per-effect, on the sender** and proven against real effects before any of it is frozen into a standardized audio layer. When that layer arrives it should slot into this scheme as a third size-distinct struct (and is the most likely trigger for finally adding an explicit discriminator byte).

---

## See also

- Engineering ledger `bulb-imu-telemetry-wire-schema-v1` — canonical accel wire schema and design rationale.
- `lib/v1_telemetry/v1_packet.h`, `gyro_packet_v1.h` — the layer struct definitions.
- ARCHITECTURE.md §5 (Input Roles) — the abstraction layers bind to.
- INPUT_ROLE_MATRIX.md — full per-effect role assignments.
