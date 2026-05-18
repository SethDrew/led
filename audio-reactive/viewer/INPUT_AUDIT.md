# Viewer Input Audit

Read-only map of effects in the audio-reactive viewer and the physical inputs they consume. Cites paths only; proposes nothing.

## 1. Effect inventory

The viewer's effect list is built dynamically by `audio-reactive/viewer/web/server.py:_discover_effects` (line 403) which shells out to `audio-reactive/effects/runner.py --list-json`. The web UI has no static effect list of its own; every effect is defined in `audio-reactive/effects/*.py` as a subclass of `AudioReactiveEffect` (or `ScalarSignalEffect`) declared in `audio-reactive/effects/base.py`. The runner discovers them by importing modules and reading their `registry_name`. There is also a `deprecated/` subfolder scanned separately (`server.py:_scan_deprecated_effects`).

So the answer to "where defined" is: **all effects are Python-side, viewer-rendered, sent to LEDs via serial.** Firmware-side effects (e.g. `firmware/strip`, festicorn boards) are not selectable from the viewer — they run autonomously. The viewer's `runner.py` runs effects on the laptop and pushes RGB frames over USB.

Active effects (top-level `audio-reactive/effects/`): `absint_predictive`, `band_prop`, `band_sparkles`, `band_tempo_sparkles`, `band_tendrils`, `band_zone_pulse`, `basic_sparkles`, `bass_pulse`, `centroid_blue_dual`, `color_pass`, `compare`, `composed`, `diamond_voices`, `energy_color`, `energy_waterfall`, `energy_waterfall_rap`, `firefly_sync`, `flicker_flame_warmth`, `gas_crawlers`, `heartbeat_tempo`, `heat_burst`, `heat_diffusion`, `jellyfish`, `leaf_gust`, `leaf_wind`, `mel_height_magma`, `mfcc_chroma_rainbow`, `nebula`, `nebula_explosions`, `polycule`, `polycule_rainbow`, `pot_particle`, `rap_pulse`, `rc_particle`, `spectro_chroma`, `tempo_pulse`, `tilt_gravity`, `tilt_pendulum`, `timbral_chroma_split`, `two_snakes`, `worley_voronoi`. Plus signal-only effects from `signals.py` wrapped in `ComposedEffect`.

## 2. Per-effect input requirements

Each effect file declares two metadata fields on its class (defined `base.py:30-33`): `ref_input` (human-readable input list) and `ref_interactivity` (one of `audio` | `sensor` | `hybrid` | `visual`; default `audio`). Effects that need non-audio inputs additionally implement `set_pot_value(raw)`, `set_imu_data(data)`, or `set_v1_data(data)` — `runner.py:566-571` calls these only `if hasattr(effect, ...)`.

Non-audio-dependent effects:

- `nebula_explosions.py:23` — `ref_input='v1 telemetry IMU (tap → explosion)'`, `sensor`. Calls `set_v1_data`.
- `gas_crawlers.py:22` — `pot rotation (agitation) + accel (gravity)`, `sensor`. Calls `set_pot_value` and `set_imu_data`.
- `pot_particle.py:23` — `pot (position) + gyro (rotation speed) + keyboard (explosion)`, `sensor`. `set_pot_value`, `set_imu_data`.
- `rc_particle.py:27` — `v1 accel (tilt → position) + v1 tap (explosion)`, `sensor`. `set_v1_data`.
- `tilt_gravity.py:21` — `accel (tilt) + pot (spawn/erase)`, `sensor`. `set_pot_value`, `set_imu_data`.
- `tilt_pendulum.py:19` — `pot (oscillation) + accel (gravity) + RMS (particle width)`, `hybrid`. Audio + pot + IMU.
- `nebula.py:18` — `none (visual only)`, `visual`. No input.
- `basic_sparkles.py:19` — `none (visual only)`, default audio. No actual input consumed.
- `heat_burst.py:27` — `none (static snapshot)`. No input.
- `band_tendrils.py` — declares no `ref_interactivity` override but implements `set_pot_value` (line 201). Hidden pot dependency.
- `jellyfish.py:161` — `ref_input='tempo tracker'`, but also has `set_pot_value` (line 206) for parameter tweaking.
- Deprecated `percussive_tendrils.py` has `set_pot_value`.

All other effects are mic-audio-only (RMS, FFT bands, onset, abs-integral, MFCCs, HPSS, mel spectrogram, etc.). See the `ref_input` field per file for specifics.

## 3. Input wiring today

- **Mic / audio**: BlackHole virtual device or selected input via `sounddevice` in `runner.py`; `--wav FILE` flag plays back files. Enters effects through `process_audio(mono_chunk)`.
- **Pot (potentiometer)**: 10-bit ADC sampled on a controller MCU, framed as `[0xFC][hi][lo]` and written to USB serial. Emitters: `firmware/strip/src/streaming_receiver.cpp:205` (strip controller) and `festicorn/noodles/src/main.cpp:180` (noodles ESP32 — `controllers.json` mac `f4:2d:c9:6d:b2:58`, baud 460800, port hint `11430`). Parsed in `audio-reactive/effects/runner.py:380-382` (`SerialLEDOutput._drain_and_parse`), stored as `pot_value` (0-1023), pushed each frame to effects via `set_pot_value`.
- **Duck IMU (v0.1 SensorPacket, 15 B)**: original-duck sender ESP32 broadcasts via ESP-NOW (mac `38:44:be:45:d9:cc`), the noodles controller receives and re-emits to USB as `[0xFB][15 bytes]` (`noodles/src/main.cpp:191`). Runner parses at `runner.py:383-391` into `imu_data = {ax, ay, az, gx, gy, gz, rms, mic_on}`, delivered via `set_imu_data`.
- **v1 TelemetryPacketV1 (16 B, gyro-sense / aggregated motion stats)**: gyro-sense sender broadcasts ESP-NOW; noodles forwards to USB as `[0xFA][mac 6B][packet 16B]` (`noodles/src/main.cpp:196-206`). Runner parses at `runner.py:392-411` into `v1_data` dict; delivered via `set_v1_data` (only when non-None — `runner.py:570`).
- **Keyboard**: `pot_particle.py` mentions keyboard (explosion); handled inside the runner loop (key listener).
- **Board telemetry / controller identification**: at startup the viewer queries each candidate port for EEPROM `device_id` (`runner.py:75`, `server.py:_query_device_id`) and MAC (`server.py:_query_mac`), matched against `audio-reactive/hardware/controllers.json`.

Note: the `festicorn/espnow-bridge` board uses a different framing (`0xA5 0x5A LEN PAYLOAD XOR8`) and is **not** the path that feeds the runner. It is used by `festicorn` recorder tools, not by the viewer.

## 4. Detection feasibility per input

- **Audio input**: live device enumerable via `sounddevice.query_devices()`; BlackHole presence is detectable. Frame energy can also be monitored — silence vs absent device are distinguishable.
- **Controller USB serial**: enumerated at startup; `server.py:_resolve_controller_ports` already determines "connected vs not connected" by `vid/pid + device_id/MAC` match. This is the strongest signal — if the noodles or strip controller port is absent, **no pot, IMU, or v1 data can possibly arrive.**
- **Pot**: noodles emits pot at a fixed cadence; absent CMD_POT (0xFC) bytes for N seconds while the noodles controller is connected → pot disconnected/no movement. There is no explicit heartbeat — silence is the only signal, and the firmware emits unconditionally when polled, so "no movement" vs "unplugged" are indistinguishable without a timestamp on last delivery.
- **Duck IMU (0xFB)**: emitted by noodles only when ESP-NOW receives a v0.1 packet from the duck (`noodles/src/main.cpp` `len == sizeof(SensorPacket)` branch). Silence ≥ ~1 s → duck off/out of range. Detectable via "last 0xFB timestamp" in `SerialLEDOutput`.
- **v1 telemetry (0xFA)**: same pattern — only emitted when gyro-sense sender broadcasts. Currently runner.py guards `if led_output.v1_data is not None` (`runner.py:570`) but never clears it on staleness, so effects keep using the last frame indefinitely.
- **Keyboard**: trivially detectable (always present where viewer runs).

All four ESP-NOW-fed inputs share the same single-point-of-failure: the **noodles controller**. If it is not the selected target, those bytes never reach the runner at all.

## 5. Existing checks

- **Controller resolution**: `server.py:_resolve_controller_ports` (line 129) prints `"not connected"` when a controller fails MAC/vid-pid match, but the result is not surfaced to the UI in a way that gates effect selection.
- **`isEffectAudioReactive`** (`audio-reactive/viewer/web/static/app.js:1773`) — the only effect-vs-input gate in the UI today. It treats `ref_input` starting with the string `"none"` as "non-audio-reactive". It does not consider sensor effects at all; a `sensor` effect with `ref_input='pot rotation (agitation) + accel (gravity)'` is classified as audio-reactive.
- **`ref_interactivity` field** exists on the base class and on the sensor/hybrid/visual effects, and is plumbed through `_discover_effects` into the effects-list API payload — but the UI does not branch on it.
- **`set_v1_data` guard** (`runner.py:570`) — the only "input present" gate inside the runner. Pot and IMU are pushed every frame regardless (with default values 512 / zeros). So a `set_pot_value`-based effect run without the noodles controller sees a stuck mid-position pot and renders silently/incorrectly.
- No timestamp-based staleness check exists for any input.
- No warnings or fallbacks anywhere in `effects/` for missing pot / IMU / v1. Two effects (`band_tendrils`, `jellyfish`) consume the pot without declaring it in `ref_interactivity` or `ref_input`, so even a UI built on those fields would miss them.
