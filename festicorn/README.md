Umbrella project for all LED installation firmware. Each installation is a separate PlatformIO project under this directory.

## Installations

- **butterfly/** — WiFi-controlled WS2812B sculpture (ESP32-C3). Web UI served from LittleFS for real-time effect switching, OKLCH palettes, and OTA updates.
- **original-duck/** — v0.1 deployable installation: handheld mic+gyro sender (outlier SDA=20/SCL=21 pinout) paired with an SK6812 RGBW receiver. Frozen, in the field.
- **gyro-sense/** — v1 IMU telemetry project: 16-byte gyro-only sender (canonical SDA=8/SCL=7), fleet registry, LittleFS recorder, ESP-NOW streaming pipeline, dataset captures, and the analysis that derived the v1 wire schema.
- **biolum/** — v1 production receiver. Bioluminescence installation: 6 strips × 100 LEDs (or RGBW variant) on classic ESP32. Pairs with `gyro-sense/`'s sender.cpp; renders quiet_bloom and tap-driven wave_pulse.
- **budget-skylight/** — autonomous SK6812 RGBW bloom on classic ESP32 (no sender, WiFi+OTA, web UI). Independent of the duck/bulbs ecosystems.
- **stick/** — Dual-strip diffuser stick. Streams audio-reactive frames over serial from a host.

## Shared Libraries

`lib/` contains libraries shared across installations:

- **oklch_lut** — 256-entry OKLCH rainbow LUT with hue-dependent lightness
- **delta_sigma** — First-order delta-sigma modulator for sub-byte brightness
- **streaming_protocol** — Packet framing and checksum for serial frame streaming

## Python Tooling

- `gen_palettes.py` — generates OKLCH hue-arc gradients and chroma sweeps
- `gradient_server.py` — web UI gradient picker that streams frames to a Nano over serial

The OKLCH rainbow LUT (`lib/oklch_lut/oklch_lut.cpp`) is generated from the
shared Python module — see `audio-reactive/effects/color/gen_firmware_lut.py`.

## Build

```
cd festicorn/butterfly && pio run -e butterfly
cd festicorn/original-duck && pio run -e receiver
cd festicorn/gyro-sense && pio run -e sender
cd festicorn/biolum && pio run
cd festicorn/budget-skylight && pio run
cd festicorn/stick && pio run
```
