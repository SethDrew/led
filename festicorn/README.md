Umbrella project for all LED installation firmware. Each installation is a separate PlatformIO project under this directory.

## Installations

- **butterfly/** — WiFi-controlled WS2812B sculpture (ESP32-C3). Web UI served from LittleFS for real-time effect switching, OKLCH palettes, and OTA updates.
- **bulbs/** — Standalone gyro+mic reactive LED bulbs. IMU-driven color mapping with audio reactivity, no WiFi required.
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
cd festicorn/bulbs && pio run
cd festicorn/stick && pio run
```
