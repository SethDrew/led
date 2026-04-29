# bench-bulbs

Bench/lab firmware. **Not production** — for A/B testing the two festicorn
animation pipelines side-by-side on one ESP32-C3 driven by two simultaneous
ESP-NOW senders.

The production target for the bioluminescence sculpture is `festicorn/biolum/`.
This project is allowed to drift; don't refactor it in lockstep.

## Hardware

- **Receiver**: ESP32-C3 super-mini
  - MAC: `10:00:3B:B0:6E:04`
  - Port: `/dev/cu.usbmodem114401`
- **Strip A — SK6812 RGBW on GPIO 10**, 50 LEDs
  - Driven from the original-duck v0.1 sender (mic + gyro, 15-byte `SensorPacket`)
  - Animations: `sparkle_burst`, `fire_meld`, `fire_flicker`, `quiet_bloom`
  - Effect cycling is automatic from tilt + audio onset (verbatim duck logic)
- **Strip B — WS2812B RGB on GPIO 21**, 50 LEDs
  - Driven from the gyro-sense v1 sender (gyro only, 16-byte `TelemetryPacketV1`)
  - Animation: `bloom` — collapsed from biolum's 6 strips × 100 LEDs to
    one 50-LED strip with 2 colonies (verbatim per-LED bloom semantics)

## Dual-sender topology

```
  duck sender (38:44:BE:45:D9:CC) ─┐
    [mic + gyro] 15 B SensorPacket │
                                   │   broadcast on same Wi-Fi channel
                                   ├───────────────► bench-bulbs receiver
                                   │                 (10:00:3B:B0:6E:04)
  gyro-sense v1 sender             │                       │
    [gyro] 16 B TelemetryPacketV1 ─┘                       │
                                                           ├── strip A (SK6812, GPIO 10)
                                                           └── strip B (WS2812B, GPIO 21)
```

Both senders broadcast simultaneously. The receiver's ESP-NOW `onReceive`
callback dispatches by `len`:

- `len == 15` (`sizeof(SensorPacket)`)        → updates the duck packet state
- `len == 16` (`sizeof(TelemetryPacketV1)`)   → updates the biolum packet state

Each animation pipeline reads its own state every render frame.

### Cross-talk

There is none. The length filter in `onReceive()` is strict — packets of the
wrong size are dropped silently. Simultaneous broadcasts from both senders
are safe.

## Channel discovery

At boot the receiver scans for SSID `cuteplant` and locks the radio to the
channel where it has the strongest RSSI (or channel 1 if it can't find the
AP). The same scan is repeated every 5 minutes to heal if the router moves
channels — same logic as the senders. There is no AP join; ESP-NOW only.

## Library / frame rate

NeoPixelBus drives both strips via parallel RMT (channel 0 = RGBW, channel 1 = RGB).
The render loop targets ~200 FPS so delta-sigma dithering is temporally
invisible. With only 50 LEDs/strip and parallel RMT, both `Show()`
transmissions overlap (~2.0 ms total) leaving the rest of the 5-ms frame
budget for animation logic.

If 200 FPS turns out infeasible at runtime, that's fine — the animations
still look correct at lower rates, dithering just becomes more visible at
very low brightnesses.

## Build / flash

```
cd festicorn/bench-bulbs
../../.venv/bin/pio run               # build
../../.venv/bin/pio run -t upload     # flash + monitor
../../.venv/bin/pio device monitor    # monitor only
```
