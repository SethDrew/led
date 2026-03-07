# Serial Protocol

Communication between host (Python/macOS) and LED controllers (ESP32/Arduino Nano).

## Frame format

```
[0xFF] [0xAA] [R G B × N] [XOR checksum]
```

- **Header**: 2 bytes (`0xFF 0xAA`)
- **Payload**: 3 bytes per LED (RGB order), N = pixel count for the target controller
- **Checksum**: XOR of all payload bytes (not header)

## Timing

- **Baud rate**: 1 Mbps
- **Frame rate**: 30 FPS (33.3ms per frame)
- **Latency budget**: 18–58ms end-to-end (audio capture → LED update)

## Architecture constraint

Audio callback stores state → fixed-rate main loop reads state → serial flush. Serial write must complete before WS2812B timing-critical `show()` call. This ordering is non-negotiable — WS2812B disables interrupts during data transmission.

## Controller-specific

| Controller | Pixels | Baud | PlatformIO env |
|------------|--------|------|----------------|
| ESP32 tree | 197 (3 strips: pins 27/15/14) | 1 Mbps | `tree` |
| Arduino Nano (sculpture/diamond) | 80 | 1 Mbps | `sculpture_stream` |

## Audio input

BlackHole 2ch virtual loopback device via `sounddevice` (Python). System audio → BlackHole → our process.
