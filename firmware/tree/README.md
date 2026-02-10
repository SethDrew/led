# LED Tree - Physical Hardware Control

This directory contains the Arduino code for controlling the physical LED tree with 197 nodes across 3 NeoPixel strips.

## Hardware Setup

- **Board**: Arduino Nano (ATmega328, 2KB RAM)
- **Port**: `/dev/cu.usbserial-11240`
- **LEDs**: 197 total (Strip 1: 92 LEDs on pin 13, Strip 2: 6 LEDs on pin 12, Strip 3: 99 LEDs on pin 11)

## Build Environments

### `tree` - Effects Mode (Default)
Runs preset animations directly on the Arduino.

```bash
# Compile
pio run -e tree

# Upload
pio upload -e tree

# Monitor
pio device monitor -e tree
```

Available animations in `src/main.cpp`:
- Classic wave (green)
- Sap flow (green particles)
- Blue wave
- Orange wave
- White sap
- Solid white

### `tree_stream` - Streaming Mode
Receives RGB frame data from a computer via serial at 1Mbps.

```bash
# Compile
pio run -e tree_stream

# Upload
pio upload -e tree_stream

# Monitor (note: 1Mbps baud rate)
pio device monitor -e tree_stream
```

**Protocol**: `[0xFF] [0xAA] [R0] [G0] [B0] ... [R196] [G196] [B196]`
- Start bytes: `0xFF 0xAA`
- RGB data: 197 pixels × 3 bytes = 591 bytes
- Pixel order matches node order from `TreeTopology.h`

**Memory Usage**: ~982 bytes RAM (47.9% of 2KB)
- NeoPixel strip buffers: ~591 bytes
- Topology mapping: ~591 bytes (stored in flash/PROGMEM, not RAM)
- No frame buffer needed (reads directly into strip pixels)

## Tree Topology

See `src/TreeTopology.h` for the complete node-to-strip mapping. Key structure:

- **Strip 1 (Pin 13)**: Lower trunk (39 LEDs) + Branch A (24 LEDs) + Branch B (30 LEDs) = 92 LEDs
- **Strip 2 (Pin 12)**: Branch C (6 LEDs)
- **Strip 3 (Pin 11)**: Upper trunk (71 LEDs) + Branch D (2 LEDs) + Branch E (26 LEDs) = 99 LEDs

Each node has:
- `stripId` (0, 1, or 2)
- `stripIndex` (position on that strip)
- `depth` (vertical position in tree, 0-70)

## Python Streaming Controller (TODO)

The streaming mode is designed to work with a Python controller that:
1. Generates RGB frames (197 pixels)
2. Sends frames via serial at 1Mbps
3. Can drive audio-reactive effects, simulations, etc.

Example workflow:
```python
import serial
import numpy as np

# Open serial connection
ser = serial.Serial('/dev/cu.usbserial-11240', 1000000)

# Generate frame (197 pixels)
frame = np.zeros((197, 3), dtype=np.uint8)
# ... populate frame with RGB data ...

# Send frame
ser.write(b'\xFF\xAA')  # Start bytes
ser.write(frame.tobytes())  # RGB data
```

## Development Notes

- The ATmega328 has only 2KB RAM - be extremely careful with memory usage
- The streaming receiver reads serial bytes directly into strip pixel memory (no intermediate frame buffer)
- Effects mode uses memory-optimized compact node representation
- TreeTopology.h should not be modified - it's the single source of truth for the physical layout
