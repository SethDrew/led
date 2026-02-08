# Single Strip Streaming

Stream effects to a single LED strip connected to Arduino.

## Structure

```
single-strip/
├── receiver/         # Arduino firmware
│   ├── src/
│   │   └── streaming_receiver.ino
│   └── platformio.ini
└── controller/       # Python stream controller
    ├── nebula_stream.py
    ├── white_test.py
    └── venv/
```

## Setup

### 1. Upload Receiver Firmware

```bash
cd receiver
pio run -e controller1_stream --target upload  # Port 11230
pio run -e controller2_stream --target upload  # Port 11240
```

The receiver:
- Listens on serial at 1 Mbps
- Expects 150 LEDs on pin 12
- Displays received RGB frames

### 2. Stream Effects

```bash
cd controller
source venv/bin/activate
python nebula_stream.py --port /dev/cu.usbserial-11230
```

## Default Settings

- **LEDs**: 150
- **FPS**: 60
- **Brightness**: 50%
- **Speed**: 1.0x
- **Tail Length**: 15 (medium)

## Customization

```bash
python nebula_stream.py \
  --port /dev/cu.usbserial-11230 \
  --leds 150 \
  --fps 60 \
  --brightness 0.8 \
  --speed 1.5 \
  --tail-length 30 \
  --orbs 8
```

## Hardware

- **Arduino Nano** (ATmega328P)
- **LED Strip**: WS2812B (NeoPixel)
- **Data Pin**: GPIO 12
- **Serial**: 1 Mbps (high speed)
