# LED Streaming Effects

Stream LED effects from computer to Arduino for high-performance animations.

## Project Structure

```
streaming/
├── single-strip/     # Single LED strip streaming
│   ├── receiver/     # Arduino firmware (receives frames via serial)
│   ├── controller/   # Python controller (generates and streams frames)
│   └── README.md     # Single-strip documentation
├── multi-strip/      # (Future) Multiple independent strips
└── wokwi/           # (Future) Wokwi simulations
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Single Strip Setup

1. **Upload receiver firmware to Arduino:**
   ```bash
   cd single-strip/receiver
   pio run -e controller1_stream --target upload  # Controller 1 (port 11230)
   pio run -e controller2_stream --target upload  # Controller 2 (port 11240)
   ```

2. **Start streaming:**
   ```bash
   cd single-strip/controller
   source venv/bin/activate
   python nebula_stream.py --port /dev/cu.usbserial-11230
   ```

## Available Effects

- **Nebula**: Breathing blue-magenta waves with glowing orbs

## Why Stream?

- **Faster computation**: Python on computer vs Arduino
- **Complex effects**: More memory and processing power
- **Real-time control**: Adjust parameters on the fly
- **High FPS**: 60+ FPS streaming
