# Audio-Reactive LED Streaming

Real-time audio-reactive LED animations for the LED tree, streaming over serial.

## Projects

This directory contains two different audio-reactive implementations:

### 1. Basic Audio-Reactive (`basic-audio-reactive/`)
Simple bass detection with pulsating background.
- ✅ Easy to understand and tune
- ✅ Threshold-based bass detection
- ✅ Pulsating nebula background with bass flashes
- 25 FPS

**Use when:** You want simple, reliable bass-reactive lighting.

[→ See basic-audio-reactive README](basic-audio-reactive/README.md)

### 2. Spectrum Analyzer (`scottlawsonbc-spectrum/`)
Advanced frequency spectrum visualization inspired by scottlawsonbc/audio-reactive-led-strip.
- ✅ Mel-scale FFT analysis (24 bins)
- ✅ Multiple visualization modes (energy, spectrum, scroll)
- ✅ Configurable threshold for energy mode
- ✅ Real-time spectrum visualization
- 60 FPS

**Use when:** You want frequency-specific effects or spectrum visualization.

[→ See scottlawsonbc-spectrum README](scottlawsonbc-spectrum/README.md)

## Shared Components

Both projects share the same Arduino streaming receiver:

### Arduino Streaming Receiver
- **Code**: `src/main.cpp`
- **Config**: `platformio.ini`
- **Upload**: `pio run --target upload --environment tree`

The receiver accepts serial data at 1 Mbps and updates all 197 LEDs:
- Protocol: `[0xFF] [0xAA] [R0 G0 B0] [R1 G1 B1] ... [R196 G196 B196]`
- Baud rate: 1000000 (1 Mbps)
- No local processing - pure streaming receiver

## Audio Setup

Both projects use **BlackHole 2ch** for system audio capture:

1. Install BlackHole: `brew install blackhole-2ch`
2. Restart Core Audio: `sudo killall coreaudiod`
3. Create Multi-Output Device in Audio MIDI Setup:
   - Add BlackHole 2ch + your speakers
   - Set as system output
4. Python scripts use device 5 (BlackHole 2ch input)

## Quick Start

```bash
# 1. Upload Arduino receiver (once)
cd led-tree/streaming-audio
pio run --target upload --environment tree

# 2. Choose a project and follow its README:

# Option A: Basic bass-reactive
cd basic-audio-reactive
python audio_stream.py

# Option B: Spectrum analyzer
cd scottlawsonbc-spectrum
python spectrum_stream.py energy
```

## Important: Timing and LED Stability

**Critical insight for audio streaming:**

WS2812B LEDs require **consistent timing** between frame updates. Audio callbacks fire at irregular intervals, causing serial jitter that corrupts LED data (blinking).

**Solution:** Decouple audio analysis from frame sending:
```python
# ❌ WRONG - Irregular timing causes LED blinking
def audio_callback(audio):
    frame = generate_frame(analyze(audio))
    send_frame(frame)  # Timing varies with audio buffer!

# ✅ CORRECT - Fixed-rate sending prevents blinking
def audio_callback(audio):
    self.latest_spectrum = analyze(audio)  # Store only

def main_loop():
    while True:
        frame = generate_frame(self.latest_spectrum)
        send_frame(frame)  # Consistent timing
        sleep(1/FPS)
```

Both implementations follow this pattern for stable LEDs.

## Directory Structure

```
streaming-audio/
├── README.md                    (this file)
├── platformio.ini               (Arduino config)
├── src/
│   └── main.cpp                 (Arduino streaming receiver)
├── basic-audio-reactive/        (simple bass detection)
│   ├── audio_stream.py
│   ├── audio_visualizer.py
│   └── README.md
└── scottlawsonbc-spectrum/      (advanced spectrum analyzer)
    ├── spectrum_stream.py
    ├── spectrum_visualizer.py
    └── README.md
```

## Troubleshooting

**Serial port errors:**
- Only run ONE streaming script at a time
- Stop streaming before uploading Arduino code
- Check port: `ls /dev/cu.usb*`

**No audio detection:**
- Verify BlackHole routing in System Settings → Sound
- Check Multi-Output Device is active
- Use visualizer scripts to debug

**LED blinking:**
- This should not happen with current implementations
- If it does, verify fixed-rate loop pattern (see above)
- Test with simple animations first
