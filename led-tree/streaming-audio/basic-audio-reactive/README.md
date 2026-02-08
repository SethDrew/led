# Basic Audio-Reactive LED Streaming

Simple bass-reactive LED animation that responds to bass hits from your computer audio.

## Effect

- **Background**: Pulsating dark green (nebula-style breathing)
- **Bass hits**: Quick bright flash on strong bass

## Setup

### 1. Upload Arduino Code

The Arduino streaming receiver is shared with the spectrum analyzer:

```bash
cd led-tree/streaming-audio
pio run --target upload --environment tree
```

### 2. Install Python Dependencies

```bash
cd led-tree/streaming-audio/basic-audio-reactive
pip install -r requirements.txt
```

### 3. Run Audio Streaming

```bash
python audio_stream.py /dev/cu.usbserial-1240
```

Or use default port:
```bash
python audio_stream.py
```

## Visualization

To see what the audio analysis is detecting in real-time:

```bash
python audio_visualizer.py
```

This shows:
- Total RMS level (white line)
- Bass energy (red line, normalized)
- Mid/high energy (green/blue lines)
- Bass threshold line (yellow)
- Detection events (bottom plot)

## Tuning

Edit `audio_stream.py` to adjust:

- **BASS_THRESHOLD**: Sensitivity (default 2.0)
  - Lower = more sensitive to bass
  - Higher = only reacts to strong bass

- **BASS_LOW_FREQ / BASS_HIGH_FREQ**: Bass frequency range (20-250 Hz)

- **Colors**:
  - NEBULA_GREEN: Dark base color
  - NEBULA_BRIGHT: Bright pulse color
  - WHITE_FLASH: Bass flash color

- **FPS**: Frame rate (default 25)

## Protocol

Same streaming protocol as spectrum analyzer:
- Baud: 1 Mbps
- Format: `[0xFF] [0xAA] [R0 G0 B0] [R1 G1 B1] ... [R196 G196 B196]`
- 197 LEDs × 3 bytes = 591 bytes per frame

## Files

- `audio_stream.py` - Basic bass-reactive streamer with pulsating background
- `audio_visualizer.py` - Real-time visualization for debugging/tuning

## Troubleshooting

**"Could not open serial port"**
- Check Arduino is connected
- Try different port: `ls /dev/cu.usb*`
- Make sure no other streaming script is running

**Not reacting to bass**
- Lower BASS_THRESHOLD (try 1.5 or 1.0)
- Check BlackHole audio routing in System Settings
- Play louder music with more bass
- Use visualizer to see what's being detected

**Choppy/laggy**
- Normal latency: 50-150ms
- Lower FPS if needed
- Check CPU usage

## Difference from Spectrum Analyzer

This is the **basic** audio-reactive system with simple bass detection:
- ✅ Simple threshold-based bass detection
- ✅ Pulsating background with bass flashes
- ✅ Easy to understand and tune

For more advanced frequency spectrum visualization, see `../` (scottlawsonbc-inspired).
