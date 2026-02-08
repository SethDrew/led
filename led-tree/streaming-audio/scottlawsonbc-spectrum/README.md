# Spectrum Audio-Reactive LED Streaming

Advanced frequency spectrum visualization for LED tree, inspired by scottlawsonbc/audio-reactive-led-strip.

Multiple visualization modes:
- **spectrum**: Full frequency spectrum mapped to tree
- **energy**: Bass energy reactive with threshold
- **scroll**: Scrolling spectrum effect (TODO)

## Setup

### 1. Upload Arduino Code

```bash
cd led-tree/streaming-audio
pio run --target upload --environment tree
```

### 2. Install Python Dependencies

```bash
cd led-tree/streaming-audio/scottlawsonbc-spectrum
pip install -r requirements.txt
```

### 3. Run Spectrum Streaming

```bash
# Energy mode (default) - bass-reactive with threshold
python spectrum_stream.py energy

# Spectrum mode - full frequency visualization
python spectrum_stream.py spectrum

# Scroll mode - scrolling spectrum (TODO)
python spectrum_stream.py scroll
```

## Visualization

To see the spectrum analysis in real-time:

```bash
python spectrum_visualizer.py
```

This shows:
- Current frequency spectrum (24 mel-scale bins)
- Energy bands over time (bass/mid/high)
- Spectrogram waterfall view

## Modes

### Energy Mode
Bass-reactive animation with configurable threshold:
- LEDs stay dark when bass energy is below threshold
- Bright pulse when bass exceeds threshold
- Gradient effect (brighter at bottom)

**Tuning**: Edit `BASS_THRESHOLD` in spectrum_stream.py (line 36)
- Lower (0.70-0.75): More sensitive, triggers more often
- Default: 0.80 (top ~10% of bass hits)
- Higher (0.85-0.90): Less sensitive, only strong bass

### Spectrum Mode
Full frequency spectrum visualization:
- 24 mel-scale bins (20Hz - 8kHz)
- Color-coded by frequency:
  - Bass (red): First 8 bins
  - Mid (green): Middle 8 bins
  - High (blue): Last 8 bins
- Spectrum mapped across all 197 LEDs

## Configuration

**Frequency Analysis:**
- Sample rate: 44100 Hz
- FFT bins: 24 (mel-scale)
- Frequency range: 20Hz - 8kHz
- Window: Hann

**Performance:**
- FPS: 60 (smoother than basic audio)
- Audio device: BlackHole 2ch (device 5)

## Protocol

Same streaming protocol:
- Baud: 1 Mbps
- Format: `[0xFF] [0xAA] [R0 G0 B0] [R1 G1 B1] ... [R196 G196 B196]`
- 197 LEDs × 3 bytes = 591 bytes per frame

## Files

- `spectrum_stream.py` - Spectrum analyzer streamer
- `spectrum_visualizer.py` - Real-time spectrum visualization
- `requirements.txt` - Python dependencies
- `../src/main.cpp` - Arduino streaming receiver (shared)
- `../platformio.ini` - Build configuration (shared)

## Troubleshooting

**"Could not open serial port"**
- Check Arduino is connected
- Try different port: `ls /dev/cu.usb*`
- Make sure no other streaming script is running

**Not reacting to bass (energy mode)**
- Lower BASS_THRESHOLD (edit line 36 in spectrum_stream.py)
- Check BlackHole audio routing in System Settings
- Play louder music with more bass
- Use spectrum_visualizer.py to see what's being detected

**Choppy/laggy**
- Normal latency: 50-100ms at 60 FPS
- Lower FPS if needed (edit line 27)
- Check CPU usage

## Important Gotcha: Timing and LED Stability

**Problem:** Sending LED frames directly from audio callbacks causes LED blinking/glitching.

**Why:** Audio callbacks fire at irregular intervals (~46ms ± jitter) based on when audio buffer is ready. This timing jitter in serial communication causes marginal LEDs to glitch, even though data is correct.

**Solution:** **Decouple audio analysis from frame sending**:
```python
# ❌ WRONG - Causes LED blinking
def audio_callback(audio):
    spectrum = analyze(audio)
    frame = generate_frame(spectrum)
    send_frame(frame)  # Irregular timing!

# ✅ CORRECT - Stable LEDs
def audio_callback(audio):
    spectrum = analyze(audio)
    self.latest_spectrum = spectrum  # Just store

def main_loop():
    while True:
        frame = generate_frame(self.latest_spectrum)
        send_frame(frame)  # Fixed-rate timing!
        sleep(1/FPS)
```

**Key insight:** WS2812B LEDs need **consistent timing** between updates. Irregular callback timing causes serial jitter → LED data corruption → blinking. Always use fixed-rate loops for sending frames, even if audio analysis happens asynchronously.

This issue manifests as:
- Single LED blinking while others work fine
- Blinking only during streaming, not with static colors
- Works fine at very low FPS (1-5) but breaks at higher rates

**Debug test:** If a LED blinks with audio streaming but works with time-based pulsation at the same FPS, it's a timing issue, not hardware.

## Difference from Basic Audio

For simpler bass-reactive animation without spectrum analysis, see `../basic-audio-reactive/`:
- ✅ Simple threshold-based bass detection
- ✅ Pulsating background with bass flashes
- ✅ Easy to understand and tune

This project (spectrum analyzer):
- ✅ Advanced mel-scale FFT analysis
- ✅ Multiple visualization modes
- ✅ Frequency-specific effects
- ✅ Inspired by scottlawsonbc/audio-reactive-led-strip
