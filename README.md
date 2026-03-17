# LED

Interactive LED installation development. Work is ongoing across seven axes of design:

1. **Topologies** — sculpture shapes and spatial mapping
2. **Musical Events** — what happens in music that LEDs should respond to
3. **Audio Features** — what we can compute from audio in real time
4. **LED Behaviors** — the visual vocabulary (sparkle, pulse, flow, growth...)
5. **Temporal Scope** — frame-level transients to song-level arcs
6. **Composition** — how effects layer, blend, and transition
7. **Perceptual Mapping** — bridging audio features to visual parameters, e.g. color, brightness

Commits are welcome across any of these.

## Active Projects

**Audio Viewer** — browser-based audio analysis and visualization testbed. Waveform, mel spectrogram, tap annotations, source separation (Demucs, HPSS), and real-time LED effect preview. Used for research and effect development.

**Festicorn** — ESP32-C3 firmware for finalized sculpture installations. Drives WS2812B strips with OKLCH color palettes, gamma-corrected output, OTA updates, and a wireless web UI for live control.

## Local Setup

```bash
git clone https://github.com/SethDrew/led.git
cd led
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the viewer:
```bash
cd audio-reactive/viewer
python explore.py
```

Optionally install Demucs for 4-stem source separation:
```bash
pip install demucs
```

## License

MIT
