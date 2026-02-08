# Audio-Reactive Effect Comparison Framework

A/B test different audio-reactive algorithms on the same audio through the same LED hardware.

## Quick Start

```bash
# List available effects
python runner.py --list

# Run an effect on live audio (BlackHole)
python runner.py wled_volume
python runner.py wled_geq
python runner.py wled_beat

# Terminal-only (no LEDs)
python runner.py wled_volume --no-leds

# WAV file playback
python runner.py wled_geq --wav ../audio-segments/fa_br_drop1.wav --no-leds
```

## Available Effects

### WLED Sound Reactive (`wled_sr/`)
Reimplementation of WLED's audio processing algorithms in Python.
See `wled_sr/notes.md` for full algorithm analysis.

- **wled_volume** — RMS volume → brightness (simplest baseline)
- **wled_geq** — 16 FFT bands → colored bars per frequency range
- **wled_beat** — Dominant FFT bin threshold → pulse/decay flash

## Architecture

```
base.py      — AudioReactiveEffect abstract class
runner.py    — Audio capture + LED driving loop
wled_sr/     — WLED Sound Reactive algorithms
ours/        — Our custom detectors (TODO: wrap in same interface)
```

Key constraint (from WS2812B timing): audio processing and LED rendering
are decoupled. `process_audio()` runs in the audio callback thread,
`render()` runs in the fixed-rate main loop.

## Options

- `--no-leds` — Terminal visualization only (no serial)
- `--wav FILE` — Play WAV file instead of live BlackHole capture
- `--port PORT` — Serial port (auto-detected if omitted)
- `--leds N` — Number of LEDs (default 197 for tree)
- `--brightness F` — Brightness cap 0-1 (default 0.03 = 3%)
