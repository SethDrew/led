# Audio-Reactive Effect Comparison Framework

A/B test different audio-reactive algorithms on the same audio through the same LED hardware.

## Quick Start

```bash
# List available effects
python runner.py --list

# Run an effect on live audio (BlackHole)
python runner.py bass_pulse
python runner.py energy_color

# Terminal-only (no LEDs)
python runner.py bass_pulse --no-leds

# WAV file playback
python runner.py bass_pulse --wav ../audio-segments/fa_br_drop1.wav --no-leds
```

## Available Effects

Run `python runner.py --list` for the current registry.

### WLED Sound Reactive (`wled_sr/`)
Reimplementation of WLED's audio processing algorithms in Python — kept as a
research reference. Classes here do not register as runnable effects; see
`wled_sr/notes.md` for the algorithm analysis.

## Architecture

```
base.py      — AudioReactiveEffect abstract class
runner.py    — Audio capture + LED driving loop
color/       — Shared OKLCH color machinery (RAINBOW_LUT, swatch, palette)
wled_sr/     — WLED Sound Reactive reference algorithms
deprecated/  — Soft-archived effects (managed by the viewer UI)
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
