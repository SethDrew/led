# Tools

Executable scripts for the audio-reactive LED pipeline. All tools use `venv/` at the repo root.

```bash
source ../../venv/bin/activate  # from tools/
```

---

## Real-Time LED Controllers

### realtime_beat_led.py — Bass Flux Detector

Bass-band spectral flux (20-250 Hz) beat detector. Flashes the LED tree red on kick drums.

```bash
python realtime_beat_led.py              # LED tree (auto-detect serial)
python realtime_beat_led.py --no-leds    # Terminal meter only
```

### realtime_onset_led.py — Onset A/B Test

Dual detector: bass flux vs full-spectrum onset. Color-coded for visual comparison.

```bash
python realtime_onset_led.py --detector both     # RED=bass, BLUE=onset, WHITE=both agree
python realtime_onset_led.py --detector bass      # Bass only (RED)
python realtime_onset_led.py --detector onset     # Onset only (BLUE)
python realtime_onset_led.py --no-leds --detector both  # Terminal only
```

**Threshold tuning** (higher = fewer beats):
```bash
python realtime_onset_led.py --detector both --bass-threshold 2.0 --onset-threshold 3.0
```

**Troubleshooting:**
- "BlackHole not found" — System Settings > Sound > Output > BlackHole 2ch
- "No serial port" — plug in Arduino or use `--no-leds`
- No beats — lower thresholds (`--bass-threshold 1.0 --onset-threshold 1.0`)

---

## Audio Capture & Annotation

### record_segment.py — Record Audio

Records audio from BlackHole for analysis.

### annotate_segment.py — Tap Annotation

Play audio and press a key at musical moments. Multiple passes with different layer names build multi-layer annotations.

```bash
python annotate_segment.py ../audio-segments/opiate_intro.wav beat
python annotate_segment.py ../audio-segments/opiate_intro.wav air
```

### playback_segment.py / playback_visualizer.py — Playback

Play back recorded segments, optionally with real-time visualization.

---

## Visualization

### visualize_segment.py — Static Analysis

Generates a multi-panel PNG: waveform, mel spectrogram, band energy (5 bands), onset/beat detection, spectral centroid, RMS energy.

```bash
python visualize_segment.py ../audio-segments/electronic_beat.wav   # Single file
python visualize_segment.py                                          # All WAVs
```

### trim_audio.py — Audio Trimming

Trim audio segments to specific time ranges.

---

## Architecture (Onset A/B Test)

Both detectors share the same interface: `process_chunk(audio, use_realtime=True) -> (is_beat, strength, threshold)`

```
Audio Callback Thread (23ms chunks)     Main Thread (30 FPS fixed rate)
  ├─ BassFluxDetector                     ├─ Read shared state (mutex)
  │   └─ FFT → bass bins → flux           ├─ Update decay
  ├─ OnsetDetector                        ├─ Render frame
  │   └─ FFT → mel bands → flux           ├─ Send serial (591 bytes)
  └─ Store in shared state                └─ Sleep to next frame target
```

**Key design:** Audio callback ONLY stores state. Main loop ONLY reads state and sends frames. Decoupled to prevent WS2812B timing corruption.

### Detector Parameters

| Parameter | Bass Flux | Onset |
|-----------|-----------|-------|
| Frequency range | 20-250 Hz | 20-8000 Hz (mel) |
| FFT size | 2048 | 2048 |
| Mel bands | — | 40 |
| Threshold | 1.5x std | 2.0x std |
| Min interval | 0.3s (200 BPM) | 0.1s (600 BPM) |
| History window | 3s | 3s |
| Decay | 0.12 | 0.12 |
| Gamma | 2.2 | 2.2 |

### Performance (Opiate Intro)

| Detector | Detected | True Positives | F1 |
|----------|----------|----------------|-----|
| Bass flux | 60 | 8 | 0.158 |
| Onset | 62 | 13 | 0.252 |

Both over-detect (expected — tuned for sensitivity). Onset scores 60% better on rock.

### Offline Validation

```bash
cd ../analysis/scripts
python validate_onset_detector.py
```
