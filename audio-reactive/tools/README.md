# Tools

Audio segment management CLI. All tools use `venv/` at the repo root.

```bash
source ../../venv/bin/activate  # from tools/
```

---

## Quick Reference

```bash
python segment.py web                               # Browser-based explorer
python segment.py list                              # Show catalog
python segment.py record                            # Record from BlackHole
python segment.py trim "Opiate Intro.wav"           # Auto-trim silence
python segment.py trim "Opiate Intro.wav" 2.5       # Manual trim at 2.5s
python segment.py play "Opiate Intro.wav"           # Interactive viewer
python segment.py play "Opiate Intro.wav" --annotate beat   # Tap annotation
python segment.py stems electronic_beat.wav          # Stem viewer (demucs)
python segment.py hpss electronic_beat.wav           # Harmonic/percussive viewer
python segment.py export "Opiate Intro.wav"         # Static PNG
python segment.py export                            # All WAVs
```

---

## Subcommands

### `web` — Browser-Based Explorer

Opens a web-based audio analysis viewer with perfect audio sync. Scans `audio-segments/` and `harmonix/` for WAV files.

Three tabs:
- **Analysis**: waveform, spectrogram, band energy, onset, centroid panels
- **Annotations**: same panels + annotation layers (if `.annotations.yaml` exists)
- **Stems**: 4 stem spectrograms (drums/bass/vocals/other via demucs, lazy)

Controls: Space play/pause, arrow keys +-5s, click panel or progress bar to seek.

```bash
python segment.py web              # Auto-pick port, opens browser
python segment.py web --port 8080  # Fixed port
```

### `list` — Show Catalog

Reads `audio-segments/catalog.yaml` and prints a table of all recorded segments.

### `record` — Record Audio

Records stereo audio from BlackHole at 44.1kHz. Press ENTER to stop, then fill in metadata (song, artist, genre, BPM, notes). Saves WAV + updates `catalog.yaml`.

### `trim <file> [start]` — Trim Silence

Removes silence from the start of a WAV file. Auto-detects where music begins (RMS threshold), or accepts a manual start time in seconds. Creates a `_trimmed.wav` file. Uses only stdlib `wave` — no heavy dependencies.

### `play <file>` — Interactive Visualizer

Opens a matplotlib window with synced audio playback and analysis panels:
- Waveform
- Mel spectrogram (20 Hz - 22 kHz)
- Band energy (sub-bass, bass, mids, high-mids, treble)
- User annotations (if present)
- Features: onset strength, spectral centroid, RMS (toggleable with O/C/E/B keys)

**Normal mode controls:**
| Key | Action |
|-----|--------|
| SPACE | Play / Pause |
| CLICK | Seek to position |
| O | Toggle onset strength + markers |
| C | Toggle spectral centroid |
| E | Toggle RMS energy |
| B | Toggle librosa-beats |
| Q | Quit |

**Options:**
- `--panel <name>` — Maximize one panel (waveform, spectrogram, bands, features, annotations)
- `--show-beats` — Show librosa-beats markers on startup (also toggleable with B key)
- `--save-png` — Save static PNG instead of opening viewer

### `play <file> --annotate <layer>` — Tap Annotation

Same visualizer, but in annotation mode. Tap SPACE while listening to mark moments. Multiple runs with different layer names build rich multi-layer annotations.

**Annotation mode controls:**
| Key | Action |
|-----|--------|
| SPACE | Record a tap |
| P | Play / Pause |
| R | Restart + clear taps |
| Q | Save & quit |

Taps are saved to `<file>.annotations.yaml` next to the WAV. New taps replace the target layer; other layers are preserved.

**Common layers:** beat, kick, snare, airy, heavy, tension, drop

### `stems <file>` — Instrument Decomposition

Separates audio into 4 stems (drums, bass, vocals, other) using [demucs](https://github.com/adefossez/demucs) and opens an interactive 4-row spectrogram viewer with synced playback.

First run separates the audio (~25s on CPU); subsequent runs use cached stems from `audio-segments/separated/htdemucs/<name>/`.

**Controls:**
| Key | Action |
|-----|--------|
| SPACE | Play / Pause |
| CLICK | Seek to position |
| 1 / 2 / 3 / 4 | Toggle drums / bass / vocals / other |
| A | All stems on (reset) |
| Q | Quit |

Active stems play summed audio. Muted stems show dimmed spectrograms.

### `hpss <file>` — Harmonic/Percussive Separation

Splits audio into harmonic (pitched/tonal) and percussive (transient/drums) components using librosa's HPSS — no ML model, no downloads, instant computation. Same interactive viewer as `stems` but with 2 rows.

**Controls:**
| Key | Action |
|-----|--------|
| SPACE | Play / Pause |
| CLICK | Seek to position |
| 1 / 2 | Toggle harmonic / percussive |
| A | All on (reset) |
| Q | Quit |

Useful for evaluating whether HPSS separation quality is sufficient for real-time LED effects (HPSS is frame-by-frame, trivially real-time on ESP32).

### `export [file]` — Static PNG

Generates a 6-panel analysis PNG (same layout as `play` but without interactivity). When no file is given, exports all WAV files in `audio-segments/`.

---

## LED Effect Testing

Real-time LED effects have moved to `comparison/` alongside WLED-SR algorithms:

```bash
cd ../comparison
python runner.py --list                  # Show all effects
python runner.py bass_flux --no-leds     # Bass flux detector (terminal only)
python runner.py onset --no-leds         # Onset strength detector
python runner.py wled_beat --no-leds     # WLED-SR beat reactive
```

See `comparison/` for full documentation on the A/B testing framework.

---

## Architecture

```
segment.py                     viewer.py              web_viewer.py
  ├─ web    → lazy imports ─────────────────────────── run_server()
  │                               │                      ├─ Agg PNG rendering
  │                               │                      ├─ WAV Range serving
  │                               │                      └─ HTML SPA + cursor sync
  ├─ list   (inline, yaml)      ├─ SyncedVisualizer (interactive)
  ├─ record (inline, sounddevice)│   ├─ Normal mode (play/pause/seek)
  ├─ trim   (inline, stdlib wave)│   └─ Annotation mode (tap/save)
  ├─ play   → lazy imports ──────┤
  ├─ stems  → lazy imports ──────├─ StemVisualizer (4-stem viewer)
  └─ export → lazy imports ──────└─ export_static_png() (batch)
```

Light commands (list, record, trim) have no heavy dependencies. Heavy commands (play, export) lazy-import `viewer.py` which pulls in librosa + matplotlib.
