# Audio Explorer

An interactive audio visualization and audio interactivity research testbed. It's configured to require no user accounts. It allows you to upload or record and analyze any audio you would like. This audio is stored on your machine only for copyright reasons.

**Live demo:** [audio.sethdrew.com](https://audio.sethdrew.com)

## What it does

- **Waveform + mel spectrogram + band energy** — core audio visualization with cursor sync
- **Tap annotations** — record what you feel while listening, compare against algorithms
- **Source separation** — Demucs (deep learning, 4-stem), HPSS (real-time, harmonic/percussive)
- **Lab experiments** — NMF decomposition, REPET pattern extraction, spectral features
- **Browser recording** — record system audio via BlackHole or microphone, no install needed
- **Audio-reactive LED effects** — drive WS2812B LEDs from audio analysis (local only)

## Your data

All uploaded and recorded audio is cached in your browser's **IndexedDB** storage. You can manage your files from the **Record** tab in the app, or clear everything manually:

- **Chrome/Edge:** Settings → Privacy → Site Data → `audio.sethdrew.com` → Remove
- **Firefox:** Settings → Privacy → Manage Data → `audio.sethdrew.com` → Remove
- **Safari:** Settings → Privacy → Manage Website Data → `audio.sethdrew.com` → Remove

Analysis results (spectrograms, stems, etc.) are also cached locally so repeat visits are instant.

## Run locally with Docker

Docker gives you the full application including Demucs source separation, which requires more resources than the public demo server provides.

### Quick start

```bash
docker run -p 8080:8080 -v ~/Music:/app/audio-reactive/research/audio-segments ghcr.io/sethdrew/led-viewer
```

Then open [http://localhost:8080](http://localhost:8080).

The `-v ~/Music:/app/...` flag mounts your music folder into the viewer. Change `~/Music` to wherever your WAV files are.

### Build from source

```bash
git clone https://github.com/SethDrew/led.git
cd led
docker build -t led-viewer .
docker run -p 8080:8080 -v ~/Music:/app/audio-reactive/research/audio-segments led-viewer
```

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac, Windows, Linux)
- WAV files to analyze (the viewer works with `.wav` format)

### What's included

| Feature | Docker (local) | Public site |
|---------|---------------|-------------|
| Analysis + spectrogram | Yes | Yes |
| Tap annotations | Yes | Yes |
| Upload / record audio | Yes | Yes |
| Demucs (4-stem separation) | Yes | No (needs ~4GB RAM) |
| HPSS (harmonic/percussive) | Yes | Yes (with passcode) |
| Lab (NMF, REPET, features) | Yes | Yes (with passcode) |
| Browser recording | Yes | Yes |
| LED effects | Yes (with hardware) | No |

## Run without Docker

```bash
git clone https://github.com/SethDrew/led.git
cd led
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install demucs  # optional, for source separation
cd audio-reactive/tools
python segment.py web
```

### Record system audio (macOS)

To record what's playing on your computer, install [BlackHole](https://existential.audio/blackhole/) (free virtual audio driver):

1. Install BlackHole 2ch
2. Set BlackHole 2ch as your system sound output
3. Use the Record tab in the viewer

To hear the audio while recording, create a **Multi-Output Device** in Audio MIDI Setup that includes both your speakers and BlackHole, and set that as your output instead.

## Project structure

```
audio-reactive/
  tools/          — segment.py CLI, web_viewer.py, viewer.py
  effects/        — audio-reactive LED effects
  research/       — analysis scripts, separation algorithms, datasets
firmware/         — Arduino/ESP32 LED controller firmware
static-animations/ — non-audio-reactive LED effects
```

## Contact

Questions, ideas, or improvements — reach out to seth at sethdrew dot com.

## License

MIT License.
