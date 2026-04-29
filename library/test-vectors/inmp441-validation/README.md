# INMP441 Frequency-Response Validation

Two synchronized 60-second captures comparing the on-board INMP441 MEMS microphone (acoustic pickup) against system audio (BlackHole loopback) while music plays. Captured to validate the `inmp441-frequency-response` ledger entry, which claimed the INMP441 is ~24 dB quieter overall with severe bass loss but treble "comparable" to system audio. Two mic positions tested: ambient (~3 ft from speaker, off-axis) and near-field (touching the speaker face).

## Per-band Δ dB (inmp441 − system, negative = mic quieter)

| band | range | ledger claim | ambient | near-field |
|---|---|---|---|---|
| sub-bass | 20-80 Hz | -15 | -16.5 | -12.4 |
| bass | 80-250 Hz | -28 | -30.5 | -16.4 |
| mids | 250-2000 Hz | -21 | -35.8 | -16.2 |
| high-mid | 2-6 kHz | (none) | -40.5 | -27.7 |
| treble | 6-8 kHz | "comparable" (~0) | -43.0 | -36.1 |
| broadband RMS | — | -24 | -21.8 | -15.7 |
| envelope Pearson | — | — | 0.575 | 0.824 |
| alignment lag | — | — | 270 ms | 110 ms |

## Key findings

- **Bass loss is real and dominant.** ~16 dB at zero distance, ~30 dB at three feet — the bass deficit is the load-bearing artistic constraint and reproduces robustly.
- **Distance matters a lot.** Going from 3 ft to 0 cm cut broadband loss from -22 to -16 dB and pulled mids/bass much closer to the system signal. The envelope-tracking quality (Pearson) jumped from 0.58 to 0.82 — at three feet the mic is tracking a heavily filtered room reflection rather than the source.
- **The ledger's "treble comparable" claim does NOT reproduce.** Treble (6-8 kHz) is the most attenuated band in both runs (-36 to -43 dB), consistent with INMP441 datasheet roll-off above 6 kHz at moderate SPL. The original measurement may have been clipped, used a different band definition, or used a much louder source.
- **The system-audio reference shifted between the two runs** (peak 0.58 → 0.37). Inter-run absolute deltas should be read with care; per-band ratios within each run are still valid.

## Capture metadata

| | ambient | near-field |
|---|---|---|
| mic position | 3 ft from speaker, on a table, not in direct point-source path | literally on speaker face (~0 cm) |
| room | small bedroom | small bedroom |
| speaker | unspecified by user | same as ambient |
| volume | ambient | unspecified |
| timestamp | 2026-04-26 18:28:15 | 2026-04-26 18:38:45 |
| firmware | raw_audio_sender (16 kHz / 32-bit / int16 over USB CDC) | same |
| sample rate | 16000 Hz mono | 16000 Hz mono |
| duration | 60.10 s | 60.10 s |

Full metadata in `ambient/metadata.yaml` and `nearfield/metadata.yaml`, including PSD numbers per band.

## How to reproduce

Three artifacts make up the full toolchain. Refer to them by name — locate the
current files with grep (paths in prose go stale on every refactor).

- **Firmware**: PIO env `raw_audio_sender` — INMP441 → 16 kHz int16 PCM over
  USB CDC @ 460800 baud. ESP32-C3 super-mini, I2S pins SCK=6 WS=5 SD=0.
  Find with `rg '\[env:raw_audio_sender\]' festicorn/` to get the current
  `platformio.ini` and pass its parent dir to `pio run -d <dir>`.
- **Capture script**: `dual_capture.py` — two thread-synchronized streams
  (serial PCM + BlackHole loopback) gated on a shared `threading.Event` for
  ~ms-level start alignment. BlackHole 2ch is downmixed to mono and resampled
  to 16 kHz to match.
- **Comparison script**: `compare_inmp441_vs_system.py` — Welch PSD per band,
  envelope cross-correlation for alignment, Pearson correlation, dB delta
  table, and spectrogram PNGs.

Pre-flight: INMP441 wired to ESP32-C3 (SCK=6 WS=5 SD=0); device enumerated on
a USB CDC port; macOS BlackHole 2ch installed with a Multi-Output Device
routing music to both speakers and BlackHole; music playing audibly.

Steps:

1. Flash the `raw_audio_sender` env to the ESP32-C3.
2. Run `dual_capture.py 60` to capture 60 s into this directory.
3. Run `compare_inmp441_vs_system.py` (auto-picks the newest pair).
4. Re-flash the regular `stream_sender` env when done.

## File manifest

```
ambient/
  inmp441_20260426_182815.wav     1.92 MB  60.17 s @ 16 kHz mono int16
  system_20260426_182815.wav      1.92 MB  60.00 s @ 16 kHz mono (resampled from 44.1 kHz)
  metadata.yaml                   per-band PSD, broadband, alignment, firmware sha
  spec_inmp441_20260426_182815.png   spectrogram (log freq, 80 dB range)
  spec_system_20260426_182815.png    spectrogram (same)
  psd_compare_20260426_182815.png    overlaid Welch PSD curves

nearfield/
  inmp441_20260426_183845.wav     same shape as ambient
  system_20260426_183845.wav
  metadata.yaml
  spec_inmp441_20260426_183845.png
  spec_system_20260426_183845.png
  psd_compare_20260426_183845.png
```
