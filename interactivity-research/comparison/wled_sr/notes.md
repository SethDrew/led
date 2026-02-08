# WLED Sound Reactive — Algorithm Analysis

Source: `github.com/Aircoookie/WLED`, main branch, `usermods/audioreactive/audio_reactive.cpp` + `wled00/FX.cpp`

## Audio Pipeline

- **Sample rate**: 22050 Hz (we use 44100 Hz)
- **FFT size**: 512 samples (we use 1024 → same 43.07 Hz frequency resolution)
- **Windowing**: not documented in source (likely Hann or rectangular)
- **Min cycle**: 21ms between FFT runs (~47 Hz update rate)

## Data Exposed to Effects

Effects access audio data via `getAudioData()` → `um_data_t`:

| Index | Variable | Type | Description |
|-------|----------|------|-------------|
| 0 | volumeSmth | float | 16-sample EMA of volume |
| 2 | fftResult | uint8_t[16] | 16 frequency bands, 0-255, log-mapped |
| 3 | samplePeak | uint8_t | 1 if peak detected this frame |
| 4 | FFT_MajorPeak | float | Dominant frequency (Hz) |
| 5 | FFT_Magnitude | float | Magnitude of dominant frequency |
| 6 | maxVol | uint8_t* | Peak threshold (adjustable) |
| 7 | binNum | uint8_t* | Selected FFT bin (adjustable) |

## 16 Frequency Bands (fftResult)

At 22050/512 = 43.07 Hz per bin (our 44100/1024 is identical):

| Band | Bins | Freq Range | Pink Noise Factor |
|------|------|------------|-------------------|
| 0 | 1-2 | 43-86 Hz | 1.70 |
| 1 | 2-3 | 86-129 Hz | 1.71 |
| 2 | 3-5 | 129-216 Hz | 1.73 |
| 3 | 5-7 | 216-301 Hz | 1.78 |
| 4 | 7-10 | 301-430 Hz | 1.68 |
| 5 | 10-13 | 430-560 Hz | 1.56 |
| 6 | 13-19 | 560-818 Hz | 1.55 |
| 7 | 19-26 | 818-1120 Hz | 1.63 |
| 8 | 26-33 | 1120-1421 Hz | 1.79 |
| 9 | 33-44 | 1421-1895 Hz | 1.62 |
| 10 | 44-56 | 1895-2412 Hz | 1.80 |
| 11 | 56-70 | 2412-3015 Hz | 2.06 |
| 12 | 70-86 | 3015-3704 Hz | 2.47 |
| 13 | 86-104 | 3704-4479 Hz | 3.35 |
| 14 | 104-165 | 4479-7106 Hz | 6.83 (×0.88) |
| 15 | 165-215 | 7106-9259 Hz | 9.55 (×0.70) |

## Key Algorithm Details

### Volume (getSample)
- 16-sample EMA: `sampleAvg = (sampleAvg * 15 + sample) / 16`
- Noise gate: `squelch` threshold (default 10)
- AGC: PI controller with 3 presets (normal/vivid/lazy)

### FFT Post-Processing
- **Asymmetric smoothing**: attack 0.75/0.25, decay 0.22/0.78
- **Scaling modes**: linear (`*0.30 - 4`), log (`log(x*0.42 - 8)`), sqrt (`sqrt(x*0.38 - 6)`)
- **Pink noise compensation**: per-band multipliers (see table above)
- **FFT_DOWNSCALE**: 0.46 (for 22kHz sampling)

### Beat/Peak Detection (samplePeak)
- Simple threshold: `vReal[binNum] > maxVol`
- Requires: `sampleAvg > 1 && maxVol > 0 && binNum > 4`
- Min interval: 100ms between peaks
- Auto-reset: 50ms or frame time after peak

## 1D Audio Effects (12 total)

| Effect | Audio Data Used | Visual |
|--------|----------------|--------|
| Juggles | volumeSmth | Colored dots juggling, brightness = volume |
| Midnoise | volumeSmth, FFT_MajorPeak | Noise pattern colored by frequency |
| Noisemeter | volumeSmth, samplePeak | Volume meter with peak flash |
| Plasmoid | volumeSmth, samplePeak | Plasma effect modulated by volume |
| Blurz | fftResult[16] | Blurred frequency bands |
| DJLight | fftResult[16] | Frequency-colored light show |
| Freqmap | FFT_MajorPeak, volumeSmth | Map frequency to LED position |
| Freqmatrix | FFT_MajorPeak, volumeSmth | Scrolling frequency matrix |
| Freqpixels | FFT_MajorPeak, volumeSmth | Random pixels colored by frequency |
| Freqwave | FFT_MajorPeak, volumeSmth | Center-outward waves, color = frequency |
| Noisemove | fftResult[16] | Perlin noise positioned by frequency |
| Rocktaves | FFT_MajorPeak, magnitude | Musical octave colors |

## Key Observations

1. **Beat detection is extremely simple** — just "is this bin louder than threshold?"
   No spectral flux, no tempo tracking, no onset detection. Our bass flux is more sophisticated.

2. **Most effects use volume OR frequency, not both** — they don't combine features
   like derivatives, deviation from context, or multi-timescale analysis.

3. **No "feeling layer"** — no concept of airiness, build detection, or section awareness.
   Every effect reacts to the current audio frame independently.

4. **The real magic is in the visual effects**, not the audio analysis. WLED has
   beautiful scrolling, blurring, and color mapping that makes simple audio data look good.

5. **AGC is their most sophisticated audio feature** — a full PI controller with
   3 presets. This is genuinely useful and worth stealing.

6. **Pink noise compensation is thoughtful** — they account for the fact that higher
   frequencies naturally have less energy. This flattens the GEQ response.
