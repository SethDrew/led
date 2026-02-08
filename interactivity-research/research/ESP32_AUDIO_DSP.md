# ESP32 Audio DSP Feasibility Report
## Standalone Audio-Reactive LED Controller Research

**Date:** February 6, 2026
**Purpose:** Evaluate feasibility of running audio-reactive beat detection and feature extraction directly on ESP32 without computer streaming

---

## Executive Summary

**TL;DR: It's totally doable, but you'll trade flexibility for portability.**

The ESP32 can absolutely run real-time audio analysis for LED control:
- ✅ **FFT up to 2048 points** at frame rates suitable for LED control (30-60 FPS)
- ✅ **I2S microphones** can capture at 44.1kHz or 48kHz with good quality
- ✅ **Basic beat detection** (spectral flux, onset detection) is practical
- ✅ **Frequency band analysis** (bass/mid/treble energy) works well
- ⚠️ **Advanced features** (tempo tracking, harmonic analysis, sentiment) are limited
- ❌ **Complex ML models** and multi-second analysis windows are out of reach

**Best use case:** Standalone installation where you want audio-reactive effects without a computer. The ESP32 can handle the core algorithms from our research, but won't match the sophistication of PC-based analysis.

---

## 1. ESP32 Audio Input Options

### 1.1 I2S Digital Microphones (RECOMMENDED)

**Top Choices:**
- **INMP441** — Omnidirectional MEMS, 18-bit, up to 61kHz sample rate, $3-5
  - Most popular in ESP32 audio projects
  - Good SNR (61 dBA), low power (1.4mA)
  - PDM or I2S output

- **SPH0645LM4H** — I2S MEMS from Knowles/Adafruit, 18-bit
  - Better low-frequency response than INMP441
  - More expensive (~$7-10)
  - Used in Adafruit breakout boards

- **ICS-43434** — I2S MEMS from TDK InvenSense, 24-bit
  - Excellent SNR (65 dBA)
  - Flat frequency response 50Hz-20kHz
  - Best quality option (~$5-8)

**Sample Rates:** All support standard rates:
- 8kHz, 16kHz, 22.05kHz, 32kHz, **44.1kHz**, **48kHz**
- ESP32 I2S peripheral can handle up to 96kHz theoretically
- **Practical choice: 44.1kHz or 48kHz at 16-bit** (matches CD quality, familiar tooling)

**Bit Depth:**
- Hardware: 18-24 bit
- ESP32 I2S DMA: Can handle 16, 24, or 32-bit samples
- **Practical choice: 16-bit** (more than enough for beat detection, saves memory)

**Pros:**
- Digital signal (immune to ADC noise)
- Direct I2S DMA to memory (no CPU polling)
- Low latency
- Good SNR

**Cons:**
- Requires I2S-compatible mic (not standard 3.5mm audio)
- 3 GPIO pins (BCK, WS, DATA)
- Need to handle mic placement for best pickup

### 1.2 I2S PDM Microphones

Same as above but use Pulse Density Modulation:
- Single data line (saves GPIO)
- ESP32 has hardware PDM→PCM conversion
- Slightly noisier than standard I2S
- Examples: SPM1423, MP34DT05

**Verdict:** Standard I2S is better for audio quality.

### 1.3 Analog Microphone + ADC

Use ESP32's built-in ADC with MAX4466, MAX9814, or electret capsule:

**Pros:**
- Cheap ($1-3 for electret + amp)
- Familiar analog audio path

**Cons:**
- ESP32 ADC is **TERRIBLE** for audio:
  - Non-linear (needs calibration curves)
  - Noisy (especially when WiFi is active)
  - Max effective resolution ~9-10 bits (not 12-bit)
  - Sample rate limited to ~10kHz without artifacts
- Not recommended for quality audio analysis

**Verdict:** Only use for prototyping. I2S mics are worth the extra $3.

### 1.4 Line-In Audio Capture

Capture line-level audio (3.5mm aux, RCA):

**Option A: Analog line-in → ADC**
- Use voltage divider + DC bias to center signal at 1.65V (ESP32 ADC midpoint)
- Same ADC problems as mic input
- **Not recommended**

**Option B: I2S ADC chip (PCM1808, CS5343)**
- External stereo ADC with I2S output
- High quality (24-bit, 96kHz capable)
- More expensive ($5-15 per chip)
- Requires level shifting (3.3V I2S)
- **Good for DJ booth / aux input scenarios**

**Option C: I2S Audio Codec (WM8731, CS4344)**
- Full codec with ADC + DAC
- Can do line-in AND headphone monitoring
- Most complex but most flexible
- $8-20

**Verdict:** For aux input, use external I2S ADC. The quality jump is worth it.

### Audio Input Recommendation

**For standalone mic-based installation:**
- **ICS-43434** I2S microphone (best quality)
- or **INMP441** (most popular, proven)
- **48kHz sample rate, 16-bit**
- I2S pins: GPIO25 (BCK), GPIO26 (WS), GPIO27 (DATA)

**For line-in from mixer/phone:**
- **PCM1808** I2S ADC for stereo line input
- 44.1kHz or 48kHz, 16 or 24-bit

---

## 2. ESP32 DSP Libraries

### 2.1 ESP-DSP (Espressif Official) ⭐ RECOMMENDED

**What it is:**
- Official DSP library from Espressif
- Optimized for ESP32 Xtensa LX6 CPU
- Uses hardware accelerators where available (MAC instructions)
- Part of ESP-IDF (works in Arduino too via component)

**FFT Implementation:**
- Real FFT (rfft) and Complex FFT (cfft)
- Supports radix-2, radix-4 algorithms
- **Sizes: 64, 128, 256, 512, 1024, 2048, 4096**
- Uses lookup tables for speed
- Float32 or Int16 fixed-point

**Performance (ESP32 @ 240MHz):**
| FFT Size | Float32 | Int16 Fixed-Point |
|----------|---------|-------------------|
| 512      | ~1.5ms  | ~0.8ms            |
| 1024     | ~3.2ms  | ~1.7ms            |
| 2048     | ~7.0ms  | ~3.8ms            |
| 4096     | ~15ms   | ~8ms              |

*(These are approximate based on Espressif benchmarks; actual varies with cache hits)*

**Other Functions:**
- FIR/IIR filters
- Matrix operations
- Convolution
- Window functions (Hann, Hamming, Blackman)

**Memory:**
- 1024-point FFT: ~16KB for buffers + lookup tables
- 2048-point FFT: ~32KB
- Fits comfortably in ESP32's 520KB SRAM

**Verdict:** This is the gold standard for ESP32 DSP. Use this.

### 2.2 arduinoFFT

**What it is:**
- Popular Arduino library by @kosme
- Pure C++ implementation
- Works on any Arduino-compatible platform
- Available via Arduino Library Manager

**Performance:**
- **Slower than ESP-DSP** (no hardware optimization)
- 1024-point FFT: ~8-12ms on ESP32 @ 240MHz
- Only float64 (double precision) or float32
- No fixed-point option

**Pros:**
- Easy to install (Arduino Library Manager)
- Well-documented examples
- Widely used (lots of community examples)

**Cons:**
- 2-3x slower than ESP-DSP
- Higher memory usage (no optimized buffers)

**Verdict:** Fine for prototyping or if you're Arduino-only. But ESP-DSP is better.

### 2.3 kiss_fft

**What it is:**
- "Keep It Simple, Stupid" FFT library
- Portable C library (works anywhere)
- Small code size (~2KB compiled)

**Performance:**
- Middle ground: faster than arduinoFFT, slower than ESP-DSP
- 1024-point: ~5-7ms on ESP32
- Float or fixed-point (user-configured)

**Verdict:** Good option if you need portability or simpler codebase than ESP-DSP.

### 2.4 fix_fft (Fixed-Point FFT)

**What it is:**
- Ancient 8-bit fixed-point FFT for AVR/8051
- 16-bit input, 8-bit twiddle factors
- **Max size: 256 points**

**Verdict:** Too limited. Don't use on ESP32 (we have floating-point!).

### 2.5 CMSIS-DSP

**What it is:**
- ARM's DSP library for Cortex-M processors
- Highly optimized for ARM SIMD instructions

**Problem:**
- ESP32 uses **Xtensa LX6**, not ARM
- No official ESP32 port
- Some community attempts exist but unmaintained

**Verdict:** Doesn't apply to ESP32. (Would be great for Raspberry Pi Pico W.)

### DSP Library Recommendation

**Use ESP-DSP:**
- Fastest FFT (3-7ms for 2048 points)
- Official support
- Optimized for ESP32 architecture
- Good documentation

**Example code structure:**
```cpp
#include "esp_dsp.h"

// Setup
float* input = (float*)malloc(N * sizeof(float));
float* output = (float*)malloc(N * sizeof(float));
dsps_fft2r_init_fc32(NULL, N); // Init lookup tables

// In loop
dsps_wind_hann_f32(input, N); // Window
dsps_fft2r_fc32(input, N); // FFT
dsps_bit_rev_fc32(input, N); // Bit reversal
dsps_cplx2reC_fc32(input, N); // Convert to magnitude
```

---

## 3. Beat Detection on ESP32

### 3.1 Can Our Spectral Flux Algorithm Run?

**Our algorithm from research:**
1. Capture audio chunk (1024 samples @ 44.1kHz = 23ms)
2. Apply window (Hann)
3. Compute FFT (1024-point)
4. Extract bass bins (20-200 Hz)
5. Compute spectral flux (sum of positive differences)
6. Apply attack/decay envelope
7. Compare to adaptive threshold
8. Trigger beat if crossed

**ESP32 feasibility:**

✅ **Step 1-3: FFT** — 3.2ms (ESP-DSP 1024-point float32)
✅ **Step 4: Bass extraction** — <0.1ms (just summing ~10 bins)
✅ **Step 5: Spectral flux** — <0.1ms (subtract previous frame, half-wave rectify, sum)
✅ **Step 6: Envelope** — <0.05ms (simple exponential moving average)
✅ **Step 7-8: Threshold** — <0.05ms (comparison + adaptive tracking)

**Total CPU time per frame: ~3.5ms**
**Available time budget (23ms chunks): 23ms**
**CPU utilization: ~15%**

**Verdict: YES, spectral flux beat detection is totally practical on ESP32.**

You could even run MULTIPLE detectors simultaneously:
- Bass beat detector (20-200 Hz)
- Snare detector (200-800 Hz)
- Hi-hat detector (5-10 kHz)

### 3.2 Memory Budget

**ESP32 SRAM: 520KB total**

**Our beat detection memory:**
- Audio input buffer: 1024 samples × 2 bytes (int16) = 2KB
- FFT input: 1024 floats × 4 bytes = 4KB
- FFT output: 1024 floats × 4 bytes = 4KB
- Previous spectrum: 1024 floats = 4KB
- Spectral flux history (for adaptive threshold): 100 floats = 0.4KB
- ESP-DSP lookup tables: ~8KB

**Total: ~22KB**

**Remaining for LED buffers, network, system: ~500KB**

**Verdict: Memory is not a constraint. We're using <5% for audio analysis.**

### 3.3 Latency Analysis

**Latency sources:**
1. I2S DMA buffer fill: 1024 samples ÷ 44.1kHz = **23ms** (inherent)
2. FFT computation: **3.2ms**
3. Beat detection logic: **0.3ms**
4. LED update (FastLED or NeoPixelBus): **<1ms** for 300 LEDs

**Total latency: ~27ms**

This is **well under our 50ms target** from research. Humans perceive audio-visual sync below 50ms as "instant."

### 3.4 Frame Rate

With 23ms audio chunks + 3.5ms processing:
- We can process **~38 audio frames per second**
- LED update can be decoupled (60 FPS with interpolation)
- Or lock LED updates to audio frames (38 FPS — still smooth)

**Verdict: 30-60 FPS is achievable.**

### 3.5 Existing ESP32 Beat Detection Projects

While I can't search in real-time, here's what exists as of my knowledge cutoff:

**WLED Sound Reactive (SR-WLED):**
- Most popular ESP32 audio-reactive project
- Fork of WLED (LED controller firmware)
- Uses **I2S microphone (INMP441 or SPH0645)**
- Implements:
  - FFT (256 or 512 bins using arduinoFFT)
  - Frequency band extraction (bass, mid, treble)
  - Simple beat detection (bass energy threshold)
  - Volume reactive effects
  - AGC (automatic gain control)
- Runs at **40-60 FPS** depending on FFT size
- **Limitation:** Beat detection is basic (energy threshold, no tempo tracking)
- GitHub: `atuline/WLED` or `blazoncek/WLED` (multiple forks)

**FastLED + Audio Reactive Examples:**
- Many Arduino/ESP32 projects on GitHub
- Common pattern:
  - MSGEQ7 analog chip (7-band graphic equalizer) + ADC
  - or FFT with arduinoFFT
  - Map frequency bands to LED colors/brightness
- Examples: `NoiseMaker`, `aurora`, various `sound-reactive-led` repos

**ESP32-audioI2S:**
- Not beat detection, but shows I2S audio input patterns
- By @schreibfaul1
- Demonstrates clean I2S microphone integration

**LightLeaf:**
- Standalone ESP32 audio visualizer
- Uses FFT and beat detection
- Custom PCB with INMP441
- GitHub: `jasoncoon/LightLeaf` (approximate name)

**Key Insight from Existing Projects:**
- Everyone uses **simple energy-based beat detection** (bass threshold)
- No one implements **tempo tracking** or **onset detection with backtracking**
- Why? Memory and CPU for multi-second analysis windows
- Our spectral flux algorithm would be **state-of-the-art** for ESP32

---

## 4. aubio on ESP32

**What is aubio?**
- C library for audio analysis
- Features: onset detection, tempo tracking, pitch detection, beat tracking
- Used in our research as a reference implementation
- Designed for desktop/server (assumes plenty of memory)

**Can it run on ESP32?**

**Cross-compilation: YES, but...**
- aubio is standard C99 (no OS dependencies)
- Can compile with ESP-IDF toolchain
- aubio's dependencies:
  - **FFTW** (not available) → Need to replace with ESP-DSP
  - **libsamplerate** (optional, for resampling) → Can skip
  - Standard math library → Available

**Memory Requirements:**
- aubio's tempo tracker needs **several seconds of history** (6-10 seconds typical)
- At 44.1kHz: 6 seconds = 265K samples = **1 MB of audio buffer**
- ESP32 has **520KB SRAM**
- **Won't fit.**

**What COULD work:**
- **Onset detection only** (aubio's `aubio_onset` module)
  - Needs 2-3 frames of FFT history: ~8KB
  - This fits!
  - No tempo tracking, but still better than energy threshold
- Would need to replace aubio's FFT backend with ESP-DSP
- Significant porting effort (1-2 weeks for someone familiar with aubio internals)

**Existing Ports:**
- I'm not aware of any maintained aubio ESP32 port as of January 2025
- Some experiments on forums, but nothing production-ready

**Verdict: Onset detection is feasible with porting effort, but tempo tracking is out of reach due to memory.**

Alternative: Use our **spectral flux algorithm** (which is essentially onset detection) — easier to implement from scratch than porting aubio.

---

## 5. TensorFlow Lite Micro on ESP32

**What is TFLite Micro?**
- Stripped-down TensorFlow for microcontrollers
- Runs inference only (no training)
- Designed for low-memory, low-power devices

**ESP32 Support:**
- Official TFLite Micro supports ESP32
- Examples in TensorFlow repo: `tensorflow/lite/micro/examples/`
- Works with ESP-IDF and Arduino

**Model Size Limits:**
- **Practical limit: ~100-200KB model** (fits in flash + working memory)
- Larger models exist but start hitting SRAM limits during inference

**Latency:**
- Small models (10-50KB): **10-100ms inference** depending on complexity
- Medium models (100-200KB): **100-500ms**
- For real-time audio (30-60 FPS), we need **<30ms inference**

**Audio Classification Examples:**
- Google's "Micro Speech" example (wake word detection)
  - Model: 18KB
  - Input: 1-second audio window (16kHz)
  - Output: Yes/No/Unknown
  - Inference: ~15ms on ESP32
- This proves **small audio models can run in real-time**

**Beat Detection with ML:**

**Hypothetical approach:**
1. Train small CNN on Harmonix dataset (from our research)
2. Input: Mel spectrogram (24 bins × 10 frames = 240 features)
3. Output: Beat probability (single neuron)
4. Model architecture: 3 conv layers + 2 dense → ~50-100KB

**Feasibility:**
- ✅ Model would fit
- ⚠️ Inference might be **20-50ms** (too slow for 60 FPS, okay for 30 FPS)
- ⚠️ Training requires desktop + dataset (can't train on ESP32)
- ❌ Less transparent than spectral flux (harder to debug)

**Verdict: Possible but not practical.**

For beat detection, **handcrafted DSP algorithms** (our spectral flux) are:
- Faster (3ms vs 20ms)
- More transparent (tunable parameters)
- Don't require training data
- Work out of the box

ML makes sense for **genre classification** or **mood detection** where hand-crafted features fail — but that's beyond scope for LED control.

---

## 6. What We'd LOSE Going Standalone

### 6.1 Feature Extraction Limitations

**What works on ESP32:**
- ✅ FFT (up to 2048 points)
- ✅ Frequency band energy (bass, mid, treble)
- ✅ Spectral centroid (brightness)
- ✅ Spectral flux (onset detection)
- ✅ RMS energy (volume)
- ✅ Zero-crossing rate (noisiness)
- ✅ Basic beat detection (energy threshold or spectral flux)

**What's LIMITED on ESP32:**
- ⚠️ Mel filterbank (possible, but need to precompute filters)
- ⚠️ Spectral flatness (feasible but takes CPU)
- ⚠️ Chroma features (possible with 12 filters, but CPU-heavy)

**What WON'T work on ESP32:**
- ❌ **Tempo tracking** (needs multi-second history → memory)
- ❌ **HPSS** (harmonic-percussive separation — too complex)
- ❌ **Pitch tracking** (aubio's YIN needs large buffers)
- ❌ **Genre/sentiment classification** (needs complex ML or long analysis)
- ❌ **Multi-second analysis** (e.g., detecting song structure changes)
- ❌ **Dynamic time warping** (aligning beats across tempo changes)

### 6.2 The "Feeling Layer" Problem

From our research, the **feeling layer** (mapping audio features → subjective qualities like "airy" or "tense") requires:
- Multiple features combined
- User annotation and iterative tuning
- Potentially ML models trained on custom datasets

**On PC:**
- We can run 20+ features simultaneously
- Store minutes of audio history
- Retrain models on the fly
- Experiment rapidly

**On ESP32:**
- Limited to 3-5 features (CPU budget)
- No history beyond a few frames
- No runtime ML
- Experimentation requires reflashing firmware

**Impact:** ESP32 can do **reactive** effects (beat flashes, volume pulsing) but struggles with **interpretive** effects (mood-based color shifts, structure-aware transitions).

### 6.3 Development Velocity

**PC streaming (current approach):**
- Edit Python script → see changes instantly
- Add new features in minutes
- Debug with print statements, plots, logs
- Hot-reload effects without restarting

**ESP32 standalone:**
- Edit C++ code → compile → upload → test (2-5 minute cycle)
- Adding features requires understanding ESP-IDF + DSP math
- Debugging via Serial.print() or JTAG (slower)
- No hot-reload (must reboot ESP32)

**Impact:** Iteration speed drops by 10-100x.

### 6.4 Quality vs. Portability Tradeoff

| Aspect               | PC Streaming (librosa + Python) | ESP32 Standalone (ESP-DSP + C++) |
|----------------------|----------------------------------|----------------------------------|
| **Audio Features**   | 20+ features, unlimited depth    | 5-10 features, shallow analysis  |
| **Beat Detection**   | aubio's full tempo tracking      | Spectral flux or energy threshold|
| **Latency**          | 30-50ms (USB + processing)       | 20-30ms (all on-chip)            |
| **Portability**      | Needs PC nearby (not portable)   | Fully standalone (battery even)  |
| **Flexibility**      | Rapid experimentation            | Slow firmware updates            |
| **Cost**             | PC required (~$500+)             | ESP32 + mic (~$10 total)         |
| **Complexity**       | High (learn librosa/numpy)       | High (learn ESP-IDF/DSP)         |

---

## 7. Recommended Hardware Setup

### Option A: Best Quality (Recommended)

**Microcontroller:**
- **ESP32-DevKitC-32E** or **ESP32-WROVER-E** (with PSRAM)
  - WROVER has 4MB or 8MB PSRAM (external RAM)
  - Not needed for beat detection, but nice for LED buffers + WiFi
  - ~$8-12

**Microphone:**
- **ICS-43434** I2S MEMS microphone breakout
  - Best SNR and frequency response
  - ~$6-8 (Adafruit, Tindie)
  - or **INMP441** if budget-constrained (~$3)

**LED Output:**
- WS2812B (NeoPixel) or APA102 (DotStar)
- Use **RMT peripheral** (ESP32 hardware for WS2812 timing) or **I2S parallel output**
- Library: **FastLED** or **NeoPixelBus**

**Power:**
- 5V supply (USB or dedicated)
- ESP32 runs at 3.3V (onboard regulator)
- WS2812B needs 5V logic (ESP32's 3.3V is usually okay for first LED)

**Wiring:**
```
ESP32          ICS-43434
GPIO25   -->   SCK (bit clock)
GPIO26   -->   WS (word select / LRCLK)
GPIO27   -->   SD (serial data)
3.3V     -->   VDD
GND      -->   GND

ESP32          LED Strip
GPIO21   -->   Data In (WS2812B)
5V       -->   VCC (from power supply)
GND      -->   GND (common ground)
```

**Total cost: ~$20 (ESP32 + mic + 1m LED strip)**

### Option B: Line-In Version (DJ Booth / AUX Input)

**Microcontroller:**
- Same ESP32-WROVER-E

**Audio Input:**
- **PCM1808** I2S Stereo ADC module (~$8-12)
  - 24-bit, up to 96kHz
  - 3.5mm line input
  - I2S output to ESP32
  - or **MAX98357A** I2S module (has ADC version)

**Wiring:**
```
ESP32          PCM1808
GPIO25   -->   BCK
GPIO26   -->   LRCK
GPIO27   -->   DOUT (I2S data)
3.3V     -->   VDD
GND      -->   GND

Audio Source (phone/mixer) --> 3.5mm to PCM1808 line-in
```

**Total cost: ~$30 (ESP32 + ADC + LEDs)**

### Option C: Budget Prototype (Not Recommended)

- ESP32-DevKitC (~$6)
- MAX4466 electret mic + ADC (~$2)
- **Warning:** Poor audio quality due to ESP32 ADC issues
- Only for proof-of-concept

---

## 8. Recommended Software Stack

### 8.1 Development Environment

**Use ESP-IDF (not Arduino)** for best performance:
- ESP-IDF is Espressif's official SDK
- Better control over I2S, DMA, interrupts
- ESP-DSP integrates seamlessly
- Arduino is built on top of ESP-IDF (adds overhead)

**But Arduino is easier for prototyping:**
- If starting with Arduino, use **ESP32 Arduino Core**
- Can use ESP-DSP as an Arduino component
- Migrate to ESP-IDF later if needed

### 8.2 Core Libraries

**Audio Input:**
- **ESP32-I2S driver** (built into ESP-IDF)
- Example: `esp-idf/examples/peripherals/i2s/`
- Configure for I2S microphone mode (I2S_MODE_MASTER | I2S_MODE_RX)

**DSP:**
- **ESP-DSP** (install via ESP-IDF component or Arduino library)
  ```bash
  idf.py add-dependency "espressif/esp-dsp"
  ```

**LED Control:**
- **FastLED** (Arduino library) — easiest, works with ESP32
- **NeoPixelBus** (better for ESP32, uses RMT)
- **ESP32 RMT driver** (ESP-IDF) — lowest level, most control

**Networking (optional):**
- **ESPAsyncWebServer** — host web UI for controls
- **ArduinoJSON** — parse config files
- **MQTT** — integrate with smart home

### 8.3 Code Architecture

```
main.cpp
├── setup()
│   ├── i2s_init()           // Configure I2S microphone
│   ├── dsp_init()           // Init ESP-DSP FFT tables
│   ├── led_init()           // Init FastLED/NeoPixelBus
│   └── wifi_init()          // Optional web UI
│
└── loop()
    ├── audio_read()         // Read chunk from I2S DMA
    ├── audio_process()      // FFT + feature extraction
    │   ├── fft()
    │   ├── extract_bands()  // Bass, mid, treble
    │   └── detect_beat()    // Spectral flux algorithm
    ├── map_to_leds()        // Audio features → LED states
    └── led_update()         // Push to LED strip
```

**Key pattern:** Use **FreeRTOS tasks** to parallelize:
- Task 1: Audio input (high priority, real-time)
- Task 2: DSP processing (medium priority)
- Task 3: LED rendering (medium priority)
- Task 4: WiFi/UI (low priority, optional)

This prevents WiFi from blocking audio (WiFi is notoriously interrupt-heavy on ESP32).

### 8.4 Example Code Skeleton

```cpp
#include <Arduino.h>
#include <FastLED.h>
#include "driver/i2s.h"
#include "esp_dsp.h"

#define I2S_WS 26
#define I2S_SD 27
#define I2S_SCK 25
#define LED_PIN 21
#define NUM_LEDS 150
#define SAMPLE_RATE 44100
#define SAMPLE_BUFFER_SIZE 1024

CRGB leds[NUM_LEDS];
int16_t sampleBuffer[SAMPLE_BUFFER_SIZE];
float fftInput[SAMPLE_BUFFER_SIZE];
float fftOutput[SAMPLE_BUFFER_SIZE];

void setup() {
  Serial.begin(115200);

  // Init I2S microphone
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = SAMPLE_BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);

  // Init ESP-DSP FFT
  dsps_fft2r_init_fc32(NULL, SAMPLE_BUFFER_SIZE);

  // Init LEDs
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(50);
}

void loop() {
  // Read audio from I2S
  size_t bytesRead;
  i2s_read(I2S_NUM_0, sampleBuffer, sizeof(sampleBuffer), &bytesRead, portMAX_DELAY);

  // Convert int16 to float and apply window
  for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
    fftInput[i] = (float)sampleBuffer[i];
  }
  dsps_wind_hann_f32(fftInput, SAMPLE_BUFFER_SIZE);

  // Compute FFT
  dsps_fft2r_fc32(fftInput, SAMPLE_BUFFER_SIZE);
  dsps_bit_rev_fc32(fftInput, SAMPLE_BUFFER_SIZE);
  dsps_cplx2reC_fc32(fftInput, SAMPLE_BUFFER_SIZE);

  // Extract bass energy (bins 1-10 for 20-200Hz at 44.1kHz)
  float bassEnergy = 0;
  for (int i = 1; i < 10; i++) {
    bassEnergy += fftInput[i];
  }

  // Map to LEDs (simple example: flash on bass)
  uint8_t brightness = constrain(bassEnergy / 100, 0, 255);
  fill_solid(leds, NUM_LEDS, CHSV(160, 255, brightness)); // Purple flash

  FastLED.show();
}
```

**This is 80 lines and already has:**
- Real-time I2S audio input
- FFT processing
- Bass extraction
- LED control

Extend with spectral flux, beat detection, etc.

---

## 9. Performance Estimates

### 9.1 FFT Speed

Using **ESP-DSP Float32 FFT on ESP32 @ 240MHz:**

| FFT Size | Time    | CPU % @ 44.1kHz (chunks every 23ms) |
|----------|---------|-------------------------------------|
| 512      | 1.5ms   | 6.5%                                |
| 1024     | 3.2ms   | 14%                                 |
| 2048     | 7.0ms   | 30%                                 |
| 4096     | 15ms    | 65%                                 |

**Recommendation:** Use **1024-point FFT** (14% CPU, 23ms chunks).

This leaves **86% CPU** for:
- LED effects rendering
- Beat detection logic
- WiFi (if needed)
- Web UI

### 9.2 Feature Extraction Speed

Estimates for common features (after FFT):

| Feature                  | Time per frame |
|--------------------------|----------------|
| Frequency band energy    | 0.05ms         |
| Spectral centroid        | 0.1ms          |
| Spectral flux            | 0.1ms          |
| RMS energy               | 0.02ms         |
| Zero-crossing rate       | 0.03ms         |
| Beat detection (spectral flux) | 0.15ms   |

**All features combined: <1ms**

### 9.3 Total Latency Budget

| Stage                | Time    |
|----------------------|---------|
| I2S DMA buffer fill  | 23ms    |
| FFT                  | 3.2ms   |
| Feature extraction   | 0.5ms   |
| Beat detection       | 0.15ms  |
| LED mapping          | 0.2ms   |
| LED output (300 LEDs)| 0.9ms   |
| **Total**            | **28ms**|

**Latency: 28ms** (well under 50ms perceptual threshold)

### 9.4 Frame Rate

- Audio frames: **~38 FPS** (23ms chunks + 5ms processing)
- LED updates: Can run at **60 FPS** with interpolation/smoothing
- Or lock to audio frames: **38 FPS** (still smooth for human eye)

---

## 10. Algorithms from Our Research That Transfer to ESP32

### ✅ WORKS GREAT

1. **Spectral Flux Beat Detection**
   - Our core algorithm
   - 1024-point FFT → bass bins → flux → adaptive threshold
   - **3.5ms per frame** on ESP32
   - This is **better than most existing ESP32 projects**

2. **Frequency Band Extraction**
   - Bass (20-200 Hz), Mid (200-2000 Hz), Treble (2000-8000 Hz)
   - Simple bin summing after FFT
   - Used for color mapping (bass=red, mid=green, treble=blue)

3. **Attack/Decay Envelopes**
   - Asymmetric EMA smoothing
   - α_attack = 0.7-0.99, α_decay = 0.1-0.3
   - Makes LED effects feel "musical" (sharp attack, slow fade)

4. **RMS Normalization with AGC**
   - Rolling window of RMS values
   - Adaptive gain to keep dynamic range
   - Prevents quiet songs from being invisible

5. **Onset Strength (simplified)**
   - Half-wave rectified spectral difference
   - Less sophisticated than aubio, but 95% as good for beat detection

### ⚠️ POSSIBLE WITH MODIFICATIONS

6. **Mel Filterbank**
   - Precompute triangular filters for 24 mel bins
   - Store in flash (~2KB)
   - Apply after FFT (matrix multiply)
   - Adds ~1ms per frame
   - **Worth it** if you want perceptually-weighted frequency analysis

7. **Spectral Centroid & Flatness**
   - Centroid: Weighted average of bin frequencies (brightness)
   - Flatness: Geometric mean / arithmetic mean (noisiness)
   - Each adds ~0.1ms
   - **Worth it** for mood-based color shifts

8. **Chroma Features (12-bin pitch)**
   - Map FFT bins to 12 musical pitches (C, C#, D, ...)
   - Useful for music-theory-aware effects (colors per key)
   - Adds ~0.5ms per frame
   - **Niche use case** but doable

### ❌ WON'T WORK (Memory/CPU Limits)

9. **Tempo Tracking (aubio's beat_track)**
   - Needs 6-10 seconds of audio history (1MB buffer)
   - ESP32 has 520KB SRAM
   - **Not feasible**

10. **HPSS (Harmonic-Percussive Separation)**
    - Requires multiple FFT passes + median filtering
    - Too CPU-intensive for real-time
    - **Skip it**

11. **Dynamic Time Warping**
    - Aligns beats across tempo changes
    - Needs large history buffers
    - **Not feasible**

12. **Genre/Sentiment Classification**
    - Requires large ML models (>500KB)
    - Or hand-crafted features over long time windows
    - **Not feasible in real-time**

### 💡 NEW POSSIBILITIES ON ESP32

13. **Multi-Band Beat Detection**
    - Run 3 spectral flux detectors simultaneously:
      - Bass beats (kick drum)
      - Mid beats (snare)
      - High beats (hi-hat)
    - Map to different LED zones or colors
    - Only **10.5ms total** (3.5ms × 3 detectors)
    - **Totally feasible** and would look amazing

14. **Directional Sound (Stereo Mics)**
    - Use 2 I2S mics (left/right)
    - Detect which direction bass is coming from
    - Map to LED strip position
    - **Unique to ESP32** (PC doesn't know LED positions)

---

## 11. Comparison Table

| Aspect                  | PC Streaming (librosa)      | ESP32 Standalone (ESP-DSP)  |
|-------------------------|-----------------------------|------------------------------|
| **Portability**         | Not portable (needs PC)     | Fully standalone, battery-powered |
| **Cost**                | ~$500 (PC) + $10 (LEDs)     | $20 (ESP32 + mic + LEDs)     |
| **Setup Complexity**    | Medium (USB cable, scripts) | Low (plug and play)          |
| **Audio Latency**       | 30-50ms (USB + processing)  | 20-30ms (all on-chip)        |
| **Beat Detection**      | aubio tempo tracking (advanced) | Spectral flux (good)     |
| **Feature Count**       | 20+ features                | 5-10 features                |
| **Tempo Tracking**      | Yes (aubio)                 | No (memory limit)            |
| **Mood/Genre Detection**| Possible (librosa + ML)     | Not practical                |
| **Development Speed**   | Fast (Python hot-reload)    | Slow (compile + upload)      |
| **Effect Flexibility**  | Unlimited (Python)          | Limited (firmware changes)   |
| **Power Consumption**   | 50-200W (PC)                | 1-3W (ESP32 + LEDs)          |
| **WiFi/Control UI**     | Easy (web server)           | Possible (ESPAsyncWebServer) |
| **Best For**            | Studio, experimentation, complex effects | Installations, portability, simplicity |

---

## 12. Recommendations

### 12.1 When to Use ESP32 Standalone

✅ **Choose ESP32 if:**
- Portability matters (no PC available)
- Installation is permanent (once configured, runs forever)
- Budget is tight (<$50 total)
- You want simple, reliable beat-reactive effects
- Power consumption matters (battery or solar)
- You're okay with **reactive** effects (not interpretive/mood-based)

### 12.2 When to Keep PC Streaming

✅ **Choose PC streaming if:**
- You're still experimenting with algorithms
- You want **tempo tracking** or complex analysis
- You want **mood-based** effects (the "feeling layer")
- Development speed matters (rapid iteration)
- You have access to a PC/laptop nearby
- You want to integrate with other software (DAWs, streaming)

### 12.3 Hybrid Approach

💡 **Best of both worlds:**

**Phase 1: Develop on PC**
- Use current librosa/Python setup
- Experiment with algorithms rapidly
- Fine-tune spectral flux beat detection
- Build effect library

**Phase 2: Port to ESP32**
- Implement proven algorithms in C++/ESP-DSP
- Use ESP32 for installations
- Keep PC version for studio/development

**Phase 3: Over-The-Air Updates**
- ESP32 can update firmware via WiFi
- Push new effects without physical access
- Best of portability + flexibility

---

## 13. Conclusion

**ESP32 standalone audio-reactive LED control is absolutely feasible and practical.**

**What you GET:**
- ✅ Real-time FFT (1024-2048 points)
- ✅ Spectral flux beat detection (our algorithm works great)
- ✅ Frequency band analysis (bass/mid/treble)
- ✅ Low latency (20-30ms)
- ✅ High frame rate (30-60 FPS)
- ✅ Fully portable
- ✅ Cheap ($20)

**What you LOSE:**
- ❌ Tempo tracking
- ❌ HPSS / advanced analysis
- ❌ Mood/genre classification
- ❌ Rapid experimentation
- ❌ Multi-second analysis windows

**Recommended path:**
1. **Start with PC streaming** for R&D (your current approach)
2. **Prototype ESP32 standalone** with our spectral flux algorithm
3. **Choose per-installation** based on needs

For your project (taste/art exploration), I'd recommend **staying with PC streaming during development**, then **porting final effects to ESP32 for installations**.

The ESP32 can absolutely run the core algorithms from our research. It just can't match the sophistication of PC-based analysis — but that sophistication might not matter for LED effects. A well-tuned spectral flux beat detector looks incredible, even without tempo tracking.

---

## Appendix A: Further Reading

**ESP-DSP Documentation:**
- https://docs.espressif.com/projects/esp-dsp/

**I2S Microphone Tutorials:**
- Espressif I2S examples: `esp-idf/examples/peripherals/i2s/`
- Adafruit I2S MEMS Mic Guide: https://learn.adafruit.com/adafruit-i2s-mems-microphone-breakout

**WLED Sound Reactive:**
- GitHub: Search "WLED sound reactive" (multiple forks)
- Best starting point for ESP32 audio-reactive projects

**FastLED Library:**
- https://github.com/FastLED/FastLED
- Best LED library for effects

**TensorFlow Lite Micro:**
- ESP32 examples: https://github.com/tensorflow/tflite-micro
- Micro Speech example (audio ML)

---

**Next Steps:**
- Order an ESP32 + ICS-43434 mic (~$20)
- Flash WLED Sound Reactive to see existing capabilities
- Prototype spectral flux algorithm in Arduino + ESP-DSP
- Compare to PC streaming version

Let me know if you want code examples for any specific algorithm on ESP32.
