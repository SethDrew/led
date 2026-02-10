# Audio-Reactive LED Systems - Research Report

## Executive Summary

This report evaluates existing audio-reactive LED projects and Python audio analysis libraries to inform whether to extend an existing solution or build custom using your established streaming architecture (`nebula_stream.py`).

**Key Finding**: Build custom using your streaming architecture + a focused audio analysis library (librosa or aubio). Your existing Python→Serial→Arduino pipeline is cleaner and more extensible than adapting existing projects.

---

## Part 1: Audio-Reactive LED Projects

### Comparison Table

| Project | Stars | Activity | Use Case | Architecture | Protocol | Extensibility | Verdict |
|---------|-------|----------|----------|--------------|----------|---------------|---------|
| **scottlawsonbc/audio-reactive-led-strip** | ~2.8k | Last major update 2020, minimal recent activity | Raspberry Pi + LED strip, spectrum visualization | NumPy FFT, mel-scale filtering, real-time audio capture (pyaudio) | Direct GPIO (RPi.GPIO) or serial | Medium - monolithic design, requires understanding full system | Already tried, not happy |
| **LedFx** | ~1.4k | Active (2023-2024) | Network-based controller for WLED devices, festival/club lighting | Plugin architecture, FFT-based effects, WebSocket GUI | Network (E1.31/Art-Net/UDP) to WLED devices | High - plugin system, but complex to learn | Overkill for your use case |
| **WLED (Audio Reactive)** | ~14k | Very active (2024+) | ESP32/ESP8266 standalone, ambient/accent lighting | On-device FFT (ESP32), I2S microphone or UDP sync | Direct GPIO (FastLED library) | Low - C++ on-device, limited by MCU constraints | Wrong architecture (on-device) |
| **Hyperion.ng** | ~3k | Active (2023-2024) | Ambilight/TV backlighting with audio support | Multi-source input, FFT effects, LED mapping | Network (various protocols) | Medium - complex setup, GUI-focused | Not audio-first, heavyweight |
| **FastLED + Spectrum Shield** | Varies | Community examples, no single repo | Arduino standalone, simple visualizers | Hardware FFT (MSGEQ7 chip) | Direct library (FastLED) | Medium - Arduino constraints, limited processing power | You already have better Python streaming |

### Detailed Analysis

#### 1. scottlawsonbc/audio-reactive-led-strip

**What it is**: Python script that captures audio, performs FFT analysis, and sends visualization data to LED strips. Originally designed for Raspberry Pi.

**Strengths**:
- Well-documented FFT → mel-scale frequency mapping
- Multiple visualization modes (scroll, spectrum, energy)
- Gaussian filtering for smooth transitions
- Proven to work with LED strips

**Weaknesses**:
- Monolithic design - tight coupling between audio capture, processing, and output
- Raspberry Pi GPIO-centric (not trivial to adapt to serial)
- No genre awareness or advanced beat detection
- Effect logic mixed with protocol logic
- Dated dependencies (pyaudio can be problematic)
- Limited customization without understanding entire codebase

**Your Experience**: "Wasn't happy out of the box" - likely because:
- Effects are pre-baked, not easily tunable
- No separation of concerns for custom effect development
- Spectrum visualization is cool but not ambient/artistic

**Code Quality**: 6/10 - Functional but not architected for extension

**Recommendation**: Mine for algorithms (mel-scale, filtering) but don't fork.

---

#### 2. LedFx

**What it is**: Network-based LED controller with web GUI. Acts as middleware between audio source and WLED/network-connected LED devices.

**Strengths**:
- Professional plugin architecture - effects are isolated modules
- Rich effect library (spectral, beat-reactive, color patterns)
- Real-time audio analysis with multiple backends
- Active community, good documentation
- WebSocket GUI for live tuning

**Weaknesses**:
- Complex setup - requires WLED devices or compatible hardware
- Network-based adds latency (not ideal for tight audio sync)
- Heavy dependency footprint (web server, multiple audio backends)
- Learning curve for plugin development
- Over-engineered for simple use cases

**Use Case Match**: 4/10 - Built for multi-device festival lighting, not single-strip prototyping

**Extensibility**: High, but requires learning their plugin API and architecture

**Recommendation**: Too complex for your needs. You'd spend more time adapting LedFx than building custom.

---

#### 3. WLED (Audio Reactive Fork)

**What it is**: ESP32/ESP8266 firmware with on-device audio processing. Microphone input or UDP audio sync.

**Strengths**:
- Standalone - no computer required after setup
- Active development, huge community
- Many built-in effects and palettes
- Web GUI for configuration
- Low latency (on-device processing)

**Weaknesses**:
- ESP32 FFT limitations (lower resolution than Python)
- Effect development in C++ (less flexible than Python)
- On-device processing = limited algorithmic complexity
- No genre awareness or ML capabilities
- Wrong architecture for your streaming setup

**Use Case Match**: 2/10 - You already have superior Python streaming architecture

**Recommendation**: Not relevant. You have more compute power in Python than ESP32 can provide.

---

#### 4. Hyperion.ng

**What it is**: Multi-purpose LED controller, primarily for TV ambilight with audio reactivity as secondary feature.

**Strengths**:
- Mature project, cross-platform
- Multiple input sources (screen capture, audio, API)
- Network-based LED control
- Good performance

**Weaknesses**:
- Audio reactivity is not the focus
- Complex configuration (JSON, network setup)
- Heavyweight for single-purpose audio-reactive use
- GUI-dependent workflow

**Use Case Match**: 3/10 - Overkill, audio is afterthought

**Recommendation**: Pass. Not audio-first.

---

#### 5. FastLED + Hardware Spectrum Analyzer (MSGEQ7)

**What it is**: Arduino-based approach using dedicated FFT chip (MSGEQ7) for frequency analysis.

**Strengths**:
- Hardware FFT is fast and reliable
- Standalone (no computer required)
- Low latency
- Simple to implement

**Weaknesses**:
- Requires additional hardware (MSGEQ7 chip, microphone circuit)
- Only 7 frequency bins (very limited)
- Arduino processing constraints
- No beat detection, genre awareness, or advanced features
- You already have better: Python streaming with full NumPy/SciPy

**Use Case Match**: 1/10 - You've already surpassed this with nebula_stream.py

**Recommendation**: Step backward from your current architecture.

---

### Key Takeaways from LED Projects

1. **Your streaming architecture is superior**: Python compute → serial → dumb Arduino receiver is cleaner and more powerful than any reviewed project.

2. **Existing projects are:**
   - Too monolithic (scottlawsonbc)
   - Too complex (LedFx, Hyperion)
   - Wrong architecture (WLED, FastLED)
   - Not extensible for your vision (genre modes, sentiment)

3. **Useful learnings to steal**:
   - Mel-scale frequency mapping (scottlawsonbc)
   - Gaussian filtering for smooth transitions (scottlawsonbc)
   - Effect separation patterns (LedFx plugin architecture concept)
   - Color palette management (WLED)

4. **What's missing from all of them**:
   - Genre-aware processing
   - Beat detection beyond simple onset detection
   - Sentiment/mood analysis
   - Flexible streaming architecture

---

## Part 2: Python Audio Analysis Libraries

### Comparison Table

| Library | Best For | Real-Time? | Ease of Use | Dependencies | Maintenance | Verdict |
|---------|----------|------------|-------------|--------------|-------------|---------|
| **librosa** | Music analysis, feature extraction, MIR | Partial - with streaming | Medium | NumPy, SciPy, soundfile, resampy | Active (2024) | Top choice for comprehensive features |
| **aubio** | Beat/pitch detection, onset detection | Yes | Easy | NumPy, minimal | Active (2024) | Best for real-time beat detection |
| **madmom** | Advanced beat tracking, tempo estimation | Partial | Hard | NumPy, SciPy, Cython | Less active (2023) | Overkill, complex |
| **sounddevice** | Audio I/O, live capture | Yes | Very easy | PortAudio | Active (2024) | Essential for audio input |
| **pyaudio** | Audio I/O (older) | Yes | Medium | PortAudio | Maintenance mode | Avoid - sounddevice is better |
| **scipy.signal** | FFT, filtering, signal processing | Yes | Easy (if you know DSP) | NumPy/SciPy | Active (2024) | Great for custom FFT |
| **essentia** | Advanced MIR, ML models | Partial | Hard | Many (C++ binding) | Active (2024) | Powerful but heavyweight |

### Detailed Analysis

#### 1. librosa

**What it is**: Comprehensive music and audio analysis library. Gold standard for Music Information Retrieval (MIR).

**Strengths**:
- Beat tracking: `librosa.beat.beat_track()` with tempo estimation
- Spectral features: mel spectrograms, chroma, MFCC, spectral centroid
- Onset detection: `librosa.onset.onset_detect()`
- Time-stretching, pitch shifting
- Genre classification features (can train ML models)
- Excellent documentation and examples

**Real-Time Capability**:
- Designed for offline analysis, but works in real-time with streaming approach:
```python
# Streaming pattern
def audio_callback(audio_chunk):
    # Real-time onset detection
    onsets = librosa.onset.onset_strength(y=audio_chunk, sr=sample_rate)
    # Real-time beat tracking (with memory)
    tempo, beats = librosa.beat.beat_track(y=audio_chunk, sr=sample_rate)
```
- Use with `sounddevice` for audio capture

**Ease of Use**:
- Medium - good docs, but need to understand MIR concepts
- Great for experimentation (notebook-friendly)
- High-level functions hide complexity

**Dependencies**:
- NumPy, SciPy, scikit-learn, numba (JIT compilation for speed)
- Install: `pip install librosa` (straightforward)

**Use Cases**:
- Beat tracking with tempo estimation
- Frequency analysis (mel spectrograms for genre fingerprinting)
- Chroma features (key/harmony detection for sentiment)
- Spectral features (brightness, flatness for mood)

**Performance**:
- Fast (numba JIT optimization)
- Can handle real-time at 60 FPS with proper chunking

**Best For Your Project**:
- Beat detection with tempo awareness
- Genre classification features
- Future sentiment/mood analysis (chroma, harmony)
- Comprehensive toolbox approach

**Recommendation**: **First choice** for comprehensive system.

---

#### 2. aubio

**What it is**: Lightweight audio analysis library focused on real-time performance. Built for music applications.

**Strengths**:
- Real-time beat detection: `aubio.tempo()`
- Onset detection: `aubio.onset()`
- Pitch detection: `aubio.pitch()`
- Minimal dependencies (NumPy only)
- Low latency, optimized C core
- Simple API

**Real-Time Capability**:
- Designed for real-time - buffers audio chunks efficiently
```python
# Real-time beat detection
tempo = aubio.tempo('default', 1024, 512, sample_rate)
while True:
    audio_chunk = get_audio()
    is_beat = tempo(audio_chunk)
    if is_beat:
        trigger_led_pulse()
```

**Ease of Use**:
- Very easy - minimal API surface
- Less documentation than librosa, but examples are clear

**Dependencies**:
- NumPy only (minimal)
- Install: `pip install aubio` (easy)

**Use Cases**:
- Simple beat detection for LED pulses
- Onset detection for percussive hits
- Pitch tracking (less relevant for your use case)

**Performance**:
- Excellent - C implementation, minimal overhead
- Lower latency than librosa

**Best For Your Project**:
- If you only need beat detection, aubio is perfect
- Simpler than librosa, less to learn
- Fast prototyping

**Recommendation**: **Best for beat-only prototype**, but less flexible for future features (genre, sentiment).

---

#### 3. madmom

**What it is**: Advanced beat tracking and music information retrieval. Research-grade algorithms (RNNs for beat tracking).

**Strengths**:
- State-of-the-art beat tracking (better than librosa/aubio for complex music)
- Handles polyrhythms, time signature changes (great for prog rock!)
- Downbeat detection (measure boundaries)
- Chord recognition

**Weaknesses**:
- Complex API, steep learning curve
- Heavier dependencies (Cython, models)
- Slower than aubio (ML models)
- Less maintained (last update 2023)

**Real-Time Capability**:
- Partial - models need context (1-2 seconds)
- Higher latency than aubio

**Use Cases**:
- Complex music analysis (Tool, prog rock)
- When simple beat detection fails
- Research-grade accuracy

**Recommendation**: **Overkill for MVP**. Consider later if aubio/librosa beat detection isn't good enough for prog rock.

---

#### 4. sounddevice

**What it is**: Modern Python audio I/O library. Replaces pyaudio.

**Strengths**:
- Simple callback-based API
- Cross-platform (PortAudio backend)
- Low latency
- Clean, Pythonic
- Active maintenance

**Example**:
```python
import sounddevice as sd

def audio_callback(indata, frames, time, status):
    audio_chunk = indata[:, 0]  # Mono
    # Process audio -> update LED state

sd.InputStream(callback=audio_callback, channels=1, samplerate=44100)
```

**Recommendation**: **Essential** for audio capture. Use with librosa/aubio for analysis.

---

#### 5. pyaudio (deprecated)

**What it is**: Older Python audio I/O wrapper.

**Why not**:
- Installation issues (compiler required)
- Less Pythonic API
- sounddevice is superior in every way

**Recommendation**: **Avoid**. Use sounddevice instead.

---

#### 6. scipy.signal

**What it is**: SciPy's signal processing module. Low-level DSP tools.

**Strengths**:
- FFT: `scipy.fft.fft()` (fast, flexible)
- Filtering: `scipy.signal.butter()`, `scipy.signal.filtfilt()`
- Spectrogram: `scipy.signal.spectrogram()`
- No additional dependencies (part of SciPy)

**Real-Time Capability**:
- Yes - manual implementation required

**Use Cases**:
- Custom FFT implementations
- Band-pass filtering (isolate bass, mids, highs)
- Low-level control

**Best For Your Project**:
- If you want full control over audio processing
- Learning DSP fundamentals
- scottlawsonbc uses scipy.signal for mel-scale filtering

**Recommendation**: **Complement to librosa/aubio**, not replacement. Use for custom filtering or if you want to understand audio processing deeply.

---

#### 7. essentia

**What it is**: Advanced MIR library with ML models. From Music Technology Group (Barcelona).

**Strengths**:
- Pre-trained models for genre classification, mood detection
- Comprehensive feature extraction
- Research-grade algorithms

**Weaknesses**:
- Complex installation (C++ binding)
- Heavy dependencies
- Steep learning curve
- Overkill for prototyping

**Recommendation**: **Consider later** for sentiment/mood analysis. Start with librosa first.

---

### Key Takeaways from Audio Libraries

1. **Recommended Stack**:
   - **Audio input**: sounddevice (callback-based capture)
   - **Beat detection**: aubio (fast, simple) OR librosa (more features)
   - **FFT/Spectrum**: librosa or scipy.signal
   - **Future (genre/sentiment)**: librosa features + scikit-learn

2. **Real-Time Pattern**:
```python
import sounddevice as sd
import librosa
import numpy as np

class AudioAnalyzer:
    def __init__(self):
        self.latest_features = {}

    def audio_callback(self, indata, frames, time, status):
        audio = indata[:, 0]

        # Update features (decoupled from LED sending)
        self.latest_features['onset'] = librosa.onset.onset_strength(y=audio)
        self.latest_features['spectrum'] = np.abs(librosa.stft(audio))

    def get_features(self):
        return self.latest_features

# Main loop (fixed rate, like nebula_stream.py)
analyzer = AudioAnalyzer()
with sd.InputStream(callback=analyzer.audio_callback):
    while True:
        features = analyzer.get_features()
        frame = generate_led_frame(features)
        streamer.send_frame(frame)
        time.sleep(1/60)  # Fixed 60 FPS
```

3. **Why librosa over aubio**:
   - More features for future expansion (genre, sentiment)
   - Better documentation
   - More flexible (can add ML models later)
   - Only slightly slower than aubio (negligible at 60 FPS)

4. **Why aubio over librosa**:
   - Simpler API, faster to prototype
   - If you only need beat detection, it's perfect
   - Lower latency

---

## Part 3: Architecture Recommendation

### Option A: Fork Existing Project (NOT RECOMMENDED)

**Candidates**: scottlawsonbc, LedFx

**Pros**:
- Proven audio analysis code
- Working examples

**Cons**:
- Your streaming architecture is better
- Major refactoring required to adapt
- Tight coupling makes customization hard
- Would spend more time understanding their system than building yours

**Verdict**: Don't fork. Your nebula_stream.py is cleaner.

---

### Option B: Build Custom on Your Streaming Architecture (RECOMMENDED)

**Approach**:
1. Extend nebula_stream.py architecture
2. Add audio analysis layer
3. Create new effect classes that respond to audio features

**Why This Wins**:
- You already have solid streaming foundation
- Complete control over effect logic
- Separation of concerns (audio → features → effects → frames → serial)
- Easy to add genre modes, sentiment analysis
- No legacy baggage

**Architecture**:
```
Audio Input (sounddevice)
    ↓
Audio Analysis (librosa/aubio)
    ↓
Feature Extraction (beat, spectrum, tempo)
    ↓
AudioReactiveEffect (like NebulaEffect)
    ↓
LED Frame (numpy array)
    ↓
LEDStreamer (existing)
    ↓
Serial → Arduino (existing)
```

**New Components to Build**:
1. `AudioAnalyzer` class - wraps librosa/aubio, outputs features
2. `AudioReactiveEffect` base class - like NebulaEffect but consumes audio features
3. Specific effects:
   - `BeatPulseEffect` - brightness responds to beats
   - `SpectrumEffect` - colors map to frequency bands
   - `GenreModeEffect` - different patterns for dance vs prog rock

**Reuse from nebula_stream.py**:
- `LEDStreamer` class (unchanged)
- Frame timing logic (fixed 60 FPS, decoupled from audio)
- Serial protocol (unchanged)

---

### Complexity Comparison

| Approach | Initial Effort | Customization | Long-Term Maintenance |
|----------|----------------|---------------|----------------------|
| Fork scottlawsonbc | Medium | Hard | Hard (legacy code) |
| Adapt LedFx | High | Medium | Medium (their API) |
| Build custom | Low-Medium | Easy | Easy (your code) |

**Build custom wins**: Less upfront effort, easier customization, cleaner long-term.

---

## Part 4: Recommended Implementation Plan

### Phase 1: Beat Detection Prototype (Simplest First)

**Goal**: Prove architecture with simplest audio-reactive feature.

**Library**: aubio (simpler) or librosa (more features)

**Effect**: Beat-triggered brightness pulses over nebula background.

**Code Structure**:
```python
# audio_reactive_stream.py
class AudioAnalyzer:
    # Wraps aubio.tempo() or librosa.beat.beat_track()
    # Callback stores latest beat info

class BeatPulseEffect:
    # Inherits timing patterns from NebulaEffect
    # Adds brightness pulse on beat detection

# Main loop (like nebula_stream.py)
analyzer = AudioAnalyzer()
effect = BeatPulseEffect(num_leds=150, analyzer=analyzer)
streamer = LEDStreamer(port, num_leds)

while True:
    frame = effect.update(dt)
    streamer.send_frame(frame)
    time.sleep(1/60)
```

**Deliverable**: `audio_reactive_stream.py` with beat detection.

**Estimated Effort**: 4-6 hours (including testing, tuning threshold).

---

### Phase 2: Frequency Spectrum Effects

**Goal**: Color zones respond to frequency bands (bass/mid/high).

**Library**: librosa (mel spectrogram)

**Effect**:
- Bass frequencies → red/purple (bottom LEDs)
- Mids → blue (middle LEDs)
- Highs → white (top LEDs)

**Estimated Effort**: 3-4 hours (after Phase 1 architecture is solid).

---

### Phase 3: Genre Modes

**Goal**: Different response patterns for different music.

**Approach**:
- Analyze tempo, spectral features
- Heuristics or simple classifier:
  - Dance: 120-130 BPM, strong bass → tight sync, saturated colors
  - Prog/Rock: variable tempo, complex → longer trails, layered effects
  - Ambient: slow tempo, spectral smoothness → slow fades, muted colors

**Estimated Effort**: 6-8 hours (experimentation required).

---

### Phase 4: Sentiment/Mood Analysis (Advanced)

**Goal**: Colors respond to emotional content.

**Approach**:
- Chroma features (key detection): major = warm, minor = cool
- Spectral brightness/flatness: bright = energetic, flat = calm
- ML model (optional): essentia pre-trained mood classifier

**Estimated Effort**: 8-12 hours (research + implementation).

---

## Part 5: Final Recommendations

### Audio Analysis Library Choice

**MVP (Beat Detection)**:
- **First choice**: aubio (simplest, fastest to prototype)
- **Second choice**: librosa (more features, easier to expand later)

**Full System (All Features)**:
- **librosa** + scipy.signal + (later) scikit-learn
- More code upfront, but better long-term

**Recommendation**: Start with **librosa** for beat detection. Slightly more complex than aubio, but you won't have to refactor when adding spectrum/genre/sentiment.

---

### LED Project Approach

**Do NOT fork**:
- scottlawsonbc (monolithic, wrong architecture)
- LedFx (overkill, complex)
- WLED (wrong platform)

**DO build custom**:
- Extend nebula_stream.py
- Add AudioAnalyzer + AudioReactiveEffect classes
- Steal algorithms (mel-scale, filtering) but not architecture

---

### Why Your Streaming Architecture Wins

1. **Separation of concerns**:
   - Audio analysis in Python (full NumPy/SciPy power)
   - Effect logic in Python (easy to iterate)
   - Arduino is dumb receiver (no complexity)

2. **Performance**:
   - Python compute >> ESP32/Arduino
   - 60 FPS streaming proven with nebula
   - Real-time audio analysis is well within Python's capability

3. **Extensibility**:
   - Adding genre modes = new effect class
   - Adding sentiment = new feature extraction, same architecture
   - No hardware changes required

4. **Your existing knowledge**:
   - You already understand nebula_stream.py
   - Adding audio is incremental, not rewrite

---

## Appendix A: Quick Start Guide

### Minimal Audio-Reactive Setup

1. **Install dependencies**:
```bash
pip install librosa sounddevice numpy scipy
```

2. **Audio routing** (already done):
- BlackHole 2ch for system audio capture

3. **Code pattern**:
```python
import sounddevice as sd
import librosa
import numpy as np
from nebula_stream import LEDStreamer  # Reuse!

# Audio analysis
def audio_callback(indata, frames, time, status):
    global latest_onset
    audio = indata[:, 0]
    latest_onset = librosa.onset.onset_strength(y=audio, sr=44100)

# Main loop
streamer = LEDStreamer(port, num_leds)
with sd.InputStream(callback=audio_callback, samplerate=44100):
    while True:
        brightness = scale_onset_to_brightness(latest_onset)
        frame = generate_frame(brightness)
        streamer.send_frame(frame)
        time.sleep(1/60)
```

---

## Appendix B: Useful Code Snippets from scottlawsonbc

### Mel-Scale Frequency Mapping
```python
# Convert FFT bins to mel-scale (perceptually uniform)
def create_mel_filterbank(n_fft, sr, n_mels=24):
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return mel_basis

# Apply to FFT
fft = np.fft.rfft(audio_chunk)
magnitude = np.abs(fft)
mel_spectrum = np.dot(mel_basis, magnitude)
```

### Gaussian Smoothing
```python
from scipy.ndimage.filters import gaussian_filter1d

# Smooth spectrum for visual effect
mel_smooth = gaussian_filter1d(mel_spectrum, sigma=1.0)
```

---

## Appendix C: Testing Methodology Notes

### 10-Second Test Segments

**Capture approach**:
```python
import soundfile as sf

# Record segment
audio, sr = sd.rec(int(10 * 44100), samplerate=44100, channels=1)
sd.wait()
sf.write('test_segment_dance.wav', audio, sr)

# Playback for testing
audio, sr = sf.read('test_segment_dance.wav')
for chunk in generate_chunks(audio, chunk_size=1024):
    # Process as if live
    features = analyze(chunk)
    frame = generate_led_frame(features)
    send_frame(frame)
```

**Evaluation schema** (for user feedback):
```yaml
segment_id: dance_01
audio_file: test_segment_dance.wav
genre: house
tempo: 128
description: Steady 4/4 beat, strong bass

expected_behavior:
  - brightness_pulse_on_beat: yes
  - color_mode: blue_purple
  - intensity: high

actual_behavior:
  - brightness_pulse_on_beat: yes
  - color_mode: blue_purple
  - intensity: medium (too subtle)

rating: 3/5

notes: |
  Beat detection works, but pulse intensity should be stronger.
  Try increasing pulse_amplitude from 0.5 to 0.8.
```

---

## Conclusion

**Clear recommendation**: Build custom using librosa + your streaming architecture.

**Effort estimate**:
- Phase 1 (beat detection): 4-6 hours
- Phase 2 (spectrum): 3-4 hours
- Phase 3 (genre modes): 6-8 hours
- Phase 4 (sentiment): 8-12 hours

**Total**: 21-30 hours for full vision. MVP in 4-6 hours.

**Why this beats forking**:
- Less effort upfront
- Complete control
- Cleaner architecture
- Your existing streaming is superior
- Easy to extend

**Next step**: Implement Phase 1 (beat detection) to validate approach.
