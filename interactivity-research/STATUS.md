# Audio-Reactive LED Research — Status & Knowledge Map

*Last updated: 2026-02-07*

## Project Goal

Build audio-reactive LED effects that feel connected to music — not just volume meters, but effects that capture the *feeling* of what's happening musically. Target: standalone ESP32 installations.

---

## The Four Pillars

### Pillar 1: Taste Research (Our Unique Asset)

**What this is:** Human-annotated data + derived insights. The user's creative judgment about what musical moments matter and why. No library or dataset provides this.

**Status: ACTIVE — small but rich dataset, several general principles extracted**

#### User Annotations (Ground Truth)

| File | Audio | Layers | Key Data |
|------|-------|--------|----------|
| `opiate_intro.annotations.yaml` | Tool - Opiate, 40s, psych rock | beat (72 taps), changes (6), air (24), consistent-beat (42), beat-flourish (76), beat-flourish2 (65) | Multi-layer tap data from single track. 5 potential keyboard repeat artifacts at 29.3-29.6s (83-85ms intervals). |
| `fa_br_drop1.annotations.yaml` | Fred again.., 129s, EDM | sections (6 boundaries), energy_peaks (125+), tease_cycles (5), drop (1) | Section-level structure annotations |
| Golden copy at `audio-segments/golden/` | — | — | Preserved original user input |

#### General Principles Discovered (from annotations + analysis)

These are the research outputs that generalize beyond the specific songs:

1. **Derivatives > Absolutes** (`analysis/taste/build_vs_climax.md`)
   - Build vs climax has nearly identical static features (RMS ±0.5%, centroid ±4.8%)
   - The difference is RATE OF CHANGE: climax brightens 58x faster
   - *General rule: Track derivatives of features, not just features*

2. **Feelings Map to Deviation From Context** (`analysis/taste/air_feature_analysis.md`)
   - Two acoustically opposite moments (dark/sparse vs bright/dense) both feel "airy"
   - Common thread: both deviate from surrounding music's local norm
   - *General rule: Subjective feelings map to deviation-from-running-average, not absolute feature values*

3. **Flourish Ratio = Mode Selector** (`analysis/taste/beat_vs_consistent.md`)
   - Comparing free-form taps vs metronomic taps reveals "flourishes" (off-grid musical moments)
   - Flourish ratio inversely correlates with rhythmic clarity
   - *Three-mode framework: Ambient (>70% flourishes), Accent (30-70%), Groove (<30%)*

4. **Taps Track Bass Peaks, Not Onsets** (`analysis/taste/beat_tap_analysis.md`)
   - 100% of user taps within 50ms of bass peaks (19ms median)
   - Only 48.5% near librosa onsets
   - Users switch tracking modes across sections

5. **Flourishes Are Quieter Than On-Beat** (`analysis/taste/flourish_audio_properties.md`)
   - Percussive energy -24.7%, RMS -20.2%, onset density -13.7%
   - Flourishes are accents in the GAPS, not louder peaks
   - *Detection requires two-pass: find beat grid first, then score off-grid activity*
   - **Caveat:** Effect weakens dramatically for high-confidence flourishes (2+ source agreement, N=19): percussive energy effect size drops from -0.533 to -0.007. Headline finding may not be robust.

6. **Build Taxonomy** (`fa_br_drop1.annotations.yaml`, user input)
   - Primer/edging: cyclic energy ramps that reset (anticipation via repetition)
   - Sustained: consistent upward trajectory (anticipation via accumulation)
   - Bridge: chaos between build and drop
   - Drop: maximum intensity
   - *All independent — any can exist without the others*
   - **Counterintuitive finding:** Bridge has HIGHER RMS (0.3156) than drop (0.2638). Drops are about *sustained* intensity, not peak intensity. Can't detect drops by looking for max RMS.

#### User Taste Inputs (Creative Decisions to Preserve)

These are the user's subjective preferences that should guide design:

- "Airiness is a way to build excitement, or to transition you to another space"
- The first occurrence of a sound feels more airy than repeats — novelty matters
- Imperfections in tapping ARE the data — don't over-correct
- Beat detection should *feel* good on LEDs, not be metronomically accurate
- Maximalist drops (everything on full) vs progressive drops (subtle groove) are different genres needing different treatment
- The "edging/primer" build is distinct from sustained build — repeated disappointment creates different anticipation than steady climb
- "I was looking not to see if you could detect when my flourishes were, but if those flourishes lined up with elements in the song that you could extract into helpful properties"

#### Key Analysis Files

| File | What It Answers |
|------|----------------|
| `analysis/taste/beat_tap_analysis.md` | What audio features do user taps actually register on? |
| `analysis/taste/air_feature_analysis.md` | What makes moments feel "airy"? |
| `analysis/taste/build_vs_climax.md` | How to distinguish build from climax computationally? |
| `analysis/taste/beat_vs_consistent.md` | How to extract flourishes from tap data? |
| `analysis/taste/flourish_audio_properties.md` | What are the audio properties of flourish-worthy moments? |
| `analysis/taste/fa_br_drop1_analysis.md` | What does a maximalist EDM drop look like in feature space? |
| `analysis/taste/electronic_drop_summary.md` | How well does automated section detection match user description? |

---

### Pillar 2: Algorithm R&D (Prototypes — Unvalidated Theory)

**What this is:** Our custom detection algorithms. The hypothesis is that algorithms tuned to our taste research will produce better LED effects than off-the-shelf solutions. **This is currently unvalidated.**

**Status: TWO DETECTORS BUILT, initial validation done, needs head-to-head comparison with existing solutions**

#### What We've Built

| Algorithm | File | Approach | Strengths | Weaknesses |
|-----------|------|----------|-----------|------------|
| Bass Spectral Flux | `tools/realtime_beat_led.py` | FFT → bass bins → half-wave rectified diff → adaptive threshold | Simple, low latency, detects kicks | Misses electronic (continuous sub-bass), false positives in quiet sections |
| Full-Spectrum Onset | `tools/realtime_onset_led.py` | Mel spectrogram → spectral diff → adaptive threshold | Detects all transients, works on electronic | May over-trigger, doesn't distinguish kick from hi-hat |

#### Validation Results (Harmonix Dataset)

| Detector | Electronic F1 | Rock F1 | Notes |
|----------|--------------|---------|-------|
| Bass Spectral Flux | 0.06 | 0.27 | F1 measured against ALL beats — intentionally bass-only, so low recall expected |
| Full-Spectrum Onset | — | 0.252 | Measured against user taps (Opiate), NOT Harmonix — not directly comparable to bass flux F1 above |

**Important caveat:** Low F1 against "all beats" ground truth doesn't mean the algorithm is bad for LED purposes. Pulsing on every beat at 120+ BPM would strobe. Bass-only may be artistically better.

#### What We Haven't Done Yet (The Validation Gap)

- **No head-to-head comparison with WLED Sound Reactive** — the most popular existing solution
- **No user evaluation** — haven't A/B tested our algorithms vs off-the-shelf with music playing and asked "which looks better?"
- **No multi-song validation** — only tested on 3 songs + Harmonix subset
- **No electronic music tuning** — hyperparameters optimized primarily on rock

#### Proposed Architecture (Designed, Not Built)

| Concept | Status | File |
|---------|--------|------|
| Interpreter Registry (pluggable detectors) | Designed | Conversation notes |
| 5-Layer Real-Time Pipeline | Designed | `analysis/algorithms/REALTIME_CONSTRAINTS.md` |
| → Layer 1: Immediate (<50ms) | — | Bass peaks, onsets, RMS, centroid (per-chunk FFT) |
| → Layer 2: Short-context (50ms-2s) | — | Tempo, beat grid, airiness deviation |
| → Layer 3: Medium-context (2-10s) | — | Flourish ratio, build detection, climax (derivatives) |
| → Layer 4: Reactive events | — | Drops, flourishes (50ms feels instant to humans) |
| → Layer 5: Heuristic prediction | — | Song structure, needs heuristics not lookahead |
| Real-Time Constraint Classification | Done | `analysis/algorithms/REALTIME_CONSTRAINTS.md` |
| Harmonix Hyperparameter Optimization | Done (limited) | `analysis/harmonix/harmonix_optimization.md` |
| Build/Climax Detector | Code examples | `analysis/algorithms/REALTIME_CONSTRAINTS.md` |
| Drop Detector | Code examples | `analysis/algorithms/REALTIME_CONSTRAINTS.md` |
| Flourish Ratio Mode Selector | Designed | `analysis/taste/beat_vs_consistent.md` |

---

### Pillar 3: External Landscape (What Already Exists)

**What this is:** Survey of libraries, datasets, and existing projects. Context for knowing what we're competing against and what we can build on.

**Status: SURVEYED — good map of the landscape**

#### Libraries Evaluated

| Library | Real-Time? | Best For | Our Verdict |
|---------|-----------|----------|-------------|
| librosa | Onset detection: YES. Beat tracking: NO (needs full file) | General MIR, onset detection | Use for onset detection in Python |
| madmom | RNN frame-by-frame: YES. Beat tracking: NO | Tempo estimation (83.3 BPM on rock — correct!) | Use for tempo estimation only |
| essentia | Unknown | Comprehensive features | Failed to install (numpy 2.x incompatible) |
| aubio | YES (C library, designed for embedded) | Real-time onset/tempo/pitch | Not tested yet — promising for ESP32 |
| BeatNet | Unknown | Neural beat tracking | Not tested (needs PyTorch) |
| WLED Sound Reactive | YES (runs on ESP32) | Existing complete solution | **NOT EVALUATED — biggest gap in our research** |

**Key file:** `analysis/algorithms/library_comparison.md`, `analysis/algorithms/LIBRARY_RESEARCH_SUMMARY.md`

#### Datasets Available

| Dataset | Size | What | Relevance | Status |
|---------|------|------|-----------|--------|
| Harmonix Set | 912 tracks | Human beat annotations | HIGH — validation + hyperparameter training | Downloaded, explored, partial optimization done |
| AIST++ | 1,408 sequences | Dance motion + music | MEDIUM — energy proxy | Not downloaded |
| DEAM | 1,802 songs | Per-second arousal/valence | MEDIUM — emotion mapping | Not downloaded |
| MagnaTagATune | 25,863 clips | Crowd-sourced feeling tags | MEDIUM — feeling vocabulary | Not downloaded |
| SALAMI | 1,359 songs | Section boundaries | LOW-MEDIUM — structure detection | Not downloaded |

**Key file:** `research/EXISTING_DATASETS.md`

#### ESP32 Feasibility

- Spectral flux runs in 3.5ms on ESP32 (14% CPU) — viable for standalone
- ICS-43434 I2S mic + ESP32-WROVER-E = ~$20 hardware
- Lose: tempo tracking, HPSS, multi-second analysis windows
- Keep: beat detection, band analysis, attack/decay, tuned hyperparameters
- WLED Sound Reactive already does basic version of this

**Key file:** `research/ESP32_AUDIO_DSP.md`

---

### Pillar 4: Working Prototypes (Code That Runs)

**What this is:** Tools and firmware that actually work on hardware.

**Status: STREAMING PIPELINE WORKS — tree flashes to music**

#### Hardware Setup

| Component | Details |
|-----------|---------|
| LED Tree | 197 LEDs: Strip 1 (pin 13) = 92 LEDs (lower trunk + 2 branches), Strip 2 (pin 12) = 6 LEDs (side branch), Strip 3 (pin 11) = 99 LEDs (upper trunk + 2 branches). Max depth = 70. Effects use depth for spatial mapping via `TreeTopology.h`. |
| Nebula Strip | 150 LEDs, single strip, Arduino Nano |
| Audio Capture | BlackHole 2ch → Python. BlackHole is a macOS virtual audio driver that creates a loopback device. You create a "Multi-Output Device" in Audio MIDI Setup combining speakers + BlackHole, set it as system output, then Python reads from BlackHole's input side via sounddevice. Device number auto-detected by name. |
| Serial Protocol | `[0xFF][0xAA][RGB × N]` at 1Mbps. No error detection — corrupted frames silently dropped after 100ms timeout. Receiver reads in 30-byte chunks (ATmega328 has only 2KB RAM). Streaming receiver maps sequential node indices to physical strip positions via `TreeTopology.h`. |
| Frame Rate | 30 FPS (reduced from 60 due to serial buffer overflow on Arduino's 64-byte hardware buffer, not bandwidth limitation) |
| Brightness | Capped at 3% for testing (convenience, not design constraint) |

#### Latency Budget (End-to-End)

| Stage | Time | Notes |
|-------|------|-------|
| Audio capture | 0-10ms | Depends on buffer size |
| FFT computation | 10-20ms | 1024-2048 samples |
| Feature detection | 1-5ms | Per detector |
| LED command generation | 1-5ms | Color calculation |
| Serial transmission | 5-15ms | 593 bytes at 1Mbps |
| Arduino processing | 1-3ms | Parse + display |
| **Total** | **18-58ms** | Within 50ms target most of the time |

**Cold-start latencies:** Tempo estimation needs 2-4s to stabilize. Flourish ratio needs 5s. Build detection needs 5-10s. Airiness baseline needs 1-2s. First few seconds of any song will have reduced capability.

#### Detector Hyperparameters

| Parameter | Bass Flux | Onset |
|-----------|-----------|-------|
| Frequency range | 20-250 Hz | 20-8000 Hz (mel scale) |
| FFT/Mel bands | Raw FFT bins | 40 mel bands |
| History window | 3.0s | 3.0s |
| Threshold multiplier | 1.5 | 2.0 |
| Min beat interval | 0.3s | 0.1s |
| Attack alpha | 0.95 | 0.95 |
| Decay alpha | 0.12 | 0.12 |
| Gamma correction | 2.2 | 2.2 |

#### Working Tools

| Tool | What It Does |
|------|-------------|
| `tools/realtime_beat_led.py` | Bass flux → red pulse on tree (197 LEDs) |
| `tools/realtime_onset_led.py` | A/B test: bass (red) vs onset (blue) vs both (white) |
| `tools/record_segment.py` | Record audio segments from BlackHole |
| `tools/annotate_segment.py` | Tap-to-annotate: run with audio file + layer name, audio plays back, press key at relevant moments, timestamps saved to YAML. Multiple passes with different layer names build multi-layer annotations. |
| `tools/visualize_segment.py` | Static spectrogram + waveform visualization |
| `tools/playback_visualizer.py` | Real-time visualization during playback |
| `led-tree/physical/src/streaming_receiver.cpp` | Tree streaming firmware (compiled, tested) |
| `led-tree/physical/src/main.cpp` | Tree standalone effects (background/foreground system) |

#### Known Issues

- LED glitching from irregular serial timing (fix applied: decoupled audio callback from frame send)
- Bass detector has false positives in quiet sections
- No automatic genre detection / algorithm switching yet

---

## Facts, Opinions, and Assumptions

### Hard Facts (measured, reproducible)

| Fact | Evidence | Caveat |
|------|----------|--------|
| User taps land within 50ms of bass peaks (19ms median) | Computed on Opiate Intro, 72 taps | **N=1 song, N=1 person** |
| Build and climax have similar static features (RMS ±0.5%) | Computed on Opiate Intro, two 6s regions | **N=1 song, two cherry-picked regions** |
| Climax centroid rises 58x faster than build | Same as above | **Same caveat — one comparison** |
| librosa beat_track doubles tempo on rock (161.5 vs ~83 BPM) | Tested on Opiate Intro | Validated across Harmonix rock tracks too |
| madmom tempo estimation: 83.3 BPM on Opiate (correct) | Tested once | Not tested across Harmonix rock set |
| Bass spectral flux F1=0.06 on electronic, 0.27 on rock | Harmonix validation, 7 tracks | F1 measures all-beat accuracy, not LED quality |
| ESP32 FFT in 3.5ms at 240MHz | ESP-DSP documentation/benchmarks | Not tested on our hardware |
| Streaming pipeline works at 30 FPS, 197 LEDs | Tested on tree | Glitching observed, fix applied but not fully re-validated |

### Opinions Presented as Findings (need scrutiny)

1. **"Derivatives > Absolutes" is a general principle**
   - *What we actually showed:* On ONE 6-second pair of regions in ONE song, derivatives were more discriminating than absolutes.
   - *Assumption:* This generalizes to all build/climax pairs across genres.
   - *To validate:* Test on the Fred again.. build/drop. Test on Harmonix tracks with section annotations. If it holds on 10+ examples, it's a principle.

2. **"Airiness = deviation from context"**
   - *What we actually showed:* Two clusters in ONE song that the user called "airy" both deviated from the local average, despite being acoustically opposite.
   - *Assumption:* Deviation-from-context is THE mechanism for perceived airiness, not just a correlate.
   - *Alternative hypothesis:* Maybe airiness = novelty (first occurrence of a sound), and deviation-from-context is just a proxy for novelty. The user said "the first time something like this happens, ESPECIALLY if it's alone on the audio, the more airy it feels." That's novelty, not deviation.
   - *To validate:* Find moments that deviate from context but DON'T feel airy. If they exist, the principle is wrong or incomplete.

3. **"Flourish ratio = mode selector"**
   - *What we actually showed:* On Opiate Intro, the ratio of off-grid to on-grid taps correlates with section type (intro=90%, groove=17%).
   - *Assumption:* This ratio can be computed in real-time from audio alone (without user taps).
   - *Problem:* We've never actually computed flourish ratio from audio. We computed it from USER TAP DATA. Computing it from audio requires: (a) finding the beat grid algorithmically, (b) identifying off-grid events, (c) computing a ratio. Step (a) already fails on rock music.
   - *To validate:* Implement the audio-only flourish ratio detector and test it.

4. **"Flourishes are quieter than on-beat"**
   - *What we actually showed:* In Opiate Intro, the computed flourish moments had lower energy than on-beat moments.
   - *Assumption:* This is because flourishes ARE musically quieter accents in the gaps.
   - *Alternative:* Maybe our beat grid was wrong and the "flourishes" are just the moments the beat detector missed (which tend to be quieter because the detector is energy-biased).
   - *To validate:* Use Harmonix ground truth beats instead of our computed grid, then re-check if off-grid moments are still quieter.

5. **"Our custom approach will be better than off-the-shelf"**
   - *What we actually know:* We haven't tested any off-the-shelf solution (WLED Sound Reactive, LedFx, etc.)
   - *Assumption:* Because we understand the music better (through taste research), our effects will feel more connected.
   - *Counter-evidence:* WLED Sound Reactive has thousands of users, years of iteration, and community feedback. Our detector scored F1=0.06-0.27 on Harmonix. We might be building something worse while assuming it's better.
   - *To validate:* Install WLED Sound Reactive on our ESP32, play the same music, compare side-by-side.

### Hardware Constraints (Non-Negotiable)

1. **WS2812B LEDs require consistent frame timing.**
   - The WS2812B protocol bit-bangs data at precise intervals on the signal line. If frames arrive irregularly (as audio callbacks do), LED data corrupts — visible as random flashing/glitching.
   - **Consequence:** Audio analysis and frame sending MUST be decoupled. Audio callback stores results only → fixed-rate main loop reads state and sends frames using absolute time targets → serial flush after write. Every real-time tool follows this pattern.
   - **Learned the hard way:** Initial `realtime_beat_led.py` called `trigger_beat()` from the audio callback. Tree exhibited glitching. Fix: moved all LED state mutation to the main loop, added `threading.Lock` for shared state, used `next_frame_time += frame_interval` for pacing.
   - See: `led-tree/streaming-audio/README.md` lines 75-97.

### Implicit Assumptions We've Been Making

1. **"More sophisticated = better LED effects"**
   - We've been building increasingly complex analysis (derivatives, flourish ratios, multi-layer pipelines) without ever testing whether simple RMS-based brightness produces 80% of the visual impact. Maybe a $20 sound-reactive LED strip from Amazon looks 90% as good as our system.
   - **Test:** Record video of simple RMS brightness vs our beat detector vs WLED. Show to 5 people. See if anyone notices the difference.

2. **"One person's annotations generalize"**
   - All taste data comes from ONE person tapping on TWO songs. We call these "general principles" but they might be "things one person felt about two songs."
   - **Mitigation:** Get 2-3 more people to annotate the same tracks. If they agree, the principles strengthen. If they disagree, the principles are personal preferences (still useful for this installation, but not general).

3. **"Real-time constraints are absolute"**
   - We designed everything around "no lookahead." But many DJ sets and live shows have setlists. If the song is known, we could pre-analyze and have full structure awareness.
   - **Question:** Is the no-lookahead constraint real, or self-imposed? If installations could use known playlists, the entire architecture changes.

4. **"Bass detection is the foundation"**
   - We started with bass spectral flux and built everything on top. But the Harmonix results show it's the weakest approach. We might be optimizing the wrong starting point.
   - **Alternative:** Start from full-spectrum onset detection (better F1) and ADD bass isolation on top, not the reverse.

5. **"We need custom algorithms at all"**
   - The ESP32 research showed WLED Sound Reactive already does: FFT, beat detection, frequency band reactive effects, 10+ visualization modes, runs standalone.
   - **Question:** What specifically would our system do that WLED can't? If we can't answer this concretely, we're reinventing a wheel.

6. **"The feeling layer is unsolved"**
   - We've stated that "mapping features to feelings is not solved by any library." But we haven't proven this either. Spotify's audio features (energy, danceability, valence) are a form of feeling mapping. DEAM has per-second arousal annotations. MusicCaps has natural language descriptions. Maybe the feeling layer IS partially solved, and we should build on existing work rather than starting from scratch.

### What We'd Need to Prove the Theory

To move from "interesting R&D project" to "justified custom approach," we'd need:

1. **A concrete comparison** showing our effects look/feel better than WLED Sound Reactive on the same hardware
2. **Multi-song validation** of at least the "derivatives > absolutes" principle (5+ songs, 2+ genres)
3. **Multi-person validation** of at least one taste finding (do others agree about airiness?)
4. **A working multi-feature detector** (not just beat pulse — build detection, mode switching, etc.) running in real-time
5. **An audience test** — do people watching the LEDs notice/prefer the sophisticated approach?

---

## File Organization Guide

```
interactivity-research/
├── STATUS.md                          ← YOU ARE HERE
├── LEARNINGS.md                       ← Session-by-session discovery log
├── KNOWLEDGE_AUDIT.md                 ← Facts vs opinions audit
│
├── research/                          ← PILLAR 3: External landscape
│   ├── RESEARCH_FINDINGS.md           ← Audio feature fundamentals
│   ├── RESEARCH_AUDIO_ANALYSIS.md     ← FFT, beat detection, onset theory
│   ├── RESEARCH_AUDIO_VISUAL_MAPPING.md ← Feature→visual mapping strategies
│   ├── EXISTING_DATASETS.md           ← Available datasets survey
│   └── ESP32_AUDIO_DSP.md             ← Standalone ESP32 feasibility
│
├── analysis/                          ← PILLARS 1+2: Our research + algorithm validation
│   ├── taste/                         ← Pillar 1: Findings from user annotations
│   │   ├── beat_tap_analysis.*        ← What taps register on
│   │   ├── air_feature_analysis.*     ← What makes moments feel "airy"
│   │   ├── build_vs_climax.md         ← Build vs climax (derivatives)
│   │   ├── beat_vs_consistent.*       ← Flourish ratio extraction
│   │   ├── flourish_*.md              ← Flourish properties + comparisons
│   │   ├── fa_br_drop1_analysis.md    ← EDM drop in feature space
│   │   └── electronic_drop_summary.md ← Automated section detection
│   │
│   ├── algorithms/                    ← Pillar 2: Algorithm evaluation
│   │   ├── REALTIME_CONSTRAINTS.md    ← What's detectable without lookahead
│   │   ├── beat_tracker_comparison.*  ← 5-approach comparison
│   │   ├── library_comparison.md      ← librosa vs madmom vs essentia
│   │   ├── madmom_*.md                ← madmom evaluation
│   │   └── LED_MAPPING_GUIDE.md       ← Feature→LED mapping
│   │
│   ├── harmonix/                      ← Harmonix dataset work
│   │   ├── harmonix_exploration.md    ← Dataset structure + contents
│   │   ├── harmonix_optimization.*    ← Hyperparameter optimization
│   │   └── validation_tracks.md       ← Selected validation tracks
│   │
│   └── scripts/                       ← Reproducible analysis scripts
│
├── tools/                             ← PILLAR 4: Working prototypes
│   ├── README.md                      ← Tool documentation (consolidated)
│   ├── realtime_beat_led.py           ← Bass flux LED controller
│   ├── realtime_onset_led.py          ← Onset LED controller (A/B test)
│   ├── record_segment.py             ← Audio recording
│   ├── annotate_segment.py           ← Tap annotation
│   └── visualize_segment.py          ← Visualization
│
├── audio-segments/                    ← Raw data
│   ├── catalog.yaml                   ← Audio file catalog
│   ├── *.wav                          ← Audio recordings
│   ├── *.annotations.yaml            ← User annotations (TASTE DATA)
│   └── golden/                        ← Preserved original annotations
│
└── datasets/                          ← External datasets
    └── harmonix/                      ← 912 tracks, beat annotations
```

---

## Recommended Next Steps (Prioritized)

1. **Validate against WLED Sound Reactive** — biggest gap. Download it, run it, compare to our approach. If it's already good enough, our custom work needs to add clear value.

2. **A/B test on tree** — play diverse music through both detectors (`--detector both`), form opinions on what feels right.

3. **Record more annotations** — 2 songs is a thin dataset. Even 5-10 more (across genres) would strengthen the general principles.

4. **Build the interpreter registry** — make algorithm switching a first-class feature, not just different scripts.

5. **Test ESP32 standalone** — prove the spectral flux algorithm runs on actual ESP32 hardware with a mic.
