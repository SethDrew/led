# Audio Analysis for Music-Reactive LED Projects: Math, Tradeoffs, and Failure Modes

**Goal:** Understand what each audio analysis approach actually does mathematically, what it's optimized for, where it fails, and when to use it for LED visualizations.

**Last updated:** 2026-02-05

---

## 1. Approaches by Problem

### 1.1 Beat Detection ("Flash LEDs on kick drum")

#### **Energy-Based Thresholding** (Simple, real-time friendly)
**What it does:**
- Compute FFT on each audio chunk (typically 1024 samples at 44.1kHz = 23ms)
- Sum energy in bass band (60-130 Hz) and low-mid (301-750 Hz)
- Track 1-second history (~43 frames at 44.1kHz)
- Calculate: `avg(history)` and `variance(history)`
- Detect beat when: `current_energy > (-15 × variance + 1.55) × avg`

**Optimized for:**
- Kick drums and snare hits in typical pop/EDM music
- Ultra-low latency (single buffer = 23ms)
- Simple embedded implementation (no ML, minimal memory)

**Failure modes:**
- **Sustained bass:** A long bass note has same frequency content as kick drum but shouldn't trigger beats
- **Tempo changes:** Variance-based threshold adapts too slowly to sudden tempo shifts
- **Quiet sections:** Low variance makes threshold hypersensitive, triggers false beats on noise
- **Complex rhythms:** Only looks at energy, misses syncopation and polyrhythms

**When to use:**
- Simple real-time LED projects where "flash on loud kick" is good enough
- Microcontrollers with limited CPU (ESP32, Arduino)
- When you need <50ms latency

**Latency:** 23-46ms (1-2 buffers)

---

#### **Spectral Flux Onset Detection** (Better accuracy, still real-time)
**What it does:**
- Compute STFT (Short-Time Fourier Transform) on overlapping windows
- Calculate difference between current spectrum and previous: `sum(max(0, S[f,t] - S[f,t-1]))`
- Half-wave rectification (only positive changes = energy increases)
- Threshold on onset strength envelope (often adaptive like median + constant)

**Optimized for:**
- Percussive transients (drums, plucks, attacks)
- Better separation of harmonic vs. percussive content than raw energy

**Failure modes:**
- **Tonal onsets:** Misses sustained note changes (e.g., organ chord -> different chord)
- **Vibrato/tremolo:** Rapid spectral changes can false-trigger as onsets
- **Harmonic interference:** Cymbal crashes have huge spectral spread, can mask kick drum onsets

**When to use:**
- You want to distinguish kick/snare from bass guitar
- Moderate CPU available (Python on Pi, or C++ with FFT library)
- Still need low latency but want better accuracy than energy threshold

**Latency:** ~50-100ms (depends on hop size, typically 512 samples = 11.6ms hop)

**Implementation:**
- **aubio**: `specflux` method
- **librosa**: `onset_strength` with default mel-spectrogram
- **Essentia**: `OnsetDetection` with `flux` or `melflux`

---

#### **High Frequency Content (HFC)**
**What it does:**
- Weight each FFT bin by its frequency: `sum(frequency[f] × magnitude[f])`
- Higher frequencies contribute more to score
- Percussive hits have sharp high-frequency transients

**Optimized for:**
- Snare drums, hi-hats, claps (high-frequency percussion)

**Failure modes:**
- **Bass-heavy music:** Kick drums are low-frequency, HFC barely notices them
- **Cymbal-heavy:** Constant ride cymbal keeps HFC elevated, hard to detect individual hits

**When to use:**
- You specifically want to react to snares/hi-hats, not bass
- Complement to bass energy detection (use both for full drum kit response)

**Implementation:**
- **aubio**: `hfc` method
- **Essentia**: `OnsetDetection` with `hfc`

---

#### **Complex Domain / Phase Deviation**
**What it does:**
- Track both magnitude AND phase changes in spectrum
- Phase changes indicate new sound sources starting (phase resets at attack)
- Formula: `sum(weighted_phase_derivative × magnitude)`

**Optimized for:**
- Polyphonic music (multiple instruments attacking at once)
- Tonal onsets (new note in piano, guitar string pluck)

**Failure modes:**
- **Noisy phase:** Requires clean audio; MP3 compression artifacts create phase noise
- **Over-detection:** Tremolo, vibrato, and phase modulation effects false-trigger
- **CPU cost:** More expensive than magnitude-only methods

**When to use:**
- Classical, jazz, or other polyphonic genres
- You need to detect melodic note changes, not just drum hits
- CPU not a constraint

**Implementation:**
- **aubio**: `complex` or `phase` methods
- **Essentia**: `OnsetDetection` with `complex` or `complex_phase`

---

#### **Dynamic Programming Beat Tracking (librosa.beat.beat_track)**
**What it does:**
1. Compute onset strength envelope (spectral flux on mel-spectrogram)
2. Autocorrelation to estimate global tempo (find periodicity in onset envelope)
3. Dynamic programming: pick beat times that are (a) high onset strength, (b) roughly periodic with estimated tempo
4. Tightness parameter controls how strictly it enforces periodicity

**Optimized for:**
- Steady-tempo Western pop/rock music
- Finding all beats in a song (offline analysis)
- Beats that align with strong onsets

**Failure modes:**
- **Tempo changes:** Assumes single global tempo; half-time or double-time sections throw it off
- **Rubato:** Classical music with tempo fluctuations confuses the periodicity assumption
- **Weak beats:** Off-beat hi-hats or syncopation may have higher onset strength than actual beats
- **Lookahead required:** Needs several seconds of audio to estimate tempo (NOT real-time)

**When to use:**
- Offline pre-analysis (e.g., sync LED show to pre-recorded song)
- Music with clear, steady beats
- You want beat positions, not just "a beat happened now"

**Latency:** NOT REAL-TIME (needs 5-10 seconds lookahead)

**Parameters:**
- `hop_length=512` (11.6ms time resolution)
- `tightness=100` (higher = stricter tempo enforcement, may miss syncopation)
- `start_bpm=120` (initial guess, affects convergence)

---

#### **RNN Beat Tracking (madmom)**
**What it does:**
- Recurrent Neural Network trained on beat-annotated music datasets
- Input: multi-band spectral features (mel-spectrogram or similar)
- Output: probability of beat at each time step
- Post-processing: dynamic Bayesian network (DBN) or HMM to enforce tempo consistency

**Optimized for:**
- Complex rhythms (syncopation, polyrhythms, odd meters)
- Tempo changes within song
- Genres covered in training data (pop, rock, electronic, classical)

**Failure modes:**
- **Unknown genres:** Trained on specific datasets (likely Western music-heavy); fails on unfamiliar genres (gamelan, free jazz)
- **CPU cost:** RNN inference is slow; not suitable for real-time on embedded devices
- **Black box:** Can't explain why it fails, hard to tune
- **Latency:** Requires context window (typically 3-10 seconds)

**When to use:**
- Offline analysis of complex music
- You have CPU to burn (Python on desktop, not microcontroller)
- Need best-in-class accuracy on Western music

**Latency:** 3-10 seconds lookahead

**Training data:** Not fully documented, but madmom papers reference:
- SMC MIREX datasets (pop, rock, electronic)
- Ballroom dance music datasets
- Classical music corpora

**Implementation:**
- **madmom**: `RNNBeatProcessor` + `DBNBeatTrackingProcessor`

---

#### **Multi-Feature Beat Tracking (Essentia)**
**What it does:**
- Compute 5 onset detection functions: complex spectral diff, energy flux, melflux, beat emphasis, spectral info gain
- Run TempoTapDegara on each (tempo induction via autocorrelation)
- TempoTapMaxAgreement: pick beats where multiple features agree

**Optimized for:**
- Robustness across genres (ensemble of methods reduces failure modes of any single one)
- 44.1kHz audio only

**Failure modes:**
- **Sample rate dependent:** MUST be 44.1kHz (hard-coded filter parameters)
- **CPU cost:** 5 onset functions is expensive
- **Still assumes periodicity:** TempoTap relies on autocorrelation = steady tempo assumption

**When to use:**
- Need robustness but not RNN complexity
- Audio is 44.1kHz (resample if needed)
- Offline or real-time with powerful CPU

**Confidence score:** 0-5.32 (>3.5 = excellent, ~80% accuracy)

---

### 1.2 Frequency Band Energy ("Color shift based on bass vs. treble")

#### **Linear FFT Bins**
**What it does:**
- Standard FFT: N time-domain samples → N/2 + 1 frequency bins
- Bin frequency: `f[i] = i × sample_rate / N`
- Example: 1024-point FFT at 44.1kHz → bins at 0 Hz, 43 Hz, 86 Hz, ..., 22.05 kHz
- Linear spacing: each bin is same Hz width (43 Hz in this example)

**Optimized for:**
- Simple, fast (hardware FFT on many microcontrollers)
- Uniform frequency resolution across spectrum

**Failure modes:**
- **Bass resolution:** 43 Hz bins can't distinguish 60 Hz kick from 80 Hz bass synth
- **High-frequency waste:** Humans don't perceive 10 kHz vs. 10.043 kHz as different; wasting bins
- **Octave mismatch:** Music is octave-based (A4=440 Hz, A5=880 Hz, doubling); linear bins don't align

**When to use:**
- Microcontroller with hardware FFT
- You're going to manually group bins into bands anyway (e.g., bass = bins 1-3, mid = bins 4-10)
- Need absolute minimum CPU

**Typical band grouping for LED visualization:**
- Bass: 60-250 Hz (bins 1-5 in 1024-point FFT at 44.1kHz)
- Mid: 250-2000 Hz (bins 6-46)
- Treble: 2000-8000 Hz (bins 47-186)

---

#### **Mel-Scale Filterbanks**
**What it does:**
- Convert linear FFT bins to perceptually-spaced bands
- Mel scale formula: `mel = 2595 × log10(1 + f/700)`
- Below ~1 kHz: approximately linear spacing
- Above 1 kHz: logarithmic spacing (more like octaves)
- Typical: 40-128 mel bands covering 0-8000 Hz

**Optimized for:**
- Speech recognition (designed for human vocal perception)
- Balancing bass and treble resolution (better than linear FFT)

**Failure modes:**
- **Not true octaves:** Mel scale is empirical fit to speech perception, not music theory
- **Bass bias:** Still oversamples low frequencies vs. musical octaves
- **Over-smoothing:** Fewer bands than FFT bins = lose fine frequency detail

**When to use:**
- You want perceptually-balanced frequency bands without heavy computation
- Most music genres (not just speech)
- Default choice for onset detection (librosa, aubio use mel-spectrogram)

**Implementation:**
- **librosa**: `melspectrogram(n_mels=128, fmin=0, fmax=8000)` (default fmax = sr/2)
- **Essentia**: `MelBands`

---

#### **Constant-Q Transform (CQT)**
**What it does:**
- Frequency bins spaced by musical intervals (e.g., 12 bins per octave = semitones)
- Each bin has constant Q = center_frequency / bandwidth
- Example: 7 octaves × 12 bins/octave = 84 bins covering 32 Hz to 4186 Hz (C1 to C8)

**Optimized for:**
- Music with pitched instruments (piano, guitar, vocals)
- Chromagram (pitch class detection)
- Equal resolution per octave (bass and treble treated equally in musical sense)

**Failure modes:**
- **CPU cost:** CQT is much slower than FFT (no Fast CQT like there is FFT)
- **Time-frequency tradeoff:** Low notes need long windows (good frequency res, poor time res); high notes vice versa
- **Percussive content:** Drums are not pitched; CQT doesn't help, use onset detection instead

**When to use:**
- Offline analysis where CPU doesn't matter
- You care about pitch/harmony (e.g., color LEDs by musical key)
- NOT for real-time beat detection (use mel-spectrogram instead)

**Implementation:**
- **librosa**: `cqt(bins_per_octave=12, n_bins=84)` (default 7 octaves)

---

#### **Octave Bands (Spectral Contrast)**
**What it does:**
- Divide spectrum into 6-7 octave bands (e.g., 200-400 Hz, 400-800 Hz, 800-1600 Hz, ...)
- For each band, compute difference between peak energy (top 2% of bins) and valley energy (bottom 2%)
- High contrast = clear, narrow-band signals (melodic instruments)
- Low contrast = broadband noise (distortion, cymbals)

**Optimized for:**
- Distinguishing harmonic vs. noisy content
- Music genre classification (metal has high mid-range contrast, ambient has low)

**Failure modes:**
- **Not an energy measure:** Contrast is relative; a quiet flute can have higher contrast than a loud kick drum
- **Slow changes:** Designed for frame-level analysis (100ms), not beat-level (milliseconds)

**When to use:**
- You want to detect "vibe" changes (e.g., shift LED color palette when song goes from melodic to distorted)
- NOT for beat detection or real-time energy visualization

**Implementation:**
- **librosa**: `spectral_contrast(n_bands=6, fmin=200)`

---

### 1.3 Pitch / Harmony ("Detect musical key, color by chord")

#### **Chromagram (Pitch Class Profile)**
**What it does:**
- Map all frequencies to 12 pitch classes (C, C#, D, ..., B)
- Fold octaves together (all A notes contribute to "A" bin, regardless of octave)
- Typically computed from CQT or STFT with tuning adjustment

**Optimized for:**
- Detecting chords and harmonic content
- Key detection (major vs. minor, tonic pitch)
- Works for Western 12-tone equal temperament music

**Failure modes:**
- **Percussive content:** Drums have no pitch; chromagram is just noise
- **Microtonal music:** Non-Western scales, quarter-tones → forced into 12-bin grid
- **Tuning drift:** Live recordings may not be A=440 Hz; need tuning parameter
- **Overlapping harmonics:** Complex chords blend into mush (C major + F major = hard to parse)

**When to use:**
- LED color mapped to musical key (e.g., red = C, orange = D, ...)
- Detect chord changes (change LED pattern on new chord)
- NOT for rhythm or beat detection

**Implementation:**
- **librosa**: `chroma_stft` or `chroma_cqt`
- **Essentia**: `HPCP` (Harmonic Pitch Class Profile)

---

### 1.4 Tempo Estimation ("Detect BPM changes in prog rock")

#### **Autocorrelation of Onset Envelope** (librosa default)
**What it does:**
- Compute onset strength envelope (how "attack-y" is the audio at each time)
- Autocorrelation: does the envelope repeat every X seconds?
- Peak autocorrelation at lag = 0.5 seconds → tempo = 120 BPM

**Optimized for:**
- Music with clear, periodic onsets (4/4 pop, techno)
- Single global tempo

**Failure modes:**
- **Tempo changes:** Autocorrelation averages over several seconds; can't track rapid changes
- **Syncopation:** Off-beat accents create autocorrelation peaks at half or double tempo
- **Ballads:** Sparse onsets (e.g., 1 piano note per beat) → weak autocorrelation

**When to use:**
- Offline BPM detection for steady-tempo music
- Quick estimate (faster than RNN)

**Latency:** 5-10 seconds of audio needed

---

#### **Tempogram** (Local autocorrelation)
**What it does:**
- Sliding-window autocorrelation of onset envelope
- Each frame: "what are the tempo-like periodicities in the last ~9 seconds?"
- 2D output: time × tempo_lag

**Optimized for:**
- Visualizing tempo changes over time
- Detecting ritardando, accelerando, half-time sections

**Failure modes:**
- **Post-hoc only:** Not a tempo estimate, just a visualization/analysis tool
- **Window size tradeoff:** Long window = good tempo resolution, poor time resolution; vice versa

**When to use:**
- Analyze where tempo changes happen (for manual LED timing adjustments)
- NOT for real-time control

**Implementation:**
- **librosa**: `tempogram(win_length=384)`

---

#### **BPM Histogram (Essentia)**
**What it does:**
- Compute tempo estimates across entire song
- Build histogram of BPM values
- Peaks in histogram = dominant tempos

**Optimized for:**
- Songs with multiple sections at different tempos
- Detecting if a song is ambiguous (could be 120 BPM or 60 BPM)

**When to use:**
- Offline analysis to decide if song has steady tempo
- Detect songs that need manual tempo correction

---

### 1.5 Mood / Timbre ("Detect if song feels happy or sad")

#### **MFCCs (Mel-Frequency Cepstral Coefficients)**
**What it does:**
- Compute mel-spectrogram
- Take DCT (discrete cosine transform) of log-mel spectrum
- Keep first 13-20 coefficients
- MFCCs capture "spectral envelope" (overall shape of frequency content)

**Optimized for:**
- Speech recognition (designed to capture vocal timbre)
- Music genre classification (distinguish guitar from synth)

**Failure modes:**
- **Not semantic:** MFCCs don't know "happy" vs. "sad"; need ML classifier on top
- **Training data bias:** Mood classifiers trained on Western pop won't generalize to other cultures
- **Subjective:** "Sad" is culturally dependent (minor key in West, different in Middle Eastern music)

**When to use:**
- Train a classifier on labeled music (happy/sad, energetic/calm)
- NOT plug-and-play; requires ML pipeline

**Implementation:**
- **librosa**: `mfcc(n_mfcc=13)`
- **Essentia**: `MFCC`

---

#### **Spectral Features (Rolloff, Centroid, Flatness)**
**What it does:**
- **Rolloff:** Frequency below which 99% of energy is concentrated (bright vs. dull sound)
- **Centroid:** "Center of mass" of spectrum (higher = brighter)
- **Flatness:** Ratio of geometric mean to arithmetic mean (1 = white noise, 0 = pure tone)

**Optimized for:**
- Heuristic timbre description
- Simple rules (e.g., high centroid = bright = happy; low centroid = dark = sad)

**Failure modes:**
- **Oversimplified:** Brightness doesn't always mean happy (harsh industrial music is bright but not happy)
- **No context:** Single-frame features ignore temporal patterns (major vs. minor chord needs harmony analysis, not spectral features)

**When to use:**
- Quick heuristics for LED effects (brighter sound = brighter LEDs)
- NOT for semantic mood (happy/sad/energetic)

**Implementation:**
- **librosa**: `spectral_rolloff`, `spectral_centroid`, `spectral_flatness`

---

## 2. Algorithm Deep Dives

### 2.1 Spectral Flux (The Workhorse of Onset Detection)

**Full algorithm:**
```
1. STFT: Compute FFT on overlapping windows (e.g., 1024 samples, 512 hop)
   → Time-frequency matrix S[frequency, time]

2. Log-power mel-spectrogram (librosa default):
   - Take magnitude spectrum: |S[f, t]|
   - Apply mel filterbank: sum FFT bins into ~128 mel bands
   - Convert to log scale: log(mel_energy + epsilon)

3. First-order difference:
   diff[mel, t] = S[mel, t] - S[mel, t - lag]
   (lag = 1 frame by default)

4. Half-wave rectification:
   onset_strength[t] = mean_over_mel_bands(max(0, diff[mel, t]))
   (Only keep positive changes = energy increases)

5. Optional: Local max filtering
   - Replace S[mel, t-1] with local max in frequency neighborhood
   - Suppresses vibrato (rapid frequency modulation)

6. Output: Onset strength envelope (1D signal, one value per time frame)
```

**Why this math?**
- **Log scale:** Human perception is logarithmic; 10 dB change sounds similar across frequencies
- **Mel scale:** Matches auditory perception; more resolution in bass (where kicks live)
- **Half-wave rectification:** Energy *decreases* (note offset) aren't onsets
- **Mean over frequency:** Drums hit many frequencies at once; averaging captures this

**Assumptions:**
- Onsets = sudden energy increases
- Energy increases happen across multiple frequency bands
- Mel-scale weighting is musically appropriate (mostly true for Western music)

**What it misses:**
- Tonal onsets where energy stays constant (organ chord change)
- Onsets masked by louder sounds (quiet hi-hat during loud bass)
- Sub-bass (<60 Hz) if mel filterbank starts at 60 Hz

---

### 2.2 Ellis Dynamic Programming Beat Tracker (librosa.beat.beat_track)

**Full algorithm:**
```
1. Onset strength envelope (spectral flux, see above)

2. Tempo estimation via autocorrelation:
   - Autocorrelate onset envelope
   - Find peaks in autocorrelation → candidate tempos
   - Weight by prior (Gaussian centered on start_bpm=120)
   - Pick tempo with highest weighted autocorrelation

3. Dynamic programming:
   - Define cost function: beat_cost[t] = -onset_strength[t] + tempo_penalty
   - tempo_penalty = tightness × |time_since_last_beat - expected_beat_interval|²
   - Use Viterbi algorithm to find beat path minimizing total cost

4. Post-processing:
   - Trim weak leading/trailing beats (if trim=True)
   - Convert frame indices to time/samples
```

**Why this math?**
- **Autocorrelation finds periodicity:** If onsets repeat every 0.5 sec, autocorrelation peaks at 0.5 sec lag
- **Dynamic programming enforces consistency:** Can't have beats randomly spaced; must be quasi-periodic
- **Tightness parameter balances:** High tightness = strict metronome; low = flexible to syncopation

**Assumptions:**
- Beats are roughly periodic (true for 90% of Western pop/rock/EDM)
- Tempo is constant over the ~10-second analysis window
- High onset strength → likely beat (true for downbeat, less true for off-beats)

**Failure modes:**
- **Tempo changes:** DP assumes single tempo; half-time section locks to wrong tempo
- **Syncopation:** Off-beat hi-hats may have higher onset strength than on-beat bass (beat tracker locks to wrong phase)
- **Waltz (3/4 time):** DP may lock to every beat or every measure depending on onset pattern
- **Sparse beats:** Classical music with long held notes → weak onset envelope → tempo estimation fails

**Latency:** Needs ~10 seconds to estimate tempo reliably (autocorrelation needs multiple beat cycles)

---

### 2.3 Harmonic-Percussive Source Separation (HPSS)

**Full algorithm:**
```
1. Compute STFT magnitude spectrogram S[frequency, time]

2. Median filtering:
   - Horizontal median filter (across time): detects horizontal streaks = harmonic content
     H_soft = median_filter(S, kernel_size=(1, 17))  # 17 frames ≈ 200ms
   - Vertical median filter (across frequency): detects vertical streaks = percussive content
     P_soft = median_filter(S, kernel_size=(17, 1))

3. Soft masking (if margin=1):
   - H_mask = H_soft / (H_soft + P_soft)
   - P_mask = P_soft / (H_soft + P_soft)
   - H = H_mask × S
   - P = P_mask × S

4. Hard masking (if margin > 1):
   - H_mask = (H_soft > margin × P_soft)
   - P_mask = (P_soft > margin × H_soft)
   - H = H_mask × S
   - P = P_mask × S

5. Inverse STFT to get time-domain harmonic and percussive signals
```

**Why this math?**
- **Harmonic = horizontal:** Sustained notes have constant frequency over time → horizontal lines in spectrogram
- **Percussive = vertical:** Drum hits have broadband energy at instant of attack → vertical lines
- **Median filter finds structure:** Median preserves streaks, suppresses noise
- **Margin > 1:** Cleaner separation but loses some energy (safer for beat tracking to avoid harmonic interference)

**Use cases for LED visualization:**
- Separate beat tracking (percussive) from pitch detection (harmonic)
- Example: detect kick drum beats in presence of heavy bass guitar
- Can improve onset detection by running spectral flux on percussive component only

**Limitations:**
- **Not perfect:** Kick drum has tonal component (50-60 Hz fundamental); HPSS may split it wrong
- **Latency:** Median filter is causal but needs kernel_size frames (17 frames = ~200ms)
- **Tuning required:** kernel_size and margin depend on musical style

---

### 2.4 Constant-Q Transform (CQT)

**Math:**
- **FFT:** Fixed window size → constant time resolution, frequency resolution = sr/window_size
- **CQT:** Variable window size per frequency
  - Low frequencies: long window (good freq resolution, poor time resolution)
  - High frequencies: short window (poor freq resolution, good time resolution)
- **Q factor:** Q = f_center / bandwidth (constant across all bins)
- **Typical:** Q ≈ 34.5 for 12 bins/octave (each bin covers 1 semitone)

**Example:**
- C2 (65.4 Hz): window = 4096 samples (93ms at 44.1kHz), bandwidth = 1.9 Hz
- C7 (2093 Hz): window = 128 samples (2.9ms), bandwidth = 60 Hz

**Why use for music?**
- Musical intervals are logarithmic (A4=440 Hz, A5=880 Hz, ratio of 2)
- CQT bins align with semitones → easy to build chromagram

**Why NOT use for beat detection?**
- **Slow:** No fast algorithm like FFT (CQT is O(N × num_bins) vs. FFT's O(N log N))
- **Low-frequency latency:** Long windows for bass = delay
- **Percussive content has no pitch:** CQT wastes computation on drums

**Use for LED visualization:**
- Offline: compute CQT → chromagram → detect chord changes → trigger LED scene change
- Real-time: NOT recommended (too slow)

---

## 3. Failure Modes Table

| **Problem** | **Algorithm** | **Failure Mode** | **Musical Example** | **Symptom** |
|-------------|---------------|------------------|---------------------|-------------|
| **Beat detection** | Energy threshold | Sustained bass | Daft punk - "Around the World" (bass synth holds notes) | LEDs flash on held bass notes, not just kicks |
| | Energy threshold | Quiet section | Acoustic ballad intro | False triggers on background noise |
| | Spectral flux | Tonal onset | Organ chord change | Misses chord changes (no percussive attack) |
| | Spectral flux | Vibrato | Opera vocals | False onsets on pitch wobbles |
| | HFC | Bass-heavy | Hip-hop with kick-only beat | Misses most kicks (low freq) |
| | librosa beat_track | Tempo change | Led Zeppelin - "Stairway to Heaven" (speeds up) | Locks to first tempo, misses later beats |
| | librosa beat_track | Syncopation | Reggae (off-beat emphasis) | Locks to off-beats instead of downbeats |
| | librosa beat_track | Rubato | Chopin piano (tempo fluctuates) | Averages tempo, beats drift out of sync |
| | RNN beat tracker | Unknown genre | Balinese gamelan | Trained on Western pop → fails |
| **Frequency bands** | Linear FFT | Bass resolution | Distinguish 60 Hz kick from 80 Hz bass | Both in same FFT bin, can't separate |
| | Mel filterbank | Over-smoothing | Detect specific guitar note | Mel bands too wide, multiple notes blend |
| | CQT | Percussive content | Drum kit | Wasting CPU on non-pitched sounds |
| **Pitch detection** | Chromagram | Microtonal music | Middle Eastern maqam | Forced into 12-tone grid, sounds wrong |
| | Chromagram | Percussive | Electronic music (drums only, no melody) | Chromagram is just noise |
| **Tempo** | Autocorrelation | Syncopation | Funk (strong off-beats) | Estimates half or double the actual tempo |
| | Autocorrelation | Sparse onsets | Ambient (long sustained notes) | No clear periodicity → fails |
| **Mood** | MFCCs + classifier | Cultural bias | Japanese enka (minor key but celebratory) | Western "sad" classifier mislabels as sad |
| | Spectral centroid | Genre mismatch | Harsh industrial (bright but aggressive) | High centroid suggests "happy" but song is dark |
| **Normalization** | Per-frame max norm | Absolute level | Quiet breakdown vs. loud chorus | Both normalize to same brightness, lose dynamic range |
| | Global max norm | Dynamic range | Song with one peak scream | Rest of song looks too dim |
| **Harmonic-Percussive** | HPSS | Tonal percussion | Tabla, toms (pitched drums) | Splits drum into harmonic + percussive, loses attack |
| | HPSS | Distorted guitar | Metal (sustain + crunch) | Distortion is broadband = classified as percussive |

---

## 4. Recommendation Matrix

### "I want to..."

#### **Flash LEDs on every kick drum**
→ **Spectral flux onset detection** on bass band (60-250 Hz)
- **Why:** Kick drums are percussive (good for flux) and low-frequency (isolate in bass band)
- **Library:** aubio `onset` with `specflux`, or librosa `onset_detect` with mel-spectrogram (fmax=250)
- **Real-time:** Yes (50-100ms latency with hop_length=512)
- **Fails on:** Sustained bass notes, very quiet kicks

**Alternative for extreme low-latency:**
→ **Energy threshold** on 60-130 Hz band
- **Why:** Simpler, faster (23ms latency)
- **Library:** Raw FFT (scipy, numpy) + manual thresholding
- **Fails on:** Sustained bass, false triggers on noise

---

#### **Flash LEDs on snare/hi-hat**
→ **HFC onset detection** or **spectral flux on mid/high bands** (500+ Hz)
- **Why:** Snare/hi-hat have high-frequency content
- **Library:** aubio `onset` with `hfc`, or librosa `onset_detect` with fmin=500
- **Real-time:** Yes (50-100ms)
- **Fails on:** Cymbal crashes may cause continuous triggering

---

#### **Detect all drum hits (kick + snare + hi-hat)**
→ **HPSS + spectral flux on percussive component**
- **Why:** Separate drums from bass guitar/synth
- **Library:** librosa `hpss` → `onset_detect` on percussive signal
- **Real-time:** Marginal (~200ms latency due to HPSS median filter)
- **Fails on:** Tonal percussion (tabla, congas)

---

#### **Color LEDs by bass vs. treble energy**
→ **Mel filterbanks** (3-5 bands: sub-bass, bass, mid, high-mid, treble)
- **Why:** Perceptually balanced, fast to compute
- **Library:** librosa `melspectrogram(n_mels=5, fmin=60, fmax=8000)` → sum energy per band
- **Real-time:** Yes (23ms latency, same as FFT)
- **Fails on:** Nothing major (robust approach)

**Example band mapping:**
- Sub-bass (20-60 Hz) → Deep purple LEDs
- Bass (60-250 Hz) → Blue
- Mid (250-2000 Hz) → Green
- High-mid (2000-6000 Hz) → Yellow
- Treble (6000+ Hz) → Red

---

#### **Change LED color based on musical key**
→ **Chromagram** (offline) or **simplified pitch detection** (real-time)
- **Offline:** librosa `chroma_cqt` → aggregate over time → detect dominant pitch class
- **Real-time:** Track strongest FFT bin in each octave → map to pitch class (crude but fast)
- **Fails on:** Percussive music (no pitch), microtonal music

**Example:** C=red, C#=orange, D=yellow, ..., B=violet (color wheel)

---

#### **Detect tempo changes (prog rock, classical)**
→ **Tempogram** (offline analysis) or **RNN beat tracker** (if CPU available)
- **Why:** Need local tempo estimation; global methods fail
- **Offline:** librosa `tempogram` → visualize where tempo changes → manually adjust LED timing
- **Real-time:** madmom `RNNBeatProcessor` (if you have desktop CPU)
- **Fails on:** Rubato, unmeasured music

**Practical for LED projects:** Pre-analyze song, save beat times to file, sync LEDs to file (not live detection)

---

#### **Detect "energy" for brightness control**
→ **RMS energy** or **spectral rolloff**
- **RMS:** `sqrt(mean(audio_chunk²))`
- **Rolloff:** Frequency below which 99% of energy lies
- **Why:** Simple, robust, real-time
- **Library:** librosa `rms` or `spectral_rolloff`
- **Real-time:** Yes (23ms)
- **Fails on:** Mastered music (heavily compressed → always loud)

**Normalization:** Use adaptive threshold (like beat detection variance method) to avoid saturation

---

#### **Detect "mood" (happy vs. sad)**
→ **Don't.** This is ML-hard and culturally subjective.
- **Heuristic:** High spectral centroid + major key (from chromagram) = happier
- **Real solution:** Train classifier on labeled dataset (MFCCs + chroma features → SVM/neural net)
- **Better approach for LEDs:** Use proxy features:
  - Bright/dark: spectral centroid
  - Energetic/calm: RMS energy + onset rate
  - Harmonic/noisy: spectral flatness

---

#### **Sync LED show to pre-recorded song**
→ **Offline beat tracking** → save beat times → real-time playback with lookahead
- **Process:**
  1. librosa `beat_track` on full song → get beat times
  2. Save to file (JSON: `{"beats": [0.5, 1.0, 1.5, ...]}`)
  3. Real-time: read file, trigger LEDs at beat times (with ~50ms lookahead to account for LED latency)
- **Why:** Offline = use slow but accurate methods (RNN, multi-feature)
- **Fails on:** If song is played back at different tempo (DJ pitch shift)

---

#### **Real-time LED visualization with <50ms latency (live music, DJ set)**
→ **Energy threshold** or **spectral flux** on frequency bands
- **Why:** Only causal methods with single-buffer latency work
- **Avoid:** beat_track, RNN, CQT (all need lookahead or are slow)
- **Library:** aubio (C library, real-time optimized) or custom scipy/numpy FFT
- **Process:**
  1. Audio callback: 1024 samples (23ms at 44.1kHz)
  2. FFT → mel filterbank (5-10 bands)
  3. Spectral flux on each band → threshold → trigger LED
  4. Total latency: 23ms (audio) + 5ms (computation) + 10ms (LED serial) = 38ms

---

## 5. Library Quick Reference

### **librosa** (Python, offline-first)
- **Best for:** Feature extraction, beat tracking, spectral analysis
- **Real-time:** Possible but not optimized (use aubio instead)
- **Strengths:** Comprehensive, well-documented, great for research/prototyping
- **Weaknesses:** Slow (pure Python + numpy), assumes full audio in memory
- **Typical workflow:** Load audio → compute features → analyze offline
- **LED use case:** Pre-analyze song, generate beat times, sync LEDs to timeline
- **Install:** `pip install librosa` (requires scipy, numpy, soundfile)

**Key functions:**
- `beat_track()`: DP beat tracker
- `onset_detect()`: Spectral flux onset detection
- `melspectrogram()`: Mel-scale STFT
- `chroma_cqt()`: Pitch class profile
- `hpss()`: Harmonic-percussive separation
- `spectral_contrast()`: Octave-band peak-valley contrast

---

### **aubio** (C library, Python bindings, real-time)
- **Best for:** Real-time onset/beat/pitch detection
- **Real-time:** Yes (designed for it)
- **Strengths:** Fast, low latency, simple API
- **Weaknesses:** Less comprehensive than librosa, fewer analysis tools
- **Typical workflow:** Streaming audio → process chunks → detect onsets/beats
- **LED use case:** Live onset detection → trigger LED flash
- **Install:** `pip install aubio` (C library, may need build tools)

**Key classes:**
- `onset(method='specflux')`: Onset detection (7 methods: energy, hfc, complex, phase, specdiff, kl, mkl, specflux)
- `tempo()`: Beat tracking
- `pitch()`: Pitch detection (yin, yinfft, schmitt, fcomb, mcomb)

**Real-time pattern:**
```python
import aubio
o = aubio.onset("specflux", 1024, 512, 44100)  # buf_size, hop_size, samplerate
for audio_chunk in stream:
    if o(audio_chunk):
        trigger_led()
```

---

### **madmom** (Python, research-grade)
- **Best for:** State-of-art beat/onset/chord detection (RNN-based)
- **Real-time:** No (RNN inference + lookahead = seconds of latency)
- **Strengths:** Best accuracy on complex music
- **Weaknesses:** Slow, CPU-heavy, black-box ML
- **Typical workflow:** Offline analysis → save annotations
- **LED use case:** Pre-analyze complex song (prog rock, jazz) → extract beats
- **Install:** `pip install madmom` (requires numpy, scipy, Cython)

**Key classes:**
- `RNNBeatProcessor`: RNN-based beat activation
- `DBNBeatTrackingProcessor`: Bayesian beat tracking post-processing
- `OnsetDetectorLL`: Low-latency onset detection (still ~100ms)

---

### **Essentia** (C++, Python bindings, comprehensive)
- **Best for:** Music Information Retrieval research, ML-based analysis
- **Real-time:** Some algorithms yes, some no (check docs per-algorithm)
- **Strengths:** Huge library (>400 algorithms), production-ready (used by Spotify)
- **Weaknesses:** Steep learning curve, complex API
- **Typical workflow:** Build processing graph → extract features
- **LED use case:** Multi-feature beat tracking with confidence score
- **Install:** `pip install essentia` (precompiled wheels available)

**Key algorithms:**
- `OnsetDetection(method='hfc')`: 6 onset methods
- `BeatTrackerMultiFeature`: Ensemble beat tracker (5 features)
- `TempoCNN`: Deep learning tempo estimation
- `MelBands`, `MFCC`, `HPCP`: Spectral features

---

### **scipy.signal** (Python standard, low-level)
- **Best for:** Custom DSP, understanding algorithms
- **Real-time:** If you write the code yourself
- **Strengths:** No dependencies (built on numpy), full control
- **Weaknesses:** No high-level music algorithms (just FFT, filters, etc.)
- **Typical workflow:** Implement your own beat detector using FFT + thresholding
- **LED use case:** Microcontroller-like processing in Python (for prototyping)
- **Install:** `pip install scipy` (standard scientific Python)

**Key functions:**
- `spectrogram()`: STFT
- `welch()`: Power spectral density
- `find_peaks()`: Peak detection (for onset detection)
- `butter()`, `sosfilt()`: Bandpass filters

**Example simple beat detector:**
```python
import numpy as np
from scipy.signal import spectrogram, find_peaks

def detect_beats(audio, sr=44100):
    f, t, Sxx = spectrogram(audio, sr, nperseg=1024, noverlap=512)
    bass_energy = np.sum(Sxx[1:5, :], axis=0)  # 60-250 Hz
    peaks, _ = find_peaks(bass_energy, distance=sr//512*10)  # Min 10 frames apart
    return t[peaks]
```

---

## 6. Real-Time Considerations

### Latency Budget for LED Visualization
- **Audio capture:** 23ms (1024 samples at 44.1kHz)
- **FFT + feature extraction:** 1-5ms (depends on algorithm)
- **Beat detection logic:** <1ms (threshold, peak detection)
- **LED serial transmission:** 5-30ms (depends on protocol: WS2812B ≈ 30µs/LED)
- **Human perception:** <50ms feels instant; 50-100ms noticeable; >100ms feels laggy

**Total target:** <50ms end-to-end

### Which Algorithms Are Real-Time Feasible?

| **Algorithm** | **Latency** | **Real-time?** | **Notes** |
|---------------|-------------|----------------|-----------|
| Energy threshold | 23ms (1 buffer) | ✅ Yes | Fastest |
| Spectral flux (aubio) | 50ms (1 buffer + hop) | ✅ Yes | Good balance |
| HFC (aubio) | 50ms | ✅ Yes | Same as spectral flux |
| HPSS + onset | 200ms (median filter) | ⚠️ Marginal | Noticeable lag |
| librosa beat_track | 5-10 seconds | ❌ No | Needs lookahead |
| RNN beat tracker | 3-10 seconds | ❌ No | Needs context |
| CQT | 100ms (low freq windows) | ⚠️ Marginal | Slow to compute |
| Chromagram (from STFT) | 50ms | ✅ Yes | If using STFT, not CQT |
| Mel-spectrogram | 23ms | ✅ Yes | Same as FFT |
| Spectral contrast | 50ms | ✅ Yes | But too slow-changing for beats |

### Buffer Size Tradeoffs

| **Buffer Size (samples)** | **Time (44.1kHz)** | **Frequency Resolution** | **Time Resolution** | **Use Case** |
|---------------------------|-------------------|--------------------------|---------------------|--------------|
| 512 | 11.6ms | 86 Hz/bin | Excellent | High-frequency percussion |
| 1024 | 23ms | 43 Hz/bin | Good | General purpose |
| 2048 | 46ms | 21 Hz/bin | OK | Bass-heavy music |
| 4096 | 93ms | 11 Hz/bin | Poor | Offline analysis only |

**For LED visualization:** 1024 is standard (good bass resolution, acceptable latency)

---

## 7. Practical Code Snippets

### Energy Threshold Beat Detection (Ultra-Low-Latency)
```python
import numpy as np
from collections import deque

class SimpleBeatDetector:
    def __init__(self, sample_rate=44100, buffer_size=1024, history_len=43):
        self.sr = sample_rate
        self.buffer_size = buffer_size
        self.history = deque(maxlen=history_len)  # 1 second history
        self.bass_bins = (1, 6)  # 60-250 Hz for 1024-point FFT

    def process(self, audio_chunk):
        # FFT
        spectrum = np.abs(np.fft.rfft(audio_chunk))

        # Bass energy
        bass_energy = np.mean(spectrum[self.bass_bins[0]:self.bass_bins[1]])

        # Update history
        self.history.append(bass_energy)

        # Threshold
        if len(self.history) < 10:
            return False  # Not enough history

        avg = np.mean(self.history)
        var = np.var(self.history)
        threshold = (-15 * var + 1.55) * avg

        return bass_energy > threshold

# Usage
detector = SimpleBeatDetector()
for chunk in audio_stream:  # 1024 samples at a time
    if detector.process(chunk):
        flash_leds()
```

---

### Spectral Flux Onset Detection (librosa)
```python
import librosa
import numpy as np

# Load audio
y, sr = librosa.load('song.mp3', sr=44100)

# Onset strength envelope (spectral flux on mel-spectrogram)
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

# Detect onsets
onsets = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=512,
    backtrack=True,  # Refine onset times to local minimum (more accurate)
    units='time'
)

print(f"Detected {len(onsets)} onsets")
print(f"First 10 onset times: {onsets[:10]}")

# For real-time: compute onset_strength on chunks, threshold manually
```

---

### Multi-Band Energy for LED Color (Real-Time)
```python
import numpy as np
import librosa

def compute_band_energies(audio_chunk, sr=44100):
    """
    Returns dict of energy in 5 frequency bands (for LED color mapping)
    """
    # Mel spectrogram (5 bands)
    S = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=sr,
        n_fft=1024,
        hop_length=1024,  # No overlap for single-chunk analysis
        n_mels=5,
        fmin=60,
        fmax=8000
    )

    # Energy per band (mean over time, though we only have 1 frame)
    energies = np.mean(S, axis=1)

    return {
        'sub_bass': energies[0],  # 60-120 Hz
        'bass': energies[1],      # 120-250 Hz
        'mid': energies[2],       # 250-1000 Hz
        'high_mid': energies[3],  # 1000-4000 Hz
        'treble': energies[4]     # 4000-8000 Hz
    }

# Usage
for chunk in audio_stream:
    bands = compute_band_energies(chunk)
    led_color = map_bands_to_color(bands)  # Your LED mapping logic
    set_led_color(led_color)
```

---

### Offline Beat Tracking → Save for Real-Time Playback
```python
import librosa
import json

# Offline analysis
y, sr = librosa.load('song.mp3')
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')

# Save beat times
beat_data = {
    'tempo': float(tempo),
    'beats': beats.tolist()
}
with open('song_beats.json', 'w') as f:
    json.dump(beat_data, f)

# Real-time playback
import time
with open('song_beats.json') as f:
    beat_data = json.load(f)

start_time = time.time()
for beat_time in beat_data['beats']:
    # Wait until beat time
    while time.time() - start_time < beat_time:
        time.sleep(0.001)
    flash_leds()
```

---

## 8. Key Takeaways

1. **There is no one-size-fits-all algorithm.** Beat detection on EDM needs different math than beat detection on jazz.

2. **Real-time = causal + low latency.** Avoid: librosa beat_track, RNN, CQT. Use: aubio, scipy FFT, mel-spectrogram.

3. **Onset detection ≠ beat detection.** Onsets are *any* note attack; beats are *periodic* rhythmic pulses. Most "beat detectors" are onset detectors + tempo estimation.

4. **Spectral flux is the workhorse.** For 80% of LED projects, spectral flux (or HFC) onset detection on mel-spectrogram is the right answer.

5. **Normalize carefully.** Per-frame max normalization destroys absolute loudness. Use adaptive thresholding (variance-based) instead.

6. **Test on your target genre.** Algorithms trained/designed for pop music fail on metal, classical, or non-Western music.

7. **CPU budget matters.** Microcontroller (ESP32): energy threshold only. Raspberry Pi: aubio onset detection. Desktop: librosa, madmom, anything goes.

8. **Latency budget:** Audio (23ms) + processing (5ms) + LED transmission (10ms) = 38ms minimum. Humans tolerate up to ~50ms.

9. **Offline is easier.** If you can pre-analyze the song, do it. Save beat times to file, playback with lookahead. Removes all the hard real-time constraints.

10. **Frequency bands:** Mel-scale is good default. Linear FFT if you're on a microcontroller. CQT only if you need pitch detection.

11. **Don't try to detect "mood" directly.** Use proxy features (spectral centroid = brightness, RMS = energy, onset rate = rhythmic activity).

12. **HPSS is underrated.** Separate drums from bass before onset detection → much cleaner beat tracking in dense mixes.

---

## 9. Further Reading

**Papers (if you want the math):**
- Ellis (2007): "Beat Tracking by Dynamic Programming" - librosa's beat_track algorithm
- Böck et al. (2016): "Joint Beat and Downbeat Tracking with RNNs" - madmom RNN approach
- Fitzgerald (2010): "Harmonic/Percussive Separation using Median Filtering" - HPSS algorithm

**Practical Tutorials:**
- librosa tutorial: https://librosa.org/doc/main/tutorial.html
- aubio onset detection examples: https://github.com/aubio/aubio/tree/master/python/demos
- Music Information Retrieval course (Stanford): https://musicinformationretrieval.com/

**Real-World Projects:**
- Audio-reactive LED strip (Python + ESP32): https://github.com/scottlawsonbc/audio-reactive-led-strip
- BTrack (C++ real-time beat tracker): https://github.com/adamstark/BTrack
- YAAFE (comprehensive feature extractor): https://yaafe.github.io/Yaafe/

---

**Author's Note:** This document prioritizes practical understanding over academic rigor. Formulas are simplified; consult original papers for full derivations. All latency numbers assume 44.1kHz audio and typical hardware (Raspberry Pi 4, ESP32). Your mileage may vary.
