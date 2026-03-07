# Audio Features Reference

Comprehensive catalog of audio features implemented or identified for the LED effect system. Organized into standard features (librosa/numpy), custom calculus features (signals.py), per-band expansion rules, and dead ends.

---

## Standard Features (librosa / numpy)

| Feature | What it measures | Real-time? | Temporal scope |
|---|---|---|---|
| RMS | Loudness (energy) | Yes | Frame (46ms) |
| Onset strength | Instantaneous spectral flux | Yes | Frame |
| Spectral centroid | Brightness / timbral center | Yes | Frame |
| Spectral flatness | Noise-like vs tonal | Yes | Frame |
| ZCR (zero-crossing rate) | Noisiness | Yes | Frame |
| MFCCs | Timbral fingerprint (13 coefficients) | Yes | Frame |
| Chroma | Pitch class distribution (12 bins) | Yes | Frame |
| HPSS | Harmonic/percussive separation | Yes (streaming temporal-median) | Frame |
| Mel spectrogram | Perceptually-weighted frequency energy | Yes | Frame |

---

## Calculus Features (custom, first-class)

Custom signal processing primitives in `effects/signals.py`. These are not standard library features -- they emerged from project-specific research and are treated as first-class alongside librosa features.

| Feature | What it measures | Real-time? | Temporal scope | Source |
|---|---|---|---|---|
| AbsIntegral | \|d(RMS)/dt\| summed over 150ms -- rate of loudness change | Yes | Beat (~150ms window) | `signals.py:AbsIntegral` |
| Multi-band onset envelope | FFT -> 6 mel bands -> log energy -> diff -> half-wave rectify -> mean | Yes | Frame | `signals.py:OnsetTempoTracker` |
| Autocorrelation tempo | Periodicity detection from 5s buffer of absint or onset signal | Yes | Phrase (5s) | `signals.py:BeatPredictor`, `OnsetTempoTracker` |
| Rolling integral | Sum of energy over 5s window per band -- sustained energy detector | Yes | Phrase (5s) | `viewer.py` (visualization only) |
| Rolling integral slope | d/dt of rolling integral -- rising (build) vs falling (breakdown) | Untested | Phrase | Concept only |

---

## Per-Band Expansion

Any scalar feature can be computed independently per frequency band. With 5 bands, this multiplies the feature space by 5x. Not all features benefit equally:

- **Benefits strongly**: RMS, onset strength, spectral flux, absint (each band has independent energy dynamics)
- **Benefits moderately**: spectral centroid (sub-band centroid gives finer frequency detail)
- **Doesn't benefit**: overall spectral flatness, chroma (already frequency-decomposed)

Standard band definitions (from `band_zone_pulse.py`): sub-bass (20-80Hz), bass (80-250Hz), low-mid (250-1kHz), high-mid (1-4kHz), treble (4-16kHz).

---

## Full Feature Catalog

Features currently implemented or identified. This table is a snapshot -- new features will be added as research progresses.

| Feature | Category | Measures | Real-time? | Temporal Scope | Events it detects | Behaviors it drives |
|---|---|---|---|---|---|---|
| RMS | Standard | Loudness | Yes | Frame | Silence, loud passages | Volume meter, brightness |
| Onset strength | Standard | Spectral flux (one-sided) | Yes | Frame | Transients, percussion | Sparkle triggers, flash |
| Spectral centroid | Standard | Timbral brightness | Yes | Frame | Instrument changes, filter sweeps | Color temperature shift |
| Spectral flatness | Standard | Noise vs tone | Yes | Frame | Noise bursts, tonal sections | Sparkle density |
| ZCR | Standard | High-frequency content | Yes | Frame | Percussion, sibilance | Treble sparkle rate |
| MFCCs (13) | Standard | Timbral fingerprint | Yes | Frame | Instrument/section changes | Similarity detection |
| Chroma (12) | Standard | Pitch class energy | Yes | Frame | Key changes, chord progressions | Color palette selection |
| HPSS harmonic | Standard | Sustained tonal content | Yes | Frame | Chords, pads, vocals | Background color, glow |
| HPSS percussive | Standard | Transient content | Yes | Frame | Drums, clicks, consonants | Sparkle triggers |
| AbsIntegral | Calculus | \|d(RMS)/dt\| over 150ms | Yes | Beat | Beats, energy changes | Pulse, breathe, snake, meter |
| Onset envelope | Calculus | Multi-band onset flux | Yes | Beat | Onsets (better than absint on dense material) | Tempo estimation input |
| Beat prediction | Calculus | Autocorrelation + threshold | Yes | Phrase | Beat timing | Pulse, predicted beats |
| Tempo estimate | Calculus | Onset autocorrelation period | Yes | Phrase | Tempo (octave-ambiguous) | Pulse rate, crawl speed |
| Rolling integral (5s) | Calculus | Sustained energy per band | Untested | Phrase | Drops, builds, sections | Background color, growth |
| Per-band RMS (x5) | Expansion | Energy per frequency zone | Yes | Frame | Band-specific events | Zone coloring, band prop |
| Per-band absint (x5) | Expansion | Energy change per zone | Yes | Beat | Band-specific beats | Zone pulse, band sparkles |
| Per-band onset (x5) | Expansion | Onset per frequency zone | Yes | Frame | Band-specific transients | Zone sparkle, band flux |
| Dominant band (5s vote) | Derived | Which band has most energy | Yes | Phrase | Instrument dominance shifts | Background color selection |
| Autocorrelation confidence | Derived | Periodicity strength | Yes | Phrase | Rhythmic vs arrhythmic sections | Beat prediction enable/disable |

---

## Dead Ends

Features that don't work for our use case:

- **Equal-loudness weighting on mastered music** -- double-corrects because mix engineers already compensate (ledger: `equal-loudness-weighting-dead-end`)
- **Absint autocorrelation for tempo on percussive music** -- symmetric bump-dip destroys timing (ledger: `absint-fails-percussive`)
- **Spectral centroid for sparkle position** -- clusters mid-zone, tree feels underutilized (ledger: `centroid-position-clustering`)
- **NMF for band-level transient detection** -- overkill, less stable in streaming (ledger: `nmf-not-suited-for-sparkles`)
