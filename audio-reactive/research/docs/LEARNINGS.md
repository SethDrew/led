# Learnings & Key Insights

## Session 1 (2026-02-05)

### Discovery: Bass Detection is Fundamentally Broken in Naive Implementations
The existing `audio_stream.py` and `spectrum_stream.py` both fail to detect bass reliably:
- **Self-normalizing history** (dividing by rolling average): Steady bass reads as ~1.0x average, never triggering. Only detects bass *changes*, not bass *presence*. A four-on-the-floor kick doesn't register because it's always there.
- **Per-frame max normalization** (`spectrum / max(spectrum)`): If any other frequency band is louder, bass gets squashed proportionally. A dominant guitar or vocal makes bass invisible.
- **Root cause**: Both approaches destroy absolute energy information. The "bass" we hear is partly perceptual (A-weighting, Fletcher-Munson curves) and partly about transients vs sustain. A kick drum at 60Hz and a bass synth at 60Hz have identical frequency content but different temporal profiles.
- **Fix direction**: Rolling peak normalization with bounded adaptation range. Track peaks per-band independently. Consider HPSS to separate percussive (kick) from harmonic (bass guitar) content.

### Discovery: librosa beat_track Doubles Tempo on Syncopated Rock
- On Tool's "Opiate," librosa detected 161.5 BPM. Actual tempo is ~101-109 BPM (confirmed by user tap annotations).
- The autocorrelation-based tempo estimator locked onto hi-hat/eighth-note onsets instead of kick/quarter-note beats.
- This is a known failure mode of onset-strength + autocorrelation beat tracking — it can't distinguish metrical levels (quarter vs eighth vs sixteenth).
- **Potential fixes**: Constrain BPM search range, use bass-band-only onset detection, try madmom's RNN tracker (trained on rock), or pre-analyze offline with manual correction.

### Insight: User Tap Annotations Are Richer Than Beat Positions
User tapped along to Opiate and the data revealed:
1. **Intro hits at ~160 BPM** (0-1.5s) — fast rhythmic element caught attention
2. **Groove at ~105 BPM** (5-17s) — consistent quarter-note feel
3. **Rapid bursts at 29s and 32s** — responding to fills or intensity peaks (5+ taps in <0.5s)
4. **Subdivision shifts** (33-40s) — mixing quarter and eighth note taps, feel changing

The *changes in tapping behavior* are the most valuable signal — they mark where the song's character shifts. An LED system should probably change behavior at these same transition points.

User also noted they "changed what they were triggering on" partway through — this is expected and valuable. The song invites different types of rhythmic engagement at different moments. Don't smooth this out; use it.

### Insight: Two Independent Quality Axes
The initial research over-conflated "LED projects" with "audio analysis." These are separate concerns:
1. **Audio analysis quality**: How well can we extract features from audio? (beat, frequency, onset, mood) — This is well-served by existing libraries (librosa, aubio).
2. **Audio→visual mapping quality**: How well do extracted features translate to compelling LED effects? (decay curves, color mapping, normalization) — This is where the art lives and existing projects provide useful starting constants but no "right answers."

### Insight: The "Feeling Layer" Requires Human-in-the-Loop Development
No library provides `detect_airiness()` or `detect_tension()`. The computable features exist:
- Spectral flatness ≈ "noise-like vs tonal" (could map to airiness)
- Spectral centroid ≈ "brightness" (higher = brighter sound)
- RMS energy ≈ "loudness"
- HPSS harmonic component ≈ "sustained texture" (persists across beat changes)
- Onset rate ≈ "rhythmic activity"

But the mapping from feature combinations → human feelings is subjective and contextual. User described "airiness" in the Fred again.. track that persists even after the beat returns — this is the harmonic layer (HPSS) remaining consistent while the percussive layer changes. We need to:
1. Have user annotate feelings on segments (tap-to-annotate tool built)
2. Correlate annotations with computed features
3. Build personalized mapping rules
4. Validate with user on LEDs

### Discovery: Feelings Map to Derivatives, Not Absolute Values
Comparing two regions of Opiate with identical vocal phrases ("like meeeeee" x4):
- **Build region** (28.9-33.5s): User tapped as "airy" — anticipation, preparing for what comes next
- **Climax region** (34-40s): User did NOT tap — feels like arrival/culmination

Static features are nearly identical (RMS ±0.5%, centroid ±4.8%, harmonic ratio ±3.1%). But trajectories diverge massively:
- **Centroid trajectory**: Build +0.04 Hz/frame, Climax +2.11 Hz/frame (58x faster brightening)
- **Harmonic trajectory**: Build becoming more tonal (+), Climax becoming more percussive (−) — direction reversal
- **Bass energy drops 69%** from build to climax; **mids rise 64%**

**General principle**: Human perception of "build" vs "arrival" tracks rate-of-change (derivatives), not absolute position. Two moments at the same loudness/brightness feel completely different if one is accelerating and the other is steady. LED effects must track derivatives of features, not just features themselves.

### Discovery: "Airiness" Is Deviation From Local Context, Not Absolute Acoustic Properties
Two air-tapped clusters in Opiate are acoustically opposite:
- Cluster 1 (guitar trail-off): dark (575 Hz centroid), quiet, sparse, highly tonal
- Cluster 2 (vocal build): bright (4018 Hz), loud, dense, more percussive

Both feel "airy" because both **deviate from surrounding music**. The common thread is *difference from context*, not any fixed feature value. For LED mapping, this means a deviation-from-running-average detector rather than fixed thresholds.

User's insight on airiness function: "a way to build excitement, or to transition you to another space, ready to accept and engage with what will come." Airiness = novelty + isolation OR anticipation + sustained texture. Different acoustic forms, same narrative function.

### Discovery: User Taps Track Bass Peaks, Not Onsets (and Switch Modes)
100% of beat taps land within 50ms of a bass energy peak (19ms median). Only 48.5% align with librosa-detected onsets. Onset-snapping would be wrong.

User switches between tracking modes across song sections:
- **Groove sections**: Taps align with spectral flux peaks (overall energy changes)
- **Intro/fills**: Taps align with centroid peaks (timbral brightness changes)
- User's "changes" layer markers capture exactly where these switches occur

### Discovery: All Algorithmic Beat Trackers Fail on Dense Rock
Tested against user taps on Opiate (steady groove sections, 8-24s):
| Approach | Detected BPM | F1 Score |
|----------|-------------|----------|
| librosa default | 161.5 (doubled) | 0.486 |
| librosa constrained (start_bpm=105) | 161.5 (constraint ignored!) | 0.486 |
| librosa bass-only onset | 110.0 (closest) | 0.500 |
| Custom spectral flux | 82.0 (undershot) | 0.327 |
| madmom RNN | Could not install | — |

Best algorithmic F1 = 0.500. User taps are the most accurate beat tracking available for this music.

### Research Conclusions
- **Build custom** on existing nebula streaming architecture
- **Use librosa** as primary analysis library (comprehensive, extensible)
- **Don't fork** scottlawsonbc/LedFx/WLED — our pipeline is cleaner
- **Steal algorithms** (mel-scale, ExpFilter with asymmetric attack/decay, gaussian smoothing) but not architecture
- **Key constants from research**: attack α=0.7-0.99, decay α=0.1-0.3, gamma 2.2, mel bins 24-64, latency <50ms

### Tools Built
| Tool | Purpose |
|------|---------|
| `record_segment.py` | Capture audio from BlackHole, save WAV + catalog metadata |
| `playback_segment.py` | Play segments to speakers or back to BlackHole |
| `visualize_segment.py` | 5-panel static analysis (mel spectrogram, bands, onset, centroid) |
| `playback_visualizer.py` | Same 5 panels with synced audio playback + cursor + click-to-seek |
| `annotate_segment.py` | Tap-to-annotate: record feeling-based timestamps by layer |
