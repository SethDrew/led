# Divisive Normalization for Audio-Reactive Visualization

## Problem

Audio-reactive LED effects need a real-time normalizer that produces usable 0-1 values from raw audio features (RMS energy, band energy, spectral derivatives). The dominant approach in the LED/VJ ecosystem is peak-decay normalization: track a running maximum that decays exponentially, divide the signal by it. This works for steady-state music but fails for section-aware dynamics. When a loud drop sets the peak, subsequent quieter sections are crushed toward zero because the denominator remembers the loudest moment. Section differences — the contrast between a breakdown and a drop, between a verse and a chorus — are erased. The normalizer preserves beat-level detail within the current loudness regime but destroys the macro structure that makes music feel like it has shape.

## Prior work

**Peak-decay normalization** is ubiquitous. WLED Sound Reactive [1], scottlawsonbc's audio-reactive-led-strip [2], and FastLED community projects all use variants of `peak = max(value, peak * decay)`. Tuline's FastLED-SoundReactive [1] initially implemented mean-tracking normalization (`multAgc = targetAgc / sampleAvg`), but this was later replaced by peak-tracking in the WLED codebase. Decay constants vary (0.998-0.9999) but the architecture is the same: instant attack, slow exponential release.

**EBU R128** [3] replaced peak normalization with integrated mean-loudness normalization for broadcast audio, recognizing that peak-based leveling produces inconsistent perceived loudness across programs. The shift from peak to mean as the reference signal is the same conceptual move we make here, applied to a different domain.

**PCEN** (Per-Channel Energy Normalization) [4] applies `E(t) / M(t)^alpha` where `M(t)` is a first-order IIR filter (exponential moving average) of the energy. The core step — dividing by the running mean rather than the running max — is identical to our approach. Lostanlen and Salamon show this outperforms static normalization for sound event detection because it adapts to local context.

**Divisive normalization** is a canonical computation in biological neural circuits [5]. Neurons divide their response by a weighted sum of nearby activity, producing responses that adapt to local statistical context. This is the same operation: the denominator tracks context, the quotient preserves local contrast relative to that context.

**Auditory adaptation** in biological hearing uses the same principle at a systems level. Dean, Harper, and McAlpine [6] showed that auditory neurons shift their rate-level functions to match the local mean of stimulus intensity, maintaining sensitivity across a wide dynamic range. The adaptation time constants (seconds to tens of seconds) match the range we find optimal.

## Our approach

We tested 13 normalization algorithms on electronic music segments with clear section structure (breakdowns, builds, drops), evaluating two independent objectives: intra-section coefficient of variation (beat visibility within sections) and inter-section range (contrast between section means). The joint metric is their product.

### Finding 1: Mean-tracking replaces peak-tracking

EMA normalization — dividing the signal by its exponential moving average — outperforms peak-decay for section-aware dynamics. The core difference is what the denominator tracks:

- **Peak-decay**: denominator is dominated by the loudest moment encountered. After a loud drop, quieter sections divide by a slowly-decaying large number. The output compresses toward zero and stays there until the peak decays enough or a louder moment arrives.
- **EMA (mean-tracking)**: denominator tracks the current energy context. When music moves from a loud drop to a quiet breakdown, the EMA adapts downward, allowing the breakdown's internal dynamics to fill the 0-1 range. Loud sections still read louder than quiet sections (the EMA is higher, so the ratio is lower for the same absolute energy), preserving inter-section contrast.

The EMA ratio produces values centered around 1.0 (since the signal oscillates around its own mean). A `/2` rescaling maps this to approximately 0-1 with headroom for transients. This is simpler than peak-decay's cold-start and warm-start problems: EMA has no concept of a "peak" that can get stuck.

### Finding 2: Time constant is section-relative

The optimal EMA time constant is approximately 0.5-0.7x the typical section length of the music:

| Genre | Typical section | Optimal EMA time constant |
|-------|----------------|--------------------------|
| EDM (4-bar phrases) | 30-40s | 20s |
| Rock/pop | 45-60s | 30-40s |
| DJ sets (long blends) | 90-120s | 60-90s |

Too short (< 0.3x section length): the EMA tracks within-section dynamics too closely, erasing beat contrast. The denominator follows the signal so tightly that the ratio becomes nearly constant.

Too long (> 1x section length): the EMA behaves like peak-decay, remembering previous sections and crushing current dynamics.

The frame-rate-independent formula is: `alpha = 2.0 / (tc_seconds * fps + 1)`, where `tc_seconds` is the time constant in seconds. This is the standard EMA smoothing factor that gives consistent behavior regardless of frame rate (tested at 43-86 fps).

### Finding 3: Section boundary behavior is a feature

EMA's adaptation lag at section transitions is not a deficiency — it is the section contrast signal itself.

When a breakdown drops to 20% energy after a loud section, the EMA denominator is still high (it was tracking the loud section). The quotient drops sharply — producing a visible dip in normalized output. Over several seconds, the EMA adapts downward and the breakdown's internal beats become visible. This mirrors human auditory adaptation: after a loud passage, quiet sounds initially seem very quiet, then gradually become audible as the ear re-sensitizes [6].

When a build climbs into a drop, the EMA is still tracking the build's lower energy. The first moments of the drop produce a spike — a transient overshoot where the signal exceeds the denominator. This is the "drop hit" that should flash the LEDs.

Peak-decay's instant attack absorbs these transitions in a single frame. The peak jumps to the new maximum, the ratio returns to near-1.0, and the transition is invisible. Peak-decay is optimized for steady-state accuracy; EMA is optimized for transition expressiveness.

## Validation

Tested on 3 electronic music segments: Fred Again.. "Billie (Loving Arms)" drop section (clear build-drop-breakdown structure), Four Tet "Lostvillage Screech" (sustained texture with dynamic layering), and a complex beat collage with irregular section boundaries. Two additional tracks from a different artist (Nathaniel) confirmed generalization within electronic music.

Signals tested: RMS energy, absolute integral (spectral flux), 5-band mel energy (sub-bass through treble), rolling RMS integral. All showed the same pattern: EMA-ratio preserves both intra-section CV and inter-section range where peak-decay preserves only intra-section CV.

## Caveats

- **Metric bias.** The joint metric (intra-CV x inter-range) has demonstrated bias toward faster-adapting algorithms. An algorithm that adapts instantly to each section would score perfectly on inter-range but produce constant output within sections. The metric captures this tradeoff imperfectly.
- **Perceptual validation pending.** All comparisons are quantitative (waveform analysis). Perceptual validation — human observers rating LED output on actual hardware — has not yet been performed. The metric is a proxy for visual quality, not a measurement of it.
- **Genre generalization.** Tested primarily on electronic music with clear section structure. Rock, jazz, classical, and ambient music have different section dynamics and may require different time constants or different approaches entirely.
- **Hybrid architectures unexplored.** The G1-EMA hybrid (EMA for local dynamics, linear-decay peak for section position, combined via power-law compression) showed promise in testing but adds complexity. Whether the EMA-ratio alone is sufficient or whether the hybrid is needed for certain signals remains open.

## References

[1] Tuline, A. et al. WLED Sound Reactive / FastLED-SoundReactive. GitHub. https://github.com/atuline/WLED. Originally implemented `multAgc = targetAgc / sampleAvg` (mean-tracking); later releases moved to peak-tracking AGC.

[2] S. Lawson. audio-reactive-led-strip. GitHub. https://github.com/scottlawsonbc/audio-reactive-led-strip. Per-filterbank asymmetric EMA normalization.

[3] European Broadcasting Union. "Loudness normalisation and permitted maximum level of audio signals." EBU R 128, 2011. Replaced peak normalization with integrated mean-loudness (LUFS) for broadcast leveling.

[4] Lostanlen, V. & Salamon, J. "Per-Channel Energy Normalization: Why and How." IEEE Signal Processing Letters, 26(1), 39-43, 2019. Core operation: divide energy by its exponential moving average, then apply gain and offset.

[5] Carandini, M. & Heeger, D.J. "Normalization as a canonical neural computation." Nature Reviews Neuroscience, 13(1), 51-62, 2012. Divisive normalization as a unifying principle across sensory systems.

[6] Dean, I., Harper, N.S. & McAlpine, D. "Neural population coding of sound level adapts to stimulus statistics." Nature Neuroscience, 8(12), 1684-1689, 2005. Auditory neurons shift rate-level functions to match local mean intensity; adaptation time constants of seconds to tens of seconds.
