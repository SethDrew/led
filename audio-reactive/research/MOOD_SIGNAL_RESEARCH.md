# MOOD Signal Research: Beyond Spectral Centroid

Research into what audio features can fill the MOOD input role — the slow-moving qualitative character signal that shifts color temperature, palette, warmth/coolness, tension/relaxation in LED effects.

**Date:** 2026-03-02
**Status:** Research complete, awaiting review
**Context:** The MOOD role (from `architecture/AUTOMATED-LIGHTSHOW.md`) controls qualitative character — warm/cool, tense/relaxed, major/minor. Currently filled by spectral centroid alone (`centroid_color.py`, `feature_computer.py`). This document proposes concrete alternatives and improvements.

---

## 1. What Centroid Does Well

Spectral centroid is the "center of mass" of the frequency spectrum — a single scalar that tracks where the energy is concentrated. It has a robust psychoacoustic connection to the perception of **brightness**.

**Strengths of the current implementation (`centroid_color.py`):**

- Proven VJ prior art: spectral centroid to color temperature is the single most validated audio-visual correspondence across scottlawsonbc, LedFx, and WLED (ARCHITECTURE.md Axis 7)
- Real-time computable: one FFT, one weighted mean — trivial computational cost
- Context-adaptive: the effect uses slow EMA to normalize centroid within a running range, so the mapping adapts per-song
- Perceptually intuitive: bright sounds look cool/blue, warm/bass-heavy sounds look red/orange
- ESP32 feasible: just a weighted sum over FFT bins

## 2. Where Centroid Falls Short

### 2.1 Single-Dimensional Collapse

Centroid reduces the entire spectral shape to one number — "where is the center?" This loses critical information:

- **Two spectra with identical centroid but different character**: A sine wave at 2kHz and white noise filtered to have its centroid at 2kHz have the same centroid value but sound completely different. One is pure and focused, the other is diffuse and rough.
- **Symmetry blindness**: Centroid can't distinguish between "energy spread broadly" vs "energy concentrated at a point." A violin and a synth pad can have the same centroid but radically different timbral character.

### 2.2 Narrow Effective Range

From ledger entry `centroid-position-clustering`: the observed centroid range in real music is much narrower than the theoretical range. Even with adaptive normalization, centroid tends to cluster around mid-values for most musical content. In a 200-10000 Hz log mapping, typical pop/electronic music centroid stays between 800-4000 Hz. The warm/cool palette rarely reaches its extremes, making color changes subtle to the point of imperceptibility.

### 2.3 Insensitivity to Harmonic Structure

Centroid ignores whether the spectral energy is **harmonic** (pitched, tonal) or **inharmonic** (noisy, percussive). A bright harmonic synth and a bright crash cymbal can have similar centroids but evoke completely different moods. The harmonic/inharmonic distinction is perceptually one of the most salient timbral properties.

### 2.4 No Valence Information

Research consistently finds that arousal (energy/intensity) is well-predicted by temporal and energy features (RMS, onset rate, tempo), but **valence** (positive/negative, happy/sad) requires **harmonic and tonal features** — key, mode, chord strength, dissonance. Centroid captures brightness, which correlates with arousal but barely correlates with valence. The MOOD role description explicitly includes "major/minor" — centroid cannot address this.

### 2.5 Frame-Level Noise

Raw per-frame centroid is noisy. The current implementation uses ~7s EMA smoothing, which helps but creates a tradeoff: responsive enough to track filter sweeps vs. smooth enough for stable color. A different signal architecture could provide stability without sacrificing responsiveness.

---

## 3. Proposed Alternative Signals

Each proposal includes: what it measures, why it's better than centroid alone, what timescale it operates at, computational cost, and ESP32 feasibility.

### 3.1 Spectral Contrast (Peak-Valley Ratio per Sub-Band)

**What it measures:** The difference between spectral peaks and spectral valleys within each frequency sub-band (typically 6-7 bands). High contrast = clear, tonal, narrowband signals. Low contrast = broad, noisy, diffuse signals.

**Why it's better than centroid:** Spectral contrast captures **texture and tonal quality** — the peak-valley dynamic that centroid completely ignores. Research shows spectral contrast outperforms centroid for mood classification tasks (lower RMSE in valence prediction). It provides 6-7 values per frame instead of 1, giving a multi-dimensional timbral fingerprint.

**Perceptual mapping:**
- High contrast in low bands + low in high bands = warm, full, grounded (bass instruments dominating)
- High contrast across all bands = clean, defined, focused (well-mixed mastered music, clear mix)
- Low contrast across all bands = washy, ambient, reverberant (atmospheric pads, ambient sections)
- The ratio of low-band contrast to high-band contrast could serve as a "warmth" axis

**Computation:**
```
For each sub-band (6 mel-spaced bands):
  Sort FFT bins within the band by magnitude
  peak = mean of top quantile (top 20%)
  valley = mean of bottom quantile (bottom 20%)
  contrast = peak - valley (in log domain)
```

**Timescale:** Frame-level (per FFT frame), but should be smoothed over ~2-5s for MOOD role.

**ESP32 feasibility:** YES. Same FFT already needed for centroid. Sorting within sub-bands adds O(n log n) per band, but bands are small (10-30 bins each). Alternatively, approximate with max/min instead of quantile.

**Integration with centroid:** Spectral contrast is *orthogonal* to centroid. Centroid says "where is the energy?" Contrast says "how focused is the energy?" Together they span a 2D mood space: bright-focused vs bright-diffuse vs warm-focused vs warm-diffuse.

---

### 3.2 Spectral Slope (Spectral Tilt)

**What it measures:** The linear regression slope of the log-magnitude spectrum from low to high frequencies. Negative slope (energy decreases with frequency) = warm, muffled. Flatter slope = bright, open. Positive slope (unusual) = thin, harsh.

**Why it's better than centroid:** Spectral slope captures the *overall shape* of the spectrum rather than just its center of mass. It's more robust to narrow-band peaks (a single loud instrument doesn't skew it as much). It correlates directly with the psychoacoustic perception of "warmth" vs "brightness" as a tilt rather than a point.

**Perceptual mapping:**
- Steep negative slope: warm, muffled, dark (lo-fi, bass-heavy, subwoofer content)
- Moderate negative slope: natural, balanced (most acoustic music)
- Flat slope: bright, open, airy (hi-hats, cymbals, noise-rich content)
- Spectral slope *derivative* (d/dt of slope): filter sweeps! A rising slope = filter opening (brightness increasing). A falling slope = filter closing. This directly detects the DJ filter sweep technique mentioned in the ledger.

**Computation:**
```
freqs = log(fft_frequencies)
mags = log(fft_magnitudes + eps)
slope = linear_regression_slope(freqs, mags)
```

**Timescale:** Frame-level, smooth over ~1-3s for MOOD. The derivative of slope has beat/phrase-level utility for detecting filter sweeps.

**ESP32 feasibility:** YES. One linear regression over ~100 FFT bins is trivial (sum of products, sum of squares — 4 accumulators).

**Advantage over centroid:** Spectral slope is fundamentally more stable than centroid because it's a global fit rather than a local mean. A single prominent harmonic won't shift the slope as much as it shifts the centroid. This means less smoothing needed, more responsive.

---

### 3.3 Harmonic-to-Noise Ratio (HNR) / Harmonic Ratio

**What it measures:** The proportion of the signal's energy that is harmonic (pitched, periodic) vs. inharmonic (noisy, aperiodic). High HNR = tonal, sustained, melodic. Low HNR = noisy, percussive, textured.

**Why it's better than centroid:** HNR captures a dimension of timbral character that centroid is completely blind to: **tonality vs. noise**. A high-HNR section (strings, pads, vocals) and a low-HNR section (drums, atmospheric effects, textured synths) should evoke different visual moods even if they have identical spectral centroids.

**Perceptual mapping:**
- High HNR: smooth, singing, lyrical, focused → visually: flowing, coherent, luminous
- Low HNR: rough, breathy, textured, diffuse → visually: sparkly, grainy, organic noise
- HNR derivative: transition from harmonic to noisy (breakdown → drop with noise sweeps) or noisy to harmonic (intro evolving into melodic section)

**Computation:**
Already have HPSS in the project (`band_zone_pulse.py` uses streaming HPSS). The ratio of harmonic energy to percussive energy IS the harmonic ratio:
```
H, P = HPSS(spectrogram)
harmonic_ratio = sum(H) / (sum(H) + sum(P) + eps)
```
Or simpler: autocorrelation peak height of the audio frame. A strongly periodic signal has a high autocorrelation peak.

**Timescale:** Frame-level computation, smooth over 2-5s for MOOD.

**ESP32 feasibility:** MAYBE. Full HPSS requires a spectrogram buffer (memory intensive). But a simple autocorrelation-based HNR on a single frame is feasible. Alternative: use the already-computed autocorrelation confidence from `FeatureComputer` as a rough proxy for harmonicity.

**Key insight:** The existing `autocorr_conf` feature in `feature_computer.py` already partially measures this! High autocorrelation confidence = strongly periodic = high HNR. But `autocorr_conf` is computed on the abs-integral signal (RMS derivative), not the audio itself. Computing it on the raw audio waveform would give a more direct harmonic ratio.

---

### 3.4 Spectral Spread (Bandwidth)

**What it measures:** The standard deviation of the spectrum around the spectral centroid. Wide spread = energy distributed broadly across frequencies. Narrow spread = energy concentrated near the centroid.

**Why it's better than centroid:** Spectral spread is the *complement* to centroid — literature recommends always using them together. While centroid says "where is the center?", spread says "how wide is the distribution?" A narrow-spread signal at 2kHz (a pure tone) and a wide-spread signal at 2kHz (broadband noise centered at 2kHz) have the same centroid but completely different character.

**Perceptual mapping:**
- Narrow spread: focused, clear, pure, intimate → single-instrument solo, close-mic vocal
- Wide spread: diffuse, rich, full, immersive → orchestra, full band, dense mix, reverb
- Spread *relative to centroid*: this is already available as "bandwidth" and directly maps to the perceptual quality of "fullness" vs. "thinness"

**Computation:**
```
centroid = sum(freqs * spectrum) / sum(spectrum)
spread = sqrt(sum((freqs - centroid)^2 * spectrum) / sum(spectrum))
```

**Timescale:** Frame-level, smooth over 2-5s for MOOD.

**ESP32 feasibility:** YES. Same FFT, one additional pass for variance computation (or computed alongside centroid in the same loop).

---

### 3.5 Chroma Entropy (Harmonic Complexity)

**What it measures:** The Shannon entropy of the chroma (pitch class) distribution. Low entropy = one or two dominant pitch classes (clear key, simple harmony). High entropy = energy spread across many pitch classes (complex chords, atonality, multiple instruments in different keys).

**Why it's better than centroid:** Chroma entropy addresses the **valence gap** — the dimension of MOOD that centroid cannot reach. Research identifies HPCP entropy (closely related) as one of the most important features for valence prediction. Simple harmony (low entropy) tends to feel resolved, calm, or cheerful. Complex harmony (high entropy) tends to feel tense, uncertain, or dark.

**Perceptual mapping:**
- Low chroma entropy: calm, resolved, clear, stable → major key sections, simple melodies
- High chroma entropy: tense, complex, uncertain, rich → jazz chords, dense orchestration, chromatic passages, dissonant buildup
- Entropy derivative: increasing entropy signals growing tension (builds), decreasing signals resolution (drops, choruses that simplify to a hook)

**Computation:**
```
chroma = chromagram(spectrum)  # 12 pitch classes
chroma_norm = chroma / sum(chroma)  # probability distribution
entropy = -sum(chroma_norm * log(chroma_norm + eps))  # Shannon entropy
# Max entropy = log(12) = 2.485 for uniform distribution
# Normalize: entropy / log(12) → 0-1
```

**Timescale:** Frame-level, but best smoothed over 2-8s (chord changes happen at bar level).

**ESP32 feasibility:** MAYBE. Computing a chromagram requires mapping FFT bins to pitch classes. Not as cheap as centroid, but feasible with a pre-computed filter bank. The entropy calculation itself is trivial (12 multiplies + 12 logs).

---

### 3.6 Sensory Dissonance

**What it measures:** The total roughness/beating perceived when multiple frequencies are close together within a critical bandwidth. Based on Plomp-Levelt dissonance curves. High dissonance = tense, harsh, grating. Low dissonance = consonant, smooth, pleasant.

**Why it's better than centroid:** Dissonance is one of the strongest predictors of perceived musical tension, which is at the core of what MOOD should track. It captures something fundamentally different from spectral shape — the *interaction* between simultaneous frequencies, not just their distribution.

**Perceptual mapping:**
- Low dissonance: smooth, consonant, relaxed → octaves, fifths, simple intervals
- High dissonance: tense, rough, aggressive → cluster chords, distortion, dense builds
- Dissonance *trajectory* (d/dt over 5-10s): building dissonance = tension building, resolving dissonance = release

**Computation:**
```
peaks = find_spectral_peaks(spectrum, N=20)
dissonance = 0
for each pair (f1, f2, a1, a2) in peaks:
    s = abs(f2 - f1) / (0.24 * (f1 + f2) / 2 + 0.0207)  # critical bandwidth scaling
    dissonance += a1 * a2 * (exp(-3.5 * s) - exp(-5.75 * s))  # Plomp-Levelt curve
```

**Timescale:** Frame-level, smooth over 2-5s for MOOD.

**ESP32 feasibility:** CHALLENGING. Finding spectral peaks and computing N^2 pairwise distances is expensive. Could limit to top 10 peaks (45 pairs) which is manageable. Essentia implements this efficiently in C++.

---

### 3.7 Band Energy Ratio (Low-to-High Energy Balance)

**What it measures:** The ratio of energy in low frequency bands to high frequency bands. A simpler, more robust alternative to spectral centroid that directly measures the bass/treble balance.

**Why it's better than centroid:** Band energy ratio is more stable and interpretable than centroid. A loud bass note shifting centroid down by 200Hz is hard to interpret. A 3dB shift in bass/treble ratio is immediately meaningful. Research finds that the ratio of high-frequency to low-frequency energy is one of the top 3 predictors of arousal-valence dimensions (predicting 50-57% of variance).

**Perceptual mapping:**
- High bass ratio: warm, heavy, grounded, powerful
- High treble ratio: bright, airy, delicate, sparkly
- Mid-heavy: present, forward, vocal-focused
- Could use 3 ratios (bass/mid, mid/high, bass/high) for a 3D warmth space

**Computation:**
```
bass_energy = sum(spectrum[20-250 Hz]^2)
mid_energy = sum(spectrum[250-4000 Hz]^2)
high_energy = sum(spectrum[4000-16000 Hz]^2)
total = bass_energy + mid_energy + high_energy + eps
bass_ratio = bass_energy / total
mid_ratio = mid_energy / total
high_ratio = high_energy / total
# Or simpler: low_high_ratio = bass_energy / (high_energy + eps)
```

**Timescale:** Frame-level, smooth over 2-5s.

**ESP32 feasibility:** YES. Trivially cheap — sum FFT bins in three ranges. Already essentially done in the per-band energy computation.

---

## 4. The Multi-Dimensional MOOD Proposal

Rather than replacing centroid with a single alternative, the research points toward a **multi-dimensional MOOD vector**. The MOOD role should output 2-4 dimensions, not 1.

### 4.1 Proposed MOOD Dimensions

| Dimension | What It Captures | Primary Signal | Secondary Signal | Perceptual Axis |
|---|---|---|---|---|
| **Brightness** | Warm ↔ Cool | Spectral slope | Spectral centroid | Color temperature |
| **Texture** | Smooth ↔ Rough | Harmonic ratio (HNR) | Spectral flatness | Visual coherence vs. grain |
| **Tension** | Relaxed ↔ Tense | Chroma entropy | Sensory dissonance | Palette saturation, contrast |
| **Fullness** | Thin ↔ Full | Spectral spread | Band energy ratio | Visual density, bloom |

### 4.2 Why These Four

These four dimensions map to the two axes of the circumplex model (Russell, 1980) that the MIR community has validated:

- **Arousal** (energy/activation) is already served by INTENSITY (RMS, absint). Not MOOD's job.
- **Valence** (positive/negative) is the hard one. Research shows it requires harmonic/tonal features. Our **Tension** dimension (chroma entropy + dissonance) approximates this.
- **Brightness** and **Fullness** are orthogonal timbral qualities that shape the "character" of the sound — exactly what MOOD should capture.
- **Texture** (harmonic ratio) distinguishes smooth melodic sections from rough percussive ones — a distinction invisible to centroid.

### 4.3 Visual Mapping

Each dimension maps to a distinct visual parameter:

| MOOD Dimension | Visual Parameter | Example |
|---|---|---|
| Brightness (warm↔cool) | Color temperature | Red-orange ↔ blue-white (same as current centroid_color) |
| Texture (smooth↔rough) | Visual coherence | Smooth gradients ↔ sparkly/grainy noise |
| Tension (relaxed↔tense) | Color saturation + contrast | Muted pastels ↔ vivid saturated colors |
| Fullness (thin↔full) | Visual density / bloom | Sparse dim ↔ dense glowing |

### 4.4 Dimensionality for Different Contexts

Not every effect needs all four dimensions. The role system handles this:

- **Simple effects** (ambient glow, breathing): bind MOOD.brightness only
- **Medium complexity** (noise field, plasma): bind MOOD.brightness + MOOD.texture
- **Rich effects** (seasonal cycle, reaction-diffusion): bind all four

---

## 5. Implementation Priority

Ordered by impact/effort ratio and alignment with existing code:

### Phase 1: Quick Wins (use existing FFT, minutes to implement)

1. **Spectral slope** — one linear regression, more stable than centroid, detects filter sweeps
2. **Band energy ratio** — three sums of existing FFT bins, more interpretable than centroid
3. **Spectral spread** — one additional variance pass alongside centroid computation

### Phase 2: Moderate Effort (need additional computation)

4. **Spectral contrast** — needs sub-band sorting or quantile estimation, 6-7 values per frame
5. **Harmonic ratio from HPSS** — already have streaming HPSS in `band_zone_pulse.py`, need to extract the ratio as a scalar

### Phase 3: Harder / Future

6. **Chroma entropy** — needs chromagram computation (pitch-class filter bank)
7. **Sensory dissonance** — needs peak finding + pairwise computation, O(N^2)

### ESP32 Feasibility Summary

| Signal | ESP32 Feasible? | Extra Cost Beyond Existing FFT |
|---|---|---|
| Spectral slope | YES | ~200 multiplies (linear regression) |
| Band energy ratio | YES | ~3 partial sums (already have per-band) |
| Spectral spread | YES | ~100 multiplies (variance) |
| Spectral contrast | YES | Sort within 6 bands (~30 bins each) |
| Harmonic ratio | MAYBE | Autocorrelation on audio frame, or simplify HPSS |
| Chroma entropy | MAYBE | 12-bin filter bank + 12 logs |
| Sensory dissonance | HARD | Peak finding + N^2 pairwise, limit to 10 peaks |

---

## 6. What This Means for `centroid_color.py`

The current `centroid_color` effect maps centroid → color temperature, RMS → brightness. With multi-dimensional MOOD, this would become:

```
spectral_slope → color temperature (replaces centroid — more stable, wider range)
harmonic_ratio → visual coherence (smooth gradient vs. sparkle overlay)
chroma_entropy → saturation (relaxed = pastel, tense = vivid)
spectral_spread → glow radius (thin = tight, full = blooming)
RMS → brightness (unchanged)
```

The effect would become richer without becoming more complex — each dimension independently modulates one visual parameter.

---

## 7. Research Sources

- Russell's circumplex model of affect (1980) — the valence-arousal framework that music emotion research uses
- Eerola & Vuoskoski — audio features for valence/arousal prediction; found that harmonic/tonal features are principal drivers of valence, while temporal/timbral-flux features drive arousal
- Jiang et al. (2002) "Music type classification by spectral contrast feature" — demonstrated spectral contrast's superiority over centroid for classification tasks
- Essentia library (MTG/UPF) — implements sensory dissonance, spectral contrast, HNR, and other timbral features in C++
- librosa — spectral_contrast, spectral_bandwidth, chroma features
- McAdams (2019) "Timbre as a Structuring Force in Music" — timbral dimensions: brightness, roughness, attack time, spectral flux
- Alluri & Toiviainen — found 50-57% of valence/arousal variance predicted by ratio of high/low frequency energy, attack slope, spectral regularity
- Spotify audio features — valence computation uses undisclosed combination of spectral, harmonic, and tonal features; validates that single-scalar mood requires multi-feature input
- Project ledger entries: `centroid-position-clustering`, `feelings-are-contextual`, `airiness-context-deviation`, `derivatives-over-absolutes`
