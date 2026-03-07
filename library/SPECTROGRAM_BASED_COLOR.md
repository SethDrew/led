# Spectrogram-Based Color Mapping

Strategies for mapping mel spectrograms and MFCCs to LED color, organized by mapping philosophy.

## Core Strategies

### 1. Whole-Strip Uniform Color
Reduce the mel vector to a single scalar (weighted centroid, dominant bin, or vector norm), index into a color gradient, paint the whole strip. Simple but surprisingly expressive — you see the *character* of the sound as color temperature.

### 2. Frequency-to-Position
Each LED = one mel bin (or interpolated range). Intensity at that bin → position on a color gradient. This is a physical spectrogram — low frequencies at the base, highs at the tip. The strip becomes a living EQ display.

### 3. Scrolling History (Waterfall)
Current frame enters at one end, old frames scroll away. Each pixel's color = that frame's spectral character. You get a spatial time-series — patterns like verse/chorus become visible as repeating color bands.

### 4. MFCC → RGB Directly
MFCCs are decorrelated by design. Map MFCC1 → R, MFCC2 → G, MFCC3 → B (after normalizing). No gradient needed — the *timbre itself* becomes the color. Spoken word, synth pads, and drums would each get a distinct color fingerprint without any hand-tuning.

### 5. Spectral Centroid + Bandwidth
Centroid picks the *hue* on the gradient. Bandwidth (spectral spread) controls how many LEDs light up — narrow tones light a focused point, broad noise fills the whole strip. Combines frequency and texture in one mapping.

### 6. Dominant Bin Winner-Takes-All
Find the peak mel bin, map its index to hue, its intensity to brightness. Only the loudest frequency matters. Creates dramatic color shifts on melodic content — every note change = color change.

### 7. Difference/Novelty Coloring
Color based on *change* from the previous frame, not the frame itself. Frame-to-frame spectral flux maps to a warm↔cool gradient. Static drones are dim/cool; transients and note changes flash bright/warm. Highlights musical events.

### 8. Dual-Axis 2D Colormap
Position on strip = frequency, but use a 2D colormap (like `turbo` or `inferno`) where one axis is frequency position and the other is intensity. A quiet high note and a loud low note get completely different colors rather than just dim/bright versions of the same hue.

### 9. Harmonic Richness Gradient
Compute spectral flatness per-frame. Pure tones → monochromatic (single hue, high saturation). Noisy/complex spectra → rainbow spread across the strip. The harmonic complexity of the sound controls the *diversity* of color on the strip.

### 10. Comet Waterfall
Like scrolling history but with exponential brightness decay. New frame enters bright at one end, fades as it scrolls. Creates comet-tail trails. Rhythmic content produces pulsing waves that travel down the strip.

## Web Viewer Visualization Approaches

- **Interactive spectrogram with swappable colormaps** — render the mel/MFCC spectrogram on canvas with user-selectable gradients (turbo, inferno, magma, viridis, custom OKLCH palettes)
- **Simulated LED strip** — a horizontal row of circles below the spectrogram, showing what each mapping strategy would look like on hardware in real-time as the playhead moves
- **Side-by-side compare** — show 3-4 mapping strategies simultaneously for visual A/B comparison
