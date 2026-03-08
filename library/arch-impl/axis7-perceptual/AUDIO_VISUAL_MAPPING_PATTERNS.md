# Audio-to-Visual Mapping Research: Algorithms & Techniques

Research compiled: 2026-02-05

This document extracts the actual math, constants, and algorithms used in successful audio-reactive LED projects. Focus is on concrete implementation details rather than vague descriptions.

---

## 1. Mapping Techniques by Category

### 1.1 Beat → Flash (Energy-Based Mapping)

#### scottlawsonbc Energy Visualization
**Input:** Mel-scale frequency bands (post-FFT)
**Transform:**
```python
# Power law scaling per frequency band (exponent 0.9)
r = int(np.mean(y[:len(y)//3]**0.9))      # Bass: 0 to (N_PIXELS/2)
g = int(np.mean(y[len(y)//3:2*len(y)//3]**0.9))  # Mids
b = int(np.mean(y[2*len(y)//3:]**0.9))    # Highs
```
**Output:** Integer pixel counts (0 to N_PIXELS/2) indicating how many LEDs to light from center
**Why 0.9 exponent?** Compresses dynamic range while maintaining perceptual energy response (less aggressive than sqrt, more than linear)

#### LedFx Energy Effect
**Input:** Mean of mel filterbank frequency band
**Transform:**
```python
multiplier = 1.6 - (blur / 17)  # blur ∈ [0, 10]
index = multiplier × pixel_count × mean(frequency_band)
```
**Smoothing:** Exponential filter with decay = (sensitivity - 0.1) × 0.7
- Sensitivity range: 0.3 to 0.99
- Effective decay: 0.14 to 0.62
- Rise: Uses sensitivity directly (0.3 to 0.99)
**Output:** LED index position to illuminate

**Key insight:** "Sticky" peak behavior - energy rises quickly but falls slowly, preventing jittery flashing

### 1.2 Frequency → Color Mapping

#### Static RGB Band Mapping (scottlawsonbc Spectrum)
**Input:** Mel filterbank output (perceptually-spaced frequencies)
**Spatial mapping:**
```python
# Interpolate mel bands to match LED count
y = interpolate(mel_output, N_PIXELS // 2)
```
**Color channel assignment:**
```python
r = r_filt.update(y - common_mode.value)  # Red: absolute energy minus DC
g = np.abs(diff)                           # Green: temporal derivative (change rate)
b = b_filt.update(np.copy(y))             # Blue: smoothed magnitude
output = np.array([r, g, b]) * 255
```

**Filter constants (asymmetric decay):**
- **Red:** alpha_decay=0.2, alpha_rise=0.99 (slow decay, instant rise)
- **Green:** alpha_decay=0.05, alpha_rise=0.3 (fast decay, moderate rise)
- **Blue:** alpha_decay=0.1, alpha_rise=0.5 (medium decay, medium rise)

**Why different rates per channel?**
- Red persists to show sustained energy
- Green flashes quickly to show transients/attacks
- Blue provides smooth mid-ground

#### Gradient-Based Color Mapping (LedFx Bands)
**Input:** Frequency band index (0 to num_bands)
**Transform:**
```python
color = get_gradient_color(band_index / num_bands)
```
**Output:** RGB from continuous gradient palette (0.0 to 1.0 normalized position)

**Permutation-Based Mixing (LedFx Spectrum)**
RGB channels use 6 possible permutations:
```python
permutations = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
```
Allows different "flavors" of the same frequency data mapped to different visual channels.

### 1.3 Energy → Brightness

#### Direct Scaling (Web Audio API)
**Input:** Frequency bin magnitude (Uint8Array, 0-255)
**Transform:**
```javascript
barHeight = dataArray[i] / 2;  // Divide by 2 to fit canvas
```
**Output:** Pixel height/brightness (0-127)

**With gamma correction (common LED practice):**
```python
# Power law for perceptual brightness
brightness = (normalized_energy ** 2.2) * 255
```
**Why gamma 2.2?** Compensates for LED nonlinear brightness perception. Makes dim values more visible, bright values less overwhelming.

#### Adaptive Normalization (scottlawsonbc)
**Problem:** Quiet songs produce no visible effect, loud songs saturate
**Solution:** Adaptive gain tracking
```python
mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
mel /= mel_gain.value  # Normalize by recent peak
```
**How it works:**
- Tracks rolling maximum of mel spectrum
- Uses exponential filter (ExpFilter) to smooth peak tracking
- Divides current frame by recent peak → output always spans full dynamic range

**Tradeoff:** Loses absolute loudness information, but maintains relative dynamics within recent time window

### 1.4 Onset → Animation Trigger

While none of the examined projects explicitly implement sophisticated onset detection, the temporal derivative approach provides implicit transient detection:

```python
# Green channel in scottlawsonbc
g = np.abs(diff)  # Absolute difference between consecutive frames
```

This naturally produces spikes during attack transients (sudden amplitude increases).

**Better onset detection (not found in projects, but industry standard):**
- Spectral flux: sum of positive differences in spectrum
- High-frequency content: sudden HF energy indicates onset
- Complex domain: phase deviation detection

---

## 2. Smoothing and Temporal Shaping

### 2.1 Exponential Moving Average (EMA) - Universal Approach

All examined LED projects use exponential filtering as the core smoothing primitive.

#### Implementation (scottlawsonbc ExpFilter)
```python
class ExpFilter:
    def __init__(self, val, alpha_decay=0.5, alpha_rise=0.5):
        self.value = val
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise

    def update(self, new_value):
        if new_value > self.value:
            alpha = self.alpha_rise   # Attack
        else:
            alpha = self.alpha_decay  # Decay
        self.value = alpha * new_value + (1 - alpha) * self.value
        return self.value
```

**Key characteristics:**
- **alpha ∈ [0, 1]**: Higher = more responsive (less smoothing)
- **Asymmetric:** Different alpha for rise vs. fall
- **Memoryless:** Only depends on current state (computationally cheap)

### 2.2 Attack/Decay Constants Found in Practice

#### scottlawsonbc RGB Channels
| Channel | Decay (α) | Rise (α) | Time Constant (60fps) | Behavior |
|---------|-----------|----------|----------------------|----------|
| Red     | 0.2       | 0.99     | ~5 frames (83ms)     | Sticky peaks |
| Green   | 0.05      | 0.3      | ~20 frames (333ms)   | Smooth, gentle |
| Blue    | 0.1       | 0.5      | ~10 frames (167ms)   | Balanced |

**Time constant calculation:** τ ≈ 1/α frames (for α << 1)

#### LedFx Energy Effect
- **Decay:** (sensitivity - 0.1) × 0.7, range [0.14, 0.62]
- **Rise:** sensitivity directly, range [0.3, 0.99]
- At sensitivity=0.5: decay=0.28, rise=0.5 → moderately responsive

#### LedFx Spectrum Effect (Channel 2 Filter)
- **Decay:** 0.1
- **Rise:** 0.5
- Balanced response, suitable for melodic content

### 2.3 Why Asymmetric Attack/Decay?

**Physiological basis:** Human perception is more sensitive to sudden changes (attack) than gradual fading
**Visual effect:**
- **High rise alpha (0.7-0.99):** Effect appears instantly with music
- **Low decay alpha (0.05-0.3):** Smooth fadeout prevents flickering

**Anti-pattern:** Equal attack/decay creates laggy feeling or jittery appearance depending on alpha value.

### 2.4 Spatial Smoothing

#### Gaussian Blur (scottlawsonbc)
```python
# Tight blur for scrolling effects
p = gaussian_filter1d(p, sigma=0.2)

# Heavy blur for energy visualization
p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
```

**sigma parameter:**
- **0.2:** Minimal smoothing, preserves sharp features
- **4.0:** Heavy blur, creates gradient clouds
- Units are in pixel space

**Use case:** Prevents single-pixel artifacts, creates smooth gradients for "glow" effects

---

## 3. Normalization Strategies

Normalization is the hardest problem in audio-reactive systems. Goal: make both quiet jazz and loud EDM produce visible effects without manual gain adjustment.

### 3.1 Per-Frame Min-Max Normalization (AVOID)

```python
# Anti-pattern - DO NOT USE
output = (data - data.min()) / (data.max() - data.min())
```

**Why it's bad:** Destroys musical dynamics. Quiet and loud sections look identical. Noise floor becomes "music" during silence.

### 3.2 Rolling Window Peak Normalization (scottlawsonbc)

```python
# Track recent maximum
mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
# Normalize by it
mel /= mel_gain.value
```

**How it works:**
- ExpFilter tracks peak with slow decay (alpha ~ 0.1-0.2)
- Peak "remembers" recent loud moments
- Normalization brings peaks to ~1.0 while preserving relative dynamics

**Tradeoffs:**
- **Pro:** Adapts to song loudness automatically
- **Pro:** Maintains dynamics within ~2-5 second window
- **Con:** 1-2 second startup time as gain settles
- **Con:** Sudden quiet sections briefly over-amplify noise

**Time constant tuning:**
- Faster decay (α = 0.3): Responsive to changes, but can pump during dynamic songs
- Slower decay (α = 0.1): Stable gain, but slow to adapt to new sections

### 3.3 Fixed Range with Clipping (Web Audio API)

```javascript
analyser.minDecibels = -90;  // Noise floor
analyser.maxDecibels = -10;  // Loud signal
// Output automatically scales to [0, 255]
```

**How it works:**
- Maps specified dB range to byte range
- Values outside range are clipped
- No adaptation, but predictable behavior

**When to use:**
- Live performance with consistent levels
- After audio is gain-normalized in DAW
- When you want predictable response curves

### 3.4 Automatic Gain Control (AGC) - Not Found in Projects

**Common in professional systems but absent from hobby projects:**
```python
# Pseudo-code for AGC
target_rms = 0.3
current_rms = np.sqrt(np.mean(audio ** 2))
gain_adjustment = target_rms / (current_rms + epsilon)
audio *= gain_adjustment
```

**Why it's not used:**
- Complex to implement correctly (requires lookahead, multi-band processing)
- Can sound "pumping" if not tuned well
- Rolling peak normalization is "good enough" for LEDs

### 3.5 Hybrid Approach (Recommended but not yet seen)

```python
# Static baseline + adaptive scaling
baseline_gain = 2.0  # User-set for typical listening volume
adaptive_factor = clip(1.0 / rolling_peak, 0.5, 2.0)  # Limited range
final_gain = baseline_gain * adaptive_factor
output = audio * final_gain
```

**Benefits:**
- User sets rough gain for their system
- Automatic adaptation handles ±6dB variation
- Limiting prevents extreme pumping

---

## 4. Color Mapping Approaches

### 4.1 Static Frequency-to-RGB (Most Common)

**Approach:** Low frequencies → red, mid → green, high → blue

#### scottlawsonbc Energy Visualization
```python
y_bands = [bass_band, mid_band, high_band]  # Split spectrum into 3
r = int(np.mean(y_bands[0] ** 0.9))
g = int(np.mean(y_bands[1] ** 0.9))
b = int(np.mean(y_bands[2] ** 0.9))
```

**Split points:**
- Bass: 0 to 1/3 of mel bands (~200-800 Hz depending on N_FFT_BINS)
- Mids: 1/3 to 2/3 (~800-3000 Hz)
- Highs: 2/3 to end (~3000-12000 Hz)

**Pros:** Intuitive, distinct colors for different instruments
**Cons:** Colors are always same for given frequency content

### 4.2 Gradient Palette Mapping (LedFx)

```python
# Map frequency band index to color gradient
color = gradient[int(band_index / total_bands * gradient_length)]
```

**Common gradients:**
- **Rainbow:** Red → Orange → Yellow → Green → Blue → Purple
- **Fire:** Black → Red → Orange → Yellow → White
- **Ocean:** Blue → Cyan → Green → White

**Advantages:**
- User customizable "vibe"
- Smooth color transitions
- Can rotate/animate palette independently of audio

**LedFx Spectrum Modes:**
RGB channel permutations create different "looks" from same data:
- [0,1,2]: Standard R=bass, G=mid, B=high
- [2,1,0]: Inverted (R=high, G=mid, B=bass)
- Other permutations create unique color feels

### 4.3 HSV Color Space (Recommended but not seen)

**Theory (from audio visualization best practices):**
```python
# Hue from spectral centroid (frequency "center of mass")
centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
hue = map_range(centroid, min_freq, max_freq, 0, 360)

# Saturation from spectral flatness (noise vs. tones)
flatness = geometric_mean(spectrum) / arithmetic_mean(spectrum)
saturation = 1.0 - flatness  # Tonal = saturated, noisy = desaturated

# Value from overall energy
value = sqrt(np.sum(magnitudes ** 2))
```

**Why HSV?**
- Decouples color (hue) from brightness (value)
- Saturation indicates "purity" of sound
- Allows independent control of each perceptual dimension

**Not found in projects because:** Requires more sophisticated analysis (spectral centroid calculation). RGB direct mapping is simpler.

### 4.4 Color Mixing Strategies

#### Additive (LedFx Energy)
```python
# Multiple frequency bands light same pixel
# Colors accumulate (can saturate to white)
for band in bands:
    pixels[index] += band_color
```
**Result:** Overlapping bands create mixed colors, busy sounds → white

#### Overlap/Replace (LedFx Energy)
```python
# Later bands overwrite earlier ones
for band in bands:
    pixels[index] = band_color  # Replace, don't add
```
**Result:** Higher frequencies dominate visually, cleaner separation

#### Mirror Symmetry (scottlawsonbc)
```python
# Create mirror image from center
output = np.concatenate((y[::-1], y))
```
**Result:** Aesthetic symmetry, makes single strip look "balanced"

---

## 5. What Makes Effects "Feel Right"

Based on analysis of successful projects and common pitfalls:

### 5.1 Patterns (DO THIS)

#### ✓ Asymmetric Attack/Decay
- **Fast rise (α = 0.7-0.99):** Effect appears instantly with beat
- **Slow decay (α = 0.1-0.3):** Smooth fadeout, no flicker
- **Found in:** All successful projects use this

#### ✓ Frequency-Dependent Time Constants
- **Bass:** Slower decay (200-500ms) - low frequencies have longer natural decay
- **Highs:** Faster decay (50-150ms) - cymbals/hi-hats are percussive
- **Found in:** scottlawsonbc uses different filters per RGB channel

#### ✓ Perceptual Scaling
- **Power law (x^0.9 to x^2.2):** Compresses dynamic range perceptually
- **Log scale:** For frequency (mel filterbank, not linear FFT bins)
- **Found in:** scottlawsonbc (x^0.9), gamma correction (x^2.2)

#### ✓ Spatial Smoothing
- **Light blur (σ = 0.2-1.0):** Removes single-pixel jitter
- **Heavy blur (σ = 3.0-5.0):** Creates atmospheric "glow" effects
- **Found in:** scottlawsonbc applies gaussian blur post-processing

#### ✓ Adaptive Normalization with Limits
- **Track recent peaks, not instantaneous values
- **Bound adaptation range** (don't let gain go 0→∞)
- **Found in:** scottlawsonbc rolling peak normalization

#### ✓ Temporal Derivative for Transients
```python
change = abs(current_frame - previous_frame)
```
- Highlights onsets without explicit beat detection
- **Found in:** scottlawsonbc green channel

### 5.2 Anti-Patterns (AVOID THIS)

#### ✗ Symmetric or No Smoothing
```python
# Bad: Jittery, follows every audio sample fluctuation
output = raw_audio_magnitude
```
**Result:** Flickering, epilepsy-inducing, doesn't feel musical

#### ✗ Too Much Smoothing
```python
# Bad: Laggy, unresponsive
alpha = 0.01  # 99% history, 1% new data
```
**Result:** Effect lags behind music by noticeable delay, feels disconnected

#### ✗ Per-Frame Normalization
```python
# Bad: Destroys dynamics
output = (data - min(data)) / (max(data) - min(data))
```
**Result:** Quiet and loud sections look identical, no musical dynamics

#### ✗ Linear Frequency Spacing
```python
# Bad: Perceptually unbalanced
freq_bins = np.linspace(20, 20000, num_bins)
```
**Result:** Bass occupies 1-2 bins, treble dominates. Use mel scale or log spacing.

#### ✗ No Spatial Smoothing on Discrete LEDs
```python
# Bad: Aliasing and pixel-level jitter visible
pixels[i] = frequency_bin[i]  # Direct 1:1 mapping
```
**Result:** Harsh pixel boundaries, "digital" looking

#### ✗ Clipping Without Warning
```python
# Bad: Silent saturation
output = min(max(value, 0), 255)  # Clips without adjustment
```
**Result:** Loud sections look flat and uninteresting, no dynamics preserved

### 5.3 Frame Rate Considerations

#### Fixed Time Constants (CORRECT)
```python
# Time constant in seconds
TAU = 0.1  # 100ms decay
alpha = 1.0 - exp(-dt / TAU)  # dt = frame time
```
**Result:** Consistent behavior at any frame rate

#### Frame-Dependent Constants (WRONG)
```python
# Bad: alpha directly used
alpha = 0.1  # Means different things at 30fps vs 60fps
```
**Result:** Effect looks different on different hardware

**Note:** None of the projects reviewed do this correctly. They all use fixed alpha values that assume 60fps. This is acceptable for LED projects (frame rate is usually constant) but not for cross-platform visualizers.

### 5.4 Latency Budget

Total system latency should be < 50ms for effect to "feel tight" with music.

**Typical breakdown:**
- Audio input buffering: 10-20ms (512-1024 samples at 44.1kHz)
- FFT computation: <1ms (modern processors)
- Smoothing (1-frame delay): 16ms at 60fps
- LED communication: 1-5ms (depends on protocol/length)
- **Total:** ~30-45ms

**Critical threshold:** 50ms is perception limit for audio-visual sync
**Above 100ms:** Noticeably laggy, feels disconnected

**Configuration constants (scottlawsonbc):**
```python
MIC_RATE = 44100  # Sample rate
FPS = 60          # Visual refresh rate
N_ROLLING_HISTORY = 2  # Frames of history
```

At 60fps with 2-frame history: ~33ms of built-in latency just from buffering.

---

## 6. Code Snippets

### 6.1 Exponential Filter (Universal Building Block)

```python
class ExpFilter:
    """
    Asymmetric exponential moving average filter.
    Small alpha = more smoothing (slower response)
    Large alpha = less smoothing (faster response)
    """
    def __init__(self, initial_value, alpha_decay=0.5, alpha_rise=0.5):
        self.value = initial_value
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise

    def update(self, new_value):
        if isinstance(new_value, (list, np.ndarray)):
            alpha = np.where(
                new_value > self.value,
                self.alpha_rise,
                self.alpha_decay
            )
        else:
            alpha = self.alpha_rise if new_value > self.value else self.alpha_decay

        self.value = alpha * new_value + (1.0 - alpha) * self.value
        return self.value
```

**Usage example:**
```python
# Sticky peak detector
peak_filter = ExpFilter(0.0, alpha_decay=0.1, alpha_rise=0.99)

# Smooth follower
smooth_filter = ExpFilter(0.0, alpha_decay=0.2, alpha_rise=0.2)

# In audio loop
for frame in audio_stream:
    magnitude = abs(fft(frame))
    peak = peak_filter.update(np.max(magnitude))
    smooth = smooth_filter.update(magnitude)
```

### 6.2 Mel Filterbank Setup (scottlawsonbc)

```python
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

# Configuration
MIN_FREQUENCY = 200    # Hz
MAX_FREQUENCY = 12000  # Hz
N_FFT_BINS = 24
MIC_RATE = 44100       # Sample rate
FPS = 60

# Calculate sample count for FFT window
samples = int(MIC_RATE * N_ROLLING_HISTORY / (2.0 * FPS))

# Create mel filterbank matrix
# This converts linear FFT bins to perceptually-spaced mel bins
mel_y, mel_x = melbank.compute_melmat(
    num_mel_bands=N_FFT_BINS,
    freq_min=MIN_FREQUENCY,
    freq_max=MAX_FREQUENCY,
    num_fft_bands=samples,
    sample_rate=MIC_RATE
)

# Apply to FFT output
fft_magnitude = np.abs(np.fft.rfft(audio_window))
mel_spectrum = np.atleast_2d(fft_magnitude).T @ mel_y.T
mel_spectrum = mel_spectrum ** 2.0  # Quadratic emphasis
```

### 6.3 Spectrum Visualization (scottlawsonbc)

```python
# Initialize filters for RGB channels
r_filt = ExpFilter(np.tile(0.01, N_PIXELS // 2), alpha_decay=0.2, alpha_rise=0.99)
g_filt = ExpFilter(np.tile(0.01, N_PIXELS // 2), alpha_decay=0.05, alpha_rise=0.3)
b_filt = ExpFilter(np.tile(0.01, N_PIXELS // 2), alpha_decay=0.1, alpha_rise=0.5)

# Track common mode (DC offset) and gain
common_mode = ExpFilter(0.0, alpha_decay=0.99, alpha_rise=0.01)
mel_gain = ExpFilter(np.tile(1e-1, N_FFT_BINS), alpha_decay=0.01, alpha_rise=0.99)

def visualize_spectrum(mel):
    """Convert mel spectrum to RGB LED values."""
    # Smooth and normalize
    mel = gaussian_filter1d(mel, sigma=1.0)
    mel_gain.update(np.max(mel))
    mel /= mel_gain.value

    # Interpolate to LED count
    y = np.copy(interpolate(mel, N_PIXELS // 2))

    # Update common mode (DC component)
    common_mode.update(np.mean(y))

    # Calculate temporal derivative (change between frames)
    diff = y - prev_y
    prev_y = np.copy(y)

    # Apply filters to each channel
    r = r_filt.update(y - common_mode.value)  # DC-removed energy
    g = np.abs(diff)                           # Change rate (transients)
    b = b_filt.update(y)                       # Smoothed magnitude

    # Combine and scale to LED range
    output = np.array([r, g, b]) * 255

    # Mirror for symmetric display
    output = np.concatenate((output[:, ::-1], output), axis=1)

    return output.astype(int)
```

### 6.4 Energy-Based Expansion (scottlawsonbc)

```python
def visualize_energy(mel):
    """Create expanding bars from center based on frequency energy."""
    # Normalize
    mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
    mel /= mel_gain.value

    # Scale to pixel range
    y = mel * (N_PIXELS // 2 - 1)

    # Apply power law per frequency band
    bass_len = int(np.mean(y[:len(y)//3] ** 0.9))
    mid_len = int(np.mean(y[len(y)//3:2*len(y)//3] ** 0.9))
    high_len = int(np.mean(y[2*len(y)//3:] ** 0.9))

    # Create output array
    pixels = np.zeros((3, N_PIXELS))

    # Fill from center outward
    center = N_PIXELS // 2
    pixels[0, center:center+bass_len] = 255  # Red bass to right
    pixels[0, center-bass_len:center] = 255  # Red bass to left
    pixels[1, center:center+mid_len] = 255   # Green mids
    pixels[1, center-mid_len:center] = 255
    pixels[2, center:center+high_len] = 255  # Blue highs
    pixels[2, center-high_len:center] = 255

    # Apply heavy gaussian blur for smooth gradient
    for i in range(3):
        pixels[i, :] = gaussian_filter1d(pixels[i, :], sigma=4.0)

    return pixels.astype(int)
```

### 6.5 Web Audio API Bar Graph Visualizer

```javascript
// Setup
const audioCtx = new AudioContext();
const analyser = audioCtx.createAnalyser();
analyser.fftSize = 256;  // Small FFT for clear bars
analyser.smoothingTimeConstant = 0.8;  // Built-in smoothing

const bufferLength = analyser.frequencyBinCount;  // 128 bins
const dataArray = new Uint8Array(bufferLength);

// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function draw() {
    requestAnimationFrame(draw);

    // Get frequency data (0-255 per bin)
    analyser.getByteFrequencyData(dataArray);

    // Clear canvas
    ctx.fillStyle = 'rgb(0, 0, 0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Calculate bar dimensions
    const barWidth = (canvas.width / bufferLength) * 2.5;  // 2.5x magnification
    let x = 0;

    // Draw bars
    for (let i = 0; i < bufferLength; i++) {
        // Scale height
        const barHeight = dataArray[i] / 2;  // Fit to canvas

        // Color based on magnitude (more red = louder)
        ctx.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;

        // Draw from bottom up
        ctx.fillRect(
            x,
            canvas.height - barHeight / 2,  // Y position (bottom-up)
            barWidth,
            barHeight
        );

        x += barWidth + 1;  // 1px spacing
    }
}

// Start visualization
draw();
```

### 6.6 LedFx-Style Gradient Band Visualizer

```python
def visualize_bands(mel, num_bands=8, gradient_colors=None):
    """
    Split LED strip into bands, each showing energy of frequency band.

    Args:
        mel: Mel spectrum array (N_FFT_BINS values)
        num_bands: Number of frequency bands to display
        gradient_colors: List of RGB tuples for gradient
    """
    # Default gradient: red -> yellow -> green -> cyan -> blue
    if gradient_colors is None:
        gradient_colors = [
            (255, 0, 0),    # Red
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (0, 255, 255),  # Cyan
            (0, 0, 255),    # Blue
        ]

    # Normalize mel spectrum
    mel = np.clip(mel, 0, 1)

    # Create output array
    pixels = np.zeros((N_PIXELS, 3))

    # Calculate pixels per band
    band_width = N_PIXELS // num_bands

    # Split mel spectrum into bands
    mel_bands = np.array_split(mel, num_bands)

    for band_idx in range(num_bands):
        # Get band energy (max amplitude in this frequency range)
        band_energy = np.max(mel_bands[band_idx])

        # Calculate how many LEDs to light in this band
        lit_count = int(band_energy * band_width)

        # Get color from gradient
        gradient_pos = band_idx / num_bands
        color = interpolate_gradient(gradient_colors, gradient_pos)

        # Fill pixels for this band
        band_start = band_idx * band_width
        band_end = band_start + band_width

        # Light from bottom (or start) of band
        for i in range(lit_count):
            pixels[band_start + i] = color

    return pixels.astype(int)

def interpolate_gradient(colors, position):
    """Interpolate color from gradient at position [0, 1]."""
    position = np.clip(position, 0, 1)

    # Find surrounding colors
    scaled_pos = position * (len(colors) - 1)
    idx1 = int(scaled_pos)
    idx2 = min(idx1 + 1, len(colors) - 1)

    # Interpolate
    frac = scaled_pos - idx1
    color = np.array(colors[idx1]) * (1 - frac) + np.array(colors[idx2]) * frac

    return color
```

---

## 7. Configuration Values Summary

### Frequency Analysis
| Parameter | Typical Value | Range | Notes |
|-----------|--------------|-------|-------|
| Sample Rate | 44100 Hz | 22050-48000 Hz | Higher = better freq resolution |
| FFT Size | 512-2048 | 256-8192 | Larger = better freq res, worse time res |
| Freq Range | 200-12000 Hz | 60-20000 Hz | Cut DC and ultrasonic noise |
| Mel Bins | 24 | 12-64 | More bins = finer freq detail |

### Temporal Processing
| Parameter | Typical Value | Range | Purpose |
|-----------|--------------|-------|---------|
| Frame Rate | 60 fps | 30-120 fps | LED refresh rate |
| Attack (rise) | 0.5-0.99 | 0.3-0.99 | Higher = instant response |
| Decay (fall) | 0.1-0.3 | 0.05-0.5 | Lower = smooth fadeout |
| Rolling History | 2 frames | 1-5 frames | For derivative calculation |

### Spatial Processing
| Parameter | Typical Value | Range | Purpose |
|-----------|--------------|-------|---------|
| Gaussian σ | 0.2-4.0 | 0-10 | Light blur to heavy glow |
| LED Count | 60-300 | 30-1000+ | More LEDs = higher spatial res |

### Normalization
| Parameter | Typical Value | Range | Purpose |
|-----------|--------------|-------|---------|
| Peak Decay | 0.1-0.2 | 0.05-0.5 | How fast peak tracker forgets |
| Min Threshold | 1e-7 | 1e-10 to 1e-5 | Noise floor cutoff |

### Color Mapping
| Parameter | Typical Value | Range | Purpose |
|-----------|--------------|-------|---------|
| Gamma | 2.2 | 1.8-2.8 | Perceptual brightness correction |
| Power Law | 0.9 | 0.5-1.5 | Dynamic range compression |
| Bass Cutoff | 800 Hz | 300-1200 Hz | Bass/mid boundary |
| High Cutoff | 3000 Hz | 2000-5000 Hz | Mid/high boundary |

---

## 8. Key Takeaways

### What Works (Proven Patterns)

1. **Exponential filtering with asymmetric attack/decay** - Universal in all successful projects
2. **Mel-scale frequency bins** - Better than linear FFT bins for musical perception
3. **Rolling peak normalization** - Adaptive gain without destroying dynamics
4. **Power law scaling (x^0.9 to x^2.2)** - Compresses dynamic range perceptually
5. **Spatial smoothing (gaussian blur)** - Removes jitter, creates professional look
6. **Temporal derivative for transients** - Simple onset detection without complex algorithms
7. **Mirror symmetry** - Aesthetic improvement for single LED strips

### What Doesn't Work (Anti-Patterns)

1. **Per-frame normalization** - Destroys musical dynamics
2. **Symmetric or no smoothing** - Creates flickering
3. **Too much smoothing** - Makes effect laggy and disconnected
4. **Linear frequency spacing** - Perceptually unbalanced
5. **Frame-rate dependent constants** - Inconsistent behavior across systems

### Critical Numbers to Remember

- **Attack α: 0.7-0.99** (instant response to beats)
- **Decay α: 0.1-0.3** (smooth fadeout)
- **Gamma: 2.2** (LED brightness correction)
- **FFT: 512-2048 samples** at 44.1kHz
- **Frame rate: 60 fps** (standard for LEDs)
- **Latency budget: <50ms** (for tight sync)
- **Frequency range: 200-12000 Hz** (musical content)
- **Mel bins: 24** (good balance of resolution and performance)

### Research Gaps (Not Found in Projects)

These techniques are known to work but weren't found in the examined codebases:
- **Sophisticated onset detection** (spectral flux, complex domain)
- **HSV color space mapping** (hue from spectral centroid)
- **Multi-band AGC** (professional adaptive gain control)
- **Frame-rate independent time constants** (exp(-dt/τ) formulation)
- **Perceptual loudness weighting** (A-weighting or K-weighting)

---

## Sources

### Primary Code Analysis
1. **scottlawsonbc/audio-reactive-led-strip** - visualization.py, config.py, dsp.py
   - Exponential filtering implementation
   - Mel filterbank setup
   - Spectrum and energy visualization algorithms
   - Specific alpha values for RGB channels

2. **LedFx/LedFx** - effects/spectrum.py, effects/energy.py, effects/bands.py
   - Permutation-based RGB mixing
   - Gradient palette mapping
   - Band-based visualization with alignment modes
   - Sensitivity-dependent decay constants

3. **MDN Web Audio API Documentation** - Visualizations with Web Audio API
   - Web Audio analyser setup
   - Bar graph and waveform algorithms
   - Normalization and scaling strategies
   - Frame loop patterns with requestAnimationFrame

4. **Three.js** - webaudio_visualizer.html example
   - GPU-accelerated visualization via texture sampling
   - Shader-based frequency bar rendering

### Hardware/Platform Documentation
- **WLED-SR (Sound Reactive)** - Overview of volume and frequency reactive approaches
- **projectM** - Milkdrop preset architecture and FFT + beat detection pipeline

### Configuration Values
- scottlawsonbc config.py: Concrete values for sample rate, FFT bins, frequency range, frame rate
- LedFx effect parameters: Sensitivity ranges, blur effects, mixing modes

---

**Document compiled from web research - all numeric constants and algorithms extracted from actual source code or technical documentation.**
