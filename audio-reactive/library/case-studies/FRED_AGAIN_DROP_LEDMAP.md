# LED Mapping Guide for Electronic Drop Structure

How to translate the detected drop structure into real-time LED effects.

## Quick Reference

Based on analysis of `fa_br_drop1.wav`, here's how to map each detected section to LED behavior:

| Section | Duration | LED Strategy | Key Features |
|---------|----------|--------------|--------------|
| Normal Song | 0-29s | Standard reactive | Beat sync, full palette |
| Tease | 29-83s | Sparse + cyclic surges | Mini-builds at 42s, 53s, 56s, 64s, 70s |
| Real Build | 83-113s | Gradual ramp | Brightness/color temperature climb |
| Bridge | 113-121s | Chaotic/glitchy | Random patterns, high variation |
| Drop | 121-129s | Maximum intensity | Full brightness, bass-driven |

## Phase-by-Phase Implementation

### Phase 1: Normal Song (0-29.4s)

**Audio Characteristics:**
- Balanced frequency content (bass ≈ mids)
- Moderate RMS energy (0.26)
- Stable spectral centroid (1640 Hz)

**LED Behavior:**

```python
# Baseline mode
mode = "beat_reactive"
brightness_base = 0.6  # 60% brightness
color_palette = "full"  # All colors available
effect_speed = "moderate"

# Sync to onset strength
if onset_strength > threshold:
    flash_brightness = 1.0
    flash_duration = 50ms  # Quick pulse
```

**Effect Examples:**
- Beat-synchronized color changes
- Running chases on strong beats
- Brightness pulses on kicks/snares
- Moderate color cycling

**Goal:** Establish baseline visual energy that matches the musical groove.

---

### Phase 2: Tease/Edge Section (29.4-83.5s)

**Audio Characteristics:**
- Bass drops 26% from normal
- Sparse arrangement
- 5 cyclic builds that don't pay off
- Mid energy varies cyclically

**LED Behavior:**

```python
# Minimal baseline
mode = "sparse_anticipation"
brightness_base = 0.3  # Dim to create contrast
color_palette = "restricted"  # Monochrome or limited colors
effect_speed = "slow"

# Cyclic build detection
tease_cycle_times = [42.0, 52.8, 56.2, 64.5, 70.3]  # From annotations

for cycle_time in tease_cycle_times:
    if current_time >= cycle_time - 2.0:  # 2s before cycle peak
        # Build anticipation
        brightness_target = 0.7
        ramp_duration = 2.0s

    if current_time >= cycle_time:  # At cycle peak
        # Brief surge (don't fully pay off)
        brightness_target = 0.8  # NOT full brightness
        hold_duration = 0.3s

    if current_time >= cycle_time + 0.3:
        # Fade back down
        brightness_target = 0.3
        fade_duration = 1.0s
```

**Effect Examples:**
- **Breathing patterns** — slow rise/fall in brightness
- **Color temperature cycling** — cool to warm and back
- **Sparse pixel activation** — only partial strip lit
- **Slow color shifts** — long transitions between colors
- **Mini-surges** on detected cycle times (don't sustain)

**Key Insight:** Create visual anticipation without satisfaction. Each cycle should tease the viewer, building expectation for the eventual payoff.

**Pattern Ideas:**
- Start with only 10-20% of LEDs active
- On each cycle, briefly increase to 40-50% then back down
- Use unsaturated colors until the real build
- Slow, pulsing effects rather than sharp changes

---

### Phase 3: Real Build (83.5-113.2s)

**Audio Characteristics:**
- Sustained upward trend in mid-energy
- Spectral centroid rises (brighter)
- Bass nearly absent (focus on highs/mids)
- Clear trajectory toward climax

**LED Behavior:**

```python
# Gradual buildup
mode = "sustained_build"
build_start_time = 83.5
build_end_time = 113.2
build_duration = 29.7

# Calculate build progress (0.0 to 1.0)
build_progress = (current_time - build_start_time) / build_duration

# Map to LED parameters
brightness = 0.3 + (0.7 * build_progress)  # 30% → 100%
color_temperature = cool_to_warm(build_progress)  # Blue → Red
effect_density = 0.2 + (0.8 * build_progress)  # Sparse → Full
effect_speed = slow_to_fast(build_progress)  # Gradual acceleration

# Use smoothed mid-energy for micro-adjustments
brightness *= normalize(mid_energy, 0.8, 1.2)
```

**Effect Examples:**
- **Brightness ramp:** Smooth 0% → 100% over 30 seconds
- **Color temperature shift:** Blue/cool → Red/warm
- **Density increase:** Activate more LEDs progressively
- **Speed acceleration:** Effects get faster as build progresses
- **Pattern evolution:** Start simple, become complex

**Visual Progression Ideas:**

```
83s:  ░░░░░░░░░░ (sparse, cool blue, slow)
90s:  ▒░▒░▒░▒░▒░ (more pixels, blue-teal, moderate)
95s:  ▒▒▓▒▒▓▒▒▓▒ (denser, teal-yellow, faster)
100s: ▓▓▓▓▓▓▓▓▓▓ (very dense, yellow-orange, fast)
110s: ██████████ (full, orange-red, rapid)
```

**Key Insight:** This must be visually distinguishable from the tease cycles. Use a **sustained, consistent upward trajectory** that the viewer can track. No backsliding — only forward momentum.

---

### Phase 4: Bridge (113.2-120.7s)

**Audio Characteristics:**
- High spectral flatness (noise/unpitched)
- Unusual centroid behavior
- Short duration (7.4s)
- NOT the drop itself

**LED Behavior:**

```python
# Chaotic transition
mode = "bridge_chaos"
spectral_flatness_threshold = 0.001  # Detect unusual sounds

if spectral_flatness > spectral_flatness_threshold:
    # Randomized, glitchy effects
    pattern = random.choice([
        "strobe",
        "reverse_chase",
        "random_pixels",
        "color_glitch",
        "fast_flicker"
    ])

    # High variation
    brightness = random.uniform(0.3, 1.0)
    color = random_color()
    duration = random.uniform(50ms, 200ms)
```

**Effect Examples:**
- **Strobing** — rapid on/off
- **Color glitches** — rapid, unpredictable color changes
- **Random pixel activation** — chaotic, not synchronized
- **Reverse patterns** — break expected direction
- **Fast flicker** — intentionally jarring

**Key Insight:** This should feel **unsettling and different** from both the build and the drop. It's a moment of controlled chaos that signals "something is about to change."

**DO NOT:**
- Go full brightness sustained (that's the drop)
- Use smooth, predictable patterns (that's the build)
- Return to minimal (that's the tease)

**DO:**
- Create visual confusion/surprise
- Break the build's pattern
- Prepare viewer for massive change

---

### Phase 5: Drop (120.7-128.6s)

**Audio Characteristics:**
- Bass energy peaks (0.15)
- Full-spectrum content
- High onset strength
- SUSTAINED maximum intensity

**LED Behavior:**

```python
# Maximum intensity
mode = "drop_maximum"
brightness_base = 1.0  # FULL brightness
saturation = 1.0  # FULL saturation
effect_speed = "maximum"

# Bass-driven pulses
bass_energy = get_bass_band_energy()  # 20-250 Hz
if bass_energy > bass_threshold:
    strobe_all_leds()
    duration = 100ms

# Keep energy HIGH throughout
brightness_floor = 0.8  # Never drop below 80%

# Full-strip effects
pattern = random.choice([
    "full_flash",
    "fast_chase",
    "wave_pulse",
    "full_spectrum_cycle"
])
```

**Effect Examples:**
- **Full-strip strobes** on bass hits
- **Synchronized pulses** across entire strip
- **Fast chases** — maximum speed
- **Color cycling** — rapid, full saturation
- **Wave effects** — full amplitude

**Key Characteristics:**
- **Sustained high intensity** (not just a spike)
- **Full LED activation** (all pixels active)
- **Bass synchronization** (react to low frequencies)
- **High contrast** (bright/dark, saturated colors)

**Pattern Ideas:**

```python
# Bass pulse
def bass_pulse(bass_energy):
    brightness = map(bass_energy, 0, 1, 0.7, 1.0)
    color = hot_color  # Red, orange, white
    duration = 100ms

# Fast chase on beat
def drop_chase(onset_strength):
    if onset_detected:
        run_chase(speed="maximum", direction="random")
```

---

## Real-Time Detection Strategy

### Using Annotations (Offline)

If you have pre-analyzed tracks with annotations:

```python
import yaml

# Load annotations
with open('track.annotations_simple.yaml') as f:
    annotations = yaml.safe_load(f)

sections = annotations['sections']
tease_cycles = annotations['tease_cycles']
build_time = annotations['build'][0]
drop_time = annotations['drop'][0]

# During playback
def update_leds(current_time, audio_features):
    if current_time < sections[1]:  # Normal song
        return normal_song_mode(audio_features)
    elif current_time < sections[2]:  # Tease
        return tease_mode(audio_features, current_time, tease_cycles)
    elif current_time < sections[3]:  # Build
        return build_mode(audio_features, current_time, build_time)
    elif current_time < sections[4]:  # Bridge
        return bridge_mode(audio_features)
    else:  # Drop
        return drop_mode(audio_features)
```

### Real-Time Detection (Live Audio)

For unknown tracks, detect sections in real-time:

```python
# Maintain rolling buffers
rms_buffer = RollingBuffer(size=100)  # ~2 seconds
mid_energy_buffer = RollingBuffer(size=100)
bass_buffer = RollingBuffer(size=50)

def detect_section_transition(current_features, buffers):
    # Tease detection: bass drop
    if bass_energy < 0.7 * bass_buffer.mean():
        return "entering_tease"

    # Build detection: sustained mid-energy trend
    trend = calculate_trend(mid_energy_buffer, window=50)
    if trend > threshold and trend_duration > 5s:
        return "entering_build"

    # Bridge detection: spectral flatness spike
    if spectral_flatness > 10 * flatness_buffer.mean():
        return "entering_bridge"

    # Drop detection: bass × RMS peak
    if bass_energy * rms > threshold and just_after_bridge:
        return "entering_drop"
```

---

## Parameter Mapping Tables

### Brightness

| Section | Base | Range | Behavior |
|---------|------|-------|----------|
| Normal | 0.6 | 0.4-0.8 | Beat-reactive |
| Tease | 0.3 | 0.2-0.8 | Cyclic surges |
| Build | 0.3→1.0 | Linear ramp | Sustained climb |
| Bridge | Random | 0.3-1.0 | Chaotic |
| Drop | 1.0 | 0.8-1.0 | Sustained high |

### Color Strategy

| Section | Palette | Saturation | Behavior |
|---------|---------|------------|----------|
| Normal | Full | 0.6-0.8 | Varied |
| Tease | Limited | 0.3-0.6 | Monochrome/restricted |
| Build | Gradient | 0.5→1.0 | Cool→warm |
| Bridge | Random | 0.5-1.0 | Unpredictable |
| Drop | Hot | 1.0 | Reds/oranges/whites |

### Effect Speed

| Section | Speed | Pattern Type |
|---------|-------|--------------|
| Normal | Moderate | Beat-sync chases |
| Tease | Slow | Breathing, fades |
| Build | Slow→Fast | Accelerating |
| Bridge | Very Fast | Strobes, glitches |
| Drop | Maximum | Full-strip pulses |

---

## Testing Recommendations

1. **Listen first:** Play the track and mentally note when you'd expect each phase
2. **Check annotations:** Do detected boundaries match your perception?
3. **Test phase by phase:** Implement one section at a time
4. **Verify transitions:** Ensure smooth handoffs between phases
5. **Adjust thresholds:** Tune parameters to match track energy
6. **Compare to other tracks:** Does the same mapping work for similar drops?

---

## Advanced: Feature-Driven Modulation

Instead of hard section boundaries, use features for smooth transitions:

```python
# Smooth brightness based on smoothed mid-energy
brightness = base_brightness + (mid_energy_smooth * modulation_depth)

# Color temperature based on spectral centroid
color_temp = map(spectral_centroid, 1000, 4000, cool, warm)

# Effect density based on overall RMS
led_density = map(rms, 0.1, 0.5, sparse, full)

# Pattern speed based on onset strength
speed_multiplier = map(onset_strength, 0, 10, 0.5, 2.0)
```

This creates LED behavior that **flows with the music** rather than jumping between discrete modes.

---

## Summary

**The key to effective drop mapping:**

1. **Tease without payoff** — build anticipation but don't satisfy
2. **Build with trajectory** — clear, sustained upward momentum
3. **Bridge as chaos** — break expectations, signal change
4. **Drop as climax** — deliver full intensity, sustain it

Each phase must be **visually distinct** and **musically appropriate**. The viewer should be able to identify the structure through the LEDs alone.
