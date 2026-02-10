# Flourish Detection: The Counterintuitive Finding

## The Surprising Result

**Flourishes are QUIETER and LESS active than on-beat moments.**

This is counterintuitive because we think of flourishes as "emphasis" — special moments that deserve extra attention. But the audio data reveals the opposite pattern.

## Top Discriminating Features (All Negative Effect Sizes)

| Feature | Effect Size | Flourish vs On-Beat |
|---------|-------------|---------------------|
| percussive_energy | -0.533 | -24.7% quieter |
| rms | -0.475 | -20.2% quieter |
| onset_density | -0.469 | -13.7% less activity |
| spectral_flux | -0.459 | -22.3% less change |
| spectral_novelty | -0.419 | -20.9% less novelty |

**All top 5 features show flourishes as having LESS of something compared to on-beat.**

## Why This Makes Sense

### Hypothesis 1: Flourishes as "Gaps in the Grid"
- On-beat moments are the **main rhythmic engine** — loud, consistent, driving
- Flourishes are the **spaces between** — accents, fills, grace notes
- They're noteworthy precisely BECAUSE they break the steady pattern
- Example: in Opiate, the consistent beat is the heavy kick/snare groove
- Flourishes are the cymbal taps, tom fills, guitar accents AROUND that groove

### Hypothesis 2: Human Perception vs Raw Energy
- Humans don't tap flourishes because they're LOUD
- They tap them because they're INTERESTING (timbre, timing, context)
- A quiet hi-hat accent can be more "flourish-worthy" than a loud kick drum
- Surprise value > absolute energy

### Hypothesis 3: The "Fill Moment" Effect
- Many flourishes are drum fills or transitions
- Fills often have LOWER energy than the main groove
- Example: drummer plays steady heavy hits, then does a quick quiet roll before the next section
- The roll is a flourish (noteworthy, off-beat) but quieter than the main hits

## Comparison: Flourishes vs Non-Events

When we compare flourishes to random non-event baseline moments, we get the OPPOSITE pattern:

| Feature | Effect Size | Flourish vs Baseline |
|---------|-------------|----------------------|
| spectral_bandwidth | +1.319 | Much wider |
| spectral_contrast | +1.240 | Much higher |
| spectral_centroid | +1.162 | Much brighter |
| onset_density | +1.054 | More active |
| rms | +0.848 | Louder |
| percussive_energy | +0.831 | More percussive |

**Flourishes are MUCH more active than silence or background texture.**

## The Full Picture: A Two-Tier System

```
Loudness/Activity Spectrum:

[Baseline/Background] << [Flourishes] << [On-Beat]
      ^                      ^                ^
   quiet drone         interesting       main groove
   sustained note         fills          kick/snare
   guitar wash         cymbal taps       driving rhythm
```

**Flourishes occupy the MIDDLE ground:**
- More interesting than background
- Less intense than the main beat
- Noteworthy precisely because they're NOT the beat

## What This Means for Detection

### The Wrong Approach (what we expected)
```python
# Detect flourishes as "extra loud moments"
if energy > high_threshold:
    flourish = True  # WRONG!
```

### The Right Approach (what the data shows)
```python
# Detect flourishes as "moderate moments that aren't on the beat"
if beat_detector.is_beat(t):
    # It's on the beat, not a flourish
    pass
elif energy > background_threshold and energy < beat_energy:
    # It's in the "interesting but not main beat" range
    if spectral_novelty > threshold or onset_strength > threshold:
        flourish = True  # This is the gap-filling accent
```

**Key insight**: Flourishes are defined RELATIONALLY, not absolutely.
- They're not "the loudest moments"
- They're "the moments that stand out FROM THE BEAT"
- Detection requires knowing where the beat is first

## Detection Strategy v2

### Step 1: Find the Beat Grid
- Use beat tracking or consistent-beat taps
- Identify the main rhythmic pulse (the loud, steady events)

### Step 2: Find Off-Beat Activity
- Look for onsets that are >150ms from any beat
- Filter by moderate energy (not too quiet, not as loud as beat)
- Score by contextual interest:
  - Is it different from surrounding frames? (spectral novelty)
  - Is it a clear transient? (onset strength)
  - Is it percussive but not on the beat? (percussive energy in off-beat window)

### Step 3: The "Interesting Gap" Score
```python
def flourish_score(t, features, beat_grid):
    # Must be off-beat
    if is_near_beat(t, beat_grid, tolerance=150ms):
        return 0

    score = 0

    # Must have some activity (not background)
    if features['rms'] < 0.05:
        return 0

    # But not as much as the beat
    beat_rms = get_beat_rms(beat_grid)
    if features['rms'] > beat_rms * 0.8:
        # Too loud, probably an undetected beat
        score -= 1

    # Prefer moderate energy
    if 0.08 < features['rms'] < beat_rms * 0.7:
        score += 1

    # Novelty is good (unusual timbre)
    if features['spectral_novelty'] > 120:
        score += 1

    # Percussive but quieter than beat
    if features['percussive_energy'] > 0.02 and features['percussive_energy'] < beat_rms * 0.6:
        score += 1

    return score
```

## Genre Considerations

This pattern (flourishes quieter than beat) is likely **genre-specific**:

### Likely to hold:
- **Rock/metal** (like Tool): Heavy beat, lighter fills and accents
- **Funk**: Strong downbeat, ghost notes and fills between
- **Jazz**: Clear ride/hi-hat pulse, quieter comping and fills

### Likely to invert:
- **Electronic/EDM**: Steady kick might be moderate, fills might be LOUD
- **Ambient**: No strong beat, flourishes might be the LOUDEST events
- **Classical**: Extremely dynamic, flourishes could go either way

### Recommendation
Test on:
1. `electronic_beat.wav` (Fred again..) — expect different pattern
2. `ambient.wav` (Fred again..) — expect inverted pattern
3. Rock/metal examples — expect same pattern as Opiate

## The "Air" vs "Flourish" Hypothesis

We don't have air tap analysis yet, but prediction:

**Air taps**: Sustained, quiet, textural
- Low percussive energy (sustained notes)
- Low onset density (steady, not punctuated)
- Possibly low RMS (soft textures)
- High spectral bandwidth or flatness (complex/noisy textures)

**Flourish taps**: Transient, moderate energy, punctuated
- Moderate percussive energy (accents/hits, but not main beat)
- Moderate onset density (some activity)
- Moderate RMS (audible but not dominating)
- Variable spectral features depending on instrument

**Prediction**: Air and flourish will share LOW energy, but differ in:
- Air = sustained, flourish = transient
- Air = smooth, flourish = punctuated
- Air = background layer, flourish = foreground accent

## LED Mapping Implications

### Old thinking (wrong)
```python
# Make flourishes the BRIGHTEST moments
if is_flourish(t):
    led.brightness = MAX
```

### New thinking (correct)
```python
# Flourishes are ACCENTS, not PEAKS
if is_beat(t):
    led.brightness = HIGH  # Main pulse, strong and steady
    led.color = PRIMARY
elif is_flourish(t):
    led.brightness = MEDIUM  # Accent, noticeable but not overpowering
    led.color = SECONDARY  # Differentiate from beat
    led.effect = SPARKLE  # Draw attention with motion, not just brightness
else:
    led.brightness = map_to_range(rms, LOW, MED)  # Background
    led.color = AMBIENT
```

**Key principle**: Flourishes should be visually distinct from the beat, not necessarily brighter.

Options:
- Different color (beat = warm, flourish = cool)
- Different motion (beat = solid, flourish = sparkle/chase)
- Different spatial pattern (beat = whole tree, flourish = branch accents)
- Brightness relative to context (if beat is quiet, flourish can be brighter; if beat is loud, flourish is mid-level accent)

## Next Steps

1. **Test the relational model** on this track:
   - Compute beat_rms as baseline
   - Score moments by their relationship to beat energy
   - See if medium-energy off-beat moments correlate with flourish taps

2. **Test on different genres**:
   - Electronic: expect flourishes might be LOUDER than steady kick
   - Ambient: expect flourishes might be the ONLY loud moments
   - Rock: expect same pattern as Opiate

3. **Temporal context**:
   - Are flourishes preceded/followed by steady beat?
   - Is there a "breathing" pattern (beat → flourish → beat)?

4. **Multi-scale flourishes**:
   - Micro: individual accents within a bar
   - Macro: section transitions, fills before drops

5. **Compare to air taps** to understand the full "annotation vocabulary":
   - Beat = main pulse (loud, steady)
   - Flourish = accent/fill (moderate, transient, off-beat)
   - Air = texture/atmosphere (quiet, sustained)
   - Changes = structural transitions (large-scale)

## Conclusion

**Flourishes are not "extra emphasis" in absolute terms.**
**They are "relative contrasts" to the main beat.**

The detection strategy must be:
1. Find the beat (the loudest, most regular events)
2. Find off-beat activity (anything not aligned with beat)
3. Filter by moderate energy (not background, not beat-level)
4. Score by interest (novelty, percussiveness, transients)

This is fundamentally a **two-pass algorithm**:
- First pass: beat tracking
- Second pass: flourish detection in the gaps

You cannot detect flourishes without knowing where the beat is.

---

Generated by flourish_audio_properties.py
Dataset: 91 flourish taps from 3 annotation sources
Track: Tool - Opiate Intro (40.6s, psych rock)
