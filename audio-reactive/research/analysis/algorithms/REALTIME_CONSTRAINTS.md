# Real-Time Processing Constraints Analysis

## Executive Summary

This document re-analyzes all our audio-reactive findings through the lens of **real-time streaming constraints**. The system captures audio from a live stream (BlackHole) and must react immediately with **no lookahead**. Only past data is available.

**Key insight**: Many of our findings are REACTIVE (detect-after-it-happens), not PREDICTIVE (detect-before-it-happens). This is OK — the reaction itself can be the effect. But we must distinguish what can be anticipated vs what can only be responded to.

---

## Classification of Key Findings

### 1. IMMEDIATELY DETECTABLE (< 50ms latency)

These can be detected the moment they happen, within a single audio chunk.

#### 1.1 Bass Energy Peak
- **Finding**: User taps track bass peaks (19ms median alignment, 100% within 50ms)
- **Implementation**:
  - Compute bass band energy (20-250 Hz) per frame
  - Track with exponential peak follower (attack α=0.95, decay α=0.2)
  - Trigger when `current > peak * 0.9`
- **Window**: 1 frame (23ms @ 44.1kHz, hop=1024)
- **Latency**: ~25-50ms (1-2 frames)
- **LED behavior**: Beat flash, pulse start
- **Code**:
```python
bass_energy = np.mean(mel_spectrum[0:3])  # 20-250 Hz bins
peak = peak * decay + max(bass_energy, peak) * (1 - decay)
if bass_energy > peak * 0.9:
    trigger_beat_flash()
```

#### 1.2 Spectral Flux Peak (Onset Proxy)
- **Finding**: User taps in groove sections align with spectral flux peaks (4.7-20ms median)
- **Implementation**:
  - Compute spectral flux: `sum(max(0, spectrum[t] - spectrum[t-1]))`
  - Apply asymmetric smoothing (attack=0.9, decay=0.3)
  - Trigger when flux crosses threshold
- **Window**: 2 frames (46ms)
- **Latency**: ~50ms
- **LED behavior**: Color change, brightness spike
- **Code**:
```python
flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
smooth_flux = smooth_flux * decay + flux * (1 - decay)
if flux > smooth_flux * 1.5:
    trigger_onset()
```

#### 1.3 RMS Energy Level
- **Finding**: Core feature for loudness tracking
- **Implementation**:
  - Compute RMS per frame
  - Apply gamma correction (2.2) for LED brightness
  - Map to brightness with floor/ceiling
- **Window**: 1 frame (23ms)
- **Latency**: ~25ms
- **LED behavior**: Overall brightness level
- **Code**:
```python
rms = np.sqrt(np.mean(audio_chunk**2))
brightness = (rms / max_rms) ** (1/2.2)  # gamma correction
led.set_brightness(brightness)
```

#### 1.4 Spectral Centroid (Brightness)
- **Finding**: Correlates with timbral brightness, used in many sections
- **Implementation**:
  - Compute weighted frequency mean
  - Normalize to 0-1 range (map 0-8000 Hz)
  - Use for color temperature or filter cutoff
- **Window**: 1 frame (23ms)
- **Latency**: ~25ms
- **LED behavior**: Color warmth/coolness
- **Code**:
```python
centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
color_temp = np.clip((centroid - 1000) / 7000, 0, 1)
led.set_color_temp(color_temp)  # 0=warm, 1=cool
```

---

### 2. SHORT-CONTEXT DETECTABLE (50ms - 2s)

These need a short rolling window but are still responsive enough for LED effects.

#### 2.1 Tempo Estimation (Beat Period)
- **Finding**: Consistent-beat layer shows 82 BPM in groove sections
- **Implementation**:
  - Track inter-beat intervals (IBIs) in rolling buffer (last 8 beats)
  - Compute median IBI → tempo
  - Requires detecting several beats first (cold start: ~4-8 beats = 2-4 seconds)
- **Window**: 2-4 seconds (4-8 beats)
- **Latency**: 2-4s to stabilize, then real-time tracking
- **LED behavior**: Sets pulse rate for rhythmic effects
- **Code**:
```python
class TempoTracker:
    def __init__(self):
        self.last_beat_time = None
        self.ibis = deque(maxlen=8)  # last 8 intervals

    def on_beat(self, t):
        if self.last_beat_time is not None:
            ibi = t - self.last_beat_time
            self.ibis.append(ibi)
        self.last_beat_time = t

    def get_tempo(self):
        if len(self.ibis) < 4:
            return None  # not enough data
        return 60.0 / np.median(self.ibis)
```

#### 2.2 Beat Grid (Phase-Locked Loop)
- **Finding**: Needed for flourish detection (off-grid events)
- **Implementation**:
  - After tempo stabilizes, track expected beat times
  - Adjust phase when actual beats detected (phase correction)
  - Use as reference for on-grid vs off-grid classification
- **Window**: 2-4s to lock, then predictive
- **Latency**: 2-4s cold start
- **LED behavior**: Drives metronomic pulse layer
- **Code**:
```python
class BeatGrid:
    def __init__(self):
        self.phase = 0
        self.period = None
        self.last_correction = 0

    def update(self, t, beat_detected, tempo):
        if tempo and beat_detected:
            self.period = 60.0 / tempo
            expected = self.phase + self.period
            error = t - expected
            # Phase-lock: gradually correct drift
            self.phase = t - error * 0.1  # 10% correction

    def is_near_beat(self, t, tolerance=0.15):
        if not self.period:
            return False
        beat_phase = (t - self.phase) % self.period
        return beat_phase < tolerance or beat_phase > (self.period - tolerance)
```

#### 2.3 Spectral Centroid Rolling Average (for "Airiness")
- **Finding**: Airiness = deviation from local context (effect size -0.247)
- **Implementation**:
  - Track rolling mean of centroid (last 1-2 seconds)
  - Compute deviation: `abs(current - mean) / std`
  - Trigger when deviation > 2σ
- **Window**: 1-2 seconds (43-86 frames)
- **Latency**: 1-2s to establish baseline
- **LED behavior**: "Airy" mode trigger (sparse, spacey effects)
- **Code**:
```python
class DeviationDetector:
    def __init__(self, window_size=86):  # ~2 sec
        self.history = deque(maxlen=window_size)

    def update(self, value):
        self.history.append(value)
        if len(self.history) < 20:  # need minimum samples
            return 0
        mean = np.mean(self.history)
        std = np.std(self.history)
        if std < 1e-6:
            return 0
        deviation = abs(value - mean) / std
        return deviation

    def is_airy(self, value):
        return self.update(value) > 2.0  # 2σ threshold
```

#### 2.4 RMS Rolling Average (for Energy Context)
- **Finding**: Used for normalization, build/climax detection
- **Implementation**: Same as centroid deviation
- **Window**: 1-2 seconds
- **Latency**: 1-2s to establish baseline
- **LED behavior**: Dynamic range adaptation

---

### 3. MEDIUM-CONTEXT DETECTABLE (2-10s)

These need more history but can still detect patterns as they emerge.

#### 3.1 Flourish Ratio (Mode Selection)
- **Finding**: Ratio of off-grid to on-grid events correlates with song structure
  - Intro: 90.9% flourishes (no clear pulse)
  - Strong groove: 16.7% flourishes (tight lock-in)
- **Implementation**:
  - Detect onsets in 5-second rolling window
  - Classify each as on-grid or off-grid (using beat grid from 2.2)
  - Compute ratio: `off_grid / (on_grid + off_grid)`
- **Window**: 5 seconds
- **Latency**: 5s to compute first ratio
- **LED behavior**: Mode selection
  - High ratio (>70%): "Ambient mode" — respond to timbre, not beat
  - Medium (30-70%): "Accent mode" — pulse + highlights
  - Low (<30%): "Groove mode" — strong rhythmic structure
- **Code**:
```python
class FlourishRatioTracker:
    def __init__(self):
        self.events = deque(maxlen=50)  # ~5 sec of events
        self.beat_grid = BeatGrid()

    def add_event(self, t, is_beat):
        self.events.append((t, is_beat))

    def get_ratio(self):
        if len(self.events) < 10:
            return None
        off_grid = sum(1 for t, is_beat in self.events if not is_beat)
        return off_grid / len(self.events)

    def get_mode(self):
        ratio = self.get_ratio()
        if ratio is None or ratio > 0.7:
            return "ambient"
        elif ratio > 0.3:
            return "accent"
        else:
            return "groove"
```

#### 3.2 Build Detection (Energy/Centroid Trajectory)
- **Finding**: Builds show positive derivatives in centroid (+0.04 Hz/frame), climaxes show much steeper (+2.11 Hz/frame, 58x faster)
- **Implementation**:
  - Track centroid and RMS over 5-10 second window
  - Compute linear regression slope (trajectory)
  - Detect sustained positive slope = building
  - Detect large positive derivative = climax onset
- **Window**: 5-10 seconds (215-430 frames)
- **Latency**: 5-10s to detect trend
- **LED behavior**:
  - Build: Gradually increase brightness/intensity
  - Climax detected: Flash/peak brightness trigger
- **Code**:
```python
class BuildDetector:
    def __init__(self, window_sec=7):
        self.window_size = int(window_sec * 43)  # frames
        self.centroid_history = deque(maxlen=self.window_size)
        self.rms_history = deque(maxlen=self.window_size)
        self.time_history = deque(maxlen=self.window_size)

    def update(self, t, centroid, rms):
        self.time_history.append(t)
        self.centroid_history.append(centroid)
        self.rms_history.append(rms)

    def get_centroid_trajectory(self):
        if len(self.centroid_history) < 50:
            return 0
        # Linear regression slope
        x = np.arange(len(self.centroid_history))
        y = np.array(self.centroid_history)
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def is_building(self):
        slope = self.get_centroid_trajectory()
        return slope > 0.5  # sustained brightening

    def is_climax(self):
        slope = self.get_centroid_trajectory()
        return slope > 2.0  # rapid brightening (58x threshold)
```

#### 3.3 Section Change Detection (User "Changes" Layer)
- **Finding**: User marked structural transitions at specific times
- **Implementation**:
  - Detect large changes in multiple features simultaneously
  - Track: RMS variance, centroid variance, tempo stability, onset density
  - When 3+ features change significantly within 2-second window → section change
- **Window**: 5-10 seconds (compare before/after)
- **Latency**: 2-5s after change starts
- **LED behavior**: Scene transition (crossfade to new palette/pattern)
- **Code**:
```python
class SectionChangeDetector:
    def __init__(self):
        self.feature_detectors = {
            'rms_variance': VarianceTracker(window_sec=5),
            'centroid_variance': VarianceTracker(window_sec=5),
            'onset_rate': RateTracker(window_sec=5),
            'tempo_stability': TempoStabilityTracker()
        }
        self.last_change_time = 0

    def update(self, t, features):
        # Update all trackers
        change_signals = []
        for name, tracker in self.feature_detectors.items():
            if tracker.detect_change(features):
                change_signals.append(name)

        # Require 3+ features changing + cooldown (no changes in last 8s)
        if len(change_signals) >= 3 and (t - self.last_change_time) > 8:
            self.last_change_time = t
            return True
        return False
```

#### 3.4 Harmonic Ratio Trajectory (Build vs Climax Distinction)
- **Finding**: Build becomes more harmonic (+0.000055/frame), climax becomes more percussive (-0.000046/frame) — direction reversal
- **Implementation**:
  - Run HPSS (harmonic-percussive separation) per frame
  - Track harmonic ratio over 5-10s window
  - Compute slope
  - Positive = build (getting more tonal), negative = climax (getting more percussive)
- **Window**: 5-10 seconds
- **Latency**: 5-10s to detect trend
- **LED behavior**:
  - Build (more harmonic): Smooth, flowing patterns
  - Climax (more percussive): Sharp, staccato patterns
- **Code**:
```python
class HarmonicTrendDetector:
    def __init__(self):
        self.harmonic_ratio_history = deque(maxlen=430)  # 10 sec

    def update(self, audio_chunk):
        # HPSS
        H, P = librosa.effects.hpss(audio_chunk)
        h_energy = np.sum(H**2)
        p_energy = np.sum(P**2)
        ratio = h_energy / (h_energy + p_energy + 1e-6)
        self.harmonic_ratio_history.append(ratio)

    def get_trend(self):
        if len(self.harmonic_ratio_history) < 100:
            return 0
        x = np.arange(len(self.harmonic_ratio_history))
        y = np.array(self.harmonic_ratio_history)
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def is_build(self):
        return self.get_trend() > 0.00003  # getting more harmonic

    def is_climax(self):
        return self.get_trend() < -0.00003  # getting more percussive
```

---

### 4. REACTIVE ONLY (detected after the fact)

These can only be detected AFTER they happen. The reaction IS the effect.

#### 4.1 Drop (Electronic Music)
- **Finding**: Not explicitly in our data, but common in electronic music
- **Implementation**:
  - Detect sudden large increase in bass energy + RMS after quiet section
  - Pattern: sustained low energy (>2s) → sudden jump (>2x increase)
- **Window**: Compare last 2-4 seconds to current frame
- **Latency**: Detected ~50ms after drop hits
- **LED behavior**: Immediate full-brightness flash, then sync to new beat
- **Reaction type**: FLASH-ON-DROP (reactive)
- **Code**:
```python
class DropDetector:
    def __init__(self):
        self.rms_history = deque(maxlen=86)  # 2 sec
        self.bass_history = deque(maxlen=86)
        self.last_drop_time = 0

    def update(self, t, rms, bass):
        self.rms_history.append(rms)
        self.bass_history.append(bass)

        if len(self.rms_history) < 86:
            return False

        # Check for sudden jump
        recent_avg = np.mean(list(self.rms_history)[-86:-4])  # exclude last 4 frames
        current = np.mean(list(self.rms_history)[-4:])  # last 4 frames

        # Drop = quiet section followed by sudden loud
        if current > recent_avg * 2.0 and recent_avg < 0.1 and (t - self.last_drop_time) > 10:
            self.last_drop_time = t
            return True
        return False
```

#### 4.2 Flourish Events (Off-Beat Accents)
- **Finding**: Flourishes are quieter than beat (effect size -0.533 on percussive energy), occur off-grid
- **Implementation**:
  - Detect onset that's NOT near expected beat time
  - Filter by moderate energy (not background, not beat-level)
  - Trigger accent effect
- **Window**: Requires beat grid (from 2.2)
- **Latency**: ~50ms (onset detection latency)
- **LED behavior**: Color flash, sparkle, secondary pulse
- **Reaction type**: ACCENT-ON-FLOURISH (reactive)
- **Code**:
```python
class FlourishDetector:
    def __init__(self, beat_grid):
        self.beat_grid = beat_grid
        self.beat_rms = 0.15  # learned from on-beat energy

    def detect(self, t, onset_detected, rms, percussive_energy):
        if not onset_detected:
            return False

        # Must be off-grid
        if self.beat_grid.is_near_beat(t, tolerance=0.15):
            return False

        # Must have moderate energy (not too quiet, not beat-level)
        if rms < 0.05:
            return False  # too quiet
        if rms > self.beat_rms * 0.8:
            return False  # too loud, probably missed beat

        # Moderate energy + off-beat = flourish
        return True
```

#### 4.3 Airy Moment (Deviation Spike)
- **Finding**: Airiness = deviation from local context, happens suddenly
- **Implementation**: See 2.3 (Spectral Centroid Deviation)
- **Window**: 1-2s for baseline, then immediate detection
- **Latency**: 1-2s to establish context, then <50ms per event
- **LED behavior**: Switch to sparse/spacey mode, reduce brightness, slower motion
- **Reaction type**: MODE-CHANGE-ON-AIR (reactive)

---

### 5. REQUIRES LOOKAHEAD (not available in real-time)

These fundamentally need future data. Some can be approximated with heuristics.

#### 5.1 Anticipating Drops/Climaxes (PREDICTIVE)
- **Finding**: Not possible with audio alone
- **Approximation**: Detect builds and PREDICT drop is coming
  - If centroid slope > 0.5 for >5 seconds → likely climaxing soon
  - Start ramping up LED intensity during build
  - When climax hits (reactive), you're already primed
- **Heuristic latency**: 3-5s warning (if build is long enough)
- **LED behavior**: Pre-ramp intensity during build, then reactive flash on actual drop
- **Code**:
```python
class ClimaxPredictor:
    def __init__(self):
        self.build_detector = BuildDetector()
        self.build_duration = 0
        self.last_building_state = False

    def update(self, t, dt, features):
        is_building = self.build_detector.is_building()

        if is_building:
            if self.last_building_state:
                self.build_duration += dt
            else:
                self.build_duration = dt
        else:
            self.build_duration = 0

        self.last_building_state = is_building

        # Predict climax if building for >5 sec
        return self.build_duration > 5.0

    def get_anticipation_level(self):
        # 0-1 scale based on build duration
        return min(self.build_duration / 10.0, 1.0)
```

#### 5.2 Pre-Section-Change Preparation (PREDICTIVE)
- **Finding**: Cannot predict section changes without lookahead
- **Approximation**: Detect instability in features
  - If tempo becomes unstable (large variance in IBIs)
  - If multiple features show increasing variance
  - If flourish ratio rising rapidly
  - → Probably transitioning soon
- **Heuristic latency**: 2-4s warning (if transition is gradual)
- **LED behavior**: Pre-fade current scene, prepare for transition
- **Note**: This is WEAK prediction. Most section changes will still be reactive.

#### 5.3 Song-Global Context (Structure, Mood)
- **Finding**: Impossible without full-song analysis or metadata
- **Approximation**: None. Real-time systems don't know "verse vs chorus" or "intro vs outro"
- **Workaround**: Build up statistical profile over time
  - After 30s of audio, you know typical RMS range, tempo, spectral profile
  - Can detect "this section is brighter/louder/faster than average so far"
  - But you don't know if it's "the climax" in absolute terms
- **Not available**: Pre-programmed scene changes, lyrics-synced effects, structure-aware patterns

---

## Real-Time Feature Pipeline Design

### Layer 1: IMMEDIATE (< 50ms) — Reactive Beat Response
**Update rate**: Every frame (23ms)

| Feature | Window | Latency | Drives LED Behavior |
|---------|--------|---------|---------------------|
| Bass peak | 1 frame | 25ms | Beat flash |
| RMS energy | 1 frame | 25ms | Overall brightness |
| Spectral centroid | 1 frame | 25ms | Color temperature |
| Spectral flux peak | 2 frames | 50ms | Onset flash |

**Example LED effect**: On bass peak → white flash, decay over 300ms

---

### Layer 2: SHORT-CONTEXT (50ms - 2s) — Rhythmic Sync
**Update rate**: Per beat or 1/sec

| Feature | Window | Latency | Drives LED Behavior |
|---------|--------|---------|---------------------|
| Tempo (BPM) | 2-4s | 2-4s cold start | Sets pulse rate |
| Beat grid | 2-4s | 2-4s to lock | Metronomic pulse layer |
| Centroid deviation | 1-2s | 1-2s baseline | Airy mode trigger |
| RMS deviation | 1-2s | 1-2s baseline | Dynamic range adaptation |

**Example LED effect**: Once tempo locks → pulse entire tree at detected BPM

---

### Layer 3: MEDIUM-CONTEXT (2-10s) — Mode & Scene
**Update rate**: Every 1-5 seconds

| Feature | Window | Latency | Drives LED Behavior |
|---------|--------|---------|---------------------|
| Flourish ratio | 5s | 5s | Mode selection (ambient/accent/groove) |
| Centroid trajectory | 5-10s | 5-10s | Build detection (ramp up) |
| Harmonic trajectory | 5-10s | 5-10s | Build (smooth) vs climax (sharp) |
| Section change | 5-10s | 2-5s | Scene transition |

**Example LED effect**: If flourish ratio > 70% for 5s → switch from beat-driven pulse to timbral color shifts (ambient mode)

---

### Layer 4: REACTIVE EVENTS — Special Triggers
**Update rate**: Event-driven (when detected)

| Event | Detection Window | Latency | LED Response |
|-------|------------------|---------|--------------|
| Drop (electronic) | 2-4s context | 50ms | Full-brightness flash |
| Flourish (off-beat accent) | Beat grid + onset | 50ms | Sparkle/secondary color |
| Airy moment | 1-2s context | 50ms | Mode switch to sparse |
| Climax onset | 5-10s trajectory | 50ms | Peak brightness trigger |

**Example LED effect**: Drop detected → instant white flash → transition to high-energy chase pattern

---

### Layer 5: HEURISTIC PREDICTION — Anticipation
**Update rate**: Every 1-2 seconds

| Heuristic | Lookback | Warning Time | LED Pre-Response |
|-----------|----------|--------------|------------------|
| Build → climax | 5-10s | 3-5s | Gradual intensity ramp |
| Instability → section change | 5-10s | 2-4s | Pre-fade current scene |

**Example LED effect**: If centroid rising for 6 seconds → start ramping brightness from 50% → 100% over next 4s, so when climax hits, you're already at peak

---

## Minimum Rolling Window Sizes

| Feature/Detector | Minimum Window | Recommended Window | Why |
|------------------|----------------|-------------------|-----|
| Bass peak | 1 frame | 3 frames (70ms) | Need recent history for peak detection |
| Onset (flux) | 2 frames | 5 frames (115ms) | Compare current to recent |
| Tempo | 4 beats | 8 beats | Need several cycles to average |
| Beat grid | 4 beats | 8 beats | Lock to phase |
| Centroid deviation | 0.5s (20 frames) | 2s (86 frames) | Establish local mean/std |
| Flourish ratio | 3s | 5s | Need enough events to compute ratio |
| Build detection | 3s | 7-10s | Detect sustained trend |
| Section change | 5s | 10s | Compare before/after |

**Memory requirement**: For 10-second windows @ 44.1kHz, hop=1024:
- 430 frames × (num features)
- Example: 430 frames × 10 features × 4 bytes = ~17 KB per detector

---

## Latency Budget Breakdown

**Target**: < 50ms end-to-end for beat response

| Stage | Latency | Notes |
|-------|---------|-------|
| Audio capture | 0-10ms | BlackHole → Python (sounddevice) |
| FFT/Feature extraction | 10-20ms | librosa on 1024-sample chunk |
| Feature detection | 1-5ms | Peak detection, thresholds |
| LED command formation | 1-5ms | Map to RGB values |
| Serial transmission | 5-15ms | 115200 baud, ~300 bytes |
| Arduino processing | 1-3ms | Parse + update strip |
| **Total** | **18-58ms** | Within budget if optimized |

**Optimization targets**:
1. Pre-compute mel filterbank (don't recompute each frame)
2. Use incremental FFT if possible (rolling window)
3. Batch LED updates (send all LEDs in one serial packet)
4. Avoid Python loops (vectorize with numpy)

---

## Feature Availability Table

| Finding/Feature | Detectable? | Category | Latency | Can Anticipate? |
|----------------|-------------|----------|---------|-----------------|
| Bass peaks (beat) | ✓ | Immediate | 25ms | No (reactive only) |
| Spectral flux (onset) | ✓ | Immediate | 50ms | No (reactive only) |
| RMS energy | ✓ | Immediate | 25ms | No (reactive only) |
| Spectral centroid | ✓ | Immediate | 25ms | No (reactive only) |
| Tempo (BPM) | ✓ | Short-context | 2-4s | No (need history first) |
| Beat grid | ✓ | Short-context | 2-4s | Yes (predict next beat after lock) |
| Airiness (deviation) | ✓ | Short-context | 1-2s | No (reactive only) |
| Flourish ratio | ✓ | Medium-context | 5s | Weak (rising ratio → transition soon) |
| Build (trajectory) | ✓ | Medium-context | 5-10s | Yes (build → climax likely) |
| Climax (trajectory spike) | ✓ | Medium-context | 5-10s | Weak (if build detected) |
| Section change | ✓ | Medium-context | 2-5s | Weak (instability → change soon) |
| Flourish event | ✓ | Reactive | 50ms | No (reactive only) |
| Drop (electronic) | ✓ | Reactive | 50ms | Weak (if build detected) |
| Song structure | ✗ | Requires lookahead | N/A | No |
| Lyrics sync | ✗ | Requires lookahead | N/A | No |
| "Knowing climax is coming" | ✗ | Requires lookahead | N/A | Heuristic only (build detection) |

---

## Answers to Specific Questions

### Q: Can we detect build-to-climax transitions in real-time using derivative thresholds?

**A: YES, but reactively, not predictively.**

- **Detection method**: Track centroid trajectory over 7-10 second window
- **Build threshold**: Slope > 0.5 Hz/frame (sustained brightening)
- **Climax threshold**: Slope > 2.0 Hz/frame (58x faster than build)
- **Window size**: 7-10 seconds (300-430 frames)
- **Latency**: 5-10 seconds (need history to compute trend)

**What you CAN do**:
1. Detect "currently building" (centroid rising steadily) → ramp LED intensity
2. Detect "climax just started" (centroid accelerating) → trigger peak brightness

**What you CANNOT do**:
3. Detect "climax will start in 2 seconds" (no lookahead)

**But you can FAKE prediction**:
- If build has been going for 5+ seconds, assume climax is imminent
- Start ramping up intensity during build
- When climax actually hits (reactive), you're already at high intensity
- This gives the ILLUSION of anticipation

### Q: For "airiness = deviation from context," what's the ideal running average window?

**A: 1-2 seconds (43-86 frames).**

- **Too short** (< 0.5s): Noise in the deviation signal, every brief moment looks "different"
- **Too long** (> 4s): Slow to adapt, misses fast transitions
- **Sweet spot**: 1-2s captures "local context" without being too slow

**Implementation**:
```python
class AirDetector:
    def __init__(self):
        self.centroid_history = deque(maxlen=86)  # 2 sec

    def update(self, centroid):
        self.centroid_history.append(centroid)
        if len(self.centroid_history) < 20:
            return False
        mean = np.mean(self.centroid_history)
        std = np.std(self.centroid_history)
        deviation = abs(centroid - mean) / (std + 1e-6)
        return deviation > 2.0  # 2 sigma = airy
```

**Can we detect it fast enough?**
- After 2-second cold start: YES, detection is immediate (<50ms per frame)
- During the airy moment: You're detecting it AS IT HAPPENS
- This is reactive, not predictive — but that's OK
- LED response: When deviation spike detected, immediately switch to airy mode (dim, sparse, slow)

### Q: For drops in electronic music, is pure reactivity good enough?

**A: YES, and it's often the BEST approach.**

**Why reactive is good**:
1. **Surprise value**: Drops are meant to be impactful. A reactive flash feels synchronized.
2. **Low latency**: 50ms detection → flash is perceived as instant
3. **Simple implementation**: No complex prediction logic

**What makes a good reactive drop effect**:
1. **Immediate full-brightness flash** (0-50ms)
2. **Hold for 100-200ms** (let the impact register)
3. **Decay over 300-500ms** (fade to new steady state)
4. **Sync to new beat** (drop usually brings stronger, clearer beat)

**When prediction helps**:
- If the buildup is long and obvious (>8 seconds)
- Ramp up intensity during buildup
- When drop hits, you're already at 80% → flash to 100% feels like climax

**When prediction fails**:
- Sudden drops (no buildup) — can't predict
- False alarms (buildup doesn't lead to drop) — bad UX if you pre-flash

**Recommendation**: Start with pure reactive. Add predictive ramping only if buildups are long and reliable.

### Q: What are minimum rolling window sizes for each feature to be useful?

**See "Minimum Rolling Window Sizes" table above.**

**Key principle**: Window size depends on WHAT you're measuring:
- **Instantaneous features** (RMS, centroid): 1 frame is enough
- **Local context** (peaks, deviations): 0.5-2 seconds
- **Rhythmic features** (tempo, beat grid): 2-4 seconds (multiple beat cycles)
- **Trends** (build, trajectory): 5-10 seconds (need sustained change)

### Q: Which findings ONLY work with lookahead?

**See "Category 5: Requires Lookahead" section above.**

**Summary**:
1. **Song structure** (verse/chorus/bridge): Impossible without full song or metadata
2. **Lyrics sync**: Impossible without lyrics and timing data
3. **Pre-anticipating drops/climaxes**: Impossible without knowing future audio
4. **Global context** ("this is the climax of the song"): Impossible without knowing full song arc

**What you CAN do instead**:
- Build up RELATIVE context over time ("this is the loudest/brightest moment SO FAR")
- Use heuristics (if building for >5s, climax probably soon)
- Accept that real-time is REACTIVE, not PREDICTIVE — and that's OK!

---

## Recommended Implementation Order

### Phase 1: Core Real-Time Engine (Week 1)
1. Audio capture pipeline (BlackHole → Python)
2. Frame-by-frame feature extraction (FFT, mel spectrum, RMS, centroid)
3. Immediate detectors (bass peak, spectral flux)
4. LED command transmission (serial to Arduino)
5. **Test**: Beat-reactive flash on bass peaks

### Phase 2: Rhythmic Sync (Week 2)
6. Tempo estimation (IBI tracking)
7. Beat grid (phase-locked loop)
8. Flourish detection (off-beat events)
9. **Test**: Metronomic pulse + flourish sparkles

### Phase 3: Contextual Adaptation (Week 3)
10. Deviation detectors (airiness)
11. Flourish ratio tracker (mode selection)
12. Build detector (trajectory)
13. **Test**: Ambient mode (high flourish ratio) vs groove mode (low ratio)

### Phase 4: Advanced Detection (Week 4)
14. Section change detector
15. Drop detector (electronic music)
16. Climax predictor (build → climax heuristic)
17. **Test**: Full song with mode transitions, builds, climaxes

### Phase 5: Optimization & Tuning (Week 5)
18. Latency optimization (batch processing, vectorization)
19. Parameter tuning (thresholds, window sizes)
20. Genre-specific profiles (rock vs electronic vs ambient)
21. **Test**: Multiple genres, live performance

---

## Final Thoughts

**Key insight**: Most of our findings are REACTIVE, not PREDICTIVE — and that's OK.

**The LED system should**:
1. React instantly to beats, onsets, and energy changes (< 50ms)
2. Adapt to medium-term patterns like builds and mode changes (2-10s)
3. Use weak heuristics for anticipation (build → climax likely soon)
4. Accept that some things (song structure, drops) can only be reacted to

**The system CANNOT**:
1. Know the future
2. Understand song structure without metadata
3. Predict drops with certainty

**But the REACTION IS THE MAGIC**:
- A perfectly-timed reactive flash on a drop FEELS like anticipation (because 50ms is below perception threshold)
- Building up intensity during a detected build FEELS like you know what's coming
- Adapting mode based on flourish ratio FEELS like you understand the music

**The illusion of prediction is good enough.**

---

## Next Steps

1. **Implement Phase 1** (core real-time engine)
2. **Test latency** on actual hardware (measure end-to-end)
3. **Validate detectors** on our annotated segments
4. **Tune thresholds** per genre (rock, electronic, ambient)
5. **Build mode system** (ambient/accent/groove) with smooth transitions
6. **User testing** with live audio (user judges if effects "feel right")

---

*Analysis completed: 2026-02-06*
*Based on: LEARNINGS.md, beat_vs_consistent.md, build_vs_climax.md, air_feature_analysis.md, beat_tap_analysis.md, FLOURISH_FINDINGS_SUMMARY.md, flourish_audio_properties.md*
