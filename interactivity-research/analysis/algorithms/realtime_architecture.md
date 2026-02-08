# Real-Time Audio-Reactive Architecture for LEDs

**Date**: 2026-02-06

---

## The Core Problem

Audio analysis libraries are optimized for **offline analysis** (process entire file, look ahead/behind for context). LED systems need **real-time processing** (process audio as it arrives, no lookahead, <50ms latency).

### Offline vs Real-Time Algorithms

| Algorithm | Offline | Real-Time | Why |
|-----------|---------|-----------|-----|
| **librosa.beat_track()** | ✓ | ✗ | Dynamic programming needs full audio |
| **librosa.onset_detect()** | ✓ | ✓ | Can work with `backtrack=False` |
| **madmom RNNBeatProcessor** | ✓ | ✓ | Frame-by-frame RNN inference |
| **madmom DBNBeatTrackingProcessor** | ✓ | ⚠️ | Needs activation buffer (100-200 frames) |
| **madmom TempoEstimationProcessor** | ✓ | ⚠️ | Needs activation buffer for histogram |

✓ = Can process incrementally
⚠️ = Needs buffering (introduces latency)
✗ = Requires full audio

---

## Architecture #1: Onset-Based Reactive (Low Latency)

### Overview
Detect transients (onsets) and react immediately. No beat tracking, no tempo estimation.

### Pipeline
```
Audio Input (44.1kHz)
    ↓
Frame Buffer (2048 samples = 46ms)
    ↓
Windowing + FFT
    ↓
Mel Spectrogram (24 bands)
    ↓
Spectral Flux (onset strength)
    ↓
Peak Detection (threshold)
    ↓
Attack/Decay Smoothing
    ↓
LED Brightness/Color
```

### Latency Budget
- Audio buffering: 46ms (2048 samples @ 44.1kHz)
- FFT + Mel: ~1ms
- Onset detection: <1ms
- EMA smoothing: <1ms
- **Total: ~48ms** ✓ Under 50ms target

### Implementation (librosa)
```python
import numpy as np
import librosa

class OnsetReactiveLEDs:
    def __init__(self, sr=44100, hop_length=512, n_mels=24):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels

        # State for incremental processing
        self.prev_mel = None

        # EMA smoothing parameters
        self.alpha_attack = 0.9  # Fast attack
        self.alpha_decay = 0.3   # Slow decay
        self.smoothed_value = 0.0

    def process_chunk(self, audio_chunk):
        """
        Process one chunk of audio (e.g., 2048 samples).
        Returns: LED brightness value (0.0 to 1.0)
        """
        # Compute STFT
        stft = librosa.stft(audio_chunk, n_fft=2048, hop_length=self.hop_length)

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(stft)**2,
            sr=self.sr,
            n_mels=self.n_mels
        )

        # Onset strength (spectral flux)
        if self.prev_mel is not None:
            # Positive difference (increase in energy)
            flux = np.sum(np.maximum(0, mel_spec - self.prev_mel))
        else:
            flux = 0.0

        self.prev_mel = mel_spec

        # Normalize flux (empirical scaling)
        onset_strength = np.clip(flux / 1000.0, 0.0, 1.0)

        # EMA smoothing (asymmetric attack/decay)
        if onset_strength > self.smoothed_value:
            alpha = self.alpha_attack  # Fast attack
        else:
            alpha = self.alpha_decay   # Slow decay

        self.smoothed_value = alpha * onset_strength + (1 - alpha) * self.smoothed_value

        # Apply gamma correction for LED brightness
        led_brightness = self.smoothed_value ** (1/2.2)

        return led_brightness

    def process_chunk_multiband(self, audio_chunk):
        """
        Process chunk with frequency band separation (bass, mids, highs).
        Returns: dict with brightness for each band
        """
        # Compute STFT and mel spectrogram (same as above)
        stft = librosa.stft(audio_chunk, n_fft=2048, hop_length=self.hop_length)
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(stft)**2,
            sr=self.sr,
            n_mels=self.n_mels
        )

        if self.prev_mel is None:
            self.prev_mel = mel_spec
            return {'bass': 0.0, 'mids': 0.0, 'highs': 0.0}

        # Split into frequency bands
        # Mel bin 0-7 = ~20-250 Hz (bass)
        # Mel bin 8-15 = ~250-2000 Hz (mids)
        # Mel bin 16-23 = ~2000-8000 Hz (highs)
        bass_flux = np.sum(np.maximum(0, mel_spec[0:8] - self.prev_mel[0:8]))
        mids_flux = np.sum(np.maximum(0, mel_spec[8:16] - self.prev_mel[8:16]))
        highs_flux = np.sum(np.maximum(0, mel_spec[16:24] - self.prev_mel[16:24]))

        self.prev_mel = mel_spec

        return {
            'bass': np.clip(bass_flux / 500.0, 0.0, 1.0),
            'mids': np.clip(mids_flux / 500.0, 0.0, 1.0),
            'highs': np.clip(highs_flux / 500.0, 0.0, 1.0)
        }
```

### Tunable Parameters
- `hop_length`: Frame hop (512 = 12ms latency, 256 = 6ms latency)
- `n_mels`: Number of mel bands (24 = good balance, 64 = more detail)
- `alpha_attack`: Attack smoothing (0.9 = fast, 0.7 = slower)
- `alpha_decay`: Decay smoothing (0.3 = slow, 0.5 = faster)
- Flux threshold: Minimum energy change to trigger
- Normalization factor: Scale flux to 0-1 range

### Pros
- Low latency (~48ms)
- Fast processing (39x real-time)
- Simple to implement
- Works on any music genre
- No tempo/beat tracking needed

### Cons
- No beat synchronization
- Lots of false positives on complex music
- Doesn't capture "beat" vs "onset"
- No tempo information for periodic patterns

---

## Architecture #2: Beat-Synced Reactive (Buffered)

### Overview
Track beats and tempo, synchronize LED patterns to beat grid. Requires buffering for tempo estimation.

### Pipeline
```
Audio Input (44.1kHz)
    ↓
Frame Buffer (2048 samples = 46ms)
    ↓
RNN Beat Activation (madmom)
    ↓
Activation Buffer (2-4 seconds)
    ↓
Tempo Estimation (every 2s)
    ↓
Beat Phase Tracking (0.0 to 1.0 per beat)
    ↓
Beat-Synced Pattern Generator
    ↓
LED Pattern
```

### Latency Budget
- Audio buffering: 46ms
- RNN inference: ~10ms per frame
- Tempo buffer: 2000ms (required for accuracy)
- **Total: ~2046ms** ⚠️ Over 50ms, but acceptable for beat-synced effects

### Implementation (madmom)
```python
import madmom
import numpy as np
from collections import deque

class BeatSyncedLEDs:
    def __init__(self, buffer_duration=2.0):
        self.rnn = madmom.features.beats.RNNBeatProcessor()

        # State
        self.activation_buffer = deque(maxlen=int(buffer_duration * 100))  # 100 fps
        self.current_tempo = 120.0  # BPM
        self.beat_phase = 0.0       # 0.0 to 1.0 (position in current beat)
        self.last_beat_time = 0.0

        # Peak detection threshold
        self.beat_threshold = 0.3

    def process_activation_frame(self, activation_value, current_time):
        """
        Process one frame of beat activation.
        activation_value: float (0.0 to 1.0) from RNN
        current_time: float (seconds)
        """
        self.activation_buffer.append(activation_value)

        # Simple peak detection for immediate beat
        is_beat = False
        if (len(self.activation_buffer) >= 3 and
            activation_value > self.beat_threshold and
            activation_value > self.activation_buffer[-2] and
            activation_value > self.activation_buffer[-3]):

            # Beat detected!
            is_beat = True
            interval = current_time - self.last_beat_time
            if interval > 0.2:  # Ignore beats <200ms apart
                # Update tempo (exponential moving average)
                detected_tempo = 60.0 / interval
                self.current_tempo = 0.8 * self.current_tempo + 0.2 * detected_tempo
                self.last_beat_time = current_time
                self.beat_phase = 0.0

        # Update beat phase
        beat_period = 60.0 / self.current_tempo
        time_since_beat = current_time - self.last_beat_time
        self.beat_phase = (time_since_beat / beat_period) % 1.0

        return {
            'is_beat': is_beat,
            'beat_phase': self.beat_phase,
            'tempo': self.current_tempo,
            'activation': activation_value
        }

    def estimate_tempo_from_buffer(self):
        """
        Estimate tempo from activation buffer (call every 2 seconds).
        Uses autocorrelation to find periodic pattern.
        """
        if len(self.activation_buffer) < 200:
            return self.current_tempo

        activation = np.array(self.activation_buffer)

        # Autocorrelation
        autocorr = np.correlate(activation, activation, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags

        # Find peaks in autocorrelation (periodic pattern)
        # Look for peaks between 40-200 BPM (0.3 to 1.5 seconds at 100fps)
        min_lag = int(0.3 * 100)  # 0.3s = 200 BPM
        max_lag = int(1.5 * 100)  # 1.5s = 40 BPM

        peak_lag = np.argmax(autocorr[min_lag:max_lag]) + min_lag
        period_seconds = peak_lag / 100.0
        tempo = 60.0 / period_seconds

        # Smooth tempo update
        self.current_tempo = 0.7 * self.current_tempo + 0.3 * tempo

        return self.current_tempo

    def get_beat_synced_brightness(self, beat_phase, pattern='pulse'):
        """
        Generate LED brightness based on beat phase.
        beat_phase: 0.0 to 1.0 (position in current beat)
        pattern: 'pulse', 'fade', 'strobe', 'sine'
        """
        if pattern == 'pulse':
            # Pulse on beat, fade out
            return max(0.0, 1.0 - beat_phase)

        elif pattern == 'fade':
            # Fade in to beat, pulse at beat
            if beat_phase < 0.1:
                return 1.0  # Peak at beat
            else:
                return beat_phase  # Fade in

        elif pattern == 'strobe':
            # Strobe on beat
            return 1.0 if beat_phase < 0.1 else 0.0

        elif pattern == 'sine':
            # Sine wave (smooth oscillation)
            return 0.5 + 0.5 * np.sin(2 * np.pi * beat_phase - np.pi/2)

        else:
            return beat_phase
```

### Tunable Parameters
- `buffer_duration`: How long to buffer for tempo estimation (2-4 seconds)
- `beat_threshold`: Activation threshold for beat detection (0.2-0.4)
- `tempo_smoothing`: How much to smooth tempo updates (0.7-0.9)
- `min_beat_interval`: Minimum time between beats (200ms = 300 BPM max)
- Pattern type: pulse, fade, strobe, sine, etc.

### Pros
- Beat-synchronized patterns
- Tempo-aware effects
- Smooth beat phase interpolation
- Works well on steady tempo music

### Cons
- 2 second latency for accurate tempo
- Breaks on tempo changes
- RNN slower than simple onset detection
- Still has tempo doubling issue on rock

---

## Architecture #3: Hybrid (Best of Both)

### Overview
Use onset detection for immediate reaction, tempo estimation for beat-synced patterns. Combine both signals.

### Pipeline
```
                    Audio Input
                         ↓
                 Frame Buffer (2048)
                    ↙         ↘
        [Fast Path]           [Slow Path]
         Onset Detection       RNN Beat Activation
              ↓                      ↓
         EMA Smoothing         Activation Buffer
              ↓                      ↓
      Reactive Brightness      Tempo Estimation
              ↓                      ↓
              └──────→ Mix ←────────┘
                         ↓
                    LED Output
```

### Implementation
```python
class HybridReactiveLEDs:
    def __init__(self):
        self.onset_detector = OnsetReactiveLEDs()
        self.beat_tracker = BeatSyncedLEDs()

        # Mix parameters
        self.reactive_weight = 0.7   # Weight for onset-reactive
        self.beat_weight = 0.3       # Weight for beat-synced

    def process_frame(self, audio_chunk, beat_activation, current_time):
        """
        Process one frame with both onset and beat tracking.
        """
        # Fast path: onset detection
        onset_brightness = self.onset_detector.process_chunk(audio_chunk)

        # Slow path: beat tracking
        beat_info = self.beat_tracker.process_activation_frame(
            beat_activation,
            current_time
        )

        # Generate beat-synced pattern
        beat_brightness = self.beat_tracker.get_beat_synced_brightness(
            beat_info['beat_phase'],
            pattern='pulse'
        )

        # Mix both signals
        # Option 1: Weighted average
        mixed = self.reactive_weight * onset_brightness + self.beat_weight * beat_brightness

        # Option 2: Maximum (take brightest)
        # mixed = max(onset_brightness, beat_brightness)

        # Option 3: Multiply (beat modulates onset)
        # mixed = onset_brightness * (0.5 + 0.5 * beat_brightness)

        return {
            'brightness': mixed,
            'onset': onset_brightness,
            'beat': beat_brightness,
            'tempo': beat_info['tempo'],
            'is_beat': beat_info['is_beat']
        }

    def set_mode(self, mode):
        """
        Switch between modes dynamically.
        mode: 'reactive', 'beat_synced', 'hybrid'
        """
        if mode == 'reactive':
            self.reactive_weight = 1.0
            self.beat_weight = 0.0
        elif mode == 'beat_synced':
            self.reactive_weight = 0.0
            self.beat_weight = 1.0
        elif mode == 'hybrid':
            self.reactive_weight = 0.7
            self.beat_weight = 0.3
```

### Pros
- Low latency for immediate reaction
- Beat synchronization for patterns
- Flexible mixing strategies
- Can switch modes dynamically

### Cons
- More complex code
- Two processing pipelines
- Need to tune mixing weights

---

## Performance Considerations

### Processing Speed (from tests)
- **librosa onset detection**: 855 chunks/sec (39x real-time)
- **madmom RNN**: ~17x real-time (2.4s for 40s audio)

### Memory Usage
- **Onset detector**: ~1 MB (one mel spectrogram frame)
- **Beat tracker**: ~20 MB (activation buffer + RNN model)

### CPU Usage (estimated)
- **Onset detection**: ~5% on modern CPU
- **RNN inference**: ~15-20% on modern CPU

### Latency Breakdown

**Onset-Based (48ms total)**
- Audio buffering: 46ms (fixed by chunk size)
- Processing: 2ms

**Beat-Synced (2046ms total)**
- Audio buffering: 46ms
- Activation buffering: 2000ms (for accurate tempo)
- Processing: <10ms

**Hybrid (2046ms for tempo, 48ms for reactive)**
- Immediate reaction from onset path
- Tempo converges after 2 seconds

---

## Comparison to Original nebula streaming code

From the user's memory, the old code had problems:
1. **Self-normalizing history** - Bass stayed at 1.0, only detected changes
2. **Per-frame normalization** - `spectrum / max(spectrum)` killed absolute levels

### How these architectures fix it

**Onset Detection**
- Uses **spectral flux** (change in spectrum), not absolute level
- But flux is computed incrementally, not normalized per-frame
- Maintains attack/decay EMA for smooth response
- Doesn't normalize to max - uses fixed scaling factor

**Beat Tracking**
- Uses **RNN activation** trained on diverse music
- RNN output is calibrated probability (not normalized)
- Doesn't rely on absolute bass level
- Peak picking uses fixed threshold, not relative

### Key Difference
Old code: `spectrum / max(spectrum)` → all frames normalized to 0-1
New code: `flux = max(0, current - previous)` → absolute change measured

---

## Recommendations

### For MVP (Minimum Viable Product)
**Use Architecture #1: Onset-Based**
- Simple to implement
- Low latency
- Works on any music
- Good enough for "energy" reactive effects

### For Production (Best User Experience)
**Use Architecture #3: Hybrid**
- Onset detection for immediate reaction
- Beat tracking for synchronized patterns
- User can choose mode (reactive vs beat-synced vs hybrid)
- Best of both worlds

### For Specific Genres
- **Electronic/EDM**: Beat-synced works great (steady tempo)
- **Rock/Metal**: Onset-based better (tempo doubling issues)
- **Jazz/Classical**: Onset-based (tempo changes)
- **Hip-hop**: Hybrid (strong beat + lots of percussion)

---

## Next Steps for Implementation

1. **Test onset detection on live audio**
   - Implement chunked processing
   - Tune attack/decay parameters
   - Test on different music genres

2. **Test RNN beat activation incremental processing**
   - madmom RNN should work frame-by-frame
   - Verify real-time performance
   - Test simple peak picking vs DBN

3. **Build hybrid system**
   - Combine both pipelines
   - Test mixing strategies
   - Add mode switching

4. **Optimize for target hardware**
   - Profile CPU usage
   - Optimize buffer sizes
   - Consider GPU acceleration for RNN

5. **Add "feel" features**
   - Spectral centroid (brightness)
   - Spectral flatness (noise vs tone)
   - RMS energy (loudness)
   - Combine with user annotations

---

## References

- **librosa documentation**: https://librosa.org/doc/latest/index.html
- **madmom documentation**: https://madmom.readthedocs.io/
- **Research paper**: "Deep Learning for Beat and Downbeat Tracking" (Böck et al., 2016)
- **User's annotation tool**: `/Users/KO16K39/Documents/led/interactivity-research/tools/annotate_segment.py`
