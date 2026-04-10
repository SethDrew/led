"""
Reusable signal processing primitives for audio-reactive effects.

Building blocks that effects compose via has-a:

  OverlapFrameAccumulator — feeds audio chunks, yields overlapped frames
  AbsIntegral             — computes normalized abs-integral of RMS derivative
  BeatPredictor           — autocorrelation tempo + predicted/confirmed beats
  OnsetTempoTracker       — onset-envelope autocorrelation for tempo estimation
  SpeedSignal             — spectral evolution rate (ambient-friendly 0-1 speed)
  detect_timbral_sections — batch MFCC novelty detector with eagerness curve

Usage:
    accum = OverlapFrameAccumulator()
    absint = AbsIntegral(sample_rate=44100)
    predictor = BeatPredictor(rms_fps=absint.rms_fps)

    # For tempo-only (no individual beat detection):
    tracker = OnsetTempoTracker(sample_rate=44100)

    def process_audio(self, chunk):
        for frame in accum.feed(chunk):
            normalized = absint.update(frame)
            beats = predictor.feed(normalized, absint.time_acc)
            tracker.feed_frame(frame)
            ...
"""

import numpy as np


class OverlapFrameAccumulator:
    """Accumulates audio chunks into overlapped analysis frames.

    Handles the bookkeeping of filling a buffer, yielding complete frames,
    and shifting by hop_size for overlap. Every effect that processes RMS
    or FFT frames uses this same loop.
    """

    def __init__(self, frame_len: int = 2048, hop: int = 512):
        self.frame_len = frame_len
        self.hop = hop
        self.buf = np.zeros(frame_len, dtype=np.float32)
        self.pos = 0

    def feed(self, chunk: np.ndarray) -> list:
        """Feed an audio chunk, return list of complete overlapped frames."""
        frames = []
        n = len(chunk)
        pos = self.pos
        offset = 0

        while n > 0:
            space = self.frame_len - pos
            take = min(n, space)
            self.buf[pos:pos + take] = chunk[offset:offset + take]
            offset += take
            pos += take
            n -= take

            if pos >= self.frame_len:
                frames.append(self.buf.copy())
                self.buf[:self.frame_len - self.hop] = self.buf[self.hop:]
                pos = self.frame_len - self.hop

        self.pos = pos
        return frames


class StickyFloorRMS:
    """RMS normalization via sticky noise floor + log dB mapping.

    Ported from the bulbs firmware (gyro_mic_fade.cpp). Three-stage pipeline:

    1. Sticky floor EMA — tracks the ambient noise level.
       Fast down (4x alpha), very sticky up (0.1x alpha).
       During constant music the floor slowly creeps upward, causing the
       output to decay. Transient spikes punch through because they're
       far above the rising floor.

    2. Log dB above floor — 20 * log10((rms - floor) / floor), clamped
       to a dB_window (default 15 dB) and mapped to [0, 1].

    3. Output is a 0-1 value suitable for brightness mapping.

    Key property: constant energy → signal decays (floor catches up).
    This is fundamentally different from peak-decay normalization where
    constant energy → signal stays at ~1.0 forever.
    """

    def __init__(self, floor_tc: float = 10.0, fps: float = 86.0,
                 db_window: float = 15.0,
                 up_mult: float = 0.1, down_mult: float = 4.0,
                 warm_start: bool = False):
        """
        Args:
            floor_tc:   Base time constant for floor EMA in seconds.
            fps:        Rate at which update() is called (audio frames/sec).
                        Default 86 ≈ 44100/512 for standard hop.
            db_window:  dB range mapped to 0-1. Default 15 dB means the
                        signal hits 1.0 when rms is ~30x above the floor.
            up_mult:    Multiplier on alpha for floor rising (sticky).
                        0.1 = very sticky (~50s to adapt). 1.0 = same as base TC.
            down_mult:  Multiplier on alpha for floor falling (fast).
                        4.0 = drops ~4x faster than base TC.
            warm_start: If True, initialize floor to first RMS reading so
                        output starts near zero and ramps up gradually.
                        Default False = start floor near zero (eager/hot start).
        """
        self._alpha_base = 2.0 / (floor_tc * fps + 1.0)
        self._db_window = db_window
        self._up_mult = up_mult
        self._down_mult = down_mult
        self._warm_start = warm_start
        self._floor = 0.0
        self._initialized = False
        self.value = 0.0  # latest 0-1 output

    def update(self, frame: np.ndarray) -> float:
        """Process one audio frame, return 0-1 normalized value."""
        rms = float(np.sqrt(np.mean(frame ** 2)))

        if not self._initialized:
            if self._warm_start:
                # Reluctant start: floor = first reading, output starts near 0
                # and ramps up as floor adapts. Good for effects that should
                # fade in rather than blast on immediately.
                self._floor = max(rms, 1e-6)
            else:
                # Eager start: floor near zero, output spikes immediately.
                self._floor = 1e-6
            self._initialized = True

        # Asymmetric floor EMA: fast down, sticky up
        if rms < self._floor:
            self._floor += self._alpha_base * self._down_mult * (rms - self._floor)
        else:
            self._floor += self._alpha_base * self._up_mult * (rms - self._floor)

        # Prevent floor from hitting zero
        floor = max(self._floor, 1e-10)

        # Log dB above floor → [0, 1]
        above_floor = max(rms - floor, 1e-10)
        db = 20.0 * np.log10(above_floor / floor)
        self.value = float(np.clip(db / self._db_window, 0.0, 1.0))

        return self.value


class AbsIntegral:
    """Computes normalized absolute integral of RMS derivative.

    The core signal used by most absint_* effects. Tracks how much the
    loudness is *changing* over a short window (default 150ms).

    Returns a normalized 0-1 value via update(). Also exposes raw values
    for effects that need them (e.g. for autocorrelation input).
    """

    def __init__(self, frame_len: int = 2048, sample_rate: int = 44100,
                 window_sec: float = 0.15, peak_decay: float = 0.998):
        self.frame_len = frame_len
        self.sample_rate = sample_rate
        self.dt = frame_len / sample_rate
        self.rms_fps = sample_rate / 512  # hop-based FPS (for autocorrelation)

        self.prev_rms = 0.0

        # Ring buffer of |d(RMS)/dt| values
        self.window_frames = max(1, int(window_sec / self.dt))
        self.deriv_buf = np.zeros(self.window_frames, dtype=np.float32)
        self.deriv_buf_pos = 0

        # Signal state
        self.raw = 0.0              # raw abs-integral value
        self.peak = 1e-10           # slow-decay peak for normalization
        self.peak_decay = peak_decay
        self.normalized = 0.0       # 0-1 normalized value

        # Time tracking
        self.time_acc = 0.0
        self.rms_dt = 512 / sample_rate  # time per hop

    def update(self, frame: np.ndarray) -> float:
        """Process one frame, return normalized 0-1 value."""
        rms = np.sqrt(np.mean(frame ** 2))

        # RMS derivative
        rms_deriv = (rms - self.prev_rms) / self.dt
        self.prev_rms = rms

        # Store |derivative| in ring buffer
        self.deriv_buf[self.deriv_buf_pos % self.window_frames] = abs(rms_deriv)
        self.deriv_buf_pos += 1

        # Abs-integral: sum of ring buffer * dt
        self.raw = np.sum(self.deriv_buf) * self.dt

        # Slow-decay peak normalization
        self.peak = max(self.raw, self.peak * self.peak_decay)
        self.normalized = self.raw / self.peak if self.peak > 0 else 0.0

        # Advance time
        self.time_acc += self.rms_dt

        return self.normalized

    @property
    def current_rms(self):
        """Last computed RMS value."""
        return self.prev_rms


class BeatPredictor:
    """Autocorrelation-based tempo estimation and beat prediction.

    Feed it the abs-integral signal value each frame. It detects confirmed
    beats (threshold crossing), estimates tempo via autocorrelation, and
    predicts future beats.

    Returns a list of beat events per frame:
      [{'type': 'confirmed', 'strength': 0.85}, ...]
      [{'type': 'predicted', 'strength': 0.80}, ...]
      []  # no beat this frame
    """

    def __init__(self, rms_fps: float, threshold: float = 0.30,
                 cooldown: float = 0.25, ac_window_sec: float = 5.0,
                 min_bpm: float = 40, max_bpm: float = 300,
                 confidence_threshold: float = 0.3,
                 predicted_strength: float = 0.8,
                 max_missed: int = 4):
        self.rms_fps = rms_fps
        self.threshold = threshold
        self.cooldown = cooldown
        self.predicted_strength = predicted_strength
        self.max_missed = max_missed

        # Beat detection state
        self.last_beat_time = -1.0
        self.beat_count = 0
        self.predicted_beat_count = 0

        # Autocorrelation buffer
        self.ac_window_frames = int(ac_window_sec * rms_fps)
        self.ac_buf = np.zeros(self.ac_window_frames, dtype=np.float32)
        self.ac_buf_pos = 0
        self.ac_buf_filled = 0

        # Period bounds
        self.min_period_frames = max(1, int(rms_fps * 60.0 / max_bpm))
        self.max_period_frames = int(rms_fps * 60.0 / min_bpm)

        # Autocorrelation result
        self.confidence = 0.0
        self.min_confidence = confidence_threshold
        self.estimated_period = 0.0  # seconds

        # Prediction state
        self.next_predicted_beat = 0.0
        self.prediction_active = False
        self.last_detection_time = -1.0
        self.missed_beats = 0

    def feed(self, abs_integral_value: float, normalized: float,
             time_acc: float) -> list:
        """Feed one frame of signal. Returns list of beat events."""
        events = []

        # Store for autocorrelation
        self.ac_buf[self.ac_buf_pos % self.ac_window_frames] = abs_integral_value
        self.ac_buf_pos += 1
        self.ac_buf_filled = min(self.ac_buf_filled + 1, self.ac_window_frames)

        # Late beat detection
        time_since_beat = time_acc - self.last_beat_time
        beat_detected = False

        if normalized > self.threshold and time_since_beat > self.cooldown:
            beat_detected = True
            self.last_beat_time = time_acc
            self.last_detection_time = time_acc
            self.beat_count += 1

            self._update_autocorrelation()
            self._update_prediction_phase()

            events.append({
                'type': 'confirmed',
                'strength': min(1.0, normalized),
            })

        # Check predicted beats
        if not beat_detected:
            pred = self._check_predicted_beat(time_acc)
            if pred:
                events.append(pred)

        return events

    def _update_autocorrelation(self):
        """Compute autocorrelation of abs-integral signal to estimate period."""
        if self.ac_buf_filled < self.min_period_frames * 3:
            return

        n = self.ac_buf_filled
        if n >= self.ac_window_frames:
            start = self.ac_buf_pos % self.ac_window_frames
            signal = np.concatenate([self.ac_buf[start:], self.ac_buf[:start]])
        else:
            signal = self.ac_buf[:n].copy()

        signal = signal - np.mean(signal)
        norm = np.dot(signal, signal)
        if norm < 1e-20:
            return

        min_lag = self.min_period_frames
        max_lag = min(self.max_period_frames, len(signal) // 2)
        if min_lag >= max_lag:
            return

        autocorr = np.zeros(max_lag - min_lag, dtype=np.float64)
        for i, lag in enumerate(range(min_lag, max_lag)):
            autocorr[i] = np.dot(signal[:-lag], signal[lag:]) / norm

        best_lag = -1
        best_corr = 0.0
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > best_corr:
                    best_corr = autocorr[i]
                    best_lag = min_lag + i
                    if best_corr > self.min_confidence:
                        break

        self.confidence = best_corr

        if best_corr > self.min_confidence and best_lag > 0:
            new_period = best_lag / self.rms_fps

            # Octave correction + smoothing (from tempo_pulse's mature version)
            if self.estimated_period > 0:
                ratio = new_period / self.estimated_period
                if 0.8 < ratio < 1.2:
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * new_period
                elif 0.45 < ratio < 0.55:
                    # Detected half-period — double it
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * (new_period * 2)
                elif 1.8 < ratio < 2.2:
                    # Detected double-period — halve it
                    self.estimated_period = 0.8 * self.estimated_period + 0.2 * (new_period / 2)
                # else: too far off, ignore this estimate
            else:
                self.estimated_period = new_period

    def _update_prediction_phase(self):
        """Phase-lock prediction to the most recent late detection."""
        if self.estimated_period <= 0 or self.confidence < self.min_confidence:
            self.prediction_active = False
            return
        self.next_predicted_beat = self.last_detection_time + self.estimated_period
        self.prediction_active = True
        self.missed_beats = 0

    def _check_predicted_beat(self, time_acc: float):
        """Check if a predicted beat should fire now."""
        if not self.prediction_active or self.estimated_period <= 0:
            return None

        if time_acc >= self.next_predicted_beat:
            time_since_detection = time_acc - self.last_detection_time
            if time_since_detection < self.estimated_period * 0.3:
                # Too close to a real detection — skip
                self.next_predicted_beat += self.estimated_period
                self.missed_beats = 0
                return None

            self.predicted_beat_count += 1
            self.next_predicted_beat += self.estimated_period
            self.missed_beats += 1

            if self.missed_beats >= self.max_missed:
                self.prediction_active = False
                self.missed_beats = 0

            return {
                'type': 'predicted',
                'strength': self.predicted_strength,
            }
        return None

    @property
    def bpm(self) -> float:
        """Current estimated BPM, or 0 if unknown."""
        return 60.0 / self.estimated_period if self.estimated_period > 0 else 0.0


class OnsetTempoTracker:
    """Tempo estimation via onset-envelope autocorrelation.

    Designed for music where the abs-integral beat detector struggles —
    dense percussive content like rap, where constant hi-hats and 808s
    make the abs-integral signal noisy. Instead of detecting individual
    beats, this estimates the underlying tempo period.

    == How autocorrelation works ==

    Take 5 seconds of the onset signal. Make a copy and slide it forward
    by some amount ("lag"). Multiply the original and shifted copy
    point-by-point, sum the products. If the signal repeats at that lag,
    peaks align with peaks → big positive sum. If the lag is wrong,
    peaks hit valleys → small or negative sum.

    Try every plausible lag (corresponding to 40-300 BPM). The lag with
    the biggest sum is the estimated beat period.

    The catch: if the beat repeats every T, there's also correlation at
    2T, 3T, etc. (sub-harmonics). A kick every 2 beats creates a peak
    at half-tempo. This "octave ambiguity" is inherent — autocorrelation
    can't tell which is the "real" beat. We use a Gaussian tempo prior
    to bias toward plausible tempos.

    == Signal: multi-band onset envelope ==

    Per frame: FFT → group into 6 mel-spaced frequency bands → log
    energy per band → first-order difference → half-wave rectify
    (only energy increases = onsets) → mean across bands.

    This is a streaming approximation of librosa.onset.onset_strength.
    The multi-band decomposition matters: a kick at 80Hz and a hi-hat
    at 8kHz both register as separate onsets in their bands rather than
    cancelling out in a single-band RMS. 6 bands is enough — tested
    against 128 mel bands, same autocorrelation peak structure.

    Why not abs-integral (|d(RMS)/dt| summed over a window)?
    The abs-integral counts both energy arriving AND leaving. In rap,
    kick and snare create symmetric bump-dip patterns that the absolute
    value merges, destroying onset timing. The onset envelope preserves
    the one-sided onset peaks that autocorrelation needs.

    Tested on 5 tracks (82-146 BPM rap + 129 BPM glue), streaming:
      - Without prior: always locks to a clean power-of-2 multiple
        (0.5x or 2x) of the true tempo. Consistent and predictable.
      - With prior: can get exact tempo but also introduces non-octave
        errors (2/3, 4/3) when the prior center doesn't match the track.
        Worse for LED effects where any octave is fine.

    == No prior by default ==

    Without a prior, autocorrelation picks the strongest peak, which
    is always a sub-harmonic (power-of-2 related to the true beat).
    For tempo-driven LED effects this is ideal: the estimated period
    is always a clean multiple of the real beat, so fades stay locked
    to the music's periodicity regardless of which octave was chosen.

    A prior (prior_sigma < 99999) can be enabled to target exact BPM,
    but risks non-octave errors when the prior doesn't match the track.

    == Output ==

    estimated_period (seconds), confidence (0-1), bpm.
    No beat events — this is for effects that need tempo, not timing.
    """

    def __init__(self, sample_rate: int = 44100, frame_len: int = 2048,
                 ac_window_sec: float = 5.0, update_interval_sec: float = 0.5,
                 min_bpm: float = 40, max_bpm: float = 300,
                 prior_center: float = 100, prior_sigma: float = 99999,
                 n_bands: int = 6, full_wave: bool = False):
        self.sample_rate = sample_rate
        self.full_wave = full_wave
        self.frame_len = frame_len
        self.dt = frame_len / sample_rate
        self.rms_fps = sample_rate / 512  # hop-based rate
        self.rms_dt = 512 / sample_rate

        # Multi-band onset envelope state.
        # We split the FFT into n_bands mel-spaced bands and track
        # spectral flux (positive energy increase) in each band, then
        # average across bands. This is a streaming approximation of
        # librosa.onset.onset_strength.
        self.n_bands = n_bands
        fft_size = frame_len // 2 + 1
        freqs = np.linspace(0, sample_rate / 2, fft_size)
        # Mel-spaced band edges
        mel_lo = 2595 * np.log10(1 + 20 / 700)
        mel_hi = 2595 * np.log10(1 + (sample_rate / 2) / 700)
        mel_edges = np.linspace(mel_lo, mel_hi, n_bands + 1)
        hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
        # Precompute bin ranges for each band
        self._band_slices = []
        for b in range(n_bands):
            lo_bin = np.searchsorted(freqs, hz_edges[b])
            hi_bin = np.searchsorted(freqs, hz_edges[b + 1])
            hi_bin = max(hi_bin, lo_bin + 1)  # at least 1 bin
            self._band_slices.append(slice(lo_bin, hi_bin))
        self._prev_band_energy = np.zeros(n_bands, dtype=np.float64)

        # Ring buffer of onset values for autocorrelation
        self.ac_window_frames = int(ac_window_sec * self.rms_fps)
        self.onset_buf = np.zeros(self.ac_window_frames, dtype=np.float32)
        self.buf_pos = 0
        self.buf_filled = 0

        # How often to re-run autocorrelation (in frames)
        self.update_interval = int(update_interval_sec * self.rms_fps)
        self.frames_since_update = 0

        # Autocorrelation lag bounds
        self.min_lag = max(1, int(self.rms_fps * 60.0 / max_bpm))
        self.max_lag = int(self.rms_fps * 60.0 / min_bpm)

        # Tempo prior: precompute Gaussian weights for each lag
        self._prior_weights = np.zeros(self.max_lag - self.min_lag)
        for i, lag in enumerate(range(self.min_lag, self.max_lag)):
            bpm = 60.0 * self.rms_fps / lag
            self._prior_weights[i] = np.exp(
                -0.5 * ((bpm - prior_center) / prior_sigma) ** 2
            )

        # Result state
        self.estimated_period = 0.0  # seconds
        self.confidence = 0.0
        self.time_acc = 0.0

    def feed_frame(self, frame: np.ndarray):
        """Feed one audio frame (2048 samples). Call once per frame from
        OverlapFrameAccumulator."""

        # Multi-band onset envelope:
        # 1. FFT → magnitude spectrum
        # 2. Sum energy in each mel-spaced band (log scale)
        # 3. First-order difference vs previous frame (spectral flux)
        # 4. Half-wave rectify (only increases = onsets)
        # 5. Mean across bands
        spectrum = np.abs(np.fft.rfft(frame))
        band_energy = np.zeros(self.n_bands, dtype=np.float64)
        for b, sl in enumerate(self._band_slices):
            band_energy[b] = np.log1p(np.sum(spectrum[sl] ** 2))
        flux = band_energy - self._prev_band_energy
        self._prev_band_energy = band_energy
        # Rectify + mean: onset strength this frame
        # Half-wave (default): only energy increases (onsets)
        # Full-wave: both increases and decreases (onsets + offsets)
        onset = np.mean(np.abs(flux)) if self.full_wave else np.mean(np.maximum(0, flux))

        # Store in ring buffer
        self.onset_buf[self.buf_pos % self.ac_window_frames] = onset
        self.buf_pos += 1
        self.buf_filled = min(self.buf_filled + 1, self.ac_window_frames)
        self.time_acc += self.rms_dt

        # Periodically run autocorrelation
        self.frames_since_update += 1
        if self.frames_since_update >= self.update_interval:
            self.frames_since_update = 0
            self._update_autocorrelation()

    def _update_autocorrelation(self):
        """Run autocorrelation on the onset buffer, find best period."""
        # Need enough data for autocorrelation to be meaningful.
        # min_lag * 3 is the mathematical minimum, but in practice we need
        # at least a full window for reliable peaks. Without this, the first
        # estimate from a nearly-empty buffer can be garbage and then the
        # smoothing logic permanently rejects the correct tempo.
        if self.buf_filled < self.ac_window_frames:
            return

        # Unroll ring buffer into contiguous array
        n = self.buf_filled
        if n >= self.ac_window_frames:
            start = self.buf_pos % self.ac_window_frames
            signal = np.concatenate([self.onset_buf[start:],
                                     self.onset_buf[:start]])
        else:
            signal = self.onset_buf[:n].copy()

        # Subtract mean so autocorrelation measures periodicity, not DC offset
        signal = signal - np.mean(signal)
        norm = np.dot(signal, signal)
        if norm < 1e-20:
            return

        max_lag = min(self.max_lag, len(signal) // 2)
        if self.min_lag >= max_lag:
            return

        n_lags = max_lag - self.min_lag

        # Core autocorrelation: for each candidate lag, compute
        #   sum(signal[t] * signal[t + lag]) / norm
        # High value = signal correlates well with itself shifted by that lag.
        autocorr = np.zeros(n_lags, dtype=np.float64)
        for i, lag in enumerate(range(self.min_lag, max_lag)):
            autocorr[i] = np.dot(signal[:-lag], signal[lag:]) / norm

        # Find peaks (local maxima) in the autocorrelation
        # Weight each peak by the tempo prior
        best_score = -1.0
        best_lag = -1
        best_raw_conf = 0.0

        for i in range(1, n_lags - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > 0.05:
                    score = autocorr[i] * self._prior_weights[i]
                    if score > best_score:
                        best_score = score
                        best_lag = self.min_lag + i
                        best_raw_conf = autocorr[i]

        if best_lag < 0:
            return

        self.confidence = best_raw_conf
        new_period = best_lag / self.rms_fps

        # Smooth into existing estimate (80/20 blend),
        # with octave correction for 2x/0.5x jumps.
        if self.estimated_period > 0:
            ratio = new_period / self.estimated_period
            if 0.8 < ratio < 1.2:
                self.estimated_period = 0.8 * self.estimated_period + 0.2 * new_period
            elif 0.45 < ratio < 0.55:
                self.estimated_period = (0.8 * self.estimated_period
                                         + 0.2 * (new_period * 2))
            elif 1.8 < ratio < 2.2:
                self.estimated_period = (0.8 * self.estimated_period
                                         + 0.2 * (new_period / 2))
            # else: too far off, ignore this estimate
        else:
            self.estimated_period = new_period

    @property
    def bpm(self) -> float:
        """Current estimated BPM, or 0 if unknown."""
        return 60.0 / self.estimated_period if self.estimated_period > 0 else 0.0




class SpeedSignal:
    """Spectral evolution rate — ambient-friendly speed signal.

    Blends spectral flux, centroid rate, and chroma flux into a single
    0-1 value. Works on ambient music where tempo tracking fails.

    Three uncorrelated spectral features are computed per frame:
      1. Spectral flux (multi-band): FFT → 6 mel bands → log energy →
         diff → half-wave rectify → mean. Same band setup as
         OnsetTempoTracker.feed_frame().
      2. Centroid rate of change: |centroid[t] - centroid[t-1]| where
         centroid = weighted mean of FFT magnitudes.
      3. Chroma flux: L2 distance between consecutive 12-bin chroma
         vectors. Chroma L2 distance between consecutive frames.

    Each feature is peak-normalized to 0-1 (fast attack, configurable
    decay), then averaged and EMA-smoothed.
    """

    def __init__(self, sample_rate=44100, frame_len=2048,
                 ema_seconds=1.5, peak_decay=0.998, n_bands=6):
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self._peak_decay = peak_decay

        # EMA smoothing: alpha from time constant
        fps = sample_rate / 512  # hop-based frame rate
        self._ema_alpha = 2.0 / (ema_seconds * fps + 1.0)

        # ── Spectral flux state (multi-band, same as OnsetTempoTracker) ──
        fft_size = frame_len // 2 + 1
        freqs = np.linspace(0, sample_rate / 2, fft_size)
        mel_lo = 2595 * np.log10(1 + 20 / 700)
        mel_hi = 2595 * np.log10(1 + (sample_rate / 2) / 700)
        mel_edges = np.linspace(mel_lo, mel_hi, n_bands + 1)
        hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
        self._band_slices = []
        for b in range(n_bands):
            lo_bin = np.searchsorted(freqs, hz_edges[b])
            hi_bin = np.searchsorted(freqs, hz_edges[b + 1])
            hi_bin = max(hi_bin, lo_bin + 1)
            self._band_slices.append(slice(lo_bin, hi_bin))
        self._prev_band_energy = np.zeros(n_bands, dtype=np.float64)
        self._flux_peak = 1e-10

        # ── Centroid rate state ──
        self._prev_centroid = 0.0
        self._centroid_peak = 1e-10
        # Precompute frequency weights for centroid calculation
        self._freqs = np.linspace(0, sample_rate / 2, fft_size)

        # ── Chroma flux state ──
        self._prev_chroma = np.zeros(12, dtype=np.float64)
        self._chroma_peak = 1e-10
        chroma_freqs = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)
        self._chroma_bins = [[] for _ in range(12)]
        for i, f in enumerate(chroma_freqs):
            if f < 65 or f > sample_rate / 2 * 0.9:
                continue
            midi = 69 + 12 * np.log2(f / 440.0)
            pc = int(round(midi)) % 12
            self._chroma_bins[pc].append(i)

        # ── Output state ──
        self.value = 0.0            # latest 0-1 speed value (EMA-smoothed)
        self.spectral_flux = 0.0    # raw normalized spectral flux
        self.centroid_rate = 0.0    # raw normalized centroid rate
        self.chroma_flux = 0.0      # raw normalized chroma flux

    def update(self, frame: np.ndarray) -> float:
        """Process one audio frame, return 0-1 speed value."""
        spectrum = np.abs(np.fft.rfft(frame))

        # ── 1. Spectral flux (multi-band, half-wave rectified) ──
        n_bands = len(self._band_slices)
        band_energy = np.zeros(n_bands, dtype=np.float64)
        for b, sl in enumerate(self._band_slices):
            band_energy[b] = np.log1p(np.sum(spectrum[sl] ** 2))
        flux = band_energy - self._prev_band_energy
        self._prev_band_energy = band_energy
        raw_flux = float(np.mean(np.maximum(0, flux)))
        # Peak-normalize
        self._flux_peak = max(raw_flux, self._flux_peak * self._peak_decay)
        self.spectral_flux = (raw_flux / self._flux_peak
                              if self._flux_peak > 1e-10 else 0.0)

        # ── 2. Centroid rate of change ──
        mag_sum = np.sum(spectrum)
        if mag_sum > 1e-10:
            centroid = float(np.sum(self._freqs * spectrum) / mag_sum)
        else:
            centroid = 0.0
        raw_centroid_rate = abs(centroid - self._prev_centroid)
        self._prev_centroid = centroid
        self._centroid_peak = max(raw_centroid_rate,
                                  self._centroid_peak * self._peak_decay)
        self.centroid_rate = (raw_centroid_rate / self._centroid_peak
                              if self._centroid_peak > 1e-10 else 0.0)

        # ── 3. Chroma flux (L2 distance) ──
        chroma = np.zeros(12, dtype=np.float64)
        for pc in range(12):
            bins = self._chroma_bins[pc]
            if bins:
                chroma[pc] = np.sum(spectrum[bins] ** 2)
        chroma = np.log1p(chroma)
        total = np.sum(chroma)
        if total > 1e-10:
            chroma /= total
        raw_chroma_flux = float(np.linalg.norm(chroma - self._prev_chroma))
        self._prev_chroma = chroma.copy()
        self._chroma_peak = max(raw_chroma_flux,
                                self._chroma_peak * self._peak_decay)
        self.chroma_flux = (raw_chroma_flux / self._chroma_peak
                            if self._chroma_peak > 1e-10 else 0.0)

        # ── Blend and smooth ──
        raw_speed = (self.spectral_flux + self.centroid_rate
                     + self.chroma_flux) / 3.0
        self.value += self._ema_alpha * (raw_speed - self.value)

        return self.value


class FixedRangeRMS:
    """Fixed-range log mapping from raw RMS to 0-1.

    Maps a calibrated RMS range to brightness via log scale. No adaptive
    floor — same input always produces the same output. A slow peak-decay
    ceiling handles unexpectedly loud input without clipping.

    Based on INMP441 sensitivity (-26 dBFS at 94 dB SPL) and typical
    speech levels at arm's length (~30cm):
        Quiet speech  ~65 dB SPL → RMS ~0.002
        Normal speech ~75 dB SPL → RMS ~0.007
        Loud speech   ~85 dB SPL → RMS ~0.02
        Shouting      ~95 dB SPL → RMS ~0.06

    The floor acts as a hard gate (below = 0). The ceiling defines full
    brightness. Above the ceiling, a slow peak decay gently expands the
    range so loud input still maps to ~1.0 rather than clipping.
    """

    def __init__(self, floor_rms: float = 0.005, ceiling_rms: float = 0.06,
                 peak_decay: float = 0.9999, fps: float = 86.0):
        """
        Args:
            floor_rms:    Hard gate — below this RMS, output is 0.
            ceiling_rms:  Base ceiling — this RMS maps to 1.0.
            peak_decay:   Per-frame decay for the adaptive ceiling.
                          0.9999 at 86fps ≈ 80s half-life.
            fps:          Frame rate for time-constant reference.
        """
        self.floor = floor_rms
        self.base_ceiling = ceiling_rms
        self.ceiling = ceiling_rms
        self._peak_decay = peak_decay
        self.value = 0.0
        self.raw_rms = 0.0

    def update(self, frame: np.ndarray) -> float:
        """Process one audio frame, return 0-1 mapped value."""
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self.raw_rms = rms

        # Peak decay on ceiling — expands for loud input, decays back to base
        self.ceiling = max(self.base_ceiling,
                          self.ceiling * self._peak_decay)
        if rms > self.ceiling:
            self.ceiling = rms

        # Hard gate
        if rms < self.floor:
            self.value = 0.0
            return 0.0

        # Log mapping over the current range
        db = 20.0 * np.log10(rms / self.floor)
        db_range = 20.0 * np.log10(self.ceiling / self.floor)
        self.value = float(np.clip(db / db_range, 0.0, 1.0))

        return self.value


class EnergyDelta:
    """Frame-to-frame RMS change with peak normalization.

    Detects onsets/transients by measuring how much the RMS energy changes
    between consecutive frames. The absolute delta is normalized via a
    slow-decay peak to produce a 0-1 value — high on sudden changes
    (syllable onsets, percussive hits), low during sustained energy.
    """

    def __init__(self, peak_decay: float = 0.998):
        """
        Args:
            peak_decay: Per-frame decay for the running peak. 0.998 gives
                        ~500 frames (~6s at 86 fps) half-life.
        """
        self._prev_rms = 0.0
        self._peak = 1e-10
        self._peak_decay = peak_decay
        self.value = 0.0  # latest 0-1 output

    def update(self, frame: np.ndarray) -> float:
        """Process one audio frame, return 0-1 normalized delta."""
        rms = float(np.sqrt(np.mean(frame ** 2)))
        delta = abs(rms - self._prev_rms)
        self._prev_rms = rms

        # Slow-decay peak normalization
        self._peak = max(delta, self._peak * self._peak_decay)
        self.value = delta / self._peak if self._peak > 1e-10 else 0.0

        return self.value


class RollingIntegral:
    """Windowed average of RMS over a configurable time window.

    Maintains a ring buffer of per-frame RMS values and returns the
    running mean. Useful as a smoothed energy envelope — less noisy
    than single-frame RMS, less laggy than EMA at long time constants.
    """

    def __init__(self, window_sec: float = 0.3, fps: float = 86.0):
        """
        Args:
            window_sec: Window duration in seconds.
            fps:        Rate at which update() is called (frames/sec).
                        Default 86 ≈ 44100/512 for standard hop.
        """
        self._window_size = max(1, int(window_sec * fps))
        self._buf = np.zeros(self._window_size, dtype=np.float32)
        self._pos = 0
        self._filled = 0
        self.value = 0.0  # latest output (average RMS over window)

    def update(self, frame: np.ndarray) -> float:
        """Process one audio frame, return average RMS over window."""
        rms = float(np.sqrt(np.mean(frame ** 2)))

        self._buf[self._pos % self._window_size] = rms
        self._pos += 1
        self._filled = min(self._filled + 1, self._window_size)

        self.value = float(np.sum(self._buf[:self._filled]) / self._filled)
        return self.value


# ── Batch / offline utilities ────────────────────────────────────────────


def detect_timbral_sections(mfcc_matrix, fps, l2_normalize=False,
                            ema_tc=5.0, cooldown_s=15.0,
                            sigma_ceiling=3.0, sigma_floor=1.0,
                            eagerness_tau=120.0, power=3.0):
    """Single-EMA novelty detector with power-curve eagerness decay.

    Runs causally on a pre-computed MFCC matrix (batch/offline).
    Detects timbral section boundaries by tracking distance from a
    smoothed EMA reference, with an adaptive threshold that decays
    over time (eagerness curve).

    Threshold stays near ceiling then drops at rate controlled by power.
    power=3 (cubic): very patient. power=1: linear. power=0.5: eager early.

    Parameters
    ----------
    mfcc_matrix : ndarray, shape (n_coeffs, n_frames)
        MFCC coefficients over time.
    fps : float
        Analysis frame rate (sr / hop_length).
    l2_normalize : bool
        If True, L2-normalize each frame.
    ema_tc : float
        EMA time constant in seconds.
    cooldown_s : float
        Minimum seconds between transitions.
    sigma_ceiling, sigma_floor : float
        Threshold range (multiples of running std dev).
    eagerness_tau : float
        Seconds for threshold to decay from ceiling to floor.
    power : float
        Eagerness curve shape (3=cubic/patient, 1=linear, 0.5=eager).

    Returns
    -------
    transitions : list of float
        Timestamps (seconds) of detected section boundaries.
    """
    n_coeffs, n_frames = mfcc_matrix.shape
    alpha = 1.0 / (ema_tc * fps)
    cooldown = int(cooldown_s * fps)
    sigma_range = sigma_ceiling - sigma_floor

    x0 = mfcc_matrix[:, 0].astype(np.float64)
    if l2_normalize:
        x0 = x0 / (np.linalg.norm(x0) + 1e-10)

    ema = x0.copy()
    dist_ema = 0.0
    dist_var_ema = 0.0
    dist_alpha = 1.0 / (ema_tc * fps)

    frames_since_switch = cooldown + 1
    transitions = []

    for i in range(n_frames):
        x = mfcc_matrix[:, i].astype(np.float64)
        if l2_normalize:
            x = x / (np.linalg.norm(x) + 1e-10)

        ema += alpha * (x - ema)

        diff = x - ema
        dist = np.sqrt(np.dot(diff, diff))

        dist_ema += dist_alpha * (dist - dist_ema)
        dist_var_ema += dist_alpha * ((dist - dist_ema) ** 2 - dist_var_ema)
        dist_std = np.sqrt(max(dist_var_ema, 1e-20))

        frames_since_switch += 1

        # Eagerness curve: stays near ceiling, drops at rate controlled by power
        t = frames_since_switch / fps
        progress = min(t / eagerness_tau, 1.0)
        N = sigma_ceiling - sigma_range * (progress ** power)

        if frames_since_switch > cooldown and dist > dist_ema + N * dist_std:
            transitions.append(i / fps)
            frames_since_switch = 0

    return transitions
