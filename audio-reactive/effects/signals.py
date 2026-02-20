"""
Reusable signal processing primitives for audio-reactive effects.

Building blocks that effects compose via has-a:

  OverlapFrameAccumulator — feeds audio chunks, yields overlapped frames
  AbsIntegral             — computes normalized abs-integral of RMS derivative
  BeatPredictor           — autocorrelation tempo + predicted/confirmed beats
  OnsetTempoTracker       — onset-envelope autocorrelation for tempo estimation

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
                 n_bands: int = 6):
        self.sample_rate = sample_rate
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
        # Half-wave rectify + mean: onset strength this frame
        onset = np.mean(np.maximum(0, flux))

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
