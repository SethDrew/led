#!/usr/bin/env python3
"""
Synced Audio Playback + Visualization + Annotation

Interactive viewer with waveform, spectrogram, band energy, onset detection,
spectral centroid, and user annotation panels. Supports tap-to-annotate mode
where you can record feeling-based annotations while seeing the analysis.

Annotations can be point taps (list of timestamps) or labeled segment spans
(list of {start, end, label} dicts).

Usage (via explore.py):
    python explore.py

Dependencies:
    pip install sounddevice soundfile numpy librosa matplotlib pyyaml
"""

import sys
import time
import threading
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
try:
    import sounddevice as sd
except OSError:
    sd = None  # PortAudio not available (headless server)
import soundfile as sf
import yaml

# ── Shared Constants ──────────────────────────────────────────────────
# Single source of truth for frequency bands and colors

FREQUENCY_BANDS = {
    'Sub-bass': (20, 80),
    'Bass': (80, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 6000),
    'Treble': (6000, 8000),
}

BAND_COLORS = {
    'Sub-bass': '#FF1744',
    'Bass': '#FF9100',
    'Mids': '#FFEA00',
    'High-mids': '#00E676',
    'Treble': '#00B0FF',
}

ANNOTATION_COLORS = ['#FF4081', '#40C4FF', '#69F0AE', '#FFD740', '#E040FB', '#FF6E40']


def _feature_flux(features):
    """Half-wave rectified feature flux — peaks at abrupt timbral/harmonic changes.

    O(n) causal alternative to Foote's O(n²) checkerboard novelty.
    """
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1
    normed = features / norms
    diff = np.diff(normed, axis=1)
    flux = np.mean(np.maximum(0, diff), axis=0)
    flux = np.concatenate([[0], flux])
    mx = np.max(flux)
    if mx > 0:
        flux /= mx
    return flux


def _ema_deviation(features, alpha=0.02):
    """EMA deviation — peaks during gradual drift from recent context.

    O(n) causal complement to feature flux.  Catches buildups and texture
    shifts that flux misses.
    """
    # Normalize so all coefficients contribute equally (matches _feature_flux)
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1
    normed = features / norms

    n_frames = normed.shape[1]
    ema = normed[:, 0].copy().astype(np.float64)
    deviation = np.zeros(n_frames)
    for t in range(1, n_frames):
        deviation[t] = np.linalg.norm(normed[:, t] - ema)
        ema = alpha * normed[:, t] + (1 - alpha) * ema
    mx = np.max(deviation)
    if mx > 0:
        deviation /= mx
    return deviation


def _foote_novelty(features, kernel_size=64):
    """Foote's checkerboard novelty — offline O(n^2) structural boundary detector.

    Builds a self-similarity matrix, convolves with a checkerboard kernel
    along the diagonal. Peaks indicate where music before differs maximally
    from music after — i.e., section boundaries.

    Offline only — requires full audio. Cannot run in real-time.
    """
    # L2-normalize features (same as flux)
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1
    normed = features / norms

    # Self-similarity matrix (cosine similarity since features are L2-normed)
    S = normed.T @ normed  # (n_frames, n_frames)

    # Checkerboard kernel (Foote 2000)
    # Diagonal blocks (+1): within-section similarity (high at boundaries)
    # Off-diagonal blocks (-1): cross-section similarity (low at boundaries)
    # Result: positive at boundaries, zero in homogeneous regions
    half = kernel_size // 2
    kernel = -np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = 1    # top-left: before vs before
    kernel[half:, half:] = 1    # bottom-right: after vs after

    n = S.shape[0]
    novelty = np.zeros(n)
    for i in range(half, n - half):
        patch = S[i - half:i + half, i - half:i + half]
        novelty[i] = np.sum(patch * kernel)

    # Half-wave rectify (only positive = boundary-like)
    novelty = np.maximum(0, novelty)

    # Normalize
    mx = np.max(novelty)
    if mx > 0:
        novelty /= mx

    return novelty


def _multiscale_ema_deviation(features, alphas=(0.004, 0.0002, 0.00004)):
    """EMA deviation at multiple timescales — drift from context at ~3s, ~1min, ~5min.

    Returns list of 3 arrays, each normalized independently to [0, 1].
    O(1) per frame per scale. ESP32-feasible.
    """
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1
    normed = features / norms

    n_frames = normed.shape[1]
    results = []
    for alpha in alphas:
        ema = normed[:, 0].copy().astype(np.float64)
        deviation = np.zeros(n_frames)
        for t in range(1, n_frames):
            deviation[t] = np.linalg.norm(normed[:, t] - ema)
            ema = alpha * normed[:, t] + (1 - alpha) * ema
        mx = np.max(deviation)
        if mx > 0:
            deviation /= mx
        results.append(deviation)
    return results


def _cumulative_zscore(features):
    """Running mean + variance z-score — how unusual vs everything heard so far.

    Uses Welford's online algorithm. O(1) per frame, O(d) memory.
    Normalized to [0, 1]. Note: z-scores dilute over time as variance grows.
    """
    n_dims, n_frames = features.shape
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1
    normed = features / norms

    running_mean = np.zeros(n_dims)
    running_var = np.zeros(n_dims)
    scores = np.zeros(n_frames)

    for t in range(n_frames):
        x = normed[:, t]
        if t == 0:
            running_mean = x.copy()
            continue
        delta = x - running_mean
        running_mean += delta / (t + 1)
        running_var += delta * (x - running_mean)  # Welford's
        if t > 1:
            std = np.sqrt(running_var / t)
            std[std == 0] = 1
            scores[t] = np.linalg.norm((x - running_mean) / std)

    mx = np.max(scores)
    if mx > 0:
        scores /= mx
    return scores


def _knn_reservoir_novelty(features, reservoir_size=500, k=5):
    """Reservoir sampling + KNN distance — 'have I heard this timbre before?'

    Each frame's score = distance to K-th nearest neighbor among reservoir entries.
    Reservoir sampling ensures uniform coverage of the song's history.
    O(N) per frame. ESP32-feasible at N<=500.
    """
    n_dims, n_frames = features.shape
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1
    normed = features / norms

    reservoir = []
    scores = np.zeros(n_frames)
    rng = np.random.RandomState(42)  # deterministic for reproducibility

    for t in range(n_frames):
        x = normed[:, t]
        if len(reservoir) >= k:
            dists = np.linalg.norm(np.array(reservoir) - x, axis=1)
            dists.sort()
            scores[t] = dists[k - 1]  # K-th nearest

        # Reservoir sampling
        if len(reservoir) < reservoir_size:
            reservoir.append(x.copy())
        else:
            j = rng.randint(0, t + 1)
            if j < reservoir_size:
                reservoir[j] = x.copy()

    mx = np.max(scores)
    if mx > 0:
        scores /= mx
    return scores


# ── Interactive Visualizer ────────────────────────────────────────────

class SyncedVisualizer:
    def __init__(self, filepath, focus_panel=None, show_beats=False,
                 annotations_path=None, annotate_layer=None,
                 led_effect=None, led_output=None, led_brightness=1.0,
                 panels=None, event_algorithm='b'):
        self.filepath = filepath
        self.focus_panel = focus_panel
        self.panels = panels  # optional list of panel names to show
        self.event_algorithm = event_algorithm
        self.show_beats = show_beats
        self.annotations_path_override = annotations_path
        self.annotate_layer = annotate_layer  # None = normal mode, str = annotation mode
        self.playback_start_time = None
        self.playback_offset = 0.0
        self.is_playing = False

        # LED output (optional — driven in sync with playback cursor)
        self.led_effect = led_effect
        self.led_output = led_output
        self.led_brightness = led_brightness
        self.led_sample_pos = 0  # tracks how far we've fed audio to the effect

        # Annotation state
        self.new_taps = []
        self.feature_toggle = {}
        self.feature_keys = {}

        print(f"Loading: {filepath}")

        # Load for analysis (mono, librosa resampled)
        self.y, self.sr = librosa.load(filepath, sr=None, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

        # Load for playback (original channels)
        self.y_playback, self.sr_playback = sf.read(filepath)

        print(f"Duration: {self.duration:.1f}s, Sample rate: {self.sr} Hz")

        # Load annotations if present
        self.annotations = {}
        if self.annotations_path_override:
            ann_path = Path(self.annotations_path_override)
        else:
            ann_path = Path(filepath).with_suffix('.annotations.yaml')
            if not ann_path.exists():
                ann_path = Path(filepath).parent / (Path(filepath).stem.rsplit('_short', 1)[0] + '.annotations.yaml')
        self.annotations_path = Path(filepath).with_suffix('.annotations.yaml')
        if ann_path.exists():
            with open(ann_path) as f:
                self.annotations = yaml.safe_load(f) or {}
            print(f"Annotations: {ann_path.name} — {len(self.annotations)} layers ({', '.join(self.annotations.keys())})")

        # Load algorithm annotations (separate directory, merged into display)
        algo_ann_path = Path(filepath).parent / 'algorithm-annotations' / (Path(filepath).stem + '.yaml')
        if algo_ann_path.exists():
            with open(algo_ann_path) as f:
                algo_annotations = yaml.safe_load(f) or {}
            self.annotations.update(algo_annotations)
            print(f"Algorithm annotations: {algo_ann_path.name} — {len(algo_annotations)} layers ({', '.join(algo_annotations.keys())})")

        # Analysis parameters
        self.n_fft = 2048
        self.hop_length = 512

        self._compute_features()
        self._build_figure()

    def _compute_features(self):
        print("Computing features...")

        # Mel spectrogram - full frequency range
        mel_spec = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=64, fmin=20, fmax=None
        )
        self.mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Band energies
        mel_spec_128 = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=128, fmin=20, fmax=8000
        )
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)

        self.band_energies = {}
        self.band_ratios = {}
        for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
            band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)
            energy = np.sum(mel_spec_128[band_mask, :], axis=0)
            self.band_ratios[band_name] = np.mean(energy)
            if np.max(energy) > 0:
                self.band_energies[band_name] = energy / np.max(energy)
            else:
                self.band_energies[band_name] = energy

        # A-weighted band energies (perceptually weighted before mel grouping)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        f2 = freqs ** 2
        a_weight_db = (
            20 * np.log10(
                (12194**2 * f2**2) /
                ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
                + 1e-20
            )
            + 2.0
        )
        a_weight_linear = 10 ** (a_weight_db / 20.0)
        a_weight_linear[0] = 0  # DC bin

        S_power = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length)) ** 2
        S_weighted = S_power * (a_weight_linear[:, np.newaxis] ** 2)
        mel_basis_128 = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=128, fmin=20, fmax=8000)
        mel_weighted_128 = mel_basis_128 @ S_weighted

        # Collect raw A-weighted band energies first, then normalize all bands
        # against a shared max so cross-band comparison is preserved
        band_energies_aw_raw = {}
        self.band_ratios_aw = {}
        for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
            band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)
            energy = np.sum(mel_weighted_128[band_mask, :], axis=0)
            band_energies_aw_raw[band_name] = energy
            self.band_ratios_aw[band_name] = np.mean(energy)

        # Single shared max across all bands — preserves A-weighted balance
        global_max = max(np.max(e) for e in band_energies_aw_raw.values()) + 1e-10
        self.band_energies_aw = {
            name: energy / global_max for name, energy in band_energies_aw_raw.items()
        }

        # Onset detection
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length
        )
        self.onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sr, hop_length=self.hop_length
        )

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length
        )
        self.beat_times = librosa.frames_to_time(
            beat_frames, sr=self.sr, hop_length=self.hop_length
        )
        if isinstance(tempo, np.ndarray):
            self.tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            self.tempo = float(tempo)

        # Spectral centroid & RMS
        self.spectral_centroid = librosa.feature.spectral_centroid(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        self.rms_energy = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0]

        # RMS derivative (rate-of-change of loudness)
        self.rms_derivative = np.diff(self.rms_energy, prepend=self.rms_energy[0])
        dt = self.hop_length / self.sr
        self.rms_derivative = self.rms_derivative / dt  # units per second

        # Spectral centroid derivative (rate-of-change of brightness)
        self.centroid_derivative = np.diff(self.spectral_centroid, prepend=self.spectral_centroid[0])
        self.centroid_derivative = self.centroid_derivative / dt  # Hz per second

        # Per-band energy derivatives
        self.band_derivatives = {}
        for band_name, energy in self.band_energies.items():
            deriv = np.diff(energy, prepend=energy[0]) / dt
            self.band_derivatives[band_name] = deriv

        # Time axis
        self.times = librosa.frames_to_time(
            np.arange(len(self.onset_env)), sr=self.sr, hop_length=self.hop_length
        )

        # ── Novelty: feature flux + EMA deviation (causal, O(n)) ─────
        print("Computing novelty...")
        fps = self.sr / self.hop_length
        self.mfccs = librosa.feature.mfcc(
            y=self.y, sr=self.sr, n_mfcc=13,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.chroma = librosa.feature.chroma_cqt(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )

        # Spectral flatness (low = tonal, high = noisy)
        self.spectral_flatness = librosa.feature.spectral_flatness(
            y=self.y, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        self.novelty_mfcc = _feature_flux(self.mfccs)
        self.novelty_chroma = _feature_flux(self.chroma)
        self.ema_mfcc = _ema_deviation(self.mfccs, alpha=0.02)
        self.ema_chroma = _ema_deviation(self.chroma, alpha=0.02)

        print("Computing Foote's novelty (offline)...")
        self.foote_mfcc = _foote_novelty(self.mfccs)
        self.foote_chroma = _foote_novelty(self.chroma)

        # ── Per-band normalization: asymmetric EMA + dropout detection ──
        # (ledger: per-band-normalization-with-dropout-handling)
        # Reference tracks sliding mean via instant attack / constant decay.
        # Dropout freezes the reference so reintroduction produces large values.
        from scipy.ndimage import uniform_filter1d

        precompute_frames = int(30 * fps)  # seed from first ~30s
        decay_time_sec = 30  # reference decays from mean to ~0 in this many seconds
        dropout_percentile = 5  # frames below this percentile = dropout

        raw_energies = {}
        for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
            band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)
            raw_energies[band_name] = np.sum(mel_spec_128[band_mask, :], axis=0)

        self.band_energy_rt = {}
        for band_name, energy in raw_energies.items():
            n_frames = len(energy)

            # Absolute dropout threshold: low percentile of non-zero energy
            nonzero = energy[energy > 0]
            dropout_thresh = np.percentile(nonzero, dropout_percentile) if len(nonzero) > 0 else 0

            # Seed reference from mean of first ~30s (not max)
            seed = np.mean(energy[:min(precompute_frames, n_frames)]) if n_frames > 0 else 1e-10
            seed = max(seed, 1e-10)

            # Constant decay: reference drops by this much per frame
            constant_decay = seed / (decay_time_sec * fps)
            constant_decay = max(constant_decay, 1e-12)

            ref = seed
            normalized = np.empty(n_frames)
            for i in range(n_frames):
                e = energy[i]
                if e < dropout_thresh:
                    pass  # dropout: freeze reference
                elif e > ref:
                    ref = e  # instant attack
                else:
                    ref = max(ref - constant_decay, 1e-10)  # constant (linear) decay
                normalized[i] = e / ref

            self.band_energy_rt[band_name] = normalized

        # Total raw energy for dual-axis overlay
        self.total_raw_energy = sum(raw_energies.values())

        # ── 5s rolling integral of RT-normalized energy ──
        integral_window = int(5 * fps)
        self.band_integral_5s = {}
        for band_name, energy in self.band_energy_rt.items():
            self.band_integral_5s[band_name] = uniform_filter1d(
                energy, size=integral_window, mode='reflect') * integral_window

        # ── Band deviation from long-term context ────────────────────
        self.band_deviation_context_seconds = 60
        context_frames = max(3, int(self.band_deviation_context_seconds * fps))
        self.band_deviations = {}
        for band_name, energy in self.band_energies.items():
            running_mean = uniform_filter1d(energy.astype(np.float64), size=context_frames, mode='reflect')
            running_sq = uniform_filter1d((energy ** 2).astype(np.float64), size=context_frames, mode='reflect')
            running_std = np.sqrt(np.maximum(running_sq - running_mean ** 2, 0))
            self.band_deviations[band_name] = np.abs(energy - running_mean) / (running_std + 1e-10)
        all_z = np.stack([self.band_deviations[b][:len(self.times)] for b in FREQUENCY_BANDS], axis=0)
        self.composite_deviation = np.max(all_z, axis=0)

        self._compute_events()
        self._compute_calculus()
        self._compute_vibes()

        print(f"Tempo: {self.tempo:.1f} BPM, {len(self.onset_times)} onsets, {len(self.beat_times)} beats")

    @staticmethod
    def _find_spans(mask, times, min_frames):
        """Find contiguous True runs >= min_frames, return [(start_time, end_time), ...]."""
        spans = []
        in_span = False
        start_idx = 0
        for i, v in enumerate(mask):
            if v and not in_span:
                in_span = True
                start_idx = i
            elif not v and in_span:
                if i - start_idx >= min_frames:
                    spans.append((times[start_idx], times[min(i, len(times) - 1)]))
                in_span = False
        if in_span and len(mask) - start_idx >= min_frames:
            spans.append((times[start_idx], times[len(mask) - 1]))
        return spans

    def _compute_vibes(self):
        """Compute perceptual 'vibes' features: warmth, roughness, fluctuation, fullness.

        Based on Zwicker psychoacoustic model (sharpness, roughness, fluctuation
        strength) and timbral dimension research (Grey 1977, McAdams 1995,
        Zacharakis 2014).

        Warmth: spectral slope — how fast energy drops with frequency.
                Orthogonal to brightness (Zacharakis et al. 2014).
        Roughness: amplitude modulation energy in band envelopes (15+ Hz).
                   Correlates with perceived tension/dissonance.
        Fluctuation: slow AM in band envelopes (~1-8 Hz).
                     Captures pulse, groove, breathing quality.
        Fullness: spectral spread (std dev around centroid).
        Lightness: centroid remapped to 0-1 perceptual scale.
        """
        from scipy.ndimage import gaussian_filter1d, uniform_filter1d

        print("Computing vibes...")
        fps = self.sr / self.hop_length
        dt = self.hop_length / self.sr
        n = len(self.times)

        # --- Warmth: spectral slope (tilt) per frame ---
        # Compute on 64-mel spectrogram, 50-8000Hz range
        mel_spec_w = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=64, fmin=50, fmax=8000
        )
        mel_db_w = librosa.power_to_db(mel_spec_w + 1e-10, ref=np.max)
        mel_freqs_w = librosa.mel_frequencies(n_mels=64, fmin=50, fmax=8000)
        log_freqs = np.log2(mel_freqs_w + 1)

        # Vectorized linear regression: slope per frame
        x = log_freqs
        x_mean = np.mean(x)
        x_centered = x - x_mean
        x_var = np.sum(x_centered ** 2)

        y_frames = mel_db_w[:, :n]
        y_mean = np.mean(y_frames, axis=0)
        slopes = (x_centered @ (y_frames - y_mean[np.newaxis, :])) / (x_var + 1e-10)

        # Negate: negative acoustic slope = warm sound = positive warmth value
        self.vibes_warmth = -slopes

        # --- Fullness: spectral spread (bandwidth) ---
        self.vibes_fullness = librosa.feature.spectral_bandwidth(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0][:n]

        # --- Roughness proxy: fast AM energy in band envelopes ---
        # Local variance in ~100ms windows captures modulation > ~10Hz
        rough_window = max(3, int(0.1 / dt))
        band_roughness = []
        for band_name in FREQUENCY_BANDS:
            energy = self.band_energies[band_name][:n]
            local_mean = uniform_filter1d(energy, size=rough_window, mode='reflect')
            local_sq_mean = uniform_filter1d(energy ** 2, size=rough_window, mode='reflect')
            local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
            # Coefficient of variation: scale-independent modulation depth
            band_roughness.append(np.sqrt(local_var) / (local_mean + 1e-10))

        self.vibes_roughness = np.mean(band_roughness, axis=0)

        # --- Fluctuation Strength proxy: slow AM energy (~1-8Hz) ---
        # Local variance in ~1s windows captures slow modulation
        fluct_window = max(3, int(1.0 / dt))
        band_fluctuation = []
        for band_name in FREQUENCY_BANDS:
            energy = self.band_energies[band_name][:n]
            local_mean = uniform_filter1d(energy, size=fluct_window, mode='reflect')
            local_sq_mean = uniform_filter1d(energy ** 2, size=fluct_window, mode='reflect')
            local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
            band_fluctuation.append(np.sqrt(local_var) / (local_mean + 1e-10))

        self.vibes_fluctuation = np.mean(band_fluctuation, axis=0)

        # --- Vassilakis sensory roughness (Plomp-Levelt polynomial) ---
        # Peak-based pairwise dissonance, computed every 4th frame for speed.
        # References: Vassilakis 2001, Essentia dissonance.cpp, dissonant library
        from scipy.signal import find_peaks

        print("Computing Vassilakis roughness...")
        S_mag = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length))
        stft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        n_stft = min(S_mag.shape[1], n)
        n_peaks_max = 20
        skip = 4  # compute every 4th frame, interpolate the rest

        # Precompute Bark values for all FFT frequencies
        bark_all = 13.0 * np.arctan(0.00076 * stft_freqs) + 3.5 * np.arctan((stft_freqs / 7500.0) ** 2)

        roughness_sparse = np.zeros(n_stft)
        computed_indices = list(range(0, n_stft, skip))

        for idx in computed_indices:
            mag = S_mag[:, idx]
            noise_floor = np.max(mag) * 0.01
            if noise_floor < 1e-10:
                continue

            # Find spectral peaks (local maxima above noise floor)
            peak_idx, props = find_peaks(mag, height=noise_floor, distance=2)
            if len(peak_idx) < 2:
                continue

            # Keep top N by amplitude
            if len(peak_idx) > n_peaks_max:
                top = np.argpartition(mag[peak_idx], -n_peaks_max)[-n_peaks_max:]
                peak_idx = peak_idx[top]

            # Sort by frequency for early-exit optimization
            order = np.argsort(peak_idx)
            pf = stft_freqs[peak_idx[order]]
            pa = mag[peak_idx[order]]
            pb = bark_all[peak_idx[order]]
            np_peaks = len(pf)

            # Vectorized pairwise computation
            # Build upper-triangle index pairs
            ii, jj = np.triu_indices(np_peaks, k=1)
            bark_diff = np.abs(pb[jj] - pb[ii])

            # Early-exit: only pairs within 1.18 Bark
            close = bark_diff <= 1.18
            if not np.any(close):
                continue

            ii, jj, bark_diff = ii[close], jj[close], bark_diff[close]
            a1, a2 = pa[ii], pa[jj]

            # Plomp-Levelt polynomial dissonance curve
            x = bark_diff / 1.18
            x2 = x * x
            x3 = x2 * x
            d = np.maximum(0.0,
                -1.2866 * x2 * x3 + 6.3694 * x2 * x2 - 11.7275 * x3
                + 9.7166 * x2 - 2.7059 * x - 0.1984)

            # Weight by amplitude product (simplified from Vassilakis full model)
            roughness_sparse[idx] = np.sum(a1 * a2 * d)

        # Interpolate skipped frames
        roughness_full = np.interp(
            np.arange(n_stft),
            computed_indices,
            roughness_sparse[computed_indices]
        )

        # Normalize to 0-1
        rmax = np.percentile(roughness_full[roughness_full > 0], 99) if np.any(roughness_full > 0) else 1.0
        self.vibes_roughness_vassilakis = np.clip(roughness_full[:n] / max(rmax, 1e-10), 0, 1)

        # --- Lightness: centroid remapped to 0-1 perceptual scale ---
        # Log-scale mapping: 200Hz → 0.0, 10kHz → 1.0
        cent = self.spectral_centroid[:n]
        log_lo, log_hi = np.log2(200), np.log2(10000)
        self.vibes_lightness = np.clip(
            (np.log2(np.maximum(cent, 1)) - log_lo) / (log_hi - log_lo), 0, 1
        )

        print(f"Vibes: warmth μ={np.mean(self.vibes_warmth):.2f}, "
              f"roughness(AM) μ={np.mean(self.vibes_roughness):.3f}, "
              f"roughness(Vass) μ={np.mean(self.vibes_roughness_vassilakis):.3f}, "
              f"fluctuation μ={np.mean(self.vibes_fluctuation):.3f}, "
              f"fullness μ={np.mean(self.vibes_fullness):.0f}Hz")

        # --- Timbral section transitions (online detection, simulated) ---
        # Uses the same anchored-reference algorithm as timbral_chroma_split
        # to detect section boundaries via L2-normalized MFCC shape drift.
        # Runs causally on the pre-computed MFCC matrix (no future data).
        try:
            _effects_dir = str(Path(__file__).resolve().parent.parent / 'effects')
            if _effects_dir not in sys.path:
                sys.path.insert(0, _effects_dir)
            from timbral_section_detect import detect_log_eagerness
            self.timbral_transitions = detect_log_eagerness(
                self.mfccs[0:13], fps, l2_normalize=False, power=3.0
            )
            print(f"Timbral sections: {len(self.timbral_transitions)} transitions detected")
        except Exception as e:
            print(f"Timbral section detection skipped: {e}")
            self.timbral_transitions = []

    def _compute_events(self):
        """Dispatch to algorithm A or B based on self.event_algorithm."""
        from scipy.ndimage import gaussian_filter1d

        fps = self.sr / self.hop_length
        t = self.times
        n = len(t)

        # ── Risers: shared by both algorithms ──
        cent_d = gaussian_filter1d(self.centroid_derivative[:n], sigma=10)
        cent_d_max = np.max(np.abs(cent_d)) + 1e-10
        riser_mask = cent_d > 0.08 * cent_d_max
        min_riser_frames = max(1, int(1.0 * fps))
        self.event_risers = self._find_spans(riser_mask, t, min_riser_frames)

        if self.event_algorithm == 'a':
            self._compute_events_a()
        else:
            self._compute_events_b()

        total_dropouts = sum(len(v) for v in self.event_dropouts.values())
        dropout_bands = ', '.join(f'{b}:{len(v)}' for b, v in self.event_dropouts.items())
        print(f"Events ({self.event_algorithm.upper()}): {len(self.event_drops)} drops, "
              f"{len(self.event_risers)} risers, "
              f"{total_dropouts} dropouts ({dropout_bands}), {len(self.event_harmonic)} harmonic")

    def _compute_events_a(self):
        """Algorithm A: additive scoring (prominence-based), stricter thresholds."""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

        fps = self.sr / self.hop_length
        t = self.times
        n = len(t)

        # ── Drops: additive prominence-based scoring ──
        nov_m = self.novelty_mfcc[:n]
        rms = self.rms_energy[:n]
        rms_max = np.max(rms) + 1e-10

        min_dist = max(1, int(2.0 * fps))
        nov_peaks, nov_props = find_peaks(nov_m, prominence=0.15, distance=min_dist)

        self.event_drops = []
        self.event_drop_scores = []
        half_sec = max(1, int(0.5 * fps))

        for i, peak_idx in enumerate(nov_peaks):
            # Base score from prominence (0-1)
            base = nov_props['prominences'][i]

            # Booster 1: RMS discontinuity in ±0.5s window
            lo = max(0, peak_idx - half_sec)
            hi = min(n, peak_idx + half_sec)
            rms_before = np.mean(rms[lo:peak_idx]) if peak_idx > lo else 0
            rms_after = np.mean(rms[peak_idx:hi]) if hi > peak_idx else 0
            rms_disc = abs(rms_after - rms_before) / rms_max
            rms_boost = 0.20 * min(rms_disc / 0.3, 1.0)

            # Booster 2: band deviation spike near peak
            dev = self.composite_deviation
            dev_window = dev[lo:hi] if hi > lo else np.array([0])
            dev_spike = np.max(dev_window)
            dev_boost = 0.15 * min(dev_spike / 4.0, 1.0)

            score = base + rms_boost + dev_boost
            if score >= 0.15:
                self.event_drops.append(float(t[peak_idx]))
                self.event_drop_scores.append(float(score))

        # ── Dropouts: vectorized causal trailing mean ──
        trailing_window = int(10.0 * fps)
        min_dropout_frames = max(1, int(0.3 * fps))
        self.event_dropouts = {}
        for band_name in FREQUENCY_BANDS:
            energy_rt = self.band_energy_rt[band_name][:n]
            padded = np.concatenate([np.zeros(trailing_window), energy_rt])
            trailing_mean = uniform_filter1d(padded, size=trailing_window, mode='reflect')[trailing_window:]
            was_present = trailing_mean > 0.20
            is_low = energy_rt < 0.08
            dropout_mask = was_present & is_low
            spans = self._find_spans(dropout_mask, t, min_dropout_frames)
            if spans:
                self.event_dropouts[band_name] = spans

        # ── Harmonic: strict ratio thresholds, median-relative flatness ──
        onset = self.onset_env[:n]
        sigma_03s = max(1, int(0.3 * fps))
        onset_smooth = gaussian_filter1d(onset, sigma=sigma_03s)
        median_size = max(3, int(15.0 * fps))
        if median_size % 2 == 0:
            median_size += 1
        onset_local_median = median_filter(onset, size=median_size, mode='reflect')
        self.event_harmonic_ratio = onset_smooth / (onset_local_median + 1e-10)

        flatness = self.spectral_flatness[:n]
        sigma_05s = max(1, int(0.5 * fps))
        self.event_harmonic_flatness = gaussian_filter1d(flatness, sigma=sigma_05s)
        flatness_median = np.median(self.event_harmonic_flatness)

        ratio = self.event_harmonic_ratio
        flat = self.event_harmonic_flatness

        harmonic_mask = (
            (ratio < 0.25)
            | ((ratio < 0.40) & (flat < flatness_median))
        )
        silence_gate = rms > 0.02 * rms_max
        harmonic_mask = harmonic_mask & silence_gate

        min_harmonic_frames = max(1, int(1.5 * fps))
        self.event_harmonic = self._find_spans(harmonic_mask, t, min_harmonic_frames)

    def _compute_events_b(self):
        """Algorithm B: multiplicative scoring (novelty-value-based), lenient thresholds."""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d, median_filter

        fps = self.sr / self.hop_length
        t = self.times
        n = len(t)

        # ── Drops: multiplicative novelty-value scoring ──
        nov_m = self.novelty_mfcc[:n]
        rms_d = self.rms_derivative[:n]
        rms_d_norm = np.abs(rms_d) / (np.max(np.abs(rms_d)) + 1e-10)

        comp_dev = self.composite_deviation[:n]
        comp_dev_norm = comp_dev / (np.max(comp_dev) + 1e-10)

        min_peak_dist = max(1, int(2.0 * fps))
        nov_peaks, nov_props = find_peaks(nov_m, prominence=0.12, distance=min_peak_dist)

        self.event_drops = []
        self.event_drop_scores = []
        for idx in nov_peaks:
            base_score = nov_m[idx]

            window = int(0.5 * fps)
            lo = max(0, idx - window)
            hi = min(n, idx + window)
            rms_boost = np.max(rms_d_norm[lo:hi])

            dev_window = int(1.0 * fps)
            lo_d = max(0, idx - dev_window)
            hi_d = min(n, idx + dev_window)
            dev_boost = np.max(comp_dev_norm[lo_d:hi_d])

            score = base_score * (1.0 + 0.3 * rms_boost + 0.2 * dev_boost)

            self.event_drops.append(t[idx])
            self.event_drop_scores.append(score)

        # ── Dropouts: cumsum-based causal trailing mean ──
        trailing_window = int(8.0 * fps)
        presence_thresh = 0.15
        low_thresh = 0.05
        min_dropout_frames = max(1, int(0.5 * fps))

        self.event_dropouts = {}
        for band_name, energy_rt in self.band_energy_rt.items():
            e = energy_rt[:n]

            cumsum = np.cumsum(np.insert(e, 0, 0))
            trailing_mean = np.empty_like(e)
            for i in range(n):
                lo = max(0, i - trailing_window)
                trailing_mean[i] = (cumsum[i] - cumsum[lo]) / max(i - lo, 1)

            was_present = trailing_mean > presence_thresh
            is_low = e < low_thresh
            dropout_mask = was_present & is_low

            spans = self._find_spans(dropout_mask, t, min_dropout_frames)
            if spans:
                self.event_dropouts[band_name] = spans

        # ── Harmonic: lenient ratio thresholds, fixed flatness threshold ──
        onset = self.onset_env[:n]

        onset_sigma = max(1, int(0.3 * fps))
        onset_smooth = gaussian_filter1d(onset, sigma=onset_sigma)

        median_window = max(3, int(15.0 * fps))
        if median_window % 2 == 0:
            median_window += 1
        onset_median = median_filter(onset_smooth, size=median_window, mode='reflect')

        self.event_harmonic_ratio = onset_smooth / (onset_median + 1e-10)

        flatness_sigma = max(1, int(2.0 * fps))
        self.event_harmonic_flatness = gaussian_filter1d(
            self.spectral_flatness[:n], sigma=flatness_sigma)

        rms = self.rms_energy[:n]
        rms_max = np.max(rms) + 1e-10
        silence_mask = rms > 0.02 * rms_max

        ratio_thresh = 0.4
        primary_harmonic = (self.event_harmonic_ratio < ratio_thresh) & silence_mask

        flatness_thresh = 0.15
        ratio_thresh_lenient = 0.65
        secondary_harmonic = (
            (self.event_harmonic_ratio < ratio_thresh_lenient)
            & (self.event_harmonic_flatness < flatness_thresh)
            & silence_mask
        )

        harmonic_mask = primary_harmonic | secondary_harmonic
        min_harmonic_frames = max(1, int(1.0 * fps))
        self.event_harmonic = self._find_spans(harmonic_mask, t, min_harmonic_frames)

    def _compute_calculus(self):
        """Compute second derivatives and time-bounded integrals for calculus tab."""
        from scipy.ndimage import gaussian_filter1d, uniform_filter1d

        fps = self.sr / self.hop_length
        dt = 1.0 / fps

        # Trailing rolling sums of RMS (causal — no look-ahead)
        # output[i] = sum(rms[max(0, i-w+1) : i+1])
        cs = np.concatenate([[0.0], np.cumsum(self.rms_energy)])
        indices = np.arange(len(self.rms_energy))

        int_window_10s = int(10 * fps)
        starts_10 = np.maximum(0, indices - int_window_10s + 1)
        self.rms_integral_10s = cs[indices + 1] - cs[starts_10]

        int_window_5s = int(5 * fps)
        starts_5 = np.maximum(0, indices - int_window_5s + 1)
        self.rms_integral_5s = cs[indices + 1] - cs[starts_5]

        # Integral slope: d/dt of 10s integral, smoothed
        raw_slope = np.diff(self.rms_integral_10s, prepend=self.rms_integral_10s[0]) / dt
        self.integral_slope = gaussian_filter1d(raw_slope, sigma=15)

        # Integral curvature: d²/dt² of 10s integral
        self.integral_curvature = gaussian_filter1d(
            np.diff(self.integral_slope, prepend=self.integral_slope[0]) / dt, sigma=10)

        # Slope derivative for zero-crossing peak detection
        # Heavy smoothing (σ=60 ≈ 0.7s) eliminates phrase-level jitter,
        # keeping only section-level transitions. RT-feasible via 3 cascaded box filters.
        slope_smooth = gaussian_filter1d(self.integral_slope, sigma=60)
        self.slope_derivative = np.diff(slope_smooth, prepend=slope_smooth[0]) / dt
        self.slope_smooth = slope_smooth

        # Find ALL zero crossings of the smoothed slope derivative
        sd = self.slope_derivative
        raw_build_zc = np.where((sd[:-1] > 0) & (sd[1:] <= 0))[0]
        raw_decay_zc = np.where((sd[:-1] < 0) & (sd[1:] >= 0))[0]

        # Filter: build peaks only when slope is positive and large,
        # decay troughs only when slope is negative and large
        slope_max = np.max(np.abs(slope_smooth)) + 1e-10
        slope_thresh = 0.20 * slope_max
        raw_build_zc = raw_build_zc[slope_smooth[raw_build_zc] > slope_thresh]
        raw_decay_zc = raw_decay_zc[slope_smooth[raw_decay_zc] < -slope_thresh]

        # Filter: minimum 8s gap between consecutive same-type markers
        min_gap_frames = int(8 * fps)

        def _dedupe(indices):
            if len(indices) == 0:
                return indices
            keep = [indices[0]]
            for idx in indices[1:]:
                if idx - keep[-1] >= min_gap_frames:
                    keep.append(idx)
            return np.array(keep)

        self.slope_peak_build_idx = _dedupe(raw_build_zc)
        self.slope_peak_decay_idx = _dedupe(raw_decay_zc)

        # Multi-scale integrals (2s, 5s, 15s of RT-normalized total RMS)
        total_rt = sum(self.band_energy_rt[b] for b in FREQUENCY_BANDS) / len(FREQUENCY_BANDS)
        self.multi_scale_integrals = {}
        for window_sec in (2, 5, 15):
            w = int(window_sec * fps)
            self.multi_scale_integrals[window_sec] = uniform_filter1d(
                total_rt, size=w, mode='reflect') * w

        # Onset second derivative (smoothed at multiple scales)
        self.onset_d2 = {}
        for sigma in (1, 3, 10, 30):
            smoothed = gaussian_filter1d(self.onset_env, sigma=sigma)
            d2 = np.diff(smoothed, n=2, prepend=[smoothed[0], smoothed[0]]) / (dt ** 2)
            self.onset_d2[sigma] = d2

        # RMS second derivatives at multiple smoothing scales (for jitter panel)
        self.rms_d2 = {}
        for sigma in (1, 3, 10, 30):
            smoothed = gaussian_filter1d(self.rms_energy, sigma=sigma)
            d2 = np.diff(smoothed, n=2, prepend=[smoothed[0], smoothed[0]]) / (dt ** 2)
            self.rms_d2[sigma] = d2

        # AbsInt: rolling sum of |d(RMS)/dt| over 150ms window (offline equivalent of AbsIntegral)
        absint_window = max(1, int(0.15 * fps))
        abs_deriv = np.abs(self.rms_derivative)
        absint_cs = np.concatenate([[0.0], np.cumsum(abs_deriv)])
        absint_idx = np.arange(len(abs_deriv))
        absint_starts = np.maximum(0, absint_idx - absint_window + 1)
        self.absint_signal = (absint_cs[absint_idx + 1] - absint_cs[absint_starts]) * dt

        # AbsInt second derivatives at multiple smoothing scales
        self.absint_d2 = {}
        for sigma in (1, 3, 10, 30):
            smoothed = gaussian_filter1d(self.absint_signal, sigma=sigma)
            d2 = np.diff(smoothed, n=2, prepend=[smoothed[0], smoothed[0]]) / (dt ** 2)
            self.absint_d2[sigma] = d2

        # Build detector spans
        self._detect_builds()

        print(f"Calculus: integral_slope range [{self.integral_slope.min():.4f}, {self.integral_slope.max():.4f}], "
              f"{len(self.build_spans)} builds, {len(self.decay_spans)} decays")

    def _detect_builds(self):
        """Classify spans as building/decaying/steady from integral slope."""
        fps = self.sr / self.hop_length
        slope = self.integral_slope
        t = self.times

        # Threshold: 5% of the max absolute slope
        threshold = 0.05 * np.max(np.abs(slope))

        min_build_frames = max(1, int(5.0 * fps))  # 5s minimum for builds
        min_decay_frames = max(1, int(3.0 * fps))   # 3s minimum for decays

        build_mask = slope > threshold
        decay_mask = slope < -threshold

        self.build_spans = self._find_spans(build_mask, t, min_build_frames)
        self.decay_spans = self._find_spans(decay_mask, t, min_decay_frames)

    def _overlay_sections(self, ax):
        """Draw section boundaries from annotations on any panel."""
        sections = self.annotations.get('sections', self.annotations.get('segments', []))
        if not sections:
            return
        for i, sec in enumerate(sections):
            t = sec.get('time', sec.get('start', None))
            if t is None:
                continue
            label = sec.get('label', '')
            color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            ax.axvline(x=t, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax.text(t + 0.3, 0.97, label, transform=ax.get_xaxis_transform(),
                    va='top', fontsize=7, color=color, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    # ── Calculus panels ──────────────────────────────────────────────

    def _build_calc_energy_integral_panel(self, ax):
        """Panel 1: RMS energy + trailing 5s/10s rolling integrals (dual y-axis)."""
        t = self.times
        n = len(t)
        ax.plot(t, self.rms_energy[:n], color='#FFFFFF', linewidth=0.8, alpha=0.4, label='RMS Energy')
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('RMS Energy', color='#FFFFFF')
        ax.tick_params(axis='y', labelcolor='#FFFFFF')

        ax2 = ax.twinx()
        ax2.plot(t, self.rms_integral_5s[:n], color='#69F0AE', linewidth=1.5, alpha=0.7, label='5s Trailing')
        ax2.plot(t, self.rms_integral_10s[:n], color='#FFD740', linewidth=2.5, alpha=0.9, label='10s Trailing')
        ax2.set_ylabel('Trailing Integral', color='#FFD740')
        ax2.tick_params(axis='y', labelcolor='#FFD740')

        ax.set_title('Energy + Trailing Integrals (causal — no look-ahead)', fontsize=11)
        ax.legend(loc='upper left', framealpha=0.8, fontsize=8)
        ax2.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        ax.text(0.5, 0.02,
                'Trailing sums: output[i] = sum(rms[i-w..i])  ·  '
                '5s reacts ~2.5s faster than 10s  ·  '
                'Downstream slope/curvature add 0.1–0.7s Gaussian (replaceable with exponential for RT)',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=7, color='#888', alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.4))

        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_integral_slope_panel(self, ax):
        """Panel 2: Integral slope — energy momentum (d/dt of 10s integral)."""
        t = self.times
        n = len(t)
        slope = self.integral_slope[:n]
        pos = np.maximum(slope, 0)
        neg = np.minimum(slope, 0)

        ax.fill_between(t, pos, color='#FF5252', alpha=0.7, linewidth=0, label='Rising')
        ax.fill_between(t, neg, color='#448AFF', alpha=0.7, linewidth=0, label='Falling')
        ax.axhline(y=0, color='#666', linewidth=0.5)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('d(Integral)/dt')
        ax.set_title('Integral Slope — red = energy building, blue = energy decaying', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_slope_peaks_panel(self, ax):
        """Slope peaks: heavily smoothed slope with structural zero-crossing markers."""
        t = self.times
        n = len(t)
        slope_raw = self.integral_slope[:n]
        slope_smooth = self.slope_smooth[:n]

        # Raw slope as faint context
        ax.fill_between(t, np.maximum(slope_raw, 0), color='#FF5252', alpha=0.15, linewidth=0)
        ax.fill_between(t, np.minimum(slope_raw, 0), color='#448AFF', alpha=0.15, linewidth=0)

        # Smoothed slope (σ=60) — the signal we derive
        ax.plot(t, slope_smooth, color='#FFFFFF', linewidth=2.0, alpha=0.9, label='Slope (σ=60)')
        ax.axhline(y=0, color='#666', linewidth=0.5)

        # Peak build markers
        peak_builds = self.slope_peak_build_idx[self.slope_peak_build_idx < n]
        if len(peak_builds) > 0:
            ax.scatter(t[peak_builds], slope_smooth[peak_builds],
                       color='#FF4081', s=100, zorder=6, marker='v',
                       edgecolors='white', linewidths=0.5, label='Peak build')
            for idx in peak_builds:
                ax.annotate(f'{t[idx]:.0f}s', (t[idx], slope_smooth[idx]),
                            textcoords='offset points', xytext=(0, -16),
                            fontsize=8, color='#FF4081', ha='center', fontweight='bold')

        # Peak decay markers
        peak_decays = self.slope_peak_decay_idx[self.slope_peak_decay_idx < n]
        if len(peak_decays) > 0:
            ax.scatter(t[peak_decays], slope_smooth[peak_decays],
                       color='#40C4FF', s=100, zorder=6, marker='^',
                       edgecolors='white', linewidths=0.5, label='Peak decay')
            for idx in peak_decays:
                ax.annotate(f'{t[idx]:.0f}s', (t[idx], slope_smooth[idx]),
                            textcoords='offset points', xytext=(0, 12),
                            fontsize=8, color='#40C4FF', ha='center', fontweight='bold')

        n_b = len(peak_builds)
        n_d = len(peak_decays)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Smoothed Slope')
        ax.set_title(
            f'Slope Peaks (σ=60, >25% thresh, >8s gap) — '
            f'{n_b} build peaks, {n_d} decay troughs', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_integral_curvature_panel(self, ax):
        """Panel 3: Energy acceleration — d²/dt² of 10s integral."""
        t = self.times
        n = len(t)
        curv = self.integral_curvature[:n]
        pos = np.maximum(curv, 0)
        neg = np.minimum(curv, 0)

        ax.fill_between(t, pos, color='#FF9100', alpha=0.7, linewidth=0, label='Accelerating')
        ax.fill_between(t, neg, color='#00E5FF', alpha=0.7, linewidth=0, label='Decelerating')
        ax.axhline(y=0, color='#666', linewidth=0.5)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('d²(Integral)/dt²')
        ax.set_title('Energy Acceleration — orange = build accelerating, cyan = plateauing', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_multi_scale_panel(self, ax):
        """Panel 4: Multi-scale integrals (2s, 5s, 15s)."""
        t = self.times
        n = len(t)
        styles = {2: ('#69F0AE', 1.0), 5: ('#40C4FF', 1.8), 15: ('#E040FB', 2.5)}
        for window_sec in (2, 5, 15):
            color, lw = styles[window_sec]
            ax.plot(t, self.multi_scale_integrals[window_sec][:n],
                    color=color, linewidth=lw, alpha=0.85,
                    label=f'{window_sec}s')

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Integrated Energy')
        ax.set_title('Multi-Scale Integrals — 2s phrase, 5s passage, 15s section arc', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_onset_d2_panel(self, ax):
        """Panel 5: Onset second derivative — peak detection via zero-crossings."""
        from scipy.ndimage import gaussian_filter1d

        t = self.times
        n = len(t)

        # Onset envelope (smoothed σ=3) with zero-crossing peaks
        onset = gaussian_filter1d(self.onset_env[:n], sigma=3)
        d2 = self.onset_d2[3][:n]

        ax.plot(t, onset / (np.max(onset) + 1e-10), color='#00E5FF', linewidth=1.5, alpha=0.9, label='Onset (σ=3)')

        # Detect peaks: zero-crossings of d2 from positive to negative
        zero_crossings = np.where((d2[:-1] > 0) & (d2[1:] <= 0))[0]
        if len(zero_crossings) > 0:
            onset_norm = onset / (np.max(onset) + 1e-10)
            ax.scatter(t[zero_crossings], onset_norm[zero_crossings],
                       color='#FF4081', s=15, zorder=5, marker='v', label='d² peaks')

        # Beat annotation ticks if available
        beats = self.annotations.get('beats', self.annotations.get('beat', []))
        if beats and isinstance(beats, list):
            beat_times = [b if isinstance(b, (int, float)) else b.get('time', b.get('start', 0)) for b in beats]
            beat_times = [b for b in beat_times if isinstance(b, (int, float))]
            if beat_times:
                for bt in beat_times:
                    ax.axvline(x=bt, color='#FFD740', alpha=0.3, linewidth=0.8)

                # Compute hit rate at 50ms tolerance
                hits = 0
                for zc_t in t[zero_crossings]:
                    if any(abs(zc_t - bt) < 0.05 for bt in beat_times):
                        hits += 1
                precision = hits / max(len(zero_crossings), 1)
                recall = sum(1 for bt in beat_times
                             if any(abs(bt - t[zc]) < 0.05 for zc in zero_crossings)) / max(len(beat_times), 1)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                ax.text(0.02, 0.95, f'P={precision:.2f} R={recall:.2f} F1={f1:.2f} @50ms',
                        transform=ax.transAxes, va='top', fontsize=8, color='#FFD740',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Onset Strength')
        ax.set_title('Onset Second Derivative — peak detection via d² zero-crossings', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_jitter_panel(self, ax):
        """Panel 6: RMS second derivative at 4 smoothing levels — jitter vs smoothing tradeoff."""
        t = self.times
        n = len(t)
        fps = self.sr / self.hop_length
        dt_ms = 1000.0 / fps  # ms per frame

        alphas = {1: 0.3, 3: 0.5, 10: 0.7, 30: 0.9}
        colors = {1: '#FF8A65', 3: '#FF5722', 10: '#E91E63', 30: '#9C27B0'}

        for sigma in (1, 3, 10, 30):
            d2 = self.rms_d2[sigma][:n]
            # Normalize each to its own range for visual comparison
            mx = np.max(np.abs(d2)) + 1e-10
            d2_norm = d2 / mx
            ms = sigma * dt_ms
            ax.plot(t, d2_norm, color=colors[sigma], linewidth=1.2,
                    alpha=alphas[sigma], label=f'σ={sigma} ({ms:.0f}ms)')

        ax.axhline(y=0, color='#666', linewidth=0.5)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('d²RMS/dt² (normalized)')
        ax.set_title('Jitter vs Smoothing — lighter = jittery (σ=1), darker = smooth (σ=30)', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_absint_d2_panel(self, ax):
        """AbsInt second derivative — peak detection via d² zero-crossings."""
        from scipy.ndimage import gaussian_filter1d

        t = self.times
        n = len(t)

        # AbsInt signal (smoothed σ=3) with zero-crossing peaks
        absint = gaussian_filter1d(self.absint_signal[:n], sigma=3)
        d2 = self.absint_d2[3][:n]

        ax.plot(t, absint / (np.max(absint) + 1e-10), color='#76FF03', linewidth=1.5, alpha=0.9, label='AbsInt (σ=3)')

        # Detect peaks: zero-crossings of d2 from positive to negative
        zero_crossings = np.where((d2[:-1] > 0) & (d2[1:] <= 0))[0]
        if len(zero_crossings) > 0:
            absint_norm = absint / (np.max(absint) + 1e-10)
            ax.scatter(t[zero_crossings], absint_norm[zero_crossings],
                       color='#FF4081', s=15, zorder=5, marker='v', label='d² peaks')

        # Beat annotation ticks if available
        beats = self.annotations.get('beats', self.annotations.get('beat', []))
        if beats and isinstance(beats, list):
            beat_times = [b if isinstance(b, (int, float)) else b.get('time', b.get('start', 0)) for b in beats]
            beat_times = [b for b in beat_times if isinstance(b, (int, float))]
            if beat_times:
                for bt in beat_times:
                    ax.axvline(x=bt, color='#FFD740', alpha=0.3, linewidth=0.8)

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('AbsInt Strength')
        ax.set_title('AbsInt Second Derivative — peak detection via d² zero-crossings', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_calc_absint_jitter_panel(self, ax):
        """AbsInt second derivative at 4 smoothing levels — jitter vs smoothing tradeoff."""
        t = self.times
        n = len(t)
        fps = self.sr / self.hop_length
        dt_ms = 1000.0 / fps

        alphas = {1: 0.3, 3: 0.5, 10: 0.7, 30: 0.9}
        colors = {1: '#B2FF59', 3: '#76FF03', 10: '#64DD17', 30: '#33691E'}

        for sigma in (1, 3, 10, 30):
            d2 = self.absint_d2[sigma][:n]
            mx = np.max(np.abs(d2)) + 1e-10
            d2_norm = d2 / mx
            ms = sigma * dt_ms
            ax.plot(t, d2_norm, color=colors[sigma], linewidth=1.2,
                    alpha=alphas[sigma], label=f'σ={sigma} ({ms:.0f}ms)')

        ax.axhline(y=0, color='#666', linewidth=0.5)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('d²AbsInt/dt² (normalized)')
        ax.set_title('AbsInt Jitter vs Smoothing — lighter = jittery (σ=1), darker = smooth (σ=30)', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _sections_to_spans(self):
        """Convert section boundary annotations to (start, end, label) spans.

        Handles both formats:
        - Point boundaries: [{time: 0, label: 'intro'}, {time: 30, label: 'build'}]
          → spans from each time to the next
        - Segment spans: [{start: 0, end: 30, label: 'intro'}]
          → used directly
        """
        sections = self.annotations.get('sections', self.annotations.get('segments', []))
        if not sections:
            return []
        spans = []
        if isinstance(sections[0], dict) and 'start' in sections[0]:
            for sec in sections:
                spans.append((sec['start'], sec.get('end', sec['start']), sec.get('label', '')))
        elif isinstance(sections[0], dict) and 'time' in sections[0]:
            for i, sec in enumerate(sections):
                start = sec['time']
                end = sections[i + 1]['time'] if i + 1 < len(sections) else self.duration
                spans.append((start, end, sec.get('label', '')))
        return spans

    def _build_calc_build_detector_panel(self, ax):
        """Panel 7: Build detector — simple classifier from integral slope."""
        t = self.times

        # Algorithmic detection spans (top row, y=0.7)
        for start, end in self.build_spans:
            ax.axvspan(start, end, ymin=0.45, ymax=1.0, color='#69F0AE', alpha=0.4)
        for start, end in self.decay_spans:
            ax.axvspan(start, end, ymin=0.45, ymax=1.0, color='#FF5252', alpha=0.4)

        # Section annotation spans (bottom row, y=0.15)
        section_spans = self._sections_to_spans()
        for i, (start, end, label) in enumerate(section_spans):
            color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            ax.barh(0.2, width=end - start, left=start, height=0.3,
                    color=color, alpha=0.5, edgecolor=color, linewidth=0.5)
            mid = (start + min(end, self.duration)) / 2
            ax.text(mid, 0.2, label, ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold', alpha=0.9)

        # Point-tap event markers from rich annotation layers
        event_layers = {
            'drop': ('#FF1744', 'v', 12),
            'riser': ('#76FF03', '^', 10),
            'finalbuild': ('#FFD740', 'D', 10),
            'doubledown': ('#E040FB', 's', 10),
            'bridge': ('#40C4FF', 'o', 10),
            'tease_cycles': ('#FF9100', '|', 8),
        }
        for layer_name, (color, marker, size) in event_layers.items():
            taps = self.annotations.get(layer_name, [])
            if taps and isinstance(taps, list):
                for tap in taps:
                    if isinstance(tap, (int, float)):
                        ax.plot(tap, 0.7, marker, color=color, markersize=size,
                                markeredgewidth=1.5, alpha=0.9, zorder=6)
                        ax.text(tap, 0.85, layer_name, ha='center', va='bottom',
                                fontsize=5, color=color, alpha=0.7, rotation=45)

        ax.set_xlim([0, self.duration])
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([])
        ax.text(0.001, 0.72, 'algorithm', transform=ax.transAxes, fontsize=7,
                color='#aaa', va='center', alpha=0.6)
        ax.text(0.001, 0.22, 'annotated', transform=ax.transAxes, fontsize=7,
                color='#aaa', va='center', alpha=0.6)

        # Summary stats
        n_builds = len(self.build_spans)
        n_decays = len(self.decay_spans)
        build_dur = sum(e - s for s, e in self.build_spans)
        decay_dur = sum(e - s for s, e in self.decay_spans)
        ax.set_title(
            f'Build Detector — {n_builds} builds ({build_dur:.1f}s), '
            f'{n_decays} decays ({decay_dur:.1f}s)', fontsize=11)
        ax.legend(
            [plt.Rectangle((0, 0), 1, 1, fc='#69F0AE', alpha=0.4),
             plt.Rectangle((0, 0), 1, 1, fc='#FF5252', alpha=0.4),
             plt.Rectangle((0, 0), 1, 1, fc='#666', alpha=0.3)],
            ['Building (slope > thresh, >5s)', 'Decaying (slope < -thresh, >3s)', 'Steady'],
            loc='upper right', framealpha=0.8, fontsize=8
        )
        ax.grid(True, axis='x', alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    # ── Vibes panels ─────────────────────────────────────────────

    def _build_vibes_lightness_panel(self, ax):
        """Lightness: centroid remapped to perceptual brightness (0-1)."""
        from scipy.ndimage import gaussian_filter1d
        t = self.times
        n = len(t)
        raw = self.vibes_lightness[:n]
        smooth = gaussian_filter1d(raw, sigma=int(7 * self.sr / self.hop_length))  # 7s EMA-like

        ax.fill_between(t, raw, alpha=0.15, color='#FFD54F')
        ax.plot(t, raw, color='#FFD54F', linewidth=0.5, alpha=0.3)
        ax.plot(t, smooth, color='#FFD54F', linewidth=2.5, label='Lightness (7s smooth)')

        ax.set_xlim([0, self.duration])
        ax.set_ylim([-0.05, 1.05])
        ax.set_ylabel('Lightness (0=dark, 1=bright)')
        ax.set_title('Lightness — centroid mapped to L channel (not hue)', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_vibes_warmth_panel(self, ax):
        """Warmth: spectral slope — orthogonal to brightness (Zacharakis 2014)."""
        from scipy.ndimage import gaussian_filter1d
        t = self.times
        n = len(t)
        raw = self.vibes_warmth[:n]
        smooth = gaussian_filter1d(raw, sigma=int(3 * self.sr / self.hop_length))  # 3s

        # Color: warm (positive) = orange/red, cold (negative) = blue
        pos = np.maximum(smooth, 0)
        neg = np.minimum(smooth, 0)
        ax.fill_between(t, pos, color='#FF7043', alpha=0.5, linewidth=0)
        ax.fill_between(t, neg, color='#42A5F5', alpha=0.5, linewidth=0)
        ax.plot(t, smooth, color='white', linewidth=1.5, alpha=0.8)
        ax.axhline(0, color='#666', linewidth=0.5)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Warmth (slope)')
        ax.set_title('Warmth — spectral slope (orange=warm, blue=cold) — independent of brightness',
                      fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_vibes_roughness_panel(self, ax):
        """Roughness: AM proxy vs Vassilakis sensory roughness side by side."""
        from scipy.ndimage import gaussian_filter1d
        t = self.times
        n = len(t)
        sigma_frames = int(0.5 * self.sr / self.hop_length)  # 0.5s

        # AM proxy (quick, band-envelope modulation depth)
        am_raw = self.vibes_roughness[:n]
        am_smooth = gaussian_filter1d(am_raw, sigma=sigma_frames)
        # Normalize AM to 0-1 for visual comparison
        am_max = np.percentile(am_smooth[am_smooth > 0], 99) if np.any(am_smooth > 0) else 1.0
        am_norm = np.clip(am_smooth / max(am_max, 1e-10), 0, 1)

        # Vassilakis (peak-based Plomp-Levelt)
        vass_raw = self.vibes_roughness_vassilakis[:n]
        vass_smooth = gaussian_filter1d(vass_raw, sigma=sigma_frames)

        ax.fill_between(t, am_norm, alpha=0.2, color='#EF5350')
        ax.plot(t, am_norm, color='#EF5350', linewidth=1.5, alpha=0.7,
                label='AM proxy (band-envelope)')
        ax.plot(t, vass_smooth, color='#FFD740', linewidth=2.5,
                label='Vassilakis (peak-based)')

        ax.set_xlim([0, self.duration])
        ax.set_ylim([-0.02, 1.05])
        ax.set_ylabel('Roughness (0-1)')
        ax.set_title('Roughness — AM proxy (red) vs Vassilakis sensory dissonance (gold)',
                      fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_vibes_fluctuation_panel(self, ax):
        """Fluctuation strength proxy: slow AM energy (~groove/breathing/pulse)."""
        from scipy.ndimage import gaussian_filter1d
        t = self.times
        n = len(t)
        raw = self.vibes_fluctuation[:n]
        smooth = gaussian_filter1d(raw, sigma=int(2 * self.sr / self.hop_length))  # 2s

        ax.fill_between(t, smooth, alpha=0.4, color='#AB47BC')
        ax.plot(t, raw, color='#AB47BC', linewidth=0.5, alpha=0.2)
        ax.plot(t, smooth, color='#AB47BC', linewidth=2, label='Fluctuation (2s smooth)')

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Fluctuation (slow AM)')
        ax.set_title('Fluctuation Strength — slow amplitude modulation ≈ pulse/groove/breathing',
                      fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_vibes_fullness_panel(self, ax):
        """Fullness: spectral spread (bandwidth around centroid)."""
        from scipy.ndimage import gaussian_filter1d
        t = self.times
        n = len(t)
        raw = self.vibes_fullness[:n]
        smooth = gaussian_filter1d(raw, sigma=int(3 * self.sr / self.hop_length))  # 3s

        ax.fill_between(t, smooth, alpha=0.3, color='#26A69A')
        ax.plot(t, raw, color='#26A69A', linewidth=0.5, alpha=0.2)
        ax.plot(t, smooth, color='#26A69A', linewidth=2, label='Fullness (3s smooth)')

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Bandwidth (Hz)')
        ax.set_title('Fullness — spectral spread (wide=full/rich, narrow=thin/focused)', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_vibes_timbral_panel(self, ax):
        """Timbral shape: MFCC heatmap showing spectral envelope character over time.
        Uses global percentile [2,98] normalization per coefficient."""
        t = self.times
        n = len(t)

        # MFCCs 1-12 (skip 0 = overall energy, already captured by other panels)
        mfcc_display = self.mfccs[1:13, :n]

        # Normalize each coefficient independently so all rows are visible
        mfcc_norm = np.zeros_like(mfcc_display, dtype=float)
        for i in range(mfcc_display.shape[0]):
            row = mfcc_display[i]
            lo, hi = np.percentile(row, [2, 98])
            span = hi - lo if hi > lo else 1.0
            mfcc_norm[i] = np.clip((row - lo) / span, 0, 1)

        ax.imshow(mfcc_norm, aspect='auto', origin='lower', cmap='magma',
                  extent=[0, self.duration, 1, 13], interpolation='bilinear')
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('MFCC')
        ax.set_yticks(np.arange(1, 13) + 0.5)
        ax.set_yticklabels([str(i) for i in range(1, 13)])

        # Draw timbral section transition markers
        self._draw_timbral_markers(ax)

        ax.set_title('Timbral Shape — global normalization (percentile [2,98])',
                     fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _draw_timbral_markers(self, ax):
        """Draw timbral section transition markers."""
        for i, tt in enumerate(getattr(self, 'timbral_transitions', [])):
            ax.axvline(x=tt, color='#00FF88', linewidth=1.5, alpha=0.85, linestyle='--')
            ax.text(tt, 12.6, f'T{i+1}', color='#00FF88', fontsize=7,
                    fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='black',
                              edgecolor='#00FF88', alpha=0.7, linewidth=0.5))

    def _build_vibes_timbral_local_panel(self, ax):
        """Timbral shape: MFCC heatmap with local windowed normalization + gamma.
        30-second sliding window min/max per coefficient, gamma 0.6 for midtone contrast."""
        t = self.times
        n = len(t)

        # MFCCs 1-12 (skip 0 = overall energy, already captured by other panels)
        mfcc_display = self.mfccs[1:13, :n]

        # Local windowed normalization per coefficient (30s sliding window).
        # Global percentile [2,98] crushes contrast for ambient tracks where
        # the full-song range dwarfs local variation. A sliding window preserves
        # local contrast so subtle timbral shifts remain visible.
        from scipy.ndimage import minimum_filter1d, maximum_filter1d
        fps = n / self.duration if self.duration > 0 else 43.0
        win_frames = max(1, int(30.0 * fps))  # 30s window
        mfcc_norm = np.zeros_like(mfcc_display, dtype=float)
        for i in range(mfcc_display.shape[0]):
            row = mfcc_display[i]
            local_min = minimum_filter1d(row, size=win_frames, mode='nearest')
            local_max = maximum_filter1d(row, size=win_frames, mode='nearest')
            span = np.maximum(local_max - local_min, 1e-6)
            mfcc_norm[i] = np.clip((row - local_min) / span, 0, 1)
        # Gamma 0.6 boosts midtone contrast — dark regions become more
        # visible without clipping bright regions.
        mfcc_norm = np.power(mfcc_norm, 0.6)

        ax.imshow(mfcc_norm, aspect='auto', origin='lower', cmap='magma',
                  extent=[0, self.duration, 1, 13], interpolation='bilinear')
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('MFCC')
        ax.set_yticks(np.arange(1, 13) + 0.5)
        ax.set_yticklabels([str(i) for i in range(1, 13)])

        # Draw timbral section transition markers
        self._draw_timbral_markers(ax)

        ax.set_title(u'Timbral Shape \u2014 local contrast (30s window, \u03b3=0.6)',
                     fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_figure(self):
        print("Building visualization...")
        plt.style.use('dark_background')

        # Clear matplotlib default keybindings that conflict with our feature toggles
        for param in ['keymap.zoom', 'keymap.back', 'keymap.forward']:
            plt.rcParams[param] = []

        # Determine if we need annotation panel
        has_annotations = bool(self.annotations) or self.annotate_layer is not None

        if self.panels is not None:
            pass  # figure created below in panels branch
        elif self.focus_panel is not None:
            self.fig = plt.figure(figsize=(18, 10))
        elif has_annotations:
            self.fig = plt.figure(figsize=(18, 36))
        else:
            self.fig = plt.figure(figsize=(18, 32))

        # Determine which panels to show
        # Focus panel mode: show one panel maximized
        focus_builders = {
            'waveform': self._build_waveform_panel,
            'spectrogram': self._build_spectrogram_panel,
            'bands': self._build_band_energy_panel,
            'bands-aw': self._build_band_energy_aw_panel,
            'rms-derivative': self._build_rms_derivative_panel,
            'centroid': self._build_centroid_panel,
            'centroid-derivative': self._build_centroid_derivative_panel,
            'band-derivative': self._build_band_derivative_panel,
            'band-rt': self._build_band_rt_panel,
            'band-integral': self._build_band_integral_panel,
            'mfcc': self._build_mfcc_panel,
            'novelty': self._build_novelty_panel,
            'foote': self._build_foote_panel,
            'band-deviation': self._build_band_deviation_panel,
            'event-drops': self._build_event_drops_panel,
            'event-risers': self._build_event_risers_panel,
            'event-dropouts': self._build_event_dropouts_panel,
            'event-harmonic': self._build_event_harmonic_panel,
            'calc-energy-integral': self._build_calc_energy_integral_panel,
            'calc-integral-slope': self._build_calc_integral_slope_panel,
            'calc-slope-peaks': self._build_calc_slope_peaks_panel,
            'calc-integral-curvature': self._build_calc_integral_curvature_panel,
            'calc-multi-scale': self._build_calc_multi_scale_panel,
            'calc-onset-d2': self._build_calc_onset_d2_panel,
            'calc-jitter': self._build_calc_jitter_panel,
            'calc-absint-d2': self._build_calc_absint_d2_panel,
            'calc-absint-jitter': self._build_calc_absint_jitter_panel,
            'calc-build-detector': self._build_calc_build_detector_panel,
            'vibes-lightness': self._build_vibes_lightness_panel,
            'vibes-warmth': self._build_vibes_warmth_panel,
            'vibes-roughness': self._build_vibes_roughness_panel,
            'vibes-fluctuation': self._build_vibes_fluctuation_panel,
            'vibes-fullness': self._build_vibes_fullness_panel,
            'vibes-timbral': self._build_vibes_timbral_panel,
            'vibes-timbral-local': self._build_vibes_timbral_local_panel,
        }
        if has_annotations:
            focus_builders['annotations'] = self._build_annotation_panel

        if self.panels is not None:
            # Custom panel selection (e.g. annotate tab: waveform + spectrogram only)
            builders = [(name, focus_builders[name]) for name in self.panels
                        if name in focus_builders]
            n_panels = len(builders)
            height_map = {'waveform': 1, 'spectrogram': 2, 'bands': 1, 'bands-aw': 1,
                          'rms-derivative': 1, 'centroid': 1, 'centroid-derivative': 1,
                          'band-derivative': 1, 'mfcc': 1.5, 'novelty': 1.5, 'foote': 1.5,
                          'band-deviation': 1.5, 'annotations': None,
                          'event-drops': 1.0, 'event-risers': 0.5,
                          'event-dropouts': 1.0, 'event-harmonic': 1.0,
                          'calc-energy-integral': 1, 'calc-integral-slope': 1,
                          'calc-slope-peaks': 1.2, 'calc-integral-curvature': 1,
                          'calc-multi-scale': 1,
                          'calc-onset-d2': 1, 'calc-jitter': 1,
                          'calc-build-detector': 1.2,
                          'vibes-lightness': 1, 'vibes-warmth': 1,
                          'vibes-roughness': 1, 'vibes-fluctuation': 1,
                          'vibes-fullness': 1,
                          'vibes-timbral': 1.5,
                          'vibes-timbral-local': 1.5}
            ratios = [height_map.get(name, 1) for name, _ in builders]
            if has_annotations:
                n_layers = len(self.annotations) + (1 if self.annotate_layer else 0)
                ann_height = max(0.8, 0.4 * n_layers)
                builders.append(('annotations', self._build_annotation_panel))
                ratios.append(ann_height)
            total = len(builders)
            fig_height = sum(ratios) * 2 + 2
            self.fig = plt.figure(figsize=(18, fig_height))
            gs = gridspec.GridSpec(total, 1, height_ratios=ratios, hspace=0.3)
            for i, (name, builder) in enumerate(builders):
                builder(self.fig.add_subplot(gs[i]))
        elif self.focus_panel in focus_builders:
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            focus_builders[self.focus_panel](self.fig.add_subplot(gs[0]))
        elif has_annotations:
            n_layers = len(self.annotations) + (1 if self.annotate_layer else 0)
            ann_height = max(0.8, 0.4 * n_layers)
            gs = gridspec.GridSpec(11, 1, height_ratios=[1, 2, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, ann_height], hspace=0.3)
            self._build_waveform_panel(self.fig.add_subplot(gs[0]))
            self._build_spectrogram_panel(self.fig.add_subplot(gs[1]))
            self._build_band_energy_panel(self.fig.add_subplot(gs[2]))
            self._build_rms_derivative_panel(self.fig.add_subplot(gs[3]))
            self._build_centroid_panel(self.fig.add_subplot(gs[4]))
            self._build_centroid_derivative_panel(self.fig.add_subplot(gs[5]))
            self._build_band_derivative_panel(self.fig.add_subplot(gs[6]))
            self._build_mfcc_panel(self.fig.add_subplot(gs[7]))
            self._build_novelty_panel(self.fig.add_subplot(gs[8]))
            self._build_band_deviation_panel(self.fig.add_subplot(gs[9]))
            self._build_annotation_panel(self.fig.add_subplot(gs[10]))
        else:
            gs = gridspec.GridSpec(10, 1, height_ratios=[1, 2, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5], hspace=0.3)
            self._build_waveform_panel(self.fig.add_subplot(gs[0]))
            self._build_spectrogram_panel(self.fig.add_subplot(gs[1]))
            self._build_band_energy_panel(self.fig.add_subplot(gs[2]))
            self._build_rms_derivative_panel(self.fig.add_subplot(gs[3]))
            self._build_centroid_panel(self.fig.add_subplot(gs[4]))
            self._build_centroid_derivative_panel(self.fig.add_subplot(gs[5]))
            self._build_band_derivative_panel(self.fig.add_subplot(gs[6]))
            self._build_mfcc_panel(self.fig.add_subplot(gs[7]))
            self._build_novelty_panel(self.fig.add_subplot(gs[8]))
            self._build_band_deviation_panel(self.fig.add_subplot(gs[9]))

        self._finalize_figure()

    def _build_waveform_panel(self, ax):
        waveform_times = np.linspace(0, self.duration, len(self.y))
        ax.plot(waveform_times, self.y, color='#FFFFFF', linewidth=0.5, alpha=0.8)

        # RMS energy overlay (normalized to waveform amplitude range, hidden by default)
        rms_norm = self.rms_energy / np.max(self.rms_energy)
        y_max = np.max(np.abs(self.y)) if np.max(np.abs(self.y)) > 0 else 1.0
        rms_scaled = rms_norm * y_max
        rms_line, = ax.plot(self.times, rms_scaled,
                            color='#FFD740', linewidth=2, label='RMS [E]',
                            alpha=0.8, visible=False)
        # Store for feature toggle system
        if not hasattr(self, 'feature_toggle'):
            self.feature_toggle = {}
            self.feature_keys = {}
        self.feature_toggle['rms'] = {'artists': [rms_line], 'visible': False}
        self.feature_keys['e'] = 'rms'

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform — [E] toggle RMS overlay', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_spectrogram_panel(self, ax):
        librosa.display.specshow(
            self.mel_spec_db, sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='mel', fmin=20, fmax=None,
            ax=ax, cmap='magma'
        )
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Mel Spectrogram — full frequency range (20 Hz - 22 kHz)', fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_energy_panel(self, ax):
        for band_name, energy in self.band_energies.items():
            ax.plot(self.times, energy, label=band_name,
                    color=BAND_COLORS[band_name], linewidth=2, alpha=0.9)
        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Normalized Energy')
        ax.set_title('Band Energy', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        max_ratio = max(self.band_ratios.values())
        ratio_text = "Magnitude ratios:\n"
        for name, ratio in self.band_ratios.items():
            ratio_text += f"  {name}: {ratio / max_ratio:.2f}\n"
        ax.text(0.02, 0.98, ratio_text, transform=ax.transAxes,
                va='top', fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_energy_aw_panel(self, ax):
        """A-weighted band energy: perceptually weighted before mel grouping, shared normalization."""
        for band_name, energy in self.band_energies_aw.items():
            ax.plot(self.times, energy, label=band_name,
                    color=BAND_COLORS[band_name], linewidth=2, alpha=0.9)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Energy (shared scale)')
        ax.set_title('Band Energy (A-weighted, shared norm) — '
                     'all bands on same scale, bass attenuated to match hearing', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        max_ratio = max(self.band_ratios_aw.values()) if self.band_ratios_aw else 1
        ratio_text = "A-weighted ratios:\n"
        for name, ratio in self.band_ratios_aw.items():
            ratio_text += f"  {name}: {ratio / max_ratio:.2f}\n"
        ax.text(0.02, 0.98, ratio_text, transform=ax.transAxes,
                va='top', fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_rms_derivative_panel(self, ax):
        """RMS derivative: rate-of-change of loudness. Positive = getting louder, negative = getting quieter."""
        pos = np.maximum(self.rms_derivative, 0)
        neg = np.minimum(self.rms_derivative, 0)

        ax.fill_between(self.times, pos, color='#FF5252', alpha=0.7, linewidth=0)
        ax.fill_between(self.times, neg, color='#448AFF', alpha=0.7, linewidth=0)
        ax.axhline(y=0, color='#666', linewidth=0.5)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('dRMS/dt')
        ax.set_title('RMS Derivative — red = getting louder, blue = getting quieter', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_centroid_panel(self, ax):
        """Spectral centroid: center of mass of the frequency spectrum."""
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(self.spectral_centroid, sigma=5)

        ax.plot(self.times, self.spectral_centroid, color='#B388FF', linewidth=0.5, alpha=0.3)
        ax.plot(self.times, smoothed, color='#B388FF', linewidth=2)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Center of Mass of Frequency — how bright the sound is right now', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_centroid_derivative_panel(self, ax):
        """Spectral centroid derivative: rising = getting brighter, falling = getting darker."""
        # Smooth to reduce noise (centroid is noisy frame-to-frame)
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(self.centroid_derivative, sigma=5)

        pos = np.maximum(smoothed, 0)
        neg = np.minimum(smoothed, 0)

        ax.fill_between(self.times, pos, color='#B388FF', alpha=0.7, linewidth=0)
        ax.fill_between(self.times, neg, color='#00BFA5', alpha=0.7, linewidth=0)
        ax.axhline(y=0, color='#666', linewidth=0.5)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('dCentroid/dt (Hz/s)')
        ax.set_title('Centroid Derivative — purple = brighter, teal = darker', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_derivative_panel(self, ax):
        """Per-band energy derivatives: rate-of-change per frequency band."""
        from scipy.ndimage import gaussian_filter1d

        for band_name, deriv in self.band_derivatives.items():
            smoothed = gaussian_filter1d(deriv, sigma=5)
            ax.plot(self.times, smoothed, label=band_name,
                    color=BAND_COLORS[band_name], linewidth=1.5, alpha=0.85)

        ax.axhline(y=0, color='#666', linewidth=0.5)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('dEnergy/dt')
        ax.set_title('Band Energy Derivative — per-band rate of change', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_mfcc_panel(self, ax):
        """MFCC heatmap: the timbral fingerprint over time."""
        ax.imshow(self.mfccs, aspect='auto', origin='lower', cmap='magma',
                  extent=[0, self.duration, 0, 13])
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('MFCC')
        ax.set_yticks(np.arange(13) + 0.5)
        ax.set_yticklabels([str(i) for i in range(13)])
        ax.set_title('Timbral Shape (MFCC) — vertical color shifts = the sound character changed', fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_novelty_panel(self, ax):
        """Causal novelty: flux (sharp edges) + EMA deviation (gradual drift)
        + Foote's checkerboard (offline structural boundaries).
        Chroma on left axis, MFCC + Foote on right axis (independent scales)."""
        from scipy.signal import find_peaks

        t = self.times
        n = len(t)
        nov_m = self.novelty_mfcc[:n]
        nov_c = self.novelty_chroma[:n]
        ema_m = self.ema_mfcc[:n]
        ema_c = self.ema_chroma[:n]
        foote_m = self.foote_mfcc[:n]

        fps = self.sr / self.hop_length
        peak_dist = max(1, int(fps * 1.5))

        # Chroma on left axis
        line_c, = ax.plot(t, nov_c, color='#4DD0E1', linewidth=1.2, alpha=0.9, label='Chroma flux')
        ax.plot(t, ema_c, color='#4DD0E1', linewidth=1.0, alpha=0.4, linestyle='--', label='Chroma EMA')
        peaks_c, _ = find_peaks(nov_c, prominence=0.15, distance=peak_dist)
        ax.scatter(t[peaks_c], nov_c[peaks_c], color='#4DD0E1', s=25, zorder=5, marker='v')
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Chroma novelty', color='#4DD0E1')
        ax.tick_params(axis='y', labelcolor='#4DD0E1')
        ax.grid(True, alpha=0.2)

        # MFCC on right axis (independent scale)
        ax2 = ax.twinx()
        line_m, = ax2.plot(t, nov_m, color='#FF8A65', linewidth=1.2, alpha=0.9, label='MFCC flux')
        ax2.plot(t, ema_m, color='#FF8A65', linewidth=1.0, alpha=0.4, linestyle='--', label='MFCC EMA')
        peaks_m, _ = find_peaks(nov_m, prominence=0.15, distance=peak_dist)
        ax2.scatter(t[peaks_m], nov_m[peaks_m], color='#FF8A65', s=25, zorder=5, marker='v')
        ax2.set_ylabel('MFCC novelty', color='#FF8A65')
        ax2.tick_params(axis='y', labelcolor='#FF8A65')

        # Foote's offline novelty (gold standard reference) on MFCC axis
        line_f, = ax2.plot(t, foote_m, color='#FFD740', linewidth=1.8, alpha=0.85,
                           label="Foote's (offline)")
        ax2.fill_between(t, foote_m, alpha=0.1, color='#FFD740')
        peaks_f, _ = find_peaks(foote_m, prominence=0.15, distance=peak_dist)
        ax2.scatter(t[peaks_f], foote_m[peaks_f], color='#FFD740', s=25, zorder=5, marker='D')

        ax.set_title(
            "Novelty — causal flux (solid) vs Foote's checkerboard (gold, offline)  "
            "[V] events (todo)", fontsize=11)
        ax.legend([line_c, line_m, line_f],
                  ['Chroma flux (harmonic)', 'MFCC flux (timbral)',
                   "Foote's (offline, structural)"],
                  loc='upper right', framealpha=0.8, fontsize=8)

        # V:events — placeholder for future real-time event overlay (todo)
        if not hasattr(self, 'feature_toggle'):
            self.feature_toggle = {}
            self.feature_keys = {}
        self.feature_toggle['events'] = {'artists': [], 'visible': False}
        self.feature_keys['v'] = 'events'

        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_foote_panel(self, ax):
        """Foote's checkerboard novelty — chroma (teal) + MFCC (gold), offline reference."""
        from scipy.signal import find_peaks

        t = self.times
        n = len(t)
        foote_m = self.foote_mfcc[:n]
        foote_c = self.foote_chroma[:n]

        fps = self.sr / self.hop_length
        peak_dist = max(1, int(fps * 1.5))

        # Chroma Foote
        ax.plot(t, foote_c, color='#4DD0E1', linewidth=1.5, alpha=0.9, label='Foote chroma')
        ax.fill_between(t, foote_c, alpha=0.1, color='#4DD0E1')
        peaks_c, _ = find_peaks(foote_c, prominence=0.15, distance=peak_dist)
        ax.scatter(t[peaks_c], foote_c[peaks_c], color='#4DD0E1', s=25, zorder=5, marker='D')

        # MFCC Foote
        ax.plot(t, foote_m, color='#FFD740', linewidth=1.8, alpha=0.85, label='Foote MFCC')
        ax.fill_between(t, foote_m, alpha=0.1, color='#FFD740')
        peaks_m, _ = find_peaks(foote_m, prominence=0.15, distance=peak_dist)
        ax.scatter(t[peaks_m], foote_m[peaks_m], color='#FFD740', s=25, zorder=5, marker='D')

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Novelty')
        ax.set_title("Foote's Checkerboard — chroma (teal, harmonic) + MFCC (gold, timbral)  "
                     "[OFFLINE · O(n²)]", fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    # ── Event timeline panels (one per event type) ─────────────────

    def _build_event_drops_panel(self, ax):
        """Drops: MFCC novelty peaks scored with RMS + deviation boosters."""
        t = self.times
        n = len(t)
        nov_m = self.novelty_mfcc[:n]

        # MFCC novelty as context trace
        ax.plot(t, nov_m, color='#FF8A65', linewidth=1.0, alpha=0.5, label='MFCC novelty')
        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, max(1.1, max(self.event_drop_scores) * 1.1) if self.event_drop_scores else 1.1])

        # Drop lines with visual weight proportional to score
        max_score = max(self.event_drop_scores) if self.event_drop_scores else 1.0
        for drop_t, score in zip(self.event_drops, self.event_drop_scores):
            # Linewidth 1-4 based on score
            lw = 1.0 + 3.0 * (score / max_score)
            alpha = 0.5 + 0.5 * (score / max_score)
            ax.axvline(x=drop_t, color='#FF1744', linestyle='--',
                       linewidth=lw, alpha=alpha)
            # Score label
            ax.text(drop_t, ax.get_ylim()[1] * 0.92, f'{score:.2f}',
                    ha='center', va='top', fontsize=7, color='#FF1744',
                    fontweight='bold', rotation=45,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))

        nd = len(self.event_drops)
        ax.set_ylabel('Novelty / Score')
        ax.set_title(f'Drops — {nd} detected  '
                     f'(MFCC novelty peaks, scored with RMS + deviation boosters)', fontsize=11)
        ax.legend(loc='upper left', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_event_risers_panel(self, ax):
        """Risers: sustained upward spectral centroid movement."""
        for start, end in self.event_risers:
            ax.axvspan(start, end, color='#76FF03', alpha=0.4)
        ax.set_xlim([0, self.duration])
        ax.set_yticks([])
        n = len(self.event_risers)
        ax.set_title(f'Risers — {n} detected  '
                     f'(centroid derivative positive >1s)', fontsize=11)
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc='#76FF03', alpha=0.4)],
                  ['Riser (pitch/brightness climbing)'],
                  loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, axis='x', alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_event_dropouts_panel(self, ax):
        """Dropouts: presence-gated per-band transitions with RT energy traces."""
        t = self.times
        n = len(t)

        # Algorithm-dependent thresholds for display
        if self.event_algorithm == 'a':
            presence_display, low_display = 0.20, 0.08
        else:
            presence_display, low_display = 0.15, 0.05

        # RT-normalized band energy as thin traces (all 5 bands for context)
        for band_name in FREQUENCY_BANDS:
            e_rt = self.band_energy_rt[band_name][:n]
            ax.plot(t, e_rt, color=BAND_COLORS[band_name],
                    linewidth=0.8, alpha=0.4, label=band_name)

        # Threshold reference lines
        ax.axhline(y=presence_display, color='#888', linewidth=0.5, linestyle=':', alpha=0.6)
        ax.axhline(y=low_display, color='#888', linewidth=0.5, linestyle=':', alpha=0.6)
        ax.text(self.duration * 0.99, presence_display, 'presence', ha='right', va='bottom',
                fontsize=6, color='#888', alpha=0.6)
        ax.text(self.duration * 0.99, low_display, 'low', ha='right', va='bottom',
                fontsize=6, color='#888', alpha=0.6)

        # Dropout spans as colored overlays
        handles = []
        labels = []
        for band_name, spans in self.event_dropouts.items():
            color = BAND_COLORS.get(band_name, '#E040FB')
            for start, end in spans:
                ax.axvspan(start, end, color=color, alpha=0.3)
            if spans:
                handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3))
                labels.append(f'{band_name} dropout ({len(spans)})')

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('RT Energy')
        total = sum(len(v) for v in self.event_dropouts.values())
        ax.set_title(f'Dropouts — {total} detected  '
                     f'(presence-gated: was >{presence_display:.0%}, now <{low_display:.0%})', fontsize=11)
        if handles:
            ax.legend(handles, labels, loc='upper right',
                      framealpha=0.8, fontsize=8, ncol=min(len(handles), 3))
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_event_harmonic_panel(self, ax):
        """Harmonic sections: onset/median ratio + spectral flatness diagnostic."""
        t = self.times
        n = len(t)

        # Algorithm-dependent thresholds for display
        if self.event_algorithm == 'a':
            primary_ratio, lenient_ratio = 0.25, 0.40
        else:
            primary_ratio, lenient_ratio = 0.4, 0.65

        # Harmonic spans as background shading
        for start, end in self.event_harmonic:
            ax.axvspan(start, end, color='#FFD740', alpha=0.15)

        # Onset ratio trace (primary signal)
        ratio = self.event_harmonic_ratio[:n]
        ax.plot(t, ratio, color='#00E5FF', linewidth=1.2, alpha=0.9, label='Onset/median ratio')

        # Threshold lines
        ax.axhline(y=primary_ratio, color='#FF5252', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.axhline(y=lenient_ratio, color='#FF9100', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.text(self.duration * 0.99, primary_ratio, 'primary', ha='right', va='bottom',
                fontsize=6, color='#FF5252', alpha=0.7)
        ax.text(self.duration * 0.99, lenient_ratio, 'lenient', ha='right', va='bottom',
                fontsize=6, color='#FF9100', alpha=0.6)

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, max(2.0, np.percentile(ratio, 99) * 1.1)])
        ax.set_ylabel('Onset/Median Ratio', color='#00E5FF')
        ax.tick_params(axis='y', labelcolor='#00E5FF')

        # Spectral flatness on twin y-axis
        ax2 = ax.twinx()
        flatness = self.event_harmonic_flatness[:n]
        ax2.plot(t, flatness, color='#E040FB', linewidth=1.0, alpha=0.7, label='Spectral flatness')
        ax2.axhline(y=0.15, color='#E040FB', linewidth=0.5, linestyle=':', alpha=0.4)
        ax2.set_ylabel('Spectral Flatness', color='#E040FB')
        ax2.tick_params(axis='y', labelcolor='#E040FB')

        nh = len(self.event_harmonic)
        harmonic_dur = sum(e - s for s, e in self.event_harmonic)
        ax.set_title(f'Harmonic — {nh} spans ({harmonic_dur:.1f}s)  '
                     f'(onset < local median + flatness corroboration)', fontsize=11)

        lines1 = [plt.Line2D([0], [0], color='#00E5FF', linewidth=1.2),
                   plt.Line2D([0], [0], color='#E040FB', linewidth=1.0),
                   plt.Rectangle((0, 0), 1, 1, fc='#FFD740', alpha=0.15)]
        ax.legend(lines1, ['Onset/median ratio', 'Spectral flatness', 'Harmonic span'],
                  loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        self._overlay_sections(ax)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_deviation_panel(self, ax):
        """Band deviation from long-term context: per-band z-scores."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_deviations[band_name][:n], color=BAND_COLORS[band_name],
                    linewidth=1.0, alpha=0.7, label=band_name)
        ax.plot(t, self.composite_deviation[:n], color='#FFFFFF',
                linewidth=1.5, alpha=0.9, label='Max (any band)')

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Z-score')
        cs = self.band_deviation_context_seconds
        ax.set_title(
            f'Band Deviation from {cs}s Running Context '
            f'— high = spectral balance shifted', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    # ── Band Analysis panels ─────────────────────────────────────────

    def _build_band_rt_panel(self, ax):
        """Per-band normalized energy: asymmetric EMA + dropout detection."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_energy_rt[band_name][:n],
                    color=BAND_COLORS[band_name], linewidth=1.5, alpha=0.85,
                    label=band_name)
        ax.set_xlim([0, self.duration])
        # Auto-scale y: values can exceed 1.0 after dropout reintroduction
        all_vals = np.concatenate([self.band_energy_rt[b][:n] for b in FREQUENCY_BANDS])
        y_max = min(np.percentile(all_vals, 99.5) * 1.1, 5.0)
        ax.set_ylim([0, max(y_max, 1.1)])
        ax.set_ylabel('Normalized Energy')
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        # Dual y-axis: total raw energy as thin white line
        ax2 = ax.twinx()
        total = self.total_raw_energy[:n]
        ax2.plot(t, total, color='#FFFFFF', linewidth=0.8, alpha=0.5)
        ax2.set_ylabel('Total Raw', color='#888')
        ax2.tick_params(axis='y', labelcolor='#888')

        ax.set_title('RT Normalized (asymmetric EMA, dropout detection)', fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_integral_panel(self, ax):
        """5s rolling integral of real-time normalized energy."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_integral_5s[band_name][:n],
                    color=BAND_COLORS[band_name], linewidth=1.5, alpha=0.85,
                    label=band_name)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Integrated Energy')
        ax.set_title('5s Rolling Integral', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_features_panel(self, ax):
        """Second-class features: onset strength, centroid (hidden by default)."""
        self.features_ax = ax

        # Onset strength line + detection markers (hidden by default)
        onset_norm = self.onset_env / np.max(self.onset_env)
        onset_line, = ax.plot(self.times, onset_norm,
                              color='#00E5FF', linewidth=1.5, label='Onset [O]',
                              alpha=0.9, visible=False)
        onset_markers = []
        for t in self.onset_times:
            m = ax.axvline(x=t, color='#FFC400', alpha=0.15, linewidth=0.8, visible=False)
            onset_markers.append(m)

        # Spectral centroid (normalized, hidden by default)
        centroid_norm = (self.spectral_centroid - np.min(self.spectral_centroid))
        centroid_range = np.max(self.spectral_centroid) - np.min(self.spectral_centroid)
        if centroid_range > 0:
            centroid_norm = centroid_norm / centroid_range
        centroid_line, = ax.plot(self.times, centroid_norm,
                                 color='#B388FF', linewidth=2, label='Centroid [C]',
                                 alpha=0.9, visible=False)

        # Librosa-beats markers (hidden by default unless --show-beats)
        beat_markers = []
        for t in self.beat_times:
            m = ax.axvline(x=t, color='#FF1744', alpha=0.6, linewidth=2,
                           visible=self.show_beats)
            beat_markers.append(m)

        # Feature toggle state (onset, centroid, beats — RMS is on waveform panel)
        self.feature_toggle.update({
            'onset': {'artists': [onset_line] + onset_markers, 'visible': False},
            'centroid': {'artists': [centroid_line], 'visible': False},
            'librosa-beats': {'artists': beat_markers, 'visible': self.show_beats},
        })
        self.feature_keys.update({'o': 'onset', 'c': 'centroid', 'b': 'librosa-beats'})

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Normalized')
        self._update_features_title()
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _update_features_title(self):
        """Update features panel title to reflect toggle states."""
        if not hasattr(self, 'features_ax'):
            return

    def _toggle_feature(self, name):
        """Toggle visibility of a feature overlay."""
        feat = self.feature_toggle[name]
        feat['visible'] = not feat['visible']
        for artist in feat['artists']:
            artist.set_visible(feat['visible'])
        self._update_features_title()
        self.fig.canvas.draw_idle()
        print(f"{name}: {'on' if feat['visible'] else 'off'}")

    def _is_segment_layer(self, layer_data):
        """Check if a layer contains segment spans (list of dicts) vs point taps (list of floats)."""
        return (isinstance(layer_data, list) and len(layer_data) > 0
                and isinstance(layer_data[0], dict) and 'start' in layer_data[0])

    def _draw_layer(self, ax, layer_name, layer_data, y_pos, color):
        """Draw a single annotation layer — handles both point taps and segment spans."""
        if self._is_segment_layer(layer_data):
            # Segment spans: colored bars with centered labels
            for seg in layer_data:
                start = seg['start']
                end = seg['end']
                label = seg.get('label', '')
                if start <= self.duration:
                    ax.barh(y_pos, width=end - start, left=start, height=0.6,
                            color=color, alpha=0.25, edgecolor=color, linewidth=0.5)
                    mid = (start + min(end, self.duration)) / 2
                    ax.text(mid, y_pos, label, ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold', alpha=0.9)
        else:
            # Point taps: vertical markers
            taps = layer_data if isinstance(layer_data, list) else []
            for t in taps:
                if isinstance(t, (int, float)) and t <= self.duration:
                    ax.plot(t, y_pos, '|', color=color, markersize=12, markeredgewidth=2, alpha=0.8)

    def _build_annotation_panel(self, ax):
        """Annotation panel: existing layers + live annotation layer."""
        self.annotation_ax = ax  # Save reference for live tap rendering

        layer_names = list(self.annotations.keys())
        # Add the live annotation layer slot if annotating
        if self.annotate_layer and self.annotate_layer not in layer_names:
            layer_names.append(self.annotate_layer)

        total_layers = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            y_pos = total_layers - i

            layer_data = self.annotations.get(layer_name, [])
            self._draw_layer(ax, layer_name, layer_data, y_pos, color)

            ax.text(-self.duration * 0.01, y_pos, layer_name, ha='right', va='center',
                    fontsize=8, color=color, fontweight='bold')

            # Remember the y_pos and color for the live annotation layer
            if layer_name == self.annotate_layer:
                self._annotate_y_pos = y_pos
                self._annotate_color = color

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0.3, total_layers + 0.7])
        ax.set_yticks([])

        # Count taps and segments separately
        total_taps = 0
        total_segs = 0
        for v in self.annotations.values():
            if self._is_segment_layer(v):
                total_segs += len(v)
            else:
                total_taps += len(v)
        parts = []
        if total_taps:
            parts.append(f'{total_taps} taps')
        if total_segs:
            parts.append(f'{total_segs} segments')
        count_str = ', '.join(parts) if parts else 'empty'
        ax.set_title(
            f'Annotations — {count_str} across {len(self.annotations)} layers',
            fontsize=11
        )
        ax.grid(True, axis='x', alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _finalize_figure(self):
        # Create cursor lines
        self.cursor_lines = []
        for ax in self.axes:
            line = ax.axvline(x=0, color='#FFFFFF', linewidth=1.5, alpha=0.9, linestyle='-')
            self.cursor_lines.append(line)

        filename = Path(self.filepath).name

        if self.annotate_layer:
            title = (
                f'{filename}  —  Layer: {self.annotate_layer}  —  '
                f'SPACE: tap | P: play/pause | R: restart | Q: save & quit'
            )
        else:
            title = f'{filename}  —  SPACE: play  |  CLICK: seek  |  E: RMS overlay  |  Q: quit'

        self.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        if self.annotate_layer:
            print(f"\nAnnotation mode: Layer '{self.annotate_layer}'")
            print("  SPACE  - Record a tap")
            print("  P      - Play / Pause")
            print("  R      - Restart + clear taps")
            print("  Q      - Save & quit")
        else:
            print("Ready. SPACE: play/pause  CLICK: seek  E: RMS overlay  Q: quit")

    def _on_key(self, event):
        if self.annotate_layer:
            self._on_key_annotate(event)
        else:
            self._on_key_normal(event)

    def _on_key_normal(self, event):
        if event.key == ' ':
            if not self.is_playing:
                self._start_playback()
            else:
                self._stop_playback()
        elif event.key in self.feature_keys:
            self._toggle_feature(self.feature_keys[event.key])
        elif event.key == 'q':
            self._stop_playback()
            plt.close(self.fig)

    def _on_key_annotate(self, event):
        if event.key == ' ':
            # Record tap
            if self.is_playing and self.playback_start_time is not None:
                tap_time = (time.time() - self.playback_start_time) + self.playback_offset
                if 0 <= tap_time <= self.duration:
                    self.new_taps.append(tap_time)
                    # Render tap marker on annotation panel
                    if hasattr(self, 'annotation_ax') and hasattr(self, '_annotate_y_pos'):
                        self.annotation_ax.plot(
                            tap_time, self._annotate_y_pos, '|',
                            color=self._annotate_color,
                            markersize=12, markeredgewidth=2, alpha=0.8
                        )
                        self.fig.canvas.draw_idle()
                    print(f"\r  [{tap_time:6.2f}s] TAP #{len(self.new_taps)}", end='', flush=True)
        elif event.key == 'p':
            if not self.is_playing:
                self._start_playback()
            else:
                self._stop_playback()
        elif event.key == 'r':
            # Restart: stop, clear taps, rewind, restart
            self._stop_playback()
            self.new_taps = []
            self.playback_offset = 0.0
            # Clear tap markers from annotation panel by redrawing
            if hasattr(self, 'annotation_ax'):
                ax = self.annotation_ax
                # Remove only the new tap markers (they are the most recent artists)
                # Simpler: just redraw the annotation panel
                ax.clear()
                self._rebuild_annotation_panel_contents(ax)
                self.fig.canvas.draw_idle()
            print("\r  Restarted. Taps cleared.                    ", end='', flush=True)
            time.sleep(0.1)
            self._start_playback(from_offset=0.0)
        elif event.key in self.feature_keys:
            self._toggle_feature(self.feature_keys[event.key])
        elif event.key == 'q':
            self._stop_playback()
            self._save_annotations()
            plt.close(self.fig)

    def _rebuild_annotation_panel_contents(self, ax):
        """Redraw annotation panel contents (used after restart clears taps)."""
        layer_names = list(self.annotations.keys())
        if self.annotate_layer and self.annotate_layer not in layer_names:
            layer_names.append(self.annotate_layer)

        total_layers = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            y_pos = total_layers - i

            layer_data = self.annotations.get(layer_name, [])
            self._draw_layer(ax, layer_name, layer_data, y_pos, color)

            ax.text(-self.duration * 0.01, y_pos, layer_name, ha='right', va='center',
                    fontsize=8, color=color, fontweight='bold')

            if layer_name == self.annotate_layer:
                self._annotate_y_pos = y_pos
                self._annotate_color = color

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0.3, total_layers + 0.7])
        ax.set_yticks([])
        total_taps = 0
        total_segs = 0
        for v in self.annotations.values():
            if self._is_segment_layer(v):
                total_segs += len(v)
            else:
                total_taps += len(v)
        parts = []
        if total_taps:
            parts.append(f'{total_taps} taps')
        if total_segs:
            parts.append(f'{total_segs} segments')
        count_str = ', '.join(parts) if parts else 'empty'
        ax.set_title(
            f'Annotations — {count_str} across {len(self.annotations)} layers',
            fontsize=11
        )
        ax.grid(True, axis='x', alpha=0.2)

        # Re-add cursor line for this axis
        line = ax.axvline(x=self.playback_offset, color='#FFFFFF', linewidth=1.5, alpha=0.9, linestyle='-')
        # Replace the cursor line for this axis in cursor_lines
        for idx, a in enumerate(self.axes):
            if a is ax:
                self.cursor_lines[idx] = line
                break

    def _save_annotations(self):
        """Save new taps to annotations YAML (additive layer format)."""
        if not self.new_taps:
            print(f"\n\n  No taps recorded for layer '{self.annotate_layer}'.")
            return

        # Load existing annotations from the canonical path
        all_annotations = {}
        if self.annotations_path.exists():
            with open(self.annotations_path) as f:
                all_annotations = yaml.safe_load(f) or {}

        # Add/replace the target layer
        all_annotations[self.annotate_layer] = [round(t, 3) for t in sorted(self.new_taps)]

        with open(self.annotations_path, 'w') as f:
            yaml.dump(all_annotations, f, default_flow_style=False, sort_keys=False)

        print(f"\n\n  Saved {len(self.new_taps)} taps to layer '{self.annotate_layer}'")
        print(f"  File: {self.annotations_path}")
        print(f"  Layers: {list(all_annotations.keys())}")
        for layer, taps in all_annotations.items():
            print(f"    {layer}: {len(taps)} marks")

    def _on_click(self, event):
        if self.annotate_layer:
            return  # Clicks don't seek in annotation mode

        seek_time = None

        if event.inaxes in self.axes and event.xdata is not None:
            seek_time = event.xdata
        elif event.x is not None and event.y is not None:
            try:
                inv = self.axes[0].transData.inverted()
                data_coords = inv.transform((event.x, event.y))
                seek_time = data_coords[0]
            except Exception:
                pass

        if seek_time is not None and 0 <= seek_time <= self.duration:
            self._seek_to(seek_time)
            print(f"Seeking to {seek_time:.2f}s")
        elif seek_time is not None:
            print(f"Click out of range: {seek_time:.2f}s (0-{self.duration:.1f}s)")

    def _start_playback(self, from_offset=None):
        if from_offset is not None:
            self.playback_offset = from_offset

        self.is_playing = True
        self.playback_start_time = time.time()

        start_sample = int(self.playback_offset * self.sr_playback)
        audio_to_play = self.y_playback[start_sample:]

        def play_audio():
            try:
                sd.play(audio_to_play, self.sr_playback)
                sd.wait()
                self.is_playing = False
            except Exception as e:
                print(f"Playback error: {e}")
                self.is_playing = False

        self.playback_thread = threading.Thread(target=play_audio, daemon=True)
        self.playback_thread.start()
        print(f"Playing from {self.playback_offset:.2f}s...")

    def _stop_playback(self):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            self.playback_offset = min(self.playback_offset + elapsed, self.duration)

        self.is_playing = False
        sd.stop()
        time.sleep(0.05)

        print(f"Stopped at {self.playback_offset:.2f}s.")

    def _seek_to(self, time_position):
        was_playing = self.is_playing

        if self.is_playing:
            self._stop_playback()

        self.playback_offset = time_position
        self.led_sample_pos = int(time_position * self.sr)

        for line in self.cursor_lines:
            line.set_xdata([time_position, time_position])
        self.fig.canvas.draw_idle()

        if was_playing:
            time.sleep(0.05)
            self._start_playback()

    def _update_cursor(self, frame):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            current_time = self.playback_offset + elapsed

            if current_time > self.duration:
                self.is_playing = False
                current_time = self.duration

            for line in self.cursor_lines:
                line.set_xdata([current_time, current_time])

            # Feed audio to LED effect in sync with playback
            if self.led_effect is not None:
                current_sample = int(current_time * self.sr)
                while self.led_sample_pos < current_sample and self.led_sample_pos < len(self.y):
                    chunk_end = min(self.led_sample_pos + 1024, len(self.y), current_sample)
                    chunk = self.y[self.led_sample_pos:chunk_end]
                    if len(chunk) > 0:
                        self.led_effect.process_audio(chunk)
                    self.led_sample_pos = chunk_end
                led_frame = self.led_effect.render(1.0 / 30)
                led_frame = (led_frame.astype(np.float32) * self.led_brightness).astype(np.uint8)
                self.led_output.send_frame(led_frame)
        elif not self.is_playing:
            for line in self.cursor_lines:
                line.set_xdata([self.playback_offset, self.playback_offset])

        return self.cursor_lines

    def run(self):
        # Use blit=False in annotation mode so new markers render without manual blit management
        use_blit = not bool(self.annotate_layer)
        self.anim = FuncAnimation(
            self.fig, self._update_cursor,
            interval=33, blit=use_blit, cache_frame_data=False
        )
        try:
            plt.show()
        finally:
            if self.led_output is not None:
                self.led_output.close()


# ── Stem Visualizer ─────────────────────────────────────────────────


class StemVisualizer:
    """Interactive N-row stem spectrogram viewer with solo/mute and synced playback.

    Args:
        filepath: Original audio file path (for display name).
        stem_names: Ordered list of stem names (e.g. ['drums', 'bass', ...]).
        stems_playback: Dict of name -> (ndarray, sr) for audio playback.
        stems_mono: Dict of name -> (ndarray, sr) for spectrogram analysis.
    """

    def __init__(self, filepath, stem_names, stems_playback, stems_mono):
        self.filepath = filepath
        self.stem_names = stem_names
        self.stem_keys = {str(i + 1): name for i, name in enumerate(stem_names)}
        self.playback_start_time = None
        self.playback_offset = 0.0
        self.is_playing = False
        self.active_stems = {name: True for name in stem_names}

        self.stems_playback = stems_playback
        self.stems_mono = stems_mono
        self.mel_specs = {}

        # Use first stem to get duration/sr
        first = stem_names[0]
        self.sr_playback = stems_playback[first][1]
        self.sr = stems_mono[first][1]
        self.duration = librosa.get_duration(y=stems_mono[first][0], sr=self.sr)

        # Spectrogram params
        self.n_fft = 2048
        self.hop_length = 512

        self._compute_spectrograms()
        self._compute_mix()
        self._build_figure()

    def _compute_spectrograms(self):
        print("Computing spectrograms...")
        all_db = []
        for name in self.stem_names:
            y, sr = self.stems_mono[name]
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=64, fmin=20, fmax=None
            )
            db = librosa.power_to_db(mel, ref=np.max)
            self.mel_specs[name] = db
            all_db.append(db)

        self.vmin = min(d.min() for d in all_db)
        self.vmax = max(d.max() for d in all_db)
        print(f"Duration: {self.duration:.1f}s, Sample rate: {self.sr} Hz")

    def _compute_mix(self):
        """Sum active stem playback arrays into current_mix."""
        active = [name for name, on in self.active_stems.items() if on]
        if not active:
            ref = self.stems_playback[self.stem_names[0]][0]
            self.current_mix = np.zeros_like(ref)
        else:
            self.current_mix = sum(
                self.stems_playback[name][0] for name in active
            )

    def _build_figure(self):
        print("Building visualization...")
        plt.style.use('dark_background')

        n = len(self.stem_names)
        fig_height = max(6, n * 2.5)
        self.fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(n, 1, hspace=0.35)

        self.axes = []
        self.dim_overlays = {}
        self.stem_labels = {}

        for i, name in enumerate(self.stem_names):
            ax = self.fig.add_subplot(gs[i])
            librosa.display.specshow(
                self.mel_specs[name], sr=self.sr, hop_length=self.hop_length,
                x_axis='time', y_axis='mel', fmin=20, fmax=None,
                ax=ax, cmap='magma', vmin=self.vmin, vmax=self.vmax
            )

            label = ax.set_ylabel(f'{name.capitalize()}', fontsize=12, fontweight='bold')
            self.stem_labels[name] = label

            if i < n - 1:
                ax.set_xlabel('')

            from matplotlib.patches import Rectangle
            overlay = Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                facecolor='black', alpha=0.7, zorder=10, visible=False
            )
            ax.add_patch(overlay)
            self.dim_overlays[name] = overlay

            self.axes.append(ax)

        self._finalize_figure()

    def _update_visual_state(self):
        """Update dim overlays and label colors to match active_stems."""
        for name in self.stem_names:
            active = self.active_stems[name]
            self.dim_overlays[name].set_visible(not active)
            color = 'white' if active else '#666666'
            self.stem_labels[name].set_color(color)
        self.fig.canvas.draw_idle()

    def _format_title(self):
        filename = Path(self.filepath).name
        n = len(self.stem_names)
        active_str = '  '.join(
            f'{"★" if self.active_stems[name] else " "} {i+1}:{name}'
            for i, name in enumerate(self.stem_names)
        )
        return (
            f'{filename}  —  {active_str}  —  '
            f'SPACE: play  1-{n}: toggle  A: all  Q: quit'
        )

    def _finalize_figure(self):
        self.cursor_lines = []
        for ax in self.axes:
            line = ax.axvline(x=0, color='#FFFFFF', linewidth=1.5, alpha=0.9, zorder=20)
            self.cursor_lines.append(line)

        self.fig.suptitle(self._format_title(), fontsize=12, fontweight='bold', y=0.995)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        n = len(self.stem_names)
        print(f"Ready. Press SPACE to start playback, 1-{n} to toggle stems.")

    def _update_title(self):
        self.fig.suptitle(self._format_title(), fontsize=12, fontweight='bold', y=0.995)

    def _on_key(self, event):
        if event.key == ' ':
            if not self.is_playing:
                self._start_playback()
            else:
                self._stop_playback()
        elif event.key in self.stem_keys:
            self._toggle_stem(self.stem_keys[event.key])
        elif event.key == 'a':
            self._all_stems_on()
        elif event.key == 'q':
            self._stop_playback()
            plt.close(self.fig)

    def _toggle_stem(self, name):
        was_playing = self.is_playing
        if was_playing:
            self._stop_playback()

        self.active_stems[name] = not self.active_stems[name]
        self._compute_mix()
        self._update_visual_state()
        self._update_title()

        active_names = [n for n, on in self.active_stems.items() if on]
        print(f"Active: {', '.join(active_names) if active_names else '(none)'}")

        if was_playing:
            self._start_playback()

    def _all_stems_on(self):
        was_playing = self.is_playing
        if was_playing:
            self._stop_playback()

        for name in self.stem_names:
            self.active_stems[name] = True
        self._compute_mix()
        self._update_visual_state()
        self._update_title()
        print("All stems active")

        if was_playing:
            self._start_playback()

    def _on_click(self, event):
        seek_time = None
        if event.inaxes in self.axes and event.xdata is not None:
            seek_time = event.xdata
        elif event.x is not None and event.y is not None:
            try:
                inv = self.axes[0].transData.inverted()
                data_coords = inv.transform((event.x, event.y))
                seek_time = data_coords[0]
            except Exception:
                pass

        if seek_time is not None and 0 <= seek_time <= self.duration:
            self._seek_to(seek_time)
            print(f"Seeking to {seek_time:.2f}s")

    def _start_playback(self, from_offset=None):
        if from_offset is not None:
            self.playback_offset = from_offset

        self.is_playing = True
        self.playback_start_time = time.time()

        start_sample = int(self.playback_offset * self.sr_playback)
        audio_to_play = self.current_mix[start_sample:]

        def play_audio():
            try:
                sd.play(audio_to_play, self.sr_playback)
                sd.wait()
                self.is_playing = False
            except Exception as e:
                print(f"Playback error: {e}")
                self.is_playing = False

        self.playback_thread = threading.Thread(target=play_audio, daemon=True)
        self.playback_thread.start()
        print(f"Playing from {self.playback_offset:.2f}s...")

    def _stop_playback(self):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            self.playback_offset = min(self.playback_offset + elapsed, self.duration)

        self.is_playing = False
        sd.stop()
        time.sleep(0.05)
        print(f"Stopped at {self.playback_offset:.2f}s.")

    def _seek_to(self, time_position):
        was_playing = self.is_playing
        if self.is_playing:
            self._stop_playback()

        self.playback_offset = time_position
        for line in self.cursor_lines:
            line.set_xdata([time_position, time_position])
        self.fig.canvas.draw_idle()

        if was_playing:
            time.sleep(0.05)
            self._start_playback()

    def _update_cursor(self, frame):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            current_time = self.playback_offset + elapsed

            if current_time > self.duration:
                self.is_playing = False
                current_time = self.duration

            for line in self.cursor_lines:
                line.set_xdata([current_time, current_time])
        elif not self.is_playing:
            for line in self.cursor_lines:
                line.set_xdata([self.playback_offset, self.playback_offset])

        return self.cursor_lines

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self._update_cursor,
            interval=33, blit=True, cache_frame_data=False
        )
        plt.show()
