"""
REPET: REPeating Pattern Extraction Technique

Separates audio into "repeating" (background) and "non-repeating" (foreground)
layers by detecting cyclic patterns in the spectrogram.

Based on Rafii & Pardo 2012, with extensions for aggressive separation:
  - Multi-period detection (catches repeats at beat, bar, and phrase scales)
  - Mask sharpening via exponentiation (pushes soft mask toward 0/1)
  - Conservative template (lower percentile instead of median)

Core idea:
  1. STFT → magnitude spectrogram
  2. Beat spectrum (autocorrelation across time) → find repeating periods
  3. For each period: segment spectrogram, take percentile → template
  4. Combine templates (element-wise max across periods)
  5. Soft mask = min(template, original) / original, raised to power
  6. Apply mask to complex STFT → repeating audio
  7. (1 - mask) → non-repeating audio

Usage:
    from repet import repet
    y_rep, y_nonrep, info = repet(y_mono, sr)
"""

import numpy as np
import librosa


def beat_spectrum(magnitude):
    """Compute beat spectrum via FFT-based autocorrelation.

    For each frequency bin, autocorrelate its energy over time,
    then sum across all bins. Peaks indicate repeating periods.

    Args:
        magnitude: STFT magnitude array (n_freq, n_frames)

    Returns:
        beat_spec: 1D array of length n_frames (normalized, [0] = 1.0)
    """
    power = magnitude ** 2
    n_frames = power.shape[1]
    # Zero-pad for linear (not circular) autocorrelation
    n_fft = 2 * n_frames
    X = np.fft.rfft(power, n=n_fft, axis=1)
    auto = np.fft.irfft(X * np.conj(X), axis=1)[:, :n_frames]
    bs = np.sum(auto, axis=0)
    if bs[0] > 0:
        bs = bs / bs[0]
    return bs


def find_periods(beat_spec, sr, hop_length, min_period_sec=0.15, max_periods=4):
    """Find multiple dominant repeating periods from the beat spectrum.

    Returns periods at different time scales (beat, bar, phrase) so the
    combined template catches repeats at all levels.

    Args:
        beat_spec: beat spectrum array
        sr: sample rate
        hop_length: STFT hop length
        min_period_sec: minimum period to consider
        max_periods: maximum number of periods to return

    Returns:
        periods: list of (period_frames, period_seconds) tuples, sorted short→long
    """
    from scipy.signal import find_peaks

    n_frames = len(beat_spec)
    min_lag = max(2, int(min_period_sec * sr / hop_length))
    max_lag = n_frames // 2

    search_region = beat_spec[min_lag:max_lag]
    if len(search_region) < 3:
        period_frames = n_frames // 2
        return [(period_frames, period_frames * hop_length / sr)]

    # Find all significant peaks
    peaks, props = find_peaks(search_region, height=0.03, distance=min_lag // 2 + 1)
    peaks = peaks + min_lag

    if len(peaks) == 0:
        period_frames = int(min_lag + np.argmax(search_region))
        return [(period_frames, period_frames * hop_length / sr)]

    # Score peaks by beat spectrum height
    heights = beat_spec[peaks]

    # Select top peaks, but skip near-harmonics of already-selected peaks
    # (e.g., if we pick period=43, don't also pick 86 unless it scores higher)
    selected = []
    sorted_indices = np.argsort(-heights)  # highest first

    for idx in sorted_indices:
        p = int(peaks[idx])
        # Check if this is a near-harmonic of an already selected period
        is_harmonic = False
        for sp, _ in selected:
            ratio = p / sp
            # Check if ratio is close to an integer (harmonic) or simple fraction
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) < 0.15:
                is_harmonic = True
                break
        if not is_harmonic:
            selected.append((p, p * hop_length / sr))
            if len(selected) >= max_periods:
                break

    # Sort short to long
    selected.sort(key=lambda x: x[0])
    return selected


def build_template_for_period(magnitude, period_frames, percentile=25):
    """Build a repeating template for a single period.

    Uses a lower percentile instead of median for a more conservative
    estimate — only attributes energy to "repeating" if it's consistently
    present across most repetitions.

    Args:
        magnitude: STFT magnitude (n_freq, n_frames)
        period_frames: period length in STFT frames
        percentile: percentile for template (lower = more conservative).
                    50 = median (original REPET), 25 = conservative.

    Returns:
        model: repeating model tiled to match magnitude shape, or None if <2 segments
    """
    n_freq, n_frames = magnitude.shape
    n_segments = n_frames // period_frames

    if n_segments < 2:
        return None

    usable_frames = n_segments * period_frames
    mag_segments = magnitude[:, :usable_frames].reshape(
        n_freq, n_segments, period_frames
    )

    # Lower percentile = more conservative = only what's truly consistent
    template = np.percentile(mag_segments, percentile, axis=1)

    # Tile to cover full spectrogram
    n_tiles = (n_frames + period_frames - 1) // period_frames
    model = np.tile(template, n_tiles)[:, :n_frames]
    return model


def repet(y, sr, period_sec=None, n_fft=2048, hop_length=512,
          sharpness=3.0, percentile=25, max_periods=4):
    """Run REPET separation on mono audio.

    Args:
        y: mono audio signal (1D numpy array)
        sr: sample rate
        period_sec: repeating period in seconds (auto-detect if None).
                    If given, only that single period is used.
        n_fft: FFT window size
        hop_length: hop length for STFT
        sharpness: mask exponent (1.0 = original soft mask, higher = sharper).
                   3.0 pushes mask values toward 0 or 1 for cleaner separation.
        percentile: percentile for template building (50 = median, lower = more
                    conservative, attributes less to repeating layer)
        max_periods: max number of repeating periods to detect and combine

    Returns:
        y_repeating: the repeating (background) signal
        y_nonrepeating: the non-repeating (foreground) signal
        info: dict with diagnostics
    """
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    n_freq, n_frames = magnitude.shape

    # Beat spectrum
    bs = beat_spectrum(magnitude)

    # Find repeating periods
    if period_sec is not None:
        periods = [(int(period_sec * sr / hop_length), period_sec)]
    else:
        periods = find_periods(bs, sr, hop_length, max_periods=max_periods)

    for pf, ps in periods:
        n_seg = n_frames // pf
        print(f"  Period: {ps:.3f}s ({pf} frames, {n_seg} segments)")

    # Build templates for each period and combine
    models = []
    for period_frames, period_seconds in periods:
        model = build_template_for_period(magnitude, period_frames, percentile)
        if model is not None:
            models.append(model)

    if not models:
        print("  WARNING: no usable periods found — separation will be poor")
        info = {
            'periods': periods,
            'n_segments': 0,
            'beat_spec': bs,
            'magnitude': magnitude,
            'repeating_model': magnitude,
            'mask': np.ones_like(magnitude),
            'hop_length': hop_length,
            'sr': sr,
            'sharpness': sharpness,
            'percentile': percentile,
        }
        return y.copy(), np.zeros_like(y), info

    # Combine models: element-wise maximum across all periods
    # Each model captures repeats at its time scale; the max captures all of them
    combined_model = models[0]
    for m in models[1:]:
        combined_model = np.maximum(combined_model, m)

    # Soft mask with sharpening
    raw_mask = np.minimum(combined_model, magnitude) / (magnitude + 1e-10)
    raw_mask = np.clip(raw_mask, 0, 1)

    # Sharpen: raise to power pushes values toward 0 or 1
    mask = raw_mask ** sharpness

    # Apply mask to complex STFT (preserving original phase)
    D_repeating = mask * D
    D_nonrepeating = (1 - mask) * D

    # Inverse STFT
    y_repeating = librosa.istft(D_repeating, hop_length=hop_length, length=len(y))
    y_nonrepeating = librosa.istft(D_nonrepeating, hop_length=hop_length, length=len(y))

    # Use primary period for backward-compat info fields
    primary_pf, primary_ps = periods[0]
    info = {
        'period_seconds': primary_ps,
        'period_frames': primary_pf,
        'periods': periods,
        'n_segments': n_frames // primary_pf,
        'beat_spec': bs,
        'magnitude': magnitude,
        'repeating_model': combined_model,
        'mask': mask,
        'raw_mask': raw_mask,
        'hop_length': hop_length,
        'sr': sr,
        'sharpness': sharpness,
        'percentile': percentile,
    }

    return y_repeating, y_nonrepeating, info
