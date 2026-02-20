#!/usr/bin/env python3
"""
Rap tempo analysis — testing autocorrelation approaches for tempo estimation.

The core question: Can we get accurate tempo from autocorrelation alone,
without reliable threshold-crossing beat detections?

Current BeatPredictor only runs autocorrelation on confirmed beat detections.
For rap (dense percussion, noisy abs-integral), we test running autocorrelation
continuously on the signal buffer.

Tests:
  1. Abs-integral signal (current approach, but continuous autocorrelation)
  2. Low-pass filtered RMS (kick drum isolation, <150Hz)
  3. Full-band RMS signal
  4. How quickly each converges to correct tempo

Ground truth BPM from Harmonix annotations:
  - Black and Yellow: ~82 BPM (732ms)
  - In Da Club: ~91 BPM (variable, see notes)
  - Love The Way You Lie: ~87 BPM (690ms)
  - Can't Hold Us: ~146 BPM (411ms)
"""

import numpy as np
import librosa
import yaml
from pathlib import Path
from scipy import signal as scipy_signal

SEGMENTS = Path(__file__).resolve().parents[2] / 'audio-segments'
HARMONIX = SEGMENTS / 'harmonix'

SR = 44100
HOP = 512
FRAME_LEN = 2048


def load_track(name):
    """Load wav + annotations, return (audio, beat_times, true_bpm)."""
    wav = HARMONIX / f'{name}.wav'
    ann = HARMONIX / f'{name}.annotations.yaml'
    if not wav.exists():
        return None, None, None
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    with open(ann) as f:
        # Some annotation files use numpy binary (unsafe yaml needed)
        try:
            data = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            f.seek(0)
            data = yaml.unsafe_load(f)
    beats = np.array(data['harmonix_beats'], dtype=float)
    if len(beats) > 1:
        true_bpm = 60.0 / np.median(np.diff(beats))
    else:
        true_bpm = 0
    return audio, beats, true_bpm


def compute_abs_integral_stream(audio, window_sec=0.15):
    """Simulate real-time abs-integral computation, return (times, signal)."""
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LEN, hop_length=HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    dt = HOP / SR

    # RMS derivative
    rms_deriv = np.diff(rms, prepend=rms[0]) / dt

    # Absolute integral over window
    window_frames = max(1, int(window_sec / dt))
    abs_deriv = np.abs(rms_deriv)
    cumsum = np.cumsum(np.insert(abs_deriv * dt, 0, 0))
    abs_integral = np.zeros(len(abs_deriv))
    for i in range(len(abs_deriv)):
        start = max(0, i - window_frames)
        abs_integral[i] = cumsum[i + 1] - cumsum[start]

    return times, abs_integral


def compute_lowpass_rms(audio, cutoff_hz=150):
    """Low-pass filter audio then compute RMS — isolates kick drum."""
    # Design low-pass filter
    nyq = SR / 2
    sos = scipy_signal.butter(4, cutoff_hz / nyq, btype='low', output='sos')
    filtered = scipy_signal.sosfilt(sos, audio)
    rms = librosa.feature.rms(y=filtered, frame_length=FRAME_LEN, hop_length=HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    return times, rms


def compute_fullband_rms(audio):
    """Full-band RMS."""
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LEN, hop_length=HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    return times, rms


def autocorrelation_tempo(signal_buf, rms_fps, min_bpm=40, max_bpm=300):
    """
    Run autocorrelation on a signal buffer, return (estimated_period_sec, confidence).

    This is the same algorithm as BeatPredictor._update_autocorrelation(),
    but callable on any signal buffer.
    """
    sig = signal_buf - np.mean(signal_buf)
    norm = np.dot(sig, sig)
    if norm < 1e-20:
        return 0, 0

    min_lag = max(1, int(rms_fps * 60.0 / max_bpm))
    max_lag = min(int(rms_fps * 60.0 / min_bpm), len(sig) // 2)
    if min_lag >= max_lag:
        return 0, 0

    autocorr = np.zeros(max_lag - min_lag, dtype=np.float64)
    for i, lag in enumerate(range(min_lag, max_lag)):
        autocorr[i] = np.dot(sig[:-lag], sig[lag:]) / norm

    # Find first strong peak
    best_lag = -1
    best_corr = 0.0
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > best_corr:
                best_corr = autocorr[i]
                best_lag = min_lag + i
                if best_corr > 0.3:
                    break

    if best_lag > 0:
        return best_lag / rms_fps, best_corr
    return 0, 0


def autocorrelation_all_peaks(signal_buf, rms_fps, min_bpm=40, max_bpm=300, n_peaks=5):
    """Return top N autocorrelation peaks as (bpm, confidence) pairs."""
    sig = signal_buf - np.mean(signal_buf)
    norm = np.dot(sig, sig)
    if norm < 1e-20:
        return []

    min_lag = max(1, int(rms_fps * 60.0 / max_bpm))
    max_lag = min(int(rms_fps * 60.0 / min_bpm), len(sig) // 2)
    if min_lag >= max_lag:
        return []

    autocorr = np.zeros(max_lag - min_lag, dtype=np.float64)
    for i, lag in enumerate(range(min_lag, max_lag)):
        autocorr[i] = np.dot(sig[:-lag], sig[lag:]) / norm

    # Find all peaks
    peaks = []
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            lag = min_lag + i
            bpm = 60.0 * rms_fps / lag
            peaks.append((bpm, autocorr[i]))

    # Sort by confidence descending
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:n_peaks]


def compute_onset_envelope(audio):
    """Librosa onset strength envelope — good for percussive content."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=SR, hop_length=HOP)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=SR, hop_length=HOP)
    return times, onset_env


def simulate_continuous_tempo(times, signal, ac_window_sec=5.0, update_interval_sec=0.5):
    """
    Simulate real-time continuous autocorrelation.

    Instead of only running autocorrelation on beat detections,
    run it every update_interval_sec on the trailing ac_window_sec buffer.

    Returns: list of (time, estimated_bpm, confidence)
    """
    rms_fps = SR / HOP
    dt = HOP / SR
    ac_window_frames = int(ac_window_sec * rms_fps)
    update_interval_frames = int(update_interval_sec * rms_fps)

    results = []
    estimated_period = 0.0

    for i in range(ac_window_frames, len(signal), update_interval_frames):
        buf = signal[max(0, i - ac_window_frames):i]
        period, conf = autocorrelation_tempo(buf, rms_fps)

        if conf > 0.3 and period > 0:
            # Octave correction + smoothing
            if estimated_period > 0:
                ratio = period / estimated_period
                if 0.8 < ratio < 1.2:
                    estimated_period = 0.8 * estimated_period + 0.2 * period
                elif 0.45 < ratio < 0.55:
                    estimated_period = 0.8 * estimated_period + 0.2 * (period * 2)
                elif 1.8 < ratio < 2.2:
                    estimated_period = 0.8 * estimated_period + 0.2 * (period / 2)
                # else: ignore outlier
            else:
                estimated_period = period

        bpm = 60.0 / estimated_period if estimated_period > 0 else 0
        results.append((times[min(i, len(times)-1)], bpm, conf))

    return results


def analyze_track(name, title):
    """Full analysis of one track."""
    audio, beats, true_bpm = load_track(name)
    if audio is None:
        print(f"  SKIP: {title} (no wav)")
        return

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  True BPM: {true_bpm:.1f}  ({len(beats)} beats in {len(audio)/SR:.1f}s)")
    if len(beats) > 1:
        ibi = np.diff(beats)
        print(f"  IBI: median={np.median(ibi)*1000:.0f}ms  "
              f"std={np.std(ibi)*1000:.1f}ms  "
              f"range=[{np.min(ibi)*1000:.0f}, {np.max(ibi)*1000:.0f}]ms")
    print(f"{'='*70}")

    # Compute signals
    t_absint, sig_absint = compute_abs_integral_stream(audio)
    t_lp, sig_lp = compute_lowpass_rms(audio, cutoff_hz=150)
    t_full, sig_full = compute_fullband_rms(audio)
    t_onset, sig_onset = compute_onset_envelope(audio)

    signals = [
        ("Abs-integral", t_absint, sig_absint),
        ("Low-pass RMS (<150Hz)", t_lp, sig_lp),
        ("Full-band RMS", t_full, sig_full),
        ("Onset envelope", t_onset, sig_onset),
    ]

    # Show autocorrelation peaks for each signal (using full 30s buffer from middle)
    rms_fps = SR / HOP
    print(f"\n  --- Autocorrelation peak analysis (30s buffer, middle of track) ---")
    for sig_name, t, sig in signals:
        mid = len(sig) // 2
        buf_frames = int(30.0 * rms_fps)
        start = max(0, mid - buf_frames // 2)
        end = min(len(sig), start + buf_frames)
        buf = sig[start:end]
        peaks = autocorrelation_all_peaks(buf, rms_fps, n_peaks=5)
        if peaks:
            peak_strs = [f"{bpm:.0f}BPM({conf:.3f})" for bpm, conf in peaks]
            print(f"    {sig_name:25s}: {', '.join(peak_strs)}")
        else:
            print(f"    {sig_name:25s}: no peaks")

    for sig_name, t, sig in signals:
        results = simulate_continuous_tempo(t, sig, ac_window_sec=5.0, update_interval_sec=0.5)

        if not results:
            print(f"\n  {sig_name}: no results")
            continue

        bpms = [r[1] for r in results if r[1] > 0]
        confs = [r[2] for r in results]
        times_r = [r[0] for r in results]

        # Convergence: first time within 5% of true BPM
        converge_time = None
        for t_val, bpm, conf in results:
            if bpm > 0 and abs(bpm - true_bpm) / true_bpm < 0.05:
                converge_time = t_val
                break

        # Steady-state accuracy (after 10s)
        late_bpms = [r[1] for r in results if r[0] > 10 and r[1] > 0]
        if late_bpms:
            median_bpm = np.median(late_bpms)
            err_pct = abs(median_bpm - true_bpm) / true_bpm * 100
            # Check for octave error
            octave_err = None
            if abs(median_bpm - true_bpm * 2) / (true_bpm * 2) < 0.05:
                octave_err = "2x (double)"
            elif abs(median_bpm - true_bpm / 2) / (true_bpm / 2) < 0.05:
                octave_err = "0.5x (half)"
        else:
            median_bpm = 0
            err_pct = 100
            octave_err = None

        print(f"\n  {sig_name}:")
        print(f"    Converge to 5%:  {converge_time:.1f}s" if converge_time else
              f"    Converge to 5%:  NEVER")
        print(f"    Steady BPM:      {median_bpm:.1f} (true: {true_bpm:.1f}, err: {err_pct:.1f}%)")
        if octave_err:
            print(f"    OCTAVE ERROR:    {octave_err}")
        print(f"    Confidence:      median={np.median(confs):.3f}")

        # Show BPM evolution at key timepoints
        checkpoints = [5, 10, 15, 30, 45]
        bpm_at = []
        for cp in checkpoints:
            closest = min(results, key=lambda r: abs(r[0] - cp))
            bpm_at.append(f"{cp}s:{closest[1]:.0f}")
        print(f"    BPM over time:   {', '.join(bpm_at)}")

    # Also test: different low-pass cutoffs
    print(f"\n  --- Low-pass cutoff sweep ---")
    for cutoff in [80, 100, 120, 150, 200, 250]:
        t_lp, sig_lp = compute_lowpass_rms(audio, cutoff_hz=cutoff)
        results = simulate_continuous_tempo(t_lp, sig_lp)
        late_bpms = [r[1] for r in results if r[0] > 10 and r[1] > 0]
        if late_bpms:
            med = np.median(late_bpms)
            err = abs(med - true_bpm) / true_bpm * 100
            conv = None
            for t_val, bpm, conf in results:
                if bpm > 0 and abs(bpm - true_bpm) / true_bpm < 0.05:
                    conv = t_val
                    break
            conv_str = f"{conv:.1f}s" if conv else "never"
            print(f"    {cutoff:>4d}Hz:  BPM={med:5.1f}  err={err:4.1f}%  converge={conv_str}")
        else:
            print(f"    {cutoff:>4d}Hz:  no estimates")

    # Test: different AC window sizes
    print(f"\n  --- AC window size sweep ---")
    for ac_win in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        results = simulate_continuous_tempo(t_absint, sig_absint, ac_window_sec=ac_win)
        late_bpms = [r[1] for r in results if r[0] > max(ac_win + 1, 10) and r[1] > 0]
        if late_bpms:
            med = np.median(late_bpms)
            err = abs(med - true_bpm) / true_bpm * 100
            conv = None
            for t_val, bpm, conf in results:
                if bpm > 0 and abs(bpm - true_bpm) / true_bpm < 0.05:
                    conv = t_val
                    break
            conv_str = f"{conv:.1f}s" if conv else "never"
            print(f"    {ac_win:4.1f}s:   BPM={med:5.1f}  err={err:4.1f}%  converge={conv_str}")
        else:
            print(f"    {ac_win:4.1f}s:   no estimates")


def autocorrelation_octave_corrected(signal_buf, rms_fps, min_bpm=40, max_bpm=300,
                                      prefer_range=(70, 180)):
    """
    Autocorrelation with octave-aware peak selection.

    Strategy: find all peaks, then prefer peaks where:
    1. The peak's double (2x BPM) also exists as a peak — pick the higher one
    2. Peaks in the "preferred" BPM range get a boost
    3. Among candidates, pick the one with highest weighted score

    This handles rap's sub-harmonic problem: kick on beats 1&3 creates
    a half-tempo peak that's often stronger than the true beat peak.
    """
    sig = signal_buf - np.mean(signal_buf)
    norm = np.dot(sig, sig)
    if norm < 1e-20:
        return 0, 0

    min_lag = max(1, int(rms_fps * 60.0 / max_bpm))
    max_lag = min(int(rms_fps * 60.0 / min_bpm), len(sig) // 2)
    if min_lag >= max_lag:
        return 0, 0

    autocorr = np.zeros(max_lag - min_lag, dtype=np.float64)
    for i, lag in enumerate(range(min_lag, max_lag)):
        autocorr[i] = np.dot(sig[:-lag], sig[lag:]) / norm

    # Find all peaks with positive correlation
    peaks = []
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.05:  # minimum threshold
                lag = min_lag + i
                bpm = 60.0 * rms_fps / lag
                peaks.append((bpm, autocorr[i], lag))

    if not peaks:
        return 0, 0

    # Score each peak: base confidence + octave bonus + range bonus
    scored = []
    for bpm, conf, lag in peaks:
        score = conf

        # Bonus if double-tempo also has a peak (confirms this is a sub-harmonic)
        for bpm2, conf2, _ in peaks:
            ratio = bpm2 / bpm
            if 1.9 < ratio < 2.1 and conf2 > 0.1:
                # This peak has a harmonic at 2x — the 2x is more likely the beat
                score *= 0.5  # penalize sub-harmonic
                break

        # Bonus if half-tempo also has a peak (confirms this peak IS the beat)
        for bpm2, conf2, _ in peaks:
            ratio = bpm / bpm2
            if 1.9 < ratio < 2.1 and conf2 > 0.1:
                score *= 1.3  # boost: has sub-harmonic support
                break

        # Range prior: prefer peaks in typical rap tempo range
        if prefer_range[0] <= bpm <= prefer_range[1]:
            score *= 1.2

        scored.append((bpm, score, conf, lag))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_bpm, best_score, best_conf, best_lag = scored[0]
    return best_lag / rms_fps, best_conf


def simulate_octave_corrected(times, signal, ac_window_sec=5.0, update_interval_sec=0.5):
    """Like simulate_continuous_tempo but using octave-corrected peak selection."""
    rms_fps = SR / HOP
    ac_window_frames = int(ac_window_sec * rms_fps)
    update_interval_frames = int(update_interval_sec * rms_fps)

    results = []
    estimated_period = 0.0

    for i in range(ac_window_frames, len(signal), update_interval_frames):
        buf = signal[max(0, i - ac_window_frames):i]
        period, conf = autocorrelation_octave_corrected(buf, rms_fps)

        if conf > 0.1 and period > 0:
            if estimated_period > 0:
                ratio = period / estimated_period
                if 0.8 < ratio < 1.2:
                    estimated_period = 0.8 * estimated_period + 0.2 * period
                elif 0.45 < ratio < 0.55:
                    estimated_period = 0.8 * estimated_period + 0.2 * (period * 2)
                elif 1.8 < ratio < 2.2:
                    estimated_period = 0.8 * estimated_period + 0.2 * (period / 2)
            else:
                estimated_period = period

        bpm = 60.0 / estimated_period if estimated_period > 0 else 0
        results.append((times[min(i, len(times)-1)], bpm, conf))

    return results


def test_octave_corrected(name, title):
    """Test octave-corrected onset envelope on one track."""
    audio, beats, true_bpm = load_track(name)
    if audio is None:
        print(f"  SKIP: {title}")
        return

    t_onset, sig_onset = compute_onset_envelope(audio)
    results = simulate_octave_corrected(t_onset, sig_onset, ac_window_sec=5.0)

    if not results:
        print(f"  {title}: no results")
        return

    # Convergence
    converge_time = None
    for t_val, bpm, conf in results:
        if bpm > 0 and abs(bpm - true_bpm) / true_bpm < 0.05:
            converge_time = t_val
            break

    late_bpms = [r[1] for r in results if r[0] > 10 and r[1] > 0]
    if late_bpms:
        median_bpm = np.median(late_bpms)
        err_pct = abs(median_bpm - true_bpm) / true_bpm * 100
    else:
        median_bpm = 0
        err_pct = 100

    checkpoints = [5, 10, 15, 30, 45]
    bpm_at = []
    for cp in checkpoints:
        closest = min(results, key=lambda r: abs(r[0] - cp))
        bpm_at.append(f"{cp}s:{closest[1]:.0f}")

    conv_str = f"{converge_time:.1f}s" if converge_time else "NEVER"
    status = "OK" if err_pct < 5 else "OCTAVE" if err_pct > 30 else "DRIFT"
    print(f"  [{status:6s}] {title:35s} true={true_bpm:.0f}  est={median_bpm:.0f}  "
          f"err={err_pct:.1f}%  conv={conv_str}")
    print(f"           BPM over time: {', '.join(bpm_at)}")


def autocorrelation_with_prior(signal_buf, rms_fps, min_bpm=40, max_bpm=300,
                                prior_center=100, prior_sigma=40):
    """
    Autocorrelation with Rayleigh-like tempo prior.

    Instead of taking the first strong peak, weight all peaks by a
    Gaussian prior centered at prior_center BPM. This is what librosa's
    beat tracker does internally (default prior at 120 BPM).

    For rap: prior_center=100, prior_sigma=40 covers 60-140 BPM well.
    """
    sig = signal_buf - np.mean(signal_buf)
    norm = np.dot(sig, sig)
    if norm < 1e-20:
        return 0, 0

    min_lag = max(1, int(rms_fps * 60.0 / max_bpm))
    max_lag = min(int(rms_fps * 60.0 / min_bpm), len(sig) // 2)
    if min_lag >= max_lag:
        return 0, 0

    autocorr = np.zeros(max_lag - min_lag, dtype=np.float64)
    for i, lag in enumerate(range(min_lag, max_lag)):
        autocorr[i] = np.dot(sig[:-lag], sig[lag:]) / norm

    # Find all peaks
    peaks = []
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.05:
                lag = min_lag + i
                bpm = 60.0 * rms_fps / lag
                # Gaussian prior weight
                prior_weight = np.exp(-0.5 * ((bpm - prior_center) / prior_sigma) ** 2)
                score = autocorr[i] * prior_weight
                peaks.append((bpm, score, autocorr[i], lag))

    if not peaks:
        return 0, 0

    peaks.sort(key=lambda x: x[1], reverse=True)
    best_bpm, best_score, best_conf, best_lag = peaks[0]
    return best_lag / rms_fps, best_conf


def simulate_with_prior(times, signal, ac_window_sec=5.0, update_interval_sec=0.5,
                         prior_center=100, prior_sigma=40):
    """Simulate continuous tempo with prior-weighted autocorrelation."""
    rms_fps = SR / HOP
    ac_window_frames = int(ac_window_sec * rms_fps)
    update_interval_frames = int(update_interval_sec * rms_fps)

    results = []
    estimated_period = 0.0

    for i in range(ac_window_frames, len(signal), update_interval_frames):
        buf = signal[max(0, i - ac_window_frames):i]
        period, conf = autocorrelation_with_prior(
            buf, rms_fps, prior_center=prior_center, prior_sigma=prior_sigma
        )

        if conf > 0.1 and period > 0:
            if estimated_period > 0:
                ratio = period / estimated_period
                if 0.8 < ratio < 1.2:
                    estimated_period = 0.8 * estimated_period + 0.2 * period
                elif 0.45 < ratio < 0.55:
                    estimated_period = 0.8 * estimated_period + 0.2 * (period * 2)
                elif 1.8 < ratio < 2.2:
                    estimated_period = 0.8 * estimated_period + 0.2 * (period / 2)
            else:
                estimated_period = period

        bpm = 60.0 / estimated_period if estimated_period > 0 else 0
        results.append((times[min(i, len(times)-1)], bpm, conf))

    return results


def test_prior_approach(name, title, prior_center=100, prior_sigma=40):
    """Test prior-weighted onset autocorrelation on one track."""
    audio, beats, true_bpm = load_track(name)
    if audio is None:
        print(f"  SKIP: {title}")
        return

    t_onset, sig_onset = compute_onset_envelope(audio)
    results = simulate_with_prior(t_onset, sig_onset,
                                   prior_center=prior_center,
                                   prior_sigma=prior_sigma)

    if not results:
        print(f"  {title}: no results")
        return

    converge_time = None
    for t_val, bpm, conf in results:
        if bpm > 0 and abs(bpm - true_bpm) / true_bpm < 0.05:
            converge_time = t_val
            break

    late_bpms = [r[1] for r in results if r[0] > 10 and r[1] > 0]
    if late_bpms:
        median_bpm = np.median(late_bpms)
        err_pct = abs(median_bpm - true_bpm) / true_bpm * 100
    else:
        median_bpm = 0
        err_pct = 100

    checkpoints = [5, 10, 15, 30]
    bpm_at = []
    for cp in checkpoints:
        closest = min(results, key=lambda r: abs(r[0] - cp))
        bpm_at.append(f"{cp}s:{closest[1]:.0f}")

    conv_str = f"{converge_time:.1f}s" if converge_time else "NEVER"
    status = "OK" if err_pct < 5 else "CLOSE" if err_pct < 15 else "FAIL"
    print(f"  [{status:5s}] {title:35s} true={true_bpm:.0f}  est={median_bpm:.0f}  "
          f"err={err_pct:.1f}%  conv={conv_str}  [{', '.join(bpm_at)}]")


def main():
    print("\n  Rap Tempo Analysis — Continuous Autocorrelation")
    print("  " + "=" * 50)

    tracks = [
        ('0026_blackandyellow_60s', 'Black and Yellow (Wiz Khalifa)'),
        ('0141_indaclub_60s', 'In Da Club (50 Cent)'),
        ('0439_lovethewayyoulie_60s', 'Love The Way You Lie (Eminem)'),
        ('0607_cantholdus_60s', "Can't Hold Us (Macklemore)"),
    ]

    for name, title in tracks:
        analyze_track(name, title)

    # ===== Test octave-corrected onset envelope approach =====
    print("\n" + "=" * 70)
    print("  OCTAVE-CORRECTED ONSET ENVELOPE TEST")
    print("=" * 70)
    for name, title in tracks:
        test_octave_corrected(name, title)

    # ===== Test prior-weighted onset envelope =====
    print("\n" + "=" * 70)
    print("  PRIOR-WEIGHTED ONSET ENVELOPE (center=100, sigma=40)")
    print("=" * 70)
    for name, title in tracks:
        test_prior_approach(name, title, prior_center=100, prior_sigma=40)

    print(f"\n  --- Sweep prior centers ---")
    for center in [80, 90, 100, 110, 120]:
        print(f"\n  Prior center={center} BPM, sigma=40:")
        for name, title in tracks:
            test_prior_approach(name, title, prior_center=center, prior_sigma=40)

    # Also test what librosa.beat.tempo gives (offline ground truth)
    print("\n" + "=" * 70)
    print("  LIBROSA BEAT.TEMPO (offline reference)")
    print("=" * 70)
    for name, title in tracks:
        audio, beats, true_bpm = load_track(name)
        if audio is None:
            continue
        tempo = librosa.beat.tempo(y=audio, sr=SR)[0]
        err = abs(tempo - true_bpm) / true_bpm * 100
        print(f"  {title:35s} true={true_bpm:.0f}  librosa={tempo:.1f}  err={err:.1f}%")

    # Test abs-integral + prior (reuse existing signal, no new computation)
    print("\n" + "=" * 70)
    print("  ABS-INTEGRAL + PRIOR (center=100, sigma=40) — reuses existing signal")
    print("=" * 70)
    for name, title in tracks:
        audio, beats, true_bpm = load_track(name)
        if audio is None:
            continue
        t_absint, sig_absint = compute_abs_integral_stream(audio)
        results = simulate_with_prior(t_absint, sig_absint,
                                       prior_center=100, prior_sigma=40)
        late_bpms = [r[1] for r in results if r[0] > 10 and r[1] > 0]
        median_bpm = np.median(late_bpms) if late_bpms else 0
        err_pct = abs(median_bpm - true_bpm) / true_bpm * 100 if true_bpm else 100
        conv = None
        for t_val, bpm, conf in results:
            if bpm > 0 and abs(bpm - true_bpm) / true_bpm < 0.05:
                conv = t_val
                break
        conv_str = f"{conv:.1f}s" if conv else "NEVER"
        status = "OK" if err_pct < 5 else "CLOSE" if err_pct < 15 else "FAIL"
        checkpoints = [5, 10, 15, 30]
        bpm_at = [f"{cp}s:{min(results, key=lambda r: abs(r[0]-cp))[1]:.0f}" for cp in checkpoints]
        print(f"  [{status:5s}] {title:35s} true={true_bpm:.0f}  est={median_bpm:.0f}  "
              f"err={err_pct:.1f}%  conv={conv_str}  [{', '.join(bpm_at)}]")

    # Test half-wave rectified RMS derivative (cheap onset approximation)
    print("\n" + "=" * 70)
    print("  HALF-WAVE RMS DERIVATIVE + PRIOR — cheap real-time onset envelope")
    print("=" * 70)
    for name, title in tracks:
        audio, beats, true_bpm = load_track(name)
        if audio is None:
            continue
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LEN, hop_length=HOP)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
        dt = HOP / SR
        rms_deriv = np.diff(rms, prepend=rms[0]) / dt
        # Half-wave rectify: only positive (energy arriving = onset)
        hw_deriv = np.maximum(0, rms_deriv)

        results = simulate_with_prior(times, hw_deriv,
                                       prior_center=100, prior_sigma=40)
        late_bpms = [r[1] for r in results if r[0] > 10 and r[1] > 0]
        median_bpm = np.median(late_bpms) if late_bpms else 0
        err_pct = abs(median_bpm - true_bpm) / true_bpm * 100 if true_bpm else 100
        conv = None
        for t_val, bpm, conf in results:
            if bpm > 0 and abs(bpm - true_bpm) / true_bpm < 0.05:
                conv = t_val
                break
        conv_str = f"{conv:.1f}s" if conv else "NEVER"
        status = "OK" if err_pct < 5 else "CLOSE" if err_pct < 15 else "FAIL"
        checkpoints = [5, 10, 15, 30]
        bpm_at = [f"{cp}s:{min(results, key=lambda r: abs(r[0]-cp))[1]:.0f}" for cp in checkpoints]
        print(f"  [{status:5s}] {title:35s} true={true_bpm:.0f}  est={median_bpm:.0f}  "
              f"err={err_pct:.1f}%  conv={conv_str}  [{', '.join(bpm_at)}]")

    print("\n" + "=" * 70)
    print("  Summary & Recommendations")
    print("=" * 70)


if __name__ == '__main__':
    main()
