#!/usr/bin/env python3
"""
Normalization Test: Three-Way Comparison

Computes 5-band energy three ways and quantifies the gap:
  (a) Offline global-max (viewer.py style: energy / np.max(energy))
  (b) Offline causal RT (viewer.py band_energy_rt: linear decay, seed from 30s, dropout)
  (c) Real-time peak-decay (effects style: peak = max(energy, peak * decay), cold start)

Uses the same mel spectrogram computation as the viewer.
"""

import os
import sys
import numpy as np
import librosa

# ── Paths ──
SEGMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio-segments')

# ── Band definitions (matching viewer.py exactly) ──
FREQUENCY_BANDS = {
    'Sub-bass': (20, 80),
    'Bass': (80, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 6000),
    'Treble': (6000, 8000),
}

BAND_ORDER = ['Sub-bass', 'Bass', 'Mids', 'High-mids', 'Treble']


def compute_three_ways(wav_path):
    """Compute 5-band energy normalized three ways.

    Returns:
        raw_energies: dict {band_name: np.array} of raw mel band energy
        offline_global: dict {band_name: np.array} normalized by per-band global max
        offline_causal: dict {band_name: np.array} causal RT from viewer.py
        rt_peak_decay: dict {band_name: np.array} peak-decay with 0.9995
        sr, hop_length, n_frames: metadata
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    n_fft = 2048
    hop_length = 512
    fps = sr / hop_length

    print(f"  Duration: {duration:.1f}s, SR: {sr}, FPS: {fps:.1f}")

    # ── Mel spectrogram (matching viewer.py exactly) ──
    mel_spec_128 = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=128, fmin=20, fmax=8000
    )
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)

    # ── Raw band energies ──
    raw_energies = {}
    for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
        band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)
        raw_energies[band_name] = np.sum(mel_spec_128[band_mask, :], axis=0)

    n_frames = len(next(iter(raw_energies.values())))

    # ═══════════════════════════════════════════════════════
    # (a) Offline global-max: energy / np.max(energy) per band
    # ═══════════════════════════════════════════════════════
    offline_global = {}
    for band_name, energy in raw_energies.items():
        mx = np.max(energy)
        if mx > 0:
            offline_global[band_name] = energy / mx
        else:
            offline_global[band_name] = energy.copy()

    # ═══════════════════════════════════════════════════════
    # (b) Offline causal RT (viewer.py band_energy_rt, lines 458-486)
    #     Linear decay, seed from first 30s mean, dropout detection
    # ═══════════════════════════════════════════════════════
    precompute_frames = int(30 * fps)
    decay_time_sec = 30
    dropout_percentile = 5

    offline_causal = {}
    for band_name, energy in raw_energies.items():
        nf = len(energy)

        # Dropout threshold: low percentile of non-zero energy
        nonzero = energy[energy > 0]
        dropout_thresh = np.percentile(nonzero, dropout_percentile) if len(nonzero) > 0 else 0

        # Seed from mean of first ~30s
        seed = np.mean(energy[:min(precompute_frames, nf)]) if nf > 0 else 1e-10
        seed = max(seed, 1e-10)

        # Constant (linear) decay per frame
        constant_decay = seed / (decay_time_sec * fps)
        constant_decay = max(constant_decay, 1e-12)

        ref = seed
        normalized = np.empty(nf)
        for i in range(nf):
            e = energy[i]
            if e < dropout_thresh:
                pass  # dropout: freeze reference
            elif e > ref:
                ref = e  # instant attack
            else:
                ref = max(ref - constant_decay, 1e-10)  # constant decay
            normalized[i] = e / ref

        offline_causal[band_name] = normalized

    # ═══════════════════════════════════════════════════════
    # (c) Real-time peak-decay (effects style)
    #     peak = max(energy, peak * decay), cold start from 1e-10
    # ═══════════════════════════════════════════════════════
    decay_constants = {
        '0.9995': 0.9995,  # band_sparkles.py default
        '0.998': 0.998,    # absint_band_pulse.py
        '0.9999': 0.9999,  # best from initial tests
    }

    rt_peak_decay = {}
    for dc_name, dc_val in decay_constants.items():
        rt_peak_decay[dc_name] = {}
        for band_name, energy in raw_energies.items():
            nf = len(energy)
            peak = 1e-10
            normalized = np.empty(nf)
            for i in range(nf):
                peak = max(float(energy[i]), peak * dc_val)
                normalized[i] = energy[i] / peak if peak > 1e-10 else 0.0
            rt_peak_decay[dc_name][band_name] = normalized

    return {
        'raw': raw_energies,
        'offline_global': offline_global,
        'offline_causal': offline_causal,
        'rt_peak_decay': rt_peak_decay,
        'sr': sr,
        'hop_length': hop_length,
        'n_frames': n_frames,
        'duration': duration,
        'fps': fps,
    }


def compute_cold_start_duration(causal, rt, fps, tolerance=0.10):
    """How many seconds until RT stabilizes within `tolerance` of causal.

    Scans from the start. Returns the time at which |rt[i] - causal[i]| < tolerance
    for at least 1 second continuously.
    """
    hold_frames = int(fps)  # need 1s of stability
    streak = 0
    for i in range(len(causal)):
        if abs(rt[i] - causal[i]) < tolerance:
            streak += 1
            if streak >= hold_frames:
                return (i - hold_frames + 1) / fps
        else:
            streak = 0
    return len(causal) / fps  # never stabilized


def print_results(data, track_name):
    """Print correlation and MAE tables."""
    offline_global = data['offline_global']
    offline_causal = data['offline_causal']
    rt_decay = data['rt_peak_decay']
    fps = data['fps']
    n_frames = data['n_frames']

    print(f"\n{'='*90}")
    print(f"  {track_name} — {data['duration']:.1f}s, {n_frames} frames, {fps:.1f} fps")
    print(f"{'='*90}")

    # ── Correlation Table ──
    print(f"\n  PEARSON CORRELATION")
    print(f"  {'Band':<12} {'Global↔Causal':>14} {'Global↔PD0.998':>15} "
          f"{'Global↔PD0.9995':>16} {'Global↔PD0.9999':>16} "
          f"{'Causal↔PD0.9995':>16}")
    print(f"  {'-'*85}")

    for band in BAND_ORDER:
        g = offline_global[band]
        c = offline_causal[band]
        r998 = rt_decay['0.998'][band]
        r9995 = rt_decay['0.9995'][band]
        r9999 = rt_decay['0.9999'][band]

        def corr(a, b):
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        gc = corr(g, c)
        g998 = corr(g, r998)
        g9995 = corr(g, r9995)
        g9999 = corr(g, r9999)
        c9995 = corr(c, r9995)

        print(f"  {band:<12} {gc:>14.4f} {g998:>15.4f} {g9995:>16.4f} "
              f"{g9999:>16.4f} {c9995:>16.4f}")

    # ── MAE Table ──
    print(f"\n  MEAN ABSOLUTE ERROR")
    print(f"  {'Band':<12} {'Global↔Causal':>14} {'Global↔PD0.998':>15} "
          f"{'Global↔PD0.9995':>16} {'Global↔PD0.9999':>16} "
          f"{'Causal↔PD0.9995':>16}")
    print(f"  {'-'*85}")

    for band in BAND_ORDER:
        g = offline_global[band]
        c = offline_causal[band]
        r998 = rt_decay['0.998'][band]
        r9995 = rt_decay['0.9995'][band]
        r9999 = rt_decay['0.9999'][band]

        gc = float(np.mean(np.abs(g - c)))
        g998 = float(np.mean(np.abs(g - r998)))
        g9995 = float(np.mean(np.abs(g - r9995)))
        g9999 = float(np.mean(np.abs(g - r9999)))
        c9995 = float(np.mean(np.abs(c - r9995)))

        print(f"  {band:<12} {gc:>14.4f} {g998:>15.4f} {g9995:>16.4f} "
              f"{g9999:>16.4f} {c9995:>16.4f}")

    # ── Cold Start Transient Duration ──
    print(f"\n  COLD START TRANSIENT (seconds until RT within 10% of causal)")
    print(f"  {'Band':<12} {'PD 0.998':>12} {'PD 0.9995':>12} {'PD 0.9999':>12}")
    print(f"  {'-'*50}")

    for band in BAND_ORDER:
        c = offline_causal[band]
        t998 = compute_cold_start_duration(c, rt_decay['0.998'][band], fps)
        t9995 = compute_cold_start_duration(c, rt_decay['0.9995'][band], fps)
        t9999 = compute_cold_start_duration(c, rt_decay['0.9999'][band], fps)

        print(f"  {band:<12} {t998:>11.1f}s {t9995:>11.1f}s {t9999:>11.1f}s")

    # ── Value Range Comparison ──
    print(f"\n  VALUE RANGES (mean / p95 / max)")
    print(f"  {'Band':<12} {'Offline Global':>18} {'Offline Causal':>18} "
          f"{'RT PD 0.9995':>18}")
    print(f"  {'-'*70}")

    for band in BAND_ORDER:
        g = offline_global[band]
        c = offline_causal[band]
        r = rt_decay['0.9995'][band]

        def stats(arr):
            return f"{np.mean(arr):.3f}/{np.percentile(arr,95):.3f}/{np.max(arr):.3f}"

        print(f"  {band:<12} {stats(g):>18} {stats(c):>18} {stats(r):>18}")

    # ── Where they diverge most ──
    print(f"\n  WORST DIVERGENCE WINDOWS (Global vs RT PD 0.9995)")
    print(f"  {'Band':<12} {'Worst 1s Window':>20} {'Error in Window':>16} {'When (sec)':>12}")
    print(f"  {'-'*65}")

    window_frames = int(fps)
    for band in BAND_ORDER:
        g = offline_global[band]
        r = rt_decay['0.9995'][band]
        errors = np.abs(g - r)

        # Rolling 1s mean error
        if len(errors) > window_frames:
            rolling = np.convolve(errors, np.ones(window_frames) / window_frames, mode='valid')
            worst_idx = np.argmax(rolling)
            worst_err = rolling[worst_idx]
            worst_time = worst_idx / fps
        else:
            worst_err = np.mean(errors)
            worst_time = 0.0

        print(f"  {band:<12} {'':>20} {worst_err:>16.4f} {worst_time:>11.1f}s")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Technique Comparison (4 RT-feasible algorithms)
# ═══════════════════════════════════════════════════════════════════════

def norm_A_baseline_peak_decay(energy, fps):
    """A. Baseline peak-decay: cold start, 0.9995 decay."""
    n = len(energy)
    peak = 1e-10
    out = np.empty(n)
    for i in range(n):
        peak = max(float(energy[i]), peak * 0.9995)
        out[i] = energy[i] / peak if peak > 1e-10 else 0.0
    return out


def norm_B_warmstart_peak_decay(energy, fps):
    """B. Warm-start peak-decay: seed from first 5s mean, 0.9995 decay."""
    n = len(energy)
    warmup = int(5 * fps)
    peak = float(np.mean(energy[:min(warmup, n)]))
    peak = max(peak, 1e-10)
    out = np.empty(n)
    for i in range(n):
        peak = max(float(energy[i]), peak * 0.9995)
        out[i] = energy[i] / peak if peak > 1e-10 else 0.0
    return out


def norm_C_linear_decay(energy, fps):
    """C. Linear decay: seed from first 5s mean, constant decay over 30s."""
    n = len(energy)
    warmup = int(5 * fps)
    seed = float(np.mean(energy[:min(warmup, n)]))
    seed = max(seed, 1e-10)
    constant_decay = seed / (30 * fps)
    constant_decay = max(constant_decay, 1e-12)
    ref = seed
    out = np.empty(n)
    for i in range(n):
        if energy[i] > ref:
            ref = float(energy[i])  # instant attack
        else:
            ref = max(ref - constant_decay, 1e-10)
        out[i] = energy[i] / ref
    return out


def norm_D_multi_timescale(energy, fps, fast_weight=0.7):
    """D. Multi-timescale blend: two peak-decay at different speeds.

    fast: 0.998 (~12s half-life), slow: 0.9998 (~5min half-life).
    Both warm-started from first 5s mean.
    """
    n = len(energy)
    warmup = int(5 * fps)
    seed = float(np.mean(energy[:min(warmup, n)]))
    seed = max(seed, 1e-10)
    peak_fast = seed
    peak_slow = seed
    out = np.empty(n)
    slow_weight = 1.0 - fast_weight
    for i in range(n):
        e = float(energy[i])
        peak_fast = max(e, peak_fast * 0.998)
        peak_slow = max(e, peak_slow * 0.9998)
        norm_fast = e / peak_fast if peak_fast > 1e-10 else 0.0
        norm_slow = e / peak_slow if peak_slow > 1e-10 else 0.0
        out[i] = fast_weight * norm_fast + slow_weight * norm_slow
    return out


def corr(a, b):
    """Pearson correlation, safe for constant signals."""
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_technique_comparison(wav_path, track_name):
    """Compare 4 RT-feasible normalization algorithms against offline global-max."""
    from scipy.ndimage import uniform_filter1d

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    n_fft = 2048
    hop_length = 512
    fps = sr / hop_length
    smooth_size = 30  # frames for arc preservation

    # Mel spectrogram
    mel_spec_128 = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=128, fmin=20, fmax=8000
    )
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)

    # Raw band energies
    raw_energies = {}
    for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
        band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)
        raw_energies[band_name] = np.sum(mel_spec_128[band_mask, :], axis=0)

    # Offline global-max reference
    offline_global = {}
    for band_name, energy in raw_energies.items():
        mx = np.max(energy)
        offline_global[band_name] = energy / mx if mx > 0 else energy.copy()

    # ── Run all 4 algorithms per band ──
    algorithms = {
        'A: Baseline PD': norm_A_baseline_peak_decay,
        'B: Warmstart PD': norm_B_warmstart_peak_decay,
        'C: Linear Decay': norm_C_linear_decay,
        'D: Multi-TS 70/30': lambda e, f: norm_D_multi_timescale(e, f, 0.7),
    }

    results = {}  # {algo_name: {band_name: normalized_array}}
    for algo_name, algo_fn in algorithms.items():
        results[algo_name] = {}
        for band_name, energy in raw_energies.items():
            results[algo_name][band_name] = algo_fn(energy, fps)

    # ── Print comparison table ──
    print(f"\n{'='*100}")
    print(f"  TECHNIQUE COMPARISON — {track_name} ({duration:.1f}s)")
    print(f"{'='*100}")

    # Correlation table
    print(f"\n  PEARSON CORRELATION vs Offline Global-Max")
    header = f"  {'Band':<12}"
    for algo_name in algorithms:
        header += f" {algo_name:>18}"
    print(header)
    print(f"  {'-'*len(header)}")

    for band in BAND_ORDER:
        ref = offline_global[band]
        row = f"  {band:<12}"
        for algo_name in algorithms:
            c = corr(ref, results[algo_name][band])
            row += f" {c:>18.4f}"
        print(row)

    # MAE table
    print(f"\n  MEAN ABSOLUTE ERROR vs Offline Global-Max")
    print(header)
    print(f"  {'-'*len(header)}")

    for band in BAND_ORDER:
        ref = offline_global[band]
        row = f"  {band:<12}"
        for algo_name in algorithms:
            mae = float(np.mean(np.abs(ref - results[algo_name][band])))
            row += f" {mae:>18.4f}"
        print(row)

    # Arc preservation: correlation of smoothed versions
    print(f"\n  ARC PRESERVATION (correlation of {smooth_size}-frame smoothed signals)")
    print(header)
    print(f"  {'-'*len(header)}")

    for band in BAND_ORDER:
        ref = offline_global[band]
        ref_smooth = uniform_filter1d(ref, size=smooth_size)
        row = f"  {band:<12}"
        for algo_name in algorithms:
            cand = results[algo_name][band]
            cand_smooth = uniform_filter1d(cand, size=smooth_size)
            c = corr(ref_smooth, cand_smooth)
            row += f" {c:>18.4f}"
        print(row)

    # ── Algorithm D blend ratio sweep ──
    print(f"\n  {'='*90}")
    print(f"  MULTI-TIMESCALE BLEND RATIO SWEEP (Algorithm D)")
    print(f"  {'='*90}")

    blend_ratios = [
        (0.3, 0.7, '30/70 (slow-heavy)'),
        (0.5, 0.5, '50/50 (equal)'),
        (0.7, 0.3, '70/30 (fast-heavy)'),
        (0.85, 0.15, '85/15 (mostly fast)'),
    ]

    # Correlation
    print(f"\n  CORRELATION vs Offline Global-Max")
    hdr = f"  {'Band':<12}"
    for _, _, label in blend_ratios:
        hdr += f" {label:>20}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    blend_results = {}
    for fast_w, slow_w, label in blend_ratios:
        blend_results[label] = {}
        for band_name, energy in raw_energies.items():
            blend_results[label][band_name] = norm_D_multi_timescale(energy, fps, fast_w)

    for band in BAND_ORDER:
        ref = offline_global[band]
        row = f"  {band:<12}"
        for _, _, label in blend_ratios:
            c = corr(ref, blend_results[label][band])
            row += f" {c:>20.4f}"
        print(row)

    # MAE
    print(f"\n  MAE vs Offline Global-Max")
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    for band in BAND_ORDER:
        ref = offline_global[band]
        row = f"  {band:<12}"
        for _, _, label in blend_ratios:
            mae = float(np.mean(np.abs(ref - blend_results[label][band])))
            row += f" {mae:>20.4f}"
        print(row)

    # Arc preservation
    print(f"\n  ARC PRESERVATION (smoothed correlation)")
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    for band in BAND_ORDER:
        ref = offline_global[band]
        ref_smooth = uniform_filter1d(ref, size=smooth_size)
        row = f"  {band:<12}"
        for _, _, label in blend_ratios:
            cand = blend_results[label][band]
            cand_smooth = uniform_filter1d(cand, size=smooth_size)
            c = corr(ref_smooth, cand_smooth)
            row += f" {c:>20.4f}"
        print(row)

    # ── Mean across bands summary ──
    print(f"\n  MEAN ACROSS ALL BANDS")
    print(f"  {'Metric':<20}", end='')
    for _, _, label in blend_ratios:
        print(f" {label:>20}", end='')
    print()
    print(f"  {'-'*100}")

    for metric_name, metric_fn in [
        ('Correlation', lambda ref, cand: corr(ref, cand)),
        ('MAE', lambda ref, cand: float(np.mean(np.abs(ref - cand)))),
        ('Arc Preservation', lambda ref, cand: corr(
            uniform_filter1d(ref, size=smooth_size),
            uniform_filter1d(cand, size=smooth_size))),
    ]:
        row = f"  {metric_name:<20}"
        for _, _, label in blend_ratios:
            vals = []
            for band in BAND_ORDER:
                ref = offline_global[band]
                cand = blend_results[label][band]
                vals.append(metric_fn(ref, cand))
            row += f" {np.mean(vals):>20.4f}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Transition Behavior Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_transition_analysis(wav_path, track_name):
    """Analyze normalizer behavior at dynamic transitions."""
    from scipy.ndimage import uniform_filter1d

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    n_fft = 2048
    hop_length = 512
    fps = sr / hop_length

    # Compute RMS per frame
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    n_frames = len(rms)
    times = np.arange(n_frames) / fps

    # Smooth RMS to find structural transitions (not beat-level)
    rms_smooth = uniform_filter1d(rms, size=int(2 * fps))  # 2s smoothing

    # Find transitions via RMS derivative
    rms_deriv = np.diff(rms_smooth, prepend=rms_smooth[0]) * fps
    rms_deriv_smooth = uniform_filter1d(rms_deriv, size=int(1 * fps))

    # Quiet-to-loud: largest positive derivative
    qtl_frame = np.argmax(rms_deriv_smooth)
    qtl_time = qtl_frame / fps

    # Loud-to-quiet: largest negative derivative
    ltq_frame = np.argmin(rms_deriv_smooth)
    ltq_time = ltq_frame / fps

    print(f"\n{'='*100}")
    print(f"  TRANSITION BEHAVIOR ANALYSIS — {track_name} ({duration:.1f}s)")
    print(f"{'='*100}")
    print(f"\n  Detected transitions:")
    print(f"    Quiet-to-loud: {qtl_time:.1f}s (RMS deriv = {rms_deriv_smooth[qtl_frame]:.6f})")
    print(f"    Loud-to-quiet: {ltq_time:.1f}s (RMS deriv = {rms_deriv_smooth[ltq_frame]:.6f})")
    print(f"    Peak RMS: {np.max(rms):.6f} at {times[np.argmax(rms)]:.1f}s")
    print(f"    Mean RMS: {np.mean(rms):.6f}")

    # Mel spectrogram for band analysis
    mel_spec_128 = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=128, fmin=20, fmax=8000
    )
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)

    # Use Mids band as representative
    band_mask = (mel_freqs >= 250) & (mel_freqs <= 2000)
    raw_mids = np.sum(mel_spec_128[band_mask, :], axis=0)

    # Offline reference
    offline = raw_mids / np.max(raw_mids) if np.max(raw_mids) > 0 else raw_mids.copy()

    # All normalization strategies to test
    decay_values = {
        'PD 0.998': 0.998,
        'PD 0.9995': 0.9995,
        'PD 0.9999': 0.9999,
        'PD 0.99995': 0.99995,
    }

    strategies = {}
    for name, dc in decay_values.items():
        peak = 1e-10
        out = np.empty(len(raw_mids))
        for i in range(len(raw_mids)):
            peak = max(float(raw_mids[i]), peak * dc)
            out[i] = raw_mids[i] / peak if peak > 1e-10 else 0.0
        strategies[name] = out

    # Add linear decay
    warmup = int(5 * fps)
    seed = float(np.mean(raw_mids[:min(warmup, len(raw_mids))]))
    seed = max(seed, 1e-10)
    constant_decay = seed / (30 * fps)
    ref = seed
    out = np.empty(len(raw_mids))
    for i in range(len(raw_mids)):
        if raw_mids[i] > ref:
            ref = float(raw_mids[i])
        else:
            ref = max(ref - constant_decay, 1e-10)
        out[i] = raw_mids[i] / ref
    strategies['Linear Decay'] = out

    # ── Quiet-to-loud analysis ──
    print(f"\n  QUIET-TO-LOUD TRANSITION (at {qtl_time:.1f}s, Mids band)")
    print(f"  {'Strategy':<20} {'First 0.5s mean':>16} {'Offline 0.5s mean':>18} {'Ratio (flash)':>14}")
    print(f"  {'-'*72}")

    window_05s = int(0.5 * fps)
    qtl_start = max(0, qtl_frame - int(0.25 * fps))  # center on transition
    qtl_end = min(len(raw_mids), qtl_start + window_05s)

    offline_qtl = np.mean(offline[qtl_start:qtl_end])
    for name, norm_data in strategies.items():
        strat_qtl = np.mean(norm_data[qtl_start:qtl_end])
        ratio = strat_qtl / offline_qtl if offline_qtl > 1e-10 else 0.0
        print(f"  {name:<20} {strat_qtl:>16.4f} {offline_qtl:>18.4f} {ratio:>14.2f}x")

    # ── Loud-to-quiet analysis ──
    # Find the loud peak region (within 10s before the loud-to-quiet transition)
    search_start = max(0, ltq_frame - int(10 * fps))
    loud_peak_frame = search_start + np.argmax(rms_smooth[search_start:ltq_frame+1])
    loud_peak_time = loud_peak_frame / fps

    print(f"\n  LOUD-TO-QUIET TRANSITION (loud peak at {loud_peak_time:.1f}s, drop at {ltq_time:.1f}s)")
    print(f"  Adaptation lag: seconds after loud peak until normalizer output reaches")
    print(f"  50% of what offline shows (i.e., quiet section looks reasonably bright)")
    print(f"")
    print(f"  {'Strategy':<20} {'Re-sensitize time':>18} {'Mean quiet output':>18} {'Offline quiet mean':>18}")
    print(f"  {'-'*78}")

    # Define "quiet region" as 5s starting 3s after the loud-to-quiet transition
    quiet_start = min(len(raw_mids) - 1, ltq_frame + int(3 * fps))
    quiet_end = min(len(raw_mids), quiet_start + int(5 * fps))

    # If the track ends before we get a quiet section, use whatever's left
    if quiet_end <= quiet_start:
        quiet_start = max(0, len(raw_mids) - int(2 * fps))
        quiet_end = len(raw_mids)

    offline_quiet_mean = np.mean(offline[quiet_start:quiet_end]) if quiet_end > quiet_start else 0

    for name, norm_data in strategies.items():
        strat_quiet_mean = np.mean(norm_data[quiet_start:quiet_end]) if quiet_end > quiet_start else 0

        # Find re-sensitization time: scan from loud peak forward
        # until normalizer output reaches 50% of offline for 0.5s
        target = 0.5
        hold_frames = int(0.5 * fps)
        streak = 0
        resensitize_time = -1
        for i in range(loud_peak_frame, min(len(raw_mids), loud_peak_frame + int(60 * fps))):
            off_val = offline[i]
            strat_val = norm_data[i]
            if off_val > 0.05:  # only check when offline says there's signal
                ratio = strat_val / off_val if off_val > 1e-10 else 0
                if ratio >= target:
                    streak += 1
                    if streak >= hold_frames:
                        resensitize_time = (i - hold_frames + 1 - loud_peak_frame) / fps
                        break
                else:
                    streak = 0

        if resensitize_time < 0:
            rs_str = "never (>60s)"
        else:
            rs_str = f"{resensitize_time:.1f}s"

        print(f"  {name:<20} {rs_str:>18} {strat_quiet_mean:>18.4f} {offline_quiet_mean:>18.4f}")

    # ── Decay constant sweet spot analysis ──
    print(f"\n  DECAY CONSTANT SWEEP (Mids band, full track)")
    print(f"  {'Decay':<12} {'Half-life':>12} {'Corr':>8} {'MAE':>8} {'Arc Pres':>10}")
    print(f"  {'-'*54}")

    for dc_val in [0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99993, 0.99995, 0.99998, 0.99999]:
        half_life_frames = np.log(2) / (1 - dc_val) if dc_val < 1 else float('inf')
        half_life_sec = half_life_frames / fps

        peak = 1e-10
        out = np.empty(len(raw_mids))
        for i in range(len(raw_mids)):
            peak = max(float(raw_mids[i]), peak * dc_val)
            out[i] = raw_mids[i] / peak if peak > 1e-10 else 0.0

        c = corr(offline, out)
        mae = float(np.mean(np.abs(offline - out)))
        # Arc preservation
        off_s = uniform_filter1d(offline, size=30)
        out_s = uniform_filter1d(out, size=30)
        arc = corr(off_s, out_s)

        print(f"  {dc_val:<12} {half_life_sec:>10.1f}s {c:>8.4f} {mae:>8.4f} {arc:>10.4f}")

    # Also test linear decay
    c_lin = corr(offline, strategies['Linear Decay'])
    mae_lin = float(np.mean(np.abs(offline - strategies['Linear Decay'])))
    off_s = uniform_filter1d(offline, size=30)
    lin_s = uniform_filter1d(strategies['Linear Decay'], size=30)
    arc_lin = corr(off_s, lin_s)
    print(f"  {'Linear':<12} {'(adaptive)':>12} {c_lin:>8.4f} {mae_lin:>8.4f} {arc_lin:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════
# PART 4: RMS Integral Normalization
# ═══════════════════════════════════════════════════════════════════════

def run_rms_integral_test(wav_path, track_name):
    """Test self-relative normalization for RMS integral."""
    from scipy.ndimage import uniform_filter1d

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    n_fft = 2048
    hop_length = 512
    fps = sr / hop_length

    # Compute RMS per frame
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    n_frames = len(rms)
    times = np.arange(n_frames) / fps

    # Rolling RMS integral (10s window)
    window_frames = int(10 * fps)
    rms_ring = np.zeros(window_frames, dtype=np.float64)
    rms_ring_pos = 0
    rms_ring_filled = 0
    rms_integral = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        rms_ring[rms_ring_pos % window_frames] = rms[i]
        rms_ring_pos += 1
        rms_ring_filled = min(rms_ring_filled + 1, window_frames)
        rms_integral[i] = float(np.sum(rms_ring[:rms_ring_filled]))

    # Offline ground truth
    integral_max = np.max(rms_integral)
    offline_global = rms_integral / integral_max if integral_max > 1e-10 else rms_integral.copy()

    print(f"\n{'='*100}")
    print(f"  RMS INTEGRAL NORMALIZATION — {track_name} ({duration:.1f}s)")
    print(f"{'='*100}")
    print(f"  Window: {window_frames} frames ({window_frames/fps:.1f}s)")
    print(f"  Max integral: {integral_max:.6f}")

    # ── Strategy A: Peak-decay on the integral itself ──
    strategies = {}
    for dc_name, dc_val in [('PD 0.998', 0.998), ('PD 0.9995', 0.9995),
                              ('PD 0.9999', 0.9999), ('PD 0.99995', 0.99995)]:
        peak = 1e-10
        out = np.empty(n_frames)
        for i in range(n_frames):
            peak = max(float(rms_integral[i]), peak * dc_val)
            out[i] = rms_integral[i] / peak if peak > 1e-10 else 0.0
        strategies[dc_name] = out

    # ── Strategy B: Linear decay on the integral ──
    warmup = int(5 * fps)
    seed = float(np.mean(rms_integral[:min(warmup, n_frames)]))
    seed = max(seed, 1e-10)
    constant_decay = seed / (30 * fps)
    ref = seed
    out = np.empty(n_frames)
    for i in range(n_frames):
        if rms_integral[i] > ref:
            ref = float(rms_integral[i])
        else:
            ref = max(ref - constant_decay, 1e-10)
        out[i] = rms_integral[i] / ref
    strategies['Linear Decay'] = out

    # ── Strategy C: Self-relative (integral / (rms_peak * window_frames)) ──
    for dc_name, dc_val in [('Self-rel 0.9999', 0.9999),
                              ('Self-rel 0.99995', 0.99995)]:
        rms_peak = 1e-10
        out = np.empty(n_frames)
        for i in range(n_frames):
            rms_peak = max(float(rms[i]), rms_peak * dc_val)
            theoretical_max = rms_peak * window_frames
            out[i] = rms_integral[i] / theoretical_max if theoretical_max > 1e-10 else 0.0
        strategies[dc_name] = out

    # ── Strategy D: Self-relative with warm-start ──
    warmup = int(5 * fps)
    rms_peak = float(np.mean(rms[:min(warmup, n_frames)]))
    rms_peak = max(rms_peak, 1e-10)
    out = np.empty(n_frames)
    for i in range(n_frames):
        rms_peak = max(float(rms[i]), rms_peak * 0.9999)
        theoretical_max = rms_peak * window_frames
        out[i] = rms_integral[i] / theoretical_max if theoretical_max > 1e-10 else 0.0
    strategies['Self-rel warm 0.9999'] = out

    # ── Strategy E: Self-relative with linear decay on RMS peak ──
    warmup = int(5 * fps)
    seed_rms = float(np.mean(rms[:min(warmup, n_frames)]))
    seed_rms = max(seed_rms, 1e-10)
    rms_decay = seed_rms / (30 * fps)
    rms_ref = seed_rms
    out = np.empty(n_frames)
    for i in range(n_frames):
        if rms[i] > rms_ref:
            rms_ref = float(rms[i])
        else:
            rms_ref = max(rms_ref - rms_decay, 1e-10)
        theoretical_max = rms_ref * window_frames
        out[i] = rms_integral[i] / theoretical_max if theoretical_max > 1e-10 else 0.0
    strategies['Self-rel linear RMS'] = out

    # ── Print results ──
    print(f"\n  {'Strategy':<25} {'Corr':>8} {'MAE':>8} {'Arc Pres':>10} "
          f"{'Mean':>8} {'P95':>8} {'Max':>8}")
    print(f"  {'-'*80}")

    for name, norm_data in strategies.items():
        c = corr(offline_global, norm_data)
        mae = float(np.mean(np.abs(offline_global - norm_data)))
        off_s = uniform_filter1d(offline_global, size=30)
        out_s = uniform_filter1d(norm_data, size=30)
        arc = corr(off_s, out_s)
        mn = float(np.mean(norm_data))
        p95 = float(np.percentile(norm_data, 95))
        mx = float(np.max(norm_data))

        print(f"  {name:<25} {c:>8.4f} {mae:>8.4f} {arc:>10.4f} "
              f"{mn:>8.3f} {p95:>8.3f} {mx:>8.3f}")

    # Offline reference stats for comparison
    mn = float(np.mean(offline_global))
    p95 = float(np.percentile(offline_global, 95))
    print(f"\n  {'Offline global-max':<25} {'1.000':>8} {'0.000':>8} {'1.000':>10} "
          f"{mn:>8.3f} {p95:>8.3f} {'1.000':>8}")


def main():
    test_tracks = {
        'fa_br_drop1': os.path.join(SEGMENTS_DIR, 'fa_br_drop1.wav'),
        'fourtet_lostvillage_screech': os.path.join(SEGMENTS_DIR, 'fourtet_lostvillage_screech.wav'),
        'complex_beat_glue': os.path.join(SEGMENTS_DIR, 'complex_beat_glue.wav'),
    }

    # Part 1: Three-way comparison (all tracks)
    for tid, path in test_tracks.items():
        if not os.path.exists(path):
            print(f"  SKIP: {tid} — file not found at {path}")
            continue

        print(f"\nLoading {tid}...")
        data = compute_three_ways(path)
        print_results(data, tid)

    # Part 2: Technique comparison (fa_br_drop1 primary, then others)
    print(f"\n\n{'#'*100}")
    print(f"# PART 2: TECHNIQUE COMPARISON")
    print(f"{'#'*100}")

    for tid, path in test_tracks.items():
        if not os.path.exists(path):
            continue
        run_technique_comparison(path, tid)

    # Part 3: Transition behavior (fa_br_drop1 only — has clear structure)
    print(f"\n\n{'#'*100}")
    print(f"# PART 3: TRANSITION BEHAVIOR ANALYSIS")
    print(f"{'#'*100}")

    primary_path = test_tracks['fa_br_drop1']
    if os.path.exists(primary_path):
        run_transition_analysis(primary_path, 'fa_br_drop1')

    # Part 4: RMS integral (all tracks)
    print(f"\n\n{'#'*100}")
    print(f"# PART 4: RMS INTEGRAL NORMALIZATION")
    print(f"{'#'*100}")

    for tid, path in test_tracks.items():
        if not os.path.exists(path):
            continue
        run_rms_integral_test(path, tid)


if __name__ == '__main__':
    main()
