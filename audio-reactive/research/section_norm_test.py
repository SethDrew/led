#!/usr/bin/env python3
"""
Section-Aware Normalization Test

Goal: normalization that makes within-section dynamics visible while
preserving section-level differences.

Algorithms:
  A. Baseline peak-decay (current effects approach)
  B. Integral-referenced (trailing mean as reference)
  C. EMA-ratio (exponential moving average as section envelope)
  D. Dual-track (section + detail via geometric mean)
  E. Percentile-windowed (local min-max range normalization)
  F. Causal RT (viewer.py-style linear decay with dropout detection)

Metrics:
  - Intra-section CV: coefficient of variation within 10s windows (beat visibility)
  - Inter-section range: range of window means across track (section contrast)
  - Product: intra_CV * inter_range (joint quality score)
"""

import os
import sys
import numpy as np
import librosa
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──
SEGMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio-segments')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normalization-tests')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FREQUENCY_BANDS = {
    'Sub-bass': (20, 80),
    'Bass': (80, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 6000),
    'Treble': (6000, 8000),
}
BAND_ORDER = ['Sub-bass', 'Bass', 'Mids', 'High-mids', 'Treble']


# ═══════════════════════════════════════════════════════════════════════
# Normalization Algorithms
# ═══════════════════════════════════════════════════════════════════════

def norm_A_peak_decay(energy, fps):
    """A. Baseline peak-decay at ~65s half-life (matching current band_sparkles)."""
    half_life = 65.0
    decay = 2.0 ** (-1.0 / (half_life * fps))
    n = len(energy)
    peak = 1e-10
    out = np.empty(n)
    for i in range(n):
        peak = max(float(energy[i]), peak * decay)
        out[i] = energy[i] / peak if peak > 1e-10 else 0.0
    return out


def norm_B_integral_ref(energy, fps, window_sec=5.0):
    """B. Integral-referenced: normalize by trailing rolling mean."""
    window = int(window_sec * fps)
    n = len(energy)
    ring = np.zeros(window, dtype=np.float64)
    ring_pos = 0
    ring_sum = 0.0
    out = np.empty(n)

    for i in range(n):
        idx = ring_pos % window
        ring_sum -= ring[idx]
        ring[idx] = float(energy[i])
        ring_sum += float(energy[i])
        ring_pos += 1

        local_mean = ring_sum / min(ring_pos, window)
        raw = energy[i] / local_mean if local_mean > 1e-10 else 0.0
        # Scale: values center around 1.0, clip to 0-1 via /2
        out[i] = min(raw / 2.0, 1.0)

    return out


def norm_C_ema_ratio(energy, fps, window_sec=5.0):
    """C. EMA-ratio: normalize by exponential moving average."""
    alpha = 2.0 / (window_sec * fps + 1)
    n = len(energy)
    ema = float(energy[0]) if n > 0 else 1e-10
    out = np.empty(n)

    for i in range(n):
        ema = alpha * float(energy[i]) + (1 - alpha) * ema
        raw = energy[i] / ema if ema > 1e-10 else 0.0
        out[i] = min(raw / 2.0, 1.0)

    return out


def norm_D_dual_track(energy, fps):
    """D. Dual-track: geometric mean of section-level and detail-level."""
    decay_slow = 2.0 ** (-1.0 / (60 * fps))   # 60s half-life
    decay_fast = 2.0 ** (-1.0 / (5 * fps))    # 5s half-life
    n = len(energy)
    peak_slow = 1e-10
    peak_fast = 1e-10
    out = np.empty(n)

    for i in range(n):
        e = float(energy[i])
        peak_slow = max(e, peak_slow * decay_slow)
        peak_fast = max(e, peak_fast * decay_fast)

        section_level = e / peak_slow if peak_slow > 1e-10 else 0.0
        detail_level = e / peak_fast if peak_fast > 1e-10 else 0.0

        out[i] = np.sqrt(section_level * detail_level)

    return out


def norm_E_percentile_window(energy, fps, window_sec=10.0):
    """E. Percentile-windowed: normalize to local min-max range."""
    window = int(window_sec * fps)
    n = len(energy)
    max_deque = deque()  # monotonic decreasing
    min_deque = deque()  # monotonic increasing
    out = np.empty(n)

    for i in range(n):
        e = float(energy[i])

        # Update max deque
        while max_deque and e >= energy[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(i)
        if max_deque[0] <= i - window:
            max_deque.popleft()

        # Update min deque
        while min_deque and e <= energy[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(i)
        if min_deque[0] <= i - window:
            min_deque.popleft()

        local_max = float(energy[max_deque[0]])
        local_min = float(energy[min_deque[0]])
        span = local_max - local_min
        out[i] = (e - local_min) / span if span > 1e-10 else 0.5

    return out


def norm_F_causal_rt(energy, fps):
    """F. Causal RT: viewer.py-style linear decay with dropout detection.

    5s warmup seed, 30s linear decay, instant attack, dropout freeze.
    """
    n = len(energy)
    warmup_frames = min(int(5 * fps), n)
    out = np.empty(n)

    # Seed from first 5s mean
    seed = float(np.mean(energy[:warmup_frames]))
    seed = max(seed, 1e-10)
    constant_decay = seed / (30 * fps)

    # Dropout threshold: 5th percentile of first 5s (non-zero values)
    warmup = energy[:warmup_frames]
    positive = warmup[warmup > 0]
    dropout_thresh = float(np.percentile(positive, 5)) if len(positive) > 0 else 0.0

    ref = seed
    for i in range(n):
        e = float(energy[i])
        if e < dropout_thresh:
            pass  # freeze reference during dropout
        elif e > ref:
            ref = e  # instant attack
        else:
            ref = max(ref - constant_decay, 1e-10)  # linear decay
        out[i] = e / ref if ref > 1e-10 else 0.0

    return out


def _hybrid_components(energy, fps, window_sec=10.0, decay_sec=60.0):
    """Shared component computation for G-family hybrid algorithms.

    Uses linear decay with recalibration for the slow peak tracker.
    When a new peak arrives, the decay rate is recalibrated so the peak
    always takes ~decay_sec to reach zero from the current level.

    Returns (local_dynamics, section_position) arrays.
      local_dynamics[i] = energy[i] / trailing_mean  (centered ~1.0)
      section_position[i] = trailing_mean / slow_peak  (0-1 range)
    """
    window = int(window_sec * fps)
    n = len(energy)

    ring = np.zeros(window, dtype=np.float64)
    ring_pos = 0
    ring_sum = 0.0

    # Seed slow peak from first 5s mean
    warmup_frames = min(int(5 * fps), n)
    seed = float(np.mean(energy[:warmup_frames]))
    seed = max(seed, 1e-10)
    peak_slow = seed
    slow_decay_rate = seed / (decay_sec * fps)

    local_dynamics = np.empty(n)
    section_position = np.empty(n)

    for i in range(n):
        e = float(energy[i])

        # Update trailing mean
        idx = ring_pos % window
        ring_sum -= ring[idx]
        ring[idx] = e
        ring_sum += e
        ring_pos += 1
        local_mean = ring_sum / min(ring_pos, window)

        # Update slow peak with linear decay + recalibration
        if e > peak_slow:
            peak_slow = e
            # Recalibrate: decay from new peak to zero in ~decay_sec
            slow_decay_rate = peak_slow / (decay_sec * fps)
        else:
            peak_slow = max(peak_slow - slow_decay_rate, 1e-10)

        local_dynamics[i] = e / local_mean if local_mean > 1e-10 else 0.0
        section_position[i] = local_mean / peak_slow if peak_slow > 1e-10 else 0.0

    return local_dynamics, section_position


def norm_G1_power_compress(energy, fps, gamma=0.3):
    """G1. Power-law compression of section position.

    section_compressed = section_position^gamma (lift quiet sections)
    normalized = section_compressed * local_dynamics
    gamma=0.3: quiet at 10% of peak → 50% baseline (generous)
    gamma=0.5: quiet at 10% of peak → 32% baseline (moderate)
    gamma=0.7: quiet at 10% of peak → 20% baseline (conservative)
    """
    local_dyn, sec_pos = _hybrid_components(energy, fps)
    sec_compressed = np.power(sec_pos, gamma)
    raw = sec_compressed * local_dyn
    return np.clip(raw / 2.0, 0.0, 1.0)


def norm_G1_ema_hybrid(energy, fps, gamma=0.3, ema_sec=20.0, decay_sec=60.0):
    """G1-EMA: Power-law compression with EMA as local envelope.

    Uses EMA (not trailing ring buffer) for local_dynamics,
    and linear-decay peak for section_position.
    """
    alpha = 2.0 / (ema_sec * fps + 1)
    n = len(energy)

    # Seed
    warmup_frames = min(int(5 * fps), n)
    seed = float(np.mean(energy[:warmup_frames]))
    seed = max(seed, 1e-10)

    ema = seed
    peak_slow = seed
    slow_decay_rate = seed / (decay_sec * fps)

    out = np.empty(n)

    for i in range(n):
        e = float(energy[i])

        # Update EMA
        ema = alpha * e + (1 - alpha) * ema

        # Update linear decay peak with recalibration
        if e > peak_slow:
            peak_slow = e
            slow_decay_rate = peak_slow / (decay_sec * fps)
        else:
            peak_slow = max(peak_slow - slow_decay_rate, 1e-10)

        local_dynamics = e / ema if ema > 1e-10 else 0.0
        section_position = ema / peak_slow if peak_slow > 1e-10 else 0.0

        out[i] = (section_position ** gamma) * local_dynamics

    # No /2 scaling — section_position^gamma already compresses below 1.0
    # Clip only at 1.0 for display; report raw stats before clipping
    return out


def norm_G2_additive_blend(energy, fps):
    """G2. Additive blend: 60% local dynamics + 40% section context.

    normalized = 0.6 * clip(local_dynamics/2, 0, 1) + 0.4 * section_position
    """
    local_dyn, sec_pos = _hybrid_components(energy, fps)
    local_01 = np.clip(local_dyn / 2.0, 0.0, 1.0)
    return np.clip(0.6 * local_01 + 0.4 * sec_pos, 0.0, 1.0)


def norm_G3_section_range(energy, fps):
    """G3. Section-modulated range.

    Maps local dynamics to [section_floor, 1.0] where floor depends on section level.
    Quiet sections: floor at 0.3. Loud sections: floor at 0.0.
    """
    local_dyn, sec_pos = _hybrid_components(energy, fps)
    local_01 = np.clip(local_dyn / 2.0, 0.0, 1.0)
    section_floor = 0.3 * (1.0 - sec_pos)
    return section_floor + local_01 * (1.0 - section_floor)


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_section_metrics(normalized, fps, window_sec=10.0):
    """Compute section-aware metrics.

    Returns dict with:
      intra_cv: mean coefficient of variation within 10s windows
      inter_range: range of window means
      product: intra_cv * inter_range
    """
    window = int(window_sec * fps)
    n = len(normalized)

    # Split into non-overlapping windows
    window_means = []
    window_cvs = []

    for start in range(0, n - window + 1, window):
        chunk = normalized[start:start + window]
        mn = np.mean(chunk)
        sd = np.std(chunk)
        window_means.append(mn)
        cv = sd / mn if mn > 1e-10 else 0.0
        window_cvs.append(cv)

    if not window_means:
        return {'intra_cv': 0.0, 'inter_range': 0.0, 'product': 0.0}

    intra_cv = float(np.mean(window_cvs))
    inter_range = float(np.max(window_means) - np.min(window_means))
    product = intra_cv * inter_range

    return {
        'intra_cv': intra_cv,
        'inter_range': inter_range,
        'product': product,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Test
# ═══════════════════════════════════════════════════════════════════════

def compute_absint(y, sr, n_fft=2048, hop_length=512, window_sec=0.15):
    """Compute AbsInt signal matching signals.py AbsIntegral.

    Returns raw abs-integral array (un-normalized), one value per hop frame.
    """
    dt = n_fft / sr  # time per frame (for derivative scaling)

    # Frame-level RMS
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    n_frames = len(rms)

    # Ring buffer of |d(RMS)/dt| values
    window_frames = max(1, int(window_sec / dt))
    deriv_buf = np.zeros(window_frames, dtype=np.float64)

    absint = np.empty(n_frames)
    prev_rms = 0.0

    for i in range(n_frames):
        rms_deriv = (float(rms[i]) - prev_rms) / dt
        prev_rms = float(rms[i])

        deriv_buf[i % window_frames] = abs(rms_deriv)
        absint[i] = np.sum(deriv_buf) * dt

    return absint


def run_signal_test(signal, signal_name, track_name, fps, times, algo_names, algo_fns):
    """Run algorithms on a signal, print metrics table + output range, generate plot."""
    results = {}
    for name in algo_names:
        results[name] = algo_fns[name](signal, fps)

    # Print metrics table with output distribution
    print(f"\n  {signal_name}")
    print(f"  {'Algorithm':<20} {'Intra-CV':>10} {'Inter-Range':>12} {'Product':>10}  "
          f"{'Mean':>6} {'P5':>6} {'P25':>6} {'P75':>6} {'P95':>6}  "
          f"{'%>1':>5} {'Max':>6}")
    print(f"  {'-'*108}")

    for name in algo_names:
        m = compute_section_metrics(results[name], fps)
        d = results[name]
        mn = float(np.mean(d))
        p5 = float(np.percentile(d, 5))
        p25 = float(np.percentile(d, 25))
        p75 = float(np.percentile(d, 75))
        p95 = float(np.percentile(d, 95))
        pct_over1 = float(np.mean(d > 1.0) * 100)
        mx = float(np.max(d))
        print(f"  {name:<20} {m['intra_cv']:>10.4f} {m['inter_range']:>12.4f} {m['product']:>10.4f}  "
              f"{mn:>6.3f} {p5:>6.3f} {p25:>6.3f} {p75:>6.3f} {p95:>6.3f}  "
              f"{pct_over1:>5.1f} {mx:>6.3f}")

    # Generate plot: raw + all algorithms
    n_rows = 1 + len(algo_names)
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 2.2 * n_rows), sharex=True)
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#1a1a2e')

    algo_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD',
        '#98FB98', '#FFD700', '#FFA07A', '#FF69B4', '#00CED1',
        '#87CEEB', '#F0E68C',
    ]

    # Row 0: Raw signal
    raw_norm = signal / np.max(signal) if np.max(signal) > 0 else signal
    axes[0].plot(times, raw_norm, color='#AAAAAA', linewidth=0.5)
    axes[0].set_ylabel('Raw')
    axes[0].set_title(f'{track_name} — {signal_name} normalization comparison', fontsize=12)
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].grid(True, alpha=0.15)

    for idx, name in enumerate(algo_names):
        ax = axes[idx + 1]
        norm_data = results[name]
        m = compute_section_metrics(norm_data, fps)
        c = algo_colors[idx % len(algo_colors)]

        ax.plot(times, norm_data, color=c, linewidth=0.5, alpha=0.7)
        ax.set_ylabel(name.split(':')[0])
        y_max = max(1.1, float(np.percentile(norm_data, 99.5)) * 1.05)
        ax.set_ylim(-0.05, y_max)
        ax.grid(True, alpha=0.15)
        # Add a reference line at 1.0 if data exceeds it
        if float(np.max(norm_data)) > 1.0:
            ax.axhline(y=1.0, color='white', linewidth=0.3, alpha=0.3, linestyle='--')
        ax.set_title(f'{name} — CV={m["intra_cv"]:.3f}, Range={m["inter_range"]:.3f}, '
                     f'Product={m["product"]:.4f}', fontsize=9, loc='left')

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()

    safe_name = signal_name.lower().replace(' ', '_')
    plot_path = os.path.join(OUTPUT_DIR, f'{track_name}_{safe_name}_comparison.png')
    fig.savefig(plot_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {plot_path}")


def run_section_test(wav_path, track_name):
    """Run algorithms on total RMS and AbsInt, with G1 gamma sweep."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    n_fft = 2048
    hop_length = 512
    fps = sr / hop_length

    print(f"\n{'='*100}")
    print(f"  {track_name} — {duration:.1f}s, SR: {sr}, FPS: {fps:.1f}")
    print(f"{'='*100}")

    # Compute signals
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    absint = compute_absint(y, sr, n_fft, hop_length)
    n_frames = len(rms)
    times = np.arange(n_frames) / fps

    # ── Test 0: RMS full comparison with B/C window sweep ──
    rms_full_names = [
        'A: Peak-Decay',
        'B: IntRef 5s',
        'B: IntRef 10s',
        'B: IntRef 20s',
        'C: EMA 5s',
        'C: EMA 10s',
        'C: EMA 20s',
        'D: Dual-Track',
        'E: MinMax 10s',
        'F: Causal RT',
        'G1: PwrCompress',
        'G2: AddBlend',
        'G3: SectRange',
    ]
    rms_full_fns = {
        'A: Peak-Decay': norm_A_peak_decay,
        'B: IntRef 5s': lambda e, f: norm_B_integral_ref(e, f, 5.0),
        'B: IntRef 10s': lambda e, f: norm_B_integral_ref(e, f, 10.0),
        'B: IntRef 20s': lambda e, f: norm_B_integral_ref(e, f, 20.0),
        'C: EMA 5s': lambda e, f: norm_C_ema_ratio(e, f, 5.0),
        'C: EMA 10s': lambda e, f: norm_C_ema_ratio(e, f, 10.0),
        'C: EMA 20s': lambda e, f: norm_C_ema_ratio(e, f, 20.0),
        'D: Dual-Track': norm_D_dual_track,
        'E: MinMax 10s': lambda e, f: norm_E_percentile_window(e, f, 10.0),
        'F: Causal RT': norm_F_causal_rt,
        'G1: PwrCompress': lambda e, f: norm_G1_power_compress(e, f, gamma=0.3),
        'G2: AddBlend': norm_G2_additive_blend,
        'G3: SectRange': norm_G3_section_range,
    }

    run_signal_test(rms, 'RMS — Full Comparison', track_name, fps, times,
                    rms_full_names, rms_full_fns)

    # ── Test 1: G1-EMA gamma sweep + EMA window comparison ──
    g1ema_algo_names = [
        'C: EMA 20s',
        'C: EMA 30s',
        'G1ema γ=0.3',
        'G1ema γ=0.5',
        'G1ema γ=0.7',
        'G1ema γ=1.0',
        'A: Peak-Decay',
    ]
    g1ema_algo_fns = {
        'C: EMA 20s': lambda e, f: norm_C_ema_ratio(e, f, 20.0),
        'C: EMA 30s': lambda e, f: norm_C_ema_ratio(e, f, 30.0),
        'G1ema γ=0.3': lambda e, f: norm_G1_ema_hybrid(e, f, gamma=0.3),
        'G1ema γ=0.5': lambda e, f: norm_G1_ema_hybrid(e, f, gamma=0.5),
        'G1ema γ=0.7': lambda e, f: norm_G1_ema_hybrid(e, f, gamma=0.7),
        'G1ema γ=1.0': lambda e, f: norm_G1_ema_hybrid(e, f, gamma=1.0),
        'A: Peak-Decay': norm_A_peak_decay,
    }

    # Test on RMS
    run_signal_test(rms, 'RMS — G1-EMA Gamma Sweep', track_name, fps, times,
                    g1ema_algo_names, g1ema_algo_fns)

    # Test on AbsInt
    run_signal_test(absint, 'AbsInt — G1-EMA Gamma Sweep', track_name, fps, times,
                    g1ema_algo_names, g1ema_algo_fns)

    # ── Test 2: AbsInt proportional percussion (B/C window sweep) ──
    absint_algo_names = [
        'A: Peak-Decay',
        'B: IntRef 5s',
        'B: IntRef 10s',
        'B: IntRef 20s',
        'C: EMA 5s',
        'C: EMA 10s',
        'C: EMA 20s',
        'E: MinMax 10s',
        'F: Causal RT',
    ]
    absint_algo_fns = {
        'A: Peak-Decay': norm_A_peak_decay,
        'B: IntRef 5s': lambda e, f: norm_B_integral_ref(e, f, 5.0),
        'B: IntRef 10s': lambda e, f: norm_B_integral_ref(e, f, 10.0),
        'B: IntRef 20s': lambda e, f: norm_B_integral_ref(e, f, 20.0),
        'C: EMA 5s': lambda e, f: norm_C_ema_ratio(e, f, 5.0),
        'C: EMA 10s': lambda e, f: norm_C_ema_ratio(e, f, 10.0),
        'C: EMA 20s': lambda e, f: norm_C_ema_ratio(e, f, 20.0),
        'E: MinMax 10s': lambda e, f: norm_E_percentile_window(e, f, 10.0),
        'F: Causal RT': norm_F_causal_rt,
    }

    run_signal_test(absint, 'AbsInt — Proportional', track_name, fps, times,
                    absint_algo_names, absint_algo_fns)


def main():
    test_tracks = {
        'fa_br_drop1': os.path.join(SEGMENTS_DIR, 'fa_br_drop1.wav'),
        'nathaniel_black_ice': os.path.join(SEGMENTS_DIR, 'nathaniel', 'black_ice.wav'),
    }

    for tid, path in test_tracks.items():
        if not os.path.exists(path):
            print(f"  SKIP: {tid} — not found at {path}")
            continue
        run_section_test(path, tid)


if __name__ == '__main__':
    main()
