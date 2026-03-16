#!/usr/bin/env python3
"""
Normalization Comparison: Offline vs Real-Time (Streaming)

Compares offline (full-file) normalization against simulated real-time
(causal-only, peak-decay) normalization for key audio features:
  1. AbsIntegral (|d(RMS)/dt| integrated 150ms)
  2. RMS energy
  3. Band energy (5 bands: sub-bass, bass, mids, high-mids, treble)
  4. RMS energy integral (rolling 10s sum)

Normalization strategies tested:
  A. Peak-decay (current): peak = max(val, peak * decay); norm = val / peak
  B. Percentile-based: track running percentiles via reservoir/window
  C. Adaptive decay: decay rate changes based on signal variance
  D. Two-speed: fast peak (attack) + slow peak (sustain), blend

Output: plots + quantitative metrics saved to normalization-tests/
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Audio loading
import soundfile as sf

# ── Paths ──
SEGMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio-segments')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normalization-tests')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Band definitions (matching effects/band_sparkles.py) ──
BANDS = [
    ('Sub-bass', 20, 80),
    ('Bass', 80, 250),
    ('Mids', 250, 2000),
    ('High-mids', 2000, 6000),
    ('Treble', 6000, 8000),
]

BAND_COLORS = ['#FF1744', '#FF9100', '#FFEA00', '#00E676', '#00B0FF']


# ═══════════════════════════════════════════════════════════════════════
# Feature Computation (shared, streaming frame-by-frame)
# ═══════════════════════════════════════════════════════════════════════

def compute_raw_features(audio, sr, frame_len=2048, hop=512):
    """Compute raw (unnormalized) features frame-by-frame, simulating streaming.

    Returns dict of {feature_name: np.array of raw values per frame}.
    """
    n_frames = (len(audio) - frame_len) // hop + 1
    if n_frames <= 0:
        return {}

    window = np.hanning(frame_len).astype(np.float32)
    freq_bins = np.fft.rfftfreq(frame_len, 1.0 / sr)
    dt = frame_len / sr

    # AbsIntegral state
    absint_window_sec = 0.15
    absint_window_frames = max(1, int(absint_window_sec / dt))
    absint_deriv_buf = np.zeros(absint_window_frames, dtype=np.float32)
    absint_deriv_pos = 0
    prev_rms = 0.0

    # Band masks
    band_masks = []
    for _, lo, hi in BANDS:
        band_masks.append((freq_bins >= lo) & (freq_bins < hi))

    # Outputs
    rms_raw = np.zeros(n_frames, dtype=np.float64)
    absint_raw = np.zeros(n_frames, dtype=np.float64)
    band_energy_raw = np.zeros((n_frames, len(BANDS)), dtype=np.float64)
    rms_integral_raw = np.zeros(n_frames, dtype=np.float64)  # rolling 10s sum

    # Rolling RMS integral state (10s window)
    integral_window = max(1, int(10.0 * sr / hop))
    rms_ring = np.zeros(integral_window, dtype=np.float64)
    rms_ring_pos = 0
    rms_ring_filled = 0

    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len]

        # RMS
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_raw[i] = rms

        # RMS derivative -> abs-integral
        rms_deriv = (rms - prev_rms) / dt
        prev_rms = rms
        absint_deriv_buf[absint_deriv_pos % absint_window_frames] = abs(rms_deriv)
        absint_deriv_pos += 1
        absint_raw[i] = float(np.sum(absint_deriv_buf) * dt)

        # Band energy from FFT
        spec = np.abs(np.fft.rfft(frame * window))
        for b, mask in enumerate(band_masks):
            band_energy_raw[i, b] = float(np.sum(spec[mask] ** 2))

        # Rolling RMS integral
        rms_ring[rms_ring_pos % integral_window] = rms
        rms_ring_pos += 1
        rms_ring_filled = min(rms_ring_filled + 1, integral_window)
        rms_integral_raw[i] = float(np.sum(rms_ring[:rms_ring_filled]))

    return {
        'rms': rms_raw,
        'absint': absint_raw,
        'band_energy': band_energy_raw,
        'rms_integral': rms_integral_raw,
        'n_frames': n_frames,
        'sr': sr,
        'hop': hop,
        'frame_len': frame_len,
    }


# ═══════════════════════════════════════════════════════════════════════
# Normalization Strategies
# ═══════════════════════════════════════════════════════════════════════

def normalize_offline_global_max(raw):
    """Offline: divide by global max. The gold standard."""
    mx = np.max(raw)
    if mx < 1e-10:
        return np.zeros_like(raw)
    return raw / mx


def normalize_offline_percentile(raw, lo=2, hi=98):
    """Offline: map [percentile_lo, percentile_hi] to [0, 1]."""
    p_lo = np.percentile(raw, lo)
    p_hi = np.percentile(raw, hi)
    if p_hi - p_lo < 1e-10:
        return np.zeros_like(raw)
    return np.clip((raw - p_lo) / (p_hi - p_lo), 0, 1)


def normalize_peak_decay(raw, decay):
    """Streaming: peak = max(val, peak * decay); norm = val / peak."""
    out = np.zeros_like(raw)
    peak = 1e-10
    for i in range(len(raw)):
        peak = max(float(raw[i]), peak * decay)
        out[i] = raw[i] / peak if peak > 1e-10 else 0.0
    return out


def normalize_percentile_streaming(raw, window_size=1000):
    """Streaming: running percentile via sliding window.

    Track 2nd and 98th percentile over a sliding window, map to [0,1].
    Uses numpy's percentile on window (exact, but O(window_size) per frame).
    """
    out = np.zeros_like(raw)
    for i in range(len(raw)):
        start = max(0, i - window_size + 1)
        window = raw[start:i + 1]
        p2 = np.percentile(window, 2)
        p98 = np.percentile(window, 98)
        if p98 - p2 > 1e-10:
            out[i] = np.clip((raw[i] - p2) / (p98 - p2), 0, 1)
        else:
            out[i] = 0.0
    return out


def normalize_adaptive_decay(raw, base_decay=0.998, fast_decay=0.99,
                              variance_window=100):
    """Streaming: decay rate adapts based on signal variance.

    When variance is high (dynamic section), use faster decay.
    When variance is low (steady section), use slower decay for stability.
    """
    out = np.zeros_like(raw)
    peak = 1e-10
    for i in range(len(raw)):
        # Compute local variance
        start = max(0, i - variance_window + 1)
        local_var = np.var(raw[start:i + 1]) if i > 0 else 0.0

        # Map variance to decay: higher variance -> faster decay
        # Normalize variance by running estimate of max variance
        if i == 0:
            max_var = local_var + 1e-10
        else:
            max_var = max(max_var * 0.9999, local_var)

        var_ratio = local_var / max_var if max_var > 1e-10 else 0.0
        decay = base_decay + (fast_decay - base_decay) * var_ratio
        # decay ranges from base_decay (slow, low var) to fast_decay (fast, high var)

        peak = max(float(raw[i]), peak * decay)
        out[i] = raw[i] / peak if peak > 1e-10 else 0.0
    return out


def normalize_two_speed(raw, fast_decay=0.99, slow_decay=0.9998, blend=0.5):
    """Streaming: two parallel peak-decay trackers, blended.

    Fast tracker: responds quickly, good for transients
    Slow tracker: stable envelope, good for sustained sections
    Output: blend between the two normalized values.
    """
    out = np.zeros_like(raw)
    fast_peak = 1e-10
    slow_peak = 1e-10
    for i in range(len(raw)):
        val = float(raw[i])
        fast_peak = max(val, fast_peak * fast_decay)
        slow_peak = max(val, slow_peak * slow_decay)

        fast_norm = val / fast_peak if fast_peak > 1e-10 else 0.0
        slow_norm = val / slow_peak if slow_peak > 1e-10 else 0.0

        out[i] = blend * fast_norm + (1 - blend) * slow_norm
    return out


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(reference, candidate):
    """Compute comparison metrics between offline reference and streaming candidate."""
    # Skip initial frames (peak-decay needs warmup)
    skip = min(100, len(reference) // 10)
    ref = reference[skip:]
    cand = candidate[skip:]

    if len(ref) == 0 or np.std(ref) < 1e-10:
        return {'correlation': 0.0, 'mse': 0.0, 'mae': 0.0,
                'p90_error': 0.0, 'max_error': 0.0}

    corr = float(np.corrcoef(ref, cand)[0, 1]) if np.std(cand) > 1e-10 else 0.0
    mse = float(np.mean((ref - cand) ** 2))
    mae = float(np.mean(np.abs(ref - cand)))
    errors = np.abs(ref - cand)
    p90 = float(np.percentile(errors, 90))
    mx = float(np.max(errors))

    return {
        'correlation': corr,
        'mse': mse,
        'mae': mae,
        'p90_error': p90,
        'max_error': mx,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Comparison
# ═══════════════════════════════════════════════════════════════════════

def run_comparison(track_id, wav_path):
    """Run full normalization comparison for one track."""
    print(f"\n{'='*60}")
    print(f"Track: {track_id}")
    print(f"File:  {wav_path}")
    print(f"{'='*60}")

    # Load audio
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s, SR: {sr}")

    # Compute raw features
    feats = compute_raw_features(audio, sr)
    n_frames = feats['n_frames']
    times = np.arange(n_frames) * feats['hop'] / sr
    print(f"Frames: {n_frames}")

    # ── Define normalization strategies ──
    strategies = {
        'peak_decay_0.998': lambda r: normalize_peak_decay(r, 0.998),
        'peak_decay_0.9995': lambda r: normalize_peak_decay(r, 0.9995),
        'peak_decay_0.9999': lambda r: normalize_peak_decay(r, 0.9999),
        'adaptive_decay': lambda r: normalize_adaptive_decay(r),
        'two_speed': lambda r: normalize_two_speed(r),
        'percentile_5s': lambda r: normalize_percentile_streaming(r, window_size=int(5 * sr / feats['hop'])),
        'percentile_30s': lambda r: normalize_percentile_streaming(r, window_size=int(30 * sr / feats['hop'])),
    }

    # ── Features to test ──
    scalar_features = {
        'absint': feats['absint'],
        'rms': feats['rms'],
        'rms_integral': feats['rms_integral'],
    }

    # ── Compute all normalizations ──
    all_results = {}
    all_metrics = {}

    for feat_name, raw in scalar_features.items():
        offline_ref = normalize_offline_global_max(raw)
        all_results[feat_name] = {'offline': offline_ref}
        all_metrics[feat_name] = {}

        for strat_name, strat_fn in strategies.items():
            normalized = strat_fn(raw)
            all_results[feat_name][strat_name] = normalized
            metrics = compute_metrics(offline_ref, normalized)
            all_metrics[feat_name][strat_name] = metrics

    # Band energy: normalize each band independently
    for b, (band_name, _, _) in enumerate(BANDS):
        feat_key = f'band_{band_name.lower().replace("-", "_")}'
        raw = feats['band_energy'][:, b]
        offline_ref = normalize_offline_global_max(raw)
        all_results[feat_key] = {'offline': offline_ref}
        all_metrics[feat_key] = {}

        for strat_name, strat_fn in strategies.items():
            normalized = strat_fn(raw)
            all_results[feat_key][strat_name] = normalized
            metrics = compute_metrics(offline_ref, normalized)
            all_metrics[feat_key][strat_name] = metrics

    # ── Print metrics table ──
    print(f"\n{'Feature':<20} {'Strategy':<25} {'Corr':>6} {'MSE':>8} {'MAE':>6} {'P90':>6} {'Max':>6}")
    print('-' * 80)
    for feat_name in sorted(all_metrics.keys()):
        for strat_name, m in sorted(all_metrics[feat_name].items()):
            print(f"{feat_name:<20} {strat_name:<25} {m['correlation']:>6.3f} "
                  f"{m['mse']:>8.5f} {m['mae']:>6.3f} {m['p90_error']:>6.3f} {m['max_error']:>6.3f}")

    # ── Generate plots ──
    # Plot 1: Scalar features comparison
    fig, axes = plt.subplots(len(scalar_features), 1, figsize=(18, 4 * len(scalar_features)),
                             sharex=True)
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#1a1a2e')

    if len(scalar_features) == 1:
        axes = [axes]

    for ax, (feat_name, raw) in zip(axes, scalar_features.items()):
        offline = all_results[feat_name]['offline']
        ax.plot(times, offline, color='white', linewidth=1.5, alpha=0.8, label='Offline (global max)')

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        for idx, (strat_name, _) in enumerate(strategies.items()):
            strat_data = all_results[feat_name][strat_name]
            c = colors[idx % len(colors)]
            m = all_metrics[feat_name][strat_name]
            ax.plot(times, strat_data, color=c, linewidth=0.8, alpha=0.6,
                    label=f'{strat_name} (r={m["correlation"]:.3f}, MAE={m["mae"]:.3f})')

        ax.set_ylabel(feat_name)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.15)
        ax.set_title(f'{feat_name} — normalization comparison', fontsize=11)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{track_id} — Scalar Feature Normalization', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'{track_id}_scalar_comparison.png')
    fig.savefig(plot_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {plot_path}")

    # Plot 2: Band energy comparison (top 3 strategies only)
    top_strategies = ['peak_decay_0.998', 'peak_decay_0.9995', 'two_speed']
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(18, 3 * len(BANDS)), sharex=True)
    fig.patch.set_facecolor('#1a1a2e')

    for ax, (b, (band_name, _, _)) in zip(axes, enumerate(BANDS)):
        feat_key = f'band_{band_name.lower().replace("-", "_")}'
        offline = all_results[feat_key]['offline']
        ax.plot(times, offline, color='white', linewidth=1.5, alpha=0.8, label='Offline')

        for idx, strat_name in enumerate(top_strategies):
            if strat_name in all_results[feat_key]:
                strat_data = all_results[feat_key][strat_name]
                m = all_metrics[feat_key][strat_name]
                c = ['#FF6B6B', '#4ECDC4', '#45B7D1'][idx]
                ax.plot(times, strat_data, color=c, linewidth=0.8, alpha=0.6,
                        label=f'{strat_name} (r={m["correlation"]:.3f})')

        ax.set_ylabel(band_name)
        ax.set_ylim(-0.05, 1.15)
        ax.set_facecolor('#1a1a2e')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{track_id} — Band Energy Normalization', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'{track_id}_band_comparison.png')
    fig.savefig(plot_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # Plot 3: Error heatmap (features x strategies)
    feat_names = sorted(all_metrics.keys())
    strat_names = sorted(next(iter(all_metrics.values())).keys())
    corr_matrix = np.zeros((len(feat_names), len(strat_names)))
    mae_matrix = np.zeros((len(feat_names), len(strat_names)))

    for i, fn in enumerate(feat_names):
        for j, sn in enumerate(strat_names):
            corr_matrix[i, j] = all_metrics[fn][sn]['correlation']
            mae_matrix[i, j] = all_metrics[fn][sn]['mae']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(feat_names) * 0.6)))
    fig.patch.set_facecolor('#1a1a2e')

    im1 = ax1.imshow(corr_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xticks(range(len(strat_names)))
    ax1.set_xticklabels(strat_names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(len(feat_names)))
    ax1.set_yticklabels(feat_names, fontsize=8)
    ax1.set_title('Correlation with Offline', fontsize=11)
    for i in range(len(feat_names)):
        for j in range(len(strat_names)):
            ax1.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                     color='black' if corr_matrix[i,j] > 0.5 else 'white')
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(mae_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.3)
    ax2.set_xticks(range(len(strat_names)))
    ax2.set_xticklabels(strat_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(len(feat_names)))
    ax2.set_yticklabels(feat_names, fontsize=8)
    ax2.set_title('MAE from Offline', fontsize=11)
    for i in range(len(feat_names)):
        for j in range(len(strat_names)):
            ax2.text(j, i, f'{mae_matrix[i,j]:.3f}', ha='center', va='center', fontsize=7,
                     color='black' if mae_matrix[i,j] < 0.15 else 'white')
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(f'{track_id} — Normalization Quality Heatmap', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'{track_id}_heatmap.png')
    fig.savefig(plot_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {plot_path}")

    return all_metrics


def main():
    # Test tracks
    test_tracks = {
        'fa_br_drop1': os.path.join(SEGMENTS_DIR, 'fa_br_drop1.wav'),
        'fourtet_lostvillage_screech': os.path.join(SEGMENTS_DIR, 'fourtet_lostvillage_screech.wav'),
        'complex_beat_glue': os.path.join(SEGMENTS_DIR, 'complex_beat_glue.wav'),
    }

    # Verify files exist
    available = {}
    for tid, path in test_tracks.items():
        if os.path.exists(path):
            available[tid] = path
            print(f"  Found: {tid}")
        else:
            print(f"  Missing: {tid} ({path})")

    if not available:
        print("No test tracks found!")
        return

    # Run comparisons
    all_track_metrics = {}
    for tid, path in available.items():
        metrics = run_comparison(tid, path)
        all_track_metrics[tid] = metrics

    # ── Cross-track summary ──
    print(f"\n{'='*60}")
    print("CROSS-TRACK SUMMARY")
    print(f"{'='*60}")

    # Compute mean correlation per strategy across all features and tracks
    strat_names = sorted(next(iter(next(iter(all_track_metrics.values())).values())).keys())
    print(f"\n{'Strategy':<25} {'Mean Corr':>10} {'Mean MAE':>10} {'Worst P90':>10}")
    print('-' * 60)

    for sn in strat_names:
        corrs = []
        maes = []
        p90s = []
        for tid, feat_metrics in all_track_metrics.items():
            for fn, strat_metrics in feat_metrics.items():
                if sn in strat_metrics:
                    corrs.append(strat_metrics[sn]['correlation'])
                    maes.append(strat_metrics[sn]['mae'])
                    p90s.append(strat_metrics[sn]['p90_error'])
        print(f"{sn:<25} {np.mean(corrs):>10.4f} {np.mean(maes):>10.4f} {np.max(p90s):>10.4f}")

    # Save metrics JSON
    # Convert numpy types to Python types for JSON serialization
    def to_python(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_metrics = {}
    for tid, feat_metrics in all_track_metrics.items():
        json_metrics[tid] = {}
        for fn, strat_metrics in feat_metrics.items():
            json_metrics[tid][fn] = {}
            for sn, m in strat_metrics.items():
                json_metrics[tid][fn][sn] = {k: to_python(v) for k, v in m.items()}

    json_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"\nSaved metrics: {json_path}")


if __name__ == '__main__':
    main()
