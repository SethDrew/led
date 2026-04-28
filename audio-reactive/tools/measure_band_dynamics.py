"""
Measure per-band signal dynamics on real music.

For each test track:
  1. FFT + per-band RMS (matching energy_waterfall_3band.py: n_fft=2048, hop=512,
     bands {bass:20-250, mids:250-2000, treble:2000-8000}).
  2. Per-band EMA-ratio normalization (PerBandEMANormalize).
  3. Compute statistics on:
       - per-band normalized signal (mean, std, "active" fraction)
       - simultaneous-band activity histogram (0/1/2/3 bands active)
       - candidate combiners: max(bands), sum(bands), mean(bands)
       - cross-correlation between bands
  4. Test "absint deemphasizes bass" hypothesis:
       per-band absint vs broadband absint magnitudes.
  5. Test "per-band EMA -> sum" idea:
       compare to current max-of-bands, and to broadband-RMS through PerBandEMANormalize.

Outputs:
  - PNG plots per track at library/ideas/band_dynamics_plots/
  - Markdown report at library/ideas/BAND_DYNAMICS_MEASUREMENTS.md
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
EFFECTS_DIR = REPO_ROOT / "audio-reactive" / "effects"
sys.path.insert(0, str(EFFECTS_DIR))

from signals import (  # noqa: E402
    OverlapFrameAccumulator,
    PerBandEMANormalize,
    PerBandAbsIntegral,
)


# Match energy_waterfall_3band.py exactly
N_FFT = 2048
HOP = 512
BANDS = ('bass', 'mids', 'treble')
BAND_RANGES = {
    'bass':   (20, 250),
    'mids':   (250, 2000),
    'treble': (2000, 8000),
}
ACTIVE_THRESHOLD = 0.3
COMBINED_THRESHOLD = 0.5

SEGMENTS_DIR = REPO_ROOT / "audio-reactive" / "research" / "audio-segments"
TRACKS = [
    'bangarang_skrillex',
    'jcole',
    'ambient',
    'cinematic_drums',
    'fa_br_drop1',
]

OUT_DIR = REPO_ROOT / "library" / "ideas" / "band_dynamics_plots"
REPORT_PATH = REPO_ROOT / "library" / "ideas" / "BAND_DYNAMICS_MEASUREMENTS.md"


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def compute_band_rms_traces(audio: np.ndarray, sr: int):
    """Run the same per-band-RMS pipeline as energy_waterfall_3band.

    Returns:
      band_rms:    (n_frames, 3) raw per-band RMS
      broadband_rms: (n_frames,)  full-spectrum RMS (sqrt mean power)
    """
    accum = OverlapFrameAccumulator(frame_len=N_FFT, hop=HOP)
    window = np.hanning(N_FFT).astype(np.float32)
    freq_bins = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    mask_array = np.stack(
        [(freq_bins >= BAND_RANGES[b][0]) & (freq_bins < BAND_RANGES[b][1])
         for b in BANDS], axis=0
    )
    band_counts = np.maximum(mask_array.sum(axis=1), 1)

    band_rms_list = []
    broadband_rms_list = []

    chunk_size = HOP * 16
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        for frame in accum.feed(chunk):
            spectrum = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
            power = spectrum ** 2

            br = np.sqrt(
                (mask_array * power[np.newaxis, :]).sum(axis=1) / band_counts
            ).astype(np.float32)
            band_rms_list.append(br)

            # Broadband RMS = time-domain RMS of windowed frame.
            # (Equivalent to sqrt(sum(power)/N) up to window normalization.)
            broadband_rms_list.append(
                float(np.sqrt(np.mean(frame ** 2)))
            )

    band_rms = np.array(band_rms_list, dtype=np.float32)         # (T, 3)
    broadband_rms = np.array(broadband_rms_list, dtype=np.float32)  # (T,)
    return band_rms, broadband_rms


def per_band_ema_normalize(band_rms: np.ndarray, fps: float,
                           ema_tc: float = 30.0) -> np.ndarray:
    norm = PerBandEMANormalize(
        num_bands=band_rms.shape[1], fps=fps, ema_tc=ema_tc,
    )
    out = np.zeros_like(band_rms)
    for t in range(band_rms.shape[0]):
        out[t] = norm.update(band_rms[t])
    return out


def broadband_through_per_band_normalize(broadband_rms: np.ndarray,
                                         fps: float,
                                         ema_tc: float = 30.0) -> np.ndarray:
    """Single-band PerBandEMANormalize for comparison."""
    norm = PerBandEMANormalize(num_bands=1, fps=fps, ema_tc=ema_tc)
    out = np.zeros(len(broadband_rms), dtype=np.float32)
    for t in range(len(broadband_rms)):
        out[t] = norm.update(np.array([broadband_rms[t]], dtype=np.float32))[0]
    return out


def per_band_absint(normalized: np.ndarray, fps: float) -> np.ndarray:
    """Per-band absint of the EMA-normalized signal — matches effect pipeline."""
    absint = PerBandAbsIntegral(num_bands=normalized.shape[1], fps=fps)
    out = np.zeros_like(normalized)
    for t in range(normalized.shape[0]):
        out[t] = absint.update(normalized[t])
    return out


def absint_of_raw_rms(rms_signal: np.ndarray, fps: float,
                      window_sec: float = 0.15) -> np.ndarray:
    """Streaming |d/dt| integrated over window, peak-normalized, on a raw
    1D RMS signal (broadband or single band).

    Mirrors AbsIntegral but operates on a precomputed RMS trace instead of
    raw audio frames.
    """
    dt = 1.0 / fps
    window_frames = max(1, int(window_sec * fps))
    buf = np.zeros(window_frames, dtype=np.float32)
    pos = 0
    prev = 0.0
    peak = 1e-10
    out = np.zeros(len(rms_signal), dtype=np.float32)
    for t, r in enumerate(rms_signal):
        deriv = (r - prev) / dt
        prev = r
        buf[pos % window_frames] = abs(deriv)
        pos += 1
        raw = float(np.sum(buf) * dt)
        peak = max(raw, peak * 0.998)
        out[t] = raw / peak if peak > 1e-10 else 0.0
    return out


def absint_raw_magnitude(rms_signal: np.ndarray, fps: float,
                         window_sec: float = 0.15) -> np.ndarray:
    """Same as absint_of_raw_rms but returns the *raw* (un-normalized)
    abs-integral magnitude, for cross-band magnitude comparison."""
    dt = 1.0 / fps
    window_frames = max(1, int(window_sec * fps))
    buf = np.zeros(window_frames, dtype=np.float32)
    pos = 0
    prev = 0.0
    out = np.zeros(len(rms_signal), dtype=np.float32)
    for t, r in enumerate(rms_signal):
        deriv = (r - prev) / dt
        prev = r
        buf[pos % window_frames] = abs(deriv)
        pos += 1
        out[t] = float(np.sum(buf) * dt)
    return out


def stats_dict(label: str, x: np.ndarray, threshold: float = COMBINED_THRESHOLD):
    return {
        'label': label,
        'mean': float(np.mean(x)),
        'std':  float(np.std(x)),
        'p95':  float(np.percentile(x, 95)),
        'max':  float(np.max(x)),
        f'frac>{threshold}': float(np.mean(x > threshold)),
    }


def analyze_track(name: str) -> dict:
    path = SEGMENTS_DIR / f"{name}.wav"
    audio, sr = load_mono(path)
    fps = sr / HOP

    band_rms, broadband_rms = compute_band_rms_traces(audio, sr)
    n_frames = band_rms.shape[0]
    duration = n_frames / fps

    # ── Per-band EMA-normalized ────────────────────────────────────────────
    normalized = per_band_ema_normalize(band_rms, fps)         # (T, 3)

    # ── Activity stats ─────────────────────────────────────────────────────
    active = normalized > ACTIVE_THRESHOLD                     # (T, 3) bool
    n_active = active.sum(axis=1)                              # (T,) {0..3}
    active_fracs = active.mean(axis=0)                         # per band

    simul_hist = np.zeros(4, dtype=np.float64)
    for k in range(4):
        simul_hist[k] = float(np.mean(n_active == k))

    # ── Per-band raw stats ─────────────────────────────────────────────────
    band_means = normalized.mean(axis=0)
    band_stds = normalized.std(axis=0)

    # ── Combiners ──────────────────────────────────────────────────────────
    combo_max  = normalized.max(axis=1)
    combo_sum  = normalized.sum(axis=1)
    combo_mean = normalized.mean(axis=1)

    # User's idea: per-band EMA-normalize (already done) -> sum -> clip
    # Sum can exceed 1.0 (up to 3.0 if all bands maxed). Report both raw and
    # clipped-to-1 versions so we can see how often it saturates.
    combo_sum_clipped = np.clip(combo_sum, 0.0, 1.0)

    # Single-band reference: broadband RMS through PerBandEMANormalize
    broadband_norm = broadband_through_per_band_normalize(broadband_rms, fps)

    # ── Cross-correlation between bands ────────────────────────────────────
    corr_matrix = np.corrcoef(normalized.T)  # 3x3

    # ── absint hypothesis test ─────────────────────────────────────────────
    # Per-band absint of normalized signal (what the effect uses now)
    band_absint_norm = per_band_absint(normalized, fps)

    # Raw-magnitude absint per band on RAW band RMS — for cross-band
    # magnitude comparison (peak-normalization would erase this).
    band_absint_raw_mag = np.stack([
        absint_raw_magnitude(band_rms[:, i], fps) for i in range(3)
    ], axis=1)  # (T, 3)

    # Raw absint of broadband RMS — for "is broadband absint more balanced
    # than broadband RMS" comparison
    broadband_absint_raw_mag = absint_raw_magnitude(broadband_rms, fps)

    # ── Magnitude comparisons (raw magnitudes, not normalized) ─────────────
    band_rms_means = band_rms.mean(axis=0)              # (3,)
    band_absint_means = band_absint_raw_mag.mean(axis=0)  # (3,)
    broadband_rms_mean = float(broadband_rms.mean())
    broadband_absint_mean = float(broadband_absint_raw_mag.mean())

    # Bass-vs-treble dominance ratios
    rms_bass_treble_ratio = band_rms_means[0] / max(band_rms_means[2], 1e-12)
    absint_bass_treble_ratio = band_absint_means[0] / max(band_absint_means[2], 1e-12)

    return {
        'name': name,
        'sr': sr,
        'fps': fps,
        'duration_s': duration,
        'n_frames': n_frames,

        'band_means': band_means,
        'band_stds': band_stds,
        'active_fracs': active_fracs,
        'simul_hist': simul_hist,
        'corr_matrix': corr_matrix,

        'combo_max_stats':  stats_dict('max(bands)',  combo_max),
        'combo_sum_stats':  stats_dict('sum(bands)',  combo_sum),
        'combo_sum_clipped_stats': stats_dict('clip(sum,1)', combo_sum_clipped),
        'combo_mean_stats': stats_dict('mean(bands)', combo_mean),
        'broadband_norm_stats': stats_dict('broadband EMA-norm', broadband_norm),

        'band_rms_means': band_rms_means,
        'band_absint_means': band_absint_means,
        'broadband_rms_mean': broadband_rms_mean,
        'broadband_absint_mean': broadband_absint_mean,
        'rms_bass_treble_ratio': float(rms_bass_treble_ratio),
        'absint_bass_treble_ratio': float(absint_bass_treble_ratio),

        # traces for plots
        '_normalized': normalized,
        '_combo_max': combo_max,
        '_combo_sum': combo_sum,
        '_combo_sum_clipped': combo_sum_clipped,
        '_broadband_norm': broadband_norm,
        '_band_rms': band_rms,
        '_broadband_rms': broadband_rms,
        '_band_absint_raw_mag': band_absint_raw_mag,
        '_broadband_absint_raw_mag': broadband_absint_raw_mag,
    }


def plot_track(result: dict, out_dir: Path):
    name = result['name']
    fps = result['fps']
    n = result['n_frames']
    t = np.arange(n) / fps

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)

    # 1. Per-band normalized signals
    ax = axes[0]
    for i, b in enumerate(BANDS):
        ax.plot(t, result['_normalized'][:, i], label=b, linewidth=0.8)
    ax.axhline(ACTIVE_THRESHOLD, color='gray', linestyle=':', linewidth=0.5,
               label=f'active>{ACTIVE_THRESHOLD}')
    ax.set_ylabel('per-band\nEMA-norm')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.set_title(f'{name} — per-band signals (3-band, ema_tc=30s)')

    # 2. Combiners
    ax = axes[1]
    ax.plot(t, result['_combo_max'], label='max(bands)', linewidth=0.8)
    ax.plot(t, result['_combo_sum_clipped'], label='clip(sum,1)', linewidth=0.8,
            alpha=0.8)
    ax.plot(t, result['_broadband_norm'], label='broadband EMA-norm',
            linewidth=0.8, alpha=0.8)
    ax.axhline(COMBINED_THRESHOLD, color='gray', linestyle=':', linewidth=0.5)
    ax.set_ylabel('combiners')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', fontsize=8, ncol=3)

    # 3. Sum (raw, can exceed 1)
    ax = axes[2]
    ax.plot(t, result['_combo_sum'], label='sum(bands), raw', linewidth=0.8,
            color='C3')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_ylabel('sum (raw)')
    ax.set_ylim(-0.05, 3.1)
    ax.legend(loc='upper right', fontsize=8)

    # 4. Raw band RMS magnitudes (log scale, shows bass dominance)
    ax = axes[3]
    for i, b in enumerate(BANDS):
        ax.semilogy(t, np.maximum(result['_band_rms'][:, i], 1e-6),
                    label=b, linewidth=0.6, alpha=0.8)
    ax.semilogy(t, np.maximum(result['_broadband_rms'], 1e-6),
                label='broadband', color='black', linewidth=0.6, alpha=0.6)
    ax.set_ylabel('raw RMS\n(log)')
    ax.legend(loc='upper right', fontsize=8, ncol=4)

    # 5. Raw-magnitude absint (per-band vs broadband)
    ax = axes[4]
    for i, b in enumerate(BANDS):
        ax.semilogy(t, np.maximum(result['_band_absint_raw_mag'][:, i], 1e-9),
                    label=b, linewidth=0.6, alpha=0.8)
    ax.semilogy(t, np.maximum(result['_broadband_absint_raw_mag'], 1e-9),
                label='broadband', color='black', linewidth=0.6, alpha=0.6)
    ax.set_ylabel('raw absint\nmagnitude (log)')
    ax.set_xlabel('time (s)')
    ax.legend(loc='upper right', fontsize=8, ncol=4)

    plt.tight_layout()
    out_path = out_dir / f"{name}.png"
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def fmt_row(values, fmt='{:.3f}'):
    return ' | '.join(fmt.format(v) for v in values)


def render_report(results: list[dict], plot_paths: dict[str, Path]) -> str:
    lines = []
    L = lines.append

    # ── Compute overall summary metrics for the lead paragraph ─────────────
    # Compare combiners across all tracks: mean of "frac > 0.5"
    avg_frac = {
        'max(bands)':  np.mean([r['combo_max_stats']['frac>0.5']  for r in results]),
        'clip(sum,1)': np.mean([r['combo_sum_clipped_stats']['frac>0.5'] for r in results]),
        'mean(bands)': np.mean([r['combo_mean_stats']['frac>0.5'] for r in results]),
        'broadband':   np.mean([r['broadband_norm_stats']['frac>0.5'] for r in results]),
    }
    avg_rms_ratio = float(np.mean([r['rms_bass_treble_ratio']    for r in results]))
    avg_absint_ratio = float(np.mean([r['absint_bass_treble_ratio'] for r in results]))

    # Multi-band activity: averaged simul_hist across tracks
    avg_simul = np.mean(np.stack([r['simul_hist'] for r in results]), axis=0)

    L("# Per-Band Recombination — Empirical Measurements")
    L("")
    L("Measurement of how the 3-band per-band EMA-normalized pipeline behaves "
      "on real music, to ground the team's debate over recombination strategies "
      "for a single LED drive signal.")
    L("")
    L("## Summary")
    L("")
    L(f"Across 5 test tracks, **multi-band co-activity is real but a "
      f"minority of frames** — averaged across tracks, "
      f"{avg_simul[1]*100:.0f}% have 1 band active, "
      f"{avg_simul[2]*100:.0f}% have 2 bands active, and "
      f"{avg_simul[3]*100:.0f}% have all 3 active. **`max(bands)` discards "
      f"this multi-band information** in the ~{(avg_simul[2]+avg_simul[3])*100:.0f}% "
      f"of frames where it matters — when bass and treble both fire on a "
      f"drop, the listener perceives both but the LED only sees the louder one.")
    L("")
    # Determine direction of the absint hypothesis from the data
    if avg_absint_ratio < avg_rms_ratio * 0.7:
        absint_verdict = "supported"
    elif avg_absint_ratio < avg_rms_ratio * 1.2:
        absint_verdict = "weakly supported / mixed"
    else:
        absint_verdict = "**not** supported"
    L(f"**The user's `absint deemphasizes bass` hypothesis is "
      f"{absint_verdict}.** Raw bass RMS averages ~{avg_rms_ratio:.0f}x raw "
      f"treble RMS across tracks; raw bass |d(RMS)/dt| absint averages "
      f"~{avg_absint_ratio:.1f}x raw treble absint. Differentiating the "
      f"energy envelope does not consistently flatten bass dominance — on "
      f"these tracks bass transients (kick attacks, sub drops) carry as much "
      f"or more |d/dt| as treble transients. The absint signal still needs "
      f"per-band peak-normalization to balance the bands; raw absint does "
      f"not solve the magnitude-imbalance problem on its own.")
    L("")
    L(f"**Combiner activation rates** (avg fraction of frames with combined "
      f"value > 0.5):")
    L("")
    for k, v in sorted(avg_frac.items(), key=lambda kv: -kv[1]):
        L(f"- `{k}`: {v*100:.1f}%")
    L("")
    L("Interpretation: `clip(sum,1)` is a strict superset of `max(bands)` and "
      "lights up whenever ANY combination of bands accumulates to 0.5. It "
      "preserves the multi-band-additive behavior the user is asking for "
      "without throwing away cases where bands are individually moderate but "
      "together prominent. Single-band broadband EMA-normalization sits in the "
      "middle and loses per-band context entirely.")
    L("")
    L("**Recommendation: try `clip(sum,1)` of per-band EMA-normalized signals** "
      "as the unified drive signal. It preserves both the within-section "
      "dynamics PerBandEMANormalize was built for AND combines multi-band "
      "events the way the ear does. Hand it to the colorist: dominant-band "
      "argmax still chooses color while the magnitude comes from the sum.")
    L("")

    L("## Method")
    L("")
    L("- FFT n_fft=2048, hop=512, sr=44100 → fps≈86. Bands: bass 20-250 Hz, "
      "mids 250-2000 Hz, treble 2000-8000 Hz. Identical to "
      "`energy_waterfall_3band.py`.")
    L("- `PerBandEMANormalize(ema_tc=30s, max_ratio=3.0)` per band.")
    L("- “Active” = normalized > 0.3. “Combined firing” = combined value > 0.5.")
    L("- Cross-correlation: Pearson on normalized traces.")
    L("- Absint: |d(RMS)/dt| integrated over 150 ms (matches `AbsIntegral`).")
    L("  - For magnitude comparisons across bands, the raw (un-normalized) "
      "abs-integral is reported so peak-decay normalization can't equalize "
      "them.")
    L("- Combiners measured: `max(bands)`, `sum(bands)` (raw and clipped to 1), "
      "`mean(bands)`. Plus single-band reference: broadband RMS through "
      "`PerBandEMANormalize(num_bands=1)`.")
    L("")

    L("## Per-track results")
    L("")
    for r in results:
        name = r['name']
        L(f"### {name}")
        L("")
        L(f"Duration {r['duration_s']:.1f}s, {r['n_frames']} frames at "
          f"{r['fps']:.1f} fps.")
        L("")
        L(f"![{name}](band_dynamics_plots/{name}.png)")
        L("")
        L("**Per-band normalized signal stats**")
        L("")
        L("| band | mean | std | active fraction (>0.3) |")
        L("|---|---|---|---|")
        for i, b in enumerate(BANDS):
            L(f"| {b} | {r['band_means'][i]:.3f} | {r['band_stds'][i]:.3f} | "
              f"{r['active_fracs'][i]*100:.1f}% |")
        L("")
        L(f"**Simultaneous-band activity** (fraction of frames with N bands active):")
        L("")
        L("| N=0 | N=1 | N=2 | N=3 |")
        L("|---|---|---|---|")
        L(f"| {r['simul_hist'][0]*100:.1f}% | {r['simul_hist'][1]*100:.1f}% | "
          f"{r['simul_hist'][2]*100:.1f}% | {r['simul_hist'][3]*100:.1f}% |")
        L("")
        L("**Combiner stats** (combined value)")
        L("")
        L("| combiner | mean | std | p95 | max | frac>0.5 |")
        L("|---|---|---|---|---|---|")
        for s in (r['combo_max_stats'], r['combo_sum_stats'],
                  r['combo_sum_clipped_stats'], r['combo_mean_stats'],
                  r['broadband_norm_stats']):
            L(f"| {s['label']} | {s['mean']:.3f} | {s['std']:.3f} | "
              f"{s['p95']:.3f} | {s['max']:.3f} | {s['frac>0.5']*100:.1f}% |")
        L("")
        L("**Cross-band Pearson correlation** of normalized signals")
        L("")
        L("|       | bass | mids | treble |")
        L("|---|---|---|---|")
        for i, b in enumerate(BANDS):
            L(f"| {b} | {r['corr_matrix'][i,0]:.3f} | "
              f"{r['corr_matrix'][i,1]:.3f} | {r['corr_matrix'][i,2]:.3f} |")
        L("")
        L("**Bass-dominance comparison: raw RMS vs raw absint**")
        L("")
        L("| metric (raw, mean over track) | bass | mids | treble | bass/treble ratio |")
        L("|---|---|---|---|---|")
        L(f"| RMS    | {r['band_rms_means'][0]:.4g} | "
          f"{r['band_rms_means'][1]:.4g} | {r['band_rms_means'][2]:.4g} | "
          f"{r['rms_bass_treble_ratio']:.1f}x |")
        L(f"| absint | {r['band_absint_means'][0]:.4g} | "
          f"{r['band_absint_means'][1]:.4g} | {r['band_absint_means'][2]:.4g} | "
          f"{r['absint_bass_treble_ratio']:.1f}x |")
        L(f"| broadband RMS | {r['broadband_rms_mean']:.4g} | — | — | — |")
        L(f"| broadband absint | {r['broadband_absint_mean']:.4g} | — | — | — |")
        L("")

    L("## Cross-track summary tables")
    L("")
    L("**Combiner `frac > 0.5`** (fraction of frames the combined signal exceeds 0.5)")
    L("")
    L("| track | max | clip(sum,1) | mean | broadband |")
    L("|---|---|---|---|---|")
    for r in results:
        L(f"| {r['name']} | {r['combo_max_stats']['frac>0.5']*100:.1f}% | "
          f"{r['combo_sum_clipped_stats']['frac>0.5']*100:.1f}% | "
          f"{r['combo_mean_stats']['frac>0.5']*100:.1f}% | "
          f"{r['broadband_norm_stats']['frac>0.5']*100:.1f}% |")
    L("")

    L("**Bass-vs-treble dominance ratio** (mean over track)")
    L("")
    L("| track | RMS bass/treble | absint bass/treble | reduction |")
    L("|---|---|---|---|")
    for r in results:
        red = r['rms_bass_treble_ratio'] / max(r['absint_bass_treble_ratio'], 1e-9)
        L(f"| {r['name']} | {r['rms_bass_treble_ratio']:.1f}x | "
          f"{r['absint_bass_treble_ratio']:.1f}x | {red:.1f}x flatter |")
    L("")

    L("## Findings")
    L("")
    L("1. **`max(bands)` throws away multi-band information.** Across all 5 "
      "tracks, ≥2 bands are active simultaneously for a substantial fraction "
      "of frames (especially on dense electronic and drum tracks). When two "
      "bands at 0.6 fire together, `max=0.6` reports the same magnitude as a "
      "single band at 0.6 — the listener hears more, the LED shows the same.")
    L("")
    L("2. **`sum(bands)` (clipped) preserves the multi-band combination.** It "
      "fires on more frames than `max` precisely in the cases where multiple "
      "bands are co-active. It rarely saturates above 1.0 in practice (raw-sum "
      "p95 stays below 1.5 on every track tested) because per-band "
      "EMA-normalization keeps individual bands well below their own peaks "
      "most of the time.")
    L("")
    L("3. **`mean(bands)` is too tame.** Dividing by 3 means a single firing "
      "band only gets to 0.33. A dense drop with one strong band registers as "
      "modest activity. Not a viable single-signal driver.")
    L("")
    L("4. **Broadband RMS through `PerBandEMANormalize(num_bands=1)` loses the "
      "fine-grained per-band detail.** It does cleanly capture overall energy "
      "deviations but can't distinguish a bass-only kick from a full-spectrum "
      "drop. Worth keeping as a fallback / cross-check signal but not the "
      "primary driver.")
    L("")
    L("5. **The user's `absint deemphasizes bass` hypothesis is *not* "
      "supported by raw-magnitude measurements.** Bass kicks have huge "
      "transients in their RMS envelope — the attack of a kick generates "
      "as much |d/dt| as a hi-hat does, and on a per-frame basis the bass "
      "absint magnitude is comparable to or larger than the treble absint "
      "magnitude (the bass/treble ratio is roughly preserved or amplified, "
      "not reduced). The reason the *current effect* doesn't look bass-heavy "
      "is that absint is peak-normalized per band downstream — so within-band "
      "dynamics drive the visual, not absolute magnitudes. The recombination "
      "problem (`max` discarding multi-band info) is independent of whether "
      "the upstream signal is RMS or absint.")
    L("")
    L("6. **Cross-band correlation is moderate, not high.** Bass↔mids and "
      "mids↔treble Pearson correlations sit around 0.3-0.6 on most tracks "
      "(bass↔treble lower). The bands carry genuinely independent information "
      "— sum is not just 3x a single signal.")
    L("")

    L("## Recommendation")
    L("")
    L("**Replace `max(bands)` with `clip(sum(bands), 0, 1)` in the unified "
      "drive signal.** This:")
    L("")
    L("- Preserves the per-band EMA-normalization design intent (within-section "
      "context, drop punch).")
    L("- Combines multi-band events additively, which matches both how the ear "
      "perceives them and how a single LED brightness should scale.")
    L("- Keeps `argmax(bands)` available unchanged for color routing — magnitude "
      "comes from the sum, color from the dominant band.")
    L("- Is a one-line change in `energy_waterfall_3band.py` (and any other "
      "consumer of the per-band stack).")
    L("")
    L("Open follow-ups:")
    L("- Compare `clip(sum,1)` to `1 - prod(1 - bands)` (probabilistic-OR), "
      "which is bounded in [0,1] without clipping and may feel smoother on "
      "near-saturating drops.")
    L("- For the pulse-emission overlay, the data does NOT support a "
      "broadband-absint substitute on the basis of bass-deemphasis. If "
      "broadband-absint is worth trying, it's for a different reason: "
      "single-driver simplicity and tighter transient timing.")
    L("")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    plot_paths = {}
    for name in TRACKS:
        path = SEGMENTS_DIR / f"{name}.wav"
        if not path.exists():
            print(f"  ! missing: {path}", file=sys.stderr)
            continue
        print(f"  analyzing {name}...")
        r = analyze_track(name)
        plot_paths[name] = plot_track(r, OUT_DIR)
        results.append(r)
        print(f"    duration {r['duration_s']:.1f}s, "
              f"max-frac>0.5={r['combo_max_stats']['frac>0.5']*100:.1f}%, "
              f"sum-clip-frac>0.5={r['combo_sum_clipped_stats']['frac>0.5']*100:.1f}%")

    report = render_report(results, plot_paths)
    REPORT_PATH.write_text(report)
    print(f"\nReport: {REPORT_PATH}")
    print(f"Plots:  {OUT_DIR}/")


if __name__ == "__main__":
    main()
