"""
Compare HPSS-based percussive detection vs per-band spectral flux.

Hypothesis: spectral flux produces ~same events as the HPSS residual approach
used in band_sparkle_band_fade_bg.py, but with zero latency and simpler code.

Runs both detectors on the same audio files, outputs:
  - Event counts per band per method
  - Timing agreement (events within ±50ms tolerance)
  - Background color (dominant band) agreement over time
"""

import numpy as np
import librosa
import sys
import os
from pathlib import Path

# ── Band definitions (same as effect) ──
BANDS = [
    ('Sub-bass', 20, 80),
    ('Bass', 80, 250),
    ('Mids', 250, 2000),
    ('High-mids', 2000, 6000),
    ('Treble', 6000, 8000),
]
BAND_NAMES = ['Sub', 'Bas', 'Mid', 'HiM', 'Tre']
N_BANDS = len(BANDS)

N_FFT = 2048
HOP = 512
SR = 44100


def make_band_masks():
    freq_bins = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    masks = []
    for _, lo, hi in BANDS:
        masks.append((freq_bins >= lo) & (freq_bins < hi))
    return freq_bins, masks


def run_hpss_detector(audio):
    """Reproduce the exact HPSS-based detection from band_sparkle_band_fade_bg.py."""
    freq_bins, band_masks = make_band_masks()
    window = np.hanning(N_FFT).astype(np.float32)

    # HPSS state
    hpss_buf_size = 17
    spec_buf = np.zeros((hpss_buf_size, N_FFT // 2 + 1), dtype=np.float32)
    spec_buf_idx = 0
    spec_buf_filled = 0

    # Harmonic band energy (background color)
    band_peaks = np.full(N_BANDS, 1e-10, dtype=np.float32)
    band_peak_decay = 0.9995
    band_window_len = int(5 * SR / N_FFT)
    band_ring = np.zeros((band_window_len, N_BANDS), dtype=np.float32)
    band_ring_pos = 0
    band_ring_filled = 0

    # Percussive per-band peak detection
    perc_history_len = 8
    perc_history = np.zeros((perc_history_len, N_BANDS), dtype=np.float32)
    perc_hist_idx = 0
    perc_hist_filled = 0
    perc_cooldown_frames = 3
    perc_cooldown_counters = np.zeros(N_BANDS, dtype=np.int32)
    perc_band_peaks = np.full(N_BANDS, 1e-10, dtype=np.float32)
    perc_peak_decay = 0.998

    # Centroid range tracking
    centroid_min = np.array([lo for _, lo, _ in BANDS], dtype=np.float32)
    centroid_max = np.array([hi for _, _, hi in BANDS], dtype=np.float32)
    centroid_decay = 0.9998

    events = []  # (time_sec, band_idx, strength)
    bg_bands = []  # (time_sec, dominant_band_idx)

    # Process frames with overlap
    n_frames = (len(audio) - N_FFT) // HOP + 1
    for f in range(n_frames):
        start = f * HOP
        frame = audio[start:start + N_FFT]
        time_sec = start / SR

        spec = np.abs(np.fft.rfft(frame * window))

        # Streaming HPSS
        spec_buf[spec_buf_idx] = spec
        spec_buf_idx = (spec_buf_idx + 1) % hpss_buf_size
        spec_buf_filled = min(spec_buf_filled + 1, hpss_buf_size)

        if spec_buf_filled >= 3:
            buf_slice = spec_buf[:spec_buf_filled]
            harmonic_mask = np.median(buf_slice, axis=0)
            percussive = np.maximum(spec - harmonic_mask, 0)
        else:
            harmonic_mask = spec * 0.5
            percussive = spec * 0.5

        # Harmonic band energy → background color
        harm_energies = np.array(
            [np.sum(harmonic_mask[m] ** 2) for m in band_masks], dtype=np.float32)

        for i in range(N_BANDS):
            band_peaks[i] = max(harm_energies[i], band_peaks[i] * band_peak_decay)
        normalized = harm_energies / band_peaks

        idx = band_ring_pos % band_window_len
        band_ring[idx] = normalized
        band_ring_pos += 1
        band_ring_filled = min(band_ring_filled + 1, band_window_len)

        filled = band_ring[:band_ring_filled]
        integrals = np.sum(filled, axis=0)
        total = np.sum(integrals)
        if total > 0:
            proportions = integrals / total
            sharpened = proportions ** 3
            s_total = np.sum(sharpened)
            weights = sharpened / s_total if s_total > 0 else proportions
        else:
            weights = np.ones(N_BANDS) / N_BANDS

        dominant = int(np.argmax(weights))
        bg_bands.append((time_sec, dominant))

        # Percussive per-band peak detection
        perc_energies = np.array(
            [np.sum(percussive[m] ** 2) for m in band_masks], dtype=np.float32)

        h_idx = perc_hist_idx % perc_history_len
        perc_history[h_idx] = perc_energies
        perc_hist_idx += 1
        perc_hist_filled = min(perc_hist_filled + 1, perc_history_len)

        perc_cooldown_counters = np.maximum(perc_cooldown_counters - 1, 0)

        for i in range(N_BANDS):
            perc_band_peaks[i] = max(perc_energies[i], perc_band_peaks[i] * perc_peak_decay)

        total_perc = np.sum(perc_energies)
        if total_perc > 0:
            band_shares = perc_energies / total_perc
        else:
            band_shares = np.zeros(N_BANDS)

        if perc_hist_filled >= 3:
            hist = perc_history[:perc_hist_filled]
            means = np.mean(hist, axis=0)
            stds = np.std(hist, axis=0)

            for i in range(N_BANDS):
                if perc_cooldown_counters[i] > 0:
                    continue
                if band_shares[i] < 0.15:
                    continue
                threshold = means[i] + 2.5 * stds[i]
                floor = perc_band_peaks[i] * 0.05
                if perc_energies[i] > threshold and perc_energies[i] > floor:
                    excess = perc_energies[i] - threshold
                    headroom = perc_band_peaks[i] - threshold
                    raw_strength = min(1.0, excess / (headroom + 1e-10))
                    strength = raw_strength * band_shares[i] / max(band_shares)
                    events.append((time_sec, i, max(0.3, strength)))
                    perc_cooldown_counters[i] = perc_cooldown_frames

    return events, bg_bands


def run_flux_detector(audio, avg_frames=1):
    """Per-band half-wave rectified spectral flux.

    avg_frames=1: compare vs previous frame (original flux)
    avg_frames=4: compare vs 4-frame running average (less chatty)
    """
    freq_bins, band_masks = make_band_masks()
    window = np.hanning(N_FFT).astype(np.float32)

    # Background color state (same as HPSS version, but on raw spectrum)
    band_peaks = np.full(N_BANDS, 1e-10, dtype=np.float32)
    band_peak_decay = 0.9995
    band_window_len = int(5 * SR / N_FFT)
    band_ring = np.zeros((band_window_len, N_BANDS), dtype=np.float32)
    band_ring_pos = 0
    band_ring_filled = 0

    # Spectral flux state — ring buffer per band for running average
    spec_rings = [np.zeros((avg_frames, np.sum(m)), dtype=np.float32) for m in band_masks]
    spec_ring_idx = 0
    spec_ring_filled = 0

    # Per-band flux peak detection (same threshold logic as HPSS version)
    flux_history_len = 8
    flux_history = np.zeros((flux_history_len, N_BANDS), dtype=np.float32)
    flux_hist_idx = 0
    flux_hist_filled = 0
    flux_cooldown_frames = 3
    flux_cooldown_counters = np.zeros(N_BANDS, dtype=np.int32)
    flux_band_peaks = np.full(N_BANDS, 1e-10, dtype=np.float32)
    flux_peak_decay = 0.998

    events = []
    bg_bands = []

    n_frames = (len(audio) - N_FFT) // HOP + 1
    for f in range(n_frames):
        start = f * HOP
        frame = audio[start:start + N_FFT]
        time_sec = start / SR

        spec = np.abs(np.fft.rfft(frame * window))

        # ── Background color: raw band energy (no HPSS) ──
        raw_energies = np.array(
            [np.sum(spec[m] ** 2) for m in band_masks], dtype=np.float32)

        for i in range(N_BANDS):
            band_peaks[i] = max(raw_energies[i], band_peaks[i] * band_peak_decay)
        normalized = raw_energies / band_peaks

        idx = band_ring_pos % band_window_len
        band_ring[idx] = normalized
        band_ring_pos += 1
        band_ring_filled = min(band_ring_filled + 1, band_window_len)

        filled = band_ring[:band_ring_filled]
        integrals = np.sum(filled, axis=0)
        total = np.sum(integrals)
        if total > 0:
            proportions = integrals / total
            sharpened = proportions ** 3
            s_total = np.sum(sharpened)
            weights = sharpened / s_total if s_total > 0 else proportions
        else:
            weights = np.ones(N_BANDS) / N_BANDS

        dominant = int(np.argmax(weights))
        bg_bands.append((time_sec, dominant))

        # ── Per-band half-wave rectified spectral flux ──
        ring_idx = spec_ring_idx % avg_frames
        flux_values = np.zeros(N_BANDS, dtype=np.float32)
        for i, m in enumerate(band_masks):
            band_spec = spec[m]
            if spec_ring_filled >= 1:
                avg = np.mean(spec_rings[i][:spec_ring_filled], axis=0)
                diff = band_spec - avg
                flux_values[i] = np.sum(np.maximum(diff, 0) ** 2)
            spec_rings[i][ring_idx] = band_spec
        spec_ring_idx += 1
        spec_ring_filled = min(spec_ring_filled + 1, avg_frames)

        # Same threshold logic as HPSS version
        h_idx = flux_hist_idx % flux_history_len
        flux_history[h_idx] = flux_values
        flux_hist_idx += 1
        flux_hist_filled = min(flux_hist_filled + 1, flux_history_len)

        flux_cooldown_counters = np.maximum(flux_cooldown_counters - 1, 0)

        for i in range(N_BANDS):
            flux_band_peaks[i] = max(flux_values[i], flux_band_peaks[i] * flux_peak_decay)

        total_flux = np.sum(flux_values)
        if total_flux > 0:
            band_shares = flux_values / total_flux
        else:
            band_shares = np.zeros(N_BANDS)

        if flux_hist_filled >= 3:
            hist = flux_history[:flux_hist_filled]
            means = np.mean(hist, axis=0)
            stds = np.std(hist, axis=0)

            for i in range(N_BANDS):
                if flux_cooldown_counters[i] > 0:
                    continue
                if band_shares[i] < 0.15:
                    continue
                threshold = means[i] + 2.5 * stds[i]
                floor = flux_band_peaks[i] * 0.05
                if flux_values[i] > threshold and flux_values[i] > floor:
                    excess = flux_values[i] - threshold
                    headroom = flux_band_peaks[i] - threshold
                    raw_strength = min(1.0, excess / (headroom + 1e-10))
                    strength = raw_strength * band_shares[i] / max(band_shares)
                    events.append((time_sec, i, max(0.3, strength)))
                    flux_cooldown_counters[i] = flux_cooldown_frames

    return events, bg_bands


def match_events(events_a, events_b, tolerance=0.050):
    """Match events between two lists within timing tolerance, per-band.
    Returns (matched_pairs, unmatched_a, unmatched_b)."""
    matched = []
    used_b = set()

    for t_a, band_a, s_a in events_a:
        best_j = None
        best_dt = tolerance + 1
        for j, (t_b, band_b, s_b) in enumerate(events_b):
            if j in used_b:
                continue
            if band_a != band_b:
                continue
            dt = abs(t_a - t_b)
            if dt <= tolerance and dt < best_dt:
                best_dt = dt
                best_j = j

        if best_j is not None:
            matched.append(((t_a, band_a, s_a), events_b[best_j], best_dt))
            used_b.add(best_j)

    unmatched_a = [(t, b, s) for i, (t, b, s) in enumerate(events_a)
                   if not any(m[0] == (t, b, s) for m in matched)]
    unmatched_b = [(t, b, s) for j, (t, b, s) in enumerate(events_b)
                   if j not in used_b]

    return matched, unmatched_a, unmatched_b


def compare_bg(bg_a, bg_b):
    """Compare background band sequences. Returns fraction of frames that agree."""
    n = min(len(bg_a), len(bg_b))
    if n == 0:
        return 1.0
    agree = sum(1 for i in range(n) if bg_a[i][1] == bg_b[i][1])
    return agree / n


def analyze_file(filepath):
    """Run all three detectors on one file and print comparison."""
    name = Path(filepath).stem
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    audio, sr = librosa.load(filepath, sr=SR, mono=True)
    duration = len(audio) / SR
    print(f"  Duration: {duration:.1f}s")

    hpss_events, hpss_bg = run_hpss_detector(audio)
    flux1_events, flux1_bg = run_flux_detector(audio, avg_frames=1)
    flux4_events, flux4_bg = run_flux_detector(audio, avg_frames=4)

    # Event counts per band — all three
    print(f"\n  Events per band:")
    print(f"  {'Band':<8} {'HPSS':>6} {'Flux1':>6} {'Flux4':>6}")
    print(f"  {'-'*30}")
    for i in range(N_BANDS):
        h = sum(1 for _, b, _ in hpss_events if b == i)
        f1 = sum(1 for _, b, _ in flux1_events if b == i)
        f4 = sum(1 for _, b, _ in flux4_events if b == i)
        print(f"  {BAND_NAMES[i]:<8} {h:>6} {f1:>6} {f4:>6}")

    h_total = len(hpss_events)
    f1_total = len(flux1_events)
    f4_total = len(flux4_events)
    print(f"  {'TOTAL':<8} {h_total:>6} {f1_total:>6} {f4_total:>6}")

    # Match each variant against HPSS
    def match_summary(label, test_events, test_bg):
        matched, only_hpss, only_test = match_events(hpss_events, test_events)
        n_matched = len(matched)
        hpss_r = n_matched / h_total if h_total > 0 else 1.0
        test_r = n_matched / len(test_events) if len(test_events) > 0 else 1.0
        bg = compare_bg(hpss_bg, test_bg)
        if matched:
            dts = [dt * 1000 for _, _, dt in matched]
            timing = f"median {np.median(dts):.1f}ms"
        else:
            timing = "n/a"
        return {
            'label': label,
            'count': len(test_events),
            'matched': n_matched,
            'only_hpss': len(only_hpss),
            'only_test': len(only_test),
            'hpss_recall': hpss_r,
            'test_recall': test_r,
            'bg_agree': bg,
            'timing': timing,
        }

    s1 = match_summary('Flux(1)', flux1_events, flux1_bg)
    s4 = match_summary('Flux(4)', flux4_events, flux4_bg)

    print(f"\n  vs HPSS (±50ms, same band):")
    print(f"  {'':>12} {'Flux(1)':>10} {'Flux(4)':>10}")
    print(f"  {'Matched':>12} {s1['matched']:>10} {s4['matched']:>10}")
    print(f"  {'HPSS-only':>12} {s1['only_hpss']:>10} {s4['only_hpss']:>10}")
    print(f"  {'Extra':>12} {s1['only_test']:>10} {s4['only_test']:>10}")
    print(f"  {'HPSS recall':>12} {s1['hpss_recall']:>9.0%} {s4['hpss_recall']:>9.0%}")
    print(f"  {'Precision':>12} {s1['test_recall']:>9.0%} {s4['test_recall']:>9.0%}")
    print(f"  {'BG agree':>12} {s1['bg_agree']:>9.0%} {s4['bg_agree']:>9.0%}")
    print(f"  {'Timing':>12} {s1['timing']:>10} {s4['timing']:>10}")

    # Also compare flux4 vs flux1 directly
    m41, only1, only4 = match_events(flux1_events, flux4_events)
    f1_in_f4 = len(m41) / f1_total if f1_total > 0 else 1.0
    print(f"\n  Flux(4) vs Flux(1): {len(m41)} matched, "
          f"Flux(1) recall={f1_in_f4:.0%}, "
          f"{len(only1)} dropped, {len(only4)} new")

    return {
        'name': name,
        'duration': duration,
        'hpss': h_total,
        'flux1': f1_total,
        'flux4': f4_total,
        'hpss_recall_1': s1['hpss_recall'],
        'hpss_recall_4': s4['hpss_recall'],
        'precision_1': s1['test_recall'],
        'precision_4': s4['test_recall'],
        'bg1': s1['bg_agree'],
        'bg4': s4['bg_agree'],
    }


def main():
    seg_dir = Path(__file__).parent.parent / 'audio-segments'

    files = [
        'opiate_intro.wav',
        'fa_br_drop1.wav',
        'electronic_beat.wav',
        'amen_break.wav',
        'cinematic_drums.wav',
        'lorn_sega_sunset.wav',
        'ambient.wav',
        'fourtet_set.wav',
    ]

    results = []
    for f in files:
        path = seg_dir / f
        if not path.exists():
            print(f"\n  SKIP: {f} not found")
            continue
        results.append(analyze_file(str(path)))

    if results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY — all three detectors")
        print(f"{'='*70}")
        print(f"  {'Track':<20} {'HPSS':>5} {'F(1)':>5} {'F(4)':>5}  "
              f"{'H-rec1':>6} {'H-rec4':>6}  {'BG1':>4} {'BG4':>4}")
        print(f"  {'-'*62}")
        for r in results:
            print(f"  {r['name'][:20]:<20} {r['hpss']:>5} {r['flux1']:>5} {r['flux4']:>5}  "
                  f"{r['hpss_recall_1']:>5.0%} {r['hpss_recall_4']:>5.0%}  "
                  f"{r['bg1']:>3.0%} {r['bg4']:>3.0%}")

        # Totals
        th = sum(r['hpss'] for r in results)
        t1 = sum(r['flux1'] for r in results)
        t4 = sum(r['flux4'] for r in results)
        ah1 = np.mean([r['hpss_recall_1'] for r in results])
        ah4 = np.mean([r['hpss_recall_4'] for r in results])
        ab1 = np.mean([r['bg1'] for r in results])
        ab4 = np.mean([r['bg4'] for r in results])
        print(f"  {'-'*62}")
        print(f"  {'AVG/TOTAL':<20} {th:>5} {t1:>5} {t4:>5}  "
              f"{ah1:>5.0%} {ah4:>5.0%}  {ab1:>3.0%} {ab4:>3.0%}")
        print()
        print(f"  H-rec = % of HPSS events also found by flux variant")
        print(f"  BG    = % of frames where background band matches HPSS")


if __name__ == '__main__':
    main()
