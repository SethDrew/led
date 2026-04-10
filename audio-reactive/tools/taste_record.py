#!/usr/bin/env python3
"""Record a taste calibration entry from a replay JSONL file.

Computes standard metrics from replay data (produced by replay_effect.py)
and appends a record to taste.jsonl. Works with any effect — metrics that
require specific replay fields (e.g., mfcc_norm) are skipped when absent.

Usage:
    # From replay data (replay_effect.py output):
    python taste_record.py replay.jsonl \
        --effect energy_waterfall_rap \
        --variant "baseline" \
        --music "hip hop, heavy 808s" \
        --applies-to led_strip 60_leds dark_room 1m_viewing \
        --verdict good \
        --notes "Bass hits land well, treble detail present"

    # Seed entry with manually provided metrics (no replay file):
    python taste_record.py --seed \
        --effect mfcc_chroma_rainbow \
        --variant "original_br6.93" \
        --music "jazz, walking bass, full ensemble" \
        --applies-to led_strip 60_leds dark_room 2m_viewing \
        --verdict rejected \
        --notes "Flicker at 11/s too flashy for LEDs" \
        --metric flicker_rate_per_s=11.0 \
        --metric black_frame_pct=0.31 \
        --param BR_DECAY=6.93 --param ATTACK=55.0

Metrics computed from replay data (availability depends on effect):

  Universal (RGB-only, always computed):
    flicker_rate_per_s     visible→black→visible transitions within 100ms, per LED per second
    black_frame_pct        fraction of LED-frames with RGB sum = 0
    mean_rgb_brightness    mean perceived brightness from RGB output (0-1)

  Requires smooth_br in replay:
    dynamic_range          [p10, p90] of smooth_br across all LED-frames

  Requires smooth_sat / smooth_bright in replay:
    mean_saturation        mean of smooth_sat across frames
    mean_brightness        mean of smooth_bright across frames

  Requires mfcc_norm in replay (MFCC-based effects only):
    bass_treble_ratio      mean of MFCC coefficients 1-4 / mean of 9-12
    binary_mfcc_pct        fraction of frames where all MFCCs are at 0 or 1
    inter_coeff_correlation mean pairwise correlation between MFCC time series
"""

import sys
import os
import json
import argparse
from datetime import date

import numpy as np


TASTE_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'research', 'taste.jsonl'
)

VALID_VERDICTS = ('rejected', 'acceptable', 'good', 'great')


def load_replay(path):
    """Load replay JSONL into list of dicts."""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _has_field(frames, field):
    """Check if a field exists in replay data."""
    return field in frames[0]


def compute_metrics(frames):
    """Compute metrics from replay data. Skips metrics whose required fields are absent."""
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("Empty replay file")

    n_leds = len(frames[0]['rgb'])
    fps = 1.0 / frames[0]['dt'] if frames[0]['dt'] > 0 else 30.0
    duration = frames[-1]['t'] - frames[0]['t'] + frames[0]['dt']

    metrics = {}

    # ── Universal metrics (RGB-only) ──

    rgb_all = np.array([f['rgb'] for f in frames], dtype=np.uint8)  # (n_frames, n_leds, 3)
    rgb_sum = rgb_all.sum(axis=2)  # (n_frames, n_leds)

    # flicker_rate_per_s: visible→black→visible within 100ms
    max_gap_frames = max(1, int(0.1 * fps))
    is_visible = rgb_sum > 0
    flicker_count = 0
    for led in range(n_leds):
        vis = is_visible[:, led]
        i = 0
        while i < n_frames:
            if vis[i]:
                j = i + 1
                while j < n_frames and vis[j]:
                    j += 1
                if j < n_frames:
                    k = j + 1
                    while k < n_frames and not vis[k] and (k - j) <= max_gap_frames:
                        k += 1
                    if k < n_frames and vis[k] and (k - j) <= max_gap_frames:
                        flicker_count += 1
                        i = k
                        continue
                i = j
            else:
                i += 1
    metrics['flicker_rate_per_s'] = round(
        flicker_count / max(duration, 1e-10) / max(n_leds, 1), 2
    )

    # black_frame_pct: fraction of LED-frames with RGB sum = 0
    total_led_frames = n_frames * n_leds
    metrics['black_frame_pct'] = round(
        int((rgb_sum == 0).sum()) / total_led_frames, 4
    )

    # mean_rgb_brightness: perceived brightness from RGB (0-1)
    # Uses luminance weights (0.299R + 0.587G + 0.114B) / 255
    lum = (rgb_all[:, :, 0] * 0.299 + rgb_all[:, :, 1] * 0.587
           + rgb_all[:, :, 2] * 0.114) / 255.0
    metrics['mean_rgb_brightness'] = round(float(lum.mean()), 4)

    # ── Metrics requiring smooth_br ──

    if _has_field(frames, 'smooth_br'):
        smooth_br_all = np.array([f['smooth_br'] for f in frames])
        all_br = smooth_br_all.flatten()
        p10 = float(np.percentile(all_br, 10))
        p90 = float(np.percentile(all_br, 90))
        metrics['dynamic_range'] = [round(p10, 4), round(p90, 4)]

    # ── Metrics requiring smooth_sat / smooth_bright ──

    if _has_field(frames, 'smooth_sat'):
        smooth_sat = np.array([f['smooth_sat'] for f in frames])
        metrics['mean_saturation'] = round(float(smooth_sat.mean()), 4)

    if _has_field(frames, 'smooth_bright'):
        smooth_bright = np.array([f['smooth_bright'] for f in frames])
        metrics['mean_brightness'] = round(float(smooth_bright.mean()), 4)

    # ── MFCC-specific metrics (only when mfcc_norm present) ──

    if _has_field(frames, 'mfcc_norm'):
        mfcc_series = np.array([f['mfcc_norm'] for f in frames])
        n_coeffs = mfcc_series.shape[1]

        # bass_treble_ratio: low-order vs high-order MFCCs
        if n_coeffs >= 12:
            bass = mfcc_series[:, 0:4].mean()
            treble = mfcc_series[:, 8:12].mean()
            metrics['bass_treble_ratio'] = round(
                float(bass / max(treble, 1e-10)), 3
            )

        # binary_mfcc_pct: fraction of frames with all MFCCs pegged to 0 or 1
        binary_frames = 0
        for i in range(n_frames):
            mfcc = mfcc_series[i]
            if np.all((mfcc <= 0.001) | (mfcc >= 0.999)):
                binary_frames += 1
        metrics['binary_mfcc_pct'] = round(binary_frames / n_frames, 4)

        # inter_coeff_correlation: mean pairwise Pearson between MFCC time series
        if n_frames > 2 and n_coeffs > 1:
            corr_matrix = np.corrcoef(mfcc_series.T)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            metrics['inter_coeff_correlation'] = round(
                float(np.nanmean(corr_matrix[mask])), 4
            )

    return metrics


def extract_parameters(frames):
    """Extract effect parameters that are visible in replay data.

    Returns best-guess parameters from the replay — these are observable
    state values, not the source constants. The caller should override
    with known parameter values via --param flags.
    """
    # We can infer some parameters from the data patterns but not all.
    # Return an empty dict — the user provides params via CLI.
    return {}


def build_record(effect, variant, music, applies_to, verdict, notes,
                 metrics, parameters):
    """Build a taste record dict."""
    return {
        'date': date.today().isoformat(),
        'effect': effect,
        'variant': variant,
        'music': music,
        'applies_to': applies_to,
        'verdict': verdict,
        'notes': notes,
        'metrics': metrics,
        'parameters': parameters,
    }


def append_record(record, taste_path=None):
    """Append a record to taste.jsonl."""
    path = taste_path or TASTE_FILE
    path = os.path.abspath(path)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return path


def parse_key_value(s):
    """Parse 'key=value' into (key, numeric_or_string_value)."""
    if '=' not in s:
        raise argparse.ArgumentTypeError(f"Expected key=value, got: {s}")
    key, val = s.split('=', 1)
    # Try numeric
    try:
        return key, float(val)
    except ValueError:
        return key, val


def main():
    parser = argparse.ArgumentParser(
        description='Record a taste calibration entry from replay data'
    )
    parser.add_argument('replay', nargs='?',
                        help='Path to replay JSONL file')
    parser.add_argument('--seed', action='store_true',
                        help='Seed mode: no replay file, provide metrics manually')
    parser.add_argument('--effect', required=True,
                        help='Effect name')
    parser.add_argument('--variant', required=True,
                        help='Variant description')
    parser.add_argument('--music', required=True,
                        help='Music description')
    parser.add_argument('--applies-to', nargs='+', default=[],
                        help='Context tags (hardware, viewing distance, etc.)')
    parser.add_argument('--verdict', required=True, choices=VALID_VERDICTS,
                        help='Quality verdict')
    parser.add_argument('--notes', default='',
                        help='Free text notes')
    parser.add_argument('--metric', action='append', type=parse_key_value,
                        default=[], dest='manual_metrics',
                        help='Manual metric: key=value (repeatable)')
    parser.add_argument('--param', action='append', type=parse_key_value,
                        default=[], dest='manual_params',
                        help='Effect parameter: key=value (repeatable)')
    parser.add_argument('--taste-file', default=None,
                        help='Path to taste.jsonl (default: audio-reactive/research/taste.jsonl)')
    parser.add_argument('--date', default=None,
                        help='Override date (YYYY-MM-DD)')

    args = parser.parse_args()

    if not args.seed and not args.replay:
        parser.error("Provide a replay JSONL file, or use --seed for manual entry")

    if args.seed:
        # Seed mode: metrics come from --metric flags
        metrics = {}
        for key, val in args.manual_metrics:
            # Handle list-like values (dynamic_range)
            if isinstance(val, str) and val.startswith('['):
                metrics[key] = json.loads(val)
            else:
                metrics[key] = val
    else:
        # Compute from replay data
        if not os.path.isfile(args.replay):
            print(f"Error: file not found: {args.replay}", file=sys.stderr)
            sys.exit(1)
        frames = load_replay(args.replay)
        metrics = compute_metrics(frames)
        # Override with any manual metrics
        for key, val in args.manual_metrics:
            metrics[key] = val

    # Parameters
    parameters = {}
    for key, val in args.manual_params:
        parameters[key] = val

    record = build_record(
        effect=args.effect,
        variant=args.variant,
        music=args.music,
        applies_to=args.applies_to,
        verdict=args.verdict,
        notes=args.notes,
        metrics=metrics,
        parameters=parameters,
    )

    if args.date:
        record['date'] = args.date

    # Append
    path = append_record(record, taste_path=args.taste_file)

    # Print for confirmation
    print(json.dumps(record, indent=2))
    print(f"\nAppended to: {path}", file=sys.stderr)


if __name__ == '__main__':
    main()
