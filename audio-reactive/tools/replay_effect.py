#!/usr/bin/env python3
"""Offline replay of any AudioReactiveEffect — produces per-frame JSONL.

Usage:
    python replay_effect.py <audio.wav> --effect <module>:<Class> [-n NUM_LEDS] [-o output.jsonl] [--fps FPS]

Examples:
    python replay_effect.py song.wav --effect mfcc_chroma_rainbow:MfccChromaRainbowEffect
    python replay_effect.py song.wav --effect energy_waterfall_rap:EnergyWaterfallRapEffect -n 150

Feeds audio through the effect's process_audio/render loop, capturing
per-frame output and internal state as JSONL. Suitable for jq queries,
Python analysis, or taste_record.py calibration.

Output fields (always present):
    frame       frame number (0-indexed)
    t           timestamp in seconds
    dt          render timestep
    rgb         list of [r, g, b] per LED

Output fields (when available):
    diagnostics dict from effect.get_diagnostics()
    <internal>  any readable internal state attrs (effect-specific, see below)

The replay automatically probes for known internal state attributes
(smooth_br, smooth_bright, smooth_sat, mfcc_norm, rms_norm, etc.) and
includes whatever the effect exposes. This makes the output useful for
taste_record.py metrics without requiring effects to implement a replay API.

Example queries:
    # Brightness over time (via diagnostics)
    jq '{frame: .frame, bright: .diagnostics.brightness}' out.jsonl

    # Frames where saturation < 0.3
    jq 'select(.smooth_sat < 0.3)' out.jsonl

    # Feed to taste calibration
    python taste_record.py out.jsonl --effect my_effect --variant v1 \\
        --music "jazz" --verdict good
"""

import sys
import os
import json
import argparse
import importlib

# Add effects dir to path so we can import effect classes.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'effects'))

import numpy as np
import soundfile as sf


# Internal state attributes to probe on each effect instance.
# (attr_name, output_key, transform)
# The output key strips leading underscores and "shared_" prefixes.
KNOWN_STATE_ATTRS = [
    ('_shared_mfcc_norm',     'mfcc_norm',     lambda v: [round(float(x), 5) for x in v]),
    ('_shared_mfcc_contrast', 'contrast',      lambda v: round(float(v), 5)),
    ('_shared_rms_norm',      'rms_norm',      lambda v: round(float(v), 5)),
    ('_rms_peak',             'rms_peak',      lambda v: round(float(v), 8)),
    ('_adapt_count',          'adapt_count',   lambda v: int(v)),
    ('_smooth_sat',           'smooth_sat',    lambda v: round(float(v), 5)),
    ('_smooth_bright',        'smooth_bright', lambda v: round(float(v), 5)),
    ('_silence_factor',       'silence_factor', lambda v: round(float(v), 5)),
    ('_smooth_br',            'smooth_br',     lambda v: [round(float(x), 5) for x in v]),
]


def load_effect_class(spec):
    """Load an effect class from 'module_name:ClassName' spec.

    Example: 'mfcc_chroma_rainbow:MfccChromaRainbowEffect'
    """
    if ':' not in spec:
        raise ValueError(
            f"Effect spec must be 'module:Class', got: {spec}\n"
            f"Example: mfcc_chroma_rainbow:MfccChromaRainbowEffect"
        )
    module_name, class_name = spec.split(':', 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def capture_state(effect, frame_num, timestamp, render_dt, rgb_frame):
    """Snapshot output and readable internal state from the effect.

    Always includes frame/t/dt/rgb. Probes for known internal attributes
    and includes whatever exists on this particular effect instance.
    """
    record = {
        'frame': frame_num,
        't': round(timestamp, 6),
        'dt': round(render_dt, 6),
        'rgb': rgb_frame.tolist(),
    }

    # Standard diagnostics interface
    try:
        diag = effect.get_diagnostics()
        if diag:
            record['diagnostics'] = diag
    except Exception:
        pass

    # Probe known internal state attributes (under lock if available)
    lock = getattr(effect, '_lock', None)
    if lock:
        try:
            lock.acquire()
        except Exception:
            lock = None

    try:
        for attr_name, output_key, transform in KNOWN_STATE_ATTRS:
            val = getattr(effect, attr_name, None)
            if val is not None:
                try:
                    record[output_key] = transform(
                        val.copy() if hasattr(val, 'copy') else val
                    )
                except Exception:
                    pass
    finally:
        if lock:
            try:
                lock.release()
            except Exception:
                pass

    return record


def replay(audio_path, effect_spec, num_leds=60, fps=30, output_path=None):
    """Run the effect on an audio file and emit per-frame JSONL."""
    audio, sr = sf.read(audio_path, dtype='float32', always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    duration = len(audio) / sr
    effect_cls = load_effect_class(effect_spec)
    effect = effect_cls(num_leds=num_leds, sample_rate=sr)

    print(f"Audio: {audio_path}", file=sys.stderr)
    print(f"  {sr} Hz, {len(audio)} samples, {duration:.2f}s", file=sys.stderr)
    print(f"  Effect: {effect.name} ({effect_spec})", file=sys.stderr)
    print(f"  {num_leds} LEDs, {fps} fps render", file=sys.stderr)

    chunk_size = 1024
    render_dt = 1.0 / fps
    audio_pos = 0
    total_samples = len(audio)

    out_file = open(output_path, 'w') if output_path else sys.stdout
    frame_count = 0

    try:
        t = 0.0
        while audio_pos < total_samples:
            # Advance audio to current render time
            target_sample = min(int(t * sr) + chunk_size, total_samples)
            while audio_pos < target_sample:
                chunk_end = min(audio_pos + chunk_size, target_sample)
                chunk = audio[audio_pos:chunk_end]
                if len(chunk) > 0:
                    effect.process_audio(chunk)
                audio_pos = chunk_end

            # Render and capture
            rgb = effect.render(render_dt)
            record = capture_state(effect, frame_count, t, render_dt, rgb)
            out_file.write(json.dumps(record) + '\n')

            frame_count += 1
            t += render_dt

    finally:
        if output_path and out_file is not sys.stdout:
            out_file.close()

    print(f"  {frame_count} frames written", file=sys.stderr)
    return frame_count


def main():
    parser = argparse.ArgumentParser(
        description='Replay audio through any AudioReactiveEffect, output per-frame JSONL'
    )
    parser.add_argument('audio', help='Path to WAV audio file')
    parser.add_argument('--effect', required=True,
                        help='Effect spec: module:Class '
                             '(e.g., mfcc_chroma_rainbow:MfccChromaRainbowEffect)')
    parser.add_argument('-n', '--num-leds', type=int, default=60,
                        help='Number of LEDs (default: 60)')
    parser.add_argument('-o', '--output',
                        help='Output JSONL file (default: stdout)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Render FPS (default: 30)')
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    replay(args.audio, args.effect,
           num_leds=args.num_leds, fps=args.fps,
           output_path=args.output)


if __name__ == '__main__':
    main()
