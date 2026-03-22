#!/usr/bin/env python3
"""
LED Effect Preview CLI

Runs an audio-reactive effect headlessly, renders a composite preview PNG,
and prints metrics as JSON to stdout.

Usage:
    python tools/preview_effect.py energy_waterfall \
        --wav audio-reactive/research/audio-segments/booster_skrillex.wav \
        --out /tmp/preview_energy_waterfall.png \
        --leds 150 --duration 15 --start 0
"""

import sys
import os
import argparse
import json
import base64

import numpy as np

# Add effects directory to sys.path so runner.py can import its siblings
EFFECTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio-reactive', 'effects')
EFFECTS_DIR = os.path.abspath(EFFECTS_DIR)
sys.path.insert(0, EFFECTS_DIR)

# Import our preview library
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)


def main():
    parser = argparse.ArgumentParser(description='Preview an LED effect as a composite PNG')
    parser.add_argument('effect_name', help='Name of the effect to run')
    parser.add_argument('--wav', required=True, help='Path to WAV file')
    parser.add_argument('--out', required=True, help='Output PNG path')
    parser.add_argument('--leds', type=int, default=150, help='Number of LEDs (default: 150)')
    parser.add_argument('--duration', type=float, default=15.0, help='Duration in seconds (default: 15)')
    parser.add_argument('--start', type=float, default=0.0, help='Start time in seconds (default: 0)')
    parser.add_argument('--zoom', action='store_true',
                        help='Also render a zoomed-in 2-3s window centered on peak energy')

    args = parser.parse_args()

    wav_path = os.path.abspath(args.wav)
    out_path = os.path.abspath(args.out)

    if not os.path.exists(wav_path):
        print(json.dumps({'error': f'WAV file not found: {wav_path}'}))
        sys.exit(1)

    # Step 1: Run effect headlessly via runner.analyze_effect()
    print(f"Running effect '{args.effect_name}' on {os.path.basename(wav_path)}...",
          file=sys.stderr)

    from runner import analyze_effect

    result = analyze_effect(
        effect_name=args.effect_name,
        wav_path=wav_path,
        num_leds=args.leds,
    )

    if 'error' in result:
        print(json.dumps({'error': result['error']}))
        sys.exit(1)

    # Step 2: Decode base64 LED data into numpy array
    led_b64 = result['led_data']
    num_frames = result['num_frames']
    num_leds = result['num_leds']
    fps = result['fps']
    audio_duration = result['duration']

    if not led_b64 or num_frames == 0:
        print(json.dumps({'error': 'No frames rendered'}))
        sys.exit(1)

    raw_bytes = base64.b64decode(led_b64)
    total_values = len(raw_bytes)
    # Infer actual num_leds from data (some effects override the requested count)
    actual_leds = total_values // (num_frames * 3)
    if actual_leds != num_leds:
        print(f"Note: effect uses {actual_leds} LEDs (requested {num_leds})",
              file=sys.stderr)
        num_leds = actual_leds
    frames = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(num_frames, num_leds, 3)

    print(f"Got {num_frames} frames, {num_leds} LEDs, {fps} fps, "
          f"{audio_duration:.1f}s audio", file=sys.stderr)

    # Step 3: Render composite preview
    from preview_lib import render_preview

    duration_to_render = min(args.duration, num_frames / fps)

    composite, metrics = render_preview(
        frames=frames,
        audio_path=wav_path,
        fps=fps,
        duration=duration_to_render,
        start_time=args.start,
    )

    # Step 4: Save PNG
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    composite.save(out_path, 'PNG')
    print(f"Saved preview to {out_path}", file=sys.stderr)

    # Step 4b: Zoom preview (if requested)
    zoom_path = None
    zoom_metrics = None
    if args.zoom:
        from preview_lib import render_zoom_preview

        zoom_composite, zoom_metrics, zoom_start = render_zoom_preview(
            frames=frames,
            audio_path=wav_path,
            fps=fps,
            duration=duration_to_render,
            start_time=args.start,
        )

        # Save as <output>_zoom.png
        base, ext = os.path.splitext(out_path)
        zoom_path = f"{base}_zoom{ext}"
        zoom_composite.save(zoom_path, 'PNG')
        print(f"Saved zoom preview to {zoom_path} (window starts at {zoom_start:.2f}s)",
              file=sys.stderr)

    # Step 5: Print JSON to stdout
    output = {
        'png': out_path,
        'num_frames': num_frames,
        'num_leds': num_leds,
        'duration_rendered': round(duration_to_render, 1),
        'metrics': metrics,
    }
    if zoom_path:
        output['zoom_png'] = zoom_path
        output['zoom_metrics'] = zoom_metrics
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
