#!/usr/bin/env python3
"""
Play back recorded audio segments.

Usage:
    python playback_segment.py <segment_name> [speakers|blackhole]

Examples:
    python playback_segment.py tool_lateralus_intro            # play to speakers
    python playback_segment.py tool_lateralus_intro blackhole  # play to BlackHole (for testing effects)

Dependencies:
    pip install sounddevice soundfile
"""

import sounddevice as sd
import soundfile as sf
import sys
import os

SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio-segments')


def find_device_by_name(name_fragment):
    """Find output device by name fragment."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if name_fragment.lower() in d['name'].lower() and d['max_output_channels'] >= 2:
            return i, d['name']
    return None, None


def list_segments():
    """List available audio segments."""
    if not os.path.exists(SEGMENTS_DIR):
        print("No segments directory found.")
        return
    wavs = sorted(f for f in os.listdir(SEGMENTS_DIR) if f.endswith('.wav'))
    if wavs:
        print("Available segments:")
        for w in wavs:
            print(f"  {w.replace('.wav', '')}")
    else:
        print("No segments recorded yet.")


def play(segment_name, output='speakers'):
    if not segment_name.endswith('.wav'):
        segment_name += '.wav'
    filepath = os.path.join(SEGMENTS_DIR, segment_name)

    if not os.path.exists(filepath):
        print(f"Not found: {filepath}")
        list_segments()
        sys.exit(1)

    audio, sr = sf.read(filepath)
    duration = len(audio) / sr

    # Choose output device
    if output == 'blackhole':
        device_id, device_name = find_device_by_name('blackhole')
        if device_id is None:
            print("ERROR: Could not find BlackHole output device")
            print("\nAvailable output devices:")
            for i, d in enumerate(sd.query_devices()):
                if d['max_output_channels'] > 0:
                    print(f"  [{i}] {d['name']} ({d['max_output_channels']}ch)")
            sys.exit(1)
    else:
        device_id = None  # system default
        device_name = "default speakers"

    print(f"Playing: {segment_name} ({duration:.1f}s)")
    print(f"Output: {device_name}")
    print("Press Ctrl+C to stop\n")

    try:
        sd.play(audio, sr, device=device_id)
        sd.wait()
        print("Done.")
    except KeyboardInterrupt:
        sd.stop()
        print("\nStopped.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python playback_segment.py <segment_name> [speakers|blackhole]")
        print()
        list_segments()
        sys.exit(1)

    segment = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'speakers'
    play(segment, output)
