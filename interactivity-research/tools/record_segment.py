#!/usr/bin/env python3
"""
Record audio segment from BlackHole for testing.

Usage:
    python record_segment.py

Records stereo audio from BlackHole at 44.1kHz.
Press ENTER to stop recording, then fill in metadata.
Saves WAV + updates catalog.yaml.

Dependencies:
    pip install sounddevice soundfile numpy pyyaml
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import sys
import os
import yaml
from datetime import datetime

SAMPLE_RATE = 44100
CHANNELS = 2
SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio-segments')


def find_blackhole_device():
    """Auto-detect BlackHole input device by name."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if 'blackhole' in d['name'].lower() and d['max_input_channels'] >= 2:
            return i, d['name']
    return None, None


def record():
    device_id, device_name = find_blackhole_device()
    if device_id is None:
        print("ERROR: Could not find BlackHole device")
        print("\nAvailable input devices:")
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']} ({d['max_input_channels']}ch)")
        sys.exit(1)

    print(f"Recording from: {device_name} (device {device_id})")
    print(f"Format: {SAMPLE_RATE}Hz, {CHANNELS}ch, WAV")
    print("\nStart your music, then press ENTER to stop recording...")
    print("Recording NOW\n")

    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"  Audio status: {status}")
        frames.append(indata.copy())

    stream = sd.InputStream(
        device=device_id,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=callback
    )

    with stream:
        input()  # Block until enter

    if not frames:
        print("No audio recorded.")
        return

    audio = np.concatenate(frames)
    duration = len(audio) / SAMPLE_RATE
    print(f"Recorded {duration:.1f} seconds")

    # Get segment name
    name = input("Segment name (e.g., tool_lateralus_intro): ").strip()
    if not name:
        name = f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filename = f"{name}.wav"
    filepath = os.path.join(SEGMENTS_DIR, filename)

    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"Saved: {filepath}")

    # Optional metadata
    song = input("Song name (optional): ").strip()
    artist = input("Artist (optional): ").strip()
    genre = input("Genre (optional): ").strip()
    bpm = input("BPM if known (optional): ").strip()
    notes = input("Notes - what makes this interesting? (optional): ").strip()

    # Update catalog
    catalog_path = os.path.join(SEGMENTS_DIR, 'catalog.yaml')
    catalog = []
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f) or []

    entry = {
        'id': name,
        'filename': filename,
        'duration_seconds': round(duration, 1),
        'sample_rate': SAMPLE_RATE,
        'channels': CHANNELS,
        'recorded': datetime.now().isoformat(),
    }
    if song:
        entry['song'] = song
    if artist:
        entry['artist'] = artist
    if genre:
        entry['genre'] = genre
    if bpm:
        entry['bpm'] = int(bpm)
    if notes:
        entry['notes'] = notes

    catalog.append(entry)
    with open(catalog_path, 'w') as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False)

    print(f"\nCatalog updated: {catalog_path}")
    print("Done!")


if __name__ == '__main__':
    record()
