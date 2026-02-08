#!/usr/bin/env python3
"""
Tap-to-annotate: Record feeling-based annotations on audio segments.

Play a song and tap SPACE to mark moments. Each run records one "layer"
(e.g., beat, airy, heavy, tension). Multiple runs build up a rich
multi-layer annotation.

Usage:
    python annotate_segment.py <audio_file.wav> <layer_name>

Examples:
    python annotate_segment.py "Opiate Intro.wav" beat
    python annotate_segment.py "Opiate Intro.wav" airy
    python annotate_segment.py "Opiate Intro.wav" heavy

Annotations are saved as YAML next to the WAV file.

Controls:
    SPACE  - Mark a moment (tap in rhythm!)
    r      - Restart from beginning
    q      - Quit and save

Dependencies:
    pip install sounddevice soundfile pyyaml
"""

import sys
import os
import time
import threading
import tty
import termios
import select
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml

SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio-segments')


def get_key_nonblocking(timeout=0.01):
    """Read a single keypress without blocking, or return None."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class TapAnnotator:
    def __init__(self, filepath, layer_name):
        self.filepath = filepath
        self.layer_name = layer_name
        self.annotations_path = Path(filepath).with_suffix('.annotations.yaml')

        # Load audio
        self.audio, self.sr = sf.read(filepath)
        self.duration = len(self.audio) / self.sr

        # State
        self.taps = []
        self.playback_start_time = None
        self.is_playing = False
        self.is_done = False

        # Load existing annotations
        self.all_annotations = {}
        if self.annotations_path.exists():
            with open(self.annotations_path) as f:
                self.all_annotations = yaml.safe_load(f) or {}

        existing_count = len(self.all_annotations.get(layer_name, []))
        if existing_count > 0:
            print(f"\nNote: Layer '{layer_name}' already has {existing_count} marks.")
            print("New taps will REPLACE the existing layer.")

    def save(self):
        """Save annotations to YAML file."""
        if self.taps:
            # Round to millisecond precision
            self.all_annotations[self.layer_name] = [round(t, 3) for t in sorted(self.taps)]
        elif self.layer_name in self.all_annotations:
            # User ran but didn't tap — don't delete existing
            pass

        with open(self.annotations_path, 'w') as f:
            yaml.dump(self.all_annotations, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved to: {self.annotations_path}")
        print(f"Layers: {list(self.all_annotations.keys())}")
        for layer, taps in self.all_annotations.items():
            print(f"  {layer}: {len(taps)} marks")

    def play_audio(self):
        """Play audio in background thread."""
        import sounddevice as sd
        try:
            sd.play(self.audio, self.sr)
            sd.wait()
            self.is_playing = False
        except Exception as e:
            print(f"\nPlayback error: {e}")
            self.is_playing = False

    def restart(self):
        """Restart playback from beginning."""
        import sounddevice as sd
        sd.stop()
        self.taps = []
        time.sleep(0.1)
        self.is_playing = True
        self.playback_start_time = time.time()
        self.playback_thread = threading.Thread(target=self.play_audio, daemon=True)
        self.playback_thread.start()
        print("\r\033[K  Restarted. Taps cleared.", end='', flush=True)

    def run(self):
        import sounddevice as sd

        filename = Path(self.filepath).name
        print(f"\n{'='*60}")
        print(f"  Annotating: {filename}")
        print(f"  Layer: {self.layer_name}")
        print(f"  Duration: {self.duration:.1f}s")
        print(f"{'='*60}")
        print(f"\n  SPACE = tap a moment  |  R = restart  |  Q = save & quit\n")
        print(f"  Playback starting in 2 seconds...")
        time.sleep(2)

        # Start playback
        self.is_playing = True
        self.playback_start_time = time.time()
        self.playback_thread = threading.Thread(target=self.play_audio, daemon=True)
        self.playback_thread.start()

        tap_count = 0

        try:
            while not self.is_done:
                key = get_key_nonblocking(timeout=0.005)

                if key == ' ':
                    if self.is_playing and self.playback_start_time:
                        tap_time = time.time() - self.playback_start_time
                        if 0 <= tap_time <= self.duration:
                            self.taps.append(tap_time)
                            tap_count += 1
                            # Show progress inline
                            print(f"\r\033[K  [{tap_time:6.2f}s] TAP #{tap_count}", end='', flush=True)

                elif key == 'r':
                    self.restart()
                    tap_count = 0

                elif key == 'q' or key == '\x03':  # q or Ctrl+C
                    self.is_done = True

                # Check if playback finished
                if not self.is_playing and self.playback_start_time:
                    print(f"\n\n  Playback finished. {len(self.taps)} taps recorded.")
                    print(f"  Press Q to save, R to redo.")

                    # Wait for q or r
                    while not self.is_done:
                        key = get_key_nonblocking(timeout=0.05)
                        if key == 'q' or key == '\x03':
                            self.is_done = True
                        elif key == 'r':
                            self.restart()
                            tap_count = 0
                            break

        except KeyboardInterrupt:
            pass
        finally:
            sd.stop()

        print(f"\n\n  Final: {len(self.taps)} taps for layer '{self.layer_name}'")
        self.save()


def main():
    if len(sys.argv) < 3:
        print("Usage: python annotate_segment.py <audio_file> <layer_name>")
        print("\nExamples:")
        print('  python annotate_segment.py "Opiate Intro.wav" beat')
        print('  python annotate_segment.py "Opiate Intro.wav" airy')
        print('  python annotate_segment.py "Opiate Intro.wav" heavy')
        print("\nCommon layers:")
        print("  beat    - tap on every beat you feel")
        print("  kick    - tap on kick drum hits")
        print("  snare   - tap on snare hits")
        print("  airy    - tap when it feels airy/open")
        print("  heavy   - tap when it feels heavy/dense")
        print("  tension - tap when tension is building")
        print("  drop    - tap on drops/releases")
        sys.exit(1)

    filepath = sys.argv[1]

    # If not absolute, check in segments dir
    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        alt = os.path.join(SEGMENTS_DIR, filepath)
        if os.path.exists(alt):
            filepath = alt

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    layer = sys.argv[2]
    annotator = TapAnnotator(filepath, layer)
    annotator.run()


if __name__ == '__main__':
    main()
