#!/usr/bin/env python3
"""Record audio clips from the Mac mic, saved to clips/.

Usage:
    python record.py                  # prompts for name, records until Enter
    python record.py my_clip          # records clips/my_clip.wav until Enter
    python record.py my_clip 10       # records 10 seconds
"""

import sys
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 44100
CLIPS_DIR = os.path.join(os.path.dirname(__file__), 'clips')


def record(name, duration=None):
    path = os.path.join(CLIPS_DIR, f'{name}.wav')
    print(f'  Recording to: {path}')
    print(f'  {"Recording " + str(duration) + "s..." if duration else "Recording... press Enter to stop."}')

    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    stream = sd.InputStream(channels=1, samplerate=RATE, blocksize=1024,
                            callback=callback)
    stream.start()

    try:
        if duration:
            time.sleep(duration)
        else:
            # Poll instead of blocking on input() so signals work
            import select
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.readline()
                    break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stream.stop()
        stream.close()
        # Always save whatever we captured
        if chunks:
            audio = np.concatenate(chunks, axis=0)
            sf.write(path, audio, RATE)
            dur = len(audio) / RATE
            print(f'\n  Saved: {name}.wav ({dur:.1f}s)')
        else:
            print('\n  No audio captured.')

    return path


def main():
    os.makedirs(CLIPS_DIR, exist_ok=True)

    if len(sys.argv) >= 2:
        name = sys.argv[1]
    else:
        name = input('  Clip name: ').strip()
        if not name:
            name = f'clip_{int(time.time())}'

    duration = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    record(name, duration)


if __name__ == '__main__':
    main()
