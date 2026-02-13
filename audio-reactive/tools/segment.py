#!/usr/bin/env python3
"""
Audio segment management CLI.

Usage:
    python segment.py web                           # Browser-based explorer
    python segment.py list                          # Show catalog
    python segment.py record                        # Record from BlackHole
    python segment.py trim <file> [start_seconds]   # Trim silence
    python segment.py play <file> [--annotate layer] [--panel P] [--show-beats]

Dependencies:
    pip install sounddevice soundfile numpy pyyaml
    (play also needs: librosa matplotlib)
"""

import sys
import os
import argparse
import wave
import struct
import math
from pathlib import Path
from datetime import datetime

SEGMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research', 'audio-segments')


def resolve_segment_path(name):
    """Resolve a segment name/filename/path to an absolute filepath.

    Accepts:
      - Absolute path: /Users/.../foo.wav
      - Relative path: ../audio-segments/foo.wav
      - Bare name: "Opiate Intro" or "Opiate Intro.wav"
    """
    # Absolute or existing relative path
    if os.path.isabs(name) and os.path.exists(name):
        return name
    if os.path.exists(name):
        return os.path.abspath(name)

    # Try in segments dir
    candidate = os.path.join(SEGMENTS_DIR, name)
    if os.path.exists(candidate):
        return candidate

    # Try appending .wav
    if not name.endswith('.wav'):
        candidate = os.path.join(SEGMENTS_DIR, name + '.wav')
        if os.path.exists(candidate):
            return candidate

    return None


def find_blackhole_device():
    """Auto-detect BlackHole input device by name."""
    import sounddevice as sd
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if 'blackhole' in d['name'].lower() and d['max_input_channels'] >= 2:
            return i, d['name']
    return None, None


# ── list ──────────────────────────────────────────────────────────────

def cmd_list(args):
    """Print all WAV files in audio-segments, with catalog metadata where available."""
    import yaml

    # Load catalog for metadata lookup
    catalog_path = os.path.join(SEGMENTS_DIR, 'catalog.yaml')
    catalog = []
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f) or []

    # Index catalog by filename
    catalog_by_file = {}
    for entry in catalog:
        fname = entry.get('filename', '')
        catalog_by_file[fname] = entry

    # Discover all WAV files on disk
    wav_files = sorted(Path(SEGMENTS_DIR).glob('*.wav'))
    if not wav_files:
        print("No WAV files found in audio-segments/")
        return

    # Header
    print(f"\n  {'Name':<25s} {'Duration':>8s}  {'Artist':<20s} {'Song':<25s} {'Notes'}")
    print(f"  {'-'*25} {'-'*8}  {'-'*20} {'-'*25} {'-'*30}")

    cataloged = 0
    for wav_path in wav_files:
        fname = wav_path.name
        entry = catalog_by_file.get(fname)

        if entry:
            cataloged += 1
            name = entry.get('id', wav_path.stem)
            dur = entry.get('duration_seconds', 0)
            artist = entry.get('artist', '')
            song = entry.get('song', '')
            notes = entry.get('notes', '')
        else:
            name = wav_path.stem
            artist = ''
            song = ''
            notes = '(uncataloged)'
            # Get duration from WAV header
            try:
                with wave.open(str(wav_path), 'rb') as wf:
                    dur = wf.getnframes() / wf.getframerate()
            except Exception:
                dur = 0

        if len(notes) > 50:
            notes = notes[:47] + '...'
        print(f"  {name:<25s} {dur:>6.1f}s  {artist:<20s} {song:<25s} {notes}")

    uncataloged = len(wav_files) - cataloged
    print(f"\n  {len(wav_files)} files ({cataloged} cataloged, {uncataloged} uncataloged)\n")


# ── record ────────────────────────────────────────────────────────────

def cmd_record(args):
    """Record audio from BlackHole. Interactive prompts for metadata."""
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import yaml

    SAMPLE_RATE = 44100
    CHANNELS = 2

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


# ── trim ──────────────────────────────────────────────────────────────

def analyze_audio(filepath, threshold=500, preserve_long_silence=10.0):
    """Find where audio becomes significant. Returns start time or None."""
    with wave.open(filepath, 'rb') as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()

        print(f"Analyzing audio: {rate}Hz, {channels}ch, {sampwidth}byte")

        window_size = int(rate * 0.1)  # 100ms windows
        frame_count = 0

        while True:
            frames = wav.readframes(window_size)
            if not frames:
                break

            if sampwidth == 2:
                values = struct.unpack(f'<{len(frames)//2}h', frames)
            else:
                values = struct.unpack(f'<{len(frames)}B', frames)

            sum_sq = sum(v * v for v in values)
            rms = math.sqrt(sum_sq / len(values)) if values else 0

            if rms > threshold:
                start_time = frame_count / rate

                if start_time >= preserve_long_silence:
                    print(f"Long silence detected ({start_time:.2f}s >= {preserve_long_silence}s)")
                    print(f"Preserving original file (likely intentional silence)")
                    return None

                print(f"Music detected at: {start_time:.2f} seconds")
                print(f"RMS level: {rms:.1f}")
                return start_time

            frame_count += window_size

        print(f"No audio detected in file")
        return None


def trim_audio(input_path, output_path, start_seconds):
    """Trim audio file from start_seconds onwards."""
    with wave.open(input_path, 'rb') as wav_in:
        params = wav_in.getparams()
        rate = wav_in.getframerate()

        start_frame = int(start_seconds * rate)
        wav_in.setpos(start_frame)

        remaining_frames = params.nframes - start_frame
        audio_data = wav_in.readframes(remaining_frames)

        with wave.open(output_path, 'wb') as wav_out:
            wav_out.setparams(params)
            wav_out.setnframes(remaining_frames)
            wav_out.writeframes(audio_data)

        new_duration = remaining_frames / rate
        print(f"\nTrimmed audio saved to: {output_path}")
        print(f"New duration: {new_duration:.2f} seconds")
        print(f"Removed: {start_seconds:.2f} seconds from start")


def cmd_trim(args):
    """Trim silence from audio file."""
    filepath = resolve_segment_path(args.file)
    if filepath is None:
        print(f"File not found: {args.file}")
        sys.exit(1)

    if args.start is not None:
        start_time = args.start
        print(f"Manual trim at {start_time:.2f} seconds")
    else:
        print(f"Analyzing: {filepath}")
        start_time = analyze_audio(filepath)
        if start_time is None:
            print("Skipping trim (preserving original)")
            return

    output_file = filepath.replace('.wav', '_trimmed.wav')
    trim_audio(filepath, output_file, start_time)


# ── play ──────────────────────────────────────────────────────────────

def cmd_play(args):
    """Launch interactive viewer (lazy-imports viewer.py)."""
    filepath = resolve_segment_path(args.file)
    if filepath is None:
        print(f"File not found: {args.file}")
        sys.exit(1)

    from viewer import SyncedVisualizer

    # Optional LED output
    led_effect = None
    led_output = None
    if args.port and args.effect:
        comparison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'comparison')
        sys.path.insert(0, comparison_dir)
        from runner import SerialLEDOutput, get_effect_registry

        effects = get_effect_registry()
        if args.effect not in effects:
            print(f"Unknown effect: {args.effect}. Available: {', '.join(effects.keys())}")
            sys.exit(1)

        led_effect = effects[args.effect](num_leds=args.leds, sample_rate=44100)
        led_output = SerialLEDOutput(args.port, args.leds)
        print(f"  LED effect: {led_effect.name} → {args.port} ({args.leds} LEDs)")

    viz = SyncedVisualizer(
        filepath,
        focus_panel=args.panel,
        show_beats=args.show_beats,
        annotate_layer=args.annotate,
        led_effect=led_effect,
        led_output=led_output,
        led_brightness=args.brightness,
    )
    viz.run()


# ── stems ─────────────────────────────────────────────────────────────

def cmd_stems(args):
    """Launch stem visualizer (demucs separation + interactive viewer)."""
    import subprocess
    import soundfile as sf
    import librosa

    filepath = resolve_segment_path(args.file)
    if filepath is None:
        print(f"File not found: {args.file}")
        sys.exit(1)

    # Stem output directory: audio-segments/separated/htdemucs/<stem_name>/
    separated_dir = os.path.join(SEGMENTS_DIR, 'separated')
    stem_name = Path(filepath).stem
    stem_dir = os.path.join(separated_dir, 'htdemucs', stem_name)

    # Check if all 4 stems exist
    stem_names = ['drums', 'bass', 'vocals', 'other']
    expected = [f'{n}.wav' for n in stem_names]
    if not all(os.path.exists(os.path.join(stem_dir, f)) for f in expected):
        print(f"Running demucs separation (first run may take ~25s on CPU)...")
        subprocess.run(
            [sys.executable, '-m', 'demucs', '-n', 'htdemucs',
             '-o', separated_dir, filepath],
            check=True
        )

    # Load stems into dicts
    print(f"Loading stems from: {stem_dir}")
    stems_playback = {}
    stems_mono = {}
    for name in stem_names:
        stem_path = os.path.join(stem_dir, f'{name}.wav')
        stems_playback[name] = sf.read(stem_path)
        stems_mono[name] = librosa.load(stem_path, sr=None, mono=True)

    from viewer import StemVisualizer
    viz = StemVisualizer(filepath, stem_names, stems_playback, stems_mono)
    viz.run()


# ── hpss ─────────────────────────────────────────────────────────────

def cmd_hpss(args):
    """Launch HPSS viewer (harmonic/percussive separation, no ML)."""
    import numpy as np
    import soundfile as sf
    import librosa

    filepath = resolve_segment_path(args.file)
    if filepath is None:
        print(f"File not found: {args.file}")
        sys.exit(1)

    print(f"Computing HPSS for: {Path(filepath).name}")

    # Load audio
    y_play, sr_play = sf.read(filepath)
    y_mono, sr_mono = librosa.load(filepath, sr=None, mono=True)

    # HPSS on mono for spectrograms
    D = librosa.stft(y_mono)
    H, P = librosa.decompose.hpss(D)
    y_h_mono = librosa.istft(H, length=len(y_mono))
    y_p_mono = librosa.istft(P, length=len(y_mono))

    # HPSS per channel for stereo playback
    if y_play.ndim == 1:
        y_play = y_play[:, np.newaxis]
    n_ch = y_play.shape[1]
    y_h_play = np.zeros_like(y_play)
    y_p_play = np.zeros_like(y_play)
    for ch in range(n_ch):
        D_ch = librosa.stft(y_play[:, ch])
        H_ch, P_ch = librosa.decompose.hpss(D_ch)
        y_h_play[:, ch] = librosa.istft(H_ch, length=y_play.shape[0])
        y_p_play[:, ch] = librosa.istft(P_ch, length=y_play.shape[0])

    # Squeeze back to 1D if original was mono
    if n_ch == 1:
        y_h_play = y_h_play.squeeze()
        y_p_play = y_p_play.squeeze()

    stem_names = ['harmonic', 'percussive']
    stems_playback = {
        'harmonic': (y_h_play, sr_play),
        'percussive': (y_p_play, sr_play),
    }
    stems_mono = {
        'harmonic': (y_h_mono, sr_mono),
        'percussive': (y_p_mono, sr_mono),
    }

    print("HPSS complete.")

    from viewer import StemVisualizer
    viz = StemVisualizer(filepath, stem_names, stems_playback, stems_mono)
    viz.run()


# ── web ──────────────────────────────────────────────────────────────

def cmd_web(args):
    """Launch browser-based audio explorer."""
    from web_viewer import run_server
    run_server(port=args.port, host=args.host, no_browser=args.no_browser)


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Audio segment management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python segment.py web                                   # Browser explorer
  python segment.py list
  python segment.py record
  python segment.py trim "Opiate Intro.wav"
  python segment.py trim "Opiate Intro.wav" 2.5
  python segment.py play "Opiate Intro.wav"
  python segment.py play "Opiate Intro.wav" --annotate beat
  python segment.py play "Opiate Intro.wav" --panel onset --show-beats
  python segment.py stems electronic_beat.wav
  python segment.py hpss electronic_beat.wav
""",
    )

    subparsers = parser.add_subparsers(dest='command')

    # web
    p_web = subparsers.add_parser('web', help='Browser-based audio explorer')
    p_web.add_argument('--port', type=int, default=0, help='Server port (0=auto)')
    p_web.add_argument('--host', default='127.0.0.1', help='Bind address (default localhost)')
    p_web.add_argument('--no-browser', action='store_true', help='Skip opening browser')

    # list
    subparsers.add_parser('list', help='Show segment catalog')

    # record
    subparsers.add_parser('record', help='Record audio from BlackHole')

    # trim
    p_trim = subparsers.add_parser('trim', help='Trim silence from audio')
    p_trim.add_argument('file', help='Audio file (name, filename, or path)')
    p_trim.add_argument('start', nargs='?', type=float, default=None,
                        help='Start time in seconds (auto-detect if omitted)')

    # play
    p_play = subparsers.add_parser('play', help='Interactive visualizer + annotator')
    p_play.add_argument('file', help='Audio file (name, filename, or path)')
    p_play.add_argument('--annotate', metavar='LAYER',
                        help='Enable annotation mode for this layer (e.g., beat, airy)')
    p_play.add_argument('--panel',
                        choices=['waveform', 'spectrogram', 'bands', 'features', 'annotations'],
                        help='Maximize a specific panel')
    p_play.add_argument('--show-beats', action='store_true',
                        help='Show beat detection markers')
    p_play.add_argument('--port', help='Serial port for LED output (e.g., rfc2217://localhost:9012)')
    p_play.add_argument('--effect',
                        help='Audio effect to drive LEDs (e.g., wled_volume, wled_geq, wled_beat)')
    p_play.add_argument('--leds', type=int, default=30, help='Number of LEDs (default 30)')
    p_play.add_argument('--brightness', type=float, default=1.0,
                        help='LED brightness cap (0-1, default 1.0)')

    # stems
    p_stems = subparsers.add_parser('stems', help='Instrument stem visualizer (demucs)')
    p_stems.add_argument('file', help='Audio file (name, filename, or path)')

    # hpss
    p_hpss = subparsers.add_parser('hpss', help='Harmonic/percussive separation viewer')
    p_hpss.add_argument('file', help='Audio file (name, filename, or path)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        'web': cmd_web,
        'list': cmd_list,
        'record': cmd_record,
        'trim': cmd_trim,
        'play': cmd_play,
        'stems': cmd_stems,
        'hpss': cmd_hpss,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
