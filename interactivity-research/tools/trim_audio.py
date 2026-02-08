#!/usr/bin/env python3
"""
Audio trimming tool - removes silence/intro from wav files
"""
import wave
import struct
import sys
import math

def analyze_audio(filepath, threshold=500, preserve_long_silence=10.0):
    """Find where audio becomes significant

    Args:
        filepath: Path to wav file
        threshold: RMS threshold for detecting audio
        preserve_long_silence: If silence is longer than this (seconds), don't trim

    Returns:
        Start time in seconds, or None if should not trim (long silence)
    """
    with wave.open(filepath, 'rb') as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()

        print(f"Analyzing audio: {rate}Hz, {channels}ch, {sampwidth}byte")

        # Read in chunks
        window_size = int(rate * 0.1)  # 100ms windows
        frame_count = 0

        while True:
            frames = wav.readframes(window_size)
            if not frames:
                break

            # Convert to values
            if sampwidth == 2:  # 16-bit
                values = struct.unpack(f'<{len(frames)//2}h', frames)
            else:
                values = struct.unpack(f'<{len(frames)}B', frames)

            # Calculate RMS
            sum_sq = sum(v*v for v in values)
            rms = math.sqrt(sum_sq / len(values)) if values else 0

            if rms > threshold:
                start_time = frame_count / rate

                # Check if silence is too long (intentional)
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
    """Trim audio file from start_seconds onwards"""
    with wave.open(input_path, 'rb') as wav_in:
        params = wav_in.getparams()
        rate = wav_in.getframerate()

        # Skip to start position
        start_frame = int(start_seconds * rate)
        wav_in.setpos(start_frame)

        # Read remaining frames
        remaining_frames = params.nframes - start_frame
        audio_data = wav_in.readframes(remaining_frames)

        # Write trimmed file
        with wave.open(output_path, 'wb') as wav_out:
            wav_out.setparams(params)
            wav_out.setnframes(remaining_frames)
            wav_out.writeframes(audio_data)

        new_duration = remaining_frames / rate
        print(f"\nTrimmed audio saved to: {output_path}")
        print(f"New duration: {new_duration:.2f} seconds")
        print(f"Removed: {start_seconds:.2f} seconds from start")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python trim_audio.py <input.wav> [start_seconds]")
        print("  If start_seconds not provided, auto-detects music start")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        # Manual start time
        start_time = float(sys.argv[2])
        print(f"Manual trim at {start_time:.2f} seconds")
    else:
        # Auto-detect
        print(f"Analyzing: {input_file}")
        start_time = analyze_audio(input_file)

        if start_time is None:
            print("Skipping trim (preserving original)")
            sys.exit(0)

    # Create output filename
    output_file = input_file.replace('.wav', '_trimmed.wav')

    trim_audio(input_file, output_file, start_time)
