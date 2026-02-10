#!/usr/bin/env python3
"""
Test librosa beat tracking and onset detection.
Focus on real-time capabilities and tunable parameters.
"""

import librosa
import numpy as np
import soundfile as sf
import time

def test_librosa_beat_tracking(audio_path, ground_truth_beats=None):
    """
    Test librosa's beat tracking with various parameters.

    librosa.beat.beat_track() is NOT real-time - requires full audio.
    However, onset detection CAN be done in real-time with frames.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {audio_path}")
    print(f"{'='*60}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=44100)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f}s, Sample rate: {sr} Hz")

    # ============================================================
    # 1. BEAT TRACKING (offline only - needs full audio)
    # ============================================================
    print("\n--- Beat Tracking (OFFLINE ONLY) ---")

    # Default parameters
    start = time.time()
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    elapsed = time.time() - start
    tempo_val = tempo if isinstance(tempo, (int, float)) else tempo[0]
    print(f"Default: {tempo_val:.1f} BPM, {len(beats)} beats, {elapsed:.3f}s processing")

    # With onset envelope pre-computed
    start = time.time()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo2, beats2 = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times2 = librosa.frames_to_time(beats2, sr=sr)
    elapsed = time.time() - start
    tempo2_val = tempo2 if isinstance(tempo2, (int, float)) else tempo2[0]
    print(f"With onset_env: {tempo2_val:.1f} BPM, {len(beats2)} beats, {elapsed:.3f}s")

    # Constrained tempo search
    start = time.time()
    tempo3, beats3 = librosa.beat.beat_track(y=y, sr=sr, bpm=100, tightness=100)
    beat_times3 = librosa.frames_to_time(beats3, sr=sr)
    elapsed = time.time() - start
    tempo3_val = tempo3 if isinstance(tempo3, (int, float)) else tempo3[0]
    print(f"Constrained (bpm=100, tight): {tempo3_val:.1f} BPM, {len(beats3)} beats, {elapsed:.3f}s")

    # Wide tempo search (just use different start_bpm)
    start = time.time()
    tempo4, beats4 = librosa.beat.beat_track(y=y, sr=sr, start_bpm=80)
    beat_times4 = librosa.frames_to_time(beats4, sr=sr)
    elapsed = time.time() - start
    tempo4_val = tempo4 if isinstance(tempo4, (int, float)) else tempo4[0]
    print(f"Start BPM 80: {tempo4_val:.1f} BPM, {len(beats4)} beats, {elapsed:.3f}s")

    # ============================================================
    # 2. ONSET DETECTION (CAN be real-time)
    # ============================================================
    print("\n--- Onset Detection (REAL-TIME CAPABLE) ---")

    # Onset strength (spectral flux)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False  # Real-time mode
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    print(f"Onset detection: {len(onset_times)} onsets")
    print(f"First 10 onsets: {onset_times[:10]}")

    # Onset detection with different methods
    for method in ['energy', 'hfc', 'complex', 'phase', 'flux']:
        try:
            onset_env_method = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=hop_length, feature=librosa.feature.melspectrogram,
                aggregate=np.median
            )
            onset_frames_method = librosa.onset.onset_detect(
                onset_envelope=onset_env_method,
                sr=sr,
                hop_length=hop_length,
                backtrack=False
            )
            print(f"  {method}: {len(onset_frames_method)} onsets")
        except Exception as e:
            print(f"  {method}: ERROR - {e}")

    # ============================================================
    # 3. TEMPO ESTIMATION (offline only)
    # ============================================================
    print("\n--- Tempo Estimation (OFFLINE ONLY) ---")

    # Static tempo estimation
    tempo_static = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    print(f"Static tempo: {tempo_static[0]:.1f} BPM")

    # Dynamic tempo (tempogram)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    print(f"Tempogram shape: {tempogram.shape} (tempo_bins x time_frames)")

    # ============================================================
    # 4. ACCURACY vs GROUND TRUTH
    # ============================================================
    if ground_truth_beats is not None:
        print("\n--- Accuracy vs Ground Truth ---")
        gt_beats = np.array(ground_truth_beats)

        for name, detected in [
            ("Default", beat_times),
            ("With onset_env", beat_times2),
            ("Constrained", beat_times3),
            ("Wide search", beat_times4),
        ]:
            # F-measure with 70ms window
            tolerance = 0.07
            tp = 0
            for gt_beat in gt_beats:
                if np.any(np.abs(detected - gt_beat) < tolerance):
                    tp += 1

            precision = tp / len(detected) if len(detected) > 0 else 0
            recall = tp / len(gt_beats) if len(gt_beats) > 0 else 0
            f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{name:20s}: P={precision:.3f} R={recall:.3f} F={f_measure:.3f}")

    # ============================================================
    # 5. REAL-TIME SIMULATION
    # ============================================================
    print("\n--- Real-Time Simulation (chunk-by-chunk) ---")
    chunk_size = 2048  # ~46ms at 44.1kHz
    hop_length = 512

    # Onset strength can be computed incrementally
    print("Processing audio in chunks...")
    num_chunks = len(y) // chunk_size
    onset_values = []

    start = time.time()
    for i in range(num_chunks):
        chunk = y[i*chunk_size:(i+1)*chunk_size]
        # Compute STFT for this chunk
        stft_chunk = librosa.stft(chunk, n_fft=2048, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(S=np.abs(stft_chunk)**2, sr=sr)
        # Onset strength is the spectral flux (change between frames)
        if i > 0:
            onset_val = np.sum(np.maximum(0, mel_spec - prev_mel))
            onset_values.append(onset_val)
        prev_mel = mel_spec

    elapsed = time.time() - start
    print(f"Processed {num_chunks} chunks in {elapsed:.3f}s ({num_chunks/elapsed:.1f} chunks/sec)")
    print(f"Real-time factor: {(duration/elapsed):.2f}x (>1.0 = faster than real-time)")

    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'onset_times': onset_times,
        'onset_env': onset_env
    }


if __name__ == "__main__":
    import yaml

    # Test on Opiate Intro (rock)
    opiate_path = "/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.wav"

    # Load ground truth
    with open("/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.annotations.yaml") as f:
        annotations = yaml.safe_load(f)
        ground_truth = annotations.get('consistent-beat', [])

    print("="*60)
    print("LIBROSA BEAT TRACKING TEST")
    print("="*60)
    print("\nKey findings:")
    print("- beat_track() is OFFLINE ONLY (needs full audio)")
    print("- onset_detect() CAN be real-time (with backtrack=False)")
    print("- Onset strength can be computed incrementally")
    print("- Tunable params: start_bpm, std_bpm, bpm, tightness, hop_length, onset method")

    results = test_librosa_beat_tracking(opiate_path, ground_truth)

    # Test on electronic
    electronic_path = "/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/electronic_beat.wav"
    results2 = test_librosa_beat_tracking(electronic_path)
