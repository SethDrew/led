#!/usr/bin/env python3
"""
Test Essentia beat tracking and audio feature extraction.
Focus on real-time streaming mode and extensive feature set.
"""

import essentia
import essentia.standard as es
import essentia.streaming as ess
import numpy as np
import time

def test_essentia_beat_tracking(audio_path, ground_truth_beats=None):
    """
    Test Essentia's beat tracking with standard and streaming modes.

    Essentia has two modes:
    - standard: process entire audio (offline)
    - streaming: process frame-by-frame (real-time capable)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {audio_path}")
    print(f"{'='*60}")

    # Load audio with Essentia
    loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
    audio = loader()
    duration = len(audio) / 44100
    print(f"Duration: {duration:.2f}s, Sample rate: 44100 Hz")

    # ============================================================
    # 1. BEAT TRACKING (Standard Mode - Offline)
    # ============================================================
    print("\n--- Beat Tracking (Standard Mode - OFFLINE) ---")

    # BeatTrackerMultiFeature (most robust)
    start = time.time()
    bt = es.BeatTrackerMultiFeature()
    beats = bt(audio)
    elapsed = time.time() - start
    print(f"BeatTrackerMultiFeature: {len(beats)} beats, {elapsed:.3f}s")
    print(f"First 10 beats: {beats[:10]}")

    # BeatTrackerDegara (Degara's method)
    start = time.time()
    bt_degara = es.BeatTrackerDegara()
    beats_degara = bt_degara(audio)
    elapsed = time.time() - start
    print(f"BeatTrackerDegara: {len(beats_degara)} beats, {elapsed:.3f}s")

    # RhythmExtractor2013 (comprehensive rhythm analysis)
    start = time.time()
    rhythm = es.RhythmExtractor2013()
    bpm, beat_times, confidence, estimates, intervals = rhythm(audio)
    elapsed = time.time() - start
    print(f"RhythmExtractor2013: {bpm:.1f} BPM, {len(beat_times)} beats, confidence: {confidence:.3f}, {elapsed:.3f}s")

    # Percival2014 (tempo estimation)
    try:
        start = time.time()
        percival = es.Percival2014(sampleRate=44100)
        bpm_percival, ticks = percival(audio)
        elapsed = time.time() - start
        print(f"Percival2014: {bpm_percival:.1f} BPM, {len(ticks)} ticks, {elapsed:.3f}s")
    except Exception as e:
        print(f"Percival2014: ERROR - {e}")

    # ============================================================
    # 2. TEMPO ESTIMATION
    # ============================================================
    print("\n--- Tempo Estimation ---")

    # PercivalBpmEstimator
    start = time.time()
    percival_bpm = es.PercivalBpmEstimator(sampleRate=44100)
    bpm_est = percival_bpm(audio)
    elapsed = time.time() - start
    print(f"PercivalBpmEstimator: {bpm_est:.1f} BPM, {elapsed:.3f}s")

    # ============================================================
    # 3. ONSET DETECTION (can be real-time)
    # ============================================================
    print("\n--- Onset Detection ---")

    # OnsetDetection with different methods
    for method in ['hfc', 'complex', 'flux', 'melflux', 'rms']:
        try:
            # Compute onset detection function
            w = es.Windowing(type='hann')
            fft = es.FFT()
            c2p = es.CartesianToPolar()
            onset_det = es.OnsetDetection(method=method)

            # Frame-by-frame processing
            frame_size = 1024
            hop_size = 512
            onsets = []

            for frame_idx in range(0, len(audio) - frame_size, hop_size):
                frame = audio[frame_idx:frame_idx+frame_size]
                spectrum = fft(w(frame))
                mag, phase = c2p(spectrum)
                onset_val = onset_det(spectrum, phase)
                onsets.append(onset_val)

            # Peak picking
            onset_times = []
            threshold = np.mean(onsets) + 1.5 * np.std(onsets)
            for i in range(1, len(onsets)-1):
                if onsets[i] > threshold and onsets[i] > onsets[i-1] and onsets[i] > onsets[i+1]:
                    onset_times.append(i * hop_size / 44100)

            print(f"  {method:10s}: {len(onset_times)} onsets")
        except Exception as e:
            print(f"  {method:10s}: ERROR - {e}")

    # OnsetRate (estimates onset rate)
    try:
        onset_rate = es.OnsetRate()
        rate, onset_times_auto = onset_rate(audio)
        print(f"OnsetRate: {len(onset_times_auto)} onsets, rate: {rate:.2f} onsets/sec")
    except Exception as e:
        print(f"OnsetRate: ERROR - {e}")

    # ============================================================
    # 4. STREAMING MODE (Real-Time Capable)
    # ============================================================
    print("\n--- Streaming Mode (REAL-TIME CAPABLE) ---")

    # Create streaming network
    pool = essentia.Pool()

    # Audio source
    audio_gen = ess.VectorInput(audio)

    # Windowing and FFT
    framecutter = ess.FrameCutter(frameSize=1024, hopSize=512)
    window = ess.Windowing(type='hann')
    fft = ess.FFT()
    c2p = ess.CartesianToPolar()

    # Onset detection
    onset_hfc = ess.OnsetDetection(method='hfc')

    # Connect network
    audio_gen.data >> framecutter.signal
    framecutter.frame >> window.frame >> fft.frame
    fft.fft >> c2p.complex
    (c2p.magnitude, c2p.phase) >> onset_hfc.spectrum, onset_hfc.phase
    onset_hfc.onsetDetection >> (pool, 'onsets.hfc')

    # Run streaming
    start = time.time()
    essentia.run(audio_gen)
    elapsed = time.time() - start

    onset_values = pool['onsets.hfc']
    print(f"Streaming mode processed {len(onset_values)} frames in {elapsed:.3f}s")
    print(f"Real-time factor: {(duration/elapsed):.2f}x (>1.0 = faster than real-time)")

    # ============================================================
    # 5. ADVANCED FEATURES (Essentia's strength)
    # ============================================================
    print("\n--- Advanced Audio Features ---")

    # Spectral features
    spec_centroid = es.Centroid(range=22050)
    spec_rolloff = es.RollOff()
    spec_flux = es.Flux()

    # MFCC
    mfcc_extractor = es.MFCC()

    # Dissonance (harmony)
    dissonance = es.Dissonance()

    # Dynamic complexity
    dynamic_complexity = es.DynamicComplexity()

    print("Essentia provides 100+ audio features including:")
    print("  - Spectral: centroid, rolloff, flux, flatness, crest, entropy")
    print("  - Temporal: energy, zero-crossing rate, RMS")
    print("  - Rhythm: beat tracking, tempo, rhythm patterns")
    print("  - Tonal: pitch, harmony, dissonance, key estimation")
    print("  - Timbre: MFCC, spectral contrast, tristimulus")
    print("  - Loudness: EBU R128, replay gain")

    # ============================================================
    # 6. ACCURACY vs GROUND TRUTH
    # ============================================================
    if ground_truth_beats is not None:
        print("\n--- Accuracy vs Ground Truth ---")
        gt_beats = np.array(ground_truth_beats)

        for name, detected in [
            ("BeatTrackerMultiFeature", beats),
            ("BeatTrackerDegara", beats_degara),
            ("RhythmExtractor2013", beat_times),
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

            print(f"{name:25s}: P={precision:.3f} R={recall:.3f} F={f_measure:.3f}")

    return {
        'tempo': bpm,
        'beats': beat_times,
        'confidence': confidence,
        'onset_values': onset_values
    }


def test_essentia_parameters(audio_path):
    """
    Test different Essentia parameters.
    """
    print(f"\n{'='*60}")
    print("Testing Parameter Variations")
    print(f"{'='*60}")

    loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
    audio = loader()

    # Different beat tracking parameters
    print("\nBeatTrackerMultiFeature with different parameters:")

    # Default
    bt1 = es.BeatTrackerMultiFeature()
    beats1 = bt1(audio)
    print(f"Default: {len(beats1)} beats")

    # Adjust max/min tempo
    bt2 = es.BeatTrackerMultiFeature(minTempo=80, maxTempo=160)
    beats2 = bt2(audio)
    print(f"Tempo range 80-160: {len(beats2)} beats")

    # RhythmExtractor2013 parameters
    print("\nRhythmExtractor2013 with different parameters:")

    r1 = es.RhythmExtractor2013()
    bpm1, beats1, conf1, est1, int1 = r1(audio)
    print(f"Default: {bpm1:.1f} BPM")

    r2 = es.RhythmExtractor2013(method='multifeature', minTempo=80, maxTempo=160)
    bpm2, beats2, conf2, est2, int2 = r2(audio)
    print(f"Constrained (80-160): {bpm2:.1f} BPM")


if __name__ == "__main__":
    import yaml

    # Test on Opiate Intro (rock)
    opiate_path = "/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.wav"

    # Load ground truth
    with open("/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.annotations.yaml") as f:
        annotations = yaml.safe_load(f)
        ground_truth = annotations.get('consistent-beat', [])

    print("="*60)
    print("ESSENTIA BEAT TRACKING TEST")
    print("="*60)
    print("\nKey findings:")
    print("- Standard mode: Full-featured but offline")
    print("- Streaming mode: Real-time capable, frame-by-frame processing")
    print("- 100+ audio features (spectral, temporal, rhythm, tonal, timbre)")
    print("- Multiple beat tracking algorithms (MultiFeature, Degara, etc.)")
    print("- Excellent for comprehensive audio analysis")

    results = test_essentia_beat_tracking(opiate_path, ground_truth)

    # Test parameter variations
    test_essentia_parameters(opiate_path)

    # Test on electronic
    print("\n\n")
    electronic_path = "/Users/KO16K39/Documents/led/interactivity-research/audio-segments/electronic_beat.wav"
    results2 = test_essentia_beat_tracking(electronic_path)
