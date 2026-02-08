#!/usr/bin/env python3
"""
Beat Detector Validation Script

Tests the bass-band spectral flux beat detection algorithm from realtime_beat_led.py
against electronic and ambient music to identify failure modes and tuning needs.

Usage:
    source /Users/KO16K39/Documents/led/venv/bin/activate
    python beat_detector_validation.py
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ── Algorithm Parameters (from realtime_beat_led.py) ──
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024          # ~23ms per chunk
N_FFT = 2048
BASS_LOW_HZ = 20
BASS_HIGH_HZ = 250
FLUX_HISTORY_SEC = 3.0
MIN_BEAT_INTERVAL_SEC = 0.3  # ~200 BPM max
THRESHOLD_MULTIPLIER = 1.5

# ── Paths ──
BASE_DIR = Path("/Users/KO16K39/Documents/led/interactivity-research")
AUDIO_DIR = BASE_DIR / "audio-segments"
OUTPUT_DIR = BASE_DIR / "analysis"

# Audio files
AUDIO_FILES = {
    'electronic': AUDIO_DIR / "electronic_beat.wav",
    'ambient': AUDIO_DIR / "ambient.wav"
}


class BeatDetectorOffline:
    """
    Offline version of BeatDetector from realtime_beat_led.py
    Processes audio chunk-by-chunk to simulate real-time behavior.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT):
        self.sr = sample_rate
        self.n_fft = n_fft

        # Frequency bin indices for bass range
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self.bass_bins = np.where((freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ))[0]

        # State
        self.prev_spectrum = None
        self.flux_history = []
        self.max_history = int(FLUX_HISTORY_SEC * sample_rate / CHUNK_SIZE)
        self.last_beat_idx = -999999  # Use chunk index instead of time
        self.beat_count = 0

        # Windowing
        self.window = np.hanning(n_fft)

        # Audio buffer for overlapping FFT
        self.audio_buffer = np.zeros(n_fft)

        # Record all beats and flux values
        self.beat_timestamps = []
        self.all_flux = []
        self.all_thresholds = []

    def process_chunk(self, audio_chunk, chunk_idx):
        """Process an audio chunk. Returns (is_beat, flux_value, threshold)."""
        # Shift buffer and add new audio
        chunk_len = len(audio_chunk)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = audio_chunk

        # Windowed FFT
        windowed = self.audio_buffer * self.window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Extract bass band
        bass_spectrum = spectrum[self.bass_bins]

        if self.prev_spectrum is None:
            self.prev_spectrum = bass_spectrum
            self.all_flux.append(0.0)
            self.all_thresholds.append(0.0)
            return False, 0.0, 0.0

        # Spectral flux: sum of positive differences (half-wave rectified)
        diff = bass_spectrum - self.prev_spectrum
        flux = np.sum(np.maximum(diff, 0))
        self.prev_spectrum = bass_spectrum

        # Adaptive threshold
        self.flux_history.append(flux)
        if len(self.flux_history) > self.max_history:
            self.flux_history.pop(0)

        if len(self.flux_history) < 10:
            self.all_flux.append(flux)
            self.all_thresholds.append(0.0)
            return False, flux, 0.0

        mean_flux = np.mean(self.flux_history)
        std_flux = np.std(self.flux_history)
        threshold = mean_flux + THRESHOLD_MULTIPLIER * std_flux

        # Beat detection with minimum interval (in chunks)
        min_interval_chunks = int(MIN_BEAT_INTERVAL_SEC * self.sr / CHUNK_SIZE)
        is_beat = (flux > threshold and
                   (chunk_idx - self.last_beat_idx) > min_interval_chunks)

        if is_beat:
            self.last_beat_idx = chunk_idx
            self.beat_count += 1
            timestamp = chunk_idx * CHUNK_SIZE / self.sr
            self.beat_timestamps.append(timestamp)

        self.all_flux.append(flux)
        self.all_thresholds.append(threshold)

        return is_beat, flux, threshold


def analyze_beats(beat_timestamps, audio_duration):
    """Compute statistics from detected beats."""
    if len(beat_timestamps) < 2:
        return {
            'total_beats': len(beat_timestamps),
            'avg_bpm': 0.0,
            'interval_mean': 0.0,
            'interval_std': 0.0,
            'interval_cv': 0.0,
            'false_positive_periods': [],
            'long_gap_periods': []
        }

    # Beat intervals
    intervals = np.diff(beat_timestamps)

    # BPM from mean interval
    avg_bpm = 60.0 / np.mean(intervals) if len(intervals) > 0 else 0.0

    # Coefficient of variation (normalized std)
    interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0.0

    # Find suspicious periods
    # False positives: many beats in quick succession (< 0.2s)
    false_positive_periods = []
    for i in range(len(intervals)):
        if intervals[i] < 0.2:
            false_positive_periods.append((beat_timestamps[i], beat_timestamps[i+1]))

    # Long gaps: > 2 seconds without a beat (potential false negatives)
    long_gap_periods = []
    for i in range(len(intervals)):
        if intervals[i] > 2.0:
            long_gap_periods.append((beat_timestamps[i], beat_timestamps[i+1]))

    return {
        'total_beats': len(beat_timestamps),
        'avg_bpm': avg_bpm,
        'interval_mean': np.mean(intervals),
        'interval_std': np.std(intervals),
        'interval_cv': interval_cv,
        'false_positive_periods': false_positive_periods,
        'long_gap_periods': long_gap_periods
    }


def visualize_results(audio, sr, detector, stats, title, output_path):
    """Create visualization showing waveform, flux, and beat intervals."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'Beat Detection Analysis: {title}', fontsize=16, fontweight='bold')

    time_axis = np.arange(len(audio)) / sr
    chunk_time_axis = np.arange(len(detector.all_flux)) * CHUNK_SIZE / sr

    # 1. Waveform with detected beats
    ax = axes[0]
    ax.plot(time_axis, audio, color='gray', alpha=0.5, linewidth=0.5)
    for beat_time in detector.beat_timestamps:
        ax.axvline(beat_time, color='red', alpha=0.7, linewidth=2)
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Waveform with Detected Beats (n={stats["total_beats"]})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(audio) / sr)

    # 2. Spectral flux with threshold
    ax = axes[1]
    ax.plot(chunk_time_axis, detector.all_flux, label='Spectral Flux', color='blue', linewidth=1)
    ax.plot(chunk_time_axis, detector.all_thresholds, label='Adaptive Threshold',
            color='orange', linestyle='--', linewidth=1.5)
    for beat_time in detector.beat_timestamps:
        ax.axvline(beat_time, color='red', alpha=0.3, linewidth=1)
    ax.set_ylabel('Flux Value')
    ax.set_title('Bass-Band Spectral Flux with Adaptive Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(audio) / sr)

    # 3. Beat intervals histogram
    ax = axes[2]
    if len(detector.beat_timestamps) > 1:
        intervals = np.diff(detector.beat_timestamps)
        ax.hist(intervals, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(intervals), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(intervals):.3f}s')
        ax.axvline(MIN_BEAT_INTERVAL_SEC, color='orange', linestyle=':', linewidth=2,
                   label=f'Min Allowed: {MIN_BEAT_INTERVAL_SEC}s')
        ax.set_xlabel('Interval (seconds)')
        ax.set_ylabel('Count')
        ax.set_title(f'Beat Interval Distribution (BPM={stats["avg_bpm"]:.1f}, CV={stats["interval_cv"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough beats detected', ha='center', va='center', fontsize=14)
        ax.set_title('Beat Interval Distribution')

    # 4. Flux/threshold ratio over time (shows how close we are to threshold)
    ax = axes[3]
    # Avoid division by zero
    threshold_arr = np.array(detector.all_thresholds)
    flux_arr = np.array(detector.all_flux)
    ratio = np.where(threshold_arr > 0, flux_arr / threshold_arr, 0)
    ax.plot(chunk_time_axis, ratio, color='green', linewidth=1)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Threshold Ratio = 1.0')
    ax.fill_between(chunk_time_axis, 0, 1.0, alpha=0.1, color='red', label='Below Threshold')
    for beat_time in detector.beat_timestamps:
        ax.axvline(beat_time, color='red', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Flux / Threshold')
    ax.set_title('Flux/Threshold Ratio (>1.0 = potential beat)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(audio) / sr)
    ax.set_ylim(0, max(3.0, np.percentile(ratio, 99)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {output_path}")


def process_audio_file(audio_path, label):
    """Process a single audio file and return analysis results."""
    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"{'='*60}")

    # Load audio
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Mix to mono

    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Samples: {len(audio)}")

    # Initialize detector
    detector = BeatDetectorOffline(sample_rate=sr)

    # Process chunk-by-chunk
    num_chunks = len(audio) // CHUNK_SIZE
    print(f"  Processing {num_chunks} chunks...")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = start_idx + CHUNK_SIZE
        chunk = audio[start_idx:end_idx]
        detector.process_chunk(chunk, chunk_idx)

    # Analyze results
    stats = analyze_beats(detector.beat_timestamps, len(audio) / sr)

    print(f"\n  Results:")
    print(f"    Total beats detected: {stats['total_beats']}")
    print(f"    Average BPM: {stats['avg_bpm']:.1f}")
    if stats['total_beats'] > 1:
        print(f"    Beat interval: {stats['interval_mean']:.3f}s ± {stats['interval_std']:.3f}s")
        print(f"    Coefficient of variation: {stats['interval_cv']:.3f}")
        print(f"    Suspicious rapid beats: {len(stats['false_positive_periods'])} occurrences")
        print(f"    Long gaps (>2s): {len(stats['long_gap_periods'])} occurrences")

    # Visualize
    output_path = OUTPUT_DIR / f"beat_detector_{label.lower()}.png"
    visualize_results(audio, sr, detector, stats, label, output_path)

    return {
        'label': label,
        'stats': stats,
        'detector': detector,
        'audio_duration': len(audio) / sr
    }


def write_analysis_report(results):
    """Write detailed analysis to markdown file."""
    output_path = OUTPUT_DIR / "beat_detector_validation.md"

    with open(output_path, 'w') as f:
        f.write("# Beat Detector Validation Report\n\n")
        f.write("Analysis of bass-band spectral flux beat detection algorithm on electronic and ambient music.\n\n")

        f.write("## Algorithm Configuration\n\n")
        f.write(f"- **Sample Rate**: {SAMPLE_RATE} Hz\n")
        f.write(f"- **Chunk Size**: {CHUNK_SIZE} samples (~{CHUNK_SIZE/SAMPLE_RATE*1000:.1f}ms)\n")
        f.write(f"- **FFT Size**: {N_FFT}\n")
        f.write(f"- **Bass Range**: {BASS_LOW_HZ}-{BASS_HIGH_HZ} Hz\n")
        f.write(f"- **Flux History Window**: {FLUX_HISTORY_SEC}s\n")
        f.write(f"- **Minimum Beat Interval**: {MIN_BEAT_INTERVAL_SEC}s (max ~{60/MIN_BEAT_INTERVAL_SEC:.0f} BPM)\n")
        f.write(f"- **Threshold Multiplier**: {THRESHOLD_MULTIPLIER}×σ above mean\n\n")

        f.write("---\n\n")

        for result in results:
            label = result['label']
            stats = result['stats']
            duration = result['audio_duration']

            f.write(f"## {label}\n\n")

            f.write("### Summary\n\n")
            f.write(f"- **Duration**: {duration:.1f}s\n")
            f.write(f"- **Total Beats Detected**: {stats['total_beats']}\n")
            f.write(f"- **Average BPM**: {stats['avg_bpm']:.1f}\n")

            if stats['total_beats'] > 1:
                f.write(f"- **Mean Beat Interval**: {stats['interval_mean']:.3f}s\n")
                f.write(f"- **Interval StdDev**: {stats['interval_std']:.3f}s\n")
                f.write(f"- **Coefficient of Variation**: {stats['interval_cv']:.3f}\n\n")

                f.write("### Interval Quality\n\n")
                if stats['interval_cv'] < 0.15:
                    f.write("✅ **Very consistent** - CV < 0.15 (good rhythm tracking)\n\n")
                elif stats['interval_cv'] < 0.30:
                    f.write("⚠️ **Moderately consistent** - CV 0.15-0.30 (acceptable but some drift)\n\n")
                else:
                    f.write("❌ **Inconsistent** - CV > 0.30 (poor rhythm tracking)\n\n")

                f.write("### Potential Issues\n\n")

                # False positives
                if len(stats['false_positive_periods']) > 0:
                    f.write(f"**Rapid Beat Clusters (potential false positives)**: {len(stats['false_positive_periods'])} occurrences\n\n")
                    if len(stats['false_positive_periods']) <= 10:
                        for i, (start, end) in enumerate(stats['false_positive_periods'], 1):
                            f.write(f"{i}. {start:.2f}s - {end:.2f}s (Δ={end-start:.3f}s)\n")
                    else:
                        f.write(f"First 10 occurrences:\n")
                        for i, (start, end) in enumerate(stats['false_positive_periods'][:10], 1):
                            f.write(f"{i}. {start:.2f}s - {end:.2f}s (Δ={end-start:.3f}s)\n")
                        f.write(f"... and {len(stats['false_positive_periods'])-10} more\n")
                    f.write("\n")
                else:
                    f.write("**Rapid Beat Clusters**: None detected ✅\n\n")

                # Long gaps
                if len(stats['long_gap_periods']) > 0:
                    f.write(f"**Long Gaps >2s (potential false negatives)**: {len(stats['long_gap_periods'])} occurrences\n\n")
                    for i, (start, end) in enumerate(stats['long_gap_periods'], 1):
                        f.write(f"{i}. {start:.2f}s - {end:.2f}s (gap={end-start:.2f}s)\n")
                    f.write("\n")
                else:
                    f.write("**Long Gaps >2s**: None detected ✅\n\n")

            else:
                f.write("\n⚠️ **Insufficient beats detected for interval analysis**\n\n")

            f.write(f"### Visualization\n\n")
            f.write(f"See `beat_detector_{label.lower()}.png` for detailed plots.\n\n")
            f.write("---\n\n")

        # Overall conclusions
        f.write("## Analysis & Recommendations\n\n")

        f.write("### Algorithm Behavior\n\n")

        electronic_stats = next(r['stats'] for r in results if r['label'] == 'Electronic')
        ambient_stats = next(r['stats'] for r in results if r['label'] == 'Ambient')

        # Compare results
        f.write("#### Electronic Music (Fred again.. - Tanya maybe life)\n\n")
        if electronic_stats['total_beats'] > 10:
            f.write(f"- Detected {electronic_stats['total_beats']} beats at {electronic_stats['avg_bpm']:.1f} BPM\n")
            if electronic_stats['interval_cv'] < 0.2:
                f.write("- ✅ Beat tracking is **stable** — algorithm works well on electronic music\n")
            else:
                f.write("- ⚠️ Beat tracking is **inconsistent** — may need tuning\n")

            if len(electronic_stats['false_positive_periods']) > 5:
                f.write(f"- ⚠️ **{len(electronic_stats['false_positive_periods'])} rapid beat clusters** — threshold may be too low\n")

            if len(electronic_stats['long_gap_periods']) > 3:
                f.write(f"- ⚠️ **{len(electronic_stats['long_gap_periods'])} long gaps** — missing beats during breakdowns or quiet sections\n")
        else:
            f.write(f"- ❌ Only detected {electronic_stats['total_beats']} beats — algorithm likely **failing**\n")
            f.write("- Bass content may be weak or threshold is too high\n")

        f.write("\n#### Ambient Music (Fred again.. - tayla every night)\n\n")
        if ambient_stats['total_beats'] > 5:
            f.write(f"- Detected {ambient_stats['total_beats']} beats at {ambient_stats['avg_bpm']:.1f} BPM\n")
            if ambient_stats['total_beats'] < 10:
                f.write("- ⚠️ Sparse beat detection — expected for ambient music with weak/irregular rhythm\n")

            if len(ambient_stats['false_positive_periods']) > 10:
                f.write(f"- ❌ **{len(ambient_stats['false_positive_periods'])} false positives** — algorithm reacting to non-rhythmic bass content\n")
                f.write("- Adaptive threshold may be over-sensitive to low-frequency texture changes\n")
        else:
            f.write(f"- Detected {ambient_stats['total_beats']} beats\n")
            f.write("- ✅ Expected behavior — ambient music often lacks clear rhythmic pulse\n")

        f.write("\n### Tuning Recommendations\n\n")

        # Generate recommendations based on results
        if electronic_stats['total_beats'] < 20:
            f.write("1. **Threshold too high for electronic music** — try reducing `THRESHOLD_MULTIPLIER` from 1.5 to 1.2\n")

        if len(electronic_stats['false_positive_periods']) > 10 or len(ambient_stats['false_positive_periods']) > 10:
            f.write("2. **Too many false positives** — consider:\n")
            f.write("   - Increase `MIN_BEAT_INTERVAL_SEC` from 0.3s to 0.4s\n")
            f.write("   - Increase `THRESHOLD_MULTIPLIER` to reduce sensitivity\n")
            f.write("   - Add onset strength smoothing before thresholding\n")

        if len(electronic_stats['long_gap_periods']) > 3:
            f.write("3. **Missing beats during quiet sections** — adaptive threshold may be drifting too low\n")
            f.write("   - Consider absolute minimum threshold floor\n")
            f.write("   - Use longer `FLUX_HISTORY_SEC` (e.g., 5.0s) for more stable threshold\n")

        f.write("\n### Genre Suitability\n\n")
        f.write("Based on this validation:\n\n")

        if electronic_stats['interval_cv'] < 0.25:
            f.write("- ✅ **Electronic music**: Algorithm works well\n")
        else:
            f.write("- ⚠️ **Electronic music**: Needs tuning for better consistency\n")

        if ambient_stats['total_beats'] < 10 and len(ambient_stats['false_positive_periods']) < 5:
            f.write("- ✅ **Ambient music**: Correctly ignores non-rhythmic content\n")
        elif len(ambient_stats['false_positive_periods']) > 10:
            f.write("- ❌ **Ambient music**: Too many false positives — over-sensitive to texture\n")
        else:
            f.write("- ⚠️ **Ambient music**: Mixed results\n")

        f.write("\n---\n\n")
        f.write("*Report generated by beat_detector_validation.py*\n")

    print(f"\n  Analysis report written to: {output_path}")


def main():
    print("\n" + "="*60)
    print("Beat Detector Validation")
    print("="*60)
    print(f"\nTesting bass-band spectral flux algorithm on:")
    print(f"  1. Electronic music (Fred again..)")
    print(f"  2. Ambient music (Fred again..)")
    print()

    results = []

    # Process electronic music
    results.append(process_audio_file(AUDIO_FILES['electronic'], 'Electronic'))

    # Process ambient music
    results.append(process_audio_file(AUDIO_FILES['ambient'], 'Ambient'))

    # Write analysis report
    print(f"\n{'='*60}")
    print("Generating Analysis Report")
    print("="*60)
    write_analysis_report(results)

    print(f"\n{'='*60}")
    print("Validation Complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_DIR}/beat_detector_electronic.png")
    print(f"  - {OUTPUT_DIR}/beat_detector_ambient.png")
    print(f"  - {OUTPUT_DIR}/beat_detector_validation.md")
    print()


if __name__ == '__main__':
    main()
