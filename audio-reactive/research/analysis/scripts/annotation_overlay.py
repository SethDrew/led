#!/usr/bin/env python3
"""
Opiate Intro Annotation Overlay Visualization

Creates a comprehensive 5-panel visualization showing user tap annotations
overlaid on various audio analysis features. This helps validate that
user taps correlate with specific audio characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import yaml
from pathlib import Path

# File paths
AUDIO_PATH = Path("/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.wav")
ANNOTATION_PATH = Path("/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/opiate_intro.annotations.yaml")
OUTPUT_PATH = Path("/Users/KO16K39/Documents/led/audio-reactive/research/analysis/opiate_annotation_overlay.png")

# Constants
SR = 44100  # Target sample rate
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
FMIN = 20
FMAX = 8000

# Color scheme (dark background compatible)
COLORS = {
    'beat': '#FF4444',          # Red
    'changes': '#FFFFFF',       # White
    'air': '#00DDDD',           # Cyan
    'waveform': '#FFFFFF',      # White
    'algo_beat': '#FFDD44',     # Yellow
    'onset': '#00DDDD',         # Cyan
    'harmonic': '#BB44FF',      # Purple
    'percussive': '#FF8844',    # Orange
}

# Band definitions (Hz)
BANDS = {
    'Sub-bass': (20, 60),
    'Bass': (60, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 4000),
    'Treble': (4000, 8000),
}

BAND_COLORS = ['#FF4444', '#FF8844', '#FFDD44', '#44DD44', '#4444FF']


def load_annotations(path):
    """Load tap annotations from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return {
        'beat': np.array(data.get('beat', [])),
        'changes': np.array(data.get('changes', [])),
        'air': np.array(data.get('air', [])),
    }


def compute_band_energies(y, sr, hop_length):
    """Compute RMS energy for each frequency band over time."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    band_energies = {}
    for band_name, (fmin, fmax) in BANDS.items():
        # Find frequency bins in this band
        mask = (freqs >= fmin) & (freqs <= fmax)
        # Compute RMS over frequency bins
        band_mag = S[mask, :]
        band_energy = np.sqrt(np.mean(band_mag**2, axis=0))
        band_energies[band_name] = band_energy

    return band_energies


def normalize_band(energy):
    """Normalize band energy to 0-1 range."""
    min_val = np.min(energy)
    max_val = np.max(energy)
    if max_val > min_val:
        return (energy - min_val) / (max_val - min_val)
    return energy


def create_visualization():
    """Create the complete 5-panel visualization."""

    # Load audio
    print(f"Loading audio from {AUDIO_PATH}...")
    y, sr = librosa.load(AUDIO_PATH, sr=SR)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f}s, Sample rate: {sr} Hz")

    # Load annotations
    print(f"Loading annotations from {ANNOTATION_PATH}...")
    annotations = load_annotations(ANNOTATION_PATH)
    print(f"Beat taps: {len(annotations['beat'])}")
    print(f"Change markers: {len(annotations['changes'])}")
    print(f"Air taps: {len(annotations['air'])}")

    # Compute audio features
    print("Computing audio features...")

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    # Beat tracking (algorithmic)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    # Tempo might be an array, extract scalar
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    print(f"Detected tempo: {tempo_val:.1f} BPM ({len(beat_times)} beats)")

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Band energies
    band_energies = compute_band_energies(y, sr, HOP_LENGTH)

    # HPSS separation
    print("Computing harmonic-percussive separation...")
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Compute RMS for harmonic and percussive
    harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=HOP_LENGTH)[0]
    percussive_rms = librosa.feature.rms(y=y_percussive, hop_length=HOP_LENGTH)[0]

    # Time axes
    time_frames = librosa.frames_to_time(
        np.arange(mel_spec.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )
    time_samples = np.arange(len(y)) / sr

    # Create figure
    print("Creating visualization...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(5, 1, figsize=(16, 24), dpi=150)
    fig.suptitle('Opiate Intro — User Annotation Overlay', fontsize=20, y=0.995)

    # ========== PANEL 1: Waveform + Beat taps ==========
    ax = axes[0]
    ax.plot(time_samples, y, color=COLORS['waveform'], alpha=0.7, linewidth=0.5)
    ax.set_xlim(0, duration)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Waveform + Beat Taps + Section Markers', fontsize=14, pad=10)
    ax.grid(alpha=0.2)

    # Beat taps
    for tap in annotations['beat']:
        ax.axvline(tap, color=COLORS['beat'], alpha=0.5, linewidth=1, linestyle='-')

    # Change markers (thick dashed white lines)
    changes = annotations['changes']
    for change in changes:
        ax.axvline(change, color=COLORS['changes'], alpha=0.8, linewidth=2, linestyle='--')

    # Label sections between changes
    section_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    section_starts = np.concatenate([[0], changes, [duration]])
    for i in range(len(section_starts) - 1):
        mid_point = (section_starts[i] + section_starts[i+1]) / 2
        if i < len(section_labels):
            ax.text(mid_point, ax.get_ylim()[1] * 0.85, section_labels[i],
                   fontsize=16, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # ========== PANEL 2: Mel Spectrogram + Air taps ==========
    ax = axes[1]
    img = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel',
        sr=sr, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX,
        ax=ax, cmap='magma'
    )
    ax.set_title('Mel Spectrogram + Air Taps', fontsize=14, pad=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Air taps with highlight regions
    for tap in annotations['air']:
        # Vertical line
        ax.axvline(tap, color=COLORS['air'], alpha=0.8, linewidth=1.5, linestyle='-')
        # Semi-transparent rectangle (±250ms)
        ax.axvspan(tap - 0.25, tap + 0.25,
                  color=COLORS['air'], alpha=0.15, linewidth=0)

    # Change markers
    for change in annotations['changes']:
        ax.axvline(change, color=COLORS['changes'], alpha=0.6, linewidth=2, linestyle='--')

    # ========== PANEL 3: Band Energy ==========
    ax = axes[2]
    for (band_name, energy), color in zip(band_energies.items(), BAND_COLORS):
        normalized_energy = normalize_band(energy)
        ax.plot(time_frames, normalized_energy, color=color,
               label=band_name, linewidth=2, alpha=0.8)

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Normalized Energy', fontsize=12)
    ax.set_title('Frequency Band Energies', fontsize=14, pad=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.2)

    # Change markers
    for change in annotations['changes']:
        ax.axvline(change, color=COLORS['changes'], alpha=0.8, linewidth=2, linestyle='--')

    # ========== PANEL 4: Onset Strength + Beat Comparison ==========
    ax = axes[3]

    # Onset strength envelope
    onset_norm = onset_env / np.max(onset_env)
    ax.plot(time_frames, onset_norm, color=COLORS['onset'],
           linewidth=2, alpha=0.8, label='Onset Strength')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Normalized Strength', fontsize=12)
    ax.set_title('Onset Strength: User Taps (Red ↑) vs Algorithm (Yellow ↓)', fontsize=14, pad=10)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right', fontsize=10)

    # User beat taps (red tick marks on top)
    for tap in annotations['beat']:
        ax.plot([tap, tap], [1.05, 1.15], color=COLORS['beat'],
               linewidth=2, alpha=0.9)

    # Algorithmic beats (yellow tick marks on bottom)
    for beat in beat_times:
        ax.plot([beat, beat], [-0.05, -0.15], color=COLORS['algo_beat'],
               linewidth=2, alpha=0.9)

    # Add text labels
    ax.text(duration * 0.98, 1.10, 'User', color=COLORS['beat'],
           fontsize=10, ha='right', va='center', weight='bold')
    ax.text(duration * 0.98, -0.10, 'Algorithm', color=COLORS['algo_beat'],
           fontsize=10, ha='right', va='center', weight='bold')

    # Change markers
    for change in annotations['changes']:
        ax.axvline(change, color=COLORS['changes'], alpha=0.6, linewidth=2, linestyle='--')

    # ========== PANEL 5: HPSS Components + Air taps ==========
    ax = axes[4]

    # Normalize RMS values
    harmonic_norm = normalize_band(harmonic_rms)
    percussive_norm = normalize_band(percussive_rms)

    ax.plot(time_frames, harmonic_norm, color=COLORS['harmonic'],
           linewidth=2, alpha=0.8, label='Harmonic RMS')
    ax.plot(time_frames, percussive_norm, color=COLORS['percussive'],
           linewidth=2, alpha=0.8, label='Percussive RMS')

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Normalized RMS', fontsize=12)
    ax.set_title('Harmonic-Percussive Separation + Air Taps', fontsize=14, pad=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.2)

    # Air taps
    for tap in annotations['air']:
        ax.axvline(tap, color=COLORS['air'], alpha=0.7, linewidth=1.5, linestyle='-')

    # Change markers
    for change in annotations['changes']:
        ax.axvline(change, color=COLORS['changes'], alpha=0.8, linewidth=2, linestyle='--')

    # ========== Legend Box ==========
    legend_text = (
        "Legend:\n"
        f"• Beat taps (n={len(annotations['beat'])}) — {COLORS['beat']} thin lines\n"
        f"• Change markers (n={len(annotations['changes'])}) — {COLORS['changes']} thick dashed lines\n"
        f"• Air taps (n={len(annotations['air'])}) — {COLORS['air']} lines + highlight regions\n"
        f"• Algorithmic beats — {COLORS['algo_beat']} (detected {tempo_val:.1f} BPM)"
    )

    fig.text(0.99, 0.01, legend_text, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.995])

    # Save
    print(f"Saving visualization to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"✓ Saved to {OUTPUT_PATH}")

    # Print analysis summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    # User tap tempo analysis
    beat_intervals = np.diff(annotations['beat'])
    mean_interval = np.mean(beat_intervals)
    user_bpm = 60 / mean_interval if mean_interval > 0 else 0
    print(f"\nUser Beat Taps:")
    print(f"  Count: {len(annotations['beat'])}")
    print(f"  Mean interval: {mean_interval:.3f}s")
    print(f"  Implied tempo: {user_bpm:.1f} BPM")
    print(f"  Std deviation: {np.std(beat_intervals):.3f}s")

    print(f"\nAlgorithmic Beat Detection:")
    print(f"  Detected tempo: {tempo_val:.1f} BPM")
    print(f"  Beat count: {len(beat_times)}")
    print(f"  Tempo ratio: {tempo_val / user_bpm:.2f}x user tempo")

    print(f"\nSection Changes:")
    section_durations = np.diff(np.concatenate([[0], annotations['changes'], [duration]]))
    for i, dur in enumerate(section_durations):
        label = section_labels[i] if i < len(section_labels) else '?'
        print(f"  Section {label}: {dur:.2f}s")

    print(f"\nAir Taps:")
    print(f"  Count: {len(annotations['air'])}")
    if len(annotations['air']) > 0:
        air_ranges = []
        # Find continuous air regions (within 2 seconds)
        current_start = annotations['air'][0]
        for i in range(1, len(annotations['air'])):
            if annotations['air'][i] - annotations['air'][i-1] > 2.0:
                air_ranges.append((current_start, annotations['air'][i-1]))
                current_start = annotations['air'][i]
        air_ranges.append((current_start, annotations['air'][-1]))

        print(f"  Air regions detected: {len(air_ranges)}")
        for i, (start, end) in enumerate(air_ranges, 1):
            print(f"    Region {i}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")

    print("\n" + "="*60)


if __name__ == '__main__':
    create_visualization()
