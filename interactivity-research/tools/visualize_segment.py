#!/usr/bin/env python3
"""
Audio Visualization for LED Reactive Processing

Visualizes audio segments in a way that maps to how audio-reactive LED systems
process sound. Shows the features that LED controllers typically extract:
- Frequency content (mel spectrogram)
- Band energy (for frequency-to-color mapping)
- Onsets and beats (for trigger events)
- Spectral characteristics (for brightness/color temperature)

Usage:
    python visualize_segment.py <audio_file.wav>
    python visualize_segment.py  # processes all WAV files in audio-segments/
"""

import sys
import os
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf


# Define frequency bands typical for LED audio-reactive systems
FREQUENCY_BANDS = {
    'Sub-bass': (20, 80),
    'Bass': (80, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 6000),
    'Treble': (6000, 8000),
}


def load_audio(filepath):
    """Load audio file and convert to mono if needed."""
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr


def compute_band_energies(y, sr, n_fft=2048, hop_length=512):
    """
    Compute energy in each frequency band using mel filterbanks.
    This simulates what an LED system would extract for frequency-to-color mapping.
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=128, fmin=20, fmax=8000
    )

    # Get mel frequencies
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)

    # Extract energy for each band
    band_energies = {}
    band_ratios = {}

    for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
        # Find mel bins in this frequency range
        band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)

        # Sum energy in this band
        band_energy = np.sum(mel_spec[band_mask, :], axis=0)
        band_energies[band_name] = band_energy

        # Store ratio for normalization info
        band_ratios[band_name] = np.mean(band_energy)

    # Normalize each band independently for visualization
    band_energies_norm = {}
    for band_name, energy in band_energies.items():
        if np.max(energy) > 0:
            band_energies_norm[band_name] = energy / np.max(energy)
        else:
            band_energies_norm[band_name] = energy

    return band_energies_norm, band_ratios


def visualize_audio(filepath, save_output=True, show_plot=True):
    """
    Create comprehensive visualization of audio features for LED processing.
    """
    print(f"\nProcessing: {filepath}")

    # Load audio
    y, sr = librosa.load(filepath, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")

    # Analysis parameters
    n_fft = 2048
    hop_length = 512

    # Compute features
    print("  Computing mel spectrogram...")
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=64, fmin=20, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    print("  Computing band energies...")
    band_energies, band_ratios = compute_band_energies(y, sr, n_fft, hop_length)

    print("  Computing onset detection...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    print("  Computing beat tracking...")
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Extract tempo as scalar if it's an array
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    print(f"  Detected tempo: {tempo:.1f} BPM")
    print(f"  Detected {len(onset_times)} onsets, {len(beat_times)} beats")

    print("  Computing spectral features...")
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Create time axis
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

    # Set up the plot with dark style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1.5, 1.5, 1.5, 1.5, 0.3], hspace=0.3)

    # Color palette
    band_colors = {
        'Sub-bass': '#FF1744',    # Red
        'Bass': '#FF9100',        # Orange
        'Mids': '#FFEA00',        # Yellow
        'High-mids': '#00E676',   # Green
        'Treble': '#00B0FF',      # Blue
    }

    # Panel 1: Waveform
    ax1 = fig.add_subplot(gs[0])
    waveform_times = np.linspace(0, duration, len(y))
    ax1.plot(waveform_times, y, color='#FFFFFF', linewidth=0.5, alpha=0.8)
    ax1.set_xlim([0, duration])
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Panel 1: Waveform (Ground Truth)\nRaw audio signal amplitude over time',
                  fontsize=12, pad=10)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Mel Spectrogram
    ax2 = fig.add_subplot(gs[1])
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', fmin=20, fmax=8000,
        ax=ax2, cmap='magma'
    )
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Panel 2: Mel Spectrogram (Primary Input to LED Systems)\n' +
                  'Frequency content over time - shows which frequencies are active when',
                  fontsize=12, pad=10)
    cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.set_label('Power (dB)', rotation=270, labelpad=20)

    # Panel 3: Band Energy Over Time
    ax3 = fig.add_subplot(gs[2])
    for band_name, energy in band_energies.items():
        ax3.plot(times, energy, label=band_name, color=band_colors[band_name],
                linewidth=2, alpha=0.9)

    ax3.set_xlim([0, duration])
    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('Normalized Energy')
    ax3.set_title('Panel 3: Frequency Band Energy (For Color Mapping)\n' +
                  'Energy in each band (independently normalized) - drives LED color choices',
                  fontsize=12, pad=10)
    ax3.legend(loc='upper right', framealpha=0.8)
    ax3.grid(True, alpha=0.2)

    # Add magnitude ratio info
    max_ratio = max(band_ratios.values())
    ratio_text = "Actual magnitude ratios:\n"
    for band_name, ratio in band_ratios.items():
        normalized_ratio = ratio / max_ratio
        ratio_text += f"{band_name}: {normalized_ratio:.2f}\n"
    ax3.text(0.02, 0.98, ratio_text, transform=ax3.transAxes,
            verticalalignment='top', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # Panel 4: Onset Strength & Beat Detection
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(times, onset_env / np.max(onset_env), color='#00E5FF',
            linewidth=1.5, label='Onset Strength')

    # Mark onset detections
    for onset_time in onset_times:
        ax4.axvline(x=onset_time, color='#FFC400', alpha=0.3, linewidth=0.8)

    # Mark beat detections
    for beat_time in beat_times:
        ax4.axvline(x=beat_time, color='#FF1744', alpha=0.6, linewidth=2)

    ax4.set_xlim([0, duration])
    ax4.set_ylim([0, 1.1])
    ax4.set_ylabel('Normalized Strength')
    ax4.set_title('Panel 4: Onset Detection & Beat Tracking\n' +
                  'Yellow lines = onsets (note attacks), Red lines = beats - triggers for LED events',
                  fontsize=12, pad=10)
    ax4.legend(loc='upper right', framealpha=0.8)
    ax4.grid(True, alpha=0.2)

    # Panel 5: Spectral Centroid & RMS Energy
    ax5 = fig.add_subplot(gs[4])

    # Plot spectral centroid (normalized to 0-1)
    ax5_twin = ax5.twinx()
    centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / \
                    (np.max(spectral_centroid) - np.min(spectral_centroid))
    line1 = ax5.plot(times, centroid_norm, color='#B388FF', linewidth=2,
                     label='Spectral Centroid (brightness)', alpha=0.9)

    # Plot RMS energy
    rms_norm = rms_energy / np.max(rms_energy)
    line2 = ax5_twin.plot(times, rms_norm, color='#FFD740', linewidth=2,
                          label='RMS Energy (loudness)', alpha=0.9)

    ax5.set_xlim([0, duration])
    ax5.set_ylim([0, 1.1])
    ax5_twin.set_ylim([0, 1.1])
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_ylabel('Spectral Centroid\n(normalized)', color='#B388FF')
    ax5_twin.set_ylabel('RMS Energy\n(normalized)', color='#FFD740')
    ax5.set_title('Panel 5: Spectral Centroid & RMS Energy\n' +
                  'Centroid = "brightness" of sound, RMS = loudness - drives LED intensity',
                  fontsize=12, pad=10)
    ax5.tick_params(axis='y', labelcolor='#B388FF')
    ax5_twin.tick_params(axis='y', labelcolor='#FFD740')
    ax5.grid(True, alpha=0.2)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper right', framealpha=0.8)

    # Info panel at bottom
    ax_info = fig.add_subplot(gs[5])
    ax_info.axis('off')

    filename = Path(filepath).name
    info_text = (
        f"File: {filename}  |  "
        f"Duration: {duration:.2f}s  |  "
        f"Sample Rate: {sr} Hz  |  "
        f"Detected Tempo: {tempo:.1f} BPM  |  "
        f"Onsets: {len(onset_times)}  |  "
        f"Beats: {len(beat_times)}"
    )
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                         edgecolor='#404040', linewidth=1))

    # Overall title
    fig.suptitle(f'Audio-Reactive LED Processing Analysis\n{filename}',
                fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    if save_output:
        output_path = Path(filepath).with_suffix('.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='#0a0a0a', edgecolor='none')
        print(f"  Saved visualization to: {output_path}")

    # Show plot
    if show_plot:
        plt.show()

    plt.close()
    print("  Done!\n")


def main():
    """Main entry point."""

    # Show help message
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        print("\nUsage:")
        print("  python visualize_segment.py <audio_file.wav>  # Process specific file")
        print("  python visualize_segment.py                   # Process all WAV files")
        sys.exit(0)

    # Determine which files to process
    if len(sys.argv) > 1:
        # Process specific file
        audio_files = [sys.argv[1]]
    else:
        # Process all WAV files in audio-segments directory
        script_dir = Path(__file__).parent
        segments_dir = script_dir.parent / 'audio-segments'

        if not segments_dir.exists():
            print(f"Error: Audio segments directory not found: {segments_dir}")
            sys.exit(1)

        audio_files = sorted(segments_dir.glob('*.wav'))

        if not audio_files:
            print(f"Error: No WAV files found in {segments_dir}")
            sys.exit(1)

        print(f"Found {len(audio_files)} WAV files to process")

    # Process each file
    for audio_file in audio_files:
        audio_file = Path(audio_file)

        if not audio_file.exists():
            print(f"Error: File not found: {audio_file}")
            continue

        try:
            visualize_audio(str(audio_file), save_output=True, show_plot=True)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
