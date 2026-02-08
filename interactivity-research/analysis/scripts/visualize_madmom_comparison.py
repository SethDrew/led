#!/usr/bin/env python3
"""
Visualize madmom beat detection vs. user tap annotations
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

def load_annotations(yaml_path):
    """Load tap annotations from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def main():
    # Paths
    audio_path = Path("/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.wav")
    annotations_path = Path("/Users/KO16K39/Documents/led/interactivity-research/audio-segments/opiate_intro.annotations.yaml")
    output_path = Path("/Users/KO16K39/Documents/led/interactivity-research/analysis/madmom_comparison.png")

    # Load annotations
    print("Loading user annotations...")
    annotations = load_annotations(annotations_path)
    user_beats = np.array(annotations.get('consistent-beat', []))

    # Run madmom
    print("Running madmom beat detection...")
    rnn = RNNBeatProcessor()
    activations = rnn(str(audio_path))
    dbn = DBNBeatTrackingProcessor(fps=100)
    madmom_beats = dbn(activations)

    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Get audio duration (use max of all beat times)
    duration = max(max(user_beats), max(madmom_beats))

    # Plot 1: User taps (consistent-beat layer)
    ax = axes[0]
    ax.eventplot([user_beats], lineoffsets=0.5, linelengths=0.8, colors='blue', linewidths=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('User Taps\n(81.9 BPM)', fontweight='bold')
    ax.set_title('Beat Detection Comparison: madmom vs. User Annotations\nTool - Opiate (Intro)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])

    # Plot 2: Madmom beats
    ax = axes[1]
    ax.eventplot([madmom_beats], lineoffsets=0.5, linelengths=0.8, colors='red', linewidths=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Madmom\n(162.2 BPM)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])

    # Plot 3: Overlay comparison
    ax = axes[2]
    ax.eventplot([user_beats], lineoffsets=0.7, linelengths=0.4, colors='blue',
                 linewidths=2, label='User taps (ground truth)')
    ax.eventplot([madmom_beats], lineoffsets=0.3, linelengths=0.4, colors='red',
                 linewidths=2, label='Madmom detected')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Overlay', fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)

    # Focus on steady groove section (11-40s)
    for ax in axes:
        ax.axvspan(11, 40, alpha=0.1, color='green', label='Steady groove')
        ax.set_xlim(0, duration)

    # Add text annotations
    fig.text(0.02, 0.96, '✅ User taps: 41 beats @ 81.9 BPM (steady groove pulse)',
             fontsize=10, color='blue', fontweight='bold', transform=fig.transFigure)
    fig.text(0.02, 0.93, '❌ Madmom: 106 beats @ 162.2 BPM (tempo doubled - tracking subdivisions)',
             fontsize=10, color='red', fontweight='bold', transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to:\n   {output_path}")

    # Show stats
    print(f"\nStats:")
    print(f"  User taps: {len(user_beats)} beats")
    print(f"  Madmom: {len(madmom_beats)} beats")
    print(f"  Ratio: {len(madmom_beats) / len(user_beats):.2f}x (expected ~2x for tempo doubling)")


if __name__ == '__main__':
    main()
