#!/usr/bin/env python3
"""
Visualize beat tracking comparison results.

Creates a timeline visualization showing detected beats from each approach
overlaid with user tap annotations.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Paths
DATA_FILE = "/Users/KO16K39/Documents/led/interactivity-research/analysis/beat_tracker_comparison.yaml"
OUTPUT_FILE = "/Users/KO16K39/Documents/led/interactivity-research/analysis/beat_tracker_comparison.png"

# Focus on steady groove section
STEADY_GROOVE_START = 8.0
STEADY_GROOVE_END = 24.0


def main():
    # Load data
    with open(DATA_FILE, 'r') as f:
        data = yaml.safe_load(f)

    user_taps = data['reference']['all_user_taps']
    steady_taps = data['reference']['steady_groove_taps']
    changes = [8.073, 17.105, 24.271, 28.696, 34.495, 37.371]  # From annotations

    # Select approaches to visualize (exclude redundant constrained versions)
    approaches_to_plot = [
        ('User Taps (Reference)', user_taps, 'black', 's', 8),
        ('librosa_default', data['beat_times']['librosa_default'], '#e74c3c', 'o', 4),
        ('librosa_bass_onset', data['beat_times']['librosa_bass_onset'], '#2ecc71', '^', 4),
        ('librosa_constrained_105', data['beat_times']['librosa_constrained_105'], '#3498db', 'v', 4),
        ('spectral_flux', data['beat_times']['spectral_flux'], '#9b59b6', 'd', 4),
    ]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Plot 1: Full timeline
    y_offset = 0
    for name, beats, color, marker, size in approaches_to_plot:
        beats_arr = np.array(beats)
        ax1.scatter(beats_arr, np.ones_like(beats_arr) * y_offset,
                   c=color, marker=marker, s=size**2, alpha=0.7, label=name)
        y_offset += 1

    # Add change markers
    for change in changes:
        ax1.axvline(change, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Highlight steady groove region
    ax1.axvspan(STEADY_GROOVE_START, STEADY_GROOVE_END, alpha=0.1, color='green',
                label='Steady Groove (Evaluation Region)')

    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Approach', fontsize=12)
    ax1.set_title('Beat Tracker Comparison - Tool "Opiate Intro" (Full Timeline)',
                  fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(approaches_to_plot)))
    ax1.set_yticklabels([name for name, _, _, _, _ in approaches_to_plot])
    ax1.grid(True, alpha=0.2, axis='x')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.set_xlim(0, 41)

    # Plot 2: Zoomed into steady groove section
    y_offset = 0
    for name, beats, color, marker, size in approaches_to_plot:
        beats_arr = np.array(beats)
        # Filter to steady groove section
        mask = (beats_arr >= STEADY_GROOVE_START) & (beats_arr <= STEADY_GROOVE_END)
        steady_beats = beats_arr[mask]

        ax2.scatter(steady_beats, np.ones_like(steady_beats) * y_offset,
                   c=color, marker=marker, s=size**2, alpha=0.7)
        y_offset += 1

    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Approach', fontsize=12)
    ax2.set_title('Steady Groove Section (8-24s) - Zoomed View',
                  fontsize=12, fontweight='bold')
    ax2.set_yticks(range(len(approaches_to_plot)))
    ax2.set_yticklabels([name for name, _, _, _, _ in approaches_to_plot])
    ax2.grid(True, alpha=0.2, axis='x')
    ax2.set_xlim(STEADY_GROOVE_START, STEADY_GROOVE_END)

    # Add annotations
    ax1.text(2, 4.5, 'Section A\n(Intro)', ha='center', va='center',
            fontsize=9, alpha=0.7, style='italic')
    ax1.text(12.5, 4.5, 'Section B\n(Steady)', ha='center', va='center',
            fontsize=9, alpha=0.7, style='italic')
    ax1.text(20.5, 4.5, 'Section C\n(Steady)', ha='center', va='center',
            fontsize=9, alpha=0.7, style='italic')
    ax1.text(26.5, 4.5, 'Section D\n(Fills)', ha='center', va='center',
            fontsize=9, alpha=0.7, style='italic')
    ax1.text(35, 4.5, 'Section E+\n(Variation)', ha='center', va='center',
            fontsize=9, alpha=0.7, style='italic')

    # Add F1 scores to legend
    f1_scores = {
        'librosa_default': 0.486,
        'librosa_bass_onset': 0.500,
        'librosa_constrained_105': 0.486,
        'spectral_flux': 0.327
    }

    textstr = 'F1 Scores (Steady Groove):\n'
    textstr += f"• librosa_bass_onset: {f1_scores['librosa_bass_onset']:.3f} ★\n"
    textstr += f"• librosa_default: {f1_scores['librosa_default']:.3f}\n"
    textstr += f"• librosa_constrained: {f1_scores['librosa_constrained_105']:.3f}\n"
    textstr += f"• spectral_flux: {f1_scores['spectral_flux']:.3f}"

    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
