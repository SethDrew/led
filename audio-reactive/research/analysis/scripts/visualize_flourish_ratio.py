#!/usr/bin/env python3
"""
Visualize flourish ratio and density over time
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/KO16K39/Documents/led/audio-reactive/research")
ANNOTATIONS_PATH = BASE_DIR / "audio-segments" / "opiate_intro.annotations.yaml"
OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_PATH = OUTPUT_DIR / "beat_vs_consistent_visualization.png"

def load_annotations():
    with open(ANNOTATIONS_PATH, 'r') as f:
        return yaml.safe_load(f)

def load_classifications():
    """Parse classifications from the YAML file (avoiding numpy objects)"""
    classifications_path = OUTPUT_DIR / "beat_vs_consistent.yaml"
    classifications = []

    with open(classifications_path, 'r') as f:
        lines = f.readlines()
        in_classifications = False
        current = {}

        for line in lines:
            if 'classifications:' in line:
                in_classifications = True
                continue
            if in_classifications and line.startswith('- index:'):
                current = {}
            elif in_classifications and '  timestamp:' in line:
                current['timestamp'] = float(line.split(':')[1].strip())
            elif in_classifications and '  type:' in line:
                current['type'] = line.split(':')[1].strip()
                classifications.append(current)
            elif in_classifications and line.startswith('flourish_timestamps:'):
                break

    return classifications

def compute_flourish_ratio_timeseries(classifications, changes, window_size=5.0, step=0.5):
    """
    Compute flourish ratio over time using sliding window.
    """
    times = []
    ratios = []

    max_time = max(c['timestamp'] for c in classifications)

    t = window_size / 2
    while t < max_time:
        window_start = t - window_size / 2
        window_end = t + window_size / 2

        window_taps = [c for c in classifications if window_start <= c['timestamp'] < window_end]

        if len(window_taps) > 0:
            flourish_count = sum(1 for c in window_taps if c['type'] == 'flourish')
            ratio = flourish_count / len(window_taps) * 100
        else:
            ratio = np.nan

        times.append(t)
        ratios.append(ratio)
        t += step

    return times, ratios

def main():
    print("Loading data...")
    annotations = load_annotations()
    classifications = load_classifications()

    changes = annotations['changes']
    beat_taps = annotations['beat']
    consistent_taps = annotations['consistent-beat']

    print(f"Loaded {len(classifications)} classifications")

    # Compute flourish ratio timeseries
    print("Computing flourish ratio timeseries...")
    times, ratios = compute_flourish_ratio_timeseries(classifications, changes)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Tap annotations
    ax1 = axes[0]
    beat_times = [c['timestamp'] for c in classifications]
    beat_types = [1 if c['type'] == 'on-grid' else 2 for c in classifications]

    on_grid = [t for t, ty in zip(beat_times, beat_types) if ty == 1]
    flourishes = [t for t, ty in zip(beat_times, beat_types) if ty == 2]

    ax1.scatter(on_grid, [1]*len(on_grid), c='blue', s=50, alpha=0.6, label='On-grid taps')
    ax1.scatter(flourishes, [2]*len(flourishes), c='red', s=50, alpha=0.6, label='Flourish taps')
    ax1.scatter(consistent_taps, [0.5]*len(consistent_taps), c='green', s=30, marker='x', alpha=0.8, label='Consistent-beat')

    # Add section boundaries
    for change in changes:
        ax1.axvline(change, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    ax1.set_ylabel('Tap Type')
    ax1.set_ylim(0, 2.5)
    ax1.set_yticks([0.5, 1, 2])
    ax1.set_yticklabels(['Consistent', 'On-grid', 'Flourish'])
    ax1.legend(loc='upper right')
    ax1.set_title('Beat vs Consistent-Beat Annotation Comparison (Tool - Opiate Intro)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # Plot 2: Flourish ratio over time
    ax2 = axes[1]
    ax2.plot(times, ratios, linewidth=2, color='darkred', alpha=0.8)
    ax2.fill_between(times, ratios, alpha=0.3, color='red')

    # Add horizontal lines for mode boundaries
    ax2.axhline(70, color='orange', linestyle=':', alpha=0.5, label='Ambient threshold (70%)')
    ax2.axhline(30, color='green', linestyle=':', alpha=0.5, label='Groove threshold (30%)')

    # Add section boundaries
    for change in changes:
        ax2.axvline(change, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Annotate sections
    section_names = ['Intro', 'Build 1', 'Groove', 'Build 2', 'Peak 1', 'Peak 2', 'Outro']
    section_starts = [0] + changes

    for i, (start, name) in enumerate(zip(section_starts, section_names)):
        if i < len(changes):
            end = changes[i]
        else:
            end = max(beat_taps)

        mid = (start + end) / 2
        ax2.text(mid, 95, name, ha='center', va='top', fontsize=9, alpha=0.6, fontweight='bold')

    ax2.set_ylabel('Flourish Ratio (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    ax2.set_title('Flourish Ratio Over Time (5-second window)', fontsize=12)

    # Plot 3: Section-level flourish ratio
    ax3 = axes[2]

    section_ratios = []
    section_labels = []

    for i, name in enumerate(section_names):
        if i == 0:
            start, end = 0, changes[0]
        elif i < len(changes):
            start, end = changes[i-1], changes[i]
        else:
            start, end = changes[-1], 100

        section_taps = [c for c in classifications if start <= c['timestamp'] < end]
        if len(section_taps) > 0:
            flourish_count = sum(1 for c in section_taps if c['type'] == 'flourish')
            ratio = flourish_count / len(section_taps) * 100
            section_ratios.append(ratio)
            section_labels.append(f"{name}\n({len(section_taps)} taps)")
        else:
            section_ratios.append(0)
            section_labels.append(f"{name}\n(0 taps)")

    colors = ['darkred' if r > 70 else 'orange' if r > 30 else 'green' for r in section_ratios]
    bars = ax3.bar(range(len(section_ratios)), section_ratios, color=colors, alpha=0.7)

    # Add percentage labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, section_ratios)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)

    ax3.axhline(70, color='orange', linestyle=':', alpha=0.5)
    ax3.axhline(30, color='green', linestyle=':', alpha=0.5)

    ax3.set_ylabel('Flourish Ratio (%)', fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.set_xticks(range(len(section_labels)))
    ax3.set_xticklabels(section_labels, fontsize=9)
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.set_title('Flourish Ratio by Section', fontsize=12)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', alpha=0.7, label='Ambient mode (>70%)'),
        Patch(facecolor='orange', alpha=0.7, label='Accent mode (30-70%)'),
        Patch(facecolor='green', alpha=0.7, label='Groove mode (<30%)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.xlabel('Time (seconds)', fontsize=11)
    plt.tight_layout()

    print(f"Saving visualization to {OUTPUT_PATH}...")
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print("Done!")
    print(f"\nVisualization saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
