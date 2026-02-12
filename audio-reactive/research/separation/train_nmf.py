#!/usr/bin/env python3
"""
Train NMF dictionaries from demucs-separated stems.

Scans audio-segments/separated/htdemucs/ for available stems,
trains per-source dictionaries, and saves to dictionaries.npz.

Usage:
    python train_nmf.py
    python train_nmf.py --components 15 --mels 128
"""

import argparse
import os
import sys
from pathlib import Path

SEGMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'audio-segments')


def find_stem_dirs():
    """Find all directories with complete demucs stems."""
    htdemucs_dir = os.path.join(SEGMENTS_DIR, 'separated', 'htdemucs')
    if not os.path.exists(htdemucs_dir):
        return []

    stem_dirs = []
    required = {'drums.wav', 'bass.wav', 'vocals.wav', 'other.wav'}

    for name in sorted(os.listdir(htdemucs_dir)):
        d = os.path.join(htdemucs_dir, name)
        if not os.path.isdir(d):
            continue
        files = set(os.listdir(d))
        if required.issubset(files):
            stem_dirs.append(d)

    return stem_dirs


def main():
    parser = argparse.ArgumentParser(description='Train NMF dictionaries')
    parser.add_argument('--components', type=int, default=10,
                        help='Components per source (default 10)')
    parser.add_argument('--mels', type=int, default=64,
                        help='Mel frequency bins (default 64)')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='NMF iterations (default 200)')
    args = parser.parse_args()

    from nmf_separation import train_dictionaries, save_dictionaries

    stem_dirs = find_stem_dirs()
    if not stem_dirs:
        print("No demucs stems found. Run: python segment.py stems <file>")
        sys.exit(1)

    print(f"Found {len(stem_dirs)} tracks with demucs stems:")
    for d in stem_dirs:
        print(f"  {Path(d).name}")

    print(f"\nTraining with K={args.components} components, "
          f"{args.mels} mel bands, {args.max_iter} iterations...")

    dicts = train_dictionaries(
        stem_dirs,
        n_components=args.components,
        n_mels=args.mels,
        max_iter=args.max_iter,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'dictionaries.npz')
    save_dictionaries(dicts, out_path)

    # Print summary
    W = dicts['W']
    print(f"\nDictionary summary:")
    print(f"  Shape: {W.shape} (mel_bins x total_components)")
    for name in dicts['source_names']:
        start, end = dicts['source_ranges'][name]
        print(f"  {name}: components {start}-{end-1}")


if __name__ == '__main__':
    main()
