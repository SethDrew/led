#!/usr/bin/env python3
"""VJ Visual-Audio Correlation PoC.

Tests whether VJ lighting decisions in concert videos correlate with audio
features — proving YouTube concert footage can serve as "human-annotated"
training data for audio-reactive LED effects.

Pipeline: download → segment → video features → audio features → correlate → plot
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import cv2
import librosa

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'output'
MEDIA_DIR = OUTPUT_DIR / 'media'
FEATURES_DIR = OUTPUT_DIR / 'features'
PLOTS_DIR = OUTPUT_DIR / 'plots'

# ── Config ─────────────────────────────────────────────────────────────────

VIDEO_URL = 'https://www.youtube.com/watch?v=Ca6pjR2TLns'

SEGMENTS = [
    {'id': 'seg_01', 'start_s': 299,  'label': '4:59'},
    {'id': 'seg_02', 'start_s': 412,  'label': '6:52'},
    {'id': 'seg_03', 'start_s': 537,  'label': '8:57'},
    {'id': 'seg_04', 'start_s': 1311, 'label': '21:51 (zoomed out)'},
    {'id': 'seg_05', 'start_s': 1942, 'label': '32:22'},
    {'id': 'seg_06', 'start_s': 2170, 'label': '36:10'},
    {'id': 'seg_07', 'start_s': 3024, 'label': '50:24'},
    {'id': 'seg_08', 'start_s': 3552, 'label': '59:12'},
    {'id': 'seg_09', 'start_s': 4124, 'label': '1:08:44'},
]
SEGMENT_DURATION = 30  # seconds

# Audio analysis params (matching normalization_test.py)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000
TARGET_SR = 44100

FREQUENCY_BANDS = {
    'Sub-bass': (20, 80),
    'Bass': (80, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 6000),
    'Treble': (6000, 8000),
}
BAND_ORDER = ['Sub-bass', 'Bass', 'Mids', 'High-mids', 'Treble']

# Correlation pairs to test
CORR_PAIRS = [
    ('brightness', 'rms',              'Brightness ↔ RMS'),
    ('brightness', 'onset_strength',   'Brightness ↔ Onset'),
    ('dominant_hue', 'spectral_centroid', 'Hue ↔ Centroid'),
    ('flicker', 'onset_strength',      'Flicker ↔ Onset'),
]

# Matplotlib style
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD',
    '#98FB98', '#FFD700', '#FFA07A', '#FF69B4',
]


# ── Helpers ────────────────────────────────────────────────────────────────

def run_cmd(cmd, desc=''):
    """Run a shell command, raise on failure."""
    print(f'  {desc}' if desc else f'  $ {cmd[0]}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  STDERR: {result.stderr[:500]}')
        raise RuntimeError(f'Command failed: {" ".join(cmd[:3])}...')
    return result


def fmt_time(s):
    """Format seconds as MM:SS."""
    return f'{int(s)//60}:{int(s)%60:02d}'


# ── Step 1: Download + Segment Extraction ──────────────────────────────────

def download_video():
    """Download full video at 720p via yt-dlp."""
    full_video = MEDIA_DIR / 'full_video.mp4'
    if full_video.exists():
        print(f'  [cached] {full_video.name}')
        return full_video

    print('  Downloading video at 720p...')
    run_cmd([
        'yt-dlp',
        '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        '--merge-output-format', 'mp4',
        '-o', str(full_video),
        VIDEO_URL,
    ], 'yt-dlp download')
    return full_video


def extract_full_audio(full_video):
    """Extract mono 44.1kHz WAV from full video."""
    full_audio = MEDIA_DIR / 'full_audio.wav'
    if full_audio.exists():
        print(f'  [cached] {full_audio.name}')
        return full_audio

    run_cmd([
        'ffmpeg', '-y', '-i', str(full_video),
        '-vn', '-ac', '1', '-ar', str(TARGET_SR),
        '-acodec', 'pcm_s16le', str(full_audio),
    ], 'ffmpeg extract audio')
    return full_audio


def extract_segments(full_video):
    """Extract 30s video + audio clips for each segment."""
    for seg in SEGMENTS:
        vid_clip = MEDIA_DIR / f'{seg["id"]}_video.mp4'
        aud_clip = MEDIA_DIR / f'{seg["id"]}_audio.wav'

        if vid_clip.exists() and aud_clip.exists():
            print(f'  [cached] {seg["id"]} ({seg["label"]})')
            continue

        start = seg['start_s']
        print(f'  Extracting {seg["id"]} @ {seg["label"]} ...')

        # Video clip (re-encode for clean cuts)
        run_cmd([
            'ffmpeg', '-y', '-ss', str(start), '-i', str(full_video),
            '-t', str(SEGMENT_DURATION),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-an', str(vid_clip),
        ], f'  video clip {seg["id"]}')

        # Audio clip
        run_cmd([
            'ffmpeg', '-y', '-ss', str(start), '-i', str(full_video),
            '-t', str(SEGMENT_DURATION),
            '-vn', '-ac', '1', '-ar', str(TARGET_SR),
            '-acodec', 'pcm_s16le', str(aud_clip),
        ], f'  audio clip {seg["id"]}')


# ── Step 2: Video Features ────────────────────────────────────────────────

def extract_video_features(seg):
    """Extract per-frame video features for a segment.

    Returns dict with arrays: brightness, dominant_hue, flicker, spatial_uniformity
    """
    cache_path = FEATURES_DIR / f'{seg["id"]}_video.npz'
    if cache_path.exists():
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    vid_path = MEDIA_DIR / f'{seg["id"]}_video.mp4'
    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    brightness_list = []
    hue_list = []
    flicker_list = []
    uniformity_list = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Grayscale for brightness, flicker, uniformity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        brightness_list.append(float(np.mean(gray)))
        uniformity_list.append(float(np.std(gray)))

        # Flicker: mean absolute frame-to-frame difference
        if prev_gray is not None:
            flicker_list.append(float(np.mean(np.abs(gray - prev_gray))))
        else:
            flicker_list.append(0.0)
        prev_gray = gray

        # Dominant hue: HSV H-channel histogram mode, weighted by S*V
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)  # 0-179 in OpenCV
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        weights = s * v  # Focus on bright, colorful pixels (LEDs)

        # Weighted histogram over hue bins
        hist, _ = np.histogram(h.ravel(), bins=180, range=(0, 180),
                               weights=weights.ravel())
        if hist.sum() > 0:
            dominant_hue = float(np.argmax(hist)) / 180.0  # Normalize 0-1
        else:
            dominant_hue = 0.0
        hue_list.append(dominant_hue)

    cap.release()

    features = {
        'brightness': np.array(brightness_list),
        'dominant_hue': np.array(hue_list),
        'flicker': np.array(flicker_list),
        'spatial_uniformity': np.array(uniformity_list),
        'video_fps': np.array([fps]),
    }
    np.savez(cache_path, **features)
    return features


# ── Step 3: Audio Features ─────────────────────────────────────────────────

def extract_audio_features(seg):
    """Extract per-frame audio features for a segment.

    Returns dict with arrays: rms, onset_strength, spectral_centroid,
    and per-band energies.
    """
    cache_path = FEATURES_DIR / f'{seg["id"]}_audio.npz'
    if cache_path.exists():
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    aud_path = MEDIA_DIR / f'{seg["id"]}_audio.wav'
    y, sr = librosa.load(str(aud_path), sr=TARGET_SR, mono=True)

    audio_fps = sr / HOP_LENGTH

    # Core features
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]

    # Mel spectrogram for band energies
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)

    features = {
        'rms': rms,
        'onset_strength': onset,
        'spectral_centroid': centroid,
        'audio_fps': np.array([audio_fps]),
    }

    for band_name in BAND_ORDER:
        flo, fhi = FREQUENCY_BANDS[band_name]
        mask = (mel_freqs >= flo) & (mel_freqs <= fhi)
        features[f'band_{band_name}'] = np.sum(mel_spec[mask, :], axis=0)

    np.savez(cache_path, **features)
    return features


# ── Step 4: Alignment + Correlation ────────────────────────────────────────

def align_timeseries(video_feat, audio_feat, video_fps, audio_fps):
    """Resample video features (~30fps) to audio framerate (~86fps) via interp."""
    n_video = len(video_feat)
    n_audio = len(audio_feat)
    video_times = np.arange(n_video) / video_fps
    audio_times = np.arange(n_audio) / audio_fps
    return np.interp(audio_times, video_times, video_feat)


def normalize_01(x):
    """Min-max normalize to [0, 1]."""
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-10:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def compute_correlations(video_feats, audio_feats):
    """Compute Pearson r and p-value for each correlation pair.

    Returns list of dicts with keys: video_key, audio_key, label, r, p
    """
    video_fps = float(video_feats['video_fps'][0])
    audio_fps = float(audio_feats['audio_fps'][0])

    results = []
    for v_key, a_key, label in CORR_PAIRS:
        v = video_feats[v_key]
        a = audio_feats[a_key]

        # Align video to audio framerate
        v_aligned = align_timeseries(v, a, video_fps, audio_fps)

        # Normalize both to 0-1 for comparable correlation
        v_norm = normalize_01(v_aligned)
        a_norm = normalize_01(a)

        # Truncate to same length (edge case: off-by-one from interp)
        n = min(len(v_norm), len(a_norm))
        v_norm = v_norm[:n]
        a_norm = a_norm[:n]

        r, p = stats.pearsonr(v_norm, a_norm)
        results.append({
            'video_key': v_key,
            'audio_key': a_key,
            'label': label,
            'r': float(r),
            'p': float(p),
            'v_norm': v_norm,
            'a_norm': a_norm,
        })
    return results


# ── Step 5: Plotting ──────────────────────────────────────────────────────

def setup_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'axes.edgecolor': '#555',
        'grid.color': '#333',
        'grid.alpha': 0.3,
        'text.color': '#eee',
        'axes.labelcolor': '#ccc',
        'xtick.color': '#999',
        'ytick.color': '#999',
        'font.size': 9,
    })


def plot_segment_timeseries(seg, video_feats, audio_feats, corr_results):
    """Per-segment plot: 2 rows × 4 cols (video top, audio bottom)."""
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(20, 6))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f'{seg["id"]} @ {seg["label"]}', fontsize=14, color='white')

    video_fps = float(video_feats['video_fps'][0])
    audio_fps = float(audio_feats['audio_fps'][0])

    # Top row: video features
    v_keys = ['brightness', 'dominant_hue', 'flicker', 'spatial_uniformity']
    v_colors = ['#FFEAA7', '#FF6B6B', '#4ECDC4', '#DDA0DD']
    for i, (key, color) in enumerate(zip(v_keys, v_colors)):
        ax = axes[0, i]
        data = video_feats[key]
        t = np.arange(len(data)) / video_fps
        ax.plot(t, data, color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(key, fontsize=9)
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.set_ylabel('Video', fontsize=10)

    # Bottom row: audio features
    a_keys = ['rms', 'onset_strength', 'spectral_centroid', 'band_Bass']
    a_colors = ['#45B7D1', '#98FB98', '#FFD700', '#FFA07A']
    for i, (key, color) in enumerate(zip(a_keys, a_colors)):
        ax = axes[1, i]
        data = audio_feats[key]
        t = np.arange(len(data)) / audio_fps
        ax.plot(t, data, color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.set_ylabel('Audio', fontsize=10)

    # Add r-values as annotations
    for cr in corr_results:
        # Find matching column
        for i, key in enumerate(v_keys):
            if key == cr['video_key']:
                axes[0, i].text(
                    0.98, 0.02, f'r={cr["r"]:.3f}',
                    transform=axes[0, i].transAxes,
                    ha='right', va='bottom', fontsize=8,
                    color='white', bbox=dict(boxstyle='round,pad=0.2',
                                             facecolor='#333', alpha=0.7))

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f'{seg["id"]}_timeseries.png', dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_summary_scatter(all_corr_data):
    """2×2 scatter plot: all segments pooled, color-coded by segment."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('VJ Visual-Audio Correlation (all segments pooled)',
                 fontsize=14, color='white')

    for idx, (v_key, a_key, label) in enumerate(CORR_PAIRS):
        ax = axes[idx // 2, idx % 2]
        all_v = []
        all_a = []

        for seg_idx, (seg_id, seg_corrs) in enumerate(all_corr_data.items()):
            for cr in seg_corrs:
                if cr['video_key'] == v_key and cr['audio_key'] == a_key:
                    # Subsample for scatter readability
                    step = max(1, len(cr['v_norm']) // 200)
                    v_sub = cr['v_norm'][::step]
                    a_sub = cr['a_norm'][::step]
                    ax.scatter(a_sub, v_sub, s=3, alpha=0.3,
                               color=COLORS[seg_idx % len(COLORS)],
                               label=seg_id)
                    all_v.extend(cr['v_norm'].tolist())
                    all_a.extend(cr['a_norm'].tolist())

        # Pooled correlation
        if all_v and all_a:
            r, p = stats.pearsonr(all_v, all_a)
            ax.set_title(f'{label}\nr={r:.3f}, p={p:.2e}', fontsize=10)
        else:
            ax.set_title(label, fontsize=10)

        ax.set_xlabel(a_key, fontsize=9)
        ax.set_ylabel(v_key, fontsize=9)
        ax.grid(True, alpha=0.15)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=7, markerscale=3)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(PLOTS_DIR / 'summary_scatter.png', dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_summary_heatmap(all_results):
    """Segments × correlation pairs heatmap with r-values annotated."""
    setup_style()

    seg_ids = list(all_results.keys())
    pair_labels = [label for _, _, label in CORR_PAIRS]

    # Build r-value matrix: segments × pairs
    r_matrix = np.zeros((len(seg_ids), len(pair_labels)))
    for i, seg_id in enumerate(seg_ids):
        for j, (v_key, a_key, _) in enumerate(CORR_PAIRS):
            for cr in all_results[seg_id]:
                if cr['video_key'] == v_key and cr['audio_key'] == a_key:
                    r_matrix[i, j] = cr['r']

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#1a1a2e')

    im = ax.imshow(r_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, label='Pearson r')
    cbar.ax.yaxis.label.set_color('#ccc')
    cbar.ax.tick_params(colors='#999')

    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(seg_ids)))
    ax.set_yticklabels([f'{sid} ({SEGMENTS[i]["label"]})'
                        for i, sid in enumerate(seg_ids)], fontsize=9)
    ax.set_title('Correlation Heatmap (segments × pairs)', fontsize=13,
                 color='white')

    # Annotate cells
    for i in range(len(seg_ids)):
        for j in range(len(pair_labels)):
            val = r_matrix[i, j]
            color = 'white' if abs(val) > 0.4 else '#ccc'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'summary_heatmap.png', dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Verdict + Results ──────────────────────────────────────────────────────

def compute_verdict(pooled_results):
    """Determine verdict based on cross-segment pooled correlations."""
    signal_found = False
    weak_signal = False

    for pr in pooled_results:
        r, p = abs(pr['r']), pr['p']
        if r > 0.3 and p < 0.01:
            signal_found = True
        elif r > 0.15 and p < 0.05:
            weak_signal = True

    if signal_found:
        return 'SIGNAL FOUND'
    elif weak_signal:
        return 'WEAK SIGNAL'
    else:
        return 'NO SIGNAL'


def print_summary(all_results, pooled_results, verdict):
    """Print terminal summary table."""
    print('\n' + '=' * 80)
    print('VJ VISUAL-AUDIO CORRELATION — RESULTS')
    print('=' * 80)

    # Per-segment table
    header = f'{"Segment":<28}'
    for _, _, label in CORR_PAIRS:
        header += f' {label:>20}'
    print(header)
    print('-' * len(header))

    for seg_id, corrs in all_results.items():
        seg_info = next(s for s in SEGMENTS if s['id'] == seg_id)
        row = f'{seg_id} ({seg_info["label"]:<15})'
        for v_key, a_key, _ in CORR_PAIRS:
            for cr in corrs:
                if cr['video_key'] == v_key and cr['audio_key'] == a_key:
                    r_str = f'{cr["r"]:+.3f}'
                    sig = '*' if cr['p'] < 0.05 else ' '
                    sig = '**' if cr['p'] < 0.01 else sig
                    row += f' {r_str}{sig:>17}'
        print(row)

    # Pooled
    print('-' * len(header))
    row = f'{"POOLED (cross-segment)":<28}'
    for pr in pooled_results:
        r_str = f'{pr["r"]:+.3f}'
        sig = '*' if pr['p'] < 0.05 else ' '
        sig = '**' if pr['p'] < 0.01 else sig
        row += f' {r_str}{sig:>17}'
    print(row)

    print()
    print(f'  Verdict: {verdict}')
    print(f'  (* p<0.05, ** p<0.01)')
    print()
    if verdict == 'SIGNAL FOUND':
        print('  → VJ responses correlate with audio features.')
        print('    YouTube concert footage can serve as training data.')
    elif verdict == 'WEAK SIGNAL':
        print('  → Suggestive but not conclusive.')
        print('    Next steps: ROI masking or lag compensation.')
    else:
        print('  → No correlation at frame level.')
        print('    Next steps: ROI masking, then lagged cross-correlation.')
    print('=' * 80)


# ── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    # Ensure directories
    for d in [MEDIA_DIR, FEATURES_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Download + extract
    print('\n[Step 1] Download + segment extraction')
    full_video = download_video()
    extract_full_audio(full_video)
    extract_segments(full_video)

    # Steps 2-4: Features + correlation per segment
    print('\n[Step 2-3] Extracting features...')
    all_results = {}  # seg_id -> list of corr dicts
    all_video_feats = {}
    all_audio_feats = {}

    for seg in SEGMENTS:
        print(f'\n  Processing {seg["id"]} ({seg["label"]})...')

        print(f'    Video features...')
        vf = extract_video_features(seg)
        all_video_feats[seg['id']] = vf

        print(f'    Audio features...')
        af = extract_audio_features(seg)
        all_audio_feats[seg['id']] = af

        print(f'    Correlating...')
        corrs = compute_correlations(vf, af)
        all_results[seg['id']] = corrs

        # Per-segment timeseries plot
        plot_segment_timeseries(seg, vf, af, corrs)
        print(f'    → {seg["id"]}_timeseries.png')

    # Step 4: Pooled cross-segment correlation
    print('\n[Step 4] Pooled cross-segment correlation...')
    pooled_results = []
    for v_key, a_key, label in CORR_PAIRS:
        all_v = []
        all_a = []
        for seg_id, corrs in all_results.items():
            for cr in corrs:
                if cr['video_key'] == v_key and cr['audio_key'] == a_key:
                    all_v.extend(cr['v_norm'].tolist())
                    all_a.extend(cr['a_norm'].tolist())
        r, p = stats.pearsonr(all_v, all_a)
        pooled_results.append({
            'video_key': v_key, 'audio_key': a_key,
            'label': label, 'r': float(r), 'p': float(p),
        })

    # Step 5: Summary plots
    print('\n[Step 5] Generating summary plots...')
    plot_summary_scatter(all_results)
    print('  → summary_scatter.png')
    plot_summary_heatmap(all_results)
    print('  → summary_heatmap.png')

    # Verdict
    verdict = compute_verdict(pooled_results)

    # Save results.json
    json_results = {
        'video_url': VIDEO_URL,
        'segments': SEGMENTS,
        'per_segment': {
            seg_id: [
                {'video_key': cr['video_key'], 'audio_key': cr['audio_key'],
                 'label': cr['label'], 'r': cr['r'], 'p': cr['p']}
                for cr in corrs
            ]
            for seg_id, corrs in all_results.items()
        },
        'pooled': [
            {'video_key': pr['video_key'], 'audio_key': pr['audio_key'],
             'label': pr['label'], 'r': pr['r'], 'p': pr['p']}
            for pr in pooled_results
        ],
        'verdict': verdict,
    }
    results_path = OUTPUT_DIR / 'results.json'
    results_path.write_text(json.dumps(json_results, indent=2))
    print(f'\n  → {results_path}')

    # Terminal summary
    print_summary(all_results, pooled_results, verdict)


if __name__ == '__main__':
    main()
