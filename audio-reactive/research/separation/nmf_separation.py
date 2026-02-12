"""
Online NMF Source Separation

Supervised NMF: train spectral dictionaries from demucs stems offline,
then decompose new audio frames in real-time using fixed dictionaries.

Training:
    Learns K spectral basis vectors per source (drums, bass, vocals, other)
    from demucs-separated stems across multiple tracks.

Inference (per-frame):
    Given magnitude spectrum v, solve v ≈ W @ h (h ≥ 0) with fixed W.
    Group activations by source to get per-source energy or Wiener masks.

ESP32 profile (64 mel bands, K=10 per source = 40 total):
    Memory: ~20KB (dictionary + precomputed matrices)
    Compute: ~80K ops/frame at 20 iterations → trivial at 240MHz

Usage:
    # Train
    from nmf_separation import train_dictionaries
    dicts = train_dictionaries(stem_dirs, n_components=10)

    # Inference
    from nmf_separation import OnlineNMF
    nmf = OnlineNMF.from_file('dictionaries.npz')
    for frame in audio_frames:
        result = nmf.process_frame(frame)
        print(result['energy'])  # {'drums': 0.3, 'bass': 0.8, ...}
"""

import numpy as np
import librosa
import os
from pathlib import Path


# ── Training ─────────────────────────────────────────────────────────

def train_dictionaries(stem_dirs, source_names=('drums', 'bass', 'vocals', 'other'),
                       n_components=10, n_mels=64, sr=44100, n_fft=2048,
                       hop_length=512, max_iter=200):
    """Learn NMF dictionaries from demucs-separated stems.

    Args:
        stem_dirs: list of directories, each containing drums.wav, bass.wav, etc.
        source_names: which stems to learn dictionaries for
        n_components: spectral templates per source (K)
        n_mels: mel frequency bins
        sr: sample rate for loading
        n_fft: FFT size
        hop_length: STFT hop
        max_iter: NMF iterations

    Returns:
        dict with:
            'W': combined dictionary (n_mels, K_total)
            'source_names': list of source names
            'source_ranges': dict mapping name -> (start_idx, end_idx) in W
            'n_mels': mel bins used
            'sr': sample rate
            'n_fft': FFT size
            'hop_length': hop length
            'W_T': precomputed W.T
            'W_T_W': precomputed W.T @ W
    """
    from sklearn.decomposition import NMF

    dictionaries = {}

    for source in source_names:
        print(f"  Training {source} dictionary ({n_components} components)...")

        # Concatenate mel spectrograms from all training tracks
        all_mels = []
        for stem_dir in stem_dirs:
            stem_path = os.path.join(stem_dir, f'{source}.wav')
            if not os.path.exists(stem_path):
                print(f"    Skipping {stem_dir} (no {source}.wav)")
                continue

            y, sr_loaded = librosa.load(stem_path, sr=sr, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, fmin=20, fmax=sr // 2
            )
            all_mels.append(mel)

        if not all_mels:
            print(f"    WARNING: no stems found for {source}, skipping")
            continue

        # Stack all frames horizontally: (n_mels, total_frames)
        V = np.hstack(all_mels)
        print(f"    {V.shape[1]} frames from {len(all_mels)} tracks")

        # Run NMF: V ≈ W @ H
        model = NMF(n_components=n_components, max_iter=max_iter,
                     init='nndsvda', random_state=42)
        W = model.fit_transform(V)  # (n_mels, n_components)
        print(f"    Reconstruction error: {model.reconstruction_err_:.1f}")

        # Normalize columns to unit norm (keeps activations interpretable)
        norms = np.linalg.norm(W, axis=0, keepdims=True)
        norms[norms == 0] = 1
        W = W / norms

        dictionaries[source] = W

    # Combine into single dictionary matrix
    source_list = [s for s in source_names if s in dictionaries]
    W_combined = np.hstack([dictionaries[s] for s in source_list])

    # Build source range index
    idx = 0
    source_ranges = {}
    for s in source_list:
        k = dictionaries[s].shape[1]
        source_ranges[s] = (idx, idx + k)
        idx += k

    # Precompute for online inference
    W_T = W_combined.T
    W_T_W = W_T @ W_combined

    result = {
        'W': W_combined,
        'source_names': source_list,
        'source_ranges': source_ranges,
        'n_mels': n_mels,
        'sr': sr,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'W_T': W_T,
        'W_T_W': W_T_W,
    }

    return result


def save_dictionaries(dicts, path):
    """Save trained dictionaries to .npz file."""
    # Convert source_ranges to arrays for numpy serialization
    names = dicts['source_names']
    ranges = np.array([dicts['source_ranges'][n] for n in names])

    np.savez(path,
             W=dicts['W'],
             source_names=np.array(names),
             source_ranges=ranges,
             n_mels=dicts['n_mels'],
             sr=dicts['sr'],
             n_fft=dicts['n_fft'],
             hop_length=dicts['hop_length'])
    print(f"Saved dictionaries to {path}")
    print(f"  Shape: {dicts['W'].shape} ({len(names)} sources, "
          f"{dicts['W'].shape[1]} total components)")


def load_dictionaries(path):
    """Load dictionaries from .npz file."""
    data = np.load(path, allow_pickle=True)
    W = data['W']
    names = list(data['source_names'])
    ranges_arr = data['source_ranges']

    source_ranges = {}
    for i, name in enumerate(names):
        source_ranges[name] = (int(ranges_arr[i, 0]), int(ranges_arr[i, 1]))

    return {
        'W': W,
        'source_names': names,
        'source_ranges': source_ranges,
        'n_mels': int(data['n_mels']),
        'sr': int(data['sr']),
        'n_fft': int(data['n_fft']),
        'hop_length': int(data['hop_length']),
        'W_T': W.T,
        'W_T_W': W.T @ W,
    }


# ── Online Inference ─────────────────────────────────────────────────

class OnlineNMF:
    """Per-frame NMF source separation with pre-trained dictionaries.

    Solves v ≈ W @ h (h ≥ 0) for each audio frame using multiplicative
    updates with fixed dictionary W.
    """

    def __init__(self, dicts, n_iter=20):
        """
        Args:
            dicts: dictionary from train_dictionaries() or load_dictionaries()
            n_iter: multiplicative update iterations per frame
        """
        self.W = dicts['W']              # (n_mels, K_total)
        self.W_T = dicts['W_T']          # (K_total, n_mels) - precomputed
        self.W_T_W = dicts['W_T_W']      # (K_total, K_total) - precomputed
        self.source_names = dicts['source_names']
        self.source_ranges = dicts['source_ranges']
        self.n_mels = dicts['n_mels']
        self.sr = dicts['sr']
        self.n_fft = dicts['n_fft']
        self.hop_length = dicts['hop_length']
        self.n_iter = n_iter
        self.K = self.W.shape[1]

        # Mel filterbank for converting STFT to mel
        self.mel_basis = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft,
            n_mels=self.n_mels, fmin=20, fmax=self.sr // 2
        )

    @classmethod
    def from_file(cls, path, n_iter=20):
        """Load from saved .npz dictionary file."""
        return cls(load_dictionaries(path), n_iter=n_iter)

    def process_frame(self, v_mel):
        """Decompose a single mel magnitude frame.

        Args:
            v_mel: mel magnitude spectrum (n_mels,) — non-negative

        Returns:
            dict with:
                'h': full activation vector (K_total,)
                'energy': dict of source_name -> scalar energy
                'masks': dict of source_name -> (n_mels,) soft mask
        """
        v = np.maximum(v_mel, 1e-10)  # ensure non-negative

        # Multiplicative updates: h <- h * (W^T @ v) / (W^T @ W @ h + eps)
        h = np.ones(self.K) * 0.1
        W_T_v = self.W_T @ v  # precompute numerator part

        for _ in range(self.n_iter):
            denom = self.W_T_W @ h + 1e-10
            h = h * W_T_v / denom

        # Reconstruct per-source spectrograms and compute energies/masks
        total_reconstruction = self.W @ h + 1e-10
        energy = {}
        masks = {}

        for name in self.source_names:
            start, end = self.source_ranges[name]
            h_source = h[start:end]
            v_source = self.W[:, start:end] @ h_source  # (n_mels,)
            energy[name] = float(np.sum(v_source))
            masks[name] = v_source / total_reconstruction  # Wiener mask

        return {
            'h': h,
            'energy': energy,
            'masks': masks,
        }

    def process_stft_frame(self, stft_mag_frame):
        """Process a raw STFT magnitude frame (n_fft//2 + 1,).

        Converts to mel internally, then runs NMF inference.
        """
        v_mel = self.mel_basis @ stft_mag_frame
        return self.process_frame(v_mel)

    def process_audio_offline(self, y, sr=None):
        """Process full audio signal offline. Returns per-source activation matrix.

        Args:
            y: mono audio signal
            sr: sample rate (uses self.sr if None)

        Returns:
            dict with:
                'activations': dict of source_name -> (n_frames,) energy over time
                'masks': dict of source_name -> (n_mels, n_frames) Wiener masks
                'mel_spec': (n_mels, n_frames) original mel spectrogram
                'times': (n_frames,) time axis
        """
        if sr is None:
            sr = self.sr

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=20, fmax=sr // 2
        )
        n_frames = mel.shape[1]
        times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=self.hop_length
        )

        # Process each frame
        activations = {name: np.zeros(n_frames) for name in self.source_names}
        masks = {name: np.zeros((self.n_mels, n_frames)) for name in self.source_names}

        for t in range(n_frames):
            result = self.process_frame(mel[:, t])
            for name in self.source_names:
                activations[name][t] = result['energy'][name]
                masks[name][:, t] = result['masks'][name]

        return {
            'activations': activations,
            'masks': masks,
            'mel_spec': mel,
            'times': times,
        }
