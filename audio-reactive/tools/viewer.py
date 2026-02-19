#!/usr/bin/env python3
"""
Synced Audio Playback + Visualization + Annotation

Interactive viewer with waveform, spectrogram, band energy, onset detection,
spectral centroid, and user annotation panels. Supports tap-to-annotate mode
where you can record feeling-based annotations while seeing the analysis.

Annotations can be point taps (list of timestamps) or labeled segment spans
(list of {start, end, label} dicts).

Usage (via segment.py):
    python segment.py play "Opiate Intro.wav"
    python segment.py play "Opiate Intro.wav" --annotate beat

Dependencies:
    pip install sounddevice soundfile numpy librosa matplotlib pyyaml
"""

import sys
import time
import threading
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
try:
    import sounddevice as sd
except OSError:
    sd = None  # PortAudio not available (headless server)
import soundfile as sf
import yaml

# ── Shared Constants ──────────────────────────────────────────────────
# Single source of truth for frequency bands and colors

FREQUENCY_BANDS = {
    'Sub-bass': (20, 80),
    'Bass': (80, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 6000),
    'Treble': (6000, 8000),
}

BAND_COLORS = {
    'Sub-bass': '#FF1744',
    'Bass': '#FF9100',
    'Mids': '#FFEA00',
    'High-mids': '#00E676',
    'Treble': '#00B0FF',
}

ANNOTATION_COLORS = ['#FF4081', '#40C4FF', '#69F0AE', '#FFD740', '#E040FB', '#FF6E40']


def _checkerboard_novelty(S, kernel_size):
    """Foote's (2000) checkerboard kernel novelty along the diagonal of a similarity matrix.

    Places a Gaussian-tapered checkerboard kernel at each point along the diagonal.
    High values where the music's character changes (cross-block similarity drops).
    """
    from scipy.signal import windows as sig_windows

    w = kernel_size // 2
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:w, w:] = -1   # top-right: past vs future
    kernel[w:, :w] = -1   # bottom-left: future vs past
    g = sig_windows.gaussian(kernel_size, std=kernel_size / 4)
    kernel *= np.outer(g, g)

    n = S.shape[0]
    novelty = np.zeros(n)
    for i in range(n):
        r0 = max(0, i - w)
        r1 = min(n, i + w)
        k_r0 = w - (i - r0)
        k_r1 = w + (r1 - i)
        patch = S[r0:r1, r0:r1]
        novelty[i] = np.sum(patch * kernel[k_r0:k_r1, k_r0:k_r1])

    novelty = np.maximum(novelty, 0)
    mx = np.max(novelty)
    if mx > 0:
        novelty /= mx
    return novelty


# ── Interactive Visualizer ────────────────────────────────────────────

class SyncedVisualizer:
    def __init__(self, filepath, focus_panel=None, show_beats=False,
                 annotations_path=None, annotate_layer=None,
                 led_effect=None, led_output=None, led_brightness=1.0,
                 panels=None):
        self.filepath = filepath
        self.focus_panel = focus_panel
        self.panels = panels  # optional list of panel names to show
        self.show_beats = show_beats
        self.annotations_path_override = annotations_path
        self.annotate_layer = annotate_layer  # None = normal mode, str = annotation mode
        self.playback_start_time = None
        self.playback_offset = 0.0
        self.is_playing = False

        # LED output (optional — driven in sync with playback cursor)
        self.led_effect = led_effect
        self.led_output = led_output
        self.led_brightness = led_brightness
        self.led_sample_pos = 0  # tracks how far we've fed audio to the effect

        # Annotation state
        self.new_taps = []
        self.feature_toggle = {}
        self.feature_keys = {}

        print(f"Loading: {filepath}")

        # Load for analysis (mono, librosa resampled)
        self.y, self.sr = librosa.load(filepath, sr=None, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

        # Load for playback (original channels)
        self.y_playback, self.sr_playback = sf.read(filepath)

        print(f"Duration: {self.duration:.1f}s, Sample rate: {self.sr} Hz")

        # Load annotations if present
        self.annotations = {}
        if self.annotations_path_override:
            ann_path = Path(self.annotations_path_override)
        else:
            ann_path = Path(filepath).with_suffix('.annotations.yaml')
            if not ann_path.exists():
                ann_path = Path(filepath).parent / (Path(filepath).stem.rsplit('_short', 1)[0] + '.annotations.yaml')
        self.annotations_path = Path(filepath).with_suffix('.annotations.yaml')
        if ann_path.exists():
            with open(ann_path) as f:
                self.annotations = yaml.safe_load(f) or {}
            print(f"Annotations: {ann_path.name} — {len(self.annotations)} layers ({', '.join(self.annotations.keys())})")

        # Load algorithm annotations (separate directory, merged into display)
        algo_ann_path = Path(filepath).parent / 'algorithm-annotations' / (Path(filepath).stem + '.yaml')
        if algo_ann_path.exists():
            with open(algo_ann_path) as f:
                algo_annotations = yaml.safe_load(f) or {}
            self.annotations.update(algo_annotations)
            print(f"Algorithm annotations: {algo_ann_path.name} — {len(algo_annotations)} layers ({', '.join(algo_annotations.keys())})")

        # Analysis parameters
        self.n_fft = 2048
        self.hop_length = 512

        self._compute_features()
        self._build_figure()

    def _compute_features(self):
        print("Computing features...")

        # Mel spectrogram - full frequency range
        mel_spec = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=64, fmin=20, fmax=None
        )
        self.mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Band energies
        mel_spec_128 = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=128, fmin=20, fmax=8000
        )
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)

        self.band_energies = {}
        self.band_ratios = {}
        for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
            band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)
            energy = np.sum(mel_spec_128[band_mask, :], axis=0)
            self.band_ratios[band_name] = np.mean(energy)
            if np.max(energy) > 0:
                self.band_energies[band_name] = energy / np.max(energy)
            else:
                self.band_energies[band_name] = energy

        # Onset detection
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length
        )
        self.onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sr, hop_length=self.hop_length
        )

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length
        )
        self.beat_times = librosa.frames_to_time(
            beat_frames, sr=self.sr, hop_length=self.hop_length
        )
        if isinstance(tempo, np.ndarray):
            self.tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            self.tempo = float(tempo)

        # Spectral centroid & RMS
        self.spectral_centroid = librosa.feature.spectral_centroid(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        self.rms_energy = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0]

        # RMS derivative (rate-of-change of loudness)
        self.rms_derivative = np.diff(self.rms_energy, prepend=self.rms_energy[0])
        dt = self.hop_length / self.sr
        self.rms_derivative = self.rms_derivative / dt  # units per second

        # Spectral centroid derivative (rate-of-change of brightness)
        self.centroid_derivative = np.diff(self.spectral_centroid, prepend=self.spectral_centroid[0])
        self.centroid_derivative = self.centroid_derivative / dt  # Hz per second

        # Per-band energy derivatives
        self.band_derivatives = {}
        for band_name, energy in self.band_energies.items():
            deriv = np.diff(energy, prepend=energy[0]) / dt
            self.band_derivatives[band_name] = deriv

        # Time axis
        self.times = librosa.frames_to_time(
            np.arange(len(self.onset_env)), sr=self.sr, hop_length=self.hop_length
        )

        # ── Foote's checkerboard novelty ──────────────────────────────
        print("Computing novelty...")
        fps = self.sr / self.hop_length
        self.mfccs = librosa.feature.mfcc(
            y=self.y, sr=self.sr, n_mfcc=13,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.chroma = librosa.feature.chroma_stft(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        def _cosine_sim(A):
            norms = np.linalg.norm(A, axis=0, keepdims=True)
            norms[norms == 0] = 1
            An = A / norms
            return (An.T @ An).astype(np.float32)

        self.novelty_kernel_seconds = 3
        self.novelty_kernel_size = max(16, int(self.novelty_kernel_seconds * fps))
        if self.novelty_kernel_size % 2 != 0:
            self.novelty_kernel_size += 1

        S = _cosine_sim(self.mfccs)
        self.novelty_mfcc = _checkerboard_novelty(S, self.novelty_kernel_size)
        del S
        S = _cosine_sim(self.chroma)
        self.novelty_chroma = _checkerboard_novelty(S, self.novelty_kernel_size)
        del S

        # ── Slow-decay peak normalized band energy (real-time sim) ──
        from scipy.ndimage import uniform_filter1d

        decay = 0.9995
        precompute_frames = int(30 * fps)  # seed from first ~30s

        raw_energies = {}
        for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
            band_mask = (mel_freqs >= fmin) & (mel_freqs <= fmax)
            raw_energies[band_name] = np.sum(mel_spec_128[band_mask, :], axis=0)

        self.band_energy_rt = {}
        for band_name, energy in raw_energies.items():
            peaks = np.empty_like(energy)
            seed = np.max(energy[:min(precompute_frames, len(energy))]) if len(energy) > 0 else 1e-10
            peak = max(seed, 1e-10)
            for i in range(len(energy)):
                peak = max(energy[i], peak * decay)
                peaks[i] = peak
            self.band_energy_rt[band_name] = energy / peaks

        # Total raw energy for dual-axis overlay
        self.total_raw_energy = sum(raw_energies.values())

        # ── Band share (each band as % of total per frame) ──
        all_bands = np.stack([raw_energies[b] for b in FREQUENCY_BANDS])
        totals = np.sum(all_bands, axis=0) + 1e-10
        self.band_share = {b: raw_energies[b] / totals * 100 for b in FREQUENCY_BANDS}

        # ── Band RT derivatives (rate-of-change of real-time normalized) ──
        self.band_rt_derivatives = {}
        for band_name, energy in self.band_energy_rt.items():
            deriv = np.diff(energy, prepend=energy[0]) / dt
            self.band_rt_derivatives[band_name] = deriv

        # ── 5s rolling integral of RT-normalized energy ──
        integral_window = int(5 * fps)
        self.band_integral_5s = {}
        for band_name, energy in self.band_energy_rt.items():
            self.band_integral_5s[band_name] = uniform_filter1d(
                energy, size=integral_window, mode='reflect') * integral_window

        # ── Band deviation from long-term context ────────────────────
        self.band_deviation_context_seconds = 60
        context_frames = max(3, int(self.band_deviation_context_seconds * fps))
        self.band_deviations = {}
        for band_name, energy in self.band_energies.items():
            running_mean = uniform_filter1d(energy.astype(np.float64), size=context_frames, mode='reflect')
            running_sq = uniform_filter1d((energy ** 2).astype(np.float64), size=context_frames, mode='reflect')
            running_std = np.sqrt(np.maximum(running_sq - running_mean ** 2, 0))
            self.band_deviations[band_name] = np.abs(energy - running_mean) / (running_std + 1e-10)
        all_z = np.stack([self.band_deviations[b][:len(self.times)] for b in FREQUENCY_BANDS], axis=0)
        self.composite_deviation = np.max(all_z, axis=0)

        self._compute_events()

        print(f"Tempo: {self.tempo:.1f} BPM, {len(self.onset_times)} onsets, {len(self.beat_times)} beats")

    @staticmethod
    def _find_spans(mask, times, min_frames):
        """Find contiguous True runs >= min_frames, return [(start_time, end_time), ...]."""
        spans = []
        in_span = False
        start_idx = 0
        for i, v in enumerate(mask):
            if v and not in_span:
                in_span = True
                start_idx = i
            elif not v and in_span:
                if i - start_idx >= min_frames:
                    spans.append((times[start_idx], times[min(i, len(times) - 1)]))
                in_span = False
        if in_span and len(mask) - start_idx >= min_frames:
            spans.append((times[start_idx], times[len(mask) - 1]))
        return spans

    def _compute_events(self):
        """Detect structural audio events: drops, risers, dropouts, harmonic sections."""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d

        fps = self.sr / self.hop_length
        t = self.times
        n = len(t)

        # ── Drops: MFCC novelty peak within ±0.5s of positive RMS derivative peak ──
        nov_m = self.novelty_mfcc[:n]
        rms_d = self.rms_derivative[:n]
        rms_d_norm = rms_d / (np.max(np.abs(rms_d)) + 1e-10)

        nov_peaks, _ = find_peaks(nov_m, prominence=0.2)
        rms_peaks, _ = find_peaks(rms_d_norm, height=0.3)

        self.event_drops = []
        for np_idx in nov_peaks:
            for rp_idx in rms_peaks:
                if abs(t[np_idx] - t[rp_idx]) <= 0.5:
                    self.event_drops.append(t[np_idx])
                    break

        # ── Risers: smoothed centroid derivative sustained positive ──
        cent_d = gaussian_filter1d(self.centroid_derivative[:n], sigma=10)
        cent_d_max = np.max(np.abs(cent_d)) + 1e-10
        riser_mask = cent_d > 0.08 * cent_d_max
        min_riser_frames = max(1, int(1.0 * fps))
        self.event_risers = self._find_spans(riser_mask, t, min_riser_frames)

        # ── Dropouts: normalized RMS below threshold ──
        rms_norm = self.rms_energy[:n] / (np.max(self.rms_energy[:n]) + 1e-10)
        dropout_mask = rms_norm < 0.05
        min_dropout_frames = max(1, int(0.1 * fps))
        self.event_dropouts = self._find_spans(dropout_mask, t, min_dropout_frames)

        # ── Harmonic: smoothed onset strength below threshold ──
        onset_smooth = gaussian_filter1d(self.onset_env[:n], sigma=0.5 * fps)
        onset_max = np.max(onset_smooth) + 1e-10
        harmonic_mask = onset_smooth < 0.10 * onset_max
        min_harmonic_frames = max(1, int(1.0 * fps))
        self.event_harmonic = self._find_spans(harmonic_mask, t, min_harmonic_frames)

        print(f"Events: {len(self.event_drops)} drops, {len(self.event_risers)} risers, "
              f"{len(self.event_dropouts)} dropouts, {len(self.event_harmonic)} harmonic")

    def _build_figure(self):
        print("Building visualization...")
        plt.style.use('dark_background')

        # Clear matplotlib default keybindings that conflict with our feature toggles
        for param in ['keymap.zoom', 'keymap.back', 'keymap.forward']:
            plt.rcParams[param] = []

        # Determine if we need annotation panel
        has_annotations = bool(self.annotations) or self.annotate_layer is not None

        if self.panels is not None:
            pass  # figure created below in panels branch
        elif self.focus_panel is not None:
            self.fig = plt.figure(figsize=(18, 10))
        elif has_annotations:
            self.fig = plt.figure(figsize=(18, 36))
        else:
            self.fig = plt.figure(figsize=(18, 32))

        # Determine which panels to show
        # Focus panel mode: show one panel maximized
        focus_builders = {
            'waveform': self._build_waveform_panel,
            'spectrogram': self._build_spectrogram_panel,
            'bands': self._build_band_energy_panel,
            'rms-derivative': self._build_rms_derivative_panel,
            'centroid': self._build_centroid_panel,
            'centroid-derivative': self._build_centroid_derivative_panel,
            'band-derivative': self._build_band_derivative_panel,
            'band-rt': self._build_band_rt_panel,
            'band-share': self._build_band_share_panel,
            'band-context-deviation': self._build_band_context_deviation_panel,
            'band-rt-derivative': self._build_band_rt_derivative_panel,
            'band-integral': self._build_band_integral_panel,
            'mfcc': self._build_mfcc_panel,
            'novelty': self._build_novelty_panel,
            'band-deviation': self._build_band_deviation_panel,
        }
        if has_annotations:
            focus_builders['annotations'] = self._build_annotation_panel

        if self.panels is not None:
            # Custom panel selection (e.g. annotate tab: waveform + spectrogram only)
            builders = [(name, focus_builders[name]) for name in self.panels
                        if name in focus_builders]
            n_panels = len(builders)
            height_map = {'waveform': 1, 'spectrogram': 2, 'bands': 1,
                          'rms-derivative': 1, 'centroid': 1, 'centroid-derivative': 1,
                          'band-derivative': 1, 'mfcc': 1.5, 'novelty': 1.5,
                          'band-deviation': 1.5, 'annotations': None}
            ratios = [height_map.get(name, 1) for name, _ in builders]
            if has_annotations:
                n_layers = len(self.annotations) + (1 if self.annotate_layer else 0)
                ann_height = max(0.8, 0.4 * n_layers)
                builders.append(('annotations', self._build_annotation_panel))
                ratios.append(ann_height)
            total = len(builders)
            fig_height = sum(ratios) * 2 + 2
            self.fig = plt.figure(figsize=(18, fig_height))
            gs = gridspec.GridSpec(total, 1, height_ratios=ratios, hspace=0.3)
            for i, (name, builder) in enumerate(builders):
                builder(self.fig.add_subplot(gs[i]))
        elif self.focus_panel in focus_builders:
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            focus_builders[self.focus_panel](self.fig.add_subplot(gs[0]))
        elif has_annotations:
            n_layers = len(self.annotations) + (1 if self.annotate_layer else 0)
            ann_height = max(0.8, 0.4 * n_layers)
            gs = gridspec.GridSpec(11, 1, height_ratios=[1, 2, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, ann_height], hspace=0.3)
            self._build_waveform_panel(self.fig.add_subplot(gs[0]))
            self._build_spectrogram_panel(self.fig.add_subplot(gs[1]))
            self._build_band_energy_panel(self.fig.add_subplot(gs[2]))
            self._build_rms_derivative_panel(self.fig.add_subplot(gs[3]))
            self._build_centroid_panel(self.fig.add_subplot(gs[4]))
            self._build_centroid_derivative_panel(self.fig.add_subplot(gs[5]))
            self._build_band_derivative_panel(self.fig.add_subplot(gs[6]))
            self._build_mfcc_panel(self.fig.add_subplot(gs[7]))
            self._build_novelty_panel(self.fig.add_subplot(gs[8]))
            self._build_band_deviation_panel(self.fig.add_subplot(gs[9]))
            self._build_annotation_panel(self.fig.add_subplot(gs[10]))
        else:
            gs = gridspec.GridSpec(10, 1, height_ratios=[1, 2, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5], hspace=0.3)
            self._build_waveform_panel(self.fig.add_subplot(gs[0]))
            self._build_spectrogram_panel(self.fig.add_subplot(gs[1]))
            self._build_band_energy_panel(self.fig.add_subplot(gs[2]))
            self._build_rms_derivative_panel(self.fig.add_subplot(gs[3]))
            self._build_centroid_panel(self.fig.add_subplot(gs[4]))
            self._build_centroid_derivative_panel(self.fig.add_subplot(gs[5]))
            self._build_band_derivative_panel(self.fig.add_subplot(gs[6]))
            self._build_mfcc_panel(self.fig.add_subplot(gs[7]))
            self._build_novelty_panel(self.fig.add_subplot(gs[8]))
            self._build_band_deviation_panel(self.fig.add_subplot(gs[9]))

        self._finalize_figure()

    def _build_waveform_panel(self, ax):
        waveform_times = np.linspace(0, self.duration, len(self.y))
        ax.plot(waveform_times, self.y, color='#FFFFFF', linewidth=0.5, alpha=0.8)

        # RMS energy overlay (normalized to waveform amplitude range, hidden by default)
        rms_norm = self.rms_energy / np.max(self.rms_energy)
        y_max = np.max(np.abs(self.y)) if np.max(np.abs(self.y)) > 0 else 1.0
        rms_scaled = rms_norm * y_max
        rms_line, = ax.plot(self.times, rms_scaled,
                            color='#FFD740', linewidth=2, label='RMS [E]',
                            alpha=0.8, visible=False)
        # Store for feature toggle system
        if not hasattr(self, 'feature_toggle'):
            self.feature_toggle = {}
            self.feature_keys = {}
        self.feature_toggle['rms'] = {'artists': [rms_line], 'visible': False}
        self.feature_keys['e'] = 'rms'

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform — [E] toggle RMS overlay', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_spectrogram_panel(self, ax):
        librosa.display.specshow(
            self.mel_spec_db, sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='mel', fmin=20, fmax=None,
            ax=ax, cmap='magma'
        )
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Mel Spectrogram — full frequency range (20 Hz - 22 kHz)', fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_energy_panel(self, ax):
        for band_name, energy in self.band_energies.items():
            ax.plot(self.times, energy, label=band_name,
                    color=BAND_COLORS[band_name], linewidth=2, alpha=0.9)
        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Normalized Energy')
        ax.set_title('Band Energy', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        max_ratio = max(self.band_ratios.values())
        ratio_text = "Magnitude ratios:\n"
        for name, ratio in self.band_ratios.items():
            ratio_text += f"  {name}: {ratio / max_ratio:.2f}\n"
        ax.text(0.02, 0.98, ratio_text, transform=ax.transAxes,
                va='top', fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_rms_derivative_panel(self, ax):
        """RMS derivative: rate-of-change of loudness. Positive = getting louder, negative = getting quieter."""
        pos = np.maximum(self.rms_derivative, 0)
        neg = np.minimum(self.rms_derivative, 0)

        ax.fill_between(self.times, pos, color='#FF5252', alpha=0.7, linewidth=0)
        ax.fill_between(self.times, neg, color='#448AFF', alpha=0.7, linewidth=0)
        ax.axhline(y=0, color='#666', linewidth=0.5)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('dRMS/dt')
        ax.set_title('RMS Derivative — red = getting louder, blue = getting quieter', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_centroid_panel(self, ax):
        """Spectral centroid: center of mass of the frequency spectrum."""
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(self.spectral_centroid, sigma=5)

        ax.plot(self.times, self.spectral_centroid, color='#B388FF', linewidth=0.5, alpha=0.3)
        ax.plot(self.times, smoothed, color='#B388FF', linewidth=2)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Center of Mass of Frequency — how bright the sound is right now', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_centroid_derivative_panel(self, ax):
        """Spectral centroid derivative: rising = getting brighter, falling = getting darker."""
        # Smooth to reduce noise (centroid is noisy frame-to-frame)
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(self.centroid_derivative, sigma=5)

        pos = np.maximum(smoothed, 0)
        neg = np.minimum(smoothed, 0)

        ax.fill_between(self.times, pos, color='#B388FF', alpha=0.7, linewidth=0)
        ax.fill_between(self.times, neg, color='#00BFA5', alpha=0.7, linewidth=0)
        ax.axhline(y=0, color='#666', linewidth=0.5)

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('dCentroid/dt (Hz/s)')
        ax.set_title('Centroid Derivative — purple = brighter, teal = darker', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_derivative_panel(self, ax):
        """Per-band energy derivatives: rate-of-change per frequency band."""
        from scipy.ndimage import gaussian_filter1d

        for band_name, deriv in self.band_derivatives.items():
            smoothed = gaussian_filter1d(deriv, sigma=5)
            ax.plot(self.times, smoothed, label=band_name,
                    color=BAND_COLORS[band_name], linewidth=1.5, alpha=0.85)

        ax.axhline(y=0, color='#666', linewidth=0.5)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('dEnergy/dt')
        ax.set_title('Band Energy Derivative — per-band rate of change', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_mfcc_panel(self, ax):
        """MFCC heatmap: the timbral fingerprint over time."""
        ax.imshow(self.mfccs, aspect='auto', origin='lower', cmap='magma',
                  extent=[0, self.duration, 0, 13])
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('MFCC')
        ax.set_yticks(np.arange(13) + 0.5)
        ax.set_yticklabels([str(i) for i in range(13)])
        ax.set_title('Timbral Shape (MFCC) — vertical color shifts = the sound character changed', fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_novelty_panel(self, ax):
        """Foote's checkerboard novelty: peaks where the music's character changes.
        Chroma on left axis, MFCC on right axis (independent scales)."""
        from scipy.signal import find_peaks

        t = self.times
        n = len(t)
        nov_m = self.novelty_mfcc[:n]
        nov_c = self.novelty_chroma[:n]

        fps = self.sr / self.hop_length
        peak_dist = max(1, int(fps * 1.5))

        # Chroma on left axis
        line_c, = ax.plot(t, nov_c, color='#4DD0E1', linewidth=1.2, alpha=0.9, label='Chroma (harmonic)')
        peaks_c, _ = find_peaks(nov_c, prominence=0.15, distance=peak_dist)
        ax.scatter(t[peaks_c], nov_c[peaks_c], color='#4DD0E1', s=25, zorder=5, marker='v')
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Chroma novelty', color='#4DD0E1')
        ax.tick_params(axis='y', labelcolor='#4DD0E1')
        ax.grid(True, alpha=0.2)

        # MFCC on right axis (independent scale)
        ax2 = ax.twinx()
        line_m, = ax2.plot(t, nov_m, color='#FF8A65', linewidth=1.2, alpha=0.9, label='MFCC (timbral)')
        peaks_m, _ = find_peaks(nov_m, prominence=0.15, distance=peak_dist)
        ax2.scatter(t[peaks_m], nov_m[peaks_m], color='#FF8A65', s=25, zorder=5, marker='v')
        ax2.set_ylabel('MFCC novelty', color='#FF8A65')
        ax2.tick_params(axis='y', labelcolor='#FF8A65')

        ks = self.novelty_kernel_size
        ksec = self.novelty_kernel_seconds
        ax.set_title(
            f"Foote's Checkerboard Novelty (kernel={ks} frames, {ksec}s) "
            f"— peaks = section boundaries  [V] events", fontsize=11)
        ax.legend([line_c, line_m], ['Chroma (harmonic)', 'MFCC (timbral)'],
                  loc='upper right', framealpha=0.8, fontsize=8)

        # ── Event overlays (hidden by default, toggled with V) ──
        event_artists = []
        for drop_t in self.event_drops:
            a = ax.axvline(x=drop_t, color='#FF1744', linestyle='--',
                           linewidth=2, alpha=0.8, visible=False)
            event_artists.append(a)
        for start, end in self.event_risers:
            a = ax.axvspan(start, end, color='#76FF03', alpha=0.15, visible=False)
            event_artists.append(a)
        for start, end in self.event_dropouts:
            a = ax.axvspan(start, end, color='#E040FB', alpha=0.15, visible=False)
            event_artists.append(a)
        for start, end in self.event_harmonic:
            a = ax.axvspan(start, end, color='#FFD740', alpha=0.10, visible=False)
            event_artists.append(a)

        if not hasattr(self, 'feature_toggle'):
            self.feature_toggle = {}
            self.feature_keys = {}
        self.feature_toggle['events'] = {'artists': event_artists, 'visible': False}
        self.feature_keys['v'] = 'events'

        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_deviation_panel(self, ax):
        """Band deviation from long-term context: per-band z-scores."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_deviations[band_name][:n], color=BAND_COLORS[band_name],
                    linewidth=1.0, alpha=0.7, label=band_name)
        ax.plot(t, self.composite_deviation[:n], color='#FFFFFF',
                linewidth=1.5, alpha=0.9, label='Max (any band)')

        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Z-score')
        cs = self.band_deviation_context_seconds
        ax.set_title(
            f'Band Deviation from {cs}s Running Context '
            f'— high = spectral balance shifted', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    # ── Band Analysis panels ─────────────────────────────────────────

    def _build_band_rt_panel(self, ax):
        """Slow-decay peak normalized band energy (simulating real-time effects view)."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_energy_rt[band_name][:n],
                    color=BAND_COLORS[band_name], linewidth=1.5, alpha=0.85,
                    label=band_name)
        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Normalized Energy')
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        # Dual y-axis: total raw energy as thin white line
        ax2 = ax.twinx()
        total = self.total_raw_energy[:n]
        ax2.plot(t, total, color='#FFFFFF', linewidth=0.8, alpha=0.5)
        ax2.set_ylabel('Total Raw', color='#888')
        ax2.tick_params(axis='y', labelcolor='#888')

        ax.set_title('Real-Time View (slow-decay peak, 30s precompute)', fontsize=11)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_share_panel(self, ax):
        """Band share: each band as % of total energy per frame."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_share[band_name][:n],
                    color=BAND_COLORS[band_name], linewidth=1.5, alpha=0.85,
                    label=band_name)
        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 100])
        ax.set_ylabel('% of Total')
        ax.set_title('Band Share of Total Energy (%)', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_context_deviation_panel(self, ax):
        """Running context deviation: per-band z-scores from 60s rolling mean."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_deviations[band_name][:n],
                    color=BAND_COLORS[band_name], linewidth=1.0, alpha=0.7,
                    label=band_name)
        ax.plot(t, self.composite_deviation[:n], color='#FFFFFF',
                linewidth=1.5, alpha=0.9, label='Max (any band)')
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Z-score')
        ax.set_title('Band Deviation from 60s Context (z-score)', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_rt_derivative_panel(self, ax):
        """Band derivative of real-time normalized energy."""
        from scipy.ndimage import gaussian_filter1d
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            smoothed = gaussian_filter1d(self.band_rt_derivatives[band_name][:n], sigma=5)
            ax.plot(t, smoothed, color=BAND_COLORS[band_name],
                    linewidth=1.5, alpha=0.85, label=band_name)
        ax.axhline(y=0, color='#666', linewidth=0.5)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('dEnergy/dt')
        ax.set_title('Band Energy Derivative (real-time normalized)', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_band_integral_panel(self, ax):
        """5s rolling integral of real-time normalized energy."""
        t = self.times
        n = len(t)
        for band_name in FREQUENCY_BANDS:
            ax.plot(t, self.band_integral_5s[band_name][:n],
                    color=BAND_COLORS[band_name], linewidth=1.5, alpha=0.85,
                    label=band_name)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Integrated Energy')
        ax.set_title('5s Rolling Integral', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_features_panel(self, ax):
        """Second-class features: onset strength, centroid (hidden by default)."""
        self.features_ax = ax

        # Onset strength line + detection markers (hidden by default)
        onset_norm = self.onset_env / np.max(self.onset_env)
        onset_line, = ax.plot(self.times, onset_norm,
                              color='#00E5FF', linewidth=1.5, label='Onset [O]',
                              alpha=0.9, visible=False)
        onset_markers = []
        for t in self.onset_times:
            m = ax.axvline(x=t, color='#FFC400', alpha=0.15, linewidth=0.8, visible=False)
            onset_markers.append(m)

        # Spectral centroid (normalized, hidden by default)
        centroid_norm = (self.spectral_centroid - np.min(self.spectral_centroid))
        centroid_range = np.max(self.spectral_centroid) - np.min(self.spectral_centroid)
        if centroid_range > 0:
            centroid_norm = centroid_norm / centroid_range
        centroid_line, = ax.plot(self.times, centroid_norm,
                                 color='#B388FF', linewidth=2, label='Centroid [C]',
                                 alpha=0.9, visible=False)

        # Librosa-beats markers (hidden by default unless --show-beats)
        beat_markers = []
        for t in self.beat_times:
            m = ax.axvline(x=t, color='#FF1744', alpha=0.6, linewidth=2,
                           visible=self.show_beats)
            beat_markers.append(m)

        # Feature toggle state (onset, centroid, beats — RMS is on waveform panel)
        self.feature_toggle.update({
            'onset': {'artists': [onset_line] + onset_markers, 'visible': False},
            'centroid': {'artists': [centroid_line], 'visible': False},
            'librosa-beats': {'artists': beat_markers, 'visible': self.show_beats},
        })
        self.feature_keys.update({'o': 'onset', 'c': 'centroid', 'b': 'librosa-beats'})

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Normalized')
        self._update_features_title()
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _update_features_title(self):
        """Update features panel title to reflect toggle states."""
        if not hasattr(self, 'features_ax'):
            return

    def _toggle_feature(self, name):
        """Toggle visibility of a feature overlay."""
        feat = self.feature_toggle[name]
        feat['visible'] = not feat['visible']
        for artist in feat['artists']:
            artist.set_visible(feat['visible'])
        self._update_features_title()
        self.fig.canvas.draw_idle()
        print(f"{name}: {'on' if feat['visible'] else 'off'}")

    def _is_segment_layer(self, layer_data):
        """Check if a layer contains segment spans (list of dicts) vs point taps (list of floats)."""
        return (isinstance(layer_data, list) and len(layer_data) > 0
                and isinstance(layer_data[0], dict) and 'start' in layer_data[0])

    def _draw_layer(self, ax, layer_name, layer_data, y_pos, color):
        """Draw a single annotation layer — handles both point taps and segment spans."""
        if self._is_segment_layer(layer_data):
            # Segment spans: colored bars with centered labels
            for seg in layer_data:
                start = seg['start']
                end = seg['end']
                label = seg.get('label', '')
                if start <= self.duration:
                    ax.barh(y_pos, width=end - start, left=start, height=0.6,
                            color=color, alpha=0.25, edgecolor=color, linewidth=0.5)
                    mid = (start + min(end, self.duration)) / 2
                    ax.text(mid, y_pos, label, ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold', alpha=0.9)
        else:
            # Point taps: vertical markers
            taps = layer_data if isinstance(layer_data, list) else []
            for t in taps:
                if isinstance(t, (int, float)) and t <= self.duration:
                    ax.plot(t, y_pos, '|', color=color, markersize=12, markeredgewidth=2, alpha=0.8)

    def _build_annotation_panel(self, ax):
        """Annotation panel: existing layers + live annotation layer."""
        self.annotation_ax = ax  # Save reference for live tap rendering

        layer_names = list(self.annotations.keys())
        # Add the live annotation layer slot if annotating
        if self.annotate_layer and self.annotate_layer not in layer_names:
            layer_names.append(self.annotate_layer)

        total_layers = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            y_pos = total_layers - i

            layer_data = self.annotations.get(layer_name, [])
            self._draw_layer(ax, layer_name, layer_data, y_pos, color)

            ax.text(-self.duration * 0.01, y_pos, layer_name, ha='right', va='center',
                    fontsize=8, color=color, fontweight='bold')

            # Remember the y_pos and color for the live annotation layer
            if layer_name == self.annotate_layer:
                self._annotate_y_pos = y_pos
                self._annotate_color = color

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0.3, total_layers + 0.7])
        ax.set_yticks([])

        # Count taps and segments separately
        total_taps = 0
        total_segs = 0
        for v in self.annotations.values():
            if self._is_segment_layer(v):
                total_segs += len(v)
            else:
                total_taps += len(v)
        parts = []
        if total_taps:
            parts.append(f'{total_taps} taps')
        if total_segs:
            parts.append(f'{total_segs} segments')
        count_str = ', '.join(parts) if parts else 'empty'
        ax.set_title(
            f'Annotations — {count_str} across {len(self.annotations)} layers',
            fontsize=11
        )
        ax.grid(True, axis='x', alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _finalize_figure(self):
        # Create cursor lines
        self.cursor_lines = []
        for ax in self.axes:
            line = ax.axvline(x=0, color='#FFFFFF', linewidth=1.5, alpha=0.9, linestyle='-')
            self.cursor_lines.append(line)

        filename = Path(self.filepath).name

        if self.annotate_layer:
            title = (
                f'{filename}  —  Layer: {self.annotate_layer}  —  '
                f'SPACE: tap | P: play/pause | R: restart | Q: save & quit'
            )
        else:
            title = f'{filename}  —  SPACE: play  |  CLICK: seek  |  E: RMS overlay  |  Q: quit'

        self.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        if self.annotate_layer:
            print(f"\nAnnotation mode: Layer '{self.annotate_layer}'")
            print("  SPACE  - Record a tap")
            print("  P      - Play / Pause")
            print("  R      - Restart + clear taps")
            print("  Q      - Save & quit")
        else:
            print("Ready. SPACE: play/pause  CLICK: seek  E: RMS overlay  Q: quit")

    def _on_key(self, event):
        if self.annotate_layer:
            self._on_key_annotate(event)
        else:
            self._on_key_normal(event)

    def _on_key_normal(self, event):
        if event.key == ' ':
            if not self.is_playing:
                self._start_playback()
            else:
                self._stop_playback()
        elif event.key in self.feature_keys:
            self._toggle_feature(self.feature_keys[event.key])
        elif event.key == 'q':
            self._stop_playback()
            plt.close(self.fig)

    def _on_key_annotate(self, event):
        if event.key == ' ':
            # Record tap
            if self.is_playing and self.playback_start_time is not None:
                tap_time = (time.time() - self.playback_start_time) + self.playback_offset
                if 0 <= tap_time <= self.duration:
                    self.new_taps.append(tap_time)
                    # Render tap marker on annotation panel
                    if hasattr(self, 'annotation_ax') and hasattr(self, '_annotate_y_pos'):
                        self.annotation_ax.plot(
                            tap_time, self._annotate_y_pos, '|',
                            color=self._annotate_color,
                            markersize=12, markeredgewidth=2, alpha=0.8
                        )
                        self.fig.canvas.draw_idle()
                    print(f"\r  [{tap_time:6.2f}s] TAP #{len(self.new_taps)}", end='', flush=True)
        elif event.key == 'p':
            if not self.is_playing:
                self._start_playback()
            else:
                self._stop_playback()
        elif event.key == 'r':
            # Restart: stop, clear taps, rewind, restart
            self._stop_playback()
            self.new_taps = []
            self.playback_offset = 0.0
            # Clear tap markers from annotation panel by redrawing
            if hasattr(self, 'annotation_ax'):
                ax = self.annotation_ax
                # Remove only the new tap markers (they are the most recent artists)
                # Simpler: just redraw the annotation panel
                ax.clear()
                self._rebuild_annotation_panel_contents(ax)
                self.fig.canvas.draw_idle()
            print("\r  Restarted. Taps cleared.                    ", end='', flush=True)
            time.sleep(0.1)
            self._start_playback(from_offset=0.0)
        elif event.key in self.feature_keys:
            self._toggle_feature(self.feature_keys[event.key])
        elif event.key == 'q':
            self._stop_playback()
            self._save_annotations()
            plt.close(self.fig)

    def _rebuild_annotation_panel_contents(self, ax):
        """Redraw annotation panel contents (used after restart clears taps)."""
        layer_names = list(self.annotations.keys())
        if self.annotate_layer and self.annotate_layer not in layer_names:
            layer_names.append(self.annotate_layer)

        total_layers = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            y_pos = total_layers - i

            layer_data = self.annotations.get(layer_name, [])
            self._draw_layer(ax, layer_name, layer_data, y_pos, color)

            ax.text(-self.duration * 0.01, y_pos, layer_name, ha='right', va='center',
                    fontsize=8, color=color, fontweight='bold')

            if layer_name == self.annotate_layer:
                self._annotate_y_pos = y_pos
                self._annotate_color = color

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0.3, total_layers + 0.7])
        ax.set_yticks([])
        total_taps = 0
        total_segs = 0
        for v in self.annotations.values():
            if self._is_segment_layer(v):
                total_segs += len(v)
            else:
                total_taps += len(v)
        parts = []
        if total_taps:
            parts.append(f'{total_taps} taps')
        if total_segs:
            parts.append(f'{total_segs} segments')
        count_str = ', '.join(parts) if parts else 'empty'
        ax.set_title(
            f'Annotations — {count_str} across {len(self.annotations)} layers',
            fontsize=11
        )
        ax.grid(True, axis='x', alpha=0.2)

        # Re-add cursor line for this axis
        line = ax.axvline(x=self.playback_offset, color='#FFFFFF', linewidth=1.5, alpha=0.9, linestyle='-')
        # Replace the cursor line for this axis in cursor_lines
        for idx, a in enumerate(self.axes):
            if a is ax:
                self.cursor_lines[idx] = line
                break

    def _save_annotations(self):
        """Save new taps to annotations YAML (additive layer format)."""
        if not self.new_taps:
            print(f"\n\n  No taps recorded for layer '{self.annotate_layer}'.")
            return

        # Load existing annotations from the canonical path
        all_annotations = {}
        if self.annotations_path.exists():
            with open(self.annotations_path) as f:
                all_annotations = yaml.safe_load(f) or {}

        # Add/replace the target layer
        all_annotations[self.annotate_layer] = [round(t, 3) for t in sorted(self.new_taps)]

        with open(self.annotations_path, 'w') as f:
            yaml.dump(all_annotations, f, default_flow_style=False, sort_keys=False)

        print(f"\n\n  Saved {len(self.new_taps)} taps to layer '{self.annotate_layer}'")
        print(f"  File: {self.annotations_path}")
        print(f"  Layers: {list(all_annotations.keys())}")
        for layer, taps in all_annotations.items():
            print(f"    {layer}: {len(taps)} marks")

    def _on_click(self, event):
        if self.annotate_layer:
            return  # Clicks don't seek in annotation mode

        seek_time = None

        if event.inaxes in self.axes and event.xdata is not None:
            seek_time = event.xdata
        elif event.x is not None and event.y is not None:
            try:
                inv = self.axes[0].transData.inverted()
                data_coords = inv.transform((event.x, event.y))
                seek_time = data_coords[0]
            except Exception:
                pass

        if seek_time is not None and 0 <= seek_time <= self.duration:
            self._seek_to(seek_time)
            print(f"Seeking to {seek_time:.2f}s")
        elif seek_time is not None:
            print(f"Click out of range: {seek_time:.2f}s (0-{self.duration:.1f}s)")

    def _start_playback(self, from_offset=None):
        if from_offset is not None:
            self.playback_offset = from_offset

        self.is_playing = True
        self.playback_start_time = time.time()

        start_sample = int(self.playback_offset * self.sr_playback)
        audio_to_play = self.y_playback[start_sample:]

        def play_audio():
            try:
                sd.play(audio_to_play, self.sr_playback)
                sd.wait()
                self.is_playing = False
            except Exception as e:
                print(f"Playback error: {e}")
                self.is_playing = False

        self.playback_thread = threading.Thread(target=play_audio, daemon=True)
        self.playback_thread.start()
        print(f"Playing from {self.playback_offset:.2f}s...")

    def _stop_playback(self):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            self.playback_offset = min(self.playback_offset + elapsed, self.duration)

        self.is_playing = False
        sd.stop()
        time.sleep(0.05)

        print(f"Stopped at {self.playback_offset:.2f}s.")

    def _seek_to(self, time_position):
        was_playing = self.is_playing

        if self.is_playing:
            self._stop_playback()

        self.playback_offset = time_position
        self.led_sample_pos = int(time_position * self.sr)

        for line in self.cursor_lines:
            line.set_xdata([time_position, time_position])
        self.fig.canvas.draw_idle()

        if was_playing:
            time.sleep(0.05)
            self._start_playback()

    def _update_cursor(self, frame):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            current_time = self.playback_offset + elapsed

            if current_time > self.duration:
                self.is_playing = False
                current_time = self.duration

            for line in self.cursor_lines:
                line.set_xdata([current_time, current_time])

            # Feed audio to LED effect in sync with playback
            if self.led_effect is not None:
                current_sample = int(current_time * self.sr)
                while self.led_sample_pos < current_sample and self.led_sample_pos < len(self.y):
                    chunk_end = min(self.led_sample_pos + 1024, len(self.y), current_sample)
                    chunk = self.y[self.led_sample_pos:chunk_end]
                    if len(chunk) > 0:
                        self.led_effect.process_audio(chunk)
                    self.led_sample_pos = chunk_end
                led_frame = self.led_effect.render(1.0 / 30)
                led_frame = (led_frame.astype(np.float32) * self.led_brightness).astype(np.uint8)
                self.led_output.send_frame(led_frame)
        elif not self.is_playing:
            for line in self.cursor_lines:
                line.set_xdata([self.playback_offset, self.playback_offset])

        return self.cursor_lines

    def run(self):
        # Use blit=False in annotation mode so new markers render without manual blit management
        use_blit = not bool(self.annotate_layer)
        self.anim = FuncAnimation(
            self.fig, self._update_cursor,
            interval=33, blit=use_blit, cache_frame_data=False
        )
        try:
            plt.show()
        finally:
            if self.led_output is not None:
                self.led_output.close()


# ── Stem Visualizer ─────────────────────────────────────────────────


class StemVisualizer:
    """Interactive N-row stem spectrogram viewer with solo/mute and synced playback.

    Args:
        filepath: Original audio file path (for display name).
        stem_names: Ordered list of stem names (e.g. ['drums', 'bass', ...]).
        stems_playback: Dict of name -> (ndarray, sr) for audio playback.
        stems_mono: Dict of name -> (ndarray, sr) for spectrogram analysis.
    """

    def __init__(self, filepath, stem_names, stems_playback, stems_mono):
        self.filepath = filepath
        self.stem_names = stem_names
        self.stem_keys = {str(i + 1): name for i, name in enumerate(stem_names)}
        self.playback_start_time = None
        self.playback_offset = 0.0
        self.is_playing = False
        self.active_stems = {name: True for name in stem_names}

        self.stems_playback = stems_playback
        self.stems_mono = stems_mono
        self.mel_specs = {}

        # Use first stem to get duration/sr
        first = stem_names[0]
        self.sr_playback = stems_playback[first][1]
        self.sr = stems_mono[first][1]
        self.duration = librosa.get_duration(y=stems_mono[first][0], sr=self.sr)

        # Spectrogram params
        self.n_fft = 2048
        self.hop_length = 512

        self._compute_spectrograms()
        self._compute_mix()
        self._build_figure()

    def _compute_spectrograms(self):
        print("Computing spectrograms...")
        all_db = []
        for name in self.stem_names:
            y, sr = self.stems_mono[name]
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=64, fmin=20, fmax=None
            )
            db = librosa.power_to_db(mel, ref=np.max)
            self.mel_specs[name] = db
            all_db.append(db)

        self.vmin = min(d.min() for d in all_db)
        self.vmax = max(d.max() for d in all_db)
        print(f"Duration: {self.duration:.1f}s, Sample rate: {self.sr} Hz")

    def _compute_mix(self):
        """Sum active stem playback arrays into current_mix."""
        active = [name for name, on in self.active_stems.items() if on]
        if not active:
            ref = self.stems_playback[self.stem_names[0]][0]
            self.current_mix = np.zeros_like(ref)
        else:
            self.current_mix = sum(
                self.stems_playback[name][0] for name in active
            )

    def _build_figure(self):
        print("Building visualization...")
        plt.style.use('dark_background')

        n = len(self.stem_names)
        fig_height = max(6, n * 2.5)
        self.fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(n, 1, hspace=0.35)

        self.axes = []
        self.dim_overlays = {}
        self.stem_labels = {}

        for i, name in enumerate(self.stem_names):
            ax = self.fig.add_subplot(gs[i])
            librosa.display.specshow(
                self.mel_specs[name], sr=self.sr, hop_length=self.hop_length,
                x_axis='time', y_axis='mel', fmin=20, fmax=None,
                ax=ax, cmap='magma', vmin=self.vmin, vmax=self.vmax
            )

            label = ax.set_ylabel(f'{name.capitalize()}', fontsize=12, fontweight='bold')
            self.stem_labels[name] = label

            if i < n - 1:
                ax.set_xlabel('')

            from matplotlib.patches import Rectangle
            overlay = Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                facecolor='black', alpha=0.7, zorder=10, visible=False
            )
            ax.add_patch(overlay)
            self.dim_overlays[name] = overlay

            self.axes.append(ax)

        self._finalize_figure()

    def _update_visual_state(self):
        """Update dim overlays and label colors to match active_stems."""
        for name in self.stem_names:
            active = self.active_stems[name]
            self.dim_overlays[name].set_visible(not active)
            color = 'white' if active else '#666666'
            self.stem_labels[name].set_color(color)
        self.fig.canvas.draw_idle()

    def _format_title(self):
        filename = Path(self.filepath).name
        n = len(self.stem_names)
        active_str = '  '.join(
            f'{"★" if self.active_stems[name] else " "} {i+1}:{name}'
            for i, name in enumerate(self.stem_names)
        )
        return (
            f'{filename}  —  {active_str}  —  '
            f'SPACE: play  1-{n}: toggle  A: all  Q: quit'
        )

    def _finalize_figure(self):
        self.cursor_lines = []
        for ax in self.axes:
            line = ax.axvline(x=0, color='#FFFFFF', linewidth=1.5, alpha=0.9, zorder=20)
            self.cursor_lines.append(line)

        self.fig.suptitle(self._format_title(), fontsize=12, fontweight='bold', y=0.995)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        n = len(self.stem_names)
        print(f"Ready. Press SPACE to start playback, 1-{n} to toggle stems.")

    def _update_title(self):
        self.fig.suptitle(self._format_title(), fontsize=12, fontweight='bold', y=0.995)

    def _on_key(self, event):
        if event.key == ' ':
            if not self.is_playing:
                self._start_playback()
            else:
                self._stop_playback()
        elif event.key in self.stem_keys:
            self._toggle_stem(self.stem_keys[event.key])
        elif event.key == 'a':
            self._all_stems_on()
        elif event.key == 'q':
            self._stop_playback()
            plt.close(self.fig)

    def _toggle_stem(self, name):
        was_playing = self.is_playing
        if was_playing:
            self._stop_playback()

        self.active_stems[name] = not self.active_stems[name]
        self._compute_mix()
        self._update_visual_state()
        self._update_title()

        active_names = [n for n, on in self.active_stems.items() if on]
        print(f"Active: {', '.join(active_names) if active_names else '(none)'}")

        if was_playing:
            self._start_playback()

    def _all_stems_on(self):
        was_playing = self.is_playing
        if was_playing:
            self._stop_playback()

        for name in self.stem_names:
            self.active_stems[name] = True
        self._compute_mix()
        self._update_visual_state()
        self._update_title()
        print("All stems active")

        if was_playing:
            self._start_playback()

    def _on_click(self, event):
        seek_time = None
        if event.inaxes in self.axes and event.xdata is not None:
            seek_time = event.xdata
        elif event.x is not None and event.y is not None:
            try:
                inv = self.axes[0].transData.inverted()
                data_coords = inv.transform((event.x, event.y))
                seek_time = data_coords[0]
            except Exception:
                pass

        if seek_time is not None and 0 <= seek_time <= self.duration:
            self._seek_to(seek_time)
            print(f"Seeking to {seek_time:.2f}s")

    def _start_playback(self, from_offset=None):
        if from_offset is not None:
            self.playback_offset = from_offset

        self.is_playing = True
        self.playback_start_time = time.time()

        start_sample = int(self.playback_offset * self.sr_playback)
        audio_to_play = self.current_mix[start_sample:]

        def play_audio():
            try:
                sd.play(audio_to_play, self.sr_playback)
                sd.wait()
                self.is_playing = False
            except Exception as e:
                print(f"Playback error: {e}")
                self.is_playing = False

        self.playback_thread = threading.Thread(target=play_audio, daemon=True)
        self.playback_thread.start()
        print(f"Playing from {self.playback_offset:.2f}s...")

    def _stop_playback(self):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            self.playback_offset = min(self.playback_offset + elapsed, self.duration)

        self.is_playing = False
        sd.stop()
        time.sleep(0.05)
        print(f"Stopped at {self.playback_offset:.2f}s.")

    def _seek_to(self, time_position):
        was_playing = self.is_playing
        if self.is_playing:
            self._stop_playback()

        self.playback_offset = time_position
        for line in self.cursor_lines:
            line.set_xdata([time_position, time_position])
        self.fig.canvas.draw_idle()

        if was_playing:
            time.sleep(0.05)
            self._start_playback()

    def _update_cursor(self, frame):
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            current_time = self.playback_offset + elapsed

            if current_time > self.duration:
                self.is_playing = False
                current_time = self.duration

            for line in self.cursor_lines:
                line.set_xdata([current_time, current_time])
        elif not self.is_playing:
            for line in self.cursor_lines:
                line.set_xdata([self.playback_offset, self.playback_offset])

        return self.cursor_lines

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self._update_cursor,
            interval=33, blit=True, cache_frame_data=False
        )
        plt.show()
