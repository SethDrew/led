#!/usr/bin/env python3
"""
Synced Audio Playback + Visualization

Plays audio through speakers while showing a cursor sweeping
across all analysis panels in real-time. Lets you hear the music
and see what the audio processor "sees" simultaneously.

Usage:
    python playback_visualizer.py <audio_file.wav>

Dependencies:
    pip install sounddevice soundfile numpy librosa matplotlib
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
import sounddevice as sd
import soundfile as sf
import yaml

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


class SyncedVisualizer:
    def __init__(self, filepath, focus_panel=None, show_beats=False, annotations_path=None):
        self.filepath = filepath
        self.focus_panel = focus_panel  # Which panel to maximize (None = show all)
        self.show_beats = show_beats  # Whether to show beat detection
        self.annotations_path_override = annotations_path
        self.playback_start_time = None
        self.playback_offset = 0.0  # Current playback position in seconds
        self.is_playing = False

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
                # Also check without _short suffix variant
                ann_path = Path(filepath).parent / (Path(filepath).stem.rsplit('_short', 1)[0] + '.annotations.yaml')
        if ann_path.exists():
            with open(ann_path) as f:
                self.annotations = yaml.safe_load(f) or {}
            print(f"Annotations: {ann_path.name} — {len(self.annotations)} layers ({', '.join(self.annotations.keys())})")

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
            n_mels=64, fmin=20, fmax=None  # None = Nyquist (sr/2, ~22kHz)
        )
        self.mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Band energies - keep focused on musical range for LED mapping
        mel_spec_128 = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=128, fmin=20, fmax=8000  # Keep band analysis at 8kHz for consistency
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

        # Time axis
        self.times = librosa.frames_to_time(
            np.arange(len(self.onset_env)), sr=self.sr, hop_length=self.hop_length
        )

        print(f"Tempo: {self.tempo:.1f} BPM, {len(self.onset_times)} onsets, {len(self.beat_times)} beats")

    def _build_figure(self):
        print("Building visualization...")
        plt.style.use('dark_background')

        # If focusing on one panel, make it bigger
        if self.focus_panel is not None:
            self.fig = plt.figure(figsize=(18, 10))
        elif self.annotations:
            self.fig = plt.figure(figsize=(18, 24))
        else:
            self.fig = plt.figure(figsize=(18, 20))

        # Determine which panels to show
        if self.focus_panel == 'onset':
            # Show only onset/beat panel
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            self._build_onset_panel(self.fig.add_subplot(gs[0]))
            self.axes = [self.axes[0]]  # Keep only onset axis
        elif self.focus_panel == 'waveform':
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            self._build_waveform_panel(self.fig.add_subplot(gs[0]))
            self.axes = [self.axes[0]]
        elif self.focus_panel == 'spectrogram':
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            self._build_spectrogram_panel(self.fig.add_subplot(gs[0]))
            self.axes = [self.axes[0]]
        elif self.focus_panel == 'bands':
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            self._build_band_energy_panel(self.fig.add_subplot(gs[0]))
            self.axes = [self.axes[0]]
        elif self.focus_panel == 'annotations' and self.annotations:
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            self._build_annotation_panel(self.fig.add_subplot(gs[0]))
            self.axes = [self.axes[0]]
        elif self.focus_panel == 'centroid':
            gs = gridspec.GridSpec(1, 1, hspace=0.3)
            self._build_centroid_panel(self.fig.add_subplot(gs[0]))
            self.axes = [self.axes[0]]
        else:
            # Show all panels (default), add annotation panel if annotations exist
            if self.annotations:
                n_layers = len(self.annotations)
                ann_height = max(0.8, 0.4 * n_layers)  # scale with layer count
                gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1.5, 1.5, 1.5, ann_height, 1.5], hspace=0.3)
                self._build_waveform_panel(self.fig.add_subplot(gs[0]))
                self._build_spectrogram_panel(self.fig.add_subplot(gs[1]))
                self._build_band_energy_panel(self.fig.add_subplot(gs[2]))
                self._build_onset_panel(self.fig.add_subplot(gs[3]))
                self._build_annotation_panel(self.fig.add_subplot(gs[4]))
                self._build_centroid_panel(self.fig.add_subplot(gs[5]))
            else:
                gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1.5, 1.5, 1.5, 1.5], hspace=0.3)
                self._build_waveform_panel(self.fig.add_subplot(gs[0]))
                self._build_spectrogram_panel(self.fig.add_subplot(gs[1]))
                self._build_band_energy_panel(self.fig.add_subplot(gs[2]))
                self._build_onset_panel(self.fig.add_subplot(gs[3]))
                self._build_centroid_panel(self.fig.add_subplot(gs[4]))

        # Finalize figure (cursor lines, events, title)
        self._finalize_figure()

    def _build_waveform_panel(self, ax):
        """Panel 1: Waveform"""
        waveform_times = np.linspace(0, self.duration, len(self.y))
        ax.plot(waveform_times, self.y, color='#FFFFFF', linewidth=0.5, alpha=0.8)
        ax.set_xlim([0, self.duration])
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_spectrogram_panel(self, ax):
        """Panel 2: Mel Spectrogram"""
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
        """Panel 3: Band Energy"""
        for band_name, energy in self.band_energies.items():
            ax.plot(self.times, energy, label=band_name,
                     color=BAND_COLORS[band_name], linewidth=2, alpha=0.9)
        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Normalized Energy')
        ax.set_title('Band Energy — drives LED color mapping', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

        # Magnitude ratios
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

    def _build_onset_panel(self, ax):
        """Panel 4: Onset & Beat Detection"""
        ax.plot(self.times, self.onset_env / np.max(self.onset_env),
                 color='#00E5FF', linewidth=1.5, label='Onset Strength')

        # Optional: show onset markers
        for t in self.onset_times:
            ax.axvline(x=t, color='#FFC400', alpha=0.3, linewidth=0.8)

        # Optional: show beat markers (only if flag enabled)
        if self.show_beats:
            for t in self.beat_times:
                ax.axvline(x=t, color='#FF1744', alpha=0.6, linewidth=2)

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Strength')

        # Update title based on whether beats are shown
        if self.show_beats:
            title = f'Onset & Beat Detection — {self.tempo:.0f} BPM detected'
        else:
            title = 'Onset Strength'
        ax.set_title(title, fontsize=11)

        ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_annotation_panel(self, ax):
        """Panel: User tap annotations from annotate_segment.py"""
        layer_colors = ['#FF4081', '#40C4FF', '#69F0AE', '#FFD740', '#E040FB', '#FF6E40']
        layer_names = list(self.annotations.keys())

        for i, (layer_name, taps) in enumerate(self.annotations.items()):
            color = layer_colors[i % len(layer_colors)]
            y_pos = len(layer_names) - i  # stack layers bottom to top
            for t in taps:
                if t <= self.duration:
                    ax.plot(t, y_pos, '|', color=color, markersize=12, markeredgewidth=2, alpha=0.8)
            # Label on the left
            ax.text(-self.duration * 0.01, y_pos, layer_name, ha='right', va='center',
                    fontsize=8, color=color, fontweight='bold')

        ax.set_xlim([0, self.duration])
        ax.set_ylim([0.3, len(layer_names) + 0.7])
        ax.set_yticks([])
        ax.set_title(f'User Annotations — {sum(len(v) for v in self.annotations.values())} taps across {len(layer_names)} layers', fontsize=11)
        ax.grid(True, axis='x', alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _build_centroid_panel(self, ax):
        """Panel 5: Centroid & RMS"""
        ax_twin = ax.twinx()
        centroid_norm = (self.spectral_centroid - np.min(self.spectral_centroid))
        centroid_range = np.max(self.spectral_centroid) - np.min(self.spectral_centroid)
        if centroid_range > 0:
            centroid_norm = centroid_norm / centroid_range
        ax.plot(self.times, centroid_norm, color='#B388FF', linewidth=2,
                 label='Spectral Centroid', alpha=0.9)
        rms_norm = self.rms_energy / np.max(self.rms_energy)
        ax_twin.plot(self.times, rms_norm, color='#FFD740', linewidth=2,
                      label='RMS Energy', alpha=0.9)
        ax.set_xlim([0, self.duration])
        ax.set_ylim([0, 1.1])
        ax_twin.set_ylim([0, 1.1])
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Centroid', color='#B388FF')
        ax_twin.set_ylabel('RMS', color='#FFD740')
        ax.set_title('Spectral Centroid (brightness) & RMS (loudness)', fontsize=11)
        ax.grid(True, alpha=0.2)
        if not hasattr(self, 'axes'):
            self.axes = []
        self.axes.append(ax)

    def _finalize_figure(self):

        # Create cursor lines (one per panel)
        self.cursor_lines = []
        for ax in self.axes:
            line = ax.axvline(x=0, color='#FFFFFF', linewidth=1.5, alpha=0.9, linestyle='-')
            self.cursor_lines.append(line)

        filename = Path(self.filepath).name
        self.fig.suptitle(
            f'{filename}  —  SPACE: play/pause  |  CLICK: seek  |  Q: quit',
            fontsize=14, fontweight='bold', y=0.995
        )

        # Connect keyboard and mouse events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        print("Ready. Press SPACE to start playback, or CLICK to seek.")

    def _on_key(self, event):
        if event.key == ' ':
            if not self.is_playing:
                self._start_playback()
            else:
                self._stop_playback()
        elif event.key == 'q':
            self._stop_playback()
            plt.close(self.fig)

    def _on_click(self, event):
        """Seek to clicked position"""
        seek_time = None

        # Try to get time from clicked axis
        if event.inaxes in self.axes and event.xdata is not None:
            seek_time = event.xdata
        # If click was outside axes (margins/gaps), map figure coords to time
        elif event.x is not None and event.y is not None:
            # Use first axis (waveform) to map x position to time
            try:
                # Convert figure pixel coords to data coords using first axis
                inv = self.axes[0].transData.inverted()
                data_coords = inv.transform((event.x, event.y))
                seek_time = data_coords[0]
            except:
                pass

        # Seek if we got a valid time
        if seek_time is not None and 0 <= seek_time <= self.duration:
            self._seek_to(seek_time)
            print(f"✓ Seeking to {seek_time:.2f}s")
        elif seek_time is not None:
            print(f"✗ Click out of range: {seek_time:.2f}s (0-{self.duration:.1f}s)")
        else:
            print(f"✗ Click didn't register (try clicking on a graph)")

    def _start_playback(self, from_offset=None):
        """Start playback from current offset or specified position"""
        if from_offset is not None:
            self.playback_offset = from_offset

        self.is_playing = True
        self.playback_start_time = time.time()

        # Calculate which samples to play
        start_sample = int(self.playback_offset * self.sr_playback)
        audio_to_play = self.y_playback[start_sample:]

        # Play audio in background thread
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
        """Stop playback and remember current position"""
        if self.is_playing and self.playback_start_time is not None:
            elapsed = time.time() - self.playback_start_time
            self.playback_offset = min(self.playback_offset + elapsed, self.duration)

        self.is_playing = False
        sd.stop()

        # Wait a bit for audio device to fully stop (fixes Core Audio errors)
        time.sleep(0.05)

        print(f"Stopped at {self.playback_offset:.2f}s.")

    def _seek_to(self, time_position):
        """Seek to specific time and start playing"""
        was_playing = self.is_playing

        # Stop playback if playing
        if self.is_playing:
            self._stop_playback()

        self.playback_offset = time_position

        # Update cursor immediately
        for line in self.cursor_lines:
            line.set_xdata([time_position, time_position])
        self.fig.canvas.draw_idle()

        # Restart if it was playing (with slight delay for audio cleanup)
        if was_playing:
            time.sleep(0.05)  # Let audio device fully stop
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
            # Show cursor at stopped position
            for line in self.cursor_lines:
                line.set_xdata([self.playback_offset, self.playback_offset])

        return self.cursor_lines

    def run(self):
        # Animate cursor at ~30fps
        self.anim = FuncAnimation(
            self.fig, self._update_cursor,
            interval=33, blit=True, cache_frame_data=False
        )
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Audio visualization with synced playback')
    parser.add_argument('audio_file', help='Path to audio file (.wav)')
    parser.add_argument('--panel', choices=['waveform', 'spectrogram', 'bands', 'onset', 'centroid', 'annotations'],
                        help='Maximize specific panel (default: show all)')
    parser.add_argument('--show-beats', action='store_true',
                        help='Show beat detection markers (red lines)')
    parser.add_argument('--annotations', help='Path to .annotations.yaml file (auto-detected if omitted)')
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"File not found: {args.audio_file}")
        sys.exit(1)

    print("\nControls:")
    print("  SPACE  - Play / Pause")
    print("  CLICK  - Seek to position")
    print("  Q      - Quit")
    if args.panel:
        print(f"\nShowing: {args.panel} panel (maximized)")
    if args.show_beats:
        print("Beat detection: ON (red markers)")
    print()

    viz = SyncedVisualizer(args.audio_file, focus_panel=args.panel, show_beats=args.show_beats,
                           annotations_path=args.annotations)
    viz.run()


if __name__ == '__main__':
    main()
