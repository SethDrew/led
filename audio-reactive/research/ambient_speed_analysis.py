"""
Analyze ambient song for "speed" features — what rate-of-change signals
are present in ten_ambient.wav?
"""
import sys
sys.path.insert(0, '/Users/sethdrew/Documents/projects/led')

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# --- Load ---
wav_path = '/Users/sethdrew/Documents/projects/led/audio-reactive/research/audio-segments/ten_ambient.wav'
y, sr = librosa.load(wav_path, sr=None, mono=True)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Loaded: {duration:.1f}s, sr={sr}")

# --- Parameters ---
n_fft = 2048
hop_length = 512
dt = hop_length / sr
times = librosa.frames_to_time(np.arange(int(np.ceil(len(y) / hop_length))), sr=sr, hop_length=hop_length)

# --- Core features ---
rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]

# Trim times to match feature lengths
n_frames = min(len(rms), len(centroid), len(onset_env), len(times))
rms = rms[:n_frames]
centroid = centroid[:n_frames]
onset_env = onset_env[:n_frames]
spectral_bw = spectral_bw[:n_frames]
spectral_flatness = spectral_flatness[:n_frames]
spectral_rolloff = spectral_rolloff[:n_frames]
times = times[:n_frames]

# --- Derivatives (rate of change = "speed") ---
rms_deriv = np.diff(rms, prepend=rms[0]) / dt
centroid_deriv = np.diff(centroid, prepend=centroid[0]) / dt
bw_deriv = np.diff(spectral_bw, prepend=spectral_bw[0]) / dt
rolloff_deriv = np.diff(spectral_rolloff, prepend=spectral_rolloff[0]) / dt

# --- AbsIntegral (offline approx) ---
window_sec = 0.15
window_frames = max(1, int(window_sec / dt))
abs_rms_deriv = np.abs(rms_deriv)
abs_integral = np.convolve(abs_rms_deriv, np.ones(window_frames) / window_frames, mode='same')

# --- Smoothed versions (for ambient, longer smoothing) ---
smooth_sigma = int(1.0 / dt)  # 1 second gaussian
rms_smooth = gaussian_filter1d(rms, sigma=smooth_sigma)
centroid_smooth = gaussian_filter1d(centroid, sigma=smooth_sigma)
bw_smooth = gaussian_filter1d(spectral_bw, sigma=smooth_sigma)

# Smooth derivatives (what we actually care about for ambient)
rms_deriv_smooth = np.diff(rms_smooth, prepend=rms_smooth[0]) / dt
centroid_deriv_smooth = np.diff(centroid_smooth, prepend=centroid_smooth[0]) / dt
bw_deriv_smooth = np.diff(bw_smooth, prepend=bw_smooth[0]) / dt

# --- Rolling integral (5s and 15s windows) ---
window_5s = max(1, int(5.0 / dt))
window_15s = max(1, int(15.0 / dt))
rolling_rms_5s = uniform_filter1d(rms, size=window_5s, mode='constant')
rolling_rms_15s = uniform_filter1d(rms, size=window_15s, mode='constant')

# Slope of rolling integral = build/decay detector
rolling_slope_5s = np.diff(rolling_rms_5s, prepend=rolling_rms_5s[0]) / dt
rolling_slope_15s = np.diff(rolling_rms_15s, prepend=rolling_rms_15s[0]) / dt

# --- Band energies ---
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
    n_mels=128, fmin=20, fmax=8000
)[:, :n_frames]
mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=8000)

bands = {
    'sub_bass': (20, 80),
    'bass': (80, 250),
    'mids': (250, 2000),
    'high_mids': (2000, 6000),
    'treble': (6000, 8000),
}
band_energy = {}
for name, (lo, hi) in bands.items():
    mask = (mel_freqs >= lo) & (mel_freqs <= hi)
    band_energy[name] = np.sum(mel_spec[mask, :], axis=0)

# --- Spectral flux (frame-to-frame change in spectrum) ---
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))[:, :n_frames]
spectral_flux = np.sqrt(np.mean(np.diff(S, axis=1, prepend=S[:, :1]) ** 2, axis=0))
spectral_flux_smooth = gaussian_filter1d(spectral_flux, sigma=smooth_sigma)

# --- Summary stats ---
print(f"\n=== SIGNAL SUMMARY ===")
print(f"Duration: {duration:.1f}s")
print(f"Frames: {n_frames}, dt={dt*1000:.1f}ms")
print(f"\nRMS: mean={np.mean(rms):.4f}, std={np.std(rms):.4f}, range=[{np.min(rms):.4f}, {np.max(rms):.4f}]")
print(f"RMS dynamic range: {np.max(rms)/max(np.mean(rms), 1e-10):.2f}x mean")
print(f"\nCentroid: mean={np.mean(centroid):.0f}Hz, std={np.std(centroid):.0f}Hz, range=[{np.min(centroid):.0f}, {np.max(centroid):.0f}]Hz")
print(f"Centroid variability: {np.std(centroid)/max(np.mean(centroid), 1):.2%} CV")
print(f"\nSpectral bandwidth: mean={np.mean(spectral_bw):.0f}Hz, std={np.std(spectral_bw):.0f}Hz")
print(f"Spectral flatness: mean={np.mean(spectral_flatness):.4f} (0=tonal, 1=noise)")
print(f"\nOnset envelope: mean={np.mean(onset_env):.4f}, max={np.max(onset_env):.4f}")
print(f"AbsIntegral: mean={np.mean(abs_integral):.6f}, max={np.max(abs_integral):.6f}")
print(f"Spectral flux: mean={np.mean(spectral_flux):.4f}, max={np.max(spectral_flux):.4f}")

# Band energy distribution
total_energy = sum(np.mean(e) for e in band_energy.values())
print(f"\nBand energy distribution:")
for name, energy in band_energy.items():
    pct = np.mean(energy) / max(total_energy, 1e-10) * 100
    print(f"  {name:12s}: {pct:5.1f}%  (mean={np.mean(energy):.4f})")

# Rate-of-change summary
print(f"\n=== RATE-OF-CHANGE ('SPEED') FEATURES ===")
print(f"RMS velocity (smoothed): mean={np.mean(np.abs(rms_deriv_smooth)):.6f}/s")
print(f"Centroid velocity (smoothed): mean={np.mean(np.abs(centroid_deriv_smooth)):.1f} Hz/s")
print(f"Bandwidth velocity (smoothed): mean={np.mean(np.abs(bw_deriv_smooth)):.1f} Hz/s")
print(f"Rolling integral slope (5s): range=[{np.min(rolling_slope_5s):.6f}, {np.max(rolling_slope_5s):.6f}]")
print(f"Rolling integral slope (15s): range=[{np.min(rolling_slope_15s):.6f}, {np.max(rolling_slope_15s):.6f}]")

# --- Detect pulse cycles in RMS ---
# Zero crossings of smoothed RMS derivative = peaks and troughs
rms_smooth_long = gaussian_filter1d(rms, sigma=int(2.0 / dt))
rms_deriv_long = np.diff(rms_smooth_long, prepend=rms_smooth_long[0]) / dt
zero_crossings = np.where(np.diff(np.sign(rms_deriv_long)))[0]
if len(zero_crossings) > 1:
    cycle_lengths = np.diff(zero_crossings) * dt
    full_cycle = cycle_lengths[::2] if len(cycle_lengths) > 2 else cycle_lengths
    print(f"\n=== PULSE CYCLE DETECTION (2s smoothing) ===")
    print(f"Zero crossings found: {len(zero_crossings)}")
    print(f"Half-cycle durations: mean={np.mean(cycle_lengths):.2f}s, range=[{np.min(cycle_lengths):.2f}, {np.max(cycle_lengths):.2f}]s")
    if len(full_cycle) > 1:
        print(f"Full-cycle durations: mean={np.mean(full_cycle):.2f}s, range=[{np.min(full_cycle):.2f}, {np.max(full_cycle):.2f}]s")
        print(f"Implied pulse rate: ~{60/np.mean(full_cycle):.1f} BPM (or {np.mean(full_cycle):.1f}s period)")

# --- PLOT ---
fig, axes = plt.subplots(8, 1, figsize=(16, 24), sharex=True)
fig.suptitle('Ambient Speed Analysis — ten_ambient.wav', fontsize=14, fontweight='bold')

# 1. Waveform + RMS
ax = axes[0]
ax.plot(np.arange(len(y)) / sr, y, alpha=0.3, color='gray', linewidth=0.5)
ax.plot(times, rms, color='blue', linewidth=1, label='RMS')
ax.plot(times, rms_smooth, color='red', linewidth=2, label='RMS (1s smooth)')
ax.set_ylabel('Amplitude')
ax.set_title('Waveform + RMS')
ax.legend(loc='upper right')

# 2. RMS derivative (the "speed" of volume change)
ax = axes[1]
ax.plot(times, rms_deriv, alpha=0.2, color='gray', linewidth=0.5)
ax.plot(times, rms_deriv_smooth, color='red', linewidth=1.5, label='d(RMS)/dt (1s smooth)')
ax.axhline(0, color='black', linewidth=0.5)
ax.fill_between(times, rms_deriv_smooth, 0, alpha=0.3,
                where=rms_deriv_smooth > 0, color='green', label='Swelling')
ax.fill_between(times, rms_deriv_smooth, 0, alpha=0.3,
                where=rms_deriv_smooth < 0, color='red', label='Fading')
ax.set_ylabel('d(RMS)/dt')
ax.set_title('Volume Speed — how fast is loudness changing?')
ax.legend(loc='upper right')

# 3. AbsIntegral + onset envelope
ax = axes[2]
ax_norm = abs_integral / max(np.max(abs_integral), 1e-10)
onset_norm = onset_env / max(np.max(onset_env), 1e-10)
ax.plot(times, ax_norm, color='purple', linewidth=1.5, label='AbsIntegral (|dRMS/dt| avg)')
ax.plot(times, onset_norm, color='orange', alpha=0.6, linewidth=1, label='Onset envelope')
ax.set_ylabel('Normalized')
ax.set_title('Change detectors — AbsIntegral vs Onset Envelope')
ax.legend(loc='upper right')

# 4. Spectral centroid + its velocity
ax = axes[3]
ax2 = ax.twinx()
ax.plot(times, centroid, alpha=0.3, color='gray', linewidth=0.5)
ax.plot(times, centroid_smooth, color='blue', linewidth=2, label='Centroid (1s smooth)')
ax.set_ylabel('Centroid (Hz)', color='blue')
ax2.plot(times, centroid_deriv_smooth, color='red', linewidth=1, alpha=0.7, label='d(Centroid)/dt')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('Hz/sec', color='red')
ax.set_title('Tonal drift speed — centroid + its velocity')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# 5. Spectral bandwidth (is sound expanding or contracting?)
ax = axes[4]
ax.plot(times, spectral_bw, alpha=0.3, color='gray', linewidth=0.5)
ax.plot(times, bw_smooth, color='teal', linewidth=2, label='Bandwidth (1s smooth)')
ax.set_ylabel('Bandwidth (Hz)')
ax.set_title('Sound width — is the frequency spread expanding or contracting?')
ax.legend(loc='upper right')

# 6. Spectral flux (texture morphing speed)
ax = axes[5]
ax.plot(times, spectral_flux, alpha=0.3, color='gray', linewidth=0.5)
ax.plot(times, spectral_flux_smooth, color='magenta', linewidth=2, label='Spectral flux (1s smooth)')
ax.set_ylabel('Flux')
ax.set_title('Texture speed — how fast is the spectrum rearranging?')
ax.legend(loc='upper right')

# 7. Band energies
ax = axes[6]
colors_band = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
for (name, energy), color in zip(band_energy.items(), colors_band):
    energy_smooth = gaussian_filter1d(energy, sigma=smooth_sigma)
    norm = energy_smooth / max(np.max(energy_smooth), 1e-10)
    ax.plot(times, norm, color=color, linewidth=1.5, label=name, alpha=0.8)
ax.set_ylabel('Normalized energy')
ax.set_title('Band energy distribution over time')
ax.legend(loc='upper right', ncol=3)

# 8. Rolling integral + slope (build/decay detector)
ax = axes[7]
ax2 = ax.twinx()
ax.plot(times, rolling_rms_5s, color='blue', linewidth=2, label='Rolling RMS (5s)')
ax.plot(times, rolling_rms_15s, color='navy', linewidth=2, alpha=0.7, label='Rolling RMS (15s)')
ax.set_ylabel('Rolling RMS', color='blue')
ax2.plot(times, rolling_slope_5s, color='green', linewidth=1, alpha=0.7, label='Slope (5s)')
ax2.plot(times, rolling_slope_15s, color='darkgreen', linewidth=1.5, alpha=0.8, label='Slope (15s)')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('Slope', color='green')
ax.set_xlabel('Time (s)')
ax.set_title('Section-level energy arc — rolling integral + slope')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/Users/sethdrew/Documents/projects/led/audio-reactive/research/ambient_speed_analysis.png', dpi=150)
print(f"\nPlot saved to ambient_speed_analysis.png")
plt.close()
