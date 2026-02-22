#!/usr/bin/env python3
"""Visual comparison of weighting curves for the audit report."""

import numpy as np
import matplotlib.pyplot as plt

def a_weight_db(f):
    """A-weighting (IEC 61672) - 40 phons"""
    f = np.asarray(f, dtype=np.float64)
    f2 = f ** 2
    return (
        20 * np.log10(
            (12194**2 * f2**2) /
            ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
            + 1e-20
        )
        + 2.0
    )

def c_weight_db(f):
    """C-weighting (IEC 61672) - 80-100 dB SPL"""
    f = np.asarray(f, dtype=np.float64)
    f2 = f ** 2
    numerator = 12194**2 * f2
    denominator = (f2 + 20.6**2) * (f2 + 12194**2)
    Rc = numerator / (denominator + 1e-20)
    return 20 * np.log10(Rc + 1e-20) + 0.06

# Frequency range
freqs = np.logspace(np.log10(20), np.log10(20000), 1000)

# Compute curves
a_curve = a_weight_db(freqs)
c_curve = c_weight_db(freqs)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Curve comparison
ax1.semilogx(freqs, a_curve, 'r-', linewidth=2, label='A-weighting (40 phons)')
ax1.semilogx(freqs, c_curve, 'b-', linewidth=2, label='C-weighting (80-100 dB SPL)')
ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax1.grid(True, which='both', alpha=0.3)
ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_ylabel('Weighting (dB)', fontsize=12)
ax1.set_title('Weighting Curves: A vs C', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_xlim([20, 20000])
ax1.set_ylim([-60, 10])

# Add band regions
band_colors = {
    'Sub-bass': ('#FF6B6B', 20, 80),
    'Bass': ('#4ECDC4', 80, 250),
    'Mids': ('#45B7D1', 250, 2000),
    'High-mids': ('#FFA07A', 2000, 6000),
    'Treble': ('#98D8C8', 6000, 8000),
}

for i, (name, (color, fmin, fmax)) in enumerate(band_colors.items()):
    ax1.axvspan(fmin, fmax, alpha=0.1, color=color)
    # Add label at top
    mid_freq = np.sqrt(fmin * fmax)
    ax1.text(mid_freq, 8, name, ha='center', fontsize=9, fontweight='bold')

# Power factor comparison
a_power = 10 ** (a_curve / 10.0)
c_power = 10 ** (c_curve / 10.0)

ax2.semilogx(freqs, a_power, 'r-', linewidth=2, label='A-weighting power factor')
ax2.semilogx(freqs, c_power, 'b-', linewidth=2, label='C-weighting power factor')
ax2.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.3, label='No attenuation')
ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Power Multiplication Factor', fontsize=12)
ax2.set_title('Power Attenuation Factors', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.set_xlim([20, 20000])
ax2.set_ylim([0.0001, 2])
ax2.set_yscale('log')

# Add band regions
for name, (color, fmin, fmax) in band_colors.items():
    ax2.axvspan(fmin, fmax, alpha=0.1, color=color)

# Add annotations for key frequencies
key_freqs = [50, 100, 1000, 4000]
for freq in key_freqs:
    a_val = 10 ** (a_weight_db(freq) / 10.0)
    c_val = 10 ** (c_weight_db(freq) / 10.0)
    ax2.plot(freq, a_val, 'ro', markersize=6)
    ax2.plot(freq, c_val, 'bo', markersize=6)
    ax2.text(freq * 1.2, a_val, f'{a_val:.4f}', fontsize=8, color='red')
    ax2.text(freq * 1.2, c_val, f'{c_val:.2f}', fontsize=8, color='blue')

plt.tight_layout()
plt.savefig('weighting_comparison.png', dpi=150, bbox_inches='tight')
print("Saved weighting_comparison.png")

# Print summary table
print("\n" + "="*80)
print("WEIGHTING COMPARISON SUMMARY")
print("="*80)

FREQUENCY_BANDS = {
    'Sub-bass': (20, 80),
    'Bass': (80, 250),
    'Mids': (250, 2000),
    'High-mids': (2000, 6000),
    'Treble': (6000, 8000),
}

print(f"\n{'Band':<15} {'A-weight (dB)':<15} {'A power':<15} {'C-weight (dB)':<15} {'C power':<15}")
print("-" * 80)

for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
    freqs_band = np.linspace(fmin, fmax, 100)
    a_avg = np.mean(a_weight_db(freqs_band))
    c_avg = np.mean(c_weight_db(freqs_band))
    a_pow = 10 ** (a_avg / 10.0)
    c_pow = 10 ** (c_avg / 10.0)

    print(f"{band_name:<15} {a_avg:<15.1f} {a_pow:<15.6f} {c_avg:<15.1f} {c_pow:<15.4f}")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("\nA-weighting reduces sub-bass power by factor of ~1600 (0.0006x)")
print("C-weighting reduces sub-bass power by factor of ~1.6 (0.64x)")
print("\nFor music at normal listening levels, C-weighting or no weighting")
print("is much more appropriate than A-weighting.")
