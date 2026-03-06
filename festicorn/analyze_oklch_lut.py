#!/usr/bin/env python3
"""
Analyze OKLCH rainbow LUT for dark red and purple representation.
"""

import colorsys
import numpy as np
from collections import defaultdict

# Parse the LUT from effects.cpp
oklch_lut = [
    [252,  53, 102], [252,  53,  99], [252,  54,  95], [252,  54,  92],
    [252,  55,  89], [252,  55,  86], [252,  56,  83], [252,  56,  80],
    [252,  57,  77], [252,  57,  75], [252,  57,  72], [252,  58,  69],
    [252,  58,  67], [252,  58,  64], [252,  59,  62], [252,  59,  59],
    [252,  59,  57], [252,  60,  55], [252,  60,  52], [252,  60,  50],
    [252,  61,  48], [252,  61,  45], [252,  61,  43], [252,  62,  41],
    [252,  62,  39], [252,  62,  37], [252,  63,  35], [252,  63,  32],
    [252,  63,  30], [252,  64,  28], [252,  64,  26], [252,  64,  24],
    [252,  65,  22], [252,  65,  20], [252,  65,  17], [252,  65,  15],
    [252,  66,  13], [252,  66,  11], [252,  66,   9], [252,  67,   7],
    [252,  67,   4], [252,  67,   2], [249,  69,   1], [244,  70,   1],
    [240,  72,   1], [235,  74,   1], [231,  75,   1], [227,  77,   1],
    [223,  78,   1], [219,  80,   1], [215,  81,   1], [211,  83,   1],
    [207,  84,   1], [204,  86,   1], [200,  87,   1], [196,  88,   1],
    [193,  90,   1], [189,  91,   1], [186,  92,   1], [183,  94,   1],
    [179,  95,   1], [176,  96,   1], [173,  98,   1], [169,  99,   1],
    [166, 100,   1], [163, 101,   1], [159, 103,   1], [156, 104,   1],
    [153, 105,   1], [149, 107,   1], [146, 108,   1], [143, 109,   1],
    [139, 110,   1], [136, 112,   1], [133, 113,   1], [129, 114,   1],
    [126, 116,   1], [122, 117,   1], [119, 118,   1], [115, 120,   1],
    [111, 121,   1], [108, 123,   1], [104, 124,   1], [100, 126,   1],
    [ 96, 127,   1], [ 92, 129,   1], [ 88, 131,   1], [ 83, 132,   1],
    [ 79, 134,   1], [ 74, 136,   1], [ 70, 138,   1], [ 65, 139,   1],
    [ 60, 141,   1], [ 55, 143,   1], [ 49, 146,   1], [ 44, 148,   1],
    [ 38, 150,   1], [ 32, 153,   1], [ 25, 155,   1], [ 19, 158,   1],
    [ 12, 161,   1], [  4, 164,   1], [  2, 164,   6], [  2, 163,  12],
    [  2, 162,  17], [  2, 161,  23], [  2, 161,  28], [  2, 160,  33],
    [  2, 159,  38], [  2, 159,  43], [  2, 158,  47], [  2, 157,  52],
    [  2, 157,  56], [  2, 156,  60], [  2, 156,  64], [  2, 155,  68],
    [  2, 155,  72], [  2, 154,  76], [  2, 154,  79], [  2, 153,  83],
    [  2, 153,  86], [  2, 152,  90], [  2, 152,  93], [  2, 151,  96],
    [  2, 151, 100], [  2, 150, 103], [  2, 150, 106], [  2, 149, 109],
    [  2, 149, 112], [  2, 148, 115], [  2, 148, 118], [  2, 148, 121],
    [  2, 147, 125], [  2, 147, 128], [  2, 146, 131], [  2, 146, 134],
    [  2, 145, 137], [  2, 145, 140], [  2, 145, 143], [  2, 144, 146],
    [  2, 144, 149], [  2, 143, 152], [  2, 143, 155], [  2, 142, 158],
    [  2, 142, 161], [  2, 141, 165], [  2, 141, 168], [  2, 141, 171],
    [  2, 140, 175], [  2, 140, 178], [  2, 139, 182], [  2, 139, 185],
    [  2, 138, 189], [  2, 137, 193], [  2, 137, 196], [  2, 136, 200],
    [  2, 136, 204], [  2, 135, 209], [  2, 135, 213], [  2, 134, 217],
    [  2, 133, 222], [  2, 133, 227], [  2, 132, 232], [  2, 131, 237],
    [  2, 130, 243], [  2, 130, 248], [  3, 129, 252], [  6, 127, 252],
    [  9, 126, 252], [ 12, 125, 252], [ 15, 124, 252], [ 18, 122, 252],
    [ 21, 121, 252], [ 24, 120, 252], [ 26, 119, 252], [ 29, 118, 252],
    [ 31, 117, 252], [ 34, 116, 252], [ 37, 115, 252], [ 39, 114, 252],
    [ 42, 113, 252], [ 44, 112, 252], [ 46, 111, 252], [ 49, 110, 252],
    [ 51, 109, 252], [ 54, 108, 252], [ 56, 107, 252], [ 58, 106, 252],
    [ 61, 105, 252], [ 63, 104, 252], [ 65, 104, 252], [ 68, 103, 252],
    [ 70, 102, 252], [ 72, 101, 252], [ 75, 100, 252], [ 77,  99, 252],
    [ 80,  98, 252], [ 82,  97, 252], [ 85,  96, 252], [ 87,  95, 252],
    [ 90,  94, 252], [ 92,  93, 252], [ 95,  92, 252], [ 98,  91, 252],
    [100,  90, 252], [103,  89, 252], [106,  87, 252], [109,  86, 252],
    [112,  85, 252], [115,  84, 252], [118,  83, 252], [122,  81, 252],
    [125,  80, 252], [128,  79, 252], [132,  77, 252], [136,  76, 252],
    [140,  74, 252], [144,  73, 252], [148,  71, 252], [152,  69, 252],
    [157,  68, 252], [162,  66, 252], [167,  64, 252], [172,  62, 252],
    [178,  60, 252], [184,  57, 252], [190,  55, 252], [197,  52, 252],
    [204,  49, 252], [212,  46, 252], [221,  43, 252], [230,  40, 252],
    [240,  36, 252], [251,  32, 252], [252,  33, 241], [252,  34, 229],
    [252,  36, 219], [252,  37, 209], [252,  39, 200], [252,  40, 191],
    [252,  41, 183], [252,  42, 176], [252,  43, 169], [252,  44, 163],
    [252,  45, 156], [252,  46, 151], [252,  47, 145], [252,  47, 140],
    [252,  48, 135], [252,  49, 130], [252,  49, 126], [252,  50, 121],
    [252,  51, 117], [252,  51, 113], [252,  52, 109], [252,  52, 106],
]

print("=" * 80)
print("OKLCH Rainbow LUT Analysis: Dark Red and Purple Representation")
print("=" * 80)
print()

# Convert all RGB to HSV
hsv_data = []
for idx, (r, g, b) in enumerate(oklch_lut):
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    hsv_data.append({
        'idx': idx,
        'rgb': (r, g, b),
        'h': h * 360,  # Convert to degrees
        's': s,
        'v': v
    })

# Create hue histogram (30° buckets)
hue_buckets = {
    'red (0-30°)': [],
    'orange (30-60°)': [],
    'yellow (60-90°)': [],
    'green (90-150°)': [],
    'cyan (150-210°)': [],
    'blue (210-270°)': [],
    'purple (270-310°)': [],
    'magenta (310-340°)': [],
    'dark_red (340-360°)': []
}

for data in hsv_data:
    h = data['h']
    if 0 <= h < 30:
        hue_buckets['red (0-30°)'].append(data)
    elif 30 <= h < 60:
        hue_buckets['orange (30-60°)'].append(data)
    elif 60 <= h < 90:
        hue_buckets['yellow (60-90°)'].append(data)
    elif 90 <= h < 150:
        hue_buckets['green (90-150°)'].append(data)
    elif 150 <= h < 210:
        hue_buckets['cyan (150-210°)'].append(data)
    elif 210 <= h < 270:
        hue_buckets['blue (210-270°)'].append(data)
    elif 270 <= h < 310:
        hue_buckets['purple (270-310°)'].append(data)
    elif 310 <= h < 340:
        hue_buckets['magenta (310-340°)'].append(data)
    else:  # 340-360
        hue_buckets['dark_red (340-360°)'].append(data)

print("1. HUE DISTRIBUTION HISTOGRAM")
print("-" * 80)
for bucket_name, entries in hue_buckets.items():
    count = len(entries)
    bar = '█' * (count // 2)  # Visual bar
    print(f"{bucket_name:25s}: {count:3d} entries {bar}")
print()

# Detailed analysis of red region (340-360 and 0-20)
print("2. DARK RED REGION ANALYSIS (HSV hue 340-360° and 0-20°)")
print("-" * 80)
dark_red_entries = hue_buckets['dark_red (340-360°)'] + [
    d for d in hue_buckets['red (0-30°)'] if d['h'] <= 20
]

if dark_red_entries:
    print(f"Found {len(dark_red_entries)} LUT entries in dark red region:")
    print()
    for data in dark_red_entries:
        r, g, b = data['rgb']
        print(f"  Index {data['idx']:3d}: RGB({r:3d}, {g:3d}, {b:3d}) | "
              f"HSV(H={data['h']:6.2f}°, S={data['s']:.3f}, V={data['v']:.3f})")

    # Average saturation and value
    avg_s = np.mean([d['s'] for d in dark_red_entries])
    avg_v = np.mean([d['v'] for d in dark_red_entries])
    print()
    print(f"  Average Saturation: {avg_s:.3f}")
    print(f"  Average Value (brightness): {avg_v:.3f}")
    print()

    # Compare to target "dark red" (180, 0, 0)
    target_r, target_g, target_b = 180, 0, 0
    target_h, target_s, target_v = colorsys.rgb_to_hsv(target_r/255.0, target_g/255.0, target_b/255.0)
    print(f"  Target 'dark red' RGB({target_r}, {target_g}, {target_b}):")
    print(f"    HSV(H={target_h*360:.2f}°, S={target_s:.3f}, V={target_v:.3f})")
    print()
    print(f"  LUT dark reds are {'DARKER' if avg_v < target_v else 'BRIGHTER'} than target")
    print(f"    (V={avg_v:.3f} vs target V={target_v:.3f}, delta={abs(avg_v - target_v):.3f})")
else:
    print("WARNING: No entries found in dark red region!")
print()

# Detailed analysis of purple region (270-310)
print("3. PURPLE REGION ANALYSIS (HSV hue 270-310°)")
print("-" * 80)
purple_entries = hue_buckets['purple (270-310°)']

if purple_entries:
    print(f"Found {len(purple_entries)} LUT entries in purple region:")
    print()
    for data in purple_entries:
        r, g, b = data['rgb']
        print(f"  Index {data['idx']:3d}: RGB({r:3d}, {g:3d}, {b:3d}) | "
              f"HSV(H={data['h']:6.2f}°, S={data['s']:.3f}, V={data['v']:.3f})")

    # Average saturation and value
    avg_s = np.mean([d['s'] for d in purple_entries])
    avg_v = np.mean([d['v'] for d in purple_entries])
    print()
    print(f"  Average Saturation: {avg_s:.3f}")
    print(f"  Average Value (brightness): {avg_v:.3f}")
    print()

    # Compare to target "dark purple" (100, 0, 100)
    target_r, target_g, target_b = 100, 0, 100
    target_h, target_s, target_v = colorsys.rgb_to_hsv(target_r/255.0, target_g/255.0, target_b/255.0)
    print(f"  Target 'dark purple' RGB({target_r}, {target_g}, {target_b}):")
    print(f"    HSV(H={target_h*360:.2f}°, S={target_s:.3f}, V={target_v:.3f})")
    print()
    print(f"  LUT purples are {'DARKER' if avg_v < target_v else 'BRIGHTER'} than target")
    print(f"    (V={avg_v:.3f} vs target V={target_v:.3f}, delta={abs(avg_v - target_v):.3f})")
else:
    print("WARNING: No entries found in purple region!")
print()

# OKLCH hue analysis
print("4. OKLCH HUE RANGE ESTIMATION FOR RED AND PURPLE")
print("-" * 80)
print("The LUT was generated at L=0.75 with per-hue max chroma (98% gamut boundary).")
print()
print("OKLCH hue relationship to RGB:")
print("  - OKLCH hue 0° = magenta (not red!)")
print("  - OKLCH hue ~30° = red")
print("  - OKLCH hue ~270° = blue")
print("  - OKLCH hue ~300° = purple/magenta")
print()
print("The LUT maps OKLCH hue linearly across 256 entries:")
print(f"  - Index 0 = OKLCH hue 0° (magenta/pink)")
print(f"  - Index 21-22 ≈ OKLCH hue 30° (red)")
print(f"  - Index 192 ≈ OKLCH hue 270° (blue)")
print(f"  - Index 213 ≈ OKLCH hue 300° (purple)")
print()

# Check what HSV hues map to these OKLCH indices
print("Actual HSV hues at key OKLCH positions:")
idx_checks = [0, 21, 22, 192, 213, 255]
for idx in idx_checks:
    if idx < len(hsv_data):
        data = hsv_data[idx]
        oklch_hue = (idx / 256.0) * 360
        r, g, b = data['rgb']
        print(f"  Index {idx:3d} (OKLCH ≈{oklch_hue:5.1f}°): "
              f"RGB({r:3d},{g:3d},{b:3d}) → HSV hue {data['h']:6.2f}°")
print()

# Brightness analysis
print("5. L=0.75 BRIGHTNESS ANALYSIS")
print("-" * 80)
print("At L=0.75 in OKLCH (75% lightness), all colors are fairly bright.")
print()
print("For 'dark' reds and purples, we need LOWER lightness values:")
print("  - Dark red (180,0,0) has V=0.706 (70.6% brightness)")
print("  - Dark purple (100,0,100) has V=0.392 (39.2% brightness)")
print()

all_v_values = [d['v'] for d in hsv_data]
min_v = min(all_v_values)
max_v = max(all_v_values)
avg_v = np.mean(all_v_values)

print(f"Current LUT V (brightness) range:")
print(f"  Min: {min_v:.3f} ({min_v*100:.1f}%)")
print(f"  Max: {max_v:.3f} ({max_v*100:.1f}%)")
print(f"  Avg: {avg_v:.3f} ({avg_v*100:.1f}%)")
print()
print("FINDING: L=0.75 is too bright for 'dark' colors.")
print()

# Distribution analysis
print("6. HUE DISTRIBUTION PROBLEM")
print("-" * 80)
total_entries = len(hsv_data)
red_purple_count = len(dark_red_entries) + len(purple_entries)
red_purple_pct = (red_purple_count / total_entries) * 100

print(f"Red + Purple entries: {red_purple_count} / {total_entries} = {red_purple_pct:.1f}%")
print()
print("The OKLCH hue wheel is different from HSV:")
print("  - OKLCH spreads hues based on perceptual uniformity")
print("  - Red/purple occupy less perceptual hue space than green/cyan")
print("  - This is by design in OKLCH (perceptually uniform), but means")
print("    fewer LUT entries land in red/purple regions when mapped to HSV.")
print()

# Recommendations
print("=" * 80)
print("7. RECOMMENDATIONS")
print("=" * 80)
print()
print("TWO SEPARATE ISSUES:")
print()
print("Issue 1: BRIGHTNESS (L=0.75 too high)")
print("  - Current LUT uses constant L=0.75 (75% lightness)")
print("  - This produces vibrant, bright colors but NOT 'dark' ones")
print("  - Dark red needs L≈0.50, dark purple needs L≈0.35")
print()
print("  FIX: Vary L by hue")
print("    - Use L=0.50-0.55 for red hues (OKLCH 20-40°)")
print("    - Use L=0.35-0.40 for purple hues (OKLCH 290-320°)")
print("    - Keep L=0.75 for yellow/cyan (already good)")
print()
print("Issue 2: HUE DISTRIBUTION (fewer entries in red/purple)")
print(f"  - Red/purple only occupy {red_purple_pct:.1f}% of LUT entries")
print("  - This is inherent to OKLCH's perceptual uniformity")
print("  - Can't fix without breaking perceptual uniformity elsewhere")
print()
print("  WORKAROUND: Accept the tradeoff")
print("    - OKLCH's strength is smooth gradients across all hues")
print("    - Red/purple get fewer entries but are still represented")
print("    - Lowering L for these hues will make them MORE visually distinct")
print()
print("RECOMMENDATION: Generate new LUT with variable L by hue")
print("  - Keep OKLCH perceptual hue spacing (don't adjust hue distribution)")
print("  - Lower L to 0.50 for reds, 0.35-0.40 for purples")
print("  - This will produce darker, more saturated reds and purples")
print("  - May need to adjust max chroma per-hue to stay in gamut at lower L")
print()
print("=" * 80)
