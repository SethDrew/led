#!/usr/bin/env python3
"""
Generate a 256-entry OKLCH rainbow LUT with hue-dependent lightness.

Lightness profile:
  - Red hues (~30 deg):   L ~ 0.52
  - Purple hues (~300 deg): L ~ 0.38
  - Green/cyan/yellow:     L ~ 0.75

Smooth cosine interpolation between regions.
Per-hue max chroma via binary search (98% of gamut boundary).
Output: C array format for effects.cpp.
"""
import math
import sys


# ── OKLab / OKLCH conversion (manual, no external deps) ─────────────

def oklch_to_oklab(L, C, h_deg):
    """OKLCH -> OKLab."""
    h = math.radians(h_deg)
    return (L, C * math.cos(h), C * math.sin(h))


def oklab_to_linear_srgb(L, a, b):
    """OKLab -> linear sRGB via LMS intermediate."""
    # OKLab -> LMS (cube-root space)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Un-cube-root
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    # LMS -> linear sRGB
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_out = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    return (r, g, b_out)


def in_srgb_gamut(L, C, h_deg, margin=0.02):
    """Check if OKLCH color is within sRGB gamut (with margin)."""
    lab = oklch_to_oklab(L, C, h_deg)
    r, g, b = oklab_to_linear_srgb(*lab)
    lo = -margin
    hi = 1.0 + margin
    return lo <= r <= hi and lo <= g <= hi and lo <= b <= hi


def max_chroma_for(L, h_deg, target_fraction=0.98):
    """Binary search for max chroma at given L, h that stays in sRGB gamut.
    Returns target_fraction of the gamut boundary chroma."""
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if in_srgb_gamut(L, mid, h_deg, margin=0.001):
            lo = mid
        else:
            hi = mid
    return lo * target_fraction


def oklch_to_srgb8(L, C, h_deg):
    """OKLCH -> clamped 8-bit sRGB (no gamma, OKLab's perceptual L handles it)."""
    lab = oklch_to_oklab(L, C, h_deg)
    r, g, b = oklab_to_linear_srgb(*lab)
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


# ── Hue-dependent lightness profile ─────────────────────────────────

def lightness_for_hue(h_deg):
    """Compute lightness for a given OKLCH hue angle.

    Uses sum of two cosine bumps to lower L around red and purple hues,
    keeping green/cyan/yellow at the base lightness.

    Hue reference:
      ~30 deg  = red     -> L ~ 0.52
      ~300 deg = purple  -> L ~ 0.38
      ~90-200  = yellow/green/cyan -> L ~ 0.75
    """
    L_base = 0.75

    # Red dip: centered at 30 deg, width ~60 deg (sigma-like), depth 0.23
    red_center = 30.0
    red_depth = 0.23
    red_width = 55.0  # half-width at which bump reaches zero

    # Purple dip: centered at 300 deg, width ~50 deg, depth 0.37
    purple_center = 300.0
    purple_depth = 0.37
    purple_width = 50.0

    def cosine_bump(h, center, depth, half_width):
        """Smooth cosine dip centered at `center` with given half_width."""
        # Angular distance (wrapping around 360)
        diff = (h - center + 180) % 360 - 180  # signed, -180..+180
        if abs(diff) >= half_width:
            return 0.0
        # Cosine taper: 1 at center, 0 at +/- half_width
        return depth * 0.5 * (1.0 + math.cos(math.pi * diff / half_width))

    dip = cosine_bump(h_deg, red_center, red_depth, red_width)
    dip += cosine_bump(h_deg, purple_center, purple_depth, purple_width)

    return L_base - dip


# ── Generate LUTs ───────────────────────────────────────────────────

def generate_lut(lightness_fn, name="lut"):
    """Generate 256-entry LUT with given lightness function."""
    entries = []
    for i in range(256):
        h_deg = i * 360.0 / 256.0
        L = lightness_fn(h_deg)
        C = max_chroma_for(L, h_deg)
        r, g, b = oklch_to_srgb8(L, C, h_deg)
        entries.append((r, g, b, h_deg, L, C))
    return entries


def format_c_array(entries, array_name):
    """Format entries as a C uint8_t array."""
    lines = []
    lines.append(f"static const uint8_t {array_name}[256][3] = {{")
    for i in range(0, 256, 4):
        row = []
        for j in range(i, min(i + 4, 256)):
            r, g, b = entries[j][0], entries[j][1], entries[j][2]
            row.append(f"{{{r:3d},{g:3d},{b:3d}}}")
        comma = "," if i + 4 < 256 else ""
        lines.append("    " + ", ".join(row) + ("," if i + 4 < 256 else ""))
    lines.append("};")
    return "\n".join(lines)


def print_lightness_profile(entries, label):
    """Print ASCII art lightness profile."""
    print(f"\n{'='*70}")
    print(f"  Lightness profile: {label}")
    print(f"{'='*70}")
    print(f"  {'Hue':>5s}  {'L':>5s}  {'C':>5s}  Bar")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*40}")
    # Print every 8th entry for readability
    for i in range(0, 256, 4):
        h_deg = entries[i][3]
        L = entries[i][4]
        C = entries[i][5]
        bar_len = int(L * 50)
        bar = '#' * bar_len
        print(f"  {h_deg:5.1f}  {L:.3f}  {C:.3f}  {bar}")


if __name__ == "__main__":
    print("Generating constant-L LUT (L=0.75)...")
    const_entries = generate_lut(lambda h: 0.75, "const-L")

    print("Generating variable-L LUT...")
    var_entries = generate_lut(lightness_for_hue, "var-L")

    # Print lightness profiles
    print_lightness_profile(const_entries, "Constant L=0.75")
    print_lightness_profile(var_entries, "Variable L (red/purple lowered)")

    # Output C arrays
    print("\n" + "=" * 70)
    print("  C ARRAY: constant-L")
    print("=" * 70)
    c_const = format_c_array(const_entries, "oklchConstL")
    print(c_const)

    print("\n" + "=" * 70)
    print("  C ARRAY: variable-L")
    print("=" * 70)
    c_var = format_c_array(var_entries, "oklchVarL")
    print(c_var)

    # Also write just the variable-L array to a file for easy copy-paste
    with open("/Users/KO16K39/Documents/led/festicorn/oklch_varL_lut.h", "w") as f:
        f.write("// OKLCH variable-L rainbow LUT\n")
        f.write("// Hue-dependent lightness: red ~0.52, purple ~0.38, green/cyan/yellow ~0.75\n")
        f.write("// Per-hue max chroma at 98% gamut boundary.\n")
        f.write(c_var + "\n")

    print("\nDone. Variable-L LUT written to festicorn/oklch_varL_lut.h")
