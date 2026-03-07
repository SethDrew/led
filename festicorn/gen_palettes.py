#!/usr/bin/env python3
"""
Generate OKLCH-based palettes for festicorn effects.

Uses the same OKLab math as gen_oklch_varL_lut.py to produce:
  Category 2: Two-color hue-arc gradients through OKLCH space (16 stops each)
  Category 3: Single-hue chroma sweeps from max saturation to gray (8 stops each)

All RGB values are computed from OKLCH coordinates, not guessed.
Output: C arrays + CSS gradients for the web UI.
"""
import math


# ── OKLab / OKLCH conversion (same as gen_oklch_varL_lut.py) ────────

def oklch_to_oklab(L, C, h_deg):
    h = math.radians(h_deg)
    return (L, C * math.cos(h), C * math.sin(h))


def oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_out = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return (r, g, b_out)


def in_srgb_gamut(L, C, h_deg, margin=0.02):
    lab = oklch_to_oklab(L, C, h_deg)
    r, g, b = oklab_to_linear_srgb(*lab)
    lo = -margin
    hi = 1.0 + margin
    return lo <= r <= hi and lo <= g <= hi and lo <= b <= hi


def max_chroma_for(L, h_deg, target_fraction=0.98):
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if in_srgb_gamut(L, mid, h_deg, margin=0.001):
            lo = mid
        else:
            hi = mid
    return lo * target_fraction


def oklch_to_srgb8(L, C, h_deg):
    lab = oklch_to_oklab(L, C, h_deg)
    r, g, b = oklab_to_linear_srgb(*lab)
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


# ── Hue-dependent lightness (same as gen_oklch_varL_lut.py) ─────────

def lightness_for_hue(h_deg):
    L_base = 0.75
    red_center = 30.0
    red_depth = 0.23
    red_width = 55.0
    purple_center = 300.0
    purple_depth = 0.37
    purple_width = 50.0

    def cosine_bump(h, center, depth, half_width):
        diff = (h - center + 180) % 360 - 180
        if abs(diff) >= half_width:
            return 0.0
        return depth * 0.5 * (1.0 + math.cos(math.pi * diff / half_width))

    dip = cosine_bump(h_deg, red_center, red_depth, red_width)
    dip += cosine_bump(h_deg, purple_center, purple_depth, purple_width)
    return L_base - dip


# ── Category 2: Hue-arc gradients ──────────────────────────────────

def hue_arc_stops(h_start, h_end, n_stops=16):
    """Generate n_stops along a hue arc from h_start to h_end (degrees).
    Arc goes in the positive (CCW) direction. If h_end < h_start,
    wraps through 360."""
    stops = []
    # Compute arc length
    arc = (h_end - h_start) % 360
    if arc == 0:
        arc = 360  # full circle if same hue
    for i in range(n_stops):
        t = i / (n_stops - 1)
        h = (h_start + t * arc) % 360
        L = lightness_for_hue(h)
        C = max_chroma_for(L, h)
        r, g, b = oklch_to_srgb8(L, C, h)
        stops.append((r, g, b))
    return stops


# Define gradients: (name, c_name, h_start, h_end)
# Arc direction: positive (CCW through OKLCH hue wheel)
GRADIENTS = [
    # Red -> Blue uses custom non-uniform hue stops (see hue_arc_red_blue())
    # ("Red -> Blue",      "redBlue",      0,   270),  # replaced with non-uniform version
    ("Cyan -> Gold",     "cyanGold",     195,  90),  # cool-to-warm
    ("Green -> Purple",  "greenPurple",  145, 310),  # complementary
    ("Orange -> Teal",   "orangeTeal",    50, 180),  # fire-to-ice
    ("Magenta -> Cyan",  "magentaCyan",  340, 195),  # neon sweep
]


# ── Category 3: Chroma sweeps ──────────────────────────────────────

def chroma_sweep_stops(h_deg, n_stops=8):
    """Generate n_stops from max chroma down to 0 at a fixed hue."""
    L = lightness_for_hue(h_deg)
    C_max = max_chroma_for(L, h_deg)
    stops = []
    for i in range(n_stops):
        t = i / (n_stops - 1)  # 0 = max chroma, 1 = zero chroma
        C = C_max * (1.0 - t)
        r, g, b = oklch_to_srgb8(L, C, h_deg)
        stops.append((r, g, b))
    return stops


CHROMA_SWEEPS = [
    ("Blue Wash",   "blueWash",   265),
    ("Red Wash",    "redWash",     25),
    ("Green Wash",  "greenWash",  145),
    ("Purple Wash", "purpleWash", 305),
    ("Gold Wash",   "goldWash",    90),
]


# ── Output formatters ──────────────────────────────────────────────

def format_c_array(name, stops):
    """Format as C RGB array."""
    lines = []
    lines.append(f"static const RGB {name}[] = {{")
    for i, (r, g, b) in enumerate(stops):
        comma = "," if i < len(stops) - 1 else ""
        lines.append(f"    {{{r:3d}, {g:3d}, {b:3d}}}{comma}")
    lines.append("};")
    lines.append(f"static const uint8_t {name}Count = sizeof({name}) / sizeof({name}[0]);")
    return "\n".join(lines)


def format_css_gradient(stops):
    """Format as CSS linear-gradient string."""
    colors = [f"rgb({r},{g},{b})" for r, g, b in stops]
    # For CSS preview, sample ~5 evenly spaced stops
    if len(colors) > 5:
        indices = [int(i * (len(colors) - 1) / 4) for i in range(5)]
        colors = [colors[i] for i in indices]
    return "linear-gradient(90deg, " + ", ".join(colors) + ")"


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  OKLCH-COMPUTED PALETTES")
    print("=" * 70)

    # --- Category 2: Hue-arc gradients ---
    print("\n// ── Category 2: Hue-arc gradients (OKLCH-computed) ──────────────\n")
    for label, c_name, h_start, h_end in GRADIENTS:
        stops = hue_arc_stops(h_start, h_end, 16)
        print(f"// {label}: hue {h_start} -> {h_end} deg OKLCH")
        print(format_c_array(c_name, stops))
        print()

    # --- Category 3: Chroma sweeps ---
    print("\n// ── Category 3: Chroma sweeps (OKLCH-computed) ─────────────────\n")
    for label, c_name, h_deg in CHROMA_SWEEPS:
        stops = chroma_sweep_stops(h_deg, 8)
        L = lightness_for_hue(h_deg)
        C_max = max_chroma_for(L, h_deg)
        print(f"// {label}: hue={h_deg} deg, L={L:.3f}, C={C_max:.3f} -> 0")
        print(format_c_array(c_name, stops))
        print()

    # --- CSS gradients for web UI ---
    print("\n" + "=" * 70)
    print("  CSS GRADIENTS FOR WEB UI")
    print("=" * 70)
    print()

    print("// Hue-arc gradients")
    for label, c_name, h_start, h_end in GRADIENTS:
        stops = hue_arc_stops(h_start, h_end, 16)
        css = format_css_gradient(stops)
        # Convert c_name to snake_case for palette IDs
        snake = ''.join(['_' + c.lower() if c.isupper() else c for c in c_name]).lstrip('_')
        # Actually let's just use the c_name lowered with underscores
        parts = []
        for c in c_name:
            if c.isupper() and parts:
                parts.append('_')
            parts.append(c.lower())
        pid = ''.join(parts)
        print(f"  {{id:'{pid}', css:\"{css}\"}},")

    print()
    print("// Chroma sweeps")
    for label, c_name, h_deg in CHROMA_SWEEPS:
        stops = chroma_sweep_stops(h_deg, 8)
        css = format_css_gradient(stops)
        parts = []
        for c in c_name:
            if c.isupper() and parts:
                parts.append('_')
            parts.append(c.lower())
        pid = ''.join(parts)
        print(f"  {{id:'{pid}', css:\"{css}\"}},")

    # --- Palette enum entries ---
    print("\n" + "=" * 70)
    print("  PALETTE ENUM ENTRIES")
    print("=" * 70)
    print()
    for _, c_name, _, _ in GRADIENTS:
        # Convert camelCase to UPPER_SNAKE
        upper = ''.join(['_' + c if c.isupper() else c.upper() for c in c_name]).lstrip('_')
        print(f"    {upper},")
    for _, c_name, _ in CHROMA_SWEEPS:
        upper = ''.join(['_' + c if c.isupper() else c.upper() for c in c_name]).lstrip('_')
        print(f"    {upper},")
