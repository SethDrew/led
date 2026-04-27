"""
OKLCH→sRGB color machinery and the variable-L rainbow LUT.

Originally inlined in worley_voronoi.py and duplicated in festicorn
tooling. Centralized here so all Python consumers share a single
canonical implementation. The festicorn firmware LUT
(`festicorn/lib/oklch_lut/oklch_lut.cpp`) is now a build artifact of
this module — see `gen_firmware_lut.py`.

See ledger: oklch-perceptual-rainbow, oklch-color-solid-coverage.
"""

import math
import numpy as np


def _oklch_to_oklab(L, C, h_deg):
    h = math.radians(h_deg)
    return (L, C * math.cos(h), C * math.sin(h))


def _oklab_to_linear_srgb(L, a, b):
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


def _in_srgb_gamut(L, C, h_deg):
    lab = _oklch_to_oklab(L, C, h_deg)
    r, g, b = _oklab_to_linear_srgb(*lab)
    return -0.001 <= r <= 1.001 and -0.001 <= g <= 1.001 and -0.001 <= b <= 1.001


def _max_chroma_for(L, h_deg):
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _in_srgb_gamut(L, mid, h_deg):
            lo = mid
        else:
            hi = mid
    return lo * 0.98


def lightness_for_hue(h_deg):
    """Hue-dependent lightness: cosine dips at red (30 deg) and purple (300 deg)."""
    L_base = 0.75
    def cosine_bump(h, center, depth, half_width):
        diff = (h - center + 180) % 360 - 180
        if abs(diff) >= half_width:
            return 0.0
        return depth * 0.5 * (1.0 + math.cos(math.pi * diff / half_width))
    dip = cosine_bump(h_deg, 30.0, 0.23, 55.0)
    dip += cosine_bump(h_deg, 300.0, 0.37, 50.0)
    return L_base - dip


def constant_lightness(_h_deg):
    """L=0.75 everywhere — used by the firmware constant-L LUT."""
    return 0.75


def _build_rainbow_lut(lightness_fn=lightness_for_hue):
    """Generate 256-entry OKLCH rainbow as (256, 3) uint8 numpy array."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        h_deg = i * 360.0 / 256.0
        L = lightness_fn(h_deg)
        C = _max_chroma_for(L, h_deg)
        lab = _oklch_to_oklab(L, C, h_deg)
        r, g, b = _oklab_to_linear_srgb(*lab)
        lut[i] = (
            int(round(max(0, min(1, r)) * 255)),
            int(round(max(0, min(1, g)) * 255)),
            int(round(max(0, min(1, b)) * 255)),
        )
    return lut


RAINBOW_LUT = _build_rainbow_lut()
