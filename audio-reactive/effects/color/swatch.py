"""
Named-color swatch helper backed by the OKLCH variable-L rainbow LUT.

`swatch(hue_deg)` returns the RGB tuple for a hue (degrees) by indexing
the 256-entry RAINBOW_LUT. `NAMED_HUES` provides a small set of common
hue degrees so callers can write `swatch(NAMED_HUES["red"])` rather than
repeating the magic numbers.

See ledger: oklch-perceptual-rainbow.
"""

from .oklch import RAINBOW_LUT


NAMED_HUES = {
    "red":    0,
    "orange": 30,
    "yellow": 60,
    "green":  120,
    "blue":   240,
    "purple": 270,
}


def swatch(hue_deg):
    """Return RGB tuple from RAINBOW_LUT for `hue_deg` (degrees)."""
    idx = int(round(hue_deg / (360.0 / 256.0))) % 256
    return tuple(int(x) for x in RAINBOW_LUT[idx])
