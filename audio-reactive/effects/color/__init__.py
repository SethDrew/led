"""Shared color machinery for audio-reactive effects."""

from .oklch import RAINBOW_LUT
from .swatch import NAMED_HUES, swatch

__all__ = ['RAINBOW_LUT', 'NAMED_HUES', 'swatch']
