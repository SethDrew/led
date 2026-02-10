"""
Base class for audio-reactive LED effects.

All effects implement the same interface so we can A/B test different
algorithms (WLED-SR, our custom detectors, LedFx, etc.) on the same
audio through the same LED hardware.

The key architectural constraint (from WS2812B timing):
  Audio processing happens in a callback thread.
  LED rendering happens in a fixed-rate main loop.
  These MUST be decoupled.

So the interface has two methods:
  process_audio()  — called from audio thread, updates internal state
  render()         — called from main loop at LED_FPS, returns RGB frame
"""

import numpy as np
from abc import ABC, abstractmethod


class AudioReactiveEffect(ABC):
    """Base class for all audio-reactive LED effects."""

    def __init__(self, num_leds: int, sample_rate: int = 44100):
        self.num_leds = num_leds
        self.sample_rate = sample_rate

    @property
    def name(self) -> str:
        """Human-readable name for display."""
        return self.__class__.__name__

    @abstractmethod
    def process_audio(self, mono_chunk: np.ndarray):
        """Process a chunk of mono audio. Called from audio thread.

        Args:
            mono_chunk: float32 array of audio samples, typically 1024 samples
                        (~23ms at 44100 Hz). Values roughly in [-1, 1].

        Must be thread-safe and fast (< 5ms).
        Store results in internal state for render() to read.
        """

    @abstractmethod
    def render(self, dt: float) -> np.ndarray:
        """Render current LED frame. Called from main loop at fixed rate.

        Args:
            dt: seconds since last render call (typically 1/30)

        Returns:
            np.ndarray of shape (num_leds, 3), dtype uint8, RGB values 0-255.
            Brightness capping is handled by the runner, not here.
        """

    def get_diagnostics(self) -> dict:
        """Optional: return diagnostic info for terminal display.

        Override to provide effect-specific metrics (beat count, energy, etc.)
        """
        return {}
