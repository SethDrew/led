"""
ComposedEffect — glues a ScalarSignalEffect to a PaletteMap.

The result is a standard AudioReactiveEffect so nothing downstream changes.
"""

from base import AudioReactiveEffect


class ComposedEffect(AudioReactiveEffect):
    """Wraps a ScalarSignalEffect + PaletteMap as a standard AudioReactiveEffect."""

    def __init__(self, signal, palette):
        super().__init__(signal.num_leds, signal.sample_rate)
        self.signal = signal
        self.palette = palette
        self.palette.setup(signal.num_leds)

    @property
    def name(self):
        return self.signal.name

    @property
    def description(self):
        return self.signal.description

    def process_audio(self, mono_chunk):
        self.signal.process_audio(mono_chunk)

    def render(self, dt):
        intensity = self.signal.get_intensity(dt)
        return self.palette.colorize(intensity, self.num_leds)

    def get_diagnostics(self):
        return self.signal.get_diagnostics()
