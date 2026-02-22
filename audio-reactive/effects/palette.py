"""
PaletteMap — separates color mapping from signal effects.

A PaletteMap converts a scalar intensity (0-1) into an RGB LED frame.
It handles color interpolation, gamma correction, brightness caps,
and spatial modes (uniform, fibonacci).

Any ScalarSignalEffect can be composed with any PaletteMap via ComposedEffect.
"""

import json
import os

import numpy as np

USER_PALETTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_palettes.json')


def fibonacci_sections(total_leds):
    """Generate Fibonacci-sized sections that fit in total_leds.
    Returns list of (start, end, section_index) from start of strip."""
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= total_leds:
        fibs.append(fibs[-1] + fibs[-2])

    sections = []
    pos = total_leds
    for i, size in enumerate(fibs):
        if pos <= 0:
            break
        start = max(0, pos - size)
        sections.append((start, pos, i))
        pos = start

    if pos > 0 and sections:
        last_start, last_end, last_idx = sections[-1]
        sections[-1] = (0, last_end, last_idx)

    return sections


def _sample_colors(colors, t):
    """Sample color from Nx3 color array at position t (0-1)."""
    t = np.clip(t, 0, 1)
    n = len(colors) - 1
    idx = t * n
    lo = int(idx)
    hi = min(lo + 1, n)
    frac = idx - lo
    return colors[lo] * (1 - frac) + colors[hi] * frac


class PaletteMap:
    """Converts scalar intensity to RGB LED frame.

    Args:
        colors: Nx3 array of color stops (float32, 0-255 range).
        gamma: Power curve for perceived brightness (lower = more contrast).
        brightness_cap: Maximum intensity (0-1). 1.0 = full, 0.20 = night mode.
        spatial_mode: 'uniform' | 'fibonacci'
    """

    def __init__(self, colors, gamma=0.7, brightness_cap=1.0,
                 spatial_mode='uniform', fill_from='start'):
        self.colors = np.array(colors, dtype=np.float32)
        self.gamma = gamma
        self.brightness_cap = brightness_cap
        self.spatial_mode = spatial_mode
        self.fill_from = fill_from  # kept for serialization compat

        # Precomputed in setup()
        self._num_leds = 0
        self._led_colors = None  # (num_leds, 3) for fibonacci modes

    # Backward compat alias
    @property
    def palette(self):
        return self.colors

    def setup(self, num_leds):
        """Precompute per-LED colors for spatial modes."""
        self._num_leds = num_leds

        if self.spatial_mode == 'fibonacci':
            sections = fibonacci_sections(num_leds)
            n_groups = len(sections)
            self._led_colors = np.zeros((num_leds, 3), dtype=np.float32)
            for start, end, idx in sections:
                t = idx / max(n_groups - 1, 1)
                color = _sample_colors(self.colors, t)
                self._led_colors[start:end] = color


    def colorize(self, intensity, num_leds):
        """Convert scalar intensity (0-1) to RGB frame (num_leds, 3) uint8.

        Args:
            intensity: float 0-1, current signal intensity
            num_leds: number of LEDs

        Returns:
            np.ndarray shape (num_leds, 3), dtype uint8
        """
        if self.spatial_mode == 'uniform':
            return self._colorize_uniform(intensity, num_leds)
        elif self.spatial_mode == 'fibonacci':
            return self._colorize_fibonacci(intensity, num_leds)
        else:
            return self._colorize_uniform(intensity, num_leds)

    def _colorize_uniform(self, intensity, num_leds):
        """All LEDs same color, sampled by intensity."""
        b = min(intensity, 1.0) * self.brightness_cap
        display_b = b ** self.gamma
        color = _sample_colors(self.colors, intensity)
        pixel = (color * display_b).clip(0, 255).astype(np.uint8)
        return np.tile(pixel, (num_leds, 1))

    def _colorize_fibonacci(self, intensity, num_leds):
        """Per-LED color from fibonacci sections, uniform brightness."""
        if self._led_colors is None or len(self._led_colors) != num_leds:
            self.setup(num_leds)
        b = min(intensity, 1.0) * self.brightness_cap
        display_b = b ** self.gamma
        return (self._led_colors * display_b).clip(0, 255).astype(np.uint8)


# ── Built-in presets ────────────────────────────────────────────────

PALETTE_PRESETS = {
    'amber': PaletteMap(
        colors=[[180, 80, 0], [255, 140, 0], [255, 200, 20]],
        gamma=0.7, brightness_cap=1.0, spatial_mode='uniform',
    ),
    'reds': PaletteMap(
        colors=[
            [40,  5,  0],
            [160, 50, 0],
            [200, 20, 0],
            [180, 0,  60],
        ],
        gamma=0.6, brightness_cap=0.80, spatial_mode='uniform',
    ),
    'greens': PaletteMap(
        colors=[
            [0,  60, 10],
            [0,  150, 30],
            [20, 220, 60],
            [60, 255, 100],
        ],
        gamma=0.7, brightness_cap=0.80, spatial_mode='uniform',
    ),
    'blues': PaletteMap(
        colors=[
            [0,  0,  100],
            [0,  40, 180],
            [0,  100, 255],
            [40, 160, 255],
        ],
        gamma=0.7, brightness_cap=0.80, spatial_mode='uniform',
    ),
    'yellows': PaletteMap(
        colors=[
            [200, 100, 0],
            [255, 160, 0],
            [255, 220, 0],
            [255, 255, 50],
        ],
        gamma=0.7, brightness_cap=0.80, spatial_mode='uniform',
    ),
    'purples': PaletteMap(
        colors=[
            [40,  0,  100],
            [80,  0,  180],
            [140, 0,  255],
            [200, 40, 255],
        ],
        gamma=0.7, brightness_cap=0.80, spatial_mode='uniform',
    ),
    'rainbow': PaletteMap(
        colors=[
            [255, 0,   0],
            [255, 150, 0],
            [255, 255, 0],
            [0,   255, 0],
            [0,   100, 255],
            [120, 0,   255],
        ],
        gamma=0.7, brightness_cap=0.80, spatial_mode='uniform',
    ),
    'night_dim': PaletteMap(
        colors=[
            [30, 0, 5],
            [180, 0, 100],
        ],
        gamma=0.7, brightness_cap=0.20, spatial_mode='uniform',
    ),
    'fib_orange_purple': PaletteMap(
        colors=[
            [220, 80,  0],
            [200, 30,  0],
            [180, 10,  20],
            [170, 0,   80],
            [150, 0,  140],
            [120, 10, 200],
        ],
        gamma=0.6, brightness_cap=0.80, spatial_mode='fibonacci',
    ),
    'energy_bloom': PaletteMap(
        colors=[
            [50,  15, 35],    # dusty mauve (quiet)
            [160, 30, 70],    # warm rose (mid)
            [255, 0,  120],   # vivid magenta (loud)
        ],
        gamma=0.7, brightness_cap=1.0, spatial_mode='uniform',
    ),
    'fib_rainbow': PaletteMap(
        colors=[
            [255, 0,   0],
            [255, 150, 0],
            [255, 255, 0],
            [0,   255, 0],
            [0,   100, 255],
            [120, 0,   255],
        ],
        gamma=0.7, brightness_cap=0.80, spatial_mode='fibonacci',
    ),
}

# Backward compat aliases for old preset names
_PALETTE_ALIASES = {
    'warm_white': 'amber',
    'night_reds': 'reds',
}


# ── User palettes (persisted in JSON) ─────────────────────────────

def palette_to_dict(pal):
    """Serialize a PaletteMap to a JSON-safe dict."""
    return {
        'colors': pal.colors.astype(int).tolist(),
        'gamma': pal.gamma,
        'brightness_cap': pal.brightness_cap,
        'spatial_mode': pal.spatial_mode,
        'fill_from': pal.fill_from,
    }


def _spec_to_palette(spec):
    """Convert a JSON spec dict to a PaletteMap instance."""
    return PaletteMap(
        colors=spec.get('colors', spec.get('palette', [[255, 255, 255]])),
        gamma=spec.get('gamma', 0.7),
        brightness_cap=spec.get('brightness_cap', 1.0),
        spatial_mode=spec.get('spatial_mode', 'uniform'),
        fill_from=spec.get('fill_from', 'start'),
    )


def load_user_palettes():
    """Read user_palettes.json, return dict of {name: PaletteMap}."""
    if not os.path.exists(USER_PALETTES_PATH):
        return {}
    try:
        with open(USER_PALETTES_PATH) as f:
            data = json.load(f)
        return {name: _spec_to_palette(spec) for name, spec in data.items()}
    except (json.JSONDecodeError, Exception):
        return {}


def _load_user_palettes_raw():
    """Read user_palettes.json as raw dict (for save/delete operations)."""
    if not os.path.exists(USER_PALETTES_PATH):
        return {}
    try:
        with open(USER_PALETTES_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return {}


def save_user_palette(name, spec):
    """Write/update a user palette entry in the JSON file."""
    data = _load_user_palettes_raw()
    data[name] = spec
    with open(USER_PALETTES_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def delete_user_palette(name):
    """Remove a user palette entry from the JSON file."""
    data = _load_user_palettes_raw()
    if name in data:
        del data[name]
        with open(USER_PALETTES_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    return False


def resolve_palette_name(name):
    """Resolve a palette name, handling aliases for renamed presets."""
    return _PALETTE_ALIASES.get(name, name)


def all_palettes():
    """Merge built-in presets + user palettes. User can't shadow built-in names."""
    merged = dict(PALETTE_PRESETS)
    for name, pal in load_user_palettes().items():
        if name not in PALETTE_PRESETS:
            merged[name] = pal
    return merged
