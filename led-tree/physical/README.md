# Tree Effects - Modular System

Modular LED effects system for the physical tree (197 LEDs across 3 strips).

## Architecture

```
physical/
├── TreeTopology.h          # Tree structure (197 LEDs, 3 strips, depth mapping)
├── tree_final.ino          # Simple wave animation
├── tree_effects.ino        # Modular effects system ← Start here
├── src/
│   ├── TreeEffect.h        # Base classes
│   ├── backgrounds/        # Background effects (REPLACE blend)
│   │   ├── SolidColorBackground.h
│   │   └── DepthGradientBackground.h
│   └── foregrounds/        # Foreground effects (ADD blend)
│       ├── DepthWaveForeground.h
│       ├── SapFlowForeground.h
│       └── TwinkleForeground.h
└── platformio.ini
```

## Effect System

### Background Effects (REPLACE blend)
Set the base look of the tree. Only one background active at a time.

- **SolidColorBackground**: Fill entire tree with one color
- **DepthGradientBackground**: Color gradient from base to tips

### Foreground Effects (ADD blend)
Layer on top of backgrounds. Multiple foregrounds can be active.

- **DepthWaveForeground**: Wave flowing up/down tree (classic animation)
- **SapFlowForeground**: Particles rising through tree like sap
- **TwinkleForeground**: Random sparkles

## Pre-made Animations

In `tree_effects.ino`:

1. **classicWaveAnimation()** - Green wave (original tree_final.ino effect)
2. **sapFlowAnimation()** - Earth tones with rising green sap
3. **twinkleTreeAnimation()** - Gradient with white sparkles
4. **autumnWaveAnimation()** - Purple/orange gradient with orange wave
5. **springAwakeningAnimation()** - Sap flow + yellow sparkles
6. **pureSapAnimation()** - Just sap on black background
7. **rainbowTreeAnimation()** - Rainbow gradient with white wave

## Usage

### Upload to Arduino

```bash
cd led-tree/physical
pio run --target upload
```

### Switch Animations

In `tree_effects.ino`, uncomment the animation you want in `loop()`:

```cpp
void loop() {
  // classicWaveAnimation();      // ← Comment out current
  springAwakeningAnimation();     // ← Uncomment desired animation
  delay(40);
}
```

### Create New Effect

1. Create header file in `src/backgrounds/` or `src/foregrounds/`
2. Inherit from `TreeBackgroundEffect` or `TreeForegroundEffect`
3. Implement required methods:

```cpp
class MyEffect : public TreeForegroundEffect {
public:
  MyEffect(Tree* tree) : TreeForegroundEffect(tree) {}

  void update() override {
    // Update animation state each frame
  }

  void render(uint8_t buffer[][3], BlendMode blend = ADD) override {
    // Render to buffer
    for (uint8_t i = 0; i < tree->getNumLEDs(); i++) {
      uint8_t depth = tree->getDepth(i);
      // ... calculate colors based on depth, time, etc.
      buffer[i][0] = r;
      buffer[i][1] = g;
      buffer[i][2] = b;
    }
  }
};
```

4. Include and use it:

```cpp
#include "src/foregrounds/MyEffect.h"

void myAnimation() {
  static MyEffect effect(&tree);
  effect.update();
  renderLayeredFrame(NULL, &effect);
}
```

## Tree Topology

The tree has 197 LEDs across 3 physical strips with a depth-based structure:

- **Depth range**: 0 (base) to 70 (tips)
- **Branches**: 5 branches splitting at depths 25, 38, and 43
- **Strips**:
  - Strip 1 (Pin 13): 92 LEDs - Lower trunk + 2 branches
  - Strip 2 (Pin 12): 6 LEDs - Side branch
  - Strip 3 (Pin 11): 99 LEDs - Upper trunk + 2 branches

See `TreeTopology.h` for detailed structure.

## Memory Usage

Current: **46% RAM** (942 bytes of 2048 bytes)

Optimizations applied:
- Compact 3-byte node representation (591 bytes for 197 LEDs)
- Shared rendering buffer
- Effects instantiated as static locals
