# Input Role Matrix

The full mapping of which input roles each effect consumes, plus timescale variants for each role. Companion to the input roles overview in ARCHITECTURE.md.

---

## Timescale Variants

Most roles have natural timescale variants. An effect can request the variant that matches its needs:

| Role | Fast Variant (frame/onset) | Medium Variant (beat/bar) | Slow Variant (phrase/song) |
|---|---|---|---|
| **INTENSITY** | Per-frame RMS | Beat-averaged energy | Rolling integral (phrase energy) |
| **RATE** | Instantaneous onset density | Tempo (BPM) | Tempo trend (accelerating/decelerating) |
| **MOOD** | Per-frame spectral centroid | Bar-averaged harmonic ratio | Song-level genre/mood classifier |
| **TEXTURE** | Per-frame spectral flatness | Beat-averaged noisiness | Section-level texture character |
| **DENSITY** | Per-frame onset count | Beats-per-bar activity | Phrase-level arrangement density |
| **TRAJECTORY** | RMS derivative (rising/falling now) | 4-bar energy slope | Song-level energy arc |

An effect like Seasonal Cycle would bind TRAJECTORY at the slow (phrase/song) timescale, while Sidechain Pump would bind INTENSITY at the fast (frame) timescale.

---

## The Role Matrix

Which roles each effect consumes. **P** = primary (the effect is meaningless without it), **S** = secondary (enhances the effect but not required).

| Effect | EVT | RAT | INT | MOD | TRJ | TXT | NOV | DEN | SPC | POS | FAM | SUR | PHA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chase / Theater | | P | | S | | | | | | | | | S |
| Comet / Meteor | P | S | P | | | | | | | S | | | |
| Larson Scanner | | P | S | | | | | | | | | | |
| Wipe | P | S | | | S | | | | | | | | |
| Running Lights | | P | | | | | | | | | | | S |
| Perlin Noise | | P | S | P | | S | | | S | | | | |
| Plasma | | P | | S | | | | | S | | | | S |
| Fire / Flame | P | | P | | | | | | | | | | |
| Lava / Magma | | P | P | | | | | | | | | | |
| Sparkle / Twinkle | P | | | | | P | | S | | | | | |
| Fireworks | P | | P | | | | | | | P | | | |
| Fireflies | | | S | | | | | P | S | | | | |
| Rain / Drip | P | | P | | | | | S | | | | | |
| Confetti | | | | | | | | P | | | | | |
| Breathing / Pulse | | P | S | | | | | | | | | | P |
| Ripple | P | P | | | | | | | | P | | | |
| Gradient Shift | | | | P | S | | | | P | | | | |
| Zone Coloring | | | P | | | | | | P | | | | |
| Palette Cycling | | P | | P | | | | | | | | | |
| Sidechain Pump | P | | P | | | | | | | | | | |
| Sap Flow / Growth | | | | | P | | | | | | | | |
| Root Pulse | P | P | P | | | | | | | | | | |
| Branch Lightning | P | | P | | | | | | | S | | S | |
| Capillary Fill | | | P | | S | | | | S | | | | |
| Bloom | | | P | S | | | | | | | | | |
| Heartbeat | P | P | | | | | | | | | | | P |
| Branching Cascade | P | P | S | | | | | | | | | | |
| Seasonal Cycle | | | | P | P | | | | | | | | |
| Tide / Water Level | | | P | | | | | | | | | | |
| Stratification | | | P | | | | | | P | | | | |
| Rising Heat | | | | P | S | | | | P | | | | |
| Gravity Drops | P | | P | | | | | | | | | | |
| Eruption | P | | P | | | | | | | | | | |
| Mirror Pulse | P | | | | | | | | | | | | P |
| Convergence | | | | | P | | | | | | | | |
| Divergence | | | | | P | | | | | | | | |
| Metaballs / Blobs | | | S | | | | | P | | | | | |
| Reaction-Diffusion | | | P | | | | | | | | | | |
| Cellular Automata | | | | | | | P | | | | | | |
| Flow Field | | P | | | | | | | | | | | |
| Feedback / Echo | | | | | P | | | | | | | | |
| Candle Flicker | | | P | S | | P | | | | | | | |
| Bioluminescence | | P | | S | | | | P | S | | | | |
| Electrical Storm | P | | P | | S | | | | | S | | S | |
| Embers / Coals | P | | P | | | | | | | | | | |
| Frost Crystallization | | P | | | S | | P | | | P | | | |
| Breathing Organism | | P | S | S | | S | | | | | | | P |
| Mycelium Network | P | P | | | | S | | | | | | S | |
| **Interactions** | | | | | | | | | | | | | |
| Collision | P | P | P | | | | | | | | | S | |
| Attraction/Repulsion | | | P | P | S | | | S | | | | | |
| Communication | P | P | | | | S | | | | | | S | |
| Competition | | | P | | S | | S | | P | | | | |
| Symbiosis | | S | P | P | | S | | | | | | | |
| Predator/Prey | P | P | | | | | | P | | | | S | |
| Synchronization | | P | P | | S | | S | | | | | | P |
| Inheritance | | | P | S | S | | | | | | | | |
| Decay/Growth | | | S | | P | S | | S | | | | | |

Reading the matrix: Sap Flow's only primary role is TRAJECTORY -- the effect is fundamentally about directional change over time. Root Pulse needs EVENT (when to fire), RATE (propagation speed), and INTENSITY (how bright). This tells you exactly what to bind when integrating a new signal source.
