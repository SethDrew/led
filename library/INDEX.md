# Library Index

Reference documents for the synchronous LED system. Organized by the seven design axes from ARCHITECTURE.md.

## Root

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — The design space framework: seven axes, key principles, input roles, show architecture.

## arch-impl/

Deep-dive documents organized by the seven axes defined in ARCHITECTURE.md. Browse `arch-impl/` by axis folder.

### axis1-topologies/

- **NATURE_TOPOLOGY_EFFECTS_CATALOG.md** — Effects that exploit physical sculpture form: hub-and-spoke, height-mapped, radial, and diamond-specific idioms with audio hooks.

### axis3-audio-features/

- **AUDIO_FEATURES.md** — Full catalog of computable audio features.
- **AUDIO_ANALYSIS_ALGORITHMS.md** — Algorithm details for audio analysis.
- **SIGNAL_NORMALIZATION.md** — The normalization primitive: EMA multi-output design (peak_normalized, ema_ratio, sigmoid), time constants in wall-clock seconds, feature-specific normalization strategies.

### axis4-led-behaviors/

- **MATHY_EFFECTS_CATALOG.md** — Generative art patterns adapted for 1D LED arrays: noise, waves, cellular automata, particle systems, mathematical functions.
- **ENTITY_INTERACTIONS.md** — Multi-entity interaction types: collision, attraction/repulsion, communication, competition, synchronization.

### axis6-composition/

- **PER_BAND_NORMALIZATION.md** — Per-band peak-decay normalization as composition preprocessing (0-1 scaling across frequency bands).
- **INPUT_ROLE_MATRIX.md** — Primary/secondary input role assignments across all effects.
- **SHOW_AUTOMATION.md** — Show automation and VJ replacement architecture.

### axis7-perceptual/

- **COLOR_ENGINEERING.md** — Color pipeline: OKLCH rainbow LUT, hybrid gamma brightness, chroma/desaturation control. WS2812B-specific tuning.
- **AUDIO_VISUAL_MAPPING_PATTERNS.md** — Algorithms and constants from successful audio-reactive LED projects (energy mapping, frequency-to-color, beat detection).
- **SPECTROGRAM_BASED_COLOR.md** — Strategies for mapping mel spectrograms and MFCCs to LED color (uniform, frequency-to-position, waterfall, centroid+bandwidth).
- **VJ_AUDIO_VISUAL_MAPPING.md** — Professional VJ conventions for audio-to-visual parameter mapping (Resolume, TouchDesigner, VDMX, etc.).

## datasets/

- *(Moved to `audio-reactive/research/datasets/harmonix/`)*

## research-briefs/

- **[perceptual-color-on-ws2812b.md](research-briefs/perceptual-color-on-ws2812b.md)** — Perceptual color control for WS2812B: green dominance, gamma correction, and tested compensation approaches.
- **[divisive-normalization-for-audio-reactive-visualization.md](research-briefs/divisive-normalization-for-audio-reactive-visualization.md)** — EMA normalization outperforms peak-decay for section-aware audio-reactive LED dynamics: mean-tracking, time constants, and transition behavior.

## case-studies/

- **[FRED_AGAIN_DROP_LEDMAP.md](case-studies/FRED_AGAIN_DROP_LEDMAP.md)** — Electronic drop structure mapping as a test case for the LED system.
- **[TOOL_LAB.md](case-studies/TOOL_LAB.md)** — Metric complexity stress test using Tool's music.
