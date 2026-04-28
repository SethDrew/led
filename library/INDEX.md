# Library Index

Reference documents for the synchronous LED system. Organized by the seven design axes from ARCHITECTURE.md.

## Root

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — The design space framework: seven axes, key principles, input roles, show architecture.
- **[visual-qa-preview-system.md](visual-qa-preview-system.md)** — Internal tooling: composite PNG preview for AI-driven effect QA, team pattern, iteration protocol.
- **[BULB_INTERACTION_TELEMETRY.md](BULB_INTERACTION_TELEMETRY.md)** — Hanging-bulb gesture pipeline: IMU capture, gesture taxonomy (kind × severity), locked 16 B / 25 Hz / 400 B/s ESP-NOW wire schema, classifier validation, residual gaps.

## arch-impl/

Deep-dive documents organized by the seven axes defined in ARCHITECTURE.md. Browse `arch-impl/` by axis folder.

### axis1-topologies/

- **NATURE_TOPOLOGY_EFFECTS_CATALOG.md** — Effects that exploit physical sculpture form: hub-and-spoke, height-mapped, radial, and diamond-specific idioms with audio hooks.

### axis3-audio-features/

- **AUDIO_FEATURES.md** — Full catalog of computable audio features.
- **AUDIO_ANALYSIS_ALGORITHMS.md** — Algorithm details for audio analysis.
- **SIGNAL_NORMALIZATION.md** — The normalization primitive: EMA multi-output design (peak_normalized, ema_ratio, sigmoid), time constants in wall-clock seconds, feature-specific normalization strategies.
- **PER_BAND_NORMALIZATION.md** — Per-band EMA-ratio normalization with noise-floor gating. Standard pipeline (PerBandEMANormalize → PerBandAbsIntegral → PulseDriver) for band-decomposition RMS amplitude effects.

### axis4-led-behaviors/

- **MATHY_EFFECTS_CATALOG.md** — Generative art patterns adapted for 1D LED arrays: noise, waves, cellular automata, particle systems, mathematical functions.
- **ENTITY_INTERACTIONS.md** — Multi-entity interaction types: collision, attraction/repulsion, communication, competition, synchronization.

### axis6-composition/

- **INPUT_ROLE_MATRIX.md** — Primary/secondary input role assignments across all effects.
- **SHOW_AUTOMATION.md** — Show automation and VJ replacement architecture.

### axis7-perceptual/

- **COLOR_ENGINEERING.md** — Color pipeline: OKLCH rainbow LUT, hybrid gamma brightness, chroma/desaturation control. WS2812B-specific tuning.
- **SK6812_OUTPUT_PIPELINE.md** — SK6812 RGBW output stage: gamma 2.4, pure-W low-brightness routing, 8.8 fixed-point delta-sigma dithering, sub-LSB noise gate. The float-to-wire pipeline.
- **AUDIO_VISUAL_MAPPING_PATTERNS.md** — Algorithms and constants from successful audio-reactive LED projects (energy mapping, frequency-to-color, beat detection).
- **SPECTROGRAM_BASED_COLOR.md** — Strategies for mapping mel spectrograms and MFCCs to LED color (uniform, frequency-to-position, waterfall, centroid+bandwidth).
- **VJ_AUDIO_VISUAL_MAPPING.md** — Professional VJ conventions for audio-to-visual parameter mapping (Resolume, TouchDesigner, VDMX, etc.).

## datasets/

- *(Moved to `audio-reactive/research/datasets/harmonix/`)*

## test-vectors/

- **[inmp441-validation/](test-vectors/inmp441-validation/README.md)** — INMP441 vs system-audio synchronized 60s captures (ambient and near-field). Reusable WAV test vectors plus per-band frequency-response analysis. Validates the `inmp441-frequency-response` ledger entry: bass loss reproduces robustly with distance-dependence; "treble comparable" claim does NOT reproduce.

## research-briefs/

- **[perceptual-color-on-ws2812b.md](research-briefs/perceptual-color-on-ws2812b.md)** — Perceptual color control for WS2812B: green dominance, gamma correction, and tested compensation approaches.
- **[divisive-normalization-for-audio-reactive-visualization.md](research-briefs/divisive-normalization-for-audio-reactive-visualization.md)** — EMA normalization outperforms peak-decay for section-aware audio-reactive LED dynamics: mean-tracking, time constants, and transition behavior.
- **[vj-practitioner-interview-charles.md](research-briefs/vj-practitioner-interview-charles.md)** — VJ practitioner interview: ear-to-hand over auto-reactivity, 1-4 dimensions of control, gain/fall as the audio reactivity interface, "humans are smoothing machines."

## case-studies/

- **[FRED_AGAIN_DROP_LEDMAP.md](case-studies/FRED_AGAIN_DROP_LEDMAP.md)** — Electronic drop structure mapping as a test case for the LED system.
- **[TOOL_LAB.md](case-studies/TOOL_LAB.md)** — Metric complexity stress test using Tool's music.
