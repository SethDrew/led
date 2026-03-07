# Library Index

Reference documents for the synchronous LED system. Keep this index updated when adding, removing, or modifying documents.

## Documents

### HARMONIX_QUICKSTART.md
Setup guide for the Harmonix Set (912 tracks, beat annotations). How to load JAMS files, extract beat times/segments, run validation. Priority track list and expected results by genre.

### VJ_AUDIO_VISUAL_MAPPING.md
How VJ systems map audio to visuals. The 6 visual parameters (brightness, color, speed, complexity, spatial position, rhythm sync) and their audio feature sources. Perceptual science (arousal-valence model), MER dimensions and real-time feasibility, proven smoothing/normalization constants. ZCR redundancy confirmed via FMA dataset (r=0.85 with spectral centroid).

### COLOR_ENGINEERING.md
Quick reference for color on LED strips. Three validated tools: hue (OKLCH variable-L rainbow LUT), brightness (hybrid gamma on scalar multiplier), and chroma (BT.601 luminance desaturation). WS2812B board-specific findings, source file pointers, external attribution. Short — designed to be loaded during color/effect design work.

### SPECTROGRAM_BASED_COLOR.md
Strategies for mapping mel spectrograms and MFCCs to LED color. Ten mapping philosophies: whole-strip uniform color, frequency-to-position, scrolling history (waterfall), MFCC→RGB direct, spectral centroid+bandwidth, dominant bin winner-takes-all, difference/novelty coloring, dual-axis 2D colormap, harmonic richness gradient, and comet waterfall. Includes web viewer visualization approaches (interactive spectrogram, simulated LED strip, side-by-side compare).

### MATHY_EFFECTS_CATALOG.md
Generative art and math-based visual patterns adapted for 1D LED arrays. Noise, waves, cellular automata, physics sims, chaos attractors, FastLED primitives, and more. Each rated on physical/spatial quality for sculpture suitability.

### AUDIO_FEATURES.md
Complete audio feature catalog for the LED effect system. Standard features (librosa/numpy: RMS, onset, centroid, flatness, ZCR, MFCCs, chroma, HPSS, mel spectrogram), custom calculus features (AbsIntegral, onset envelope, autocorrelation tempo, rolling integral), per-band expansion rules (5 bands, which features benefit), full 19-row feature catalog with categories/measures/temporal scope/behaviors, and documented dead ends (equal-loudness, absint-on-percussive, centroid-position, NMF).

### INPUT_ROLE_MATRIX.md
The full primary/secondary role mapping for all cataloged effects and interaction types (50+ rows x 13 roles). Timescale variants for each role (fast/medium/slow). Companion to the input roles overview in ARCHITECTURE.md.

### ENTITY_INTERACTIONS.md
Nine multi-entity interaction types for LED sculpture effects: collision, attraction/repulsion, communication, competition, symbiosis, predator/prey, synchronization, inheritance, and decay/growth/lifecycle. Each with description, topology fit, audio role mapping, variants. Includes topology suitability matrix, implementation complexity (ESP32 feasibility), and a cheat sheet for designing new multi-entity effects.

### NATURE_TOPOLOGY_EFFECTS_CATALOG.md
Effects that exploit the physical form of LED sculptures. Hub-and-spoke idioms (sap flow, root pulse, branch lightning, capillary fill, bloom, heartbeat, branching cascade, seasonal cycle), height-mapped idioms (tide, stratification, rising heat, gravity drops, eruption), symmetry idioms (mirror pulse, counterpoint, convergence, divergence), nature simulations (candle, bioluminescence, electrical storm, embers, frost, magma, breathing organism, mycelium), effects requiring 3D coordinates (directional wave, spherical expansion, 3D noise, lateral awareness, shadow/occlusion, depth layering), and interactions in world space vs topology space.

### case-studies/
Deep dives on specific artists/songs as test cases for the system. Fred Again (electronic drop structure mapping), Tool (metric complexity stress test).
