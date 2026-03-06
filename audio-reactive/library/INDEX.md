# Library Index

Reference documents for the synchronous LED system. Keep this index updated when adding, removing, or modifying documents.

## Documents

### LIBRARY_RESEARCH_SUMMARY.md
Audio analysis library comparison (librosa, madmom, essentia, aubio, BeatNet). Test results on rock and electronic audio. Recommendations by use case (onset-reactive, beat-synced, hybrid). Installation status and code examples.

### LED_MAPPING_GUIDE.md
How to map electronic drop structure to LED effects. Phase-by-phase strategies (normal, tease, build, bridge, drop) with code, brightness/color/speed parameter tables, and real-time detection heuristics.

### HARMONIX_QUICKSTART.md
Setup guide for the Harmonix Set (912 tracks, beat annotations). How to load JAMS files, extract beat times/segments, run validation. Priority track list and expected results by genre.

### VJ_AUDIO_VISUAL_MAPPING.md
How VJ systems map audio to visuals. The 6 visual parameters (brightness, color, speed, complexity, spatial position, rhythm sync) and their audio feature sources. Perceptual science (arousal-valence model), MER dimensions and real-time feasibility, proven smoothing/normalization constants. ZCR redundancy confirmed via FMA dataset (r=0.85 with spectral centroid).

### TOOL_LAB.md
Tool as a stress test for audio-reactive LED systems. Academic literature on metric dissonance in rock (Biamonte 2014, Kozak 2021, Dorhauer 2025), the MIR dataset gap (Tool absent from all standard datasets, no dataset annotates changing meters), structured data source inventory (GuitarPro tabs, MuseScore, DrumsTheWord), six reference audio clips covering meter alternation/polyrhythm/polymeter, and five actionable research paths from meter ground truth extraction to beat tracker evaluation.

### COLOR_ON_WS2812B.md
How we produce perceptually correct color on WS2812B strips. Three independent axes: hue (OKLCH variable-L rainbow LUT), brightness (hybrid gamma correction on scalar multiplier), and chroma/saturation (unimplemented, design sketched). Covers WS2812B hardware characteristics, why per-channel gamma is wrong, the community research gap, and how the axes combine. Generator script reference, C array format, implementation code pointers.

### GENERATIVE_PATTERNS_CATALOG.md
Comprehensive catalog of visual patterns from generative art and creative coding adapted for 1D LED arrays (73-197 pixels). Covers 10 categories: noise-based (Perlin, Simplex, flow fields, Worley), wave interference (standing waves, beats, traveling waves, Lissajous), cellular automata (Rule 30/110, reaction-diffusion), physics simulations (springs, pendulums, wave equation, heat diffusion), chaos attractors (Lorenz, Rössler), mathematical sequences (Fibonacci, primes), color theory (palettes, harmonics), FastLED primitives (fire, sinelon, juggle, BPM-sync, confetti, comet, plasma, pride), and advanced patterns (metaballs, fractals, particles, easing). Each pattern rated on "physical/spatial quality" (★☆☆☆☆ to ★★★★★) for suitability to the "takes the form of the structure" aesthetic. Includes implementation approaches, code examples, ESP32 feasibility notes, and extensive references to research and creative coding resources.
