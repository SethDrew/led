# Library Index

Reference documents for the synchronous LED system. Keep this index updated when adding, removing, or modifying documents.

## Documents

### LIBRARY_RESEARCH_SUMMARY.md
Audio analysis library comparison (librosa, madmom, essentia, aubio, BeatNet). Test results on rock and electronic audio. Recommendations by use case (onset-reactive, beat-synced, hybrid). Installation status and code examples.

### LED_MAPPING_GUIDE.md
Case study: mapping a Fred Again drop (`fa_br_drop1.wav`) to LED effects. Phase-by-phase breakdown (normal, tease, build, bridge, drop) with code, brightness/color/speed parameter tables, and detection heuristics. One worked example, not a general guide.

### HARMONIX_QUICKSTART.md
Setup guide for the Harmonix Set (912 tracks, beat annotations). How to load JAMS files, extract beat times/segments, run validation. Priority track list and expected results by genre.

### VJ_AUDIO_VISUAL_MAPPING.md
How VJ systems map audio to visuals. The 6 visual parameters (brightness, color, speed, complexity, spatial position, rhythm sync) and their audio feature sources. Perceptual science (arousal-valence model), MER dimensions and real-time feasibility, proven smoothing/normalization constants. ZCR redundancy confirmed via FMA dataset (r=0.85 with spectral centroid).

### TOPOLOGY_NATIVE_EFFECTS.md
Catalog of effects that exploit sculpture spatial properties (height, branch, neighbor connectivity, distance-from-root) rather than treating LEDs as a flat array. Nature simulations section covers 14 biomimicry effects (fireflies, rain, aurora, fire, breathing, lightning, water, wind, bioluminescence, heartbeat, swarm, vines, snowfall, sunrise/sunset) with topology property usage and input role mappings.
