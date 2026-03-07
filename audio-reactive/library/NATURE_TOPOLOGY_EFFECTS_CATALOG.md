# Topology-Native Effects Reference

Effects that exploit the physical form of LED sculptures -- the ones that can't exist on a flat screen. Organized by what topology property they use, plus nature simulations and 3D coordinate requirements.

The organizing principle: the best effects **can't be rendered on a screen** -- they take the form of the object they're mapped onto. A sap-flow effect on a tree sculpture *is* sap flowing up a tree. Sparkles on a diamond *are* gemstone refractions. Filter the entire VJ/LED/shader vocabulary through this lens: **does this effect gain something from the physical form, or is it just a screen effect squished onto a strip?**

---

## Hub-and-Spoke / Tree Topology

Effects that use the fact that branches share a common trunk/base point.

| Idiom | Description | What Makes It Physical | Audio Hook |
|---|---|---|---|
| **Sap Flow / Growth** | LEDs light progressively from base to tips | Energy literally *rises* through the sculpture | Rolling integral drives fill height; builds = growth, breakdowns = retreat |
| **Root Pulse** | Flash originates at base, travels outward to all branch tips | Energy radiates from a center point | Bass hit triggers pulse at trunk, propagates to tips at speed proportional to onset strength |
| **Branch Lightning** | Random branch illuminates tip->base, quick flash + slow fade | Electricity crackles through structure | Triggered by strong transients; which branch = random or frequency-mapped |
| **Capillary Fill** | Slow fill from base, each branch fills at different rate | Like water being drawn up through vessels | Per-band energy drives per-branch fill rate (bass=main trunk, treble=fine branches) |
| **Bloom** | Tips light up first (like flowers opening), then fills inward | Growth from the extremities | Treble energy drives tip brightness; harmonic sections activate bloom |
| **Heartbeat** | Pulsing wave from center outward, like a circulatory system | The sculpture has a pulse | Bass-locked double-pulse (lub-dub), travels at physiological rates |
| **Branching Cascade** | Event at trunk cascades outward, splitting at branch points | Like a chain reaction through the structure | Beat triggers cascade; intensity diminishes at each branch point or amplifies |
| **Seasonal Cycle** | Slow color shift base->tip (green trunk, autumn tips) | Tree *has* seasons | Long-horizon energy drives warmth; high energy = summer (green), low = autumn (amber/red) |

## Height-Mapped Topology (Diamond, Tree)

Effects that use the vertical axis -- base is "ground," tips are "sky."

| Idiom | Description | What Makes It Physical | Audio Hook |
|---|---|---|---|
| **Tide / Water Level** | Color fills from bottom to a level, with surface shimmer | Water filling a vessel | RMS drives water level; surface sparkle from high-freq onset |
| **Stratification** | Layers of color at different heights, like geological strata | Horizontal bands in 3D space | Per-band energy maps to height bands (bass=ground, treble=sky) |
| **Rising Heat** | Warm colors at base, cool at top, boundary shifts | Heat rises | Spectral centroid drives the warm/cool boundary height |
| **Gravity Drops** | Lit pixels fall from tips to base and pool | Physical gravity | Onset spawns drops at tips; they accelerate downward and accumulate at base |
| **Eruption** | Base suddenly launches particles upward | Volcanic energy | Strong bass transient triggers upward particle burst from base |

## Symmetry Topology (Diamond, any multi-branch)

Effects that use the fact that branches mirror each other.

| Idiom | Description | What Makes It Physical | Audio Hook |
|---|---|---|---|
| **Mirror Pulse** | Identical animations on symmetric branches | Emphasizes the sculpture's geometry | Beat-synced pulse, mirrored perfectly on both sides |
| **Asymmetric Counterpoint** | Different-but-related patterns on each branch | Musical counterpoint made visible | Bass drives one side, treble drives the other |
| **Convergence** | Patterns flow inward from tips to center | Energy gathering | Build sections drive convergence; drop triggers release |
| **Divergence** | Patterns flow outward from center to tips | Energy radiating | Drop sections drive divergence; breakdowns = convergence |

---

## Nature Simulations

Effects inspired by natural phenomena that feel *correct* on physical objects in a room.

| Idiom | Natural Phenomenon | Implementation Sketch | Audio Hook | Why Physical |
|---|---|---|---|---|
| **Candle Flicker** | Flame | Perlin noise with warm palette, asymmetric (fast flare, slow dim) | Flicker rate from spectral flatness; brightness from RMS | Sculpture becomes a candle |
| **Bioluminescence** | Deep-sea creatures | Random soft glows that pulse independently, blue-green | Pulse rate from tempo, density from onset rate | Sculpture glows like it's alive |
| **Electrical Storm** | Lightning + afterglow | Quick white flash on random segment + slow purple afterglow | Flash from strong transient; frequency increases during builds | Sculpture crackles with energy |
| **Embers / Coals** | Dying fire | Deep red base with occasional orange flares | Flare rate from onset, base brightness from RMS | Sculpture smolders |
| **Frost Crystallization** | Ice forming | White pixels spread from a seed point, branching | Spread rate from tempo (slow), trigger from section boundary | Ice creeps across the sculpture |
| **Magma / Lava Flow** | Molten rock | Slow noise field, red-orange palette, occasional bright fissures | Flow speed from tempo, fissure rate from onset | Sculpture looks molten |
| **Breathing Organism** | Slow organic pulse | Sine brightness + subtle color shift, very slow | Rate slightly offset from tempo (uncanny living feel) | Sculpture breathes |
| **Mycelium Network** | Fungal signal propagation | Signal travels between nodes, activating neighbors | Signal = onset detection, propagation speed from tempo | Network of life under the surface |

---

## Effects Requiring 3D Coordinates

These effects are meaningless on a flat strip but become powerful on a 3D sculpture. Requires world-space (x,y,z) positions in `sculptures.json`.

| Effect | Why It Needs 3D | Audio Role |
|---|---|---|
| **Directional wave** | A plane wave sweeps through 3D space (left-to-right wash). LEDs activate as the wavefront passes their position. On a 3D sculpture, the wavefront reveals the sculpture's shape. | EVENT triggers wave, RATE controls speed, MOOD or stereo field sets direction |
| **Spherical expansion** | An event at a 3D point creates an expanding sphere of light. All LEDs at the same radius activate simultaneously, regardless of branch. | EVENT triggers expansion, RATE controls speed |
| **3D noise cross-section** | The sculpture "slices" through a 4D noise field (x, y, z, t). Each LED samples noise at its world position. | RATE controls noise evolution, MOOD shifts scale/octaves |
| **Laterally-aware effects** | Effects that distinguish "left branch vs right branch" spatially, not just by branch ID. A directional wash sweeping left-to-right across the tree. | MOOD or stereo field maps to lateral axis |
| **Shadow / occlusion** | A "light source" illuminates the sculpture from a direction. LEDs facing the light are bright; LEDs behind are dark. Move the light source with audio. | MOOD positions the light source, INTENSITY controls brightness, RATE rotates |
| **Depth layering** | Near-viewer LEDs render differently from far-viewer LEDs. Creates a sense of depth. | SPACE maps depth to visual layer |

## Interactions in World Space

The multi-entity interactions become richer with 3D coordinates:

| Interaction | In Topology Space | In World Space |
|---|---|---|
| **Collision** | Entities collide when at the same branch position. Two entities on different branches can never collide. | Entities collide within physical proximity. Two entities on adjacent branches at the same height can collide across branches. |
| **Attraction** | Force proportional to branch-relative distance. Entities on different branches only interact through the trunk. | Force proportional to physical distance. Adjacent LEDs in world space interact directly, creating cross-branch dynamics. |
| **Communication** | Signal propagates along branch paths, must traverse trunk to reach other branches. | Signal propagates through 3D space. Adjacent LEDs communicate directly regardless of wiring. |
| **Competition** | Territory boundaries are branch-relative ranges. Cannot span across branches. | Territory boundaries are 3D surfaces. A frequency band can "own" a spatial region spanning multiple branches. |
| **Synchronization** | Coupling based on branch-relative distance. Entities on separate branches couple only through the trunk. | Coupling based on physical distance. Entities cluster and synchronize based on spatial proximity. |
