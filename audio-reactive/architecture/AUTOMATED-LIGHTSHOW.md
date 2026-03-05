# Automated Light Show — Implementation Architecture

Companion to `ARCHITECTURE.md`. That document defines the design space; this one explores the axis of **"cool visual effect first, then layer in sound reactivity"** — the WLED/Fluora quadrant pushed toward musical intelligence.

Three parts: (1) a catalog of visual idioms worth stealing, organized by what makes them work on physical sculptures rather than screens, (2) the **input role** abstraction that decouples effects from specific audio signals, and (3) what it takes to automate a multi-hour show without a human VJ.

---

## Part 1: Visual Effect Idioms

### The Organizing Principle: Topology-Native Effects

The user's insight: the best effects **can't be rendered on a screen** — they take the form of the object they're mapped onto. A sap-flow effect on a tree sculpture *is* sap flowing up a tree. Sparkles on a diamond *are* gemstone refractions. This is the opposite of VJ work, where effects are projected onto flat surfaces.

This means we should filter the entire VJ/LED/shader vocabulary through a lens: **does this effect gain something from the physical form, or is it just a screen effect squished onto a strip?**

### Taxonomy: Where Visual Idioms Come From

| Source Domain | What They're Good At | Sculpture Fit | Examples |
|---|---|---|---|
| **WLED / FastLED** | Proven strip-native effects, ESP32-tested | Direct | Fire, comet, chase, pride, pacifica |
| **VJ / Resolume** | Rich parametric animations, layering | Better with 3D coordinates (see Part 3) | Plasma, tunnel, feedback loops |
| **Shader Art** | Mathematical beauty, infinite variation | Excellent with 3D coordinates; 1D slice for some | Noise fields, voronoi, SDF blobs |
| **Generative Art** | Emergent complexity from simple rules | Natural fit | Cellular automata, reaction-diffusion, wave interference |
| **LED Art Installations** | Spatial/physical presence, immersion | Direct inspiration | Squidsoup waves, TeamLab flows, Leo Villareal sequences |
| **Nature Simulation** | Organic movement, no repetition | Excellent on sculpture | Fire, water, aurora, bioluminescence, growth |
| **Physics Simulation** | Physically grounded motion | Good; better with 3D | Springs, pendulums, heat diffusion, wave equation |

---

### Category 1: Strip-Native Idioms (Proven on 1D LEDs)

These are the bread and butter — effects designed for LED strips that already work well. The opportunity is adding sound reactivity to each.

#### Motion Idioms

| Idiom | Description | Audio-Reactive Hook | Topology Fit |
|---|---|---|---|
| **Chase / Theater** | Lit pixels march along strip | Speed from tempo, color from centroid | All linear |
| **Comet / Meteor** | Bright head with decaying tail | Spawned on beat, speed from onset strength | Linear, tree (base->tip) |
| **Larson Scanner** | Bouncing dot (Cylon/KITT) | Bounce rate from tempo, width from RMS | Linear, diamond |
| **Wipe** | Color fills strip then retreats | Triggered on section change, fill rate from build energy | Linear, height-mapped |
| **Running Lights** | Sine wave moves along strip | Speed from tempo, wavelength from dominant band | All |

#### Noise / Organic Idioms

| Idiom | Description | Audio-Reactive Hook | Topology Fit |
|---|---|---|---|
| **Perlin Noise Field** | Smoothly evolving color landscape | Noise speed from tempo, palette from centroid, turbulence from RMS | All (excellent with 3D coords) |
| **Plasma** | Overlapping sine waves create flowing color | Phase speed from tempo, frequency offsets from per-band energy | All (excellent with 3D coords) |
| **Fire / Flame** | Heat rises and flickers (FastLED Fire2012) | Sparking rate from onset, cooling from inverse RMS | Height-mapped (tree, diamond) |
| **Lava / Magma** | Slow-moving blobs of warm color | Blob speed from tempo, brightness from RMS | All |
| **Aurora / Northern Lights** | Curtains of shifting green/purple | Curtain speed from tempo, color range from centroid | Linear, wide strips |

#### Particle Idioms

| Idiom | Description | Audio-Reactive Hook | Topology Fit |
|---|---|---|---|
| **Sparkle / Twinkle** | Random pixels flash and fade | Density from spectral flatness, triggers from onset, zones from frequency | All |
| **Fireworks** | Explosion point -> spreading particles | Point triggered on beat, spread from onset strength | Linear, hub-and-spoke |
| **Fireflies** | Slow random blinks, organic timing | Density from RMS, spatial clustering from dominant band | All, especially tree |
| **Rain / Drip** | Drops fall from top to bottom | Rate from onset density, splash from onset strength | Height-mapped |
| **Confetti** | Random pixel, random color, quick fade | Rate from spectral flux, palette from chroma | All |
| **Popcorn** | Pixels jump up with gravity | Jump height from onset strength, rate from tempo | Height-mapped |

#### Wave / Oscillation Idioms

| Idiom | Description | Audio-Reactive Hook | Topology Fit |
|---|---|---|---|
| **Breathing / Pulse** | Whole strip fades in and out | Rate locked to tempo or half-tempo | All |
| **Standing Waves** | Interference patterns from two sine waves | Frequencies from two dominant bands | Linear |
| **Ripple** | Concentric rings from a point | Trigger from onset, propagation speed from tempo | Hub-and-spoke, linear (excellent with 3D coords) |
| **Sinelon** | Sine wave oscillation of a lit dot | Frequency from tempo, trail from RMS | Linear |

#### Composition Idioms

| Idiom | Description | Audio-Reactive Hook | Topology Fit |
|---|---|---|---|
| **Gradient Shift** | Color temperature drifts along strip | Position from spectral centroid over time | All |
| **Zone Coloring** | Strip divided into colored regions | Zone colors from per-band energy, boundaries from section | Zoned |
| **Palette Cycling** | Colors rotate through a palette | Cycle speed from tempo, palette selection from mood/energy | All |
| **Sidechain Pump** | Brightness dips on kick, recovers | Bass energy suppresses background, fast attack slow decay | All |

---

### Category 2: Topology-Aware Idioms (The Good Stuff)

Effects that **exploit** the physical form — the ones that can't exist on a flat screen. Organized by what topology property they use.

#### Hub-and-Spoke / Tree Topology

These effects use the fact that branches share a common trunk/base point.

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

#### Height-Mapped Topology (Diamond, Tree)

These effects use the vertical axis — base is "ground," tips are "sky."

| Idiom | Description | What Makes It Physical | Audio Hook |
|---|---|---|---|
| **Tide / Water Level** | Color fills from bottom to a level, with surface shimmer | Water filling a vessel | RMS drives water level; surface sparkle from high-freq onset |
| **Stratification** | Layers of color at different heights, like geological strata | Horizontal bands in 3D space | Per-band energy maps to height bands (bass=ground, treble=sky) |
| **Rising Heat** | Warm colors at base, cool at top, boundary shifts | Heat rises | Spectral centroid drives the warm/cool boundary height |
| **Gravity Drops** | Lit pixels fall from tips to base and pool | Physical gravity | Onset spawns drops at tips; they accelerate downward and accumulate at base |
| **Eruption** | Base suddenly launches particles upward | Volcanic energy | Strong bass transient triggers upward particle burst from base |

#### Symmetry Topology (Diamond, any multi-branch)

These effects use the fact that branches mirror each other.

| Idiom | Description | What Makes It Physical | Audio Hook |
|---|---|---|---|
| **Mirror Pulse** | Identical animations on symmetric branches | Emphasizes the sculpture's geometry | Beat-synced pulse, mirrored perfectly on both sides |
| **Asymmetric Counterpoint** | Different-but-related patterns on each branch | Musical counterpoint made visible | Bass drives one side, treble drives the other |
| **Convergence** | Patterns flow inward from tips to center | Energy gathering | Build sections drive convergence; drop triggers release |
| **Divergence** | Patterns flow outward from center to tips | Energy radiating | Drop sections drive divergence; breakdowns = convergence |

---

### Category 3: Shader/VJ Idioms Adapted for Sculptures

Effects from the 2D/3D world. Some work as 1D slices along branch topology; many become significantly better with 3D world-space coordinates (see Part 3).

| Idiom | 2D/3D Origin | Sculpture Adaptation | Why It Works | Audio Hook |
|---|---|---|---|---|
| **Metaballs / Blobs** | SDF blob merging | Gaussian blobs along branches; spherical in 3D if coords available | Merging/separating feels organic | Blob count from onset density, positions from per-band energy |
| **Reaction-Diffusion** | Gray-Scott / Turing patterns | 1D activator-inhibitor per branch; 3D diffusion with coords | Creates evolving spots/stripes without repetition | Feed/kill rates modulated by RMS and centroid |
| **Cellular Automata** | Rule 30, Rule 110 | History scrolls along strip OR just current state | Emergent complexity from simple rules | Rule selection from section, birth rate from onset |
| **Flow Field** | Perlin noise vector field | Particles follow noise gradient (1D or 3D) | Organic directional motion | Noise scale from tempo, direction bias from spectral slope |
| **Voronoi** | Cell boundaries | 3D Voronoi cells with LED world positions; falls back to segments on flat strips | Color regions with organic boundaries | Cell center positions from per-band peaks |
| **Feedback / Echo** | Video feedback loops | Buffer history blended into current frame | Ghostly trails, accumulation | Feedback amount from rolling integral (builds = more echo) |
| **Slit Scan** | Time-delayed sampling | Each pixel shows a different time offset of a signal | Temporal smearing, reveals rhythm structure | The signal being scanned = audio feature over time |

---

### Category 4: Nature Simulations (Physical Presence)

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

### Category 5: Multi-Entity Interactions (Particle Dynamics)

Categories 1-4 catalog individual effects: a comet, a sparkle, a noise field. But the most visually compelling moments happen when *multiple entities interact*. Two comets colliding. Fireflies synchronizing. Organisms competing for territory. These interactions are not effects in themselves — they are *relationships between effects* (or between instances of the same effect).

This is the entity-interaction axis: given N entities on a sculpture, what rules govern their relationships? The interaction types below are organized from simple (pairwise, local) to complex (emergent, global).

#### Interaction Type 1: Collision

Two entities occupy the same position (or come within a threshold distance). Something happens at the collision point.

- Two traveling pulses approach from opposite ends of a branch. At the meeting point: a bright flash that scatters smaller particles in both directions. On the diamond, pulses travel up `up1` and `down` simultaneously, colliding at the apex. On the tree, a pulse from the base meets a pulse reflected from the tip.
- **Variants**: merge (two become one), bounce (reverse direction, exchange momentum), annihilate (both vanish in a flash), spawn children (collision produces smaller entities), exchange color.
- **Audio roles**: EVENT triggers spawn of colliding pairs. INTENSITY scales the collision outcome (soft = merge, hard = annihilate with flash). RATE determines entity speed, therefore collision frequency. SURPRISE can trigger annihilation on unexpected transients.
- **Topology**: Linear = straightforward 1D intersection. Diamond = collision at apex/base is geometrically meaningful. Tree = collision at trunk where all branches converge.

#### Interaction Type 2: Attraction / Repulsion

Entities exert forces on each other at a distance. Gravitational pull, magnetic repulsion, flocking behaviors.

- Glowing orbs drift along the strip. When close, they pull toward each other or push apart. With balanced forces: orbiting pairs, stable clusters. On the tree, orbs on different branches converge at the trunk — creating a gravitational well at the base.
- **Flocking/boids variant**: entities align velocity with neighbors (cohesion), avoid crowding (separation), steer toward center of mass (alignment). In 1D this produces organic clumping and dispersal.
- **Audio roles**: MOOD shifts force polarity (warm = attraction, percussive = repulsion). INTENSITY scales force magnitude. DENSITY determines entity count. TRAJECTORY shifts from attraction-dominated (builds) to repulsion-dominated (breakdowns).
- **Topology**: Tree is ideal — trunk as natural attractor. Diamond has gravity wells at vertices.

#### Interaction Type 3: Communication (Signal Propagation)

One entity signals another. The signal propagates through the sculpture's physical structure, activating neighbors in sequence. Chain reactions, cascades, bioluminescence waves.

- A single LED flashes. After a delay proportional to distance, neighbors flash in response, creating an expanding wave. On the tree, a signal at one branch tip cascades down to the trunk, then up all other branches — the sculpture's topology becomes the communication network.
- **Audio roles**: EVENT triggers the initial signal. RATE controls propagation speed (tempo-locked = signal arrives at tips on the next beat). TEXTURE modulates signal fidelity (clean = crisp cascade, noisy = stuttering propagation). SURPRISE triggers chain reactions on unexpected events.
- **Topology**: Tree is the strongest — hub-and-spoke means every signal passes through the trunk. Diamond's asymmetric branch lengths (42/20/11 LEDs) create natural rhythm in the cascade.

#### Interaction Type 4: Competition (Territory)

Entities compete for pixels. Each entity "owns" a region of the strip. Territories expand, contract, and push against each other based on relative strength.

- Two colors start at opposite ends. Each grows toward the other. Where they meet, a contested boundary forms — shifting based on which entity is stronger. The boundary shimmers, sparks, or produces interference patterns.
- **Audio roles**: INTENSITY determines territorial strength per entity (bass-entity grows when bass is high, treble-entity when treble dominates). SPACE maps entities to frequency bands. TRAJECTORY drives long-term territorial shifts. NOVELTY triggers territorial disruption at section boundaries.
- **Topology**: Tree = competition between base (bass territory) and tips (treble territory) mirrors frequency-to-position correspondence.

#### Interaction Type 5: Symbiosis (Mutual Enhancement)

Entities that enhance each other when close. Color blending, resonance amplification, constructive interference.

- Two dim entities drift along the strip. When far apart, barely visible. As they approach, both brighten — their combined presence creates something more than either alone. Colors blend into a third color neither produces on its own. On the tree, entities on adjacent branches resonate through the trunk.
- **Audio roles**: MOOD determines compatibility (harmonic sections increase symbiotic coupling, percussive sections decrease it). INTENSITY amplifies the enhancement. RATE can sync entity oscillation frequencies. TEXTURE modulates blend quality.
- **Topology**: Works everywhere. Diamond branch junctions are natural symbiosis points. Tree trunk amplifies when all branches have active entities near the base.

#### Interaction Type 6: Predator / Prey

One entity class chases another. Prey flees; predators pursue. If caught, prey is consumed (vanishes, predator grows/speeds up/changes color).

- Small bright particles (prey) scatter along the strip. Larger, slower entities (predators) track toward the nearest prey. On the tree, prey flee down branches through the trunk to escape to other branches. On the diamond, vertices are dead ends — prey cornered at the apex has nowhere to go.
- **Audio roles**: RATE controls predator speed (faster music = faster predators). DENSITY determines prey spawn rate. EVENT triggers predator spawn on strong transients. SURPRISE triggers role reversal.
- **Topology**: Tree is ideal — hub-and-spoke creates chokepoints, escape routes, natural drama.

#### Interaction Type 7: Synchronization

Entities that independently oscillate, but gradually phase-lock when near each other. Firefly synchronization, metronome entrainment.

- Multiple "fireflies" pulse independently at slightly different rates. When two are close, they nudge each other's phase. Over seconds, nearby clusters synchronize — pulsing in unison. On the tree, each branch develops its own rhythm, but the trunk region forces synchronization. The Kuramoto model: N oscillators with natural frequencies, coupled by sin(phase difference). The transition from disorder to order is visually dramatic and musically meaningful.
- **Audio roles**: RATE sets natural frequency (near detected tempo, with slight random offsets). INTENSITY scales coupling strength (louder = faster synchronization). TRAJECTORY drives the order/disorder transition (builds increase coupling, breakdowns decrease it). NOVELTY triggers phase resets at section boundaries.
- **Topology**: Tree creates a natural hierarchy — tips weakly coupled (independent), trunk strongly coupled (synchronized). A visual gradient from synchronized center to chaotic periphery.

#### Interaction Type 8: Inheritance (Spawn with Variation)

Parent entities produce child entities with modified properties. Mutation, evolution, generational change.

- A large, bright entity reaches a branch tip and splits into two smaller entities. Each child inherits parent color with a slight hue shift, velocity with random perturbation. Over generations, the sculpture's color palette emerges from lineage, not from a preset. On the tree, the physical branching structure mirrors biological branching — trunk = root organism, branches = generations.
- **Audio roles**: INTENSITY drives growth rate (louder = more frequent splits). MOOD determines mutation magnitude (harmonic = similar offspring, dissonant = divergent). TRAJECTORY influences population growth (builds) vs. die-off (breakdowns).
- **Topology**: Tree is perfect — the sculpture's topology IS the family tree.

#### Interaction Type 9: Decay / Growth / Lifecycle

Entities age. Born, grow, mature, weaken, die. Neighboring entities influence lifecycle — crowding accelerates death, isolation prevents reproduction.

- Entities appear as dim sparks, grow brighter, plateau, then fade. Overcrowding dims faster. Empty regions spawn new entities (ecological succession). On the tree, the trunk is "old growth" (long-lived, slow), tips are "new growth" (born bright, live fast, die young).
- **Seasons variant**: Spring = rapid spawning, bright greens. Summer = dense population, full brightness. Autumn = die-off, warm colors. Winter = sparse, dim, slow. Driven by rolling integral (song-level energy arc).
- **Audio roles**: TRAJECTORY drives the seasonal arc. DENSITY is both input and output. INTENSITY modulates lifespan (quiet = long-lived, loud = ephemeral). TEXTURE influences vitality (smooth harmonic content keeps entities healthy, noise/distortion accelerates aging).
- **Topology**: Diamond = height as age metaphor. Tree = old growth at trunk, new growth at tips.

#### Interaction Matrix: Topology Suitability

| Interaction | Linear Strip | Diamond (height) | Tree (hub-spoke) | Primary Audio Roles |
|---|---|---|---|---|
| **Collision** | Good (1D intersection) | Great (vertex collisions) | Great (trunk convergence) | EVENT, INTENSITY |
| **Attraction/Repulsion** | Good (1D clustering) | Good (gravity wells at vertices) | Great (trunk as attractor) | MOOD, INTENSITY |
| **Communication** | Fair (no branching) | Good (asymmetric paths) | Excellent (nervous system) | EVENT, RATE |
| **Competition** | Good (binary territory) | Good (multi-branch contest) | Great (frequency-mapped zones) | INTENSITY, SPACE |
| **Symbiosis** | Good (overlap point) | Great (vertex amplification) | Great (trunk resonance) | MOOD, TEXTURE |
| **Predator/Prey** | Fair (no escape routes) | Good (vertex traps) | Excellent (branch escape) | RATE, DENSITY |
| **Synchronization** | Good (distance coupling) | Good (height gradient) | Excellent (hierarchy) | INTENSITY, TRAJECTORY |
| **Inheritance** | Fair (binary split) | Good (junction splits) | Excellent (mirrors biology) | INTENSITY, MOOD |
| **Decay/Growth** | Good (local dynamics) | Good (height = age axis) | Great (trunk = old growth) | TRAJECTORY, TEXTURE |

#### Implementation Complexity

| Interaction | State per Entity | Pairwise Computation | ESP32 Feasibility |
|---|---|---|---|
| **Collision** | position, velocity | O(n) with spatial hash | Feasible (<30 entities) |
| **Attraction/Repulsion** | position, velocity, mass | O(n^2) naive, O(n) with cutoff | Feasible (<20 entities) |
| **Communication** | position, signal state | O(n) with propagation queue | Feasible |
| **Competition** | territory boundaries | O(pixels) per frame | Feasible (it's just a 1D diffusion) |
| **Symbiosis** | position, phase, color | O(n) with neighbor scan | Feasible |
| **Predator/Prey** | position, velocity, type | O(n*m) predators * prey | Feasible (<15+15 entities) |
| **Synchronization** | phase, natural frequency | O(n^2) for full Kuramoto, O(n) with spatial cutoff | Feasible (<30 oscillators) |
| **Inheritance** | all parent properties | O(1) per spawn event | Feasible |
| **Decay/Growth** | age, health, neighbors | O(n) per frame | Feasible |

#### Entity Interaction Cheat Sheet

For quick reference when designing a new effect with multiple entities:

```
What happens when two entities MEET?
  -> Collision: merge / bounce / annihilate / spawn / exchange

What happens when two entities are NEAR?
  -> Attraction / repulsion: approach / separate / orbit
  -> Symbiosis: mutual brightening, color blend, resonance
  -> Synchronization: phase-lock oscillations
  -> Communication: signal propagation, chain reaction

What happens when entities COMPETE for space?
  -> Territory: expanding/contracting boundaries
  -> Predator/prey: chase, consume, population dynamics
  -> Crowding: accelerated decay, resource depletion

What happens over an entity's LIFETIME?
  -> Growth: born small, grow to full size
  -> Decay: age, fade, weaken
  -> Inheritance: spawn children with variation
  -> Lifecycle: birth -> growth -> maturity -> decay -> death -> rebirth

What AUDIO drives the interaction?
  -> EVENT / SURPRISE: triggers discrete interactions (spawn, collide, die)
  -> RATE / INTENSITY: modulates continuous dynamics (speed, force, strength)
  -> MOOD / TEXTURE: shifts qualitative character (attraction vs repulsion, harmony vs chaos)
  -> TRAJECTORY: drives long-arc transitions (synchronization <-> disorder, growth <-> decay)
  -> DENSITY / SPACE: determines population and spatial distribution
```

---

### Priority Effects (User Taste Signals)

Based on stated preferences — likes effects that take the physical form, likes sparkles-on-beat, likes tree sap flow:

**Tier 1 -- Build These First:**
1. **Sap Flow + Beat Sparkle** — growth effect with sparkle overlay on transients (composite)
2. **Perlin Noise Field** — organic evolving background, parameters from audio
3. **Bioluminescence** — slow independent glows, pulsing with music
4. **Root Pulse** — bass-triggered energy radiating from trunk to tips

**Tier 2 -- High Potential:**
5. **Fire / Flame** — classic, topology-native on tree, well-documented (FastLED Fire2012)
6. **Metaballs / Blobs** — merging blobs feel organic, sound-reactive blob count
7. **Reaction-Diffusion** — emergent patterns, never repeats
8. **Electrical Storm** — dramatic, good for drops and climactic moments

**Tier 3 -- Interesting Experiments:**
9. **Feedback / Echo** — temporal smearing reveals rhythm structure
10. **Gravity Drops** — onset spawns drops, physical feel
11. **Frost Crystallization** — great for section transitions (slow reveal)
12. **Heartbeat** — double-pulse at bass rate, physiological feel
13. **Firefly Synchronization** — Kuramoto oscillators, builds create order from chaos

---

## Part 2: Input Roles — Decoupling Effects from Audio Signals

### The Problem

Every effect in Part 1 has an "Audio Hook" or "Audio-Reactive Hook" column that lists specific audio features: "speed from tempo," "density from spectral flatness," "triggered by onset." This creates a tight coupling between effect code and audio analysis code. If you want to drive the same effect from a MIDI controller, a machine learning model, or a different sensor, you have to rewrite the effect's parameter mapping.

### The Solution: Input Roles

An **input role** describes *what kind of influence* a parameter has on an effect, not *which specific signal* fills it. The same role can be filled by different sources:

- A **RATE** role might be filled by detected tempo, a user knob, a MIDI clock, or an ML model's estimate of "groove speed."
- An **EVENT** role might be filled by onset detection, beat prediction, a button press, or a motion sensor trigger.
- A **MOOD** role might be filled by spectral centroid, a genre classifier, a user's palette selection, or time of day.

This separation enables swapping signal sources without changing effect code. The effect declares "I need an EVENT, a RATE, and an INTENSITY." The show controller binds those roles to whatever sources are available.

### The 13 Input Roles

| Role | What It Controls | Timescale | Example Sources |
|---|---|---|---|
| **EVENT** | Triggers a discrete action (spawn, collide, die, flash) | Instantaneous | Onset detection, beat prediction, transient peaks, button press, MIDI note-on |
| **RATE** | Controls speed or frequency of continuous behavior | Frame-level | Tempo, onset density, spectral flux rate, user knob, MIDI clock |
| **INTENSITY** | Scales magnitude of behavior (brightness, force, size) | Frame-level | RMS, absint, per-band energy, user fader, envelope follower |
| **MOOD** | Shifts qualitative character (warm/cool, tense/relaxed, major/minor) | Slow-moving | Spectral centroid, dominant band, harmonic ratio, genre classifier, user palette selection |
| **TRAJECTORY** | Drives directional change over time (rising, falling, stable) | Phrase-level | Rolling integral slope, RMS derivative sign, section detector state |
| **TEXTURE** | Modulates roughness, density, or granularity | Frame-level | Spectral flatness, onset density, ZCR, noisiness estimate |
| **NOVELTY** | Responds to deviation from recent context | Phrase-level | MFCC distance from running mean, spectral flux variance, section boundary confidence |
| **DENSITY** | Controls how many entities or events coexist | Beat-level | Per-band onset count, spectral complexity, polyphony estimate |
| **SPACE** | Maps to spatial position or spread | Frame-level | Per-band energy (frequency-to-position), stereo field, user spatial control |
| **POSITION** | Selects a specific point or region on the sculpture | Frame-level | Frequency-to-height mapping, spatial centroid, user pointer, random walk |
| **FAMILIARITY** | Responds to how well-known the current musical material is | Song-level | Repetition detector, chorus vs. verse confidence, self-similarity score |
| **SURPRISE** | Reacts to unexpected events (dropout, reintroduction, sudden change) | Instantaneous | Per-band dropout detection, RMS step function, deviation from predicted beat |
| **PHASE** | Provides a cyclic position within a repeating pattern (0.0-1.0) | Beat-level | Beat phase (position within current beat), bar phase, LFO, breathing cycle |

#### Timescale Variants

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

### Role Composition Patterns

Effects rarely use a single role. Common patterns emerge in how roles combine:

**Pattern A: Event + Intensity (Trigger-and-Scale)**
The most common pattern. An EVENT triggers a discrete action; INTENSITY scales how dramatic it is. Examples: beat-triggered sparkle (EVENT = onset, INTENSITY = onset strength), bass-triggered root pulse (EVENT = bass onset, INTENSITY = bass energy).

**Pattern B: Rate + Mood (Continuous Character)**
RATE controls the speed of ongoing behavior; MOOD shifts its character. Examples: noise field (RATE = tempo drives evolution speed, MOOD = centroid shifts palette), bioluminescence (RATE = tempo sets pulse frequency, MOOD = harmonic ratio sets warmth).

**Pattern C: Trajectory + Density (Arc-and-Population)**
TRAJECTORY drives the long-term direction; DENSITY determines how much is happening. Examples: lifecycle/seasons (TRAJECTORY = build/breakdown arc, DENSITY = entity count rises and falls), frost crystallization (TRAJECTORY = section build drives spread, DENSITY = crystal count).

**Pattern D: Event + Surprise + Novelty (Structural Response)**
The effect responds to musical structure. EVENT handles regular triggers, SURPRISE handles unexpected ones, NOVELTY detects context shifts. Examples: automated show transitions (NOVELTY = section boundary triggers effect change, SURPRISE = dropout triggers special reaction, EVENT = beats maintain groove).

**Pattern E: Space + Position + Phase (Spatial Animation)**
The effect creates spatially-distributed animation. SPACE determines spread, POSITION selects focus, PHASE drives cyclic motion. Examples: zone coloring (SPACE = frequency-to-zone mapping, POSITION = active zone, PHASE = rotation within zone), directional wave (POSITION = wavefront location, PHASE = beat phase for timing, SPACE = wave width).

**Pattern F: Intensity + Texture + Mood (Ambient Character)**
The effect is a continuous ambient texture with no discrete events. All three modulate the character simultaneously. Examples: candle flicker (INTENSITY = brightness, TEXTURE = flicker roughness, MOOD = warmth), breathing organism (INTENSITY = breath depth, TEXTURE = smoothness, MOOD = color temperature).

### The Role Matrix

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

Reading the matrix: Sap Flow's only primary role is TRAJECTORY — the effect is fundamentally about directional change over time. Root Pulse needs EVENT (when to fire), RATE (propagation speed), and INTENSITY (how bright). This tells you exactly what to bind when integrating a new signal source.

---

## Part 3: Addressing vs. Rendering — The Spatial Dimension

### The Three Coordinate Systems

The sculptures have three distinct spatial systems, and confusing them leads to wrong design decisions:

| System | Dimension | What It Means | When to Use It |
|---|---|---|---|
| **Index space** | 1D | `led[i]` — the array offset in the data buffer | Serial output, memory layout, simple effects that don't need spatial awareness |
| **Topology space** | 1D per branch + height | Position along a branch (0 = base, 1 = tip), branch identity, and shared height axis. This is what `apply_topology()` currently provides via `sculptures.json` | Branch-aware effects (sap flow, capillary fill), height-mapped effects (stratification, fire, gravity drops). **Already implemented.** |
| **World space** | 3D | (x, y, z) physical position of each LED in centimeters | Noise fields, wave propagation, ripples, any effect that should look coherent from multiple viewing angles. **Not yet implemented.** |

**What we already have**: Height mapping and branch identity from `sculptures.json`. Effects like Fire, Tide/Water Level, Stratification, Sap Flow, and Gravity Drops already use the vertical axis correctly — they render based on height, not raw strip index.

**What we are missing**: The **lateral dimension** — x,y positions within each branch. Two LEDs at the same height but on different branches have no spatial relationship beyond "same height, different branch." This means effects that need radial distance, angular position, or true 3D proximity cannot be computed.

### What 3D Coordinates Would Unlock

Once each LED has an (x, y, z) position, effects can render in world space. The result is correct from every viewing angle — the sculpture becomes a volumetric display. Two people standing at different positions see different 2D projections of the same 3D light field. This is impossible on a screen and impossible on a flat strip.

Effects that are currently rated as needing "1D adaptation" would work natively with 3D coordinates:

| Effect | Current State (topology space) | With 3D Coordinates | What Changes |
|---|---|---|---|
| **Perlin Noise Field** | Works well; samples noise along branch position | Excellent — sample noise(x, y, z, t) per LED | Noise field is spatially coherent in 3D. Viewed from any angle: smooth, organic. No "strip" visible. |
| **Plasma** | Works; sine waves along branch position | Excellent — sine(x)*sin(y)*sin(z+t) per LED | Color waves propagate through 3D space. Sculpture reveals cross-sections of a 3D plasma field. |
| **Voronoi** | Segments along branches | Excellent — 3D Voronoi cells with LED positions | Organic 3D regions visible from any angle. Cell boundaries follow the sculpture's form, not its wiring order. |
| **Reaction-Diffusion** | 1D activator-inhibitor per branch | Better — diffusion follows 3D adjacency | Turing patterns form on the sculpture's surface, respecting physical proximity. Two LEDs physically adjacent but on different branches correctly diffuse between each other. |
| **Metaballs** | Gaussian blobs along branches | Excellent — 3D distance fields | Blobs are spherical in 3D. From any angle, smooth merging organic shapes. |
| **Ripple** | Expands along branches from a point | Excellent — concentric 3D spheres from trigger point | All LEDs at the same distance from trigger flash simultaneously, regardless of branch. |
| **Gravity Drops** | Already uses height mapping correctly | Better — particles follow true gravity vector | On a branch that curves sideways, drops follow the physical downward direction, not the wiring direction. |

### Effects That Require 3D Coordinates

These effects are meaningless on a flat strip but become powerful on a 3D sculpture:

| Effect | Why It Needs 3D | Audio Role |
|---|---|---|
| **Directional wave** | A plane wave sweeps through 3D space (left-to-right wash). LEDs activate as the wavefront passes their position. On a 3D sculpture, the wavefront reveals the sculpture's shape. | EVENT triggers wave, RATE controls speed, MOOD or stereo field sets direction |
| **Spherical expansion** | An event at a 3D point creates an expanding sphere of light. All LEDs at the same radius activate simultaneously, regardless of branch. | EVENT triggers expansion, RATE controls speed |
| **3D noise cross-section** | The sculpture "slices" through a 4D noise field (x, y, z, t). Each LED samples noise at its world position. | RATE controls noise evolution, MOOD shifts scale/octaves |
| **Laterally-aware effects** | Effects that distinguish "left branch vs right branch" spatially, not just by branch ID. A directional wash sweeping left-to-right across the tree. | MOOD or stereo field maps to lateral axis |
| **Shadow / occlusion** | A "light source" illuminates the sculpture from a direction. LEDs facing the light are bright; LEDs behind are dark. Move the light source with audio. | MOOD positions the light source, INTENSITY controls brightness, RATE rotates |
| **Depth layering** | Near-viewer LEDs render differently from far-viewer LEDs. Creates a sense of depth. | SPACE maps depth to visual layer |

### Interactions in World Space

The multi-entity interactions from Category 5 become richer with 3D coordinates:

| Interaction | In Topology Space | In World Space |
|---|---|---|
| **Collision** | Entities collide when at the same branch position. Two entities on different branches can never collide. | Entities collide within physical proximity. Two entities on adjacent branches at the same height can collide across branches. |
| **Attraction** | Force proportional to branch-relative distance. Entities on different branches only interact through the trunk. | Force proportional to physical distance. Adjacent LEDs in world space interact directly, creating cross-branch dynamics. |
| **Communication** | Signal propagates along branch paths, must traverse trunk to reach other branches. | Signal propagates through 3D space. Adjacent LEDs communicate directly regardless of wiring. |
| **Competition** | Territory boundaries are branch-relative ranges. Cannot span across branches. | Territory boundaries are 3D surfaces. A frequency band can "own" a spatial region spanning multiple branches. |
| **Synchronization** | Coupling based on branch-relative distance. Entities on separate branches couple only through the trunk. | Coupling based on physical distance. Entities cluster and synchronize based on spatial proximity. |

### Per-Sculpture Spatial Analysis

From `sculptures.json`, what spatial structure each topology provides:

**Flat Strip (cob_flat, 80 LEDs)**: Linear. One branch. World space = index space. 3D coordinates add nothing.

**Diamond (cob_diamond, 73 LEDs)**: Three branches meeting at vertices. `up1` (42 LEDs), `down` (20 LEDs, reversed), `up2` (11 LEDs, gamma=0.55). Planar 2D at minimum (width and height). The height-mapping mode already exploits the vertical axis. The gamma correction on `up2` compensates for non-linear physical spacing — proof that the physical geometry differs from the addressing order. Key insight: the three branches have very different LED counts (42/20/11). In index space, `up2` gets 15% of the pixels. In world space, it might occupy 30% of the visible area. Effects in world space would naturally give `up2` proportional visual weight.

**150 LED Strip (strip_150)**: Linear, single branch. Same as flat strip. No benefit from 3D.

**Tree (tree_flat, 197 LEDs)**: Three physical strips (pins 27/15/14) arranged as a tree with trunk, major limbs, and sub-branches. Effects already use height as the primary spatial index, not raw strip order. Height-based effects (sap flow, gravity, stratification) work correctly. Branch identity is known. What is missing: the lateral dimension — x,y positions within each branch. Two LEDs at the same height on different branches have no spatial relationship beyond "same height, different branch." 3D coordinates would unlock noise fields with smooth organic patterns across the tree's canopy, ripples that hit nearby LEDs on adjacent branches simultaneously, and directional washes that reveal the sculpture's shape.

### Revised Topology Fit Assessment

The "Topology Fit" column throughout Part 1 can be read more precisely with the three coordinate systems in mind:

| Original Assessment | What It Means |
|---|---|
| "All linear" | Works in index space on any strip. No spatial awareness needed. |
| "Linear, tree (base->tip)" | Works in topology space (branch-relative position). |
| "Height-mapped" | Works in topology space with height mode. Already implemented. Benefits from 3D (true vertical position rather than branch-relative). |
| "Hub-and-spoke" | Requires branch-junction awareness. Benefits strongly from 3D (convergence point has a physical position). |
| "Excellent with 3D coords" | Works in topology space but significantly better with world-space rendering. |

### Practical Path: Getting 3D Coordinates

Adding 3D coordinates to `sculptures.json` doesn't require CAD modeling. It requires physically measuring (or estimating) each LED's position:

1. **Photograph the sculpture** from front and side with a ruler for scale.
2. **Annotate key LED positions** (start of each branch, tips, midpoints) in the photos.
3. **Interpolate** between annotated positions along each branch path.
4. **Store as** `"positions": [[x,y,z], ...]` in `sculptures.json`.

For the tree (197 LEDs), this could be as simple as photographing from two orthogonal angles, marking branch paths in pixel coordinates, converting to centimeters using the ruler reference, and interpolating LED positions along each path at regular spacing. Even approximate coordinates (within 1-2cm) would be sufficient for noise fields, ripples, and gradient effects.

---

## Part 4: Automated Show Architecture

### What VJs Actually Do

VJs make three categories of real-time decisions:

1. **Effect Selection** — which visual runs right now (based on musical section)
2. **Parameter Modulation** — tweaking intensity, speed, color, complexity (continuous)
3. **Transitions** — crossfading between states at musical boundaries (section changes)

The key insight: VJs **prepare** the structure (asset library, color palettes, energy arc) and **improvise** the details (which exact effect, which parameter tweaks, crowd response).

For automated LED sculpture shows, the analogy is:
- **Prepare**: effect library, palette sets, section->effect mappings, energy curves
- **Automate**: real-time section detection, parameter modulation from audio, transition logic
- **Manual override**: human can intervene via controller for taste decisions

### Detection Stack

What needs to be detected, in order of feasibility and impact:

#### Layer 1: Frame-Level (Already Have)

| Detection | Status | Source |
|---|---|---|
| RMS / loudness | Implemented | `signals.py`, `feature_computer.py` |
| Onset / transient | Implemented | Per-band spectral flux |
| Spectral centroid | Implemented | `feature_computer.py` |
| Per-band energy | Implemented | 5-band mel decomposition |
| Absint (rate of change) | Implemented | `signals.py:AbsIntegral` |

#### Layer 2: Beat-Level (Partially Have)

| Detection | Status | Gap |
|---|---|---|
| Tempo (BPM) | Implemented | Octave-ambiguous |
| Beat prediction | Implemented | `signals.py:BeatPredictor` |
| Onset envelope | Implemented | `signals.py:OnsetTempoTracker` |
| Downbeat detection | Partial | Every-4th heuristic, not true downbeat |
| Vocal presence | Not implemented | Could use spectral shape heuristics or ML |

#### Layer 3: Phrase-Level (The Critical Gap)

This is where automated show control lives — and it's largely unimplemented.

| Detection | Description | Approach | Difficulty |
|---|---|---|---|
| **Build detection** | Rising energy over 4-16 bars | Rolling integral slope > threshold for N seconds | Medium — rolling integral exists but untested |
| **Drop detection** | Sudden sustained intensity after build | RMS jump after build phase, bass energy spike | Medium — combine build detection + RMS derivative |
| **Breakdown detection** | Energy withdrawal, sparser texture | Rolling integral slope < 0, spectral flatness drops | Medium |
| **Section boundary** | Any structural change | Self-similarity novelty, spectral flux over 2-4s window | Hard — Foote's method is O(n^2), needs streaming alternative |
| **Riser detection** | Sweeping frequency buildup | Rising spectral centroid + increasing RMS over 2-8s | Medium |
| **Dropout detection** | Band goes silent then returns | Per-band energy drops to near-zero, reintroduction | Easy — per-band normalization already handles this |

**Priority**: Build detection -> drop detection -> breakdown -> section boundary. These four cover 80% of automated show control decisions.

**Streaming Section Detection Alternatives** (Foote's is non-streaming):
- Rolling integral slope: positive = build, negative = breakdown, near-zero = sustained
- Per-band onset density: sparse onsets = breakdown, dense = groove
- Spectral flux variance: high variance = transition, low = stable section
- MFCC distance from running mean: sudden jump = section boundary

#### Layer 4: Song-Level (Future / Optional)

| Detection | Description | Use Case | Approach |
|---|---|---|---|
| **Song transition** | One track fading into another (DJ context) | Reset adaptation, change palette | Detect simultaneous energy in different keys, or BPM instability |
| **Genre/mood shift** | Overall energy character changes | Select effect family | Sliding window over spectral shape, onset density, tempo stability |
| **Energy arc** | Set-wide intensity trajectory | Prevent visual fatigue, manage intensity over hours | Very long rolling average (minutes), trend detection |
| **Key detection** | Musical key of current section | Color palette selection (major=warm, minor=cool) | Chroma analysis — feasible but low priority |

### Show Controller Architecture

```
                     +-----------------------------+
                     |      Show Controller         |
                     |                               |
  Audio Features --> |  Section Detector             |
  (frame-level)      |    +-- Build/Drop/Break state  |
                     |    +-- Section boundaries       |
                     |                               |
                     |  Role Binder                   | <-- binds input roles to sources
                     |    +-- Audio feature -> role    |
                     |    +-- User knob -> role         |
                     |    +-- ML model -> role           |
                     |                               |
                     |  Effect Selector               | --> Active Effect
                     |    +-- Section -> effect map     |
                     |    +-- Transition logic          |
                     |    +-- Fatigue prevention        |
                     |                               |
                     |  Parameter Modulator           | --> Effect params (via roles)
                     |    +-- Role -> visual mapping   |
                     |    +-- Energy curve tracking     |
                     |                               |
                     |  Palette Manager               | --> Color palette
                     |    +-- Mood -> palette map        |
                     |    +-- Rotation timer             |
                     |                               |
                     +-----------------------------+
```

### Section -> Effect Mapping (Starting Point)

Not prescriptive — this is a default mapping that a human would override for taste. The idea is that it works *reasonably* without intervention.

| Detected Section | Effect Family | Intensity | Palette Temperature | Speed |
|---|---|---|---|---|
| **Intro / Low Energy** | Ambient (noise field, bioluminescence, breathing) | Low | Warm/cool neutral | Slow |
| **Verse / Groove** | Groove (tempo-locked pulse, flowing noise) | Medium | Warm | Medium, locked to tempo |
| **Build** | Convergence + rising complexity | Rising | Shifts warmer | Accelerating |
| **Drop** | Full-intensity (root pulse, fire, electrical storm) | High, sustained | Hot (reds, whites) | Fast |
| **Breakdown** | Ambient + sparkle (embers, fireflies, bioluminescence) | Low-medium, falling | Cool (blues, purples) | Slow, decoupled from tempo |
| **Climax / Peak** | Maximum complexity (composite: background + sparkle + pulse) | Maximum | White-hot | Maximum |
| **Outro / Fade** | Retreat (reverse growth, cooling embers, fading noise) | Decreasing to zero | Cooling | Decelerating |

### Preventing Visual Fatigue

For multi-hour automated shows, the system needs strategies to stay interesting:

1. **Effect Rotation** — don't use the same effect for more than N consecutive sections of the same type. After 3 drops with Fire, switch to Root Pulse or Electrical Storm.
2. **Palette Cycling** — rotate through palette families (warm, cool, monochrome, complementary) over 10-20 minute periods, independent of section detection.
3. **Complexity Curve** — track a long-horizon "visual complexity" metric. If it's been high for too long, force a simpler section. Mirror the concept of dynamic range in mastering.
4. **Spatial Rotation** — alternate between topology modes (full-strip, zoned, height-mapped, branch-independent) to keep the spatial experience fresh.
5. **Novelty Injection** — periodically introduce a rare effect (frost crystallization, reaction-diffusion) that hasn't been used recently. The surprise factor prevents habituation.

### What SoundSwitch / MaestroDMX Claim (and What's Actually True)

Commercial automated lighting systems market phrase detection:

- **SoundSwitch**: Phrase detection (intro/verse/chorus/bridge/drop/outro). Ships with 32 pre-built "Autoloops." But this is **pre-analysis** — tracks must be analyzed before playback. Not real-time.
- **MaestroDMX**: Claims real-time "autonomous lighting designer." The only commercial tool claiming streaming section detection, but algorithm is proprietary and unverifiable. User reviews describe it as "not much more sophisticated than enhanced sound-to-light."
- **Engine Lighting**: "Industry-leading automated phrase detection" — also pre-analysis, not streaming.

**The honest assessment**: Real-time phrase/section detection is **not solved**. Beat/downbeat tracking is solved (BeatNet, BEAST achieve >80% accuracy with <50ms latency). But semantic section detection (verse/chorus/build/drop) relies on self-similarity matrices that require the **full song** — fundamentally non-streaming. No published causal algorithm matches offline methods.

**What exists for real-time**: Simple heuristics — rolling integral slope, kick dropout detection, onset density gradients, spectral centroid derivative. These are crude but causal and fast (<1ms). They won't generalize across genres without tuning, and even for our use case they will likely need significant R&D to feel right. This is an active research area for this project, not a solved problem we can import.

**Song-level energy arc** (10+ minute trajectory) is actually easier — long timescales are forgiving, and multi-scale rolling averages work well. This may be more important than phrase-level detection for our use case.

The "two quality axes" framework still applies: detection quality matters, but mapping quality is what people see. We should improve both in parallel.

### Implementation Phases

**Phase 1: Streaming Section Detector**
- Implement rolling integral slope as build/drop/breakdown detector
- Add per-band dropout detection (already have normalization)
- Expose detected section as state in runner.py
- Test against annotated audio segments

**Phase 2: Effect Library Expansion**
- Implement 4-6 new effects from the Tier 1/2 priority list
- Each effect declares its input roles (not specific audio features)
- Build as composable layers (background + foreground pattern)

**Phase 3: Show Controller with Role Binding**
- Section -> effect mapping with configurable defaults
- Role binder: maps audio features, user knobs, and other sources to effect input roles
- Transition logic (crossfade, cut, morph between effects)
- Palette manager with rotation
- Fatigue prevention heuristics

**Phase 4: Spatial Rendering**
- Add 3D coordinates to `sculptures.json` for tree and diamond
- Implement world-space rendering path in effect base class
- Upgrade noise field, ripple, and metaball effects to use 3D coordinates
- Effects fall back to topology space when 3D coordinates are unavailable

**Phase 5: Multi-Hour Autonomy**
- Energy arc tracking over minutes
- Effect rotation memory (don't repeat too soon)
- Long-horizon adaptation (overall set intensity curve)
- Manual override via controller input

---

## Appendix: WLED Sound Reactive Effect Catalog

For reference — the complete WLED-SR audio-reactive effects that exist in the ecosystem:

**Volume-Reactive**: 2D Swirl, 2D Waverly, Gravcenter, Gravcentric, Gravimeter, Juggles, Matripix, Midnoise, Noisefire, Noisemeter, Pixels, Pixelwave, Plasmoid, Puddlepeak, Puddles, Ripple Peak, Waterfall

**FFT-Reactive**: 2D CenterBars, 2D Funky Plank, 2D GEQ, Binmap, Blurz, DJLight, Freqmap, Freqmatrix, Freqpixels, Freqwave, Gravfreq, Noisemove, Rocktaves

**Notable non-reactive WLED effects worth adapting**: Fire 2012, Pride 2015, Pacifica, Multi Comet, Chase, Scan, Rain, Fireworks, Meteor, Ripple, Twinkle, Candle

**FastLED community effects**: Fire2012, Pride, Pacifica, ColorWaves, JugglePal, Sinelon, Confetti, BPM, Dot Beat, Rainbow March

**Hyperion effects**: Mood Blobs (multiple color variants), Rainbow Mood/Swirl, Snake, Candle, Police, Strobe
