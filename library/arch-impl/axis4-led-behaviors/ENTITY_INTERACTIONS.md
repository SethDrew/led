# Entity Interactions Reference

Catalog of multi-entity interaction types for LED sculpture effects. Categories 1-4 in the visual idioms taxonomy catalog individual effects (a comet, a sparkle, a noise field). But the most visually compelling moments happen when *multiple entities interact*. Two comets colliding. Fireflies synchronizing. Organisms competing for territory. These interactions are not effects in themselves -- they are *relationships between effects* (or between instances of the same effect).

This is the entity-interaction axis: given N entities on a sculpture, what rules govern their relationships? The interaction types below are organized from simple (pairwise, local) to complex (emergent, global).

---

## Interaction Type 1: Collision

Two entities occupy the same position (or come within a threshold distance). Something happens at the collision point.

- Two traveling pulses approach from opposite ends of a branch. At the meeting point: a bright flash that scatters smaller particles in both directions. On the diamond, pulses travel up `up1` and `down` simultaneously, colliding at the apex. On the tree, a pulse from the base meets a pulse reflected from the tip.
- **Variants**: merge (two become one), bounce (reverse direction, exchange momentum), annihilate (both vanish in a flash), spawn children (collision produces smaller entities), exchange color.
- **Audio roles**: EVENT triggers spawn of colliding pairs. INTENSITY scales the collision outcome (soft = merge, hard = annihilate with flash). RATE determines entity speed, therefore collision frequency. SURPRISE can trigger annihilation on unexpected transients.
- **Topology**: Linear = straightforward 1D intersection. Diamond = collision at apex/base is geometrically meaningful. Tree = collision at trunk where all branches converge.

## Interaction Type 2: Attraction / Repulsion

Entities exert forces on each other at a distance. Gravitational pull, magnetic repulsion, flocking behaviors.

- Glowing orbs drift along the strip. When close, they pull toward each other or push apart. With balanced forces: orbiting pairs, stable clusters. On the tree, orbs on different branches converge at the trunk -- creating a gravitational well at the base.
- **Flocking/boids variant**: entities align velocity with neighbors (cohesion), avoid crowding (separation), steer toward center of mass (alignment). In 1D this produces organic clumping and dispersal.
- **Audio roles**: MOOD shifts force polarity (warm = attraction, percussive = repulsion). INTENSITY scales force magnitude. DENSITY determines entity count. TRAJECTORY shifts from attraction-dominated (builds) to repulsion-dominated (breakdowns).
- **Topology**: Tree is ideal -- trunk as natural attractor. Diamond has gravity wells at vertices.

## Interaction Type 3: Communication (Signal Propagation)

One entity signals another. The signal propagates through the sculpture's physical structure, activating neighbors in sequence. Chain reactions, cascades, bioluminescence waves.

- A single LED flashes. After a delay proportional to distance, neighbors flash in response, creating an expanding wave. On the tree, a signal at one branch tip cascades down to the trunk, then up all other branches -- the sculpture's topology becomes the communication network.
- **Audio roles**: EVENT triggers the initial signal. RATE controls propagation speed (tempo-locked = signal arrives at tips on the next beat). TEXTURE modulates signal fidelity (clean = crisp cascade, noisy = stuttering propagation). SURPRISE triggers chain reactions on unexpected events.
- **Topology**: Tree is the strongest -- hub-and-spoke means every signal passes through the trunk. Diamond's asymmetric branch lengths (42/20/11 LEDs) create natural rhythm in the cascade.

## Interaction Type 4: Competition (Territory)

Entities compete for pixels. Each entity "owns" a region of the strip. Territories expand, contract, and push against each other based on relative strength.

- Two colors start at opposite ends. Each grows toward the other. Where they meet, a contested boundary forms -- shifting based on which entity is stronger. The boundary shimmers, sparks, or produces interference patterns.
- **Audio roles**: INTENSITY determines territorial strength per entity (bass-entity grows when bass is high, treble-entity when treble dominates). SPACE maps entities to frequency bands. TRAJECTORY drives long-term territorial shifts. NOVELTY triggers territorial disruption at section boundaries.
- **Topology**: Tree = competition between base (bass territory) and tips (treble territory) mirrors frequency-to-position correspondence.

## Interaction Type 5: Symbiosis (Mutual Enhancement)

Entities that enhance each other when close. Color blending, resonance amplification, constructive interference.

- Two dim entities drift along the strip. When far apart, barely visible. As they approach, both brighten -- their combined presence creates something more than either alone. Colors blend into a third color neither produces on its own. On the tree, entities on adjacent branches resonate through the trunk.
- **Audio roles**: MOOD determines compatibility (harmonic sections increase symbiotic coupling, percussive sections decrease it). INTENSITY amplifies the enhancement. RATE can sync entity oscillation frequencies. TEXTURE modulates blend quality.
- **Topology**: Works everywhere. Diamond branch junctions are natural symbiosis points. Tree trunk amplifies when all branches have active entities near the base.

## Interaction Type 6: Predator / Prey

One entity class chases another. Prey flees; predators pursue. If caught, prey is consumed (vanishes, predator grows/speeds up/changes color).

- Small bright particles (prey) scatter along the strip. Larger, slower entities (predators) track toward the nearest prey. On the tree, prey flee down branches through the trunk to escape to other branches. On the diamond, vertices are dead ends -- prey cornered at the apex has nowhere to go.
- **Audio roles**: RATE controls predator speed (faster music = faster predators). DENSITY determines prey spawn rate. EVENT triggers predator spawn on strong transients. SURPRISE triggers role reversal.
- **Topology**: Tree is ideal -- hub-and-spoke creates chokepoints, escape routes, natural drama.

## Interaction Type 7: Synchronization

Entities that independently oscillate, but gradually phase-lock when near each other. Firefly synchronization, metronome entrainment.

- Multiple "fireflies" pulse independently at slightly different rates. When two are close, they nudge each other's phase. Over seconds, nearby clusters synchronize -- pulsing in unison. On the tree, each branch develops its own rhythm, but the trunk region forces synchronization. The Kuramoto model: N oscillators with natural frequencies, coupled by sin(phase difference). The transition from disorder to order is visually dramatic and musically meaningful.
- **Audio roles**: RATE sets natural frequency (near detected tempo, with slight random offsets). INTENSITY scales coupling strength (louder = faster synchronization). TRAJECTORY drives the order/disorder transition (builds increase coupling, breakdowns decrease it). NOVELTY triggers phase resets at section boundaries.
- **Topology**: Tree creates a natural hierarchy -- tips weakly coupled (independent), trunk strongly coupled (synchronized). A visual gradient from synchronized center to chaotic periphery.

## Interaction Type 8: Inheritance (Spawn with Variation)

Parent entities produce child entities with modified properties. Mutation, evolution, generational change.

- A large, bright entity reaches a branch tip and splits into two smaller entities. Each child inherits parent color with a slight hue shift, velocity with random perturbation. Over generations, the sculpture's color palette emerges from lineage, not from a preset. On the tree, the physical branching structure mirrors biological branching -- trunk = root organism, branches = generations.
- **Audio roles**: INTENSITY drives growth rate (louder = more frequent splits). MOOD determines mutation magnitude (harmonic = similar offspring, dissonant = divergent). TRAJECTORY influences population growth (builds) vs. die-off (breakdowns).
- **Topology**: Tree is perfect -- the sculpture's topology IS the family tree.

## Interaction Type 9: Decay / Growth / Lifecycle

Entities age. Born, grow, mature, weaken, die. Neighboring entities influence lifecycle -- crowding accelerates death, isolation prevents reproduction.

- Entities appear as dim sparks, grow brighter, plateau, then fade. Overcrowding dims faster. Empty regions spawn new entities (ecological succession). On the tree, the trunk is "old growth" (long-lived, slow), tips are "new growth" (born bright, live fast, die young).
- **Seasons variant**: Spring = rapid spawning, bright greens. Summer = dense population, full brightness. Autumn = die-off, warm colors. Winter = sparse, dim, slow. Driven by rolling integral (song-level energy arc).
- **Audio roles**: TRAJECTORY drives the seasonal arc. DENSITY is both input and output. INTENSITY modulates lifespan (quiet = long-lived, loud = ephemeral). TEXTURE influences vitality (smooth harmonic content keeps entities healthy, noise/distortion accelerates aging).
- **Topology**: Diamond = height as age metaphor. Tree = old growth at trunk, new growth at tips.

---

## Interaction Matrix: Topology Suitability

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

## Implementation Complexity

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

## Entity Interaction Cheat Sheet

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
