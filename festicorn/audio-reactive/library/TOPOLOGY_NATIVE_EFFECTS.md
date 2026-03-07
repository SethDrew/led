# Topology-Native Effects

Effects that exploit the spatial properties of LED sculptures — height, branching, neighbor connectivity, distance-from-root — rather than treating the strip as a flat array. These effects look fundamentally different on a diamond sculpture vs. a flat strip because they use the geometry.

Companion to ARCHITECTURE.md Axis 1 (Topologies) and Axis 4 (LED Behaviors). Input roles reference the 13-role taxonomy from the project's design space (EVENT, RATE, INTENSITY, MOOD, TRAJECTORY, TEXTURE, NOVELTY, DENSITY, SPACE, POSITION, FAMILIARITY, SURPRISE, PHASE).

---

## Topology Properties Reference

| Property | Description | Available On |
|---|---|---|
| **Height** | Normalized 0 (base) to 1 (apex), from `height_keyframes` | Diamond, tree |
| **Branch** | Named branch membership (left, right, middle) | Diamond, tree |
| **Distance-from-root** | LED index distance from base junction | All |
| **Neighbor connectivity** | Cross-branch physical adjacency (e.g., LEDs 4/63 on diamond) | Diamond |
| **XY position** | 2D coordinates from `xy_keyframes` | Diamond |
| **Landmarks** | Named positions: apex, tips, base_junction, crossover | Diamond |

---

## Nature Simulations

Effects inspired by natural phenomena. These map particularly well to branching sculptures because nature itself is branching — trees, rivers, lightning, root systems. Each entry describes the visual concept, which topology properties it exploits, and which input roles it maps to naturally.

### Fireflies

Random single-pixel twinkles with slow fade-out, scattered across the sculpture. Pixels ignite independently at random positions, glow briefly, then dim. Density and speed vary with audio energy.

**Topology**: Distance-from-root (fireflies cluster near tips or near base depending on frequency), branch (different branches can have independent firefly populations), XY position (spatial clustering).

**Input roles**: DENSITY (how many fireflies are active), RATE (twinkle speed / fade duration), INTENSITY (peak brightness of each firefly), MOOD (warm fireflies vs cool bioluminescent), EVENT (trigger a burst of simultaneous fireflies).

### Rain / Dripping

Drops spawn at the apex and travel downward along branches toward the base. Each drop is a bright pixel with a dim trail. Drops follow branch topology — they fork at junctions and pool at the base.

**Topology**: Height (drops travel from high to low), branch (drops choose a branch at junctions), landmarks (apex = spawn point, base_junction = pool point), neighbor connectivity (drops can cross between branches at crossover points like LED 46/66).

**Input roles**: RATE (drop frequency / fall speed), INTENSITY (drop brightness), DENSITY (how many simultaneous drops), EVENT (trigger a heavy downpour burst), TEXTURE (single sharp drops vs soft diffuse drizzle), TRAJECTORY (downward movement speed).

### Aurora Borealis

Slow, wide color bands that drift horizontally across branches, with gentle undulation. Colors shift through greens, teals, purples, and pinks. The bands stretch across the XY plane so all branches at the same height share color.

**Topology**: Height (vertical color gradient), XY position (horizontal drift direction), branch (parallel color bands visible across all branches simultaneously).

**Input roles**: MOOD (color palette selection — cool greens vs warm pinks), RATE (drift speed), TEXTURE (smooth vs shimmering), INTENSITY (overall brightness / saturation), PHASE (position within the slow color cycle).

### Fire / Flame

Heat rises from the base upward. Each branch is an independent flame tongue. Brightness and color temperature decrease with height — white/yellow at base, orange in the middle, red/dark at tips. Random flicker on each pixel.

**Topology**: Height (heat gradient, bright at base, dark at apex), branch (independent flame tongues per branch), distance-from-root (flame intensity falloff).

**Input roles**: INTENSITY (flame height — how far up the branches the fire reaches), RATE (flicker speed), TEXTURE (smooth candle vs chaotic bonfire), DENSITY (how many pixels are actively flickering), MOOD (color temperature — blue flame vs orange flame), EVENT (flare-ups on transients).

### Breathing / Pulsing

Organic sine-wave brightness oscillation across the whole sculpture. Unlike a flat pulse, the breath wave travels upward from base to apex with a slight delay per height level, creating a visible "inhale" that fills the sculpture from bottom to top.

**Topology**: Height (wave propagation delay — base leads, tips follow), branch (all branches breathe together, reinforcing the unified organism feel).

**Input roles**: RATE (breath speed — maps directly to tempo or half-tempo), INTENSITY (breath depth — difference between inhale peak and exhale trough), PHASE (current position in breath cycle, can be beat-locked), MOOD (color shifts between inhale and exhale).

### Lightning / Electrical Arcs

A bright flash originates at the apex and races down a randomly chosen branch path to the base. The bolt is a narrow bright core (2-3 pixels) with a wider dim glow. After the main bolt, residual flickers along the path.

**Topology**: Branch (bolt chooses a specific path through the sculpture), height (propagation from apex to base or base to apex), neighbor connectivity (bolt can jump between branches at crossover points), landmarks (apex and tips as strike points).

**Input roles**: EVENT (trigger — strong transient fires a bolt), INTENSITY (bolt brightness and width), SURPRISE (lightning should feel unpredictable — irregular timing), RATE (propagation speed), TEXTURE (clean single bolt vs branching with aftershocks).

### Flowing Water / Streams

Continuous movement of soft blue-white pixels traveling along branches. Unlike rain (discrete drops), this is a fluid — a connected, flowing ribbon of light. Speed varies with audio energy. Eddies form at junctions.

**Topology**: Branch (each branch is a stream channel), landmarks (base_junction = confluence, crossover = rapids), height (flow direction — can be gravity-driven downward or pressure-driven upward), neighbor connectivity (flow merges or splits at cross-branch adjacencies).

**Input roles**: RATE (flow speed), INTENSITY (stream brightness / volume), TEXTURE (smooth laminar vs turbulent with sparkle), TRAJECTORY (flow direction — upward during builds, downward during breakdowns), DENSITY (stream width — narrow trickle vs full river).

### Wind Through Grass / Leaves

Pixels sway side-to-side in a wave pattern. The wave propagates along the height axis — base moves first, tips follow with a delay. Color shifts subtly with sway direction (lighter on one side, darker on the other). On multi-branch sculptures, branches sway with slightly different phases.

**Topology**: Height (wave propagation axis), XY position (sway displacement), branch (independent phase offsets per branch create natural look).

**Input roles**: RATE (wind speed / wave frequency), INTENSITY (sway amplitude — gentle breeze vs gale), TEXTURE (steady wind vs gusting), PHASE (wave position), MOOD (green/natural palette vs autumn colors).

### Bioluminescence

Slow, deep-ocean glow that spreads organically from touch points. A pixel ignites in deep blue-green and the glow spreads to neighbors over several frames, creating an expanding ring that fades at the edges. Multiple glow sources overlap additively.

**Topology**: Neighbor connectivity (glow spreads to physically adjacent pixels, including cross-branch neighbors like LEDs 4/63), distance-from-root (glow radius measured in graph distance, not index distance), XY position (glow forms 2D circles on the sculpture surface).

**Input roles**: EVENT (trigger a new glow source), DENSITY (number of simultaneous glow sources), RATE (spread speed), INTENSITY (glow brightness), MOOD (color — deep ocean teal vs warm amber), NOVELTY (new glow sources appear at novel musical moments).

### Heartbeat

Two quick pulses followed by a pause — the classic lub-dub rhythm. The pulse originates at the base (the heart) and radiates outward/upward along all branches simultaneously. Color shifts from red at the base to pink at the tips as the pulse travels.

**Topology**: Height (pulse propagation from base to tips), branch (pulse radiates on all branches simultaneously from base_junction), distance-from-root (pulse wavefront position).

**Input roles**: RATE (heartbeat tempo — maps to detected BPM or half-BPM), INTENSITY (pulse strength — stronger beats = brighter, wider pulse), PHASE (position in the lub-dub-rest cycle), EVENT (extra heartbeat on strong transients), TRAJECTORY (outward propagation from center).

### Swarm / Flocking

Multiple bright pixels move independently but with coordinated behavior — they drift in similar directions, cluster together, then scatter. On a branching sculpture, the swarm flows along branches and transfers between them at junction points.

**Topology**: Branch (swarm members travel along branches), neighbor connectivity (members transfer between branches at crossover points and junctions), landmarks (members cluster around landmarks — apex, tips, base), distance-from-root (members have position along the graph).

**Input roles**: DENSITY (number of swarm members), RATE (movement speed), MOOD (swarm color), TEXTURE (tight flock vs dispersed swarm), NOVELTY (swarm direction changes on novel moments), SURPRISE (sudden scatter on transients), SPACE (spatial distribution of the swarm across the sculpture).

### Growing Vines / Tendrils

A bright point starts at the base and grows upward along a branch, leaving a green trail behind. When it reaches a junction, it can fork. Growth speed varies with audio energy — fast during builds, pausing during quiet sections. The vine can retreat/wither during breakdowns.

**Topology**: Height (growth direction — base to apex), branch (vine follows branch paths, forks at junctions), landmarks (base_junction = growth origin, apex = growth target), distance-from-root (vine length / growth progress).

**Input roles**: TRAJECTORY (growth direction — forward during builds, retreat during breakdowns), RATE (growth speed), INTENSITY (vine brightness), PHASE (growth progress 0-1 along the sculpture), POSITION (current tip of the vine), FAMILIARITY (vine grows faster in repeated sections — it "knows the path").

### Snowfall

White/blue pixels drift slowly downward from apex to base with slight random horizontal wobble. Flakes accumulate at the base — a growing bright region at the bottom of each branch. Accumulation resets on strong transients (wind gust).

**Topology**: Height (fall direction — apex to base), branch (independent snowfall per branch), landmarks (base = accumulation zone), XY position (horizontal wobble).

**Input roles**: DENSITY (snowfall intensity — light flurry vs blizzard), RATE (fall speed), INTENSITY (flake brightness), EVENT (wind gust clears accumulation), TEXTURE (fine powder vs heavy wet flakes), MOOD (pure white vs blue-tinted cold).

### Sunrise / Sunset

A slow color transition that sweeps upward from the base. The base shifts from deep blue/purple through orange/gold to bright white, and this warm front propagates up the height axis over many seconds. The top of the sculpture is still night while the base is already dawn.

**Topology**: Height (the color wavefront moves along the height axis), branch (all branches share the same height-based color, creating a unified horizon line).

**Input roles**: TRAJECTORY (sunrise = warming upward sweep, sunset = cooling downward sweep), PHASE (time-of-day position in the full cycle), MOOD (dawn palette vs dusk palette vs golden hour), RATE (transition speed — can span an entire song section), INTENSITY (overall brightness follows the sun position).
