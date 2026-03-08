# Generative Art Pattern Catalog for 1D LED Arrays

Research on visual patterns from generative art, creative coding, and mathematical visualization adapted for LED strips and sculptures (1D arrays of 73-197 RGB pixels).

**Key Insight**: Prioritize patterns that feel PHYSICAL and SPATIAL, not just visual. The user specifically likes effects that "can't be rendered on a screen" and "take the form they are mapped onto" - like sap flowing up a tree.

---

## 1. Noise-Based Patterns

### 1.1 Perlin Noise (1D Time Animation)

**Description**: Smooth, organic randomness that creates natural-feeling variations. For 1D LED arrays, animate by moving through a 2D noise field (position + time) or 3D field (position + time + evolution).

**Physical/Spatial Quality**: ★★★★☆ - Creates fluid, organic motion like wind through leaves or water flowing.

**Implementation Approach**:
- Sample `noise(x, time)` for each LED position
- Map noise values to color/brightness
- Adjust octaves for complexity (more octaves = more detail)
- Use for smooth color transitions, breathing effects, or flowing gradients

**Time Dimension**: Animate by incrementing time parameter each frame - creates smooth, continuous evolution.

**Arduino/ESP32**: SimplexNoise libraries available for Arduino and ESP32 platforms for servos, LEDs, and other physical outputs.

**Code Pattern**:
```python
for i in range(num_leds):
    noise_val = perlin_noise(i * scale, time * speed)
    color[i] = map_to_color(noise_val)
```

**References**:
- Ken Perlin's original algorithm (1982)
- [Noise in Creative Coding](https://varun.ca/noise/)
- [The Book of Shaders: Noise](https://thebookofshaders.com/11/)
- [How to Use Perlin Noise in Procedural Animation](https://palospublishing.com/how-to-use-perlin-noise-in-procedural-animation/)

### 1.2 Simplex Noise (Faster Alternative)

**Description**: Ken Perlin's improved noise algorithm - computationally faster and visually smoother than Perlin noise.

**Physical/Spatial Quality**: ★★★★☆ - Same organic quality as Perlin but more efficient for real-time.

**Advantages Over Perlin**:
- Faster computation (important for ESP32)
- No directional artifacts
- Scales better to higher dimensions

**Implementation**: Similar to Perlin but use simplex_noise() function. Particularly good for flow fields (see 1.3).

**References**:
- [Simplex Noise - Grokipedia](https://grokipedia.com/page/Simplex_noise)
- [GitHub - SimplexNoise for Arduino](https://github.com/jshaw/SimplexNoise)

### 1.3 Flow Fields (Noise-Driven Particle Motion)

**Description**: Use 2D/3D noise to generate a vector field, then move particles through it. Each particle's direction determined by noise at its current position.

**Physical/Spatial Quality**: ★★★★★ - HIGHLY PHYSICAL. Like watching leaves drift in invisible currents, smoke flowing, or sap moving through organic pathways.

**Why This Works for LEDs**:
- Particles "travel" along the 1D strip following noise-driven paths
- Creates organic clustering and dispersion
- Feels like something is flowing THROUGH the physical structure

**Implementation**:
1. Each "particle" has a position on the LED strip
2. Calculate noise at that position → get direction/velocity
3. Move particle, render as colored pixel
4. Add damping for realistic physics

**Code Pattern**:
```python
for particle in particles:
    angle = noise(particle.pos, time) * TWO_PI
    particle.velocity += vec2(cos(angle), sin(angle))
    particle.velocity *= damping
    particle.pos += particle.velocity
    led[int(particle.pos)] = particle.color
```

**References**:
- [Flow Fields and Noise Algorithms with P5.js](https://dev.to/nyxtom/flow-fields-and-noise-algorithms-with-p5-js-5g67)
- [Particles in a Simplex Noise Flow Field - CodePen](https://codepen.io/DonKarlssonSan/post/particles-in-simplex-noise-flow-field)
- [Getting Creative with Perlin Noise Fields - Sighack](https://sighack.com/post/getting-creative-with-perlin-noise-fields)

### 1.4 Worley Noise / Voronoi (Cellular Patterns)

**Description**: Distance-based noise creating cell-like organic patterns. For each point, measure distance to nearest random "seed point."

**Physical/Spatial Quality**: ★★★☆☆ - Creates organic cell boundaries, but less naturally flowing than Perlin/Simplex.

**1D Application**:
- Scatter seed points along strip
- Color each LED based on distance to nearest seed
- Animate by moving seeds or changing distance metric

**Distance Variations**:
- Euclidean distance: circular cells
- Manhattan distance: diamond cells
- Chebyshev distance: square cells
- Combine multiple distances for complex textures

**Visual Result**: Sharp boundaries between regions vs smooth gradients of Perlin. Good for cracked earth, cells, or crystalline effects.

**References**:
- [Worley Noise - Wikipedia](https://en.wikipedia.org/wiki/Worley_noise)
- [Cellular Noise - The Book of Shaders](https://thebookofshaders.com/12/)
- [Understanding the Variations of Cellular Noise](https://sangillee.com/2025-04-18-cellular-noises/)

---

## 2. Wave Interference Patterns

### 2.1 Standing Waves (Stationary Oscillation)

**Description**: Superposition of two waves traveling in opposite directions creates nodes (no motion) and antinodes (maximum motion).

**Physical/Spatial Quality**: ★★★★★ - DEEPLY PHYSICAL. Standing waves are how strings vibrate, how sound resonates. Feels tied to physical laws.

**1D LED Implementation**:
```python
for i in range(num_leds):
    wave1 = sin(i * frequency + time)
    wave2 = sin(i * frequency - time)  # opposite direction
    standing_wave = wave1 + wave2
    brightness[i] = abs(standing_wave)
```

**Key Properties**:
- Nodes always at same positions (appear frozen)
- Antinodes oscillate in place
- Distance between nodes = λ/2 (half wavelength)

**Variations**:
- Change frequency to adjust number of nodes
- Modulate amplitude over time (breathing effect)
- Multiple standing waves at harmonic frequencies

**Why This Works**: Resonates with musical understanding - every note is a standing wave pattern. Maps cleanly to harmonic structure.

**References**:
- [Standing Waves - Physics LibreTexts](https://phys.libretexts.org/Courses/University_of_California_Davis/UCD:_Physics_9B__Waves_Sound_Optics_Thermodynamics_and_Fluids/01:_Waves/1.05:_Standing_Waves)
- [Wave Interference - Standing Waves and Beats](https://phys.libretexts.org/Bookshelves/Conceptual_Physics/Introduction_to_Physics_(Park)/03:_Unit_2-_Mechanics_II_-_Energy_and_Momentum_Oscillations_and_Waves_Rotation_and_Fluids/05:_Oscillations_and_Waves/5.06:_Wave_Interference-_Standing_Waves_and_Beats)

### 2.2 Beat Frequencies (Amplitude Modulation)

**Description**: Two sine waves with slightly different frequencies create a pulsing "beat" pattern.

**Physical/Spatial Quality**: ★★★★☆ - Highly perceptual. This is what you HEAR when tuning instruments.

**Formula**: `beat_freq = |f1 - f2|`

**1D Implementation**:
```python
wave1 = sin(2 * PI * f1 * time)
wave2 = sin(2 * PI * f2 * time)
beat_pattern = wave1 + wave2
# Amplitude oscillates at beat_freq
```

**Visual Result**: Periodic swelling and fading of brightness. Whole strip "breathes" together.

**Musical Connection**:
- 440Hz + 441Hz = 1 beat per second
- Consonant intervals (octave, fifth) = simple beat ratios
- Dissonant intervals = complex beating

**References**:
- [Wave Interference - Standing Waves and Beats](https://phys.libretexts.org/Bookshelves/Conceptual_Physics/Introduction_to_Physics_(Park)/03:_Unit_2-_Mechanics_II_-_Energy_and_Momentum_Oscillations_and_Waves_Rotation_and_Fluids/05:_Oscillations_and_Waves/5.06:_Wave_Interference-_Standing_Waves_and_Beats)

### 2.3 Traveling Waves (Propagating Pulses)

**Description**: Wave pulse that moves through the medium - like a ripple traveling down the strip.

**Physical/Spatial Quality**: ★★★★★ - EXTREMELY PHYSICAL. Visualizes the actual motion of energy through space.

**Types**:
1. **Single Pulse**: One peak travels from end to end
2. **Periodic Wave**: Continuous sinusoidal wave moving along strip
3. **Waterfall/Cascade**: Sequential LED activation

**Implementation**:
```python
# Traveling sine wave
for i in range(num_leds):
    position = i / num_leds
    brightness[i] = sin(2*PI * (position - time * speed))

# Pulse effect
pulse_position = (time * speed) % num_leds
for i in range(num_leds):
    distance = abs(i - pulse_position)
    brightness[i] = exp(-distance**2 / width**2)  # Gaussian pulse
```

**Audio-Reactive Application**:
- Onset detection triggers pulse
- Pulse speed mapped to tempo
- Pulse width mapped to transient sharpness

**References**:
- [Traveling Waves - Physics](http://labman.phys.utk.edu/phys221core/modules/m11/traveling_waves.html)
- [LED strip pulse/waterfall effect - Arduino Forum](https://forum.arduino.cc/t/led-strip-pulse-waterfall-effect-on-input/417097)
- [Basic Animations - CircuitPython LED Animations](https://learn.adafruit.com/circuitpython-led-animations/basic-animations)

### 2.4 Lissajous Figures (1D Projection)

**Description**: Superposition of perpendicular oscillations. Normally 2D curves, but can project to 1D by sampling one axis or using for position modulation.

**Physical/Spatial Quality**: ★★★☆☆ - Interesting mathematically but loses physicality in 1D projection.

**1D Application**:
- Use Lissajous figure to modulate LED position over time
- `position = A*sin(f1*t) + B*sin(f2*t + phase)`
- Frequency ratios create different patterns (2:3 = figure-8, 1:1 = circle → line)

**Musical Connection**: Frequency ratios map to musical intervals:
- 1:1 = unison
- 2:1 = octave
- 3:2 = perfect fifth

**Novel Approach**: Physical pendulum with 1D LED row creates persistence-of-vision Lissajous figures.

**References**:
- [Lissajous Curve - Wikipedia](https://en.wikipedia.org/wiki/Lissajous_curve)
- [Harmonograph - Wikipedia](https://en.wikipedia.org/wiki/Harmonograph)
- [Lissajous POV approach - Evil Mad Scientist](https://www.evilmadscientist.com/2008/a-simple-persistence-of-vision-approach-to-lissajous-figures/)

---

## 3. Cellular Automata

### 3.1 Elementary CA (Rule 30, Rule 110)

**Description**: 1D cellular automaton where each cell's next state depends on current state + two neighbors. 256 possible rules (2^8).

**Physical/Spatial Quality**: ★★☆☆☆ - Mathematically fascinating but NOT physically intuitive. Feels computational, not natural.

**Why Limited Physical Appeal**:
- Evolution is discrete (blocky, not smooth)
- No obvious mapping to physical processes
- Works best when viewing history (2D grid of time evolution)

**1D LED Challenge**:
- Single row shows only current generation (loses historical pattern)
- Could scroll history down the strip, but then it's just moving pixels
- Better for 2D matrices than 1D strips

**Notable Rules**:
- **Rule 30**: Chaotic, pseudo-random patterns
- **Rule 110**: Turing complete, complex emergent behavior
- **Rule 90**: Sierpiński triangle pattern

**Code**:
```python
def evolve_rule30(cells):
    new_cells = [0] * len(cells)
    for i in range(len(cells)):
        left = cells[(i-1) % len(cells)]
        center = cells[i]
        right = cells[(i+1) % len(cells)]
        pattern = (left << 2) | (center << 1) | right
        new_cells[i] = (30 >> pattern) & 1  # Rule 30 lookup
    return new_cells
```

**References**:
- [Elementary Cellular Automaton - Wikipedia](https://en.wikipedia.org/wiki/Elementary_cellular_automaton)
- [A Guide to Cellular Automata - Rule 30 and Rule 110](https://kindatechnical.com/cellular-automata/rule-30-and-rule-110.html)
- [What the hack is Rule 30?](https://gettocode.com/2021/10/17/what-the-hack-is-rule-30-cellular-automat-explained/)

### 3.2 Continuous CA / Reaction-Diffusion (1D)

**Description**: Smooth version of CA using differential equations. Gray-Scott model simulates chemical reaction + diffusion.

**Physical/Spatial Quality**: ★★★★☆ - Models REAL PHYSICAL chemistry. Can create spots, stripes, spirals (in 2D).

**1D Gray-Scott**:
- Two chemicals U and V react and diffuse
- Parameters F (feed rate) and k (kill rate) control pattern type
- 1D version creates stripes, moving fronts, oscillations

**Patterns in 1D**:
- Stable stripes
- Traveling waves
- Oscillating regions
- Chaotic turbulence

**Code Sketch**:
```python
# Simplified 1D Gray-Scott
for i in range(num_leds):
    laplacian_u = (U[i-1] + U[i+1] - 2*U[i]) / dx**2
    laplacian_v = (V[i-1] + V[i+1] - 2*V[i]) / dx**2

    reaction = U[i] * V[i]**2

    dU = Du * laplacian_u - reaction + F * (1 - U[i])
    dV = Dv * laplacian_v + reaction - (F + k) * V[i]

    U[i] += dU * dt
    V[i] += dV * dt
```

**Challenge**: Computationally intensive for ESP32 real-time. Better for pre-computed patterns.

**References**:
- [Gray-Scott Model of Reaction-Diffusion](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/)
- [Reaction-Diffusion Tutorial - Karl Sims](https://www.karlsims.com/rd.html)
- [Gray-Scott Model - VisualPDE](https://visualpde.com/nonlinear-physics/gray-scott.html)

---

## 4. Physics Simulations

### 4.1 Spring/Mass Systems (Soft Body)

**Description**: Network of masses connected by springs. Deformations propagate as oscillations.

**Physical/Spatial Quality**: ★★★★★ - EXTREMELY PHYSICAL. Simulates actual elastic materials.

**1D Implementation**:
- Each LED position is a mass
- Springs connect adjacent masses
- Apply forces (gravity, drag, external impulses)
- Integrate to get positions → map to brightness/color

**Code**:
```python
# Verlet integration for spring chain
for i in range(1, num_leds - 1):
    force = spring_k * (pos[i+1] - 2*pos[i] + pos[i-1])
    force += -damping * velocity[i]

    acceleration = force / mass
    new_pos = 2*pos[i] - prev_pos[i] + acceleration*dt**2

    prev_pos[i] = pos[i]
    pos[i] = new_pos

    brightness[i] = map(pos[i], min_pos, max_pos, 0, 255)
```

**Effects**:
- Pluck one end → wave travels and reflects
- Audio onset → impulse at random position
- Gravity → sag in the middle
- Jiggle → ongoing oscillation

**References**:
- [Programming Soft Body Physics and Blobs](https://www.gorillasun.de/blog/soft-body-physics-and-blobs/)
- [Spring Pendulum - Javalab](https://javalab.org/en/spring_en/)

### 4.2 Pendulum (Simple/Double/Coupled)

**Description**: Mass on a string, oscillates due to gravity. Double pendulum = chaos.

**Physical/Spatial Quality**: ★★★☆☆ - Physical but hard to map to 1D strip naturally.

**1D Mapping Challenge**:
- Pendulum is inherently 2D motion (angle + length)
- Could map angle to LED position, but feels arbitrary
- Better: coupled pendulums (array of pendulums) each controlling one LED

**Coupled Pendulums**:
- Each LED has an associated pendulum
- Pendulums weakly coupled (spring between neighbors)
- Creates wave-like patterns of synchronization

**References**:
- [Simple Pendulum - myPhysicsLab](https://www.myphysicslab.com/pendulum/pendulum-en.html)
- [Double Pendulum - myPhysicsLab](https://www.myphysicslab.com/pendulum/double-pendulum-en.html)

### 4.3 Wave Equation (1D String Vibration)

**Description**: Partial differential equation governing vibrating string. Produces standing waves, harmonics.

**Physical/Spatial Quality**: ★★★★★ - THE physics equation for 1D vibration. Perfectly matched to LED strips.

**Wave Equation**: `∂²y/∂t² = c² ∂²y/∂x²`

**What It Means**: Acceleration at each point proportional to curvature.

**Finite Difference Implementation**:
```python
c = wave_speed
dx = 1.0 / num_leds
dt = timestep

for i in range(1, num_leds - 1):
    curvature = (y[i+1] - 2*y[i] + y[i-1]) / dx**2
    acceleration = c**2 * curvature - damping * velocity[i]

    velocity[i] += acceleration * dt
    y[i] += velocity[i] * dt

    brightness[i] = map(y[i], -1, 1, 0, 255)
```

**Boundary Conditions**:
- Fixed ends: `y[0] = y[n-1] = 0` (guitar string)
- Free ends: `∂y/∂x = 0` at ends (flute)
- Periodic: `y[0] = y[n-1]` (circular membrane)

**Audio-Reactive**:
- Bass hit → impulse at base
- Treble → high-frequency excitation at tips
- Let physics do the work

**References**:
- [Traveling Waves - Physics](http://labman.phys.utk.edu/phys221core/modules/m11/traveling_waves.html)
- [Spring Wave Simulation](http://physics.bu.edu/~duffy/HTML5/spring_wave.html)

### 4.4 Heat Diffusion (1D Thermal Simulation)

**Description**: Temperature diffuses from hot to cold. Smooth gradual spreading.

**Physical/Spatial Quality**: ★★★★☆ - Models real thermal process. Feels slow, smooth, organic.

**Heat Equation**: `∂T/∂t = α ∂²T/∂x²`

**1D Implementation**:
```python
alpha = thermal_diffusivity
dx = 1.0 / num_leds
dt = timestep

for i in range(1, num_leds - 1):
    laplacian = (T[i+1] - 2*T[i] + T[i-1]) / dx**2
    dT = alpha * laplacian
    T[i] += dT * dt

    color[i] = heat_to_color(T[i])  # black-red-yellow-white gradient
```

**Heat-to-Color Mapping**:
- Black body radiation: 0K = black, 1000K = red, 3000K = orange, 6000K = white
- FastLED `HeatColor()` palette does this

**Audio-Reactive**:
- Bass = add heat at base
- Treble = add heat at tips
- Let diffusion smooth it out naturally

**Visual Quality**: Slow, flowing, organic color gradients. "Lava lamp" aesthetic.

**References**:
- [1D Heat Diffusion - Finite Difference Method](https://github.com/Gray-Sword/1D-Heat-Diffusion-Simulation-Using-Finite-Difference-Method)
- [Heat Equation - VisualPDE](https://visualpde.com/basic-pdes/heat-equation.html)

---

## 5. Attractor-Inspired Patterns

### 5.1 Lorenz Attractor (Chaos)

**Description**: Deterministic chaos from 3 coupled differential equations. The famous "butterfly" shape.

**Physical/Spatial Quality**: ★★☆☆☆ - Chaotic but not obviously physical to observers. Feels mathematical.

**3D → 1D Projection**:
- Solve Lorenz equations: dx/dt, dy/dt, dz/dt
- Project to 1D: use X coordinate, or magnitude, or angle
- Map to LED position or color

**Equations**:
```python
dx = sigma * (y - x)
dy = x * (rho - z) - y
dz = x * y - beta * z
```

**Parameters** (classic):
- σ = 10, ρ = 28, β = 8/3

**1D LED Mapping**:
- LED position = x(t) mapped to 0-num_leds
- Color = y(t) or z(t)
- Brightness = speed of motion

**Visual Result**: Erratic, never-repeating movement. Organic but unpredictable.

**References**:
- [Lorenz System Attractor - GitHub](https://github.com/gboeing/lorenz-system/blob/master/lorenz-system-attractor-visualize.ipynb)
- [Strange Attractors: Lorenz, Rössler, Hénon](https://fiveable.me/chaos-theory/unit-6)

### 5.2 Rössler Attractor

**Description**: Simpler chaotic attractor with single band (twisted spiral in 3D).

**Physical/Spatial Quality**: ★★☆☆☆ - Similar to Lorenz, mathematically interesting but not physically intuitive.

**Equations**:
```python
dx = -(y + z)
dy = x + a*y
dz = b + z*(x - c)
```

**Parameters** (typical):
- a = 0.2, b = 0.2, c = 5.7

**1D Projection**: Similar strategies as Lorenz - project one axis to LED position.

**References**:
- [Rössler Attractor - Scholarpedia](http://www.scholarpedia.org/article/Rossler_attractor)
- [Circuits of Chaos - Building Strange Attractors](https://www.electronicdesign.com/technologies/analog/article/55131908/circuits-of-chaos-building-lorenz-chua-and-rossler-strange-attractors)

---

## 6. Mathematical Sequences

### 6.1 Fibonacci Sequence

**Description**: Each number is sum of previous two: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...

**Physical/Spatial Quality**: ★★★☆☆ - Found in nature (spirals, phyllotaxis) but 1D visualization is abstract.

**1D LED Applications**:

1. **Spacing Pattern**: Light LEDs at Fibonacci indices (0, 1, 2, 3, 5, 8, 13...)
2. **Timing Pattern**: Change color every Fibonacci milliseconds
3. **Brightness Ratios**: Scale brightness by Fibonacci numbers
4. **Color Steps**: Move through color wheel by golden ratio (137.5°)

**Golden Ratio φ ≈ 1.618**: Ratio between consecutive Fibonacci numbers approaches φ.

**Golden Angle ≈ 137.5°**: 360° / φ² - optimal for spiral packing.

**Code**:
```python
# Golden ratio color cycling
hue = 0
for frame in range(num_frames):
    for i in range(num_leds):
        color[i] = HSV(hue, 255, 255)
    hue = (hue + 137.5) % 360  # golden angle
```

**Visual Quality**: Subtle asymmetry, never exactly repeating. Pleasant irrational spacing.

**References**:
- [The Fibonacci Sequence and Golden Ratio - Cleveland Design](https://clevelanddesign.com/insights/the-nature-of-design-the-fibonacci-sequence-and-the-golden-ratio/)
- [Fibonacci's Hidden Code in Classical Art](https://thefusepathway.com/blog/fibonaccis-hidden-code-uncovering-the-mathematics-behind-classical-art/)

### 6.2 Prime Numbers

**Description**: Numbers divisible only by 1 and themselves: 2, 3, 5, 7, 11, 13, 17...

**Physical/Spatial Quality**: ★★☆☆☆ - Mathematically fascinating, no physical intuition.

**1D LED Visualization**:

1. **Light Primes**: LED i lights if i is prime
2. **Prime Gaps**: Brightness = gap to next prime
3. **Ulam Spiral**: 1D slice through 2D spiral pattern
4. **Prime Rhythm**: Trigger effects at prime-numbered frames

**Visual Result**: Irregular spacing with no obvious pattern. Creates interesting asymmetric rhythms.

**Code**:
```python
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

for i in range(num_leds):
    brightness[i] = 255 if is_prime(i) else 0
```

**References**:
- [Math for Artists: Visualizing Prime Numbers](https://medium.com/itp-blog/math-for-artists-visualizing-prime-numbers-c1ca6a309be2)
- [Ulam Spiral - Wikipedia](https://en.wikipedia.org/wiki/Ulam_spiral)
- [Plotting Prime Numbers - Jake Tae](https://jaketae.github.io/study/prime-spirals/)

---

## 7. Color Theory Patterns

### 7.1 Palette Cycling

**Description**: Smoothly interpolate through a predefined color palette.

**Physical/Spatial Quality**: ★★☆☆☆ - Purely visual, no physical metaphor. But can be VERY beautiful.

**FastLED Palettes**:
- RainbowColors_p
- OceanColors_p
- LavaColors_p
- ForestColors_p
- HeatColors_p (black-red-orange-yellow-white)

**Implementation**:
```cpp
CRGBPalette16 palette = RainbowColors_p;
uint8_t colorIndex = 0;

void loop() {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = ColorFromPalette(palette, colorIndex + i*3);
    }
    colorIndex++;
}
```

**Techniques**:
- **Spatial gradient**: Each LED samples different palette position
- **Temporal cycle**: Whole strip cycles through palette together
- **Wave**: Palette index = sine wave function

**References**:
- [FastLED ColorPalette Example](https://github.com/FastLED/FastLED/blob/master/examples/ColorPalette/ColorPalette.ino)
- [Fire2012 with Palette](https://github.com/FastLED/FastLED/blob/master/examples/Fire2012WithPalette/Fire2012WithPalette.ino)

### 7.2 Color Harmonics (Complementary/Analogous/Split)

**Description**: Color relationships based on color wheel position.

**Physical/Spatial Quality**: ★☆☆☆☆ - Pure color theory, no physicality.

**Harmony Types**:

1. **Complementary**: Opposite on color wheel (blue/orange, red/cyan)
   - High contrast, vibrant

2. **Analogous**: Adjacent colors (blue/cyan/green)
   - Low contrast, harmonious, smooth

3. **Split Complementary**: Base color + two adjacent to complement
   - High contrast with more variety

4. **Triadic**: Three colors evenly spaced (120° apart)

**LED Application**:
- Assign different strip sections to harmony colors
- Oscillate between complementary colors
- Smooth transition through analogous range

**Code**:
```python
# Complementary oscillation
base_hue = 180  # cyan
for i in range(num_leds):
    if i < num_leds // 2:
        color[i] = HSV(base_hue, 255, 255)
    else:
        color[i] = HSV((base_hue + 180) % 360, 255, 255)
```

**References**:
- [Color Harmonies - The Paper Mill](https://blog.thepapermillstore.com/color-theory-color-harmonies/)
- [Color Theory for Beginners - Color Wheel and Harmonies](https://sarahrenaeclark.com/color-theory-basics/)

### 7.3 Rainbow Distributions

**Description**: Smooth rainbow gradient mapped across the strip.

**Physical/Spatial Quality**: ★★☆☆☆ - Visually appealing but no physical meaning unless mapped to frequency (like a spectrogram).

**FastLED Approach**:
```cpp
fill_rainbow(leds, NUM_LEDS, starting_hue, delta_hue);
```

**Parameters**:
- `starting_hue`: Initial color (0-255)
- `delta_hue`: Hue change per LED (larger = faster color change)

**Techniques**:
- **Static rainbow**: Fixed gradient
- **Rotating rainbow**: Increment starting_hue each frame
- **Breathing rainbow**: Modulate brightness with sine wave
- **Compressed rainbow**: Vary delta_hue (sparse vs dense)

**References**:
- [How to Use FastLED - Program LED Strips](https://racheldebarros.com/arduino-projects/how-to-use-fastled-with-arduino-to-program-led-strips/)
- [FastLED-Patterns on GitHub](https://github.com/Resseguie/FastLED-Patterns/blob/master/fastled-patterns.ino)

---

## 8. Common LED Pattern Primitives (FastLED)

These are the building blocks found in FastLED examples and libraries.

### 8.1 Fire Effect (Heat Simulation)

**Description**: Simulates flickering flames using heat diffusion + random cooling.

**Physical/Spatial Quality**: ★★★★☆ - Mimics actual fire physics (heat rises, flickers).

**Algorithm** (Fire2012):
1. Cool down every cell a little bit (random amount)
2. Heat from each cell drifts up and diffuses
3. Randomly ignite new sparks at base
4. Map heat values to color (black-red-orange-yellow-white)

**Code Pattern**:
```cpp
// Step 1: Cool down
for (int i = 0; i < NUM_LEDS; i++) {
    heat[i] = qsub8(heat[i], random8(0, cooling));
}

// Step 2: Heat drifts up
for (int k = NUM_LEDS - 1; k >= 2; k--) {
    heat[k] = (heat[k - 1] + heat[k - 2] + heat[k - 2]) / 3;
}

// Step 3: Randomly ignite new sparks
if (random8() < sparking) {
    heat[random8(7)] = qadd8(heat[random8(7)], random8(160, 255));
}

// Step 4: Map heat to color
for (int j = 0; j < NUM_LEDS; j++) {
    leds[j] = HeatColor(heat[j]);
}
```

**Parameters**:
- Cooling: How quickly flames die down
- Sparking: How often new sparks ignite

**Why It Works**: Combines deterministic physics (diffusion) with randomness (sparks). Feels alive.

**References**:
- [FastLED Fire2012WithPalette](https://github.com/FastLED/FastLED/blob/master/examples/Fire2012WithPalette/Fire2012WithPalette.ino)
- [LED Campfire - Adafruit](https://learn.adafruit.com/led-campfire/the-code)

### 8.2 Sinelon (Oscillating Dot)

**Description**: Single bright dot oscillating back and forth using beatsin function.

**Physical/Spatial Quality**: ★★★☆☆ - Like a pendulum or bouncing ball. Simple but effective.

**Code**:
```cpp
void sinelon() {
    fadeToBlackBy(leds, NUM_LEDS, 20);  // Trails
    int pos = beatsin16(13, 0, NUM_LEDS-1);  // 13 BPM oscillation
    leds[pos] += CHSV(gHue, 255, 192);
}
```

**Key Function**: `beatsin16(BPM, min, max)` generates 16-bit sine wave oscillating between min and max.

**Variations**:
- Multiple sinelons at different speeds
- Change BPM to match music tempo
- Vary trail fade rate

**References**:
- [FastLED DemoReel100](https://github.com/FastLED/FastLED/blob/master/examples/DemoReel100/DemoReel100.ino)
- [beatsin8() and beatsin16() are very cool - Maker Forums](https://forum.makerforums.info/t/beatsin8-and-beatsin16-are-very-cool-functions/64208)

### 8.3 Juggle (Multiple Oscillators)

**Description**: Multiple sinelons at different frequencies creating juggling effect.

**Physical/Spatial Quality**: ★★★☆☆ - Like watching juggling balls. Playful.

**Code**:
```cpp
void juggle() {
    fadeToBlackBy(leds, NUM_LEDS, 20);
    byte dothue = 0;
    for (int i = 0; i < 8; i++) {
        leds[beatsin16(i+7, 0, NUM_LEDS-1)] |= CHSV(dothue, 200, 255);
        dothue += 32;
    }
}
```

**Pattern**: 8 dots oscillating at different BPMs (7, 8, 9... 14), each a different hue.

**Why It Works**: Complexity from simplicity. Periodic but never exactly repeating.

**References**:
- [FastLED DemoReel100](https://github.com/FastLED/FastLED/blob/master/examples/DemoReel100/DemoReel100.ino)

### 8.4 BPM-Synced Patterns

**Description**: Use beat generators (beat8, beatsin8, beatsin16) to sync patterns to tempo.

**Physical/Spatial Quality**: ★★★★☆ - Locked to musical time = deeply connected to audio.

**Beat Functions**:
- `beat8(BPM)`: Sawtooth wave 0-255 at BPM
- `beatsin8(BPM, min, max)`: Sine wave oscillating between min-max
- `beatsin16(BPM, min, max)`: 16-bit version for smoother motion

**Example**:
```cpp
void bpm() {
    uint8_t BeatsPerMinute = 62;
    CRGBPalette16 palette = PartyColors_p;
    uint8_t beat = beatsin8(BeatsPerMinute, 64, 255);

    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = ColorFromPalette(palette, gHue + (i*2), beat - gHue + (i*10));
    }
}
```

**Audio-Reactive**: Replace hardcoded BPM with detected tempo from audio analysis.

**References**:
- [FastLED Beat Generators Documentation](https://fastled.io/docs/d6/d6c/group___beat_generators.html)
- [beatwave demo - GitHub](https://github.com/atuline/FastLED-Demos/blob/master/beatwave/beatwave.ino)

### 8.5 Confetti (Random Sparkles)

**Description**: Random LEDs light up with random colors.

**Physical/Spatial Quality**: ★★☆☆☆ - Like sparkles or fireflies. Purely decorative but pretty.

**Code**:
```cpp
void confetti() {
    fadeToBlackBy(leds, NUM_LEDS, 10);
    int pos = random16(NUM_LEDS);
    leds[pos] += CHSV(gHue + random8(64), 200, 255);
}
```

**Audio-Reactive Version**: Sparkle rate proportional to spectral density or high-frequency energy.

**References**:
- [FastLED DemoReel100](https://github.com/FastLED/FastLED/blob/master/examples/DemoReel100/DemoReel100.ino)

### 8.6 Comet / Chase

**Description**: Bright head with fading tail moving along strip.

**Physical/Spatial Quality**: ★★★★☆ - Like a shooting star or comet. Directional motion = physical.

**Code**:
```cpp
void comet(int position, CRGB color) {
    // Fade everything
    fadeToBlackBy(leds, NUM_LEDS, 50);

    // Bright head
    leds[position] = color;

    // Trailing tail (already faded by fadeToBlackBy)
}
```

**Variations**:
- Bounce: reverse direction at ends
- Multiple comets
- Speed varies with audio tempo

**References**:
- [quickPatterns Comet](https://github.com/brimshot/quickPatterns)
- [CircuitPython Comet Animation](https://learn.adafruit.com/circuitpython-led-animations/basic-animations)

### 8.7 Plasma Effect (Smooth Color Morphing)

**Description**: Multiple sine waves combined to create flowing, organic color patterns.

**Physical/Spatial Quality**: ★★★☆☆ - Looks like flowing liquid or energy fields. Organic but not physically grounded.

**Algorithm**:
```python
for i in range(num_leds):
    x = i / num_leds

    val = sin(x * freq1 + time * speed1)
    val += sin(x * freq2 + time * speed2)
    val += sin(x * freq3 + time * speed3 + PI/4)

    hue = map(val, -3, 3, 0, 360)
    color[i] = HSV(hue, 255, 255)
```

**Key**: Combine multiple sine waves at different frequencies/phases → complex organic motion.

**References**:
- [Plasma Effect - Wikipedia](https://en.wikipedia.org/wiki/Plasma_effect)
- [LED Flame with Plasma Effect - Instructables](https://www.instructables.com/LED-Flame-Controlled-by-Noise/)
- [Plasma Effect - Rosetta Code](https://rosettacode.org/wiki/Plasma_effect)

### 8.8 Pride (FastLED's Pride2015)

**Description**: Rainbow with brightness variation creating wave-like motion.

**Physical/Spatial Quality**: ★★☆☆☆ - Beautiful but purely visual.

**What Makes It Special**: Not just a static rainbow - brightness oscillates creating apparent motion.

**References**:
- [WLED Effects List - Pride 2015](https://github.com/Aircoookie/WLED/wiki/List-of-effects-and-palettes)
- [FastLED Pride example code](https://github.com/Resseguie/FastLED-Patterns/blob/master/fastled-patterns.ino)

---

## 9. Advanced / Exotic Patterns

### 9.1 Metaballs (Distance Field Blobs)

**Description**: Soft spheres whose influence fields merge smoothly when close.

**Physical/Spatial Quality**: ★★★☆☆ - Like liquid droplets merging. Organic but computationally abstract.

**1D Implementation**:
```python
# Multiple blob centers moving along strip
blobs = [(pos1, radius1), (pos2, radius2), (pos3, radius3)]

for i in range(num_leds):
    field = 0
    for (center, radius) in blobs:
        distance = abs(i - center)
        field += radius**2 / (distance**2 + 0.01)  # Prevent divide-by-zero

    brightness[i] = min(255, field * 50)
```

**Smooth Minimum**: For soft merging, use smooth min function instead of addition:
```python
def smooth_min(a, b, k):
    h = max(0, k - abs(a - b)) / k
    return min(a, b) - h*h*k*0.25
```

**Computational Cost**: Ray marching approach is expensive. Simpler 1D version is feasible.

**References**:
- [Metaballs - Wikipedia](https://en.wikipedia.org/wiki/Metaballs)
- [Ryan's Guide to MetaBalls](http://www.geisswerks.com/ryan/BLOBS/blobs.html)
- [Create Animated Metaballs with Shaders](https://alexcodesart.com/create-animated-metaballs-with-shaders-in-p5-js-a-creative-coding-tutorial/)

### 9.2 Fractals (L-Systems / Koch Curve)

**Description**: Recursive patterns generated by string rewriting systems.

**Physical/Spatial Quality**: ★★☆☆☆ - Beautiful mathematical structure but not obviously physical.

**L-System Basics**:
- Alphabet: F (forward), + (turn right), - (turn left), [ (push), ] (pop)
- Axiom: Starting string
- Rules: F → F+F--F+F (Koch curve)
- Iterate: Apply rules N times

**1D Challenge**: L-systems naturally create 2D/3D structures (trees, ferns). 1D representation loses the magic.

**Possible 1D Approach**:
- Use stack depth as brightness
- Map turtle angle to color
- Track path length for position

**Example - Koch Curve**:
- Axiom: F
- Rule: F → F+F--F+F
- After 3 iterations: Creates fractal spike pattern

**Better for**: Tree sculptures where branching structure maps naturally.

**References**:
- [L-system - Wikipedia](https://en.wikipedia.org/wiki/L-system)
- [L-systems: Draw Fractals and Plants](https://medium.com/@hhtun21/l-systems-draw-your-first-fractals-139ed0bfcac2)
- [Nature of Code - Fractals](https://natureofcode.com/fractals/)

### 9.3 Particle Systems

**Description**: Many independent particles with position, velocity, lifecycle. Physics updates + rendering.

**Physical/Spatial Quality**: ★★★★★ - HIGHLY PHYSICAL when done well. Feels like actual particles flowing, bouncing, swirling.

**Components**:
```python
class Particle:
    def __init__(self):
        self.pos = random(num_leds)
        self.vel = random(-1, 1)
        self.life = 255
        self.color = random_color()

    def update(self, dt):
        self.vel += gravity * dt
        self.pos += self.vel * dt
        self.life -= fade_rate * dt

        # Bounce at boundaries
        if self.pos < 0 or self.pos >= num_leds:
            self.vel *= -0.8  # damping

    def render(self, leds):
        idx = int(self.pos)
        if 0 <= idx < num_leds:
            leds[idx] += self.color * self.life
```

**Audio-Reactive**:
- Emit particles on bass hits
- Particle color from spectral centroid
- Emission rate from RMS energy
- Gravity from low-frequency energy

**Why This Works**:
- Matches user's "sap flow" aesthetic perfectly
- Takes the form of the physical structure
- Can't be properly shown on a screen - needs the 3D sculpture

**References**:
- [Simple Particle System - Processing](https://processing.org/examples/simpleparticlesystem.html)
- [Procedural Art with Particle Systems - Unity](https://shahriyarshahrabi.medium.com/procedural-art-with-unity3d-particle-systems-and-vector-fields-5a0b6d4cdc65)

### 9.4 Easing Functions (Animation Curves)

**Description**: Mathematical functions that control acceleration/deceleration of motion. Makes animations feel natural.

**Physical/Spatial Quality**: ★★★★☆ - Essential for physical realism. Linear motion feels robotic; eased motion feels alive.

**Robert Penner's Easing Functions**:

- **Ease In**: Start slow, accelerate (quad, cubic, quart, quint, expo)
- **Ease Out**: Start fast, decelerate
- **Ease In-Out**: Slow → fast → slow (S-curve)

**Code Examples**:
```python
# Linear (no easing)
t = time / duration

# Ease In Quad
t = t * t

# Ease Out Quad
t = t * (2 - t)

# Ease In-Out Cubic
if t < 0.5:
    t = 4 * t * t * t
else:
    t = 1 - pow(-2 * t + 2, 3) / 2
```

**LED Application**:
- Pulse brightness with ease-in-out for breathing effect
- Move comet position with ease-out for natural deceleration
- Transition between colors with easing for smooth blends

**Why Essential**: The difference between "moving pixels" and "physics simulation" is often just easing.

**References**:
- [Robert Penner's Easing Functions](https://robertpenner.com/easing/)
- [Animating with Easing Functions - Kirupa](https://www.kirupa.com/html5/animating_with_easing_functions_in_javascript.htm)
- [Easing Functions Cheat Sheet](https://easings.net/)

---

## 10. Pattern Selection Framework

### Physical Quality Ratings Summary

**★★★★★ EXTREMELY PHYSICAL** (Best for "takes the form" aesthetic):
- Flow fields (noise-driven particles)
- Standing waves
- Traveling waves
- Spring/mass systems
- Wave equation simulation
- Particle systems
- Fire effect

**★★★★☆ HIGHLY PHYSICAL**:
- Perlin/Simplex noise (organic flow)
- Beat frequencies
- Heat diffusion
- Comet/chase
- Easing functions

**★★★☆☆ MODERATELY PHYSICAL**:
- Lissajous figures
- Fibonacci spacing
- Soft body physics
- Metaballs
- Plasma effect

**★★☆☆☆ WEAKLY PHYSICAL**:
- Cellular automata
- Prime numbers
- Rainbow gradients
- Palette cycling
- Confetti

**★☆☆☆☆ NOT PHYSICAL**:
- Color harmonics (pure aesthetics)

### Recommendations for Tree Sculpture (197 LEDs)

1. **Flow Fields + Particle Systems**: Particles rise from trunk to branches following noise field. PERFECT for "sap flow" aesthetic.

2. **Wave Equation**: Bass hits at trunk propagate as waves to tips. Pure physics.

3. **Heat Diffusion + Fire**: Heat/energy added at base, diffuses upward. Natural mapping to tree structure.

4. **Spring/Mass + Audio**: Each branch segment is a spring. Audio impulses create oscillations that propagate.

5. **Standing Waves + Harmonics**: Map musical harmonics to physical standing wave modes. Deep connection between sound and structure.

### ESP32 Feasibility Notes

**Fast Enough**:
- Perlin/Simplex noise (with lookup tables)
- Simple particle systems (<50 particles)
- Heat diffusion (1D finite difference)
- Wave equation (1D finite difference)
- FastLED primitives (fire, sinelon, juggle)
- Easing functions

**Borderline**:
- Flow fields (depends on resolution)
- Reaction-diffusion (needs optimization)
- Many particles (>100)

**Too Slow**:
- Ray marching (metaballs 3D)
- Chaos attractors with high precision
- Complex L-systems with deep recursion

**Strategy**: Pre-compute expensive patterns, store in lookup tables. Use integer math. Profile everything.

---

## Sources

### Creative Coding & Generative Art
- [Awesome Creative Coding - GitHub](https://github.com/terkelg/awesome-creative-coding)
- [Generative Art: 50 Best Examples](https://aiartists.org/generative-art-design)
- [Noise in Creative Coding - Varun Vachhar](https://varun.ca/noise/)
- [The Book of Shaders](https://thebookofshaders.com/)

### LED Programming
- [Pixelblaze Documentation](https://www.crowdsupply.com/hencke-technologies/pixelblaze-v3)
- [FastLED Library - GitHub](https://github.com/FastLED/FastLED)
- [FastLED Documentation](https://fastled.io/)
- [Tweaking4All - LED Strip Effects](https://www.tweaking4all.com/hardware/arduino/adruino-led-strip-effects/)
- [CircuitPython LED Animations](https://learn.adafruit.com/circuitpython-led-animations/basic-animations)

### Noise & Flow Fields
- [Flow Fields and Noise with P5.js](https://dev.to/nyxtom/flow-fields-and-noise-algorithms-with-p5-js-5g67)
- [Getting Creative with Perlin Noise Fields - Sighack](https://sighack.com/post/getting-creative-with-perlin-noise-fields)
- [Particles in Simplex Noise Flow Field - CodePen](https://codepen.io/DonKarlssonSan/post/particles-in-simplex-noise-flow-field)

### Physics & Mathematics
- [myPhysicsLab - Pendulum](https://www.myphysicslab.com/pendulum/pendulum-en.html)
- [Programming Soft Body Physics](https://www.gorillasun.de/blog/soft-body-physics-and-blobs/)
- [Heat Diffusion Finite Difference - GitHub](https://github.com/Gray-Sword/1D-Heat-Diffusion-Simulation-Using-Finite-Difference-Method)
- [Gray-Scott Reaction-Diffusion](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/)
- [Karl Sims Reaction-Diffusion Tutorial](https://www.karlsims.com/rd.html)

### Color Theory
- [Color Harmonies - The Paper Mill](https://blog.thepapermillstore.com/color-theory-color-harmonies/)
- [Color Theory Basics - Sarah Renae Clark](https://sarahrenaeclark.com/color-theory-basics/)

### Animation
- [Robert Penner's Easing Functions](https://robertpenner.com/easing/)
- [Easing Functions Cheat Sheet](https://easings.net/)
- [Animating with Easing Functions - Kirupa](https://www.kirupa.com/html5/animating_with_easing_functions_in_javascript.htm)

### Chaos & Fractals
- [Lorenz System Visualization - GitHub](https://github.com/gboeing/lorenz-system)
- [L-systems Wikipedia](https://en.wikipedia.org/wiki/L-system)
- [Nature of Code - Fractals](https://natureofcode.com/fractals/)

### Mathematical Art
- [Visualizing Prime Numbers - Medium](https://medium.com/itp-blog/math-for-artists-visualizing-prime-numbers-c1ca6a309be2)
- [Fibonacci and Golden Ratio - Cleveland Design](https://clevelanddesign.com/insights/the-nature-of-design-the-fibonacci-sequence-and-the-golden-ratio/)

---

## Next Steps

1. **Prototype** top 3 physical patterns (flow fields, wave equation, particle system)
2. **Profile** on ESP32 - measure FPS and memory
3. **Audio mapping** - connect to existing audio analysis pipeline
4. **Tree topology** - use sculptures.json to map physics to actual branch structure
5. **User testing** - show patterns on sculpture, get feedback on "physical" quality
