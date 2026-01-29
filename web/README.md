# LED Effects Gallery

A web-based preview and deployment interface for your LED effects library.

## Features

- **Separated Effect Categories**: Background and foreground effects are organized separately
- **Preset Combinations**: Pre-configured animation combinations
- **Two-Button Control**: Each effect has both Wokwi simulation and Arduino upload buttons
- **Screenshot Placeholders**: Ready for your effect previews

## Directory Structure

```
web/
├── index.html          # Main gallery page
├── styles.css          # Styling
├── script.js           # Interactive functionality
├── assets/
│   └── previews/       # Place your screenshots/recordings here
└── README.md           # This file
```

## How to Use

### 1. View the Gallery

Open `index.html` in your browser:

```bash
cd web
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or simply double-click index.html in your file explorer
```

### 2. Add Effect Previews

To replace the placeholder images:

1. Open Wokwi with each animation
2. Take screenshots or screen recordings
3. Save them in `web/assets/previews/` with these names:
   - Background effects: `nebula.png`, `solid-color.png`, `pulsing-color.png`
   - Foreground effects: `crawling-stars.png`, `sparks.png`, `collision.png`, etc.
   - Preset combos: `nebula-stars.png`, `nebula-sparks.png`, etc.

4. Update the `src` attributes in `index.html` to point to your images

### 3. Button Functionality (To Be Implemented)

Currently, the buttons are set to `disabled`. The planned functionality:

#### Run in Wokwi Button
- Modifies `effects.ino` to select the animation combo
- Compiles the code
- Opens/refreshes Wokwi simulation

#### Upload to Arduino Button
- Modifies `effects.ino` to select the animation
- Uploads using `pio run --target upload`

## Next Steps

1. **Capture Screenshots**: Use Wokwi to capture each effect
2. **Implement Backend**: Add server-side code to handle:
   - File modification (commenting/uncommenting animations)
   - Compilation with PlatformIO
   - Upload to Arduino
3. **Enable Buttons**: Remove the `disabled` attribute once functionality is ready

## Effect Categories

### Background Effects (REPLACE blend)
- Nebula: Breathing waves with color shifts
- Solid Color: Static solid color background
- Pulsing Color: Pulsing solid color effect

### Foreground Effects (ADD blend)
- Crawling Stars: Glowing orbs that drift
- Sparks: Random spark explosions
- Collision: Crawlers that collide and explode
- Rainbow Circle: Rainbow circle passing through
- Enhanced Crawl: Smooth wave with color modes
- Fragmentation: White crawler decomposes to RGB
- Drifting Decay: White crawlers drift and decay

### Preset Combinations
- Nebula + Stars
- Nebula + Sparks
- Nebula + Collision
- Purple + Rainbow
- Pulsing + Crawl
- Fragmentation
- Drifting Decay
- Rainbow Circle Only
