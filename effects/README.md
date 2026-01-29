# LED Effects - Modular System

Single source of truth for LED effects development and deployment.

## 🎯 Project Structure

```
effects/
├── platformio.ini         # Multi-environment configuration
├── src/                   # Source code
│   ├── effects.ino       # Main program
│   └── *.h               # Effect modules
├── diagram.json          # Wokwi circuit diagram
└── wokwi.toml           # Wokwi configuration
```

## 🔧 Hardware Configurations

This project supports multiple deployment targets via PlatformIO environments:

| Environment | LED Count | Brightness | Use Case |
|-------------|-----------|------------|----------|
| `wokwi` | 25 | 100% | Wokwi simulation |
| `nano_test` | 50 | 50% | Testing with partial strip |
| `nano_full` | 300 | 20% | Full 5m strip (60 LED/m) |

## 🚀 Deployment

### Quick Deploy

From anywhere in your terminal:

```bash
# Test with 50 LEDs (half brightness)
pio-test

# Deploy to full 300 LED strip (20% brightness)
pio-full

# Or use the old alias (defaults to full)
pio-upload
```

### Manual Deploy

From the `effects/` directory:

```bash
# Test deployment (50 LEDs)
pio run -e nano_test --target upload

# Full deployment (300 LEDs)
pio run -e nano_full --target upload

# Just compile without uploading
pio run -e nano_test
```

## 🛠️ Development Workflow

1. **Develop** → Edit effects in Wokwi or locally
2. **Test** → Deploy to Arduino with `pio-test` (50 LEDs)
3. **Production** → Deploy with `pio-full` (300 LEDs)

## 📝 Customizing LED Count

To change the LED count or brightness for any environment, edit `platformio.ini`:

```ini
[env:nano_custom]
platform = atmelavr
board = nanoatmega328new
framework = arduino
upload_port = /dev/cu.usbserial-1230
build_flags =
    -DNUM_PIXELS=150        # Your LED count
    -DGLOBAL_BRIGHTNESS_PERCENT=30  # Your brightness (0-100)
lib_deps =
    adafruit/Adafruit NeoPixel@^1.12.0
```

Then deploy: `pio run -e nano_custom --target upload`

## 🎨 Available Effects

### Background Effects (REPLACE blend)
- `NebulaBackground` - Breathing waves with color shifts
- `SolidColorBackground` - Static solid color
- `PulsingColorBackground` - Pulsing solid color

### Foreground Effects (ADD blend)
- `CrawlingStarsForeground` - Glowing orbs that drift
- `SparksForeground` - Random spark explosions
- `CollisionForeground` - Crawlers collide and explode
- `RainbowCircleForeground` - Rainbow circle passing through
- `EnhancedCrawlForeground` - Smooth wave with color modes
- `FragmentationForeground` - White crawler decomposes to RGB
- `DriftingDecayForeground` - White crawlers drift and decay

## 🔌 Hardware Info

- **Board**: Arduino Nano (ATmega328P, new bootloader)
- **LED Strip**: WS2812B (NeoPixel) @ 60 LEDs/m
- **Data Pin**: GPIO 6
- **USB Port**: `/dev/cu.usbserial-1230`

## 📊 Memory Usage

**OPTIMIZED for 300 LEDs!** ✨ Uses `uint8_t` instead of `float` (75% memory savings)

| Environment | RAM Used | Flash Used | Status |
|-------------|----------|------------|--------|
| nano_test (50 LEDs) | 497 bytes (24%) | 8,062 bytes (26%) | ✅ Plenty of headroom |
| nano_full (300 LEDs) | ~1,750 bytes (85%) | 8,082 bytes (26%) | ✅ Safe for full strip |

See [OPTIMIZATION.md](OPTIMIZATION.md) for technical details on the memory optimization.

## ✨ Adding New Effects

1. Create `YourEffect.h` in `src/` inheriting from `BackgroundEffect` or `ForegroundEffect`
2. Implement `update()` and `render()` methods
3. Include the header in `effects.ino`
4. Instantiate and use in an animation function

## 🗑️ Old Project

The old PlatformIO project at `/Users/KO16K39/Documents/PlatformIO/Projects/led1` is no longer needed and can be safely deleted.
