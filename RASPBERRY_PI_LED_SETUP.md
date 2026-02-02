# Raspberry Pi LED Strip Control Guide

Complete instructions for controlling your 150 LED strip from the Raspberry Pi 4 Model B.

## Two Approaches

### Option A: Streaming to Arduino/Microcontroller (Recommended for Complex Effects)
- **Pros**: Keeps your existing Arduino code, more stable timing, can swap controllers easily
- **Cons**: Requires Arduino/microcontroller, uses USB port, more hardware

### Option B: Direct GPIO Control (Simpler Hardware)
- **Pros**: No Arduino needed, one less component, native Pi control
- **Cons**: Timing can be affected by OS scheduling, requires running as root

---

## Option A: Streaming to Arduino via USB

### Hardware Setup
1. **Connect Arduino to Raspberry Pi**
   - Plug Arduino USB cable into any Pi USB port
   - Arduino powers from Pi USB (draws ~100-200mA)

2. **Arduino connects to LED strip** (as you have it now)
   - Arduino data pin → LED strip data pin
   - External 5V power supply → LED strip VCC/GND
   - Arduino GND → Power supply GND (common ground!)

### Software Setup on Raspberry Pi

#### Step 1: Install Dependencies
```bash
ssh pi@raspberrypi.local

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3-pip python3-numpy python3-serial

# Install via pip
pip3 install pyserial numpy
```

#### Step 2: Copy Streaming Code to Pi
From your Mac:
```bash
# Copy the streaming controller to Pi
scp ~/Documents/led/streaming/stream_controller.py pi@raspberrypi.local:~/

# Or copy entire streaming directory
scp -r ~/Documents/led/streaming pi@raspberrypi.local:~/led-streaming/
```

#### Step 3: Find Arduino Serial Port on Pi
On the Pi:
```bash
# List USB serial devices
ls -l /dev/ttyUSB* /dev/ttyACM*

# You'll see something like:
# /dev/ttyACM0  or  /dev/ttyUSB0
```

#### Step 4: Set Serial Port Permissions
```bash
# Add pi user to dialout group (for serial access)
sudo usermod -a -G dialout pi

# Log out and back in for group change to take effect
exit
ssh pi@raspberrypi.local
```

#### Step 5: Test the Connection
```bash
# Check Arduino is detected
dmesg | tail -20

# Should see something like:
# [  123.456789] cdc_acm 1-1.3:1.0: ttyACM0: USB ACM device
```

#### Step 6: Run the Streaming Controller
```bash
cd ~/led-streaming

# Run with your settings (adjust port if needed)
python3 stream_controller.py \
  --port /dev/ttyACM0 \
  --leds 150 \
  --brightness 0.3 \
  --fps 60 \
  --orbs 5 \
  --speed 1.0
```

#### Step 7: Auto-Start on Boot (Optional)
Create a systemd service to start automatically:

```bash
sudo nano /etc/systemd/system/led-stream.service
```

Paste this:
```ini
[Unit]
Description=LED Streaming Controller
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/led-streaming
ExecStart=/usr/bin/python3 /home/pi/led-streaming/stream_controller.py --port /dev/ttyACM0 --leds 150 --brightness 0.3 --fps 60
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable led-stream.service
sudo systemctl start led-stream.service

# Check status
sudo systemctl status led-stream.service

# View logs
sudo journalctl -u led-stream.service -f
```

---

## Option B: Direct GPIO Control (No Arduino)

### Hardware Setup
1. **Wire LED strip directly to Pi**
   - LED strip data pin → Pi GPIO 18 (Pin 12, PWM0)
   - LED strip GND → Pi GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
   - LED strip VCC → External 5V power supply (NOT Pi!)
   - Pi GND → Power supply GND (common ground!)

**CRITICAL**: Never power 150 LEDs from the Pi's 5V pins! Use external power supply.

### Wiring Diagram
```
External 5V PSU (5V 5A+)
  ├─ VCC ──→ LED Strip VCC (Red)
  └─ GND ──→ LED Strip GND (White/Black)
              └─ Also connect to Pi GND (Pin 6)

Raspberry Pi
  └─ GPIO 18 (Pin 12) ──→ LED Strip Data (Green/Yellow)
```

### Software Setup on Raspberry Pi

#### Step 1: Install rpi_ws281x Library
```bash
ssh pi@raspberrypi.local

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-dev python3-numpy scons swig

# Install WS281x library
sudo pip3 install rpi_ws281x
```

#### Step 2: Create LED Control Script
```bash
cd ~
mkdir led-control
cd led-control
nano nebula_gpio.py
```

Paste this code:
```python
#!/usr/bin/env python3
"""
Direct GPIO LED control for Raspberry Pi
Runs the Nebula effect directly on the Pi
"""

import time
import numpy as np
from rpi_ws281x import PixelStrip, Color
import argparse

# LED strip configuration:
LED_COUNT = 150          # Number of LED pixels
LED_PIN = 18             # GPIO pin (must support PWM)
LED_FREQ_HZ = 800000     # LED signal frequency (Hz)
LED_DMA = 10             # DMA channel
LED_BRIGHTNESS = 128     # Brightness (0-255)
LED_INVERT = False       # Invert signal
LED_CHANNEL = 0          # PWM channel

class NebulaEffect:
    """Nebula effect - breathing waves with glowing orbs"""
    def __init__(self, num_pixels: int, speed_multiplier: float = 1.0, tail_length: float = 15.0, max_orbs: int = 5):
        self.num_pixels = num_pixels
        self.elapsed_time = 0.0
        self.speed = speed_multiplier

        # Background parameters
        self.BREATH_FREQUENCY = 0.0105 * speed_multiplier
        self.BREATH_CENTER = 51
        self.BREATH_AMPLITUDE = 38
        self.SPATIAL_AMPLITUDE = 51
        self.SPATIAL_SPEED = 0.006 * speed_multiplier
        self.BACKGROUND_MAX = 153

        # Orbs with decay buffer
        self.max_orbs = max_orbs
        self.orb_size = tail_length
        self.orb_base_speed = 0.45
        self.orbs = []
        self.orb_brightness = np.zeros(num_pixels, dtype=np.float32)

    def update(self, dt: float) -> np.ndarray:
        """Calculate and return next frame"""
        self.elapsed_time += dt
        frame = np.zeros((self.num_pixels, 3), dtype=np.uint8)

        # Background: Breathing waves
        t = self.elapsed_time * 60.0
        breathing = self.BREATH_CENTER + self.BREATH_AMPLITUDE * np.sin(t * self.BREATH_FREQUENCY)

        positions = np.arange(self.num_pixels) / self.num_pixels
        phases = positions + t * self.SPATIAL_SPEED
        spatial = self.SPATIAL_AMPLITUDE * (0.5 + 0.5 * np.cos(2.0 * np.pi * phases))

        bg_brightness = np.clip(breathing + spatial, 0, self.BACKGROUND_MAX).astype(np.uint8)

        # Color variation (blue to magenta)
        color_phases = positions * 2.0 * np.pi + t * 0.009
        color_shift = 0.5 + 0.5 * np.sin(color_phases)

        bg_r = (20 + color_shift * 235).astype(np.uint8)
        bg_g = (30 - color_shift * 20).astype(np.uint8)
        bg_b = (255 - color_shift * 125).astype(np.uint8)

        frame[:, 0] = ((bg_r.astype(np.uint16) * bg_brightness) >> 8).astype(np.uint8)
        frame[:, 1] = ((bg_g.astype(np.uint16) * bg_brightness) >> 8).astype(np.uint8)
        frame[:, 2] = ((bg_b.astype(np.uint16) * bg_brightness) >> 8).astype(np.uint8)

        # Foreground: Glowing orbs with decay trails
        decay_per_frame_at_60fps = 1.0 - (1.0 / self.orb_size)
        decay_per_second = decay_per_frame_at_60fps ** 60.0
        time_based_decay = decay_per_second ** dt
        self.orb_brightness *= time_based_decay
        self.orb_brightness[self.orb_brightness < 0.01] = 0.0

        # Spawn orbs
        if len(self.orbs) < self.max_orbs and np.random.rand() < 0.03:
            self.orbs.append({
                'position': np.random.randint(0, self.num_pixels),
                'velocity': np.random.choice([-1, 1]) * self.orb_base_speed * self.speed * np.random.uniform(0.7, 1.3),
                'age': 0,
                'lifetime': np.random.randint(100, 300)
            })

        # Update and render orbs
        orbs_to_remove = []
        for orb in self.orbs:
            orb['age'] += 1
            orb['position'] += orb['velocity']
            orb['position'] %= self.num_pixels

            if orb['age'] >= orb['lifetime']:
                orbs_to_remove.append(orb)
                continue

            lifecycle = orb['age'] / orb['lifetime']
            if lifecycle < 0.4:
                t_fade = lifecycle / 0.4
                brightness = t_fade * t_fade * (3.0 - 2.0 * t_fade)
            elif lifecycle > 0.6:
                t_fade = (1.0 - lifecycle) / 0.4
                brightness = t_fade * t_fade * (3.0 - 2.0 * t_fade)
            else:
                brightness = 1.0

            pixel = int(orb['position'])
            if 0 <= pixel < self.num_pixels:
                self.orb_brightness[pixel] = min(1.0, self.orb_brightness[pixel] + brightness * 0.6)

        for orb in orbs_to_remove:
            self.orbs.remove(orb)

        # Render orb layer (warm white)
        orb_mask = self.orb_brightness > 0.01
        star_r = (255 * self.orb_brightness[orb_mask]).astype(np.uint16)
        star_g = (240 * self.orb_brightness[orb_mask]).astype(np.uint16)
        star_b = (200 * self.orb_brightness[orb_mask]).astype(np.uint16)

        frame[orb_mask, 0] = np.clip(frame[orb_mask, 0].astype(np.uint16) + star_r, 0, 255).astype(np.uint8)
        frame[orb_mask, 1] = np.clip(frame[orb_mask, 1].astype(np.uint16) + star_g, 0, 255).astype(np.uint8)
        frame[orb_mask, 2] = np.clip(frame[orb_mask, 2].astype(np.uint16) + star_b, 0, 255).astype(np.uint8)

        return frame

def main():
    parser = argparse.ArgumentParser(description='Direct GPIO LED control for Raspberry Pi')
    parser.add_argument('--leds', type=int, default=150, help='Number of LEDs')
    parser.add_argument('--fps', type=int, default=60, help='Target frames per second')
    parser.add_argument('--brightness', type=float, default=0.3, help='Brightness (0.0-1.0)')
    parser.add_argument('--speed', type=float, default=1.0, help='Animation speed multiplier')
    parser.add_argument('--tail-length', type=float, default=15.0, help='Orb tail length')
    parser.add_argument('--orbs', type=int, default=5, help='Maximum number of orbs')
    args = parser.parse_args()

    print(f"LED Direct GPIO Control")
    print(f"  LEDs: {args.leds}")
    print(f"  Target FPS: {args.fps}")
    print(f"  Brightness: {args.brightness * 100}%")
    print(f"  Speed: {args.speed}x")
    print(f"  Tail Length: {args.tail_length}")
    print(f"  Max Orbs: {args.orbs}")
    print()

    # Create LED strip object
    strip = PixelStrip(args.leds, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, int(args.brightness * 255), LED_CHANNEL)
    strip.begin()

    # Create effect
    effect = NebulaEffect(args.leds, speed_multiplier=args.speed, tail_length=args.tail_length, max_orbs=args.orbs)

    frame_time = 1.0 / args.fps
    print("Running... Press Ctrl+C to stop")

    try:
        frame_count = 0
        start_time = time.time()
        last_frame_time = start_time

        while True:
            loop_start = time.time()

            if frame_count == 0:
                dt = frame_time
            else:
                dt = time.time() - last_frame_time
            last_frame_time = time.time()

            # Calculate frame
            frame = effect.update(dt)

            # Send to LED strip
            for i in range(args.leds):
                strip.setPixelColor(i, Color(int(frame[i][1]), int(frame[i][0]), int(frame[i][2])))  # GRB order
            strip.show()

            frame_count += 1

            # Print stats every second
            if frame_count % args.fps == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"Frames: {frame_count}, Actual FPS: {actual_fps:.1f}")

            # Frame timing
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
        # Turn off all LEDs
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()

    finally:
        print("Done!")

if __name__ == '__main__':
    main()
```

Save and make executable:
```bash
chmod +x nebula_gpio.py
```

#### Step 3: Test GPIO Control
```bash
# Must run as root for GPIO access
sudo python3 nebula_gpio.py \
  --leds 150 \
  --brightness 0.3 \
  --fps 60 \
  --orbs 5
```

#### Step 4: Auto-Start on Boot (Optional)
```bash
sudo nano /etc/systemd/system/led-gpio.service
```

Paste:
```ini
[Unit]
Description=LED GPIO Controller
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/led-control
ExecStart=/usr/bin/python3 /home/pi/led-control/nebula_gpio.py --leds 150 --brightness 0.3 --fps 60
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable led-gpio.service
sudo systemctl start led-gpio.service
sudo systemctl status led-gpio.service
```

---

## Troubleshooting

### Option A (Arduino Streaming)
**Problem**: Can't find Arduino port
```bash
# Check if Arduino is detected
lsusb
dmesg | grep tty

# Try both port types
python3 stream_controller.py --port /dev/ttyACM0
python3 stream_controller.py --port /dev/ttyUSB0
```

**Problem**: Permission denied on serial port
```bash
# Add user to dialout group
sudo usermod -a -G dialout pi
# Log out and back in
```

### Option B (GPIO Direct)
**Problem**: LEDs not lighting up
- Check you're running as root (`sudo`)
- Verify GPIO 18 is connected to data pin
- Ensure common ground between Pi and power supply
- Check LED strip is getting 5V power from external supply

**Problem**: LEDs flickering or wrong colors
- Add a 330Ω resistor between GPIO 18 and data pin
- Add a 1000µF capacitor across power supply
- Shorten data wire between Pi and first LED
- Check power supply can deliver enough current (5A+ for 150 LEDs)

**Problem**: Poor performance / low FPS
- Lower FPS target: `--fps 30`
- Reduce orbs: `--orbs 2`
- Lower brightness: `--brightness 0.2`

---

## Power Consumption Reference

At 60% brightness, full white, you measured **1.6A** for 150 LEDs.

**Estimated current draw:**
- 25% brightness: ~0.7A
- 50% brightness: ~1.3A
- 60% brightness: ~1.6A (measured)
- 100% brightness: ~2.6A

**Recommended power supplies:**
- Minimum: 5V 3A
- Recommended: 5V 5A
- Ideal: 5V 10A (headroom for future expansion)

---

## Which Option Should You Choose?

**Choose Option A (Arduino Streaming) if:**
- You want maximum stability and precise timing
- You already have Arduino code working
- You want to easily swap between different microcontrollers
- You need the Pi for other tasks while LEDs run

**Choose Option B (Direct GPIO) if:**
- You want the simplest hardware setup
- You don't mind running as root
- The Pi is dedicated to LED control
- You want native Python control without serial overhead

Both work great! Option A is more robust, Option B is simpler.
