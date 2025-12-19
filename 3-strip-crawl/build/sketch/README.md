#line 1 "/Users/KO16K39/Documents/led/README.md"
# LED Development Project with Wokwi

This project is set up for developing and testing addressable LED strip code using Wokwi simulator before deploying to physical hardware.

## Project Structure

```
led/
├── led.ino           # Main Arduino sketch for LED control
├── diagram.json      # Wokwi circuit diagram (Arduino + NeoPixel strip)
├── libraries.txt     # Required libraries for Wokwi
└── README.md        # This file
```

## Current Setup

- **Microcontroller**: Arduino Uno
- **LED Type**: NeoPixel/WS2812B addressable LED strip
- **Number of LEDs**: 8 (configurable in diagram.json and sketch.ino)
- **Data Pin**: Digital Pin 6
- **Library**: Adafruit NeoPixel

## Getting Started with Wokwi

### Option 1: Online Simulator (Recommended for Quick Testing)

1. Go to https://wokwi.com
2. Create a new Arduino Uno project
3. Copy the contents of `led.ino` into the code editor
4. Click "diagram.json" tab and paste the contents of `diagram.json`
5. Click "Library Manager" and add "Adafruit NeoPixel"
6. Click the green "Start Simulation" button

### Option 2: VS Code Extension (Recommended for Development)

1. Install VS Code if you haven't already
2. Install the "Wokwi Simulator" extension from the VS Code marketplace
3. Open this project folder in VS Code
4. Press `F1` and type "Wokwi: Start Simulator"
5. The simulation will run using the local files

## Customizing Your LED Setup

### Change Number of LEDs

1. Edit `diagram.json` - change the `"count"` value in the neopixels part
2. Edit `led.ino` - change `#define NUM_PIXELS 8` to your desired count

### Change Data Pin

1. Edit `diagram.json` - change the `"pin"` value
2. Edit `led.ino` - change `#define LED_PIN 6` to match

### Adjust Brightness

In `led.ino`, modify the line:
```cpp
strip.setBrightness(50); // Range: 0-255
```

## Sample Animations Included

The starter code includes several animations:
- **Rainbow Cycle**: Flowing rainbow pattern
- **Color Wipe**: Fill strip with solid colors
- **Theater Chase**: Marquee-style chase effect

## Developing Your Own Patterns

To create custom LED patterns, add new functions following this template:

```cpp
void myCustomPattern() {
  for(int i = 0; i < NUM_PIXELS; i++) {
    // Set color for pixel i
    strip.setPixelColor(i, strip.Color(red, green, blue));
  }
  strip.show(); // Update the strip
  delay(50);    // Wait before next change
}
```

Then call `myCustomPattern()` in the `loop()` function.

## Moving to Physical Hardware

When ready to deploy to real hardware:

1. Connect your Arduino to your computer via USB
2. Install Arduino IDE (https://www.arduino.cc/en/software)
3. Install the Adafruit NeoPixel library:
   - Open Arduino IDE
   - Go to Sketch → Include Library → Manage Libraries
   - Search for "Adafruit NeoPixel" and install it
4. Wire your LED strip:
   - Strip DIN → Arduino Pin 6
   - Strip GND → Arduino GND
   - Strip VCC → External 5V power supply (for strips with many LEDs)
     - Note: Arduino 5V pin can power ~8-10 LEDs, use external power for more
5. Open `led.ino` in Arduino IDE
6. Select your board (Tools → Board → Arduino Uno)
7. Select your port (Tools → Port)
8. Click Upload

## Important Notes for Physical Hardware

- **Power**: LED strips can draw significant current. For more than 10 LEDs, use an external 5V power supply
- **Capacitor**: Add a 1000µF capacitor between VCC and GND on the strip
- **Resistor**: Add a 330Ω resistor between Arduino data pin and strip DIN
- **Common Ground**: Always connect Arduino GND to power supply GND

## Resources

- Wokwi Documentation: https://docs.wokwi.com
- Adafruit NeoPixel Guide: https://learn.adafruit.com/adafruit-neopixel-uberguide
- Arduino Reference: https://www.arduino.cc/reference/en

## Next Steps

1. Test the current setup in Wokwi simulator
2. Experiment with different colors and patterns
3. Create custom animations for your use case
4. Scale up the number of LEDs in the simulation
5. Order physical components and deploy to hardware
