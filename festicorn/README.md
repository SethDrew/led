ESP32 control platform and user interface design for addressable LED sculptures.

Firmware drives WS2812B strips with real-time effect switching, OKLCH color palettes, and gamma-corrected brightness. A web UI served from LittleFS provides wireless control. Python tools generate perceptually uniform color lookup tables and palette definitions.

## Hardware target

ESP32-C3 with OTA firmware updates over WiFi. Current build configuration targets "Butterfly" (150 LEDs, pin 1, 41-pixel offset).

## Structure

- `src/` — firmware: effect rendering, web server, WiFi/OTA
- `data/` — HTML user interfaces served from LittleFS
- `gen_palettes.py` — generates OKLCH hue-arc gradients and chroma sweeps
- `gen_oklch_varL_lut.py` — generates 256-entry rainbow LUT with hue-dependent lightness
- `analyze_oklch_lut.py` — analysis tool for LUT brightness and hue distribution
- `platformio.ini` — PlatformIO build configuration
