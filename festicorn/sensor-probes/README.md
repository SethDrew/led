# sensor-probes

Standalone bench firmware for characterizing new I2C sensors on an
ESP32-C3-DevKitM-1 before they earn a home in an installation. Each probe is
its own `build_src_filter` env; flash one, read it over USB serial, move on.

Sibling diagnostic rigs: `console-bridge`, `espnow-bridge`, `topo-tester`,
`bench-rmt-test`. Sensor pipelines tied to a specific installation live with
that installation (e.g. the duck's IMU + gesture research in `gyro-sense`).

## Probes

- **`tcs3472_probe`** — TCS3472 RGB color sensor (I2C 0x29, SDA=3/SCL=2,
  onboard LED GPIO1). Streams raw + normalized RGBC over serial. Pair with
  `tools/tcs_color_web.html` (Web Serial API in Chrome) for a live swatch view.

```bash
../../.venv/bin/pio run -e tcs3472_probe -t upload
```
