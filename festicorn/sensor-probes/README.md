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

- **`bno08x_probe`** — GY-BNO08X 9-DOF fusion IMU (BNO080/BNO085, I2C 0x4A/0x4B,
  SDA=2/SCL=3/CS=0/ADD=1, INT unwired). Streams roll-pitch-yaw, linear accel,
  gyro, step count and stability class at 10 Hz, and prints on-chip tap/shake
  events as they fire. Until the sensor answers it scans at four bus speeds
  every 6 s, then runs a deep diagnostic: per-pin pullup rise-time measurement
  and a UART-RVC sniff that catches a part whose PS1:PS0 latched into serial
  mode instead of I2C. Companion envs: `bno08x_wires` (live per-wire
  continuity readout for reseating joints), `bno08x_meter` (pin self-check +
  static SCL hold for multimeter work). Streams JSON lines consumed by
  `tools/bno08x_dash.html` — a live panel per on-chip capability: fusion
  cube, compass, linear-accel vs gravity, gyro, mag, tap/shake events, steps,
  stability + activity classifiers, host-side 0.5 g movement gate, INMP441
  level over its learned floor.

  The dashboard connects over Web Serial (Chrome/Edge). It holds the port
  exclusively, so close the tab before flashing or running a serial monitor.

```bash
../../.venv/bin/pio run -e tcs3472_probe -t upload
../../.venv/bin/pio run -e bno08x_probe -t upload
```

The pulldown reading is the one to trust for "is this wire connected" — a
floating ESP32 input reads HIGH often enough to look like a healthy pullup.
