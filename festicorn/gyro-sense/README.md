# gyro-sense

The v1 IMU telemetry project: production sender firmware, fleet inventory,
LittleFS dataset recorder, ESP-NOW streaming pipeline, and the analysis
tools that drove the v1 wire schema. Pairs with the `biolum/` receiver.

Senders are ESP32-C3 boards with MPU-6050/clone IMUs that broadcast the
v1 16-byte `TelemetryPacketV1` over ESP-NOW at 25 Hz.

The fleet catalog at `festicorn/catalog/boards.yaml` is the registry —
one entry per physical board keyed by MAC; `tools/register.py` flashes a
board, captures boot + telemetry, and upserts the entry. See
`data/recordings/V1_SCHEMA_DERIVATION.md` for the schema design rationale.

## Canonical hardware

- Board: ESP32-C3 super-mini
- IMU: MPU-6050 or 6500-family clone, I2C addr `0x68`
- Wiring: SDA=GPIO8, SCL=GPIO7, AD0=GPIO20 (driven LOW at boot)
- Telemetry: 16 B/packet v1 schema at 25 Hz, ±4g / ±1000 dps
- Wi-Fi: scans for `cuteplant` at boot, ESP-NOW on the AP's channel,
  re-scans every 5 min

The single mic+gyro outlier board (different wiring) is NOT supported.

## Catalog

`festicorn/catalog/boards.yaml` is the authoritative list across all
festicorn projects (bulb-senders, duck-sender, rgbw-receiver, test
fixtures). One entry per physical board, keyed by MAC. Schema (v2):

```yaml
firmware_by_role:
  bulb-sender:
    source: festicorn/gyro-sense/src/sender.cpp
    packet_format: v1-telemetry-16B
  # ...one block per role

boards:
  - mac: 14:63:93:6E:93:B0
    id: ""                        # human-assigned tag once the board has a home
    chip: esp32-c3
    role: bulb-sender             # selects firmware via firmware_by_role
    i2c_addr: 0x68
    first_seen: <utc iso8601>
    last_validated:
      utc: <utc iso8601>
      port: /dev/cu.usbmodemXXXXXX
      channel: <int>              # AP channel sender locked onto
      imu_status: ok              # ok | not_found | error
      first_seq: <int>            # telemetry seq at start of capture window
      last_seq: <int>             # telemetry seq at end
      git_sha: <short>            # sender.cpp commit hash at flash time
    notes: ""
```

## Tools

`tools/register.py PORT [PORT ...]` — for each port: flash the v1 sender
firmware, capture boot + a few seconds of telemetry, upsert the resulting
record into `festicorn/catalog/boards.yaml` (matched by MAC). Prints a
pass/fail summary.

Example:

```
$ ../../.venv/bin/python tools/register.py /dev/cu.usbmodem114201 /dev/cu.usbmodem114301
```

Existing entries with the same MAC are updated; new MACs are appended.
The script never deletes — pruning is a manual edit.
