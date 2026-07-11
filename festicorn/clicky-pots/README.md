# clicky-pots

LED instrument input controller: 6 push-pull ("clicky") pots + 6 buttons
salvaged from a Faroudja LD100 front panel, read by an ESP32 and broadcast
over ESP-NOW (channel 1, `ClickyPacketV1` 18-byte binary struct — see
`lib/v1_telemetry/clicky_packet_v1.h`: magic+ver, seq, pull/btn bitfields,
6× u16 positions 0–10000; on-change + 1s heartbeat).
Wiring and pin maps: `festicorn/catalog/boards.yaml`
(role `clicky-pots`, board MAC `04:B2:47:82:FA:9C`).

## How it works

Each pot exposes two sense lines, both wired as 330Ω pull-up dividers to 3.3V
with the wiper grounded:

- **red** → ADS1115 channel. Position, valid in both switch states — but the
  push-pull switch reroutes the pot's internal network, so pressed-in and
  pulled-out have *different* resistance curves. Firmware keeps two
  calibration tables per pot and switches per sample.
- **white** → ESP32 ADC1 pin. State discriminator: strictly bimodal at rest
  (pressed ≥1.4V, pulled ≈0V; pot 3 is mirror-assembled and inverted).
  Anything in between means the switch is mechanically in flight and all
  reads are garbage.

Per sample, per pot: classify white → read ADS with the matching table →
re-classify white (bracket check; discard on mismatch) → reject if white was
in the forbidden band → quarantine any position step >0.15 until one
confirming sample. Result: no latency added to rotation, transition garbage
suppressed, pull/push cycles land on the same value.

## Calibration

Pulled-state tables are fitted from *no-rotation pull/push cycles*: the
position before and after a flip must be equal, giving ground-truth
constraints; pulled end stops anchor the line. If a knob develops a jump at
pull/push: record ~6 deliberate no-rotation cycles at spread rotations plus
full sweeps (`tools/record_serial.py`), fit, update `potCal*` in main.cpp.
The session recordings under `recordings/` are the source data for the
current tables.

## Tools

- `dashboard.py` — live web UI on :8081 (positions, pull, buttons, raw
  values). Owns the serial port: kill it before flashing or recording.
- `tools/record_serial.py <port> <out.jsonl>` — raw capture of the ~40Hz
  debug stream for calibration/diagnosis.
