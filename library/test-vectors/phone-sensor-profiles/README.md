# Phone Sensor Profiles

Binary sensor packet recordings from the Zergling Android app (SensorService).
Used to calibrate RMS thresholds and QA effects via the desktop simulator.

## Format

Each `.bin` file contains raw 15-byte `SensorPacket` structs (little-endian):

| Offset | Type    | Field     |
|--------|---------|-----------|
| 0      | int16   | ax        |
| 2      | int16   | ay        |
| 4      | int16   | az        |
| 6      | int16   | gx        |
| 8      | int16   | gy        |
| 10     | int16   | gz        |
| 12     | uint16  | rawRms    |
| 14     | uint8   | micEnabled|

Optional `.times` sidecar: one timestamp per line (seconds since recording start).

## Recording tool

```bash
cd festicorn/road-bulbs/tools/sim
python3 record_sensor.py --duration 60 --out <name>.bin
```

## Profiling tool

```bash
python3 profile_mic.py --duration 60
```

## Recordings

| File | Date | Environment | Packets | Notes |
|------|------|-------------|---------|-------|
| s24_airport_ambient_30s.bin | 2026-05-17 | Airport terminal | 72 | Galaxy S24. Loud ambient + PA announcements. RMS: min=180, p75=1110, p95=2191, max=5295 |
| s24_quiet_home_60s.bin | 2026-05-18 | Quiet home | 575 | Galaxy S24. Full 60s: quiet→close talk→quiet→far talk. Unsplit master. |
| s24_home_quiet_10s.bin | 2026-05-18 | Quiet home | 96 | Galaxy S24. Silence. Mean=76, max=121 |
| s24_home_close_talk_10s.bin | 2026-05-18 | Quiet home | 96 | Galaxy S24. Talking directly into mic. Mean=738, max=11916 |
| s24_home_far_talk_10s.bin | 2026-05-18 | Quiet home | 95 | Galaxy S24. Talking from distance. Mean=382, max=3883 |

## Replay through simulator

```bash
cd festicorn/road-bulbs/tools/sim
./sim --effect sparkle --json < ../../library/test-vectors/phone-sensor-profiles/airport_ambient_30s.bin > frames.bin 2> diag.jsonl
python3 qa_analyze.py frames.bin --diag diag.jsonl --effect sparkle --timeline
```
