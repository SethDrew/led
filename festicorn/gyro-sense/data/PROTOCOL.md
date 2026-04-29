# Sender Capture Protocol

Goal: collect labeled raw IMU + audio RMS traces from each physical mounting
condition so we can choose summary features (min/max/avg vs jerk vs hybrid)
from real data instead of theory.

## Hardware

- ESP32-C3 super-mini sender, MPU-6050 + INMP441 mic
- Recorder firmware: `[env:recorder]` in `platformio.ini`
- Inner sample rate: 200 Hz IMU, ~100 Hz audio RMS (latched)
- Storage: LittleFS `/rec.bin`, max ~3 minutes per session

## Three captures

| label     | mounting                                      | dominant motion modes              |
|-----------|-----------------------------------------------|------------------------------------|
| `duck`    | inside stuffed-animal totem, held by person   | hold, tap, drum, shake, dance      |
| `hard`    | inside hard rigid object, held by person      | hold, tap, drum, shake, dance      |
| `hanging` | dangling from end of LED strip (no contact)   | rest, gentle swing, hard swing, recoil, settle |

The `duck` vs `hard` pair isolates the role of damping/coupling material.
The `hanging` capture is the unattended-passive pendulum case.

## Recording flow

1. Flash recorder firmware (with USB cable):
   ```
   pio run -e recorder -t upload
   ```
2. Disconnect from laptop. Plug into a small USB power bank.
   The firmware sees no host and **starts recording automatically after 1 s**.
   (For laptop-attached debugging, plug in to laptop instead — drops you in
   the menu mode.)
3. Run the segment script (verbal countdown to yourself, ~10 s per segment):

### Segment menu — `duck` / `hard` (90 s total)

| start | end  | label-in-notes | action                                    |
|-------|------|----------------|-------------------------------------------|
| 0:00  | 0:10 | rest           | place on flat surface, do not touch       |
| 0:10  | 0:20 | hold           | hold gently, no intentional motion        |
| 0:20  | 0:35 | tap            | finger taps on body, ~1 per second        |
| 0:35  | 0:50 | drum           | rapid drumming both hands                 |
| 0:50  | 1:05 | shake          | vigorous shake side-to-side               |
| 1:05  | 1:20 | dance          | natural body motion / wave around         |
| 1:20  | 1:30 | rest           | place down again, settle                  |

### Segment menu — `hanging` (90 s total)

| start | end  | label-in-notes | action                                    |
|-------|------|----------------|-------------------------------------------|
| 0:00  | 0:15 | still          | hang motionless, no airflow               |
| 0:15  | 0:30 | gentle         | small push, ~5 cm amplitude               |
| 0:30  | 0:45 | hard           | hard push, full ~30 cm amplitude          |
| 0:45  | 1:00 | recoil         | several sharp pushes in succession        |
| 1:00  | 1:30 | settle         | release after one push, let it decay      |

4. Stop recording either by waiting out the 3-minute cap or by plugging into
   the laptop (drops to menu, which leaves the file intact).

## Harvest

Plug the sender into the laptop (USB-CDC). After the 3 s window the firmware
prints the menu. Then:

```
python harvest.py <label>
# e.g.
python harvest.py duck
python harvest.py hard
python harvest.py hanging
```

This dumps the binary to `data/recordings/<label>/<timestamp>.bin` and a
parsed CSV alongside. Pass `--erase` to clear the device after dump.

## Notes log

Keep a plain-text note per capture with stopwatch start times:

```
data/recordings/<label>/<timestamp>.notes.txt

t=0:00 rest
t=0:10 hold
t=0:20 tap (slow ~1 Hz)
t=0:35 drum (fast)
...
```

These align timestamps to segment labels for offline slicing.
