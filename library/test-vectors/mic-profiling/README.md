# Mic Profiling Test Vectors

INMP441 MEMS mic recordings from the ESP32-C3 sender board (MAC `38:44:BE:45:D9:CC`)
via `mic_profile` firmware streaming raw 16-bit signed mono PCM at 16kHz over USB serial.

Captured 2026-05-05 using `festicorn/original-duck/tools/mic_capture.py`.

## Recordings

| File | Duration | Scenario |
|------|----------|----------|
| `direct_voice.wav` | 30s | Speaking directly into mic, ~15cm |
| `laptop_distance.wav` | 48s | Speaking at laptop distance, ~50cm |
| `mic_same_room.wav` | 170s | Conversation in same room, ~3m |
| `mic_next_room_open.wav` | 232s | Conversation in next room, door open |
| `mic_profile_closed_door.wav` | 108s | Conversation in next room, door closed |
| `outdoor.wav` | 65s | Outdoor speech at varying distances (see segments below) |
| `two_speakers_20s.wav` | 20s | Two speakers at different distances simultaneously |

## Outdoor segments (timestamps into outdoor.wav)

| Segment | Start | End | Description |
|---------|-------|-----|-------------|
| Close | 13s | 20s | Speaking ~30cm from mic, with pauses |
| Medium | 25s | 30s | Speaking ~1-2m from mic |
| Far | 34s | 38s | Speaking ~5m+ from mic |

## Two speakers scenario (two_speakers_20s.wav)

| Time | What's happening |
|------|-----------------|
| 0-5s | Background person speaking only |
| ~5s | User says "okay", pause |
| 5-13s | Simultaneous conversation, user at laptop distance |
| 13-15s | User leans in to phone distance (~15cm) |
| 15-20s | User stops, background person continues |

## Reproducing

To capture new recordings:
```
.venv/bin/python festicorn/original-duck/tools/mic_capture.py \
  --port /dev/cu.usbmodem* --seconds 60 --out output.wav
```

Requires `mic_profile` firmware flashed on sender board:
```
.venv/bin/pio run -e mic_profile --target upload
```
