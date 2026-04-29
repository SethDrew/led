# Capture firmware (test fixtures)

These two `.cpp` files are NOT production firmware. They are dataset-capture
test fixtures that produced several of the streaming recordings under
`data/recordings/`.

- `stream_sender_test.cpp` — ESP32-C3 sender. Reads MPU-6050 IMU samples and
  broadcasts batched frames over ESP-NOW.
- `stream_bridge_test.cpp` — classic ESP32 receiver. Forwards the binary
  frames over USB serial so a host script can persist them to disk.

Pairs with `stream_capture.py` at the project root.

Kept here for dataset reproducibility — anyone re-creating the streaming
captures needs the exact firmware that produced them. The PlatformIO envs
`stream_sender`, `stream_sender_ext`, and `stream_bridge` in
`../../platformio.ini` build directly from these files via
`build_src_filter = -<*> +<../data/recordings/_capture_firmware/<file>>`.

Production sender lives at `src/sender.cpp` (env:sender).
