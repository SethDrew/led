// duck console wire packet — full state, self-contained every packet.
//
// Sender:    festicorn/axismundi/src/duck_sender.cpp
// Receivers: festicorn/axismundi/src/axismundi.cpp
// Hardware:  catalog role duck-sender (ESP32-C3, INMP441 mic + MPU-6050)
//
// 14 bytes, little-endian, ESP-NOW channel 1 broadcast at 25 Hz (no
// on-change gating: audio is continuous). 14 is deliberately unique among
// fixed-size packets (kettle 13, accel TelemetryPacketV1 16, clicky 18,
// gyro 12, audio 5) — receivers gate on BOTH size and magic/ver.
//
//   magic     0xD0CC
//   ver       1
//   seq       send counter (wraps)
//   rms_mean  sqrt-companded mean mic RMS over the 40 ms window
//   rms_max   sqrt-companded max mic RMS over the window
//             encoder: byte = clamp_u8(sqrt(val/DUCK_RMS_FS) * 255)
//             decoder: val  = (byte/255)² * DUCK_RMS_FS
//   ax/ay/az  window-mean accel (rawMean >> 8) — the gravity vector, i.e.
//             which way the duck is turned. Only its DIRECTION is specified;
//             the counts-per-g scale depends on the sender's full-scale
//             setting, so receivers must use ratios (angles), never magnitude.
//   flags     bit0 = IMU absent (ax/ay/az are 0). Other bits reserved.
//   upMs      sender uptime; a regression means the sender rebooted.
//
//   Receivers: #include this header, memcpy the payload into a local struct,
//   check magic + ver before using any field.

#ifndef DUCK_PACKET_V1_H
#define DUCK_PACKET_V1_H

#include <stdint.h>

#define DUCK_MAGIC 0xD0CC
#define DUCK_VER   1
#define DUCK_RMS_FS 200000.0f

#define DUCK_FLAG_NO_IMU 0x01

struct __attribute__((packed)) DuckPacketV1 {
    uint16_t magic;
    uint8_t  ver;
    uint8_t  seq;
    uint8_t  rms_mean;
    uint8_t  rms_max;
    int8_t   ax;
    int8_t   ay;
    int8_t   az;
    uint8_t  flags;
    uint32_t upMs;
};
static_assert(sizeof(DuckPacketV1) == 14, "duck packet must be exactly 14 bytes");

#endif
