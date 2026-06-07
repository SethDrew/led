// v1 audio telemetry wire packet — companion to TelemetryPacketV1 / GyroPacketV1.
//
// Sender:    festicorn/original-duck/src/sender_v2.cpp
// Receivers: (TBD)
//
// 5 bytes, little-endian, broadcast over ESP-NOW at 25 Hz.
// Receivers disambiguate from the IMU packets (16 B / 12 B) by packet size.
//
// Each 40 ms window the sender reads the INMP441, computes per-window audio
// RMS, and tracks the window max and mean across consecutive reads. Both are
// sqrt-companded into uint8 using the same encoder as the accel amag field
// (byte = clamp_u8( sqrt(val/FS) * 255 )), so a receiver decodes with
// val = (byte/255)² * FS.
//
//   seq        shared 25 Hz tick counter (matches accel/gyro packets, wraps)
//   rms_mean   sqrt-companded mean RMS over the window
//   rms_max    sqrt-companded max RMS over the window
//   flags      bit0 = mic muted (shake-toggle on sender; rms fields are 0
//              when set). Other bits reserved (0).

#ifndef AUDIO_PACKET_V1_H
#define AUDIO_PACKET_V1_H

#include <stdint.h>

struct __attribute__((packed)) AudioPacketV1 {
    uint16_t seq;
    uint8_t  rms_mean;
    uint8_t  rms_max;
    uint8_t  flags;
};
static_assert(sizeof(AudioPacketV1) == 5, "audio packet must be exactly 5 bytes");

#endif // AUDIO_PACKET_V1_H
