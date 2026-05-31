// v1 per-axis gyro telemetry wire packet — companion to TelemetryPacketV1.
//
// Sender:    festicorn/gyro-sense/src/sender.cpp
// Receivers: festicorn/biolum/
// Spec:      same windowing as v1 accel (8 samples @ 200 Hz = 25 Hz emit)
//
// 12 bytes, little-endian, broadcast over ESP-NOW at 25 Hz.
// Receivers disambiguate from TelemetryPacketV1 (16 bytes) by packet size.
//
// Encoding matches TelemetryPacketV1 conventions:
//   max/min are AC-coupled: (rawExtreme - rawMean) >> 8
//   mean is raw (bias-corrected but NOT AC-coupled) >> 8
//   gmag fields use sqrt-companding with FS = 57000 counts
//
// At ±1000 dps, one int8 LSB = 1000/128 ≈ 7.8 dps.

#ifndef GYRO_PACKET_V1_H
#define GYRO_PACKET_V1_H

#include <stdint.h>

struct __attribute__((packed)) GyroPacketV1 {
    uint16_t seq;
    int8_t   gx_max, gy_max, gz_max;
    int8_t   gx_min, gy_min, gz_min;
    int8_t   gx_mean, gy_mean, gz_mean;
    uint8_t  gmag_max;
};
static_assert(sizeof(GyroPacketV1) == 12, "gyro packet must be exactly 12 bytes");

#endif // GYRO_PACKET_V1_H
