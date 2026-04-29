// v1 IMU telemetry wire packet — single source of truth.
//
// Sender:    festicorn/gyro-sense/src/sender.cpp
// Receivers: festicorn/biolum/, festicorn/bench-bulbs/
// Spec:      engineering/ledger.yaml entry `bulb-imu-telemetry-wire-schema-v1`
//            festicorn/gyro-sense/data/recordings/V1_SCHEMA_DERIVATION.md
//
// 16 bytes, little-endian, broadcast over ESP-NOW at 25 Hz.
//
// IMPORTANT — field grouping is PER-STAT, not per-axis. The on-the-wire byte
// order is (max-of-all-axes, then min-of-all-axes, then mean-of-all-axes),
// NOT (everything-about-x, everything-about-y, everything-about-z). A
// per-axis layout will compile and parse but yield garbage values; gravity
// (a large number that lives in ax_mean) will be read into az_max and fire
// phantom taps when the controller is rotated. See the retro in the bug
// log for the full incident.
//
// Semantics:
//   ax_max / ay_max / az_max   AC-coupled per-axis max  ((rawMax - rawMean) >> 8)
//   ax_min / ay_min / az_min   AC-coupled per-axis min  ((rawMin - rawMean) >> 8)
//   ax_mean / ay_mean / az_mean   raw per-axis mean     (rawMean >> 8)
//                                   — gravity INCLUDED
//   amag_max / amag_mean       sqrt-companded |a| (raw, gravity included),
//                               full-scale AMAG_FS = 57000 counts
//   gmag_max / gmag_mean       sqrt-companded |gyro|,
//                               full-scale GMAG_FS = 57000 counts
//   flags                      hi-nibble accel-clip count (0..8),
//                              lo-nibble gyro-saturate count (0..8)
//
// Counts → physical units depends on the sender's configured IMU range.
// Production sender locks ±4g / ±1000 dps (see ACCEL_RANGE_G / GYRO_RANGE_DPS
// in sender.cpp). Receivers convert via:
//   counts_per_g  = 32768 / ACCEL_RANGE_G
//   ax_max_g      = ax_max * 256.0f / counts_per_g
//
// Rules of engagement for receivers:
//   • #include this header. Do NOT redeclare the struct.
//   • In onReceive callbacks, gate on `len == sizeof(TelemetryPacketV1)` so
//     coexistence with other packet types (e.g. duck v0.1 SensorPacket = 15 B)
//     is safe.

#ifndef V1_PACKET_H
#define V1_PACKET_H

#include <stdint.h>

struct __attribute__((packed)) TelemetryPacketV1 {
    uint16_t seq;
    int8_t   ax_max,  ay_max,  az_max;
    int8_t   ax_min,  ay_min,  az_min;
    int8_t   ax_mean, ay_mean, az_mean;
    uint8_t  amag_max;
    uint8_t  amag_mean;
    uint8_t  gmag_max;
    uint8_t  gmag_mean;
    uint8_t  flags;
};
static_assert(sizeof(TelemetryPacketV1) == 16, "v1 packet must be exactly 16 bytes");

#endif // V1_PACKET_H
