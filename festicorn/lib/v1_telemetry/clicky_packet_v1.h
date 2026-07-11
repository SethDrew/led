// clicky-pots panel-state wire packet — full state, self-contained every packet.
//
// Sender:    festicorn/clicky-pots/src/main.cpp
// Receivers: festicorn/tree-of-record/src/tree_of_record_eth.cpp
// Hardware:  festicorn/clicky-pots/README.md; catalog role clicky-pots
//
// 18 bytes, little-endian, ESP-NOW channel 1 broadcast, on-change + 1 s
// heartbeat. 18 is deliberately unique among fixed-size packets (accel
// TelemetryPacketV1 is 16) — receivers gate on BOTH size and magic/ver.
//
//   magic     0xC11C
//   ver       1
//   seq       send counter (wraps)
//   pullBits  bit i = pot i pulled out
//   btnBits   bit i = button i pressed
//   pos[6]    pot positions, 0..10000
//
//   Receivers: #include this header, memcpy the payload into a local struct,
//   check magic + ver before using any field.

#ifndef CLICKY_PACKET_V1_H
#define CLICKY_PACKET_V1_H

#include <stdint.h>

#define CLICKY_MAGIC 0xC11C
#define CLICKY_VER   1
#define CLICKY_NUM_POTS 6

struct __attribute__((packed)) ClickyPacketV1 {
    uint16_t magic;
    uint8_t  ver;
    uint8_t  seq;
    uint8_t  pullBits;
    uint8_t  btnBits;
    uint16_t pos[CLICKY_NUM_POTS];
};

#endif
