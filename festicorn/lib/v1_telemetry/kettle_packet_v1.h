// kettle-knob console wire packet — full state, self-contained every packet.
//
// Sender:    festicorn/kettle-knob/src/main.cpp
// Receivers: festicorn/axismundi/src/axismundi.cpp
// Hardware:  festicorn/kettle-knob/README.md; catalog role kettle-knob
//
// 13 bytes, little-endian, ESP-NOW channel 1 broadcast, on-change (coalesced
// to <=50 Hz) + 1 s heartbeat. 13 is deliberately unique among fixed-size
// packets (accel TelemetryPacketV1 is 16, ClickyPacketV1 is 18) — receivers
// gate on BOTH size and magic/ver.
//
//   magic     0xEC11 (the encoder part family)
//   ver       1
//   seq       send counter (wraps)
//   btnBits   bit 0 = knob push (GPIO 21), then GPIO 9, 7, 6, 5 on bits 1-4
//   enc       cumulative quadrature count, signed; 2 counts per detent,
//             30 detents per revolution. Receivers consume deltas.
//   upMs      sender uptime; a regression means the sender rebooted and enc
//             restarted from 0 — receivers must rebase, not integrate the jump.
//
//   Receivers: #include this header, memcpy the payload into a local struct,
//   check magic + ver before using any field.

#ifndef KETTLE_PACKET_V1_H
#define KETTLE_PACKET_V1_H

#include <stdint.h>

#define KETTLE_MAGIC 0xEC11
#define KETTLE_VER   1
#define KETTLE_NUM_BTNS 5
#define KETTLE_COUNTS_PER_DETENT 2
#define KETTLE_DETENTS_PER_REV   30

#define KETTLE_BTN_HOLD   0   // knob push: freeze energy decay
#define KETTLE_BTN_TOP    1   // GPIO 9: wave, top 50 LEDs
#define KETTLE_BTN_UPMID  2   // GPIO 7: wave, upper middle
#define KETTLE_BTN_LOMID  3   // GPIO 6: wave, lower middle
#define KETTLE_BTN_BOTTOM 4   // GPIO 5: wave, bottom 20 LEDs

struct __attribute__((packed)) KettlePacketV1 {
    uint16_t magic;
    uint8_t  ver;
    uint8_t  seq;
    uint8_t  btnBits;
    int32_t  enc;
    uint32_t upMs;
};

#endif
