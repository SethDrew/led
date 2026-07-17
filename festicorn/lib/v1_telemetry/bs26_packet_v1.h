// v1 BS-26 panel-state wire packet — full state, self-contained every packet.
//
// Sender:    festicorn/kaylab-knobs/src/main.cpp
// Receivers: festicorn/tree-of-record/src/tree_of_record_eth.cpp
//            festicorn/tree-of-record/src/main.cpp
//            festicorn/original-duck/src/sender_v3.cpp   (record switch only)
// Spec:      festicorn/kaylab-knobs/KNOB_REFERENCE.md (knob decode + field ranges)
//
// 38 bytes, little-endian, ESP-NOW channel 1: broadcast + unicast copies to
// cataloged consumers, on-change (×3 burst @ 40 ms) + 1 s heartbeat.
// Replaces the JSON broadcast (removed 2026-07-10) — same fields, same
// semantics, ~2.5× less airtime per frame.
//
// Receivers gate on BOTH size and magic/version — do NOT rely on size alone.
// Unlike the sensor packets, this format is expected to grow diagnostic
// fields; a future v2 changes sizeof, and the magic keeps any two ~38 B
// packets from colliding.
//
//   magic      0xB526
//   ver        1
//   flags      bit0 = dc switch, bit1 = ac switch (since June 2026 the dead
//              AC lever is mirrored from dc — see sender TEMP note)
//   hundreds/tens/ones      knob positions (0-12 / 0-10 / 0-10)
//   decouter/decmid/decinner decade ring positions
//   ua_dac/v_dac            meter DAC outputs (0-255)
//   dec        assembled decade value
//   raw_h/raw_t/raw_o       native-ADC raw knob readings (0-4095)
//   ads_t/ads_o/ads_vcc     ADS1115 raw readings (tens / ones / Vcc ref)
//   cdt        last debounce pending→commit interval, ms (-1 = none yet)
//   up         sender millis()
//   seq        send counter
//   boots      sender boot count
//
//   Rules of engagement for receivers:
//   • #include this header. Do NOT redeclare the struct.
//   • memcpy the payload into a local struct (no aligned-pointer casts),
//     then check magic + ver before using any field.

#ifndef BS26_PACKET_V1_H
#define BS26_PACKET_V1_H

#include <stdint.h>

#define BS26_MAGIC     0xB526
#define BS26_VER       1
#define BS26_FLAG_DC   0x01
#define BS26_FLAG_AC   0x02

struct __attribute__((packed)) Bs26PacketV1 {
    uint16_t magic;
    uint8_t  ver;
    uint8_t  flags;
    uint8_t  hundreds, tens, ones;
    uint8_t  decouter, decmid, decinner;
    uint8_t  ua_dac, v_dac;
    uint16_t dec;
    uint16_t raw_h, raw_t, raw_o;
    int16_t  ads_t, ads_o, ads_vcc;
    int16_t  cdt;
    uint32_t up;
    uint32_t seq;
    uint16_t boots;
};
static_assert(sizeof(Bs26PacketV1) == 38, "bs26 packet must be exactly 38 bytes");

#endif // BS26_PACKET_V1_H
