// v1 lo-fi audio WAVEFORM stream packet — raw PCM, NOT the 5-byte envelope.
//
// Sender:    festicorn/original-duck/src/sender_audio.cpp   (TEST firmware)
// Receiver:  festicorn/bench-bulbs/src/audio_recorder.cpp   (TEST firmware)
//
// This is a bench/test path for capturing what the duck's INMP441 actually
// hears as playable audio — distinct from AudioPacketV1 (the 5-byte per-window
// RMS envelope that drives effects). The duck streams the decimated waveform,
// the recorder writes it to flash, a host pulls it and wraps a .wav.
//
// 204 bytes, broadcast over ESP-NOW on channel 1. The 200-sample payload makes
// this length unique among the live packets (16 B accel / 12 B gyro / 5 B
// audio / 15 B legacy SensorPacket), so receivers disambiguate by size alone.
// ESP-NOW caps a single payload at 250 bytes; 204 leaves margin.
//
// Pipeline (sender side):
//   INMP441 I2S @ 16 kHz, 32-bit words -> (raw >> 8) 24-bit domain
//   -> ÷2 average-decimate to 8 kHz
//   -> one-pole DC-removal HPF (mic has a large DC bias; MANDATORY)
//   -> linear quantize to unsigned 8-bit, silence = 128
//        byte = clamp_u8( (ac >> QUANT_SHIFT) + 128 )
//   QUANT_SHIFT is the one by-ear knob: lower = louder (more clipping).
//
//   seq        packet counter, wraps at 65535 (gap = dropped packet)
//   rate_code  sample-rate enum (see AUDIO_STREAM_RATE_* below)
//   n          valid samples in this packet (normally AUDIO_STREAM_SAMPLES)
//   samples[]  unsigned 8-bit PCM, mid/silence = 128

#ifndef AUDIO_STREAM_PACKET_V1_H
#define AUDIO_STREAM_PACKET_V1_H

#include <stdint.h>

#define AUDIO_STREAM_SAMPLES   200   // samples per packet

// rate_code enum — keep host (audio_pull.py) in sync.
#define AUDIO_STREAM_RATE_8K   0     // 8000 Hz
#define AUDIO_STREAM_RATE_16K  1     // 16000 Hz (reserved, not used yet)

struct __attribute__((packed)) AudioStreamPacketV1 {
    uint16_t seq;
    uint8_t  rate_code;
    uint8_t  n;
    uint8_t  samples[AUDIO_STREAM_SAMPLES];
};
static_assert(sizeof(AudioStreamPacketV1) == 204,
              "audio stream packet must be exactly 204 bytes");

#endif // AUDIO_STREAM_PACKET_V1_H
