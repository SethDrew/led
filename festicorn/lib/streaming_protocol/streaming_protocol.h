#pragma once
#include <stdint.h>

// [0xFF][0xAA][RGB x numPixels][XOR checksum] streaming frame receiver.
// Usage:
//   StreamReceiver rx;
//   streamReceiverInit(rx, numPixels, rgbBuffer);
//   // In serial loop:
//   while (Serial.available()) {
//       if (streamReceiverFeed(rx, Serial.read())) {
//           // Frame complete, data is in rgbBuffer
//       }
//   }

struct StreamReceiver {
    uint8_t *buf;
    uint16_t bufLen;     // numPixels * 3
    uint16_t pos;
    uint8_t xorCheck;
    uint8_t state;       // 0=SYNC1, 1=SYNC2, 2=DATA, 3=CHECK
};

static inline void streamReceiverInit(StreamReceiver &rx, uint16_t numPixels, uint8_t *rgbBuf) {
    rx.buf = rgbBuf;
    rx.bufLen = numPixels * 3;
    rx.pos = 0;
    rx.xorCheck = 0;
    rx.state = 0;
}

// Feed one byte. Returns true when a complete valid frame is in buf.
static inline bool streamReceiverFeed(StreamReceiver &rx, uint8_t b) {
    switch (rx.state) {
        case 0: // SYNC1
            if (b == 0xFF) rx.state = 1;
            return false;
        case 1: // SYNC2
            if (b == 0xAA) {
                rx.state = 2;
                rx.pos = 0;
                rx.xorCheck = 0;
            } else if (b != 0xFF) {
                rx.state = 0;
            }
            // consecutive 0xFF: stay in SYNC1 equivalent (state 1)
            return false;
        case 2: // DATA
            if (rx.pos < rx.bufLen) {
                rx.buf[rx.pos++] = b;
                rx.xorCheck ^= b;
            }
            if (rx.pos >= rx.bufLen) rx.state = 3;
            return false;
        case 3: // CHECK
            rx.state = 0;
            return (b == rx.xorCheck);
    }
    rx.state = 0;
    return false;
}
