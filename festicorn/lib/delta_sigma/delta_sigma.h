#pragma once
#include <stdint.h>

// First-order delta-sigma ditherer (8.8 fixed-point accumulator).
// accum: persistent accumulator state (one per channel), init to 0.
// target16: desired output in 16-bit range (e.g., brightness * color).
// Returns: dithered 8-bit output.
static inline uint8_t deltaSigma(uint16_t &accum, uint16_t target16) {
    accum += target16;
    uint8_t out = accum >> 8;
    accum &= 0xFF;
    return out;
}
